#include "graphics/guest_gpu/graphicsRun.h"

#include "common/memoryTrace.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/gpuTiming.h"
#include "common/liveControl.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/host_gpu/vulkanRecording.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <semaphore>
#include <thread>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

static thread_local Pm4Execution* g_current_execution = nullptr;
static thread_local GuestGpu*     g_gpu_state         = nullptr;

GuestGpu::GuestGpu(RenderContext& renderer): m_renderer(renderer) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	m_gfx_cp = std::make_unique<CommandProcessor>(renderer, 0);
	m_thread = std::jthread(ThreadRun, this);
}

GuestGpu::~GuestGpu() {
	Shutdown();
}

void GuestGpu::Shutdown() {
	std::lock_guard shutdown_lock(m_shutdown_mutex);
	if (m_shutdown_complete) {
		return;
	}
	{
		Common::LockGuard lock(m_queue_mutex);
		m_accepting = false;
		m_stopping  = true;
		m_work_available.SignalAll();
	}
	if (m_thread.joinable()) {
		m_thread.join();
	}
	m_shutdown_complete = true;
}

bool GuestGpu::IsStopping() {
	Common::LockGuard lock(m_queue_mutex);
	return m_stopping;
}

void GuestGpu::SendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_commands.push_back(std::move(command));
	m_pending_commands.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
}

void GuestGpu::ProcessCommands() {
	while (m_pending_commands.load(std::memory_order_acquire) != 0) {
		Common::UniqueFunction<void> command;
		{
			Common::LockGuard lock(m_queue_mutex);
			EXIT_IF(m_commands.empty());
			command = std::move(m_commands.front());
			m_commands.pop_front();
			EXIT_IF(m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
		}
		command();
	}
}

void GuestGpu::SendCommandSync(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	std::binary_semaphore done {0};
	SendCommand([operation = std::move(command), &done]() mutable {
		operation();
		done.release();
	});
	done.acquire();
}

void GuestGpu::Submit(std::span<const uint32_t> commands) {
	if (commands.empty()) {
		return;
	}
	Submission submission;
	submission.type     = SubmissionType::Commands;
	submission.queue_id = 0;
	submission.commands = commands;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitCompute(uint32_t queue, std::span<const uint32_t> commands) {
	EXIT_IF(commands.empty());

	EXIT_NOT_IMPLEMENTED(queue < ComputeQueueBase || queue >= ComputeQueueBase + ComputeQueueCount);

	const auto compute_queue = queue - ComputeQueueBase;
	Submission submission;
	submission.type     = SubmissionType::Commands;
	submission.queue_id = 1 + compute_queue;
	submission.commands = commands;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitFlipPreparation(uint64_t request_id) {
	Submission submission;
	submission.type            = SubmissionType::FlipPreparation;
	submission.queue_id        = 0;
	submission.flip_request_id = request_id;
	Enqueue(std::move(submission));
}

void GuestGpu::SuspendPoint() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	// Do not hold a queue lock while waiting: asynchronous work may be needed to
	// finish the preceding graphics frame. The first point returns immediately.
	m_suspend_point_ready->acquire();
	Submission submission;
	submission.type = SubmissionType::SuspendPoint;
	Enqueue(std::move(submission));
	m_done_num++;
}

int GuestGpu::GetFrameNum() const {
	return m_done_num;
}

CommandProcessor& GuestGpu::GetProcessor(uint32_t queue_id) {
	if (queue_id == 0) {
		return *m_gfx_cp;
	}
	auto& processor = m_compute_cp[queue_id - 1];
	if (processor == nullptr) {
		processor = std::make_unique<CommandProcessor>(m_renderer, ComputeQueueBase + queue_id - 1);
	}
	return *processor;
}

void CommandProcessor::Reset() {
	m_sh_ctx.Reset();
	m_ucfg.Reset();
	m_ctx.Reset();
	m_saved_ctx.reset();
	m_index_type_and_size              = 0;
	m_index_buffer_size                = 0;
	m_index_base_addr                  = 0;
	m_num_instances                    = 1;
	m_num_instances_addr               = 0;
	m_predicate_skip                   = false;
	m_draw_indirect_args_base_addr     = 0;
	m_dispatch_indirect_args_base_addr = 0;
}

// Pops a pushed context, then resets the register state the operation bits name.
void CommandProcessor::ResetQueue(uint32_t op) {
	if ((op & Pm4::ResetQueueOp::PopContext) != 0 && m_saved_ctx.has_value()) {
		ApplyContextStateOperation(ContextStateOperation::Pop);
	}
	if ((op & Pm4::ResetQueueOp::CxState) != 0) {
		m_ctx.Reset();
	}
	if ((op & Pm4::ResetQueueOp::ShState) != 0) {
		m_sh_ctx.Reset();
	}
	if ((op & Pm4::ResetQueueOp::UcState) != 0) {
		m_ucfg.Reset();
	}
}

void CommandProcessor::ApplyContextStateOperation(ContextStateOperation operation) {
	switch (operation) {
		case ContextStateOperation::Clear: m_ctx.Reset(); break;
		case ContextStateOperation::Push:
			EXIT_IF(m_saved_ctx.has_value());
			m_saved_ctx.emplace(m_ctx);
			break;
		case ContextStateOperation::Pop:
			EXIT_IF(!m_saved_ctx.has_value());
			m_ctx = *m_saved_ctx;
			m_saved_ctx.reset();
			break;
		case ContextStateOperation::PushClear:
			EXIT_IF(m_saved_ctx.has_value());
			m_saved_ctx.emplace(m_ctx);
			m_ctx.Reset();
			break;
	}
}

void CommandProcessor::BufferInit() {
	m_draws = 0;
	GetScheduler().Begin(m_ctx, m_ucfg, m_sh_ctx);
}

void CommandProcessor::BufferFlush() {
	GetScheduler().Flush();
	// The occlusion counters reach a guest that polls them without waiting for a later fence.
	m_renderer.GetOcclusionQueries().Publish();
}

void CommandProcessor::BufferWait() {
	BufferInit();
	GetScheduler().Finish();
}

void CommandProcessor::WaitForRewind(bool valid) {
	if (!valid) {
		SuspendPm4(Common::GpuTiming::StallKind::Rewind);
	}
}

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func) {
	switch (func) {
		case 0: return true;
		case 1: return (value & mask) < ref;
		case 2: return (value & mask) <= ref;
		case 3: return (value & mask) == ref;
		case 4: return (value & mask) != ref;
		case 5: return (value & mask) >= ref;
		case 6: return (value & mask) > ref;
		default: EXIT("unknown wait compare function: %" PRIu32 "\n", func);
	}

	return false;
}

template <typename T>
void CommandProcessor::WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t wait_op) {
	EXIT_IF(addr == nullptr);
	if ((wait_op & ~1u) != 0) {
		EXIT("unsupported wait_reg_mem operation: 0x%08" PRIx32 "\n", wait_op);
	}

	// A label the GPU work recorded before stores at the end of that work. Every command recorded
	// after a guest flush runs after it, so the stream need not stop until the GPU gets there;
	// the label itself stays pending until then (see PendingLabels).
	const auto elide = [&] {
		return m_renderer.ElideLabelWait(
		    reinterpret_cast<uint64_t>(addr), sizeof(T),
		    [&](uint64_t value) { return TestWaitRegMemValue(value, ref, mask, func); });
	};
	if (m_renderer.HasElidedLabels()) {
		// The bytes may belong to a label whose earlier wait was elided and that is not stored.
		if (elide()) {
			EmitGlobalBarrier();
			return;
		}
		SettleLabelAccess(addr, sizeof(T));
	}
	const T observed = *addr;
	// A wait on a value the GPU itself stored orders no guest store when the GPU stored it after
	// this submission began (a pending label, as elided above, is stored later still): the GPU
	// work that stored it ran after the stores its own submission ordered, and a guest store made
	// after this submission, even of the same value, raced with that store on the console. Any
	// other value may come from a guest store, which orders every store made before it.
	const bool gpu_value =
	    m_renderer.GetPendingLabels().StoredSince(reinterpret_cast<uint64_t>(addr), sizeof(T),
	                                              static_cast<uint64_t>(observed), m_store_epoch);
	if (!TestWaitRegMemValue(observed, ref, mask, func)) {
		if (elide()) {
			EmitGlobalBarrier();
			return;
		}
		SuspendPm4(Common::GpuTiming::StallKind::WaitRegMem);
	} else if (!gpu_value) {
		m_renderer.NoteCommandOrderingPoint();
	}
}

template void CommandProcessor::WaitRegMem<uint32_t>(uint32_t, const uint32_t*, uint32_t, uint32_t,
                                                     uint32_t);
template void CommandProcessor::WaitRegMem<uint64_t>(uint32_t, const uint64_t*, uint64_t, uint64_t,
                                                     uint32_t);

void CommandProcessor::MemSemaphore(uint64_t* counter, bool signal, bool write) {
	// The counter is a packet input in guest memory, maybe written by recorded GPU work.
	SettleMemoryRead(counter, sizeof(uint64_t));
	if (!signal) {
		m_renderer.NoteCommandOrderingPoint();
	}
	if (signal) {
		*counter = write ? 1u : *counter + 1u;
	} else if (*counter == 0) {
		SuspendPm4(Common::GpuTiming::StallKind::WaitRegMem);
	} else {
		(*counter)--;
	}
}

// 1: a WRITE_DATA over bytes the GPU wrote is stored on the GPU's side, and into guest memory
// too where that can be (BufferCache::StoreWords); 0: into guest memory, after the page was
// fetched back. write_data_gpu_words: the most dwords stored that way (4: only the few, as
// it was at first).
static Common::LiveControl::Switch  g_write_data_gpu("write_data_gpu", 1);
static Common::LiveControl::Switch  g_write_data_gpu_words("write_data_gpu_words", 1024);
static Common::LiveControl::Counter g_write_data_gpu_stores("write_data_gpu_stores");
// Dwords of WRITE_DATA packets a frame; the packets that stored into guest memory although some
// of their bytes were the GPU's (a wait for the GPU each); and, settle_log 1, a line for each.
static Common::LiveControl::Counter g_write_data_words("write_data_words");
static Common::LiveControl::Counter g_write_data_fetched("write_data_fetched");
static constexpr uint32_t           FilledWords = 4;
// 1: a line for each packet read or store that found bytes the GPU still wrote.
static Common::LiveControl::Switch  g_settle_log("settle_log", 0);

void CommandProcessor::WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num,
                                 uint32_t write_control) {
	const uint32_t dst_sel = ((write_control >> 30u) & 0x1u) | ((write_control >> 7u) & 0x1eu);
	const bool     write_one_address = ((write_control >> 16u) & 0x1u) != 0;

	switch (dst_sel) {
		case 0:
		case 2:
		case 4:
		case 5:
		case 6: break;
		default: EXIT("unsupported writeData destination selector 0x%02" PRIx32 "\n", dst_sel);
	}
	if (dw_num == 0) {
		return;
	}

	const uint64_t bytes = (write_one_address ? 1u : dw_num) * sizeof(uint32_t);
	SettleLabelAccess(dst, bytes);
	// A store over bytes the GPU wrote (in Saros one dword, once or twice a frame: a value a
	// compute pass leaves and the command stream sets again) is the GPU's own in the order of
	// its commands: written into the buffer that holds the bytes, it reaches guest memory with
	// the GPU's other writes of the page. Stored into guest memory here, the page had to be
	// fetched back from the GPU first (it is uploaded whole afterwards), and this thread waited
	// for everything the GPU had queued: 0.4 to 5 ms a frame, 87 ms in a frame with a backlog.
	// The same for a longer store: Saros writes a record of 56 dwords once a frame into a page
	// a compute pass wrote just before, whose download was still on its way: this thread waited
	// for that submission, 0.85 times a frame.
	g_write_data_words.Add(dw_num);
	auto&      buffer_cache = m_renderer.GetBufferCache();
	const auto address      = reinterpret_cast<uint64_t>(dst);
	if (g_write_data_gpu.On() && !write_one_address &&
	    dw_num <= std::max(g_write_data_gpu_words.Get(), FilledWords) && (address & 3u) == 0) {
		if (buffer_cache.HasOnlyGpuDirtyBytes(address, bytes)) {
			const bool both = buffer_cache.StoreWords(address, {src, dw_num});
			g_write_data_gpu_stores.Add();
			if (g_settle_log.On()) {
				LOGF("Write data: 0x%" PRIx64 " %u dwords, stored %s, first 0x%08" PRIx32 "\n", address,
				     dw_num, both ? "on both sides" : "in the buffer alone", src[0]);
			}
			return;
		}
	}
	if (Common::LiveControl::Active() && buffer_cache.HasGpuDirtyBytes(address, bytes)) {
		g_write_data_fetched.Add();
		if (g_settle_log.On()) {
			LOGF("Write data: 0x%" PRIx64 " %u dwords%s, fetched: %s\n", address, dw_num,
			     write_one_address ? " to one address" : "",
			     buffer_cache.HasOnlyGpuDirtyBytes(address, bytes) ? "all its bytes are the GPU's"
			                                                      : "some of its bytes are the GPU's");
		}
	}
	// What a write fault would do, without taking one: the caches stop watching the bytes.
	(void)m_renderer.InvalidateMemory(reinterpret_cast<uint64_t>(dst), bytes);
	// Completion labels can be reused immediately; memcpy may write a destination twice.
	auto* output = static_cast<volatile uint32_t*>(dst);
	for (uint32_t i = 0; i < dw_num; i++) {
		*output = src[i];
		output += !write_one_address;
	}
}

void CommandProcessor::WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes) {
	if (dst_address == 0 || (num_bytes != sizeof(uint32_t) && num_bytes != sizeof(uint64_t)) ||
	    (dst_address & (num_bytes - 1u)) != 0) {
		EXIT("invalid reference-clock copy, dst=0x%016" PRIx64 " size=%u\n", dst_address,
		     num_bytes);
	}
	const auto value = Sync::ReadReferenceClock();
	SettleLabelAccess(reinterpret_cast<const void*>(dst_address), num_bytes);
	std::memcpy(reinterpret_cast<void*>(dst_address), &value, num_bytes);
	static std::atomic<uint32_t> clock_log_count {0};
	if (clock_log_count.fetch_add(1) < 64) {
		LOGF("\t copy_data reference clock: dst=0x%016" PRIx64 " value=0x%016" PRIx64
		     " size=%u\n",
		     dst_address, value, num_bytes);
	}
}

void CommandProcessor::GetLodStats(volatile uint32_t* dst, uint32_t size, bool reset) {
	auto& counters = m_renderer.GetRenderExecutor().m_mip_stats;
	// Each of up to 256 2-dword slots after the 16-dword header holds the clamped sample count
	// (dword 0 bits 0-23), the slot id (bits 24-31), the unclamped sample count (dword 1 bits 0-23)
	// and the finest mip sampled (bits 24-27, 15 for none). A non-zero header dword 0 then marks
	// the report complete.
	if (dst != nullptr && size >= 64) {
		const auto slots = std::min((size - 64) / 8, 256u);
		for (uint32_t slot = 0; slot < slots; slot++) {
			const auto& counter = counters[slot];
			dst[16 + slot * 2]  = (slot << 24u) | uint32_t {counter.clamped};
			dst[16 + slot * 2 + 1] =
			    (uint32_t {counter.finest_level} << 24u) | uint32_t {counter.unclamped};
		}
		dst[0] = 1;
	}
	if (reset) {
		counters = {};
	}
}

void CommandProcessor::DmaData(uint8_t dst_sel, uint64_t dst_address_or_offset, uint8_t src_sel,
                               uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes) {
	if (num_bytes == 0 || static_cast<uint32_t>(dst_address_or_offset) == 0x3022cu) {
		return;
	}
	auto decode_gds = [](uint8_t selector, bool& is_gds) {
		switch (selector) {
			case 0:
			case 3: is_gds = false; return true;
			case 1: is_gds = true; return true;
			default: return false;
		}
	};
	if (dst_sel == 2) {
		// kNowhere discards the GL2 prefetch destination without a guest-visible write.
		if (src_sel != 3) {
			EXIT("unsupported dmaData nowhere source selector 0x%02" PRIx8 "\n", src_sel);
		}
		return;
	}
	bool dst_gds = false;
	if (!decode_gds(dst_sel, dst_gds)) {
		EXIT("unsupported dmaData destination selector 0x%02" PRIx8 "\n", dst_sel);
	}
	auto& buffer_cache = m_renderer.GetBufferCache();
	if (src_sel == 2) {
		buffer_cache.FillBuffer(
		    dst_address_or_offset, num_bytes,
		    static_cast<uint32_t>(src_address_or_offset_or_immediate & 0xffffffffu), dst_gds);
		return;
	}
	bool src_gds = false;
	if (!decode_gds(src_sel, src_gds)) {
		EXIT("unsupported dmaData source selector 0x%02" PRIx8 "\n", src_sel);
	}
	if (src_gds && dst_gds) {
		EXIT("unsupported dmaData GDS-to-GDS copy\n");
	}
	buffer_cache.CopyBuffer(dst_address_or_offset, src_address_or_offset_or_immediate, num_bytes,
	                        dst_gds, src_gds);
}

void GuestGpu::Enqueue(Submission submission) {
	// Guest stores made before the submission are ordered by its first command, later ones by a
	// later submission or a wait.
	submission.store_epoch = m_renderer.BeginGuestStoreEpoch();
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_queues[submission.queue_id].push_back(std::move(submission));
	m_submission_count++;
	m_work_available.Signal();
}

void GuestGpu::WaitForIdle() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	Common::LockGuard lock(m_queue_mutex);
	while (m_processing || !m_commands.empty() || m_submission_count != 0) {
		m_idle.Wait(&m_queue_mutex);
	}
}

void GuestGpu::ThreadRun(void* data) {
	auto* gpu = static_cast<GuestGpu*>(data);
	EXIT_IF(gpu == nullptr);
	KYTY_PROFILER_THREAD("Thread_Gpu");
	Common::LiveControl::RegisterThread("gpu");
	// This thread's Vulkan commands are replayed by a recording thread (vulkanRecording.h); the
	// scope ends, with everything replayed, when the thread does.
	const VulkanRecording::ProducerScope recording;
	// This thread's time is the frame's: it runs above the guest's threads, which otherwise hold
	// its CPUs for whole time slices while they stream a level in (KYTY_RENDER_PRIORITY=0: normal
	// priority). KYTY_RENDER_CPUS names the CPUs it may run on ("auto": the performance cores
	// without CPU 0's, which serves most interrupts); unset, the scheduler places it.
	if (const char* priority = std::getenv("KYTY_RENDER_PRIORITY");
	    priority == nullptr || std::strcmp(priority, "0") != 0) {
		Common::Thread::RaiseCurrentThreadPriority();
	}
	if (const char* cpus = std::getenv("KYTY_RENDER_CPUS"); cpus != nullptr && *cpus != '\0') {
		const bool pinned = Common::Thread::PinCurrentThreadToCpus(cpus);
		LOGF("GPU thread: KYTY_RENDER_CPUS=%s %s\n", cpus, pinned ? "applied" : "not applied");
	}
	g_gpu_state = gpu;

	bool held_for_programs = false;
	for (;;) {
		Submission                   submission;
		Common::UniqueFunction<void> command;
		bool                         should_stop = false;
		{
			Common::LockGuard lock(gpu->m_queue_mutex);
			while (gpu->m_commands.empty() && gpu->m_submission_count == 0 && !gpu->m_stopping) {
				gpu->m_processing = false;
				gpu->m_idle.Signal();
				const auto start = std::chrono::steady_clock::now();
				gpu->m_work_available.Wait(&gpu->m_queue_mutex);
				Common::GpuTiming::Sync().AddGpuThreadStarved(
				    std::chrono::duration_cast<std::chrono::nanoseconds>(
				        std::chrono::steady_clock::now() - start)
				        .count());
			}
			if (gpu->m_stopping && gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_processing = false;
				gpu->m_idle.SignalAll();
				should_stop = true;
			} else if (!gpu->m_commands.empty()) {
				command = std::move(gpu->m_commands.front());
				gpu->m_commands.pop_front();
				EXIT_IF(gpu->m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
				gpu->m_processing = true;
			} else {
				int selected_queue = -1;
				for (uint32_t offset = 0; offset < QueueCount; offset++) {
					const auto id = (gpu->m_next_queue + offset) % QueueCount;
					if (!gpu->m_queues[id].empty() && !gpu->m_queues[id].front().blocked) {
						selected_queue = static_cast<int>(id);
						break;
					}
				}
				if (selected_queue < 0) {
					gpu->m_processing = false;
					// Sleep until new work, a memory write from outside the streams, or the
					// next poll (guest CPU stores are not reported). A write after the
					// suspended streams last ran makes the sleep pointless.
					uint64_t writes_seen = UINT64_MAX;
					for (const auto& queue: gpu->m_queues) {
						if (!queue.empty() && queue.front().blocked) {
							writes_seen = std::min(writes_seen, queue.front().blocked_writes);
						}
					}
					gpu->m_waiting_blocked.store(true);
					if (gpu->m_memory_writes.load() == writes_seen) {
						const auto start = std::chrono::steady_clock::now();
						const bool signaled =
						    gpu->m_work_available.WaitFor(&gpu->m_queue_mutex, 100);
						Common::GpuTiming::Sync().AddGpuThreadBlocked(
						    std::chrono::duration_cast<std::chrono::nanoseconds>(
						        std::chrono::steady_clock::now() - start)
						        .count(),
						    !signaled);
					}
					gpu->m_waiting_blocked.store(false);
					for (auto& queue: gpu->m_queues) {
						if (!queue.empty()) {
							queue.front().blocked = false;
						}
					}
					continue;
				}
				auto& queue = gpu->m_queues[static_cast<uint32_t>(selected_queue)];
				submission  = std::move(queue.front());
				queue.pop_front();
				gpu->m_submission_count--;
				gpu->m_next_queue = (static_cast<uint32_t>(selected_queue) + 1) % QueueCount;
				gpu->m_processing = true;
			}
		}
		if (should_stop) {
			gpu->m_gfx_cp->BufferWait();
			g_gpu_state = nullptr;
			return;
		}

		if (command) {
			gpu->m_renderer.GetBufferCache().NoteCommandBoundary();
			{
				Common::DebugCounters::ScopedTimer busy(
				    Common::DebugCounters::Counter::GpuThreadBusyNs);
				command();
			}

			Common::LockGuard lock(gpu->m_queue_mutex);
			gpu->m_processing = false;
			if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_idle.SignalAll();
			}
			continue;
		}

		if (!held_for_programs) {
			// The game's first submission: the programs translated ahead of their first use
			// are finished first, so that nothing is translated beside the game.
			held_for_programs = true;
			gpu->m_renderer.HoldForPreparedPrograms();
		}
		const auto writes   = gpu->m_memory_writes.load();
		const bool complete = [&] {
			Common::DebugCounters::ScopedTimer busy(
			    Common::DebugCounters::Counter::GpuThreadBusyNs);
			return gpu->Process(submission);
		}();

		Common::LockGuard lock(gpu->m_queue_mutex);
		if (!complete) {
			submission.blocked        = true;
			submission.blocked_writes = writes;
			gpu->m_queues[submission.queue_id].push_front(std::move(submission));
			gpu->m_submission_count++;
		} else {
			for (auto& queue: gpu->m_queues) {
				if (!queue.empty()) {
					queue.front().blocked = false;
				}
			}
		}
		gpu->m_processing = false;
		if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
			gpu->m_idle.SignalAll();
		}
	}
}

void GuestGpu::NotifyMemoryWritten() {
	m_memory_writes.fetch_add(1);
	if (m_waiting_blocked.load()) {
		Common::LockGuard lock(m_queue_mutex);
		m_work_available.Signal();
	}
}

bool GuestGpu::Process(Submission& submission) {
	auto& cp = GetProcessor(submission.queue_id);
	// A resumed submission orders nothing new: the wait it stopped at runs again and orders the
	// stores it observes.
	m_renderer.NoteCommandOrderingPoint(submission.store_epoch);
	cp.SetStoreEpoch(submission.store_epoch);

	cp.BufferInit();
	bool complete = true;
	const bool commands = submission.type == SubmissionType::Commands;
	if (commands) {
		// The submissions this processor runs next continue its register state; a flip or a
		// suspend point ends what the scan may assume.
		m_lookahead_queue.clear();
		{
			Common::LockGuard lock(m_queue_mutex);
			for (const auto& queued: m_queues[submission.queue_id]) {
				if (queued.type != SubmissionType::Commands) {
					break;
				}
				m_lookahead_queue.push_back(queued.commands);
			}
		}
		cp.SetLookaheadQueue(m_lookahead_queue);
		cp.PlanLookahead(submission.execution, submission.commands);
	}

	switch (submission.type) {
		case SubmissionType::Commands:
			complete =
			    cp.Process(submission.execution, submission.commands) == Pm4ProcessResult::Complete;
			if (submission.execution.MadeProgress()) {
				const Common::GpuTiming::SyncScope sync_reason(
				    Common::GpuTiming::SyncReason::QueueSlice);
				cp.BufferFlush();
			}
			break;
		case SubmissionType::FlipPreparation:
			cp.PrepareCpuFlip(submission.flip_request_id);
			break;
		case SubmissionType::SuspendPoint: {
			// Age caches at the frame boundary, not once per command list.
			m_renderer.RunGarbageCollector();
			cp.EmitGlobalBarrier();
			m_renderer.GetCommandScheduler().DeferPriorityOperation(
			    [ready = m_suspend_point_ready] {
				    Common::MemoryTrace::Note(Common::MemoryTrace::Kind::Suspend, 0, 0);
				    ready->release();
			    },
			    GuestWrites::None());
			const Common::GpuTiming::SyncScope sync_reason(
			    Common::GpuTiming::SyncReason::FrameBoundary);
			cp.BufferFlush();
			cp.Reset();
			break;
		}
	}

	if (commands) cp.SetLookaheadQueue({});
	return complete;
}

Pm4ProcessResult CommandProcessor::Process(Pm4Execution&             execution,
                                           std::span<const uint32_t> commands) {
	KYTY_PROFILER_BLOCK("CommandProcessor::Process");
	EXIT_IF(g_current_execution != nullptr);
	EXIT_IF(commands.size() > UINT32_MAX);
	if (execution.m_buffer_stack.empty() && !commands.empty()) {
		// The parser reads the commands like a packet input (SettleMemoryRead).
		SettleMemoryRead(commands.data(), commands.size_bytes());
		execution.m_buffer_stack.push_back({commands});
	}
	execution.m_suspended     = false;
	execution.m_made_progress = false;

	struct ExecutionScope {
		explicit ExecutionScope(Pm4Execution& execution) { g_current_execution = &execution; }
		~ExecutionScope() { g_current_execution = nullptr; }
	} execution_scope(execution);

	ProcessPm4(execution);
	return execution.m_buffer_stack.empty() ? Pm4ProcessResult::Complete
	                                        : Pm4ProcessResult::Blocked;
}

// 1: the look-ahead scan is planned again where the stream leaves what the scan read.
// - A called or chained buffer that holds GPU writes when the stream enters it: the scan passed
//   it before that work was recorded and read the commands of an earlier frame (or stopped at
//   the call, which plans again as before). Saros writes its dispatch lists with compute
//   shaders; their dispatches had no planned walk and the GPU thread walked every one itself.
// - A scan stop the stream never runs: the scan stopped inside a buffer the stream then left by
//   a taken branch or chain before that packet (the branch at the end of such a list), or the
//   buffer ended. Without a new plan nothing after the buffer was planned until the submission
//   ended: hundreds of packets in Saros.
// 0: a plan only where a scan stop is executed.
static Common::LiveControl::Switch  g_lookahead_replan("lookahead_replan", 1);
static Common::LiveControl::Counter g_lookahead_replans_settled("lookahead_replans_settled");
static Common::LiveControl::Counter g_lookahead_replans_lost("lookahead_replans_lost");

// The packets whose programs ResourceLookahead plans.
static bool IsLookaheadCommand(uint32_t opcode) {
	switch (opcode) {
		case Pm4::IT_DISPATCH_DIRECT:
		case Pm4::IT_DISPATCH_INDIRECT:
		case Pm4::IT_DRAW_INDEX_2:
		case Pm4::IT_DRAW_INDEX_AUTO:
		case Pm4::IT_DRAW_INDEX_OFFSET_2:
		case Pm4::IT_DRAW_INDIRECT:
		case Pm4::IT_DRAW_INDEX_INDIRECT:
		case Pm4::IT_DRAW_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI: return true;
		default: return false;
	}
}

const uint32_t* CommandProcessor::CurrentPacket() {
	if (g_current_execution == nullptr || g_current_execution->m_buffer_stack.empty()) {
		return nullptr;
	}
	const auto& cursor = g_current_execution->m_buffer_stack.back();
	return cursor.commands.data() + cursor.offset_dw;
}

void CommandProcessor::PlanLookahead(const Pm4Execution&       execution,
                                     std::span<const uint32_t> commands) {
	auto& pipelines    = m_renderer.GetPipelineCache();
	m_lookahead_active = pipelines.LookaheadEnabled();
	if (!m_lookahead_active) {
		return;
	}
	pipelines.BeginLookaheadRun(this);
	m_lookahead_stop = nullptr;
	m_lookahead_writes.clear();
	// The scan applies the shader-register packets to this processor's state to know each
	// dispatch's registers, then restores it; nothing else changes. A scanned user-data write
	// also drops the deferred indirect-draw value of its register (NoteUserDataWrite), which
	// the draws before that packet still read.
	const auto saved_sh            = m_sh_ctx;
	const auto saved_draw_base     = m_draw_indirect_args_base_addr;
	const auto saved_dispatch_base = m_dispatch_indirect_args_base_addr;
	m_lookahead_saved_deferred.assign(m_deferred_indirect.begin(), m_deferred_indirect.end());
	m_lookahead_budget = UINT32_MAX;
	bool to_end        = true;
	if (execution.m_buffer_stack.empty()) {
		to_end = ScanLookahead(commands, 0);
	} else {
		for (auto level = execution.m_buffer_stack.rbegin();
		     level != execution.m_buffer_stack.rend() && to_end; ++level) {
			to_end = ScanLookahead(level->commands.subspan(level->offset_dw), 0);
		}
	}
	// Into the queued submissions: the parser settles a submission's buffer before it reads it
	// (Process), so one holding GPU writes ends the scan.
	m_lookahead_budget = LookaheadQueuedCommands;
	for (const auto queued: m_lookahead_queued) {
		if (!to_end || m_lookahead_budget == 0) break;
		if (queued.empty()) continue;
		if (LibKernel::Memory::HasGpuDirtyBytes(reinterpret_cast<uint64_t>(queued.data()),
		                                        queued.size_bytes())) {
			break;
		}
		to_end = ScanLookahead(queued, 0);
	}
	m_lookahead_budget                 = UINT32_MAX;
	m_sh_ctx                           = saved_sh;
	m_draw_indirect_args_base_addr     = saved_draw_base;
	m_dispatch_indirect_args_base_addr = saved_dispatch_base;
	m_deferred_indirect.swap(m_lookahead_saved_deferred);
	pipelines.KickLookahead();
}

bool CommandProcessor::ScanLookahead(std::span<const uint32_t> commands, uint32_t depth) {
	constexpr uint32_t MaxDepth  = 4;
	auto&              pipelines = m_renderer.GetPipelineCache();
	const auto         total_dw  = static_cast<uint32_t>(commands.size());
	// The stream is planned again once it executed the packet the scan stopped at.
	const auto Stop = [this](const uint32_t* packet) {
		m_lookahead_stop = packet;
		return false;
	};
	uint32_t offset = 0;
	while (offset < total_dw) {
		const auto* packet = commands.data() + offset;
		const auto  header = packet[0];
		if (header == 0x80000000u || Pm4::PacketSizeDw(header) == 1u) {
			offset++;
			continue;
		}
		const auto remaining = total_dw - offset;
		const auto size      = KYTY_PM4_LEN(header);
		if (remaining < 2 || size > remaining) return Stop(packet);
		if ((header & 1u) != 0 && ShouldSkipPredicatedPackets()) {
			offset += size;
			continue;
		}
		const auto opcode = (header >> 8u) & 0xffu;
		switch (opcode) {
			case Pm4::IT_SET_SH_REG_INDIRECT:
				// A register table recorded GPU work still writes is read back by the parser.
				if (size == 5u &&
				    LibKernel::Memory::HasGpuDirtyBytes(
				        (packet[1] & 0xfffffffcu) | (static_cast<uint64_t>(packet[2]) << 32u),
				        (packet[4] & 0x3fffu) * 2ull * sizeof(uint32_t))) {
					return Stop(packet);
				}
				[[fallthrough]];
			case Pm4::IT_SET_SH_REG:
			case Pm4::IT_SET_BASE:
				(void)g_cp_op_func[opcode](*this, header & ~1u, packet + 1, remaining, total_dw);
				break;
			case Pm4::IT_NOP:
				if (KYTY_PM4_R(header) == Pm4::R_DISPATCH_RESET) return Stop(packet);
				break;
			case Pm4::IT_SET_UCONFIG_REG:
			case Pm4::IT_SET_UCONFIG_REG_INDEX: {
				// A flip block or a flip-done wait block is one unit for the parser
				// (CpOpSetUconfigReg). Neither changes the commands after it: the flip's label
				// tells the display a buffer is shown, and the wait in the other block only keeps
				// rendering off a buffer the display still reads.
				if (KYTY_PM4_R(header) != 1u || packet[1] != Pm4::SQ_THREAD_TRACE_USERDATA_2) break;
				const auto tag   = packet[2] & 0xff000000u;
				const auto block = tag == Pm4::USERDATA_TAG_FLIP ? Pm4::FLIP_PACKET_DW
				                   : tag == Pm4::USERDATA_TAG_WAIT_FLIP_DONE
				                       ? Pm4::WAIT_FLIP_DONE_PACKET_DW
				                       : 0u;
				if (block == 0u) break;
				if (block > remaining) return Stop(packet);
				offset += block;
				continue;
			}
			case Pm4::IT_DISPATCH_DIRECT:
				if (size != 5u) return Stop(packet);
				if (m_lookahead_budget == 0) return false;
				m_lookahead_budget--;
				m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(packet[4]));
				pipelines.PlanLookaheadDispatch(
				    this, packet, m_sh_ctx.GetCs(), packet[1], packet[2], packet[3],
				    (packet[4] & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0, 0);
				break;
			case Pm4::IT_DRAW_INDEX_2:
			case Pm4::IT_DRAW_INDEX_AUTO:
			case Pm4::IT_DRAW_INDEX_OFFSET_2:
			case Pm4::IT_DRAW_INDIRECT:
			case Pm4::IT_DRAW_INDEX_INDIRECT:
			case Pm4::IT_DRAW_INDIRECT_MULTI:
			case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI:
				if (m_lookahead_budget == 0) return false;
				m_lookahead_budget--;
				pipelines.PlanLookaheadDraw(this, packet, m_sh_ctx.GetVs(), m_sh_ctx.GetPs());
				break;
			case Pm4::IT_DISPATCH_INDIRECT: {
				uint64_t args = 0;
				uint32_t mode = 0;
				if (header == 0xc0021600u) {
					args = packet[1] | (static_cast<uint64_t>(packet[2]) << 32u);
					mode = packet[3];
				} else if (header == 0xc0011600u && m_dispatch_indirect_args_base_addr != 0) {
					args = m_dispatch_indirect_args_base_addr + packet[1];
					mode = packet[2];
				} else {
					return Stop(packet);
				}
				if (m_lookahead_budget == 0) return false;
				m_lookahead_budget--;
				m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));
				// Thread dimensions are read from the arguments when the dispatch runs.
				if (args != 0 && (args & 3u) == 0 &&
				    (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) == 0) {
					pipelines.PlanLookaheadDispatch(this, packet, m_sh_ctx.GetCs(), 0, 0, 0, false,
					                                args);
				}
				break;
			}
			case Pm4::IT_INDIRECT_BUFFER: {
				if (size != 4u || depth >= MaxDepth) return Stop(packet);
				const auto* buffer = reinterpret_cast<const uint32_t*>(
				    packet[1] | (static_cast<uint64_t>(packet[2]) << 32u));
				const auto count = packet[3] & 0xfffffu;
				if (count != 0 && buffer == nullptr) return Stop(packet);
				// Recorded GPU work may still write the commands; the parser reads them back.
				if (count != 0 &&
				    LibKernel::Memory::HasGpuDirtyBytes(reinterpret_cast<uint64_t>(buffer),
				                                        count * sizeof(uint32_t))) {
					return Stop(packet);
				}
				if (count != 0 && !ScanLookahead({buffer, count}, depth + 1)) return false;
				if ((packet[3] & (1u << 20u)) != 0) return true; // a chain does not return
				break;
			}
			case Pm4::IT_REWIND:
				// The guest is still writing the commands after an invalid rewind.
				if ((packet[1] & 0x80000000u) == 0) return Stop(packet);
				break;
			case Pm4::IT_WAIT_REG_MEM:
			case Pm4::IT_WAIT_REG_MEM_64: {
				// Commands after a wait the memory does not satisfy yet may not be written yet
				// (the guest releases command buffers it is still writing this way).
				const bool wide = opcode == Pm4::IT_WAIT_REG_MEM_64;
				if (size != (wide ? 9u : 7u) || (packet[1] & 0x10u) == 0) return Stop(packet);
				const auto mask_address = wide ? 0x7ull : 0x3ull;
				const auto address      = (packet[2] & ~mask_address) |
				                          (static_cast<uint64_t>(packet[3] & 0x3ffffu) << 32u);
				uint64_t   value        = 0;
				uint64_t   ref          = packet[4];
				uint64_t   mask         = wide ? packet[6] : packet[5];
				if (wide) {
					ref |= static_cast<uint64_t>(packet[5]) << 32u;
					mask |= static_cast<uint64_t>(packet[7]) << 32u;
				}
				if (address == 0 ||
				    !LibKernel::Memory::TryReadBacking(address, &value, wide ? 8u : 4u)) {
					return Stop(packet);
				}
				if (!TestWaitRegMemValue(value, ref, mask, packet[1] & 0x7u) &&
				    !LookaheadWaitPasses(address, wide ? 8u : 4u, ref, mask, packet[1] & 0x7u)) {
					return Stop(packet);
				}
				break;
			}
			case Pm4::IT_RELEASE_MEM: {
				// As CommandProcessor::ReleaseMem: data_sel 1/2 store an immediate label, which
				// interrupt selector 4 and graphics-queue selector 1 leave out.
				if (size != 8u) break;
				const auto data_sel  = (packet[2] >> 29u) & 0x7u;
				const auto interrupt = (packet[2] >> 24u) & 0x7u;
				const auto address =
				    (packet[3] & ~0x3ull) | (static_cast<uint64_t>(packet[4]) << 32u);
				const bool label_stored =
				    interrupt != 4u && (interrupt != 1u || IsAsyncComputeQueue());
				if ((data_sel == 1u || data_sel == 2u) && address != 0 && label_stored) {
					const auto value = packet[5] | (static_cast<uint64_t>(packet[6]) << 32u);
					m_lookahead_writes.push_back(
					    {.address = address,
					     .value   = data_sel == 1u ? value & 0xffffffffu : value,
					     .size    = data_sel == 1u ? 4u : 8u});
				}
				break;
			}
			case Pm4::IT_WRITE_DATA: {
				// The CP stores the data when it executes the packet (CommandProcessor::WriteData).
				if (size < 5u) break;
				const auto control = packet[1];
				const auto address = packet[2] | (static_cast<uint64_t>(packet[3]) << 32u);
				const auto data    = std::span(packet + 4, size - 4u);
				const bool one     = ((control >> 16u) & 0x1u) != 0;
				// A long write is data, not a flag a wait could be on.
				if (data.size() > 16u) break;
				for (uint32_t i = one ? static_cast<uint32_t>(data.size()) - 1u : 0u;
				     i < data.size(); ++i) {
					m_lookahead_writes.push_back(
					    {.address = address + (one ? 0u : i * sizeof(uint32_t)),
					     .value   = data[i],
					     .size    = 4u});
				}
				break;
			}
			// Commands whose effect on the stream depends on memory or GPU results.
			case Pm4::IT_SET_PREDICATION:
			case Pm4::IT_COND_EXEC: return Stop(packet);
			default: break;
		}
		offset += size;
	}
	return true;
}

bool CommandProcessor::LookaheadWaitPasses(uint64_t address, uint32_t size, uint64_t ref,
                                           uint64_t mask, uint32_t func) {
	const auto satisfied = [&](uint64_t value) {
		return TestWaitRegMemValue(value, ref, mask, func);
	};
	// The latest scanned write overlapping the bytes decides, as the latest label does for
	// PendingLabels; one that does not cover them all leaves the wait undecided.
	for (auto it = m_lookahead_writes.rbegin(); it != m_lookahead_writes.rend(); ++it) {
		if (address >= it->address + it->size || it->address >= address + size) continue;
		if (address < it->address || address + size > it->address + it->size) return false;
		uint64_t value = 0;
		std::memcpy(&value, reinterpret_cast<const uint8_t*>(&it->value) + (address - it->address),
		            size);
		return satisfied(value);
	}
	return m_renderer.GetPendingLabels().Satisfies(address, size, satisfied);
}

void CommandProcessor::ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain) {
	EXIT_IF(g_current_execution == nullptr);
	EXIT_IF(!g_current_execution->m_next_buffer.empty());
	// Recorded GPU work may have written the commands: games generate command buffers with
	// compute shaders and call them after that work.
	m_lookahead_settled =
	    m_lookahead_active && g_lookahead_replan.On() && !commands.empty() &&
	    LibKernel::Memory::HasGpuDirtyBytes(reinterpret_cast<uint64_t>(commands.data()),
	                                        commands.size_bytes());
	SettleMemoryRead(commands.data(), commands.size_bytes());
	g_current_execution->m_next_buffer = commands;
	g_current_execution->m_chain       = chain;
}

void CommandProcessor::SuspendPm4(Common::GpuTiming::StallKind kind) {
	EXIT_IF(g_current_execution == nullptr);
	Common::GpuTiming::Sync().AddStall(kind);
	g_current_execution->m_suspended = true;
}

void CommandProcessor::ProcessPm4(Pm4Execution& execution) {
	const bool debug_dump = graphics_debug_dump_enabled();
	while (!execution.m_buffer_stack.empty()) {
		// The previous packet recorded every GPU write it marked.
		m_renderer.GetBufferCache().NoteCommandBoundary();
		if (g_gpu_state != nullptr) {
			g_gpu_state->ProcessCommands();
		}
		auto& cursor = execution.m_buffer_stack.back();
		if (cursor.offset_dw == cursor.commands.size()) {
			// A stop inside the finished buffer was never run (running it plans again).
			const bool lost = g_lookahead_replan.On() && LookaheadStoppedIn(cursor.commands);
			execution.m_buffer_stack.pop_back();
			if (lost) {
				g_lookahead_replans_lost.Add();
				PlanLookahead(execution, {});
			}
			continue;
		}

		const auto* const packet        = cursor.commands.data() + cursor.offset_dw;
		const auto        total_dw      = static_cast<uint32_t>(cursor.commands.size());
		const auto        remaining_dw  = total_dw - cursor.offset_dw;
		const auto        packet_header = packet[0];
		const auto        opcode        = (packet_header >> 8u) & 0xffu;

		if (packet_header == 0x80000000u || Pm4::PacketSizeDw(packet_header) == 1u) {
			cursor.offset_dw++;
			execution.m_made_progress = true;
			continue;
		}

		EXIT_NOT_IMPLEMENTED(remaining_dw < 2);

		if (debug_dump) {
			LOGF("CP packet: offset=0x%05" PRIx32 " cmd_id=0x%08" PRIx32 " op=0x%02" PRIx32
			     " len=%" PRIu32 "\n",
			     cursor.offset_dw, packet_header, opcode, KYTY_PM4_LEN(packet_header));
		}

		if ((packet_header & 1u) != 0 && ShouldSkipPredicatedPackets()) {
			auto packet_dw = KYTY_PM4_LEN(packet_header);
			EXIT_NOT_IMPLEMENTED(packet_dw > remaining_dw);
			cursor.offset_dw += packet_dw;
			execution.m_made_progress = true;
			continue;
		}

		auto handler = g_cp_op_func[opcode];

		if (handler == nullptr) {
			const auto offset = cursor.offset_dw;
			LOGF("unknown PM4 packet: data=0x%016" PRIx64 ", num_dw=%" PRIu32
			     ", offset=0x%05" PRIx32 ", current=0x%016" PRIx64 "\n",
			     reinterpret_cast<uint64_t>(packet - offset), total_dw, offset,
			     reinterpret_cast<uint64_t>(packet));
			const auto  dump_begin = (offset > 8 ? offset - 8 : 0);
			const auto  dump_end   = std::min<uint32_t>(total_dw, offset + 16);
			auto* const base       = packet - offset;
			for (uint32_t i = dump_begin; i < dump_end; i++) {
				LOGF("\t%05" PRIx32 "%s %08" PRIx32 "\n", i, (i == offset ? ":" : " "), base[i]);
			}
			EXIT("unknown op\n\t%05" PRIx32 ":\n\tcmd_id = %08" PRIx32 "\n", offset, packet_header);
		}

		const bool lookahead_command = m_lookahead_active && IsLookaheadCommand(opcode);
		if (lookahead_command) m_renderer.GetPipelineCache().SetLookaheadCommand(this, packet);
		const auto packet_dw =
		    handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
		if (lookahead_command) m_renderer.GetPipelineCache().SetLookaheadCommand(nullptr, nullptr);
		bool replan = packet == m_lookahead_stop;
		EXIT_IF(packet_dw > remaining_dw);
		if (execution.m_suspended) {
			return;
		}
		cursor.offset_dw += packet_dw;
		execution.m_made_progress = true;
		if (!execution.m_next_buffer.empty()) {
			// Chains and taken branches reuse the fetcher; only calls retain a return cursor.
			if (execution.m_chain) {
				// The packets left behind are not run: a scan that stopped among them is over.
				if (!replan && g_lookahead_replan.On() && LookaheadStoppedIn(cursor.commands)) {
					g_lookahead_replans_lost.Add();
					replan = true;
				}
				cursor = {execution.m_next_buffer};
			} else {
				execution.m_buffer_stack.push_back({execution.m_next_buffer});
			}
			execution.m_next_buffer = {};
			if (std::exchange(m_lookahead_settled, false) && !replan) {
				g_lookahead_replans_settled.Add();
				replan = true;
			}
		}
		if (replan) PlanLookahead(execution, {});
	}
}

void CommandProcessor::SetIndexType(uint32_t index_type_and_size) {
	m_index_type_and_size = index_type_and_size & 0x3u;
}

void CommandProcessor::SetIndexBaseAddress(uint64_t index_base_addr) {
	m_index_base_addr = index_base_addr;
}

void CommandProcessor::SetIndexBufferSize(uint32_t index_buffer_size) {
	m_index_buffer_size = index_buffer_size;
}

void CommandProcessor::SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr) {
	m_draw_indirect_args_base_addr = draw_indirect_args_base_addr;
}

void CommandProcessor::SetDispatchIndirectArgsBaseAddress(
    uint64_t dispatch_indirect_args_base_addr) {
	m_dispatch_indirect_args_base_addr = dispatch_indirect_args_base_addr;
}

void CommandProcessor::SetNumInstances(uint32_t num_instances) {
	if (num_instances == 0) {
		num_instances = 1;
	}

	m_num_instances      = num_instances;
	m_num_instances_addr = 0;
	if (!m_deferred_indirect.empty()) {
		std::erase_if(m_deferred_indirect, [](const DeferredIndirectValue& value) {
			return value.field == DeferredIndirectValue::Field::InstanceCount;
		});
	}
}

void CommandProcessor::SettleMemoryRead(const volatile void* address, uint64_t size) {
	SettleLabelAccess(address, size);
	const std::pair<uint64_t, uint64_t> range {reinterpret_cast<uint64_t>(address), size};
	LibKernel::Memory::ReadGpuDirtyBytes(std::span {&range, 1u});
}

bool CommandProcessor::TryReadSettled(uint64_t address, void* data, uint64_t size) {
	SettleLabelAccess(reinterpret_cast<const void*>(address), size);
	if (m_renderer.GetBufferCache().HasGpuDirtyBytes(address, size)) {
		return false;
	}
	// The backing holds the bytes; the guest mapping of their page may still fault on other
	// GPU-written bytes.
	if (!LibKernel::Memory::TryReadBacking(address, data, size)) {
		std::memcpy(data, reinterpret_cast<const void*>(address), size);
	}
	return true;
}

// 1: a GPU-driven draw or dispatch whose arguments say that it does nothing, and can be read
// without waiting for the GPU, is left out; 0: every one is prepared and recorded; 2: a draw's
// arguments are waited for (see IndirectDrawsNothing). Off unless asked for
// (KYTY_SKIP_EMPTY_INDIRECT=1), for the reason given at fault_ahead (bufferCache.cpp): it finds
// two or three draws a frame, which is no gain to weigh against a doubt.
static Common::LiveControl::Switch  g_skip_empty_indirect("skip_empty_indirect", 0);
static Common::LiveControl::Counter g_empty_draws("empty_draws");
static Common::LiveControl::Counter g_empty_dispatches("empty_dispatches");

// 1: a GPU-driven draw whose record the GPU still holds is left out when the same draw had
// nothing to draw in the frames before (EmptyDrawPredictor). NOT EXACT: a draw that gets
// something to draw is recorded a frame or two late. 0 (the default): every such draw is
// prepared and recorded.
static Common::LiveControl::Switch  g_predict_empty_draws("predict_empty_draws", 0);
static Common::LiveControl::Counter g_predicted_draws("predicted_draws");
static Common::LiveControl::Counter g_predicted_wrong("predicted_wrong");
static Common::LiveControl::Counter g_draw_checks("draw_checks");

// Guesses which GPU-driven draws draw nothing, from what they drew in the frames before.
//
// Such a draw's record (its index and instance counts) is written by a culling pass on the GPU
// just before the draw, so the emulator cannot know that it is empty without waiting for the GPU,
// which costs more than preparing the draw (IndirectDrawsNothing). More than half of Saros's
// GPU-driven draws are empty, and mostly the same ones frame after frame: what the camera does
// not see stays unseen for a while.
//
// Every GPU-driven draw with one record gets an identity that needs no record: a key of its
// shaders, index buffer, targets and viewport (CommandProcessor::IndirectDrawKey) and its
// place among the draws with that key in the frame. Whether it was empty is learned for every
// such draw of every frame, left out or not, without a wait: the first eight bytes of its
// record are copied into a download buffer where the draws stand in the command stream
// (CommandBuffer::QueueRead) and read when that submission has completed. A draw is left out
// when its identity was empty in each of its last Streak checks (some passes are drawn every
// other frame, so a gap of up to Recent frames continues the row), the newest of them not older
// than Recent frames, and was not seen with something to draw in the last Hold frames. A draw
// left out that had something to draw is counted (predicted_wrong), and its identity is then
// drawn for the next Hold frames whatever its record says.
//
// What it costs when it is wrong: the draw is missing from the picture until the check of its
// frame arrives, one or two frames. Nothing else depends on a draw having been recorded. At a
// heavy spot of Saros about 125 of 170 empty draws a frame are left out and one draw in ten
// frames is left out wrongly, standing or turning; stills with it on and off do not differ.
class EmptyDrawPredictor {
public:
	explicit EmptyDrawPredictor(RenderContext& renderer)
	    : m_renderer(renderer),
	      m_download(renderer.GetCommandScheduler().Graphics(), renderer.GetCommandScheduler(),
	                 MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst,
	                 Slots * SlotBytes),
	      m_shared(std::make_shared<Shared>()) {
		m_shared->mapped = m_download.Mapped().data();
	}
	~EmptyDrawPredictor() {
		std::lock_guard lock(m_shared->mutex);
		m_shared->mapped = nullptr;
	}
	KYTY_CLASS_NO_COPY(EmptyDrawPredictor);

	// Without coherent memory the download buffer would need a call before each read.
	[[nodiscard]] bool Usable() const noexcept { return m_download.IsCoherent(); }

	// The identity of the next draw with `key` in this frame.
	[[nodiscard]] uint64_t Identify(uint64_t key) {
		const auto frame = Frame();
		auto&      seen  = m_seen[key];
		if (seen.frame != frame) {
			seen = {frame, 0};
		}
		const uint64_t place = seen.count++;
		return key ^ (place * 0x9e3779b97f4a7c15ull + (key << 6u) + (key >> 2u));
	}

	// Whether the draw is left out.
	[[nodiscard]] bool Predict(uint64_t id) {
		TakeResults();
		const auto it = m_history.find(id);
		if (it == m_history.end()) {
			return false;
		}
		const auto  frame = Frame();
		const auto& draw  = it->second;
		return draw.streak >= Streak && frame - draw.checked <= Recent &&
		       (draw.drew == 0 || frame - draw.drew > Hold);
	}

	// The record is in guest memory: what it says, at once.
	void Known(uint64_t id, bool empty) { Apply(id, Frame(), empty, false); }

	// Has the record of the draw read where the command stream stands, for the frames to come.
	void Check(CommandBuffer& buffer, uint64_t id, uint64_t args_address, bool left_out) {
		auto&      scheduler = m_renderer.GetCommandScheduler();
		const auto tick      = scheduler.CurrentTick();
		if (m_batch == nullptr || m_batch_tick != tick) {
			m_batch      = std::make_shared<std::vector<Entry>>();
			m_batch_tick = tick;
			// On the completion thread, when the submission that holds the copies has finished.
			scheduler.DeferPriorityOperation(
			    [shared = m_shared, batch = m_batch] {
				    std::lock_guard lock(shared->mutex);
				    if (shared->mapped == nullptr) {
					    return;
				    }
				    for (const auto& entry: *batch) {
					    std::array<uint32_t, 2> counts {};
					    std::memcpy(counts.data(), shared->mapped + entry.slot * SlotBytes, SlotBytes);
					    shared->results.push_back(
					        {entry.id, entry.frame, counts[0] == 0 || counts[1] == 0, entry.left_out});
				    }
				    shared->pending.store(true, std::memory_order_release);
			    },
			    GuestWrites::None());
		}
		const auto [source, offset] =
		    m_renderer.GetBufferCache().ObtainBuffer(args_address, SlotBytes, false, false);
		const auto slot = m_next_slot++ % Slots;
		buffer.QueueRead(source->Handle(), offset, m_download.Handle(), slot * SlotBytes, SlotBytes);
		m_batch->push_back({id, Frame(), slot, left_out});
		g_draw_checks.Add();
	}

private:
	// Both kinds of record begin with the vertex or index count and the instance count.
	static constexpr uint64_t SlotBytes = 2 * sizeof(uint32_t);
	static constexpr uint64_t Slots     = uint64_t {1} << 16u;
	static constexpr uint32_t Streak    = 4;
	static constexpr uint64_t Recent    = 4;
	static constexpr uint64_t Hold      = 90;
	static constexpr uint64_t Forget    = 3000;

	struct Entry {
		uint64_t id       = 0;
		uint64_t frame    = 0;
		uint64_t slot     = 0;
		bool     left_out = false;
	};
	struct Result {
		uint64_t id       = 0;
		uint64_t frame    = 0;
		bool     empty    = false;
		bool     left_out = false;
	};
	struct Shared {
		std::mutex          mutex;
		const uint8_t*      mapped = nullptr;
		std::vector<Result> results;
		std::atomic<bool>   pending {false};
	};
	struct Seen {
		uint64_t frame = 0;
		uint32_t count = 0;
	};
	struct Draw {
		// The newest frame checked, the frames in a row it was empty up to that one, and the
		// newest frame it had something to draw (0: never).
		uint64_t checked = 0;
		uint32_t streak  = 0;
		uint64_t drew    = 0;
	};

	// The guest's frames are counted by its flips. (From 1: 0 stands for "never".)
	[[nodiscard]] static uint64_t Frame() noexcept { return Common::GpuTiming::GuestFlips() + 1u; }

	void Apply(uint64_t id, uint64_t frame, bool empty, bool left_out) {
		auto& draw = m_history[id];
		if (!empty) {
			draw.streak = 0;
			draw.drew   = std::max(draw.drew, frame);
			if (left_out) {
				g_predicted_wrong.Add();
			}
		} else if (frame > draw.checked) {
			// Some passes are drawn every other frame, so the checks of one identity need not be
			// of frames in a row: a gap within what counts as recent continues the row.
			draw.streak = frame - draw.checked <= Recent ? draw.streak + 1u : 1u;
		}
		draw.checked = std::max(draw.checked, frame);
	}

	void TakeResults() {
		if (!m_shared->pending.load(std::memory_order_acquire)) {
			return;
		}
		{
			std::lock_guard lock(m_shared->mutex);
			m_results.swap(m_shared->results);
			m_shared->pending.store(false, std::memory_order_release);
		}
		for (const auto& result: m_results) {
			Apply(result.id, result.frame, result.empty, result.left_out);
		}
		m_results.clear();
		// Identities not seen for a long time are forgotten, now and then.
		if (const auto frame = Frame(); frame - m_swept > Forget) {
			m_swept = frame;
			std::erase_if(m_history,
			              [&](const auto& item) { return frame - item.second.checked > Forget; });
			std::erase_if(m_seen, [&](const auto& item) { return frame - item.second.frame > Forget; });
		}
	}

	RenderContext&                         m_renderer;
	Buffer                                 m_download;
	std::shared_ptr<Shared>                m_shared;
	std::shared_ptr<std::vector<Entry>>    m_batch;
	uint64_t                               m_batch_tick = 0;
	uint64_t                               m_next_slot  = 0;
	uint64_t                               m_swept      = 0;
	std::vector<Result>                    m_results;
	std::unordered_map<uint64_t, Seen>     m_seen;
	std::unordered_map<uint64_t, Draw>     m_history;
};

uint64_t CommandProcessor::IndirectDrawKey(bool indexed) const {
	uint64_t   key = indexed ? 0x9e3779b97f4a7c15ull : 0xc2b2ae3d27d4eb4full;
	const auto add = [&key](uint64_t value) {
		key ^= value + 0x9e3779b97f4a7c15ull + (key << 6u) + (key >> 2u);
	};
	const auto pair = [](uint32_t low, uint32_t high) {
		return static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32u);
	};
	const auto& vs = m_sh_ctx.GetVs();
	add(vs.es_regs.data_addr);
	add(vs.ls_regs.data_addr);
	add(vs.hs_regs.data_addr);
	add(vs.gs_regs.data_addr);
	add(m_sh_ctx.GetPs().ps_regs.data_addr);
	add(m_index_base_addr);
	add(pair(m_index_buffer_size, m_index_type_and_size));
	// The pass: its targets, and the part of them it draws into (shadow views share an atlas).
	const auto& depth = m_ctx.GetDepthRenderTarget();
	add(depth.z_write_base_addr);
	add(depth.z_read_base_addr);
	add(m_ctx.GetRenderTarget(0).base.addr);
	const auto& screen   = m_ctx.GetScreenViewport();
	const auto& viewport = screen.viewports[0];
	add(pair(std::bit_cast<uint32_t>(viewport.xscale), std::bit_cast<uint32_t>(viewport.xoffset)));
	add(pair(std::bit_cast<uint32_t>(viewport.yscale), std::bit_cast<uint32_t>(viewport.yoffset)));
	add(pair(static_cast<uint32_t>(viewport.viewport_scissor_left),
	         static_cast<uint32_t>(viewport.viewport_scissor_top)));
	add(pair(static_cast<uint32_t>(viewport.viewport_scissor_right),
	         static_cast<uint32_t>(viewport.viewport_scissor_bottom)));
	add(pair(static_cast<uint32_t>(screen.generic_scissor_left),
	         static_cast<uint32_t>(screen.generic_scissor_top)));
	add(pair(static_cast<uint32_t>(screen.generic_scissor_right),
	         static_cast<uint32_t>(screen.generic_scissor_bottom)));
	return key;
}

bool CommandProcessor::IndirectDrawsNothing(uint64_t args_address, uint32_t max_count,
                                            const volatile uint32_t* count_addr,
                                            uint32_t                 stride_in_bytes) {
	// Saros draws its scene with GPU-driven draws: a compute pass culls the instances and writes
	// each draw's record, and the game then issues every draw, also the ones the culling left
	// without an instance: 170 of 310 such draws a frame at one spot. Each takes the whole
	// preparation of a draw (pipeline, resources, bindings, render pass) to make the GPU draw
	// nothing, and an empty draw has no effect on the console either. But the culling pass
	// was recorded just before its draws, so the records are with the GPU when the packet comes:
	// only the few that are in guest memory already (two or three a frame there) are known to be
	// empty without a wait, and those are left out.
	constexpr uint32_t MaxRecords = 64;
	if (!g_skip_empty_indirect.On()) {
		return false;
	}
	uint32_t count = max_count;
	if (count_addr != nullptr) {
		uint32_t selected = 0;
		if (!TryReadSettled(reinterpret_cast<uint64_t>(count_addr), &selected, sizeof(selected))) {
			return false;
		}
		count = std::min(selected, max_count);
	}
	if (count > MaxRecords) {
		return false;
	}
	// 2 (for measuring, slower): the records are waited for. Then every empty draw is found,
	// 170 a frame at that spot, and the waits cost more than the draws: 49.5 to 45.0 fps.
	if (g_skip_empty_indirect.Get() >= 2 && count != 0) {
		SettleMemoryRead(reinterpret_cast<const void*>(args_address),
		                 static_cast<uint64_t>(count - 1u) * stride_in_bytes + 2 * sizeof(uint32_t));
	}
	for (uint32_t i = 0; i < count; i++) {
		// Both kinds of record begin with the vertex or index count and the instance count.
		std::array<uint32_t, 2> counts {};
		static_assert(offsetof(vk::DrawIndirectCommand, instanceCount) == sizeof(uint32_t) &&
		              offsetof(vk::DrawIndexedIndirectCommand, instanceCount) == sizeof(uint32_t));
		if (!TryReadSettled(args_address + static_cast<uint64_t>(i) * stride_in_bytes,
		                    counts.data(), sizeof(counts))) {
			return false;
		}
		if (counts[0] != 0 && counts[1] != 0) {
			return false;
		}
	}
	return true;
}
uint32_t CommandProcessor::NumInstances() {
	if (m_num_instances_addr != 0) {
		SettleMemoryRead(reinterpret_cast<const void*>(m_num_instances_addr), sizeof(uint32_t));
		m_num_instances      = *reinterpret_cast<const uint32_t*>(m_num_instances_addr);
		m_num_instances_addr = 0;
	}
	return m_num_instances;
}

void CommandProcessor::SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
                                      const volatile void* address) {
	uint64_t value = 0;

	switch (op) {
		case 0x00:
			m_predicate_skip = false;
			return;
		case 0x01: {
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			m_renderer.GetOcclusionQueries().Publish(true);
			// One begin/end pair per DB; bit 63 marks each counter ready.
			constexpr uint64_t ready_bit = 1ull << 63u;
			SettleLabelAccess(address, 16u * 2u * sizeof(uint64_t));
			const auto* results = reinterpret_cast<const volatile uint64_t*>(address);
			for (uint32_t db = 0; db < 16u; db++) {
				const auto begin = results[db * 2u];
				const auto end   = results[db * 2u + 1u];
				if ((begin & end & ready_bit) == 0) {
					if (wait_op == 0) {
						SuspendPm4(Common::GpuTiming::StallKind::Predication);
					} else {
						m_predicate_skip = false;
					}
					return;
				}
				value += end - begin;
			}
		} break;
		case 0x03:
			// The wait selector applies only to Z-pass query readiness.
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			SettleMemoryRead(address, sizeof(uint64_t));
			value = *reinterpret_cast<const volatile uint64_t*>(address);
			break;
		default: EXIT("unknown predication op: 0x%08" PRIx32 "\n", op);
	}
	switch (condition) {
		case 0x00: m_predicate_skip = (value != 0); break;
		case 0x01: m_predicate_skip = (value == 0); break;
		default: EXIT("unknown predication condition: 0x%08" PRIx32 "\n", condition);
	}
}

// Submits the recorded work every N draws so the host GPU starts before a long buffer ends.
void CommandProcessor::CountDraw() {
	if (m_flush_draws == 0) {
		return;
	}
	if (m_draws == m_flush_draws) {
		GetScheduler().Flush();
		m_draws = 0;
	}
	m_draws++;
}

void CommandProcessor::DrawIndex(DrawIndexArgs args) {
	CountDraw();
	if (args.offset_source == DrawOffsetSource::DrawState) {
		SettleIndirectDrawState(true, args.instance_count == 0);
	}
	args.index_type_and_size = m_index_type_and_size;
	if (args.instance_count == 0) {
		args.instance_count = NumInstances();
	}
	m_renderer.GetRenderExecutor().DrawIndex(CurrentBuffer(), args);
}

const void* CommandProcessor::IndexAddress(uint32_t first_index) const {
	uint64_t index_size = 0;
	switch (m_index_type_and_size) {
		case 0: index_size = 2; break;
		case 1: index_size = 4; break;
		case 2: index_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
	}
	return reinterpret_cast<const void*>(m_index_base_addr + first_index * index_size);
}

void CommandProcessor::DrawIndexOffset(uint32_t index_offset, uint32_t index_count) {
	DrawIndex({.index_count = index_count, .index_addr = IndexAddress(index_offset)});
}

static void PatchIndirectDrawOffsets(CommandProcessor& cp, IndirectDrawRegisters registers,
                                     uint32_t vertex_offset, uint32_t instance_offset,
                                     uint32_t first_index, uint32_t draw_index) {
	const auto write = [&](uint32_t location, uint32_t value) {
		if (location == Pm4::SH_NOP) {
			return;
		}
		EXIT_NOT_IMPLEMENTED(location >= Pm4::SH_NUM ||
		                     g_hw_sh_indirect_func[location] == nullptr);
		g_hw_sh_indirect_func[location](cp, location, value);
	};
	write(registers.vertex_offset, vertex_offset);
	write(registers.instance_offset, instance_offset);
	write(registers.index_offset, first_index);
	write(registers.draw_index, draw_index);
}

// Writes the GS user-data slots the packet patches; false when the shaders cannot read the
// offsets from GPU records: a first-index or subdraw-index destination, or another stage's
// user data.
static bool IndirectDrawOffsets(IndirectDrawRegisters registers, ShaderIndirectOffsets& offsets) {
	if (registers.index_offset != Pm4::SH_NOP || registers.draw_index != Pm4::SH_NOP) {
		return false;
	}
	for (const auto [location, slot]: {std::pair {registers.vertex_offset, &offsets.vertex},
	                                   std::pair {registers.instance_offset, &offsets.instance}}) {
		if (location == Pm4::SH_NOP) {
			continue;
		}
		const uint32_t index = location - Pm4::SPI_SHADER_USER_DATA_GS_0;
		if (index >= HW::UserSgprInfo::SGPRS_MAX) {
			return false;
		}
		*slot = static_cast<uint8_t>(index);
	}
	return true;
}

void CommandProcessor::DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
                                         const volatile uint32_t* count_addr,
                                         uint32_t stride_in_bytes, IndirectDrawRegisters registers,
                                         uint32_t draw_initiator, bool indexed) {
	CountDraw();
	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != (indexed ? 0u : 2u));
	EXIT_NOT_IMPLEMENTED(m_draw_indirect_args_base_addr == 0);
	// The guest records match Vulkan's indirect commands byte for byte.
	EXIT_NOT_IMPLEMENTED((stride_in_bytes & 3u) != 0 ||
	                     stride_in_bytes < (indexed ? sizeof(vk::DrawIndexedIndirectCommand)
	                                                : sizeof(vk::DrawIndirectCommand)));

	if (count_addr != nullptr) {
		SettleLabelAccess(count_addr, sizeof(uint32_t));
	}
	if (DrawIndirectOnGpu(data_offset, max_count_or_count, count_addr, stride_in_bytes, registers,
	                      indexed)) {
		// The deferred state holds the last instance count.
		m_num_instances_addr = 0;
		return;
	}
	// The packet replaces the deferred state only when it draws, and its shader may read any
	// register. Its instance count replaces the deferred one, which is read only when the packet
	// may draw nothing (a GPU-written instance count is then not waited for).
	SettleIndirectDrawState(true, count_addr != nullptr || max_count_or_count == 0);

	uint32_t draw_count = max_count_or_count;
	if (count_addr != nullptr) {
		SettleMemoryRead(count_addr, sizeof(uint32_t));
		draw_count = std::min(uint32_t {*count_addr}, max_count_or_count);
	}
	if (draw_count == 0) {
		return;
	}
	std::erase_if(m_deferred_indirect, [](const DeferredIndirectValue& value) {
		return value.field == DeferredIndirectValue::Field::InstanceCount;
	});
	Common::DebugCounters::Add(Common::DebugCounters::Counter::IndirectDraws, draw_count);

	auto&      executor  = m_renderer.GetRenderExecutor();
	const auto args_addr = m_draw_indirect_args_base_addr + data_offset;
	// NUM_INSTANCES keeps the last draw's instance count.
	m_num_instances_addr = args_addr + static_cast<uint64_t>(draw_count - 1) * stride_in_bytes +
	                       offsetof(vk::DrawIndirectCommand, instanceCount);
	// The GPU reads the records if the shaders allow; the VS prolog removes Vulkan's bases
	// from the index VGPRs. A packet that patches user SGPRs was declined above because the
	// host must read its arguments (a resource walk or a fetch prolog reads the SGPRs), so only
	// packets without patched SGPRs take this path.
	DrawIndirectArgs records {args_addr, draw_count, stride_in_bytes};
	if (IndirectDrawOffsets(registers, records.offsets) &&
	    records.offsets == ShaderIndirectOffsets {}) {
		if (!indexed && executor.DrawAuto(CurrentBuffer(), {.indirect = records})) {
			return;
		}
		if (indexed && m_index_buffer_size != 0 &&
		    executor.DrawIndex(CurrentBuffer(), {.index_count         = m_index_buffer_size,
		                                         .index_addr          = IndexAddress(0),
		                                         .index_type_and_size = m_index_type_and_size,
		                                         .indirect            = records})) {
			return;
		}
	}

	SettleMemoryRead(
	    reinterpret_cast<const void*>(args_addr),
	    static_cast<uint64_t>(draw_count - 1u) * stride_in_bytes +
	        (indexed ? sizeof(vk::DrawIndexedIndirectCommand) : sizeof(vk::DrawIndirectCommand)));
	const uint32_t max_indices = m_index_buffer_size != 0 ? m_index_buffer_size : UINT32_MAX;
	for (uint32_t i = 0; i < draw_count; i++) {
		const auto addr = args_addr + static_cast<uint64_t>(i) * stride_in_bytes;
		if (!indexed) {
			const auto& args = *reinterpret_cast<const vk::DrawIndirectCommand*>(addr);
			PatchIndirectDrawOffsets(*this, registers, args.firstVertex, args.firstInstance, 0, i);
			executor.DrawAuto(CurrentBuffer(), {.vertex_count   = args.vertexCount,
			                                    .instance_count = args.instanceCount,
			                                    .first_vertex   = args.firstVertex,
			                                    .first_instance = args.firstInstance,
			                                    .offset_source  = DrawOffsetSource::IndirectArgs});
			continue;
		}

		const auto& args = *reinterpret_cast<const vk::DrawIndexedIndirectCommand*>(addr);
		PatchIndirectDrawOffsets(*this, registers, static_cast<uint32_t>(args.vertexOffset),
		                         args.firstInstance, args.firstIndex, i);
		executor.DrawIndex(CurrentBuffer(), {.index_count = std::min(args.indexCount, max_indices),
		                                     .index_addr  = IndexAddress(args.firstIndex),
		                                     .instance_count      = args.instanceCount,
		                                     .index_type_and_size = m_index_type_and_size,
		                                     .base_vertex         = args.vertexOffset,
		                                     .first_instance      = args.firstInstance,
		                                     .offset_source = DrawOffsetSource::IndirectArgs});
	}
}

bool CommandProcessor::DrawIndirectOnGpu(uint32_t data_offset, uint32_t max_count_or_count,
                                         const volatile uint32_t* count_addr,
                                         uint32_t stride_in_bytes, IndirectDrawRegisters registers,
                                         bool indexed) {
	// The offsets go to vertex user SGPRs, which the vertex shader then reads from its draw.
	const auto slot = [](uint32_t location) -> std::optional<int32_t> {
		if (location == Pm4::SH_NOP) {
			return -1;
		}
		if (location >= Pm4::SPI_SHADER_USER_DATA_GS_0 &&
		    location < Pm4::SPI_SHADER_USER_DATA_GS_0 + 32u) {
			return static_cast<int32_t>(location - Pm4::SPI_SHADER_USER_DATA_GS_0);
		}
		return std::nullopt;
	};
	const auto vertex_slot   = slot(registers.vertex_offset);
	const auto instance_slot = slot(registers.instance_offset);
	// A first-index or subdraw-index destination needs a value per record on the host.
	if (registers.index_offset != Pm4::SH_NOP || registers.draw_index != Pm4::SH_NOP ||
	    !vertex_slot || !instance_slot || max_count_or_count == 0) {
		return false;
	}
	// User SGPRs earlier packets left on the GPU and this one does not write: the renderer asks
	// for them only when the draw's vertex stage reads them.
	uint32_t stale_slots = 0;
	for (const auto& value: m_deferred_indirect) {
		if (value.field != DeferredIndirectValue::Field::InstanceCount &&
		    value.location != registers.vertex_offset &&
		    value.location != registers.instance_offset) {
			stale_slots |= 1u << (value.location - Pm4::SPI_SHADER_USER_DATA_GS_0);
		}
	}
	IndirectDrawSource args {
	    .args_address         = m_draw_indirect_args_base_addr + data_offset,
	    .stride               = stride_in_bytes,
	    .max_draw_count       = max_count_or_count,
	    .count_address        = reinterpret_cast<uint64_t>(count_addr),
	    .indexed              = indexed,
	    .index_base           = m_index_base_addr,
	    .index_buffer_size    = m_index_buffer_size,
	    .index_type_and_size  = m_index_type_and_size,
	    .vertex_offset_slot   = *vertex_slot,
	    .instance_offset_slot = *instance_slot,
	    .stale_slots          = stale_slots,
	};
	// A packet that draws nothing is done as the GPU would have done it: the state it leaves for
	// later packets (below) is the same.
	bool nothing =
	    IndirectDrawsNothing(args.args_address, max_count_or_count, count_addr, stride_in_bytes);
	if (nothing) {
		g_empty_draws.Add();
	}
	// The guess for a draw whose record the GPU holds (off by default, see EmptyDrawPredictor).
	if (g_predict_empty_draws.On() && count_addr == nullptr && max_count_or_count == 1) {
		if (m_empty_draws == nullptr) {
			m_empty_draws = std::make_shared<EmptyDrawPredictor>(m_renderer);
		}
		if (m_empty_draws->Usable()) {
			const auto              id = m_empty_draws->Identify(IndirectDrawKey(indexed));
			std::array<uint32_t, 2> counts {};
			if (nothing) {
				m_empty_draws->Known(id, true);
			} else if (TryReadSettled(args.args_address, counts.data(), sizeof(counts))) {
				m_empty_draws->Known(id, false);
			} else {
				nothing = m_empty_draws->Predict(id);
				m_empty_draws->Check(CurrentBuffer(), id, args.args_address, nothing);
				if (nothing) {
					g_predicted_draws.Add();
				}
			}
		}
	}
	auto& executor = m_renderer.GetRenderExecutor();
	auto  result =
	    nothing ? DrawIndirectResult::Done : executor.DrawIndirect(CurrentBuffer(), args);
	if (result == DrawIndirectResult::SettleRegisters) {
		SettleIndirectDrawState(true, false);
		args.stale_slots = 0;
		result           = executor.DrawIndirect(CurrentBuffer(), args);
	}
	if (result == DrawIndirectResult::ReadArguments) {
		return false;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::IndirectDraws);
	using Field = DeferredIndirectValue::Field;
	std::erase_if(m_deferred_indirect, [&](const DeferredIndirectValue& value) {
		return value.field == Field::InstanceCount || value.location == registers.vertex_offset ||
		       value.location == registers.instance_offset;
	});
	const auto defer = [&](uint32_t location, Field field) {
		m_deferred_indirect.push_back({.location     = location,
		                               .field        = field,
		                               .indexed      = indexed,
		                               .args_address = args.args_address,
		                               .stride       = stride_in_bytes,
		                               .max_count    = max_count_or_count,
		                               .count_addr   = count_addr});
	};
	if (registers.vertex_offset != Pm4::SH_NOP) {
		defer(registers.vertex_offset, Field::BaseVertex);
	}
	if (registers.instance_offset != Pm4::SH_NOP) {
		defer(registers.instance_offset, Field::FirstInstance);
	}
	defer(Pm4::SH_NOP, Field::InstanceCount);
	return true;
}

void CommandProcessor::SettleIndirectDrawState(bool registers, bool num_instances) {
	using Field = DeferredIndirectValue::Field;
	if (m_deferred_indirect.empty()) {
		return;
	}
	std::vector<DeferredIndirectValue> settled;
	std::erase_if(m_deferred_indirect, [&](const DeferredIndirectValue& value) {
		const bool take = value.field == Field::InstanceCount ? num_instances : registers;
		if (take) {
			settled.push_back(value);
		}
		return take;
	});
	for (const auto& value: settled) {
		// The draws read their count and arguments in order; the last one drawn set the state,
		// and a packet that drew nothing left it as it was.
		uint32_t count = value.max_count;
		if (value.count_addr != nullptr) {
			SettleMemoryRead(value.count_addr, sizeof(uint32_t));
			const uint32_t selected = *value.count_addr;
			count                   = std::min(selected, value.max_count);
		}
		if (count == 0) {
			continue;
		}
		const auto args_addr =
		    value.args_address + static_cast<uint64_t>(count - 1u) * value.stride;
		uint32_t word = 0;
		switch (value.field) {
			case Field::BaseVertex:
				word = value.indexed ? offsetof(vk::DrawIndexedIndirectCommand, vertexOffset) / 4u
				                     : offsetof(vk::DrawIndirectCommand, firstVertex) / 4u;
				break;
			case Field::FirstInstance:
				word = value.indexed
				           ? offsetof(vk::DrawIndexedIndirectCommand, firstInstance) / 4u
				           : offsetof(vk::DrawIndirectCommand, firstInstance) / 4u;
				break;
			case Field::InstanceCount:
				word = value.indexed ? offsetof(vk::DrawIndexedIndirectCommand, instanceCount) / 4u
				                     : offsetof(vk::DrawIndirectCommand, instanceCount) / 4u;
				break;
		}
		const auto* argument_addr = reinterpret_cast<const uint32_t*>(args_addr) + word;
		SettleMemoryRead(argument_addr, sizeof(uint32_t));
		const auto argument = *argument_addr;
		if (value.field == Field::InstanceCount) {
			m_num_instances      = argument;
			m_num_instances_addr = 0;
		} else {
			PatchIndirectDrawOffsets(*this, {value.location, Pm4::SH_NOP, Pm4::SH_NOP, Pm4::SH_NOP},
			                         argument, 0, 0, 0);
		}
	}
}

void CommandProcessor::DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y,
                                      uint32_t thread_group_z, uint32_t mode) {
	m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	m_renderer.GetRenderExecutor().DispatchDirect(CurrentBuffer(), thread_group_x, thread_group_y,
	                                              thread_group_z, mode, IsAsyncComputeQueue());
}

void CommandProcessor::ProcessPssr(const Pssr::Request& request) {
	EXIT_NOT_IMPLEMENTED(IsAsyncComputeQueue());
	m_renderer.GetRenderExecutor().ProcessPssr(CurrentBuffer(), request);
}

void CommandProcessor::DispatchIndirect(uint64_t args_addr, uint32_t mode) {
	EXIT_NOT_IMPLEMENTED(args_addr == 0 || (args_addr & 3u) != 0);
	if ((mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0) {
		// Thread dimensions are converted to workgroups on the host, so the arguments are read
		// here, after the work recorded before this packet.
		// Settled, the backing holds them; the guest mapping of the page may still fault on
		// other GPU-written bytes.
		SettleMemoryRead(reinterpret_cast<const void*>(args_addr),
		                 sizeof(vk::DispatchIndirectCommand));
		vk::DispatchIndirectCommand args {};
		if (!LibKernel::Memory::TryReadBacking(args_addr, &args, sizeof(args))) {
			std::memcpy(&args, reinterpret_cast<const void*>(args_addr), sizeof(args));
		}
		DispatchDirect(args.x, args.y, args.z, mode);
		return;
	}
	m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	// A dispatch of no workgroups runs nothing (Saros: 180 of 420 indirect dispatches a frame at
	// one spot): left out when its arguments say so without a wait, as an empty draw is. Here
	// too the arguments are mostly with the GPU still (one dispatch a frame is found).
	if (g_skip_empty_indirect.On()) {
		vk::DispatchIndirectCommand args {};
		if (TryReadSettled(args_addr, &args, sizeof(args)) &&
		    (args.x == 0 || args.y == 0 || args.z == 0)) {
			g_empty_dispatches.Add();
			return;
		}
	}
	m_renderer.GetRenderExecutor().DispatchIndirect(CurrentBuffer(), args_addr, mode,
	                                                IsAsyncComputeQueue());
}

void CommandProcessor::DrawIndexAuto(DrawAutoArgs args) {
	CountDraw();
	if (args.offset_source == DrawOffsetSource::DrawState) {
		SettleIndirectDrawState(true, args.instance_count == 0);
	}
	if (args.instance_count == 0) {
		args.instance_count = NumInstances();
	}
	m_renderer.GetRenderExecutor().DrawAuto(CurrentBuffer(), args);
}

void CommandProcessor::WaitFlipDone(const uint64_t* label) {
	BufferFlush();

	m_renderer.GetVideoOut().WaitFlipDone(label);
}

void CommandProcessor::ReleaseMem(uint32_t data_sel, uint32_t interrupt_selector, void* address,
                                  uint64_t data, uint32_t interrupt_context_id,
                                  bool gl2_writeback) {
	if (graphics_debug_dump_enabled()) {
		LOGF("CommandProcessor::ReleaseMem()\n"
		     "\t data_sel           = %" PRIu32 "\n"
		     "\t interrupt_selector = %" PRIu32 "\n"
		     "\t address            = 0x%016" PRIx64 "\n"
		     "\t data               = 0x%016" PRIx64 "\n"
		     "\t interrupt_context  = 0x%08" PRIx32 "\n",
		     data_sel, interrupt_selector, reinterpret_cast<uint64_t>(address), data,
		     interrupt_context_id);
	}

	// QueuedInterrupt: kNone, kOnly, kAfterWriteConfirm, kDataAfterWriteConfirm, kSendIntCtx.
	EXIT_NOT_IMPLEMENTED(interrupt_selector > 4);
	const bool interrupt = (interrupt_selector != 0 && interrupt_selector != 3);
	// kSendIntCtx has no label, and kOnly writes none on graphics queues.
	if (address == nullptr || interrupt_selector == 4 ||
	    (interrupt_selector == 1 && !IsAsyncComputeQueue())) {
		data_sel = 0;
	}
	// A GL2 writeback makes GPU writes CPU-visible even without a signal.
	if (gl2_writeback) {
		WriteBackImages();
	}
	if (data_sel == 0 && !interrupt) {
		return;
	}

	// Dumped occlusion counters are written before any end-of-pipe signal reaches the guest.
	m_renderer.GetOcclusionQueries().Publish();

	// EndOfPipeData: kNone, kImmediate32b, kImmediate64b, kRefClk; 5 reads GDS. A label lands
	// when its submission completes, then the interrupt follows it.
	EXIT_IF(data_sel >= 1 && data_sel <= 3 && CurrentBuffer().IsInvalid());
	switch (data_sel) {
		case 0: break;
		case 1:
			WriteLabelAtEndOfPipe(static_cast<uint32_t*>(address), static_cast<uint32_t>(data),
			                      interrupt, interrupt_context_id);
			return;
		case 2:
		case 3:
			WriteLabelAtEndOfPipe(static_cast<uint64_t*>(address),
			                      data_sel == 3 ? Sync::ReadReferenceClock() : data, interrupt,
			                      interrupt_context_id, data_sel == 3);
			return;
		case 5: {
			SynchronizeGpu();
			const auto gds = static_cast<uint32_t>(data);
			Sync::ReadGds(*m_renderer.GetBufferCache().GetGdsBuffer(),
			              static_cast<uint32_t*>(address), gds & 0xffffu, gds >> 16u);
			break;
		}
		default: EXIT("unknown release_mem data selector: %" PRIu32 "\n", data_sel);
	}

	// GuestGpu::Process's end-of-slice flush submits the work this interrupt waits on.
	if (interrupt) {
		TriggerEopEventAtEndOfPipe(interrupt_context_id);
	}
}

void CommandProcessor::WriteBackImages() {
	// GPU-written images are copied back without holding any signal: a held signal exposes the host
	// GPU's speed. Guest accesses to them wait for the copies.
	if (m_renderer.GetTextureCache().WriteBackImages()) {
		GetScheduler().Flush();
	}
}

void CommandProcessor::EmitGlobalBarrier() {
	Common::LockGuard lock(m_renderer.GetMutex());
	CurrentBuffer().GuestFlush();
}

// The reference clock when the GPU finished submission `tick`: its end-of-pipe timestamp, or the
// time the packet was processed when that submission ended before it (nothing was recorded in
// between). Without a GPU timestamp the completion bounds it.
static uint64_t EndOfPipeTimestamp(CommandScheduler& scheduler, uint64_t tick, int64_t issued_ns,
                                   uint64_t issued_clock) {
	const auto end_ns = scheduler.GetTimestamps().CompletedEndNs(tick);
	const auto at_ns  = end_ns ? std::max(*end_ns, issued_ns)
	                           : std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count();
	return Sync::MonotonicReferenceClock(std::max(Sync::ReferenceClockAt(at_ns), issued_clock));
}

bool CommandProcessor::SubmitForLabel(bool with_interrupt, bool timestamp) {
	// Some games end almost every command with a label (Saros: ~900 per frame, many after a
	// single dispatch), and a submission per label costs more than the work it carries. While
	// the GPU still runs earlier submissions, a label can ride with the commands recorded after
	// it: it lands when that larger submission completes, and the GPU has queued work meanwhile.
	// The submission follows at a later label once enough work is recorded or the GPU caught
	// up, and at the end of the queue slice at the latest (GuestGpu::Process). Interrupts and
	// timestamps are submitted at once: their time is what the guest observes.
	auto& scheduler = GetScheduler();
	if (with_interrupt || timestamp || !CurrentBuffer().HasCommands() ||
	    CurrentBuffer().CommandCount() >= LabelBatchCommands) {
		return true;
	}
	return scheduler.IsFree(scheduler.CurrentTick() - 1);
}

template <typename T>
void CommandProcessor::WriteLabelAtEndOfPipe(T* dst, T value, bool with_interrupt,
                                             uint32_t interrupt_context_id, bool timestamp) {
	// An end-of-pipe label means every command before it finished. Writing it when the
	// submission completes makes the guest's wait cover the GPU, so the results it reads next
	// are already published instead of faulting into a synchronous readback of unsubmitted
	// work. With nothing recorded since the last submission the label rides on that submission,
	// and the flush has nothing to submit.
	auto&      scheduler = GetScheduler();
	const auto issued_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
	                           std::chrono::steady_clock::now().time_since_epoch())
	                           .count();
	if (SubmitForLabel(with_interrupt, timestamp)) {
		const Common::GpuTiming::SyncScope sync_reason(Common::GpuTiming::SyncReason::Label);
		scheduler.Flush();
	}
	// Pending until stored, so a WAIT_REG_MEM on it can continue behind a barrier.
	auto&      labels = m_renderer.GetPendingLabels();
	const auto label_id = labels.Add(reinterpret_cast<uint64_t>(dst), sizeof(T),
	                                 static_cast<uint64_t>(value), !timestamp);
	// Queued after the flush: the submission queued the publication of its asynchronous
	// downloads, which then runs before the label on the completion thread.
	const auto label_tick = scheduler.DeferPriorityOperationAfterRecorded(
	    [&renderer = m_renderer, &scheduler, dst, value, with_interrupt, issued_ns, label_id,
	     event_id  = m_interrupt_event_id, interrupt_context_id, timestamp] {
		    const auto tick   = scheduler.ActivePriorityTick();
		    auto       stored = static_cast<uint64_t>(value);
		    if (timestamp) {
			    stored = EndOfPipeTimestamp(scheduler, tick, issued_ns, stored);
			    renderer.GetPendingLabels().SetValue(label_id, stored);
		    }
		    renderer.DeliverLabel(
		        dst, stored, sizeof(T),
		        [&renderer, &scheduler, tick, issued_ns, with_interrupt, event_id,
		         interrupt_context_id] {
			        scheduler.GetTimestamps().RecordLabel(tick, issued_ns);
			        if (with_interrupt) {
				        renderer.TriggerInterrupt(event_id, interrupt_context_id);
			        }
		        },
		        label_id);
	    },
	    GuestWrites {reinterpret_cast<uint64_t>(dst), sizeof(T)});
	if (label_tick == 0) {
		labels.Stored(label_id); // Shutdown: already stored.
	} else {
		labels.SetTick(label_id, label_tick);
	}
}

void CommandProcessor::TriggerEopEventAtEndOfPipe(uint32_t interrupt_context_id) {
	EXIT_IF(CurrentBuffer().IsInvalid());
	GetScheduler().DeferPriorityOperationAfterRecorded(
	    [&renderer = m_renderer, event_id = m_interrupt_event_id, interrupt_context_id] {
		    renderer.TriggerInterrupt(event_id, interrupt_context_id);
	    },
	    GuestWrites::None());
}

void CommandProcessor::TriggerEvent(uint32_t event_type, uint32_t event_index,
                                    uint64_t event_address) {
	if (graphics_debug_dump_enabled()) {
		LOGF("CommandProcessor::TriggerEvent()\n"
		     "\t event_type  = 0x%08" PRIx32 "\n"
		     "\t event_index = 0x%08" PRIx32 "\n"
		     "\t address     = 0x%016" PRIx64 "\n",
		     event_type, event_index, event_address);
	}

	const auto valid_cache_event_index = event_index == 0x00000000 || event_index == 0x00000007;
	// The command buffer records partial flushes and cache writebacks as guest flushes
	// (CommandBuffer::GuestFlush): its own barriers leave some work unordered until one.
	switch (event_type) {
		// CsPartialFlush, GsPartialFlush, PsPartialFlush.
		case 0x00000007:
		case 0x0000000f:
		case 0x00000010: EmitGlobalBarrier(); break;
		// CbDbDataWritebackInvalidate, CbDataWritebackInvalidate, DbDataWritebackInvalidate,
		// DbMetadataWritebackInvalidate, CbMetadataWritebackInvalidate.
		case 0x00000016:
		case 0x00000031:
		case 0x0000002a:
		case 0x0000002c:
		case 0x0000002e:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			EmitGlobalBarrier();
			break;
		case 0x0000000d:
		case 0x0000000e:
		case 0x00000012:
		case 0x00000017:
		case 0x00000018:
		case 0x00000019:
		case 0x0000001a:
		case 0x0000001b:
		case 0x00000038:
		case 0x0000003a:
			LOGF("\t temporary: ignoring unsupported event_write type 0x%08" PRIx32
			     ", index 0x%08" PRIx32 "\n",
			     event_type, event_index);
			break;
		case 0x00000039: {
			if (event_index != 0x00000001 || event_address == 0 || (event_address & 0x7u) != 0) {
				EXIT("invalid occlusion-counter dump: index=0x%08" PRIx32 ", address=0x%016" PRIx64
				     "\n",
				     event_index, event_address);
			}
			if (auto& queries = m_renderer.GetOcclusionQueries(); queries.Enabled()) {
				queries.Dump(event_address);
				break;
			}
			static std::once_flag warning_once;
			std::call_once(warning_once, [] {
				std::printf("Warning: game uses occlusion queries, which are treated as always "
				            "visible without --occlusion-queries; GPU usage may be higher and FPS "
				            "may be lower.\n");
			});

			// Without host occlusion queries, publish an always-visible result. The PS5 layout
			// contains one interleaved begin/end pair per DB, and bit 63 marks a result ready.
			constexpr uint64_t ready_bit    = 1ull << 63u;
			constexpr uint64_t counter_mask = ready_bit - 1u;
			auto*              results      = reinterpret_cast<volatile uint64_t*>(event_address);
			const auto         value        = ready_bit | m_synthetic_occlusion_counter;
			for (uint32_t db = 0; db < 16u; db++) {
				results[db * 2u] = value;
			}
			m_synthetic_occlusion_counter = (m_synthetic_occlusion_counter + 1u) & counter_mask;
			break;
		}
		default:
			EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type, event_index);
	}
}

void CommandProcessor::Flip(int handle, int index, int flip_mode, int64_t flip_arg) {
	if (graphics_debug_dump_enabled()) {
		LOGF("CommandProcessor::Flip()\n");
	}
	m_renderer.GetOcclusionQueries().Publish();

	auto& command = CurrentBuffer();
	auto  request = Sync::PrepareVideoOutFlip(command, handle, index, flip_mode, flip_arg);
	GetScheduler().DeferPriorityOperation(
	    [&renderer = m_renderer, request] {
		    Common::MemoryTrace::Note(Common::MemoryTrace::Kind::Flip, 0, 0,
		                              renderer.GetCommandScheduler().ActivePriorityTick());
		    renderer.GetVideoOut().CompleteFlip(request);
	    },
	    GuestWrites::None());
	const Common::GpuTiming::SyncScope sync_reason(Common::GpuTiming::SyncReason::Flip);
	GetScheduler().Flush();
}

void CommandProcessor::PrepareCpuFlip(uint64_t request_id) {
	auto& command = CurrentBuffer();
	m_renderer.GetVideoOut().PrepareFlip(request_id, command);
	GetScheduler().DeferPriorityOperation(
	    [this, request_id] { m_renderer.GetVideoOut().CompleteFlip(request_id); },
	    GuestWrites::None());
	const Common::GpuTiming::SyncScope sync_reason(Common::GpuTiming::SyncReason::Flip);
	GetScheduler().Flush();
}

void CommandProcessor::SynchronizeGpu() {
	const Common::GpuTiming::SyncScope sync_reason(Common::GpuTiming::SyncReason::GuestWait);
	GetScheduler().Finish();
}

bool GuestGpu::IsGpuThread() noexcept {
	return g_gpu_state != nullptr;
}

} // namespace Libs::Graphics
