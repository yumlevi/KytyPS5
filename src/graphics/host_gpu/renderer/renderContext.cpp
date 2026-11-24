#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/memoryTrace.h"
#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/gpuTiming.h"
#include "common/liveControl.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/presentation/videoOut.h"
#include "kernel/memory.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstring>
#include <thread>

namespace Libs::Graphics {

namespace {

// 2: the session builds pipelines beside the game unless the card is short of video memory
// (RenderContext::RenderContext); 1: on any card; 0: on none.
Common::LiveControl::Switch g_session_precompile("session_precompile", 2);
// A video memory budget under this counts as short: cards of 12 GB and less (the budget of a
// card of 16 GB is about 15 GiB).
constexpr uint64_t SmallVideoMemory = 13ull << 30u;

// 1: a guest unmap waits only for the completions that write its range (UnmapMemory); 0: for
// everything the GPU still has to run, whenever a cache knows the memory.
Common::LiveControl::Switch g_unmap_precise("unmap_precise", 1);

} // namespace

RenderContext::RenderContext(GraphicContext& graphics)
    : m_graphics(graphics), m_render_executor(*this), m_command_scheduler(*this, graphics),
      m_upload_scatter(graphics, m_command_scheduler),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_pipeline_cache(graphics, &m_image_heap, &m_command_scheduler), m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache),
      m_image_heap(*this, m_page_manager), m_bindless_table(graphics, m_command_scheduler),
      m_occlusion_queries(graphics, m_command_scheduler) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	m_command_scheduler.GetTimestamps().SetPrimary(true);
	if (graphics.ray_query_enabled) {
		m_ray_tracing = std::make_unique<RayTracing>(*this);
	}
	// After the bindless table: precompiled layouts reference its set layout.
	// The builds beside the game cost video memory: the pipelines kept for their first use, and
	// what the driver keeps of every pipeline created through its cache (1.6 GB less without
	// them at a heavy spot of Saros), and giving them up later does not give the memory back.
	// A card whose budget is under SmallVideoMemory leaves them out: what a precompile pass
	// built comes from the driver cache at its first use (a few milliseconds), anything else
	// is compiled then. session_precompile: 1 builds beside the game on any card, 0 on none.
	bool beside_the_game = Config::PrecompilePipelinesEnabled() && !Config::PrecompileOnlyEnabled();
	if (beside_the_game && g_session_precompile.Get() != 1) {
		const auto budget = graphics.SampleMemoryHeaps().device.budget;
		if (g_session_precompile.Get() == 0 || (budget != 0 && budget < SmallVideoMemory)) {
			beside_the_game = false;
			LOGF("Precompile beside the game: left out (%" PRIu64 " MiB of video memory budget; "
			     "it keeps about 1.6 GB of video memory). Pipelines come from the driver cache "
			     "at their first use: run the precompile pass before play\n",
			     budget >> 20u);
		}
	}
	if (beside_the_game) {
		// While the session boots, about half the host threads (at low priority) build what
		// earlier sessions or the list recorded, which the game is about to need; then as many
		// as the background pipeline builds use: driver compilers are multithreaded themselves,
		// and the guest needs the CPU.
		const auto host = std::thread::hardware_concurrency();
		m_pipeline_cache.StartPrecompile(std::clamp(host / 2u, 2u, 8u),
		                                 std::clamp(host / 8u, 1u, 2u));
	}
	if (!Config::PrecompileOnlyEnabled()) {
		// (Unless asked for: a quarter of the host's threads. They run at low priority, and with
		// the shaders of the game's files there are thousands of programs to translate.)
		const auto asked = Config::PrepareThreads();
		m_pipeline_cache.StartProgramPreparation(
		    asked == UINT32_MAX ? std::clamp(std::thread::hardware_concurrency() / 4u, 2u, 8u)
		                        : std::min(asked, 8u));
	}
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
	// Before the bindless table destroys the set layout that background builds reference.
	m_pipeline_cache.StopBackgroundCompiles();
}

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
	m_notified_gpu.store(m_gpu.get());
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		// The completion thread may still store labels; wait until none uses the GPU object.
		m_notified_gpu.store(nullptr);
		while (m_gpu_notifiers.load() != 0) {
			std::this_thread::yield();
		}
		m_gpu.reset();
	}
	if (m_video_out != nullptr) {
		if (m_command_scheduler.Active()) {
			m_command_scheduler.Finish();
		}
		m_command_scheduler.DrainPriorityOperations();
		m_video_out = nullptr;
	}
}

void RenderContext::NotifyGpuMemoryWritten() {
	m_gpu_notifiers.fetch_add(1);
	if (auto* gpu = m_notified_gpu.load(); gpu != nullptr) {
		gpu->NotifyMemoryWritten();
	}
	m_gpu_notifiers.fetch_sub(1);
}

GuestGpu& RenderContext::GetGpu() const {
	EXIT_IF(m_gpu == nullptr);
	return *m_gpu;
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
}

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		return false;
	}
	// The access retries once the image write-backs over its page land.
	if (WaitWriteBacks(fault_vaddr, fault_size)) {
		return true;
	}
	if (access == PageFaultAccess::Write) {
		// (Only the slow ones: a frame of Saros takes some 1,700 write faults.)
		const Common::LiveControl::GuestWait held("write-fault", fault_vaddr, 0, 500);
		// A store of the emulated command processor (on the GPU thread) is part of the current
		// work; anything else stores for the guest CPU.
		m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size, !GuestGpu::IsGpuThread(), true);
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_image_heap.InvalidateMemory(fault_vaddr, fault_size);
	} else {
		// A guest read of GPU-written memory tends to continue through the following pages, so
		// one readback fetches the GPU-modified extent from the faulting page instead of a GPU
		// round trip per page.
		constexpr uint64_t fault_window = uint64_t {1} << 20u;
		m_buffer_cache.ReadMemoryAround(fault_vaddr, fault_window);
	}
	return true;
}

bool RenderContext::WaitWriteBacks(uint64_t vaddr, uint64_t size) {
	const auto tick = m_texture_cache.WriteBackTick(vaddr, size);
	if (tick == 0) {
		return false;
	}
	const Common::LiveControl::GuestWait held("image-write-back", vaddr, size);
	// The GPU thread can fault inside a draw, so it submits without running deferred operations.
	if (GuestGpu::IsGpuThread()) {
		m_command_scheduler.Wait(tick);
	} else if (tick == m_command_scheduler.CurrentTick()) {
		GetGpu().SendCommandSync([this, tick] { m_command_scheduler.Wait(tick); });
	}
	m_command_scheduler.WaitPriorityOperations(tick);
	return true;
}

void RenderContext::DeliverLabel(void* dst, uint64_t value, uint32_t size,
                                 Common::UniqueFunction<void>&& notify, uint64_t label_id) {
	EXIT_IF(dst == nullptr || (size != sizeof(uint32_t) && size != sizeof(uint64_t)));
	Common::MemoryTrace::Note(Common::MemoryTrace::Kind::Label, reinterpret_cast<uint64_t>(dst), size,
	                          m_command_scheduler.ActivePriorityTick());
	// The completion thread runs the publications of a submission's downloads before its
	// labels, so whatever the label announces is already in guest memory. The store epoch is read
	// before the store: a wait of a submission of that epoch or earlier saw the store come later.
	const auto epoch = m_buffer_cache.CurrentBdaStoreEpoch();
	if (m_gpu_thread_labels.load(std::memory_order_acquire) == 0 &&
	    m_page_manager.TryWriteUnwatched(reinterpret_cast<uint64_t>(dst), &value, size)) {
		m_pending_labels.NoteStoredValue(reinterpret_cast<uint64_t>(dst), size, value, epoch);
		Common::GpuTiming::Sync().AddLabel(true);
		if (label_id != 0) {
			m_pending_labels.Stored(label_id);
		}
		NotifyGpuMemoryWritten();
		if (notify) {
			notify();
		}
		return;
	}
	Common::GpuTiming::Sync().AddLabel(false);
	m_gpu_thread_labels.fetch_add(1, std::memory_order_relaxed);
	GetGpu().SendCommand([this, dst, value, size, label_id, notify = std::move(notify)]() mutable {
		const auto vaddr = reinterpret_cast<uint64_t>(dst);
		const auto epoch = m_buffer_cache.CurrentBdaStoreEpoch();
		// A settled access may have stored the label, and later ones, already.
		const auto store = [&] { return label_id == 0 || m_pending_labels.ShouldStore(label_id); };
		// As upstream's RELEASE_MEM (2463e3aaf): a label beside GPU-written bytes goes to the
		// backing store, so those bytes stay with the GPU instead of being read back for a CPU
		// write to their page. GPU-written ranges are tracked by byte here, so "beside" means
		// on a page with GPU-written bytes, none of them under the label.
		const auto page   = vaddr & ~(TRACKER_PAGE_SIZE - 1u);
		const bool beside = GuestRange {vaddr, size}.Valid() &&
		                    (m_buffer_cache.IsRegionGpuModified(page, TRACKER_PAGE_SIZE) ||
		                     m_buffer_cache.HasGpuDirtyBytes(page, TRACKER_PAGE_SIZE)) &&
		                    !m_buffer_cache.HasGpuDirtyBytes(vaddr, size) &&
		                    (!store() || LibKernel::Memory::TryWriteBacking(vaddr, &value, size));
		if (!beside) {
			// What a write fault would do, without relying on one: GPU-written bytes under the
			// label are read back first (a later download would overwrite the label), then the
			// caches drop or read back what they hold of the label's pages and stop watching them.
			if (GuestRange {vaddr, size}.Valid() && m_buffer_cache.HasGpuDirtyBytes(vaddr, size)) {
				m_buffer_cache.ReadMemory(vaddr, size, true);
			}
			(void)InvalidateMemory(vaddr, size);
			if (store()) {
				if (size == sizeof(uint32_t)) {
					const auto word = static_cast<uint32_t>(value);
					std::memcpy(dst, &word, sizeof(word));
				} else {
					std::memcpy(dst, &value, sizeof(value));
				}
			}
		}
		if (store()) {
			m_pending_labels.NoteStoredValue(vaddr, size, value, epoch);
		}
		if (label_id != 0) {
			m_pending_labels.Stored(label_id);
		}
		NotifyGpuMemoryWritten();
		if (notify) {
			notify();
		}
		m_gpu_thread_labels.fetch_sub(1, std::memory_order_release);
	});
}

void RenderContext::DeliverZPassCounters(std::span<const std::pair<uint64_t, uint64_t>> dumps) {
	// The words WriteZPassCounters stores: a begin dump is one block of counters and cleared
	// ends, an end dump every other word of it.
	std::array<uint64_t, OCCLUSION_QUERY_DBS * 2u> block {};
	std::array<uint64_t, OCCLUSION_QUERY_DBS>      words {};
	for (uint32_t db = 0; db < OCCLUSION_QUERY_DBS; db++) {
		words[db] = block[db * 2u] = OCCLUSION_QUERY_READY;
	}
	bool handed = false;
	for (const auto& [address, samples]: dumps) {
		words[0] = block[0] = OCCLUSION_QUERY_READY | samples;
		const bool begin = (address & 8u) == 0;
		if (m_gpu_thread_labels.load(std::memory_order_acquire) == 0 &&
		    (begin ? m_page_manager.TryWriteUnwatched(address, block.data(), sizeof(block))
		           : m_page_manager.TryWriteUnwatchedWords(address, words.data(),
		                                                   OCCLUSION_QUERY_DBS,
		                                                   2u * sizeof(uint64_t)))) {
			continue;
		}
		// From here on the dumps, and the labels after them, follow each other on the GPU thread.
		m_gpu_thread_labels.fetch_add(1, std::memory_order_relaxed);
		std::lock_guard lock(m_zpass_mutex);
		m_zpass_pending.emplace_back(address, samples);
		handed = true;
	}
	if (handed) {
		GetGpu().SendCommand([this] { WritePendingZPassCounters(); });
	}
	NotifyGpuMemoryWritten();
}

void RenderContext::WritePendingZPassCounters() {
	std::vector<std::pair<uint64_t, uint64_t>> pending;
	{
		std::lock_guard lock(m_zpass_mutex);
		pending.swap(m_zpass_pending);
	}
	for (const auto& [address, samples]: pending) {
		// (A watched page takes the write fault here, as it did when this thread wrote them all.)
		WriteZPassCounters(reinterpret_cast<volatile uint64_t*>(address), samples,
		                   (address & 8u) == 0);
		m_gpu_thread_labels.fetch_sub(1, std::memory_order_release);
	}
}

void RenderContext::SettleElidedLabels(uint64_t vaddr, uint64_t size) {
	static thread_local bool settling = false;
	if (settling || !GuestGpu::IsGpuThread()) {
		return;
	}
	const auto tick = m_pending_labels.ElidedTick(vaddr, size);
	if (tick == 0) {
		return;
	}
	settling = true;
	{
		const Common::GpuTiming::SyncScope sync_reason(Common::GpuTiming::SyncReason::Label);
		m_command_scheduler.Wait(tick);
		// The completion thread stores the labels of that submission now ...
		m_command_scheduler.WaitPriorityOperations(tick);
	}
	// ... unless it handed them to this thread, behind the current command. Those are stored
	// here, through the backing like the completion thread's own stores, and skipped there.
	m_pending_labels.StoreCompleted(vaddr, size, tick, [](const PendingLabels::Label& label) {
		(void)LibKernel::Memory::TryWriteBacking(label.address, &label.value, label.size);
	});
	settling = false;
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	// Host I/O into the range, even one running past its mapping, would fail on its inaccessible
	// pages, and a later write-back would overwrite it.
	(void)WaitWriteBacks(vaddr, size);
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	m_image_heap.InvalidateMemory(vaddr, size);
	return true;
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(m_mapped_ranges_mutex);
	m_mapped_ranges.Add(vaddr, size);
	m_buffer_cache.QueueBdaSync(vaddr, size);
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto unmap = [this, vaddr, size] {
		const Common::GpuTiming::SyncScope sync_reason(Common::GpuTiming::SyncReason::Unmap);
		// Check cache ownership on the GPU thread. Guest-memory callbacks can still
		// write a range with no cached data, so the ones that write it must finish before it is
		// unmapped.
		if (!m_command_scheduler.Active()) {
			// Nothing in flight.
		} else if (g_unmap_precise.On()) {
			// Only they have to: the publications of downloads, the labels and the query
			// counters that land in the range (each names the memory it writes). Whatever else
			// the GPU still runs needs no wait here: an image of the range is released when
			// the commands that use it are done (DeleteImage), a cached buffer outlives the
			// mapping, an entry of the image heap is reused once its tick ended (Retire), an
			// upload still to be copied from these pages is copied below (CaptureRange), and
			// GPU-written bytes of the range are read back by InvalidateMemory with waits of
			// their own. Waiting for the GPU to finish everything (Finish) at every unmap of
			// memory a cache knows stopped this thread for 4 to 7 ms on average, in half of
			// the ten-second windows of a session played through the game; the Demon's Souls
			// and the Ghost of Yotei forks came to wait for the same things only.
			if (const auto tick = m_command_scheduler.PendingPriorityWriteTick(vaddr, size);
			    tick != 0) {
				m_command_scheduler.Wait(tick);
				m_command_scheduler.WaitPriorityOperations(tick);
			}
		} else if (m_buffer_cache.IsRegionRegistered(vaddr, size) ||
		           m_texture_cache.IsRegionRegistered(vaddr, size) ||
		           m_command_scheduler.HasPendingPriorityWrites(vaddr, size)) {
			const auto tick = m_command_scheduler.CurrentTick();
			m_command_scheduler.Finish();
			m_command_scheduler.WaitPriorityOperations(tick);
		}
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		m_image_heap.UnmapMemory(vaddr, size);
		m_pending_labels.ForgetStored(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
	};
	// Shutdown still owns the GPU while queued rendering drains, but its command lane no
	// longer accepts external work. Use the guest GPU's state for the teardown route.
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	const Common::LiveControl::GuestWait held("unmap", vaddr, size);
	m_gpu->SendCommandSync(unmap);
}

uint64_t RenderContext::BeginGuestStoreEpoch() {
	return m_buffer_cache.BeginBdaStoreEpoch();
}

void RenderContext::NoteCommandOrderingPoint(uint64_t epoch) {
	m_buffer_cache.NoteBdaOrderingPoint(epoch);
}

void RenderContext::PrepareBda() {
	[[maybe_unused]] static const bool logged = [] {
		Log::WriteToConsoleAndLog("GPU: using buffer device address (BDA) shader memory access.\n");
		return true;
	}();
	std::shared_lock lock(m_mapped_ranges_mutex);
	// The uploads of every written range share their barriers.
	const auto batch = m_command_scheduler.Current().BatchCopies();
	m_buffer_cache.SynchronizeBda(m_mapped_ranges);
	m_fault_process_pending = true;
}

void RenderContext::RunGarbageCollector() {
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_graphics.MonitorMemoryBudget();
	CollectGpuProfile(*this);
	m_image_heap.RunGarbageCollector();
	m_texture_cache.ProcessDownloadImages();
	m_texture_cache.RunGarbageCollector();
	m_buffer_cache.RunGarbageCollector();
	if (m_ray_tracing) {
		m_ray_tracing->RunGarbageCollector();
	}
}

void RenderContext::AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it != m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.push_back({eq, event_id});
}

void RenderContext::DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it == m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.erase(it);
}

void RenderContext::TriggerInterrupt(int event_id, uint32_t context_id) {
	std::vector<InterruptEqRegistration> registrations;
	{
		Common::LockGuard lock(m_interrupt_mutex);
		for (const auto& registration: m_interrupt_eqs) {
			if (registration.event_id == event_id) {
				registrations.push_back(registration);
			}
		}
	}

	for (const auto& registration: registrations) {
		const auto result = LibKernel::EventQueue::KernelTriggerEvent(
		    registration.eq, static_cast<uintptr_t>(registration.event_id),
		    LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS,
		    reinterpret_cast<void*>(static_cast<uintptr_t>(context_id)));
		if (result == LibKernel::KERNEL_ERROR_EBADF || result == LibKernel::KERNEL_ERROR_ENOENT) {
			DeleteInterruptEq(registration.eq, registration.event_id);
			continue;
		}
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

} // namespace Libs::Graphics
