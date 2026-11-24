#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/gpuTiming.h"
#include "common/liveControl.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstring>
#include <fmt/format.h>
#include <functional>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;
// GPU-modified ranges come as many small scattered islands, so a readback widens to a window.
constexpr uint64_t DownloadWindowSize = 512 * 1024;

// For `measure` of the live channel: BDA synchronizations that had ranges to take, those ranges
// and their copies; in-place readbacks and those that waited for the GPU; invalidations made by
// the GPU thread (stores of the emulated command processor).
Common::LiveControl::Counter g_bda_syncs("bda_syncs");
Common::LiveControl::Counter g_bda_ranges("bda_ranges");
Common::LiveControl::Counter g_bda_copies("bda_copies");
Common::LiveControl::Counter g_inplace_reads("inplace_reads");
Common::LiveControl::Counter g_inplace_waits("inplace_waits");
Common::LiveControl::Counter g_inplace_bytes("inplace_bytes");
Common::LiveControl::Counter g_inplace_us("inplace_us");
Common::LiveControl::Counter g_invalidations_gpu("invalidate_gpu");

// 1: a BDA synchronization leaves the write protection of the pages it takes and their copies
// into staging memory to the upload worker (uploadWorker.h); 0: the GPU thread makes both.
Common::LiveControl::Switch g_async_upload("async_upload", 0);

// 1: a readback for a store (a label of the command processor, a guest write) and one for a
// resource walk take the tracking pages of the request only; 0: the download window around it,
// as a guest read fault does.
Common::LiveControl::Switch g_readback_narrow("readback_narrow", 1);
// 1: a log line for every readback a guest read fault asks for (a census, for a few seconds).
Common::LiveControl::Switch g_readback_log("readback_log", 0);
// The most pages (of 16 KiB) the hot set holds: the pages a guest thread read back, whose next
// GPU write is downloaded with its submission so that the next read finds the bytes in guest
// memory (TouchHotPages). And 1: a guest read fault makes every page of the window it is served
// hot, not the first 16 (256 KiB of 512). Every eighth frame Saros's render thread reads back,
// from one of five buffers of 8 MiB in turn, 4 to 4.6 MiB the GPU wrote less than a frame
// before: nine read faults, each a round trip to the GPU thread with a download of half a
// megabyte, 10 to 15 ms in all, in which the GPU thread ran out of work: a frame of 30 to 38 ms
// among frames of 19, about one and a half times a second while standing still. A set of 2,048
// pages forgot those pages before their buffer came round again (some 260 readbacks a second
// pass through it, the buffer returns after 1.3 s), and half of each window never entered it.
// With 16,384 pages and whole windows (same boot, off and on twice): the slowest 1% of frames
// 35.9 -> 22.0 and 23.2 ms, the longest 40 and 76 -> 24 and 26 ms, the 1% low 24-27 -> 42-44
// fps, read-fault waits of the GPU thread 0.8 -> 0.0 a frame.
Common::LiveControl::Switch g_hot_pages_max("hot_pages_max", 16384);
Common::LiveControl::Switch g_hot_window("hot_window", 1);
// 1: a hot page that is clean when the GPU writes it stays readable, with the bytes it had,
// until the download that rode the write's submission is published (the game is taken to read
// it only after a signal of that submission, which the completion thread delivers after the
// publication). 0: the page faults from the write on, as a page outside the hot set does.
Common::LiveControl::Switch  g_stale_readable("stale_readable", 1);
// Guest read faults on bytes whose download was submitted and is not published yet: reads that
// came before the publication (with stale_readable 1 those of a page kept readable do not
// fault: they read the bytes of before the write).
Common::LiveControl::Counter g_early_reads("early_reads");
Common::LiveControl::Counter g_early_read_kb("early_read_kb");
// 1: the window of a guest read fault doubles (to MaxReadAheadWindow) while the faults of a
// read run on through a buffer.
Common::LiveControl::Switch g_readback_ahead("readback_ahead", 1);
constexpr uint64_t          MaxReadAheadWindow = 8 * DownloadWindowSize;

// The most GPU-written bytes (in KiB) a readback copies out of a buffer's own mapping. The
// mapping is device memory: a streaming load of it is a bus read per 64-byte line (RTX 4090:
// 8 us a KiB, 0.12 GB/s; the inplace_us and inplace_bytes counters measure it). Past some size
// the copy the GPU makes into the download buffer is the shorter way, its submission and wait
// (about 0.5 ms) included: at 512 KiB, the 256 KiB a guest read fault takes in Saros held the
// GPU thread for 2 ms.
Common::LiveControl::Switch g_inplace_max_kb("inplace_max_kb", 64);

// 1: a readback a guest thread waits for (its read or write faulted on GPU-written pages) is
// finished by that thread when the GPU thread has nothing to record for it: the bytes are in the
// buffer's mapping and their last GPU write is submitted (the thread waits for it and copies), or
// downloads of them are submitted (it waits for those). 0: the GPU thread serves every request
// whole, which in Saros held it for about 2 ms a frame (some 40 requests a frame, most of them
// a store of the guest into a page with a few KiB of final GPU-written bytes: 50-170 us of bus
// reads each; BufferCache::GuestCopy).
Common::LiveControl::Switch g_guest_copy("guest_copy", 1);
// The most GPU-written bytes (in KiB) such a copy takes. The requester pays the bus reads, so
// more than the in-place limit of the GPU thread's own readbacks.
Common::LiveControl::Switch g_guest_copy_max_kb("guest_copy_max_kb", 256);
// Readbacks left to their requesters: copies, their bytes, waits for submitted downloads, and
// the requests asked again because a GPU write came between.
Common::LiveControl::Counter g_guest_copies("guest_copies");
Common::LiveControl::Counter g_guest_copy_bytes("guest_copy_bytes");
Common::LiveControl::Counter g_guest_copy_waits("guest_copy_waits");
Common::LiveControl::Counter g_guest_copy_again("guest_copy_again");

// 1: a BDA synchronization uploads the guest's stores only into buffers shaders read through
// the page table (BdaUse); a buffer only bindings use takes them when a binding is obtained, and
// its stored pages stay unprotected until then. 0: every cached buffer is synchronized. In Saros
// about 3,700 pages a frame were stored into cached buffers, in thousands of scattered pieces
// (the engine's heap shares its pages with what the GPU reads); shaders read one of them through
// the page table and bindings covered about 700, and the rest was protected and uploaded again
// every frame for nothing: 40.4 -> 47.9 fps in one process, write faults 3,770 -> 840 a frame,
// uploads 16.8 -> 4.3 MiB, protection requests of the GPU thread 1,930 -> 590.
Common::LiveControl::Switch g_bda_lazy("bda_lazy", 1);
// The guest addresses the buffer address table covers at first, in GiB (0: all of them).
// The table is read by the shaders with an entry of 8 bytes per 16 KiB of guest addresses:
// whole it is 768 MiB of video memory on every card, nearly all of it zeros. 128 GiB are
// 64 MiB; a buffer registered past the table grows it (GrowBdaTable). Saros registers buffers
// up to 617 GiB at the start of a level (a table of 352 MiB).
Common::LiveControl::Switch g_bda_table_gib("bda_table_gib", 128);

uint64_t InitialBdaTableBytes(const GraphicContext& graphics) {
	constexpr uint64_t Whole = BufferCache::BDA_PAGETABLE_SIZE;
	const uint64_t     gib   = g_bda_table_gib.Get();
	// A read past the end of a smaller table has to give zero, as an entry nobody wrote does:
	// robustBufferAccess2 says so. Without it the table is whole.
	if (gib == 0 || !graphics.robust_buffer_access_enabled) {
		return Whole;
	}
	const uint64_t covered = std::min<uint64_t>(gib, 4096) << 30u;
	return std::min(Whole,
	                (covered >> BufferCache::CACHING_PAGEBITS) * sizeof(vk::DeviceAddress));
}

// 1: the uses of a small range the guest wrote share one stream copy of it (ObtainBuffer's
// stream path) instead of copying it at every use. On the console a guest store reaches GPU
// work only through an ordering point of the command stream (BdaSyncQueue), so within one such
// stretch every use reads the same bytes, unless the emulator itself stores into them. A copy
// is shared while the scheduler tick the ring keeps it for, the count of ordering points and the
// stores InvalidateMemory was told of over its pages (guest write faults, the command
// processor's own stores, unmaps) are those of the copy, and the range is still guest-written
// and holds no GPU write. In Saros 10,600 such copies a frame (21 MiB), 8,000 of them of a
// range copied before in the same tick: a frame's constants, bound by draw after draw.
// 0: a copy at every use.
Common::LiveControl::Switch  g_stream_share("stream_share", 1);
Common::LiveControl::Counter g_stream_copies("stream_copies");
Common::LiveControl::Counter g_stream_shared("stream_shared");
// The buffer parts a BDA synchronization left to their bindings.
Common::LiveControl::Counter g_bda_unread("bda_unread");

// The most pages a guest write fault opens after its own (BufferCache::FaultAhead); 0: none,
// every page of a walk takes its own fault. Off unless asked for (KYTY_FAULT_AHEAD=16): in a
// loop of test starts 2 of 26 lost the graphics device with this and skip_empty_indirect on
// against 0 of 26 with one or both off, which neither convicts nor clears it, and it saves
// CPU time without a faster frame.
Common::LiveControl::Switch  g_fault_ahead("fault_ahead", 0);
Common::LiveControl::Counter g_fault_ahead_pages("fault_ahead_pages");
Common::LiveControl::Counter g_fault_ahead_windows("fault_ahead_windows");

// The most pages (of 4 KiB) a guest write fault on GPU-written bytes has read back after its
// own (BufferCache::StoreAhead); 0: none, every page of such a run asks for its own readback.
// While a level streams in, Saros's loading and worker threads overwrite memory the GPU wrote
// before (blocks of its pools given to new data): thousands of pages in a row, each a write
// fault, a request to the GPU thread and a copy of 4 KiB out of video memory: 50 to 260 us a
// page, 50 to 250 ms for a burst, in which the GPU thread runs out of work (the frames of 60
// to 260 ms of a walk through the game). With a window that doubles along a run the same bytes
// take a few requests.
//
// The pages ahead are a guess, and other threads store into the same pools at the same time
// (a dozen of them while a level streams in), so they are read back only by the requester's own
// copy (GuestCopy), which takes each page under its region's lock and only while the page is
// still GPU-dirty there: whoever stored into a page meanwhile has cleaned it first. The GPU
// thread's download is not such a copy. It records the window's dirty bytes and writes them
// into the guest's memory after its wait for the GPU, whatever happened to the pages in
// between; for a page another thread had just been handed and had stored into, that put the
// GPU's old bytes over the new data (streamed textures and meshes: a level that looks wrong
// after a death, when the pools are given out again). A window that cannot be handed to the
// requester is therefore cut back to the request's own pages, and the limit is what one copy
// takes (guest_copy_max_kb less the request's page).
//
// As first written, with windows up to 256 pages that the GPU thread downloaded past a quarter
// megabyte: a cycle restarted from the pause menu and the walk after it, one run each way,
// 52 -> 40 slow frames, 3.3 -> 2.7 s over the usual frame time, the game's threads held
// 3.4 -> 1.8 s by such readbacks.
Common::LiveControl::Switch  g_store_ahead("store_ahead", 63);
Common::LiveControl::Counter g_store_ahead_pages("store_ahead_pages");
Common::LiveControl::Counter g_store_ahead_windows("store_ahead_windows");

std::pair<Buffer*, uint64_t> UploadMemory(GraphicContext& graphics, CommandScheduler& scheduler,
                                        StreamBuffer& staging, uint64_t size, uint64_t alignment,
                                        vk::BufferUsageFlags usage, const auto& write) {
	const auto [mapped, offset] = staging.Map(size, alignment);
	if (mapped != nullptr) {
		write(mapped, offset);
		staging.Commit();
		return {&staging, offset};
	}

	// Large transfers use the same one-time host allocation as ordinary buffer uploads.
	auto temporary = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Upload, 0,
	                                         usage, size);
	write(temporary->Mapped().data(), 0);
	temporary->Flush(0, size);
	auto* buffer = temporary.get();
	scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return {buffer, 0};
}

} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		// Copy waits for ring reuse and flushes the fresh host data before submission.
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		const auto destination_offset = buffer.Offset(address);
		EXIT_IF(destination_offset > buffer.Size() || chunk > buffer.Size() - destination_offset);
		m_scheduler.EndRendering();
		const auto command = m_scheduler.Current().Handle();
		vk::BufferMemoryBarrier2 before {};
		before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		before.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		before.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		before.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer        = buffer.Handle();
		before.offset        = destination_offset;
		before.size          = chunk;
		vk::DependencyInfo dependency {};
		dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
		dependency.bufferMemoryBarrierCount = 1;
		dependency.pBufferMemoryBarriers    = &before;
		command.pipelineBarrier2(dependency);
		const vk::BufferCopy copy {offset, destination_offset, chunk};
		command.copyBuffer(m_staging_buffer.Handle(), buffer.Handle(), 1, &copy);
		auto after          = before;
		after.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		after.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		after.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		after.dstAccessMask              = LaterAccess2;
		dependency.pBufferMemoryBarriers = &after;
		command.pipelineBarrier2(dependency);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto& buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!(GuestRange {buffer.CpuAddress(), buffer.Size()}.Valid()) ||
	        !PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	const auto table_offset = PageIndex(buffer.CpuAddress()) * sizeof(vk::DeviceAddress);
	if constexpr (insert) {
		m_buffers.emplace(buffer.CpuAddress(), id);
		m_total_used_memory += buffer.Size();
		buffer.lru_id         = m_lru_cache.Insert(id, m_gc_tick);
		buffer.bda_watch_tick = 0;
		buffer.registered_tick = m_scheduler.CurrentTick();
		WriteBdaEntries(buffer);
		QueueBdaSync(buffer.CpuAddress(), buffer.Size());
	} else {
		m_buffers.erase(m_buffers.find(buffer.CpuAddress()));
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		WriteBdaTable(table_offset / sizeof(vk::DeviceAddress), size_pages, nullptr);
		buffer.is_deleted = true;
	}
}

void BufferCache::TouchBuffer(Buffer& buffer) {
	// A deleted buffer waits only for its slot to be freed: its LRU item is free and its page
	// table entries are cleared, and neither may come back.
	if (buffer.is_deleted) {
		return;
	}
	m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
	if (buffer.bda_watch_tick != 0) {
		buffer.bda_watch_tick = 0;
		if (!IsBdaWatched(buffer)) {
			WriteBdaEntries(buffer);
		}
	}
}

void BufferCache::WriteBdaEntries(const Buffer& buffer) {
	PageTable::PageRange pages {};
	EXIT_IF(!PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	const auto flags = IsBdaWatched(buffer) ? BdaWatchBit : vk::DeviceAddress {0};
	std::vector<vk::DeviceAddress> addresses;
	addresses.reserve(pages.last_exclusive - pages.first);
	for (uint64_t i = 0; i < pages.last_exclusive - pages.first; ++i) {
		const auto address = buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS);
		EXIT_IF((address & BdaWatchBit) != 0);
		addresses.push_back(address | flags);
	}
	const auto first = PageIndex(buffer.CpuAddress());
	if ((first + addresses.size()) * sizeof(vk::DeviceAddress) > GetBdaPageTableBuffer()->Size()) {
		GrowBdaTable(first + addresses.size());
	}
	WriteBdaTable(first, addresses.size(), addresses.data());
}

void BufferCache::WriteBdaTable(uint64_t first, uint64_t count, const vk::DeviceAddress* entries) {
	for (const auto& table: m_bda_tables) {
		const uint64_t held = table->Size() / sizeof(vk::DeviceAddress);
		if (first >= held) {
			continue;
		}
		const uint64_t part = std::min(count, held - first);
		if (entries != nullptr) {
			WriteDataBuffer(*table, first * sizeof(vk::DeviceAddress), entries,
			                part * sizeof(vk::DeviceAddress));
		} else {
			table->Fill(first * sizeof(vk::DeviceAddress), part * sizeof(vk::DeviceAddress), 0);
		}
	}
}

// A buffer registered past the newest table: a table that reaches it, and an eighth further
// (the game's heaps grow upwards), takes its place, with the old entries copied and the rest
// zero. The old one stays for BdaTableKeepTicks frames and is written alike (WriteBdaTable): a
// descriptor set written with it and still bound in a command buffer that is being recorded
// reads right entries for what it covers and zero past it, as before. A command buffer binds
// its sets again, so nothing reads the old table after that (RetireBdaTables).
void BufferCache::GrowBdaTable(uint64_t entries) {
	constexpr uint64_t Step = 16 * MiB;
	auto*              old  = GetBdaPageTableBuffer();
	const uint64_t     need = entries * sizeof(vk::DeviceAddress);
	EXIT_IF(need > BDA_PAGETABLE_SIZE || need <= old->Size());
	const uint64_t bytes = std::min(BDA_PAGETABLE_SIZE, Common::AlignUp(need + need / 8, Step));
	auto table = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
	                                      AllFlags, bytes);
	SetVulkanObjectNameF(m_graphics.device, table->Handle(), "BDA Page Table Buffer");
	m_scheduler.EndRendering();
	const auto               command = m_scheduler.Current().Handle();
	vk::BufferMemoryBarrier2 barriers[2] {};
	for (auto& barrier: barriers) {
		barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		barrier.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.offset              = 0;
		barrier.size                = old->Size();
	}
	barriers[0].buffer        = old->Handle();
	barriers[0].dstAccessMask = vk::AccessFlagBits2::eTransferRead;
	barriers[1].buffer        = table->Handle();
	barriers[1].dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
	vk::DependencyInfo dependency {};
	dependency.bufferMemoryBarrierCount = 2;
	dependency.pBufferMemoryBarriers    = barriers;
	command.pipelineBarrier2(dependency);
	const vk::BufferCopy copy {0, 0, old->Size()};
	command.copyBuffer(old->Handle(), table->Handle(), 1, &copy);
	for (auto& barrier: barriers) {
		barrier.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		barrier.srcAccessMask = barrier.dstAccessMask;
		barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		barrier.dstAccessMask = LaterAccess2;
	}
	command.pipelineBarrier2(dependency);
	table->Fill(old->Size(), bytes - old->Size(), 0);
	Log::WriteToConsoleAndLog(fmt::format(
	    "Buffer address table: {} MiB for {} GiB of guest addresses (a buffer at {:#x})\n",
	    bytes >> 20u, (bytes / sizeof(vk::DeviceAddress)) >> (30u - CACHING_PAGEBITS),
	    GuestAddress((entries - 1) << CACHING_PAGEBITS)));
	m_bda_table.store(table.get(), std::memory_order_release);
	m_bda_tables.push_back(std::move(table));
	m_bda_retire_tick = m_gc_tick + BdaTableKeepTicks;
}

void BufferCache::RetireBdaTables() {
	if (m_bda_tables.size() < 2 || m_gc_tick < m_bda_retire_tick) {
		return;
	}
	// (Freed once the GPU is done with what is recorded so far.)
	for (size_t i = 0; i + 1 < m_bda_tables.size(); ++i) {
		m_scheduler.DeferOperation(
		    [owner = std::move(m_bda_tables[i])]() mutable { owner.reset(); });
	}
	m_bda_tables.erase(m_bda_tables.begin(), m_bda_tables.end() - 1);
}

bool BufferCache::NoteBdaAccess(uint64_t address, uint64_t window_start) {
	const auto* owner = m_page_table.Find(address >> PageTable::kPageBits);
	if (owner == nullptr || !*owner) {
		return false;
	}
	auto& buffer = m_slot_buffers[*owner];
	// A resident page reports only through a watched entry (or a cleared one). A buffer that is
	// no longer watched but was registered before the reporting dispatches ran was watched when
	// they accessed it: a report processed earlier unwatched it, and the access was made. One
	// registered since had no entry then, and the access was lost. (Where unread buffers stay
	// watched, g_bda_lazy, a buffer is registered watched and loses the watch only by a report:
	// an unwatched one was watched then.)
	if (!m_bda_lazy && buffer.bda_watch_tick == 0 && buffer.registered_tick >= window_start) {
		return false;
	}
	// The page is resident, so the access was made. Its watch may have ended since the shader
	// ran: the report still shows the use, and the page needs no other buffer.
	MarkBdaRead(buffer);
	if (buffer.bda_watch_tick != 0) {
		m_bda_accessed.push_back(address);
	}
	return true;
}

void BufferCache::MarkBdaRead(Buffer& buffer) {
	const auto previous = std::exchange(buffer.bda_use, BdaUse::Read);
	buffer.bda_use_tick = m_gc_tick;
	if (previous == BdaUse::Read) {
		return;
	}
	if (previous == BdaUse::Unread) {
		// The stores it did not take: synchronizations passed them over (SynchronizeBda).
		QueueBdaSync(buffer.CpuAddress(), buffer.Size());
	}
	if (!IsBdaWatched(buffer)) {
		WriteBdaEntries(buffer);
	}
}

void BufferCache::ProbeBdaReaders() {
	const bool lazy = g_bda_lazy.On();
	if (lazy != m_bda_lazy) {
		// The switch changed: the entries of the unread buffers gain or lose their watch, and
		// with every buffer synchronized again they take the stores passed over so far.
		m_bda_lazy = lazy;
		for (const auto& [vaddr, id]: m_buffers) {
			(void)vaddr;
			const auto& buffer = m_slot_buffers[id];
			if (buffer.bda_use != BdaUse::Read && buffer.bda_watch_tick == 0) {
				WriteBdaEntries(buffer);
			}
			if (!lazy && buffer.bda_use == BdaUse::Unread) {
				QueueBdaSync(buffer.CpuAddress(), buffer.Size());
			}
		}
	}
	if (!lazy || m_gc_tick % BdaProbeStride != 0) {
		return;
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		auto&      buffer = m_slot_buffers[id];
		const auto age    = m_gc_tick - buffer.bda_use_tick;
		if (buffer.bda_use == BdaUse::Read && age >= BdaProbeInterval) {
			const bool watched  = IsBdaWatched(buffer);
			buffer.bda_use      = BdaUse::Probed;
			buffer.bda_use_tick = m_gc_tick;
			if (!watched) {
				WriteBdaEntries(buffer);
			}
		} else if (buffer.bda_use == BdaUse::Probed && age >= BdaProbationTicks) {
			// Its entries stay watched: the next access through them is reported.
			buffer.bda_use = BdaUse::Unread;
		}
	}
}

void BufferCache::KeepAccessedBdaBuffers() {
	for (const auto address: m_bda_accessed) {
		// The page's buffer, which may have replaced the reported one since.
		if (const auto* owner = m_page_table.Find(address >> PageTable::kPageBits);
		    owner != nullptr && *owner) {
			TouchBuffer(m_slot_buffers[*owner]);
		}
	}
	m_bda_accessed.clear();
	// Touched and deleted buffers leave the list; an ID reused by a new buffer is not watched.
	std::erase_if(m_bda_watched, [&](BufferId id) {
		if (!m_slot_buffers.is_allocated(id)) {
			return true;
		}
		auto& buffer = m_slot_buffers[id];
		if (buffer.is_deleted) {
			return true;
		}
		if (m_bda_reports_lost) {
			TouchBuffer(buffer);
		}
		return buffer.bda_watch_tick == 0;
	});
	m_bda_reports_lost = false;
}

void BufferCache::DeleteBuffer(BufferId id) {
	WaitGuestCopies(id);
	if (m_slot_buffers[id].gpu_write_open) {
		std::erase(m_open_writes, id);
	}
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] { m_slot_buffers.erase(id); });
	} else {
		m_slot_buffers.erase(id);
	}
}

template <bool async>
bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size,
                                       std::vector<Common::UniqueFunction<void>>* batch) {
	const CommandBuffer::ProfileGuard profile(m_scheduler.Current(),
	                                          CommandBuffer::ProfileScope::Downloads);
	struct Run {
		uint64_t address;
		uint64_t bytes;
	};
	std::vector<vk::BufferCopy> copies;
	// The GPU-dirty tracker pages the copies lie in, ascending like the copies.
	std::vector<Run> runs;
	uint64_t         total_size     = 0;
	const auto       buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges.Ranges(), address, bytes,
		                                           "buffer download");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Start each packed range on its own 64-byte cache line.
			    total_size += Common::AlignUp(end - start, 64);
		    });
		    runs.push_back({address, bytes});
		    m_gpu_modified_ranges.Subtract(address, bytes);
		    ForgetColorMetadataChecks(address, bytes);
	    });
	if (copies.empty()) {
		return false;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferDownloadBytes, total_size);

	auto download = m_download_buffer.MapDownload(total_size, 64);
	for (auto& copy: copies) {
		copy.dstOffset += download.offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.copyBuffer(buffer.Handle(), download.buffer->Handle(),
	                  static_cast<uint32_t>(copies.size()), copies.data());

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download.buffer->Handle();
	after.offset        = download.offset;
	after.size          = total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
	uint64_t written_begin = UINT64_MAX;
	uint64_t written_end   = 0;
	for (const auto& copy: copies) {
		written_begin = std::min(written_begin, buffer_address + copy.srcOffset);
		written_end   = std::max(written_end, buffer_address + copy.srcOffset + copy.size);
	}
	auto publish = [this, total_size, buffer_address, copies = std::move(copies),
	                runs = std::move(runs), download = std::move(download)] {
		download.buffer->Invalidate(download.offset, total_size);
		if constexpr (async) {
			(void)this;
			(void)runs;
			for (const auto& copy: copies) {
				Libs::LibKernel::Memory::WriteBacking(
				    buffer_address + copy.srcOffset,
				    download.mapped + (copy.dstOffset - download.offset), copy.size);
			}
		} else {
			// The GPU thread's own readback: between the copies above and this publication it
			// waited for the GPU and recorded nothing, so no GPU write is newer than these
			// bytes. A guest thread may have been faster with a page all the same: one that
			// was handed the page's readback before (GuestCopy) has by now published the same
			// bytes from the mapping and gone on to store into the page. The bytes go only
			// into pages that are GPU-dirty still, under the region's lock as that thread's
			// copy takes it; written without the check they put the GPU's old bytes over the
			// thread's new ones. (A few caching pages at a time: guest threads fault on the
			// region meanwhile.)
			constexpr uint64_t Piece = 4 * CACHING_PAGESIZE;
			size_t             first = 0;
			const auto         begin_of = [&](size_t i) { return buffer_address + copies[i].srcOffset; };
			for (const auto& run: runs) {
				for (auto piece = run.address; piece < run.address + run.bytes; piece += Piece) {
					const auto piece_bytes = std::min(Piece, run.address + run.bytes - piece);
					while (first < copies.size() && begin_of(first) + copies[first].size <= piece) {
						++first;
					}
					m_memory_tracker.PublishGpuModified(
					    piece, piece_bytes, [] { return true; },
					    [&](uint64_t address, uint64_t bytes) noexcept {
						    for (size_t i = first; i < copies.size() && begin_of(i) < address + bytes;
						         ++i) {
							    const auto from = std::max(address, begin_of(i));
							    const auto to =
							        std::min(address + bytes, begin_of(i) + copies[i].size);
							    if (from < to) {
								    Libs::LibKernel::Memory::WriteBacking(
								        from,
								        download.mapped + (copies[i].dstOffset - download.offset) +
								            (from - begin_of(i)),
								        to - from);
							    }
						    }
					    });
				}
			}
		}
	};
	if constexpr (async) {
		m_scheduler.DeferPriorityOperation(
		    std::move(publish), GuestWrites {written_begin, written_end - written_begin});
	} else if (batch != nullptr) {
		batch->emplace_back(std::move(publish));
	} else {
		const auto tick = m_scheduler.CurrentTick();
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		publish();
	}
	return true;
}

bool BufferCache::ReadBackInPlace(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	if (!buffer.IsHostReadable()) {
		return false;
	}
	// A write marked by a command still being prepared may not be recorded yet.
	const auto write_tick = buffer.gpu_write_tick;
	if (buffer.gpu_write_open || write_tick >= m_scheduler.CurrentTick()) {
		return false;
	}
	uint64_t dirty = 0;
	m_gpu_modified_ranges.ForEachInRange(
	    vaddr, size, [&](uint64_t begin, uint64_t end) { dirty += end - begin; });
	if (dirty == 0 || dirty > (uint64_t {g_inplace_max_kb.Get()} << 10u)) {
		return false;
	}
	g_inplace_reads.Add();
	g_inplace_bytes.Add(dirty);
	if (!m_scheduler.IsFree(write_tick)) {
		g_inplace_waits.Add();
		m_scheduler.Wait(write_tick);
	}
	// Publications of earlier downloads of these bytes land first, so none overwrites them.
	m_scheduler.WaitPriorityOperations(write_tick);
	const auto buffer_address = buffer.CpuAddress();
	bool       copied         = false;
	const auto copy_begin     = std::chrono::steady_clock::now();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges.Ranges(), address, bytes,
		                                           "in-place readback");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    m_in_place_scratch.resize(static_cast<size_t>(end - start));
			    buffer.ReadMapped(start - buffer_address, end - start, m_in_place_scratch.data());
			    Libs::LibKernel::Memory::WriteBacking(start, m_in_place_scratch.data(),
			                                          end - start);
			    copied = true;
		    });
		    m_gpu_modified_ranges.Subtract(address, bytes);
		    ForgetColorMetadataChecks(address, bytes);
	    });
	g_inplace_us.Add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                           std::chrono::steady_clock::now() - copy_begin)
	                                           .count()));
	if (copied) {
		Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferDownloadBytes, dirty);
	}
	return copied;
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	m_bda_lazy = g_bda_lazy.On();
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	m_bda_tables.push_back(std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::DeviceLocal,
	                                                0, AllFlags, InitialBdaTableBytes(m_graphics)));
	m_bda_table.store(m_bda_tables.back().get(), std::memory_order_release);
	SetVulkanObjectNameF(m_graphics.device, GetBdaPageTableBuffer()->Handle(),
	                     "BDA Page Table Buffer");
	m_scheduler.BeginCommand();
	for (auto* buffer: {GetBdaPageTableBuffer(), m_fault_manager.GetFaultBuffer()}) {
		buffer->Fill(0, buffer->Size(), 0);
	}
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	// Bindings of empty buffers read this one, so it must hold zeros, not leftover memory.
	GetBuffer(null_id).Fill(0, 16, 0);
	UpdateGcMarks();
}

void BufferCache::UpdateGcMarks() {
	if (m_gc_marks_pinned || !m_graphics.CanReportMemoryUsage()) {
		return;
	}
	const auto live = m_graphics.UpdateCacheMarks();
	if (live.Generation() == m_gc_marks_generation) {
		return;
	}
	m_gc_marks_generation = live.Generation();
	m_trigger_gc_memory   = live.Buffer().trigger;
	m_critical_gc_memory  = live.Buffer().critical;
}

BufferCache::~BufferCache() {
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
}

void BufferCache::NoteStreamCopyStore(uint64_t vaddr, uint64_t size) {
	if (size == 0) {
		return;
	}
	constexpr uint64_t MaxPages = 64;
	const auto         first    = vaddr >> PageTable::kPageBits;
	const auto         last     = (vaddr + size - 1) >> PageTable::kPageBits;
	if (last - first >= MaxPages) {
		m_stream_wide_stores.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	for (auto page = first; page <= last; page++) {
		m_stream_page_stores[page % StreamStoreSlots].fetch_add(1, std::memory_order_relaxed);
	}
}

BufferCache::StreamCopyStamp BufferCache::StreamCopyStores(uint64_t vaddr, uint64_t size) const {
	const auto count = [this](uint64_t page) {
		return m_stream_page_stores[page % StreamStoreSlots].load(std::memory_order_relaxed);
	};
	return {.wide  = m_stream_wide_stores.load(std::memory_order_relaxed),
	        .first = count(vaddr >> PageTable::kPageBits),
	        .last  = count((vaddr + size - 1) >> PageTable::kPageBits)};
}

void BufferCache::FaultAhead(uint64_t vaddr) {
	const auto limit = g_fault_ahead.Get();
	if (limit == 0) {
		return;
	}
	// Saros takes about 1,700 write faults a frame on pages the buffer cache write-protected, and
	// three in four of them fall beside another one: guest threads filling a buffer walk through
	// its pages, each page costing an exception, a protection change by the faulting thread and
	// another by the GPU thread when the page is uploaded. A thread that faults on the page after
	// its last fault (or after the window that fault opened) is such a walk: the window after the
	// page doubles each time, up to the limit. A single fault opens nothing, so stores scattered
	// over memory make no page CPU-modified that they do not write.
	thread_local uint64_t next_page = 0;
	thread_local uint32_t window    = 0;
	const auto            page      = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto            after     = page + TRACKER_PAGE_SIZE;
	if (page != next_page) {
		next_page = after;
		window    = 0;
		return;
	}
	window            = std::min<uint32_t>(window == 0 ? 2u : window * 2u, limit);
	const auto opened = m_memory_tracker.MarkCpuModifiedAhead(page, window);
	next_page         = after + opened;
	if (opened == 0) {
		return;
	}
	// What a fault on each of the pages would have told the cache.
	NoteStreamCopyStore(after, opened);
	QueueBdaSync(after, opened, true);
	g_fault_ahead_windows.Add();
	g_fault_ahead_pages.Add(opened / TRACKER_PAGE_SIZE);
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size, bool guest_store,
                                   bool write_fault) {
	NoteStreamCopyStore(vaddr, size);
	if (!guest_store) {
		// A store of the emulated command processor, a write-back or an unmap: an upload the
		// worker still has to copy from these pages belongs to earlier commands and must not
		// see what comes now, so the pages (all of each: the invalidation below marks whole
		// pages modified again) are copied for it first. (A guest store races with the work it
		// was not ordered before.)
		const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
		const auto end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
		m_upload_worker.CaptureRange(begin, end - begin);
	}
	if (Common::LiveControl::OnGpuThread()) {
		g_invalidations_gpu.Add();
	}
	const Common::GpuTiming::SyncScope sync_reason(Common::GpuTiming::SyncReason::CpuWrite);
	m_memory_tracker.InvalidateRegion(vaddr, size, [this, vaddr, size, guest_store, write_fault] {
		ReadMemory(vaddr, size, true, write_fault && guest_store ? StoreAhead(vaddr) : 0);
	});
	QueueBdaSync(vaddr, size, guest_store);
	if (write_fault && guest_store) {
		FaultAhead(vaddr);
	}
}

uint64_t BufferCache::StoreAhead(uint64_t vaddr) {
	const auto limit = g_store_ahead.Get();
	if (limit == 0) {
		return 0;
	}
	// The run of one thread, as in FaultAhead: a fault on the page after the last one, or after
	// the window the last one was given, doubles the window. A single store asks for its own
	// page only, as before.
	thread_local uint64_t next_page = 0;
	thread_local uint32_t window    = 0;
	const auto            page      = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	if (page != next_page) {
		next_page = page + TRACKER_PAGE_SIZE;
		window    = 0;
		return 0;
	}
	window    = std::min<uint32_t>(window == 0 ? 2u : window * 2u, limit);
	next_page = page + (uint64_t {window} + 1u) * TRACKER_PAGE_SIZE;
	g_store_ahead_windows.Add();
	g_store_ahead_pages.Add(window);
	return uint64_t {window} * TRACKER_PAGE_SIZE;
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write, uint64_t ahead) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	// The readback runs on the GPU thread; it counts under the requester's reason.
	auto reason = Common::GpuTiming::CurrentSyncReason();
	if (reason == Common::GpuTiming::SyncReason::Other) {
		reason = is_write ? Common::GpuTiming::SyncReason::CpuWrite
		                  : Common::GpuTiming::SyncReason::BufferRead;
	}
	RequestGuestReadback({is_write ? GuestReadback::Kind::Write : GuestReadback::Kind::Read, vaddr,
	                      size, reason, nullptr, nullptr, ahead});
}

void BufferCache::ReadMemoryAround(uint64_t vaddr, uint64_t window) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 "\n",
		     vaddr);
	}
	RequestGuestReadback({GuestReadback::Kind::Around, vaddr, window,
	                      Common::GpuTiming::SyncReason::FaultRead, nullptr});
}

void BufferCache::RequestGuestReadback(const GuestReadback& request) {
	if (GuestGpu::IsGpuThread()) {
		ServeGuestReadbacks({&request, 1});
		return;
	}
	const Common::LiveControl::GuestWait held(
	    request.kind == GuestReadback::Kind::Write
	        ? "readback-for-store"
	        : request.kind == GuestReadback::Kind::Around ? "readback-for-read" : "readback",
	    request.vaddr, request.size);
	std::shared_ptr<GuestCopy> copy;
	QueueGuestReadback(request, g_guest_copy.On() ? &copy : nullptr);
	if (copy == nullptr) {
		return;
	}
	RunGuestCopy(*copy);
	// What the GPU thread does after a copy of its own, unless a GPU write came between (the
	// bytes are GPU-dirty still): then the request is asked again, to be served whole.
	bool again = false;
	if (request.kind == GuestReadback::Kind::Write) {
		m_memory_tracker.InvalidateRegion(request.vaddr, request.size, [&] { again = true; });
		if (!again) {
			QueueBdaSync(request.vaddr, request.size);
		}
		if (request.ahead != 0 && copy->buffer != nullptr) {
			// A store with pages ahead: the pages this copy brought the bytes of belong to the
			// CPU now, as after a fault on each. Only those: a page of the window the copy held
			// nothing for is clean (and stays write-protected, another cache may watch it) or is
			// another request's. One that is GPU-dirty again takes its own fault later.
			const auto window_end = copy->base + copy->size;
			uint64_t   run_begin  = 0;
			uint64_t   run_end    = 0;
			const auto hand_over  = [&] {
				if (run_end > run_begin) {
					m_memory_tracker.InvalidateRegion(run_begin, run_end - run_begin, [] {});
					NoteStreamCopyStore(run_begin, run_end - run_begin);
					QueueBdaSync(run_begin, run_end - run_begin);
				}
			};
			for (const auto& interval: copy->intervals) {
				const auto begin =
				    std::max(Common::AlignDown(interval.address, TRACKER_PAGE_SIZE), copy->base);
				const auto end = std::min(
				    Common::AlignUp(interval.address + interval.size, TRACKER_PAGE_SIZE), window_end);
				if (begin > run_end) {
					hand_over();
					run_begin = begin;
				}
				run_end = std::max(run_end, end);
			}
			hand_over();
		}
	} else {
		again = m_memory_tracker.IsRegionGpuModified(
		    request.vaddr, request.kind == GuestReadback::Kind::Around ? 1u : request.size);
	}
	if (again) {
		g_guest_copy_again.Add();
		auto whole  = request;
		whole.ahead = 0;
		QueueGuestReadback(whole, nullptr);
	}
}

void BufferCache::QueueGuestReadback(const GuestReadback&        request,
                                     std::shared_ptr<GuestCopy>* handoff) {
	// Every request is followed by a command that serves all pending ones, so it is served by
	// its own command at the latest; a later command finds nothing left.
	std::binary_semaphore done {0};
	{
		std::lock_guard lock(m_guest_readbacks_mutex);
		auto&           queued = m_guest_readbacks.emplace_back(request);
		queued.done            = &done;
		queued.handoff         = handoff;
	}
	m_scheduler.Context().GetGpu().SendCommand([this] {
		std::vector<GuestReadback> requests;
		{
			std::lock_guard lock(m_guest_readbacks_mutex);
			requests.swap(m_guest_readbacks);
		}
		if (!requests.empty()) {
			ServeGuestReadbacks(requests);
		}
	});
	done.acquire();
}

void BufferCache::ServeGuestReadbacks(std::span<const GuestReadback> requests) {
	struct Served {
		uint64_t base   = 0;
		uint64_t size   = 0;
		bool     copied = false;
		bool     skip   = false;
		bool     handed = false;
	};
	std::vector<Served>                          served(requests.size());
	std::vector<Common::UniqueFunction<void>>    publications;
	std::optional<Common::GpuTiming::SyncReason> submit_reason;
	for (size_t i = 0; i < requests.size(); i++) {
		const auto&                        request = requests[i];
		auto&                              result  = served[i];
		const Common::GpuTiming::SyncScope sync_reason(request.reason);
		result.base = request.vaddr;
		result.size = request.size;
		if (request.kind == GuestReadback::Kind::Write &&
		    !IsRegionRegistered(request.vaddr, request.size)) {
			result.skip = true;
			continue;
		}
		if (request.kind == GuestReadback::Kind::Around) {
			// The dirty-range query runs here: the range set is only serialized on the GPU
			// thread, and a guest thread faulting on a read arrives from anywhere.
			const auto vaddr = request.vaddr;
			const auto page  = vaddr & ~(CACHING_PAGESIZE - 1u);
			const auto query = std::min(request.size, UINT64_MAX - page);
			result.size      = 1;
			m_gpu_modified_ranges.ForEachInRange(page, query, [&](uint64_t begin, uint64_t end) {
				if (begin <= vaddr && vaddr < end) {
					result.base = begin;
					result.size = end - begin;
				}
			});
		}
		TouchHotPages(result.base, result.size);
		DrainPublished();
		const auto buffer_id = FindBuffer(result.base, result.size);
		auto&      buffer    = m_slot_buffers[buffer_id];
		// The end of the window that is a store's pages ahead (StoreAhead).
		uint64_t ahead_bytes = 0;
		if (IsRegionGpuModified(result.base, result.size)) {
			// GPU-modified ranges come as many small scattered islands, so a readback widens to
			// a window of its buffer (a fault read: the window around the faulting byte).
			const auto wanted_end = request.kind == GuestReadback::Kind::Around
			                            ? request.vaddr + 1u
			                            : request.vaddr + request.size;
			auto start = std::max(Common::AlignDown(request.vaddr, DownloadWindowSize),
			                      buffer.CpuAddress());
			auto end   = std::min(std::max(start + DownloadWindowSize, wanted_end),
			                      buffer.CpuAddress() + buffer.Size());
			// The window pays for a guest that reads on through the pages after its fault. A
			// store needs the pages it writes and no more, and a walk asks for every range it
			// reads in one batch: the window around such a request only moves bytes nothing
			// asked for (Saros stores a 4-byte label into a 512 KiB range the GPU writes each
			// frame, and the window read all of it back for that), and hands their pages to the
			// CPU until the GPU's next write takes them again. Other invalidations keep the
			// window: an unmap of part of a buffer publishes the GPU-written pages beside it.
			if (g_readback_narrow.On() &&
			    ((request.kind == GuestReadback::Kind::Write &&
			      request.reason == Common::GpuTiming::SyncReason::CpuWrite) ||
			     (request.kind == GuestReadback::Kind::Read &&
			      request.reason == Common::GpuTiming::SyncReason::WalkRead))) {
				start = std::max(Common::AlignDown(request.vaddr, TRACKER_PAGE_SIZE), start);
				end   = std::min(Common::AlignUp(wanted_end, TRACKER_PAGE_SIZE), end);
			}
			if (request.kind == GuestReadback::Kind::Write && request.ahead != 0 &&
			    request.handoff != nullptr) {
				// A store that runs on through GPU-written pages: those after its own, as far
				// as the buffer goes (StoreAhead). For the requester's own copy only, see below.
				const auto own_end = end;
				end = std::max(end, std::min(Common::AlignUp(wanted_end, TRACKER_PAGE_SIZE) +
				                                 request.ahead,
				                             buffer.CpuAddress() + buffer.Size()));
				ahead_bytes = end - own_end;
			}
			if (request.kind == GuestReadback::Kind::Around) {
				// A guest thread that reads on through what the GPU wrote faults again where
				// its window ends. While the faults follow one another the window doubles: the
				// 4.5 MiB Saros's render thread reads of an 8 MiB buffer take four round trips to
				// this thread, not nine (readback_ahead).
				uint64_t window = DownloadWindowSize;
				if (g_readback_ahead.On() && m_read_run_buffer == buffer_id &&
				    request.vaddr >= m_read_run_end &&
				    request.vaddr - m_read_run_end < DownloadWindowSize) {
					window = std::min(m_read_run_window * 2, MaxReadAheadWindow);
					end    = std::min(std::max(start + window, wanted_end),
					                  buffer.CpuAddress() + buffer.Size());
				}
				m_read_run_buffer = buffer_id;
				m_read_run_end    = end;
				m_read_run_window = window;
			}
			result.base = start;
			result.size = end - start;
			if (request.kind == GuestReadback::Kind::Around && g_hot_window.On()) {
				TouchHotPages(result.base, result.size, result.size / CACHING_PAGESIZE);
			}
		}
		if (request.kind == GuestReadback::Kind::Around && request.handoff != nullptr &&
		    Common::LiveControl::Active()) {
			if (const auto tick = SubmittedDownloadTick(result.base, result.size); tick != 0) {
				g_early_reads.Add();
				g_early_read_kb.Add(result.size >> 10u);
				if (g_readback_log.On()) {
					LOGF("Readback log: early read at 0x%" PRIx64 " window 0x%" PRIx64 "+0x%" PRIx64
					     ", its download is %" PRIu64 " ticks old and %s\n",
					     request.vaddr, result.base, result.size, m_scheduler.CurrentTick() - tick,
					     m_scheduler.IsFree(tick) ? "done on the GPU" : "still running");
				}
			}
		}
		if (g_readback_log.On() && request.kind == GuestReadback::Kind::Around) {
			uint64_t dirty     = 0;
			uint64_t intervals = 0;
			uint64_t hot       = 0;
			m_gpu_modified_ranges.ForEachInRange(result.base, result.size,
			                                     [&](uint64_t begin, uint64_t end) {
				                                     dirty += end - begin;
				                                     intervals++;
			                                     });
			for (auto page = result.base & ~(CACHING_PAGESIZE - 1u); page < result.base + result.size;
			     page += CACHING_PAGESIZE) {
				hot += m_hot_pages.contains(page) ? 1 : 0;
			}
			LOGF("Readback log: read at 0x%" PRIx64 " buffer 0x%" PRIx64 "+0x%" PRIx64
			     " window 0x%" PRIx64 "+0x%" PRIx64 " dirty %" PRIu64 " in %" PRIu64
			     " pieces, hot pages %" PRIu64 ", write %" PRIu64 " ticks ago%s%s, submitted download %" PRIu64 "\n",
			     request.vaddr, buffer.CpuAddress(), buffer.Size(), result.base, result.size, dirty,
			     intervals, hot, m_scheduler.CurrentTick() - buffer.gpu_write_tick,
			     buffer.gpu_write_open ? " (open)" : "", buffer.IsHostReadable() ? "" : " (not readable)",
			     SubmittedDownloadTick(result.base, result.size));
		}
		if (g_readback_log.On() && request.kind == GuestReadback::Kind::Write &&
		    request.handoff == nullptr) {
			uint64_t dirty = 0;
			m_gpu_modified_ranges.ForEachInRange(
			    result.base, result.size, [&](uint64_t begin, uint64_t end) { dirty += end - begin; });
			const auto submitted = SubmittedDownloadTick(result.base, result.size);
			LOGF("Readback log: store at 0x%" PRIx64 "+%" PRIu64 " by the GPU thread (%s), window 0x%"
			     PRIx64 "+0x%" PRIx64 " dirty %" PRIu64 ", the stored bytes are %s, write %" PRIu64
			     " ticks ago%s, submitted download %s\n",
			     request.vaddr, request.size, Common::GpuTiming::SyncReasonName(request.reason),
			     result.base, result.size, dirty,
			     m_gpu_modified_ranges.Intersects(request.vaddr, request.size) ? "GPU-written"
			                                                                  : "not GPU-written",
			     m_scheduler.CurrentTick() - buffer.gpu_write_tick,
			     buffer.gpu_write_open ? " (open)" : "",
			     submitted == 0 ? "none" : m_scheduler.IsFree(submitted) ? "done" : "running");
		}
		bool handed = request.handoff != nullptr &&
		              HandOffGuestCopy(request, buffer, buffer_id, result.base, result.size);
		if (!handed && ahead_bytes != 0) {
			// The pages ahead of a store are a guess about memory other threads store into as
			// well. The requester's copy takes each of them under its region's lock and only
			// while it is GPU-dirty; nothing served here does that for bytes it would write
			// after a wait for the GPU. Without that copy the request is its own pages, as it
			// was before there were pages ahead.
			result.size -= ahead_bytes;
			handed = HandOffGuestCopy(request, buffer, buffer_id, result.base, result.size);
		}
		if (handed) {
			// The requester does the rest, tracker state included, and starts now: the other
			// requests of the batch may wait for the GPU.
			result.skip   = true;
			result.handed = true;
			request.done->release();
			continue;
		}
		AwaitSubmittedDownloads(result.base, result.size);
		if (ReadBackInPlace(buffer, result.base, result.size)) {
			// Published already: only the tracker state below remains.
			result.copied = true;
			Common::GpuTiming::Sync().AddInPlaceReadback();
			continue;
		}
		result.copied =
		    DownloadBufferMemory<false>(buffer, result.base, result.size, &publications);
		if (result.copied) {
			if (submit_reason) {
				// Rides the submission of an earlier request of the batch.
				Common::GpuTiming::Sync().AddJoinedReadback(request.reason);
			} else {
				submit_reason = request.reason;
			}
		}
	}
	if (!publications.empty()) {
		const Common::GpuTiming::SyncScope sync_reason(*submit_reason);
		const auto                         tick = m_scheduler.CurrentTick();
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		for (auto& publish: publications) {
			publish();
		}
	}
	for (size_t i = 0; i < requests.size(); i++) {
		const auto& result = served[i];
		if (!result.skip) {
			if (result.copied) {
				m_memory_tracker.UnmarkRegionAsGpuModified(result.base, result.size);
			}
			if (requests[i].kind == GuestReadback::Kind::Write) {
				// (Pages ahead of a store are handed over by the requester's own copy alone, see
				// RequestGuestReadback.)
				m_memory_tracker.MarkRegionAsCpuModified(requests[i].vaddr, requests[i].size);
				QueueBdaSync(requests[i].vaddr, requests[i].size);
			}
		}
		if (requests[i].done != nullptr && !result.handed) {
			requests[i].done->release();
		}
	}
}

// The hot set: the pages the game reads back every frame (a town of Ghost of Yotei faults about
// 500 distinct pages per frame). When full, the coldest eighth is evicted at once, so an insert
// costs O(1) amortized instead of a scan of the whole set.
static constexpr uint64_t MaxHotTouchPages = 16;

void BufferCache::TouchHotPages(uint64_t vaddr, uint64_t size, uint64_t most_pages) {
	const auto first = vaddr & ~(CACHING_PAGESIZE - 1u);
	const auto last  = std::min(Common::AlignUp(vaddr + size, CACHING_PAGESIZE),
	                            first + std::max(most_pages, MaxHotTouchPages) * CACHING_PAGESIZE);
	const auto touch = ++m_hot_touch_sequence;
	for (auto page = first; page < last; page += CACHING_PAGESIZE) {
		if (auto it = m_hot_pages.find(page); it != m_hot_pages.end()) {
			it->second.touched = touch;
			it->second.faulted = touch;
			continue;
		}
		if (m_hot_pages.size() >= std::max<size_t>(g_hot_pages_max.Get(), 64)) {
			EvictColdHotPages(touch);
		}
		m_hot_pages.emplace(page, HotPage {.touched = touch,
		                                   .faulted = touch,
		                                   .epoch   = std::make_shared<std::atomic<uint64_t>>(0)});
	}
}

void BufferCache::EvictColdHotPages(uint64_t keep) {
	auto& touched = m_hot_eviction_scratch;
	touched.clear();
	for (const auto& [page, hot]: m_hot_pages) {
		(void)page;
		touched.push_back(hot.touched);
	}
	const auto coldest = touched.begin() + static_cast<ptrdiff_t>(touched.size() / 8u);
	std::nth_element(touched.begin(), coldest, touched.end());
	// Pages touched by the current request stay.
	const auto threshold = std::min(*coldest, keep - 1u);
	std::erase_if(m_hot_pages,
	              [&](const auto& entry) { return entry.second.touched <= threshold; });
}

void BufferCache::NoteGpuWrite(uint64_t vaddr, uint64_t size) {
	// A copy a guest thread is making of bytes this write may change stops publishing: this
	// comes before the write is marked GPU-dirty (SynchronizeBuffer), which waits for the
	// tracking region's lock, so a page the copy publishes was read before the write existed.
	if (!m_guest_copies.empty()) {
		std::erase_if(m_guest_copies, [&](const std::shared_ptr<GuestCopy>& copy) {
			if (copy->finished.load(std::memory_order_acquire)) {
				return true;
			}
			if (copy->base < vaddr + size && vaddr < copy->base + copy->size) {
				copy->current.store(false, std::memory_order_release);
			}
			return false;
		});
	}
	if (m_hot_pages.empty()) {
		return;
	}
	const auto first = vaddr & ~(CACHING_PAGESIZE - 1u);
	const auto touch = ++m_hot_touch_sequence;
	for (auto it = m_hot_pages.lower_bound(first);
	     it != m_hot_pages.end() && it->first < vaddr + size; ++it) {
		// A page served asynchronously never faults again, so the GPU writes keep it hot; it
		// ages out once the GPU stops writing it.
		it->second.touched = touch;
		// The epoch advances before the write is marked GPU-dirty (SynchronizeBuffer): a publish
		// that reads the old epoch under the region lock completes before the mark and a later
		// one sees the new epoch and keeps the page dirty.
		it->second.epoch->fetch_add(1, std::memory_order_acq_rel);
		m_pending_async.Add(it->first, CACHING_PAGESIZE);
		// A clean page keeps its (current) bytes readable while the recorded write is pending;
		// a page still holding an unpublished write must fault into a readback instead.
		// (stale_readable 0: every page faults until its download is published, and a read
		// that comes before counts as an early read.)
		if (!g_stale_readable.On() ||
		    m_memory_tracker.IsRegionGpuModified(it->first, CACHING_PAGESIZE)) {
			m_make_unreadable.push_back(it->first);
		} else {
			m_keep_readable.push_back(it->first);
		}
	}
}

void BufferCache::DrainPublished() {
	if (!m_published_pending.load(std::memory_order_acquire)) {
		return;
	}
	std::vector<Published> published;
	{
		std::lock_guard lock(m_published_mutex);
		published.swap(m_published);
		m_published_pending.store(false, std::memory_order_release);
	}
	// A published tracker page that is still clean holds no GPU write recorded since the
	// download (recording one marks the page first), so all its dirty intervals were published.
	// One that is dirty again keeps them: the download that cleans it takes the whole page. The
	// caching page's epoch cannot decide this: a write to another tracker page of the caching
	// page moves it (and the published bytes stayed dirty with a clean tracker page, so no
	// download ever took them), and a hot page evicted and written again starts a new epoch.
	for (const auto& entry: published) {
		for (auto page = entry.address; page < entry.address + entry.size;
		     page += TRACKER_PAGE_SIZE) {
			if (!m_memory_tracker.IsRegionGpuModified(page, TRACKER_PAGE_SIZE)) {
				m_gpu_modified_ranges.Subtract(page, TRACKER_PAGE_SIZE);
			}
		}
	}
}

bool BufferCache::HandOffGuestCopy(const GuestReadback& request, Buffer& buffer, BufferId id,
                                   uint64_t base, uint64_t size) {
	if (!g_guest_copy.On()) {
		return false;
	}
	auto copy  = std::make_shared<GuestCopy>();
	copy->base = base;
	copy->size = size;
	// Downloads of every GPU-dirty byte are submitted and still running: the requester waits
	// for them. (Finished ones are published by now: the request is served here, without a wait.)
	if (const auto tick = SubmittedDownloadTick(base, size); tick != 0) {
		if (m_scheduler.IsFree(tick)) {
			return false;
		}
		copy->wait_tick = tick;
		g_guest_copy_waits.Add();
		*request.handoff = std::move(copy);
		return true;
	}
	// The bytes are read from the mapping: whole tracking pages (their GPU-dirty state ends with
	// the copy), a write that is submitted, and not more bytes than the bus reads are worth.
	const auto write_tick = buffer.gpu_write_tick;
	if (((base | size) & (TRACKER_PAGE_SIZE - 1u)) != 0 || !buffer.IsHostReadable() ||
	    buffer.gpu_write_open || write_tick >= m_scheduler.CurrentTick() ||
	    !buffer.IsInBounds(base, size)) {
		return false;
	}
	uint64_t dirty = 0;
	m_gpu_modified_ranges.ForEachInRange(base, size, [&](uint64_t begin, uint64_t end) {
		copy->intervals.push_back({begin, end - begin});
		dirty += end - begin;
	});
	if (dirty == 0 || dirty > (uint64_t {g_guest_copy_max_kb.Get()} << 10u)) {
		return false;
	}
	copy->buffer         = &buffer;
	copy->buffer_id      = id;
	copy->buffer_address = buffer.CpuAddress();
	copy->wait_tick      = write_tick;
	ForgetColorMetadataChecks(base, size);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferDownloadBytes, dirty);
	Common::GpuTiming::Sync().AddInPlaceReadback();
	g_guest_copies.Add();
	g_guest_copy_bytes.Add(dirty);
	std::erase_if(m_guest_copies, [](const std::shared_ptr<GuestCopy>& other) {
		return other->finished.load(std::memory_order_acquire);
	});
	m_guest_copies.push_back(copy);
	*request.handoff = std::move(copy);
	return true;
}

void BufferCache::RunGuestCopy(GuestCopy& copy) {
	// As ReadBackInPlace: the write, then the publications of earlier downloads of these bytes.
	(void)m_scheduler.GetMasterSemaphore().Wait(copy.wait_tick);
	m_scheduler.WaitPriorityOperations(copy.wait_tick);
	if (copy.buffer != nullptr) {
		const auto& intervals = copy.intervals;
		uint64_t    largest   = 0;
		for (const auto& interval: intervals) {
			largest = std::max(largest, interval.size);
		}
		thread_local std::vector<uint8_t> scratch;
		scratch.resize(static_cast<size_t>(largest));
		std::vector<Published> published;
		published.reserve(static_cast<size_t>(copy.size / TRACKER_PAGE_SIZE));
		const auto end  = copy.base + copy.size;
		size_t     next = 0;
		// A caching page at a time: the region's lock is held for one page's bus reads.
		for (auto chunk = copy.base; chunk < end && next < intervals.size();
		     chunk += CACHING_PAGESIZE) {
			const auto chunk_end = std::min(chunk + CACHING_PAGESIZE, end);
			while (next < intervals.size() &&
			       intervals[next].address + intervals[next].size <= chunk) {
				++next;
			}
			if (next == intervals.size() || intervals[next].address >= chunk_end) {
				continue;
			}
			// Of the chunk, the runs of pages this copy holds bytes for, and no other page. A
			// page that is GPU-dirty in the tracker and has no bytes here is another request's:
			// the GPU thread recorded a download of it before this copy was handed over (its
			// byte ranges left the set then) and publishes it after its wait for the GPU.
			// Cleaned here, the page took a store without a fault, and that publication then
			// wrote the GPU's old bytes over it.
			size_t at = next;
			for (auto page = chunk; page < chunk_end;) {
				while (at < intervals.size() && intervals[at].address + intervals[at].size <= page) {
					++at;
				}
				if (at == intervals.size() || intervals[at].address >= chunk_end) {
					break;
				}
				const auto run_begin =
				    std::max(page, Common::AlignDown(intervals[at].address, TRACKER_PAGE_SIZE));
				auto   run_end = run_begin;
				size_t scan    = at;
				while (run_end < chunk_end) {
					while (scan < intervals.size() &&
					       intervals[scan].address + intervals[scan].size <= run_end) {
						++scan;
					}
					if (scan == intervals.size() ||
					    intervals[scan].address >= run_end + TRACKER_PAGE_SIZE) {
						break;
					}
					run_end += TRACKER_PAGE_SIZE;
				}
				m_memory_tracker.PublishGpuModified(
				    run_begin, run_end - run_begin,
				    [&] { return copy.current.load(std::memory_order_acquire); },
				    [&](uint64_t address, uint64_t bytes) noexcept {
					    for (size_t i = at;
					         i < intervals.size() && intervals[i].address < address + bytes; ++i) {
						    const auto from = std::max(address, intervals[i].address);
						    const auto to   = std::min(address + bytes,
						                               intervals[i].address + intervals[i].size);
						    if (from < to) {
							    copy.buffer->ReadMapped(from - copy.buffer_address, to - from,
							                            scratch.data());
							    Libs::LibKernel::Memory::WriteBacking(from, scratch.data(),
							                                          to - from);
						    }
					    }
					    published.push_back({address, bytes});
				    });
				page = run_end;
			}
		}
		if (!published.empty()) {
			std::lock_guard lock(m_published_mutex);
			m_published.insert(m_published.end(), published.begin(), published.end());
			m_published_pending.store(true, std::memory_order_release);
		}
	}
	copy.finished.store(true, std::memory_order_release);
	copy.finished.notify_all();
}

void BufferCache::WaitGuestCopies(BufferId id) {
	std::erase_if(m_guest_copies, [&](const std::shared_ptr<GuestCopy>& copy) {
		if (copy->buffer_id == id) {
			copy->finished.wait(false, std::memory_order_acquire);
		}
		return copy->finished.load(std::memory_order_acquire);
	});
}

uint64_t BufferCache::SubmittedDownloadTick(uint64_t vaddr, uint64_t size) {
	uint64_t wait_tick = 0;
	bool     covered   = true;
	m_gpu_modified_ranges.ForEachInRange(vaddr, size, [&](uint64_t begin, uint64_t end) {
		for (auto page = begin & ~(CACHING_PAGESIZE - 1u); covered && page < end;
		     page += CACHING_PAGESIZE) {
			const auto hot = m_hot_pages.find(page);
			// The epoch moves on every GPU write recorded for the page, so an unchanged one
			// means the download was recorded after the last write and will be published.
			if (hot == m_hot_pages.end() || hot->second.download_tick == 0 ||
			    hot->second.epoch->load(std::memory_order_acquire) != hot->second.download_epoch) {
				covered = false;
			} else {
				wait_tick = std::max(wait_tick, hot->second.download_tick);
			}
		}
	});
	return covered && wait_tick < m_scheduler.CurrentTick() ? wait_tick : 0;
}

void BufferCache::AwaitSubmittedDownloads(uint64_t vaddr, uint64_t size) {
	const auto wait_tick = SubmittedDownloadTick(vaddr, size);
	if (wait_tick == 0) {
		return;
	}
	m_scheduler.Wait(wait_tick);
	m_scheduler.WaitPriorityOperations(wait_tick);
	DrainPublished();
	// Whatever the publications did not clean is read back by the caller as before.
	if (!m_gpu_modified_ranges.Intersects(vaddr, size)) {
		Common::GpuTiming::Sync().AddServedBySubmitted(Common::GpuTiming::CurrentSyncReason());
	}
}

void BufferCache::RecordAsyncDownloads() {
	if (!GuestGpu::IsGpuThread()) {
		return;
	}
	DrainPublished();
	const CommandBuffer::ProfileGuard profile(m_scheduler.Current(),
	                                          CommandBuffer::ProfileScope::Downloads);
	if (m_pending_async.Empty() || m_scheduler.Current().IsInvalid()) {
		return;
	}
	struct Interval {
		uint64_t address;
		uint64_t size;
		uint64_t offset;
		BufferId buffer;
	};
	struct PageRecord {
		uint64_t                               page;
		uint64_t                               epoch;
		std::shared_ptr<std::atomic<uint64_t>> epoch_ptr;
		uint64_t                               faulted;
		size_t                                 first_interval;
		size_t                                 interval_count;
		uint64_t                               bytes;
		// The tracker pages of the caching page the download holds bytes for, a bit each.
		uint32_t tracker_pages;
	};
	static_assert(CACHING_PAGESIZE / TRACKER_PAGE_SIZE <= 32);
	std::vector<Interval>   page_intervals;
	std::vector<PageRecord> pages;
	uint64_t                total = 0;
	m_pending_async.ForEach([&](uint64_t start, uint64_t end) {
		for (auto page = start; page < end; page += CACHING_PAGESIZE) {
			if (!m_memory_tracker.IsRegionGpuModified(page, CACHING_PAGESIZE)) {
				continue;
			}
			const auto  hot   = m_hot_pages.find(page);
			const auto* owner = m_page_table.Find(page >> PageTable::kPageBits);
			if (hot == m_hot_pages.end() || owner == nullptr || !*owner) {
				// Evicted since its write was recorded: no download will refresh its bytes.
				if (m_memory_tracker.IsGpuModifiedReadable(page, CACHING_PAGESIZE)) {
					m_memory_tracker.SetGpuModifiedReadable(page, CACHING_PAGESIZE, false);
				}
				continue;
			}
			PageRecord record {.page           = page,
			                   .epoch          = hot->second.epoch->load(std::memory_order_acquire),
			                   .epoch_ptr      = hot->second.epoch,
			                   .faulted        = hot->second.faulted,
			                   .first_interval = page_intervals.size(),
			                   .interval_count = 0,
			                   .bytes          = 0,
			                   .tracker_pages  = 0};
			m_gpu_modified_ranges.ForEachInRange(
			    page, CACHING_PAGESIZE, [&](uint64_t begin, uint64_t last) {
				    page_intervals.push_back({begin, last - begin, 0, *owner});
				    record.bytes += Common::AlignUp(last - begin, 64);
				    for (auto at = (std::max(begin, page) - page) / TRACKER_PAGE_SIZE;
				         at < CACHING_PAGESIZE / TRACKER_PAGE_SIZE && page + at * TRACKER_PAGE_SIZE < last;
				         ++at) {
					    record.tracker_pages |= 1u << at;
				    }
			    });
			record.interval_count = page_intervals.size() - record.first_interval;
			if (record.interval_count != 0) {
				total += record.bytes;
				pages.push_back(std::move(record));
			}
		}
	});
	m_pending_async.Clear();
	if (pages.empty()) {
		return;
	}
	// A page left out keeps its GPU write pending for the next submission's downloads. Until
	// then a guest read must fault into a readback: a page kept readable while its write was
	// pending would otherwise show stale bytes after the write ran.
	const auto defer = [&](const PageRecord& record) {
		m_pending_async.Add(record.page, CACHING_PAGESIZE);
		if (m_memory_tracker.IsGpuModifiedReadable(record.page, CACHING_PAGESIZE)) {
			m_memory_tracker.SetGpuModifiedReadable(record.page, CACHING_PAGESIZE, false);
		}
	};
	const auto cap      = std::min(m_async_download_cap, m_download_buffer.Size() / 4u);
	uint64_t   deferred = 0;
	if (total > cap) {
		// Pages the guest read back most recently first: the likeliest to be read again soon.
		std::ranges::stable_sort(pages, std::greater {}, &PageRecord::faulted);
		total = 0;
		std::erase_if(pages, [&](const PageRecord& record) {
			if (total + record.bytes <= cap) {
				total += record.bytes;
				return false;
			}
			defer(record);
			++deferred;
			return true;
		});
	}
	uint8_t* mapped = nullptr;
	uint64_t offset = 0;
	if (!pages.empty()) {
		std::tie(mapped, offset) = m_download_buffer.Map(total, 64, false);
	}
	if (mapped == nullptr) {
		// The ring is still held by submissions in flight.
		for (const auto& record: pages) {
			defer(record);
		}
		Common::GpuTiming::Sync().AddAsyncDownloads(0, deferred + pages.size());
		return;
	}
	m_download_buffer.Commit();
	Common::GpuTiming::Sync().AddAsyncDownloads(pages.size(), deferred);
	std::vector<Interval> intervals;
	uint64_t              packed = 0;
	for (const auto& record: pages) {
		for (size_t i = 0; i < record.interval_count; ++i) {
			auto interval   = page_intervals[record.first_interval + i];
			interval.offset = packed;
			packed += Common::AlignUp(interval.size, 64);
			intervals.push_back(interval);
		}
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto native = command.Handle();
	std::sort(intervals.begin(), intervals.end(),
	          [](const Interval& a, const Interval& b) { return a.buffer < b.buffer; });
	std::vector<vk::BufferCopy> copies;
	for (size_t i = 0; i < intervals.size();) {
		const auto  id     = intervals[i].buffer;
		const auto& buffer = m_slot_buffers[id];
		copies.clear();
		for (; i < intervals.size() && intervals[i].buffer == id; ++i) {
			const auto& interval = intervals[i];
			copies.emplace_back(interval.address - buffer.CpuAddress(), offset + interval.offset,
			                    interval.size);
		}
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		before.dstAccessMask = vk::AccessFlagBits::eTransferRead;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
		                       nullptr);
		native.copyBuffer(buffer.Handle(), m_download_buffer.Handle(),
		                  static_cast<uint32_t>(copies.size()), copies.data());
	}
	vk::BufferMemoryBarrier after {};
	after.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask       = vk::AccessFlagBits::eHostRead;
	after.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	after.buffer              = m_download_buffer.Handle();
	after.offset              = offset;
	after.size                = total;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);

	// RecordAsyncDownloads runs right before the submission of the current tick. Only pages whose
	// copies were recorded take the new download; a deferred page keeps its earlier one, which its
	// epoch already rejects once the GPU wrote the page again.
	const auto download_tick = m_scheduler.CurrentTick();
	for (const auto& page: pages) {
		auto& hot          = m_hot_pages.at(page.page);
		hot.download_tick  = download_tick;
		hot.download_epoch = page.epoch;
	}
	uint64_t written_begin = UINT64_MAX;
	uint64_t written_end   = 0;
	for (const auto& interval: intervals) {
		written_begin = std::min(written_begin, interval.address);
		written_end   = std::max(written_end, interval.address + interval.size);
	}
	m_scheduler.DeferPriorityOperation(
	    [this, mapped, offset, total, intervals = std::move(intervals), pages = std::move(pages)] {
		    m_download_buffer.Invalidate(offset, total);
		    std::vector<Published> published;
		    constexpr uint32_t     TrackerPages = CACHING_PAGESIZE / TRACKER_PAGE_SIZE;
		    for (const auto& page: pages) {
			    // Of the caching page, the runs of tracker pages this download holds bytes for,
			    // and no other. A tracker page that is GPU-dirty and has no bytes here was taken
			    // by a readback the GPU thread recorded before this download (its byte ranges had
			    // left the set) and still has to publish: cleaned here without its bytes, the page
			    // could be read and stored into with the GPU's bytes still to come.
			    for (uint32_t first = 0; first < TrackerPages;) {
				    if ((page.tracker_pages & (1u << first)) == 0) {
					    ++first;
					    continue;
				    }
				    uint32_t last = first + 1;
				    while (last < TrackerPages && (page.tracker_pages & (1u << last)) != 0) {
					    ++last;
				    }
				    m_memory_tracker.PublishGpuModified(
				        page.page + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE,
				        [&] { return page.epoch_ptr->load(std::memory_order_acquire) == page.epoch; },
				        [&](uint64_t address, uint64_t bytes) noexcept {
					        for (const auto& interval: intervals) {
						        const auto begin = std::max(address, interval.address);
						        const auto end =
						            std::min(address + bytes, interval.address + interval.size);
						        if (begin < end) {
							        LibKernel::Memory::WriteBacking(
							            begin, mapped + interval.offset + (begin - interval.address),
							            end - begin);
						        }
					        }
					        published.push_back({address, bytes});
				        });
				    first = last;
			    }
		    }
		    if (!published.empty()) {
			    std::lock_guard lock(m_published_mutex);
			    m_published.insert(m_published.end(), published.begin(), published.end());
			    m_published_pending.store(true, std::memory_order_release);
		    }
	    },
	    GuestWrites {written_begin, written_end - written_begin});
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size, bool by_address) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			if (by_address) {
				MarkBdaRead(buffer);
			}
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size, by_address);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, (vaddr < LOWER_ADDRESS_SIZE ? LOWER_ADDRESS_SIZE
				                                       : LibKernel::Memory::kExtendedMemoryBase +
				                                             LibKernel::Memory::kExtendedMemorySize) - end);
			}
			if (expands_right) {
				const auto minimum = vaddr < LOWER_ADDRESS_SIZE
				                         ? CACHING_PAGESIZE * 2
				                         : LibKernel::Memory::kExtendedMemoryBase;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size, bool by_address) {
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

	const auto id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	                          AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress,
	                          overlap.end - overlap.begin, true);
	auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	// Shaders read the new buffer through the page table when they read a buffer it replaces.
	bool read = by_address;
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		read              = read || m_slot_buffers[old_id].bda_use != BdaUse::Unread;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	buffer.bda_use      = read ? BdaUse::Read : BdaUse::Unread;
	buffer.bda_use_tick = m_gc_tick;
	if (overlap.first != overlap.last) {
		// The joined contents arrive by GPU copies recorded now.
		NoteOpenWrite(buffer, id);
	}
	UploadConsumedUnbuffered(buffer);
	Register(id);
	return id;
}

bool BufferCache::UploadCpuModified(Buffer& buffer, uint64_t vaddr, uint64_t size,
                                    bool is_written) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    copies.emplace_back(total_size, buffer.Offset(address), bytes);
		    total_size += bytes;
	    },
	    [&]() noexcept { source = UploadCopies(buffer, copies, total_size); });
	if (source) {
		RecordUpload(buffer, source, copies);
	}
	return !copies.empty();
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	(void)UploadCpuModified(buffer, vaddr, size, is_written);
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

void BufferCache::RecordUpload(Buffer& buffer, vk::Buffer source,
                               std::span<const vk::BufferCopy> copies) {
	// The staging ring and game buffers have device addresses: batched uploads from the ring
	// may be scattered by one dispatch.
	if (source == m_staging_buffer.Handle()) {
		m_scheduler.Current().CopyToBuffer(source, buffer.Handle(), copies,
		                                   m_staging_buffer.BufferDeviceAddress(),
		                                   buffer.BufferDeviceAddress());
	} else {
		m_scheduler.Current().CopyToBuffer(source, buffer.Handle(), copies);
	}
}

vk::Buffer BufferCache::ReserveUploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                            uint64_t total_size, UploadWorker::Job& job) {
	if (copies.empty()) {
		return nullptr;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferUploadBytes, total_size);
	Buffer*  source      = nullptr;
	uint8_t* mapped      = nullptr;
	uint64_t base_offset = 0;
	if (const auto [ring, offset] = m_staging_buffer.Map(total_size, 4); ring != nullptr) {
		m_staging_buffer.Commit();
		source      = &m_staging_buffer;
		mapped      = ring;
		base_offset = offset;
	} else {
		// As UploadMemory: a transfer the ring cannot hold gets a buffer of its own, which
		// lives until the submission that copies from it has completed.
		auto temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 0,
		                                          vk::BufferUsageFlagBits::eTransferSrc, total_size);
		source         = temporary.get();
		mapped         = temporary->Mapped().data();
		m_scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	}
	for (auto& copy: copies) {
		job.copies.push_back(
		    {buffer.CpuAddress() + copy.dstOffset, copy.size, mapped + copy.srcOffset});
		copy.srcOffset += base_offset;
	}
	if (!source->IsCoherent()) {
		job.flushes.push_back({source, base_offset, total_size});
	}
	return source->Handle();
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size) {
	if (copies.empty()) {
		return nullptr;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferUploadBytes, total_size);

	const auto source = UploadMemory(m_graphics, m_scheduler, m_staging_buffer, total_size, 4,
	    vk::BufferUsageFlagBits::eTransferSrc, [&](uint8_t* mapped, uint64_t base_offset) {
		for (auto& copy: copies) {
			const auto address = buffer.CpuAddress() + copy.dstOffset;
			if (!LibKernel::Memory::CopySparseMemory(address, mapped + copy.srcOffset, copy.size)) {
				EXIT("BufferCache: failed to read sparse guest buffer memory\n");
			}
			copy.srcOffset += base_offset;
		}
	});
	return source.first->Handle();
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	EXIT_IF(!(GuestRange {vaddr, size}.Valid()));

	if (!is_written && size <= CACHING_PAGESIZE &&
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
	    m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		// The stores are read before the bytes: one that lands during the copy leaves a stamp
		// no later use matches.
		const auto tick   = m_scheduler.CurrentTick();
		const auto stores = StreamCopyStores(vaddr, size);
		auto&      shared = m_stream_copies[(vaddr >> 4u) % m_stream_copies.size()];
		if (g_stream_share.On() && shared.vaddr == vaddr && shared.size == size &&
		    shared.tick == tick && shared.order == m_stream_copy_order &&
		    shared.stores == stores) {
			g_stream_shared.Add();
			return {&m_stream_buffer, shared.offset};
		}
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
		if (mapped != nullptr) {
			if (!LibKernel::Memory::CopySparseMemory(vaddr, mapped, size)) {
				EXIT("BufferCache: failed to read sparse guest buffer memory\n");
			}
			m_stream_buffer.Commit();
			shared = {vaddr, size, offset, tick, m_stream_copy_order, stores};
			g_stream_copies.Add();
			return {&m_stream_buffer, offset};
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	if (is_written) {
		NoteGpuWrite(vaddr, size);
	}
	(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	m_memory_tracker.SetGpuModifiedReadable(m_make_unreadable, CACHING_PAGESIZE, false);
	m_memory_tracker.SetGpuModifiedReadable(m_keep_readable, CACHING_PAGESIZE, true);
	m_make_unreadable.clear();
	m_keep_readable.clear();
	if (is_written) {
		NoteOpenWrite(buffer, id);
		m_gpu_modified_ranges.Add(vaddr, size);
		ForgetColorMetadataChecks(vaddr, size);
		++m_gpu_write_sequence;
		ForEachWatchedRange(vaddr, size, [&](WatchedRange& watched) {
			watched.sequence = ++m_watched_write_sequence;
			m_watched_write_pending |= watched.read_on_host;
		});
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}
	return ReadGuestBytesForImage(vaddr, size);
}

std::pair<Buffer*, uint64_t> BufferCache::ReadGuestBytesForImage(uint64_t vaddr, uint64_t size) {
	return UploadMemory(m_graphics, m_scheduler, m_staging_buffer, size, 16,
	    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
	    [&](uint8_t* mapped, uint64_t) {
		if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, mapped, size)) {
			EXIT("BufferCache: failed to read mapped guest image backing\n");
		}
	});
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (!is_gds) {
		LibKernel::Memory::SettleLabelAccess(vaddr, size);
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMetaByWrite(vaddr, size);
	if (!IsRegionGpuModified(vaddr, size)) {
		// What the write faults of the guest mapping would do, without taking one per page:
		// cached buffers and images of the range are invalidated and stop watching it.
		(void)m_scheduler.Context().InvalidateMemory(vaddr, size);
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	m_texture_cache.PrepareGpuBufferWrite(vaddr, size, TextureCache::BufferWriteExtent::Exact);
	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if (dst_memory) {
		LibKernel::Memory::SettleLabelAccess(dst_vaddr, size);
	}
	if (src_memory) {
		LibKernel::Memory::SettleLabelAccess(src_vaddr, size);
	}
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	// A copy into whole levels of a GPU-written image lands on its GPU contents, not stale guest
	// bytes, while no buffer holds the range.
	if (src_memory && dst_memory && !IsRegionRegistered(dst_vaddr, size) &&
	    m_texture_cache.CopyToImage(dst_vaddr, src_vaddr, size)) {
		return;
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		// As FillBuffer: invalidate the destination instead of faulting on each of its pages.
		(void)m_scheduler.Context().InvalidateMemory(dst_vaddr, size);
		std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		            size);
		return;
	}

	if (dst_memory) {
		m_texture_cache.PrepareGpuBufferWrite(dst_vaddr, size,
		                                      TextureCache::BufferWriteExtent::Exact);
	}
	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

void BufferCache::MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0) {
			m_checked_color_metadata[slot].Add(vaddr, size);
		}
	}
}

bool BufferCache::IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0 && !m_checked_color_metadata[slot].Contains(vaddr, size)) {
			return false;
		}
	}
	return true;
}

void BufferCache::ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size) {
	for (auto& checked: m_checked_color_metadata) {
		checked.Subtract(vaddr, size);
	}
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

uint64_t BufferCache::WatchedWriteSequence(uint64_t vaddr, uint64_t size) {
	uint64_t sequence = 0;
	bool     watched  = false;
	ForEachWatchedRange(vaddr, size, [&](const WatchedRange& range) {
		watched  = watched || (range.address == vaddr && range.size == size);
		sequence = std::max(sequence, range.sequence);
	});
	if (!watched) {
		// Sorted by address, so a write finds its watched ranges by a search.
		const auto position = std::upper_bound(
		    m_watched_ranges.begin(), m_watched_ranges.end(), vaddr,
		    [](uint64_t address, const WatchedRange& range) { return address < range.address; });
		m_watched_ranges.insert(position, {vaddr, size, 0u});
		m_watched_max_size = std::max(m_watched_max_size, size);
	}
	return sequence;
}

void BufferCache::NoteWatchedRead(uint64_t vaddr, uint64_t size, bool on_host) {
	ForEachWatchedRange(vaddr, size, [&](WatchedRange& watched) {
		if (watched.address == vaddr && watched.size == size) {
			watched.read_on_host = on_host;
		}
	});
}

void BufferCache::NoteHostWrite(uint64_t vaddr, uint64_t size) {
	ForEachWatchedRange(
	    vaddr, size, [&](WatchedRange& watched) { watched.sequence = ++m_watched_write_sequence; });
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	// What a guest thread's own copy or a finished download published is not dirty any more.
	DrainPublished();
	return m_gpu_modified_ranges.Intersects(vaddr, size);
}

bool BufferCache::HasOnlyGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	DrainPublished();
	return m_gpu_modified_ranges.Contains(vaddr, size);
}

void BufferCache::ReadGpuDirtyBytes(std::span<const std::pair<uint64_t, uint64_t>> ranges) {
	const Common::GpuTiming::SyncScope sync_reason(Common::GpuTiming::SyncReason::WalkRead);
	// The dirty intervals of all ranges, each clipped to its range; overlapping ones (ranges that
	// overlap each other) are merged so no byte is read back twice.
	std::vector<std::pair<uint64_t, uint64_t>> dirty;
	for (const auto& [vaddr, size]: ranges) {
		if (size == 0) continue;
		m_gpu_modified_ranges.ForEachInRange(
		    vaddr, size, [&](uint64_t begin, uint64_t end) { dirty.emplace_back(begin, end); });
	}
	if (dirty.empty()) {
		return;
	}
	std::ranges::sort(dirty);
	size_t merged = 0;
	for (const auto& interval: dirty) {
		if (merged != 0 && interval.first <= dirty[merged - 1].second) {
			dirty[merged - 1].second = std::max(dirty[merged - 1].second, interval.second);
		} else {
			dirty[merged++] = interval;
		}
	}
	dirty.resize(merged);
	if (!GuestGpu::IsGpuThread()) {
		for (const auto& [begin, end]: dirty) {
			ReadMemory(begin, end - begin);
		}
		return;
	}
	// One batch: every interval's copy is recorded, then a single wait covers them all.
	std::vector<GuestReadback> requests;
	requests.reserve(dirty.size());
	for (const auto& [begin, end]: dirty) {
		requests.push_back({GuestReadback::Kind::Read, begin, end - begin,
		                    Common::GpuTiming::SyncReason::WalkRead, nullptr});
	}
	ServeGuestReadbacks(requests);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::UploadConsumedUnbuffered(Buffer& buffer) {
	const auto begin = buffer.CpuAddress();
	const auto size  = buffer.Size();
	if (!m_consumed_unbuffered.Intersects(begin, size)) {
		return;
	}
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	m_consumed_unbuffered.ForEachInRange(begin, size, [&](uint64_t first, uint64_t last) {
		copies.emplace_back(total_size, buffer.Offset(first), last - first);
		total_size += last - first;
	});
	m_consumed_unbuffered.Subtract(begin, size);
	if (const auto source = UploadCopies(buffer, copies, total_size)) {
		RecordUpload(buffer, source, copies);
	}
}

bool BufferCache::ConsumeCpuModified(uint64_t vaddr, uint64_t size) {
	// Taking the CPU-modified state of the range also takes it from the buffers that hold the
	// range, so their parts are synchronized first: the guest's stores reach those buffers
	// instead of being dropped with the state. Pages no buffer holds are remembered: a buffer
	// created over them later takes their guest bytes, which their state no longer reports.
	bool modified = false;
	ForEachBufferPart(vaddr, vaddr + size, [&](BufferId id, uint64_t first, uint64_t last) {
		modified |= UploadCpuModified(m_slot_buffers[id], first, last - first, false);
	});
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, false,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    modified = true;
		    m_consumed_unbuffered.Add(address, bytes);
	    },
	    []() noexcept {});
	return modified;
}

void BufferCache::RunGarbageCollector() {
	RetireBdaTables();
	ProbeBdaReaders();
	KeepAccessedBdaBuffers();
	const auto tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		UpdateGcMarks();
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	// Below the memory trigger a cached buffer costs nothing but memory, unless BDA accesses are in
	// use: then every guest store into its pages is synchronized into it at the next command
	// ordering point, and its pages are write-protected again, whether or not anything still reads
	// the buffer. Clean buffers no access has touched for the collection age are retired then too.
	if (m_total_used_memory < m_trigger_gc_memory && !m_bda_in_use) {
		return;
	}

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	const uint64_t plain_age  = aggressive ? 80 : 160;
	const uint64_t age =
	    std::min<uint64_t>(m_bda_in_use ? std::min(plain_age, BdaCollectionAge) : plain_age, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges.Ranges(),
		                                           buffer.CpuAddress(), buffer.Size(),
		                                           "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (m_bda_in_use) {
			// Shaders may still use the buffer through the page table: they report it while it
			// is watched, and only a buffer nobody reported is evicted.
			if (buffer.bda_watch_tick == 0) {
				const bool watched    = IsBdaWatched(buffer);
				buffer.bda_watch_tick = tick + 1;
				if (!watched) {
					WriteBdaEntries(buffer);
				}
				m_bda_watched.push_back(id);
				return ++retire_count == limit;
			}
			if (tick + 1 - buffer.bda_watch_tick < BdaWatchTicks) {
				return false;
			}
		}
		if (dirty) {
			EXIT_NOT_IMPLEMENTED(!DownloadBufferMemory<true>(buffer, buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
		} else {
			// No tracker page of the buffer is GPU-dirty: intervals left here were published
			// (DrainPublished drops them later) and must not outlive the buffer.
			m_gpu_modified_ranges.Subtract(buffer.CpuAddress(), buffer.Size());
			m_memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
			Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferEvictions);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const Common::GpuTiming::SyncScope sync_reason(Common::GpuTiming::SyncReason::GarbageCollect);
	const auto completion_tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		    m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			// GPU-dirty bytes the download did not take (no tracker page claimed them). The
			// buffer stays registered and is collected at a later pass, when its pages are
			// clean, instead of ending the process.
			if (m_gc_deferred++ < 8) {
				LOGF("BufferCache: collection of 0x%016" PRIx64 " (0x%" PRIx64
				     " bytes) deferred: GPU-dirty bytes outside its dirty pages\n",
				     buffer.CpuAddress(), buffer.Size());
			}
			continue;
		}
		m_memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::NoteOpenWrite(Buffer& buffer, BufferId id) {
	if (!buffer.gpu_write_open) {
		buffer.gpu_write_open = true;
		m_open_writes.push_back(id);
	}
}

void BufferCache::NoteCommandBoundary() {
	if (m_open_writes.empty()) {
		return;
	}
	const auto tick = m_scheduler.CurrentTick();
	for (const auto id: m_open_writes) {
		// A slot freed and reused since is stamped late, which only makes a reader wait longer.
		auto& buffer          = m_slot_buffers[id];
		buffer.gpu_write_tick = tick;
		buffer.gpu_write_open = false;
	}
	m_open_writes.clear();
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::QueueBdaSync(uint64_t vaddr, uint64_t size, bool at_ordering_point) {
	const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	std::lock_guard lock(m_bda_sync_mutex);
	m_bda_sync.Queue(begin, end - begin, at_ordering_point);
}

uint64_t BufferCache::BeginBdaStoreEpoch() {
	std::lock_guard lock(m_bda_sync_mutex);
	const auto      epoch = m_bda_sync.BeginStoreEpoch();
	m_bda_store_epoch.store(epoch, std::memory_order_release);
	return epoch;
}

void BufferCache::NoteBdaOrderingPoint(uint64_t epoch) {
	// The ordered epoch belongs to the GPU thread, which also takes (only guest stores queue
	// elsewhere).
	m_bda_sync.NoteOrderingPoint(epoch);
	// The work after the point reads the guest's stores before it: no stream copy is shared
	// across it.
	m_stream_copy_order++;
}

void BufferCache::TakeBdaSync(BdaSyncDue& due) {
	std::lock_guard lock(m_bda_sync_mutex);
	m_bda_sync.Take(due);
}

template <typename Func>
void BufferCache::ForEachBufferPart(uint64_t start, uint64_t finish, Func&& func) {
	// Buffers start and end at page boundaries and never overlap (CreateBuffer joins overlaps),
	// so the page table names the one buffer of a page: one lookup per buffer part, and a map
	// lookup only to skip a gap between buffers.
	for (auto address = start; address < finish;) {
		const auto* owner = address < PageTable::kAddressSpaceSize
		                        ? m_page_table.Find(address >> PageTable::kPageBits)
		                        : nullptr;
		if (owner != nullptr && *owner) {
			const auto& buffer = m_slot_buffers[*owner];
			const auto  last   = std::min(buffer.CpuAddress() + buffer.Size(), finish);
			func(*owner, address, last);
			address = last;
			continue;
		}
		const auto next = m_buffers.lower_bound(address);
		if (next == m_buffers.end() || next->first >= finish) {
			break;
		}
		address = next->first;
	}
}

void BufferCache::SynchronizeBda(const RangeSet& mapped) {
	m_bda_in_use = true;
	// Take before walking: writes queued during synchronization remain pending.
	auto& due = m_bda_due;
	TakeBdaSync(due);
	auto& ranges = m_bda_ranges;
	auto& owners = m_bda_owners;
	ranges.clear();
	owners.clear();
	// The new buffers among the owners (Buffer::bda_new).
	auto& created = m_bda_created;
	created.clear();
	// The due ranges come in address order, so a mapping usually holds several in a row.
	RangeSet::Cursor mapping;
	due.ForEach([&](uint64_t begin, uint64_t end) {
		mapped.ForEachInRange(begin, end - begin, mapping, [&](uint64_t start, uint64_t finish) {
			ForEachBufferPart(start, finish, [&](BufferId id, uint64_t first, uint64_t last) {
				// A buffer no shader reads through the page table keeps the stores as modified
				// pages: its bindings upload what they cover, and MarkBdaRead queues the rest.
				// A new buffer takes its guest memory first (Register queued all of it), so an
				// access through the page table that nothing announced reads the guest's bytes
				// of that time at worst, never memory the buffer was allocated with.
				auto& buffer = m_slot_buffers[id];
				if (m_bda_lazy && buffer.bda_use == BdaUse::Unread && !buffer.bda_new) {
					g_bda_unread.Add();
					return;
				}
				if (buffer.bda_new) {
					created.push_back(id);
				}
				ranges.push_back({first, last - first});
				owners.push_back(id);
			});
		});
	});
	for (const auto id: created) {
		m_slot_buffers[id].bda_new = false;
	}
	due.Clear();
	if (ranges.empty()) {
		return;
	}
	g_bda_syncs.Add();
	g_bda_ranges.Add(ranges.size());
	// One protection change per tracking region for all the written pages (they are typically
	// many small ranges: the guest's writes since the last synchronization), then one upload
	// per buffer.
	auto& copies      = m_bda_copies;
	auto& copy_owners = m_bda_copy_owners;
	auto& sources     = m_bda_sources;
	copies.clear();
	copy_owners.clear();
	sources.clear();
	const auto take = [&](size_t index, uint64_t address, uint64_t bytes) noexcept {
		copies.emplace_back(0, m_slot_buffers[owners[index]].Offset(address), bytes);
		copy_owners.push_back(index);
	};
	// The copies of each buffer get one piece of staging memory: `upload` fills it, or leaves
	// that to the upload worker.
	const auto for_each_buffer = [&](const auto& upload) {
		for (size_t first = 0; first < copies.size();) {
			const auto id    = owners[copy_owners[first]];
			size_t     last  = first;
			uint64_t   total = 0;
			for (; last < copies.size() && owners[copy_owners[last]] == id; last++) {
				copies[last].srcOffset = total;
				total += copies[last].size;
			}
			sources.push_back(
			    upload(m_slot_buffers[id], std::span {copies}.subspan(first, last - first), total));
			first = last;
		}
	};
	if (g_async_upload.On()) {
		// The GPU thread only takes the pages and records the GPU copies (below); the worker
		// write-protects the pages again and then copies them into the staging memory.
		UploadWorker::Job job;
		m_memory_tracker.TakeUploadRanges(ranges, take, job.regions);
		for_each_buffer([&](Buffer& buffer, std::span<vk::BufferCopy> part, uint64_t total) {
			return ReserveUploadCopies(buffer, part, total, job);
		});
		if (!job.regions.empty() || !job.copies.empty()) {
			m_upload_worker.Push(std::move(job));
		}
	} else {
		m_memory_tracker.ForEachUploadRanges(ranges, take, [&]() noexcept {
			for_each_buffer([&](Buffer& buffer, std::span<vk::BufferCopy> part, uint64_t total) {
				return UploadCopies(buffer, part, total);
			});
		});
	}
	g_bda_copies.Add(copies.size());
	size_t group = 0;
	for (size_t first = 0; first < copies.size(); group++) {
		const auto id   = owners[copy_owners[first]];
		size_t     last = first;
		while (last < copies.size() && owners[copy_owners[last]] == id) {
			last++;
		}
		RecordUpload(m_slot_buffers[id], sources[group],
		             std::span {copies}.subspan(first, last - first));
		first = last;
	}
}

} // namespace Libs::Graphics
