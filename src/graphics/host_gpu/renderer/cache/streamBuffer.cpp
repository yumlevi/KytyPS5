#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/gpuTiming.h"
#include "common/liveControl.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstring>
#include <deque>
#include <fmt/format.h>
#include <immintrin.h>
#include <map>
#include <mutex>
#include <numeric>
#include <vk_mem_alloc.h>

namespace Libs::Graphics {

namespace {

// What ReadMapped read in calls of 4 KiB or more and how long that took: read_mapped_us over
// read_mapped_kb is the microseconds a KiB of a buffer's mapping costs to read.
Common::LiveControl::Counter g_read_mapped_kb("read_mapped_kb");
Common::LiveControl::Counter g_read_mapped_us("read_mapped_us");

// The buffers with a GPU address, and the last of them destroyed (DescribeGpuAddress).
struct AddressedBuffer {
	uint64_t                              address     = 0;
	uint64_t                              size        = 0;
	uint64_t                              cpu_address = 0;
	MemoryUsage                           usage       = MemoryUsage::DeviceLocal;
	std::chrono::steady_clock::time_point destroyed {};
};
constexpr size_t                    DESTROYED_KEPT = 65536;
std::mutex                          g_addressed_mutex;
std::map<uint64_t, AddressedBuffer> g_addressed;
std::deque<AddressedBuffer>         g_destroyed;

std::string NameOf(const AddressedBuffer& buffer) {
	return buffer.cpu_address != 0
	           ? fmt::format("the buffer of guest memory {:#x} ({} KiB, GPU address {:#x})",
	                         buffer.cpu_address, buffer.size >> 10u, buffer.address)
	           : fmt::format("an emulator buffer (kind {}, {} KiB, GPU address {:#x})",
	                         static_cast<uint32_t>(buffer.usage), buffer.size >> 10u,
	                         buffer.address);
}

constexpr size_t WATCHES_INITIAL_RESERVE = 0x4000;
constexpr size_t WATCHES_RESERVE_CHUNK   = 0x1000;

constexpr vk::AccessFlags MEMORY_ACCESS =
    vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;

[[nodiscard]] VmaAllocationCreateFlags AllocationFlags(MemoryUsage usage) {
	switch (usage) {
		case MemoryUsage::Upload:
		case MemoryUsage::Stream:
			return VMA_ALLOCATION_CREATE_MAPPED_BIT |
			       VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
		case MemoryUsage::Download:
			return VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
		case MemoryUsage::DeviceLocal: return {};
	}
	return {};
}

[[nodiscard]] VmaMemoryUsage AllocationUsage(MemoryUsage usage) {
	switch (usage) {
		case MemoryUsage::DeviceLocal:
		case MemoryUsage::Stream: return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		case MemoryUsage::Upload:
		case MemoryUsage::Download: return VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	}
	return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
}

[[nodiscard]] bool AlignUp(uint64_t value, uint64_t alignment, uint64_t& result) {
	if (alignment == 0) {
		result = value;
		return true;
	}
	const auto aligned = Common::AlignUp(value, alignment);
	if (aligned < value) {
		return false;
	}
	result = aligned;
	return true;
}

} // namespace

std::string DescribeGpuAddress(uint64_t address, uint64_t precision) {
	std::lock_guard lock(g_addressed_mutex);
	const auto      end = address + std::max<uint64_t>(precision, 1u);
	const auto      overlaps = [&](const AddressedBuffer& buffer) {
		return buffer.address < end && address < buffer.address + buffer.size;
	};
	std::string text;
	const auto  after = g_addressed.upper_bound(address);
	if (after != g_addressed.begin() && overlaps(std::prev(after)->second)) {
		const auto& buffer = std::prev(after)->second;
		text = fmt::format("inside {}, {:#x} bytes in", NameOf(buffer), address - buffer.address);
	} else if (after != g_addressed.end() && overlaps(after->second)) {
		text = fmt::format("at the start of {}", NameOf(after->second));
	}
	// The newest destroyed buffer that held it: a write through an address kept too long.
	const auto now = std::chrono::steady_clock::now();
	for (auto buffer = g_destroyed.rbegin(); buffer != g_destroyed.rend(); ++buffer) {
		if (overlaps(*buffer)) {
			const auto ms =
			    std::chrono::duration_cast<std::chrono::milliseconds>(now - buffer->destroyed).count();
			text += fmt::format("{}inside {}, destroyed {} ms before", text.empty() ? "" : "; also ",
			                    NameOf(*buffer), ms);
			break;
		}
	}
	if (!text.empty()) {
		return text;
	}
	text = "in no buffer the emulator has or lately had";
	if (after != g_addressed.begin()) {
		const auto& below = std::prev(after)->second;
		text += fmt::format("; {:#x} bytes past the end of {}",
		                    address - (below.address + below.size), NameOf(below));
	}
	if (after != g_addressed.end()) {
		text += fmt::format("; {:#x} bytes before {}", after->second.address - address,
		                    NameOf(after->second));
	}
	return text;
}

Buffer::Buffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
               uint64_t cpu_address, vk::BufferUsageFlags flags, uint64_t size, bool host_readable)
    : m_graphics(&graphics), m_scheduler(&scheduler), m_usage(usage), m_cpu_address(cpu_address),
      m_size(size) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(graphics.allocator == nullptr || size == 0);

	vk::BufferCreateInfo buffer_info {};
	buffer_info.size        = size;
	buffer_info.usage       = flags;

	const bool with_bda = bool(flags & vk::BufferUsageFlagBits::eShaderDeviceAddress);
	VmaAllocationCreateInfo allocation_info {};
	allocation_info.flags =
	    VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | AllocationFlags(usage);
	allocation_info.usage = AllocationUsage(usage);
	allocation_info.preferredFlags = usage == MemoryUsage::DeviceLocal
	                                     ? VkMemoryPropertyFlags {}
	                                     : VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	const bool want_mapping        = host_readable && usage == MemoryUsage::DeviceLocal;
	if (want_mapping) {
		allocation_info.flags |=
		    VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
		allocation_info.requiredFlags =
		    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
	}

	// Shaders reach device buffers through device addresses without a cache lookup that could
	// mark them used, so they keep the highest residency priority. Host-side staging and
	// download buffers live in system memory, where the priority does not matter.
	const auto native_info = static_cast<const VkBufferCreateInfo*>(buffer_info);
	const bool device_side = usage == MemoryUsage::DeviceLocal || usage == MemoryUsage::Stream;
	const auto memory_class =
	    device_side ? MemoryPriority::Class::Pinned : MemoryPriority::Class::Cache;
	VmaAllocationInfo allocation_result {};
	VkBuffer          native_buffer = VK_NULL_HANDLE;
	auto              pooled_info   = allocation_info;
	const auto        allocate      = [&] {
		uint32_t memory_type = UINT32_MAX;
		if (device_side && graphics.memory_priority_enabled &&
		    vmaFindMemoryTypeIndexForBufferInfo(graphics.allocator, native_info, &allocation_info,
		                                        &memory_type) != VK_SUCCESS) {
			memory_type = UINT32_MAX;
		}
		pooled_info = allocation_info;
		graphics.PreparePriorityAllocation(pooled_info, memory_class,
		                                   device_side ? memory_type : UINT32_MAX);
		auto result = static_cast<vk::Result>(vmaCreateBuffer(graphics.allocator, native_info,
		                                                      &pooled_info, &native_buffer,
		                                                      &m_allocation, &allocation_result));
		if (result != vk::Result::eSuccess && pooled_info.pool != nullptr) {
			// The pool could not grow; the default pools may still have room.
			pooled_info.pool = nullptr;
			result = static_cast<vk::Result>(vmaCreateBuffer(graphics.allocator, native_info,
			                                                 &pooled_info, &native_buffer,
			                                                 &m_allocation, &allocation_result));
		}
		return result;
	};
	// A device buffer is made in video memory: within the budget if it can be, over it if not.
	// Left to the allocator, a buffer made while the budget was full went to its next memory
	// type, system memory, and stayed there: the GPU then read it over the bus at every draw
	// (a simulated 12 GB card: 2.2 GB of buffers there after a level loaded, 24 fps where the
	// same spot gives 60). Over the budget, Windows moves out what has the lowest priority
	// instead, textures nobody used lately, and the texture collector frees them.
	const bool device_local = usage == MemoryUsage::DeviceLocal;
	if (device_local && !want_mapping) {
		allocation_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	}
	auto result = allocate();
	if (result != vk::Result::eSuccess && want_mapping) {
		// No mappable device-local memory, or no budget left in it: an ordinary device buffer.
		allocation_info.flags &=
		    ~(VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
		allocation_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		result                        = allocate();
	}
	if (result != vk::Result::eSuccess && device_local) {
		allocation_info.flags &= ~static_cast<VmaAllocationCreateFlags>(
		    VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT);
		result = allocate();
		if (result != vk::Result::eSuccess) {
			// No video memory left at all: system memory keeps the game running.
			allocation_info.requiredFlags = 0;
			result                        = allocate();
		}
	}
	if (result != vk::Result::eSuccess) {
		graphics.LogMemoryBudget();
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	if (device_side) {
		m_memory =
		    graphics.TrackAllocation(m_allocation, memory_class, pooled_info.pool != nullptr);
	}

	m_buffer = native_buffer;
	if (with_bda) {
		vk::BufferDeviceAddressInfo address_info {};
		address_info.buffer = m_buffer;
		m_device_address    = graphics.device.getBufferAddress(address_info);
		EXIT_IF(m_device_address == 0);
		std::lock_guard lock(g_addressed_mutex);
		g_addressed[m_device_address] = {m_device_address, size, cpu_address, usage, {}};
	}

	VkMemoryPropertyFlags properties = 0;
	vmaGetAllocationMemoryProperties(graphics.allocator, m_allocation, &properties);
	m_coherent = (properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
	if (device_local && (properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0) {
		static std::atomic<uint64_t> count {0};
		static std::atomic<uint64_t> bytes {0};
		const auto                   total = bytes.fetch_add(size) + size;
		if (const auto n = count.fetch_add(1) + 1; (n & (n - 1)) == 0) {
			LOGF("Warning: a GPU buffer of %" PRIu64 " KiB is in system memory, where the GPU reads "
			     "it over the bus (%" PRIu64 " so far, %" PRIu64 " MiB): video memory is full\n",
			     size >> 10u, n, total >> 20u);
		}
	}
	if (allocation_result.pMappedData != nullptr) {
		m_mapped = {static_cast<uint8_t*>(allocation_result.pMappedData),
		            static_cast<size_t>(size)};
	}
}

Buffer::~Buffer() {
	if (m_device_address != 0) {
		std::lock_guard lock(g_addressed_mutex);
		if (const auto it = g_addressed.find(m_device_address); it != g_addressed.end()) {
			it->second.destroyed = std::chrono::steady_clock::now();
			g_destroyed.push_back(it->second);
			if (g_destroyed.size() > DESTROYED_KEPT) {
				g_destroyed.pop_front();
			}
			g_addressed.erase(it);
		}
	}
	if (m_buffer != nullptr) {
		m_graphics->UntrackAllocation(m_memory, MemoryPriority::Class::Pinned);
		vmaDestroyBuffer(m_graphics->allocator, m_buffer, m_allocation);
	}
}

void Buffer::Flush(uint64_t offset, uint64_t size) {
	EXIT_IF(m_mapped.empty() || offset > Size() || size > Size() - offset);
	if (!IsCoherent() && size != 0) {
		const auto result =
		    vmaFlushAllocation(m_graphics->allocator, m_allocation, offset, size);
		EXIT_NOT_IMPLEMENTED(static_cast<vk::Result>(result) != vk::Result::eSuccess);
	}
}

void Buffer::ReadMapped(uint64_t offset, uint64_t size, void* destination) const {
	EXIT_IF(!IsHostReadable() || offset > Size() || size > Size() - offset);
	if (!IsCoherent() && size != 0) {
		const auto result =
		    vmaInvalidateAllocation(m_graphics->allocator, m_allocation, offset, size);
		EXIT_NOT_IMPLEMENTED(static_cast<vk::Result>(result) != vk::Result::eSuccess);
	}
	const auto* map_begin = m_mapped.data();
	const auto* map_end   = map_begin + m_mapped.size();
	const auto* source    = map_begin + offset;
	const auto* finish    = source + size;
	auto*       target    = static_cast<uint8_t*>(destination);
	const auto timed  = size >= 4096;
	const auto before = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
	const struct Count {
		bool                                  timed;
		std::chrono::steady_clock::time_point before;
		uint64_t                              size;
		~Count() {
			if (timed) {
				g_read_mapped_kb.Add(size >> 10u);
				g_read_mapped_us.Add(static_cast<uint64_t>(
				    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - before)
				        .count()));
			}
		}
	} count {timed, before, size};
	// Streaming loads may be served from fill buffers that predate the GPU's write: the fence
	// orders them after every earlier load and store, so they observe the completed write.
	_mm_mfence();
	// Every byte comes from a streaming load of its aligned 16-byte word, the bytes before the
	// first aligned word and after the last one too: the GPU-written ranges read here are often
	// islands shorter than a word, and a plain load of this (write-combined) memory costs a bus
	// read of its own. Only a word that is not wholly inside the mapping is read with plain
	// loads. (What it comes to in play, by read_mapped_us over read_mapped_kb in a session of
	// Saros: 10.5 to 12.3 us a KiB, 80 to 93 MB/s, and about a gigabyte read that way in two and a
	// half minutes. Reading whole lines through a small cached buffer, a block at a time, was
	// slower: 17.7 to 19.7 us a KiB.)
	while (source < finish) {
		const auto* word = reinterpret_cast<const uint8_t*>(reinterpret_cast<uintptr_t>(source) &
		                                                    ~uintptr_t {15});
		const auto  skip = static_cast<size_t>(source - word);
		const auto  take = std::min<size_t>(16u - skip, static_cast<size_t>(finish - source));
		if (word >= map_begin && word + 16 <= map_end) {
			const auto value =
			    _mm_stream_load_si128(reinterpret_cast<__m128i*>(const_cast<uint8_t*>(word)));
			if (take == 16u) {
				_mm_storeu_si128(reinterpret_cast<__m128i*>(target), value);
			} else {
				alignas(16) uint8_t bytes[16];
				_mm_store_si128(reinterpret_cast<__m128i*>(bytes), value);
				std::memcpy(target, bytes + skip, take);
			}
		} else {
			std::memcpy(target, source, take);
		}
		source += take;
		target += take;
	}
}

void Buffer::Invalidate(uint64_t offset, uint64_t size) {
	EXIT_IF(m_usage != MemoryUsage::Download || offset > Size() || size > Size() - offset);
	if (!IsCoherent() && size != 0) {
		const auto result =
		    vmaInvalidateAllocation(m_graphics->allocator, m_allocation, offset, size);
		EXIT_NOT_IMPLEMENTED(static_cast<vk::Result>(result) != vk::Result::eSuccess);
	}
}

vk::BufferMemoryBarrier Buffer::Barrier(uint64_t offset, uint64_t size, vk::AccessFlags source,
                                        vk::AccessFlags destination) const {
	if (Handle() == nullptr || size == 0 || offset > Size() || size > Size() - offset) {
		EXIT("Buffer: invalid DMA barrier, handle=%p offset=0x%016" PRIx64 " size=0x%016" PRIx64
		     " capacity=0x%016" PRIx64 "\n",
		     static_cast<const void*>(Handle()), offset, size, Size());
	}
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask       = source;
	barrier.dstAccessMask       = destination;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = Handle();
	barrier.offset              = offset;
	barrier.size                = size;
	return barrier;
}

void Buffer::CopyFrom(CommandBuffer& command, const Buffer& source, uint64_t source_offset,
                      uint64_t destination_offset, uint64_t size) {
	if (size == 0 || source_offset > source.Size() || size > source.Size() - source_offset ||
	    destination_offset > Size() || size > Size() - destination_offset) {
		EXIT("Buffer: invalid copy range\n");
	}
	if (source.Handle() == Handle() && source_offset < destination_offset + size &&
	    destination_offset < source_offset + size) {
		EXIT("Buffer: overlapping self-copy\n");
	}
	command.EndRendering();
	const vk::BufferMemoryBarrier before[] = {
	    source.Barrier(source_offset, size, vk::AccessFlagBits::eMemoryWrite,
	                   vk::AccessFlagBits::eTransferRead),
	    Barrier(destination_offset, size, MEMORY_ACCESS, vk::AccessFlagBits::eTransferWrite),
	};
	const auto native = command.Handle();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion,
	                       0, nullptr, 2, before, 0, nullptr);
	const vk::BufferCopy copy {source_offset, destination_offset, size};
	native.copyBuffer(source.Handle(), Handle(), 1, &copy);
	const vk::BufferMemoryBarrier after[] = {
	    source.Barrier(source_offset, size, vk::AccessFlagBits::eTransferRead, LaterAccess),
	    Barrier(destination_offset, size, vk::AccessFlagBits::eTransferWrite, LaterAccess),
	};
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 2, after, 0, nullptr);
}

void Buffer::Fill(uint64_t offset, uint64_t size, uint32_t value) {
	if (((offset | size) & 3u) != 0) {
		EXIT("Buffer: fill range must be dword aligned\n");
	}
	auto& command = Scheduler().Current();
	command.EndRendering();
	const auto before = Barrier(offset, size, MEMORY_ACCESS, vk::AccessFlagBits::eTransferWrite);
	const auto native = command.Handle();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion,
	                       0, nullptr, 1, &before, 0, nullptr);
	native.fillBuffer(Handle(), offset, size, value);
	const auto after = Barrier(offset, size, vk::AccessFlagBits::eTransferWrite, LaterAccess);
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
}

StreamBuffer::StreamBuffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
                           uint64_t size)
    : Buffer(graphics, scheduler, usage, 0,
             AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, size),
      m_current_watches(WATCHES_INITIAL_RESERVE), m_previous_watches(WATCHES_INITIAL_RESERVE) {}

bool StreamBuffer::NormalizeReservation(bool coherent, uint64_t atom, uint64_t& size,
                                        uint64_t& alignment) {
	if (coherent) {
		return true;
	}
	if (!AlignUp(size, atom, size)) {
		return false;
	}
	const auto divisor = std::gcd(alignment, atom);
	if (alignment != 0 && alignment / divisor > UINT64_MAX / atom) {
		return false;
	}
	alignment = alignment == 0 ? atom : alignment / divisor * atom;
	return true;
}

std::pair<uint8_t*, uint64_t> StreamBuffer::Map(uint64_t size, uint64_t alignment,
                                                bool allow_wait) {
	if (Mapped().empty()) {
		return {nullptr, 0};
	}
	uint64_t   mapped_size = size;
	const auto atom        = Graphics().physical_device_properties.limits.nonCoherentAtomSize;
	if (!NormalizeReservation(IsCoherent(), atom, mapped_size, alignment)) {
		return {nullptr, 0};
	}
	if (mapped_size > Size()) {
		return {nullptr, 0};
	}

	uint64_t aligned_offset = 0;
	if (!AlignUp(m_offset, alignment, aligned_offset)) {
		return {nullptr, 0};
	}

	const bool wrap = aligned_offset > Size() - mapped_size;
	if (wrap) {
		aligned_offset = 0;
	}

	auto wait_cursor = wrap ? size_t {0} : m_wait_cursor;
	auto wait_bound  = wrap ? uint64_t {0} : m_wait_bound;
	auto invalidation_mark =
	    wrap ? std::optional<size_t> {m_current_watch_cursor} : m_invalidation_mark;
	auto& pending_watches = wrap ? m_current_watches : m_previous_watches;
	if (!WaitPendingOperations(pending_watches, invalidation_mark, aligned_offset + mapped_size,
	                           allow_wait, wait_cursor, wait_bound)) {
		return {nullptr, 0};
	}

	if (wrap) {
		m_invalidation_mark    = invalidation_mark;
		m_current_watch_cursor = 0;
		std::swap(m_previous_watches, m_current_watches);
	}
	m_wait_cursor = wait_cursor;
	m_wait_bound  = wait_bound;
	m_offset      = aligned_offset;
	m_mapped_size = mapped_size;
	return {Mapped().data() + m_offset, m_offset};
}

void StreamBuffer::Commit() {
	if (Usage() != MemoryUsage::Download && m_mapped_size != 0) {
		Flush(m_offset, m_mapped_size);
	}

	m_offset += m_mapped_size;
	const auto tick = Scheduler().CurrentTick();
	if (m_current_watch_cursor != 0 && m_current_watches[m_current_watch_cursor - 1].tick == tick) {
		m_current_watches[m_current_watch_cursor - 1].upper_bound = m_offset;
		return;
	}
	if (m_current_watch_cursor + 1 >= m_current_watches.size()) {
		m_current_watches.resize(m_current_watches.size() + WATCHES_RESERVE_CHUNK);
	}
	auto& watch       = m_current_watches[m_current_watch_cursor++];
	watch.upper_bound = m_offset;
	watch.tick        = tick;
}

StreamBuffer::Download StreamBuffer::MapDownload(uint64_t size, uint64_t alignment) {
	if (const auto [mapped, offset] = Map(size, alignment); mapped != nullptr) {
		Commit();
		return {this, mapped, offset, nullptr};
	}
	auto owner = std::make_unique<Buffer>(Graphics(), Scheduler(), Usage(), 0, AllFlags, size);
	return {owner.get(), owner->Mapped().data(), 0, std::move(owner)};
}

uint64_t StreamBuffer::Copy(const void* source, uint64_t size, uint64_t alignment) {
	EXIT_IF(source == nullptr);
	const auto [data, offset] = Map(size, alignment);
	EXIT_IF(data == nullptr);
	std::memcpy(data, source, static_cast<size_t>(size));
	Commit();
	return offset;
}

bool StreamBuffer::WaitPendingOperations(const std::vector<Watch>& watches,
                                         std::optional<size_t>     invalidation_mark,
                                         uint64_t requested_upper_bound, bool allow_wait,
                                         size_t& wait_cursor, uint64_t& wait_bound) {
	if (!invalidation_mark.has_value()) {
		return true;
	}
	while (requested_upper_bound > wait_bound && wait_cursor < *invalidation_mark) {
		const auto& watch = watches[wait_cursor];
		if (!Scheduler().IsFree(watch.tick) && !allow_wait) {
			return false;
		}
		const Common::GpuTiming::SyncScope sync_reason(Common::GpuTiming::SyncReason::StreamWrap);
		Scheduler().Wait(watch.tick);
		if (Usage() == MemoryUsage::Download) {
			Scheduler().WaitPriorityOperations(watch.tick);
		}
		wait_bound = watch.upper_bound;
		++wait_cursor;
	}
	return true;
}

} // namespace Libs::Graphics
