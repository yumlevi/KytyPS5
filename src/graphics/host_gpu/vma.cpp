#include "graphics/host_gpu/vulkanCommon.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/upscaler/dlss.h"
#include "graphics/host_gpu/submissionNotes.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <fmt/format.h>
#include <mutex>
#include <string>
#include <vector>

namespace Libs::Graphics {

bool GraphicContext::CreateAllocator() {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(instance == nullptr || physical_device == nullptr || device == nullptr ||
	        allocator != nullptr);

	VmaVulkanFunctions functions {};
	functions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr   = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;

	VmaAllocatorCreateInfo info {};
	info.instance         = instance;
	info.physicalDevice   = physical_device;
	info.device           = device;
	info.pVulkanFunctions = &functions;
	info.vulkanApiVersion = VULKAN_TARGET_API_VERSION;
	info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	if (memory_budget_ext_enabled) {
		info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
	}
	if (memory_priority_enabled) {
		info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT;
	}
	// The allocator's blocks: 64 MiB on a card with less than 13 GiB of video memory
	// (KYTY_SIMULATE_VRAM_MB counts), its own 256 MiB on bigger ones. A card that runs past its
	// budget has part of the game's memory moved to system memory by the driver, block by
	// block: with smaller blocks more of what a frame reads stays in video memory (the DS fork
	// measured 2 to 6% on a card of 8 GB). KYTY_VMA_BLOCK_MB=<n>: n MiB.
	{
		uint64_t video = 0;
		for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
			if (physical_device_memory_properties.memoryHeaps[heap].flags &
			    vk::MemoryHeapFlagBits::eDeviceLocal) {
				video = std::max<uint64_t>(video,
				                           physical_device_memory_properties.memoryHeaps[heap].size);
			}
		}
		if (const char* simulate = std::getenv("KYTY_SIMULATE_VRAM_MB");
		    simulate != nullptr && *simulate != '\0') {
			video = std::min<uint64_t>(video, std::strtoull(simulate, nullptr, 10) << 20u);
		}
		const char*    text  = std::getenv("KYTY_VMA_BLOCK_MB");
		const uint64_t asked = text != nullptr ? std::strtoull(text, nullptr, 10) : 0;
		if (asked != 0) {
			info.preferredLargeHeapBlockSize = asked << 20u;
		} else if (video != 0 && video < (uint64_t {13} << 30u)) {
			info.preferredLargeHeapBlockSize = uint64_t {64} << 20u;
		}
	}

	const auto result = static_cast<vk::Result>(vmaCreateAllocator(&info, &allocator));
	if (result != vk::Result::eSuccess) {
		LOGF("vmaCreateAllocator failed: %s\n", vk::to_string(result).c_str());
		return false;
	}
	TakeSimulationBallast();
	return true;
}

// KYTY_SIMULATE_VRAM_MB=<n>: a card with n MiB of video memory, for tests on a bigger one. A
// limit in the allocator would not do it: what the allocator then puts elsewhere the driver
// still keeps in video memory while there is room. So the rest of the largest device-local
// heap is taken here, at the highest priority, and kept: the driver and Windows then move
// the game's memory to system memory as on such a card.
void GraphicContext::TakeSimulationBallast() {
	const char* text = std::getenv("KYTY_SIMULATE_VRAM_MB");
	if (text == nullptr || *text == '\0') {
		return;
	}
	const uint64_t simulated = std::strtoull(text, nullptr, 10) << 20u;
	const auto&    memory    = physical_device_memory_properties;
	uint32_t       heap      = UINT32_MAX;
	for (uint32_t i = 0; i < memory.memoryHeapCount; i++) {
		if ((memory.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) &&
		    (heap == UINT32_MAX || memory.memoryHeaps[i].size > memory.memoryHeaps[heap].size)) {
			heap = i;
		}
	}
	uint32_t type = UINT32_MAX;
	for (uint32_t i = 0; i < memory.memoryTypeCount && type == UINT32_MAX && heap != UINT32_MAX; i++) {
		if (memory.memoryTypes[i].heapIndex == heap &&
		    memory.memoryTypes[i].propertyFlags == vk::MemoryPropertyFlagBits::eDeviceLocal) {
			type = i;
		}
	}
	if (type == UINT32_MAX || simulated == 0) {
		return;
	}
	const uint64_t size = memory.memoryHeaps[heap].size;
	simulation_ballast_heap = heap;
	for (uint64_t left = simulated < size ? size - simulated : 0; left > 0;) {
		const uint64_t                    chunk = std::min<uint64_t>(left, uint64_t {1} << 30u);
		vk::MemoryPriorityAllocateInfoEXT priority {};
		priority.priority = 1.0f;
		vk::MemoryAllocateInfo info {};
		info.pNext           = memory_priority_enabled ? &priority : nullptr;
		info.allocationSize  = chunk;
		info.memoryTypeIndex = type;
		vk::DeviceMemory taken = nullptr;
		if (device.allocateMemory(&info, nullptr, &taken) != vk::Result::eSuccess) {
			break;
		}
		simulation_ballast.push_back(taken);
		simulation_ballast_bytes += chunk;
		left -= chunk;
	}
	LOGF("Simulated video memory: %" PRIu64 " MiB of heap %u (%" PRIu64
	     " MiB) taken, so %" PRIu64 " MiB are left; priorities %s\n",
	     simulation_ballast_bytes >> 20u, heap, size >> 20u,
	     (size - simulation_ballast_bytes) >> 20u, memory_priority_enabled ? "on" : "off");
}

// The allocator's budgets without the simulation's ballast: the budget and the usage of its
// heap as a card without that memory would report them.
void GraphicContext::HeapBudgets(VmaBudget* budgets) const {
	vmaGetHeapBudgets(allocator, budgets);
	if (simulation_ballast_bytes != 0) {
		auto& heap = budgets[simulation_ballast_heap];
		heap.usage -= std::min<uint64_t>(heap.usage, simulation_ballast_bytes);
		heap.budget -= std::min<uint64_t>(heap.budget, simulation_ballast_bytes);
	}
}

void GraphicContext::DestroyAllocator() {
	if (allocator == nullptr) {
		return;
	}
	DestroyRecycledImages();
	for (auto& pool: priority_pools) {
		if (pool != nullptr) {
			vmaDestroyPool(allocator, pool);
			pool = nullptr;
		}
	}
	for (const auto memory: simulation_ballast) {
		device.freeMemory(memory);
	}
	simulation_ballast.clear();
	simulation_ballast_bytes = 0;
	vmaDestroyAllocator(allocator);
	allocator = nullptr;
}

void GraphicContext::DestroyRecycledImages() {
	std::scoped_lock lock(recycled_images_mutex);
	for (const auto& recycled: recycled_images) {
		UntrackAllocation(recycled.memory, recycled.memory_class);
		vmaDestroyImage(allocator, recycled.image, recycled.allocation);
	}
	recycled_images.clear();
	recycled_image_bytes = 0;
}

uint64_t GraphicContext::TrimRecycledImages(uint64_t bytes) {
	std::scoped_lock lock(recycled_images_mutex);
	uint64_t         freed = 0;
	size_t           count = 0;
	for (; count < recycled_images.size() && freed < bytes; count++) {
		const auto& recycled = recycled_images[count];
		UntrackAllocation(recycled.memory, recycled.memory_class);
		vmaDestroyImage(allocator, recycled.image, recycled.allocation);
		freed += recycled.bytes;
	}
	recycled_images.erase(recycled_images.begin(),
	                      recycled_images.begin() + static_cast<ptrdiff_t>(count));
	recycled_image_bytes -= freed;
	return freed;
}

void GraphicContext::PreparePriorityAllocation(VmaAllocationCreateInfo& info,
                                               MemoryPriority::Class    memory_class,
                                               uint32_t                 memory_type) {
	if (!memory_priority_enabled) {
		return;
	}
	// Only dedicated allocations use it; the others take the priority of their block.
	info.priority = MemoryPriority::Value(MemoryPriority::InUse(memory_class));
	if (memory_class == MemoryPriority::Class::Cache || memory_type >= VK_MAX_MEMORY_TYPES) {
		return;
	}
	std::scoped_lock lock(priority_pools_mutex);
	auto&            pool = priority_pools[memory_type];
	if (pool == nullptr) {
		VmaPoolCreateInfo pool_info {};
		pool_info.memoryTypeIndex = memory_type;
		pool_info.priority        = MemoryPriority::Value(MemoryPriority::Level::High);
		const auto result = static_cast<vk::Result>(vmaCreatePool(allocator, &pool_info, &pool));
		if (result != vk::Result::eSuccess) {
			LOGF("vmaCreatePool failed for memory type %u: %s\n", memory_type,
			     vk::to_string(result).c_str());
			pool = nullptr;
			return;
		}
	}
	info.pool = pool;
}

MemoryPriority::Object* GraphicContext::TrackAllocation(VmaAllocation         allocation,
                                                        MemoryPriority::Class memory_class,
                                                        bool                  pooled) {
	if (!pageable_memory_enabled || allocation == nullptr) {
		return nullptr;
	}
	VmaAllocationInfo2 info {};
	vmaGetAllocationInfo2(allocator, allocation, &info);
	const auto type = info.allocationInfo.memoryType;
	if (type >= physical_device_memory_properties.memoryTypeCount ||
	    !(physical_device_memory_properties.memoryTypes[type].propertyFlags &
	      vk::MemoryPropertyFlagBits::eDeviceLocal)) {
		return nullptr;
	}
	// The priority the memory object was created with: the pool's, the one passed for a
	// dedicated allocation, or the default pools' 0.5.
	const auto initial = pooled || info.dedicatedMemory != VK_FALSE
	                         ? MemoryPriority::InUse(memory_class)
	                         : MemoryPriority::Level::Normal;
	return memory_priority.Acquire(reinterpret_cast<uint64_t>(info.allocationInfo.deviceMemory),
	                               info.blockSize, memory_class, initial);
}

void GraphicContext::UntrackAllocation(MemoryPriority::Object* object,
                                       MemoryPriority::Class   memory_class) {
	memory_priority.Release(object, memory_class, [this](uint64_t memory, float priority) {
		SetMemoryPriority(memory, priority);
	});
}

void GraphicContext::SetMemoryPriority(uint64_t memory, float priority) const {
	device.setMemoryPriorityEXT(vk::DeviceMemory(reinterpret_cast<VkDeviceMemory>(memory)),
	                            priority);
}

void GraphicContext::LogMemoryBudget() const {
	if (allocator == nullptr || physical_device == nullptr) {
		return;
	}

	const auto& properties = physical_device_memory_properties;
	VmaBudget   budgets[VK_MAX_MEMORY_HEAPS] {};
	HeapBudgets(budgets);
	for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
		LOGF("VMA heap %u: usage=%" PRIu64 ", budget=%" PRIu64 ", allocation=%" PRIu64
		     ", blocks=%" PRIu64 "\n",
		     i, static_cast<uint64_t>(budgets[i].usage), static_cast<uint64_t>(budgets[i].budget),
		     static_cast<uint64_t>(budgets[i].statistics.allocationBytes),
		     static_cast<uint64_t>(budgets[i].statistics.blockBytes));
	}
}

// What a report of a lost device needs and the driver knows: how full the memory was, and with
// VK_EXT_device_fault whether the GPU faulted on an address (and which) or stopped answering.
// To the console as well: a launcher's log holds only that.
void GraphicContext::LogLostDevice(uint64_t gpu_tick) const {
	if (device == nullptr) {
		return;
	}
	// One record: the threads that wait for the GPU all fail on a lost device, and each comes
	// here before it ends the process. A later one waits for the record and adds nothing.
	static std::mutex mutex;
	static bool       written = false;
	std::lock_guard   lock(mutex);
	if (written) {
		return;
	}
	written = true;
	using MemoryBudget::MiB;
	if (const auto heaps = SampleMemoryHeaps(); heaps.device.budget != 0) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "Lost device: video memory {} of {} MiB budget ({} MiB of it the driver's own), "
		    "shared memory {} of {} MiB\n",
		    heaps.device.usage / MiB, heaps.device.budget / MiB,
		    MemoryBudget::DriverUsage(heaps.device) / MiB, heaps.host.usage / MiB,
		    heaps.host.budget / MiB));
	}
	// (The failing call names the submission that was lost: an upscaler's commands were in
	// another one unless the ticks match.)
	if (const auto [pictures, tick] = Dlss::Evaluations(); pictures != 0) {
		Log::WriteToConsoleAndLog(
		    fmt::format("Lost device: the upscaler made {} picture(s), the last for tick {}\n",
		                pictures, tick));
	}
	// What the submissions held that the GPU did not finish, the first of them above all: a GPU
	// that stopped answering stopped in that one. (Without the GPU's tick: the newest twelve.)
	uint64_t oldest = 0;
	SubmissionNotes::ForEachNewest(SubmissionNotes::Kept, [&oldest](const auto& note) {
		oldest = oldest == 0 ? note.tick : oldest;
	});
	if (gpu_tick != 0 && oldest > gpu_tick + 1) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "Lost device: the GPU finished submission {}; what {} to {} held is no longer kept\n",
		    gpu_tick, gpu_tick + 1, oldest - 1));
	} else if (gpu_tick != 0) {
		Log::WriteToConsoleAndLog(
		    fmt::format("Lost device: the GPU finished submission {}; what came after it:\n", gpu_tick));
	}
	SubmissionNotes::ForEachNewest(gpu_tick != 0 ? SubmissionNotes::Kept : 12,
	                               [gpu_tick](const SubmissionNotes::Note& note) {
		if (note.tick <= gpu_tick) {
			return;
		}
		std::string shaders;
		for (uint32_t i = 0; i < note.shader_count; i++) {
			shaders += fmt::format("{}{:#018x}", i == 0 ? "" : ", ", note.shaders[i]);
		}
		Log::WriteToConsoleAndLog(fmt::format(
		    "Lost device: submission {} held {} command(s): {} dispatch(es){}{}, {} render pass "
		    "instance(s), {} buffer upload(s){}{}{}\n",
		    note.tick, note.commands, note.dispatches, shaders.empty() ? "" : ", the last of ",
		    shaders, note.passes, note.uploads,
		    (note.scopes & (1u << 5u)) != 0 ? ", texture uploads" : "",
		    (note.scopes & (1u << 6u)) != 0 ? ", downloads" : "",
		    (note.scopes & (1u << 7u)) != 0 ? ", the upscaler" : ""));
	});
	vk::DeviceFaultCountsEXT counts {};
	if (!device_fault_enabled ||
	    device.getFaultInfoEXT(&counts, nullptr) != vk::Result::eSuccess) {
		Log::WriteToConsoleAndLog("Lost device: the driver gave no fault record\n");
		return;
	}
	counts.vendorBinarySize = 0;
	std::vector<vk::DeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
	std::vector<vk::DeviceFaultVendorInfoEXT>  vendors(counts.vendorInfoCount);
	vk::DeviceFaultInfoEXT                     info {};
	info.pAddressInfos = addresses.empty() ? nullptr : addresses.data();
	info.pVendorInfos  = vendors.empty() ? nullptr : vendors.data();
	if (const auto result = device.getFaultInfoEXT(&counts, &info);
	    result != vk::Result::eSuccess && result != vk::Result::eIncomplete) {
		Log::WriteToConsoleAndLog("Lost device: the driver gave no fault record\n");
		return;
	}
	// No address: the GPU did not fault on memory (a hang the OS ended, or the driver itself).
	Log::WriteToConsoleAndLog(
	    fmt::format("Lost device: the driver's record: \"{}\", {} address(es), {} vendor record(s)\n",
	                info.description.data(), counts.addressInfoCount, counts.vendorInfoCount));
	for (uint32_t i = 0; i < counts.addressInfoCount; ++i) {
		Log::WriteToConsoleAndLog(
		    fmt::format("Lost device: {} at {:#x} (to within {:#x} bytes): {}\n",
		                vk::to_string(addresses[i].addressType), addresses[i].reportedAddress,
		                addresses[i].addressPrecision,
		                DescribeGpuAddress(addresses[i].reportedAddress,
		                                   addresses[i].addressPrecision)));
	}
	for (uint32_t i = 0; i < counts.vendorInfoCount; ++i) {
		Log::WriteToConsoleAndLog(
		    fmt::format("Lost device: vendor record \"{}\", code {:#x}, data {:#x}\n",
		                vendors[i].description.data(), vendors[i].vendorFaultCode,
		                vendors[i].vendorFaultData));
	}
}

uint64_t GraphicContext::GetDeviceMemoryUsage() const {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	HeapBudgets(budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t usage = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local =
		    static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                      vk::MemoryHeapFlagBits::eDeviceLocal);
		if (!discrete || device_local) {
			usage += budgets[heap].usage;
		}
	}
	return usage;
}

uint32_t GraphicContext::GetHeapUsage(std::span<HeapUsage, VK_MAX_MEMORY_HEAPS> heaps) const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	HeapBudgets(budgets);
	const auto count = physical_device_memory_properties.memoryHeapCount;
	for (uint32_t heap = 0; heap < count; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		heaps[heap] = {.usage        = budgets[heap].usage,
		               .budget       = budgets[heap].budget,
		               .allocations  = budgets[heap].statistics.allocationCount,
		               .device_local = static_cast<bool>(properties.flags &
		                                                 vk::MemoryHeapFlagBits::eDeviceLocal)};
	}
	return count;
}

GraphicContext::MemoryHeaps GraphicContext::SampleMemoryHeaps() const {
	MemoryHeaps heaps;
	const bool  discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	if (!CanReportMemoryUsage() || allocator == nullptr || !discrete) {
		return heaps;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	HeapBudgets(budgets);
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local =
		    static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                      vk::MemoryHeapFlagBits::eDeviceLocal);
		auto& state = device_local ? heaps.device : heaps.host;
		state.budget += budgets[heap].budget;
		state.usage += budgets[heap].usage;
		state.blocks += budgets[heap].statistics.blockBytes;
	}
	return heaps;
}

void GraphicContext::MonitorMemoryBudget() {
	if (allocator == nullptr) {
		return;
	}
	if (pageable_memory_enabled) {
		memory_priority.Tick(
		    [this](uint64_t memory, float priority) { SetMemoryPriority(memory, priority); });
	}
	if (!CanReportMemoryUsage()) {
		return;
	}
	vmaSetCurrentFrameIndex(allocator, ++memory_budget_frame);
	const auto heaps = SampleMemoryHeaps();
	if (heaps.device.budget == 0) {
		return;
	}
	const auto now_ms =
	    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                              std::chrono::steady_clock::now().time_since_epoch())
	                              .count());
	const auto priorities = memory_priority.GetStats();
	const auto low        = priorities.bytes[static_cast<size_t>(MemoryPriority::Level::Low)];
	const auto report     = memory_budget_monitor.Sample(heaps.device, heaps.host, now_ms, low);
	if (report == MemoryBudget::Monitor::Report::None) {
		return;
	}
	using MemoryBudget::MiB;
	const auto& device = heaps.device;
	LOGF("%sGPU memory: device %" PRIu64 " of %" PRIu64 " MiB budget, driver %" PRIu64
	     " MiB, caches %" PRIu64 " MiB (limit %" PRIu64
	     " MiB); driver memory in the host heap %" PRIu64 " MiB\n",
	     report == MemoryBudget::Monitor::Report::OverBudget
	         ? "Warning: device memory over budget (the OS pages it over the bus; expect low frame "
	           "rates). "
	     : report == MemoryBudget::Monitor::Report::WithinBudget
	         ? "Device memory back within budget. "
	     : report == MemoryBudget::Monitor::Report::Demoted
	         ? "Warning: driver memory grew in the host heap (demoted from device memory?). "
	         : "",
	     device.usage / MiB, device.budget / MiB, MemoryBudget::DriverUsage(device) / MiB,
	     device.blocks / MiB, MemoryBudget::CacheLimit(device) / MiB,
	     MemoryBudget::DriverUsage(heaps.host) / MiB);
	if (pageable_memory_enabled) {
		LOGF("GPU memory priorities: high %" PRIu64 " MiB, normal %" PRIu64 " MiB, low %" PRIu64
		     " MiB in %zu memory objects, %" PRIu64 " changes\n",
		     priorities.bytes[static_cast<size_t>(MemoryPriority::Level::High)] / MiB,
		     priorities.bytes[static_cast<size_t>(MemoryPriority::Level::Normal)] / MiB, low / MiB,
		     memory_priority.ObjectCount(), priorities.changes);
	}
}

uint64_t GraphicContext::GetTotalMemoryBudget() const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	HeapBudgets(budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t budget = 0;
	uint64_t local  = 0;
	uint64_t usage  = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		const bool  device_local =
		    static_cast<bool>(properties.flags & vk::MemoryHeapFlagBits::eDeviceLocal);
		if (device_local) {
			local += properties.size;
		}
		if (!discrete || device_local) {
			budget += CanReportMemoryUsage() ? budgets[heap].budget : properties.size;
			usage += CanReportMemoryUsage() ? budgets[heap].usage : 0;
		}
	}
	if (discrete) {
		return budget - std::min<uint64_t>(budget / 8, 1024ull * 1024 * 1024);
	}
	constexpr uint64_t system_reserve = 8ull * 1024 * 1024 * 1024;
	const auto         available      = budget > usage ? budget - usage : uint64_t {0};
	return std::max(local, available > system_reserve ? available - system_reserve : uint64_t {0});
}

MemoryBudget::LiveMarks GraphicContext::UpdateCacheMarks() {
	const auto budget = GetTotalMemoryBudget();
	const auto now_ms =
	    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                              std::chrono::steady_clock::now().time_since_epoch())
	                              .count());
	std::scoped_lock lock(cache_marks_mutex);
	// No budget before the allocator exists: the caches keep their initial marks.
	if (budget != 0) {
		(void)cache_marks.Update(budget, now_ms);
	}
	return cache_marks;
}

bool GraphicContext::CreateImage(const vk::ImageCreateInfo& image_info, VulkanImage& image,
                                 MemoryPriority::Class memory_class) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);

	{
		std::scoped_lock lock(recycled_images_mutex);
		const auto       recycled = std::find_if(
		    recycled_images.begin(), recycled_images.end(), [&](const RecycledImage& candidate) {
			    return candidate.memory_class == memory_class && candidate.info == image_info;
		    });
		if (recycled != recycled_images.end()) {
			image.image      = recycled->image;
			image.allocation = recycled->allocation;
			image.memory     = recycled->memory;
			recycled_image_bytes -= recycled->bytes;
			recycled_images.erase(recycled);
		}
	}
	if (image.image == nullptr) {
		const auto native_info = static_cast<const vk::ImageCreateInfo::NativeType*>(image_info);
		const auto create      = [&](VkMemoryPropertyFlags required, bool pooled) {
			VmaAllocationCreateInfo alloc_info {};
			alloc_info.requiredFlags  = required;
			alloc_info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
			uint32_t memory_type      = UINT32_MAX;
			if (pooled && memory_priority_enabled && memory_class != MemoryPriority::Class::Cache &&
			    vmaFindMemoryTypeIndexForImageInfo(allocator, native_info, &alloc_info,
			                                       &memory_type) != VK_SUCCESS) {
				memory_type = UINT32_MAX;
			}
			PreparePriorityAllocation(alloc_info, memory_class, pooled ? memory_type : UINT32_MAX);

			vk::Image::CType native_image = VK_NULL_HANDLE;
			const auto       result       = static_cast<vk::Result>(vmaCreateImage(
			    allocator, native_info, &alloc_info, &native_image, &image.allocation, nullptr));
			image.image                   = native_image;
			if (result != vk::Result::eSuccess) {
				return false;
			}
			image.memory =
			    TrackAllocation(image.allocation, memory_class, alloc_info.pool != nullptr);
			return true;
		};
		bool created = create(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);
		if (!created) {
			// Device memory is exhausted. The recycled backings are the only memory freed
			// right away; next, any memory type the image supports (system memory over the bus)
			// keeps the title running while the caches release what they no longer use.
			DestroyRecycledImages();
			created = create(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
			if (!created) {
				created          = create(0, false);
				const auto count = created ? ++image_fallback_count : uint64_t {0};
				if (count % 256 == 1) {
					LOGF("Image placed outside device-local memory (%" PRIu64 " so far)\n", count);
					LogMemoryBudget();
				}
			}
		}
		if (!created) {
			LogMemoryBudget();
			return false;
		}
	}

	image.format     = image_info.format;
	image.image_type = image_info.imageType;
	image.extent     = image_info.extent;
	image.layers     = image_info.arrayLayers;
	image.mip_levels = image_info.mipLevels;
	image.samples    = static_cast<uint32_t>(image_info.samples);
	image.usage      = image_info.usage;
	image.flags      = image_info.flags;
	image.create_info = image_info;
	image.memory_class = memory_class;
	image.state      = {.layout = image_info.initialLayout};
	image.subresource_states.clear();

	return true;
}

void GraphicContext::DeleteImage(VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image == nullptr || image.allocation == nullptr);

	VmaAllocationInfo allocation_info {};
	vmaGetAllocationInfo(allocator, image.allocation, &allocation_info);
	const uint64_t bytes = allocation_info.size;
	// Only a chain-free create info can be compared by value.
	if (image.create_info.pNext == nullptr && bytes <= RecycledImageBytesLimit) {
		std::scoped_lock lock(recycled_images_mutex);
		while (!recycled_images.empty() && (recycled_images.size() >= RecycledImageLimit ||
		                                    recycled_image_bytes + bytes > RecycledImageBytesLimit)) {
			const auto& oldest = recycled_images.front();
			UntrackAllocation(oldest.memory, oldest.memory_class);
			vmaDestroyImage(allocator, oldest.image, oldest.allocation);
			recycled_image_bytes -= oldest.bytes;
			recycled_images.erase(recycled_images.begin());
		}
		recycled_images.push_back({.info         = image.create_info,
		                           .image        = image.image,
		                           .allocation   = image.allocation,
		                           .bytes        = bytes,
		                           .memory       = image.memory,
		                           .memory_class = image.memory_class});
		recycled_image_bytes += bytes;
	} else {
		UntrackAllocation(image.memory, image.memory_class);
		vmaDestroyImage(allocator, image.image, image.allocation);
	}
	image.image      = nullptr;
	image.allocation = nullptr;
	image.memory     = nullptr;
}

} // namespace Libs::Graphics
