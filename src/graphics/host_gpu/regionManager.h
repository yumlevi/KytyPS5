#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_

#include "common/assert.h"
#include "graphics/host_gpu/backoffSpinLock.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <atomic>
#include <mutex>
#include <span>
#include <utility>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h>
#undef MemoryBarrier
#elif defined(__APPLE__)
#include <pthread.h>
#elif defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Libs::Graphics {

class TrackingSpinLock final {
public:
	void lock() noexcept {
		const auto thread = CurrentThread();
		// (Not recursive: the thread that holds it would wait for itself.)
		EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) == thread);
		m_lock.lock();
		m_owner.store(thread, std::memory_order_relaxed);
	}
	void unlock() noexcept {
		EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) != CurrentThread());
		m_owner.store(0, std::memory_order_relaxed);
		m_lock.unlock();
	}

private:
	static uint32_t CurrentThread() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		return GetCurrentThreadId();
#elif defined(__APPLE__)
		// mach thread port is a nonzero per-thread id (0 is the "no owner" sentinel).
		static thread_local const uint32_t tid =
		    static_cast<uint32_t>(pthread_mach_thread_np(pthread_self()));
		return tid;
#elif defined(__linux__)
		static thread_local const uint32_t tid = static_cast<uint32_t>(::syscall(SYS_gettid));
		return tid;
#else
		EXIT("region tracking thread identity is unsupported on this platform\n");
#endif
	}

	BackoffSpinLock      m_lock;
	std::atomic_uint32_t m_owner {0};
};

static_assert(std::atomic_uint32_t::is_always_lock_free);

class RegionManager final {
public:
	RegionManager(PageManager& page_manager, uint64_t cpu_addr)
	    : m_page_manager(page_manager), m_cpu_addr(cpu_addr) {
		m_cpu_dirty.Fill();
		m_writable.Fill();
		m_readable.Fill();
	}

	KYTY_CLASS_NO_COPY(RegionManager);

	template <DirtySource source>
	[[nodiscard]] bool IsModified(uint64_t offset, uint64_t size) const {
		const auto [start, end] = GetPageRange(offset, size);
		return GetBits<source>().Any(start, end);
	}

	// Answers a query of the tracking state without the lock, as a query under the lock would,
	// or returns false when the state changed meanwhile (the caller then takes the lock). The
	// GPU thread asks such questions for every buffer it binds while guest threads take the lock
	// for their write faults; a query that does not write the lock does not contend with them.
	template <typename Query>
	[[nodiscard]] bool TryQueryUnlocked(Query&& query, bool& result) const {
		const auto version = m_version.load(std::memory_order_acquire);
		if ((version & 1u) != 0u) {
			return false;
		}
		result = query();
		std::atomic_thread_fence(std::memory_order_acquire);
		return m_version.load(std::memory_order_relaxed) == version;
	}

	template <DirtySource source, bool enable>
	void ChangeState(uint64_t offset, uint64_t size) {
		const auto [start, end] = GetPageRange(offset, size);
		if constexpr (source == DirtySource::Cpu && enable) {
			if (m_gpu_dirty.Any(start, end)) {
				EXIT("CPU dirty state conflicts with GPU dirty state\n");
			}
		}
		const StateChange change(*this);
		auto&             bits = GetBits<source>();
		if constexpr (enable) {
			bits.SetRange(start, end);
		} else {
			bits.UnsetRange(start, end);
			if constexpr (source == DirtySource::Gpu) {
				m_stale_readable.UnsetRange(start, end);
			}
		}
		if constexpr (source == DirtySource::Cpu) {
			UpdateProtection<!enable, false>();
		} else {
			UpdateProtection<enable, true>();
		}
	}

	[[nodiscard]] uint64_t Address() const noexcept { return m_cpu_addr; }

	// A GPU write of the range is recorded: its pages become GPU-dirty, and those of `keep` that
	// were clean stay readable with their bytes of before (stale readable, see SetStaleReadable)
	// in the same protection update. Marked first and made readable again afterwards, each such
	// page took two protection requests that cancel (with the hot set of 16,384 pages about
	// eight page changes per page a frame, a millisecond of the GPU thread). A page that was
	// GPU-dirty already holds a write whose bytes are not in guest memory: it is not kept.
	void MarkGpuWritten(uint64_t offset, uint64_t size, const RegionBits* keep) {
		const auto [start, end] = GetPageRange(offset, size);
		RegionBits range;
		range.SetRange(start, end);
		const auto        fresh = range & ~m_gpu_dirty;
		const StateChange change(*this);
		m_gpu_dirty.SetRange(start, end);
		if (keep != nullptr) {
			m_stale_readable |= fresh & *keep;
		}
		UpdateProtection<true, true>();
	}

	// A GPU-dirty page whose guest bytes are still current (every earlier GPU write of it was
	// published before the pending one was recorded) stays readable: the guest sees what the
	// console would show before the pending write executes, and the publish of that write
	// refreshes the bytes in place. Pages that lose that property become unreadable again.
	void SetStaleReadable(uint64_t offset, uint64_t size, bool readable) {
		const auto [start, end] = GetPageRange(offset, size);
		if (readable) {
			if (RegionBits(m_gpu_dirty, start, end).None()) {
				return;
			}
			const StateChange change(*this);
			// Only the GPU-dirty pages of the range: the bit is taken back with the dirty state
			// (ChangeState, ForEachModifiedRange), so on a clean page it would stay, and a later
			// GPU write of that page, with no download to bring its bytes, would leave it
			// readable with the bytes of before for good.
			m_stale_readable |= RegionBits(m_gpu_dirty, start, end);
			UpdateProtection<false, true>();
		} else {
			const StateChange change(*this);
			m_stale_readable.UnsetRange(start, end);
			UpdateProtection<true, true>();
		}
	}
	// SetStaleReadable for each page (a guest address) of `page_size` bytes, with one protection
	// update.
	void SetStaleReadable(std::span<const uint64_t> pages, uint64_t page_size, bool readable) {
		const StateChange change(*this);
		for (const auto page: pages) {
			const auto [start, end] = GetPageRange(page - m_cpu_addr, page_size);
			if (!readable) {
				m_stale_readable.UnsetRange(start, end);
			} else {
				// (Only the GPU-dirty pages, as above.)
				m_stale_readable |= RegionBits(m_gpu_dirty, start, end);
			}
		}
		if (readable) {
			UpdateProtection<false, true>();
		} else {
			UpdateProtection<true, true>();
		}
	}
	[[nodiscard]] bool IsStaleReadable(uint64_t offset, uint64_t size) const {
		const auto [start, end] = GetPageRange(offset, size);
		return m_stale_readable.FirstRangeFrom(start).first < end;
	}

	template <DirtySource source, bool clear, typename Func>
	void ForEachModifiedRange(uint64_t offset, uint64_t size, Func&& func) {
		const auto [start, end] = GetPageRange(offset, size);
		auto&      bits         = GetBits<source>();
		if (!bits.Any(start, end)) {
			return;
		}
		RegionBits mask(bits, start, end);
		if constexpr (clear && source == DirtySource::Gpu) {
			// A publication: the bytes first (they go through the backing, whatever the pages'
			// protection), then the pages. Made readable first, a page showed a guest thread
			// that read it at that moment the bytes of before the GPU's write.
			for (const auto [first, last]: mask) {
				func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
			}
			const StateChange change(*this);
			bits.UnsetRange(start, end);
			m_stale_readable.UnsetRange(start, end);
			UpdateProtection<false, true>();
			return;
		}
		if constexpr (clear) {
			const StateChange change(*this);
			bits.UnsetRange(start, end);
			if constexpr (source == DirtySource::Gpu) {
				m_stale_readable.UnsetRange(start, end);
			}
			if constexpr (source == DirtySource::Cpu) {
				UpdateProtection<true, false>();
			} else {
				UpdateProtection<false, true>();
			}
		}
		for (const auto [first, last]: mask) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}

	// ForEachModifiedRange<DirtySource::Cpu, true> without the protection update: the caller
	// takes the CPU-modified pages of several ranges of this region under one hold of `lock`, then
	// write-protects them all with ProtectCpuClean before it releases the lock and copies them.
	template <typename Func>
	void TakeCpuModifiedRanges(uint64_t offset, uint64_t size, Func&& func) {
		const auto [start, end] = GetPageRange(offset, size);
		if (m_cpu_dirty.FirstRangeFrom(start).first >= end) {
			return;
		}
		RegionBits mask(m_cpu_dirty, start, end);
		{
			const StateChange change(*this);
			m_cpu_dirty.UnsetRange(start, end);
		}
		for (const auto [first, last]: mask) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}
	// Write-protects the pages TakeCpuModifiedRanges took. They can lie anywhere in the region:
	// one request per cluster of them, since a request costs about as much as reaching over
	// MaxProtectionGap pages whose protection it does not change (Windows: ~1.5 us per request,
	// ~25 ns per page).
	// The caller may also release the lock in between and call this later, from another thread
	// (the upload worker): until then the taken pages are clean and still writable, a guest
	// store into them takes no fault, and the copy that follows this call carries it.
	void ProtectCpuClean() {
		const StateChange change(*this);
		const auto        mask = m_writable & ~m_cpu_dirty;
		if (mask.None()) {
			return;
		}
		m_writable &= m_cpu_dirty;
		RegionBits cluster;
		size_t     cluster_end = 0;
		bool       open        = false;
		for (const auto [first, last]: mask) {
			if (open && first - cluster_end > MaxProtectionGap) {
				m_page_manager.UpdatePageWatchersForRegion<true, false>(m_cpu_addr, cluster);
				cluster.Clear();
			}
			cluster.SetRange(first, last);
			cluster_end = last;
			open        = true;
		}
		m_page_manager.UpdatePageWatchersForRegion<true, false>(m_cpu_addr, cluster);
	}
	static constexpr size_t MaxProtectionGap = 64;

	// The pages after the page of a guest write fault at `offset`, up to `pages` of them: the
	// clean ones become CPU-modified and writable with one protection update, as their own write
	// faults would make them one at a time. CPU-modified says "may have changed, upload before
	// the GPU uses it", so a page marked early costs an upload of bytes that did not change and
	// nothing else. The window ends before a page the GPU owns (its bytes are not the guest's),
	// before one that another cache watches too, and at the region's end. Returns the pages it
	// covers, those modified already included; 0 when the faulting page is not CPU-modified
	// (the GPU owns it). Under `lock`.
	[[nodiscard]] size_t MarkCpuModifiedAhead(uint64_t offset, size_t pages) {
		const auto page = static_cast<size_t>(offset / TRACKER_PAGE_SIZE);
		if (!m_cpu_dirty.Get(page)) {
			return 0;
		}
		const auto first = page + 1;
		auto       end   = std::min(first + pages, TRACKER_REGION_PAGES);
		end              = std::min(end, m_gpu_dirty.FirstRangeFrom(first).first);
		if (end <= first) {
			return 0;
		}
		// Nor past a page another cache watches (an image's): it keeps its protection whatever
		// this tracker does, and its bytes in a buffer may be the image's, not the guest's.
		end = first + m_page_manager.SoleWriteWatcherRun(m_cpu_addr + first * TRACKER_PAGE_SIZE,
		                                                 end - first);
		if (end <= first) {
			return 0;
		}
		RegionBits window;
		window.SetRange(first, end);
		const auto clean = window & ~m_cpu_dirty;
		if (!clean.None()) {
			const StateChange change(*this);
			m_cpu_dirty |= clean;
			UpdateProtection<false, false>();
		}
		return end - first;
	}

	TrackingSpinLock lock;

private:
	// Brackets a change of the tracking state and the protection that follows it (under `lock`):
	// m_version is odd while it runs, for TryQueryUnlocked.
	class StateChange {
	public:
		explicit StateChange(RegionManager& region): m_region(region) {
			const auto version = m_region.m_version.load(std::memory_order_relaxed);
			m_region.m_version.store(version + 1u, std::memory_order_relaxed);
			std::atomic_thread_fence(std::memory_order_release);
		}
		~StateChange() {
			const auto version = m_region.m_version.load(std::memory_order_relaxed);
			m_region.m_version.store(version + 1u, std::memory_order_release);
		}
		KYTY_CLASS_NO_COPY(StateChange);

	private:
		RegionManager& m_region;
	};

	template <bool track, bool is_read>
	void UpdateProtection() {
		if constexpr (is_read) {
			const auto protection = ~m_gpu_dirty | m_stale_readable;
			auto       mask       = protection ^ m_readable;
			if (mask.None()) {
				return;
			}
			m_readable = protection;
			m_page_manager.UpdatePageWatchersForRegion<track, true>(m_cpu_addr, mask);
		} else {
			// One direction at a time: pages that are clean and still writable get their write
			// watch, or pages that are dirty and watched lose it. Pages taken for a deferred
			// protection (clean, still writable until ProtectCpuClean) are protected by the
			// first, which only comes early for them, and left alone by the second.
			auto mask = track ? m_writable & ~m_cpu_dirty : m_cpu_dirty & ~m_writable;
			if (mask.None()) {
				return;
			}
			if constexpr (track) {
				m_writable &= m_cpu_dirty;
			} else {
				m_writable |= mask;
			}
			m_page_manager.UpdatePageWatchersForRegion<track, false>(m_cpu_addr, mask);
		}
	}

	template <DirtySource source>
	RegionBits& GetBits() {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	template <DirtySource source>
	const RegionBits& GetBits() const {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	[[nodiscard]] static std::pair<size_t, size_t> GetPageRange(uint64_t offset, uint64_t size) {
		return {static_cast<size_t>(offset / TRACKER_PAGE_SIZE),
		        static_cast<size_t>((offset + size + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE)};
	}

	PageManager& m_page_manager;
	uint64_t     m_cpu_addr = 0;
	RegionBits   m_cpu_dirty;
	RegionBits   m_gpu_dirty;
	RegionBits   m_stale_readable;
	RegionBits   m_writable;
	RegionBits   m_readable;
	// See StateChange.
	std::atomic<uint32_t> m_version {0};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
