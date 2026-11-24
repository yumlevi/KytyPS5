#ifndef KYTY_COMMON_MEMORY_TRACE_H_
#define KYTY_COMMON_MEMORY_TRACE_H_

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>

// The last things the emulator did to guest memory the GPU writes, kept in a ring for a fatal
// report (mem_trace 1 / KYTY_MEM_TRACE=1; off, a note is one relaxed load). A game thread that
// faults on a value it read from such memory read it too early or too late: the report then
// lists, for the memory each register points to, when the GPU's writes were recorded, when
// their bytes were brought into guest memory and by which way, when the pages were readable,
// and what was announced to the guest (labels, flips) around that time.
namespace Common::MemoryTrace {

enum class Kind : uint8_t {
	None,
	GpuWrite,       // The GPU thread recorded a write of the range (tick: its submission).
	KeepReadable,   // ... and left the page readable with its bytes of before (stale readable).
	MakeUnreadable, // ... and took the page from the guest until its bytes are brought.
	Deferred,       // The page's download did not fit the submission: unreadable until the next.
	HotPublish,     // The completion thread wrote a hot page's downloaded bytes.
	HotReject,      // ... or left them: the page was written again since the download.
	Publish,        // The GPU thread wrote downloaded bytes after its wait for the GPU.
	GuestCopy,      // A guest thread copied the bytes itself.
	Readback,       // A guest fault asked for the range (read, or a store).
	GuestStore,     // A guest store took the page (write fault).
	Copy,           // A copy command of the guest (DMA) with this destination.
	Label,          // A label was stored at the address.
	Flip,           // A flip was completed.
	Suspend,        // A suspend point was released.
};

struct Event {
	int64_t  time_ns;
	uint64_t address;
	uint64_t size;
	uint64_t tick;
	Kind     kind;
};

inline constexpr size_t      Events = size_t {1} << 17u;
inline std::atomic<bool>     g_on {false};
inline std::atomic<uint64_t> g_next {0};
inline Event                 g_events[Events];

[[nodiscard]] inline int64_t NowNs() noexcept {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

inline void Note(Kind kind, uint64_t address, uint64_t size, uint64_t tick = 0) noexcept {
	if (!g_on.load(std::memory_order_relaxed)) {
		return;
	}
	auto& event   = g_events[g_next.fetch_add(1, std::memory_order_relaxed) % Events];
	event.kind    = Kind::None;
	event.time_ns = NowNs();
	event.address = address;
	event.size    = size;
	event.tick    = tick;
	event.kind    = kind;
}

[[nodiscard]] inline const char* Name(Kind kind) noexcept {
	switch (kind) {
		case Kind::GpuWrite: return "GPU write recorded";
		case Kind::KeepReadable: return "page left readable through the write";
		case Kind::MakeUnreadable: return "page taken from the guest";
		case Kind::Deferred: return "download left to the next submission";
		case Kind::HotPublish: return "hot page's bytes written (completion thread)";
		case Kind::HotReject: return "hot page's bytes left out (written again since)";
		case Kind::Publish: return "downloaded bytes written (GPU thread)";
		case Kind::GuestCopy: return "bytes copied by a guest thread";
		case Kind::Readback: return "guest fault asked for the bytes";
		case Kind::GuestStore: return "guest store took the page";
		case Kind::Copy: return "guest copy command into the range";
		case Kind::Label: return "label stored";
		case Kind::Flip: return "flip completed";
		case Kind::Suspend: return "suspend point released";
		default: return "?";
	}
}

// Writes, without allocating, the events since `from_ns` that touch [address, address + size)
// (announcements: all of them when `announcements`), oldest first, each with its time against
// `fault_ns`. Returns the characters written; at most `most` events (the newest).
inline size_t Describe(char* out, size_t capacity, uint64_t address, uint64_t size, int64_t fault_ns,
                       int64_t from_ns, bool announcements, size_t most) noexcept {
	if (capacity == 0) {
		return 0;
	}
	const uint64_t next  = g_next.load(std::memory_order_relaxed);
	const uint64_t count = next < Events ? next : Events;
	// The matching events, newest first, then written oldest first.
	static constexpr size_t Most = 96;
	const Event*            found[Most];
	size_t                  n = 0;
	most                      = most < Most ? most : Most;
	for (uint64_t back = 1; back <= count && n < most; back++) {
		const auto& event = g_events[(next - back) % Events];
		if (event.kind == Kind::None || event.time_ns < from_ns) {
			if (event.time_ns < from_ns && event.kind != Kind::None) {
				break;
			}
			continue;
		}
		const bool signal = event.kind == Kind::Label || event.kind == Kind::Flip || event.kind == Kind::Suspend;
		const bool touches = event.size != 0 && event.address < address + size && address < event.address + event.size;
		if (announcements ? signal : (touches && !(signal && event.kind != Kind::Label))) {
			found[n++] = &event;
		}
	}
	size_t written = 0;
	for (size_t i = n; i-- > 0 && written + 160 < capacity;) {
		const auto& event = *found[i];
		const auto  us    = (event.time_ns - fault_ns) / 1000;
		written += static_cast<size_t>(std::snprintf(
		    out + written, capacity - written, "  %8" PRId64 " us %s the fault: %s 0x%010" PRIx64 "+0x%" PRIx64 " tick %" PRIu64 "\n",
		    us < 0 ? -us : us, us < 0 ? "before" : "after", Name(event.kind), event.address, event.size, event.tick));
	}
	out[written < capacity ? written : capacity - 1] = '\0';
	return written;
}

} // namespace Common::MemoryTrace

#endif // KYTY_COMMON_MEMORY_TRACE_H_
