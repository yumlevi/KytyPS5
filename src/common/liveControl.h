#ifndef KYTY_COMMON_LIVE_CONTROL_H_
#define KYTY_COMMON_LIVE_CONTROL_H_

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// A measurement channel for one boot and many same-process experiments. With KYTY_LIVE_FILE set
// to a command file, a side thread reads the file every 100 ms and runs its commands once per new
// "id <n>" first line, one command per following line, in order:
//   measure <seconds> [label]      frames, frame times and per-frame work of the render threads
//   fps <seconds> [label]          the frame rate alone, without switching the counters on
//   prof <seconds> [thread] [top]  wall-time samples of a registered thread (default gpu), by function
//   prof <seconds> <thread> <top> <text>   the callers of the functions whose name contains <text>
//   set <switch> <value>           a live switch (Switch below); `switches` lists them
//   pad <button> [ms]              press and release a controller button (cross, circle, up, ...)
//   stick <lx> <ly> <rx> <ry> <ms> hold the sticks (0..255, 128 centred), then centre them
//   replay <log> <from> [seconds]  play the gamepad events of a session's log (its "Controller
//                                  axis:" and "Controller button:" lines) from the event time
//                                  <from> on, each as long after this command as it came after
//                                  <from> then; other commands go on meanwhile. `replay stop`
//                                  ends it. Played by the clock alone the game leaves the
//                                  session's course within seconds (a stick held a frame longer
//                                  turns the camera a few degrees further, and every later step
//                                  goes that much wrong), so where the log has the camera's path
//                                  (its "Camera:" lines) the sticks are corrected towards it: the
//                                  camera is turned to where it looked then and the player walked
//                                  to where they stood then. "Replay track:" lines tell how far
//                                  off the path it is. With `oncamera` as the last word the
//                                  session's clock starts with the level's first frame (the camera
//                                  coming on after the loading screen) instead of with the
//                                  command: <from> is that frame's time in the log.
//   sleep <ms>                     wait before the next command
//   wait [seconds]                 wait (at most that long, default 300) until the emulator has no
//                                  shader translation or pipeline build left that it runs ahead of
//                                  first uses, so that a measurement does not see them
//   threads                        the registered threads and their CPU time
//   quit                           close the emulator window
// Answers go to the log and, one line each, to <command file>.out. Without KYTY_LIVE_FILE nothing
// runs: the flip hook is one relaxed load, a switch read is one relaxed load.
namespace Common::LiveControl {

// A named run-time switch of a code path, for A/B measurements in one process. It starts from
// the environment variable KYTY_<NAME> (the name in upper case) when that is set, else from the
// default, and `set <name> <value>` changes it while the game runs. Define switches at namespace
// scope: they register themselves before main.
class Switch {
public:
	Switch(const char* name, uint32_t default_value) noexcept;
	Switch(const Switch&)            = delete;
	Switch& operator=(const Switch&) = delete;

	[[nodiscard]] uint32_t Get() const noexcept { return m_value.load(std::memory_order_relaxed); }
	[[nodiscard]] bool     On() const noexcept { return Get() != 0; }
	void Set(uint32_t value) noexcept { m_value.store(value, std::memory_order_relaxed); }

	[[nodiscard]] const char*   Name() const noexcept { return m_name; }
	[[nodiscard]] const Switch* Next() const noexcept { return m_next; }

private:
	const char*           m_name;
	std::atomic<uint32_t> m_value;
	Switch*               m_next;
};

// A named event counter that `measure` reports per frame beside its own numbers. Define counters
// at namespace scope, like switches. Adding is one relaxed increment, taken only while the
// channel is active.
class Counter {
public:
	explicit Counter(const char* name) noexcept;
	Counter(const Counter&)            = delete;
	Counter& operator=(const Counter&) = delete;

	void Add(uint64_t value = 1) noexcept;

	[[nodiscard]] uint64_t Get() const noexcept { return m_value.load(std::memory_order_relaxed); }
	[[nodiscard]] const char*    Name() const noexcept { return m_name; }
	[[nodiscard]] const Counter* Next() const noexcept { return m_next; }

private:
	const char*           m_name;
	std::atomic<uint64_t> m_value {0};
	Counter*              m_next;
};

// The host actions the commands need, set by the emulator before Start.
struct HostHooks {
	void (*pad_button)(uint32_t button, bool down)             = nullptr;
	void (*pad_sticks)(int left_x, int left_y, int right_x, int right_y) = nullptr;
	void (*quit)()                                              = nullptr;
	// Shader translations and pipeline builds the emulator runs ahead of their first use and
	// has not finished (for `wait`).
	uint64_t (*pending_builds)()                                = nullptr;
	// Guest memory for `mem`: the mapping at or after an address, and reads and writes of mapped
	// bytes through the backing store (page protection is neither needed nor changed).
	bool (*guest_next_mapping)(uint64_t address, uint64_t* start, uint64_t* end) = nullptr;
	bool (*guest_read)(uint64_t address, void* data, uint64_t size)             = nullptr;
	bool (*guest_write)(uint64_t address, const void* data, uint64_t size)      = nullptr;
	std::vector<uint64_t> (*guest_find)(const uint8_t* pattern, size_t size, uint64_t align,
	                                    size_t limit, uint64_t from, uint64_t to,
	                                    uint64_t* scanned) = nullptr;
	// The guest's own threads, as host thread ids and names (for the CPU times of a hitch line).
	std::vector<std::pair<uint64_t, std::string>> (*guest_threads)() = nullptr;
	// One gamepad event as a session's log has it (SDL's axis or button number and value), for
	// `replay`: translated as the live event was.
	void (*replay_axis)(int axis, int value)      = nullptr;
	void (*replay_button)(int button, bool down) = nullptr;
	// The window's event clock in seconds: the time base of the log's input lines.
	double (*input_seconds)() = nullptr;
	// A RenderDoc capture of the next two guest flips, as the F1 key asks for one (`capture`;
	// the session was started with --rd).
	void (*capture)() = nullptr;
};
void SetHostHooks(const HostHooks& hooks) noexcept;

// The game's camera in the frame being rendered, as its view constants have it (the renderer's
// thread, once a frame): its position in the world and the direction it looks in. Goes to the log
// as a "Camera:" line with the event clock's time (path_log 0: not), so that a session's log
// holds the path the player took beside what they pressed, and is what `replay` steers by.
void NoteCamera(const float* position, const float* forward) noexcept;

// Starts the command thread when KYTY_LIVE_FILE names a command file.
void Start();

// The player marks a moment (the F9 key): a "Mark <time of day> #<n>" line goes to the log, and
// a "MarkPerf #<n>" line with the numbers of `measure` for the two seconds after it. For telling
// afterwards where in a long log something was seen on the screen.
void Mark() noexcept;

// Names the calling thread for `prof`, `threads` and the per-thread CPU time of `measure`
// ("gpu" is the thread that translates guest commands).
void RegisterThread(const char* name) noexcept;

namespace Detail {
inline constexpr size_t      FLIP_TIMES = 8192;
inline std::atomic<bool>     g_active {false};
inline std::atomic<bool>     g_guest_waits {false};
void NoteGuestWait(const char* kind, uint64_t address, uint64_t bytes, int64_t begin_ns,
                   int64_t least_ns) noexcept;
[[nodiscard]] inline int64_t SteadyNs() noexcept {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}
inline std::atomic<uint64_t> g_flips {0};
inline std::atomic<int64_t>  g_flip_times_ns[FLIP_TIMES] {};
inline thread_local bool     t_gpu_thread = false;
} // namespace Detail

// The calling thread registered itself as "gpu" (for counters kept apart for that thread).
[[nodiscard]] inline bool OnGpuThread() noexcept {
	return Detail::t_gpu_thread;
}

[[nodiscard]] inline bool Active() noexcept {
	return Detail::g_active.load(std::memory_order_relaxed);
}

inline void Counter::Add(uint64_t value) noexcept {
	if (Active()) {
		m_value.fetch_add(value, std::memory_order_relaxed);
	}
}

// A wait the emulator puts a guest thread through (a readback of GPU-written memory, an unmap
// that waits for the GPU, ...). With the hitch log on (KYTY_HITCH_LOG_MS) the line of a slow
// frame names the waits that ended in it, by thread and kind: what the game's threads were held
// up by while the GPU thread had nothing to do. `kind` is a literal; waits shorter than
// `least_us` are left out. Without the hitch log a scope is one relaxed load.
class GuestWait {
public:
	explicit GuestWait(const char* kind, uint64_t address = 0, uint64_t bytes = 0,
	                   uint32_t least_us = 0) noexcept
	    : m_kind(kind), m_address(address), m_bytes(bytes), m_least_us(least_us),
	      m_begin_ns(Detail::g_guest_waits.load(std::memory_order_relaxed) ? Detail::SteadyNs()
	                                                                         : 0) {}
	~GuestWait() {
		if (m_begin_ns != 0) {
			Detail::NoteGuestWait(m_kind, m_address, m_bytes, m_begin_ns,
			                      int64_t {m_least_us} * 1000);
		}
	}
	GuestWait(const GuestWait&)            = delete;
	GuestWait& operator=(const GuestWait&) = delete;

private:
	const char* m_kind;
	uint64_t    m_address;
	uint64_t    m_bytes;
	uint32_t    m_least_us;
	int64_t     m_begin_ns;
};

// A guest flip on the main bus (one caller thread).
inline void NoteFlip() noexcept {
	if (!Active()) {
		return;
	}
	const auto flip = Detail::g_flips.load(std::memory_order_relaxed);
	Detail::g_flip_times_ns[flip % Detail::FLIP_TIMES].store(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(
	        std::chrono::steady_clock::now().time_since_epoch())
	        .count(),
	    std::memory_order_relaxed);
	Detail::g_flips.store(flip + 1, std::memory_order_release);
}

} // namespace Common::LiveControl

#endif /* KYTY_COMMON_LIVE_CONTROL_H_ */
