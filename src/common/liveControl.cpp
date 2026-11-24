#include "common/liveControl.h"

#include "common/config.h"
#include "common/debugCounters.h"
#include "common/gpuTiming.h"
#include "common/logging/log.h"
#include "common/threads.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <chrono>
#include <fstream>
#include <functional>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
// clang-format off
#include <dbghelp.h>
#include <intrin.h>
#include <psapi.h>
// clang-format on
#endif

namespace Common::LiveControl {

namespace {

// Heads of the lists the switches' and counters' constructors build (before main, one thread).
Switch*  g_switches = nullptr;
Counter* g_counters = nullptr;

HostHooks g_hooks;

struct NamedThread {
	std::string name;
	uint64_t    handle     = 0; // A thread HANDLE on Windows.
	uint32_t    tid        = 0;
	uint64_t    stack_low  = 0;
	uint64_t    stack_high = 0;
};

std::mutex               g_threads_mutex;
std::vector<NamedThread> g_threads;

std::string g_command_path;
std::string g_answer_path;

int64_t NowNs() noexcept {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

void Answer(const std::string& line) {
	LOGF("%s\n", line.c_str());
	if (auto* out = std::fopen(g_answer_path.c_str(), "ab")) {
		std::fwrite(line.data(), 1, line.size(), out);
		std::fputc('\n', out);
		std::fclose(out);
	}
}

template <typename... Args>
std::string Format(const char* format, Args... args) {
	const int size = std::snprintf(nullptr, 0, format, args...);
	if (size <= 0) {
		return {};
	}
	std::string text(static_cast<size_t>(size), '\0');
	std::snprintf(text.data(), static_cast<size_t>(size) + 1, format, args...);
	return text;
}

std::vector<NamedThread> Threads() {
	std::scoped_lock lock(g_threads_mutex);
	return g_threads;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// The rate of the cycle counts Windows keeps per thread (the time stamp counter's), measured
// once against the performance counter.
double CyclesPerSecond() noexcept {
	static const double rate = [] {
		LARGE_INTEGER frequency {};
		LARGE_INTEGER begin {};
		LARGE_INTEGER end {};
		QueryPerformanceFrequency(&frequency);
		QueryPerformanceCounter(&begin);
		const uint64_t first = __rdtsc();
		Sleep(20);
		QueryPerformanceCounter(&end);
		const uint64_t last = __rdtsc();
		if (end.QuadPart <= begin.QuadPart || last <= first) {
			return 0.0;
		}
		return static_cast<double>(last - first) * static_cast<double>(frequency.QuadPart) /
		       static_cast<double>(end.QuadPart - begin.QuadPart);
	}();
	return rate;
}

// CPU time the thread of a handle has used, in seconds; negative when unknown. From its cycle
// count where that can be had: the times Windows keeps move by whole clock ticks (15.6 ms),
// which says nothing about one frame.
double HandleCpuSeconds(HANDLE handle) noexcept {
	ULONG64 cycles = 0;
	if (const double rate = CyclesPerSecond(); rate > 0.0 && QueryThreadCycleTime(handle, &cycles) != 0) {
		return static_cast<double>(cycles) / rate;
	}
	FILETIME creation {};
	FILETIME exit {};
	FILETIME kernel {};
	FILETIME user {};
	if (GetThreadTimes(handle, &creation, &exit, &kernel, &user) == 0) {
		return -1.0;
	}
	const auto ticks = [](const FILETIME& time) {
		return (static_cast<uint64_t>(time.dwHighDateTime) << 32u) | time.dwLowDateTime;
	};
	return static_cast<double>(ticks(kernel) + ticks(user)) / 1e7;
}
#endif

// CPU time a thread has used, in seconds; negative when unknown.
double ThreadCpuSeconds(const NamedThread& thread) noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// NOLINTNEXTLINE(performance-no-int-to-ptr)
	return HandleCpuSeconds(reinterpret_cast<HANDLE>(thread.handle));
#else
	(void)thread;
	return -1.0;
#endif
}

double ProcessCpuSeconds() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	ULONG64 cycles = 0;
	if (const double rate = CyclesPerSecond();
	    rate > 0.0 && QueryProcessCycleTime(GetCurrentProcess(), &cycles) != 0) {
		return static_cast<double>(cycles) / rate;
	}
	FILETIME creation {};
	FILETIME exit {};
	FILETIME kernel {};
	FILETIME user {};
	if (GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user) == 0) {
		return -1.0;
	}
	const auto ticks = [](const FILETIME& time) {
		return (static_cast<uint64_t>(time.dwHighDateTime) << 32u) | time.dwLowDateTime;
	};
	return static_cast<double>(ticks(kernel) + ticks(user)) / 1e7;
#else
	return -1.0;
#endif
}

// ---------------------------------------------------------------------------------------------
// measure

using DebugCounter = DebugCounters::Counter;

constexpr std::array MEASURED_COUNTERS {
    std::pair {DebugCounter::Draws, "draws"},
    std::pair {DebugCounter::IndirectDraws, "indirect_draws"},
    std::pair {DebugCounter::Dispatches, "dispatches"},
    std::pair {DebugCounter::QueueSubmits, "submits"},
    std::pair {DebugCounter::GpuWaits, "gpu_waits"},
    std::pair {DebugCounter::GpuFaults, "write_faults"},
    std::pair {DebugCounter::ShadersCompiled, "shaders"},
    std::pair {DebugCounter::PipelinesCreated, "pipelines"},
};

// More of them for the hitch lines alone (HitchLog): counts, and times in nanoseconds.
constexpr std::array HITCH_COUNTERS {
    std::pair {DebugCounter::DrawsSkippedPending, "draws_skipped"},
    std::pair {DebugCounter::TextureEvictions, "texture_evictions"},
    std::pair {DebugCounter::BufferEvictions, "buffer_evictions"},
    std::pair {DebugCounter::ColorMetadataReadbacks, "metadata_readbacks"},
    std::pair {DebugCounter::SwapchainRecreations, "swapchain_rebuilds"},
    std::pair {DebugCounter::DroppedPresents, "dropped_presents"},
    std::pair {DebugCounter::LateVblanks, "late_vblanks"},
    std::pair {DebugCounter::AudioUnderruns, "audio_underruns"},
    std::pair {DebugCounter::GuestFileReadBytes, "file_read_bytes"},
    std::pair {DebugCounter::TextureDownloadBytes, "texture_download_bytes"},
};
constexpr std::array HITCH_TIMES {
    std::pair {DebugCounter::ShaderCompileNs, "shader_compile"},
    std::pair {DebugCounter::PipelineCreateNs, "pipeline_create"},
    std::pair {DebugCounter::ColorMetadataReadbackNs, "metadata_readback"},
    std::pair {DebugCounter::PresentNs, "present"},
    std::pair {DebugCounter::PresentAcquireNs, "present_acquire"},
    std::pair {DebugCounter::OverlayNs, "overlay"},
};

struct Snapshot {
	int64_t                                          time_ns = 0;
	uint64_t                                         flips   = 0;
	std::array<uint64_t, MEASURED_COUNTERS.size()>   counters {};
	std::array<uint64_t, HITCH_COUNTERS.size()>      hitch_counters {};
	std::array<uint64_t, HITCH_TIMES.size()>         hitch_times_ns {};
	uint64_t                                         gpu_thread_busy_ns = 0;
	uint64_t                                         gpu_wait_ns        = 0;
	uint64_t                                         buffer_upload      = 0;
	uint64_t                                         texture_upload     = 0;
	uint64_t                                         buffer_download    = 0;
	GpuTiming::SyncCounts                            sync;
	GpuTiming::Totals                                gpu;
	double                                           process_cpu = 0.0;
	std::vector<std::pair<std::string, double>>      thread_cpu;
	std::vector<uint64_t>                            named; // The Counters, in list order.
};

Snapshot Take() {
	Snapshot snapshot;
	snapshot.time_ns = NowNs();
	snapshot.flips   = Detail::g_flips.load(std::memory_order_acquire);
	for (size_t i = 0; i < MEASURED_COUNTERS.size(); i++) {
		snapshot.counters[i] = DebugCounters::Get(MEASURED_COUNTERS[i].first);
	}
	for (size_t i = 0; i < HITCH_COUNTERS.size(); i++) {
		snapshot.hitch_counters[i] = DebugCounters::Get(HITCH_COUNTERS[i].first);
	}
	for (size_t i = 0; i < HITCH_TIMES.size(); i++) {
		snapshot.hitch_times_ns[i] = DebugCounters::Get(HITCH_TIMES[i].first);
	}
	snapshot.gpu_thread_busy_ns = DebugCounters::Get(DebugCounter::GpuThreadBusyNs);
	snapshot.gpu_wait_ns        = DebugCounters::Get(DebugCounter::GpuWaitNs);
	snapshot.buffer_upload      = DebugCounters::Get(DebugCounter::BufferUploadBytes);
	snapshot.texture_upload     = DebugCounters::Get(DebugCounter::TextureUploadBytes);
	snapshot.buffer_download    = DebugCounters::Get(DebugCounter::BufferDownloadBytes);
	snapshot.sync               = GpuTiming::Sync().Snapshot();
	snapshot.gpu                = GpuTiming::RunningTotals();
	snapshot.process_cpu        = ProcessCpuSeconds();
	for (const auto* counter = g_counters; counter != nullptr; counter = counter->Next()) {
		snapshot.named.push_back(counter->Get());
	}
	for (const auto& thread: Threads()) {
		const double seconds = ThreadCpuSeconds(thread);
		const auto   known   = std::find_if(snapshot.thread_cpu.begin(), snapshot.thread_cpu.end(),
		                                    [&](const auto& entry) { return entry.first == thread.name; });
		if (known != snapshot.thread_cpu.end()) {
			known->second += std::max(seconds, 0.0);
		} else {
			snapshot.thread_cpu.emplace_back(thread.name, std::max(seconds, 0.0));
		}
	}
	return snapshot;
}

// One window of measurement as a line: "<head> seconds=... flips=... fps=... | ..." (Measure).
std::string MeasureLine(const std::string& head, double seconds) {
	// The counters are collected only while something reads them (the overlay, or this).
	const bool counters_were_on = DebugCounters::Enabled();
	DebugCounters::SetEnabled(true);
	const auto begin = Take();
	std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
	const auto end = Take();
	DebugCounters::SetEnabled(counters_were_on);

	const double elapsed = static_cast<double>(end.time_ns - begin.time_ns) / 1e9;
	const auto   flips   = end.flips - begin.flips;
	const double frames  = static_cast<double>(std::max<uint64_t>(flips, 1));
	const double fps     = static_cast<double>(flips) / elapsed;

	// Frame times of the window from the flip time ring (the newest FLIP_TIMES flips).
	std::vector<double> frame_ms;
	const auto          first = flips + 1 > Detail::FLIP_TIMES ? end.flips - Detail::FLIP_TIMES + 1
	                                                           : begin.flips + 1;
	for (auto flip = std::max<uint64_t>(first, 1); flip < end.flips; flip++) {
		const auto earlier =
		    Detail::g_flip_times_ns[(flip - 1) % Detail::FLIP_TIMES].load(std::memory_order_relaxed);
		const auto later =
		    Detail::g_flip_times_ns[flip % Detail::FLIP_TIMES].load(std::memory_order_relaxed);
		if (later > earlier) {
			frame_ms.push_back(static_cast<double>(later - earlier) / 1e6);
		}
	}
	// How evenly the frames came: by how much a frame time differs from the one before it (a
	// frame on screen for 16.7 ms and the next for 33.3 differ by 16.7; a hitch moves only the
	// high end).
	std::vector<double> step_ms;
	step_ms.reserve(frame_ms.size());
	for (size_t i = 1; i < frame_ms.size(); i++) {
		step_ms.push_back(std::abs(frame_ms[i] - frame_ms[i - 1]));
	}
	std::sort(step_ms.begin(), step_ms.end());
	const auto step = [&](double q) {
		if (step_ms.empty()) {
			return 0.0;
		}
		return step_ms[std::min(step_ms.size() - 1,
		                        static_cast<size_t>(q * static_cast<double>(step_ms.size())))];
	};
	std::sort(frame_ms.begin(), frame_ms.end());
	const auto quantile = [&](double q) {
		if (frame_ms.empty()) {
			return 0.0;
		}
		return frame_ms[std::min(frame_ms.size() - 1,
		                         static_cast<size_t>(q * static_cast<double>(frame_ms.size())))];
	};
	// The mean of the slowest 1% of frames, as a rate.
	double low = 0.0;
	if (!frame_ms.empty()) {
		const auto count = std::max<size_t>(1, frame_ms.size() / 100);
		double     total = 0.0;
		for (size_t i = frame_ms.size() - count; i < frame_ms.size(); i++) {
			total += frame_ms[i];
		}
		low = 1000.0 * static_cast<double>(count) / total;
	}

	const auto sync       = end.sync.Since(begin.sync);
	const auto per_frame  = [&](uint64_t later, uint64_t earlier) {
		return static_cast<double>(later - earlier) / frames;
	};
	const auto ms_per_frame = [&](uint64_t later, uint64_t earlier) {
		return static_cast<double>(later - earlier) / 1e6 / frames;
	};
	const double gpu_window_s = static_cast<double>(end.gpu.window_ns - begin.gpu.window_ns) / 1e9;
	const double gpu_busy_share =
	    gpu_window_s > 0.0
	        ? static_cast<double>(end.gpu.busy_ns - begin.gpu.busy_ns) / 1e9 / gpu_window_s
	        : -1.0;

	std::string line =
	    head + Format(" seconds=%.2f flips=%" PRIu64
	                  " fps=%.2f low1=%.2f frame_ms p50=%.2f p95=%.2f p99=%.2f max=%.2f"
	                  " step_ms mid=%.2f p90=%.2f",
	                  elapsed, flips, fps, low, quantile(0.50), quantile(0.95), quantile(0.99),
	                  frame_ms.empty() ? 0.0 : frame_ms.back(), step(0.50), step(0.90));
	// The GPU thread per frame: wall time inside commands and submissions (waits included), its
	// waits on the GPU, and its time without guest work (starved) or with every stream suspended.
	line += Format(" | gpu_thread_ms busy=%.2f gpu_waits=%.2f starved=%.2f blocked=%.2f",
	               ms_per_frame(end.gpu_thread_busy_ns, begin.gpu_thread_busy_ns),
	               ms_per_frame(end.gpu_wait_ns, begin.gpu_wait_ns),
	               static_cast<double>(sync.gpu_thread_starved_ns) / 1e6 / frames,
	               static_cast<double>(sync.gpu_thread_blocked_ns) / 1e6 / frames);
	if (gpu_busy_share >= 0.0) {
		line += Format(" | gpu busy=%.1f%% busy_ms=%.2f", 100.0 * gpu_busy_share,
		               1000.0 * gpu_busy_share / std::max(fps, 0.001));
	} else {
		line += " | gpu busy=n/a";
	}
	line += " | per_frame";
	for (size_t i = 0; i < MEASURED_COUNTERS.size(); i++) {
		line += Format(" %s=%.1f", MEASURED_COUNTERS[i].second,
		               per_frame(end.counters[i], begin.counters[i]));
	}
	line += Format(" buffer_upload_kb=%.0f texture_upload_kb=%.0f buffer_download_kb=%.0f",
	               per_frame(end.buffer_upload, begin.buffer_upload) / 1024.0,
	               per_frame(end.texture_upload, begin.texture_upload) / 1024.0,
	               per_frame(end.buffer_download, begin.buffer_download) / 1024.0);
	// Pages (of 16 KiB) whose GPU writes were downloaded with their submission because a guest
	// thread read them back before, and pages left for a later submission.
	line += Format(" async_pages=%.1f async_deferred=%.1f",
	               static_cast<double>(sync.async_pages) / frames,
	               static_cast<double>(sync.async_deferred) / frames);
	if (g_counters != nullptr) {
		line += " | counters";
		size_t index = 0;
		for (const auto* counter = g_counters; counter != nullptr; counter = counter->Next()) {
			line += Format(" %s=%.1f", counter->Name(),
			               per_frame(end.named[index], begin.named[index]));
			index++;
		}
	}
	line += " | waits_ms";
	for (size_t reason = 0; reason < GpuTiming::SYNC_REASON_COUNT; reason++) {
		if (sync.waits[reason] != 0) {
			line += Format(" %s=%.2f(%.1f)",
			               GpuTiming::SyncReasonName(static_cast<GpuTiming::SyncReason>(reason)),
			               static_cast<double>(sync.wait_ns[reason]) / 1e6 / frames,
			               static_cast<double>(sync.waits[reason]) / frames);
		}
	}
	line += " | cpu_ms";
	if (begin.process_cpu >= 0.0 && end.process_cpu >= 0.0) {
		line += Format(" process=%.2f", 1000.0 * (end.process_cpu - begin.process_cpu) / frames);
	}
	for (const auto& [name, cpu]: end.thread_cpu) {
		const auto earlier = std::find_if(begin.thread_cpu.begin(), begin.thread_cpu.end(),
		                                  [&](const auto& entry) { return entry.first == name; });
		const double base  = earlier != begin.thread_cpu.end() ? earlier->second : 0.0;
		line += Format(" %s=%.2f", name.c_str(), 1000.0 * (cpu - base) / frames);
	}
	return line;
}

void Measure(uint64_t id, double seconds, const std::string& label) {
	Answer(MeasureLine(Format("LIVE measure id=%" PRIu64 " label=%s", id, label.c_str()), seconds));
}

// A window's frame rate alone. The counters stay as they are (off unless the performance panel
// is shown), where `measure` switches them on for its window: this is the rate of a session
// played without the panel.
void Fps(uint64_t id, double seconds, const std::string& label) {
	const auto begin_flips = Detail::g_flips.load(std::memory_order_acquire);
	const auto begin_ns    = NowNs();
	std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
	const auto   flips   = Detail::g_flips.load(std::memory_order_acquire) - begin_flips;
	const double elapsed = static_cast<double>(NowNs() - begin_ns) / 1e9;
	Answer(Format("LIVE fps id=%" PRIu64 " label=%s seconds=%.2f flips=%" PRIu64
	              " fps=%.2f counters=%s",
	              id, label.c_str(), elapsed, flips, static_cast<double>(flips) / elapsed,
	              DebugCounters::Enabled() ? "on" : "off"));
}

// KYTY_PERF_LOG=<seconds>: a measurement of every such window goes to the log while the game
// runs, so that a session played anywhere in the game says afterwards where it was slow and
// what the frame was made of there (the numbers of `measure`). Windows without a flip
// (loading, a menu that does not render) are left out.
void PerfLog(double seconds) {
	for (;;) {
		// The time of day the window ends at: what a player remembers a place by.
		auto line = MeasureLine("Perf", seconds);
		const auto now = std::time(nullptr);
		std::tm    local {};
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		localtime_s(&local, &now);
#else
		localtime_r(&now, &local);
#endif
		line.insert(4, Format(" %02d:%02d:%02d", local.tm_hour, local.tm_min, local.tm_sec));
		if (line.find(" flips=0 ") != std::string::npos) {
			continue;
		}
		Log::WriteToConsoleAndLog(line + "\n");
	}
}

// ---------------------------------------------------------------------------------------------
// hitches

// What a window cost: a time in milliseconds or a count, of the whole window.
struct Cost {
	std::string name;
	double      value = 0.0;
	bool        ms    = false;
};

std::vector<Cost> Costs(const Snapshot& begin, const Snapshot& end) {
	std::vector<Cost> costs;
	const auto time = [&](std::string name, uint64_t later, uint64_t earlier) {
		costs.push_back({std::move(name), static_cast<double>(later - earlier) / 1e6, true});
	};
	const auto count = [&](std::string name, uint64_t later, uint64_t earlier, double unit = 1.0) {
		costs.push_back({std::move(name), static_cast<double>(later - earlier) / unit, false});
	};
	const auto sync = end.sync.Since(begin.sync);
	time("gpu_thread", end.gpu_thread_busy_ns, begin.gpu_thread_busy_ns);
	time("gpu_wait", end.gpu_wait_ns, begin.gpu_wait_ns);
	time("starved", sync.gpu_thread_starved_ns, 0);
	time("blocked", sync.gpu_thread_blocked_ns, 0);
	for (size_t reason = 0; reason < GpuTiming::SYNC_REASON_COUNT; reason++) {
		const std::string name = GpuTiming::SyncReasonName(static_cast<GpuTiming::SyncReason>(reason));
		time("wait:" + name, sync.wait_ns[reason], 0);
		count("waits:" + name, sync.waits[reason], 0);
	}
	for (size_t i = 0; i < HITCH_TIMES.size(); i++) {
		time(HITCH_TIMES[i].second, end.hitch_times_ns[i], begin.hitch_times_ns[i]);
	}
	for (size_t i = 0; i < MEASURED_COUNTERS.size(); i++) {
		count(MEASURED_COUNTERS[i].second, end.counters[i], begin.counters[i]);
	}
	for (size_t i = 0; i < HITCH_COUNTERS.size(); i++) {
		count(HITCH_COUNTERS[i].second, end.hitch_counters[i], begin.hitch_counters[i]);
	}
	count("buffer_upload_kb", end.buffer_upload, begin.buffer_upload, 1024.0);
	count("texture_upload_kb", end.texture_upload, begin.texture_upload, 1024.0);
	count("buffer_download_kb", end.buffer_download, begin.buffer_download, 1024.0);
	size_t index = 0;
	for (const auto* counter = g_counters; counter != nullptr; counter = counter->Next()) {
		if (index < begin.named.size() && index < end.named.size()) {
			count(counter->Name(), end.named[index], begin.named[index]);
		}
		index++;
	}
	if (begin.process_cpu >= 0.0 && end.process_cpu >= 0.0) {
		costs.push_back({"cpu:process", 1000.0 * (end.process_cpu - begin.process_cpu), true});
	}
	for (const auto& [name, cpu]: end.thread_cpu) {
		const auto earlier = std::find_if(begin.thread_cpu.begin(), begin.thread_cpu.end(),
		                                  [&](const auto& entry) { return entry.first == name; });
		const double base  = earlier != begin.thread_cpu.end() ? earlier->second : 0.0;
		costs.push_back({"cpu:" + name, 1000.0 * (cpu - base), true});
	}
	return costs;
}

// The waits of guest threads (GuestWait), newest over oldest. A writer clears an entry's end
// time, fills it and stores the end time last; the reader takes an entry whose end time is the
// same before and after its copy.
struct WaitEvent {
	std::atomic<int64_t> end_ns {0};
	int64_t              begin_ns = 0;
	const char*          kind     = nullptr;
	uint64_t             address  = 0;
	uint64_t             bytes    = 0;
	uint64_t             thread   = 0;
};
constexpr size_t                  WAIT_EVENTS = 16384;
std::array<WaitEvent, WAIT_EVENTS> g_wait_events;
std::atomic<uint64_t>             g_wait_next {0};

uint64_t CurrentThreadNumber() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return GetCurrentThreadId();
#else
	return 0;
#endif
}

// " | held <thread> <kind> x<count> <ms>ms (longest <ms> at <address>, <KiB> KiB) ..." for the
// guest waits that ended in (from_ns, to_ns], the longest in total first; empty without any.
std::string HeldText(int64_t from_ns, int64_t to_ns) {
	struct Held {
		uint64_t    thread  = 0;
		const char* kind    = nullptr;
		uint64_t    count   = 0;
		int64_t     ns      = 0;
		int64_t     longest = 0;
		uint64_t    address = 0;
		uint64_t    bytes   = 0;
	};
	std::vector<Held> held;
	for (const auto& event: g_wait_events) {
		const auto end = event.end_ns.load(std::memory_order_acquire);
		if (end <= from_ns || end > to_ns) {
			continue;
		}
		const auto  begin   = event.begin_ns;
		const auto* kind    = event.kind;
		const auto  address = event.address;
		const auto  bytes   = event.bytes;
		const auto  thread  = event.thread;
		if (event.end_ns.load(std::memory_order_acquire) != end || kind == nullptr || begin > end) {
			continue;
		}
		auto known = std::find_if(held.begin(), held.end(), [&](const Held& other) {
			return other.thread == thread && other.kind == kind;
		});
		if (known == held.end()) {
			held.push_back({thread, kind});
			known = held.end() - 1;
		}
		known->count++;
		known->ns += end - begin;
		known->bytes += bytes;
		if (end - begin >= known->longest) {
			known->longest = end - begin;
			known->address = address;
		}
	}
	if (held.empty()) {
		return {};
	}
	std::sort(held.begin(), held.end(), [](const Held& a, const Held& b) { return a.ns > b.ns; });
	std::unordered_map<uint64_t, std::string> names;
	if (g_hooks.guest_threads != nullptr) {
		for (auto& [id, name]: g_hooks.guest_threads()) {
			std::replace(name.begin(), name.end(), ' ', '_');
			names[id] = std::move(name);
		}
	}
	std::string text = " | held";
	size_t      shown = 0;
	for (const auto& item: held) {
		if (shown++ == 8) {
			break;
		}
		const auto name = names.find(item.thread);
		text += Format(" %s %s x%" PRIu64 " %.1fms (longest %.1f at 0x%" PRIx64 ", %" PRIu64 " KiB);",
		               name != names.end() && !name->second.empty()
		                   ? name->second.c_str()
		                   : Format("thread%" PRIu64, item.thread).c_str(),
		               item.kind, item.count, static_cast<double>(item.ns) / 1e6,
		               static_cast<double>(item.longest) / 1e6, item.address, item.bytes >> 10u);
	}
	return text;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// The CPU times of the guest's own threads for a snapshot, as "guest:<name>" (threads of one
// name together). A handle is kept per thread and closed when the thread is gone or renamed.
class GuestThreadTimes {
public:
	void AddTo(Snapshot& snapshot) {
		if (g_hooks.guest_threads == nullptr) {
			return;
		}
		std::unordered_map<uint64_t, Kept> kept;
		for (const auto& [id, name]: g_hooks.guest_threads()) {
			HANDLE handle = nullptr;
			if (const auto known = m_kept.find(id); known != m_kept.end() && known->second.name == name) {
				handle = known->second.handle;
				m_kept.erase(known);
			} else {
				handle = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, static_cast<DWORD>(id));
			}
			if (handle == nullptr) {
				continue;
			}
			kept.emplace(id, Kept {handle, name});
			const double seconds = HandleCpuSeconds(handle);
			if (seconds < 0.0) {
				continue;
			}
			std::string label = "guest:" + (name.empty() ? std::to_string(id) : name);
			std::replace(label.begin(), label.end(), ' ', '_');
			const auto known = std::find_if(snapshot.thread_cpu.begin(), snapshot.thread_cpu.end(),
			                                [&](const auto& entry) { return entry.first == label; });
			if (known != snapshot.thread_cpu.end()) {
				known->second += seconds;
			} else {
				snapshot.thread_cpu.emplace_back(std::move(label), seconds);
			}
		}
		for (const auto& [id, gone]: m_kept) {
			CloseHandle(gone.handle);
		}
		m_kept = std::move(kept);
	}

private:
	struct Kept {
		HANDLE      handle = nullptr;
		std::string name;
	};
	std::unordered_map<uint64_t, Kept> m_kept;
};
#else
class GuestThreadTimes {
public:
	void AddTo(Snapshot& /*snapshot*/) {}
};
#endif

// KYTY_HITCH_LOG_MS=<milliseconds>: a line goes to the log for every frame that stayed on screen
// at least that long and at least half as long again as the frames before it (their median),
// with what it was made of beside the usual frame of the last second: the GPU thread's time, its
// waits by reason, what was compiled, uploaded, read and evicted, and the CPU time of the named
// threads. `measure` and KYTY_PERF_LOG average over seconds, in which one frame of 80 ms among
// 500 does not show; a player feels that one. A side thread takes a snapshot of the counters at
// every flip (it looks for one each millisecond), so the frame's own threads do no more for it
// than count. A value reads "<this frame>(<usual frame>)"; apart from the first three, only those
// far above the usual are listed, the times first.
void HitchLog(double min_ms) {
	constexpr size_t HISTORY     = 64;
	constexpr size_t MOST_COUNTS = 14;

	std::vector<Snapshot> history; // Oldest first; the last is of the newest flip seen.
	GuestThreadTimes      guest_times;
	uint64_t              seen = Detail::g_flips.load(std::memory_order_acquire);
	for (;;) {
		Thread::SleepNano(1'000'000);
		const auto flips = Detail::g_flips.load(std::memory_order_acquire);
		if (flips == seen) {
			continue;
		}
		seen     = flips;
		auto now = Take();
		guest_times.AddTo(now);
		const auto flip_ms = [](uint64_t flip) {
			const auto earlier = Detail::g_flip_times_ns[(flip - 1) % Detail::FLIP_TIMES].load(
			    std::memory_order_relaxed);
			const auto later =
			    Detail::g_flip_times_ns[flip % Detail::FLIP_TIMES].load(std::memory_order_relaxed);
			return static_cast<double>(later - earlier) / 1e6;
		};
		if (history.size() >= 8 && now.flips > history.back().flips && history.front().flips >= 1) {
			const auto& last  = history.back();
			const auto& first = history.front();
			// The longest frame since the last snapshot (one, unless this thread was late).
			double frame = 0.0;
			for (auto flip = last.flips; flip < now.flips; flip++) {
				frame = std::max(frame, flip_ms(flip));
			}
			std::vector<double> before;
			for (auto flip = first.flips; flip < last.flips; flip++) {
				before.push_back(flip_ms(flip));
			}
			double usual = 0.0;
			if (!before.empty()) {
				const auto middle = before.begin() + static_cast<std::ptrdiff_t>(before.size() / 2);
				std::nth_element(before.begin(), middle, before.end());
				usual = *middle;
			}
			if (frame >= min_ms && usual > 0.0 && frame >= 1.5 * usual) {
				const double usual_frames = static_cast<double>(last.flips - first.flips);
				std::unordered_map<std::string, double> usual_costs;
				for (const auto& cost: Costs(first, last)) {
					usual_costs[cost.name] = cost.value / std::max(usual_frames, 1.0);
				}
				struct Listed {
					std::string text;
					double      weight = 0.0;
				};
				std::vector<Listed> times;
				std::vector<Listed> counts;
				std::string         head;
				for (const auto& cost: Costs(last, now)) {
					const double usually = usual_costs[cost.name];
					const auto   text    = Format(cost.ms ? " %s=%.1f(%.1f)" : " %s=%.0f(%.1f)",
					                              cost.name.c_str(), cost.value, usually);
					if (cost.name == "gpu_thread" || cost.name == "gpu_wait" ||
					    cost.name == "starved") {
						head += text;
					} else if (cost.ms && cost.value - usually >= 1.0) {
						times.push_back({text, cost.value - usually});
					} else if (!cost.ms && cost.value >= 1.5 * usually + 1.0) {
						counts.push_back({text, (cost.value + 1.0) / (usually + 1.0)});
					}
				}
				const auto heavier = [](const Listed& a, const Listed& b) {
					return a.weight > b.weight;
				};
				std::sort(times.begin(), times.end(), heavier);
				std::sort(counts.begin(), counts.end(), heavier);
				if (counts.size() > MOST_COUNTS) {
					counts.resize(MOST_COUNTS);
				}
				const auto clock = std::time(nullptr);
				std::tm    local {};
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
				localtime_s(&local, &clock);
#else
				localtime_r(&clock, &local);
#endif
				auto line = Format("Hitch %02d:%02d:%02d frame_ms=%.1f usual=%.1f flip=%" PRIu64
				                   " ms this(usual) |%s |",
				                   local.tm_hour, local.tm_min, local.tm_sec, frame, usual,
				                   now.flips, head.c_str());
				for (const auto& item: times) {
					line += item.text;
				}
				line += " | counts";
				for (const auto& item: counts) {
					line += item.text;
				}
				line += HeldText(last.time_ns, now.time_ns);
				Log::WriteToConsoleAndLog(line + "\n");
			}
		}
		history.push_back(std::move(now));
		if (history.size() > HISTORY) {
			history.erase(history.begin());
		}
	}
}

// ---------------------------------------------------------------------------------------------
// prof

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

constexpr size_t SAMPLE_FRAMES = 48;

struct Sample {
	std::array<uint64_t, SAMPLE_FRAMES> frames {};
	uint32_t                            count = 0;
};

// One sample of a suspended thread: its pc and the return addresses of its callers, unwound with
// the functions' unwind data while the thread stands still (code without unwind data, generated
// or guest code, is taken as a leaf: its return address is the word at rsp). The walk stays
// inside the thread's stack. The unwind tables are read under the system's own shared lock,
// which a thread holds exclusively only while a module or a function table is added or removed.
// The walk of SampleThread, apart so that a fault in it is caught: a thread can be stopped at an
// instant its function's unwind data does not describe (a frame register that is not set up
// yet), and the system's unwinder then reads through whatever the registers hold. The frames
// found until then stand. (Such a fault ended the emulator: it reaches no handler of ours.)
void WalkSuspendedThread(CONTEXT& context, uint64_t stack_low, uint64_t stack_high,
                         Sample& sample) noexcept {
	__try {
		while (context.Rip != 0 && sample.count < SAMPLE_FRAMES) {
			sample.frames[sample.count++] = context.Rip;
			if (context.Rsp < stack_low || context.Rsp + 8 > stack_high || (context.Rsp & 7u) != 0) {
				break;
			}
			DWORD64 image_base = 0;
			auto*   function   = RtlLookupFunctionEntry(context.Rip, &image_base, nullptr);
			if (function != nullptr) {
				void*   handler_data = nullptr;
				DWORD64 establisher  = 0;
				RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, context.Rip, function, &context,
				                 &handler_data, &establisher, nullptr);
			} else {
				// NOLINTNEXTLINE(performance-no-int-to-ptr)
				context.Rip = *reinterpret_cast<const uint64_t*>(context.Rsp);
				context.Rsp += 8;
			}
		}
	} __except (EXCEPTION_EXECUTE_HANDLER) {
	}
}

bool SampleThread(const NamedThread& thread, Sample& sample) noexcept {
	// NOLINTNEXTLINE(performance-no-int-to-ptr)
	auto* handle = reinterpret_cast<HANDLE>(thread.handle);
	if (SuspendThread(handle) == static_cast<DWORD>(-1)) {
		return false;
	}
	CONTEXT context {};
	context.ContextFlags = CONTEXT_FULL;
	const bool ok        = GetThreadContext(handle, &context) != 0;
	if (ok) {
		sample.count = 0;
		WalkSuspendedThread(context, thread.stack_low, thread.stack_high, sample);
	}
	ResumeThread(handle);
	return ok;
}

// The function an address is in, "<module>" when the module has no symbol for it, "<code>" for
// memory outside every module (guest and generated code).
class Symbols {
public:
	Symbols() {
		SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_FAIL_CRITICAL_ERRORS);
		m_ready = SymInitialize(GetCurrentProcess(), nullptr, TRUE) != 0;
	}

	const std::string& Name(uint64_t address) {
		if (const auto known = m_names.find(address); known != m_names.end()) {
			return known->second;
		}
		std::string name;
		if (m_ready) {
			alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + 512] {};
			auto*                     info = reinterpret_cast<SYMBOL_INFO*>(buffer);
			info->SizeOfStruct             = sizeof(SYMBOL_INFO);
			info->MaxNameLen               = 511;
			DWORD64 displacement           = 0;
			// A symbol far before the address is the nearest export of a module without
			// symbols (a driver): the module's name says more.
			if (SymFromAddr(GetCurrentProcess(), address, &displacement, info) != 0 &&
			    displacement < 0x4000) {
				name.assign(info->Name, info->NameLen);
			}
		}
		if (name.empty()) {
			HMODULE module = nullptr;
			char    path[MAX_PATH] {};
			// NOLINTNEXTLINE(performance-no-int-to-ptr)
			if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
			                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
			                       reinterpret_cast<LPCSTR>(address), &module) != 0 &&
			    GetModuleBaseNameA(GetCurrentProcess(), module, path, MAX_PATH) != 0) {
				name = std::string("<") + path + ">";
			} else {
				name = "<code>";
			}
		}
		return m_names.emplace(address, std::move(name)).first->second;
	}

private:
	bool                                      m_ready = false;
	std::unordered_map<uint64_t, std::string> m_names;
};

// A function's name short enough for a report line.
std::string Short(const std::string& name) {
	return name.size() <= 110 ? name : name.substr(0, 107) + "...";
}

void Profile(uint64_t id, double seconds, const std::string& thread_name, size_t top,
             const std::string& focus) {
	const auto threads = Threads();
	const auto thread  = std::find_if(threads.begin(), threads.end(),
	                                  [&](const auto& entry) { return entry.name == thread_name; });
	if (thread == threads.end() || thread->handle == 0) {
		Answer(Format("LIVE error id=%" PRIu64 " prof: no thread named %s", id, thread_name.c_str()));
		return;
	}
	constexpr int64_t   PERIOD_NS = 500'000; // 2 kHz of wall time.
	std::vector<Sample> samples;
	samples.reserve(static_cast<size_t>(seconds * 2000.0) + 16);
	const auto end  = NowNs() + static_cast<int64_t>(seconds * 1e9);
	auto       next = NowNs();
	while (next < end) {
		Sample sample;
		if (SampleThread(*thread, sample)) {
			samples.push_back(sample);
		}
		next += PERIOD_NS;
		// The timer resolution is too coarse for the period: sleep while far, spin when near.
		for (auto now = NowNs(); now < next; now = NowNs()) {
			if (next - now > 2'000'000) {
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			} else {
				YieldProcessor();
			}
		}
	}
	if (samples.empty()) {
		Answer(Format("LIVE error id=%" PRIu64 " prof: no samples of %s", id, thread_name.c_str()));
		return;
	}

	static Symbols                            symbols;
	std::unordered_map<std::string, uint64_t> self;
	std::unordered_map<std::string, uint64_t> inclusive;
	// "under:<text>": what runs inside the outermost function whose name contains the text.
	static constexpr std::string_view Under = "under:";
	if (focus.starts_with(Under)) {
		const auto within  = focus.substr(Under.size());
		uint64_t   matched = 0;
		for (const auto& sample: samples) {
			uint32_t outer = UINT32_MAX;
			for (uint32_t i = 0; i < sample.count; i++) {
				if (symbols.Name(sample.frames[i]).find(within) != std::string::npos) {
					outer = i;
				}
			}
			if (outer == UINT32_MAX) {
				continue;
			}
			matched++;
			self[symbols.Name(sample.frames[0])]++;
			std::unordered_set<std::string_view> seen;
			for (uint32_t i = 0; i < outer; i++) {
				const auto& name = symbols.Name(sample.frames[i]);
				if (seen.insert(name).second) {
					inclusive[name]++;
				}
			}
		}
		Answer(Format("LIVE prof id=%" PRIu64 " thread=%s samples=%zu under %s: %.2f%% of them", id,
		              thread_name.c_str(), samples.size(), within.c_str(),
		              100.0 * static_cast<double>(matched) / static_cast<double>(samples.size())));
		const auto list = [&](const char* kind, const std::unordered_map<std::string, uint64_t>& counts) {
			std::vector<std::pair<std::string, uint64_t>> sorted(counts.begin(), counts.end());
			std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
				return a.second != b.second ? a.second > b.second : a.first < b.first;
			});
			for (size_t i = 0; i < std::min(top, sorted.size()); i++) {
				Answer(Format("LIVE prof id=%" PRIu64 " %s %5.2f%% %s", id, kind,
				              100.0 * static_cast<double>(sorted[i].second) /
				                  static_cast<double>(samples.size()),
				              Short(sorted[i].first).c_str()));
			}
		};
		list("self", self);
		list("incl", inclusive);
		Answer(Format("LIVE prof id=%" PRIu64 " end", id));
		return;
	}
	// With a focus: the callers of the innermost function whose name contains it, three deep.
	std::unordered_map<std::string, uint64_t> paths;
	for (const auto& sample: samples) {
		self[symbols.Name(sample.frames[0])]++;
		std::unordered_set<std::string_view> seen;
		bool                                 focused = focus.empty();
		for (uint32_t i = 0; i < sample.count; i++) {
			const auto& name = symbols.Name(sample.frames[i]);
			if (seen.insert(name).second) {
				inclusive[name]++;
			}
			if (!focused && name.find(focus) != std::string::npos) {
				focused = true;
				std::string        path;
				const std::string* previous = &name;
				for (uint32_t caller = i + 1, depth = 0; caller < sample.count && depth < 3;
				     caller++) {
					const auto& above = symbols.Name(sample.frames[caller]);
					if (above == *previous) {
						continue;
					}
					path += (path.empty() ? "" : " < ") + Short(above);
					previous = &above;
					depth++;
				}
				paths[path]++;
			}
		}
	}
	const auto report = [&](const char* kind, const std::unordered_map<std::string, uint64_t>& counts,
	                        bool whole = false) {
		std::vector<std::pair<std::string, uint64_t>> sorted(counts.begin(), counts.end());
		std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
			return a.second != b.second ? a.second > b.second : a.first < b.first;
		});
		for (size_t i = 0; i < std::min(top, sorted.size()); i++) {
			Answer(Format("LIVE prof id=%" PRIu64 " %s %5.2f%% %s", id, kind,
			              100.0 * static_cast<double>(sorted[i].second) /
			                  static_cast<double>(samples.size()),
			              (whole ? sorted[i].first : Short(sorted[i].first)).c_str()));
		}
	};
	Answer(Format("LIVE prof id=%" PRIu64 " thread=%s samples=%zu seconds=%.1f", id,
	              thread_name.c_str(), samples.size(), seconds));
	if (focus.empty()) {
		report("self", self);
		report("incl", inclusive);
	} else {
		report("path", paths, true);
	}
	Answer(Format("LIVE prof id=%" PRIu64 " end", id));
}

#else

void Profile(uint64_t id, double /*seconds*/, const std::string& /*thread_name*/, size_t /*top*/,
             const std::string& /*focus*/) {
	Answer(Format("LIVE error id=%" PRIu64 " prof: only on Windows", id));
}

#endif

// ---------------------------------------------------------------------------------------------
// pad

constexpr std::array PAD_BUTTONS {
    std::pair {"l3", 0x00000002u},       std::pair {"r3", 0x00000004u},
    std::pair {"options", 0x00000008u},  std::pair {"up", 0x00000010u},
    std::pair {"right", 0x00000020u},    std::pair {"down", 0x00000040u},
    std::pair {"left", 0x00000080u},     std::pair {"l2", 0x00000100u},
    std::pair {"r2", 0x00000200u},       std::pair {"l1", 0x00000400u},
    std::pair {"r1", 0x00000800u},       std::pair {"triangle", 0x00001000u},
    std::pair {"circle", 0x00002000u},   std::pair {"cross", 0x00004000u},
    std::pair {"square", 0x00008000u},   std::pair {"touchpad", 0x00100000u},
};

void Pad(uint64_t id, const std::string& button, int ms) {
	const auto known = std::find_if(PAD_BUTTONS.begin(), PAD_BUTTONS.end(),
	                                [&](const auto& entry) { return button == entry.first; });
	if (known == PAD_BUTTONS.end() || g_hooks.pad_button == nullptr) {
		Answer(Format("LIVE error id=%" PRIu64 " pad: unknown button %s", id, button.c_str()));
		return;
	}
	g_hooks.pad_button(known->second, true);
	std::this_thread::sleep_for(std::chrono::milliseconds(std::clamp(ms, 20, 60000)));
	g_hooks.pad_button(known->second, false);
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// ---------------------------------------------------------------------------------------------
// The command file

std::vector<std::string> Words(const std::string& line) {
	std::vector<std::string> words;
	std::istringstream       stream(line);
	for (std::string word; stream >> word;) {
		words.push_back(std::move(word));
	}
	return words;
}

// Guest bytes [address, address + size) into `data`; pages that are not backed read as zeros.
// False when none of it is backed.
bool ReadGuest(uint64_t address, uint8_t* data, uint64_t size) {
	if (g_hooks.guest_read(address, data, size)) {
		return true;
	}
	constexpr uint64_t Page = 0x4000;
	bool               any  = false;
	for (uint64_t at = address; at < address + size;) {
		const auto next = std::min(address + size, (at / Page + 1) * Page);
		if (g_hooks.guest_read(at, data + (at - address), next - at)) {
			any = true;
		} else {
			std::memset(data + (at - address), 0, static_cast<size_t>(next - at));
		}
		at = next;
	}
	return any;
}

// The addresses in [from, to) of the guest's mappings where `pattern` stands, at multiples of
// `align`, at most `limit` of them; `scanned` gets the bytes looked at.
std::vector<uint64_t> FindGuest(const std::vector<uint8_t>& pattern, uint64_t align, size_t limit,
                                uint64_t from, uint64_t to, uint64_t& scanned) {
	return g_hooks.guest_find(pattern.data(), pattern.size(), align, limit, from, to, &scanned);
}

std::vector<uint8_t> HexBytes(const std::string& text) {
	std::vector<uint8_t> bytes;
	for (size_t i = 0; i + 1 < text.size(); i += 2) {
		bytes.push_back(static_cast<uint8_t>(std::strtoul(text.substr(i, 2).c_str(), nullptr, 16)));
	}
	return bytes;
}

// mem maps                                   the guest's mappings: how many, their bytes
// mem find <hex bytes> [max] [from] [to]     where the bytes stand
// mem find16 <text> [max] [from] [to]        the text as a zero-terminated UTF-16 string
// mem findq <hex value> [max] [from] [to]    the value as an aligned 64-bit word (a pointer)
// mem read <hex address> <count>             up to 1024 bytes, as hex
// mem write <hex address> <hex bytes>        through the backing store
// mem dump <hex address> <hex size> <file>   the range into a file (unbacked pages as zeros)
// Addresses and values are hexadecimal. Reads and writes go through the backing store: they see
// what the guest's CPU stored, not what the GPU has written and nobody read back.
void Mem(uint64_t id, const std::vector<std::string>& words) {
	if (g_hooks.guest_read == nullptr || g_hooks.guest_next_mapping == nullptr ||
	    g_hooks.guest_write == nullptr || g_hooks.guest_find == nullptr || words.size() < 2) {
		Answer(Format("LIVE error id=%" PRIu64 " mem: not available", id));
		return;
	}
	const auto hex = [&](size_t index, uint64_t fallback) {
		return words.size() > index ? std::strtoull(words[index].c_str(), nullptr, 16) : fallback;
	};
	const auto& what = words[1];
	if (what == "maps") {
		uint64_t count = 0;
		uint64_t bytes = 0;
		uint64_t start = 0;
		uint64_t end   = 0;
		for (uint64_t at = 0; g_hooks.guest_next_mapping(at, &start, &end) && end > at; at = end) {
			count++;
			bytes += end - start;
		}
		Answer(Format("LIVE mem id=%" PRIu64 " maps: %" PRIu64 " mappings, %" PRIu64 " MiB", id, count,
		              bytes >> 20u));
	} else if ((what == "find" || what == "find16" || what == "findq") && words.size() >= 3) {
		std::vector<uint8_t> pattern;
		uint64_t             align = 1;
		if (what == "find") {
			pattern = HexBytes(words[2]);
		} else if (what == "find16") {
			for (const char c: words[2]) {
				pattern.push_back(static_cast<uint8_t>(c));
				pattern.push_back(0);
			}
			pattern.push_back(0);
			pattern.push_back(0);
			align = 2;
		} else {
			const auto value = hex(2, 0);
			pattern.resize(sizeof(value));
			std::memcpy(pattern.data(), &value, sizeof(value));
			align = 8;
		}
		if (pattern.empty()) {
			Answer(Format("LIVE error id=%" PRIu64 " mem: nothing to find", id));
			return;
		}
		const auto limit   = static_cast<size_t>(std::clamp<uint64_t>(
            words.size() > 3 ? std::strtoull(words[3].c_str(), nullptr, 10) : 16u, 1u, 4096u));
		uint64_t   scanned = 0;
		const auto begin   = std::chrono::steady_clock::now();
		const auto found   = FindGuest(pattern, align, limit, hex(4, 0), hex(5, UINT64_MAX), scanned);
		std::string line   = Format(
            "LIVE mem id=%" PRIu64 " %s: %zu found in %" PRIu64 " MiB, %.1f s:", id, what.c_str(),
            found.size(), scanned >> 20u,
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
		for (const auto address: found) {
			line += Format(" %" PRIx64, address);
		}
		Answer(line);
	} else if (what == "read" && words.size() >= 4) {
		const auto address = hex(2, 0);
		const auto count   = std::clamp<uint64_t>(std::strtoull(words[3].c_str(), nullptr, 10), 1, 1024);
		std::vector<uint8_t> bytes(static_cast<size_t>(count));
		if (!ReadGuest(address, bytes.data(), count)) {
			Answer(Format("LIVE error id=%" PRIu64 " mem: %" PRIx64 " is not mapped", id, address));
			return;
		}
		std::string line = Format("LIVE mem id=%" PRIu64 " read %" PRIx64 ":", id, address);
		for (size_t i = 0; i < bytes.size(); i++) {
			line += Format(i % 8 == 0 ? " %02x" : "%02x", bytes[i]);
		}
		Answer(line);
	} else if (what == "dump" && words.size() >= 5) {
		const auto address = hex(2, 0);
		const auto size    = std::min<uint64_t>(hex(3, 0), uint64_t {1} << 30u);
		std::vector<uint8_t> bytes(static_cast<size_t>(size));
		const bool           any = size != 0 && ReadGuest(address, bytes.data(), size);
		std::ofstream        file(words[4], std::ios::binary | std::ios::trunc);
		file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
		Answer(Format("LIVE mem id=%" PRIu64 " dump %" PRIx64 ": %zu bytes to %s%s", id, address, bytes.size(),
		              words[4].c_str(), any ? "" : " (nothing of it is mapped)"));
	} else if (what == "write" && words.size() >= 4) {
		const auto address = hex(2, 0);
		const auto bytes   = HexBytes(words[3]);
		const bool done    = !bytes.empty() && g_hooks.guest_write(address, bytes.data(), bytes.size());
		Answer(Format("LIVE mem id=%" PRIu64 " write %" PRIx64 ": %zu bytes %s", id, address, bytes.size(),
		              done ? "written" : "NOT written (not mapped)"));
	} else {
		Answer(Format("LIVE error id=%" PRIu64 " mem: maps | find | find16 | findq | read | write | dump", id));
	}
}

void Execute(uint64_t id, const std::string& line) {
	const auto words = Words(line);
	if (words.empty() || words[0].starts_with('#')) {
		return;
	}
	const auto& command = words[0];
	const auto  number  = [&](size_t index, double fallback) {
		return words.size() > index ? std::strtod(words[index].c_str(), nullptr) : fallback;
	};
	if (command == "measure" && words.size() >= 2) {
		Measure(id, std::clamp(number(1, 10.0), 0.1, 3600.0), words.size() > 2 ? words[2] : "-");
	} else if (command == "fps" && words.size() >= 2) {
		Fps(id, std::clamp(number(1, 10.0), 0.1, 3600.0), words.size() > 2 ? words[2] : "-");
	} else if (command == "prof" && words.size() >= 2) {
		Profile(id, std::clamp(number(1, 10.0), 0.1, 600.0), words.size() > 2 ? words[2] : "gpu",
		        static_cast<size_t>(std::clamp(number(3, 40.0), 1.0, 500.0)),
		        words.size() > 4 ? words[4] : std::string {});
	} else if (command == "set" && words.size() == 3) {
		for (auto* entry = g_switches; entry != nullptr; entry = const_cast<Switch*>(entry->Next())) {
			if (words[1] == entry->Name()) {
				const auto previous = entry->Get();
				entry->Set(static_cast<uint32_t>(std::strtoul(words[2].c_str(), nullptr, 0)));
				Answer(Format("LIVE set id=%" PRIu64 " %s=%u (was %u)", id, entry->Name(),
				              entry->Get(), previous));
				return;
			}
		}
		Answer(Format("LIVE error id=%" PRIu64 " set: no switch named %s", id, words[1].c_str()));
	} else if (command == "switches") {
		std::string list = Format("LIVE switches id=%" PRIu64, id);
		for (const auto* entry = g_switches; entry != nullptr; entry = entry->Next()) {
			list += Format(" %s=%u", entry->Name(), entry->Get());
		}
		Answer(list);
	} else if (command == "mem") {
		Mem(id, words);
	} else if (command == "pad" && words.size() >= 2) {
		Pad(id, words[1], static_cast<int>(number(2, 120.0)));
	} else if (command == "stick" && words.size() == 6 && g_hooks.pad_sticks != nullptr) {
		const auto axis = [&](size_t index) {
			return std::clamp(static_cast<int>(number(index, 128.0)), 0, 255);
		};
		g_hooks.pad_sticks(axis(1), axis(2), axis(3), axis(4));
		std::this_thread::sleep_for(
		    std::chrono::milliseconds(std::clamp(static_cast<int>(number(5, 0.0)), 0, 600000)));
		g_hooks.pad_sticks(128, 128, 128, 128);
	} else if (command == "sleep" && words.size() == 2) {
		std::this_thread::sleep_for(
		    std::chrono::milliseconds(std::clamp(static_cast<int>(number(1, 0.0)), 0, 600000)));
	} else if (command == "wait" && g_hooks.pending_builds != nullptr) {
		const auto start    = std::chrono::steady_clock::now();
		const auto limit    = std::chrono::duration<double>(std::clamp(number(1, 300.0), 0.0, 3600.0));
		uint64_t   pending  = g_hooks.pending_builds();
		uint64_t   at_start = pending;
		// Idle for a second: builds queue as the guest creates its shaders.
		auto idle_since = start;
		for (;;) {
			const auto now = std::chrono::steady_clock::now();
			if (pending != 0) {
				idle_since = now;
			}
			if (now - idle_since >= std::chrono::seconds {1} || now - start >= limit) {
				break;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(50));
			pending = g_hooks.pending_builds();
		}
		Answer(Format("LIVE wait id=%" PRIu64 " pending=%" PRIu64 " (was %" PRIu64 ") waited=%.1fs",
		              id, pending, at_start,
		              std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
		                  .count()));
	} else if (command == "mark") {
		Mark();
		Answer(Format("LIVE mark id=%" PRIu64, id));
	} else if (command == "threads") {
		std::string list = Format("LIVE threads id=%" PRIu64, id);
		for (const auto& thread: Threads()) {
			list += Format(" %s:%u=%.2fs", thread.name.c_str(), thread.tid, ThreadCpuSeconds(thread));
		}
		Answer(list);
	} else if (command == "quit" && g_hooks.quit != nullptr) {
		Answer(Format("LIVE quit id=%" PRIu64, id));
		g_hooks.quit();
	} else {
		Answer(Format("LIVE error id=%" PRIu64 " line=%s", id, line.c_str()));
	}
}

void Run() {
	uint64_t last_id = 0;
	for (;;) {
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		std::ifstream file(g_command_path);
		std::string   line;
		if (!file || !std::getline(file, line)) {
			continue;
		}
		const auto head = Words(line);
		if (head.size() != 2 || head[0] != "id") {
			continue;
		}
		const auto id = std::strtoull(head[1].c_str(), nullptr, 10);
		if (id == last_id) {
			continue;
		}
		last_id = id;
		std::vector<std::string> commands;
		while (std::getline(file, line)) {
			if (!line.empty() && line.back() == '\r') {
				line.pop_back();
			}
			commands.push_back(line);
		}
		file.close();
		for (const auto& command: commands) {
			Execute(id, command);
		}
		Answer(Format("LIVE done id=%" PRIu64, id));
	}
}

} // namespace

Switch::Switch(const char* name, uint32_t default_value) noexcept
    : m_name(name), m_value(default_value), m_next(g_switches) {
	g_switches = this;
	char variable[96] {};
	std::snprintf(variable, sizeof(variable), "KYTY_%s", name);
	for (char* c = variable; *c != '\0'; c++) {
		*c = static_cast<char>(std::toupper(static_cast<unsigned char>(*c)));
	}
	if (const char* text = std::getenv(variable); text != nullptr && *text != '\0') {
		m_value.store(static_cast<uint32_t>(std::strtoul(text, nullptr, 0)),
		              std::memory_order_relaxed);
	}
}

Counter::Counter(const char* name) noexcept: m_name(name), m_next(g_counters) {
	g_counters = this;
}

void SetHostHooks(const HostHooks& hooks) noexcept {
	g_hooks = hooks;
}

void Detail::NoteGuestWait(const char* kind, uint64_t address, uint64_t bytes, int64_t begin_ns,
                           int64_t least_ns) noexcept {
	const auto end = SteadyNs();
	if (end - begin_ns < least_ns) {
		return;
	}
	auto& event = g_wait_events[g_wait_next.fetch_add(1, std::memory_order_relaxed) % WAIT_EVENTS];
	event.end_ns.store(0, std::memory_order_release);
	event.begin_ns = begin_ns;
	event.kind     = kind;
	event.address  = address;
	event.bytes    = bytes;
	event.thread   = CurrentThreadNumber();
	event.end_ns.store(end, std::memory_order_release);
}

void RegisterThread(const char* name) noexcept {
	NamedThread thread;
	thread.name          = name;
	Detail::t_gpu_thread = thread.name == "gpu";
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	HANDLE handle = nullptr;
	if (DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle, 0,
	                    FALSE, DUPLICATE_SAME_ACCESS) != 0) {
		thread.handle = reinterpret_cast<uint64_t>(handle);
	}
	thread.tid = GetCurrentThreadId();
	ULONG_PTR low  = 0;
	ULONG_PTR high = 0;
	GetCurrentThreadStackLimits(&low, &high);
	thread.stack_low  = low;
	thread.stack_high = high;
#endif
	std::scoped_lock lock(g_threads_mutex);
	g_threads.push_back(std::move(thread));
}

void Mark() noexcept {
	static std::atomic<uint32_t> marks {0};
	const auto                   number = marks.fetch_add(1, std::memory_order_relaxed) + 1;
	// (Flip times are kept from the first mark on, where no log had them kept already.)
	Detail::g_active.store(true, std::memory_order_relaxed);
	const auto clock = std::time(nullptr);
	std::tm    local {};
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	localtime_s(&local, &clock);
#else
	localtime_r(&clock, &local);
#endif
	Log::WriteToConsoleAndLog(Format("Mark %02d:%02d:%02d #%u flip=%" PRIu64 "\n", local.tm_hour,
	                                 local.tm_min, local.tm_sec, number,
	                                 Detail::g_flips.load(std::memory_order_acquire)));
	// What the two seconds after it were made of.
	std::thread([number] {
		Log::WriteToConsoleAndLog(MeasureLine(Format("MarkPerf #%u", number), 2.0) + "\n");
	}).detach();
}

void Start() {
	if (const char* text = std::getenv("KYTY_PERF_LOG"); text != nullptr) {
		if (const double seconds = std::strtod(text, nullptr); seconds >= 1.0) {
			Detail::g_active.store(true, std::memory_order_relaxed);
			// (On for the whole session, without the performance panel.)
			DebugCounters::Hold();
			std::thread(PerfLog, std::min(seconds, 600.0)).detach();
		}
	}
	if (const char* text = std::getenv("KYTY_HITCH_LOG_MS"); text != nullptr) {
		if (const double ms = std::strtod(text, nullptr); ms >= 1.0) {
			Detail::g_active.store(true, std::memory_order_relaxed);
			Detail::g_guest_waits.store(true, std::memory_order_relaxed);
			DebugCounters::Hold();
			std::thread(HitchLog, ms).detach();
		}
	}
	const char* path = std::getenv("KYTY_LIVE_FILE");
	if (path == nullptr || *path == '\0') {
		return;
	}
	g_command_path = path;
	g_answer_path  = g_command_path + ".out";
	Detail::g_active.store(true, std::memory_order_relaxed);
	std::string list;
	for (const auto* entry = g_switches; entry != nullptr; entry = entry->Next()) {
		list += Format(" %s=%u", entry->Name(), entry->Get());
	}
	Answer("LIVE start file=" + g_command_path + " switches:" + list);
	std::thread(Run).detach();
}

} // namespace Common::LiveControl
