#include "common/threads.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono> // IWYU pragma: keep
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <thread>

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
#include <sys/resource.h>
#include <unistd.h>
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
// IWYU pragma: no_include <winbase.h>
constexpr DWORD KYTY_CS_SPIN_COUNT = 4000;

#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

// High-resolution waitable timers need Windows 10 1803, the floor MapViewOfFile3 already sets.
static void SleepHighResolution100ns(uint64_t units_100ns) {
	if (units_100ns == 0) {
		return;
	}
	// Closed when the thread exits.
	thread_local const std::unique_ptr<void, decltype(&CloseHandle)> timer {
	    CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
		                       TIMER_ALL_ACCESS),
	    &CloseHandle};
	LARGE_INTEGER due_time {};
	due_time.QuadPart = -static_cast<LONGLONG>(units_100ns);
	if (SetWaitableTimerEx(timer.get(), &due_time, 0, nullptr, nullptr, nullptr, 0) != 0) {
		WaitForSingleObject(timer.get(), INFINITE);
	}
}

// IWYU pragma: no_include <minwindef.h>
// IWYU pragma: no_include <synchapi.h>
// IWYU pragma: no_include <minwinbase.h>
// IWYU pragma: no_include <errhandlingapi.h>
// IWYU pragma: no_include <winerror.h>

static CRITICAL_SECTION* ToCs(unsigned char* storage) {
	return reinterpret_cast<CRITICAL_SECTION*>(storage);
}

static CONDITION_VARIABLE* ToCv(void** storage) {
	return reinterpret_cast<CONDITION_VARIABLE*>(storage);
}

#endif

namespace Common {

static wait_poll_func_t g_cond_wait_poll_callback = nullptr;

static std::thread::id g_main_thread;

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// Windows runs the threads of a process whose window is not in the foreground at reduced speed
// where it can: on a CPU with performance and efficiency cores it places them on the efficiency
// cores, and it may coarsen their timers. The emulator's work is the same whether or not its
// window has the focus (a second monitor, a capture, an overlay in front): with the window in
// the background Saros ran at 24.7 flips/s on an i9-13900K and at 37.6 once its threads were
// kept on the performance cores. Opting out of both leaves the scheduler free to use every core
// as for a foreground process. KYTY_POWER_THROTTLING=1 keeps the system's default.
static void OptOutOfPowerThrottling() {
	if (const char* keep = std::getenv("KYTY_POWER_THROTTLING");
	    keep != nullptr && std::strcmp(keep, "1") == 0) {
		return;
	}
#ifndef PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION
#define PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION 0x4
#endif
	PROCESS_POWER_THROTTLING_STATE state {};
	state.Version     = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
	state.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED |
	                    PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION;
	state.StateMask   = 0;
	if (SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &state, sizeof(state)) ==
	    0) {
		// Before Windows 11 the timer resolution control is unknown: the execution speed alone.
		state.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
		(void)SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &state,
		                            sizeof(state));
	}
}
#endif

void InitializeThreads() {
	g_main_thread = std::this_thread::get_id();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	OptOutOfPowerThrottling();
#endif
}

void Thread::SleepMicro(uint32_t micros) {
	SleepNano(uint64_t {micros} * 1000);
}

void Thread::SleepNano(uint64_t nanos) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	SleepHighResolution100ns((nanos + 99) / 100);
#elif defined(__APPLE__) // no clock_nanosleep
	std::this_thread::sleep_for(std::chrono::nanoseconds(nanos));
#else
	if (nanos == 0) {
		return;
	}
	constexpr uint64_t NANOS_PER_SEC = 1000000000;
	// An absolute deadline preserves the requested sleep across signal interruptions.
	timespec deadline {};
	clock_gettime(CLOCK_MONOTONIC, &deadline);
	auto target_nsec = static_cast<uint64_t>(deadline.tv_nsec) + nanos;
	deadline.tv_sec += static_cast<time_t>(target_nsec / NANOS_PER_SEC);
	deadline.tv_nsec = static_cast<long>(target_nsec % NANOS_PER_SEC);
	while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &deadline, nullptr) == EINTR) {
	}
#endif
}

bool Thread::IsMainThread() {
	return g_main_thread == std::this_thread::get_id();
}

void Thread::LowerCurrentThreadPriority() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	(void)SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	// Linux applies nice values per thread.
	(void)setpriority(PRIO_PROCESS, static_cast<id_t>(gettid()), 5);
#endif
}

uint32_t Thread::AvailableProcessors() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	DWORD_PTR process = 0;
	DWORD_PTR system  = 0;
	if (GetProcessAffinityMask(GetCurrentProcess(), &process, &system) != 0 && process != 0) {
		uint32_t count = 0;
		for (; process != 0; process &= process - 1) {
			count++;
		}
		// (A mask covers one processor group; the host may have more CPUs than it names.)
		return count;
	}
#endif
	return std::max(std::thread::hardware_concurrency(), 1u);
}

void Thread::RaiseCurrentThreadPriority() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	(void)SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	// Needs CAP_SYS_NICE or a nice rlimit; refused silently otherwise.
	(void)setpriority(PRIO_PROCESS, static_cast<id_t>(gettid()), -5);
#endif
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// The logical CPUs (bit n: CPU n) the process may use of the fastest core class, without the
// core that CPU 0 belongs to, or (`performance` false) of every slower class; 0 when the CPU
// has one class only or the classes are unknown.
static DWORD_PTR CoreClassCpuMask(bool performance) {
	DWORD_PTR process = 0;
	DWORD_PTR system  = 0;
	if (GetProcessAffinityMask(GetCurrentProcess(), &process, &system) == 0) {
		return 0;
	}
	ULONG length = 0;
	(void)GetSystemCpuSetInformation(nullptr, 0, &length, GetCurrentProcess(), 0);
	if (length == 0) {
		return 0;
	}
	const auto buffer = std::make_unique<unsigned char[]>(length);
	auto*      first  = reinterpret_cast<SYSTEM_CPU_SET_INFORMATION*>(buffer.get());
	if (GetSystemCpuSetInformation(first, length, &length, GetCurrentProcess(), 0) == 0) {
		return 0;
	}
	unsigned  fastest         = 0;
	unsigned  slowest         = 255;
	int       interrupt_core  = -1;
	const auto for_each = [&](const auto& visit) {
		for (ULONG offset = 0; offset < length;) {
			const auto* info =
			    reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.get() + offset);
			if (info->Type == CpuSetInformation && info->CpuSet.Group == 0 &&
			    info->CpuSet.LogicalProcessorIndex < 64 &&
			    ((process >> info->CpuSet.LogicalProcessorIndex) & 1u) != 0) {
				visit(info->CpuSet);
			}
			offset += info->Size;
		}
	};
	for_each([&](const auto& cpu) {
		fastest = std::max<unsigned>(fastest, cpu.EfficiencyClass);
		slowest = std::min<unsigned>(slowest, cpu.EfficiencyClass);
		if (cpu.LogicalProcessorIndex == 0) {
			interrupt_core = cpu.CoreIndex;
		}
	});
	if (fastest == slowest) {
		return 0;
	}
	DWORD_PTR mask = 0;
	for_each([&](const auto& cpu) {
		if (performance ? cpu.EfficiencyClass == fastest && cpu.CoreIndex != interrupt_core
		                : cpu.EfficiencyClass != fastest) {
			mask |= DWORD_PTR {1} << cpu.LogicalProcessorIndex;
		}
	});
	return mask;
}
#endif

bool Thread::PinCurrentThreadToEfficiencyCpus() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static const DWORD_PTR mask = [] {
		const auto cpus  = CoreClassCpuMask(false);
		int        count = 0;
		for (auto rest = cpus; rest != 0; rest &= rest - 1) {
			count++;
		}
		return count >= 8 ? cpus : DWORD_PTR {0};
	}();
	return mask != 0 && SetThreadAffinityMask(GetCurrentThread(), mask) != 0;
#else
	return false;
#endif
}

bool Thread::PinCurrentThreadToCpus(const char* list) {
	if (list == nullptr || *list == '\0') {
		return false;
	}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	DWORD_PTR mask = 0;
	if (std::strcmp(list, "auto") == 0) {
		mask = CoreClassCpuMask(true);
	} else {
		// "a,b,c-d".
		for (const char* text = list; *text != '\0';) {
			char*      end   = nullptr;
			const auto first = std::strtoul(text, &end, 10);
			if (end == text) {
				return false;
			}
			auto last = first;
			if (*end == '-') {
				const char* range = end + 1;
				last              = std::strtoul(range, &end, 10);
				if (end == range) {
					return false;
				}
			}
			for (auto cpu = first; cpu <= last && cpu < 64; cpu++) {
				mask |= DWORD_PTR {1} << cpu;
			}
			text = *end == ',' ? end + 1 : end;
			if (*end != ',' && *end != '\0') {
				return false;
			}
		}
	}
	return mask != 0 && SetThreadAffinityMask(GetCurrentThread(), mask) != 0;
#else
	return false;
#endif
}

int Thread::GetThreadIdUnique() {
	static std::atomic<int> counter = 0;
	static thread_local int tid     = ++counter;
	return tid;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
Mutex::Mutex() {
	static_assert(sizeof(CRITICAL_SECTION) == sizeof(m_cs) &&
	              alignof(CRITICAL_SECTION) <= alignof(Mutex));
	InitializeCriticalSectionAndSpinCount(ToCs(m_cs), KYTY_CS_SPIN_COUNT);
}

Mutex::~Mutex() {
	DeleteCriticalSection(ToCs(m_cs));
}

void Mutex::Lock() {
	EnterCriticalSection(ToCs(m_cs));
}

void Mutex::Unlock() {
	LeaveCriticalSection(ToCs(m_cs));
}

void CondVar::Signal() {
	WakeConditionVariable(ToCv(&m_cv));
}

void CondVar::SignalAll() {
	WakeAllConditionVariable(ToCv(&m_cv));
}
#endif

void CondVar::Wait(Mutex* mutex) {
	auto* callback = g_cond_wait_poll_callback;
	if (callback == nullptr) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SleepConditionVariableCS(ToCv(&m_cv), ToCs(mutex->m_cs), INFINITE);
#else
		m_cv.wait(mutex->m_mutex);
#endif
	} else if (!WaitFor(mutex, 10000)) {
		mutex->Unlock();
		callback();
		mutex->Lock();
	}
}

void CondVar::SetWaitPollCallback(wait_poll_func_t callback) {
	g_cond_wait_poll_callback = callback;
}

bool CondVar::WaitFor(Mutex* mutex, uint32_t micros) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return SleepConditionVariableCS(ToCv(&m_cv), ToCs(mutex->m_cs),
	                                micros < 1000 ? 1 : micros / 1000) != 0 ||
	       GetLastError() != ERROR_TIMEOUT;
#else
	return m_cv.wait_for(mutex->m_mutex, std::chrono::microseconds(micros)) ==
	       std::cv_status::no_timeout;
#endif
}

} // namespace Common
