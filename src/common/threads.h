#ifndef KYTY_COMMON_THREADS_H_
#define KYTY_COMMON_THREADS_H_

#include "common/common.h"

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
#include <condition_variable>
#include <mutex>
#endif

namespace Common {

void InitializeThreads();

using wait_poll_func_t = void (*)();

namespace Thread {

void SleepMicro(uint32_t micros);
void SleepNano(uint64_t nanos);
bool IsMainThread();
// Unique per thread; never reused by another thread.
int GetThreadIdUnique();
// Lets the calling thread yield the CPU to normal-priority threads (background work such as
// pipeline builds). Best effort: does nothing where the platform refuses.
void LowerCurrentThreadPriority();
// Lets the calling thread run ahead of normal-priority threads (the thread that translates guest
// GPU commands: the guest's own threads otherwise hold its CPUs for whole time slices while they
// load). Best effort.
void RaiseCurrentThreadPriority();
// The logical CPUs the process may use: those of its affinity mask where the platform has one,
// all of the host's otherwise. At least 1.
uint32_t AvailableProcessors();
// The CPUs the calling thread may run on, from a list such as "2,3,8-15" (logical CPUs 0-63 of
// its processor group); other threads are not kept off them. "auto" names the performance cores
// without the core of CPU 0, which serves most interrupts. False when nothing was changed (an
// empty or unusable list, a CPU without core classes, or a platform that refuses).
bool PinCurrentThreadToCpus(const char* list);
// Keeps the calling thread on the efficiency cores of a processor with two kinds of cores, for
// work beside the game that may take its time: the performance cores stay with the game's
// threads and the threads that feed the GPU. False, nothing changed, on a processor with one
// kind of core, with fewer than eight efficiency CPUs (the work would crowd them), or where the
// platform refuses.
bool PinCurrentThreadToEfficiencyCpus();

} // namespace Thread

// Recursive.
class Mutex {
public:
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	Mutex();
	~Mutex();

	void Lock();
	void Unlock();

	KYTY_CLASS_NO_COPY(Mutex);

private:
	// A CRITICAL_SECTION.
	alignas(8) unsigned char m_cs[40];
#else
	void Lock() { m_mutex.lock(); }
	void Unlock() { m_mutex.unlock(); }

private:
	std::recursive_mutex m_mutex;
#endif

	friend class CondVar;
};

class CondVar {
public:
	CondVar() = default;

	void Wait(Mutex* mutex);
	bool WaitFor(Mutex* mutex, uint32_t micros);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	void Signal();
	void SignalAll();
#else
	void Signal() { m_cv.notify_one(); }
	void SignalAll() { m_cv.notify_all(); }
#endif

	static void SetWaitPollCallback(wait_poll_func_t callback);

	KYTY_CLASS_NO_COPY(CondVar);

private:
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// A CONDITION_VARIABLE, whose initial state is zero.
	void* m_cv = nullptr;
#else
	std::condition_variable_any m_cv;
#endif
};

class LockGuard {
public:
	using mutex_type = Mutex;

	// NOLINTNEXTLINE(google-runtime-references)
	explicit LockGuard(mutex_type& m): m_mutex(m) { m_mutex.Lock(); }

	~LockGuard() { m_mutex.Unlock(); }

	KYTY_CLASS_NO_COPY(LockGuard);

private:
	mutex_type& m_mutex;
};

} // namespace Common

#endif /* KYTY_COMMON_THREADS_H_ */
