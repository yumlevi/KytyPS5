#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_BACKOFFSPINLOCK_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_BACKOFFSPINLOCK_H_

#include "common/liveControl.h"

#include <atomic>
#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <xmmintrin.h>
#endif

namespace Libs::Graphics {

// 1: a thread that finds a page tracking lock taken spins a little and then sleeps on the lock
// word until the holder lets go; 0: it spins until then, as before.
inline Common::LiveControl::Switch g_lock_sleep("lock_sleep", 1);

// The lock of a tracking region and of a page manager region. It is held for a few hundred
// nanoseconds as a rule, which is what a spin lock is for; but page protection changes are
// made under it (a system call each, with every CPU of the process told to drop its
// translations), and its holder can lose its CPU like any thread. Guest threads take these
// locks in their write faults, a dozen of them at once while a level streams in, and each of
// them spun for as long as that lasted: CPUs the game's other threads and the GPU thread could
// not have (the Demon's Souls fork, 02f1994c9, and the Ghost of Yotei fork, cc05e4c2a, sleep
// after a spin for the same reason). After a few microseconds of spinning the waiter sleeps on
// the lock word; a release wakes one sleeper when there is any, which costs a release without
// sleepers one load.
class BackoffSpinLock final {
public:
	void lock() noexcept {
		uint32_t spins = 0;
		while (m_flag.test_and_set(std::memory_order_acquire)) {
			// Wait on reads, so the holder's cache line is not written until the lock is free.
			while (m_flag.test(std::memory_order_relaxed)) {
				if (++spins < Spins || !g_lock_sleep.On()) {
					Pause();
					continue;
				}
				// (The count is raised before the flag is read again, inside wait, and a
				// release clears the flag before it reads the count: one of the two sees the
				// other.)
				m_sleepers.fetch_add(1, std::memory_order_seq_cst);
				m_flag.wait(true, std::memory_order_seq_cst);
				m_sleepers.fetch_sub(1, std::memory_order_relaxed);
				spins = 0;
			}
		}
	}

	void unlock() noexcept {
		m_flag.clear(std::memory_order_seq_cst);
		if (m_sleepers.load(std::memory_order_seq_cst) != 0) {
			m_flag.notify_one();
		}
	}

private:
	static constexpr uint32_t Spins = 256;

	static void Pause() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
		_mm_pause();
#elif defined(_M_ARM64)
		__yield();
#elif defined(__aarch64__)
		asm volatile("yield");
#endif
	}

	std::atomic_flag     m_flag = ATOMIC_FLAG_INIT;
	std::atomic_uint32_t m_sleepers {0};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_BACKOFFSPINLOCK_H_
