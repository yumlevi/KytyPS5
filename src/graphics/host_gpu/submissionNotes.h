#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_SUBMISSIONNOTES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_SUBMISSIONNOTES_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>

// What the last submissions held, for the record of a lost device. The failing call names the
// submission the GPU did not finish ("waiting for tick 3407, GPU at 3406"), and the driver names
// an address at best; what was in that submission (how many guest dispatches and which shaders
// last, render pass instances, uploads, downloads, the upscaler) says where to look.
namespace Libs::Graphics::SubmissionNotes {

struct Note {
	static constexpr size_t Shaders = 6;

	uint64_t tick       = 0;
	uint32_t dispatches = 0;
	uint32_t passes     = 0;
	uint32_t uploads    = 0;
	uint32_t commands   = 0;
	// Bits of CommandBuffer::ProfileScope: texture uploads, downloads, the upscaler.
	uint32_t scopes = 0;
	// The shaders of the last guest dispatches, oldest first.
	std::array<uint64_t, Shaders> shaders {};
	uint32_t                      shader_count = 0;

	void AddShader(uint64_t shader) noexcept {
		if (shader_count == Shaders) {
			for (size_t i = 1; i < Shaders; i++) {
				shaders[i - 1] = shaders[i];
			}
			shader_count--;
		}
		shaders[shader_count++] = shader;
	}
};

// (A GPU that stops answering is noticed seconds later, and the GPU thread goes on submitting
// until it has to wait for it: more than a frame's submissions, a hundred on Saros.)
inline constexpr size_t Kept = 256;

inline std::mutex             g_mutex;
inline std::array<Note, Kept> g_notes {};
inline uint64_t               g_count = 0;

// GPU thread, when a command buffer ends.
inline void Add(const Note& note) {
	std::lock_guard lock(g_mutex);
	g_notes[g_count++ % Kept] = note;
}

// Any thread: the newest `count` notes, oldest first.
template <typename Func>
void ForEachNewest(size_t count, Func&& func) {
	std::lock_guard lock(g_mutex);
	const auto      kept  = static_cast<size_t>(g_count < Kept ? g_count : Kept);
	const auto      shown = count < kept ? count : kept;
	for (uint64_t i = g_count - shown; i < g_count; i++) {
		func(g_notes[i % Kept]);
	}
}

} // namespace Libs::Graphics::SubmissionNotes

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_SUBMISSIONNOTES_H_ */
