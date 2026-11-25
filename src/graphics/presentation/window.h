#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_

#include "common/abi.h"
#include "common/common.h"

#include <filesystem>

namespace Libs::Graphics {

class Presenter;

[[nodiscard]] Presenter& WindowInit(uint32_t width, uint32_t height);
void                     WindowRun();
// --precompile-only: builds the recorded pipelines into the driver cache and saves it. Returns
// the process exit code: 0 done, 1 the driver cache was not written, 2 the window was closed.
[[nodiscard]] int WindowPrecompilePipelines();
// --export-pipeline-list: whether the list was written.
[[nodiscard]] bool       WindowExportPipelineList(const std::filesystem::path& path);
void                     WindowShutdown();
// Asks the window loop to close, as the user closing the window does (any thread).
void                     WindowRequestClose();
// One gamepad event of a session's log ("Controller axis: ..." / "Controller button: ...", with
// SDL's axis and button numbers and values) on the host-input pad: the translation the live
// event took, whatever host_gamepads says (a test run ignores the PC's pads and still takes a
// replay of a player's session). Any thread.
void WindowReplayPadAxis(int sdl_axis, int sdl_value);
void WindowReplayPadButton(int sdl_button, bool down);
// The window's event clock now, in seconds: the time base of the log's input lines ("time = ").
// Zero before the window loop runs. Any thread.
[[nodiscard]] double WindowInputSeconds();
// The renderer's background builds that have not finished (PipelineCache::
// PendingBackgroundBuilds); zero without a renderer. Any thread, while the window lives.
[[nodiscard]] uint64_t   WindowPendingBackgroundBuilds();
// The game's picture at the next present (the main bus's frame as it goes to the window, before
// scaling and overlays), written to `path` as a 32-bit BMP: the live channel's `shot`. Any thread.
void WindowRequestScreenshot(const char* path);

// Nits of the display showing the game window; the defaults apply when the host reports none.
struct DisplayLuminance {
	float max_full_frame = 80.0f;
	float max            = 1000.0f;
	float min            = 0.0f;
};

[[nodiscard]] DisplayLuminance WindowGetDisplayLuminance();

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_ */
