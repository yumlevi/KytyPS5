#include "common/archive.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/file.h"
#include "common/hostException.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "emulator.h"
#include "graphics/host_gpu/renderer/upscaler/frameGeneration.h"
#include "kytyGitVersion.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <string_view>
#include <vector>
#include <magic_enum.hpp>

#if defined(KYTY_MIMALLOC)
#include <atomic>
#include <cstdlib>
#include <malloc.h>
#include <mimalloc.h>
#include <new>
#include <windows.h> // IWYU pragma: keep

// The executable's operator new/delete (this translation unit only; after the Demon's Souls
// fork's d36975759): mimalloc's, or with KYTY_MIMALLOC=0 the CRT heap's, chosen at the first
// allocation from the environment and kept for the process (a block is always freed by the heap
// that made it). The allocations of the drivers and of the other modules stay on the process
// heap.
//
// The shader recompiler allocates a node for every value of a program and frees them all again:
// a third of a translating thread's time was the CRT heap, and with many threads at it the heap
// was what limited them (12,400 programs in 22 s on 28 threads, where eight make them in 59).
// The thread that feeds the GPU spends 2 to 3% of its time in the same heap.
//
// mimalloc keeps what it frees for longer than the CRT heap does (the Demon's Souls fork
// measured 2 to 3.6 GB more private memory over a long session, 637343ec6), so without the
// variable it is taken only on a PC with MimallocMinRamGiB of physical memory or more: a PC with
// 32 GB (Windows reports about 31.9 GiB) has it, one with 16 or 24 GB keeps the CRT heap.
// Nothing on this path may allocate: it runs inside the first operator new. The variable is
// read before main, so it has to be set by whatever starts the emulator.
constexpr unsigned long long MimallocMinRamGiB = 28;
// -1: not chosen yet; 0: the CRT heap, asked for; 1: the CRT heap, for this PC's memory;
// 2: mimalloc, asked for; 3: mimalloc.
static std::atomic<int> g_kyty_heap {-1};
static int              KytyHeapChoice() noexcept {
    if (const int chosen = g_kyty_heap.load(std::memory_order_acquire); chosen >= 0) {
        return chosen;
    }
    int         choice = 3;
    const char* value  = std::getenv("KYTY_MIMALLOC");
    if (value != nullptr && value[0] != '\0') {
        choice = value[0] != '0' ? 2 : 0;
    } else {
        MEMORYSTATUSEX memory {};
        memory.dwLength = sizeof(memory);
        if (GlobalMemoryStatusEx(&memory) != 0 && memory.ullTotalPhys < (MimallocMinRamGiB << 30u)) {
            choice = 1;
        }
    }
    // (Two threads at their first allocation: the first choice stands for both.)
    int expected = -1;
    if (!g_kyty_heap.compare_exchange_strong(expected, choice, std::memory_order_acq_rel)) {
        return expected;
    }
    return choice;
}
static bool KytyMimalloc() noexcept {
	return KytyHeapChoice() >= 2;
}
static void* KytyAlloc(std::size_t size) noexcept {
	return KytyMimalloc() ? mi_malloc(size) : std::malloc(size == 0 ? 1 : size);
}
static void* KytyAllocAligned(std::size_t size, std::align_val_t align) noexcept {
	return KytyMimalloc() ? mi_malloc_aligned(size, static_cast<std::size_t>(align))
	                      : _aligned_malloc(size == 0 ? 1 : size, static_cast<std::size_t>(align));
}
static void KytyFree(void* p) noexcept {
	if (KytyMimalloc()) {
		mi_free(p);
	} else {
		std::free(p);
	}
}
static void KytyFreeAligned(void* p) noexcept {
	if (KytyMimalloc()) {
		mi_free(p);
	} else {
		_aligned_free(p);
	}
}
// (An allocation that fails ends the process, as an uncaught std::bad_alloc would.)
static void* KytyNew(std::size_t size) {
	if (void* p = KytyAlloc(size)) {
		return p;
	}
	std::abort();
}
static void* KytyNewAligned(std::size_t size, std::align_val_t align) {
	if (void* p = KytyAllocAligned(size, align)) {
		return p;
	}
	std::abort();
}

void* operator new(std::size_t size) { return KytyNew(size); }
void* operator new[](std::size_t size) { return KytyNew(size); }
void* operator new(std::size_t size, const std::nothrow_t&) noexcept { return KytyAlloc(size); }
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept { return KytyAlloc(size); }
void* operator new(std::size_t size, std::align_val_t align) { return KytyNewAligned(size, align); }
void* operator new[](std::size_t size, std::align_val_t align) { return KytyNewAligned(size, align); }
void* operator new(std::size_t size, std::align_val_t align, const std::nothrow_t&) noexcept {
	return KytyAllocAligned(size, align);
}
void* operator new[](std::size_t size, std::align_val_t align, const std::nothrow_t&) noexcept {
	return KytyAllocAligned(size, align);
}
void operator delete(void* p) noexcept { KytyFree(p); }
void operator delete[](void* p) noexcept { KytyFree(p); }
void operator delete(void* p, std::size_t) noexcept { KytyFree(p); }
void operator delete[](void* p, std::size_t) noexcept { KytyFree(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { KytyFree(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { KytyFree(p); }
void operator delete(void* p, std::align_val_t) noexcept { KytyFreeAligned(p); }
void operator delete[](void* p, std::align_val_t) noexcept { KytyFreeAligned(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { KytyFreeAligned(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { KytyFreeAligned(p); }
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept {
	KytyFreeAligned(p);
}
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept {
	KytyFreeAligned(p);
}
#endif

#if defined(KYTY_PGO_GENERATE)
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>

// compiler-rt profile runtime, linked into instrumented builds (CMake KYTY_PGO_GENERATE).
extern "C" void __llvm_profile_set_filename(const char*);
extern "C" int  __llvm_profile_write_file(void);
#endif

using namespace Common;
using namespace Emulator;

#if defined(KYTY_PGO_GENERATE)
// The profile runtime writes its counters from an atexit handler, which the emulator skips (it
// leaves with quick_exit, a fatal error, or a killed process). Instead a thread writes them every
// KYTY_PGO_DUMP_SECONDS (default 60) to KYTY_PGO_FILE (default kyty_emulator.profraw in the
// working directory), and quick_exit writes them once more. Each dump goes to <file>.tmp first and
// is renamed over <file>, so a stopped training run leaves its last complete profile behind.
static std::mutex  g_pgo_mutex;
static std::string g_pgo_file;

static void WritePgoProfile() {
	std::scoped_lock lock {g_pgo_mutex};
	const auto       temp = g_pgo_file + ".tmp";
	__llvm_profile_set_filename(temp.c_str());
	if (__llvm_profile_write_file() != 0) {
		::fprintf(stderr, "PGO: failed to write %s\n", temp.c_str());
		return;
	}
	std::error_code error;
	std::filesystem::rename(temp, g_pgo_file, error);
	if (error) {
		::fprintf(stderr, "PGO: failed to rename %s: %s\n", temp.c_str(), error.message().c_str());
	}
}

static void StartPgoDumps() {
	const char* file    = std::getenv("KYTY_PGO_FILE");
	g_pgo_file          = (file != nullptr && *file != '\0') ? file : "kyty_emulator.profraw";
	const char* text    = std::getenv("KYTY_PGO_DUMP_SECONDS");
	const int   seconds = (text != nullptr && std::atoi(text) > 0) ? std::atoi(text) : 60;
	::fprintf(stderr, "PGO: writing the profile to %s every %d s and at exit\n", g_pgo_file.c_str(),
	          seconds);
	at_quick_exit(WritePgoProfile);
	std::thread([seconds] {
		for (;;) {
			std::this_thread::sleep_for(std::chrono::seconds(seconds));
			WritePgoProfile();
		}
	}).detach();
}
#endif

static void PrintUsage() {
	::printf("%s, %s, ver = %s, git = %s, date = %s\n",
	         KYTY_BUILD == KYTY_BUILD_DEBUG ? "Debug" : "Release", KYTY_TOOLCHAIN, KYTY_VERSION,
	         KYTY_GIT_VERSION, __DATE__);
	::printf("kyty_emulator --game <dir|elf|zar> [options]\n\n");
	::printf("Options:\n");
	::printf("  --game <dir|elf|zar>                 Game directory, ELF, or ZArchive to load.\n");
	::printf("  --game-patch <json>                  ETAHen cheat file.\n");
	::printf("  --addcont-dir <dir>                  Add-ons (DLC) folder (default: none).\n");
	::printf(
	    "  --addcont <label>[=<dir>]            Add-on (DLC) by entitlement label with its data in\n"
	    "                                       <dir>, or license only without <dir>. Repeatable.\n");
	::printf("  --screen-width <num>                 Window width. Default: 1280.\n");
	::printf("  --screen-height <num>                Window height. Default: 720.\n");
	::printf(
	    "  --user-name <name>                   Local user name (1-16 bytes). Default: Kyty.\n");
	::printf("  --user-id <num>                      Local user ID. Default: %d.\n",
	         Config::DEFAULT_USER_ID);
	::printf("  --mic <name>                        Capture from this microphone; omit for silence.\n");
	::printf("  --controller-color <#RRGGBB>        Override the controller lightbar color.\n");
	::printf("  --controller-volume <0-100>         DualSense speaker volume. Default: 50.\n");
	::printf("  --controller-vibration <0-100>      DualSense vibration intensity. Default: 100.\n");
	::printf("  --keyboard-player <0-4>             Keyboard player; 0 shares player 1 with the gamepad. Default: 0.\n");
	::printf(
	    "  --present-mode <value>               Fifo, Mailbox, or Immediate. Default: Mailbox.\n");
	::printf(
	    "  --gpu <index>                        Vulkan physical device index. Default: auto.\n");
	::printf("  --fullscreen                         Run in borderless desktop fullscreen.\n");
	::printf("  --hdr                                Enable HDR10 output when supported.\n");
	::printf(
	    "  --hide-cursor                        Hide the cursor after 2 s idle. Default: off.\n");
	::printf(
	    "  --stretch                            Stretch the image to the window. Default: off.\n");
	::printf("  --no-splash                          Disable the startup splash screen.\n");
	::printf("  --vr                                 Enable the virtual VR headset.\n");
	::printf("  --trinity                            Enable experimental PS5 Pro mode for\n"
	         "                                       supported titles. Default: off.\n");
	::printf("  --pssr-upscaler <value>              With --trinity, upscale PSSR with Linear,\n"
	         "                                       DLSS, or FSR. Default: Linear.\n");
	::printf("  --upscaler <value>                   The same; without --trinity DLSS makes the\n"
	         "                                       picture in place of the upscaler of an Unreal\n"
	         "                                       Engine 5 title (Saros).\n");
	::printf("  --dlss-folder <folder>               DLSS model folder (nvngx_dlss.dll). Default:\n"
	         "                                       NVIDIA's Streamline runtime, where it is found\n"
	         "                                       for frame generation.\n");
	::printf("  --dlss-preset <A-O>                  DLSS preset. Default: the model's own.\n");
	::printf("  --fsr-folder <folder>                FidelityFX DLL folder (Windows).\n");
	::printf("  --amd-cpu                            Apply AMD CPU instruction patches.\n");
	::printf("  --guest-tsc                          Run the guest TSC at the PS5's 1.6 GHz.\n");
	::printf("  --vblank-frequency <num>             Virtual vblank frequency. Default: 60.\n");
	::printf("  --console-language <0-29>            Console language. Default: 1 (English US).\n");
	::printf("  --vulkan-validation                  Enable Vulkan validation.\n");
	::printf("  --network-connected                  Use the host local network.\n");
	::printf("  --emulate-psn                        Simulate PSN sign-in for compatibility.\n");
	::printf("  --gpu-assisted-validation            Bounds-check shader accesses on the GPU.\n"
	         "                                       Implies --vulkan-validation; very slow.\n");
	::printf("  --shader-validation                  Enable shader validation.\n");
	::printf("  --tessellation                      Draw tessellation patches; skipped by default.\n");
	::printf("  --occlusion-queries                  Count occlusion query samples on the GPU.\n");
	::printf("  --relaxed-shader-float               Native GPU float ops in shaders; faster, can be inaccurate.\n");
	::printf(
	    "  --emulate-wave64                     Run wave64 compute shaders 32 lanes wide, two\n"
	    "                                       guest lanes per invocation, also on GPUs with\n"
	    "                                       64-lane subgroups. Default: off.\n");
	::printf(
	    "  --maximal-reconvergence <value>      Auto, On, or Off: request maximal reconvergence\n"
	    "                                       in shaders where the GPU supports it. Auto asks\n"
	    "                                       for it except on AMD's LLPC drivers. Default: "
	    "Auto.\n"
	    "  --no-maximal-reconvergence           Same as --maximal-reconvergence Off.\n");
	::printf("  --no-shader-simplification           Give the GPU driver shaders without the\n"
	         "                                       driver simplification pass. Default: off.\n");
	::printf("  --native-rt                         Use native RT on the GPU.\n");
	::printf("  --lod-stats                          Report bound textures in mip statistics.\n");
	::printf("  --shader-log-direction <value>       Silent, Console, or File.\n");
	::printf("  --shader-log-folder <path>           Shader log output folder.\n");
	::printf("  --command-buffer-dump                Enable command buffer dumps.\n");
	::printf("  --command-buffer-dump-folder <path>  Command buffer dump folder.\n");
	::printf("  --graphics-debug-dump                Enable graphics debug dumps.\n");
	::printf("  --printf-direction <value>           Silent, Console, or File.\n");
	::printf("  --printf-output-file <path>          Guest printf output file.\n");
	::printf("  --profile                            Enable the Tracy profiler.\n");
	::printf("  --spirv-debug-printf                 Enable SPIR-V debug printf.\n");
	::printf("  --readback-linear-images             Read back writable linear images on submit.\n");
	::printf("  --readback-tiled-images              Read back small writable tiled images on submit.\n");
	::printf("  --readback-storage-images            Read back storage images on GL2 writeback.\n");
	::printf("  --sync-raw-image-buffers             Synchronize raw reads of GPU images.\n");
	::printf(
	    "  --flush-draws <num>                  Submit GPU work every <num> draws; 0 is off.\n"
	    "  --no-async-pipelines                 Build missing pipelines optimized at once,\n"
	    "                                       not unoptimized first with an optimized build\n"
	    "                                       in the background.\n");
	::printf(
	    "  --skip-pending-draws                 Skip draws that only change the picture while\n"
	    "                                       their pipelines build in the background (at most\n"
	    "                                       2 s per pipeline).\n");
	::printf(
	    "  --no-precompile-pipelines            Do not build the compute and graphics pipelines\n"
	    "                                       recorded in earlier sessions, and those of compute\n"
	    "                                       shaders the game created but never dispatched, in\n"
	    "                                       the background while it runs. (The builds go into\n"
	    "                                       the driver cache, and the recorded ones are kept\n"
	    "                                       ready for their first use.)\n");
	::printf(
	    "  --lookahead-threads <n>              Threads that evaluate the resources of upcoming\n"
	    "                                       draws and dispatches ahead of the GPU thread\n"
	    "                                       (0 = off, at most 8). Default: 2.\n");
	::printf(
	    "  --prepare-threads <n>                Threads that translate the shaders that earlier\n"
	    "                                       sessions recorded, and then the compute shaders\n"
	    "                                       the precompile built from their code alone (those\n"
	    "                                       of the game's files, with --import-shaders), at\n"
	    "                                       low priority from the boot, so that their first\n"
	    "                                       use needs no translation (0 = off, at most 8).\n"
	    "                                       Default: a quarter of the host's threads, 2 to 8.\n");
	::printf(
	    "  --precompile-only                    Build the recorded compute and graphics pipelines\n"
	    "                                       and the created compute shaders into the driver\n"
	    "                                       cache, save it and exit (code 0 when done).\n");
	::printf(
	    "  --rebuild-pipeline-cache             With --precompile-only: replace the driver cache\n"
	    "                                       with only the pipelines this run builds (also\n"
	    "                                       done for a cache file larger than 3 GiB).\n");
	::printf(
	    "  --export-pipeline-list <file>        Write the pipelines recorded for the game as a\n"
	    "                                       shareable list without shader code, and exit.\n");
	::printf(
	    "  --import-pipeline-list <file>        Precompile the pipelines of a shared list whose\n"
	    "                                       shaders this game has (with --precompile-only, or\n"
	    "                                       beside the game unless --no-precompile-pipelines).\n"
	    "                                       A folder: its <title id>.pipelinelist, if any.\n");
	::printf(
	    "  --import-shaders <file>              With --precompile-only: also build the compute\n"
	    "                                       shaders of this file, which holds shader binaries\n"
	    "                                       taken from the game's files (so before the game\n"
	    "                                       has created them). A folder: its <title id>.seeds,\n"
	    "                                       if any.\n");
	::printf(
	    "  --precompile-threads <n>             With --precompile-only: how many pipelines are\n"
	    "                                       built at a time (1 to 64). Default: one for each\n"
	    "                                       logical CPU the process may use but two, fewer on\n"
	    "                                       a PC with little memory.\n");
	::printf(
	    "  --shader-pairs <file>                A list of the vertex and pixel shaders of the\n"
	    "                                       game's files that are drawn together, each with a\n"
	    "                                       recorded pipeline of the same kind. With\n"
	    "                                       --precompile-only and --import-shaders their\n"
	    "                                       pipelines are built before any session has drawn\n"
	    "                                       with them; a session takes those. A folder: its\n"
	    "                                       <title id>.pairs, if any.\n");
	::printf(
	    "  --unreal-cvar <name>=<number>        Unreal Engine titles: set a console variable of\n"
	    "                                       the running game (float and int variables that\n"
	    "                                       the engine reads as it runs), and keep it set.\n"
	    "                                       Repeatable. r.SecondaryScreenPercentage.GameViewport\n"
	    "                                       =50 makes a title's picture at half its output\n"
	    "                                       size, stretched to the output. <name>=x<factor>:\n"
	    "                                       the title's own value times the factor.\n"
	    "  --unreal-render-scale <percent>      Unreal Engine titles: the size the 3D is rendered\n"
	    "                                       at, in percent of the title's own (its screen\n"
	    "                                       percentage and dynamic resolution bounds); the\n"
	    "                                       title's upscaler still makes the output's picture.\n"
	    "                                       Less GPU time for a softer picture.\n"
	    "  --upscaler-render-percent <number>   With --upscaler DLSS in place of an Unreal Engine\n"
	    "                                       title's upscaler: the 3D at this one size, in\n"
	    "                                       percent of the output's (no dynamic resolution).\n"
	    "                                       DLSS's modes: 33.3333 (ultra performance), 50, 58,\n"
	    "                                       66.6667 (quality). Without the option the title's\n"
	    "                                       own sizes are upscaled as they come. Not applied\n"
	    "                                       when DLSS could not be loaded.\n");
	::printf("  --serialize-dispatches               Order every guest compute dispatch after the\n"
	         "                                       previous one, also where the guest sends no\n"
	         "                                       flush between them.\n");
	::printf("  --no-trophy-notifications            Disable trophy unlock toasts and their sound.\n");
	::printf(
	    "  --trophy-duration <seconds>          Trophy toast duration, 5-10. Default: 6.\n");
	::printf("  --perf-overlay                       Show the performance panel (F2 or Cmd+P).\n");
	::printf("  --playgo-hack                       Use the supplied PlayGo stub fallback.\n");
	::printf("  --skip-notice-screen                 Skip startup logos and notices in supported games.\n");
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	::printf("  --redzone                            Protect the guest SysV red zone.\n");
#endif
	::printf("  --keymap <Control=Input>             DualSense mapping; may be repeated.\n");
	::printf("  --rd                                 Enable RenderDoc capture.\n");
}

static bool NextArg(int argc, char* argv[], int& index, std::string& out) {
	if (index + 1 >= argc) {
		return false;
	}

	index++;
	out = argv[index];
	return true;
}

template <typename E>
static bool ParseEnum(const std::string& value, E& out) {
	auto enum_value = magic_enum::enum_cast<E>(value, magic_enum::case_insensitive);
	if (!enum_value.has_value()) {
		return false;
	}

	out = enum_value.value();
	return true;
}

static bool ParseConsoleLanguage(const std::string& value, uint32_t& out) {
	uint32_t language = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), language);
	if (error != std::errc {} || end != value.data() + value.size() ||
	    language > Config::MAX_CONSOLE_LANGUAGE) {
		return false;
	}
	out = language;
	return true;
}

static bool ParseUint32(const std::string& value, uint32_t& out) {
	uint32_t number   = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
	if (error != std::errc {} || end != value.data() + value.size()) {
		return false;
	}
	out = number;
	return true;
}

static bool ParseControllerColor(const std::string& value, Config::ControllerColor& out) {
	if (value.size() != 7 || value[0] != '#') {
		return false;
	}
	uint32_t rgb = 0;
	auto [end, error] = std::from_chars(value.data() + 1, value.data() + value.size(), rgb, 16);
	if (error != std::errc {} || end != value.data() + value.size()) {
		return false;
	}
	out = {static_cast<uint8_t>(rgb >> 16), static_cast<uint8_t>(rgb >> 8),
	       static_cast<uint8_t>(rgb)};
	return true;
}

static bool ParseInt32(const std::string& value, int32_t& out) {
	int32_t number    = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
	if (error != std::errc {} || end != value.data() + value.size()) {
		return false;
	}
	out = number;
	return true;
}

static bool ParseUserId(const std::string& value, int32_t& out) {
	int32_t user_id   = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), user_id);
	if (error != std::errc {} || end != value.data() + value.size() ||
	    !Config::IsConfiguredUserIdValid(user_id)) {
		return false;
	}
	out = user_id;
	return true;
}

using Config::ConfigOptions;

struct Flag {
	std::string_view name;
	bool ConfigOptions::* field;
	bool                  value = true;
};

static constexpr Flag FLAGS[] = {
    {"--rd", &ConfigOptions::renderdoc_enabled},
    {"--fullscreen", &ConfigOptions::fullscreen_enabled},
    {"--hdr", &ConfigOptions::hdr_enabled},
    {"--hide-cursor", &ConfigOptions::hide_cursor_enabled},
    {"--stretch", &ConfigOptions::stretch_enabled},
    {"--no-splash", &ConfigOptions::splash_enabled, false},
    {"--vr", &ConfigOptions::vr_enabled},
    {"--trinity", &ConfigOptions::pro_enabled},
    {"--amd-cpu", &ConfigOptions::amd_cpu_enabled},
    {"--guest-tsc", &ConfigOptions::guest_tsc_enabled},
    {"--network-connected", &ConfigOptions::network_connected},
    {"--emulate-psn", &ConfigOptions::psn_emulation_enabled},
    {"--vulkan-validation", &ConfigOptions::vulkan_validation_enabled},
    {"--gpu-assisted-validation", &ConfigOptions::gpu_assisted_validation_enabled},
    {"--shader-validation", &ConfigOptions::shader_validation_enabled},
    {"--tessellation", &ConfigOptions::tessellation_enabled},
    {"--occlusion-queries", &ConfigOptions::occlusion_queries_enabled},
    {"--relaxed-shader-float", &ConfigOptions::relaxed_shader_float},
    {"--emulate-wave64", &ConfigOptions::emulate_wave64},
    {"--no-shader-simplification", &ConfigOptions::no_shader_simplification},
    {"--native-rt", &ConfigOptions::native_rt},
    {"--lod-stats", &ConfigOptions::lod_stats_enabled},
    {"--command-buffer-dump", &ConfigOptions::command_buffer_dump_enabled},
    {"--graphics-debug-dump", &ConfigOptions::graphics_debug_dump_enabled},
    {"--profile", &ConfigOptions::profiler_enabled},
    {"--spirv-debug-printf", &ConfigOptions::spirv_debug_printf_enabled},
    {"--readback-linear-images", &ConfigOptions::readback_linear_images},
    {"--readback-tiled-images", &ConfigOptions::readback_tiled_images},
    {"--readback-storage-images", &ConfigOptions::readback_storage_images},
    {"--sync-raw-image-buffers", &ConfigOptions::sync_raw_image_buffers},
    {"--no-async-pipelines", &ConfigOptions::async_pipelines_enabled, false},
    {"--skip-pending-draws", &ConfigOptions::skip_pending_draws},
    {"--precompile-pipelines", &ConfigOptions::precompile_pipelines},
    {"--no-precompile-pipelines", &ConfigOptions::precompile_pipelines, false},
    {"--precompile-only", &ConfigOptions::precompile_only},
    {"--rebuild-pipeline-cache", &ConfigOptions::rebuild_pipeline_cache},
    {"--serialize-dispatches", &ConfigOptions::serialize_dispatches},
    {"--no-trophy-notifications", &ConfigOptions::trophy_enabled, false},
    {"--perf-overlay", &ConfigOptions::perf_overlay_enabled},
    {"--playgo-hack", &ConfigOptions::playgo_hack_enabled},
    {"--skip-notice-screen", &ConfigOptions::skip_notice_screen},
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
    {"--redzone", &ConfigOptions::red_zone_protection_enabled},
#endif
};

static bool ParseArgs(int argc, char* argv[], RunOptions& options, bool& show_help) {
	show_help = false;

	for (int i = 1; i < argc; i++) {
		std::string arg = std::string(argv[i]);
		std::string value;

		if (arg == "--help" || arg == "-h") {
			show_help = true;
			continue;
		}

		if (const auto* flag = std::ranges::find(FLAGS, arg, &Flag::name); flag != std::end(FLAGS)) {
			options.config.*flag->field = flag->value;
			continue;
		}

		if (arg == "--no-maximal-reconvergence") {
			options.config.maximal_reconvergence = Config::MaximalReconvergence::Off;
			continue;
		}

		if (!arg.starts_with("--")) {
			::printf("game input must be provided with --game\n");
			return false;
		}

		if (!NextArg(argc, argv, i, value)) {
			::printf("missing value for %s\n", arg.c_str());
			return false;
		}

		if (arg == "--game") {
			if (!options.app0_dir.empty()) {
				::printf("--game can only be specified once\n");
				return false;
			}

			value           = Common::FixFilenameSlash(value);
			const auto path = Common::PathFromUtf8(value);

			if (Common::File::IsDirectoryExisting(path)) {
				options.app0_dir = path;
				options.elf      = "/app0/eboot.bin";
			} else if (Common::IsSupportedArchive(path) && Common::File::IsFileExisting(path)) {
				const auto root = Common::MakeArchivePath(path);
				if (!Common::File::IsFileExisting(root / "eboot.bin")) {
					::printf("Archive does not contain eboot.bin: %s\n", value.c_str());
					return false;
				}
				options.app0_dir = root;
				options.elf      = "/app0/eboot.bin";
			} else if (Common::File::IsFileExisting(path)) {
				options.app0_dir = path.parent_path();

				if (options.app0_dir.empty()) {
					options.app0_dir = ".";
				}

				options.elf = std::filesystem::path("/app0") / path.filename();
			} else {
				::printf("--game must point to an existing directory, ELF, or archive: %s\n",
				         value.c_str());
				return false;
			}
		} else if (arg == "--guest-exec") {
			options.elf = Common::PathFromUtf8(Common::FixFilenameSlash(value));
			options.config.guest_arguments.assign(argv + i + 1, argv + argc);
			break;
		} else if (arg == "--game-patch") {
			if (!options.game_patch.empty()) {
				::printf("--game-patch can only be specified once\n");
				return false;
			}
			value = Common::FixFilenameSlash(value);
			const auto path = Common::PathFromUtf8(value);

			if (!Common::File::IsFileExisting(path)) {
				::printf("--game-patch must point to an existing file: %s\n", value.c_str());
				return false;
			}
			options.game_patch = path;
		} else if (arg == "--addcont-dir") {
			if (!options.addcont_dir.empty()) {
				::printf("--addcont-dir can only be specified once\n");
				return false;
			}
			options.addcont_dir = Common::PathFromUtf8(Common::FixFilenameSlash(value));
		} else if (arg == "--addcont") {
			auto entry = Common::ParseAddContent(value);
			if (!entry) {
				::printf("--addcont needs <16-character A-Z/0-9 label>[=<existing folder>]: %s\n",
				         value.c_str());
				return false;
			}
			options.addcont.push_back(std::move(*entry));
		} else if (arg == "--screen-width") {
			if (!ParseUint32(value, options.config.screen_width) ||
			    options.config.screen_width == 0) {
				::printf("invalid screen width: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--screen-height") {
			if (!ParseUint32(value, options.config.screen_height) ||
			    options.config.screen_height == 0) {
				::printf("invalid screen height: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--user-name") {
			if (value.empty() || value.size() > Config::MAX_USER_NAME_LENGTH) {
				::printf("invalid user name: must contain 1-%zu bytes\n",
				         Config::MAX_USER_NAME_LENGTH);
				return false;
			}
			options.config.user_name = value;
		} else if (arg == "--user-id") {
			if (!ParseUserId(value, options.config.user_id)) {
				::printf("invalid user ID: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--mic") {
			options.config.audio_input_device = value;
		} else if (arg == "--controller-color") {
			Config::ControllerColor color {};
			if (!ParseControllerColor(value, color)) {
				::printf("invalid controller color (expected #RRGGBB): %s\n", value.c_str());
				return false;
			}
			options.config.controller_color = color;
		} else if (arg == "--controller-volume") {
			if (!ParseUint32(value, options.config.controller_speaker_volume) ||
			    options.config.controller_speaker_volume > 100) {
				::printf("invalid controller volume: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--controller-vibration") {
			if (!ParseUint32(value, options.config.controller_vibration_intensity) ||
			    options.config.controller_vibration_intensity > 100) {
				::printf("invalid controller vibration intensity: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--keyboard-player") {
			if (!ParseUint32(value, options.config.keyboard_player) ||
			    options.config.keyboard_player > static_cast<uint32_t>(Config::MAX_PLAYERS)) {
				::printf("invalid keyboard player: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--present-mode") {
			if (!ParseEnum(value, options.config.present_mode)) {
				::printf("invalid present mode: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--maximal-reconvergence") {
			if (!ParseEnum(value, options.config.maximal_reconvergence)) {
				::printf("invalid maximal reconvergence: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--pssr-upscaler" || arg == "--upscaler") {
			if (!ParseEnum(value, options.config.pssr_upscaler)) {
				::printf("invalid upscaler: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--dlss-folder") {
			options.config.dlss_folder = Common::PathFromUtf8(value);
		} else if (arg == "--dlss-preset") {
			if (value.size() != 1 || value[0] < 'A' || value[0] > 'O') {
				::printf("invalid DLSS preset: %s\n", value.c_str());
				return false;
			}
			options.config.dlss_preset = static_cast<uint32_t>(value[0] - 'A' + 1);
		} else if (arg == "--fsr-folder") {
			options.config.fsr_folder = Common::PathFromUtf8(value);
		} else if (arg == "--gpu") {
			if (!ParseInt32(value, options.config.gpu_index)) {
				::printf("invalid gpu index: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--vblank-frequency") {
			if (!ParseUint32(value, options.config.vblank_frequency)) {
				::printf("invalid vblank frequency: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--flush-draws") {
			if (!ParseUint32(value, options.config.flush_draws)) {
				::printf("invalid draw count: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--lookahead-threads") {
			if (!ParseUint32(value, options.config.lookahead_threads) ||
			    options.config.lookahead_threads > 8u) {
				::printf("invalid lookahead thread count: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--prepare-threads") {
			if (!ParseUint32(value, options.config.prepare_threads) ||
			    options.config.prepare_threads > 8u) {
				::printf("invalid preparation thread count: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--precompile-threads") {
			if (!ParseUint32(value, options.config.precompile_threads) ||
			    options.config.precompile_threads == 0u || options.config.precompile_threads > 64u) {
				::printf("invalid precompile thread count: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--trophy-duration") {
			if (!ParseUint32(value, options.config.trophy_duration_seconds)) {
				::printf("invalid trophy duration: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--console-language") {
			if (!ParseConsoleLanguage(value, options.config.console_language)) {
				::printf("invalid console language: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--shader-log-direction") {
			if (!ParseEnum(value, options.config.shader_log_direction)) {
				::printf("invalid shader log direction: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--shader-log-folder") {
			options.config.shader_log_folder = Common::PathFromUtf8(value);
		} else if (arg == "--export-pipeline-list") {
			options.config.export_pipeline_list = Common::PathFromUtf8(value);
		} else if (arg == "--import-pipeline-list") {
			options.config.import_pipeline_list = Common::PathFromUtf8(value);
		} else if (arg == "--import-shaders") {
			options.config.import_shaders = Common::PathFromUtf8(value);
		} else if (arg == "--shader-pairs") {
			options.config.shader_pairs = Common::PathFromUtf8(value);
		} else if (arg == "--unreal-cvar") {
			options.config.unreal_cvars.push_back(value);
		} else if (arg == "--unreal-render-scale") {
			uint32_t percent = 0;
			if (!ParseUint32(value, percent) || percent < 10u || percent > 200u) {
				::printf("invalid render scale (10-200 percent): %s\n", value.c_str());
				return false;
			}
			if (percent != 100u) {
				for (const char* name: {"r.DynamicRes.MinScreenPercentage", "r.DynamicRes.MaxScreenPercentage",
				                        "r.ScreenPercentage"}) {
					options.config.unreal_cvars.push_back(std::string(name) + "=x" +
					                                      std::to_string(static_cast<double>(percent) / 100.0));
				}
			}
		} else if (arg == "--upscaler-render-percent") {
			char*        end     = nullptr;
			const double percent = std::strtod(value.c_str(), &end);
			if (end == value.c_str() || *end != '\0' || !(percent >= 10.0 && percent <= 100.0)) {
				::printf("invalid render percent (10-100): %s\n", value.c_str());
				return false;
			}
			// (Set only while the upscaler makes the picture: kernel/unrealCvars.h.)
			for (const char* name: {"r.DynamicRes.MinScreenPercentage", "r.DynamicRes.MaxScreenPercentage",
			                        "r.ScreenPercentage"}) {
				options.config.unreal_cvars.push_back(std::string("upscaler?") + name + "=" + value);
			}
		} else if (arg == "--command-buffer-dump-folder") {
			options.config.command_buffer_dump_folder = Common::PathFromUtf8(value);
		} else if (arg == "--printf-direction") {
			if (!ParseEnum(value, options.config.printf_direction)) {
				::printf("invalid printf direction: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--printf-output-file") {
			options.config.printf_output_file = Common::PathFromUtf8(value);
		} else if (arg == "--keymap") {
			const auto split = value.find('=');
			if (split == std::string::npos || split == 0 || split + 1 == value.size()) {
				::printf("invalid keymap: %s\n", value.c_str());
				return false;
			}
			options.config.keymap.push_back(value);
		} else {
			::printf("unknown option: %s\n", arg.c_str());
			return false;
		}
	}

	if (options.config.gpu_assisted_validation_enabled) {
		options.config.vulkan_validation_enabled = true;
	}

	return show_help || (!options.app0_dir.empty() && !options.elf.empty());
}

static int Main(int argc, char* argv[]) {
	InitializeThreads();
#if defined(KYTY_PGO_GENERATE)
	// After the address-space reservations: its stack must not land in the guest range.
	StartPgoDumps();
#endif

	RunOptions options;
	bool       show_help = false;

	if (argc < 2) {
		PrintUsage();
		return 0;
	}

	if (!ParseArgs(argc, argv, options, show_help)) {
		PrintUsage();
		return 1;
	}

	if (show_help) {
		PrintUsage();
		return 0;
	}

#if defined(KYTY_MIMALLOC)
	// The heap this process's operator new/delete took (fixed at the first allocation), and why.
	switch (KytyHeapChoice()) {
		case 0: std::printf("Heap: the CRT heap (KYTY_MIMALLOC=0)\n"); break;
		case 1:
			std::printf("Heap: the CRT heap (less than %llu GiB of physical memory; KYTY_MIMALLOC=1 "
			            "takes mimalloc all the same)\n",
			            MimallocMinRamGiB);
			break;
		default:
			std::printf("Heap: mimalloc %d.%d.%d (vendored)%s\n", MI_MALLOC_VERSION / 10000,
			            MI_MALLOC_VERSION / 100 % 100, MI_MALLOC_VERSION % 100,
			            KytyHeapChoice() == 2 ? " (KYTY_MIMALLOC=1)" : "");
			break;
	}
#endif

	Run(options);

	return 0;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

// NVIDIA's driver compresses the pipelines it compiles with a dictionary that it trains once a
// process has made a few hundred of them and keeps in its shader disk cache; it reads the entries
// of a pipeline cache file only with the dictionary its disk cache holds at that time. In the
// driver's default disk cache, which every program shares, that dictionary changes behind the
// emulator's back (another data folder's precompile, another game), and the title's pipeline
// cache then holds entries the driver keeps but no longer provides: every pipeline is compiled
// again at its first use (the "stale entries" of DriverCacheHistory). So the driver gets a disk
// cache of this data folder's own, next to the pipeline caches it belongs to, for the precompile
// and the game alike: what trains and holds the dictionary is then only what reads it. An
// environment that names a disk cache keeps it. Must run before the driver is loaded.
static void UsePrivateDriverDiskCache() {
    constexpr const wchar_t* Name = L"__GL_SHADER_DISK_CACHE_PATH";
    if (const wchar_t* given = _wgetenv(Name); given != nullptr && given[0] != L'\0') {
        return;
    }
    std::error_code error;
    const auto path = std::filesystem::current_path(error) / "_PipelineCache" / "nvidia";
    if (error) {
        return;
    }
    std::filesystem::create_directories(path, error);
    if (!error) {
        (void)_wputenv_s(Name, path.c_str());
    }
}

// Other programs put a Vulkan layer of their own into every game: the overlays of Steam, Epic,
// GOG Galaxy and RTSS, OBS's capture. Each of them runs inside the calls that make the instance
// and the device, between NVIDIA's Streamline and the driver when frame generation is on. A
// player's start with frame generation ended there, in the Vulkan loader, handed a device that
// was not its own, on a PC with two such layers this code was never run with. So with frame
// generation the loader is told to leave them out, all but the graphics vendors' own; it is the
// loader's own switch (VK_LOADER_LAYERS_DISABLE and _ALLOW, loader 1.3.262 and newer; an older
// loader ignores them). KYTY_VULKAN_LAYERS=1 keeps the layers, =0 leaves them out without
// frame generation too. A filter the environment already names is kept as it is. Must run
// before anything loads Vulkan (Streamline asks the loader about the GPU when it starts).
static void LeaveOutForeignVulkanLayers() {
    const wchar_t* choice           = _wgetenv(L"KYTY_VULKAN_LAYERS");
    const wchar_t* frame_generation = _wgetenv(L"KYTY_FRAMEGEN");
    const bool     chosen           = choice != nullptr && choice[0] != L'\0';
    const bool     leave_out =
        chosen ? _wtoi(choice) == 0 : frame_generation != nullptr && _wtoi(frame_generation) > 0;
    const auto given = [](const wchar_t* name) {
        const wchar_t* value = _wgetenv(name);
        return value != nullptr && value[0] != L'\0';
    };
    if (!leave_out || given(L"VK_LOADER_LAYERS_DISABLE") || given(L"VK_LOADER_LAYERS_ALLOW")) {
        return;
    }
    (void)_wputenv_s(L"VK_LOADER_LAYERS_DISABLE", L"~implicit~");
    (void)_wputenv_s(L"VK_LOADER_LAYERS_ALLOW", L"VK_LAYER_NV_*,VK_LAYER_AMD_*,VK_LAYER_INTEL_*");
    (void)_wputenv_s(L"KYTY_VULKAN_LAYERS_LEFT_OUT", L"1");
}

int wmain(int argc, wchar_t* argv[]) {
    Common::RegisterQuickExit();
    Common::HostException::InstallCrashNote();
    LeaveOutForeignVulkanLayers();
    // (Before the options are read: nothing may map memory first. operator new may have, with
    // mimalloc: its arenas are asked for at 2 TiB and above, outside the driver's room under
    // 1 TiB and below the guest's extended memory at 8 TiB.)
    bool dlss = false;
    for (int index = 1; index + 1 < argc; index++) {
        dlss = dlss || ((wcscmp(argv[index], L"--upscaler") == 0 || wcscmp(argv[index], L"--pssr-upscaler") == 0) &&
                        _wcsicmp(argv[index + 1], L"DLSS") == 0);
    }
    Libs::Graphics::FrameGen::HoldDriverAddressSpace(dlss);
    UsePrivateDriverDiskCache();
    std::vector<std::string> utf8_args;
    utf8_args.reserve(static_cast<size_t>(argc));

    for (int index = 0; index < argc; index++) {
        const std::wstring_view wide(argv[index]);
        const std::u16string utf16(wide.begin(), wide.end());

        utf8_args.push_back(Common::Utf16ToUtf8(utf16));
    }

    std::vector<char*> utf8_argv;
    utf8_argv.reserve(utf8_args.size());

    for (auto& argument: utf8_args) {
        utf8_argv.push_back(argument.data());
    }

    return Main(argc, utf8_argv.data());
}

#else

int main(int argc, char* argv[]) {
    return Main(argc, argv);
}

#endif
