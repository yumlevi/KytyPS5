#include "emulator.h"

#include "common/abi.h"
#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/hostException.h"
#include "common/liveControl.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/singleton.h"
#include "common/stringUtils.h"
#include "common/subsystems.h"
#include "common/systemInfo.h"
#include "common/threads.h"
#include "graphics/presentation/systemOverlay.h"
#include "graphics/presentation/window.h"
#include "graphics/shader/gameShaderSeeds.h"
#include "graphics/shader/shader.h"
#include "kernel/fileSystem.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "kernel/unrealCvars.h"
#include "kytyGitVersion.h"
#include "libs/agc.h"
#include "libs/audio.h"
#include "libs/controller.h"
#include "libs/libs.h"
#include "libs/network.h"
#include "loader/additionalContent.h"
#include "loader/guestFaultHistory.h"
#include "loader/runtimeLinker.h"
#include "loader/systemContent.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fmt/format.h>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace Emulator {

static void PrintSystemInfo() {
	const Common::SystemInfo info = Common::GetSystemInfo();

#if defined(__APPLE__)
	static constexpr auto platform_name = "macOS";
#elif KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static constexpr auto platform_name = "Windows";
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	static constexpr auto platform_name = "Linux";
#else
	static constexpr auto platform_name = "Unknown";
#endif

	LOGF("Build\n"
	     "  version: %s\n\n"
	     "Host\n"
	     "  os:      %s\n"
	     "  cpu:     %s\n"
	     "  threads: %u\n\n",
	     KYTY_BUILD_LABEL, platform_name, info.ProcessorName.c_str(),
	     std::thread::hardware_concurrency());
	// In every log, whatever its level: what a report of a crash or of a slow pass is read with.
	Log::WriteToConsoleAndLog(fmt::format("Host: {}, {} threads; {}\n", info.ProcessorName,
	                                      std::thread::hardware_concurrency(),
	                                      Common::HostMemoryText()));
	// From here (the guest's memory is set up, the process holds about 5 GB) a session of Saros
	// commits another 12 to 24 GB, measured. Much RAM does not mean that much can be had: a
	// session here lost the graphics device after three minutes with 0.6 GB left to commit,
	// having had 18.5 GB at this point, on a PC with 64 GB.
	constexpr uint64_t SessionCommitMiB = 24ull * 1024;
	if (const auto low = Common::LowCommitText(SessionCommitMiB); !low.empty()) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "WARNING: {}: a large game takes another 12 to 24 GB from here and may stop with a lost "
		    "graphics device or a crash when Windows runs out.\n",
		    low));
	}
}

static void KytyClose() {
	auto* rt = Common::Singleton<Loader::RuntimeLinker>::Instance();

	rt->Clear();

	LOGF("done!\n");

	Common::Subsystems::EmergencyShutdownActive();
}

static void MountOrCreateDir(const std::filesystem::path& dir, const std::string& point) {
	if (!Common::File::IsDirectoryExisting(dir)) {
		Common::File::CreateDirectories(dir);
	}

	EXIT_NOT_IMPLEMENTED(!Common::File::IsDirectoryExisting(dir));

	Libs::LibKernel::FileSystem::Mount(dir, point);
	auto dir_text = Common::PathToString(dir);
	LOGF("Mounted %s -> %s\n", point.c_str(), dir_text.c_str());
}

static void MountSandboxDirs() {
	std::string title_id;
	if (!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) {
		title_id = "UNKNOWN";
	}

	MountOrCreateDir("_DownloadData/" + title_id, "/download0");
	MountOrCreateDir("_TempData/" + title_id, "/temp0");
	MountOrCreateDir("_TempData/" + title_id, "/temp");
}

static void Init(const Config::ConfigOptions& cfg, const std::filesystem::path& param_json,
                 Common::Subsystems& subsystems) {
	EXIT_IF(!Common::Thread::IsMainThread());

	subsystems.Initialize<Config::Lifecycle>();
	Config::Load(cfg);
	subsystems.Initialize<Log::Lifecycle>();
	// (The first line of every log: a report is read with it, also one that ends early.)
	LOGF("Build: %s\n", KYTY_BUILD_LABEL);
	Common::HostException::CrashTest();

	if (Common::File::IsFileExisting(param_json)) {
		Loader::SystemContentLoadParamSfo(param_json);
	}
	if (cfg.splash_enabled) {
		Libs::Graphics::InitializeSplashScreen(param_json.parent_path());
	}

	// Initialization order is explicit; destruction is automatic and reversed.
	subsystems.Initialize<Libs::LibKernel::PthreadLifecycle>();
	subsystems.Initialize<Profiler::Lifecycle>();
	subsystems.Initialize<Libs::Network::Lifecycle>();
	subsystems.Initialize<Libs::LibKernel::Memory::Lifecycle>();
	subsystems.Initialize<Libs::LibKernel::FileSystem::Lifecycle>();
	subsystems.Initialize<Libs::Controller::Lifecycle>();
	subsystems.Initialize<Libs::Audio::Lifecycle>();
	subsystems.Initialize<Libs::Graphics::Lifecycle>();

	// The measurement channel (KYTY_LIVE_FILE): scripted pad input and a window close.
	Common::LiveControl::SetHostHooks({
	    .pad_button =
	        [](uint32_t button, bool down) {
		        Libs::Controller::SetButton(Libs::Controller::HOST_INPUT_CONTROLLER_ID, button, down);
	        },
	    .pad_sticks =
	        [](int left_x, int left_y, int right_x, int right_y) {
		        using Libs::Controller::Axis;
		        const int id = Libs::Controller::HOST_INPUT_CONTROLLER_ID;
		        Libs::Controller::SetAxis(id, Axis::LeftX, left_x);
		        Libs::Controller::SetAxis(id, Axis::LeftY, left_y);
		        Libs::Controller::SetAxis(id, Axis::RightX, right_x);
		        Libs::Controller::SetAxis(id, Axis::RightY, right_y);
	        },
	    .quit           = [] { Libs::Graphics::WindowRequestClose(); },
	    .pending_builds = [] { return Libs::Graphics::WindowPendingBackgroundBuilds(); },
	    .guest_next_mapping =
	        [](uint64_t address, uint64_t* start, uint64_t* end) {
		        return Libs::LibKernel::Memory::NextMappedRange(address, start, end);
	        },
	    .guest_read =
	        [](uint64_t address, void* data, uint64_t size) {
		        return Libs::LibKernel::Memory::TryReadBacking(address, data, size);
	        },
	    .guest_write =
	        [](uint64_t address, const void* data, uint64_t size) {
		        return Libs::LibKernel::Memory::StoreGuestBytes(address, data, size);
	        },
	    .guest_find =
	        [](const uint8_t* pattern, size_t size, uint64_t align, size_t limit, uint64_t from,
	           uint64_t to, uint64_t* scanned) {
		        return Libs::LibKernel::Memory::FindInBacking({pattern, size}, align, limit, from, to,
		                                                      scanned);
	        },
	    .guest_threads =
	        [] {
		        std::vector<std::pair<uint64_t, std::string>> threads;
		        Loader::GuestThreads().ForEach([&](uint64_t id, std::string_view name) {
			        threads.emplace_back(id, std::string(name));
		        });
		        return threads;
	        },
	});
	// --unreal-cvar: the title's own settings, once it runs.
	Libs::LibKernel::UnrealCvars::Start(Config::UnrealCvars());
	Common::LiveControl::Start();
}

static void LoadElf(const std::filesystem::path& elf, bool dbg_print_reloc = false,
                    const std::filesystem::path& save_name = {}) {
	auto* rt = Common::Singleton<Loader::RuntimeLinker>::Instance();

	auto* program = rt->LoadProgram(
	    Libs::LibKernel::FileSystem::GetRealFilename(Common::PathToGenericString(elf)));

	if (dbg_print_reloc) {
		program->dbg_print_reloc = true;
	}

	if (!save_name.empty()) {
		rt->SaveProgram(program, Libs::LibKernel::FileSystem::GetRealFilename(
		                             Common::PathToGenericString(save_name)));
	}
}

// The game's title id, from its parameters (PPSA07631).
static std::string TitleId() {
	std::string title_id;
	Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id);
	if (title_id.empty()) {
		std::string content_id;
		if (Loader::SystemContentParamSfoGetString("CONTENT_ID", &content_id) && content_id.size() == 36 &&
		    content_id[6] == '-' && content_id[16] == '_') {
			title_id = content_id.substr(7, 9);
		}
	}
	return title_id;
}

// The shaders of the game's own files, read into the seed file the precompile imports: the first
// time, and again when the game's files have changed since (graphics/shader/gameShaderSeeds.h).
static void WriteGameSeeds(const std::filesystem::path& game, const std::filesystem::path& seeds) {
	using State      = Libs::Graphics::GameShaderSeeds::State;
	const auto start = std::chrono::steady_clock::now();
	const auto made  =
	    Libs::Graphics::ShaderWriteGameSeeds(game, seeds, Libs::Graphics::ShaderSeedDecoderPath());
	const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - start;
	if (made.state == State::Written) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "Shader seeds: {} shaders from {} shader libraries of the game's files ({:.1f} MiB) "
		    "written to {} in {:.1f} s{}\n",
		    made.shaders, made.libraries, static_cast<double>(made.bytes) / (1024.0 * 1024.0),
		    Common::PathToString(seeds), seconds.count(),
		    made.no_shader != 0 ? fmt::format("; {} library entries hold no shader binary", made.no_shader)
		                        : std::string {}));
	} else if (made.state == State::Failed) {
		Log::WriteToConsoleAndLog(
		    fmt::format("Shader seeds: the game's shaders were not read from its files: {}\n", made.reason));
	}
}

static void Execute(const std::filesystem::path& game, const std::filesystem::path& game_patch) {
	if (const auto list = Config::ExportPipelineListPath(); !list.empty()) {
		// The renderer is up; the guest never starts.
		Common::QuickExit(Libs::Graphics::WindowExportPipelineList(list) ? 0 : 1);
	}
	if (Config::PrecompileOnlyEnabled()) {
		// Shaders from the game's files join the ones the game created in earlier sessions,
		// before the precompile lists what to build.
		// (A folder stands for its <title id>.seeds, when it has one.)
		auto            seeds = Config::ImportShadersPath();
		std::error_code seeds_error;
		if (std::filesystem::is_directory(seeds, seeds_error)) {
			const auto title_id = TitleId();
			seeds /= title_id + ".seeds";
			if (!title_id.empty()) {
				WriteGameSeeds(game, seeds);
			}
			if (title_id.empty() || !std::filesystem::exists(seeds, seeds_error)) {
				seeds.clear();
			}
		}
		if (!seeds.empty()) {
			const auto imported = Libs::Graphics::ShaderImportSeeds(seeds);
			Log::WriteToConsoleAndLog(fmt::format(
			    "Shader import: {} shaders ({} compute programs) from {}{}{}\n", imported.shaders,
			    imported.compute, Common::PathToString(seeds),
			    imported.rejected != 0 ? fmt::format(", {} not shader binaries", imported.rejected)
			                           : std::string {},
			    imported.read ? "" : "; the file could not be read to its end"));
		}
		// The renderer is up; the guest never starts. The driver cache is written when this
		// returns; builds abandoned past their time budget may still run, so the process ends
		// without joining them.
		Common::QuickExit(Libs::Graphics::WindowPrecompilePipelines());
	}
	std::thread([game_patch] {
		Common::Singleton<Loader::RuntimeLinker>::Instance()->Execute(game_patch);
	}).detach();
	Libs::Graphics::WindowRun();
	Common::QuickExit(0);
}

void Run(const RunOptions& options) {
	if (options.app0_dir.empty()) {
		EXIT("app0 directory is required\n");
	}

	if (options.elf.empty()) {
		EXIT("ELF is required\n");
	}

	const auto         param_json = options.app0_dir / "sce_sys" / "param.json";
	Common::Subsystems subsystems(true);
	Init(options.config, param_json, subsystems);

	PrintSystemInfo();
	std::string title_id;
	Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id);
	if (title_id.empty()) {
		std::string content_id;
		if (Loader::SystemContentParamSfoGetString("CONTENT_ID", &content_id) &&
		    content_id.size() == 36 && content_id[6] == '-' && content_id[16] == '_') {
			title_id = content_id.substr(7, 9);
		}
	}
	if (!title_id.empty()) {
		Log::WriteToConsoleAndLog(fmt::format("Title ID: {}\n", title_id));
	}

	int ok = atexit(KytyClose);
	EXIT_NOT_IMPLEMENTED(ok != 0);

	// Guest threads are still running, so skip KytyClose() and only flush emergency state.
	ok = at_quick_exit(Common::Subsystems::EmergencyShutdownActive);
	EXIT_NOT_IMPLEMENTED(ok != 0);

	Libs::LibKernel::FileSystem::Mount(options.app0_dir, "/app0");
	Libs::LibKernel::FileSystem::Mount(options.app0_dir, "/hostapp");

	MountSandboxDirs();

	Loader::AddContentLoad(options.addcont_dir, options.addcont, title_id);
	auto* rt = Common::Singleton<Loader::RuntimeLinker>::Instance();
	Libs::InitAll(rt->Symbols());

	LoadElf(options.elf);

	Execute(options.app0_dir, options.game_patch);
}

} // namespace Emulator
