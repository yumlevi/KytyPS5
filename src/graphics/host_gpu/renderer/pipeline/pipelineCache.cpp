#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/gpuTiming.h"
#include "common/liveControl.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/systemInfo.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/host_gpu/renderer/pipeline/driverCacheFile.h"
#include "graphics/host_gpu/renderer/pipeline/resourceLookahead.h"
#include "graphics/host_gpu/renderer/pipeline/scalarReadCache.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvDriverSimplify.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "kytyGitVersion.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cctype>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <cstdio>
#include <cstring>
#include <fmt/format.h>
#include <fstream>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

constexpr auto DriverCacheSaveInterval = std::chrono::minutes {5};

// 1: the precompile of a running session keeps the pipelines it builds for recorded programs
// and pipelines, and their first use takes them (WarmPipelines); 0: it destroys them, and the
// first use creates each from the driver cache on the GPU thread.
Common::LiveControl::Switch g_warm_pipelines("warm_pipelines", 1);
// 1: the code-only modules of the created compute shaders (those of the game's files among them)
// are translated in the background from the boot and kept for their first use
// (ProgramCache::StartPreparation); 0: such a shader is translated at its first use (about
// 13 ms then, its pipeline comes from the driver cache either way). 2, the default: as 1 on a PC
// with more than PrepareCreatedMemoryMiB of RAM that can still commit more than
// PrepareCreatedCommitMiB when the session starts, as 0 otherwise. The translations of a
// whole game stay in memory until they are used: for the 12,700 of Saros a session holds 6.6 GB
// more (23.1 GB against 16.5 GB at one spot), which a PC with 32 GB does not have to spare, nor
// one with 64 GB whose other programs have committed most of RAM and page file (a session then
// held 23.6 GB with 0.6 GB left to commit and lost the graphics device; it had started with
// 18.5 GB to spare and 5.5 GB held).
Common::LiveControl::Switch g_prepare_created("prepare_created", 2);
constexpr uint64_t          PrepareCreatedMemoryMiB = 40ull * 1024;
constexpr uint64_t          PrepareCreatedCommitMiB = 28ull * 1024;
// 1: on a processor with performance and efficiency cores, the threads that translate programs
// ahead and build pipelines beside the game stay on the efficiency cores
// (Thread::PinCurrentThreadToEfficiencyCpus); 0: the scheduler places them, as before. They
// run below normal priority, but sixteen of them are busy for the first minute and a half of a
// session (Saros: 20,000 translations, 250 s of CPU time, 218 of them the emission of the
// SPIR-V), and a thread of low priority still takes the second logical CPU of a core the
// thread that feeds the GPU runs on. Read when a thread starts.
//
// What that work costs the game, a replay of a session in Saros by ten-second windows from the
// level's start (three runs each): placed by the scheduler 39-42, 44-46, 40-42 fps and then,
// the work done, 51-54; on the efficiency cores 48-50, 52, 45 and, the work lasting half a
// minute longer, 44-48 where the others had finished; without the work (prepare_created 0)
// 55, 56, 48, 50, 53. So the cores make the dip half as deep and as much longer: the frames of
// the first minute come to the same. Half the threads for the shaders no session has used
// (the Demon's Souls fork's prefetch does that) made it shallower and longer again, the frame
// rate 3 lower for the whole two minutes, and is not kept. The cost is the work, wherever and
// however fast it runs; what removes it is not doing it again every session.
Common::LiveControl::Switch g_background_ecores("background_ecores", 1);
// 1: a compute dispatch that a module reading its formats at run time serves (a shader's
// code-only module, which the precompile built for every shader of the game) gets the module of
// its own formats translated and its pipeline built in the background, and later dispatches
// take that one (ProgramCache::OwnModuleFor): the run-time formats cost GPU time in every
// dispatch (measured: 41 ms instead of 17 ms of GPU work a frame where every compute shader ran
// on its code-only module). 0: the dispatch keeps the code-only module for the session.
Common::LiveControl::Switch g_upgrade_code_only("upgrade_code_only", 1);
// 1: a program whose GPU-selected image tables the host enumerates gets the candidate layout of
// the tables' key limit at once (IR::WidestCandidateLayout): one module however the tables grow,
// and for a recorded program the precompile builds it ahead; 0: a layout grows by doubling, a
// module each time (IR::WidenSpecialization).
Common::LiveControl::Switch g_widest_table_layouts("widest_table_layouts", 1);
// 1: with --skip-pending-draws, a graphics stage program that nothing has translated yet is
// translated on a helper thread and the draws that need it are left out until it is there
// (ProgramCache::Get); 0: it is translated on the GPU thread at its first draw, as without the
// option.
Common::LiveControl::Switch g_defer_stage_programs("defer_stage_programs", 1);

// Whether a guest program may store to guest memory, GDS or a storage image, going by its
// instructions alone. A draw of such a program must not be left out (PendingDrawBlocker::
// WritesMemory), and for a program nothing has translated that has to be known before it is.
// Stores through flat addresses count, although most of them go to scratch memory; a program
// that cannot be decoded here counts as one that may.
bool GuestCodeMayWriteMemory(std::span<const uint32_t> code) {
	namespace Decoder = ShaderRecompiler::Decoder;
	using Opcode      = Decoder::Opcode;
	const auto within = [](Opcode opcode, Opcode first, Opcode last) {
		return static_cast<uint32_t>(opcode) >= static_cast<uint32_t>(first) &&
		       static_cast<uint32_t>(opcode) <= static_cast<uint32_t>(last);
	};
	try {
		const Common::RecoverableFatalScope recoverable;
		Decoder::Program                    program;
		Decoder::DecodeProgram(code, program);
		return std::ranges::any_of(program.instructions, [&](const auto& inst) {
			const auto opcode = inst.opcode;
			return inst.gds ||
			       within(opcode, Opcode::BUFFER_STORE_FORMAT_X, Opcode::BUFFER_STORE_FORMAT_D16_XYZW) ||
			       within(opcode, Opcode::BUFFER_STORE_BYTE, Opcode::BUFFER_STORE_DWORDX4) ||
			       within(opcode, Opcode::TBUFFER_STORE_FORMAT_X, Opcode::TBUFFER_STORE_FORMAT_XYZW) ||
			       within(opcode, Opcode::BUFFER_ATOMIC_SWAP, Opcode::BUFFER_ATOMIC_FMAX) ||
			       within(opcode, Opcode::FLAT_STORE_BYTE, Opcode::FLAT_STORE_DWORDX4) ||
			       within(opcode, Opcode::IMAGE_STORE, Opcode::IMAGE_ATOMIC_FMAX);
		});
	} catch (const Common::RecoverableFatal&) {
		return true;
	}
}

uint8_t RemapSourceAlphaFactor(uint8_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Alpha);
		case Prospero::BlendFactor::kOneMinusSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
		default: return factor;
	}
}

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	// Driver cache entries already distinguish shader code and pipeline state across builds.
	return fmt::format("KytyPC2:{:08x}:{:08x}:{:08x}:{}\n",
	                   properties.vendorID, properties.deviceID, properties.driverVersion, uuid);
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

bool ReadShaderGuestMemory(void*, uint64_t address, std::span<uint32_t> values) {
	// Scalar and unformatted buffer dependencies use the same backing as native raw loads.
	// Image synchronization belongs to formatted buffer bindings, not these reads.
	return !values.empty() &&
	       Libs::LibKernel::Memory::TryReadBufferBacking(address, values.data(), values.size_bytes());
}

// The guest memory the walk reads on the GPU thread, synchronized with recorded GPU writes like
// the specialization reader's TryReadBufferBacking.
struct GuestScalarMemory {
	static bool Read(uint64_t address, void* data, uint64_t size) {
		return Libs::LibKernel::Memory::TryReadBacking(address, data, size);
	}
	static bool HasGpuDirtyBytes(uint64_t address, uint64_t size) {
		return Libs::LibKernel::Memory::HasGpuDirtyBytes(address, size);
	}
	static void ReadGpuDirtyBytes(std::span<const ScalarReadRange> ranges) {
		Libs::LibKernel::Memory::ReadGpuDirtyBytes(ranges);
	}
	static uint64_t GpuWriteSequence() { return Libs::LibKernel::Memory::GpuWriteSequence(); }
};
using GuestScalarReadCache = ScalarReadCache<GuestScalarMemory>;

// Out of line: the cache is several KiB, and a frame that size costs every cached read a stack
// probe.
[[gnu::noinline]] void ReadShaderScalarMemoryUncached(uint64_t            address,
                                                      std::span<uint32_t> values) {
	GuestScalarMemory    memory;
	GuestScalarReadCache uncached(memory);
	uncached.Read(address, values);
}

bool ReadShaderScalarMemory(void* userdata, uint64_t address, std::span<uint32_t> values) {
	if (auto* cache = static_cast<GuestScalarReadCache*>(userdata); cache != nullptr) {
		cache->Read(address, values);
		return true;
	}
	ReadShaderScalarMemoryUncached(address, values);
	return true;
}

// 1: a compute dispatch's walk does not wait for a DWORD the GPU still writes when only the
// shader takes its value (SrtRuntime::read_data_memory): the dispatch copies the DWORD into
// its flat buffer on the GPU (FillGpuWords, renderCompute.cpp). 0: every such DWORD is read back.
// Saros chains small compute passes, each reading a few bytes the one before wrote: 3 of the 8
// waits of a frame at a heavy spot were for a struct of six floats one of them hands the next.
Common::LiveControl::Switch g_gpu_scalars("gpu_scalars", 1);

// The reader of a walk's data-only loads on the GPU thread.
bool ReadShaderDataMemory(void* userdata, uint64_t address, uint32_t flat_offset,
                          uint32_t& value) {
	auto* cache = static_cast<GuestScalarReadCache*>(userdata);
	if (cache == nullptr) {
		ReadShaderScalarMemoryUncached(address, {&value, 1});
		return true;
	}
	if (!cache->ReadUnlessGpuDirty(address, value)) {
		cache->Deferred().emplace_back(flat_offset, address);
	}
	return true;
}

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / fmt::format("{:04d}_new_shader_{}_{:016x}.spv",
	                                                             id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(spirv.data(), spirv.size() * sizeof(uint32_t));
}

void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash,
                        std::span<const uint32_t> code) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}.bin", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(code.data(), code.size_bytes());
}

// The SPIR-V features of the host device every module of this run is emitted for.
ShaderRecompiler::Spirv::Profile HostShaderProfile(const GraphicContext& graphics) {
	// AMD GPUs (vendor 0x1002) run NMin/NMax as v_min/v_max, which give the guest's NaN and
	// signed-zero results; GLSL.std.450 alone does not order -0 below +0. NVIDIA GPUs (vendor
	// 0x10de) order the signed zeros and return the other operand of a single NaN as well, but
	// give a canonical NaN for two NaN operands.
	const auto vendor = graphics.physical_device_properties.vendorID;
	return {.maintenance8           = graphics.maintenance8_enabled,
	        .native_float_min_max   = vendor == 0x1002u,
	        .ordered_float_min_max  = vendor == 0x10deu,
	        .device_clock_period_ps = graphics.device_clock_period_ps,
	        .robust_buffer_access   = graphics.robust_buffer_access_enabled,
	        .r8_srgb_views          = graphics.r8_srgb_views,
	        .r8g8_srgb_views        = graphics.r8g8_srgb_views,
	        .relaxed_float          = Config::RelaxedShaderFloatEnabled(),
	        .ray_query              = graphics.ray_query_enabled,
	        .driver_simplify        = !Config::ShaderSimplificationDisabled()};
}

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv, bool maintenance8_enabled) {
	if (!Config::ShaderValidationEnabled()) {
		return true;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	spvtools::ValidatorOptions validator_options;
	validator_options.SetAllowOffsetTextureOperand(maintenance8_enabled);
	if (tools.Validate(spirv.data(), spirv.size(), validator_options)) {
		return true;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF_COLOR(Log::Color::BrightRed, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	           label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	return false;
}

} // namespace

std::size_t PipelineCache::GraphicsPipelineKeyHash::operator()(const GraphicsPipelineKey& key) const {
	std::size_t hash = 0;
	PipelineKeyHash::Mix(hash, key.rendering.color_count);
	for (uint32_t i = 0; i < key.rendering.color_count; i++) {
		PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.color_formats[i]));
	}
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.depth_format));
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.stencil_format));
	for (const auto id: key.vertex_shader_ids) {
		PipelineKeyHash::Mix(hash, id);
	}
	PipelineKeyHash::Mix(hash, key.ps_shader_id);
	PipelineKeyHash::Mix(hash, key.vertex_input.binding_count);
	for (uint32_t i = 0; i < key.vertex_input.binding_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].stride);
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].instance);
	}
	PipelineKeyHash::Mix(hash, key.vertex_input.attribute_count);
	for (uint32_t i = 0; i < key.vertex_input.attribute_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].offset);
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].binding);
	}
	PipelineKeyHash::Mix(hash, XXH3_64bits(&key.static_params, sizeof(key.static_params)));
	return hash;
}

struct PipelineCache::ProgramCache {
	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	struct Permutation {
		std::vector<uint32_t>                       function_code;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		ShaderProgram                                handle;
		uint64_t                                     spirv_hash = 0;
		// Prepared from the shader's code alone (StartPreparation), before any dispatch.
		bool                                         code_only = false;
	};

	struct DeferredTranslation;

	struct SourceEntry {
		SourceEntry(ShaderRecompiler::IR::ResourcePlan plan,
		            std::unique_ptr<ShaderRecompiler::ShaderSource> call_source)
		    : call_source(std::move(call_source)), resource_plan(std::move(plan)) {
			permutations.reserve(8);
		}

		std::span<const uint32_t> FunctionCode() const {
			return call_source
			    ? std::span<const uint32_t>(call_source->linked->code).subspan(call_source->code.size())
			    : std::span<const uint32_t>{};
		}

		bool RefreshCallSource(const ShaderRecompiler::IR::SrtRuntime& runtime) {
			const auto revision = call_source->revision;
			ShaderRecompiler::RefreshShaderSource(*call_source, runtime);
			resource_plan.source_reads = call_source->reads;
			return revision != call_source->revision;
		}

		// Refreshes resources and specialization for a draw. False when the draw cannot be
		// specialized: its guest data is unavailable, or the plan cannot serve one of its
		// descriptors (reported once per shader); the caller skips the draw. A dispatch is run
		// instead with the GPU-selected roots that cannot be served bound as null: its writes
		// through its other resources may be what the host or later work reads, and a skipped
		// dispatch would leave all of them stale. Once a dispatch needed that, its later walks
		// bind failed roots as null directly (a walk where every root is served is unchanged).
		bool Materialize(const ShaderRecompiler::IR::SrtRuntime& runtime, ShaderType stage) {
			auto       walk        = runtime;
			// The words the walk leaves to the dispatch (ReadShaderDataMemory), each once.
			auto* leaving = walk.read_data_memory == ReadShaderDataMemory
			                    ? static_cast<GuestScalarReadCache*>(walk.userdata)
			                    : nullptr;
			const auto materialize = [&] {
				if (leaving != nullptr) leaving->Deferred().clear();
				const bool done = ShaderRecompiler::IR::MaterializeResources(
				    resource_plan, walk, resources, specialization);
				if (leaving != nullptr) {
					for (const auto& word: leaving->Deferred()) {
						const auto same = std::ranges::find(resources.gpu_words, word.first,
						                                    &std::pair<uint32_t, uint64_t>::first);
						if (same == resources.gpu_words.end()) {
							resources.gpu_words.push_back(word);
						} else {
							same->second = word.second;
						}
					}
					std::ranges::sort(resources.gpu_words);
				}
				// The first report of each shader also goes to the log.
				if (const auto report = ShaderRecompiler::IR::TakeSpecializationFailureReport();
				    !report.empty()) {
					PipelineCacheLog("{}", report);
				}
				return done;
			};
			walk.null_failed_roots = null_failed_roots;
			if (!materialize()) {
				if (resources.unavailable || stage != ShaderType::Compute || null_failed_roots)
					return false;
				null_failed_roots      = true;
				walk.null_failed_roots = true;
				if (!materialize()) return false;
			}
			NoteUserDataReads();
			return true;
		}

		// After a refresh (Materialize, or a lookahead snapshot taken in its place).
		void NoteUserDataReads() {
			resources.user_data_reads =
			    resource_plan.user_data_reads |
			    (call_source != nullptr
			         ? ShaderRecompiler::IR::PlanUserDataReads(call_source->call_targets)
			         : 0u);
		}

		// Requires a successful Materialize for the current draw.
		template <bool HasCalls>
		Permutation* FindPermutation(uint32_t push_data_cursor) {
			// A module compiled for the draw's buffer formats before one that reads them at run
			// time (IR::PortableBufferFormats).
			for (const bool portable: {false, true}) {
				for (auto& candidate: permutations) {
					const auto& layout = candidate.program.bindings;
					if (layout.push_data_start_dword !=
					        ShaderRecompiler::IR::PushData::StartFor(push_data_cursor,
					                                                 layout.ShaderDataDwords()) ||
					    !ShaderRecompiler::IR::SpecializationServes(
					        resource_plan, candidate.specialization, specialization, portable,
					        &resources)) {
						continue;
					}
					if constexpr (HasCalls) {
						if (!std::ranges::equal(candidate.function_code, FunctionCode())) continue;
					}
					return &candidate;
				}
			}
			// A module whose candidate slots hold this draw's candidates serves it once they are
			// moved there; the latest module has the widest layout.
			for (auto it = permutations.rbegin(); it != permutations.rend(); ++it) {
				const auto& layout = it->program.bindings;
				if (layout.push_data_start_dword !=
				    ShaderRecompiler::IR::PushData::StartFor(push_data_cursor,
				                                             layout.ShaderDataDwords())) {
					continue;
				}
				if constexpr (HasCalls) {
					if (!std::ranges::equal(it->function_code, FunctionCode())) continue;
				}
				if (ShaderRecompiler::IR::AdaptToSpecialization(resource_plan, it->specialization,
				                                                resources, specialization)) {
					return &*it;
				}
			}
			return nullptr;
		}

		std::unique_ptr<ShaderRecompiler::ShaderSource> call_source;
		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		// The ranges the last walk read back (ScalarReadCache::Prefetch).
		std::vector<ScalarReadRange>                 walk_read_backs;
		// The first dispatch served by a module prepared from the code alone has been counted.
		bool                                         code_only_noted = false;
		// Modules a helper thread is compiling for draws that are left out meanwhile
		// (ProgramCache::deferral), by the specialization and push-data cursor they are for.
		struct PendingModule {
			ShaderRecompiler::IR::ResourceSpecialization specialization;
			uint32_t                                     push_data_cursor = 0;
			std::shared_ptr<DeferredTranslation>         job;
			// A compute program's own module (OwnModuleFor): the key of its record, whose
			// pipeline the precompile builds meanwhile.
			uint64_t record_key = 0;
		};
		std::vector<PendingModule> pending_modules;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		std::vector<Permutation>                    permutations;
		// The plan the lookahead helpers evaluate (null when they are off or for stages the
		// command streams do not plan).
		std::shared_ptr<LookaheadPlan>             lookahead;
		bool                                       skip_dispatch = false;
		// A dispatch of this shader had a GPU-selected root the plan cannot serve (Materialize).
		bool null_failed_roots = false;
	};

	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const {
			std::size_t hash = static_cast<std::size_t>(key.stage);
			PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash));
			if constexpr (sizeof(std::size_t) < sizeof(uint64_t)) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash >> 32u));
			}
			PipelineKeyHash::Mix(hash, key.user_data_count);
			PipelineKeyHash::Mix(hash, key.code_size);
			PipelineKeyHash::Mix(hash, key.static_state.size());
			// Bucket same-shape static variants by source. ProgramKey equality performs the one
			// exact state comparison needed on a stable hit without hashing the full state first.
			return hash;
		}
	};

	static constexpr std::size_t MaxStaticKeyWords = 32 + ShaderVertexInputInfo::RES_MAX * 6;

	// A program translated from its manifest records before its first use (StartPreparation):
	// its resource plans and the modules of the recorded specializations, as Get would have made
	// them over the session that recorded them.
	struct PreparedProgram {
		ShaderRecompiler::IR::ResourcePlan                  plan;
		std::unique_ptr<ShaderRecompiler::IR::ResourcePlan> lookahead_plan;
		std::vector<Permutation>                            permutations;
		std::vector<uint32_t>                               push_data_cursors;
	};

	// A translation a helper thread does for a draw that is left out meanwhile (see Deferral):
	// the program's plans (a program nothing has translated), or one module of a program that
	// has them. The helper writes `program` before it sets `state`; the GPU thread reads it
	// after.
	struct DeferredTranslation {
		enum State : int { Running, Done, Failed };
		std::atomic<int>                      state {Running};
		std::unique_ptr<PreparedProgram>      program;
		std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
	};

	// The records of one program key, in the order they were first added (one of the two lists).
	struct PreparationJob {
		ProgramKey                                               key;
		std::vector<std::shared_ptr<const ComputeProgramRecord>> compute;
		std::vector<std::shared_ptr<const GraphicsStageRecord>>  stages;
		// The created compute shader of this key whose code-only pipeline the precompile built
		// (code_only_records), or null: its code-only module is prepared after the recorded ones.
		std::shared_ptr<const ComputeProgramRecord> code_only;
	};

	// The programs earlier sessions recorded, translated on helper threads in the order they were
	// first needed. Get takes a program's translation at its first use: it waits for one that is
	// being translated and translates a queued one itself (the helpers then skip it).
	struct Preparation {
		enum class State : uint8_t { Queued, Running, Done, Taken };
		struct Slot {
			State                            state = State::Queued;
			std::unique_ptr<PreparedProgram> program; // null when the record cannot be translated
		};
		std::mutex                                           mutex;
		std::condition_variable                              changed;
		std::unordered_map<ProgramKey, Slot, ProgramKeyHash> slots;
		std::vector<PreparationJob>                          jobs;
		size_t                                               next    = 0;
		size_t                                               running = 0;
		bool                                                 stop    = false;
		PipelineCache::ProgramPreparationCounts              counts;
		std::chrono::steady_clock::time_point                started;
		std::vector<std::thread>                             threads;
	};

	// The key Get gives the program a record describes, with the input its translation reads (the
	// record's static input with the fields Get derives from the device).
	bool PreparationKey(const ComputeProgramRecord& record, ProgramKey& key,
	                    ShaderComputeInputInfo& input) const {
		if (record.code == nullptr || record.code->empty()) return false;
		input                    = record.input;
		input.host_subgroup_size = graphics.ComputeHostSubgroupSize();
		input.lds_storage =
		    input.lds_size_dwords * 4u >
		    graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
		key.stage           = ShaderType::Compute;
		key.hash            = record.shader_hash;
		key.user_data_count = record.user_data_count;
		// A dispatch names the program (ShaderParams::code); the record of a created shader holds
		// its whole binary, footer included.
		key.code_size = static_cast<uint32_t>(ShaderProgramWords(*record.code).size());
		BuildStageStaticKey(input, key.static_state);
		return true;
	}
	bool PreparationKey(const GraphicsStageRecord& record, ProgramKey& key,
	                    ShaderVertexInputInfo& vertex, ShaderPixelInputInfo& pixel) const {
		if (record.code == nullptr || record.code->empty() ||
		    !GraphicsStageRecord::IsGraphicsStage(record.stage))
			return false;
		key.stage           = record.stage;
		key.hash            = record.shader_hash;
		key.user_data_count = record.user_data_count;
		key.code_size       = static_cast<uint32_t>(record.code->size());
		if (record.stage == ShaderType::Pixel) {
			pixel = record.pixel;
			BuildStageStaticKey(pixel, key.static_state);
			return true;
		}
		vertex = record.vertex;
		// The fields GetGraphicsPrograms derives from the device.
		if (record.stage == ShaderType::Mesh) {
			vertex.mesh.host_subgroup_size = graphics.subgroup_size;
		}
		if (vertex.clip_space.enabled) {
			const auto& limits = graphics.GetPhysicalDeviceProperties().limits;
			vertex.clip_space.half_extent[0] =
			    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
			vertex.clip_space.half_extent[1] =
			    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		}
		BuildStageStaticKey(vertex, key.static_state);
		return true;
	}

	// Translates a record as Get translates the program for a permutation and compiles the
	// record's specialization into `prepared`; the plans come from the first record that does.
	// False when the record no longer translates to its specialization.
	// `code_only`: a created compute shader's record, which has no specialization: the module is
	// the one the precompile built from its code alone (PrecompileComputeProgram).
	// `serve_for`: the resources of the draw a stage record with code_only was made for: its
	// module is the code-only one when that is the record's specialization or serves it there,
	// and the record's own otherwise.
	template <typename Record>
	bool PrepareRecord(const Record& record, PreparedProgram& prepared, bool code_only = false,
	                   bool                                          plan_only = false,
	                   const ShaderRecompiler::IR::ResourceSnapshot* serve_for = nullptr) const {
		ProgramKey                       key;
		ShaderComputeInputInfo           compute {};
		ShaderVertexInputInfo            vertex {};
		ShaderPixelInputInfo             pixel {};
		ShaderRecompiler::CompileOptions options;
		if constexpr (std::is_same_v<Record, ComputeProgramRecord>) {
			if (!PreparationKey(record, key, compute)) return false;
			options.stage              = ShaderType::Compute;
			options.input_info.compute = &compute;
			options.wave_size          = compute.wave_size;
			options.dump_label         = "ShaderRecompiler CS";
		} else {
			if (!PreparationKey(record, key, vertex, pixel)) return false;
			options.stage = record.stage;
			if (record.stage == ShaderType::Pixel) {
				options.input_info.pixel = &pixel;
				options.wave_size        = pixel.wave_size;
				options.dump_label       = "ShaderRecompiler PS";
			} else {
				options.input_info.vertex = &vertex;
				options.wave_size         = vertex.wave_size;
				options.user_data_base    = 8;
				if (record.stage == ShaderType::Mesh ||
				    record.stage == ShaderType::TessellationControl) {
					options.user_data_base = 0;
				}
				if (record.stage == ShaderType::Mesh) {
					options.wave_size = vertex.mesh.wave_size;
				}
				options.dump_label = "ShaderRecompiler VS";
			}
			if (record.back_code != nullptr) {
				options.back_code = *record.back_code;
			}
		}
		// Translation reads the user-data count; the values only feed the resource walk.
		const std::vector<uint32_t> user_data(record.user_data_count, 0u);
		options.user_data       = user_data;
		options.profile         = &profile;
		options.shader_hash     = key.hash;
		options.dump_ir         = false;
		options.early_dump      = false;
		options.emit_debug_info = false;
		const bool                                     first = prepared.permutations.empty();
		std::optional<ShaderRecompiler::CompileResult> result;
		auto                                           specialization = record.specialization;
		// (As the precompile built it: outside the simplification store.)
		std::optional<ShaderRecompiler::Spirv::SimplifiedStoreBypass> unstored;
		if (code_only) {
			unstored.emplace();
		}
		try {
			const Common::RecoverableFatalScope recoverable;
			auto source = ShaderRecompiler::PrepareShaderSource(*record.code, options);
			if (!source.calls.empty() || source.unsupported_call_pc != UINT32_MAX) return false;
			auto translated = ShaderRecompiler::TranslateProgram(source.decoded, options);
			if (code_only) {
				auto from_code = ShaderRecompiler::IR::CodeOnlySpecialization(translated.program);
				if (!from_code.has_value()) return false;
				specialization = std::move(*from_code);
			}
			if constexpr (!std::is_same_v<Record, ComputeProgramRecord>) {
				if (record.code_only && !plan_only) {
					auto from_code = ShaderRecompiler::IR::CodeOnlySpecialization(translated.program);
					if (from_code.has_value() &&
					    (*from_code == specialization ||
					     (serve_for != nullptr &&
					      ShaderRecompiler::IR::SpecializationServes(
					          ShaderRecompiler::IR::ExtractResourcePlan(
					              translated.program, ShaderRecompiler::IR::SrtWalkerLoads::Reader),
					          *from_code, specialization, true, serve_for)))) {
						specialization = std::move(*from_code);
					}
				}
			}
			// `plan_only`: the plans of a program whose draw has no specialization yet.
			// A recompiler change may give the shader other resources than the record was made
			// for.
			if (!plan_only &&
			    !ShaderRecompiler::IR::SpecializationApplies(translated.program, specialization))
				return false;
			if (first) {
				prepared.plan = ShaderRecompiler::IR::ExtractResourcePlan(
				    translated.program, ShaderRecompiler::IR::SrtWalkerLoads::Reader);
				if (lookahead != nullptr && SlotOf(options.stage) != LookaheadSlot::None &&
				    prepared.plan.srt_code != nullptr) {
					prepared.lookahead_plan = std::make_unique<ShaderRecompiler::IR::ResourcePlan>(
					    ShaderRecompiler::IR::ExtractResourcePlan(
					        translated.program, ShaderRecompiler::IR::SrtWalkerLoads::Reader,
					        ShaderRecompiler::IR::SrtWalkerCompilation::Deferred));
				}
			}
			if (plan_only) return true;
			result = ShaderRecompiler::CompileProgram(std::move(translated), options, specialization,
			                                          record.push_data_cursor);
		} catch (const Common::RecoverableFatal& fatal) {
			// (The precompile pass has said so for a created shader.)
			if (!code_only) {
				PipelineCacheLog("Program preparation: {:016x} cannot be translated: {}", key.hash,
				                 fatal.message);
			}
			return false;
		}
		if (!ValidateShaderSpirv(options.dump_label, key.hash, result->spirv,
		                         profile.maintenance8)) {
			return false;
		}
		const auto module = CompileSPV(result->spirv, graphics.device);
		if (module == nullptr) return false;
		SetVulkanObjectNameF(graphics.device, module, "{} {:016x}",
		                     options.stage == ShaderType::Compute ? "cs"
		                     : options.stage == ShaderType::Pixel ? "ps"
		                                                          : "vs",
		                     key.hash);
		const auto spirv_hash =
		    XXH3_64bits(result->spirv.data(), result->spirv.size() * sizeof(uint32_t));
		prepared.permutations.push_back({
		    .specialization = std::move(specialization),
		    .program        = std::move(result->program).TakeCompiledInfo(),
		    .handle         = {.module = module, .spirv_hash = spirv_hash},
		    .spirv_hash     = spirv_hash,
		    .code_only      = code_only,
		});
		prepared.push_data_cursors.push_back(record.push_data_cursor);
		return true;
	}

	// Every record of the job's program; null when none of them translates any more.
	std::unique_ptr<PreparedProgram> Prepare(const PreparationJob& job) const {
		auto prepared = std::make_unique<PreparedProgram>();
		for (const auto& record: job.compute) {
			(void)PrepareRecord(*record, *prepared);
		}
		for (const auto& record: job.stages) {
			(void)PrepareRecord(*record, *prepared);
		}
		// Last: a module for the dispatch's own formats is found before it (FindPermutation).
		if (job.code_only != nullptr) {
			(void)PrepareRecord(*job.code_only, *prepared, true);
		}
		if (prepared->permutations.empty()) return nullptr;
		return prepared;
	}

	void PreparationWorker() {
		Common::Thread::LowerCurrentThreadPriority();
		if (g_background_ecores.On()) {
			(void)Common::Thread::PinCurrentThreadToEfficiencyCpus();
		}
		auto&            state = *preparation;
		std::unique_lock lock(state.mutex);
		for (;;) {
			while (!state.stop && state.next < state.jobs.size() &&
			       state.slots.at(state.jobs[state.next].key).state != Preparation::State::Queued) {
				state.next++;
			}
			if (state.stop || state.next == state.jobs.size()) break;
			const auto& job  = state.jobs[state.next++];
			auto&       slot = state.slots.at(job.key);
			slot.state       = Preparation::State::Running;
			state.running++;
			lock.unlock();
			auto program = Prepare(job);
			lock.lock();
			(program != nullptr ? state.counts.prepared : state.counts.failed)++;
			slot.program = std::move(program);
			slot.state   = Preparation::State::Done;
			state.running--;
			if (state.running == 0 && state.next == state.jobs.size()) {
				PipelineCacheLog(
				    "Programs prepared ahead: {} of {} recorded or built from their code translated "
				    "in {:.1f} s ({} failed)",
				    state.counts.prepared, state.counts.queued,
				    std::chrono::duration<double>(std::chrono::steady_clock::now() - state.started)
				        .count(),
				    state.counts.failed);
			}
			state.changed.notify_all();
		}
		state.changed.notify_all();
	}

	// Queues every recorded program for translation on `thread_count` low-priority threads, in
	// first-seen order. A program's records (the permutations an earlier session compiled) are
	// prepared together, in the order they were recorded, as that session's program ended.
	void StartPreparation(uint32_t thread_count) {
		if (thread_count == 0 || preparation != nullptr) return;
		// Dumps and logs of a translation belong to the program's first use.
		if (Config::GetShaderLogDirection() != Config::LogDirection::Silent ||
		    Config::GraphicsDebugDumpEnabled())
			return;
		std::vector<std::pair<uint64_t, PreparationJob>>       ordered;
		std::unordered_map<ProgramKey, size_t, ProgramKeyHash> index;
		const auto job_for = [&](uint64_t position, const ProgramKey& key) -> PreparationJob& {
			const auto [found, added] = index.try_emplace(key, ordered.size());
			if (added) ordered.emplace_back(position, PreparationJob {.key = key});
			return ordered[found->second].second;
		};
		if (manifest != nullptr) {
			const auto records = manifest->Records();
			for (size_t i = 0; i < records.size(); i++) {
				ProgramKey             key;
				ShaderComputeInputInfo input {};
				if (PreparationKey(*records[i], key, input))
					job_for(FirstSeenPosition(i, records.size()), key)
					    .compute.push_back(records[i]);
			}
		}
		if (pipeline_manifest != nullptr) {
			const auto records = pipeline_manifest->Stages();
			for (size_t i = 0; i < records.size(); i++) {
				ProgramKey            key;
				ShaderVertexInputInfo vertex {};
				ShaderPixelInputInfo  pixel {};
				if (PreparationKey(*records[i], key, vertex, pixel))
					job_for(FirstSeenPosition(i, records.size()), key).stages.push_back(records[i]);
			}
		}
		// The compute shaders whose pipelines the precompile built from their code alone (created
		// by the game, or imported from its files): their code-only module is translated too,
		// after everything recorded, so that a first use it serves takes it instead of
		// translating the shader then.
		// (Both: a PC with 64 GB whose other programs hold 80 GB of RAM and page file has as
		// little to spare as one with 32 GB.)
		const auto memory          = Common::PhysicalMemoryMiB();
		const auto committable     = Common::CommitAvailableMiB();
		const bool room            = (memory == 0 || memory > PrepareCreatedMemoryMiB) &&
		                             (committable == 0 || committable > PrepareCreatedCommitMiB);
		const bool prepare_created = g_prepare_created.Get() == 2 ? room : g_prepare_created.On();
		if (code_only_records && !prepare_created && g_prepare_created.Get() == 2) {
			PipelineCacheLog("Programs prepared ahead: the compute shaders no session has used are "
			                 "translated at their first use on this PC ({} GB of RAM, {} GB can still "
			                 "be committed; KYTY_PREPARE_CREATED=1 translates them all ahead)",
			                 memory / 1024, committable / 1024);
		}
		if (code_only_records && prepare_created) {
			uint64_t position = uint64_t {1} << 32u;
			for (auto& record: code_only_records()) {
				ProgramKey             key;
				ShaderComputeInputInfo input {};
				if (!PreparationKey(*record, key, input)) continue;
				auto& job = job_for(position++, key);
				if (job.code_only == nullptr) job.code_only = std::move(record);
			}
		}
		if (ordered.empty()) return;
		std::ranges::stable_sort(ordered, {}, [](const auto& item) { return item.first; });
		auto state = std::make_unique<Preparation>();
		state->jobs.reserve(ordered.size());
		for (auto& [position, job]: ordered) {
			(void)position;
			state->slots.try_emplace(job.key);
			state->jobs.push_back(std::move(job));
		}
		state->counts.queued = state->jobs.size();
		state->started       = std::chrono::steady_clock::now();
		preparation          = std::move(state);
		for (uint32_t i = 0; i < thread_count; i++) {
			preparation->threads.emplace_back([this] { PreparationWorker(); });
		}
	}

	void StopPreparation() {
		if (preparation == nullptr) return;
		{
			std::lock_guard lock(preparation->mutex);
			preparation->stop = true;
		}
		for (auto& thread: preparation->threads) {
			thread.join();
		}
		preparation->threads.clear();
	}

	// Waits until no recorded program is queued or being translated.
	void WaitForPreparation() {
		if (preparation == nullptr) return;
		std::unique_lock lock(preparation->mutex);
		preparation->changed.wait(lock, [&] {
			return preparation->running == 0 &&
			       (preparation->stop || preparation->next == preparation->jobs.size());
		});
	}

	// Takes the translation of the program `key` names, when one was prepared: waits for a
	// running translation, and keeps the helpers from translating a queued one.
	std::unique_ptr<PreparedProgram> TakePrepared(const ProgramKey& key) {
		auto&            state = *preparation;
		std::unique_lock lock(state.mutex);
		const auto       found = state.slots.find(key);
		if (found == state.slots.end()) return nullptr;
		auto& slot = found->second;
		if (slot.state == Preparation::State::Queued) {
			slot.state = Preparation::State::Taken;
			state.counts.missed++;
			return nullptr;
		}
		if (slot.state == Preparation::State::Running) {
			state.counts.waited++;
			state.changed.wait(lock, [&] { return slot.state != Preparation::State::Running; });
		}
		if (slot.state != Preparation::State::Done) return nullptr;
		slot.state = Preparation::State::Taken;
		if (slot.program != nullptr) state.counts.adopted++;
		return std::move(slot.program);
	}

	// Makes a prepared program the program of `lookup_key`, as if Get had translated it at this
	// use; the draw then specializes and finds its permutation like any later use.
	template <typename InputInfo>
	auto AdoptPrepared(ShaderType stage, const ShaderParams& params, const InputInfo& input_info) {
		auto prepared = TakePrepared(lookup_key);
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			// A shader prepared from its code alone has the input of the graphics queue
			// (ShaderGetCreatedComputeInput). The queue changes nothing in a translation: it only
			// decides whether S_GETREG_B32 has one.
			if (prepared == nullptr && input_info.async_compute) {
				auto input          = input_info;
				input.async_compute = false;
				auto key            = lookup_key;
				BuildStageStaticKey(input, key.static_state);
				prepared = TakePrepared(key);
			}
		}
		if (prepared == nullptr) return programs.end();
		return AdoptProgram(std::move(prepared), stage, params, input_info);
	}

	// Makes `prepared` the program of `lookup_key` (see AdoptPrepared).
	template <typename InputInfo>
	auto AdoptProgram(std::unique_ptr<PreparedProgram> prepared, ShaderType stage,
	                  const ShaderParams& params, const InputInfo& input_info) {
		auto entry = programs.try_emplace(lookup_key, std::move(prepared->plan), nullptr).first;
		NoteGraphicsShaderUse(stage, params.code);
		for (size_t i = 0; i < prepared->permutations.size(); i++) {
			auto& permutation =
			    entry->second.permutations.emplace_back(std::move(prepared->permutations[i]));
			permutation.handle.id = ++next_shader_id;
			const auto cursor     = prepared->push_data_cursors[i];
			if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
				if (manifest != nullptr) {
					permutation.handle.record_key =
					    RecordComputeProgram(params, input_info, cursor, permutation);
				}
			} else {
				if (pipeline_manifest != nullptr) {
					permutation.handle.record_key =
					    RecordGraphicsStage(stage, params, input_info, cursor, permutation);
				}
			}
		}
		if (prepared->lookahead_plan != nullptr && lookahead != nullptr) {
			entry->second.lookahead = std::make_shared<LookaheadPlan>(
			    walker_queue.Submit(std::move(prepared->lookahead_plan)), next_lookahead_helper++);
		}
		return entry;
	}

	// --skip-pending-draws, stage programs. A draw whose pipeline is not built yet is left out
	// while the pipeline is built in the background; the translation of a stage nothing has
	// translated used to come first, on the GPU thread (about 5 ms a stage, most of it the SPIR-V
	// emission), so every new material still cost the game a late frame or two. With `deferral`
	// set for the draw (GetGraphicsPrograms: the draw is one that may be left out), Get instead
	// hands such a program to a helper thread and reports the draw as deferred, twice: first for
	// the program's plans, which the resource walk needs, then, once a draw has been specialized,
	// for the module of that specialization. Each takes a helper a few milliseconds, so the draw
	// comes two or three frames later than it would have; nothing waits. A translation that
	// fails or takes longer than the skip window is done on the GPU thread as before.
	struct Deferral {
		std::mutex                        mutex;
		std::condition_variable           wake;
		std::deque<std::function<void()>> jobs;
		// Translations nothing waits for (OwnModuleFor), taken when `jobs` is empty.
		std::deque<std::function<void()>> later;
		std::vector<std::thread>          threads;
		bool                              stop = false;
	};
	static constexpr auto DeferralPatience = std::chrono::milliseconds {2000};

	void StartDeferral() {
		if (deferral != nullptr) return;
		deferral = std::make_unique<Deferral>();
		for (int i = 0; i < 2; i++) {
			deferral->threads.emplace_back([this] {
				auto&            state = *deferral;
				std::unique_lock lock(state.mutex);
				for (;;) {
					state.wake.wait(lock, [&] {
						return state.stop || !state.jobs.empty() || !state.later.empty();
					});
					if (state.stop) return;
					auto& queue = !state.jobs.empty() ? state.jobs : state.later;
					auto  job   = std::move(queue.front());
					queue.pop_front();
					lock.unlock();
					job();
					lock.lock();
				}
			});
		}
	}

	void StopDeferral() {
		if (deferral == nullptr) return;
		{
			std::lock_guard lock(deferral->mutex);
			deferral->stop = true;
		}
		deferral->wake.notify_all();
		for (auto& thread: deferral->threads) {
			thread.join();
		}
		deferral->threads.clear();
	}

	// The stage record of the program a draw asks for, as RecordGraphicsStage makes it.
	template <typename InputInfo>
	static std::shared_ptr<GraphicsStageRecord>
	StageRecordOf(ShaderType stage, const ShaderParams& params, const InputInfo& input,
	              uint32_t                                             push_data_cursor,
	              const ShaderRecompiler::IR::ResourceSpecialization& specialization) {
		auto record              = std::make_shared<GraphicsStageRecord>();
		record->stage            = stage;
		record->shader_hash      = params.hash;
		record->user_data_count  = params.user_data_count;
		record->push_data_cursor = push_data_cursor;
		if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			record->pixel = GraphicsStageRecord::StaticInput(input);
		} else {
			record->vertex = GraphicsStageRecord::StaticInput(input);
		}
		record->specialization = specialization;
		record->SetCode(
		    std::make_shared<const std::vector<uint32_t>>(params.code.begin(), params.code.end()),
		    nullptr);
		return record;
	}

	// Hands a stage's translation to the helpers: its plans (`plan_only`), or the module of the
	// record's specialization.
	std::shared_ptr<DeferredTranslation>
	Defer(std::shared_ptr<GraphicsStageRecord> record, bool plan_only,
	      std::shared_ptr<const ShaderRecompiler::IR::ResourceSnapshot> resources = nullptr) {
		StartDeferral();
		auto job = std::make_shared<DeferredTranslation>();
		{
			std::lock_guard lock(deferral->mutex);
			deferral->jobs.push_back([this, job, record = std::move(record), plan_only,
			                          resources = std::move(resources)] {
				auto       prepared = std::make_unique<PreparedProgram>();
				const bool done =
				    PrepareRecord(*record, *prepared, false, plan_only, resources.get()) &&
				    (plan_only || prepared->permutations.size() == 1);
				if (done) job->program = std::move(prepared);
				job->state.store(done ? DeferredTranslation::Done : DeferredTranslation::Failed,
				                 std::memory_order_release);
			});
		}
		deferral->wake.notify_one();
		return job;
	}

	// The module of a compute record, translated by a helper when it has nothing else to do.
	std::shared_ptr<DeferredTranslation> DeferLater(std::shared_ptr<const ComputeProgramRecord> record) {
		StartDeferral();
		auto job = std::make_shared<DeferredTranslation>();
		{
			std::lock_guard lock(deferral->mutex);
			deferral->later.push_back([this, job, record = std::move(record)] {
				auto       prepared = std::make_unique<PreparedProgram>();
				const bool done =
				    PrepareRecord(*record, *prepared) && prepared->permutations.size() == 1;
				if (done) job->program = std::move(prepared);
				job->state.store(done ? DeferredTranslation::Done : DeferredTranslation::Failed,
				                 std::memory_order_release);
			});
		}
		deferral->wake.notify_one();
		return job;
	}

	// A compute dispatch that `served` serves without being the module of the dispatch's own
	// formats (it reads them at run time: a shader's code-only module): the program's own
	// module, once a helper has translated it and the precompile has built its pipeline, both
	// started here at the first such dispatch. Null until then, and for a dispatch whose own
	// module `served` is. The record is the one RecordExactComputeProgram keeps for the next
	// session; this session gets its module too, because the run-time formats cost GPU time in
	// every dispatch.
	static constexpr size_t MaxOwnModules = 8;
	template <typename Entry>
	Permutation* OwnModuleFor(Entry& cached, const Permutation& served, const ShaderParams& params,
	                          const ShaderComputeInputInfo& input_info, uint32_t push_data_cursor) {
		if (!g_upgrade_code_only.On() || manifest == nullptr || cached.call_source ||
		    !build_later || !built_now || served.specialization == cached.specialization ||
		    ShaderRecompiler::IR::SpecializationServes(cached.resource_plan, served.specialization,
		                                               cached.specialization, false,
		                                               &cached.resources)) {
			return nullptr;
		}
		auto&      pending = cached.pending_modules;
		const auto found   = std::ranges::find_if(pending, [&](const auto& module) {
			return module.push_data_cursor == push_data_cursor &&
			       module.specialization == cached.specialization;
		});
		if (found == pending.end()) {
			if (pending.size() >= MaxOwnModules) return nullptr;
			auto record = std::make_shared<ComputeProgramRecord>(
			    ExactComputeRecord(params, input_info, push_data_cursor, cached.specialization));
			const auto key = record->Key();
			(void)manifest->Add(*record);
			build_later(record);
			pending.push_back(
			    {cached.specialization, push_data_cursor, DeferLater(std::move(record)), key});
			own_modules_started++;
			return nullptr;
		}
		if (found->job == nullptr) return nullptr;
		const auto state = found->job->state.load(std::memory_order_acquire);
		if (state == DeferredTranslation::Running) return nullptr;
		if (state == DeferredTranslation::Failed) {
			// Stays listed, so that it is not started again.
			found->job = nullptr;
			return nullptr;
		}
		if (!built_now(found->record_key)) return nullptr;
		auto  prepared  = std::move(found->job->program);
		auto& installed = cached.permutations.emplace_back(std::move(prepared->permutations[0]));
		installed.handle.id         = ++next_shader_id;
		installed.handle.record_key = found->record_key;
		pending.erase(found);
		own_modules_taken++;
		return &installed;
	}

	// Where a deferred translation stands; one that has been running for too long counts as
	// failed (the caller then translates on the GPU thread, and the helper's result is dropped).
	static int DeferredState(const DeferredTranslation& job) {
		const auto state = job.state.load(std::memory_order_acquire);
		if (state == DeferredTranslation::Running &&
		    std::chrono::steady_clock::now() - job.started > DeferralPatience) {
			return DeferredTranslation::Failed;
		}
		return state;
	}

	// Whether the program of a draw that may be left out may be translated by a helper: a plain
	// vertex, mesh or pixel program that stores nowhere, while nothing dumps or logs
	// translations (those belong to the first draw).
	bool MayDeferStage(ShaderType stage, const ShaderParams& params) {
		if (!defer_stage_programs || !g_defer_stage_programs.On() || !params.back_code.empty() ||
		    (stage != ShaderType::Vertex && stage != ShaderType::Pixel && stage != ShaderType::Mesh) ||
		    Config::GetShaderLogDirection() != Config::LogDirection::Silent ||
		    Config::GraphicsDebugDumpEnabled()) {
			return false;
		}
		const auto [known, added] = code_may_write.try_emplace(params.hash, false);
		if (added) known->second = GuestCodeMayWriteMemory(params.code);
		return !known->second;
	}

	void PrintShaderCounts() const {
		std::array<size_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> counts {};
		for (const auto& [key, source]: programs) {
			counts[static_cast<size_t>(key.stage)] += source.permutations.size();
		}
		// Guest geometry shaders are compiled through the host mesh stage.
		std::printf("Shaders: VS %zu | PS %zu | CS %zu | GS %zu | LS %zu | HS %zu | TES %zu\n",
		            counts[static_cast<size_t>(ShaderType::Vertex)],
		            counts[static_cast<size_t>(ShaderType::Pixel)],
		            counts[static_cast<size_t>(ShaderType::Compute)],
		            counts[static_cast<size_t>(ShaderType::Mesh)],
		            counts[static_cast<size_t>(ShaderType::Local)],
		            counts[static_cast<size_t>(ShaderType::TessellationControl)],
		            counts[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
	}

	// A draw whose GPU-selected resources cannot be enumerated (unmapped guest memory, more
	// probes than the host reads) or served (a descriptor the plan does not support) is skipped
	// rather than run with descriptors that may name the wrong memory; the shader's next draw is
	// specialized afresh.
	template <typename InputInfo>
	static ShaderProgram SkipUnavailableDraw(ShaderType stage, uint64_t hash,
	                                         InputInfo& input_info) {
		input_info.stage = {};
		static std::atomic<uint32_t> reports {0};
		if (reports.fetch_add(1u, std::memory_order_relaxed) < 64u) {
			PipelineCacheLog("Shader {} {:016x}: resources cannot be specialized for this draw, "
			                 "skipped",
			                 static_cast<uint32_t>(stage), hash);
		}
		return {};
	}

	// Names the first resource whose specialization changed between a shader's last permutation
	// and the one about to be compiled, so a shader that keeps compiling can be understood.
	static void LogPermutationDifference(const char* stage_name, uint64_t hash,
	                                     const ShaderRecompiler::IR::ResourceSpecialization& old,
	                                     const ShaderRecompiler::IR::ResourceSpecialization& now) {
		if (old.buffers.size() != now.buffers.size() || old.images.size() != now.images.size() ||
		    old.samplers.size() != now.samplers.size()) {
			LOGF("Permutation: %s %016" PRIx64 " resource counts buffers %zu->%zu images %zu->%zu "
			     "samplers %zu->%zu\n",
			     stage_name, hash, old.buffers.size(), now.buffers.size(), old.images.size(),
			     now.images.size(), old.samplers.size(), now.samplers.size());
			return;
		}
		for (size_t i = 0; i < now.buffers.size(); ++i) {
			const auto& a = old.buffers[i];
			const auto& b = now.buffers[i];
			if (a == b) continue;
			LOGF("Permutation: %s %016" PRIx64 " buffer %zu stride %u->%u format %u->%u swizzle "
			     "%u->%u zero_oob %d->%d scalar_empty %d->%d unbound %d->%d root %u->%u mapping "
			     "%u->%u iterations %u->%u\n",
			     stage_name, hash, i, a.packed_stride, b.packed_stride,
			     static_cast<uint32_t>(a.descriptor_format),
			     static_cast<uint32_t>(b.descriptor_format), a.descriptor_swizzle,
			     b.descriptor_swizzle, a.zero_stride_oob ? 1 : 0, b.zero_stride_oob ? 1 : 0,
			     a.scalar_empty ? 1 : 0, b.scalar_empty ? 1 : 0, a.unbound ? 1 : 0,
			     b.unbound ? 1 : 0, a.indirect_root, b.indirect_root, a.indirect_mapping_offset,
			     b.indirect_mapping_offset, a.indirect_search_iterations,
			     b.indirect_search_iterations);
			return;
		}
		for (size_t i = 0; i < now.images.size(); ++i) {
			const auto& a = old.images[i];
			const auto& b = now.images[i];
			if (a == b) continue;
			LOGF("Permutation: %s %016" PRIx64 " image %zu numeric %u->%u dimension %u->%u mips "
			     "%u->%u conversion %u->%u swizzle %u->%u root %u->%u mapping %u->%u iterations "
			     "%u->%u cube %d->%d fmask %d->%d bindless %d->%d\n",
			     stage_name, hash, i, static_cast<uint32_t>(a.numeric_class),
			     static_cast<uint32_t>(b.numeric_class), static_cast<uint32_t>(a.dimension),
			     static_cast<uint32_t>(b.dimension), a.mip_count, b.mip_count,
			     static_cast<uint32_t>(a.conversion_format), static_cast<uint32_t>(b.conversion_format),
			     a.shader_swizzle, b.shader_swizzle, a.indirect_root, b.indirect_root,
			     a.indirect_mapping_offset, b.indirect_mapping_offset, a.indirect_search_iterations,
			     b.indirect_search_iterations, a.cube ? 1 : 0, b.cube ? 1 : 0, a.fmask ? 1 : 0,
			     b.fmask ? 1 : 0, a.bindless ? 1 : 0, b.bindless ? 1 : 0);
			return;
		}
		for (size_t i = 0; i < now.samplers.size(); ++i) {
			const auto& a = old.samplers[i];
			const auto& b = now.samplers[i];
			if (a == b) continue;
			LOGF("Permutation: %s %016" PRIx64 " sampler %zu root %u->%u mapping %u->%u iterations "
			     "%u->%u\n",
			     stage_name, hash, i, a.indirect_root, b.indirect_root, a.indirect_mapping_offset,
			     b.indirect_mapping_offset, a.indirect_search_iterations,
			     b.indirect_search_iterations);
			return;
		}
		LOGF("Permutation: %s %016" PRIx64 " specialization equal; push data or function code "
		     "changed\n",
		     stage_name, hash);
	}

	Permutation CompilePermutation(const char*                                  stage_name,
	                               const ShaderRecompiler::CompileOptions&      options,
	                               ShaderRecompiler::TranslateResult            translated,
	                               ShaderRecompiler::IR::ResourceSpecialization specialization,
	                               uint32_t push_data_start_dword,
	                               std::span<const uint32_t> function_code) {
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::ShadersCompiled,
		                                         Common::DebugCounters::Counter::ShaderCompileNs,
		                                         Common::DebugCounters::Peak::ShaderCompileNs);
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               specialization, push_data_start_dword);
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, result.spirv,
		                         options.profile->maintenance8)) {
			DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);

		const auto module = CompileSPV(result.spirv, graphics.device);
		EXIT_IF(module == nullptr);
		// Captures (RenderDoc) show the guest shader behind each module.
		SetVulkanObjectNameF(graphics.device, module, "{} {:016x}", stage_name, options.shader_hash);
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(result.spirv.size()), options.wave_size);
		}
		const auto spirv_hash =
		    XXH3_64bits(result.spirv.data(), result.spirv.size() * sizeof(uint32_t));
		return {
		    .function_code  = {function_code.begin(), function_code.end()},
		    .specialization = std::move(specialization),
		    .program        = std::move(result.program).TakeCompiledInfo(),
		    .handle         = {.id = ++next_shader_id, .module = module, .spirv_hash = spirv_hash},
		    .spirv_hash     = spirv_hash,
		};
	}

	// Adds a compiled compute permutation to the title's manifest and returns its key. Shaders
	// with calls read guest code at run time and are not recorded.
	uint64_t RecordComputeProgram(const ShaderParams& params, const ShaderComputeInputInfo& input,
	                              uint32_t push_data_cursor, const Permutation& permutation) {
		ComputeProgramRecord record;
		record.shader_hash = params.hash;
		record.code =
		    std::make_shared<const std::vector<uint32_t>>(params.code.begin(), params.code.end());
		record.user_data_count  = params.user_data_count;
		record.input            = ComputeProgramRecord::StaticInput(input);
		record.push_data_cursor = push_data_cursor;
		record.specialization   = permutation.specialization;
		record.spirv_hash       = permutation.spirv_hash;
		const auto key          = record.Key();
		compute_recorded++;
		const bool added = manifest->Add(record);
		if (!added || (list_predicts && list_predicts(key))) {
			compute_predicted++;
		}
		// A program with enumerated image tables: the layout the tables take beyond their key
		// limit (IR::BindlessTableLayout) is recorded with it and its pipeline built in the
		// background now, for the dispatch that first finds a table that large.
		if (added && g_widest_table_layouts.On()) {
			if (auto layout = ShaderRecompiler::IR::BindlessTableLayout(record.specialization)) {
				auto other           = record;
				other.specialization = std::move(*layout);
				other.spirv_hash     = 0;
				RecordExactComputeProgram(std::move(other));
			}
		}
		return key;
	}

	// Records the specialization a dispatch asked for when its first use took the shader's
	// code-only module instead (see Get): the next session prepares and precompiles the module of
	// that specialization, and this one builds its pipeline in the background.
	void RecordExactComputeProgram(ComputeProgramRecord record) {
		if (manifest->Add(record) && build_later) {
			build_later(std::make_shared<const ComputeProgramRecord>(std::move(record)));
		}
	}

	// The record of the program a dispatch asks for (see RecordComputeProgram), without the
	// module's hash, which is not part of its key.
	static ComputeProgramRecord
	ExactComputeRecord(const ShaderParams& params, const ShaderComputeInputInfo& input,
	                   uint32_t                                             push_data_cursor,
	                   const ShaderRecompiler::IR::ResourceSpecialization& specialization) {
		ComputeProgramRecord record;
		record.shader_hash = params.hash;
		record.code =
		    std::make_shared<const std::vector<uint32_t>>(params.code.begin(), params.code.end());
		record.user_data_count  = params.user_data_count;
		record.input            = ComputeProgramRecord::StaticInput(input);
		record.push_data_cursor = push_data_cursor;
		record.specialization   = specialization;
		return record;
	}

	// Adds a compiled graphics stage permutation to the title's pipeline manifest and returns
	// its key. Shaders with calls are not recorded.
	template <typename InputInfo>
	uint64_t RecordGraphicsStage(ShaderType stage, const ShaderParams& params,
	                             const InputInfo& input, uint32_t push_data_cursor,
	                             const Permutation& permutation) {
		GraphicsStageRecord record;
		record.stage            = stage;
		record.shader_hash      = params.hash;
		record.user_data_count  = params.user_data_count;
		record.push_data_cursor = push_data_cursor;
		if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			record.pixel = GraphicsStageRecord::StaticInput(input);
		} else {
			record.vertex = GraphicsStageRecord::StaticInput(input);
		}
		record.specialization = permutation.specialization;
		record.spirv_hash     = permutation.spirv_hash;
		record.SetCode(
		    std::make_shared<const std::vector<uint32_t>>(params.code.begin(), params.code.end()),
		    params.back_code.empty() ? nullptr
		                             : std::make_shared<const std::vector<uint32_t>>(
		                                   params.back_code.begin(), params.back_code.end()));
		(void)pipeline_manifest->AddStage(record);
		return record.Key();
	}

	// Counts a graphics shader's first use, and whether the guest created it before.
	void NoteGraphicsShaderUse(ShaderType stage, std::span<const uint32_t> code) {
		if (created_code == nullptr || stage == ShaderType::Compute) return;
		const auto id = ShaderCodeId::Of(code);
		if (!census_seen.insert(id.hash).second) return;
		auto& counts = census[static_cast<size_t>(stage)];
		counts[0]++;
		if (created_code->Find(id) != nullptr) counts[1]++;
	}

	template <typename InputInfo>
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info,
	                  uint32_t& push_data_cursor) {
		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		GuestScalarMemory    scalar_memory;
		GuestScalarReadCache read_cache(scalar_memory);
		static_assert(GuestScalarReadCache::ChunkBytes ==
		                  ShaderRecompiler::IR::SrtReadCacheView::ChunkBytes &&
		              std::has_single_bit(GuestScalarReadCache::Entries));
		const ShaderRecompiler::IR::SrtReadCacheView read_cache_view {
		    .tags       = read_cache.Tags(),
		    .data       = read_cache.Data(),
		    .stride     = GuestScalarReadCache::Stride(),
		    .index_mask = static_cast<uint32_t>(GuestScalarReadCache::Entries - 1u),
		};
		ShaderRecompiler::IR::SrtRuntime runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    .read_memory                = ReadShaderScalarMemory,
		    .userdata                   = &read_cache,
		    .read_specialization_memory = ReadShaderGuestMemory,
		    .image_heap                 = image_heap,
		    .read_cache                 = &read_cache_view,
		    .mapped_range_size          = Libs::LibKernel::Memory::MappedRangeSize,
		};
		ShaderRecompiler::CompileOptions options {.wave_size = input_info.wave_size};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.stage            = input_info.logical_stage;
			options.input_info.vertex = &input_info;
			options.user_data_base   = 8;
			if (options.stage == ShaderType::Mesh ||
			    options.stage == ShaderType::TessellationControl) {
				options.user_data_base = 0;
			}
			if (options.stage == ShaderType::Mesh) {
				options.wave_size = input_info.mesh.wave_size;
			}
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			options.stage            = ShaderType::Pixel;
			options.input_info.pixel = &input_info;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			options.stage              = ShaderType::Compute;
			options.input_info.compute = &input_info;
			runtime.workgroup_counts   = input_info.workgroup_counts;
			runtime.indirect_args_address = input_info.indirect_args_address;
			// (Compute only: a draw's flat buffer is uploaded inside a render pass.)
			if (g_gpu_scalars.On()) runtime.read_data_memory = ReadShaderDataMemory;
		}
		const auto stage           = options.stage;
		lookup_key.stage           = stage;
		lookup_key.hash            = params.hash;
		lookup_key.user_data_count = params.user_data_count;
		lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, lookup_key.static_state);
		auto entry = programs.find(lookup_key);
		if (entry == programs.end() && preparation != nullptr) {
			entry = AdoptPrepared(stage, params, input_info);
		}
		deferred = false;
		if constexpr (!std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			// A program nothing has translated, of a draw that may be left out: a helper
			// thread makes its plans (see Deferral).
			if (entry == programs.end() && MayDeferStage(stage, params)) {
				auto& job = pending_plans[lookup_key];
				if (job == nullptr) {
					job = Defer(StageRecordOf(stage, params, input_info, push_data_cursor, {}), true);
					deferred_plans++;
				}
				const auto state = DeferredState(*job);
				if (state == DeferredTranslation::Running) {
					deferred = true;
					return {};
				}
				const auto finished = std::move(job);
				pending_plans.erase(lookup_key);
				if (state == DeferredTranslation::Done) {
					entry = AdoptProgram(std::move(finished->program), stage, params, input_info);
				} else {
					deferred_failed++;
				}
			}
		}
		const auto bind = [&](const Permutation& permutation) {
			input_info.stage = {.program   = &permutation.program,
			                    .resources = &entry->second.resources,
			                    .handle    = &permutation.handle};
			permutation.program.bindings.AdvancePushData(push_data_cursor);
			return permutation.handle;
		};
		bool source_changed = false;
		if (entry != programs.end()) {
			auto& cached = entry->second;
			if (cached.skip_dispatch) return {};
			Permutation* permutation = nullptr;
			if (cached.call_source) source_changed = cached.RefreshCallSource(runtime);
			if (!source_changed) {
				bool adopted = false;
				if (cached.lookahead != nullptr) {
					PublishLookahead(stage, params, cached);
					adopted =
					    lookahead_command != nullptr && AdoptLookahead(cached, params, input_info);
				}
				bool materialized = adopted;
				if (!adopted) {
					read_cache.Prefetch(cached.walk_read_backs);
					materialized = cached.Materialize(runtime, stage);
					read_cache.Prefetch({});
					if (const auto read_backs = read_cache.ReadBacks(); !read_backs.empty()) {
						cached.walk_read_backs.assign(read_backs.begin(), read_backs.end());
					}
				}
				if (!materialized) return SkipUnavailableDraw(stage, params.hash, input_info);
				permutation = cached.call_source ? cached.FindPermutation<true>(push_data_cursor)
				                                 : cached.FindPermutation<false>(push_data_cursor);
			}
			if (permutation != nullptr) {
				// The first dispatch of a shader prepared from its code alone: when that module
				// is not the one for the dispatch's own formats, that one is recorded and built in
				// the background, for the next session (as when the shader is translated here).
				if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
					if (permutation->code_only && !cached.code_only_noted) {
						cached.code_only_noted = true;
						if (permutation->specialization == cached.specialization) {
							code_only_same++;
						} else {
							code_only_taken++;
							auto record = ExactComputeRecord(params, input_info, push_data_cursor,
							                                 cached.specialization);
							if (manifest != nullptr && !cached.call_source &&
							    !(list_predicts && list_predicts(record.Key()))) {
								RecordExactComputeProgram(std::move(record));
							}
						}
					}
					if (auto* own =
					        OwnModuleFor(cached, *permutation, params, input_info, push_data_cursor)) {
						permutation = own;
					}
				}
				return bind(*permutation);
			}
		}

		const char* label = nullptr;
		const char* stage_name = nullptr;
		switch (options.stage) {
			case ShaderType::Vertex: label = "ShaderRecompiler VS"; stage_name = "vs"; break;
			case ShaderType::Mesh: label = "ShaderRecompiler MS"; stage_name = "ms"; break;
			case ShaderType::Local: label = "ShaderRecompiler LS"; stage_name = "ls"; break;
			case ShaderType::TessellationControl: label = "ShaderRecompiler HS"; stage_name = "hs"; break;
			case ShaderType::TessellationEvaluation: label = "ShaderRecompiler DS"; stage_name = "ds"; break;
			case ShaderType::Pixel: label = "ShaderRecompiler PS"; stage_name = "ps"; break;
			case ShaderType::Compute: label = "ShaderRecompiler CS"; stage_name = "cs"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		options.profile     = &profile;
		options.shader_hash = params.hash;
		options.user_data   = user_data;
		options.back_code      = params.back_code;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		options.early_dump  = options.dump_ir;
		options.emit_debug_info = Config::GraphicsDebugDumpEnabled();
		options.dump_label  = label;
		DumpShaderOriginal(stage_name, options.shader_hash, params.code);
		ShaderRecompiler::ShaderSource source;
		if (entry == programs.end()) source = ShaderRecompiler::PrepareShaderSource(params.code, options);
		if (source.unsupported_call_pc != UINT32_MAX) {
			PipelineCacheLog("!!! TEMPORARY COMPUTE DISPATCH SKIP !!! shader=0x{:016x} pc=0x{:08x} "
			                 "reason=unsupported GPU-selected callable target; rendering is incomplete",
			                 params.hash, source.unsupported_call_pc);
			programs.try_emplace(lookup_key, ShaderRecompiler::IR::ResourcePlan{}, nullptr)
			    .first->second.skip_dispatch = true;
			return {};
		}
		auto* current = entry == programs.end() ? &source : entry->second.call_source.get();
		// A program that has its plans, of a draw that may be left out: a helper thread
		// compiles the module (see Deferral, and below, where the specialization is known), and
		// nothing is translated here.
		bool defer_module = false;
		if constexpr (!std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			defer_module = entry != programs.end() && current == nullptr &&
			               MayDeferStage(stage, params);
		}
		ShaderRecompiler::TranslateResult translated;
		if (current != nullptr) {
			if (!current->calls.empty()) {
				if (entry == programs.end()) ShaderRecompiler::RefreshShaderSource(*current, runtime);
				translated = ShaderRecompiler::TranslateProgram(current->linked->decoded, options);
				translated.program.source_reads = current->reads;
			} else {
				translated = ShaderRecompiler::TranslateProgram(current->decoded, options);
			}
		} else if (!defer_module) {
			translated = ShaderRecompiler::TranslateProgram(params.code, options);
		}
		if (entry == programs.end()) {
			NoteGraphicsShaderUse(stage, params.code);
			auto retained = !source.calls.empty() ? std::make_unique<ShaderRecompiler::ShaderSource>(std::move(source)) : nullptr;
			entry = programs
			            .try_emplace(
			                lookup_key,
			                ShaderRecompiler::IR::ExtractResourcePlan(
			                    translated.program, ShaderRecompiler::IR::SrtWalkerLoads::Reader),
			                std::move(retained))
			            .first;
			if (lookahead != nullptr && SlotOf(stage) != LookaheadSlot::None &&
			    !entry->second.call_source && entry->second.resource_plan.srt_code != nullptr) {
				entry->second.lookahead = std::make_shared<LookaheadPlan>(
				    walker_queue.Submit(std::make_unique<ShaderRecompiler::IR::ResourcePlan>(
				        ShaderRecompiler::IR::ExtractResourcePlan(
				            translated.program, ShaderRecompiler::IR::SrtWalkerLoads::Reader,
				            ShaderRecompiler::IR::SrtWalkerCompilation::Deferred))),
				    next_lookahead_helper++);
				PublishLookahead(stage, params, entry->second);
			}
			if (!entry->second.Materialize(runtime, stage))
				return SkipUnavailableDraw(stage, params.hash, input_info);
		} else if (source_changed) {
			entry->second.resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(
			    translated.program, ShaderRecompiler::IR::SrtWalkerLoads::Reader);
			if (!entry->second.Materialize(runtime, stage))
				return SkipUnavailableDraw(stage, params.hash, input_info);
			if (const auto* permutation = entry->second.FindPermutation<true>(push_data_cursor)) {
				return bind(*permutation);
			}
		}
		// One line per re-translation (never per draw), so it is always on.
		if (!entry->second.permutations.empty() && !defer_module) {
			LogPermutationDifference(stage_name, params.hash,
			                         entry->second.permutations.back().specialization,
			                         entry->second.specialization);
		}
		// A shader whose GPU-selected tables change with the scene gets the union of this draw's
		// candidate layout and the last module's, so that the draws both serve need no module. A
		// layout an earlier session recorded (whose module the precompile already built) is
		// taken as it is when it serves the draw, and is the one widened before any module.
		const ShaderRecompiler::IR::ResourceSpecialization* prior    = nullptr;
		bool                                                recorded = false;
		// The widest module: a draw that a narrower module served (another candidate count) must
		// not restart the growth from its layout.
		const auto width = [](const ShaderRecompiler::IR::ResourceSpecialization& layout) {
			return layout.images.size() + layout.samplers.size();
		};
		for (const auto& permutation: entry->second.permutations) {
			if (std::ranges::equal(permutation.function_code, entry->second.FunctionCode()) &&
			    (prior == nullptr || width(permutation.specialization) >= width(*prior)))
				prior = &permutation.specialization;
		}
		std::vector<std::shared_ptr<const ComputeProgramRecord>> records;
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			if (manifest != nullptr && !entry->second.call_source) {
				records = manifest->Records();
				// This program's records, the widest first: the module of the widest one that
				// serves the draw also serves the most later draws.
				std::erase_if(
				    records, [&, state = std::vector<uint32_t> {}](const auto& record) mutable {
					    if (record->shader_hash != params.hash ||
					        record->user_data_count != params.user_data_count ||
					        record->code == nullptr || record->code->size() != params.code.size() ||
					        record->push_data_cursor != push_data_cursor)
						    return true;
					    auto input               = record->input;
					    input.host_subgroup_size = input_info.host_subgroup_size;
					    BuildStageStaticKey(input, state);
					    return state != lookup_key.static_state;
				    });
				std::ranges::stable_sort(records, std::greater {}, [&](const auto& record) {
					return width(record->specialization);
				});
				for (const auto& record: records) {
					recorded = ShaderRecompiler::IR::AdaptToSpecialization(
					    entry->second.resource_plan, record->specialization,
					    entry->second.resources, entry->second.specialization);
					if (recorded) break;
				}
				if (!records.empty() &&
				    (prior == nullptr || width(records.front()->specialization) > width(*prior)))
					prior = &records.front()->specialization;
			}
		}
		const auto images_before = entry->second.specialization.images.size();
		if (!recorded && prior != nullptr) {
			ShaderRecompiler::IR::WidenSpecialization(entry->second.resource_plan, *prior,
			                                          entry->second.resources,
			                                          entry->second.specialization);
		}
		// And the module about to be compiled is the last one such tables need: their layouts
		// stop growing at the key limit, so it takes that layout now instead of the next
		// doubling (a table that streams in took a module at 1, 4, 8, 16 and 32 candidates).
		if (!recorded && g_widest_table_layouts.On()) {
			(void)ShaderRecompiler::IR::WidenToKeyLimit(entry->second.resource_plan,
			                                            entry->second.resources,
			                                            entry->second.specialization);
		}
		if ((recorded || entry->second.specialization.images.size() != images_before) &&
		    !defer_module) {
			LOGF("Permutation: %s %016" PRIx64 " %s images %zu->%zu\n", stage_name, params.hash,
			     recorded ? "recorded layout" : "widened", images_before,
			     entry->second.specialization.images.size());
		}
		// Unbound buffers take the typed layouts already compiled for them, so this module also
		// serves those bound cases.
		auto specialization = entry->second.specialization;
		for (auto it = entry->second.permutations.rbegin(); it != entry->second.permutations.rend();
		     ++it) {
			if (std::ranges::equal(it->function_code, entry->second.FunctionCode()))
				ShaderRecompiler::IR::AdoptServingBuffers(entry->second.resource_plan,
				                                          specialization, it->specialization);
		}
		// A shader whose pipeline the precompile built from its code alone, before any dispatch
		// (a created shader, or one imported from the game's files): when that code-only module
		// serves this dispatch, it is the one taken, and the pipeline needs no compile. It reads
		// the formats of its buffers at run time; the module for the formats themselves is left
		// to the next session. Not when an imported pipeline list names the dispatch's own
		// module: the precompile has built that pipeline, or is about to.
		std::optional<ComputeProgramRecord> exact;
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			if (!recorded && manifest != nullptr && !entry->second.call_source && code_only_built &&
			    code_only_built(params.hash)) {
				auto code_only = ShaderRecompiler::IR::CodeOnlySpecialization(translated.program);
				if (code_only.has_value() && *code_only == specialization) {
					code_only_same++;
				} else if (code_only.has_value() &&
				           ShaderRecompiler::IR::SpecializationServes(
				               entry->second.resource_plan, *code_only, specialization, true,
				               &entry->second.resources)) {
					auto record =
					    ExactComputeRecord(params, input_info, push_data_cursor, specialization);
					if (list_predicts && list_predicts(record.Key())) {
						code_only_listed++;
					} else {
						exact          = std::move(record);
						specialization = std::move(*code_only);
						code_only_taken++;
					}
				} else {
					code_only_other++;
				}
			}
		}
		if constexpr (!std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			if (defer_module) {
				auto&      pending = entry->second.pending_modules;
				const auto found   = std::ranges::find_if(pending, [&](const auto& module) {
					return module.push_data_cursor == push_data_cursor &&
					       module.specialization == specialization;
				});
				if (found == pending.end()) {
					auto record =
					    StageRecordOf(stage, params, input_info, push_data_cursor, specialization);
					// A shader the precompile built pair pipelines for: its code-only module
					// when that serves this draw (see below), judged with the draw's images.
					std::shared_ptr<ShaderRecompiler::IR::ResourceSnapshot> resources;
					if (stage_code_only && stage_code_only(params.hash)) {
						record->code_only = true;
						resources = std::make_shared<ShaderRecompiler::IR::ResourceSnapshot>();
						resources->images = entry->second.resources.images;
					}
					pending.push_back({specialization, push_data_cursor,
					                   Defer(std::move(record), false, std::move(resources))});
					deferred_modules++;
					deferred = true;
					return {};
				}
				const auto state = DeferredState(*found->job);
				if (state == DeferredTranslation::Running) {
					deferred = true;
					return {};
				}
				const auto finished = std::move(found->job);
				pending.erase(found);
				if (state == DeferredTranslation::Done) {
					auto& added = entry->second.permutations.emplace_back(
					    std::move(finished->program->permutations.front()));
					added.handle.id = ++next_shader_id;
					if (pipeline_manifest != nullptr) {
						added.handle.record_key =
						    RecordGraphicsStage(stage, params, input_info, push_data_cursor, added);
					}
					PrintShaderCounts();
					return bind(added);
				}
				deferred_failed++;
				translated = ShaderRecompiler::TranslateProgram(params.code, options);
			}
			// A shader the precompile built pipelines for from the code of a shader pair no
			// session had drawn (PipelineCache::QueueShaderPairs): the module that code gives
			// by itself is taken when it is this draw's or serves it, so that the pipeline the
			// draw creates is the one the precompile built. Like a compute shader's code-only
			// module, it reads formats at run time that the draw's own module has compiled in.
			if (!entry->second.call_source && stage_code_only && stage_code_only(params.hash)) {
				auto code_only = ShaderRecompiler::IR::CodeOnlySpecialization(translated.program);
				if (code_only.has_value() &&
				    (*code_only == specialization ||
				     ShaderRecompiler::IR::SpecializationServes(
				         entry->second.resource_plan, *code_only, specialization, true,
				         &entry->second.resources))) {
					specialization = std::move(*code_only);
				}
			}
		}
		entry->second.permutations.push_back(CompilePermutation(
		    stage_name, options, std::move(translated), std::move(specialization), push_data_cursor,
		    entry->second.FunctionCode()));
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			if (manifest != nullptr && !entry->second.call_source) {
				auto& added = entry->second.permutations.back();
				added.handle.record_key =
				    RecordComputeProgram(params, input_info, push_data_cursor, added);
				if (exact.has_value()) {
					RecordExactComputeProgram(std::move(*exact));
				}
			}
		} else {
			if (pipeline_manifest != nullptr && !entry->second.call_source) {
				auto& added = entry->second.permutations.back();
				added.handle.record_key =
				    RecordGraphicsStage(stage, params, input_info, push_data_cursor, added);
			}
		}

		PrintShaderCounts();
		return bind(entry->second.permutations.back());
	}

	// Programs whose walks the command streams plan ahead, by the registers that name them.
	enum class LookaheadSlot : uint64_t { None, Compute, Vertex, Pixel };
	static LookaheadSlot SlotOf(ShaderType stage) {
		switch (stage) {
			case ShaderType::Compute: return LookaheadSlot::Compute;
			case ShaderType::Vertex:
			case ShaderType::Mesh: return LookaheadSlot::Vertex;
			case ShaderType::Pixel: return LookaheadSlot::Pixel;
			default: return LookaheadSlot::None;
		}
	}
	static uint64_t LookaheadKey(LookaheadSlot slot, uint64_t shader_address) {
		return (shader_address << 2u) | static_cast<uint64_t>(slot);
	}

	// The programs last used at a shader address: planned walks of that address use their plans,
	// and their user data starts with the same words before the user SGPRs.
	void PublishLookahead(ShaderType stage, const ShaderParams& params, const SourceEntry& entry) {
		auto& target = lookahead_targets[LookaheadKey(SlotOf(stage), params.Base())];
		target.uses++;
		auto& uses = target.programs;
		// Most recent first; a program new to the address replaces the older one.
		if (uses[0].plan != entry.lookahead) {
			std::swap(uses[0], uses[1]);
			if (uses[0].plan != entry.lookahead) uses[0].plan = entry.lookahead;
		}
		auto& use           = uses[0];
		use.last_use        = target.uses;
		use.user_data_count = params.user_data_count;
		std::copy_n(params.user_data.begin(), use.prefix.size(), use.prefix.begin());
	}

	// Takes the helper's snapshot for this command when ResourceLookahead allows it.
	template <typename InputInfo>
	bool AdoptLookahead(SourceEntry& cached, const ShaderParams& params,
	                    const InputInfo& input_info) {
		LookaheadInputs inputs;
		inputs.shader_base = params.Base();
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			inputs.indirect_args_address = input_info.indirect_args_address;
			inputs.workgroup_counts      = {input_info.workgroup_counts[0],
			                                input_info.workgroup_counts[1],
			                                input_info.workgroup_counts[2]};
		}
		inputs.user_data_count = params.user_data_count;
		std::copy_n(params.user_data.begin(), params.user_data_count, inputs.user_data.begin());
		if (lookahead->TryAdopt(lookahead_stream, lookahead_command, cached.lookahead.get(), inputs,
		                        cached.resources, cached.specialization) !=
		    ResourceLookahead::Outcome::Adopted) {
			return false;
		}
		cached.NoteUserDataReads();
		return true;
	}

	// Plans the walk of the programs used at `shader_address` with these user SGPRs: the last one,
	// and the one before it while the two alternate (the scan cannot tell which one a command
	// uses; the job of the other one is dropped at the next command).
	void PlanLookahead(const void* stream, const void* command, LookaheadSlot slot,
	                   uint64_t shader_address, std::span<const uint32_t> user_sgprs,
	                   LookaheadInputs& inputs) {
		const auto target = lookahead_targets.find(LookaheadKey(slot, shader_address));
		if (target == lookahead_targets.end()) return;
		constexpr uint64_t AlternateUses = 16;
		for (const auto& use: target->second.programs) {
			if (use.plan == nullptr || target->second.uses - use.last_use > AlternateUses) break;
			if (use.user_data_count < user_sgprs.size()) continue;
			const auto prefix = use.user_data_count - user_sgprs.size();
			if (prefix > use.prefix.size()) continue;
			inputs.shader_base     = shader_address;
			inputs.user_data_count = use.user_data_count;
			std::copy_n(use.prefix.begin(), prefix, inputs.user_data.begin());
			std::ranges::copy(user_sgprs,
			                  inputs.user_data.begin() + static_cast<ptrdiff_t>(prefix));
			lookahead->Plan(stream, command, use.plan, inputs);
		}
	}

	static bool SameBacking(uint64_t address, const void* data, uint64_t size, bool mapped) {
		const auto result = Libs::LibKernel::Memory::CompareBacking(address, data, size);
		return mapped ? result == Libs::LibKernel::Memory::BackingCompare::Equal
		              : result == Libs::LibKernel::Memory::BackingCompare::Unmapped;
	}

	ProgramCache(GraphicContext& graphics, ShaderRecompiler::IR::ImageHeapSource* image_heap,
	             uint32_t lookahead_threads)
	    : graphics(graphics), image_heap(image_heap), profile {HostShaderProfile(graphics)} {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
		if (lookahead_threads != 0) {
			lookahead = std::make_unique<ResourceLookahead>(
			    LookaheadMemory {.read        = Libs::LibKernel::Memory::TryReadBacking,
			                     .gpu_dirty   = Libs::LibKernel::Memory::HasGpuDirtyBytes,
			                     .same        = SameBacking,
			                     .mapped_size = Libs::LibKernel::Memory::MappedRangeSize},
			    lookahead_threads, image_heap);
		}
	}
	~ProgramCache() {
		StopPreparation();
		StopDeferral();
		const auto destroy = [&](const std::shared_ptr<DeferredTranslation>& job) {
			if (job == nullptr || job->program == nullptr) return;
			for (const auto& permutation: job->program->permutations) {
				graphics.device.destroyShaderModule(permutation.handle.module, nullptr);
			}
		};
		for (const auto& [key, job]: pending_plans) {
			(void)key;
			destroy(job);
		}
		for (const auto& [key, entry]: programs) {
			(void)key;
			for (const auto& module: entry.pending_modules) {
				destroy(module.job);
			}
		}
		if (preparation != nullptr) {
			for (const auto& [key, slot]: preparation->slots) {
				(void)key;
				if (slot.program == nullptr) continue;
				for (const auto& permutation: slot.program->permutations) {
					graphics.device.destroyShaderModule(permutation.handle.module, nullptr);
				}
			}
		}
		for (const auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				graphics.device.destroyShaderModule(permutation.handle.module, nullptr);
			}
		}
	}

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash> programs;
	ProgramKey                                                  lookup_key;
	GraphicContext&                                             graphics;
	ShaderRecompiler::IR::ImageHeapSource*                      image_heap;
	const ShaderRecompiler::Spirv::Profile                      profile;
	uint64_t                                                    next_shader_id = 0;
	// Receives every compiled compute permutation; null when there is no manifest.
	ProgramManifest* manifest = nullptr;
	// Receives every compiled graphics stage permutation; null when there is no manifest.
	PipelineManifest* pipeline_manifest = nullptr;
	// Whether an imported pipeline list predicts a compute program key (may be empty).
	std::function<bool(uint64_t)> list_predicts;
	// Whether the pipeline of a compute shader's code-only module is in the driver cache (the
	// precompile built it), by shader hash; and the build of a program record on the precompile
	// queue. Null without a driver cache file.
	std::function<bool(uint64_t)>                                        code_only_built;
	// Whether the precompile built pipelines with the code-only module of a graphics shader, by
	// shader hash (the shader pairs); may be empty.
	std::function<bool(uint64_t)> stage_code_only;
	// The created compute shaders whose code-only pipelines the precompile has built.
	std::function<std::vector<std::shared_ptr<const ComputeProgramRecord>>()> code_only_records;
	std::function<void(std::shared_ptr<const ComputeProgramRecord>)>     build_later;
	// Whether the session's precompile has built the pipeline of a recorded program, by record
	// key; may be empty.
	std::function<bool(uint64_t)> built_now;
	// First uses of shaders whose code-only module the precompile built: the dispatch's own
	// module is that one, that one serves the dispatch (and was taken), or neither.
	uint64_t code_only_same  = 0;
	uint64_t code_only_taken = 0;
	uint64_t code_only_other = 0;
	// ... or it serves the dispatch, but a pipeline list names the dispatch's own module.
	uint64_t code_only_listed = 0;
	// Compute permutations recorded this session, and those a record or the list predicted.
	uint64_t compute_recorded  = 0;
	uint64_t compute_predicted = 0;
	// Graphics shaders used this session, per stage, and those the guest had created first
	// (created_code; census of what a list can resolve at creation).
	const ShaderCodeStore*       created_code = nullptr;
	std::unordered_set<uint64_t> census_seen;
	std::array<std::array<uint64_t, 2>, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1>
	    census {};

	// Compute dispatches ahead of the GPU thread: the program last used at each shader address
	// with its registers, and the command the next GetComputeProgram belongs to.
	struct LookaheadUse {
		std::shared_ptr<LookaheadPlan> plan;
		uint32_t                       user_data_count = 0;
		std::array<uint32_t, 8>        prefix {};
		uint64_t                       last_use = 0;
	};
	struct LookaheadTarget {
		std::array<LookaheadUse, 2> programs; // most recent first
		uint64_t                    uses = 0;
	};
	std::unique_ptr<ResourceLookahead>            lookahead;
	std::unordered_map<uint64_t, LookaheadTarget> lookahead_targets;
	const void*                                   lookahead_stream      = nullptr;
	const void*                                   lookahead_command     = nullptr;
	uint32_t                                      next_lookahead_helper = 0;
	std::chrono::steady_clock::time_point         next_lookahead_report {};
	// Programs translated ahead of their first use (StartPreparation); null without.
	std::unique_ptr<Preparation> preparation;
	// Stage programs helper threads translate while their draws are left out (see Deferral):
	// whether the option is on, whether the draw Get is asked about may be left out
	// (GetGraphicsPrograms sets it), whether the last Get left its draw out, the plans under way
	// by program key, what is known of each program's stores, and the counts.
	bool                                                           defer_stage_programs = false;
	bool                                                           deferred             = false;
	std::unique_ptr<Deferral>                                      deferral;
	std::unordered_map<ProgramKey, std::shared_ptr<DeferredTranslation>, ProgramKeyHash>
	                                   pending_plans;
	std::unordered_map<uint64_t, bool> code_may_write;
	uint64_t                           deferred_plans   = 0;
	uint64_t                           deferred_modules = 0;
	// Own modules of dispatches a code-only module served (OwnModuleFor).
	uint64_t own_modules_started = 0;
	uint64_t own_modules_taken   = 0;
	uint64_t                           deferred_failed  = 0;
	// Compiles the lookahead plans' walkers. Last, so its thread stops before the entries holding
	// its jobs are destroyed.
	SrtWalkerCompileQueue walker_queue;
};

PrecompileOutcome PrecompileComputeProgram(GraphicContext& graphics, vk::PipelineCache driver_cache,
                                           const ComputeProgramRecord& record, uint64_t* spirv_hash,
                                           bool code_only, PipelineBuildMode mode,
                                           const std::atomic<bool>*            cancel,
                                           std::shared_ptr<DriverCacheHistory> history,
                                           KeptPipeline*                       keep) {
	const auto cancelled = [cancel] { return cancel != nullptr && cancel->load(); };
	EXIT_IF(mode != PipelineBuildMode::Precompile && mode != PipelineBuildMode::Unoptimized);
	EXIT_IF(record.code == nullptr || record.code->empty());
	// The inputs GetComputeProgram derives from the device.
	auto input               = record.input;
	input.host_subgroup_size = graphics.ComputeHostSubgroupSize();
	input.lds_storage = input.lds_size_dwords * 4u >
	                    graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	// Translation reads the user-data count; the values only feed the resource walk.
	const std::vector<uint32_t> user_data(record.user_data_count, 0u);

	const auto                       profile = HostShaderProfile(graphics);
	ShaderRecompiler::CompileOptions options;
	options.stage              = ShaderType::Compute;
	options.profile            = &profile;
	options.shader_hash        = record.shader_hash;
	options.user_data          = user_data;
	options.dump_ir            = false;
	options.early_dump         = false;
	options.emit_debug_info    = Config::GraphicsDebugDumpEnabled();
	options.dump_label         = "Precompile CS";
	options.input_info.compute = &input;
	options.wave_size          = input.wave_size;

	const auto compile = [&]() -> std::optional<ShaderRecompiler::CompileResult> {
		auto source = ShaderRecompiler::PrepareShaderSource(*record.code, options);
		if (!source.calls.empty() || source.unsupported_call_pc != UINT32_MAX) {
			return std::nullopt;
		}
		auto       translated = ShaderRecompiler::TranslateProgram(source.decoded, options);
		const auto specialization =
		    code_only ? ShaderRecompiler::IR::CodeOnlySpecialization(translated.program)
		              : std::optional(record.specialization);
		// A recompiler change may give the shader other resources than the record was made for.
		if (!specialization.has_value() ||
		    !ShaderRecompiler::IR::SpecializationApplies(translated.program, *specialization)) {
			return std::nullopt;
		}
		return ShaderRecompiler::CompileProgram(std::move(translated), options, *specialization,
		                                        record.push_data_cursor);
	};
	std::optional<ShaderRecompiler::CompileResult> result;
	// A created shader no dispatch has used may never be used: its module stays out of the
	// simplification store, which every session reads whole.
	std::optional<ShaderRecompiler::Spirv::SimplifiedStoreBypass> unstored;
	if (code_only) {
		unstored.emplace();
	}
	// No dispatch has used a created shader, and a recorded program may meet a recompiler that
	// no longer handles its code: either is skipped instead of ending the process.
	try {
		const Common::RecoverableFatalScope recoverable;
		result = compile();
	} catch (const Common::RecoverableFatal& fatal) {
		PipelineCacheLog("Precompile: {} CS {:016x} cannot be translated: {}",
		                 code_only ? "created" : "recorded", record.shader_hash, fatal.message);
		return PrecompileOutcome::Skipped;
	}
	if (!result.has_value() || !ValidateShaderSpirv(options.dump_label, record.shader_hash,
	                                                result->spirv, profile.maintenance8)) {
		return PrecompileOutcome::Skipped;
	}
	const auto module_hash =
	    XXH3_64bits(result->spirv.data(), result->spirv.size() * sizeof(uint32_t));
	if (spirv_hash != nullptr) {
		*spirv_hash = module_hash;
	}
	if (cancelled()) {
		return PrecompileOutcome::TimedOut;
	}
	const auto module = CompileSPV(result->spirv, graphics.device);
	if (module == nullptr) {
		return PrecompileOutcome::Skipped;
	}
	const auto                             program = std::move(result->program).TakeCompiledInfo();
	ShaderRecompiler::IR::ResourceSnapshot resources;
	const ShaderProgram                    handle {.module = module, .spirv_hash = module_hash};
	input.stage = {.program = &program, .resources = &resources, .handle = &handle};

	PipelineCache::Pipeline pipeline;
	bool                    from_driver_cache = false;
	(void)CreatePipelineInternal(graphics, pipeline, input, module, driver_cache, mode,
	                             &from_driver_cache, cancel, std::move(history));
	const bool built = pipeline.pipeline != nullptr;
	if (built && keep != nullptr) {
		*keep = {.pipeline = pipeline.pipeline, .fingerprint = pipeline.fingerprint};
	} else {
		graphics.device.destroyPipeline(pipeline.pipeline, nullptr);
	}
	graphics.device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
	graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout, nullptr);
	graphics.device.destroyShaderModule(module, nullptr);
	if (!built) {
		return PrecompileOutcome::TimedOut;
	}
	return from_driver_cache ? PrecompileOutcome::Cached : PrecompileOutcome::Compiled;
}

// A graphics stage translated for the precompile, shared by the recorded pipelines that use it.
struct PipelineCache::PrecompiledStage {
	std::mutex                                              mutex;
	bool                                                    done   = false;
	bool                                                    failed = false;
	vk::ShaderModule                                        module = nullptr;
	std::optional<ShaderRecompiler::IR::CompiledShaderInfo> program;
	ShaderRecompiler::IR::ResourceSnapshot                  resources;
	ShaderProgram                                           handle;
	// The input the stage was translated with: the record's, with the device's fields.
	ShaderVertexInputInfo vertex;
	ShaderPixelInputInfo  pixel;
};

// Translates a recorded graphics stage as ProgramCache::Get would. Returns false when the stage
// cannot be rebuilt from its record (calls, a specialization this recompiler no longer derives,
// or code it cannot translate).
bool PipelineCache::TranslateRecordedStage(GraphicContext&            graphics,
                                           const GraphicsStageRecord& record,
                                           PrecompiledStage&          out) {
	static std::atomic<uint64_t>     next_id {0};
	const auto                       profile = HostShaderProfile(graphics);
	ShaderRecompiler::CompileOptions options;
	options.profile         = &profile;
	options.shader_hash     = record.shader_hash;
	options.dump_ir         = false;
	options.early_dump      = false;
	options.emit_debug_info = Config::GraphicsDebugDumpEnabled();
	const std::vector<uint32_t> user_data(record.user_data_count, 0u);
	options.user_data = user_data;
	if (record.back_code != nullptr) {
		options.back_code = *record.back_code;
	}
	options.stage = record.stage;
	if (record.stage == ShaderType::Pixel) {
		out.pixel                = record.pixel;
		options.input_info.pixel = &out.pixel;
		options.wave_size        = out.pixel.wave_size;
		options.dump_label       = "Precompile PS";
	} else {
		out.vertex = record.vertex;
		// The fields GetGraphicsPrograms derives from the device.
		if (out.vertex.mesh.threads_num[0] != 0) {
			out.vertex.mesh.host_subgroup_size = graphics.subgroup_size;
		}
		if (out.vertex.clip_space.enabled) {
			const auto& limits = graphics.GetPhysicalDeviceProperties().limits;
			out.vertex.clip_space.half_extent[0] =
			    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
			out.vertex.clip_space.half_extent[1] =
			    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		}
		options.input_info.vertex = &out.vertex;
		options.wave_size         = out.vertex.wave_size;
		options.user_data_base    = 8;
		if (record.stage == ShaderType::Mesh || record.stage == ShaderType::TessellationControl) {
			options.user_data_base = 0;
			options.wave_size = record.stage == ShaderType::Mesh ? out.vertex.mesh.wave_size : 64u;
		}
		options.dump_label = "Precompile VS";
	}
	std::optional<ShaderRecompiler::CompileResult> result;
	// No draw has used the shader of a code-only stage, and it may never be used: its module
	// stays out of the simplification store, which every session reads whole.
	std::optional<ShaderRecompiler::Spirv::SimplifiedStoreBypass> unstored;
	if (record.code_only) {
		unstored.emplace();
	}
	try {
		const Common::RecoverableFatalScope recoverable;
		auto source = ShaderRecompiler::PrepareShaderSource(*record.code, options);
		if (!source.calls.empty() || source.unsupported_call_pc != UINT32_MAX) {
			return false;
		}
		auto translated     = ShaderRecompiler::TranslateProgram(source.decoded, options);
		auto specialization = record.specialization;
		if (record.code_only) {
			auto from_code = ShaderRecompiler::IR::CodeOnlySpecialization(translated.program);
			if (!from_code.has_value()) return false;
			specialization = std::move(*from_code);
		}
		if (!ShaderRecompiler::IR::SpecializationApplies(translated.program, specialization)) {
			return false;
		}
		result = ShaderRecompiler::CompileProgram(std::move(translated), options, specialization,
		                                          record.push_data_cursor);
	} catch (const Common::RecoverableFatal& fatal) {
		if (!record.code_only) {
			PipelineCacheLog("Precompile: recorded stage {:016x} cannot be translated: {}",
			                 record.shader_hash, fatal.message);
		}
		return false;
	}
	if (!ValidateShaderSpirv(options.dump_label, record.shader_hash, result->spirv,
	                         profile.maintenance8)) {
		return false;
	}
	out.module = CompileSPV(result->spirv, graphics.device);
	if (out.module == nullptr) {
		return false;
	}
	out.program.emplace(std::move(result->program).TakeCompiledInfo());
	out.handle = {.id     = ++next_id,
	              .module = out.module,
	              .spirv_hash =
	                  XXH3_64bits(result->spirv.data(), result->spirv.size() * sizeof(uint32_t))};
	return true;
}

std::shared_ptr<PipelineCache::PrecompiledStage>
PipelineCache::PrecompiledStageOf(const GraphicsStageRecord& stage) {
	std::shared_ptr<PrecompiledStage> memo;
	{
		std::lock_guard lock(m_stage_memo_mutex);
		auto&           entry = m_stage_memo[stage.Key()];
		if (entry == nullptr) entry = std::make_shared<PrecompiledStage>();
		memo = entry;
	}
	std::lock_guard lock(memo->mutex);
	if (!memo->done) {
		memo->done   = true;
		memo->failed = !TranslateRecordedStage(m_graphics, stage, *memo);
	}
	return memo->failed ? nullptr : memo;
}

// A graphics pipeline for a vertex and pixel shader no session has drawn with, patterned on a
// pipeline one has. From the pattern: the render state, the vertex input, and of the two stage
// inputs what belongs to the mesh and the pass (vertex buffers, targets, blending). From the
// pair: its code, its user data counts, and what a draw's registers hold of the two shaders'
// own headers (ShaderDeriveVertexInput, ShaderDerivePixelInput), with the stage modules the
// code gives by itself (GraphicsStageRecord::code_only). The stages are translated as a draw
// translates them: the pixel stage first, whose shader data moves the push-data cursor the
// vertex stage starts at. What comes out is the pipeline a first draw of the pair creates when
// the pair is drawn in the pattern's pass with a mesh like the pattern's; a pair drawn another
// way has only cost the precompile time. Skipped: a pair whose headers do not fit the pattern
// (a pixel shader of another pass, a merged vertex shader for a plain pattern or the reverse).
PrecompileOutcome PipelineCache::PrecompileShaderPair(const PipelinePattern& pattern,
                                                      PairShader vertex_shader, PairShader pixel_shader,
                                                      PipelineBuildMode        mode,
                                                      const std::atomic<bool>* cancel,
                                                      KeptPipeline*            keep) {
	const auto stage_of = [](const GraphicsStageRecord& like, const PairShader& shader,
	                         uint32_t cursor) {
		auto stage = std::make_shared<GraphicsStageRecord>(like);
		stage->SetCode(shader.code, nullptr);
		stage->shader_hash      = stage->code_id.hash;
		stage->user_data_count  = shader.user_data_count;
		stage->specialization   = {};
		stage->spirv_hash       = 0;
		stage->push_data_cursor = cursor;
		stage->code_only        = true;
		return stage;
	};
	const auto& like_vertex = *pattern.stages.front();
	const bool  has_pixel   = pattern.record->pixel_stage != 0;
	const auto  start = (has_pixel ? pattern.stages.back() : pattern.stages.front())->push_data_cursor;
	// What the vertex stage's input has from its own shader (the pattern's vertex shader need
	// only take the same vertex inputs). A merged shader, which runs as a mesh stage, keeps the
	// pattern's input whole: the pairs file names one only with a pattern whose shader has the
	// same registers.
	const auto vertex_input = [&](const ShaderHeaderFacts& facts, ShaderVertexInputInfo& info) {
		if (like_vertex.stage == ShaderType::Vertex) {
			return ShaderDeriveVertexInput(facts, info);
		}
		return like_vertex.stage == ShaderType::Mesh && facts.shader_stages_en.has_value() &&
		       (*facts.shader_stages_en & 0x20u) != 0;
	};
	const auto build = [&](const ShaderHeaderFacts&             vertex_facts,
	                       std::shared_ptr<GraphicsStageRecord> pixel) {
		auto cursor = start;
		if (pixel != nullptr) {
			const auto made = PrecompiledStageOf(*pixel);
			if (made == nullptr) {
				return PrecompileOutcome::Skipped;
			}
			made->program->bindings.AdvancePushData(cursor);
		}
		std::vector<std::shared_ptr<const GraphicsStageRecord>> stages;
		const auto vertex = stage_of(like_vertex, vertex_shader, cursor);
		if (!vertex_input(vertex_facts, vertex->vertex)) {
			return PrecompileOutcome::Skipped;
		}
		stages.push_back(vertex);
		if (pixel != nullptr) {
			stages.push_back(pixel);
		}
		auto record               = *pattern.record;
		record.vertex_stages      = {vertex->Key(), 0, 0};
		record.vertex_stage_count = 1;
		record.pixel_stage        = pixel != nullptr ? pixel->Key() : 0;
		return PrecompileGraphicsPipeline(record, stages, mode, cancel, keep);
	};
	// A program that ships with several headers gets a pipeline for each that fits the pattern
	// and gives other stage inputs. The pixel stage's input has from its own shader, and from
	// the vertex shader it is drawn with, what the pattern's pixel shader need not share: that
	// one need only be of the same pass.
	static const ShaderFactsList NoPixel {nullptr};
	auto                         outcome = PrecompileOutcome::Skipped;
	std::unordered_set<uint64_t> built;
	for (const auto& pixel_facts: has_pixel ? pixel_shader.facts : NoPixel) {
		for (const auto& vertex_facts: vertex_shader.facts) {
			auto probe = like_vertex.vertex;
			if (!vertex_input(*vertex_facts, probe)) {
				continue;
			}
			std::shared_ptr<GraphicsStageRecord> pixel;
			if (has_pixel) {
				pixel = stage_of(*pattern.stages.back(), pixel_shader, start);
				if (!ShaderDerivePixelInput(*pixel_facts, *vertex_facts, pixel->pixel)) {
					continue;
				}
			}
			const auto variant = (pixel != nullptr ? pixel->Key() : 0) ^
			                     (uint64_t {probe.pa_cl_vs_out_cntl} << 32u) ^
			                     (uint64_t {probe.wave_size} << 20u);
			if (!built.insert(variant).second) {
				continue;
			}
			// The outcome of the pair is that of its pipelines: compiled when any was.
			const auto one = build(*vertex_facts, std::move(pixel));
			if (one == PrecompileOutcome::TimedOut) {
				return one;
			}
			if (one == PrecompileOutcome::Compiled ||
			    (one == PrecompileOutcome::Cached && outcome == PrecompileOutcome::Skipped)) {
				outcome = one;
			}
		}
	}
	return outcome;
}

PrecompileOutcome PipelineCache::PrecompileGraphicsPipeline(
    const GraphicsPipelineRecord&                               record,
    std::span<const std::shared_ptr<const GraphicsStageRecord>> stages, PipelineBuildMode mode,
    const std::atomic<bool>* cancel, KeptPipeline* keep) {
	EXIT_IF(stages.size() != record.vertex_stage_count + (record.pixel_stage != 0 ? 1u : 0u));
	const auto cancelled = [cancel] { return cancel != nullptr && cancel->load(); };
	// Each stage once, however many pipelines use it.
	std::vector<std::shared_ptr<PrecompiledStage>> translated;
	for (const auto& stage: stages) {
		if (cancelled()) {
			return PrecompileOutcome::TimedOut;
		}
		auto memo = PrecompiledStageOf(*stage);
		if (memo == nullptr) {
			return PrecompileOutcome::Skipped;
		}
		translated.push_back(std::move(memo));
	}
	DrawRenderState state;
	for (uint32_t i = 0; i < record.vertex_stage_count; i++) {
		const auto& stage          = *translated[i];
		state.vertex_info[i]       = stage.vertex;
		state.vertex_info[i].stage = {
		    .program = &*stage.program, .resources = &stage.resources, .handle = &stage.handle};
	}
	state.ps_active = record.pixel_stage != 0;
	if (state.ps_active) {
		const auto& stage         = *translated.back();
		state.ps_input_info       = stage.pixel;
		state.ps_input_info.stage = {
		    .program = &*stage.program, .resources = &stage.resources, .handle = &stage.handle};
	}
	state.topology                 = record.static_params.topology;
	state.primitive_restart_enable = record.static_params.primitive_restart_enable;
	if (cancelled()) {
		return PrecompileOutcome::TimedOut;
	}
	// An abandoned build may return from the driver after the cache is gone: nothing below the
	// driver call reads members.
	const auto device = m_graphics.device;
	Pipeline   pipeline;
	bool       from_driver_cache = false;
	try {
		const Common::RecoverableFatalScope recoverable;
		(void)CreatePipelineInternal(m_graphics, pipeline, record.rendering, record.vertex_input,
		                             state, record.static_params, m_driver_cache, mode,
		                             &from_driver_cache, cancel, m_driver_history);
	} catch (const Common::RecoverableFatal& fatal) {
		PipelineCacheLog("Precompile: recorded pipeline {:016x} cannot be built: {}", record.Key(),
		                 fatal.message);
		return PrecompileOutcome::Skipped;
	}
	const bool built = pipeline.pipeline != nullptr;
	if (built && keep != nullptr) {
		*keep = {.pipeline = pipeline.pipeline, .fingerprint = pipeline.fingerprint};
	} else {
		device.destroyPipeline(pipeline.pipeline, nullptr);
	}
	device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
	device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout, nullptr);
	if (!built) {
		return PrecompileOutcome::TimedOut;
	}
	return from_driver_cache ? PrecompileOutcome::Cached : PrecompileOutcome::Compiled;
}

void PipelineCache::ReleasePrecompiledStages() {
	std::lock_guard lock(m_stage_memo_mutex);
	// An abandoned build may still be creating a pipeline from one of the modules: they are
	// left to the process then.
	const bool in_use = m_precompile_runs->abandoned.load() != 0;
	for (const auto& [key, stage]: m_stage_memo) {
		(void)key;
		if (stage->module != nullptr && !in_use) {
			m_graphics.device.destroyShaderModule(stage->module, nullptr);
		}
	}
	m_stage_memo.clear();
}

// The code-only builds of created compute shaders that have ended (their pipelines are in the
// driver cache, or their shaders cannot be translated) with this emulator build and this driver
// cache: later sessions do not queue them again. Without it every session would translate
// every created shader no dispatch has used just to find its pipeline in the driver cache,
// which is all the game's compute shaders once --import-shaders has brought them in (Saros:
// 11,600). A text file: the signature line, then a key per line, appended as builds end. A file
// of another signature, or a driver cache that starts empty, starts it over. Thread-safe.
class PipelineCache::FinishedBuilds {
public:
	FinishedBuilds(std::filesystem::path path, std::string signature, bool keep)
	    : m_path(std::move(path)), m_line("KytyFinishedBuilds2 " + std::move(signature)) {
		// The header is one line; the driver cache signature ends with a line break of its own,
		// which made the header two lines that never compared equal to the first line read.
		std::erase_if(m_line, [](char c) { return c == '\n' || c == '\r'; });
		if (keep) {
			std::ifstream in(m_path);
			std::string   text;
			if (std::getline(in, text) && text == m_line) {
				while (std::getline(in, text)) {
					// "b <key>": built; "s <key>": the shader does not translate.
					if (text.size() < 3 || (text[0] != 'b' && text[0] != 's') || text[1] != ' ') {
						continue;
					}
					char*      end = nullptr;
					const auto key = std::strtoull(text.c_str() + 2, &end, 16);
					if (end != text.c_str() + 2) {
						m_keys[key] = text[0] == 'b';
					}
				}
			}
		}
		Rewrite();
	}
	KYTY_CLASS_NO_COPY(FinishedBuilds);

	[[nodiscard]] bool Contains(uint64_t key) const {
		std::lock_guard lock(m_mutex);
		return m_keys.contains(key);
	}
	// The key's pipeline is in the driver cache.
	[[nodiscard]] bool Built(uint64_t key) const {
		std::lock_guard lock(m_mutex);
		const auto      entry = m_keys.find(key);
		return entry != m_keys.end() && entry->second;
	}
	[[nodiscard]] size_t Size() const {
		std::lock_guard lock(m_mutex);
		return m_keys.size();
	}
	// A build has ended. It is not queued again in this process; the file gets it with Commit.
	void Add(uint64_t key, bool built) {
		std::lock_guard lock(m_mutex);
		if (m_keys.try_emplace(key, built).second) {
			m_pending.emplace_back(key, built);
		}
	}
	using Ended = std::vector<std::pair<uint64_t, bool>>;
	// The builds that ended since the last call. They go to the file (Commit) once a driver
	// cache file that holds their pipelines is written, and back (Restore) when it could not
	// be: the file must never name a pipeline that the cache file of a pass that was cut off
	// does not have, or no later pass would build it.
	[[nodiscard]] Ended TakePending() {
		std::lock_guard lock(m_mutex);
		return std::exchange(m_pending, {});
	}
	void Commit(const Ended& ended) {
		std::lock_guard lock(m_mutex);
		if (ended.empty() || !m_file.is_open()) {
			return;
		}
		for (const auto& [key, built]: ended) {
			m_file << fmt::format("{} {:016x}\n", built ? 'b' : 's', key);
		}
		m_file.flush();
	}
	void Restore(Ended ended) {
		std::lock_guard lock(m_mutex);
		ended.insert(ended.end(), m_pending.begin(), m_pending.end());
		m_pending = std::move(ended);
	}
	// The driver cache starts over: so does this.
	void Clear() {
		std::lock_guard lock(m_mutex);
		m_keys.clear();
		m_pending.clear();
		Rewrite();
	}

private:
	void Rewrite() {
		m_file.close();
		m_file.open(m_path, std::ios::trunc);
		m_file << m_line << '\n';
		for (const auto& [key, built]: m_keys) {
			m_file << fmt::format("{} {:016x}\n", built ? 'b' : 's', key);
		}
		m_file.flush();
	}

	std::filesystem::path        m_path;
	std::string                  m_line;
	mutable std::mutex                 m_mutex;
	std::unordered_map<uint64_t, bool> m_keys;
	// Ended and not in the file yet.
	Ended                              m_pending;
	std::ofstream                      m_file;
};

PipelineCache::PipelineCache(GraphicContext&                         graphics,
                             ShaderRecompiler::IR::ImageHeapSource* image_heap,
                             CommandScheduler*                       scheduler)
    : m_graphics(graphics), m_scheduler(scheduler),
      // The lookahead plans from the command streams, which run with a scheduler.
      m_program_cache(std::make_unique<ProgramCache>(
          graphics, image_heap,
          scheduler != nullptr ? std::min(Config::LookaheadThreads(), 8u) : 0u)),
      m_next_driver_cache_save(std::chrono::steady_clock::now() + DriverCacheSaveInterval),
      m_warm_pipelines(graphics.device) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	ShaderRecompiler::Spirv::SetMaximalReconvergence(graphics.maximal_reconvergence_enabled);
	InitializeDriverCache();
	// Both modes replace or install pipeline handles while a command buffer is recording, which
	// needs the scheduler.
	m_defer_optimization = m_scheduler != nullptr && Config::AsyncPipelinesEnabled();
	m_skip_pending_draws = m_scheduler != nullptr && Config::SkipPendingDrawsEnabled();
	if (m_defer_optimization || m_skip_pending_draws) {
		// Driver compilers are multithreaded themselves; two builds at a time keep the CPU
		// available for the guest.
		const auto threads = std::clamp(std::thread::hardware_concurrency() / 8u, 1u, 2u);
		m_compile_queue    = std::make_unique<PipelineCompileQueue>(
		    threads, [device = graphics.device](uint64_t handle) {
			    device.destroyPipeline(PipelineFromCompileHandle(handle), nullptr);
		    });
	}
}

PipelineCache::~PipelineCache() {
	if (m_created != nullptr) {
		ShaderSetCreatedComputeSink({});
	}
	if (m_created_code != nullptr) {
		ShaderSetCreatedCodeSink({});
	}
	StopBackgroundCompiles();
	(void)Save();
	ShaderRecompiler::Spirv::SetSimplifiedStore(nullptr);
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	}
}

void PipelineCache::StartProgramPreparation(uint32_t thread_count) {
	m_program_cache->StartPreparation(thread_count);
}

PipelineCache::ProgramPreparationCounts PipelineCache::GetProgramPreparationCounts() const {
	if (m_program_cache->preparation == nullptr) return {};
	std::lock_guard lock(m_program_cache->preparation->mutex);
	return m_program_cache->preparation->counts;
}

void PipelineCache::WaitForProgramPreparation() {
	m_program_cache->WaitForPreparation();
}

void PipelineCache::StopBackgroundCompiles() {
	// Translations use the SPIR-V simplification store, which the destructor detaches.
	m_program_cache->StopPreparation();
	m_program_cache->StopDeferral();
	if (m_precompile_queue != nullptr) {
		{
			// Created shaders stop queueing list entries.
			std::lock_guard lock(m_list_mutex);
			if (m_list_open) {
				m_list_open = false;
				(void)m_precompile_progress.EndQueueing();
			}
		}
		{
			// Running builds stop before their next driver call.
			std::lock_guard lock(m_precompile_runs->mutex);
			for (const auto& [key, running]: m_precompile_runs->running) {
				(void)key;
				running.cancel->store(true);
			}
		}
		const auto remaining = m_precompile_queue->Remaining();
		m_precompile_queue->Stop();
		if (remaining != 0) {
			LogPrecompileStats("stopped");
		}
		m_precompile_queue.reset();
		ReleasePrecompiledStages();
	}
	if (m_compile_queue == nullptr) {
		return;
	}
	if (!m_pending_optimized.empty()) {
		LOGF("PipelineCache: dropping %zu pending pipeline builds\n", m_pending_optimized.size());
	}
	// Results that were not installed are destroyed here; their pipelines keep the unoptimized
	// handles, or none if they were never built.
	m_compile_queue->Stop();
	m_pending_optimized.clear();
	m_pending_builds.clear();
	m_skip_pending_draws = false;
}

void PipelineCache::DeferOptimization(Pipeline& pipeline, PipelineCompileQueue::Job job) {
	if (!job) {
		return;
	}
	const auto ticket = m_compile_queue->Enqueue(std::move(job));
	m_pending_optimized.emplace(ticket, &pipeline);
}

// Replaces unoptimized pipelines whose optimized builds finished. Runs on the GPU thread while
// a command buffer is being recorded: command buffers recorded so far may have bound the old
// handle, so it is destroyed once the current command buffer has completed.
void PipelineCache::InstallOptimizedPipelines() {
	// What a finished precompile run compiled is written now, not at the next save interval:
	// a session that ends before it would compile it all again.
	if (m_precompile_save_due.load(std::memory_order_relaxed) &&
	    m_precompile_save_due.exchange(false, std::memory_order_relaxed) &&
	    m_driver_cache_dirty.load(std::memory_order_relaxed)) {
		m_next_driver_cache_save = std::chrono::steady_clock::now();
		m_precompile_dirty.store(true, std::memory_order_relaxed);
	}
	if (m_precompile_dirty.load(std::memory_order_relaxed) &&
	    m_precompile_dirty.exchange(false, std::memory_order_relaxed)) {
		CheckpointDriverCache();
	}
	const auto now = std::chrono::steady_clock::now();
	if (m_precompile_queue != nullptr && now >= m_next_timeout_check) {
		m_next_timeout_check = now + std::chrono::seconds {5};
		CheckPrecompileTimeouts();
		UpdatePrecompileThreads();
		ReleaseWarmPipelinesUnderPressure();
	}
	// Sessions usually end without a save: the coverage is also logged once a minute.
	if (m_pipeline_manifest != nullptr && now >= m_next_coverage_report) {
		if (m_next_coverage_report != std::chrono::steady_clock::time_point {}) {
			LogCoverage();
			LogStaleDriverCache();
			PersistStaleMisses();
		}
		m_next_coverage_report = now + std::chrono::minutes {1};
	}
	if (m_pending_optimized.empty() || !m_scheduler->Active()) {
		return;
	}
	m_completed.clear();
	m_compile_queue->TakeCompleted(m_completed);
	for (const auto& completion: m_completed) {
		const auto entry = m_pending_optimized.find(completion.ticket);
		EXIT_IF(entry == m_pending_optimized.end());
		auto& pipeline = *entry->second;
		m_pending_optimized.erase(entry);
		if (pipeline.pipeline == nullptr) {
			// A first build: its draws were skipped until now, nothing has bound a handle.
			const auto pending = m_pending_builds.find(&pipeline);
			EXIT_IF(pending == m_pending_builds.end());
			if (completion.handle == 0) {
				EXIT("PipelineCache: background pipeline build failed (VS %016" PRIx64
				     " PS %016" PRIx64 ")\n",
				     pending->second.vs_hash, pending->second.ps_hash);
			}
			pipeline.pipeline = PipelineFromCompileHandle(completion.handle);
			const auto ms     = std::chrono::duration<double, std::milli>(
			                        std::chrono::steady_clock::now() - pending->second.queued)
			                        .count();
			LOGF("PipelineCache: pipeline of VS %016" PRIx64 " PS %016" PRIx64
			     " is ready after %.0f ms; %" PRIu64 " draws were skipped\n",
			     pending->second.vs_hash, pending->second.ps_hash, ms,
			     pending->second.skipped_draws);
			m_pending_builds.erase(pending);
			continue;
		}
		if (completion.handle == 0) {
			LOGF("PipelineCache: optimized build failed; keeping unoptimized pipeline %p\n",
			     static_cast<void*>(pipeline.pipeline));
			continue;
		}
		const auto unoptimized = pipeline.pipeline;
		pipeline.pipeline      = PipelineFromCompileHandle(completion.handle);
		m_scheduler->DeferOperation([device = m_graphics.device, unoptimized] {
			device.destroyPipeline(unoptimized, nullptr);
		});
	}
	if (!m_completed.empty()) {
		CheckpointDriverCache();
	}
}

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}
	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	if (Config::ShaderSimplificationDisabled()) {
		// Nothing is simplified: the stored simplifications are neither read nor replaced.
		PipelineCacheLog("SPIR-V simplifications: off (--no-shader-simplification)");
	} else {
		// A build's simplifier may differ from another's: the file of another build is replaced.
		m_simplified_store = std::make_unique<ShaderRecompiler::Spirv::SimplifiedStore>(
		    std::filesystem::path("_PipelineCache") / (title_id + ".v1.simplified"),
		    KYTY_GIT_VERSION);
		ShaderRecompiler::Spirv::SetSimplifiedStore(m_simplified_store.get());
		PipelineCacheLog("SPIR-V simplifications: {} stored", m_simplified_store->Loaded());
	}
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	const auto           signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	std::vector<uint8_t> initial_data;
	uint64_t             initial_hash = 0;
	std::error_code      size_error;
	const auto           file_size =
	    cache_exists ? std::filesystem::file_size(m_driver_cache_path, size_error) : uint64_t {0};
	if (cache_exists && Config::PrecompileOnlyEnabled() &&
	    DriverCacheFile::RebuildOnPrecompile(Config::RebuildPipelineCacheEnabled(),
	                                         size_error ? 0 : file_size)) {
		// Only the pipelines this run builds end up in the file; it replaces the old one when
		// the precompile has finished.
		m_driver_cache_rebuild = true;
		PipelineCacheLog(
		    "Vulkan pipeline cache: rebuilding {} ({:.2f} MB) from the pipelines this precompile "
		    "builds",
		    path, static_cast<double>(size_error ? 0 : file_size) / 1'000'000.0);
	} else if (cache_exists) {
		auto loaded = DriverCacheFile::Load(m_driver_cache_path, signature);
		switch (loaded.status) {
			case DriverCacheFile::LoadStatus::Loaded:
				initial_data = std::move(loaded.payload);
				initial_hash = loaded.payload_hash;
				break;
			case DriverCacheFile::LoadStatus::SignatureMismatch:
			case DriverCacheFile::LoadStatus::DataMismatch:
				PipelineCacheLog("Vulkan pipeline cache: invalidating {} (driver or data mismatch)",
				                 path);
				break;
			case DriverCacheFile::LoadStatus::InvalidSize:
				PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)",
				                 path);
				break;
			case DriverCacheFile::LoadStatus::Missing:
			case DriverCacheFile::LoadStatus::ReadFailed:
				PipelineCacheLog("Vulkan pipeline cache: invalidating {} (read failed)", path);
				break;
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	m_driver_cache_stored_size.store(initial_data.size(), std::memory_order_relaxed);
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {:.2f} MB from {}",
		                 static_cast<double>(initial_data.size()) / 1'000'000.0, path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}

	const auto directory  = m_driver_cache_path.parent_path();
	m_driver_history      = std::make_shared<DriverCacheHistory>();
	m_driver_history_path = DriverCacheHistory::PathFor(directory, title_id);
	if (!initial_data.empty()) {
		if (m_driver_history->Load(m_driver_history_path,
		                           DriverCacheHistory::Signature(signature, initial_hash))) {
			PipelineCacheLog("Vulkan pipeline cache: {} pipelines were given to it ({})",
			                 m_driver_history->Size(), Common::PathToString(m_driver_history_path));
			LogStaleDriverCache();
		}
		std::lock_guard lock(m_history_file_mutex);
		m_history_file_built      = m_driver_history->Snapshot();
		m_history_file_cache_hash = initial_hash;
		m_history_file_stale      = m_driver_history->StaleCount();
	}
	SetManifest(
	    std::make_unique<ProgramManifest>(ProgramManifest::PathFor(directory, title_id),
	                                      ProgramManifest::PreviousPathFor(directory, title_id)));
	const auto loaded = m_manifest->Loaded();
	PipelineCacheLog("Program manifest: {} compute programs in {}{}", loaded.records,
	                 Common::PathToString(m_manifest->Path()),
	                 loaded.rewritten ? fmt::format(" (rewritten, {} unreadable bytes dropped)",
	                                                loaded.dropped_bytes)
	                                  : std::string {});
	SetCreatedShaders(
	    std::make_unique<ProgramManifest>(ProgramManifest::CreatedPathFor(directory, title_id)));
	m_precompile_attempts = std::make_unique<PrecompileAttempts>(
	    PrecompileAttempts::PathFor(directory, title_id),
	    PrecompileAttempts::PreviousPathFor(directory, title_id));
	m_finished_created = std::make_unique<FinishedBuilds>(
	    directory / (title_id + ".v1.finished"), signature + ":" + KYTY_GIT_VERSION,
	    !initial_data.empty());
	m_program_cache->stage_code_only = [this](uint64_t shader_hash) {
		std::lock_guard lock(m_list_mutex);
		return m_pair_shaders.contains(shader_hash);
	};
	m_program_cache->code_only_built = [this](uint64_t shader_hash) {
		std::lock_guard lock(m_list_mutex);
		if (m_created_keys.empty() && m_created != nullptr) {
			// Once, at the first shader no session has used: the created shaders by hash.
			for (const auto& record: m_created->Records()) {
				m_created_keys.try_emplace(record->shader_hash, record->Key());
			}
		}
		const auto entry = m_created_keys.find(shader_hash);
		return entry != m_created_keys.end() && m_finished_created->Built(entry->second);
	};
	m_program_cache->code_only_records = [this] {
		std::vector<std::shared_ptr<const ComputeProgramRecord>> built;
		std::lock_guard                                          lock(m_list_mutex);
		if (m_created == nullptr || m_finished_created == nullptr) {
			return built;
		}
		for (auto& record: m_created->Records()) {
			if (m_finished_created->Built(record->Key())) {
				built.push_back(std::move(record));
			}
		}
		return built;
	};
	m_program_cache->built_now = [this](uint64_t key) {
		std::lock_guard lock(m_built_keys_mutex);
		return m_built_keys.contains(key);
	};
	m_program_cache->build_later = [this](std::shared_ptr<const ComputeProgramRecord> record) {
		std::lock_guard lock(m_list_mutex);
		if (m_precompile_queue == nullptr || m_precompile_only) {
			return;
		}
		// Part of the boot's precompile run while that is open, else a run of its own.
		if (m_precompile_progress.JoinQueueing()) {
			m_precompile_late.store(true, std::memory_order_relaxed);
		}
		EnqueuePrecompile(std::move(record), false, 0);
		if (m_precompile_progress.EndQueueing()) {
			PrecompileRunEnded();
		}
	};
	m_title_id = title_id;
	SetPipelineManifest(
	    std::make_unique<PipelineManifest>(PipelineManifest::PathFor(directory, title_id),
	                                       PipelineManifest::PreviousPathFor(directory, title_id)));
	m_program_cache->list_predicts = [this](uint64_t key) {
		std::lock_guard lock(m_list_mutex);
		return m_list_keys.contains(key);
	};
	const auto pipelines = m_pipeline_manifest->Loaded();
	PipelineCacheLog("Pipeline manifest: {} graphics pipelines, {} stages in {}{}{}",
	                 pipelines.pipelines, pipelines.stages,
	                 Common::PathToString(m_pipeline_manifest->Path()),
	                 pipelines.imported ? " (imported from the v6 file)" : "",
	                 pipelines.rewritten ? fmt::format(" (rewritten, {} unreadable bytes dropped)",
	                                                   pipelines.dropped_bytes)
	                                     : std::string {});
	SetCreatedCode(
	    std::make_unique<ShaderCodeStore>(ShaderCodeStore::PathFor(directory, title_id)));
	PipelineCacheLog("Created shader code: {} shaders in {}", m_created_code->Size(),
	                 Common::PathToString(ShaderCodeStore::PathFor(directory, title_id)));
	if (m_precompile_attempts->FailedCount() + m_precompile_attempts->RetriedCount() != 0) {
		PipelineCacheLog("Precompile attempts: {} builds quarantined, {} to be retried once",
		                 m_precompile_attempts->FailedCount(),
		                 m_precompile_attempts->RetriedCount());
	}
	const auto created = m_created->Loaded();
	PipelineCacheLog("Created compute shaders: {} in {}{}", created.records,
	                 Common::PathToString(m_created->Path()),
	                 created.rewritten ? fmt::format(" (rewritten, {} unreadable bytes dropped)",
	                                                 created.dropped_bytes)
	                                   : std::string {});
}

void PipelineCache::SetManifest(std::unique_ptr<ProgramManifest> manifest) {
	m_manifest                = std::move(manifest);
	m_program_cache->manifest = m_manifest.get();
}

void PipelineCache::SetCreatedShaders(std::unique_ptr<ProgramManifest> created) {
	ShaderSetCreatedComputeSink({});
	m_created = std::move(created);
	if (m_created == nullptr) {
		return;
	}
	// Runs on the creating guest thread; the manifest serializes its own state.
	ShaderSetCreatedComputeSink([created = m_created.get()](const ShaderCreatedCompute& shader) {
		ComputeProgramRecord record;
		record.shader_hash = shader.hash;
		record.code =
		    std::make_shared<const std::vector<uint32_t>>(shader.code.begin(), shader.code.end());
		record.user_data_count = shader.user_data_count;
		record.input           = ComputeProgramRecord::StaticInput(shader.input);
		(void)created->Add(record);
	});
}

namespace {

// A precompile build that has not finished after this long is abandoned: the driver may never
// finish it (seen for optimized builds of some large compute modules on NVIDIA).
constexpr auto PrecompileBuildBudget = std::chrono::seconds {300};
// Precompile order: recorded and listed programs and pipelines by their FirstSeenPosition (below
// 2^32), then the created shaders without a record by theirs.
constexpr uint64_t PrecompileCreatedOrder = uint64_t {1} << 32u;
// The precompile keeps its boot threads while recorded or listed builds were queued this
// recently: the guest creates its shaders during the boot, and the list resolves as it does.
constexpr auto PrecompileBootQuiet = std::chrono::seconds {120};
// The game renders steadily once this many 10 s windows in a row had this many guest flips.
constexpr uint32_t PrecompileRenderingWindows = 12;
constexpr uint64_t PrecompileRenderingFlips   = 10;

// The attempts key of a build's unoptimized replacement.
uint64_t UnoptimizedKey(uint64_t key) {
	return XXH3_64bits_withSeed(&key, sizeof(key), 0x756e6f7074ull);
}

} // namespace

void PipelineCache::SetPipelineManifest(std::unique_ptr<PipelineManifest> manifest) {
	m_pipeline_manifest                = std::move(manifest);
	m_program_cache->pipeline_manifest = m_pipeline_manifest.get();
}

void PipelineCache::SetCreatedCode(std::unique_ptr<ShaderCodeStore> store) {
	ShaderSetCreatedCodeSink({});
	m_created_code                = std::move(store);
	m_program_cache->created_code = m_created_code.get();
	if (m_created_code == nullptr) {
		return;
	}
	// Runs on the creating guest thread; the store serializes its own state.
	ShaderSetCreatedCodeSink([this](const ShaderCreatedCode& shader) {
		static constexpr std::array<const char*, 9> Names = {"CS",  "PS",  "GS",  "HS", "GSF",
		                                                     "HSF", "GSB", "HSB", "FS"};
		// A graphics shader's header, for the pairs (PrecompileShaderPair).
		if (static_cast<Prospero::ShaderBinaryType>(shader.binary_type) !=
		        Prospero::ShaderBinaryType::kCs &&
		    (!shader.cx_registers.empty() || !shader.input_semantics.empty() ||
		     !shader.output_semantics.empty())) {
			auto facts = std::make_shared<ShaderHeaderFacts>();
			facts->cx_registers.assign(shader.cx_registers.begin(), shader.cx_registers.end());
			facts->sh_registers.assign(shader.sh_registers.begin(), shader.sh_registers.end());
			facts->input_semantics.assign(shader.input_semantics.begin(),
			                              shader.input_semantics.end());
			facts->output_semantics.assign(shader.output_semantics.begin(),
			                               shader.output_semantics.end());
			facts->shader_stages_en = shader.shader_stages_en;
			const auto same = [](const auto& a, const auto& b) {
				return a.size() == b.size() &&
				       (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) == 0);
			};
			std::lock_guard lock(m_shader_facts_mutex);
			auto&           known = m_shader_facts[shader.hash];
			if (std::ranges::none_of(known, [&](const auto& other) {
				    return same(other->cx_registers, facts->cx_registers) &&
				           same(other->sh_registers, facts->sh_registers) &&
				           same(other->input_semantics, facts->input_semantics) &&
				           same(other->output_semantics, facts->output_semantics) &&
				           other->shader_stages_en == facts->shader_stages_en;
			    })) {
				known.push_back(std::move(facts));
			}
		}
		if (!m_created_code->Add(shader.binary_type, shader.code)) {
			return;
		}
		{
			std::lock_guard lock(m_list_mutex);
			if (m_list_open && m_list_resolver != nullptr) {
				// A list names a shader by its program.
				QueueResolved(m_list_resolver->OnCode(ShaderProgramWords(shader.code)));
			}
		}
		// A census of the created shaders, at most every 10 s.
		static std::atomic<int64_t> next_report {0};
		const auto                  now  = std::chrono::duration_cast<std::chrono::seconds>(
		                                       std::chrono::steady_clock::now().time_since_epoch())
		                                       .count();
		auto                        next = next_report.load(std::memory_order_relaxed);
		if (now >= next && next_report.compare_exchange_strong(next, now + 10)) {
			const auto  counts = m_created_code->CountByType();
			std::string line   = "Created shaders:";
			for (size_t type = 0; type < Names.size(); type++) {
				if (counts[type] != 0) line += fmt::format(" {} {}", Names[type], counts[type]);
			}
			PipelineCacheLog("{}", line);
		}
	});
}

bool PipelineCache::EnqueuePrecompileJob(uint64_t key, std::string label, PrecompileBuild build,
                                         uint64_t order) {
	auto* attempts = m_precompile_attempts.get();
	if (attempts != nullptr && attempts->Failed(key)) {
		// An optimized build that timed out is replaced by the unoptimized build, which the
		// renderer then also uses (GetComputePipeline).
		if (!attempts->TimedOut(key) || attempts->Failed(UnoptimizedKey(key))) {
			return false;
		}
		EnqueuePrecompileJob(UnoptimizedKey(key), label + " (unoptimized)", std::move(build),
		                     PipelineBuildMode::Unoptimized, order);
		return true;
	}
	EnqueuePrecompileJob(key, std::move(label), std::move(build), PipelineBuildMode::Precompile,
	                     order);
	return true;
}

void PipelineCache::EnqueuePrecompileJob(uint64_t key, std::string label, PrecompileBuild build,
                                         PipelineBuildMode mode, uint64_t order) {
	auto& progress = m_precompile_progress;
	// Counted before the job can run; a key the queue rejects is uncounted again below.
	progress.Queued();
	auto job = [this, runs = m_precompile_runs, key, label, build, mode, order,
	            device = m_graphics.device] {
		bool superseded = false;
		{
			std::lock_guard lock(runs->mutex);
			superseded = runs->superseded.erase(key) != 0;
		}
		if (superseded) {
			// The renderer needed the pipeline first and built it itself.
			if (m_precompile_progress.Finish(PrecompileOutcome::Skipped, 0)) {
				PrecompileRunEnded();
			}
			return;
		}
		// The optimized pipeline of a recorded or listed build is kept for its first use.
		const bool   keep = mode == PipelineBuildMode::Precompile && !m_precompile_only &&
		                    order < PrecompileCreatedOrder && g_warm_pipelines.On() &&
		                    !m_warm_released.load(std::memory_order_relaxed);
		KeptPipeline kept;
		// A build that ends the process or never finishes leaves an unfinished attempt, so
		// that later runs skip it (PrecompileAttempts).
		auto*      attempts = m_precompile_attempts.get();
		const auto settled  = std::make_shared<std::atomic<bool>>(false);
		const auto cancel   = std::make_shared<std::atomic<bool>>(false);
		const auto start    = std::chrono::steady_clock::now();
		if (attempts != nullptr) {
			attempts->Begin(key);
		}
		{
			std::lock_guard lock(runs->mutex);
			runs->running[key] = {.start   = start,
			                      .settled = settled,
			                      .cancel  = cancel,
			                      .label   = label,
			                      .build   = build,
			                      .mode    = mode,
			                      .order   = order};
		}
		const auto outcome = build(mode, *cancel, keep ? &kept : nullptr);
		const auto ms      = std::chrono::duration_cast<std::chrono::milliseconds>(
		                         std::chrono::steady_clock::now() - start)
		                         .count();
		{
			std::lock_guard lock(runs->mutex);
			runs->running.erase(key);
		}
		if (settled->exchange(true)) {
			// Abandoned by CheckPrecompileTimeouts; it stays quarantined. Its worker was
			// replaced and the cache may be gone: only the shared state is touched.
			runs->abandoned.fetch_sub(1, std::memory_order_relaxed);
			PipelineCacheLog("Precompile: {} finished after {} s, past the time budget", label,
			                 ms / 1000);
			if (kept.pipeline != nullptr) {
				device.destroyPipeline(kept.pipeline, nullptr);
			}
			return;
		}
		if (kept.pipeline != nullptr) {
			m_warm_pipelines.Keep(kept);
		}
		// (ProgramCache::built_now: a dispatch waits for this before it takes the module.)
		if (!m_precompile_only && order < PrecompileCreatedOrder &&
		    (outcome == PrecompileOutcome::Compiled || outcome == PrecompileOutcome::Cached)) {
			std::lock_guard lock(m_built_keys_mutex);
			m_built_keys.insert(key);
		}
		// A created shader's code-only build that ended is not queued by later sessions.
		if (order >= PrecompileCreatedOrder && outcome != PrecompileOutcome::TimedOut &&
		    m_finished_created != nullptr) {
			m_finished_created->Add(key, outcome != PrecompileOutcome::Skipped);
		}
		if (attempts != nullptr) {
			attempts->End(key);
		}
		// A probe hit may have been served by the driver's own disk cache, which then adds the
		// pipeline to the application's cache: that changes the file too.
		if (outcome == PrecompileOutcome::Compiled || outcome == PrecompileOutcome::Cached) {
			m_driver_cache_dirty.store(true, std::memory_order_relaxed);
			m_precompile_dirty.store(true, std::memory_order_relaxed);
		}
		if (m_precompile_progress.Finish(outcome, static_cast<uint64_t>(ms))) {
			PrecompileRunEnded();
		}
	};
	const bool queued = m_precompile_queue->Enqueue(key, std::move(job), order);
	if (!queued) {
		progress.Rejected();
	} else if (order < PrecompileCreatedOrder) {
		m_last_recorded_enqueue.store(std::chrono::steady_clock::now().time_since_epoch().count(),
		                              std::memory_order_relaxed);
	}
}

void PipelineCache::CheckPrecompileTimeouts() {
	const auto                                          now = std::chrono::steady_clock::now();
	std::vector<std::pair<uint64_t, RunningPrecompile>> expired;
	{
		std::lock_guard lock(m_precompile_runs->mutex);
		for (const auto& [key, running]: m_precompile_runs->running) {
			if (now - running.start >= PrecompileBuildBudget && !running.settled->exchange(true)) {
				running.cancel->store(true);
				expired.emplace_back(key, running);
			}
		}
	}
	for (auto& [key, running]: expired) {
		m_precompile_runs->abandoned.fetch_add(1, std::memory_order_relaxed);
		if (m_precompile_attempts != nullptr) {
			m_precompile_attempts->TimeOut(key);
		}
		const bool replace = running.mode == PipelineBuildMode::Precompile &&
		                     (m_precompile_attempts == nullptr ||
		                      !m_precompile_attempts->Failed(UnoptimizedKey(key)));
		PipelineCacheLog(
		    "Precompile: {} has not finished after {} s; abandoned and quarantined{}",
		    running.label,
		    std::chrono::duration_cast<std::chrono::seconds>(now - running.start).count(),
		    replace ? ", building it unoptimized" : "");
		// Queued before the abandoned build is counted as finished, so the run stays open.
		if (replace) {
			EnqueuePrecompileJob(UnoptimizedKey(key), running.label + " (unoptimized)",
			                     running.build, PipelineBuildMode::Unoptimized, running.order);
		}
		if (m_precompile_progress.Finish(PrecompileOutcome::TimedOut, 0) && !m_precompile_only) {
			LogPrecompileStats("done");
		}
		// The worker in the driver call is replaced, so the other builds keep their threads;
		// it ends by itself if the call returns (a Vulkan call cannot be cancelled).
		(void)m_precompile_queue->Abandon(key);
	}
}

void PipelineCache::UpdatePrecompileThreads() {
	if (m_precompile_queue == nullptr || m_precompile_steady_threads == 0 ||
	    m_precompile_steady_threads >= m_precompile_queue->ThreadCount()) {
		return;
	}
	const auto now = std::chrono::steady_clock::now();
	if (!m_game_rendering) {
		const auto flips = Common::GpuTiming::GuestFlips();
		if (m_flip_window_start == std::chrono::steady_clock::time_point {}) {
			m_flip_window_start       = now;
			m_flip_window_start_count = flips;
		} else if (now - m_flip_window_start >= std::chrono::seconds {10}) {
			m_rendering_windows = flips - m_flip_window_start_count >= PrecompileRenderingFlips
			                          ? m_rendering_windows + 1
			                          : 0;
			m_game_rendering    = m_rendering_windows >= PrecompileRenderingWindows;
			m_flip_window_start = now;
			m_flip_window_start_count = flips;
		}
	}
	const auto last_recorded = std::chrono::steady_clock::time_point(
	    std::chrono::steady_clock::duration(m_last_recorded_enqueue.load()));
	const bool booting =
	    !m_game_rendering && (m_precompile_queue->QueuedBelow(PrecompileCreatedOrder) != 0 ||
	                          now - last_recorded < PrecompileBootQuiet);
	const auto threads = booting ? m_precompile_queue->ThreadCount() : m_precompile_steady_threads;
	if (threads != m_precompile_queue->ActiveThreads()) {
		m_precompile_queue->SetActiveThreads(threads);
		PipelineCacheLog("Precompile: {} threads from now on ({})", threads,
		                 booting            ? "recorded builds are left"
		                 : m_game_rendering ? "the game renders"
		                                    : "no recorded build is left");
	}
}

// The pipelines kept for their first use (WarmPipelines) live in device memory: about 0.8 GB
// for the 3,700 a Saros session keeps. A first use without one costs a few milliseconds (the
// driver cache provides the pipeline); device memory over its usage limit costs the frame
// rate, and past the budget it ends in a lost device. So they go, and no more are kept, once
// usage comes within warm_headroom_mb of the limit (MemoryBudget::UsageLimit).
static Common::LiveControl::Switch g_warm_headroom_mb("warm_headroom_mb", 768);

void PipelineCache::ReleaseWarmPipelinesUnderPressure() {
	if (m_warm_released.load(std::memory_order_relaxed)) {
		return;
	}
	const auto headroom = uint64_t {g_warm_headroom_mb.Get()} << 20u;
	const auto heaps    = m_graphics.SampleMemoryHeaps();
	if (heaps.device.budget == 0 ||
	    heaps.device.usage + headroom <= MemoryBudget::UsageLimit(heaps.device.budget)) {
		return;
	}
	m_warm_released.store(true, std::memory_order_relaxed);
	const auto counts = m_warm_pipelines.GetCounts();
	m_warm_pipelines.Clear();
	// The builds the precompile beside the game has not started stay undone as well: a pipeline
	// created through the cache leaves device memory behind in the driver even when it is
	// destroyed at once (about 0.9 GB after 3,700 of them, 2.6 GB after 15,000). What they
	// would have put in the driver cache is compiled at its first use, or by the next pass.
	const auto dropped = m_precompile_queue != nullptr ? m_precompile_queue->DropQueued() : 0;
	PipelineCacheLog("Pipelines the precompile kept ready: {} released and no more kept, and {} "
	                 "builds of the precompile beside the game left undone: device memory is near "
	                 "its limit ({} of {} MiB used)",
	                 counts.kept - counts.taken, dropped, heaps.device.usage >> 20u,
	                 heaps.device.budget >> 20u);
}

void PipelineCache::EndPrecompileForExit() {
	std::lock_guard lock(m_precompile_runs->mutex);
	for (const auto& [key, running]: m_precompile_runs->running) {
		running.cancel->store(true);
		if (m_precompile_attempts != nullptr && !running.settled->load()) {
			m_precompile_attempts->Interrupt(key);
		}
	}
}

void PipelineCache::EnqueuePrecompile(std::shared_ptr<const ComputeProgramRecord> record,
                                      bool code_only, uint64_t position) {
	const auto key = record->Key();
	auto       label =
	    fmt::format("{} CS {:016x}", code_only ? "created" : "recorded", record->shader_hash);
	(void)EnqueuePrecompileJob(
	    key, std::move(label),
	    [graphics = &m_graphics, driver_cache = m_driver_cache, history = m_driver_history,
	     record = std::move(record),
	     code_only](PipelineBuildMode mode, const std::atomic<bool>& cancel, KeptPipeline* keep) {
		    return PrecompileComputeProgram(*graphics, driver_cache, *record, nullptr, code_only,
		                                    mode, &cancel, history, keep);
	    },
	    code_only ? PrecompileCreatedOrder + position : position);
}

void PipelineCache::EnqueuePrecompile(
    std::shared_ptr<const GraphicsPipelineRecord>           record,
    std::vector<std::shared_ptr<const GraphicsStageRecord>> stages, uint64_t position) {
	const auto key = record->Key();
	m_pipeline_patterns.try_emplace(key, PipelinePattern {record, stages});
	auto       label =
	    fmt::format("pipeline {:016x} (VS {:016x} PS {:016x})", key, stages.front()->shader_hash,
	                record->pixel_stage != 0 ? stages.back()->shader_hash : 0);
	(void)EnqueuePrecompileJob(
	    key, std::move(label),
	    [this, record = std::move(record),
	     stages = std::move(stages)](PipelineBuildMode mode, const std::atomic<bool>& cancel,
	                                 KeptPipeline* keep) {
		    return PrecompileGraphicsPipeline(*record, stages, mode, &cancel, keep);
	    },
	    position);
}

void PipelineCache::QueueResolved(PipelineListResolver::Resolved resolved) {
	for (size_t i = 0; i < resolved.compute.size(); i++) {
		const auto key = resolved.compute[i]->Key();
		m_list_keys.insert(key);
		// A code-only build that an earlier session of this build finished is in the driver
		// cache (see QueuePrecompile).
		if (resolved.compute_code_only[i] && m_finished_created != nullptr &&
		    m_finished_created->Contains(key)) {
			continue;
		}
		EnqueuePrecompile(std::move(resolved.compute[i]), resolved.compute_code_only[i],
		                  resolved.compute_position[i]);
	}
	for (auto& pipeline: resolved.pipelines) {
		m_list_keys.insert(pipeline.record->Key());
		auto stages = std::move(pipeline.vertex_stages);
		if (pipeline.pixel_stage != nullptr) {
			stages.push_back(std::move(pipeline.pixel_stage));
		}
		EnqueuePrecompile(std::move(pipeline.record), std::move(stages), pipeline.position);
	}
}

void PipelineCache::ImportPipelineList() {
	const auto given = Config::ImportPipelineListPath();
	if (given.empty()) {
		return;
	}
	// A folder of lists has one per title, or none for this one.
	const auto      path = PipelineList::FileFor(given, m_title_id);
	std::error_code missing;
	if (path != given && !std::filesystem::exists(path, missing)) {
		PipelineCacheLog("Pipeline list: {} has no list of {}", Common::PathToString(given),
		                 m_title_id);
		return;
	}
	PipelineList::ReadStats stats;
	auto                    contents = PipelineList::Read(path, m_title_id, &stats);
	if (!stats.valid || !stats.title_matches) {
		PipelineCacheLog("Pipeline list: {} is {}; ignored", Common::PathToString(path),
		                 !stats.valid ? "not a pipeline list" : "the list of " + contents.title_id);
		return;
	}
	const auto entries =
	    contents.compute.size() + contents.stages.size() + contents.pipelines.size();
	m_list_resolver = std::make_unique<PipelineListResolver>(std::move(contents));
	// Code this emulator has: the shaders the guest created and the recorded ones, then the
	// dumped originals (--graphics-debug-dump) for what is still missing.
	const auto local = [this](const ShaderCodeId& id) -> ShaderCode {
		if (auto code = m_created_code != nullptr ? m_created_code->Find(id) : nullptr) return code;
		if (auto code =
		        m_pipeline_manifest != nullptr ? m_pipeline_manifest->FindCode(id) : nullptr)
			return code;
		return nullptr;
	};
	// A created compute shader's record holds its whole binary, while a recorded program (and a
	// list entry made from one) names the program: binaries are also found by their program,
	// whose words are copied out when a list entry asks for them.
	std::unordered_map<ShaderCodeId, ShaderCode, ShaderCodeIdHash> compute_code;
	std::unordered_map<ShaderCodeId, ShaderCode, ShaderCodeIdHash> compute_binaries;
	for (const auto* manifest: {m_manifest.get(), m_created.get()}) {
		if (manifest == nullptr) continue;
		for (const auto& record: manifest->Records()) {
			compute_code.try_emplace(ShaderCodeId::Of(*record->code), record->code);
			if (const auto program = ShaderProgramWords(*record->code);
			    program.size() != record->code->size()) {
				compute_binaries.try_emplace(ShaderCodeId::Of(program), record->code);
			}
		}
	}
	auto   resolved   = m_list_resolver->Resolve([&](const ShaderCodeId& id) -> ShaderCode {
		if (auto code = local(id)) return code;
		if (const auto found = compute_code.find(id); found != compute_code.end()) {
			return found->second;
		}
		const auto binary = compute_binaries.find(id);
		if (binary == compute_binaries.end()) return nullptr;
		const auto program = ShaderProgramWords(*binary->second);
		auto       code =
		    std::make_shared<const std::vector<uint32_t>>(program.begin(), program.end());
		compute_code.try_emplace(id, code);
		return code;
	});
	size_t from_dumps = 0;
	size_t dumps_read = 0;
	if (m_list_resolver->Pending()) {
		// Dumped originals (--graphics-debug-dump) are named by program hash, which is the
		// code's for all but merged stages: only the dumps of a pending entry's hash are read.
		// --precompile-only, where no guest creates the rest, then offers the other dumps too.
		const auto                         wanted = m_list_resolver->PendingShaderHashes();
		std::error_code error;
		const auto      folder = Config::GetShaderLogFolder() / "original";
		std::vector<std::filesystem::path> matching;
		std::vector<std::filesystem::path> others;
		for (const auto& file: std::filesystem::directory_iterator(folder, error)) {
			if (!file.is_regular_file(error) || file.path().extension() != ".bin") continue;
			const auto stem = file.path().stem().string();
			const auto mark = stem.rfind('_');
			uint64_t   hash = 0;
			const bool named =
			    mark != std::string::npos && stem.size() - mark - 1 == 16 &&
			    std::from_chars(stem.data() + mark + 1, stem.data() + stem.size(), hash, 16).ec ==
			        std::errc {};
			(named && wanted.contains(hash) ? matching : others).push_back(file.path());
		}
		const auto offer = [&](const std::filesystem::path& path) {
			const auto size = std::filesystem::file_size(path, error);
			if (error || size == 0 || size % sizeof(uint32_t) != 0 || size > (64u << 20u)) return;
			std::vector<uint32_t> code(size / sizeof(uint32_t));
			std::ifstream         in(path, std::ios::binary);
			if (!in.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(size)))
				return;
			dumps_read++;
			auto more = m_list_resolver->OnCode(code);
			from_dumps += more.compute.size() + more.pipelines.size();
			resolved.compute.insert(resolved.compute.end(), more.compute.begin(),
			                        more.compute.end());
			resolved.compute_code_only.insert(resolved.compute_code_only.end(),
			                                  more.compute_code_only.begin(),
			                                  more.compute_code_only.end());
			resolved.compute_position.insert(resolved.compute_position.end(),
			                                 more.compute_position.begin(),
			                                 more.compute_position.end());
			std::ranges::move(more.pipelines, std::back_inserter(resolved.pipelines));
		};
		for (const auto& path: matching) {
			if (!m_list_resolver->Pending()) break;
			offer(path);
		}
		if (m_precompile_only) {
			for (const auto& path: others) {
				if (!m_list_resolver->Pending()) break;
				offer(path);
			}
		}
	}
	const auto counts = m_list_resolver->GetCounts();
	PipelineCacheLog(
	    "Pipeline list: {} entries in {}{}: compute {} of {} resolved, graphics "
	    "pipelines {} of {} ({} without their stages), {} from {} dumped shaders; {} "
	    "shaders unknown so far",
	    entries, Common::PathToString(path),
	    stats.dropped_bytes != 0 ? fmt::format(" ({} damaged bytes dropped)", stats.dropped_bytes)
	                             : std::string {},
	    counts.compute_resolved, counts.compute, counts.pipelines_resolved, counts.pipelines,
	    counts.pipelines_broken, from_dumps, dumps_read, counts.shaders_pending);
	QueueResolved(std::move(resolved));
}

// --shader-pairs <file>: "KytyShaderPairs2", then a line per pair: the record key of the
// pipeline it is patterned on (a recorded or listed one, with one vertex-side stage and no
// merged program), the vertex shader's program hash, words and user data count, and the pixel
// shader's (0 0 0 for a pattern without a pixel stage), hashes in hex. The shaders' code comes
// from the created shader store (the shaders of the game's files, with --import-shaders).
uint64_t PipelineCache::QueueShaderPairs(uint64_t& finished_before) {
	auto path = Config::ShaderPairsPath();
	if (path.empty() || m_created_code == nullptr) {
		return 0;
	}
	// A folder of lists has one per title, or none for this one.
	std::error_code error;
	if (std::filesystem::is_directory(path, error)) {
		path /= m_title_id + ".pairs";
		if (!std::filesystem::exists(path, error)) {
			return 0;
		}
	}
	const auto    given = Common::PathToString(path);
	std::ifstream in(path);
	std::string   line;
	if (!in || !std::getline(in, line) || line.rfind("KytyShaderPairs2", 0) != 0) {
		PipelineCacheLog("Shader pairs: {} is not a shader pairs file; ignored", given);
		return 0;
	}
	const auto facts_of = [this](uint64_t hash) -> ShaderFactsList {
		std::lock_guard lock(m_shader_facts_mutex);
		const auto      found = m_shader_facts.find(hash);
		return found != m_shader_facts.end() ? found->second : ShaderFactsList {};
	};
	// A check of the derivation on what is known to be right: a recorded pipeline's own stage
	// inputs have to follow from its own two shaders' headers.
	{
		uint64_t checked = 0, reproduced = 0, vertex_checked = 0, vertex_reproduced = 0;
		for (const auto& [key, pattern]: m_pipeline_patterns) {
			(void)key;
			if (pattern.record->vertex_stage_count != 1u) {
				continue;
			}
			if (const auto& recorded = *pattern.stages.front();
			    recorded.stage == ShaderType::Vertex && recorded.back_code == nullptr) {
				if (const auto facts = facts_of(recorded.shader_hash); !facts.empty()) {
					vertex_checked++;
					vertex_reproduced += std::ranges::any_of(facts, [&](const auto& vertex) {
						auto derived = recorded;
						return ShaderDeriveVertexInput(*vertex, derived.vertex) &&
						       derived.Key() == recorded.Key();
					}) ? 1 : 0;
				}
			}
			if (pattern.record->pixel_stage == 0) {
				continue;
			}
			const auto& recorded     = *pattern.stages.back();
			const auto  pixel_facts  = facts_of(recorded.shader_hash);
			const auto  vertex_facts = facts_of(pattern.stages.front()->shader_hash);
			if (pixel_facts.empty() || vertex_facts.empty()) {
				continue;
			}
			checked++;
			bool same = false;
			for (const auto& pixel: pixel_facts) {
				for (const auto& vertex: vertex_facts) {
					auto derived = recorded;
					same = same || (ShaderDerivePixelInput(*pixel, *vertex, derived.pixel) &&
					                derived.Key() == recorded.Key());
				}
			}
			reproduced += same ? 1 : 0;
		}
		if (checked != 0 || vertex_checked != 0) {
			PipelineCacheLog("Shader pairs: the pixel input of {} of {} recorded pipelines and the "
			                 "vertex input of {} of {} follow from their shaders' headers",
			                 reproduced, checked, vertex_reproduced, vertex_checked);
		}
	}
	uint64_t queued = 0, unknown_pattern = 0, unknown_code = 0, unknown_header = 0, position = 0;
	while (std::getline(in, line)) {
		uint64_t pattern_key = 0, vertex_hash = 0, pixel_hash = 0;
		uint32_t vertex_words = 0, pixel_words = 0, vertex_user_data = 0, pixel_user_data = 0;
		if (std::sscanf(line.c_str(),
		                "%" SCNx64 " %" SCNx64 " %" SCNu32 " %" SCNu32 " %" SCNx64 " %" SCNu32
		                " %" SCNu32,
		                &pattern_key, &vertex_hash, &vertex_words, &vertex_user_data, &pixel_hash,
		                &pixel_words, &pixel_user_data) != 7) {
			continue;
		}
		const auto found = m_pipeline_patterns.find(pattern_key);
		if (found == m_pipeline_patterns.end() || found->second.record->vertex_stage_count != 1u ||
		    (found->second.record->pixel_stage != 0) != (pixel_words != 0) ||
		    std::ranges::any_of(found->second.stages,
		                        [](const auto& stage) { return stage->back_code != nullptr; })) {
			unknown_pattern++;
			continue;
		}
		auto vertex_code = m_created_code->Find({.hash = vertex_hash, .words = vertex_words});
		auto pixel_code  = pixel_words != 0
		                       ? m_created_code->Find({.hash = pixel_hash, .words = pixel_words})
		                       : nullptr;
		if (vertex_code == nullptr || (pixel_words != 0 && pixel_code == nullptr)) {
			unknown_code++;
			continue;
		}
		const auto listed = [&] {
			m_pair_shaders.insert(vertex_hash);
			if (pixel_words != 0) {
				m_pair_shaders.insert(pixel_hash);
			}
		};
		const std::array<uint64_t, 4> identity {
		    pattern_key, vertex_hash, pixel_hash,
		    uint64_t {vertex_user_data} | (uint64_t {pixel_user_data} << 32u)};
		const auto key = XXH3_64bits_withSeed(identity.data(), sizeof(identity), 0x70616972ull);
		if (m_finished_created != nullptr && m_finished_created->Contains(key)) {
			finished_before++;
			listed();
			continue;
		}
		// The stages take inputs from the two shaders' headers, which are known for the shaders
		// this session has seen created or imported (the precompile pass imports the game's).
		auto vertex_facts = facts_of(vertex_hash);
		auto pixel_facts  = pixel_words != 0 ? facts_of(pixel_hash) : ShaderFactsList {};
		if (vertex_facts.empty() || (pixel_words != 0 && pixel_facts.empty())) {
			unknown_header++;
			continue;
		}
		auto label = fmt::format("pair VS {:016x} PS {:016x} like {:016x}", vertex_hash, pixel_hash,
		                         pattern_key);
		if (EnqueuePrecompileJob(
		        key, std::move(label),
		        [this, pattern = found->second,
		         vertex = PairShader {std::move(vertex_code), vertex_user_data,
		                              std::move(vertex_facts)},
		         pixel  = PairShader {std::move(pixel_code), pixel_user_data,
		                              std::move(pixel_facts)}](
		            PipelineBuildMode mode, const std::atomic<bool>& cancel, KeptPipeline* keep) {
			        return PrecompileShaderPair(pattern, vertex, pixel, mode, &cancel, keep);
		        },
		        2 * PrecompileCreatedOrder + position++)) {
			queued++;
			listed();
		}
	}
	PipelineCacheLog("Shader pairs: {} queued, {} built by earlier sessions, {} name a pipeline "
	                 "that is not recorded or listed, {} a shader that is not in the created "
	                 "shader store, {} left for a session that has the shaders' headers (the "
	                 "precompile pass with --import-shaders) ({})",
	                 queued, finished_before, unknown_pattern, unknown_code, unknown_header, given);
	return queued;
}

void PipelineCache::QueuePrecompile(uint32_t thread_count) {
	EXIT_IF(thread_count == 0);
	if (m_precompile_queue == nullptr) {
		// Next to a running guest the builds yield to it, and keep to the efficiency cores where
		// the processor has them (background_ecores); the pass by itself takes nearly every
		// CPU, and yields to whatever else the PC is used for meanwhile.
		m_precompile_queue = std::make_unique<ProgramPrecompileQueue>(
		    thread_count, true, !m_precompile_only && g_background_ecores.On());
	}
	auto&           progress = m_precompile_progress;
	std::lock_guard lock(m_list_mutex);
	progress.BeginQueueing();
	// A recorded program whose image tables the host enumerated gets a record of the layout at
	// the tables' key limit too (IR::WidestCandidateLayout), built and prepared with the rest:
	// its module serves the larger tables of a place no session has shown, where the recorded
	// layouts are outgrown and a module would be compiled at the dispatch.
	if (g_widest_table_layouts.On()) {
		const auto                   recorded = m_manifest->Records();
		std::unordered_set<uint64_t> known;
		for (const auto& record: recorded) {
			known.insert(record->Key());
		}
		// And of the layout the tables take beyond that limit, where it can be told
		// (IR::BindlessTableLayout).
		uint64_t   added = 0, present = 0, resolved_added = 0, resolved_present = 0;
		const auto add   = [&](const ComputeProgramRecord&                                    like,
		                       std::optional<ShaderRecompiler::IR::ResourceSpecialization> layout,
		                       uint64_t& added_count, uint64_t& present_count) {
			if (!layout.has_value()) {
				return;
			}
			auto other           = like;
			other.specialization = std::move(*layout);
			other.spirv_hash     = 0;
			if (!known.insert(other.Key()).second) {
				present_count++;
			} else if (m_manifest->Add(other)) {
				added_count++;
			}
		};
		for (const auto& record: recorded) {
			add(*record, ShaderRecompiler::IR::WidestCandidateLayout(record->specialization), added,
			    present);
			add(*record, ShaderRecompiler::IR::BindlessTableLayout(record->specialization),
			    resolved_added, resolved_present);
		}
		if (added + present + resolved_added + resolved_present != 0) {
			PipelineCacheLog(
			    "Precompile: of the recorded programs with image tables, {} got the layout of "
			    "the tables' key limit ({} records have theirs recorded) and {} the layout of "
			    "larger tables ({} have theirs)",
			    added, present, resolved_added, resolved_present);
		}
	}
	// Recorded programs and pipelines first, in the order they were first needed: each
	// manifest keeps first-seen order, and the two are interleaved by position.
	const auto programs = m_manifest->Records();
	for (size_t i = 0; i < programs.size(); i++) {
		EnqueuePrecompile(programs[i], false, FirstSeenPosition(i, programs.size()));
	}
	const auto recorded_compute = progress.Get().queued;
	const auto recorded_pipelines =
	    m_pipeline_manifest != nullptr
	        ? m_pipeline_manifest->Pipelines()
	        : std::vector<std::shared_ptr<const GraphicsPipelineRecord>> {};
	for (size_t i = 0; i < recorded_pipelines.size(); i++) {
		const auto&                                             pipeline = recorded_pipelines[i];
		std::vector<std::shared_ptr<const GraphicsStageRecord>> stages;
		for (const auto key: pipeline->VertexStages())
			stages.push_back(m_pipeline_manifest->FindStage(key));
		if (pipeline->pixel_stage != 0)
			stages.push_back(m_pipeline_manifest->FindStage(pipeline->pixel_stage));
		if (std::ranges::find(stages, nullptr) != stages.end()) {
			continue;
		}
		EnqueuePrecompile(pipeline, std::move(stages),
		                  FirstSeenPosition(i, recorded_pipelines.size()));
	}
	const auto recorded = progress.Get().queued;
	ImportPipelineList();
	const auto listed = progress.Get().queued - recorded;
	uint64_t   finished_before = 0;
	// A created shader gets the module of its code-only specialization, which the first
	// dispatches of most shaders no session has dispatched use. So does a shader with recorded
	// programs: a dispatch that binds it other formats or level counts than the recorded ones
	// (another place of the game, another resolution) takes that module instead of waiting for
	// one of its own.
	if (m_created != nullptr) {
		const auto created = m_created->Records();
		for (size_t i = 0; i < created.size(); i++) {
			// Built by an earlier session of this emulator build into this driver cache.
			if (m_finished_created != nullptr && m_finished_created->Contains(created[i]->Key())) {
				finished_before++;
				continue;
			}
			EnqueuePrecompile(created[i], true, FirstSeenPosition(i, created.size()));
		}
	}
	const auto created_queued = progress.Get().queued - recorded - listed;
	uint64_t   pairs_before   = 0;
	const auto pairs_queued   = QueueShaderPairs(pairs_before);
	PipelineCacheLog(
	    "Precompile: {} recorded compute programs, {} recorded graphics pipelines, {} from the "
	    "pipeline list, {} created compute shaders and {} shader pairs queued on {} threads ({} "
	    "quarantined builds skipped or rebuilt unoptimized, {} retried once; {} created compute "
	    "shaders and {} shader pairs were built by earlier sessions)",
	    recorded_compute, recorded - recorded_compute, listed, created_queued, pairs_queued,
	    thread_count, m_precompile_attempts != nullptr ? m_precompile_attempts->FailedCount() : 0,
	    m_precompile_attempts != nullptr ? m_precompile_attempts->RetriedCount() : 0,
	    finished_before, pairs_before);
	m_precompile_built_before.store(finished_before + pairs_before, std::memory_order_relaxed);
	// Entries of the list whose shaders the guest has not created stay queueable while it runs.
	m_list_open = !m_precompile_only && m_list_resolver != nullptr && m_list_resolver->Pending();
	if (!m_list_open && progress.EndQueueing()) {
		PrecompileRunEnded();
	}
}

void PipelineCache::PrecompileRunEnded() {
	// The precompile-only pass reports and saves itself (PrecompileAll).
	if (m_precompile_only || m_precompile_late.exchange(false, std::memory_order_relaxed)) {
		return;
	}
	LogPrecompileStats("done");
	m_precompile_save_due.store(true, std::memory_order_relaxed);
}

void PipelineCache::LogPrecompileStats(const char* when) {
	const auto counts = m_precompile_progress.Get();
	// A pass after the whole game was built checks only the recorded programs and pipelines
	// again; without the rest, its count reads as if the others were gone.
	const auto  before = m_precompile_built_before.load(std::memory_order_relaxed);
	std::string earlier;
	if (before != 0) {
		earlier = fmt::format("; {} more were built by earlier passes of this build and not "
		                      "checked again ({} in all)",
		                      before, before + counts.cached + counts.compiled);
	}
	PipelineCacheLog(
	    "Precompile {}: {} pipelines already in the driver cache, {} compiled ({} s of "
	    "driver time), {} skipped, {} timed out{}",
	    when, counts.cached, counts.compiled, counts.compile_ms / 1000u, counts.skipped,
	    counts.timed_out, earlier);
}

void PipelineCache::StartPrecompile(uint32_t thread_count, uint32_t steady_thread_count) {
	if (m_manifest == nullptr || m_driver_cache == nullptr) {
		PipelineCacheLog("Precompile: no driver pipeline cache for this title; nothing to build");
		return;
	}
	m_precompile_steady_threads = std::min(steady_thread_count, thread_count);
	// The boot counts as a recorded enqueue: the guest has not created its shaders yet.
	m_last_recorded_enqueue.store(std::chrono::steady_clock::now().time_since_epoch().count(),
	                              std::memory_order_relaxed);
	QueuePrecompile(thread_count);
}

PipelineCache::PrecompileResult PipelineCache::PrecompileAll(uint32_t thread_count,
                                                             const std::function<bool()>& pump) {
	m_precompile_only = true;
	// Sessions found stale entries: this precompile builds a new cache.
	if (!m_driver_cache_rebuild && m_driver_history != nullptr &&
	    m_driver_history->StaleCount() != 0 && m_driver_cache != nullptr) {
		(void)StartEmptyDriverCache();
	}
	StartPrecompile(thread_count);
	if (m_precompile_queue == nullptr) {
		return Save() ? PrecompileResult::Done : PrecompileResult::SaveFailed;
	}
	auto next_report = std::chrono::steady_clock::now() + std::chrono::seconds {30};
	bool interrupted = false;
	// What was finished at the last reports, for the pace of the last two minutes.
	struct Report {
		std::chrono::steady_clock::time_point at;
		uint64_t                              finished = 0;
	};
	std::vector<Report> reports {{std::chrono::steady_clock::now(), 0}};
	// A long pass says that it need not be sat through.
	if (m_precompile_progress.Get().queued >= 2000) {
		PipelineCacheLog("Precompile: the pass can be stopped at any time by closing the emulator's "
		                 "window: what it has built is saved then (and every few minutes while it "
		                 "runs), and the next pass goes on from there");
	}
	while (!m_precompile_progress.Complete()) {
		if (pump && !pump()) {
			interrupted = true;
			break;
		}
		CheckPrecompileTimeouts();
		// What the pass has built is written every few minutes, in the background, as a session
		// does it: a pass that is cut off (a power cut, a closed console) loses the minutes since.
		if (m_precompile_dirty.load(std::memory_order_relaxed) &&
		    m_precompile_dirty.exchange(false, std::memory_order_relaxed)) {
			CheckpointDriverCache();
		}
		if (const auto now = std::chrono::steady_clock::now(); now >= next_report) {
			next_report += std::chrono::seconds {30};
			const auto counts = m_precompile_progress.Get();
			const auto left   = counts.queued - counts.finished;
			reports.push_back({now, counts.finished});
			if (reports.size() > 5) {
				reports.erase(reports.begin());
			}
			// The kinds of builds differ in what they cost, so the estimate is of this pace only.
			const auto   built   = counts.finished - reports.front().finished;
			const double seconds = std::chrono::duration<double>(now - reports.front().at).count();
			std::string  estimate;
			if (built != 0 && seconds > 0.0) {
				const double minutes = static_cast<double>(left) * seconds / static_cast<double>(built) / 60.0;
				estimate = minutes < 1.5 ? std::string("; about a minute left at this pace")
				                         : fmt::format("; about {:.0f} min left at this pace", minutes);
			}
			PipelineCacheLog("Precompile: {} builds left of {} ({}% done{})", left, counts.queued,
			                 counts.queued != 0 ? counts.finished * 100 / counts.queued : 100, estimate);
		}
		std::this_thread::sleep_for(std::chrono::milliseconds {250});
	}
	if (interrupted) {
		// No build starts any more. The ones in a driver call cannot be stopped; they do not count
		// as attempts that ended the process (two of those quarantine a build), and what they add
		// to the cache while it is written is allowed for (WriteDriverCache).
		const auto dropped = m_precompile_queue->DropQueued();
		EndPrecompileForExit();
		PipelineCacheLog("Precompile: interrupted (window closed); {} builds were not started", dropped);
		if (m_driver_cache_rebuild) {
			// The next run rebuilds again; the old file stays meanwhile.
			m_driver_cache_dirty.store(false);
			PipelineCacheLog("Vulkan pipeline cache: not rebuilt, keeping {}",
			                 Common::PathToString(m_driver_cache_path));
		}
	}
	LogPrecompileStats(interrupted ? "stopped" : "done");
	const bool joinable = !interrupted && m_precompile_runs->abandoned.load() == 0;
	if (joinable) {
		m_precompile_queue->Stop();
		m_precompile_queue.reset();
		ReleasePrecompiledStages();
		// Builds the driver cache was given before but no longer provides compile again in every
		// session, and the driver does not replace their entries: only a new cache has them.
		if (!m_driver_cache_rebuild && m_driver_history != nullptr &&
		    m_driver_history->StaleCount() != 0 && StartEmptyDriverCache()) {
			return PrecompileAll(thread_count, pump);
		}
		return Save() ? PrecompileResult::Done : PrecompileResult::SaveFailed;
	}
	// Builds still run (abandoned, or interrupted): their threads are never joined. The driver
	// cache is written without waiting for them, at most for a bounded time in case the driver
	// holds the cache.
	auto saved   = std::make_shared<std::atomic<int>>(0);
	auto pending = std::make_shared<std::atomic<bool>>(true);
	std::thread([this, saved, pending] {
		saved->store(Save() ? 1 : 2);
		pending->store(false);
	}).detach();
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds {120};
	while (pending->load() && std::chrono::steady_clock::now() < deadline) {
		if (pump) (void)pump();
		std::this_thread::sleep_for(std::chrono::milliseconds {250});
	}
	if (pending->load()) {
		PipelineCacheLog("Vulkan pipeline cache: save did not finish within 120 s");
	}
	if (interrupted) {
		return PrecompileResult::Interrupted;
	}
	return saved->load() == 1 ? PrecompileResult::Done : PrecompileResult::SaveFailed;
}

bool PipelineCache::ExportPipelineList(const std::filesystem::path& path) {
	if (m_manifest == nullptr || m_pipeline_manifest == nullptr) {
		PipelineCacheLog("Pipeline list: no manifests for this title; nothing to export");
		return false;
	}
	auto contents = PipelineList::FromManifests(m_title_id, m_manifest.get(), m_created.get(),
	                                            m_pipeline_manifest.get());
	// With a list given for the title (--import-pipeline-list): that list first, then what this
	// folder recorded besides. Exported over the list that ships, a folder's recordings are added
	// to it; they do not take the place of what other sessions recorded.
	if (const auto given = Config::ImportPipelineListPath(); !given.empty()) {
		const auto              list = PipelineList::FileFor(given, m_title_id);
		PipelineList::ReadStats stats;
		auto                    merged = PipelineList::Read(list, m_title_id, &stats);
		if (stats.valid && stats.title_matches && stats.rejected == 0) {
			const size_t had_compute   = merged.compute.size();
			const size_t had_pipelines = merged.pipelines.size();
			const auto   added         = PipelineList::Merge(merged, contents);
			PipelineCacheLog("Pipeline list: {} had {} compute programs and {} graphics pipelines; "
			                 "this data folder's recordings add {} and {} ({} of its {} and {} "
			                 "were listed already)",
			                 Common::PathToString(list), had_compute, had_pipelines, added.compute,
			                 added.pipelines, contents.compute.size() - added.compute,
			                 contents.compute.size(), contents.pipelines.size());
			contents = std::move(merged);
		}
	}
	if (!PipelineList::Write(path, contents)) {
		PipelineCacheLog("Pipeline list: failed to write {}", Common::PathToString(path));
		return false;
	}
	size_t code_only = 0;
	for (const auto& entry: contents.compute) {
		code_only += entry.code_only ? 1u : 0u;
	}
	std::error_code error;
	PipelineCacheLog(
	    "Pipeline list: exported {} compute programs ({} created code-only), {} graphics "
	    "pipelines and {} stages to {} ({} KiB, no shader code)",
	    contents.compute.size(), code_only, contents.pipelines.size(), contents.stages.size(),
	    Common::PathToString(path), std::filesystem::file_size(path, error) / 1024u);
	return true;
}

void PipelineCache::LogStaleDriverCache() const {
	if (m_driver_history == nullptr || m_driver_history->StaleCount() == 0) {
		return;
	}
	PipelineCacheLog("Vulkan pipeline cache: {} builds ({} this session) missed pipelines the "
	                 "cache was given before (stale entries the driver keeps but does not "
	                 "provide); --precompile-only rebuilds the cache",
	                 m_driver_history->StaleCount(), m_driver_history->SessionStaleCount());
}

void PipelineCache::LogCoverage() {
	auto& programs = *m_program_cache;
	if (m_graphics_from_cache + m_graphics_compiled != 0) {
		PipelineCacheLog("Graphics pipelines this session: {} from the driver cache, {} compiled; "
		                 "{} of {} recorded were predicted (by earlier sessions or the list)",
		                 m_graphics_from_cache, m_graphics_compiled, m_graphics_predicted,
		                 m_graphics_recorded);
	}
	if (programs.code_only_same + programs.code_only_taken + programs.code_only_other +
	        programs.code_only_listed !=
	    0) {
		PipelineCacheLog("Compute programs this session: of the first uses of shaders built "
		                 "from their code alone, {} were that module, {} took it (it serves "
		                 "them), {} took their own from the pipeline list instead, {} needed "
		                 "their own",
		                 programs.code_only_same, programs.code_only_taken,
		                 programs.code_only_listed, programs.code_only_other);
	}
	if (programs.own_modules_started != 0) {
		PipelineCacheLog("Compute programs this session: {} dispatches served by a code-only "
		                 "module got their own module built in the background; {} took it",
		                 programs.own_modules_started, programs.own_modules_taken);
	}
	if (const auto warm = m_warm_pipelines.GetCounts(); warm.kept != 0) {
		PipelineCacheLog("Pipelines the precompile kept ready: {} taken at their first use of {} "
		                 "kept",
		                 warm.taken, warm.kept);
	}
	if (programs.preparation != nullptr) {
		const auto counts = GetProgramPreparationCounts();
		PipelineCacheLog("Programs prepared ahead: {} of {} recorded or built from their code "
		                 "translated ({} failed); {} taken at their first use ({} waited for), {} "
		                 "first used before their turn",
		                 counts.prepared, counts.queued, counts.failed, counts.adopted,
		                 counts.waited, counts.missed);
	}
	if (programs.deferred_plans != 0 || programs.deferred_modules != 0) {
		PipelineCacheLog("Stage programs translated off the GPU thread while their draws were "
		                 "left out: {} programs, {} modules ({} done on the GPU thread after all)",
		                 programs.deferred_plans, programs.deferred_modules,
		                 programs.deferred_failed);
	}
	if (programs.compute_recorded != 0) {
		PipelineCacheLog("Compute programs this session: {} of {} recorded were predicted",
		                 programs.compute_predicted, programs.compute_recorded);
	}
	if (programs.created_code != nullptr && !programs.census_seen.empty()) {
		std::string line = "Graphics shaders used this session (created by the guest before use):";
		const std::array<std::pair<ShaderType, const char*>, 6> stages = {
		    {{ShaderType::Vertex, "VS"},
		     {ShaderType::Pixel, "PS"},
		     {ShaderType::Mesh, "GS"},
		     {ShaderType::Local, "LS"},
		     {ShaderType::TessellationControl, "HS"},
		     {ShaderType::TessellationEvaluation, "DS"}}};
		for (const auto& [stage, name]: stages) {
			const auto& counts = programs.census[static_cast<size_t>(stage)];
			if (counts[0] != 0) line += fmt::format(" {} {} ({})", name, counts[0], counts[1]);
		}
		PipelineCacheLog("{}", line);
	}
	if (m_list_resolver != nullptr) {
		const auto counts = m_list_resolver->GetCounts();
		PipelineCacheLog("Pipeline list: compute {} of {} and graphics pipelines {} of {} resolved "
		                 "this session",
		                 counts.compute_resolved, counts.compute, counts.pipelines_resolved,
		                 counts.pipelines);
	}
}

bool PipelineCache::StartEmptyDriverCache() {
	LogStaleDriverCache();
	vk::PipelineCacheCreateInfo create {};
	vk::PipelineCache           empty = nullptr;
	const auto result = m_graphics.device.createPipelineCache(&create, nullptr, &empty);
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: cannot start an empty cache ({})",
		                 vk::to_string(result));
		return false;
	}
	m_driver_cache_writer.Wait();
	m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	m_driver_cache         = empty;
	m_driver_cache_rebuild = true;
	m_driver_cache_stored_size.store(0, std::memory_order_relaxed);
	m_driver_cache_dirty.store(false);
	m_driver_history->Clear();
	if (m_finished_created != nullptr) {
		m_finished_created->Clear();
	}
	m_precompile_progress.Reset();
	PipelineCacheLog(
	    "Vulkan pipeline cache: rebuilding {} from the pipelines this precompile builds",
	    Common::PathToString(m_driver_cache_path));
	return true;
}

void PipelineCache::CheckpointDriverCache() {
	// A driver cache without a file (tests) is not persisted.
	if (m_driver_cache == nullptr || m_driver_cache_path.empty()) {
		return;
	}
	m_driver_cache_dirty.store(true, std::memory_order_relaxed);
	// A cache being rebuilt holds only part of the old file's pipelines until the precompile
	// ends; it is written once, then.
	if (m_driver_cache_rebuild) {
		return;
	}
	const auto now = std::chrono::steady_clock::now();
	if (now < m_next_driver_cache_save || m_driver_cache_writer.Busy()) {
		return;
	}
	// Snapshotting and writing a large cache takes seconds; pipeline creation continues
	// meanwhile, and pipelines created during the snapshot mark the cache dirty again.
	m_next_driver_cache_save = now + DriverCacheSaveInterval;
	m_driver_cache_dirty.store(false, std::memory_order_relaxed);
	m_driver_cache_writer.TryStart([this] {
		if (!WriteDriverCache()) {
			m_driver_cache_dirty.store(true, std::memory_order_relaxed);
		}
	});
}

bool PipelineCache::Save() {
	if (m_compute_from_cache + m_compute_compiled != 0) {
		PipelineCacheLog("Compute pipelines this session: {} from the driver cache, {} compiled",
		                 m_compute_from_cache, m_compute_compiled);
		const auto ms = [](uint64_t ns) { return static_cast<double>(ns) / 1'000'000.0; };
		PipelineCacheLog("Pipelines created at their first use this session held the GPU thread "
		                 "for {:.0f} ms (compute) and {:.0f} ms (graphics); {} and {} of them for "
		                 "5 ms or more, the longest {:.0f} ms and {:.0f} ms",
		                 ms(m_compute_create.ns), ms(m_graphics_create.ns), m_compute_create.slow,
		                 m_graphics_create.slow, ms(m_compute_create.longest),
		                 ms(m_graphics_create.longest));
		static constexpr std::array<const char*, 7> Rules = {
		    "left out (background build)", "option off", "compute", "a stage writes memory",
		    "a query counts the draw", "a linear target", "a depth or stencil clear"};
		std::string by_rule;
		for (size_t rule = 0; rule < m_graphics_create_by_blocker.size(); rule++) {
			const auto& times = m_graphics_create_by_blocker[rule];
			if (times.count == 0) continue;
			by_rule += fmt::format("{}{}: {} pipelines, {:.0f} ms, longest {:.0f} ms",
			                       by_rule.empty() ? "" : "; ", Rules[rule], times.count,
			                       ms(times.ns), ms(times.longest));
		}
		if (!by_rule.empty()) {
			PipelineCacheLog("Graphics pipelines created at their first use, by what kept their "
			                 "draws in the frame: {}",
			                 by_rule);
		}
	}
	LogStaleDriverCache();
	LogCoverage();
	if (m_simplified_store != nullptr) {
		m_simplified_store->Flush();
	}
	m_driver_cache_writer.Wait();
	PersistStaleMisses();
	if (m_driver_cache == nullptr || m_driver_cache_path.empty() ||
	    !m_driver_cache_dirty.exchange(false)) {
		// Nothing to write: what ended since the last write changed no cache (shaders that do
		// not translate), or there is no cache file to fall behind. (A rebuild that was stopped
		// keeps the old file, which does not have what this pass built.)
		if (m_finished_created != nullptr && !m_driver_cache_rebuild) {
			m_finished_created->Commit(m_finished_created->TakePending());
		}
		return true;
	}
	m_next_driver_cache_save = std::chrono::steady_clock::now() + DriverCacheSaveInterval;
	if (!WriteDriverCache()) {
		m_driver_cache_dirty.store(true, std::memory_order_relaxed);
		return false;
	}
	return true;
}

bool PipelineCache::WriteDriverCache() {
	// Taken first: every pipeline it names is in the data below.
	const auto built =
	    m_driver_history != nullptr ? m_driver_history->Snapshot() : std::vector<uint64_t> {};
	const auto           stale = m_driver_history != nullptr ? m_driver_history->StaleCount() : 0;
	// And the builds that ended before the data is taken: their pipelines are in it.
	auto ended = m_finished_created != nullptr ? m_finished_created->TakePending()
	                                           : FinishedBuilds::Ended {};
	const auto give_back = [&] {
		if (m_finished_created != nullptr) {
			m_finished_created->Restore(std::move(ended));
		}
	};
	size_t               size  = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	// Builds may run meanwhile (a precompile's threads, a pass that was stopped and whose driver
	// calls cannot be interrupted), and each adds to the cache between the question for its size
	// and the copy, which then comes back incomplete: the buffer gets room to spare, more with
	// every attempt.
	for (uint32_t attempt = 0; attempt < 8; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0) {
			break;
		}
		// The cache holds what its file holds: a pipeline it was given makes it larger, and a
		// build it only provided does not. A session after a precompile pass provides thousands
		// and adds none, and wrote the whole file again twice (2 GB for Saros, and as much
		// memory for the copy while it did).
		if (attempt == 0 && size == m_driver_cache_stored_size.load(std::memory_order_relaxed)) {
			if (m_finished_created != nullptr) {
				m_finished_created->Commit(ended);
			}
			return true;
		}
		size += (size / 16u + (size_t {16} << 20u)) * (size_t {1} << std::min(attempt, 4u));
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {:.2f} MB)",
		                 vk::to_string(result), static_cast<double>(size) / 1'000'000.0);
		give_back();
		return false;
	}
	payload.resize(size);
	const auto signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	uint64_t   hash      = 0;
	if (!DriverCacheFile::Store(m_driver_cache_path, signature, payload.data(), payload.size(),
	                            DriverCacheFile::DefaultChunkSize, &hash)) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
		give_back();
		return false;
	}
	m_driver_cache_stored_size.store(payload.size(), std::memory_order_relaxed);
	if (m_driver_history != nullptr) {
		WriteDriverHistory(built, hash, stale);
	}
	if (m_finished_created != nullptr) {
		m_finished_created->Commit(ended);
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {:.2f} MB to {}",
	                 static_cast<double>(payload.size()) / 1'000'000.0,
	                 Common::PathToString(m_driver_cache_path));
	return true;
}

void PipelineCache::WriteDriverHistory(std::vector<uint64_t> built, uint64_t cache_hash,
                                       uint64_t stale) {
	const auto      signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	std::lock_guard lock(m_history_file_mutex);
	if (!DriverCacheHistory::Store(m_driver_history_path,
	                               DriverCacheHistory::Signature(signature, cache_hash), built,
	                               stale)) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_history_path));
		return;
	}
	m_history_file_built      = std::move(built);
	m_history_file_cache_hash = cache_hash;
	m_history_file_stale      = stale;
}

void PipelineCache::PersistStaleMisses() {
	if (m_driver_history == nullptr || m_driver_cache_rebuild) {
		return;
	}
	const auto      stale     = m_driver_history->StaleCount();
	const auto      signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	std::lock_guard lock(m_history_file_mutex);
	if (m_history_file_cache_hash == 0 || stale == m_history_file_stale) {
		return;
	}
	if (DriverCacheHistory::Store(
	        m_driver_history_path,
	        DriverCacheHistory::Signature(signature, m_history_file_cache_hash),
	        m_history_file_built, stale)) {
		m_history_file_stale = stale;
	}
}

void PipelineCache::GetGraphicsPrograms(const CommandBuffer& command, DrawRenderState& state) {
	const auto& context           = command.GetRegisters();
	const auto& user_config       = command.GetUserConfig();
	const auto& sh                = context.GetShaderRegisters();
	const auto& vertex_regs       = command.GetShaders().GetVs();
	const auto& pixel_regs        = command.GetShaders().GetPs();
	auto&       vertex_info       = state.vertex_info;
	auto&       pixel_info        = state.ps_input_info;
	const bool  object_primitive_id = context.HasObjectPrimitiveIdOutput();
	const auto  write_mask          = context.GetRenderTargetMask() & sh.m_cbShaderMask;
	uint32_t    color_output_mask   = object_primitive_id ? 1u << 7u : 0;
	for (uint32_t index = 0; index < RENDER_COLOR_ATTACHMENTS_MAX; index++) {
		const auto slot = ShaderPixelExportTarget(sh.m_cbShaderMask, index);
		if (slot >= RENDER_COLOR_ATTACHMENTS_MAX) {
			break;
		}
		if (sh.target_output_mode[index] != 0 && render_target_mask_slot(write_mask, slot) != 0) {
			color_output_mask |= 1u << slot;
		}
	}
	const auto& db = sh.db_shader_control;
	// Without color outputs, a pixel shader still runs for its depth or coverage side effects.
	const bool pixel_active =
	    pixel_regs.ps_regs.data_addr != 0 &&
	    (color_output_mask != 0 || db.shader_kill_enable || db.shader_z_export_enable ||
	     db.shader_mask_export_enable || db.shader_dual_export_enable ||
	     db.shader_execute_on_noop || sh.AlphaToMask());
	state.ps_active = pixel_active;
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	std::array<ShaderParams, 3> vertex_params;
	if (tess_active) {
		vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info);
	} else {
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	if (!tess_active && !mesh_active) {
		vertex_info[0].indirect_offsets = state.indirect_offsets;
		vertex_info[0].indirect_draw    = state.indirect_draw;
	}
	if (object_primitive_id) {
		// CxObjectIdMode: register source plus a 32-bit pipelined primitive-ID addition.
		EXIT_NOT_IMPLEMENTED(sh.m_paClObjprimIdCntl != 2u || mesh_active || tess_active ||
		                     !pixel_active || (sh.m_vgtDrawPayloadCntl & 8u) == 0 ||
		                     user_config.GetPrimType() != Prospero::PrimitiveType::kTriList);
		const auto& target = context.GetRenderTarget(7);
		EXIT_NOT_IMPLEMENTED(Prospero::ResolveRenderTargetFormat(target.info.format,
		                                                       target.info.channel_type)
		                         .buffer_format != Prospero::BufferFormat::k32UInt);
	}
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		mesh.host_subgroup_size = m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
		                          std::min(mesh.host_subgroup_size, mesh.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations ||
		    host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: dropping mesh draws the host does not support (shader 0x{:016x}).\n",
				    vertex_params[0].hash));
			}
			return;
		}
		// A task shader launches each record's groups and passes its draw parameters in a
		// payload beside the LDS, the allocation pair and the Layer array; without room for
		// it, the CPU reads the records.
		if (state.indirect_draw &&
		    (mesh.lds_size_dwords + 2u + mesh.max_vertices +
		     ShaderRecompiler::IR::PushData::MeshTaskPayloadDwordCount) * sizeof(uint32_t) <=
		        limits.maxMeshPayloadAndSharedMemorySize) {
			vertex_info[0].indirect_offsets = state.indirect_offsets;
			vertex_info[0].indirect_draw    = true;
		}
	}
	ShaderParams pixel_params;
	if (pixel_active) {
		std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX>
		    target_export_mapping {};
		for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			const auto& rt = context.GetRenderTarget(slot);
			if ((color_output_mask & (1u << slot)) != 0 && rt.base.addr != 0) {
				target_export_mapping[slot] =
				    TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type,
				                                 rt.info.channel_order)
				        .export_mapping;
			}
		}
		pixel_params      = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		pixel_info.object_primitive_id = object_primitive_id;
		for (uint32_t export_index = 0; export_index < 8u; export_index++) {
			const auto slot = ShaderPixelExportTarget(pixel_info.target_shader_mask, export_index);
			if (slot >= 8u || pixel_info.target_output_mode[export_index] == 0u) continue;
			const auto& rt = context.GetRenderTarget(slot);
			const auto target_mask = render_target_mask_slot(context.GetRenderTargetMask(), slot);
			if (rt.base.addr == 0 || target_mask == 0u) continue;
			if (rt.info.channel_type == Prospero::ChannelType::kUInt) {
				pixel_info.target_uint_mask |= 1u << export_index;
			} else if (rt.info.channel_type == Prospero::ChannelType::kSInt) {
				pixel_info.target_sint_mask |= 1u << export_index;
			}
			const auto format = Prospero::ResolveRenderTargetFormat(rt.info.format, rt.info.channel_type);
			if (format.buffer_format != Prospero::BufferFormat::k11_11_10UNorm) continue;
			const auto mapping = pixel_info.target_export_mapping[export_index];
			const auto shader_mask = render_target_mask_slot(pixel_info.target_shader_mask, slot);
			const bool blend = context.GetBlendControl(slot).enable && !rt.info.blend_bypass;
			if (blend || (mapping.ApplyMask(target_mask) & 7u) != 7u ||
			    (mapping.ApplyMask(shader_mask) & 7u) != 7u ||
			    pixel_info.target_output_mode[export_index] != 9u) {
				EXIT("unsupported packed UNorm render-target export: slot=%u export=%u mode=%u "
				     "target_mask=0x%x shader_mask=0x%x mapping=0x%02x blend=%u\n",
				     slot, export_index, pixel_info.target_output_mode[export_index], target_mask,
				     shader_mask, mapping.packed, static_cast<uint32_t>(blend));
			}
			pixel_info.target_packed_unorm_mask |= 1u << export_index;
			pixel_info.target_round_mode |= static_cast<uint8_t>(rt.info.round_mode) << export_index;
		}
		if (!mesh_active && !tess_active && Prospero::IsRectList(user_config.GetPrimType())) {
			pixel_info.parameter_mode = ShaderPixelParameterMode::Rectangle;
		} else if (context.GetModeControl().provoking_vtx_last) {
			pixel_info.parameter_mode = ShaderPixelParameterMode::LastVertex;
		} else {
			pixel_info.parameter_mode = ShaderPixelParameterMode::FirstVertex;
		}
		const auto& blend = context.GetBlendControl(0);
		pixel_info.dual_source_blending =
		    blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		    (BlendFactorIsDualSource(blend.color_srcblend) ||
		     BlendFactorIsDualSource(blend.color_destblend) ||
		     (blend.separate_alpha_blend && (BlendFactorIsDualSource(blend.alpha_srcblend) ||
		                                     BlendFactorIsDualSource(blend.alpha_destblend))));
		if (pixel_info.dual_source_blending) {
			// MRT1 supplies the second blend source for target 0; blended targets are not integer.
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
			pixel_info.target_uint_mask &= ~2u;
			pixel_info.target_sint_mask &= ~2u;
		} else if (blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		           pixel_info.target_output_mode[0] != 0 && pixel_info.target_output_mode[0] != 7 &&
		           std::all_of(std::begin(pixel_info.target_output_mode) + 1,
		                       std::end(pixel_info.target_output_mode),
		                       [](uint8_t mode) { return mode == 0; })) {
			switch (ClassifyBlendMapping(blend, pixel_info.target_export_mapping[0])) {
				case BlendMappingSupport::SourceAlpha:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlpha;
					break;
				case BlendMappingSupport::SourceAlphaOne:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaOne;
					break;
				case BlendMappingSupport::SourceAlphaZero:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaZero;
					break;
				default: break;
			}
			if (pixel_info.alpha_blend_source != ShaderAlphaBlendSource::None) {
				pixel_info.dual_source_blending     = true;
				pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
				pixel_info.target_export_mapping[1] = {};
			}
		}
	}
	auto&       last_stage   = vertex_info[tess_active ? 2u : 0u];
	const auto& clip_control = context.GetClipControl();
	if (clip_control.clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = last_stage.clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	} else if (clip_control.user_clip_planes != 0) {
		// CLIP_DISABLE also disables user planes. How they combine with shader clip and cull
		// distances is unverified; draw push constants only reach VS/mesh and PS, and the
		// generated RECTLIST stages do not forward clip distances.
		EXIT_NOT_IMPLEMENTED((last_stage.pa_cl_vs_out_cntl & 0xffffu) != 0 || tess_active ||
		                     Prospero::IsRectList(user_config.GetPrimType()));
		last_stage.user_clip_planes = clip_control.user_clip_planes;
	}
	uint32_t push_data_cursor = ShaderRecompiler::IR::PushData::GraphicsDrawDwords(
	    mesh_active, object_primitive_id, last_stage.user_clip_planes);
	// Whether this draw may be left out while a stage program of its is translated
	// (ProgramCache::Deferral): what PendingDrawSkipBlocker asks of a draw whose pipeline is
	// being built, as far as the registers say it; that no stage stores anywhere is asked of
	// each program (MayDeferStage). A draw that clears depth or stencil or renders to a linear
	// target is not, and none is without a skip window.
	bool may_defer = m_skip_pending_draws && m_skip_window > m_skip_window.zero() && !tess_active;
	if (may_defer) {
		const auto& rc = context.GetRenderControl();
		may_defer      = !rc.depth_clear_enable && !rc.stencil_clear_enable;
		for (uint32_t slot = 0; may_defer && slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			const auto& rt = context.GetRenderTarget(slot);
			if (rt.base.addr != 0 &&
			    render_target_mask_slot(context.GetRenderTargetMask(), slot) != 0 &&
			    rt.attrib3.tile_mode == Prospero::TileMode::kLinear) {
				may_defer = false;
			}
		}
	}
	m_program_cache->defer_stage_programs = may_defer;
	const auto unavailable = [&] {
		state.unavailable = true;
		if (m_program_cache->deferred) {
			Common::DebugCounters::Add(Common::DebugCounters::Counter::DrawsSkippedPending);
		}
	};
	if (pixel_active) {
		if (!m_program_cache->Get(pixel_params, pixel_info, push_data_cursor)) {
			unavailable();
			return;
		}
		for (const auto& output: pixel_info.stage.program->info.outputs) {
			if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt ||
			    output.kind == ShaderRecompiler::IR::StageOutputKind::ObjectPrimitiveId) {
				state.mrt_mask |= 1u << output.location;
			}
		}
		state.mrt_mask &= color_output_mask;
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		if (!m_program_cache->Get(vertex_params[i], vertex_info[i], push_data_cursor)) {
			unavailable();
			return;
		}
	}
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               ShaderComputeInputInfo&      input_info) {
	input_info.host_subgroup_size = m_graphics.ComputeHostSubgroupSize();
	const auto        params      = PrepareProgram(regs, input_info);
	input_info.lds_storage = input_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	uint32_t          push_data_cursor = 0;
	return m_program_cache->Get(params, input_info, push_data_cursor);
}

bool PipelineCache::LookaheadEnabled() const {
	return m_program_cache->lookahead != nullptr;
}

ResourceLookahead* PipelineCache::Lookahead() const {
	return m_program_cache->lookahead.get();
}

void PipelineCache::BeginLookaheadRun(const void* stream) {
	m_program_cache->lookahead->BeginRun(stream);
}

void PipelineCache::PlanLookaheadDispatch(const void* stream, const void* command,
                                          const HW::ComputeShaderInfo& regs, uint32_t groups_x,
                                          uint32_t groups_y, uint32_t groups_z,
                                          bool thread_dimensions, uint64_t indirect_args_address) {
	const auto& cs = regs.cs_regs;
	// The inputs RenderExecutor::DispatchDirect/DispatchIndirect give the walk.
	LookaheadInputs inputs;
	inputs.indirect_args_address = indirect_args_address;
	if (indirect_args_address == 0) {
		if (groups_x == 0 || groups_y == 0 || groups_z == 0) return;
		inputs.workgroup_counts = {groups_x, groups_y, groups_z};
		if (thread_dimensions) {
			const std::array<uint32_t, 3> sizes {cs.num_thread_x, cs.num_thread_y, cs.num_thread_z};
			for (uint32_t axis = 0; axis < 3u; ++axis) {
				const auto size               = std::max(sizes[axis], 1u);
				const auto threads            = inputs.workgroup_counts[axis];
				inputs.workgroup_counts[axis] = threads / size + (threads % size != 0u);
			}
		}
	}
	if (cs.user_sgpr > HW::UserSgprInfo::SGPRS_MAX) return;
	m_program_cache->PlanLookahead(stream, command, ProgramCache::LookaheadSlot::Compute,
	                               cs.data_addr,
	                               std::span(regs.cs_user_sgpr.value).first(cs.user_sgpr), inputs);
}

void PipelineCache::PlanLookaheadDraw(const void* stream, const void* command,
                                      const HW::VertexShaderInfo& vs,
                                      const HW::PixelShaderInfo&  ps) {
	// GetGraphicsPrograms walks the pixel shader first.
	if (ps.ps_regs.data_addr != 0 && ps.ps_regs.rsrc2.user_sgpr <= HW::UserSgprInfo::SGPRS_MAX) {
		LookaheadInputs inputs;
		m_program_cache->PlanLookahead(
		    stream, command, ProgramCache::LookaheadSlot::Pixel, ps.ps_regs.data_addr,
		    std::span(ps.ps_user_sgpr.value).first(ps.ps_regs.rsrc2.user_sgpr), inputs);
	}
	if (vs.es_regs.data_addr != 0 && vs.gs_regs.rsrc2.user_sgpr <= HW::UserSgprInfo::SGPRS_MAX) {
		LookaheadInputs inputs;
		m_program_cache->PlanLookahead(
		    stream, command, ProgramCache::LookaheadSlot::Vertex, vs.es_regs.data_addr,
		    std::span(vs.gs_user_sgpr.value).first(vs.gs_regs.rsrc2.user_sgpr), inputs);
	}
}

void PipelineCache::KickLookahead() {
	auto& programs = *m_program_cache;
	programs.lookahead->Kick();
	const auto now = std::chrono::steady_clock::now();
	if (now < programs.next_lookahead_report) return;
	if (programs.next_lookahead_report != std::chrono::steady_clock::time_point {}) {
		using Outcome    = ResourceLookahead::Outcome;
		const auto stats = programs.lookahead->TakeStats();
		const auto count = [&](Outcome outcome) {
			return static_cast<unsigned long long>(stats.outcomes[static_cast<size_t>(outcome)]);
		};
		LOGF("Lookahead: %u helpers, %llu planned, adopted %llu, not planned %llu, changed %llu, "
		     "late %llu, failed %llu, dirty %llu, moved %llu, heap %llu, kept %llu, skipped %llu "
		     "(10 s)\n",
		     programs.lookahead->HelperCount(), static_cast<unsigned long long>(stats.planned),
		     count(Outcome::Adopted), count(Outcome::NotPlanned), count(Outcome::Changed),
		     count(Outcome::Late), count(Outcome::Failed), count(Outcome::Dirty),
		     count(Outcome::Moved), count(Outcome::Heap),
		     static_cast<unsigned long long>(stats.kept),
		     static_cast<unsigned long long>(stats.skipped));
	} else {
		(void)programs.lookahead->TakeStats();
	}
	programs.next_lookahead_report = now + std::chrono::seconds(10);
}

void PipelineCache::SetLookaheadCommand(const void* stream, const void* command) {
	m_program_cache->lookahead_stream  = stream;
	m_program_cache->lookahead_command = command;
}

PipelineCache::Pipeline* PipelineCache::ResolvePendingBuild(Pipeline&          pipeline,
                                                            PendingDrawBlocker blocker) {
	const auto entry = m_pending_builds.find(&pipeline);
	EXIT_IF(entry == m_pending_builds.end());
	auto&      pending = entry->second;
	const auto waited  = std::chrono::steady_clock::now() - pending.queued;
	if (m_skip_pending_draws && PendingDrawMaySkip(blocker, waited, m_skip_window)) {
		pending.skipped_draws++;
		return nullptr;
	}
	if (!pending.wait_logged) {
		pending.wait_logged = true;
		LOGF("PipelineCache: waiting for the pipeline of VS %016" PRIx64 " PS %016" PRIx64
		     " (%s)\n",
		     pending.vs_hash, pending.ps_hash,
		     blocker == PendingDrawBlocker::None ? "skipped for too long"
		                                         : PendingDrawBlockerName(blocker));
	}
	m_compile_queue->Expedite(pending.ticket);
	InstallOptimizedPipelines();
	EXIT_NOT_IMPLEMENTED(pipeline.pipeline == nullptr);
	return &pipeline;
}

uint64_t PipelineCache::RecordGraphicsPipeline(const GraphicsPipelineKey& key,
                                               const DrawRenderState&     state) {
	GraphicsPipelineRecord record;
	const auto             vertex_stages = state.VertexStages();
	record.vertex_stage_count            = static_cast<uint32_t>(vertex_stages.size());
	for (uint32_t i = 0; i < vertex_stages.size(); i++) {
		record.vertex_stages[i] = vertex_stages[i].stage.handle->record_key;
		if (record.vertex_stages[i] == 0) return 0;
	}
	if (state.ps_active) {
		record.pixel_stage = state.ps_input_info.stage.handle->record_key;
		if (record.pixel_stage == 0) return 0;
	}
	record.rendering      = key.rendering;
	record.vertex_input   = key.vertex_input;
	record.static_params  = key.static_params;
	const auto record_key = record.Key();
	m_graphics_recorded++;
	bool predicted = !m_pipeline_manifest->AddPipeline(record);
	if (!predicted) {
		std::lock_guard lock(m_list_mutex);
		predicted = m_list_keys.contains(record_key);
	}
	if (predicted) {
		m_graphics_predicted++;
	}
	return record_key;
}

void PipelineCache::SupersedePrecompile(uint64_t key) {
	if (m_precompile_queue == nullptr || key == 0) {
		return;
	}
	std::lock_guard lock(m_precompile_runs->mutex);
	// A build that runs is past the check; only queued ones read the set.
	if (!m_precompile_runs->running.contains(key)) {
		m_precompile_runs->superseded.insert(key);
	}
}

uint64_t PipelineCache::PendingBackgroundBuilds() {
	uint64_t pending = 0;
	if (m_program_cache->preparation != nullptr) {
		std::lock_guard lock(m_program_cache->preparation->mutex);
		const auto&     counts = m_program_cache->preparation->counts;
		// A program its first use translated itself (missed) left the queue too.
		const auto done = counts.prepared + counts.failed + counts.missed;
		pending += counts.queued > done ? counts.queued - done : 0;
	}
	const auto precompile = m_precompile_progress.Get();
	pending += precompile.queued > precompile.finished ? precompile.queued - precompile.finished : 0;
	return pending;
}

PipelineCache::Pipeline& PipelineCache::GetGraphicsPipeline(const CommandBuffer&   command,
                                                            const DrawRenderState& state) {
	return *FindGraphicsPipeline(command, state, PendingDrawBlocker::Disabled);
}

PipelineCache::Pipeline* PipelineCache::FindGraphicsPipeline(const CommandBuffer&   command,
                                                             const DrawRenderState& state,
                                                             PendingDrawBlocker     blocker) {
	const auto* colors         = state.color_info;
	const auto& depth          = state.depth_info;
	const auto  vertex_info    = state.VertexStages();
	const auto& ps_input_info  = state.ps_input_info;
	const auto& vs_input_info  = vertex_info.front();
	const auto* vertex_program = vs_input_info.stage.handle;
	const auto* pixel_program  = state.ps_input_info.stage.handle;
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	EXIT_IF(vertex_program == nullptr);
	const bool ps_active = state.ps_active;
	EXIT_IF(ps_active && pixel_program == nullptr);
	const auto color_count = state.color_count;

	auto&             ctx = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	const auto vs_id = vertex_program->id;
	const auto ps_id = ps_active ? pixel_program->id : 0;

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < vertex_info.size(); i++) {
		key.vertex_shader_ids[i] = vertex_info[i].stage.handle->id;
	}
	key.ps_shader_id            = ps_id;
	auto& static_params         = key.static_params;
	auto& rendering             = key.rendering;
	rendering.color_count       = 0;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		if (colors[i].desc.info.guest_format == Prospero::BufferFormat::k11_11_10UNorm) {
			static_params.color_mask[slot] = 1u;
		}
		rendering.color_formats[slot] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		auto alpha_source = ShaderAlphaBlendSource::None;
		if (slot == 0 && ps_active) {
			alpha_source = ps_input_info.alpha_blend_source;
		}
		static_params.blend_enable[slot] = bc.enable && !rt.info.blend_bypass;
		if (static_params.blend_enable[slot] && alpha_source == ShaderAlphaBlendSource::None &&
		    ClassifyBlendMapping(bc, colors[i].export_mapping) != BlendMappingSupport::Direct) {
			static_params.blend_enable[slot] = false;
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: blending disabled for unsupported color mapping "
				    "(slot={} mapping=0x{:02x} color={}/{} alpha={}/{} separate={}).\n",
				    slot, colors[i].export_mapping.packed, bc.color_srcblend, bc.color_destblend,
				    bc.alpha_srcblend, bc.alpha_destblend, bc.separate_alpha_blend ? 1 : 0));
			}
		}
		if (static_params.blend_enable[slot]) {
			auto blend = bc;
			switch (alpha_source) {
				case ShaderAlphaBlendSource::SourceAlpha:
					blend.color_srcblend  = RemapSourceAlphaFactor(blend.color_srcblend);
					blend.color_destblend = RemapSourceAlphaFactor(blend.color_destblend);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::SourceAlphaOne:
				case ShaderAlphaBlendSource::SourceAlphaZero:
					// The second source carries the mapped source factor; its alpha stays logical Sa.
					blend.color_srcblend = static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color);
					blend.color_destblend =
					    static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::None: break;
			}
			static_params.color_srcblend[slot]       = blend.color_srcblend;
			static_params.color_comb_fcn[slot]       = blend.color_comb_fcn;
			static_params.color_destblend[slot]      = blend.color_destblend;
			static_params.separate_alpha_blend[slot] = blend.separate_alpha_blend;
			if (blend.separate_alpha_blend) {
				static_params.alpha_srcblend[slot]  = blend.alpha_srcblend;
				static_params.alpha_comb_fcn[slot]  = blend.alpha_comb_fcn;
				static_params.alpha_destblend[slot] = blend.alpha_destblend;
			}
		}
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = state.topology;
	static_params.primitive_restart_enable = state.primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info.ps_sample_shading;
	static_params.alpha_to_coverage_enable = ps_active && ctx.GetShaderRegisters().AlphaToMask();
	const bool rect_list = Prospero::IsRectList(command.GetUserConfig().GetPrimType());
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, !rect_list && mc.cull_front, !rect_list && mc.cull_back);

	if (vs_input_info.stage.program->stage != ShaderType::Mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
		}
		for (int attribute = 0; attribute < vs_input_info.resources_num; attribute++) {
			const auto binding = vs_input_info.resources_dst[attribute].buffer_index;
			EXIT_IF(binding < 0 || binding >= vs_input_info.buffers_num);
			key.vertex_input.attributes[attribute] = {
			    .offset = static_cast<uint32_t>(vs_input_info.resources[attribute].Base48() -
			                                    vs_input_info.buffers[binding].addr),
			    .binding = static_cast<uint8_t>(binding),
			};
		}
	}

	InstallOptimizedPipelines();
	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		auto& found = *iter->second;
		return found.pipeline != nullptr ? &found : ResolvePendingBuild(found, blocker);
	}
	const auto record_key = m_pipeline_manifest != nullptr ? RecordGraphicsPipeline(key, state) : 0;

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     vs_id, static_cast<void*>(vertex_program->module), ps_id,
		     ps_active ? static_cast<void*>(pixel_program->module) : nullptr);
	}

	// A draw that may be skipped leaves the whole build to the background queue.
	const auto build_mode = m_skip_pending_draws && PendingDrawMaySkip(blocker, {}, m_skip_window)
	                            ? PipelineBuildMode::Background
	                        : m_defer_optimization ? PipelineBuildMode::UnoptimizedFirst
	                                               : PipelineBuildMode::Optimized;
	auto       cached     = std::make_unique<Pipeline>();
	bool       from_driver_cache = false;
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	const auto create_start = std::chrono::steady_clock::now();
	auto job = [&] {
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		return CreatePipelineInternal(m_graphics, *cached, rendering, key.vertex_input, state,
		                              static_params, m_driver_cache, build_mode, &from_driver_cache,
		                              nullptr, m_driver_history, &m_warm_pipelines);
	}();
	(from_driver_cache ? m_graphics_from_cache : m_graphics_compiled)++;
	const auto create_ns = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
	                                                         create_start)
	        .count());
	m_graphics_create.Add(create_ns);
	if (const auto rule = static_cast<size_t>(blocker); rule < m_graphics_create_by_blocker.size()) {
		m_graphics_create_by_blocker[rule].Add(create_ns);
	}
	SupersedePrecompile(record_key);
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr && !job);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
	EXIT_IF(!inserted);
	auto& created = *iter->second;
	if (created.pipeline == nullptr) {
		const PendingBuild pending {
		    .ticket        = m_compile_queue->Enqueue(std::move(job)),
		    .queued        = std::chrono::steady_clock::now(),
		    .vs_hash       = vs_input_info.stage.program->shader_hash,
		    .ps_hash       = ps_active ? ps_input_info.stage.program->shader_hash : 0,
		    .skipped_draws = 1,
		};
		m_pending_optimized.emplace(pending.ticket, &created);
		m_pending_builds.emplace(&created, pending);
		LOGF("PipelineCache: skipping draws of VS %016" PRIx64 " PS %016" PRIx64
		     " until its pipeline is ready (built in the background; the draws only change the "
		     "picture; they wait after %" PRId64 " ms)\n",
		     pending.vs_hash, pending.ps_hash,
		     static_cast<int64_t>(
		         std::chrono::duration_cast<std::chrono::milliseconds>(m_skip_window).count()));
		return nullptr;
	}
	DeferOptimization(created, std::move(job));
	CheckpointDriverCache();

	return &created;
}

PipelineCache::Pipeline& PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(input_info.stage.handle == nullptr);
	const auto& compute_program = *input_info.stage.handle;

	InstallOptimizedPipelines();
	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	// A recorded program whose precompile has finished is taken ready-made below
	// (m_warm_pipelines). One whose precompile has not started is built here like any other
	// pipeline (from the driver cache, or unoptimized first) and its precompile is skipped; one
	// being built is built here too: waiting for an optimized build would stop the frame.
	// A program whose optimized build ran past the precompile budget is never built optimized:
	// the driver may not finish that build at all.
	const bool optimized_hangs   = compute_program.record_key != 0 &&
	                               m_precompile_attempts != nullptr &&
	                               m_precompile_attempts->TimedOut(compute_program.record_key);
	auto       cached            = std::make_unique<Pipeline>();
	bool       from_driver_cache = false;
	const auto create_start      = std::chrono::steady_clock::now();
	auto       job               = [&] {
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		return CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module,
		                              m_driver_cache,
		                              optimized_hangs        ? PipelineBuildMode::Unoptimized
		                              : m_defer_optimization ? PipelineBuildMode::UnoptimizedFirst
		                                                     : PipelineBuildMode::Optimized,
		                              &from_driver_cache, nullptr, m_driver_history,
		                              &m_warm_pipelines);
	}();
	(from_driver_cache ? m_compute_from_cache : m_compute_compiled)++;
	m_compute_create.Add(static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
	                                                         create_start)
	        .count()));
	SupersedePrecompile(compute_program.record_key);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);
	DeferOptimization(*iter->second, std::move(job));
	CheckpointDriverCache();

	return *iter->second;
}
} // namespace Libs::Graphics
