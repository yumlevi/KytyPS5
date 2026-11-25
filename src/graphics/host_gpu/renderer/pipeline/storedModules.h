#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_STOREDMODULES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_STOREDMODULES_H_

#include "graphics/host_gpu/renderer/pipeline/moduleStore.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>

namespace Libs::Graphics {

// The store the compilations of this process go through (null: none), and what every key of it
// starts from: everything outside a compilation's own arguments that changes the module it
// makes (the host's shader profile, device limits the stage inputs are derived from, settings of
// the process).
void SetModuleStore(ModuleStore* store, uint64_t variant);
// The store's counts; all zero without one.
[[nodiscard]] ModuleStore::Counts ModuleStoreCounts();
// What names the build of the emulator in a store's file: its version and, where the platform
// tells, the size and time of its executable (two builds of one working tree share a version).
[[nodiscard]] std::string ModuleStoreBuild();

struct StoredCompile {
	std::vector<uint32_t>                    spirv;
	ShaderRecompiler::IR::CompiledShaderInfo program;
	// Compiled now and refused by the caller's check: not stored, for the caller to report.
	bool invalid = false;
	// Taken from the store: nothing was compiled.
	bool from_store = false;
};

// ShaderRecompiler::CompileProgram through the module store. The module an earlier session (or
// another thread of this one) compiled from the same inputs is taken from the store; one that
// is compiled here is checked with `valid` (the SPIR-V validation of the caller) and kept.
// `code` is the guest code `translated` was made from: with the options' static inputs, the
// specialization and the push-data cursor it names the module.
//
// Compiles without the store when there is none, when the options ask for dumps or debug info
// (their output comes from the compilation), and when the specialization does not come back
// equal from the encoding the key is made of (a field the encoding does not hold would let two
// different modules share a key).
[[nodiscard]] StoredCompile
CompileThroughStore(ShaderRecompiler::TranslateResult                   translated,
                    const ShaderRecompiler::CompileOptions&             options,
                    const ShaderRecompiler::IR::ResourceSpecialization& specialization,
                    uint32_t push_data_start_dword, std::span<const uint32_t> code,
                    const std::function<bool(const std::vector<uint32_t>&)>& valid);

// The module CompileThroughStore would return for these inputs, when the store has it. Nothing
// is compiled: the caller has no translation (see FindStoredPlan). A module is in the store
// only because a compilation of this build made it from the same inputs, so the specialization
// is one the program takes.
[[nodiscard]] bool FindStoredModule(const ShaderRecompiler::CompileOptions&             options,
                                    const ShaderRecompiler::IR::ResourceSpecialization& specialization,
                                    uint32_t push_data_start_dword, std::span<const uint32_t> code,
                                    StoredCompile& result);

// The plan store: what the translation of a program gives besides its modules, kept across
// sessions like them (a second ModuleStore file of the same build; its keys start from the
// module store's variant). A session translated every recorded program at its start only to have
// these again: for Saros 12,500 programs, 15 to 18 s before the game's first frame, with the
// modules already coming from their store.
void SetPlanStore(ModuleStore* store);
[[nodiscard]] ModuleStore::Counts PlanStoreCounts();

struct StoredPlan {
	// IR::SaveResourcePlan's bytes of the plan IR::ExtractResourcePlan makes of the program.
	std::vector<uint8_t> plan;
	// IR::CodeOnlySpecialization of the program; none when it has none.
	std::optional<ShaderRecompiler::IR::ResourceSpecialization> code_only;
};
// What KeepPlan kept of the program `code` translates to with `options` (their static inputs
// name it, as they name its modules without a specialization). False when the store has none.
[[nodiscard]] bool FindStoredPlan(const ShaderRecompiler::CompileOptions& options,
                                  std::span<const uint32_t> code, StoredPlan& stored);
// Keeps them for `program`, the translation of `code` with `options`, unless the store has them
// or the plan's bytes do not load back to the same bytes. Not for a program with calls (its code
// is linked at run time; the callers do not get here with one).
void KeepPlan(const ShaderRecompiler::CompileOptions& options, std::span<const uint32_t> code,
              const ShaderRecompiler::IR::Program& program);

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_STOREDMODULES_H_ */
