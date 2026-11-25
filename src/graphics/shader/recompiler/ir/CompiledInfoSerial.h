#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_COMPILEDINFOSERIAL_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_COMPILEDINFOSERIAL_H_

#include <cstdint>
#include <span>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct CompiledShaderInfo;

// The info of a compiled module (CompiledShaderInfo: what the renderer keeps of a program once its
// IR is gone) as bytes and back, for modules kept on disk between sessions (ModuleStore). The
// bytes are this build's: they hold every field in the order the structs declare them, and a
// struct that gains or loses a field stops the serializer from compiling until its list is
// brought up to date.

// Appends the bytes of `info` to `out`.
void SaveCompiledInfo(const CompiledShaderInfo& info, std::vector<uint8_t>& out);
// False for bytes that are not exactly one info; `info` is then in an unspecified state.
[[nodiscard]] bool LoadCompiledInfo(std::span<const uint8_t> bytes, CompiledShaderInfo& info);
// Whether two infos are equal in every field.
[[nodiscard]] bool SameCompiledInfo(const CompiledShaderInfo& left, const CompiledShaderInfo& right);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_COMPILEDINFOSERIAL_H_ */
