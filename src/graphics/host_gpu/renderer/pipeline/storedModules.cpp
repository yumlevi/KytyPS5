#include "graphics/host_gpu/renderer/pipeline/storedModules.h"

#include "common/liveControl.h"
#include "graphics/host_gpu/renderer/pipeline/manifestFormat.h"
#include "graphics/shader/recompiler/ir/CompiledInfoSerial.h"
#include "graphics/shader/shaderCompiler.h"
#include "kytyGitVersion.h"

#include <atomic>
#include <filesystem>
#include <string>
#include <system_error>
#include <xxhash.h>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
#endif

namespace Libs::Graphics {

namespace {

std::atomic<ModuleStore*> g_store {nullptr};
std::atomic<uint64_t>     g_variant {0};

// 0: every module is compiled, as before the store (nothing is read from it or added to it).
// Read at every compilation.
Common::LiveControl::Switch  g_module_store("module_store", 1);
Common::LiveControl::Counter g_modules_found("module_store_found");
Common::LiveControl::Counter g_modules_compiled("module_store_compiled");

// The key of the module a compilation of these inputs makes; false when the inputs cannot be
// named safely.
bool MakeKey(const ShaderRecompiler::CompileOptions&             options,
             const ShaderRecompiler::IR::ResourceSpecialization& specialization,
             uint32_t push_data_start_dword, std::span<const uint32_t> code, uint64_t variant,
             ModuleStore::Key& key) {
	std::vector<uint8_t>   bytes;
	ManifestFormat::Writer writer(bytes);
	ManifestFormat::PutSpecialization(writer, specialization);
	{
		// The encoding is the one a program's record keeps its specialization in. A
		// specialization it does not hold whole would share its key with another.
		ShaderRecompiler::IR::ResourceSpecialization back;
		ManifestFormat::Reader                       reader(bytes);
		if (!ManifestFormat::GetSpecialization(reader, back) || !reader.AtEnd() ||
		    !(back == specialization)) {
			return false;
		}
	}
	// The stage's static inputs, as the program cache keys a translated program by them.
	std::vector<uint32_t> statics;
	if (options.stage == ShaderType::Compute) {
		if (options.input_info.compute == nullptr) return false;
		BuildStageStaticKey(*options.input_info.compute, statics);
	} else if (options.stage == ShaderType::Pixel) {
		if (options.input_info.pixel == nullptr) return false;
		BuildStageStaticKey(*options.input_info.pixel, statics);
	} else {
		if (options.input_info.vertex == nullptr) return false;
		BuildStageStaticKey(*options.input_info.vertex, statics);
	}
	writer.Put(static_cast<uint32_t>(options.stage));
	writer.Put(options.shader_hash);
	writer.Put(options.wave_size);
	writer.Put(options.user_data_base);
	// (Translation reads how many user-data words there are; their values feed the resource
	// walk alone.)
	writer.Put(static_cast<uint32_t>(options.user_data.size()));
	writer.Put(push_data_start_dword);
	writer.Put(static_cast<uint32_t>(statics.size()));
	writer.PutWords(statics);
	writer.Put(static_cast<uint32_t>(code.size()));
	writer.Put(XXH3_64bits(code.data(), code.size_bytes()));
	writer.Put(static_cast<uint32_t>(options.back_code.size()));
	writer.Put(XXH3_64bits(options.back_code.data(), options.back_code.size_bytes()));
	const auto hash = XXH3_128bits_withSeed(bytes.data(), bytes.size(), variant);
	key             = {hash.low64, hash.high64};
	return true;
}

} // namespace

void SetModuleStore(ModuleStore* store, uint64_t variant) {
	g_variant.store(variant, std::memory_order_release);
	g_store.store(store, std::memory_order_release);
}

ModuleStore::Counts ModuleStoreCounts() {
	const auto* store = g_store.load(std::memory_order_acquire);
	return store != nullptr ? store->GetCounts() : ModuleStore::Counts {};
}

std::string ModuleStoreBuild() {
	std::string           build = KYTY_GIT_VERSION;
	std::filesystem::path executable;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	std::wstring name(32768, L'\0');
	const auto   length = GetModuleFileNameW(nullptr, name.data(), static_cast<DWORD>(name.size()));
	if (length != 0 && length < name.size()) {
		name.resize(length);
		executable = name;
	}
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	std::error_code link_error;
	executable = std::filesystem::read_symlink("/proc/self/exe", link_error);
#endif
	std::error_code error;
	if (!executable.empty()) {
		const auto size = std::filesystem::file_size(executable, error);
		const auto time = error ? std::filesystem::file_time_type {}
		                        : std::filesystem::last_write_time(executable, error);
		if (!error) {
			build += ':' + std::to_string(size) + ':' +
			         std::to_string(time.time_since_epoch().count());
		}
	}
	return build;
}

StoredCompile CompileThroughStore(ShaderRecompiler::TranslateResult                   translated,
                                  const ShaderRecompiler::CompileOptions&             options,
                                  const ShaderRecompiler::IR::ResourceSpecialization& specialization,
                                  uint32_t push_data_start_dword, std::span<const uint32_t> code,
                                  const std::function<bool(const std::vector<uint32_t>&)>& valid) {
	StoredCompile    result;
	ModuleStore::Key key;
	auto*            store = g_module_store.On() ? g_store.load(std::memory_order_acquire) : nullptr;
	if (store != nullptr &&
	    (options.dump_ir || options.early_dump || options.emit_debug_info || code.empty() ||
	     !MakeKey(options, specialization, push_data_start_dword, code,
	              g_variant.load(std::memory_order_acquire), key))) {
		store = nullptr;
	}
	if (store != nullptr) {
		std::vector<uint8_t> info;
		if (store->Find(key, result.spirv, info)) {
			if (ShaderRecompiler::IR::LoadCompiledInfo(info, result.program)) {
				result.from_store = true;
				g_modules_found.Add();
				return result;
			}
			// (Its bytes passed their checksum, so this build wrote an info it cannot read:
			// compiled as without the store.)
			result = {};
			store  = nullptr;
		}
	}
	auto compiled = ShaderRecompiler::CompileProgram(std::move(translated), options, specialization,
	                                                 push_data_start_dword);
	result.spirv = std::move(compiled.spirv);
	if (!valid(result.spirv)) {
		result.invalid = true;
		// (The caller may still want the info, to report what it could not use.)
		result.program = std::move(compiled.program).TakeCompiledInfo();
		return result;
	}
	result.program = std::move(compiled.program).TakeCompiledInfo();
	g_modules_compiled.Add();
	if (store != nullptr) {
		// Kept only when the info's bytes give the same info back: a field the serializer does
		// not carry makes a module that is compiled every session, never a wrong one.
		std::vector<uint8_t> info;
		ShaderRecompiler::IR::SaveCompiledInfo(result.program, info);
		ShaderRecompiler::IR::CompiledShaderInfo back;
		if (ShaderRecompiler::IR::LoadCompiledInfo(info, back) &&
		    ShaderRecompiler::IR::SameCompiledInfo(result.program, back)) {
			store->Insert(key, result.spirv, info);
		}
	}
	return result;
}

} // namespace Libs::Graphics
