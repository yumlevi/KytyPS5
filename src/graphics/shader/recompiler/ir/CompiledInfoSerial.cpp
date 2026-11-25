#include "graphics/shader/recompiler/ir/CompiledInfoSerial.h"

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <array>
#include <concepts>
#include <cstring>
#include <string>
#include <type_traits>

namespace Libs::Graphics::ShaderRecompiler::IR {

namespace {

constexpr uint32_t Magic = 0x3149434bu; // "KCI1"

template <typename T>
concept Scalar = std::is_arithmetic_v<T> || std::is_enum_v<T>;

// The structs an info is made of. Each has a Visit below that names all of its fields through a
// structured binding, which does not compile when the number of fields changes.
template <typename T>
concept Record =
    std::same_as<T, BufferResource> || std::same_as<T, ImageResource> ||
    std::same_as<T, SamplerResource> || std::same_as<T, SampledResourcePair> ||
    std::same_as<T, StageInput> || std::same_as<T, StageOutput> || std::same_as<T, ShaderInfo> ||
    std::same_as<T, Bindings> || std::same_as<T, CompiledShaderInfo>;

template <typename A, typename R>
    requires std::same_as<std::remove_const_t<R>, BufferResource>
bool Visit(A& a, R& r);
template <typename A, typename R>
    requires std::same_as<std::remove_const_t<R>, ImageResource>
bool Visit(A& a, R& r);
template <typename A, typename R>
    requires std::same_as<std::remove_const_t<R>, SamplerResource>
bool Visit(A& a, R& r);
template <typename A, typename R>
    requires std::same_as<std::remove_const_t<R>, SampledResourcePair>
bool Visit(A& a, R& r);
template <typename A, typename R>
    requires std::same_as<std::remove_const_t<R>, StageInput>
bool Visit(A& a, R& r);
template <typename A, typename R>
    requires std::same_as<std::remove_const_t<R>, StageOutput>
bool Visit(A& a, R& r);
template <typename A, typename R>
    requires std::same_as<std::remove_const_t<R>, ShaderInfo>
bool Visit(A& a, R& r);
template <typename A, typename R>
    requires std::same_as<std::remove_const_t<R>, Bindings>
bool Visit(A& a, R& r);
template <typename A, typename R>
    requires std::same_as<std::remove_const_t<R>, CompiledShaderInfo>
bool Visit(A& a, R& r);

class Writer {
public:
	explicit Writer(std::vector<uint8_t>& out): m_out(out) {}

	template <typename... Ts>
	bool operator()(const Ts&... values) {
		(Put(values), ...);
		return true;
	}

private:
	template <Scalar T>
	void Put(const T& value) {
		const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
		m_out.insert(m_out.end(), bytes, bytes + sizeof(T));
	}
	void Put(const bool& value) { m_out.push_back(value ? 1u : 0u); }
	void Put(const std::string& value) {
		Put(static_cast<uint32_t>(value.size()));
		m_out.insert(m_out.end(), value.begin(), value.end());
	}
	template <typename T>
	void Put(const std::vector<T>& values) {
		Put(static_cast<uint32_t>(values.size()));
		for (const auto& value: values) {
			Put(value);
		}
	}
	template <typename T, size_t N>
	void Put(const std::array<T, N>& values) {
		for (const auto& value: values) {
			Put(value);
		}
	}
	template <Record T>
	void Put(const T& record) {
		(void)Visit(*this, record);
	}

	std::vector<uint8_t>& m_out;
};

class Reader {
public:
	explicit Reader(std::span<const uint8_t> bytes): m_bytes(bytes) {}

	template <typename... Ts>
	bool operator()(Ts&... values) {
		return (Get(values) && ...);
	}
	[[nodiscard]] bool AtEnd() const { return m_at == m_bytes.size(); }

private:
	[[nodiscard]] size_t Left() const { return m_bytes.size() - m_at; }

	template <Scalar T>
	bool Get(T& value) {
		if (Left() < sizeof(T)) {
			return false;
		}
		std::memcpy(&value, m_bytes.data() + m_at, sizeof(T));
		m_at += sizeof(T);
		return true;
	}
	bool Get(bool& value) {
		if (Left() < 1 || m_bytes[m_at] > 1u) {
			return false;
		}
		value = m_bytes[m_at++] != 0;
		return true;
	}
	bool Get(std::string& value) {
		uint32_t count = 0;
		if (!Get(count) || Left() < count) {
			return false;
		}
		value.assign(reinterpret_cast<const char*>(m_bytes.data() + m_at), count);
		m_at += count;
		return true;
	}
	template <typename T>
	bool Get(std::vector<T>& values) {
		uint32_t count = 0;
		// (Every element takes a byte at least: a count the bytes cannot hold is damage.)
		if (!Get(count) || Left() < count) {
			return false;
		}
		values.clear();
		values.resize(count);
		for (auto& value: values) {
			if (!Get(value)) {
				return false;
			}
		}
		return true;
	}
	template <typename T, size_t N>
	bool Get(std::array<T, N>& values) {
		for (auto& value: values) {
			if (!Get(value)) {
				return false;
			}
		}
		return true;
	}
	template <Record T>
	bool Get(T& record) {
		return Visit(*this, record);
	}

	std::span<const uint8_t> m_bytes;
	size_t                   m_at = 0;
};

template <typename A, typename R>
    requires std::same_as<std::remove_const_t<R>, BufferResource>
bool Visit(A& a, R& r) {
	auto& [source, descriptor_index, first_use_pc, max_byte_extent, packed_stride, descriptor_format,
	       descriptor_swizzle, read, written, atomic, formatted, scalar, live, indirect_root,
	       indirect_mapping_offset, indirect_search_iterations, indirect_mapping_header,
	       indirect_resources, runtime_stride, portable_format] = r;
	return a(source, descriptor_index, first_use_pc, max_byte_extent, packed_stride,
	         descriptor_format, descriptor_swizzle, read, written, atomic, formatted, scalar, live,
	         indirect_root, indirect_mapping_offset, indirect_search_iterations,
	         indirect_mapping_header, indirect_resources, runtime_stride, portable_format);
}

template <typename A, typename R>
    requires std::same_as<std::remove_const_t<R>, ImageResource>
bool Visit(A& a, R& r) {
	auto& [source, descriptor_index, first_use_pc, resource_class, numeric_class, dimension, mip_mode,
	       mip_count, conversion_format, shader_swizzle, srgb_decode, read, written, atomic, atomic64,
	       unfiltered_access_only, depth_compare, skip_degamma, cube, r128, heap, indirect_root,
	       indirect_mapping_offset, indirect_search_iterations, indirect_mapping_header,
	       indirect_resources, bindless, bindless_byte_keys] = r;
	return a(source, descriptor_index, first_use_pc, resource_class, numeric_class, dimension,
	         mip_mode, mip_count, conversion_format, shader_swizzle, srgb_decode, read, written,
	         atomic, atomic64, unfiltered_access_only, depth_compare, skip_degamma, cube, r128, heap,
	         indirect_root, indirect_mapping_offset, indirect_search_iterations,
	         indirect_mapping_header, indirect_resources, bindless, bindless_byte_keys);
}

template <typename A, typename R>
    requires std::same_as<std::remove_const_t<R>, SamplerResource>
bool Visit(A& a, R& r) {
	auto& [source, first_use_pc, snapshot_index, indirect_root, indirect_mapping_offset,
	       indirect_search_iterations, indirect_mapping_header, indirect_resources,
	       force_point_filtering, depth_compare, integer_border, gather_lod] = r;
	return a(source, first_use_pc, snapshot_index, indirect_root, indirect_mapping_offset,
	         indirect_search_iterations, indirect_mapping_header, indirect_resources,
	         force_point_filtering, depth_compare, integer_border, gather_lod);
}

template <typename A, typename R>
    requires std::same_as<std::remove_const_t<R>, SampledResourcePair>
bool Visit(A& a, R& r) {
	auto& [image, sampler, first_use_pc] = r;
	return a(image, sampler, first_use_pc);
}

template <typename A, typename R>
    requires std::same_as<std::remove_const_t<R>, StageInput>
bool Visit(A& a, R& r) {
	auto& [kind, location, component_count, debug_name, per_vertex] = r;
	return a(kind, location, component_count, debug_name, per_vertex);
}

template <typename A, typename R>
    requires std::same_as<std::remove_const_t<R>, StageOutput>
bool Visit(A& a, R& r) {
	auto& [kind, index, location, debug_name] = r;
	return a(kind, index, location, debug_name);
}

template <typename A, typename R>
    requires std::same_as<std::remove_const_t<R>, ShaderInfo>
bool Visit(A& a, R& r) {
	auto& [buffers, images, samplers, sampled_pairs, address_bases, ray_query_tlas, inputs, outputs,
	       user_data_registers, vertex_fetch_components, vertex_offset_sgpr, instance_offset_sgpr,
	       has_bitwise_xor, uses_dma, uses_swizzle, uses_lds, uses_gds, uses_flattened_srt,
	       uses_image_heap, uses_device_address, uses_dispatch_threads, bvh, bvh8, bvh_stack,
	       ray_query, subgroup_ballot, subgroup_vote, subgroup_barrier, subgroup_shuffle,
	       subgroup_arithmetic, subgroup_local_invocation_id, helper_invocation,
	       compute_derivatives, image_gather_extended, function_lds, function_lds_dwords,
	       function_scratch, pixel_valid_mask, buffer_int64_atomics, buffer_u8, buffer_u16,
	       shared_int64_atomics, coherent_buffers, float64] = r;
	return a(buffers, images, samplers, sampled_pairs, address_bases, ray_query_tlas, inputs, outputs,
	         user_data_registers, vertex_fetch_components, vertex_offset_sgpr, instance_offset_sgpr,
	         has_bitwise_xor, uses_dma, uses_swizzle, uses_lds, uses_gds, uses_flattened_srt,
	         uses_image_heap, uses_device_address, uses_dispatch_threads, bvh, bvh8, bvh_stack,
	         ray_query, subgroup_ballot, subgroup_vote, subgroup_barrier, subgroup_shuffle,
	         subgroup_arithmetic, subgroup_local_invocation_id, helper_invocation,
	         compute_derivatives, image_gather_extended, function_lds, function_lds_dwords,
	         function_scratch, pixel_valid_mask, buffer_int64_atomics, buffer_u8, buffer_u16,
	         shared_int64_atomics, coherent_buffers, float64);
}

template <typename A, typename R>
    requires std::same_as<std::remove_const_t<R>, Bindings>
bool Visit(A& a, R& r) {
	auto& [descriptor_counts, descriptor_mask, push_data_start_dword, dispatch_thread_dword,
	       ray_query_dword, memory_offset_dword, memory_strides, memory_formats, image_swizzles] = r;
	return a(descriptor_counts, descriptor_mask, push_data_start_dword, dispatch_thread_dword,
	         ray_query_dword, memory_offset_dword, memory_strides, memory_formats, image_swizzles);
}

template <typename A, typename R>
    requires std::same_as<std::remove_const_t<R>, CompiledShaderInfo>
bool Visit(A& a, R& r) {
	auto& [stage, shader_hash, wave_size, user_data_base, param_export_mask, has_address_writes,
	       info, bindings] = r;
	return a(stage, shader_hash, wave_size, user_data_base, param_export_mask, has_address_writes,
	         info, bindings);
}

} // namespace

void SaveCompiledInfo(const CompiledShaderInfo& info, std::vector<uint8_t>& out) {
	Writer writer(out);
	(void)writer(Magic, info);
}

bool LoadCompiledInfo(std::span<const uint8_t> bytes, CompiledShaderInfo& info) {
	Reader   reader(bytes);
	uint32_t magic = 0;
	return reader(magic) && magic == Magic && reader(info) && reader.AtEnd();
}

bool SameCompiledInfo(const CompiledShaderInfo& left, const CompiledShaderInfo& right) {
	return left.stage == right.stage && left.shader_hash == right.shader_hash &&
	       left.wave_size == right.wave_size && left.user_data_base == right.user_data_base &&
	       left.param_export_mask == right.param_export_mask &&
	       left.has_address_writes == right.has_address_writes && left.info == right.info &&
	       left.bindings == right.bindings;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
