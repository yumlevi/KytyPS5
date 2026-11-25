#include "graphics/host_gpu/renderer/pipeline/moduleStore.h"
#include "graphics/shader/recompiler/ir/CompiledInfoSerial.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

namespace {

using Libs::Graphics::ModuleStore;
namespace IR = Libs::Graphics::ShaderRecompiler::IR;
using namespace std::chrono_literals;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "ModuleStoreTests: failed: %s\n", text);
    std::abort();
  }
}

std::filesystem::path TempPath(const char *name) {
  auto dir = std::filesystem::temp_directory_path() / "kyty_module_store_tests";
  std::filesystem::remove_all(dir);
  return dir / "cache" / name;
}

// Words that compress (runs) with a part that does not (a hash sequence).
std::vector<uint32_t> Words(uint32_t count, uint32_t seed) {
  std::vector<uint32_t> words(count);
  for (uint32_t i = 0; i < count; i++) {
    words[i] = i % 3 == 0 ? i * 2654435761u + seed : seed;
  }
  return words;
}

std::vector<uint8_t> Bytes(uint32_t count, uint8_t seed) {
  std::vector<uint8_t> bytes(count);
  for (uint32_t i = 0; i < count; i++) {
    bytes[i] = static_cast<uint8_t>(i * 31u + seed);
  }
  return bytes;
}

void WaitWritten(ModuleStore &store, uint64_t beyond) {
  for (int i = 0; i < 2000 && store.GetCounts().file_bytes <= beyond; i++) {
    std::this_thread::sleep_for(1ms);
  }
}

void TestAcrossSessions() {
  const auto path = TempPath("title.v1.modules");
  const ModuleStore::Key key_a{1, 2};
  const ModuleStore::Key key_b{3, 4};
  const ModuleStore::Key key_c{5, 6};
  const auto words_a = Words(40000, 1);
  const auto words_b = Words(7, 2);
  const auto info_a = Bytes(300, 1);
  const auto info_b = Bytes(0, 0);
  uint64_t size_after_two = 0;
  {
    ModuleStore store(path, "build-1", 1ms);
    Check(store.GetCounts().loaded == 0, "a new file holds nothing");
    std::vector<uint32_t> words;
    std::vector<uint8_t> info;
    Check(!store.Find(key_a, words, info), "nothing is found in a new store");
    Check(!store.Contains(key_a), "a new store contains nothing");
    const auto empty = store.GetCounts().file_bytes;
    store.Insert(key_a, words_a, info_a);
    Check(store.Contains(key_a) && !store.Contains(key_b), "a key is contained once inserted");
    // Before the writer has run: served from memory.
    Check(store.Find(key_a, words, info) && words == words_a && info == info_a,
          "a module is found right after its insertion");
    store.Insert(key_b, words_b, info_b);
    store.Insert(key_a, words_b, info_b);
    WaitWritten(store, empty);
    store.Flush();
    Check(store.Find(key_a, words, info) && words == words_a && info == info_a,
          "a known key keeps its first record, read from the file");
    Check(store.Find(key_b, words, info) && words == words_b && info.empty(),
          "a module without info bytes");
    const auto counts = store.GetCounts();
    Check(counts.inserted == 2 && counts.found == 3 && counts.missed == 1 && counts.damaged == 0,
          "the counts of the first session");
    size_after_two = counts.file_bytes;
    Check(size_after_two < words_a.size() * sizeof(uint32_t), "the words are stored compressed");
  }
  Check(std::filesystem::file_size(path) == size_after_two, "the file is what the store counted");
  {
    ModuleStore store(path, "build-1", 1ms);
    Check(store.GetCounts().loaded == 2, "the next session of the build lists both records");
    std::vector<uint32_t> words;
    std::vector<uint8_t> info;
    Check(store.Find(key_a, words, info) && words == words_a && info == info_a,
          "the first module in the next session");
    Check(store.Find(key_b, words, info) && words == words_b && info.empty(),
          "the second module in the next session");
    Check(!store.Find(key_c, words, info), "an unknown key");
    Check(store.Contains(key_a) && store.Contains(key_b) && !store.Contains(key_c),
          "the next session contains what the file lists");
    store.Insert(key_c, Words(1000, 3), Bytes(10, 3));
  }
  {
    // A torn last record: the file ends inside it.
    const auto size = std::filesystem::file_size(path);
    Check(size > size_after_two, "the third record was appended when the store closed");
    std::filesystem::resize_file(path, size - 5);
    ModuleStore store(path, "build-1", 1ms);
    Check(store.GetCounts().loaded == 2, "a torn last record is dropped");
    Check(std::filesystem::file_size(path) == size_after_two, "and cut off the file");
    std::vector<uint32_t> words;
    std::vector<uint8_t> info;
    Check(!store.Find(key_c, words, info), "the torn record's key is unknown");
    Check(store.Find(key_a, words, info) && words == words_a, "the records before it are whole");
  }
  {
    // Damage inside a record: found out when it is read, and the record forgotten.
    {
      std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
      file.seekp(static_cast<std::streamoff>(size_after_two - 3));
      const char byte = 0x5a;
      file.write(&byte, 1);
    }
    ModuleStore store(path, "build-1", 1ms);
    std::vector<uint32_t> words;
    std::vector<uint8_t> info;
    Check(store.Find(key_a, words, info) && words == words_a && info == info_a,
          "an undamaged record beside a damaged one");
    Check(!store.Find(key_b, words, info), "a damaged record is not served");
    Check(store.GetCounts().damaged == 1, "and counted");
    Check(!store.Find(key_b, words, info) && store.GetCounts().damaged == 1,
          "and unknown from then on");
    // Its key can be stored again.
    store.Insert(key_b, words_b, info_b);
    Check(store.Find(key_b, words, info) && words == words_b, "a forgotten key is taken again");
  }
  {
    ModuleStore store(path, "build-2", 1ms);
    Check(store.GetCounts().loaded == 0, "the file of another build is replaced");
    std::vector<uint32_t> words;
    std::vector<uint8_t> info;
    Check(!store.Find(key_a, words, info), "and holds none of its modules");
  }
}

void TestManyThreads() {
  const auto path = TempPath("threads.v1.modules");
  constexpr uint32_t Threads = 6;
  constexpr uint32_t Each = 40;
  {
    ModuleStore store(path, "build", 1ms);
    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < Threads; t++) {
      threads.emplace_back([&store, t] {
        for (uint32_t i = 0; i < Each; i++) {
          const ModuleStore::Key key{t, i};
          const auto words = Words(500 + i * 37, t * 1000 + i);
          const auto bytes = Bytes(i, static_cast<uint8_t>(t));
          store.Insert(key, words, bytes);
          std::vector<uint32_t> found;
          std::vector<uint8_t> info;
          Check(store.Find(key, found, info) && found == words && info == bytes,
                "a thread finds what it inserted while others insert");
        }
      });
    }
    for (auto &thread : threads) {
      thread.join();
    }
  }
  ModuleStore store(path, "build", 1ms);
  Check(store.GetCounts().loaded == Threads * Each, "every thread's records are in the file");
  for (uint32_t t = 0; t < Threads; t++) {
    for (uint32_t i = 0; i < Each; i++) {
      std::vector<uint32_t> found;
      std::vector<uint8_t> info;
      Check(store.Find({t, i}, found, info) && found == Words(500 + i * 37, t * 1000 + i) &&
                info == Bytes(i, static_cast<uint8_t>(t)),
            "each of them reads back");
    }
  }
}

IR::CompiledShaderInfo SampleInfo() {
  IR::CompiledShaderInfo info;
  info.stage = static_cast<Libs::Graphics::ShaderType>(2);
  info.shader_hash = 0x1122334455667788ull;
  info.wave_size = 32;
  info.user_data_base = 8;
  info.param_export_mask = 0x00ff00f0u;
  info.has_address_writes = true;

  IR::BufferResource buffer;
  buffer.source = 3;
  buffer.descriptor_index = 5;
  buffer.first_use_pc = 0x40;
  buffer.max_byte_extent = 4096;
  buffer.packed_stride = 0x10010;
  buffer.descriptor_format = static_cast<decltype(buffer.descriptor_format)>(7);
  buffer.descriptor_swizzle = 0x688;
  buffer.read = true;
  buffer.atomic = true;
  buffer.scalar = true;
  buffer.indirect_root = 1;
  buffer.indirect_mapping_offset = 12;
  buffer.indirect_search_iterations = 9;
  buffer.indirect_mapping_header = true;
  buffer.indirect_resources = {4, 5, 6};
  buffer.portable_format = true;
  info.info.buffers = {IR::BufferResource{}, buffer};

  IR::ImageResource image;
  image.source = 11;
  image.descriptor_index = 2;
  image.first_use_pc = 0x80;
  image.resource_class = static_cast<IR::ImageResourceClass>(1);
  image.numeric_class = static_cast<decltype(image.numeric_class)>(1);
  image.dimension = static_cast<decltype(image.dimension)>(3);
  image.mip_mode = IR::ImageMipMode::Dynamic;
  image.mip_count = 9;
  image.conversion_format = static_cast<decltype(image.conversion_format)>(3);
  image.shader_swizzle = 0x123;
  image.srgb_decode = static_cast<decltype(image.srgb_decode)>(5);
  image.written = true;
  image.atomic64 = true;
  image.unfiltered_access_only = false;
  image.skip_degamma = true;
  image.r128 = true;
  image.indirect_root = 0;
  image.indirect_resources = {1};
  image.bindless = true;
  image.bindless_byte_keys = true;
  info.info.images = {image, IR::ImageResource{}};

  IR::SamplerResource sampler;
  sampler.source = 6;
  sampler.first_use_pc = 0x90;
  sampler.snapshot_index = 2;
  sampler.indirect_root = 0;
  sampler.indirect_mapping_offset = 3;
  sampler.indirect_search_iterations = 4;
  sampler.indirect_mapping_header = true;
  sampler.indirect_resources = {0, 2};
  sampler.depth_compare = true;
  sampler.gather_lod = true;
  info.info.samplers = {sampler};
  info.info.sampled_pairs = {{.image = 1, .sampler = 0, .first_use_pc = 0x94}};
  info.info.address_bases = {{1u, 2u}, {30u, 31u}};
  info.info.ray_query_tlas = {7u, 8u};
  info.info.inputs = {{.kind = static_cast<IR::StageInputKind>(1),
                       .location = 2,
                       .component_count = 3,
                       .debug_name = "an input",
                       .per_vertex = true}};
  info.info.outputs = {{.kind = IR::StageOutputKind::Parameter,
                        .index = 4,
                        .location = 5,
                        .debug_name = "an output"},
                       {}};
  info.info.user_data_registers = {0, 1, 17};
  info.info.vertex_fetch_components[0] = 3;
  info.info.vertex_fetch_components[31] = 4;
  info.info.vertex_offset_sgpr = 6;
  info.info.instance_offset_sgpr = -1;
  info.info.uses_dma = true;
  info.info.uses_gds = true;
  info.info.uses_image_heap = true;
  info.info.bvh8 = true;
  info.info.subgroup_vote = true;
  info.info.subgroup_local_invocation_id = true;
  info.info.image_gather_extended = true;
  info.info.function_lds = true;
  info.info.function_lds_dwords = 77;
  info.info.pixel_valid_mask = true;
  info.info.buffer_u16 = true;
  info.info.float64 = true;

  info.bindings.descriptor_counts[0] = 2;
  info.bindings.descriptor_counts[info.bindings.descriptor_counts.size() - 1] = 5;
  info.bindings.descriptor_mask = 0x8000000000000003ull;
  info.bindings.push_data_start_dword = 4;
  info.bindings.dispatch_thread_dword = 6;
  info.bindings.ray_query_dword = 8;
  info.bindings.memory_offset_dword = 10;
  info.bindings.memory_strides = true;
  info.bindings.image_swizzles = 3;
  return info;
}

void TestInfo() {
  const auto info = SampleInfo();
  std::vector<uint8_t> bytes = {0xaa};
  IR::SaveCompiledInfo(info, bytes);
  Check(bytes.size() > 1 && bytes[0] == 0xaa, "the bytes are appended");
  const std::span<const uint8_t> saved(bytes.data() + 1, bytes.size() - 1);
  IR::CompiledShaderInfo loaded;
  Check(IR::LoadCompiledInfo(saved, loaded), "an info loads from its bytes");
  Check(IR::SameCompiledInfo(info, loaded), "and is the info that was saved");
  Check(!IR::SameCompiledInfo(info, IR::CompiledShaderInfo{}), "which is not an empty one");

  IR::CompiledShaderInfo empty;
  std::vector<uint8_t> empty_bytes;
  IR::SaveCompiledInfo(empty, empty_bytes);
  IR::CompiledShaderInfo empty_loaded = SampleInfo();
  Check(IR::LoadCompiledInfo(empty_bytes, empty_loaded) &&
            IR::SameCompiledInfo(empty, empty_loaded),
        "an empty info replaces what the target held");

  // Every shorter prefix is refused, and so are bytes after the end.
  for (size_t size = 0; size < saved.size(); size++) {
    IR::CompiledShaderInfo partial;
    Check(!IR::LoadCompiledInfo(saved.first(size), partial), "a cut info does not load");
  }
  auto longer = std::vector<uint8_t>(saved.begin(), saved.end());
  longer.push_back(0);
  IR::CompiledShaderInfo trailing;
  Check(!IR::LoadCompiledInfo(longer, trailing), "bytes after the info are refused");
  auto other = std::vector<uint8_t>(saved.begin(), saved.end());
  other[0] ^= 1u;
  Check(!IR::LoadCompiledInfo(other, trailing), "another format's bytes are refused");

  // A changed field is a different info: each of a few, one at a time.
  auto changed = info;
  changed.info.buffers[1].indirect_resources[2] = 7;
  Check(!IR::SameCompiledInfo(info, changed), "a buffer's indirect resource");
  changed = info;
  changed.info.images[0].bindless_byte_keys = false;
  Check(!IR::SameCompiledInfo(info, changed), "an image's flag");
  changed = info;
  changed.bindings.image_swizzles = 4;
  Check(!IR::SameCompiledInfo(info, changed), "a binding count");
  changed = info;
  changed.info.outputs[0].debug_name = "another";
  Check(!IR::SameCompiledInfo(info, changed), "an output's name");
}

} // namespace

int main() {
  TestAcrossSessions();
  TestManyThreads();
  TestInfo();
  std::printf("ModuleStoreTests: passed\n");
  return 0;
}
