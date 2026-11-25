#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_MODULESTORE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_MODULESTORE_H_

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <span>
#include <stop_token>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

// The modules the shader recompiler compiled, remembered across sessions: a module's SPIR-V and
// the bytes of its info (IR::SaveCompiledInfo), under a key made of everything the compilation
// read. A session translates every program an earlier one recorded and, where memory allows,
// every compute shader of the game, ahead of its first use: for Saros 20,000 compilations, 218 s
// of CPU time and 2.7 GB of SPIR-V in the first minute and a half of every session, beside the
// game, which runs a fifth slower meanwhile. With the store a module is compiled once per build
// of the emulator.
//
// One file: a header naming the build that wrote it (a file of another build is replaced, its
// compiler may differ), then records, each a header, the info's bytes and the SPIR-V words
// compressed. Opening reads the record headers only; a record's bytes are read when it is asked
// for and checked against their checksum (a damaged record counts as absent and is forgotten).
// Records are appended by a writer thread a few seconds after they are inserted, so a session
// that is killed keeps what it wrote until then; a torn last record is dropped when the file is
// opened. Vulkan-free so that it can be tested without a device.
class ModuleStore {
public:
	struct Key {
		uint64_t low                          = 0;
		uint64_t high                         = 0;
		bool     operator==(const Key&) const = default;
	};
	struct Counts {
		uint64_t loaded     = 0; // Records the open found in the file.
		uint64_t found      = 0;
		uint64_t missed     = 0;
		uint64_t inserted   = 0;
		uint64_t damaged    = 0;
		uint64_t file_bytes = 0;
	};

	// Opens (or creates) the file at `path` for the build `build`. A path that cannot be written
	// leaves a store that holds nothing and takes nothing.
	ModuleStore(std::filesystem::path path, std::string build,
	            std::chrono::milliseconds flush_interval = std::chrono::seconds {2});
	~ModuleStore();
	ModuleStore(const ModuleStore&)            = delete;
	ModuleStore& operator=(const ModuleStore&) = delete;

	// The module of `key`: its SPIR-V into `words`, its info's bytes into `info`. False, both
	// untouched or unspecified, if the store has none or its bytes are damaged.
	[[nodiscard]] bool Find(const Key& key, std::vector<uint32_t>& words, std::vector<uint8_t>& info);
	// Whether the store has a record of `key` (its bytes are checked when they are read).
	[[nodiscard]] bool Contains(const Key& key) const;
	// Remembers the module of `key` (a key that is known already keeps its record); the writer
	// thread appends it to the file.
	void Insert(const Key& key, std::span<const uint32_t> words, std::span<const uint8_t> info);
	// Appends every record not written yet (the writer thread calls it, the destructor too).
	void Flush();

	[[nodiscard]] Counts GetCounts() const;

private:
	struct KeyHash {
		size_t operator()(const Key& key) const { return static_cast<size_t>(key.low); }
	};
	struct Entry {
		// Where the record's bytes (info, then the compressed words) begin in the file; unused
		// while `waiting` holds them.
		uint64_t offset       = 0;
		uint32_t words        = 0;
		uint32_t info_bytes   = 0;
		uint32_t stored_bytes = 0; // The compressed words.
		uint64_t checksum     = 0;
		std::shared_ptr<const std::vector<uint8_t>> waiting;
	};

	void Open();
	void WriterThread(std::stop_token stop);
	void Forget(const Key& key);

	std::filesystem::path                    m_path;
	std::string                              m_build;
	std::chrono::milliseconds                m_flush_interval;
	mutable std::mutex                       m_mutex;
	std::unordered_map<Key, Entry, KeyHash>  m_entries;
	std::vector<Key>                         m_pending;
	Counts                                   m_counts;
	bool                                     m_writable = false;
	// The appending end of the file and where its next byte goes.
	std::mutex                  m_file_mutex;
	std::ofstream               m_file;
	uint64_t                    m_file_end = 0;
	// The reading end.
	std::mutex                  m_read_mutex;
	std::ifstream               m_reader;
	std::condition_variable_any m_wake;
	std::jthread                m_writer;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_MODULESTORE_H_ */
