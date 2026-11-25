#include "graphics/host_gpu/renderer/pipeline/moduleStore.h"

#include <array>
#include <cstring>
#include <system_error>
#include <xxhash.h>
#include <zstd.h>

namespace Libs::Graphics {

namespace {

constexpr std::array<char, 8> Magic = {'K', 'y', 'M', 'o', 'd', 'S', 't', '1'};
// A larger file starts over: a title's modules of one build take about 0.6 GB.
constexpr uintmax_t MaxFileSize = uintmax_t {4} << 30u;
// What a record may claim: beyond these its header is damage.
constexpr uint32_t MaxWords     = 64u << 20u;
constexpr uint32_t MaxInfoBytes = 16u << 20u;
// Fast enough to run beside the compilations it follows (their SPIR-V shrinks to about a fifth).
constexpr int CompressionLevel = 3;

struct RecordHeader {
	uint64_t low          = 0;
	uint64_t high         = 0;
	uint32_t words        = 0;
	uint32_t info_bytes   = 0;
	uint32_t stored_bytes = 0;
	uint32_t reserved     = 0;
	uint64_t checksum     = 0;
};
static_assert(sizeof(RecordHeader) == 40);

// Of the info's bytes and the words as they were before compression.
uint64_t Checksum(const uint8_t* info, size_t info_bytes, const uint32_t* words, size_t count) {
	return XXH3_64bits_withSeed(words, count * sizeof(uint32_t), XXH3_64bits(info, info_bytes));
}

} // namespace

ModuleStore::ModuleStore(std::filesystem::path path, std::string build,
                         std::chrono::milliseconds flush_interval)
    : m_path(std::move(path)), m_build(std::move(build)), m_flush_interval(flush_interval) {
	Open();
	if (m_writable) {
		m_writer = std::jthread([this](std::stop_token stop) { WriterThread(stop); });
	}
}

ModuleStore::~ModuleStore() {
	if (m_writer.joinable()) {
		m_writer.request_stop();
		m_wake.notify_all();
		m_writer.join();
	}
	Flush();
}

void ModuleStore::Open() {
	std::error_code error;
	const auto      size      = std::filesystem::exists(m_path, error)
	                                ? std::filesystem::file_size(m_path, error)
	                                : uintmax_t {0};
	uint64_t        valid_end = 0;
	if (!error && size > 0 && size <= MaxFileSize) {
		std::ifstream                  in(m_path, std::ios::binary);
		std::array<char, Magic.size()> magic {};
		uint32_t                       build_size = 0;
		std::string                    build;
		if (in.read(magic.data(), magic.size()) && magic == Magic &&
		    in.read(reinterpret_cast<char*>(&build_size), sizeof(build_size)) &&
		    build_size == m_build.size()) {
			build.resize(build_size);
			if (in.read(build.data(), build_size) && build == m_build) {
				valid_end = magic.size() + sizeof(build_size) + build_size;
			}
		}
		// The headers of the records up to the first one that is incomplete (an interrupted
		// append) or cannot be one; their bytes are checked when they are read.
		while (valid_end != 0) {
			RecordHeader header;
			if (!in.read(reinterpret_cast<char*>(&header), sizeof(header)) ||
			    header.words == 0 || header.words > MaxWords || header.info_bytes > MaxInfoBytes ||
			    uint64_t {header.info_bytes} + header.stored_bytes >
			        size - valid_end - sizeof(header)) {
				break;
			}
			const auto payload = uint64_t {header.info_bytes} + header.stored_bytes;
			// (The last record of a key counts: a session that found one damaged wrote it again.)
			m_entries.insert_or_assign(Key {header.low, header.high},
			                           Entry {.offset       = valid_end + sizeof(header),
			                                  .words        = header.words,
			                                  .info_bytes   = header.info_bytes,
			                                  .stored_bytes = header.stored_bytes,
			                                  .checksum     = header.checksum});
			valid_end += sizeof(header) + payload;
			if (!in.seekg(static_cast<std::streamoff>(valid_end))) {
				break;
			}
		}
	}
	m_counts.loaded = m_entries.size();
	if (valid_end != 0) {
		if (valid_end != size) {
			std::filesystem::resize_file(m_path, valid_end, error);
		}
		if (!error) {
			m_file.open(m_path, std::ios::binary | std::ios::app);
		}
	} else {
		// Missing, another build's or unreadable: start a new file.
		m_entries.clear();
		m_counts.loaded = 0;
		std::filesystem::create_directories(m_path.parent_path(), error);
		m_file.open(m_path, std::ios::binary | std::ios::trunc);
		const auto build_size = static_cast<uint32_t>(m_build.size());
		if (!m_file.write(Magic.data(), Magic.size()) ||
		    !m_file.write(reinterpret_cast<const char*>(&build_size), sizeof(build_size)) ||
		    !m_file.write(m_build.data(), build_size) || !m_file.flush()) {
			m_file.close();
		}
		valid_end = Magic.size() + sizeof(build_size) + build_size;
	}
	if (m_file.is_open()) {
		m_reader.open(m_path, std::ios::binary);
	}
	m_writable = m_file.is_open() && m_reader.is_open();
	if (!m_writable) {
		// Nothing can be read back either: what the open listed is not offered.
		m_entries.clear();
		m_counts.loaded = 0;
	}
	m_file_end        = valid_end;
	m_counts.file_bytes = valid_end;
}

void ModuleStore::Forget(const Key& key) {
	std::lock_guard lock(m_mutex);
	m_counts.damaged++;
	// (A record still waiting to be written stays in the list of the writer, which finds no
	// entry for it and leaves it out.)
	m_entries.erase(key);
}

bool ModuleStore::Find(const Key& key, std::vector<uint32_t>& words, std::vector<uint8_t>& info) {
	Entry entry;
	{
		std::lock_guard lock(m_mutex);
		const auto      it = m_entries.find(key);
		if (it == m_entries.end()) {
			m_counts.missed++;
			return false;
		}
		entry = it->second;
	}
	std::vector<uint8_t> read;
	const auto*          bytes = entry.waiting.get();
	if (bytes == nullptr) {
		read.resize(size_t {entry.info_bytes} + entry.stored_bytes);
		std::lock_guard lock(m_read_mutex);
		m_reader.clear();
		if (!m_reader.seekg(static_cast<std::streamoff>(entry.offset)) ||
		    !m_reader.read(reinterpret_cast<char*>(read.data()),
		                   static_cast<std::streamsize>(read.size()))) {
			Forget(key);
			return false;
		}
		bytes = &read;
	}
	words.resize(entry.words);
	const auto size = ZSTD_decompress(words.data(), words.size() * sizeof(uint32_t),
	                                  bytes->data() + entry.info_bytes, entry.stored_bytes);
	if (ZSTD_isError(size) != 0u || size != words.size() * sizeof(uint32_t) ||
	    Checksum(bytes->data(), entry.info_bytes, words.data(), words.size()) != entry.checksum) {
		Forget(key);
		return false;
	}
	info.assign(bytes->begin(), bytes->begin() + entry.info_bytes);
	std::lock_guard lock(m_mutex);
	m_counts.found++;
	return true;
}

bool ModuleStore::Contains(const Key& key) const {
	std::lock_guard lock(m_mutex);
	return m_entries.contains(key);
}

void ModuleStore::Insert(const Key& key, std::span<const uint32_t> words,
                         std::span<const uint8_t> info) {
	if (words.empty() || words.size() > MaxWords || info.size() > MaxInfoBytes) {
		return;
	}
	{
		std::lock_guard lock(m_mutex);
		if (!m_writable || m_entries.contains(key)) {
			return;
		}
	}
	// Compressed here, by the thread that compiled the module.
	auto       payload = std::make_shared<std::vector<uint8_t>>();
	const auto bound   = ZSTD_compressBound(words.size_bytes());
	payload->resize(info.size() + bound);
	if (!info.empty()) {
		std::memcpy(payload->data(), info.data(), info.size());
	}
	const auto stored = ZSTD_compress(payload->data() + info.size(), bound, words.data(),
	                                  words.size_bytes(), CompressionLevel);
	if (ZSTD_isError(stored) != 0u || stored > UINT32_MAX) {
		return;
	}
	payload->resize(info.size() + stored);
	payload->shrink_to_fit();
	Entry entry {.offset       = 0,
	             .words        = static_cast<uint32_t>(words.size()),
	             .info_bytes   = static_cast<uint32_t>(info.size()),
	             .stored_bytes = static_cast<uint32_t>(stored),
	             .checksum     = Checksum(info.data(), info.size(), words.data(), words.size()),
	             .waiting      = std::move(payload)};
	{
		std::lock_guard lock(m_mutex);
		if (!m_writable || !m_entries.try_emplace(key, std::move(entry)).second) {
			return;
		}
		m_pending.push_back(key);
		m_counts.inserted++;
	}
	m_wake.notify_all();
}

void ModuleStore::Flush() {
	std::lock_guard file_lock(m_file_mutex);
	struct Waiting {
		Key                                         key;
		RecordHeader                                header;
		std::shared_ptr<const std::vector<uint8_t>> bytes;
		uint64_t                                    offset = 0;
	};
	std::vector<Waiting> records;
	{
		std::lock_guard lock(m_mutex);
		for (const auto& key: m_pending) {
			const auto it = m_entries.find(key);
			if (it == m_entries.end() || it->second.waiting == nullptr) {
				continue;
			}
			const auto& entry = it->second;
			records.push_back({key,
			                   RecordHeader {key.low, key.high, entry.words, entry.info_bytes,
			                                 entry.stored_bytes, 0, entry.checksum},
			                   entry.waiting});
		}
		m_pending.clear();
	}
	if (records.empty() || !m_file.is_open()) {
		return;
	}
	auto end = m_file_end;
	for (auto& record: records) {
		m_file.write(reinterpret_cast<const char*>(&record.header), sizeof(record.header));
		m_file.write(reinterpret_cast<const char*>(record.bytes->data()),
		             static_cast<std::streamsize>(record.bytes->size()));
		record.offset = end + sizeof(record.header);
		end += sizeof(record.header) + record.bytes->size();
	}
	const bool written = static_cast<bool>(m_file.flush());
	std::lock_guard lock(m_mutex);
	if (!written) {
		// The file takes no more; what waited is not offered any longer (its bytes may be
		// half in the file), what was written before stays readable.
		m_file.close();
		m_writable = false;
		for (const auto& record: records) {
			m_entries.erase(record.key);
		}
		return;
	}
	m_file_end          = end;
	m_counts.file_bytes = end;
	for (const auto& record: records) {
		if (const auto it = m_entries.find(record.key);
		    it != m_entries.end() && it->second.waiting == record.bytes) {
			// From the file from now on: the bytes leave memory with their last reader.
			it->second.offset = record.offset;
			it->second.waiting.reset();
		}
	}
}

ModuleStore::Counts ModuleStore::GetCounts() const {
	std::lock_guard lock(m_mutex);
	return m_counts;
}

void ModuleStore::WriterThread(std::stop_token stop) {
	while (!stop.stop_requested()) {
		{
			std::unique_lock lock(m_mutex);
			if (!m_wake.wait(lock, stop, [this] { return !m_pending.empty(); })) {
				return;
			}
			// Gather what arrives close together (a run of compilations) into one append.
			(void)m_wake.wait_for(lock, stop, m_flush_interval, [] { return false; });
		}
		Flush();
	}
}

} // namespace Libs::Graphics
