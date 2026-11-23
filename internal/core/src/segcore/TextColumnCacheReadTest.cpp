// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <arrow/array.h>
#include <arrow/buffer.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/io/interfaces.h>
#include <arrow/table.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "milvus-storage/format/vortex/vortex_format_reader.h"
#include "milvus-storage/lob_column/lob_column_manager.h"
#include "milvus-storage/lob_column/lob_column_writer.h"
#include "milvus-storage/lob_column/lob_reference.h"
#include "segcore/TextColumnCache.h"
#include "storage/artifact/LocalDirectory.h"

namespace milvus::segcore {
namespace {

struct TextReadRange {
    std::string path;
    int64_t offset;
    int64_t bytes;
};

struct TextReadSnapshot {
    size_t opens = 0;
    std::vector<TextReadRange> ranges;
    size_t peak_reads = 0;
    size_t peak_files = 0;
    double read_seconds = 0;

    int64_t
    Bytes() const {
        int64_t bytes = 0;
        for (const auto& range : ranges) {
            bytes += range.bytes;
        }
        return bytes;
    }
};

class TextReadTrace {
 public:
    std::atomic<int> delay_ms{0};

    void
    PauseNextRead() {
        std::lock_guard<std::mutex> lock(mutex_);
        pause_next_ = true;
        paused_ = false;
        released_ = false;
    }
    bool
    WaitForPausedRead() {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(
            lock, std::chrono::seconds(10), [&] { return paused_; });
    }
    bool
    WaitForPeakFiles(size_t count, std::chrono::milliseconds timeout) {
        std::unique_lock lock(mutex_);
        return cv_.wait_for(
            lock, timeout, [&] { return snapshot_.peak_files >= count; });
    }

    void
    ReleaseRead() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        cv_.notify_all();
    }

    void
    Begin(const std::string& path) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            ++active_reads_;
            ++active_files_[path];
            snapshot_.peak_reads =
                std::max(snapshot_.peak_reads, active_reads_);
            snapshot_.peak_files =
                std::max(snapshot_.peak_files, active_files_.size());
            cv_.notify_all();
            if (pause_next_) {
                pause_next_ = false;
                paused_ = true;
                cv_.notify_all();
                cv_.wait(lock, [&] { return released_; });
            }
        }
        const auto delay = delay_ms.load();
        if (delay != 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay));
        }
    }

    void
    End(const std::string& path, double seconds) {
        std::lock_guard<std::mutex> lock(mutex_);
        --active_reads_;
        if (--active_files_.at(path) == 0) {
            active_files_.erase(path);
        }
        snapshot_.read_seconds += seconds;
    }

    void
    Open() {
        std::lock_guard<std::mutex> lock(mutex_);
        ++snapshot_.opens;
    }

    void
    Read(const std::string& path, int64_t offset, int64_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshot_.ranges.push_back({path, offset, bytes});
    }

    TextReadSnapshot
    Take() {
        std::lock_guard<std::mutex> lock(mutex_);
        EXPECT_EQ(active_reads_, 0);
        auto result = std::move(snapshot_);
        snapshot_ = {};
        return result;
    }

 private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool pause_next_ = false;
    bool paused_ = false;
    bool released_ = true;
    TextReadSnapshot snapshot_;
    size_t active_reads_ = 0;
    std::map<std::string, size_t> active_files_;
};

class TimedTextRead {
 public:
    TimedTextRead(TextReadTrace& trace, const std::string& path)
        : trace_(trace), path_(path), start_(std::chrono::steady_clock::now()) {
        trace_.Begin(path_);
    }
    ~TimedTextRead() {
        trace_.End(path_,
                   std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - start_)
                       .count());
    }

 private:
    TextReadTrace& trace_;
    const std::string& path_;
    std::chrono::steady_clock::time_point start_;
};

// Count logical filesystem reads, even when the OS serves them from page cache.
// Both ReadAt overloads are covered; Arrow's default ReadAsync also calls ReadAt.
class CountingTextFile : public arrow::io::RandomAccessFile {
 public:
    CountingTextFile(std::shared_ptr<arrow::io::RandomAccessFile> file,
                     std::shared_ptr<TextReadTrace> trace,
                     std::string path)
        : file_(std::move(file)),
          trace_(std::move(trace)),
          path_(std::move(path)) {
    }

    arrow::Status
    Close() override {
        return file_->Close();
    }
    bool
    closed() const override {
        return file_->closed();
    }
    arrow::Result<int64_t>
    Tell() const override {
        return file_->Tell();
    }
    arrow::Status
    Seek(int64_t offset) override {
        return file_->Seek(offset);
    }
    arrow::Result<int64_t>
    GetSize() override {
        return file_->GetSize();
    }

    arrow::Result<int64_t>
    Read(int64_t bytes, void* out) override {
        TimedTextRead timed(*trace_, path_);
        ARROW_ASSIGN_OR_RAISE(auto offset, file_->Tell());
        ARROW_ASSIGN_OR_RAISE(auto read, file_->Read(bytes, out));
        trace_->Read(path_, offset, read);
        return read;
    }

    arrow::Result<std::shared_ptr<arrow::Buffer>>
    Read(int64_t bytes) override {
        TimedTextRead timed(*trace_, path_);
        ARROW_ASSIGN_OR_RAISE(auto offset, file_->Tell());
        ARROW_ASSIGN_OR_RAISE(auto buffer, file_->Read(bytes));
        trace_->Read(path_, offset, buffer->size());
        return buffer;
    }

    arrow::Result<int64_t>
    ReadAt(int64_t offset, int64_t bytes, void* out) override {
        TimedTextRead timed(*trace_, path_);
        ARROW_ASSIGN_OR_RAISE(auto read, file_->ReadAt(offset, bytes, out));
        trace_->Read(path_, offset, read);
        return read;
    }

    arrow::Result<std::shared_ptr<arrow::Buffer>>
    ReadAt(int64_t offset, int64_t bytes) override {
        TimedTextRead timed(*trace_, path_);
        ARROW_ASSIGN_OR_RAISE(auto buffer, file_->ReadAt(offset, bytes));
        trace_->Read(path_, offset, buffer->size());
        return buffer;
    }

 private:
    std::shared_ptr<arrow::io::RandomAccessFile> file_;
    std::shared_ptr<TextReadTrace> trace_;
    std::string path_;
};

class CountingTextFileSystem : public arrow::fs::LocalFileSystem {
 public:
    explicit CountingTextFileSystem(std::shared_ptr<TextReadTrace> trace)
        : trace_(std::move(trace)) {
    }

    arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>>
    OpenInputFile(const std::string& path) override {
        ARROW_ASSIGN_OR_RAISE(auto file,
                              arrow::fs::LocalFileSystem::OpenInputFile(path));
        trace_->Open();
        return std::make_shared<CountingTextFile>(
            std::move(file), trace_, path);
    }

    arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>>
    OpenInputFile(const arrow::fs::FileInfo& info) override {
        return OpenInputFile(info.path());
    }

 private:
    std::shared_ptr<TextReadTrace> trace_;
};

// Exclude the union of ranges observed during metadata-only open. No hardcoded
// footer length or assumption that every ReadAt is a payload read is needed.
int64_t
BytesOutsideMetadata(const TextReadSnapshot& reads,
                     const TextReadSnapshot& metadata) {
    int64_t bytes = 0;
    for (const auto& read : reads.ranges) {
        std::vector<std::pair<int64_t, int64_t>> covered;
        const auto end = read.offset + read.bytes;
        for (const auto& meta : metadata.ranges) {
            if (meta.path == read.path) {
                auto begin = std::max(read.offset, meta.offset);
                auto stop = std::min(end, meta.offset + meta.bytes);
                if (begin < stop) {
                    covered.emplace_back(begin, stop);
                }
            }
        }
        std::sort(covered.begin(), covered.end());
        auto cursor = read.offset;
        for (const auto& [begin, stop] : covered) {
            bytes += std::max<int64_t>(0, begin - cursor);
            cursor = std::max(cursor, stop);
        }
        bytes += end - cursor;
    }
    return bytes;
}

class TextColumnCacheReadTest : public ::testing::TestWithParam<bool> {
 protected:
    std::vector<std::string>
    Read(TextColumnCache& cache,
         const std::shared_ptr<arrow::fs::FileSystem>& fs,
         const milvus_storage::lob_column::LobColumnConfig& config,
         const std::vector<milvus_storage::lob_column::EncodedRef>& refs) {
        if (!GetParam()) {
            return cache.ReadBatch(
                config.lob_base_path, fs, config.properties, refs);
        }
        google::protobuf::RepeatedPtrField<std::string> dst;
        for (size_t i = 0; i < refs.size(); ++i) {
            dst.Add();
        }
        cache.ReadBatchInto(
            config.lob_base_path, fs, config.properties, refs, &dst);
        return {dst.begin(), dst.end()};
    }

    void
    RecordReads(const std::string& name,
                const TextReadSnapshot& snapshot,
                const TextReadSnapshot& metadata) {
        RecordProperty(name + "_opens", std::to_string(snapshot.opens));
        RecordProperty(name + "_read_calls",
                       std::to_string(snapshot.ranges.size()));
        RecordProperty(name + "_read_bytes", std::to_string(snapshot.Bytes()));
        RecordProperty(
            name + "_non_metadata_bytes",
            std::to_string(BytesOutsideMetadata(snapshot, metadata)));
    }
};

INSTANTIATE_TEST_SUITE_P(ReadAPIs,
                         TextColumnCacheReadTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                             return info.param ? "ReadBatchInto" : "ReadBatch";
                         });

TEST_P(TextColumnCacheReadTest, RepeatedLobReadsReuseReadersButReadDataAgain) {
    using namespace milvus_storage::lob_column;
    auto directory = storage::LocalDirectory::CreateOwned(
        std::filesystem::temp_directory_path().string(),
        "text-cache-read-XXXXXX",
        "TEXT cache test");
    auto trace = std::make_shared<TextReadTrace>();
    auto fs = std::make_shared<CountingTextFileSystem>(trace);
    LobColumnConfig config;
    config.lob_base_path = directory->Path() + "/lobs/105";
    config.field_id = 105;
    config.inline_threshold = 64 * 1024;
    config.max_lob_file_bytes = 8 * 1024 * 1024;
    config.flush_threshold_bytes = 4 * 1024 * 1024;

    // Deterministic, poorly compressible UTF-8 data keeps the LOB payload much
    // larger than the initial footer read; tiny/repeated strings can hide I/O.
    std::mt19937 generator(20261015);
    std::vector<std::string> texts(8, std::string(128 * 1024, ' '));
    for (auto& text : texts) {
        for (auto& ch : text) {
            ch = static_cast<char>('!' + generator() % 90);
        }
    }
    auto manager_result = LobColumnManager::Create(fs, config);
    ASSERT_TRUE(manager_result.ok()) << manager_result.status();
    auto manager = std::move(manager_result).ValueOrDie();
    auto writer_result = manager->CreateWriter();
    ASSERT_TRUE(writer_result.ok()) << writer_result.status();
    auto writer = std::move(writer_result).ValueOrDie();
    auto encoded_result = writer->WriteBatch(texts);
    ASSERT_TRUE(encoded_result.ok()) << encoded_result.status();
    auto encoded = std::move(encoded_result).ValueOrDie();
    auto files_result = writer->Close();
    ASSERT_TRUE(files_result.ok()) << files_result.status();
    auto files = std::move(files_result).ValueOrDie();
    ASSERT_EQ(files.size(), 1);
    ASSERT_EQ(encoded.size(), texts.size());
    std::vector<EncodedRef> refs;
    for (const auto& ref : encoded) {
        ASSERT_EQ(ref.size(), LOB_REFERENCE_SIZE);
        ASSERT_EQ(ref.front(), FLAG_LOB_REFERENCE);
        refs.push_back({ref.data(), ref.size()});
    }

    trace->Take();  // Exclude any writer-side activity.
    {
        milvus_storage::vortex::VortexFormatReader metadata_reader(
            fs, nullptr, files.front().path, config.properties, {});
        auto status = metadata_reader.open();
        ASSERT_TRUE(status.ok()) << status;
    }
    const auto metadata = trace->Take();
    ASSERT_GT(metadata.Bytes(), 0);

    TextColumnCache cache;
    ASSERT_EQ(Read(cache, fs, config, refs), texts);
    const auto first = trace->Take();
    RecordReads("first", first, metadata);
    ASSERT_GT(first.opens, 0);
    ASSERT_GT(BytesOutsideMetadata(first, metadata), 0);

    // Same cache, reader, file and rows, with all previous result strings gone.
    // This pins current behavior, not a desired content-cache guarantee.
    for (const auto& name : {"second", "third"}) {
        ASSERT_EQ(Read(cache, fs, config, refs), texts);
        const auto again = trace->Take();
        RecordReads(name, again, metadata);
        EXPECT_EQ(again.opens, 0) << "file readers should be reused";
        EXPECT_GT(BytesOutsideMetadata(again, metadata), 0)
            << "current LOB path caches readers, not previously read payload";
    }

    // Control: clearing the reader cache must cause the file to be opened again.
    cache.Clear();
    ASSERT_EQ(Read(cache, fs, config, refs), texts);
    const auto reopened = trace->Take();
    RecordReads("after_clear", reopened, metadata);
    EXPECT_GT(reopened.opens, 0);
    EXPECT_GT(BytesOutsideMetadata(reopened, metadata), 0);
}

TEST_P(TextColumnCacheReadTest, SharedVortexReaderActuallyOverlapsReads) {
    using namespace milvus_storage::lob_column;
    auto directory = storage::LocalDirectory::CreateOwned(
        std::filesystem::temp_directory_path().string(),
        "text-cache-concurrent-XXXXXX",
        "TEXT concurrent read test");
    auto trace = std::make_shared<TextReadTrace>();
    auto fs = std::make_shared<CountingTextFileSystem>(trace);
    LobColumnConfig config;
    config.lob_base_path = directory->Path() + "/lobs/105";
    config.inline_threshold = 64 * 1024;
    config.max_lob_file_bytes = 8 * 1024 * 1024;
    config.flush_threshold_bytes = 512 * 1024;
    std::mt19937 random(20261117);
    std::vector<std::string> texts(16, std::string(128 * 1024, ' '));
    for (auto& text : texts) {
        for (auto& ch : text) {
            ch = static_cast<char>('!' + random() % 90);
        }
    }
    auto writer_result = CreateLobColumnWriter(fs, config);
    ASSERT_TRUE(writer_result.ok()) << writer_result.status();
    auto writer = std::move(writer_result).ValueOrDie();
    auto encoded = writer->WriteBatch(texts);
    ASSERT_TRUE(encoded.ok()) << encoded.status();
    auto files = writer->Close();
    ASSERT_TRUE(files.ok()) << files.status();
    ASSERT_EQ(files->size(), 1);
    const std::vector<EncodedRef> first_refs{
        {(*encoded)[0].data(), (*encoded)[0].size()}};
    const std::vector<EncodedRef> second_refs{
        {(*encoded)[8].data(), (*encoded)[8].size()}};
    TextColumnCache cache;
    EXPECT_EQ(Read(cache, fs, config, first_refs),
              std::vector<std::string>{texts[0]});
    trace->Take();
    trace->PauseNextRead();
    auto first = std::async(std::launch::async, [&] {
        return Read(cache, fs, config, first_refs);
    });
    struct ReleaseRead {
        TextReadTrace& trace;
        ~ReleaseRead() {
            trace.ReleaseRead();
        }
    } release{*trace};
    ASSERT_TRUE(trace->WaitForPausedRead());
    auto second = std::async(std::launch::async, [&] {
        return Read(cache, fs, config, second_refs);
    });
    const bool overlaps =
        second.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
    trace->ReleaseRead();
    EXPECT_EQ(first.get(), std::vector<std::string>{texts[0]});
    EXPECT_EQ(second.get(), std::vector<std::string>{texts[8]});
    EXPECT_TRUE(overlaps);
    const auto reads = trace->Take();
    EXPECT_EQ(reads.opens, 0)
        << "both calls must share the already-open file reader";
    EXPECT_GE(reads.peak_reads, 2);
}

TEST_P(TextColumnCacheReadTest, AsyncWindowBoundsRealVortexFileReads) {
    using namespace milvus_storage::lob_column;
    auto directory = storage::LocalDirectory::CreateOwned(
        std::filesystem::temp_directory_path().string(),
        "text-cache-files-XXXXXX",
        "TEXT file window test");
    auto trace = std::make_shared<TextReadTrace>();
    auto fs = std::make_shared<CountingTextFileSystem>(trace);
    LobColumnConfig config;
    config.lob_base_path = directory->Path() + "/lobs/105";
    config.inline_threshold = 64 * 1024;
    config.max_lob_file_bytes = 8 * 1024 * 1024;
    config.flush_threshold_bytes = 512 * 1024;
    std::mt19937 random(20261123);
    std::vector<std::vector<uint8_t>> encoded;
    std::vector<std::string> expected;
    for (int file = 0; file < 3; ++file) {
        std::vector<std::string> texts(8, std::string(128 * 1024, ' '));
        for (auto& text : texts) {
            for (auto& ch : text) ch = static_cast<char>('!' + random() % 90);
        }
        auto writer_result = CreateLobColumnWriter(fs, config);
        ASSERT_TRUE(writer_result.ok()) << writer_result.status();
        auto writer = std::move(writer_result).ValueOrDie();
        auto refs = writer->WriteBatch(texts);
        ASSERT_TRUE(refs.ok()) << refs.status();
        auto files = writer->Close();
        ASSERT_TRUE(files.ok()) << files.status();
        ASSERT_EQ(files->size(), 1);
        encoded.push_back((*refs)[4]);
        expected.push_back(texts[4]);
    }
    std::vector<EncodedRef> refs;
    for (const auto& ref : encoded) refs.push_back({ref.data(), ref.size()});
    for (size_t width : {1, 2}) {
        SCOPED_TRACE(width);
        TextColumnCacheConfig cache_config;
        cache_config.max_concurrent_files = width;
        TextColumnCache cache(cache_config);
        EXPECT_EQ(Read(cache, fs, config, refs), expected);  // warm the handles
        trace->Take();
        trace->PauseNextRead();
        auto task = std::async(std::launch::async,
                               [&] { return Read(cache, fs, config, refs); });
        const bool paused = trace->WaitForPausedRead();
        const bool overlaps =
            trace->WaitForPeakFiles(2,
                                    width == 1 ? std::chrono::milliseconds(100)
                                               : std::chrono::seconds(10));
        trace->ReleaseRead();
        EXPECT_EQ(task.get(), expected);
        const auto reads = trace->Take();
        EXPECT_TRUE(paused);
        EXPECT_EQ(overlaps, width > 1);
        EXPECT_EQ(reads.opens, 0);
        EXPECT_LE(reads.peak_files, width);
    }
}

TEST_P(TextColumnCacheReadTest, InlineTextNeedsNoFileReads) {
    using namespace milvus_storage::lob_column;
    auto trace = std::make_shared<TextReadTrace>();
    auto fs = std::make_shared<CountingTextFileSystem>(trace);
    LobColumnConfig config;
    // No such file or directory is created: inline values must be self-contained.
    config.lob_base_path = "unused-inline-lob-path";
    const std::string text = "inline text";
    auto encoded = EncodeInlineText(text);
    std::vector<EncodedRef> refs{{encoded.data(), encoded.size()}};
    TextColumnCache cache;
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(Read(cache, fs, config, refs),
                  std::vector<std::string>{text});
    }
    auto snapshot = trace->Take();
    EXPECT_EQ(snapshot.opens, 0);
    EXPECT_TRUE(snapshot.ranges.empty());
}

}  // namespace
}  // namespace milvus::segcore
