// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "segcore/TextColumnCache.h"

#include <algorithm>
#include <condition_variable>
#include <chrono>
#include <exception>
#include <new>
#include <folly/OperationCancelled.h>
#include <thread>
#include <list>
#include <map>
#include <optional>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/executors/thread_factory/NamedThreadFactory.h>
#include "futures/Future.h"

#include "common/EasyAssert.h"
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/lob_column/lob_reference.h"

namespace milvus::segcore {

namespace {

// One budget for every TextColumnCache/path in this process. This bounds active
// batches, not S3 GETs or decompressed bytes. No cache mutex covers this wait.
class TextLobBatchBudget {
 public:
    void
    Configure(size_t limit) {
        std::lock_guard lock(mutex_);
        limit_ = std::max<size_t>(1, limit);
        cv_.notify_all();
    }
    void
    Acquire(const folly::CancellationToken& token) {
        folly::CancellationCallback cancelled(token, [this] {
            std::lock_guard lock(mutex_);
            cv_.notify_all();
        });
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [&] {
            return active_ < limit_ || token.isCancellationRequested();
        });
        milvus::futures::throwIfCancelled(token);
        ++active_;
    }
    void
    Release() {
        std::lock_guard lock(mutex_);
        --active_;
        cv_.notify_one();
    }

 private:
    std::mutex mutex_;
    std::condition_variable cv_;
    size_t limit_ = 4;
    size_t active_ = 0;
};

TextLobBatchBudget&
BatchBudget() {
    static TextLobBatchBudget budget;
    return budget;
}

struct BatchPermit {
    explicit BatchPermit(const folly::CancellationToken& token) {
        BatchBudget().Acquire(token);
    }
    ~BatchPermit() {
        BatchBudget().Release();
    }
    BatchPermit(const BatchPermit&) = delete;
    BatchPermit&
    operator=(const BatchPermit&) = delete;
};

std::mutex global_config_mutex;
// Preserve the upstream async-read default with bounded file admission.
TextColumnCacheConfig global_config{64, 1, false};

arrow::Result<std::unique_ptr<milvus_storage::lob_column::LobColumnReader>>
DefaultTextLobReaderFactory(
    std::shared_ptr<arrow::fs::FileSystem> fs,
    const milvus_storage::lob_column::LobColumnConfig& config) {
    return milvus_storage::lob_column::CreateLobColumnReader(std::move(fs),
                                                             config);
}

// Separate from the query/load pools whose callers synchronously wait below.
// Workers only open files and run continuations; native scans run on Rust Tokio.
folly::CPUThreadPoolExecutor&
TextLobExecutor() {
    static folly::CPUThreadPoolExecutor executor(
        4, std::make_shared<folly::NamedThreadFactory>("TextLobRead"));
    return executor;
}

using TextResult = arrow::Result<std::vector<std::string>>;

std::vector<std::string>
CheckedTexts(TextResult result, size_t expected) {
    if (!result.ok()) {
        auto error = milvus_storage::ToSegcoreError(result.status());
        ThrowInfo(error.get_error_code(),
                  "Failed to read text batch: {}",
                  error.what());
    }
    auto texts = std::move(result).ValueOrDie();
    AssertInfo(texts.size() == expected,
               "LOB reader returned {} rows, expected {}",
               texts.size(),
               expected);
    return texts;
}

struct FileRead {
    std::vector<milvus_storage::lob_column::EncodedRef> refs;
    std::vector<std::vector<size_t>> outputs;
};

struct ReadSlot {
    size_t file = 0;
    std::optional<folly::Future<TextResult>> future;
};

// The upstream API has no cancellation of admitted native scans. Keep the batch
// permit and resources until every admitted file finishes, even on exceptions.
struct DrainReads {
    std::vector<ReadSlot>& slots;
    ~DrainReads() {
        for (auto& slot : slots) {
            while (slot.future && !slot.future->isReady()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
};

}  // namespace

// Protected by reader_cache_mutex_. Extract evictions so file close/destruction
// never runs under that mutex. The small path cache needs no dependency API patch.
struct TextColumnCache::ReaderCache {
    using Entry = std::pair<std::string, std::shared_ptr<CachedTextLobReader>>;
    explicit ReaderCache(size_t capacity) : capacity(capacity) {
    }
    std::list<Entry> entries;
    size_t capacity;

    std::optional<std::shared_ptr<CachedTextLobReader>>
    get(const std::string& key) {
        auto it = std::find_if(entries.begin(),
                               entries.end(),
                               [&](const auto& e) { return e.first == key; });
        if (it == entries.end()) {
            return std::nullopt;
        }
        entries.splice(entries.begin(), entries, it);
        return entries.front().second;
    }

    std::shared_ptr<CachedTextLobReader>
    put_and_extract(const std::string& key,
                    const std::shared_ptr<CachedTextLobReader>& reader) {
        entries.emplace_front(key, reader);
        if (entries.size() <= capacity) {
            return {};
        }
        auto retired = std::move(entries.back().second);
        entries.pop_back();
        return retired;
    }
};

TextColumnCache::TextColumnCache(const TextColumnCacheConfig& config)
    : TextColumnCache(config, DefaultTextLobReaderFactory) {
}

TextColumnCache::TextColumnCache(const TextColumnCacheConfig& config,
                                 TextLobReaderFactory reader_factory)
    : config_(config),
      reader_factory_(std::move(reader_factory)),
      reader_cache_(std::make_unique<ReaderCache>(config.max_file_readers)) {
}

TextColumnCache::~TextColumnCache() = default;

std::shared_ptr<CachedTextLobReader>
TextColumnCache::GetOrCreateReader(
    const std::string& lob_base_path,
    std::shared_ptr<arrow::fs::FileSystem> fs,
    const std::shared_ptr<const milvus_storage::api::Properties>& properties) {
    {
        std::lock_guard<std::mutex> lock(reader_cache_mutex_);
        auto cached = reader_cache_->get(lob_base_path);
        if (cached.has_value()) {
            file_cache_hits_.fetch_add(1, std::memory_order_relaxed);
            return cached.value();
        }
    }

    file_cache_misses_.fetch_add(1, std::memory_order_relaxed);

    milvus_storage::lob_column::LobColumnConfig config;
    config.lob_base_path = lob_base_path;
    config.properties = properties;

    auto reader_result = reader_factory_(std::move(fs), config);
    if (!reader_result.ok()) {
        auto error = milvus_storage::ToSegcoreError(reader_result.status());
        ThrowInfo(error.get_error_code(),
                  "Failed to create LobColumnReader for {}: {}",
                  lob_base_path,
                  error.what());
    }

    auto reader = std::shared_ptr<milvus_storage::lob_column::LobColumnReader>(
        std::move(reader_result).ValueOrDie().release());
    auto cached_reader =
        std::make_shared<CachedTextLobReader>(std::move(reader));

    std::optional<std::shared_ptr<CachedTextLobReader>> retired;
    {
        std::lock_guard<std::mutex> lock(reader_cache_mutex_);
        auto cached = reader_cache_->get(lob_base_path);
        if (cached.has_value()) {
            return cached.value();
        }
        retired = reader_cache_->put_and_extract(lob_base_path, cached_reader);
    }
    // Eviction may close files; release the last reference outside the lock.

    return cached_reader;
}

std::vector<std::string>
TextColumnCache::ReadBatch(
    const std::string& lob_base_path,
    std::shared_ptr<arrow::fs::FileSystem> fs,
    const std::shared_ptr<const milvus_storage::api::Properties>& properties,
    const std::vector<milvus_storage::lob_column::EncodedRef>& encoded_refs,
    const folly::CancellationToken& cancel_token) try {
    milvus::futures::throwIfCancelled(cancel_token);
    if (encoded_refs.empty()) {
        return {};
    }

    std::vector<std::string> results(encoded_refs.size());
    using namespace milvus_storage::lob_column;
    // Group by file, sort/deduplicate rows and retain every original position.
    // A one-file async call makes the upstream unbounded fan-out safe to admit.
    std::map<std::string, std::map<int32_t, std::vector<size_t>>> grouped;
    for (size_t i = 0; i < encoded_refs.size(); ++i) {
        const auto& ref = encoded_refs[i];
        if (ref.data == nullptr || ref.size == 0) {
            continue;
        }
        if (IsInlineData(ref.data)) {
            results[i] = DecodeInlineText(ref.data, ref.size);
            continue;
        }
        if (ref.size != LOB_REFERENCE_SIZE || !IsLOBReference(ref.data)) {
            ThrowInfo(DataFormatBroken, "Invalid encoded TEXT LOB reference");
        }
        auto decoded = DecodeLOBReference(ref.data);
        if (decoded.row_offset < 0) {
            ThrowInfo(DataFormatBroken, "Negative TEXT LOB row offset");
        }
        grouped[decoded.file_id][decoded.row_offset].push_back(i);
    }
    if (grouped.empty()) {
        return results;
    }
    std::vector<FileRead> files;
    files.reserve(grouped.size());
    for (auto& [id, rows] : grouped) {
        auto& file = files.emplace_back();
        for (auto& [row, positions] : rows) {
            file.refs.push_back(encoded_refs[positions.front()]);
            file.outputs.push_back(std::move(positions));
        }
    }
    auto cached_reader = GetOrCreateReader(lob_base_path, fs, properties);
    std::unique_lock<std::mutex> fallback_lock(cached_reader->mutex,
                                               std::defer_lock);
    if (config_.serialize_reads) {
        while (!fallback_lock.try_lock()) {
            milvus::futures::throwIfCancelled(cancel_token);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    BatchPermit permit(cancel_token);
    auto materialize = [&](size_t index, TextResult result) {
        const auto& file = files[index];
        auto texts = CheckedTexts(std::move(result), file.refs.size());
        for (size_t i = 0; i < texts.size(); ++i) {
            const auto& positions = file.outputs[i];
            for (size_t j = 0; j + 1 < positions.size(); ++j) {
                results[positions[j]] = texts[i];
            }
            results[positions.back()] = std::move(texts[i]);
        }
    };
    if (config_.serialize_reads) {
        for (size_t i = 0; i < files.size(); ++i) {
            milvus::futures::throwIfCancelled(cancel_token);
            materialize(i, cached_reader->reader->ReadBatch(files[i].refs));
        }
    } else {
        auto& executor = TextLobExecutor();
        std::vector<ReadSlot> slots(std::min(
            files.size(), std::max<size_t>(1, config_.max_concurrent_files)));
        DrainReads drain{slots};
        size_t next = 0;
        size_t completed = 0;
        while (completed < files.size()) {
            // Consume all ready results before admitting more work. A failed
            // Result or exceptional future exits through DrainReads unchanged.
            for (auto& slot : slots) {
                if (slot.future && slot.future->isReady()) {
                    auto future = std::move(*slot.future);
                    slot.future.reset();
                    materialize(slot.file, std::move(future).get());
                    ++completed;
                }
            }
            milvus::futures::throwIfCancelled(cancel_token);
            for (auto& slot : slots) {
                if (!slot.future && next < files.size()) {
                    milvus::futures::throwIfCancelled(cancel_token);
                    slot.file = next++;
                    slot.future.emplace(
                        cached_reader->reader
                            ->ReadBatchAsync(files[slot.file].refs)
                            .via(&executor));
                }
            }
            if (completed < files.size()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
    milvus::futures::throwIfCancelled(cancel_token);

    return results;
} catch (const std::bad_alloc& error) {
    throw SegcoreError(ErrorCode::MemAllocateFailed, error.what());
} catch (const folly::OperationCancelled& error) {
    throw SegcoreError(ErrorCode::FollyCancel, error.what());
} catch (const folly::FutureCancellation& error) {
    throw SegcoreError(ErrorCode::FollyCancel, error.what());
} catch (const folly::FutureException& error) {
    throw SegcoreError(ErrorCode::FollyOtherException, error.what());
} catch (const std::exception&) {
    throw;
} catch (...) {
    throw SegcoreError(ErrorCode::UnexpectedError,
                       "Unknown async load exception");
}

void
TextColumnCache::ReadBatchInto(
    const std::string& lob_base_path,
    std::shared_ptr<arrow::fs::FileSystem> fs,
    const std::shared_ptr<const milvus_storage::api::Properties>& properties,
    const std::vector<milvus_storage::lob_column::EncodedRef>& encoded_refs,
    google::protobuf::RepeatedPtrField<std::string>* dst,
    const folly::CancellationToken& cancel_token) {
    AssertInfo(dst != nullptr &&
                   static_cast<size_t>(dst->size()) == encoded_refs.size(),
               "TEXT output must be sized to the reference batch");
    auto texts = ReadBatch(
        lob_base_path, std::move(fs), properties, encoded_refs, cancel_token);
    for (size_t i = 0; i < texts.size(); ++i) {
        *dst->Mutable(i) = std::move(texts[i]);
    }
}

void
TextColumnCache::Clear() {
    auto retired = std::make_unique<ReaderCache>(config_.max_file_readers);
    {
        std::lock_guard<std::mutex> lock(reader_cache_mutex_);
        retired.swap(reader_cache_);
    }
    // Readers held by active calls survive; unused readers close outside the lock.
}

static std::unique_ptr<TextColumnCache> g_text_column_cache;
static std::once_flag g_text_column_cache_init_flag;

void
ConfigureTextLobReadConcurrency(bool serialize_reads,
                                size_t max_batches,
                                size_t max_files) {
    std::lock_guard lock(global_config_mutex);
    // Startup-only settings: existing cached readers are never reconfigured.
    global_config.serialize_reads = serialize_reads;
    global_config.max_concurrent_files = std::max<size_t>(1, max_files);
    BatchBudget().Configure(max_batches);
}

TextColumnCache&
GetGlobalTextColumnCache() {
    std::call_once(g_text_column_cache_init_flag, []() {
        std::lock_guard lock(global_config_mutex);
        g_text_column_cache = std::make_unique<TextColumnCache>(global_config);
    });
    return *g_text_column_cache;
}

}  // namespace milvus::segcore
