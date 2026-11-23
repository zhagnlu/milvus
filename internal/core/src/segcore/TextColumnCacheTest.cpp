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
#include "segcore/ChunkedSegmentSealedImpl.h"

#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/status.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <future>
#include <memory>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "common/EasyAssert.h"
#include "futures/Future.h"
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/lob_column/lob_reference.h"

namespace milvus::segcore {
namespace {

using milvus_storage::lob_column::EncodedRef;
using milvus_storage::lob_column::FLAG_LOB_REFERENCE;
using milvus_storage::lob_column::LOB_REFERENCE_SIZE;
using milvus_storage::lob_column::LobColumnConfig;
using milvus_storage::lob_column::LobColumnReader;

class BlockingCallTracker {
 public:
    void
    ObserveCall() {
        std::unique_lock<std::mutex> lock(mutex_);
        ++calls_;
        ++active_;
        peak_ = std::max(peak_, active_);
        cv_.notify_all();
        cv_.wait(lock, [&] { return released_; });
        --active_;
    }

    folly::SemiFuture<folly::Unit>
    ObserveAsync() {
        folly::Promise<folly::Unit> promise;
        auto future = promise.getSemiFuture();
        {
            std::lock_guard lock(mutex_);
            ++calls_;
            ++active_;
            peak_ = std::max(peak_, active_);
            if (!released_) {
                pending_.push_back(std::move(promise));
            } else {
                --active_;
                promise.setValue();
            }
        }
        cv_.notify_all();
        return future;
    }

    bool
    WaitForCalls(int count, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, timeout, [&] { return calls_ >= count; });
    }

    void
    Release() {
        std::vector<folly::Promise<folly::Unit>> pending;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released_ = true;
            pending.swap(pending_);
            active_ -= pending.size();
        }
        cv_.notify_all();
        for (auto& promise : pending) {
            promise.setValue();
        }
    }

    int
    Peak() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return peak_;
    }

    int
    CallCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return calls_;
    }

 private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool released_{false};
    std::vector<folly::Promise<folly::Unit>> pending_;
    int calls_{0};
    int active_{0};
    int peak_{0};
};

std::vector<uint8_t>
Bytes(const std::string& value) {
    return {value.begin(), value.end()};
}

class InstrumentedLobColumnReader : public LobColumnReader {
 public:
    explicit InstrumentedLobColumnReader(
        std::shared_ptr<BlockingCallTracker> tracker,
        std::shared_ptr<BlockingCallTracker> destruction = nullptr)
        : tracker_(std::move(tracker)), destruction_(std::move(destruction)) {
    }

    ~InstrumentedLobColumnReader() override {
        if (destruction_) {
            destruction_->ObserveCall();
        }
    }

    arrow::Result<std::vector<uint8_t>>
    ReadData(const uint8_t*, size_t) override {
        tracker_->ObserveCall();
        return Bytes("mock-text");
    }

    arrow::Result<std::vector<std::vector<uint8_t>>>
    ReadBatchData(const std::vector<EncodedRef>& encoded_refs) override {
        tracker_->ObserveCall();
        return std::vector<std::vector<uint8_t>>(encoded_refs.size(),
                                                 Bytes("mock-text"));
    }

    folly::SemiFuture<arrow::Result<std::vector<std::string>>>
    ReadBatchAsync(const std::vector<EncodedRef>& refs) override {
        return tracker_->ObserveAsync().deferValue(
            [count = refs.size()](
                folly::Unit) -> arrow::Result<std::vector<std::string>> {
                return std::vector<std::string>(count, "mock-text");
            });
    }

    arrow::Result<std::shared_ptr<arrow::BinaryArray>>
    ReadArrowArray(const std::shared_ptr<arrow::BinaryArray>&) override {
        return arrow::Status::NotImplemented("unused");
    }

    arrow::Result<std::vector<std::vector<uint8_t>>>
    TakeData(const std::string&, const std::vector<int32_t>&) override {
        return arrow::Status::NotImplemented("unused");
    }

    arrow::Status
    Close() override {
        closed_ = true;
        return arrow::Status::OK();
    }

    bool
    IsClosed() const override {
        return closed_;
    }

    void
    ClearCache() override {
    }

 private:
    std::shared_ptr<BlockingCallTracker> tracker_;
    std::shared_ptr<BlockingCallTracker> destruction_;
    bool closed_{false};
};

TextLobReaderFactory
MakeReaderFactory(std::shared_ptr<BlockingCallTracker> tracker,
                  std::shared_ptr<std::atomic<int>> factory_calls) {
    return [tracker = std::move(tracker),
            factory_calls = std::move(factory_calls)](
               std::shared_ptr<arrow::fs::FileSystem>, const LobColumnConfig&)
               -> arrow::Result<std::unique_ptr<LobColumnReader>> {
        factory_calls->fetch_add(1, std::memory_order_relaxed);
        std::unique_ptr<LobColumnReader> reader =
            std::make_unique<InstrumentedLobColumnReader>(tracker);
        return std::move(reader);
    };
}

std::vector<uint8_t>
MakeLobReference() {
    std::vector<uint8_t> ref(LOB_REFERENCE_SIZE, 0);
    ref[0] = FLAG_LOB_REFERENCE;
    return ref;
}

template <typename Task>
void
ExpectConcurrentReads(const std::shared_ptr<BlockingCallTracker>& tracker,
                      Task task) {
    auto first = std::async(std::launch::async, task);
    auto second = std::async(std::launch::async, task);
    const bool overlaps = tracker->WaitForCalls(2, std::chrono::seconds(5));
    // Release before any assertion/get, including the failure case.
    tracker->Release();
    first.get();
    second.get();
    EXPECT_TRUE(overlaps) << "calls to the same cached reader were serialized";
}

using AsyncTextResult = arrow::Result<std::vector<std::string>>;

// Promise-backed native reads let tests finish any file independently, without
// occupying executor threads or relying on artificial read latency.
struct ControlledReads {
    struct Pending {
        folly::Promise<AsyncTextResult> promise;
        std::vector<std::string> values;
    };
    std::mutex mutex;
    std::condition_variable cv;
    std::map<int, Pending> pending;
    std::vector<int> admitted;
    bool complete_new = false;

    folly::SemiFuture<AsyncTextResult>
    Submit(const std::vector<EncodedRef>& refs) {
        std::lock_guard lock(mutex);
        const int file = refs.front().data[4];
        Pending read;
        int32_t previous = -1;
        for (const auto& ref : refs) {
            const auto decoded =
                milvus_storage::lob_column::DecodeLOBReference(ref.data);
            EXPECT_EQ(ref.data[4], file);
            EXPECT_GT(decoded.row_offset, previous);  // sorted and unique
            previous = decoded.row_offset;
            read.values.push_back(std::to_string(file) + ":" +
                                  std::to_string(previous));
        }
        auto future = read.promise.getSemiFuture();
        admitted.push_back(file);
        if (complete_new) {
            read.promise.setValue(AsyncTextResult(std::move(read.values)));
        } else {
            EXPECT_TRUE(pending.emplace(file, std::move(read)).second);
        }
        cv.notify_all();
        return future;
    }

    bool
    WaitFor(size_t count,
            std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
        std::unique_lock lock(mutex);
        return cv.wait_for(
            lock, timeout, [&] { return admitted.size() >= count; });
    }

    size_t
    Count() {
        std::lock_guard lock(mutex);
        return admitted.size();
    }

    void
    Finish(int file, int failure = 0) {
        std::lock_guard lock(mutex);
        auto it = pending.find(file);
        if (it == pending.end()) {
            ADD_FAILURE() << "file was not admitted: " << file;
            return;
        }
        auto& read = it->second;
        if (failure == 1) {
            read.promise.setValue(
                AsyncTextResult(milvus_storage::MakeExtendError(
                    milvus_storage::ExtendStatusCode::StorageTransientTimeout,
                    "injected read timeout")));
        } else if (failure == 2) {
            read.promise.setException(
                std::runtime_error("injected backend exception"));
        } else if (failure == 3) {
            read.promise.setValue(AsyncTextResult(std::vector<std::string>{}));
        } else {
            read.promise.setValue(AsyncTextResult(std::move(read.values)));
        }
        pending.erase(it);
    }

    void
    FinishAll() {
        std::lock_guard lock(mutex);
        complete_new = true;
        for (auto& [id, read] : pending) {
            read.promise.setValue(AsyncTextResult(std::move(read.values)));
        }
        pending.clear();
    }
};

class ControlledLobReader : public InstrumentedLobColumnReader {
 public:
    explicit ControlledLobReader(std::shared_ptr<ControlledReads> reads)
        : InstrumentedLobColumnReader(std::make_shared<BlockingCallTracker>()),
          reads_(std::move(reads)) {
    }

    folly::SemiFuture<AsyncTextResult>
    ReadBatchAsync(const std::vector<EncodedRef>& refs) override {
        return reads_->Submit(refs);
    }

 private:
    std::shared_ptr<ControlledReads> reads_;
};

TextLobReaderFactory
ControlledFactory(const std::shared_ptr<ControlledReads>& reads) {
    return
        [reads](std::shared_ptr<arrow::fs::FileSystem>, const LobColumnConfig&)
            -> arrow::Result<std::unique_ptr<LobColumnReader>> {
            return std::unique_ptr<LobColumnReader>(
                std::make_unique<ControlledLobReader>(reads));
        };
}

std::vector<uint8_t>
FileRef(uint8_t file, int32_t row) {
    uint8_t uuid[16] = {};
    uuid[0] = file;
    return milvus_storage::lob_column::EncodeLOBReference(uuid, row);
}

TEST(TextColumnCache, AsyncWindowReplenishesBeforeSlowFirstFileCompletes) {
    for (bool into : {false, true}) {
        auto reads = std::make_shared<ControlledReads>();
        TextColumnCacheConfig config;
        config.max_concurrent_files = 2;
        TextColumnCache cache(config, ControlledFactory(reads));
        auto properties =
            std::make_shared<const milvus_storage::api::Properties>();
        const std::vector<std::vector<uint8_t>> bytes{
            FileRef(4, 3),
            FileRef(1, 8),
            FileRef(2, 5),
            FileRef(1, 2),
            FileRef(3, 0),
            FileRef(1, 8),
            milvus_storage::lob_column::EncodeInlineText("inline"),
            {}};
        std::vector<EncodedRef> refs;
        for (const auto& ref : bytes) refs.push_back({ref.data(), ref.size()});
        auto task = std::async(std::launch::async, [&] {
            if (!into)
                return cache.ReadBatch("path", nullptr, properties, refs);
            google::protobuf::RepeatedPtrField<std::string> dst;
            for (size_t i = 0; i < refs.size(); ++i) dst.Add();
            cache.ReadBatchInto("path", nullptr, properties, refs, &dst);
            return std::vector<std::string>(dst.begin(), dst.end());
        });
        const bool initial = reads->WaitFor(2);
        const bool bounded = !reads->WaitFor(3, std::chrono::milliseconds(50));
        reads->Finish(2);
        const bool replenished = reads->WaitFor(3);
        reads->Finish(3);
        const bool last = reads->WaitFor(4);
        const bool slow_pending =
            task.wait_for(std::chrono::milliseconds(20)) !=
            std::future_status::ready;
        reads->FinishAll();
        EXPECT_EQ(task.get(),
                  (std::vector<std::string>{
                      "4:3", "1:8", "2:5", "1:2", "3:0", "1:8", "inline", ""}));
        EXPECT_TRUE(initial);
        EXPECT_TRUE(bounded);
        EXPECT_TRUE(replenished);
        EXPECT_TRUE(last);
        EXPECT_TRUE(slow_pending);
        EXPECT_EQ(reads->Count(), 4);
    }
}

TEST(TextColumnCache,
     AsyncFailureAndCancellationDrainWithoutAdmittingMoreFiles) {
    for (int failure : {0, 1, 2, 3}) {
        SCOPED_TRACE(failure);
        auto reads = std::make_shared<ControlledReads>();
        TextColumnCacheConfig config;
        config.max_concurrent_files = 2;
        TextColumnCache cache(config, ControlledFactory(reads));
        auto properties =
            std::make_shared<const milvus_storage::api::Properties>();
        const std::vector<std::vector<uint8_t>> bytes{
            FileRef(1, 0), FileRef(2, 0), FileRef(3, 0)};
        std::vector<EncodedRef> refs;
        for (const auto& ref : bytes) refs.push_back({ref.data(), ref.size()});
        folly::CancellationSource source;
        auto task = std::async(std::launch::async, [&] {
            return cache.ReadBatch(
                "path", nullptr, properties, refs, source.getToken());
        });
        const bool admitted = reads->WaitFor(2);
        if (failure == 0)
            source.requestCancellation();
        else
            reads->Finish(2, failure);
        const bool draining = task.wait_for(std::chrono::milliseconds(100)) !=
                              std::future_status::ready;
        const auto count = reads->Count();
        reads->FinishAll();
        if (failure == 0) {
            EXPECT_THROW(task.get(), folly::FutureCancellation);
        } else if (failure == 2) {
            try {
                task.get();
                FAIL() << "expected backend exception";
            } catch (const std::runtime_error& error) {
                EXPECT_STREQ(error.what(), "injected backend exception");
            }
        } else {
            try {
                task.get();
                FAIL() << "expected read error";
            } catch (const milvus::SegcoreError& error) {
                if (failure == 1)
                    EXPECT_EQ(error.get_error_code(),
                              milvus::ErrorCode::StorageTransientError);
            }
        }
        EXPECT_TRUE(admitted);
        EXPECT_TRUE(draining);
        EXPECT_EQ(count, 2);
        // A fresh call must still work after unwinding and releasing the permit.
        EXPECT_EQ(cache.ReadBatch("path", nullptr, properties, {refs[0]}),
                  (std::vector<std::string>{"1:0"}));
    }
}

TEST(TextColumnCache, RejectsMalformedReferencesBeforeAnyFileAdmission) {
    auto reads = std::make_shared<ControlledReads>();
    TextColumnCache cache(TextColumnCacheConfig{}, ControlledFactory(reads));
    auto properties = std::make_shared<const milvus_storage::api::Properties>();
    const auto valid = FileRef(1, 0);
    auto bad_flag = valid;
    bad_flag[0] = 2;
    for (auto invalid : {std::vector<uint8_t>{1}, bad_flag, FileRef(1, -1)}) {
        try {
            cache.ReadBatch("path",
                            nullptr,
                            properties,
                            {{valid.data(), valid.size()},
                             {invalid.data(), invalid.size()}});
            FAIL() << "expected malformed reference error";
        } catch (const milvus::SegcoreError& error) {
            EXPECT_EQ(error.get_error_code(),
                      milvus::ErrorCode::DataFormatBroken);
        }
    }
    EXPECT_EQ(reads->Count(), 0);
}

TEST(TextColumnCache, ReadBatchSharesReaderConcurrently) {
    auto tracker = std::make_shared<BlockingCallTracker>();
    auto factory_calls = std::make_shared<std::atomic<int>>(0);
    TextColumnCache cache(TextColumnCacheConfig{},
                          MakeReaderFactory(tracker, factory_calls));
    auto properties = std::make_shared<milvus_storage::api::Properties>();
    const std::string path = "/tmp/text-column-cache";
    auto ref = MakeLobReference();
    const std::vector<EncodedRef> refs{{ref.data(), ref.size()},
                                       {ref.data(), ref.size()}};
    auto cached = cache.GetOrCreateReader(path, nullptr, properties);
    ExpectConcurrentReads(tracker, [&] {
        EXPECT_EQ(cache.ReadBatch(path, nullptr, properties, refs),
                  (std::vector<std::string>{"mock-text", "mock-text"}));
    });
    EXPECT_EQ(factory_calls->load(), 1);
    EXPECT_EQ(tracker->CallCount(), 2);
    EXPECT_EQ(cache.GetOrCreateReader(path, nullptr, properties), cached);
}

TEST(TextColumnCache, ReadBatchIntoSharesReaderConcurrently) {
    auto tracker = std::make_shared<BlockingCallTracker>();
    auto factory_calls = std::make_shared<std::atomic<int>>(0);
    TextColumnCache cache(TextColumnCacheConfig{},
                          MakeReaderFactory(tracker, factory_calls));
    auto properties = std::make_shared<milvus_storage::api::Properties>();
    const std::string path = "/tmp/text-column-cache";
    auto ref = MakeLobReference();
    const std::vector<EncodedRef> refs{{ref.data(), ref.size()},
                                       {ref.data(), ref.size()}};
    auto cached = cache.GetOrCreateReader(path, nullptr, properties);
    ExpectConcurrentReads(tracker, [&] {
        google::protobuf::RepeatedPtrField<std::string> dst;
        dst.Add();
        dst.Add();
        cache.ReadBatchInto(path, nullptr, properties, refs, &dst);
        EXPECT_EQ(dst.Get(0), "mock-text");
        EXPECT_EQ(dst.Get(1), "mock-text");
    });
    EXPECT_EQ(factory_calls->load(), 1);
    EXPECT_EQ(tracker->CallCount(), 2);
    EXPECT_EQ(cache.GetOrCreateReader(path, nullptr, properties), cached);
}

TEST(TextColumnCache, SerializedFallbackCancelsQueuedReads) {
    for (bool into : {false, true}) {
        SCOPED_TRACE(into);
        auto tracker = std::make_shared<BlockingCallTracker>();
        auto factories = std::make_shared<std::atomic<int>>(0);
        TextColumnCacheConfig config;
        config.serialize_reads = true;
        TextColumnCache cache(config, MakeReaderFactory(tracker, factories));
        auto properties =
            std::make_shared<const milvus_storage::api::Properties>();
        auto ref = MakeLobReference();
        const std::vector<EncodedRef> refs{{ref.data(), ref.size()}};
        auto read = [&](const folly::CancellationToken& token) {
            if (into) {
                google::protobuf::RepeatedPtrField<std::string> dst;
                dst.Add();
                cache.ReadBatchInto(
                    "path", nullptr, properties, refs, &dst, token);
                EXPECT_EQ(dst.Get(0), "mock-text");
            } else {
                EXPECT_EQ(
                    cache.ReadBatch("path", nullptr, properties, refs, token),
                    std::vector<std::string>{"mock-text"});
            }
        };
        auto active = std::async(std::launch::async, [&] { read({}); });
        const bool entered = tracker->WaitForCalls(1, std::chrono::seconds(5));
        folly::CancellationSource source;
        std::promise<void> started;
        auto queued = std::async(std::launch::async, [&] {
            started.set_value();
            read(source.getToken());
        });
        started.get_future().wait();
        const bool waiting = queued.wait_for(std::chrono::milliseconds(50)) !=
                             std::future_status::ready;
        source.requestCancellation();
        const bool cancelled = queued.wait_for(std::chrono::seconds(5)) ==
                               std::future_status::ready;
        const auto calls = tracker->CallCount();
        tracker->Release();
        active.get();
        EXPECT_THROW(queued.get(), folly::FutureCancellation);
        EXPECT_TRUE(entered);
        EXPECT_TRUE(waiting);
        EXPECT_TRUE(cancelled);
        EXPECT_EQ(calls, 1);
        EXPECT_EQ(tracker->Peak(), 1);
        read({});  // Cancellation must leave the fallback lock usable.
    }
}

TEST(TextColumnCache, ReleasesEvictedAndClearedReadersOutsideCacheLock) {
    for (bool clear : {false, true}) {
        SCOPED_TRACE(clear);
        auto destruction = std::make_shared<BlockingCallTracker>();
        auto reads = std::make_shared<BlockingCallTracker>();
        auto factories = std::make_shared<std::atomic<int>>(0);
        TextColumnCacheConfig config;
        config.max_file_readers = 1;
        TextColumnCache cache(
            config,
            [=](std::shared_ptr<arrow::fs::FileSystem>, const LobColumnConfig&)
                -> arrow::Result<std::unique_ptr<LobColumnReader>> {
                return std::unique_ptr<LobColumnReader>(
                    std::make_unique<InstrumentedLobColumnReader>(
                        reads,
                        factories->fetch_add(1) == 0 ? destruction : nullptr));
            });
        auto properties =
            std::make_shared<const milvus_storage::api::Properties>();
        cache.GetOrCreateReader("old", nullptr, properties);
        auto retire = std::async(std::launch::async, [&] {
            if (clear) {
                cache.Clear();
            } else {
                cache.GetOrCreateReader("replacement", nullptr, properties);
            }
        });
        const bool destroying =
            destruction->WaitForCalls(1, std::chrono::seconds(5));
        auto lookup = std::async(std::launch::async, [&] {
            return cache.GetOrCreateReader("replacement", nullptr, properties);
        });
        const bool unlocked = lookup.wait_for(std::chrono::seconds(5)) ==
                              std::future_status::ready;
        destruction->Release();
        retire.get();
        EXPECT_NE(lookup.get(), nullptr);
        EXPECT_TRUE(destroying);
        EXPECT_TRUE(unlocked) << "reader destruction blocked the cache lock";
        EXPECT_EQ(factories->load(), 2);
    }
}

TEST(TextColumnCache, ClearingCachePreservesActiveRead) {
    auto reads = std::make_shared<BlockingCallTracker>();
    auto destruction = std::make_shared<BlockingCallTracker>();
    destruction->Release();
    TextColumnCache cache(
        TextColumnCacheConfig{},
        [=](std::shared_ptr<arrow::fs::FileSystem>, const LobColumnConfig&)
            -> arrow::Result<std::unique_ptr<LobColumnReader>> {
            return std::unique_ptr<LobColumnReader>(
                std::make_unique<InstrumentedLobColumnReader>(reads,
                                                              destruction));
        });
    auto ref = MakeLobReference();
    const std::vector<EncodedRef> refs{{ref.data(), ref.size()}};
    auto properties = std::make_shared<const milvus_storage::api::Properties>();
    auto active = std::async(std::launch::async, [&] {
        return cache.ReadBatch("path", nullptr, properties, refs);
    });
    const bool entered = reads->WaitForCalls(1, std::chrono::seconds(5));
    cache.Clear();
    const auto destroyed_during_read = destruction->CallCount();
    reads->Release();
    EXPECT_EQ(active.get(), std::vector<std::string>{"mock-text"});
    EXPECT_TRUE(entered);
    EXPECT_EQ(destroyed_during_read, 0);
    EXPECT_EQ(destruction->CallCount(), 1);
}

TEST(TextColumnCache, BatchBudgetIsSharedAcrossCacheInstancesAndPaths) {
    auto tracker = std::make_shared<BlockingCallTracker>();
    auto factories = std::make_shared<std::atomic<int>>(0);
    TextColumnCache first(TextColumnCacheConfig{},
                          MakeReaderFactory(tracker, factories));
    TextColumnCache second(TextColumnCacheConfig{},
                           MakeReaderFactory(tracker, factories));
    auto properties = std::make_shared<const milvus_storage::api::Properties>();
    first.GetOrCreateReader("first", nullptr, properties);
    second.GetOrCreateReader("second", nullptr, properties);
    auto ref = MakeLobReference();
    const std::vector<EncodedRef> refs{{ref.data(), ref.size()}};
    std::vector<std::future<std::vector<std::string>>> tasks;
    for (int i = 0; i < 8; ++i) {
        tasks.push_back(std::async(std::launch::async, [&, i] {
            return i % 2
                       ? first.ReadBatch("first", nullptr, properties, refs)
                       : second.ReadBatch("second", nullptr, properties, refs);
        }));
    }
    const bool full = tracker->WaitForCalls(4, std::chrono::seconds(5));
    // Inline-only reads consume no I/O permit even while the budget is full.
    const auto inline_ref =
        milvus_storage::lob_column::EncodeInlineText("inline");
    auto inline_read = std::async(std::launch::async, [&] {
        return first.ReadBatch("inline",
                               nullptr,
                               properties,
                               {{inline_ref.data(), inline_ref.size()}});
    });
    const bool inline_ready = inline_read.wait_for(std::chrono::seconds(5)) ==
                              std::future_status::ready;
    tracker->Release();
    for (auto& task : tasks)
        EXPECT_EQ(task.get(), std::vector<std::string>{"mock-text"});
    EXPECT_EQ(inline_read.get(), std::vector<std::string>{"inline"});
    EXPECT_TRUE(full);
    EXPECT_TRUE(inline_ready);
    EXPECT_EQ(tracker->Peak(), 4);
    EXPECT_EQ(factories->load(), 2);
}

TEST(TextColumnCache,
     CancellationExitsBudgetWaitWithoutReadingOrLeakingPermits) {
    auto tracker = std::make_shared<BlockingCallTracker>();
    auto factories = std::make_shared<std::atomic<int>>(0);
    TextColumnCache cache(TextColumnCacheConfig{},
                          MakeReaderFactory(tracker, factories));
    auto properties = std::make_shared<const milvus_storage::api::Properties>();
    cache.GetOrCreateReader("path", nullptr, properties);
    auto ref = MakeLobReference();
    const std::vector<EncodedRef> refs{{ref.data(), ref.size()}};
    std::vector<std::future<std::vector<std::string>>> active;
    for (int i = 0; i < 4; ++i) {
        active.push_back(std::async(std::launch::async, [&] {
            return cache.ReadBatch("path", nullptr, properties, refs);
        }));
    }
    const bool full = tracker->WaitForCalls(4, std::chrono::seconds(5));
    folly::CancellationSource source;
    std::promise<void> started;
    auto cancelled = std::async(std::launch::async, [&] {
        started.set_value();
        return cache.ReadBatch(
            "path", nullptr, properties, refs, source.getToken());
    });
    started.get_future().wait();
    const bool waiting = cancelled.wait_for(std::chrono::milliseconds(50)) !=
                         std::future_status::ready;
    source.requestCancellation();
    const bool exited = cancelled.wait_for(std::chrono::seconds(5)) ==
                        std::future_status::ready;
    const auto reads_before_release = tracker->CallCount();
    tracker->Release();
    for (auto& task : active)
        EXPECT_EQ(task.get(), std::vector<std::string>{"mock-text"});
    EXPECT_THROW(cancelled.get(), folly::FutureCancellation);
    EXPECT_TRUE(full);
    EXPECT_TRUE(waiting);
    EXPECT_TRUE(exited);
    EXPECT_EQ(reads_before_release, 4);
    EXPECT_EQ(cache.ReadBatch("path", nullptr, properties, refs),
              std::vector<std::string>{"mock-text"});
}

TEST(TextColumnCache, SharesPropertiesWithReader) {
    auto properties = std::make_shared<const milvus_storage::api::Properties>();
    std::shared_ptr<const milvus_storage::api::Properties> reader_properties;
    auto tracker = std::make_shared<BlockingCallTracker>();
    TextColumnCache cache(
        TextColumnCacheConfig{},
        [&](std::shared_ptr<arrow::fs::FileSystem>,
            const LobColumnConfig& config)
            -> arrow::Result<std::unique_ptr<LobColumnReader>> {
            reader_properties = config.properties;
            return std::unique_ptr<LobColumnReader>(
                std::make_unique<InstrumentedLobColumnReader>(tracker));
        });
    cache.GetOrCreateReader("/tmp/text-column-cache", nullptr, properties);
    EXPECT_EQ(reader_properties, properties);
}

TEST(TextColumnCache, ArrowTextOutputPropagatesCancellation) {
    struct TextOutputAccess : ChunkedSegmentSealedImpl {
        using ChunkedSegmentSealedImpl::ArrowToDataArray;
    };
    const auto encoded = milvus_storage::lob_column::EncodeInlineText("text");
    arrow::BinaryBuilder builder;
    ASSERT_TRUE(builder.Append(encoded.data(), encoded.size()).ok());
    auto array = builder.Finish();
    ASSERT_TRUE(array.ok()) << array.status();
    FieldMeta meta(FieldName("text"),
                   FieldId(101),
                   DataType::TEXT,
                   65536,
                   false,
                   std::nullopt);
    const std::string path = "/unused/cancelled-lob";
    folly::CancellationSource source;
    source.requestCancellation();
    EXPECT_THROW(TextOutputAccess::ArrowToDataArray(
                     *array, meta, {0}, 1, nullptr, &path, source.getToken()),
                 folly::FutureCancellation);
}

TEST(TextColumnCache, PreservesStorageErrorWhenCreatingReader) {
    auto status = milvus_storage::MakeExtendError(
        milvus_storage::ExtendStatusCode::StorageTransientTimeout,
        "storage timeout");
    TextColumnCache cache(
        TextColumnCacheConfig{},
        [status](std::shared_ptr<arrow::fs::FileSystem>, const LobColumnConfig&)
            -> arrow::Result<std::unique_ptr<LobColumnReader>> {
            return status;
        });
    auto properties = std::make_shared<const milvus_storage::api::Properties>();

    try {
        static_cast<void>(cache.GetOrCreateReader(
            "/tmp/text-column-cache", nullptr, properties));
        FAIL() << "expected storage error";
    } catch (const milvus::SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(),
                  milvus::ErrorCode::StorageTransientError);
        EXPECT_NE(std::string(error.what()).find("storage timeout"),
                  std::string::npos);
    }
}

}  // namespace
}  // namespace milvus::segcore
