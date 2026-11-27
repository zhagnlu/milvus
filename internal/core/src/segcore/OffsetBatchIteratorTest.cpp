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

#include <gtest/gtest.h>
#include <random>
#include <folly/futures/Future.h>
#include "segcore/InsertRecord.h"

namespace milvus::segcore {
namespace {
std::vector<int64_t>
Drain(OffsetBatchIterator next, size_t batch) {
    std::vector<int64_t> result;
    while (true) {
        auto rows = next(batch);
        EXPECT_LE(rows.size(), batch);
        if (rows.empty())
            break;
        result.insert(result.end(), rows.begin(), rows.end());
    }
    EXPECT_TRUE(next(batch).empty());
    return result;
}
}  // namespace

TEST(OffsetBatchIteratorTest, MatchesLegacyAcrossPkTypesMasksAndBatchSizes) {
    auto check = []<typename T>() {
        OffsetOrderedMap<T> growing;
        OffsetOrderedArray<T> sealed;
        std::mt19937 random(42);
        for (int64_t row = 0; row < 4099; ++row) {
            T key;
            if constexpr (std::is_same_v<T, std::string>)
                key = std::to_string(random() % 731);
            else
                key = random() % 731;
            growing.insert(key, row);
            sealed.insert(key, row);
        }
        sealed.seal();
        for (size_t count : {0, 1, 2048, 4099}) {
            for (int selectivity : {0, 3, 100}) {
                TargetBitmap excluded(count, false);
                for (size_t row = 0; row < count; ++row)
                    excluded.set(row, random() % 100 < selectivity);
                for (const OffsetMap* map :
                     std::vector<const OffsetMap*>{&growing, &sealed}) {
                    auto expected =
                        map->find_first_n(Unlimited, excluded.view()).first;
                    for (size_t batch : {1, 7, 1024, 8192})
                        EXPECT_EQ(
                            Drain(map->make_offset_iterator(excluded.view()),
                                  batch),
                            expected);
                }
            }
        }
    };
    check.template operator()<int64_t>();
    check.template operator()<std::string>();
}

TEST(OffsetBatchIteratorTest, GrowingAppendsBetweenBatchesKeepFixedPrefix) {
    OffsetOrderedMap<int64_t> map;
    for (int64_t row = 0; row < 6; ++row) map.insert(row % 3, row);
    TargetBitmap excluded(6, false);
    excluded.set(4, true);  // key 1 must fall back to offset 1
    auto cursor = map.make_offset_iterator(excluded.view());
    EXPECT_EQ(cursor(1), (std::vector<int64_t>{3}));
    // Would deadlock if a read lock survived the previous batch. New keys
    // before and after the continuation and new versions are all outside ACK.
    map.insert(int64_t{-1}, 6);
    map.insert(int64_t{1}, 7);
    map.insert(int64_t{9}, 8);
    EXPECT_EQ(Drain(std::move(cursor), 1), (std::vector<int64_t>{1, 5}));
}

TEST(OffsetBatchIteratorTest,
     SortedAndVirtualRowsPreserveDuplicatesAndEmptyTail) {
    VirtualPKOffsetMap map(19, 10000);
    TargetBitmap excluded(10000, true);
    for (size_t row : {0, 4, 2049, 5000}) excluded.set(row, false);
    const std::vector<int64_t> expected{0, 4, 2049, 5000};
    EXPECT_EQ(Drain(map.make_offset_iterator(excluded.view()), 2), expected);
    EXPECT_EQ(Drain(MakeBitmapOffsetIterator(excluded.view()), 2), expected);
    EXPECT_THROW(MakeBitmapOffsetIterator(excluded.view())(0), SegcoreError);
}

TEST(OffsetBatchIteratorTest, CancellationChecksFilteredAndDuplicateRuns) {
    OffsetOrderedMap<int64_t> map;
    OffsetOrderedArray<int64_t> array;
    for (int64_t row = 0; row < 10000; ++row) {
        map.insert(int64_t{0}, row);
        array.insert(int64_t{0}, row);
    }
    array.seal();
    TargetBitmap excluded(10000, true);
    for (int source = 0; source < 3; ++source) {
        int checks = 0;
        auto check = [&] {
            if (++checks == 3)
                throw folly::FutureCancellation();
        };
        auto cursor =
            source == 0   ? map.make_offset_iterator(excluded.view(), check)
            : source == 1 ? array.make_offset_iterator(excluded.view(), check)
                          : MakeBitmapOffsetIterator(excluded.view(), check);
        std::vector<int64_t> output;
        try {
            output = cursor(8);
            FAIL() << "expected cancellation";
        } catch (const folly::FutureCancellation&) {
        }
        EXPECT_TRUE(output.empty());
        EXPECT_EQ(checks, 3);
    }
}

}  // namespace milvus::segcore
