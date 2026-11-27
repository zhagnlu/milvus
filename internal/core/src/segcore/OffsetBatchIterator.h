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

#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <vector>
#include "common/Types.h"

namespace milvus::segcore {

// Request-local, single-consumer cursor. The exclusion bitmap is borrowed and
// must remain immutable/alive. Each call returns at most max_rows offsets;
// empty means exhausted. Abandon the cursor on any exception. Owners of PK
// maps must outlive it (segment snapshots attach their resource pins).
using OffsetBatchIterator =
    std::function<std::vector<int64_t>(size_t max_rows)>;

inline void
CheckOffsetBatchSize(size_t max_rows) {
    AssertInfo(max_rows > 0 &&
                   max_rows <=
                       static_cast<size_t>(std::numeric_limits<int32_t>::max()),
               "invalid offset batch size");
}

// Only for sources whose physical row order already equals PK order.
inline OffsetBatchIterator
MakeBitmapOffsetIterator(BitsetTypeView excluded,
                         std::function<void()> checkpoint = {}) {
    return [excluded, checkpoint = std::move(checkpoint), next = size_t{0}](
               size_t max_rows) mutable {
        CheckOffsetBatchSize(max_rows);
        if (checkpoint)
            checkpoint();
        std::vector<int64_t> offsets;
        offsets.reserve(std::min(max_rows, excluded.size() - next));
        while (next < excluded.size() && offsets.size() < max_rows) {
            if ((next % 1024) == 0 && checkpoint)
                checkpoint();
            if (!excluded[next])
                offsets.push_back(next);
            ++next;
        }
        if (checkpoint)
            checkpoint();
        return offsets;
    };
}

}  // namespace milvus::segcore
