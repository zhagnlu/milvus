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

#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <vector>
#include "common/EasyAssert.h"

namespace milvus {

// Immutable Segment row IDs for scalar execution, not batch-local RowMapping.
// Snapshot caller-owned memory once; copies of this handle share the snapshot.
// Never adopt an external vector: even an rvalue may have writable aliases.
class SegmentOffsets {
 public:
    SegmentOffsets() = default;

    static SegmentOffsets
    CopyOf(std::span<const int32_t> offsets) {
        return CopyChecked(offsets);
    }

    // Compatibility boundary for callers with wider offsets. Validate before
    // narrowing; retrieval's general int64 OffsetView API is unchanged.
    static SegmentOffsets
    CopyOf(std::span<const int64_t> offsets) {
        return CopyChecked(offsets);
    }

    size_t
    size() const {
        return offsets_ ? offsets_->size() : 0;
    }

    const int32_t*
    data() const {
        return offsets_ ? offsets_->data() : nullptr;
    }

    const int32_t*
    begin() const {
        return data();
    }

    const int32_t*
    end() const {
        return size() ? data() + size() : data();
    }

    int32_t
    operator[](size_t row) const {
        AssertInfo(row < size(), "segment offset position out of bounds");
        return (*offsets_)[row];
    }

 private:
    template <typename T>
    static SegmentOffsets
    CopyChecked(std::span<const T> offsets) {
        AssertInfo(offsets.size() <=
                       static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                   "too many scalar batch offsets");
        if (offsets.empty())
            return {};
        std::vector<int32_t> snapshot;
        snapshot.reserve(offsets.size());
        for (auto offset : offsets) {
            AssertInfo(
                offset >= 0 && offset <= std::numeric_limits<int32_t>::max(),
                "scalar segment offset outside INT32 range: {}",
                offset);
            snapshot.push_back(static_cast<int32_t>(offset));
        }
        SegmentOffsets result;
        result.offsets_ =
            std::make_shared<const std::vector<int32_t>>(std::move(snapshot));
        return result;
    }

    std::shared_ptr<const std::vector<int32_t>> offsets_;
};

}  // namespace milvus
