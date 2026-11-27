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

#include <limits>
#include <initializer_list>
#include <optional>
#include "common/SegmentOffsets.h"
#include "common/Types.h"

namespace milvus::exec {

// Immutable segment-row positions, NOT common::RowMapping's batch-local
// indices. This binding has no Reader implementation or advancing cursor.
// Query-view resources and readable bounds are owned/validated by the prepared
// source. Element addressing is intentionally not accepted by this row API.
class ReadBatchBinding {
 public:
    static ReadBatchBinding
    Range(int64_t begin, size_t count) {
        AssertInfo(
            begin >= 0 &&
                count <= static_cast<size_t>(
                             std::numeric_limits<vector_size_t>::max()) &&
                count <= static_cast<size_t>(
                             std::numeric_limits<int64_t>::max() - begin),
            "invalid scalar batch range");
        return ReadBatchBinding(begin, count, std::nullopt);
    }

    // Mutable caller buffers are snapshotted once; bindings and requests share
    // the resulting immutable INT32 storage.
    static ReadBatchBinding
    Offsets(SegmentOffsets offsets) {
        const auto count = offsets.size();
        return ReadBatchBinding(0, count, std::move(offsets));
    }

    static ReadBatchBinding
    Offsets(std::span<const int32_t> offsets) {
        return Offsets(SegmentOffsets::CopyOf(offsets));
    }

    static ReadBatchBinding
    Offsets(std::span<const int64_t> offsets) {
        return Offsets(SegmentOffsets::CopyOf(offsets));
    }

    static ReadBatchBinding
    Offsets(std::initializer_list<int32_t> offsets) {
        return Offsets(
            std::span<const int32_t>(offsets.begin(), offsets.size()));
    }

    const SegmentOffsets&
    offsets() const {
        AssertInfo(!is_range(), "range binding has no offset buffer");
        return *offsets_;
    }

    size_t
    size() const {
        return size_;
    }

    bool
    is_range() const {
        return !offsets_;
    }

    int64_t
    range_begin() const {
        AssertInfo(is_range(), "offset binding has no range start");
        return begin_;
    }

    int64_t
    offset(size_t row) const {
        AssertInfo(row < size_, "batch position out of bounds");
        return offsets_ ? (*offsets_)[row] : begin_ + static_cast<int64_t>(row);
    }

 private:
    ReadBatchBinding(int64_t begin,
                     size_t size,
                     std::optional<SegmentOffsets> offsets)
        : begin_(begin), size_(size), offsets_(std::move(offsets)) {
    }
    int64_t begin_;
    size_t size_;
    std::optional<SegmentOffsets> offsets_;
};

}  // namespace milvus::exec
