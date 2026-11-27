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

#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include "common/Selection.h"
#include "common/SegmentOffsets.h"
#include "common/Types.h"

namespace milvus {
struct OpContext;
}

namespace milvus::segcore::reader {

enum class ReadProjection { ValuesAndValidity, ValidityOnly };

// Synchronous reads borrow this context. Access nodes retain its shared owner.
// checkpoint may enforce cancellation/budget; it must not evaluate Exprs.
// No implicit retry, logical cursor, or production QueryContext wiring here.
struct ReadContext {
    std::shared_ptr<OpContext> op_context;
    std::function<void()> checkpoint;

    void
    Check() const {
        if (checkpoint)
            checkpoint();
    }
};

inline void
CheckRequestSize(size_t size) {
    AssertInfo(
        size <= static_cast<size_t>(std::numeric_limits<vector_size_t>::max()),
        "reader request too large");
}

// eval_rows and output coverage are request-local, never segment offsets.
// Constructors detach caller-owned mutable buffers. Already immutable offsets
// are shared across bindings, fields and requests.
struct RangeRequest {
    RangeRequest(int64_t begin,
                 size_t count,
                 ReadProjection projection,
                 SelectionView rows)
        : begin(begin),
          count(count),
          projection(projection),
          eval_rows(OwnedSelection::CopyOf(rows)) {
        CheckRequestSize(count);
        AssertInfo(
            begin >= 0 &&
                count <= static_cast<size_t>(
                             std::numeric_limits<int64_t>::max() - begin),
            "invalid reader range");
        AssertInfo(rows.row_count() == count,
                   "reader range selection mismatch");
    }
    const int64_t begin;
    const size_t count;
    const ReadProjection projection;
    const OwnedSelection eval_rows;
};

struct TakeRequest {
    TakeRequest(SegmentOffsets offsets,
                ReadProjection projection,
                SelectionView rows)
        : offsets(std::move(offsets)),
          projection(projection),
          eval_rows(OwnedSelection::CopyOf(rows)) {
        CheckRequestSize(this->offsets.size());
        AssertInfo(rows.row_count() == this->offsets.size(),
                   "reader take selection mismatch");
    }
    TakeRequest(std::span<const int32_t> offsets,
                ReadProjection projection,
                SelectionView rows)
        : TakeRequest(SegmentOffsets::CopyOf(offsets), projection, rows) {
    }
    TakeRequest(std::span<const int64_t> offsets,
                ReadProjection projection,
                SelectionView rows)
        : TakeRequest(SegmentOffsets::CopyOf(offsets), projection, rows) {
    }
    TakeRequest(std::initializer_list<int32_t> offsets,
                ReadProjection projection,
                SelectionView rows)
        : TakeRequest(std::span<const int32_t>(offsets.begin(), offsets.size()),
                      projection,
                      rows) {
    }
    const SegmentOffsets offsets;
    const ReadProjection projection;
    const OwnedSelection eval_rows;
};

}  // namespace milvus::segcore::reader
