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

#include "segcore/reader/ColumnReader.h"

namespace milvus {
class ChunkedColumnInterface;
}

namespace milvus::segcore::reader {

// Sealed, ordinary scalar Column adapter. The caller must bind a stable
// generation (e.g. an aliasing shared_ptr into a pinned snapshot), not resolve
// a live Segment again per batch. No index/raw fallback or ColumnFilter.
// JSON payload is copied with SIMD padding before moving a cursor/pin.
// TEXT/LOB, ARRAY, virtual PK and Growing require separate adapters.
class ColumnReaderAdapter final : public ColumnReader {
 public:
    ColumnReaderAdapter(std::shared_ptr<const ChunkedColumnInterface> column,
                        DataType type,
                        int64_t readable_rows,
                        bool recursive_array = false);
    ~ColumnReaderAdapter() override;

    DataType
    type() const override {
        return type_;
    }
    int64_t
    row_count() const override {
        return readable_rows_;
    }
    ReadResult
    ReadRange(const RangeRequest&, const ReadContext&) const override;
    ReadResult
    Take(const TakeRequest&, const ReadContext&) const override;

 private:
    std::shared_ptr<const ChunkedColumnInterface> column_;
    DataType type_;
    int64_t readable_rows_;
    bool recursive_array_;
    struct RangeSession;
    std::unique_ptr<RangeSession> range_session_;
};

}  // namespace milvus::segcore::reader
