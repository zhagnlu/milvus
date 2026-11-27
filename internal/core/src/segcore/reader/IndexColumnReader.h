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

#pragma once

#include <mutex>
#include "segcore/reader/ScalarQueryView.h"
#include "segcore/reader/PointColumnReader.h"

namespace milvus::segcore::reader {

// A raw-value source, NOT an index predicate. Only ordinary STLSORT, BITMAP
// and Trie have metadata-proven reverse lookup; inverted/nested/path indexes
// must never be used to reconstruct the original field.
template <typename T>
class IndexColumnReader final : public PointColumnReader {
 public:
    using Load = std::function<std::shared_ptr<const index::IIndexReaderBase>(
        const ReadContext&)>;
    IndexColumnReader(std::shared_ptr<const index::IIndexReaderBase> index,
                      DataType type,
                      int64_t rows)
        : PointColumnReader(type, rows),
          source_rows_(rows),
          index_(std::move(index)) {
        CheckIndex(index_);
    }

    // Bind immutable metadata now; retain the actual pin on first demand.
    // The query may address a prefix of the sealed index's full row domain.
    IndexColumnReader(DataType type,
                      int64_t rows,
                      int64_t source_rows,
                      Load load)
        : PointColumnReader(type, rows),
          source_rows_(source_rows),
          load_(std::move(load)) {
        AssertInfo(IsPointScalarType<T>(type) && load_ && source_rows >= rows,
                   "invalid deferred index value binding");
    }

 protected:
    ReadResult
    ReadPoints(SelectionView rows,
               ReadProjection projection,
               const Offset& offset,
               const ReadContext& ctx) const override {
        // Do not query the backend for empty selections.
        if (!rows.IsNone()) {
            ctx.Check();
            std::call_once(validity_once_, [&] {
                auto candidate_index = index_ ? index_ : load_(ctx);
                CheckIndex(candidate_index);
                auto valid = NullReader(*candidate_index).IsNotNull();
                AssertInfo(valid.size() == static_cast<size_t>(source_rows_),
                           "index value validity domain mismatch");
                auto candidate =
                    OwnedSelection::CopyOf(SelectionView::FromBitmap(valid));
                ctx.Check();
                index_ = std::move(candidate_index);
                validity_.emplace(std::move(candidate));
            });
            AssertInfo(index_->Count() == source_rows_,
                       "index value generation changed");
        }
        return ReadScalars<T>(
            rows,
            projection,
            [&](size_t row) {
                return validity_->view().IsSelected(offset(row));
            },
            [&](size_t row) -> T {
                auto* values = dynamic_cast<
                    const index::IScalarValueReader<QueryValue<T>>*>(
                    index_.get());
                AssertInfo(values, "prepared index has no value reader");
                auto value = values->Lookup(offset(row));
                AssertInfo(value.has_value(),
                           "index reverse lookup disagrees with validity");
                return *value;
            },
            ctx);
    }

 private:
    void
    CheckIndex(
        const std::shared_ptr<const index::IIndexReaderBase>& index) const {
        AssertInfo(index && IsPointScalarType<T>(type()),
                   "invalid index value binding");
        const auto caps = index->Caps();
        AssertInfo(caps.value_lookup && caps.cheap_value_lookup &&
                       index->CoordDomain() == index::Domain::Row &&
                       !caps.json_paths,
                   "prepared index does not provide ordinary field values");
        AssertInfo(index->Count() == source_rows_,
                   "index value snapshot domain mismatch");
    }
    int64_t source_rows_;
    Load load_;
    mutable std::shared_ptr<const index::IIndexReaderBase> index_;
    mutable std::once_flag validity_once_;
    mutable std::optional<OwnedSelection> validity_;
};
}  // namespace milvus::segcore::reader
