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

#include "exec/input/BatchInput.h"

namespace milvus::exec {

BatchInput::BatchInput(RowTypePtr schema, size_t size)
    : schema_(std::move(schema)), size_(size) {
    AssertInfo(schema_, "input schema cannot be null");
    AssertInfo(
        size <= static_cast<size_t>(std::numeric_limits<vector_size_t>::max()),
        "input exceeds vector index range");
    geometry_.resize(schema_->column_count());
    slots_.reserve(schema_->column_count());
    for (size_t i = 0; i < schema_->column_count(); ++i) {
        slots_.emplace_back(schema_->column_type(i), size);
    }
}

BatchInput::BatchInput(const RowVector& input)
    : BatchInput(input.row_type(), input.size()) {
    for (size_t i = 0; i < slots_.size(); ++i) {
        BindColumn(i, input, i, SelectionView::All(size_));
    }
}

void
BatchInput::Reset(size_t size) {
    AssertInfo(
        size <= static_cast<size_t>(std::numeric_limits<vector_size_t>::max()),
        "input exceeds vector index range");
    for (auto& slot : slots_) slot.Reset(size);
    for (auto& lookup : geometry_) lookup = {};
    size_ = size;
    started_ = false;
}

InputSlot&
BatchInput::GetSlot(column_index_t slot, SelectionView rows) {
    AssertInfo(slot < slots_.size(), "input slot out of bounds");
    AssertInfo(rows.row_count() == size_, "input selection length mismatch");
    return slots_[slot];
}

InputSlot&
BatchInput::BindSlot(column_index_t slot) {
    AssertInfo(!started_, "input binding is frozen after evaluation starts");
    return GetSlot(slot, SelectionView::None(size_));
}

void
BatchInput::BindGeometryLookup(column_index_t slot, GeometryLookup lookup) {
    BindSlot(slot);
    AssertInfo(schema_->column_type(slot) == DataType::GEOMETRY && lookup &&
                   !geometry_[slot],
               "invalid geometry cache binding");
    geometry_[slot] = std::move(lookup);
}
const BatchInput::GeometryLookup&
BatchInput::GetGeometryLookup(column_index_t slot) const {
    AssertInfo(slot < slots_.size(), "geometry cache slot out of bounds");
    return geometry_[slot];
}

void
BatchInput::BindValues(column_index_t slot, ReadOnlyVectorView values) {
    BindSlot(slot).BindValues(std::move(values));
}

void
BatchInput::BindValidity(column_index_t slot, ValidityResult validity) {
    BindSlot(slot).BindValidity(std::move(validity));
}

void
BatchInput::BindLoader(column_index_t slot,
                       ValueLoader values,
                       ValidityLoader validity) {
    BindSlot(slot).BindLoader({std::move(values), std::move(validity)});
}

void
BatchInput::BindColumn(column_index_t slot,
                       const RowVector& upstream,
                       column_index_t column,
                       SelectionView defined) {
    auto& target = BindSlot(slot);
    AssertInfo(upstream.size() == size_ && defined.row_count() == size_,
               "upstream input coverage/length mismatch");
    AssertInfo(column < upstream.row_type()->column_count() &&
                   upstream.row_type()->column_type(column) ==
                       schema_->column_type(slot),
               "upstream input column/type mismatch");
    auto valid_rows = OwnedSelection::CopyOf(defined);
    defined.ForEachSelected([&](size_t row) {
        if (!upstream.IsValid(row))
            valid_rows.SetSelected(row, false);
    });
    auto null_rows = OwnedSelection::Subtract(defined, valid_rows.view());
    auto values =
        ReadOnlyVectorView::CopyOf(upstream.child(column), valid_rows.view());
    if (!null_rows.view().IsNone()) {
        values = values.Merge(ReadOnlyVectorView::Nulls(
                                  schema_->column_type(slot), null_rows.view()),
                              null_rows.view());
    }
    target.BindValues(std::move(values));
}

ReadOnlyVectorView
BatchInput::GetColumn(column_index_t slot, SelectionView rows) {
    auto& state = GetSlot(slot, rows);
    started_ |= !rows.IsNone();
    return state.GetColumn(rows);
}

ValidityResult
BatchInput::GetValidity(column_index_t slot, SelectionView rows) {
    auto& state = GetSlot(slot, rows);
    started_ |= !rows.IsNone();
    return state.GetValidity(rows);
}

}  // namespace milvus::exec
