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

#include <memory>
#include <optional>
#include <vector>

#include "Vector.h"
#include "common/EasyAssert.h"
#include "common/Types.h"
#include "common/VectorAccess.h"

namespace milvus {

void
BaseVector::prepareForReuse() {
    null_count_ = std::nullopt;
}

void
RowVector::resize(milvus::vector_size_t new_size, bool setNotNull) {
    AssertInfo(new_size >= 0, "negative RowVector size");
    if (static_cast<size_t>(new_size) == length_) {
        return;
    }
    // Commit all columns and row validity together. Resizing one shared child
    // in place would corrupt another batch; a later child failure would leave
    // this batch with inconsistent lengths.
    auto candidate = MakeWritableCopy(*this);
    for (auto& child : candidate->childrens()) {
        child->resize(new_size, setNotNull);
    }
    if (candidate->row_validity_) {
        candidate->row_validity_->resize(new_size, setNotNull);
    } else if (!setNotNull && static_cast<size_t>(new_size) > length_) {
        candidate->row_validity_.emplace(new_size, true);
        candidate->row_validity_->reset(length_, new_size - length_);
    }
    candidate->null_count_ = candidate->row_validity_
                                 ? new_size - candidate->row_validity_->count()
                                 : 0;
    children_values_ = std::move(candidate->children_values_);
    row_validity_ = std::move(candidate->row_validity_);
    length_ = new_size;
    null_count_ = candidate->null_count_;
}

namespace {
DataType
LogicalType(const VectorPtr& child) {
    const auto* column = dynamic_cast<const ColumnVector*>(child.get());
    return column && column->IsBitmap() ? DataType::BOOL : child->type();
}
}  // namespace

RowVector::RowVector(RowTypePtr row_type,
                     size_t row_count,
                     std::vector<VectorPtr> children,
                     std::optional<TargetBitmap> validity)
    : BaseVector(DataType::ROW, row_count),
      row_type_(std::move(row_type)),
      row_validity_(std::move(validity)),
      children_values_(std::move(children)) {
    AssertInfo(
        row_count <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
        "RowVector row count exceeds size range");
    AssertInfo(
        row_type_ && row_type_->column_count() == children_values_.size(),
        "RowVector schema/column count mismatch");
    for (size_t i = 0; i < children_values_.size(); ++i) {
        const auto& child = children_values_[i];
        AssertInfo(child && child->size() == static_cast<int64_t>(row_count),
                   "RowVector child shape mismatch");
        AssertInfo(LogicalType(child) == row_type_->column_type(i),
                   "RowVector child type does not match schema");
    }
    AssertInfo(!row_validity_ || row_validity_->size() == row_count,
               "RowVector validity length mismatch");
    null_count_ = row_validity_ ? row_count - row_validity_->count() : 0;
}

void
RowVector::InferRowType() {
    std::vector<std::string> names;
    std::vector<DataType> types;
    for (size_t i = 0; i < children_values_.size(); ++i) {
        // Legacy unnamed constructors receive stable positional field names.
        names.push_back("c" + std::to_string(i));
        types.push_back(LogicalType(children_values_[i]));
    }
    row_type_ = std::make_shared<RowType>(std::move(names), std::move(types));
}

void
RowVector::InitRowValidity(std::optional<size_t> null_count) {
    AssertInfo(!null_count || *null_count == 0 || *null_count == length_,
               "partial row NULLs require an explicit validity bitmap");
    if (null_count && *null_count != 0) {
        row_validity_.emplace(length_, false);
    }
}

void
RowVector::SetRowValid(size_t row, bool valid) {
    AssertInfo(row < length_, "row index out of bounds");
    if (!row_validity_ && valid) {
        return;
    }
    if (!row_validity_) {
        row_validity_.emplace(length_, true);
    }
    const bool was_valid = (*row_validity_)[row];
    if (was_valid != valid) {
        null_count_ =
            valid ? null_count_.value_or(0) - 1 : null_count_.value_or(0) + 1;
        row_validity_->set(row, valid);
    }
}

void
RowVector::prepareForReuse() {
    std::vector<VectorPtr> children;
    for (const auto& child : children_values_) {
        children.push_back(child->cloneEmpty(0));
    }
    children_values_ = std::move(children);
    row_validity_.reset();
    length_ = 0;
    null_count_ = 0;
}

}  // namespace milvus
