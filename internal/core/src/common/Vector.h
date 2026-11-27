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
#include <memory>

#include "common/EasyAssert.h"
#include "Types.h"
#include "bitset/bitset.h"
#include "common/FieldData.h"
#include "common/FieldDataInterface.h"
#include "common/Types.h"

namespace milvus {
class BaseVector;
using VectorPtr = std::shared_ptr<BaseVector>;

/**
 * @brief base class for different type vector  
 * @todo implement full null value support
 */
class BaseVector {
 public:
    BaseVector(DataType data_type,
               size_t length,
               std::optional<size_t> null_count = std::nullopt)
        : type_kind_(data_type), length_(length), null_count_(null_count) {
    }
    virtual ~BaseVector() = default;

    int64_t
    size() const {
        return length_;
    }

    DataType
    type() const {
        return type_kind_;
    }

    int32_t
    elementSize() const {
        return GetDataTypeSize(type_kind_);
    };

    size_t
    nullCount() const {
        return null_count_.has_value() ? null_count_.value() : 0;
    }

    virtual void
    resize(vector_size_t newSize, bool setNotNull = true) {
        length_ = newSize;
    }

    // Legacy metadata reset. This does not establish exclusive ownership of
    // payload buffers. Use the VectorAccess writable-copy boundary for writes
    // to shared inputs, and VectorPool for opt-in scratch reuse.
    virtual void
    prepareForReuse();

    /// Creates a new empty instance of the same dynamic type with the given size.
    /// Used when cloning non-unique vectors to preserve the dynamic type.
    virtual VectorPtr
    cloneEmpty(vector_size_t size) const = 0;

 protected:
    DataType type_kind_;
    size_t length_;
    // todo: use null_count to skip some bitset operate
    std::optional<size_t> null_count_;
};

/**
 * SimpleVector abstracts over various Columnar Storage Formats,
 * it is used in custom functions.
 */
class SimpleVector : public BaseVector {
 public:
    SimpleVector(DataType data_type,
                 size_t length,
                 std::optional<size_t> null_count = std::nullopt)
        : BaseVector(data_type, length, null_count) {
    }

    virtual void*
    RawValueAt(size_t index, size_t size_of_element) = 0;

    virtual bool
    ValidAt(size_t index) = 0;
};

/**
 * @brief Single vector for scalar types
 * @todo using memory pool && buffer replace FieldData
 */
class ColumnVector final : public SimpleVector {
 public:
    ColumnVector(DataType data_type,
                 size_t length,
                 std::optional<size_t> null_count = std::nullopt)
        : SimpleVector(data_type, length, null_count),
          is_bitmap_(false),
          valid_values_(length,
                        !null_count.has_value() || null_count.value() == 0) {
        values_ = InitScalarFieldData(data_type, false, length);
    }

    //    ColumnVector(FixedVector<bool>&& data)
    //        : BaseVector(DataType::BOOL, data.size()) {
    //        values_ =
    //            std::make_shared<FieldData<bool>>(DataType::BOOL, std::move(data));
    //    }

    // the size is the number of bits
    // TODO: separate the usage of bitmap from scalar field data
    ColumnVector(TargetBitmap&& bitmap, TargetBitmap&& valid_bitmap)
        : SimpleVector(DataType::INT8, bitmap.size()),
          is_bitmap_(true),
          valid_values_(std::move(valid_bitmap)) {
        values_ = std::make_shared<FieldBitsetImpl<uint8_t>>(DataType::INT8,
                                                             std::move(bitmap));
    }

    ColumnVector(FieldDataPtr&& value,
                 TargetBitmap&& valid_bitmap,
                 size_t null_count)
        : SimpleVector(value ? value->get_data_type() : DataType::NONE,
                       value ? value->Length() : 0),
          is_bitmap_(false),
          valid_values_(std::move(valid_bitmap)) {
        AssertInfo(value, "ColumnVector value cannot be null");
        AssertInfo(valid_values_.size() == length_,
                   "ColumnVector valid bitmap size {} must match length {}",
                   valid_values_.size(),
                   length_);
        AssertInfo(null_count <= length_,
                   "ColumnVector null count {} must not exceed length {}",
                   null_count,
                   length_);
        values_ = std::move(value);
        null_count_ =
            null_count > 0 ? std::optional<size_t>(null_count) : std::nullopt;
    }

    virtual ~ColumnVector() override {
        values_.reset();
        valid_values_.reset();
    }

    void*
    RawValueAt(size_t index, size_t size_of_element) override {
        return reinterpret_cast<char*>(GetRawData()) + index * size_of_element;
    }

    template <typename T>
    T
    ValueAt(size_t index) const {
        return ValueAtRef<T>(index);
    }

    // Like ValueAt, but returns a reference instead of a copy. The reference
    // is valid only while this vector is alive and unchanged.
    template <typename T>
    const T&
    ValueAtRef(size_t index) const {
        AssertInfo(index < length_, "ValueAtRef index out of range");
        return *(reinterpret_cast<const T*>(GetRawData()) + index);
    }

    template <typename T>
    void
    SetValueAt(size_t index, const T& value) {
        AssertInfo(index < length_, "SetValueAt index out of range");
        *(reinterpret_cast<T*>(values_->Data()) + index) = value;
    }

    void
    nullAt(size_t index) {
        // Update null_count_ when setting a value to null
        if (valid_values_[index]) {
            // Was valid, now null: increment count
            null_count_ = null_count_.value_or(0) + 1;
        }
        valid_values_.set(index, false);
    }

    void
    clearNullAt(size_t index) {
        // Update null_count_ when clearing a null value
        if (!valid_values_[index]) {
            // Was null, now valid: decrement count
            if (null_count_.has_value()) {
                null_count_ =
                    (null_count_.value() > 0) ? null_count_.value() - 1 : 0;
            }
        }
        valid_values_.set(index, true);
    }

    bool
    ValidAt(size_t index) override {
        return valid_values_[index];
    }

    // Const access for adapters; existing mutable accessors remain unchanged.
    const TargetBitmap&
    Validity() const {
        return valid_values_;
    }

    void*
    GetRawData() const {
        return values_->Data();
    }

    void*
    GetValidRawData() {
        return valid_values_.data();
    }

    template <typename As>
    As*
    RawAsValues() const {
        return reinterpret_cast<As*>(values_->Data());
    }

    bool
    IsBitmap() const {
        return is_bitmap_;
    }

    void
    resize(vector_size_t new_size, bool setNotNull = true) override {
        AssertInfo(new_size >= 0, "negative column size");
        if (static_cast<size_t>(new_size) == length_) {
            return;
        }
        if (is_bitmap_) {
            TargetBitmap bits(new_size, false);
            const auto count = std::min(length_, static_cast<size_t>(new_size));
            if (count != 0) {
                TargetBitmap::policy_type::op_copy(
                    static_cast<const TargetBitmap::data_type*>(
                        values_->Data()),
                    0,
                    bits.data(),
                    0,
                    count);
            }
            TargetBitmap valid(valid_values_.view());
            valid.resize(new_size, setNotNull);
            auto data = std::make_shared<FieldBitsetImpl<uint8_t>>(
                DataType::INT8, std::move(bits));
            values_ = std::move(data);
            valid_values_ = std::move(valid);
            length_ = new_size;
            null_count_ = length_ - valid_values_.count();
            return;
        }
        // Allocate validity before touching values, and publish length only
        // after both resize operations have succeeded.
        TargetBitmap valid(valid_values_.view());
        valid.resize(new_size, setNotNull);
        ResizeScalarFieldData(type(), new_size, values_);
        valid_values_ = std::move(valid);
        length_ = new_size;
        null_count_ = length_ - valid_values_.count();
    }

    void
    append(const ColumnVector& other) {
        // Validate that both vectors have the same type
        AssertInfo(type() == other.type(),
                   "Cannot append ColumnVector with different type: {} != {}",
                   static_cast<int>(type()),
                   static_cast<int>(other.type()));

        auto old_size = length_;
        values_->FillFieldData(other.GetRawData(), other.size());

        auto values_length = values_->Length();
        AssertInfo(values_length >= old_size,
                   "ColumnVector append length mismatch: values_length {} < "
                   "old_size {}",
                   values_length,
                   old_size);
        length_ = values_length;
        valid_values_.resize(length_);

        // Copy validity from other
        for (size_t i = 0; i < other.size(); ++i) {
            valid_values_.set(old_size + i, other.valid_values_[i]);
        }

        // Update null_count_ by accumulating the original null_count_ and other's null count
        size_t other_nulls = 0;
        if (other.null_count_.has_value()) {
            // Use cached null_count_ if available
            other_nulls = other.null_count_.value();
        } else {
            // Compute null count from other.valid_values_ if not cached
            for (size_t i = 0; i < other.size(); ++i) {
                if (!other.valid_values_[i]) {
                    ++other_nulls;
                }
            }
        }

        // Accumulate null counts
        size_t total_nulls = null_count_.value_or(0) + other_nulls;
        null_count_ = (total_nulls > 0) ? std::optional<size_t>{total_nulls}
                                        : std::nullopt;
    }

    VectorPtr
    cloneEmpty(vector_size_t size) const override {
        AssertInfo(size >= 0, "negative column size");
        if (is_bitmap_) {
            return std::make_shared<ColumnVector>(TargetBitmap(size, false),
                                                  TargetBitmap(size, true));
        }
        return std::make_shared<ColumnVector>(
            InitScalarFieldDataWithLength(type(), size),
            TargetBitmap(size, true),
            0);
    }

    // Check if all rows are definitely TRUE (valid=1, data=1)
    // Uses early termination for efficiency
    // Can only be called when ColumnVector is a bitmap
    bool
    AllTrue() const {
        AssertInfo(is_bitmap_, "AllTrue can only be called on bitmap vectors");
        const size_t len = length_;
        if (len == 0) {
            return true;
        }

        const uint64_t* data =
            reinterpret_cast<const uint64_t*>(values_->Data());
        const uint64_t* valid =
            reinterpret_cast<const uint64_t*>(valid_values_.data());

        const size_t num_full_words = len / 64;
        const size_t tail_bits = len % 64;

        // For AllTrue: every bit must have valid=1 AND data=1
        // (data & valid) == all_ones implies both are all 1s
        constexpr uint64_t all_ones = ~0ULL;

        // Process full 64-bit words with early termination
        for (size_t i = 0; i < num_full_words; ++i) {
            if ((data[i] & valid[i]) != all_ones) {
                return false;
            }
        }

        // Process remaining bits (if any)
        if (tail_bits > 0) {
            const uint64_t mask = (1ULL << tail_bits) - 1;
            if (((data[num_full_words] & valid[num_full_words]) & mask) !=
                mask) {
                return false;
            }
        }

        return true;
    }

    // Check if all rows are definitely FALSE (valid=1, data=0)
    // Uses early termination for efficiency
    // Can only be called when ColumnVector is a bitmap
    bool
    AllFalse() const {
        AssertInfo(is_bitmap_, "AllFalse can only be called on bitmap vectors");
        const size_t len = length_;
        if (len == 0) {
            return true;
        }

        const uint64_t* data =
            reinterpret_cast<const uint64_t*>(values_->Data());
        const uint64_t* valid =
            reinterpret_cast<const uint64_t*>(valid_values_.data());

        const size_t num_full_words = len / 64;
        const size_t tail_bits = len % 64;

        // For AllFalse: every bit must have valid=1 AND data=0
        // (valid & ~data) == all_ones means all bits are definitely FALSE
        constexpr uint64_t all_ones = ~0ULL;

        // Process full 64-bit words with early termination
        for (size_t i = 0; i < num_full_words; ++i) {
            if ((valid[i] & ~data[i]) != all_ones) {
                return false;
            }
        }

        // Process remaining bits (if any)
        if (tail_bits > 0) {
            const uint64_t mask = (1ULL << tail_bits) - 1;
            if (((valid[num_full_words] & ~data[num_full_words]) & mask) !=
                mask) {
                return false;
            }
        }

        return true;
    }

 private:
    friend class VectorPool;
    bool is_bitmap_;  // TODO: remove the field after implementing BitmapVector
    FieldDataPtr values_;
    TargetBitmap valid_values_;  // false means the value is null
};

using ColumnVectorPtr = std::shared_ptr<ColumnVector>;

template <typename T>
class ConstantVector : public SimpleVector {
 public:
    ConstantVector(DataType data_type,
                   size_t length,
                   const T& val,
                   std::optional<size_t> null_count = std::nullopt)
        : SimpleVector(data_type, length),
          val_(val),
          is_null_(null_count.has_value() && null_count.value() > 0) {
    }

    void*
    RawValueAt(size_t _index, size_t _size_of_element) override {
        return &val_;
    }

    bool
    ValidAt(size_t _index) override {
        return !is_null_;
    }

    const T&
    GetValue() const {
        return val_;
    }

    bool
    IsNull() const {
        return is_null_;
    }

    VectorPtr
    cloneEmpty(vector_size_t size) const override {
        return std::make_shared<ConstantVector<T>>(
            type(), size, val_, std::nullopt);
    }

 private:
    T val_;
    bool is_null_;
};

/**
 * @brief Multi vectors for scalar types
 * mainly using it to pass internal result in segcore scalar engine system
 */
class RowVector : public BaseVector {
 public:
    // Schema-aware batch constructor. The optional validity describes the row
    // itself (e.g. a nullable struct), independently of child-column validity.
    RowVector(RowTypePtr row_type,
              size_t row_count,
              std::vector<VectorPtr> children,
              std::optional<TargetBitmap> validity = std::nullopt);

    // Explicit batch shape, including a nonzero row count with zero columns.
    // Legacy constructors retain their existing length-inference behavior.
    RowVector(size_t row_count, std::vector<VectorPtr> children)
        : BaseVector(DataType::ROW, row_count),
          children_values_(std::move(children)) {
        AssertInfo(row_count <=
                       static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                   "RowVector row count exceeds size() range");
        for (const auto& child : children_values_) {
            AssertInfo(child != nullptr, "RowVector child cannot be null");
            AssertInfo(child->size() == static_cast<int64_t>(row_count),
                       "RowVector child length {} must match row count {}",
                       child->size(),
                       row_count);
        }
        InferRowType();
    }

    RowVector(std::vector<DataType>& data_types,
              size_t length,
              std::optional<size_t> null_count = std::nullopt)
        : BaseVector(DataType::ROW, length, null_count) {
        for (auto& type : data_types) {
            children_values_.emplace_back(
                std::make_shared<ColumnVector>(type, length));
        }
        InferRowType();
        InitRowValidity(null_count);
    }

    RowVector(const std::vector<VectorPtr>& children)
        : BaseVector(DataType::ROW, 0) {
        for (auto& child : children) {
            AssertInfo(child, "RowVector child cannot be null");
            children_values_.push_back(child);
            if (child->size() > length_) {
                length_ = child->size();
            }
        }
        InferRowType();
    }

    RowVector(std::vector<VectorPtr>&& children)
        : BaseVector(DataType::ROW, 0), children_values_(std::move(children)) {
        for (auto& child : children_values_) {
            AssertInfo(child, "RowVector child cannot be null");
            if (child->size() > length_) {
                length_ = child->size();
            }
        }
        InferRowType();
    }

    RowVector(const RowTypePtr& rowType,
              size_t size,
              std::optional<size_t> nullCount = std::nullopt)
        : BaseVector(DataType::ROW, size, nullCount) {
        AssertInfo(rowType, "RowVector schema cannot be null");
        row_type_ = rowType;
        auto column_count = rowType->column_count();
        for (auto i = 0; i < column_count; i++) {
            auto column_type = rowType->column_type(i);
            children_values_.emplace_back(
                std::make_shared<ColumnVector>(column_type, size));
        }
        InitRowValidity(nullCount);
    }

    const RowTypePtr&
    row_type() const {
        return row_type_;
    }

    bool
    IsValid(size_t row) const {
        AssertInfo(row < length_, "row index out of bounds");
        return !row_validity_ || (*row_validity_)[row];
    }

    void
    SetRowValid(size_t row, bool valid);

    const std::vector<VectorPtr>&
    childrens() const {
        return children_values_;
    }

    VectorPtr
    child(int index) const {
        AssertInfo(
            index >= 0 && static_cast<size_t>(index) < children_values_.size(),
            "RowVector child index out of bounds");
        return children_values_[index];
    }

    VectorPtr
    child(const std::string& name) const {
        return child(row_type_->GetChildIndex(name));
    }

    void
    resize(vector_size_t new_size, bool setNotNull = true) override;

    // Drop this batch's children without mutating buffers still held by other
    // batches. Fresh empty children retain their schema and physical kind.
    void
    prepareForReuse() override;

    VectorPtr
    cloneEmpty(vector_size_t size) const override {
        AssertInfo(size >= 0, "negative vector size");
        std::vector<VectorPtr> children;
        for (const auto& child : children_values_) {
            children.push_back(child->cloneEmpty(size));
        }
        return std::make_shared<RowVector>(
            row_type_, size, std::move(children));
    }

 private:
    void
    InferRowType();

    void
    InitRowValidity(std::optional<size_t> null_count);

    RowTypePtr row_type_;
    std::optional<TargetBitmap> row_validity_;
    std::vector<VectorPtr> children_values_;
};

using RowVectorPtr = std::shared_ptr<RowVector>;
}  // namespace milvus
