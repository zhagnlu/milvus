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

#include <memory>
#include <string>

#include "common/FieldData.h"

namespace milvus {

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
    size() {
        return length_;
    }

    DataType
    type() {
        return type_kind_;
    }

 private:
    DataType type_kind_;
    size_t length_;
    std::optional<size_t> null_count_;
};

using VectorPtr = std::shared_ptr<BaseVector>;

/**
 * @brief Single vector for scalar types
 * @todo using memory pool && buffer replace FieldData
 */
class FlatVector final : public BaseVector {
 public:
    FlatVector(DataType data_type,
               size_t length,
               std::optional<size_t> null_count = std::nullopt)
        : BaseVector(data_type, length, null_count) {
        values_ = InitScalarFieldData(data_type, length);
    }

    FlatVector(FixedVector<bool>&& data)
        : BaseVector(DataType::BOOL, data.size()) {
        values_ =
            std::make_shared<FieldData<bool>>(DataType::BOOL, std::move(data));
    }

    virtual ~FlatVector() override {
        values_.reset();
    }

    void*
    GetRawData() {
        return values_->Data();
    }

    template <typename As>
    const As*
    RawAsValues() const {
        return reinterpret_cast<const As*>(values_->Data());
    }

 private:
    FieldDataPtr values_;
};

using FlatVectorPtr = std::shared_ptr<FlatVector>;

/**
 * @brief Multi vectors for scalar types
 * mainly using it to pass internal result in segcore scalar engine system
 */
class RowVector : public BaseVector {
 public:
    RowVector(std::vector<DataType>& data_types,
              size_t length,
              std::optional<size_t> null_count = std::nullopt)
        : BaseVector(DataType::ROW, length, null_count) {
        for (auto& type : data_types) {
            children_values_.emplace_back(
                std::make_shared<FlatVector>(type, length));
        }
    }

    RowVector(const std::vector<VectorPtr>& children)
        : BaseVector(DataType::ROW,
                     children.size() > 0 ? children[0]->size() : 0) {
        for (auto& child : children) {
            children_values_.push_back(child);
        }
    }

    const std::vector<VectorPtr>&
    childrens() {
        return children_values_;
    }

 private:
    std::vector<VectorPtr> children_values_;
};

using RowVectorPtr = std::shared_ptr<RowVector>;

}  // namespace milvus