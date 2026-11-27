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
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
#include "index/contracts/query/IIndexReaderBase.h"
#include "index/contracts/query/IScalarPredicateReader.h"
#include "index/contracts/query/INullReader.h"
#include "index/contracts/query/IScalarValueReader.h"
#include "common/EasyAssert.h"
#include "pb/plan.pb.h"

namespace milvus::segcore::reader {
template <typename T>
using QueryValue =
    std::conditional_t<std::is_same_v<T, std::string>, std::string_view, T>;

inline const index::INullReader&
NullReader(const index::IIndexReaderBase& base) {
    auto* nulls = dynamic_cast<const index::INullReader*>(&base);
    AssertInfo(nulls, "prepared index does not provide null predicates");
    return *nulls;
}

template <typename T>
class ScalarQueryView {
 public:
    explicit ScalarQueryView(const index::IIndexReaderBase& base)
        : base_(base) {
        predicate_ =
            dynamic_cast<const index::IScalarPredicateReader<QueryValue<T>>*>(
                &base);
        AssertInfo(predicate_, "prepared scalar index physical type mismatch");
    }
    TargetBitmap
    In(size_t n, const T* values) const {
        if constexpr (std::is_same_v<T, std::string>) {
            std::vector<std::string_view> views(values, values + n);
            return predicate_->In(n, views.data());
        } else
            return predicate_->In(n, values);
    }
    TargetBitmap
    NotIn(size_t n, const T* values) const {
        if constexpr (std::is_same_v<T, std::string>) {
            std::vector<std::string_view> views(values, values + n);
            return predicate_->NotIn(n, views.data());
        } else
            return predicate_->NotIn(n, values);
    }
    TargetBitmap
    Range(const T& value, proto::plan::OpType op) const {
        index::CompareOp compare;
        switch (op) {
            case proto::plan::Equal:
                compare = index::CompareOp::Equal;
                break;
            case proto::plan::NotEqual:
                compare = index::CompareOp::NotEqual;
                break;
            case proto::plan::LessThan:
                compare = index::CompareOp::LessThan;
                break;
            case proto::plan::LessEqual:
                compare = index::CompareOp::LessEqual;
                break;
            case proto::plan::GreaterThan:
                compare = index::CompareOp::GreaterThan;
                break;
            case proto::plan::GreaterEqual:
                compare = index::CompareOp::GreaterEqual;
                break;
            default:
                ThrowInfo(UnexpectedError, "invalid prepared comparison");
        }
        return predicate_->Range(value, compare);
    }
    TargetBitmap
    Range(const T& lo, bool lo_inc, const T& hi, bool hi_inc) const {
        return predicate_->Range(lo, lo_inc, hi, hi_inc);
    }
    TargetBitmap
    IsNull() const {
        return NullReader(base_).IsNull();
    }
    TargetBitmap
    IsNotNull() const {
        return NullReader(base_).IsNotNull();
    }

 private:
    const index::IIndexReaderBase& base_;
    const index::IScalarPredicateReader<QueryValue<T>>* predicate_;
};
}  // namespace milvus::segcore::reader
