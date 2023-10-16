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

#include "TermExpr.h"

namespace milvus {
namespace exec {

void
PhyTermFilterExpr::Eval(EvalCtx& context, VectorPtr& result) {
    switch (expr_->column_.data_type_) {
        case DataType::BOOL: {
            result = ExecVisitorImpl<bool>();
            break;
        }
        case DataType::INT8: {
            result = ExecVisitorImpl<int8_t>();
            break;
        }
        case DataType::INT16: {
            result = ExecVisitorImpl<int16_t>();
            break;
        }
        case DataType::INT32: {
            result = ExecVisitorImpl<int32_t>();
            break;
        }
        case DataType::INT64: {
            result = ExecVisitorImpl<int64_t>();
            break;
        }
        case DataType::FLOAT: {
            result = ExecVisitorImpl<float>();
            break;
        }
        case DataType::DOUBLE: {
            result = ExecVisitorImpl<double>();
            break;
        }
        case DataType::VARCHAR: {
            if (segment_->type() == SegmentType::Growing) {
                result = ExecVisitorImpl<std::string>();
            } else {
                result = ExecVisitorImpl<std::string_view>();
            }
            break;
        }
        default:
            PanicInfo(fmt::format("unsupported data type: {}",
                                  expr_->column_.data_type_));
    }
}

template <typename T>
VectorPtr
PhyTermFilterExpr::ExecVisitorImpl() {
    if (is_index_mode_) {
        return ExecVisitorImplForIndex<T>();
    } else {
        return ExecVisitorImplForData<T>();
    }
}

template <typename T>
VectorPtr
PhyTermFilterExpr::ExecVisitorImplForIndex() {
    typedef std::
        conditional_t<std::is_same_v<T, std::string_view>, std::string, T>
            IndexInnerType;
    using Index = index::ScalarIndex<IndexInnerType>;
    auto real_batch_size = GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }

    std::vector<IndexInnerType> vals;
    for (auto& val : expr_->vals_) {
        vals.emplace_back(GetValueFromProto<IndexInnerType>(val));
    }
    auto execute_sub_batch = [](Index* index_ptr,
                                const std::vector<IndexInnerType>& vals) {
        TermIndexFunc<T> func;
        return func(index_ptr, vals.size(), vals.data());
    };
    auto res = ProcessIndexChunks<T>(execute_sub_batch, vals);
    AssertInfo(res.size() == real_batch_size,
               fmt::format("internal error: expr processed rows {} not equal "
                           "expect batch size {}",
                           res.size(),
                           real_batch_size));
    return std::make_shared<FlatVector>(std::move(res));
}

template <>
VectorPtr
PhyTermFilterExpr::ExecVisitorImplForIndex<bool>() {
    using Index = index::ScalarIndex<bool>;

    if (current_index_chunk_ == num_index_chunk_) {
        return nullptr;
    }

    std::vector<uint8_t> vals;
    for (auto& val : expr_->vals_) {
        vals.emplace_back(GetValueFromProto<bool>(val) ? 1 : 0);
    }
    auto execute_sub_batch = [](Index* index_ptr,
                                const std::vector<uint8_t>& vals) {
        TermIndexFunc<bool> func;
        return std::move(func(index_ptr, vals.size(), (bool*)vals.data()));
    };
    auto res = ProcessIndexChunks<bool>(execute_sub_batch, vals);
    return std::make_shared<FlatVector>(std::move(res));
}

template <typename T>
VectorPtr
PhyTermFilterExpr::ExecVisitorImplForData() {
    auto real_batch_size = GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }
    auto res_vec =
        std::make_shared<FlatVector>(DataType::BOOL, real_batch_size);
    bool* res = (bool*)res_vec->GetRawData();
    std::vector<T> vals;
    for (auto& val : expr_->vals_) {
        vals.emplace_back(GetValueFromProto<T>(val));
    }
    std::unordered_set<T> vals_set(vals.begin(), vals.end());
    auto execute_sub_batch = [](const T* data,
                                const int size,
                                bool* res,
                                const std::unordered_set<T>& vals) {
        TermElementFuncSet<T> func;
        for (size_t i = 0; i < size; ++i) {
            res[i] = func(vals, data[i]);
        }
    };
    ProcessDataChunks<T>(execute_sub_batch, res, vals_set);
    return res_vec;
}

}  //namespace exec
}  // namespace milvus
