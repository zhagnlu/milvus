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

#include "UnaryExpr.h"

namespace milvus {
namespace exec {

void
PhyUnaryRangeFilterExpr::Eval(EvalCtx& context, VectorPtr& result) {
    switch (expr_->column_.data_type_) {
        case DataType::BOOL: {
            result = ExecRangeVisitorImpl<bool>();
            break;
        }
        case DataType::INT8: {
            result = ExecRangeVisitorImpl<int8_t>();
            break;
        }
        case DataType::INT16: {
            result = ExecRangeVisitorImpl<int16_t>();
            break;
        }
        case DataType::INT32: {
            result = ExecRangeVisitorImpl<int32_t>();
            break;
        }
        case DataType::INT64: {
            result = ExecRangeVisitorImpl<int64_t>();
            break;
        }
        case DataType::FLOAT: {
            result = ExecRangeVisitorImpl<float>();
            break;
        }
        case DataType::DOUBLE: {
            result = ExecRangeVisitorImpl<double>();
            break;
        }
        case DataType::VARCHAR: {
            if (segment_->type() == SegmentType::Growing) {
                result = ExecRangeVisitorImpl<std::string>();
            } else {
                result = ExecRangeVisitorImpl<std::string_view>();
            }
            break;
        }
        case DataType::JSON: {
            // switch (expr_->val_) {
            //     default:
            //         PanicInfo(
            //             fmt::format("unknown data type: {}", expr_->val_));
            // }
            break;
        }
        default:
            PanicInfo(fmt::format("unsupported data type: {}",
                                  expr_->column_.data_type_));
    }
}

template <typename T>
VectorPtr
PhyUnaryRangeFilterExpr::ExecRangeVisitorImpl() {
    if (is_index_mode_) {
        return ExecRangeVisitorImplForIndex<T>();
    } else {
        return ExecRangeVisitorImplForData<T>();
    }
}

template <typename T>
VectorPtr
PhyUnaryRangeFilterExpr::ExecRangeVisitorImplForIndex() {
    typedef std::
        conditional_t<std::is_same_v<T, std::string_view>, std::string, T>
            IndexInnerType;
    using Index = index::ScalarIndex<IndexInnerType>;

    if (current_index_chunk_ == num_index_chunk_) {
        return nullptr;
    }

    auto val = GetValueFromProto<IndexInnerType>(expr_->val_);
    const Index& index = segment_->chunk_scalar_index<IndexInnerType>(
        field_id_, current_index_chunk_++);
    auto* index_ptr = const_cast<Index*>(&index);
    FixedVector<bool> res;
    switch (expr_->op_type_) {
        case proto::plan::GreaterThan: {
            UnaryIndexFunc<T, proto::plan::GreaterThan> func;
            res = std::move(func(index_ptr, val));
            break;
        }
        case proto::plan::GreaterEqual: {
            UnaryIndexFunc<T, proto::plan::GreaterEqual> func;
            res = std::move(func(index_ptr, val));
            break;
        }
        case proto::plan::LessThan: {
            UnaryIndexFunc<T, proto::plan::LessThan> func;
            res = std::move(func(index_ptr, val));
            break;
        }
        case proto::plan::LessEqual: {
            UnaryIndexFunc<T, proto::plan::LessEqual> func;
            res = std::move(func(index_ptr, val));
            break;
        }
        case proto::plan::Equal: {
            UnaryIndexFunc<T, proto::plan::Equal> func;
            res = std::move(func(index_ptr, val));
            break;
        }
        case proto::plan::NotEqual: {
            UnaryIndexFunc<T, proto::plan::NotEqual> func;
            res = std::move(func(index_ptr, val));
            break;
        }
        default:
            PanicInfo(
                fmt::format("unsupported operator type for unary expr: {}",
                            expr_->column_.data_type_));
    }
    AssertInfo(res.size() == size_per_chunk_,
               "unary expr: size not equal to size_per_chunk");
    return std::make_shared<FlatVector>(std::move(res));
}

template <typename T>
VectorPtr
PhyUnaryRangeFilterExpr::ExecRangeVisitorImplForData() {
    int batch_size = batch_size_;
    int chunk_id = 0;
    int data_pos = 0;
    if (segment_->type() == SegmentType::Growing) {
        if (current_data_chunk_ > num_data_chunk_) {
            return nullptr;
        }
        // Multi chunks, at most one chunk every loop
        batch_size = current_data_chunk_ == num_data_chunk_
                         ? num_rows_ % size_per_chunk_
                         : size_per_chunk_;
        chunk_id = current_data_chunk_++;
    } else if (segment_->type() == SegmentType::Sealed) {
        if (current_data_chunk_pos_ >= num_rows_) {
            return nullptr;
        }
        // Only one chunk, get batch size for every loop
        batch_size = current_data_chunk_pos_ + batch_size_ <= num_rows_
                         ? batch_size_
                         : num_rows_ - current_data_chunk_pos_;
        data_pos = current_data_chunk_pos_;
        current_data_chunk_pos_ += batch_size;
    }

    auto res_vec = std::make_shared<FlatVector>(DataType::BOOL, batch_size);
    bool* res = (bool*)res_vec->GetRawData();
    auto val = GetValueFromProto<T>(expr_->val_);
    auto chunk = segment_->chunk_data<T>(field_id_, chunk_id);
    const T* data = chunk.data() + data_pos;
    switch (expr_->op_type_) {
        case proto::plan::GreaterThan: {
            UnaryElementFunc<T, proto::plan::GreaterThan> func;
            func(data, batch_size, val, res);
            break;
        }
        case proto::plan::GreaterEqual: {
            UnaryElementFunc<T, proto::plan::GreaterEqual> func;
            func(data, batch_size, val, res);
            break;
        }
        case proto::plan::LessThan: {
            UnaryElementFunc<T, proto::plan::LessThan> func;
            func(data, batch_size, val, res);
            break;
        }
        case proto::plan::LessEqual: {
            UnaryElementFunc<T, proto::plan::LessEqual> func;
            func(data, batch_size, val, res);
            break;
        }
        case proto::plan::Equal: {
            UnaryElementFunc<T, proto::plan::Equal> func;
            func(data, batch_size, val, res);
            break;
        }
        case proto::plan::NotEqual: {
            UnaryElementFunc<T, proto::plan::NotEqual> func;
            func(data, batch_size, val, res);
            break;
        }
        default:
            PanicInfo(
                fmt::format("unsupported operator type for unary expr: {}",
                            expr_->column_.data_type_));
    }
    return res_vec;
}

}  //namespace exec
}  // namespace milvus
