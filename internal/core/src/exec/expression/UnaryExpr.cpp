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
#include "common/Json.h"

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
            auto val_type = expr_->val_.val_case();
            switch (val_type) {
                case proto::plan::GenericValue::ValCase::kBoolVal:
                    result = ExecUnaryRangeVisitorDispatcherJson<bool>();
                    break;
                case proto::plan::GenericValue::ValCase::kInt64Val:
                    result = ExecUnaryRangeVisitorDispatcherJson<int64_t>();
                    break;
                case proto::plan::GenericValue::ValCase::kFloatVal:
                    result = ExecUnaryRangeVisitorDispatcherJson<double>();
                    break;
                case proto::plan::GenericValue::ValCase::kStringVal:
                    result = ExecUnaryRangeVisitorDispatcherJson<std::string>();
                    break;
                case proto::plan::GenericValue::ValCase::kArrayVal:
                    result = ExecUnaryRangeVisitorDispatcherJson<
                        proto::plan::Array>();
                    break;
                default:
                    PanicInfo(
                        DataTypeInvalid,
                        fmt::format("unknown data type: {}", int(val_type)));
            }
            break;
        }
        case DataType::ARRAY: {
            auto val_type = expr_->val_.val_case();
            switch (val_type) {
                case proto::plan::GenericValue::ValCase::kBoolVal:
                    result = ExecUnaryRangeVisitorDispatcherArray<bool>();
                    break;
                case proto::plan::GenericValue::ValCase::kInt64Val:
                    result = ExecUnaryRangeVisitorDispatcherArray<int64_t>();
                    break;
                case proto::plan::GenericValue::ValCase::kFloatVal:
                    result = ExecUnaryRangeVisitorDispatcherArray<double>();
                    break;
                case proto::plan::GenericValue::ValCase::kStringVal:
                    result =
                        ExecUnaryRangeVisitorDispatcherArray<std::string>();
                    break;
                case proto::plan::GenericValue::ValCase::kArrayVal:
                    result = ExecUnaryRangeVisitorDispatcherArray<
                        proto::plan::Array>();
                    break;
                default:
                    PanicInfo(
                        DataTypeInvalid,
                        fmt::format("unknown data type: {}", int(val_type)));
            }
            break;
        }
        default:
            PanicInfo(DataTypeInvalid,
                      fmt::format("unsupported data type: {}",
                                  expr_->column_.data_type_));
    }
}

template <typename ValueType>
VectorPtr
PhyUnaryRangeFilterExpr::ExecUnaryRangeVisitorDispatcherArray() {
    using GetType = std::conditional_t<std::is_same_v<ValueType, std::string>,
                                       std::string_view,
                                       ValueType>;
    auto real_batch_size = GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }
    auto res_vec =
        std::make_shared<FlatVector>(DataType::BOOL, real_batch_size);
    bool* res = (bool*)res_vec->GetRawData();

    ValueType val = GetValueFromProto<ValueType>(expr_->val_);
    auto op_type = expr_->op_type_;
    int index = -1;
    if (expr_->column_.nested_path_.size() > 0) {
        index = std::stoi(expr_->column_.nested_path_[0]);
    }
    auto execute_sub_batch = [op_type](const milvus::ArrayView* data,
                                       const int size,
                                       bool* res,
                                       ValueType val,
                                       int index) {
        switch (op_type) {
            case proto::plan::GreaterThan: {
                UnaryElementFuncForArray<ValueType, proto::plan::GreaterThan>
                    func;
                func(data, size, val, index, res);
                break;
            }
            case proto::plan::GreaterEqual: {
                UnaryElementFuncForArray<ValueType, proto::plan::GreaterEqual>
                    func;
                func(data, size, val, index, res);
                break;
            }
            case proto::plan::LessThan: {
                UnaryElementFuncForArray<ValueType, proto::plan::LessThan> func;
                func(data, size, val, index, res);
                break;
            }
            case proto::plan::LessEqual: {
                UnaryElementFuncForArray<ValueType, proto::plan::LessEqual>
                    func;
                func(data, size, val, index, res);
                break;
            }
            case proto::plan::Equal: {
                UnaryElementFuncForArray<ValueType, proto::plan::Equal> func;
                func(data, size, val, index, res);
                break;
            }
            case proto::plan::NotEqual: {
                UnaryElementFuncForArray<ValueType, proto::plan::NotEqual> func;
                func(data, size, val, index, res);
                break;
            }
            case proto::plan::PrefixMatch: {
                UnaryElementFuncForArray<ValueType, proto::plan::PrefixMatch>
                    func;
                func(data, size, val, index, res);
                break;
            }
            default:
                PanicInfo(
                    OpTypeInvalid,
                    fmt::format("unsupported operator type for unary expr: {}",
                                int(op_type)));
        }
    };
    int processed_size = ProcessDataChunks<milvus::ArrayView>(
        execute_sub_batch, res, val, index);
    return res_vec;
}

template <typename ExprValueType>
VectorPtr
PhyUnaryRangeFilterExpr::ExecUnaryRangeVisitorDispatcherJson() {
    using GetType =
        std::conditional_t<std::is_same_v<ExprValueType, std::string>,
                           std::string_view,
                           ExprValueType>;
    auto real_batch_size = GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }

    ExprValueType val = GetValueFromProto<ExprValueType>(expr_->val_);
    // std::cout << " real batch size" << real_batch_size << std::endl;
    auto res_vec =
        std::make_shared<FlatVector>(DataType::BOOL, real_batch_size);
    bool* res = (bool*)res_vec->GetRawData();
    auto op_type = expr_->op_type_;
    auto pointer = milvus::Json::pointer(expr_->column_.nested_path_);

#define UnaryRangeJSONCompare(cmp)                             \
    do {                                                       \
        auto x = data[i].template at<GetType>(pointer);        \
        if (x.error()) {                                       \
            if constexpr (std::is_same_v<GetType, int64_t>) {  \
                auto x = data[i].template at<double>(pointer); \
                res[i] = !x.error() && (cmp);                  \
                break;                                         \
            }                                                  \
            res[i] = false;                                    \
            break;                                             \
        }                                                      \
        res[i] = (cmp);                                        \
    } while (false)

#define UnaryRangeJSONCompareNotEqual(cmp)                     \
    do {                                                       \
        auto x = data[i].template at<GetType>(pointer);        \
        if (x.error()) {                                       \
            if constexpr (std::is_same_v<GetType, int64_t>) {  \
                auto x = data[i].template at<double>(pointer); \
                res[i] = x.error() || (cmp);                   \
                break;                                         \
            }                                                  \
            res[i] = true;                                     \
            break;                                             \
        }                                                      \
        res[i] = (cmp);                                        \
    } while (false)

    auto execute_sub_batch = [op_type, pointer](const milvus::Json* data,
                                                const int size,
                                                bool* res,
                                                ExprValueType val) {
        switch (op_type) {
            case proto::plan::GreaterThan: {
                for (size_t i = 0; i < size; ++i) {
                    if constexpr (std::is_same_v<GetType, proto::plan::Array>) {
                        res[i] = false;
                    } else {
                        UnaryRangeJSONCompare(x.value() > val);
                    }
                }
                break;
            }
            case proto::plan::GreaterEqual: {
                for (size_t i = 0; i < size; ++i) {
                    if constexpr (std::is_same_v<GetType, proto::plan::Array>) {
                        res[i] = false;
                    } else {
                        UnaryRangeJSONCompare(x.value() >= val);
                    }
                }
                break;
            }
            case proto::plan::LessThan: {
                for (size_t i = 0; i < size; ++i) {
                    if constexpr (std::is_same_v<GetType, proto::plan::Array>) {
                        res[i] = false;
                    } else {
                        UnaryRangeJSONCompare(x.value() < val);
                    }
                }
                break;
            }
            case proto::plan::LessEqual: {
                for (size_t i = 0; i < size; ++i) {
                    if constexpr (std::is_same_v<GetType, proto::plan::Array>) {
                        res[i] = false;
                    } else {
                        UnaryRangeJSONCompare(x.value() <= val);
                    }
                }
                break;
            }
            case proto::plan::Equal: {
                for (size_t i = 0; i < size; ++i) {
                    if constexpr (std::is_same_v<GetType, proto::plan::Array>) {
                        auto doc = data[i].doc();
                        auto array = doc.at_pointer(pointer).get_array();
                        if (array.error()) {
                            res[i] = false;
                            continue;
                        }
                        res[i] = CompareTwoJsonArray(array, val);
                    } else {
                        UnaryRangeJSONCompare(x.value() == val);
                    }
                }
                break;
            }
            case proto::plan::NotEqual: {
                for (size_t i = 0; i < size; ++i) {
                    if constexpr (std::is_same_v<GetType, proto::plan::Array>) {
                        std::cout << "xxxxx" << std::endl;
                        auto doc = data[i].doc();
                        auto array = doc.at_pointer(pointer).get_array();
                        if (array.error()) {
                            std::cout << "x1xx" << std::endl;
                            res[i] = false;
                            continue;
                        }
                        res[i] = !CompareTwoJsonArray(array, val);

                    } else {
                        UnaryRangeJSONCompareNotEqual(x.value() != val);
                    }
                }
                break;
            }
            default:
                PanicInfo(
                    OpTypeInvalid,
                    fmt::format("unsupported operator type for unary expr: {}",
                                int(op_type)));
        }
    };
    int processed_size =
        ProcessDataChunks<milvus::Json>(execute_sub_batch, res, val);
    return res_vec;
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
    auto real_batch_size = GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }

    auto op_type = expr_->op_type_;
    auto execute_sub_batch = [op_type](Index* index_ptr, IndexInnerType val) {
        FixedVector<bool> res;
        switch (op_type) {
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
            case proto::plan::PrefixMatch: {
                UnaryIndexFunc<T, proto::plan::PrefixMatch> func;
                res = std::move(func(index_ptr, val));
                break;
            }
            default:
                PanicInfo(
                    OpTypeInvalid,
                    fmt::format("unsupported operator type for unary expr: {}",
                                int(op_type)));
        }
        return res;
    };
    auto val = GetValueFromProto<IndexInnerType>(expr_->val_);
    auto res = ProcessIndexChunks<T>(execute_sub_batch, val);
    AssertInfo(res.size() == real_batch_size,
               fmt::format("internal error: expr processed rows {} not equal "
                           "expect batch size {}",
                           res.size(),
                           real_batch_size));
    return std::make_shared<FlatVector>(std::move(res));
}

template <typename T>
VectorPtr
PhyUnaryRangeFilterExpr::ExecRangeVisitorImplForData() {
    auto real_batch_size = GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }

    T val = GetValueFromProto<T>(expr_->val_);
    // std::cout << " real batch size" << real_batch_size << std::endl;
    auto res_vec =
        std::make_shared<FlatVector>(DataType::BOOL, real_batch_size);
    bool* res = (bool*)res_vec->GetRawData();
    auto expr_type = expr_->op_type_;
    auto execute_sub_batch = [expr_type](const T* data,
                                         const int size,
                                         bool* res,
                                         T val) {
        switch (expr_type) {
            case proto::plan::GreaterThan: {
                UnaryElementFunc<T, proto::plan::GreaterThan> func;
                func(data, size, val, res);
                break;
            }
            case proto::plan::GreaterEqual: {
                UnaryElementFunc<T, proto::plan::GreaterEqual> func;
                func(data, size, val, res);
                break;
            }
            case proto::plan::LessThan: {
                UnaryElementFunc<T, proto::plan::LessThan> func;
                func(data, size, val, res);
                break;
            }
            case proto::plan::LessEqual: {
                UnaryElementFunc<T, proto::plan::LessEqual> func;
                func(data, size, val, res);
                break;
            }
            case proto::plan::Equal: {
                UnaryElementFunc<T, proto::plan::Equal> func;
                func(data, size, val, res);
                break;
            }
            case proto::plan::NotEqual: {
                UnaryElementFunc<T, proto::plan::NotEqual> func;
                func(data, size, val, res);
                break;
            }
            case proto::plan::PrefixMatch: {
                UnaryElementFunc<T, proto::plan::PrefixMatch> func;
                func(data, size, val, res);
                break;
            }
            default:
                PanicInfo(
                    OpTypeInvalid,
                    fmt::format("unsupported operator type for unary expr: {}",
                                int(expr_type)));
        }
    };
    int processed_size = ProcessDataChunks<T>(execute_sub_batch, res, val);
    return res_vec;
}
}  //namespace exec
}  // namespace milvus
