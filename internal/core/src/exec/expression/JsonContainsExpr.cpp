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

#include "JsonContainsExpr.h"

namespace milvus {
namespace exec {

void
PhyJsonContainsFilterExpr::Eval(EvalCtx& context, VectorPtr& result) {
    switch (expr_->column_.data_type_) {
        case DataType::ARRAY:
        case DataType::JSON: {
            if (is_index_mode_) {
                PanicInfo(
                    ExprInvalid,
                    "exists expr for json or array index mode not supportted");
            }
            result = EvalJsonContainsForDataSegment();
            break;
        }
        default:
            PanicInfo(DataTypeInvalid,
                      fmt::format("unsupported data type: {}",
                                  expr_->column_.data_type_));
    }
}

VectorPtr
PhyJsonContainsFilterExpr::EvalJsonContainsForDataSegment() {
    auto data_type = expr_->column_.data_type_;
    switch (expr_->op_) {
        case proto::plan::JSONContainsExpr_JSONOp_Contains:
        case proto::plan::JSONContainsExpr_JSONOp_ContainsAny: {
            if (datatype_is_array(data_type)) {
                auto val_type = expr_->vals_[0].val_case();
                switch (val_type) {
                    case proto::plan::GenericValue::kBoolVal: {
                        return ExecArrayContains<bool>();
                    }
                    case proto::plan::GenericValue::kInt64Val: {
                        return ExecArrayContains<int64_t>();
                    }
                    case proto::plan::GenericValue::kFloatVal: {
                        return ExecArrayContains<double>();
                    }
                    case proto::plan::GenericValue::kStringVal: {
                        return ExecArrayContains<std::string>();
                    }
                    default:
                        PanicInfo(DataTypeInvalid,
                                  fmt::format("unsupported data type {}",
                                              int(val_type)));
                }
            } else {
                if (expr_->same_type_) {
                    auto val_type = expr_->vals_[0].val_case();
                    switch (val_type) {
                        case proto::plan::GenericValue::kBoolVal: {
                            return ExecJsonContains<bool>();
                        }
                        case proto::plan::GenericValue::kInt64Val: {
                            return ExecJsonContains<int64_t>();
                        }
                        case proto::plan::GenericValue::kFloatVal: {
                            return ExecJsonContains<double>();
                        }
                        case proto::plan::GenericValue::kStringVal: {
                            return ExecJsonContains<std::string>();
                        }
                        case proto::plan::GenericValue::kArrayVal: {
                            return ExecJsonContainsArray();
                        }
                        default:
                            PanicInfo(DataTypeInvalid,
                                      fmt::format("unsupported data type:{}",
                                                  int(val_type)));
                    }
                } else {
                    return ExecJsonContainsWithDiffType();
                }
            }
        }
        case proto::plan::JSONContainsExpr_JSONOp_ContainsAll: {
            if (expr_->same_type_) {
                auto val_type = expr_->vals_[0].val_case();
                switch (val_type) {
                    case proto::plan::GenericValue::kBoolVal: {
                        return ExecJsonContainsAll<bool>();
                    }
                    case proto::plan::GenericValue::kInt64Val: {
                        return ExecJsonContainsAll<int64_t>();
                    }
                    case proto::plan::GenericValue::kFloatVal: {
                        return ExecJsonContainsAll<double>();
                    }
                    case proto::plan::GenericValue::kStringVal: {
                        return ExecJsonContainsAll<std::string>();
                    }
                    case proto::plan::GenericValue::kArrayVal: {
                        return ExecJsonContainsAllArray();
                    }
                    default:
                        PanicInfo(DataTypeInvalid,
                                  fmt::format("unsupported data type:{}",
                                              int(val_type)));
                }
            } else {
                return ExecJsonContainsAllWithDiffType();
            }
        }
        default:
            PanicInfo(ExprInvalid,
                      fmt::format("unsupported json contains type"));
    }
}

template <typename ExprValueType>
VectorPtr
PhyJsonContainsFilterExpr::ExecArrayContains() {
    using GetType =
        std::conditional_t<std::is_same_v<ExprValueType, std::string>,
                           std::string_view,
                           ExprValueType>;
    auto real_batch_size = GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }
    AssertInfo(expr_->column_.nested_path_.size() == 0,
               "[ExecArrayContains]nested path must be null");

    auto res_vec =
        std::make_shared<FlatVector>(DataType::BOOL, real_batch_size);
    bool* res = (bool*)res_vec->GetRawData();
    std::unordered_set<GetType> elements;
    for (auto const& element : expr_->vals_) {
        elements.insert(GetValueFromProto<ExprValueType>(element));
    }
    auto execute_sub_batch = [](const milvus::ArrayView* data,
                                const int size,
                                bool* res,
                                const std::unordered_set<GetType>& elements) {
        for (int i = 0; i < size; ++i) {
            const auto& array = data[i];
            for (int j = 0; j < array.length(); ++j) {
                if (elements.count(array.template get_data<GetType>(j)) > 0) {
                    res[i] = true;
                    break;
                }
            }
            res[i] = false;
        }
    };

    int processed_size =
        ProcessDataChunks<milvus::ArrayView>(execute_sub_batch, res, elements);
    return res_vec;
}

template <typename ExprValueType>
VectorPtr
PhyJsonContainsFilterExpr::ExecJsonContains() {
    using GetType =
        std::conditional_t<std::is_same_v<ExprValueType, std::string>,
                           std::string_view,
                           ExprValueType>;
    auto real_batch_size = GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }

    auto res_vec =
        std::make_shared<FlatVector>(DataType::BOOL, real_batch_size);
    bool* res = (bool*)res_vec->GetRawData();
    std::unordered_set<GetType> elements;
    auto pointer = milvus::Json::pointer(expr_->column_.nested_path_);
    for (auto const& element : expr_->vals_) {
        elements.insert(GetValueFromProto<ExprValueType>(element));
    }
    auto execute_sub_batch = [](const milvus::Json* data,
                                const int size,
                                bool* res,
                                const std::string& pointer,
                                const std::unordered_set<GetType>& elements) {
        auto executor = [&](size_t i) {
            auto doc = data[i].doc();
            auto array = doc.at_pointer(pointer).get_array();
            if (array.error()) {
                return false;
            }
            for (auto&& it : array) {
                auto val = it.template get<GetType>();
                if (val.error()) {
                    continue;
                }
                if (elements.count(val.value()) > 0) {
                    return true;
                }
            }
            return false;
        };
        for (size_t i = 0; i < size; ++i) {
            res[i] = executor(i);
        }
    };

    int processed_size =
        ProcessDataChunks<Json>(execute_sub_batch, res, pointer, elements);
    return res_vec;
}

VectorPtr
PhyJsonContainsFilterExpr::ExecJsonContainsArray() {
    auto real_batch_size = GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }
    auto res_vec =
        std::make_shared<FlatVector>(DataType::BOOL, real_batch_size);
    bool* res = (bool*)res_vec->GetRawData();
    auto pointer = milvus::Json::pointer(expr_->column_.nested_path_);
    std::vector<proto::plan::Array> elements;
    for (auto const& element : expr_->vals_) {
        elements.emplace_back(GetValueFromProto<proto::plan::Array>(element));
    }
    auto execute_sub_batch =
        [](const milvus::Json* data,
           const int size,
           bool* res,
           const std::string& pointer,
           const std::vector<proto::plan::Array>& elements) {
            auto executor = [&](size_t i) -> bool {
                auto doc = data[i].doc();
                auto array = doc.at_pointer(pointer).get_array();
                if (array.error()) {
                    return false;
                }
                for (auto&& it : array) {
                    auto val = it.get_array();
                    if (val.error()) {
                        continue;
                    }
                    std::vector<
                        simdjson::simdjson_result<simdjson::ondemand::value>>
                        json_array;
                    json_array.reserve(val.count_elements());
                    for (auto&& e : val) {
                        json_array.emplace_back(e);
                    }
                    for (auto const& element : elements) {
                        if (CompareTwoJsonArray(json_array, element)) {
                            return true;
                        }
                    }
                }
                return false;
            };
            for (size_t i = 0; i < size; ++i) {
                res[i] = executor(i);
            }
        };

    int processed_size = ProcessDataChunks<milvus::Json>(
        execute_sub_batch, res, pointer, elements);
    return res_vec;
}

template <typename ExprValueType>
VectorPtr
PhyJsonContainsFilterExpr::ExecJsonContainsAll() {
    using GetType =
        std::conditional_t<std::is_same_v<ExprValueType, std::string>,
                           std::string_view,
                           ExprValueType>;
    auto real_batch_size = GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }
    auto res_vec =
        std::make_shared<FlatVector>(DataType::BOOL, real_batch_size);
    bool* res = (bool*)res_vec->GetRawData();
    auto pointer = milvus::Json::pointer(expr_->column_.nested_path_);
    std::unordered_set<GetType> elements;
    for (auto const& element : expr_->vals_) {
        elements.insert(GetValueFromProto<GetType>(element));
    }

    auto execute_sub_batch = [](const milvus::Json* data,
                                const int size,
                                bool* res,
                                const std::string& pointer,
                                const std::unordered_set<GetType>& elements) {
        auto executor = [&](const size_t i) -> bool {
            auto doc = data[i].doc();
            auto array = doc.at_pointer(pointer).get_array();
            if (array.error()) {
                return false;
            }
            std::unordered_set<GetType> tmp_elements(elements);
            // Note: array can only be iterated once
            for (auto&& it : array) {
                auto val = it.template get<GetType>();
                if (val.error()) {
                    continue;
                }
                tmp_elements.erase(val.value());
                if (tmp_elements.size() == 0) {
                    return true;
                }
            }
            return tmp_elements.size() == 0;
        };
        for (size_t i = 0; i < size; ++i) {
            res[i] = executor(i);
        }
    };

    int processed_size =
        ProcessDataChunks<Json>(execute_sub_batch, res, pointer, elements);
    return res_vec;
}

VectorPtr
PhyJsonContainsFilterExpr::ExecJsonContainsAllWithDiffType() {
    auto real_batch_size = GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }
    auto res_vec =
        std::make_shared<FlatVector>(DataType::BOOL, real_batch_size);
    bool* res = (bool*)res_vec->GetRawData();
    auto pointer = milvus::Json::pointer(expr_->column_.nested_path_);

    auto elements = expr_->vals_;
    std::unordered_set<int> elements_index;
    int i = 0;
    for (auto& element : elements) {
        elements_index.insert(i);
        i++;
    }

    auto execute_sub_batch =
        [](const milvus::Json* data,
           const int size,
           bool* res,
           const std::string& pointer,
           const std::vector<proto::plan::GenericValue>& elements,
           const std::unordered_set<int> elements_index) {
            auto executor = [&](size_t i) -> bool {
                const auto& json = data[i];
                auto doc = json.doc();
                auto array = doc.at_pointer(pointer).get_array();
                if (array.error()) {
                    return false;
                }
                std::unordered_set<int> tmp_elements_index(elements_index);
                for (auto&& it : array) {
                    int i = -1;
                    for (auto& element : elements) {
                        i++;
                        switch (element.val_case()) {
                            case proto::plan::GenericValue::kBoolVal: {
                                auto val = it.template get<bool>();
                                if (val.error()) {
                                    continue;
                                }
                                if (val.value() == element.bool_val()) {
                                    tmp_elements_index.erase(i);
                                }
                                break;
                            }
                            case proto::plan::GenericValue::kInt64Val: {
                                auto val = it.template get<int64_t>();
                                if (val.error()) {
                                    continue;
                                }
                                if (val.value() == element.int64_val()) {
                                    tmp_elements_index.erase(i);
                                }
                                break;
                            }
                            case proto::plan::GenericValue::kFloatVal: {
                                auto val = it.template get<double>();
                                if (val.error()) {
                                    continue;
                                }
                                if (val.value() == element.float_val()) {
                                    tmp_elements_index.erase(i);
                                }
                                break;
                            }
                            case proto::plan::GenericValue::kStringVal: {
                                auto val = it.template get<std::string_view>();
                                if (val.error()) {
                                    continue;
                                }
                                if (val.value() == element.string_val()) {
                                    tmp_elements_index.erase(i);
                                }
                                break;
                            }
                            case proto::plan::GenericValue::kArrayVal: {
                                auto val = it.get_array();
                                if (val.error()) {
                                    continue;
                                }
                                if (CompareTwoJsonArray(val,
                                                        element.array_val())) {
                                    tmp_elements_index.erase(i);
                                }
                                break;
                            }
                            default:
                                PanicInfo(
                                    DataTypeInvalid,
                                    fmt::format("unsupported data type {}",
                                                int(element.val_case())));
                        }
                        if (tmp_elements_index.size() == 0) {
                            return true;
                        }
                    }
                    if (tmp_elements_index.size() == 0) {
                        return true;
                    }
                }
                return tmp_elements_index.size() == 0;
            };
            for (size_t i = 0; i < size; ++i) {
                res[i] = executor(i);
            }
        };

    int processed_size = ProcessDataChunks<Json>(
        execute_sub_batch, res, pointer, elements, elements_index);
    return res_vec;
}

VectorPtr
PhyJsonContainsFilterExpr::ExecJsonContainsAllArray() {
    auto real_batch_size = GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }
    auto res_vec =
        std::make_shared<FlatVector>(DataType::BOOL, real_batch_size);
    bool* res = (bool*)res_vec->GetRawData();
    auto pointer = milvus::Json::pointer(expr_->column_.nested_path_);

    std::vector<proto::plan::Array> elements;
    for (auto const& element : expr_->vals_) {
        elements.emplace_back(GetValueFromProto<proto::plan::Array>(element));
    }
    auto execute_sub_batch =
        [](const milvus::Json* data,
           const int size,
           bool* res,
           const std::string& pointer,
           const std::vector<proto::plan::Array>& elements) {
            auto executor = [&](const size_t i) {
                auto doc = data[i].doc();
                auto array = doc.at_pointer(pointer).get_array();
                if (array.error()) {
                    return false;
                }
                std::unordered_set<int> exist_elements_index;
                for (auto&& it : array) {
                    auto val = it.get_array();
                    if (val.error()) {
                        continue;
                    }
                    std::vector<
                        simdjson::simdjson_result<simdjson::ondemand::value>>
                        json_array;
                    json_array.reserve(val.count_elements());
                    for (auto&& e : val) {
                        json_array.emplace_back(e);
                    }
                    for (int index = 0; index < elements.size(); ++index) {
                        if (CompareTwoJsonArray(json_array, elements[index])) {
                            exist_elements_index.insert(index);
                        }
                    }
                    if (exist_elements_index.size() == elements.size()) {
                        return true;
                    }
                }
                return exist_elements_index.size() == elements.size();
            };
            for (size_t i = 0; i < size; ++i) {
                res[i] = executor(i);
            }
        };

    int processed_size =
        ProcessDataChunks<Json>(execute_sub_batch, res, pointer, elements);
    return res_vec;
}

VectorPtr
PhyJsonContainsFilterExpr::ExecJsonContainsWithDiffType() {
    auto real_batch_size = GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }
    auto res_vec =
        std::make_shared<FlatVector>(DataType::BOOL, real_batch_size);
    bool* res = (bool*)res_vec->GetRawData();
    auto pointer = milvus::Json::pointer(expr_->column_.nested_path_);

    auto elements = expr_->vals_;
    std::unordered_set<int> elements_index;
    int i = 0;
    for (auto& element : elements) {
        elements_index.insert(i);
        i++;
    }

    auto execute_sub_batch =
        [](const milvus::Json* data,
           const int size,
           bool* res,
           const std::string& pointer,
           const std::vector<proto::plan::GenericValue>& elements) {
            auto executor = [&](const size_t i) {
                auto& json = data[i];
                auto doc = json.doc();
                auto array = doc.at_pointer(pointer).get_array();
                if (array.error()) {
                    return false;
                }
                // Note: array can only be iterated once
                for (auto&& it : array) {
                    for (auto const& element : elements) {
                        switch (element.val_case()) {
                            case proto::plan::GenericValue::kBoolVal: {
                                auto val = it.template get<bool>();
                                if (val.error()) {
                                    continue;
                                }
                                if (val.value() == element.bool_val()) {
                                    return true;
                                }
                                break;
                            }
                            case proto::plan::GenericValue::kInt64Val: {
                                auto val = it.template get<int64_t>();
                                if (val.error()) {
                                    continue;
                                }
                                if (val.value() == element.int64_val()) {
                                    return true;
                                }
                                break;
                            }
                            case proto::plan::GenericValue::kFloatVal: {
                                auto val = it.template get<double>();
                                if (val.error()) {
                                    continue;
                                }
                                if (val.value() == element.float_val()) {
                                    return true;
                                }
                                break;
                            }
                            case proto::plan::GenericValue::kStringVal: {
                                auto val = it.template get<std::string_view>();
                                if (val.error()) {
                                    continue;
                                }
                                if (val.value() == element.string_val()) {
                                    return true;
                                }
                                break;
                            }
                            case proto::plan::GenericValue::kArrayVal: {
                                auto val = it.get_array();
                                if (val.error()) {
                                    continue;
                                }
                                if (CompareTwoJsonArray(val,
                                                        element.array_val())) {
                                    return true;
                                }
                                break;
                            }
                            default:
                                PanicInfo(
                                    DataTypeInvalid,
                                    fmt::format("unsupported data type {}",
                                                int(element.val_case())));
                        }
                    }
                }
                return false;
            };
            for (size_t i = 0; i < size; ++i) {
                res[i] = executor(i);
            }
        };

    int processed_size =
        ProcessDataChunks<Json>(execute_sub_batch, res, pointer, elements);
    return res_vec;
}

}  //namespace exec
}  // namespace milvus
