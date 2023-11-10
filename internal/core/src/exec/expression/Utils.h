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

#include <fmt/core.h>

#include "common/EasyAssert.h"
#include "common/Types.h"
#include "common/Vector.h"
#include "exec/expression/Expr.h"
#include "segcore/SegmentInterface.h"
#include "query/Utils.h"

namespace milvus {
namespace exec {

template <typename T>
bool
CompareTwoJsonArray(T arr1, const proto::plan::Array& arr2) {
    int json_array_length = 0;
    if constexpr (std::is_same_v<
                      T,
                      simdjson::simdjson_result<simdjson::ondemand::array>>) {
        json_array_length = arr1.count_elements();
    }
    if constexpr (std::is_same_v<T,
                                 std::vector<simdjson::simdjson_result<
                                     simdjson::ondemand::value>>>) {
        json_array_length = arr1.size();
    }
    if (arr2.array_size() != json_array_length) {
        return false;
    }
    int i = 0;
    for (auto&& it : arr1) {
        switch (arr2.array(i).val_case()) {
            case proto::plan::GenericValue::kBoolVal: {
                auto val = it.template get<bool>();
                if (val.error() || val.value() != arr2.array(i).bool_val()) {
                    return false;
                }
                break;
            }
            case proto::plan::GenericValue::kInt64Val: {
                auto val = it.template get<int64_t>();
                if (val.error() || val.value() != arr2.array(i).int64_val()) {
                    return false;
                }
                break;
            }
            case proto::plan::GenericValue::kFloatVal: {
                auto val = it.template get<double>();
                if (val.error() || val.value() != arr2.array(i).float_val()) {
                    return false;
                }
                break;
            }
            case proto::plan::GenericValue::kStringVal: {
                auto val = it.template get<std::string_view>();
                if (val.error() || val.value() != arr2.array(i).string_val()) {
                    return false;
                }
                break;
            }
            default:
                PanicInfo(DataTypeInvalid,
                          fmt::format("unsupported data type {}",
                                      int(arr2.array(i).val_case())));
        }
        i++;
    }
    return true;
}

}  // namespace exec
}  // namespace milvus
