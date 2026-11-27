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
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include "common/Types.h"
#include "common/ResultColumn.h"

namespace milvus::exec {

// Raw-value dependencies only. Indexed predicates must get distinct dependency
// kinds in the predicate migration; they must not masquerade as field values.
struct InputDependency {
    std::string relation;
    int64_t field_id;
    DataType type;
    DataType element_type = DataType::NONE;
    std::vector<std::string> path;
    bool element_level = false;
    std::optional<column_index_t> upstream_column;
    bool nullable = false;
    bool needs_values = false;
    bool needs_validity = false;
    std::optional<ResultColumnId> result_column;

    bool
    SameSource(const InputDependency& other) const {
        return std::tie(relation,
                        field_id,
                        type,
                        element_type,
                        path,
                        element_level,
                        upstream_column,
                        result_column) == std::tie(other.relation,
                                                   other.field_id,
                                                   other.type,
                                                   other.element_type,
                                                   other.path,
                                                   other.element_level,
                                                   other.upstream_column,
                                                   other.result_column);
    }
};

class InputLayout {
 public:
    column_index_t
    Intern(InputDependency dependency) {
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (slots_[i].SameSource(dependency)) {
                slots_[i].nullable |= dependency.nullable;
                slots_[i].needs_values |= dependency.needs_values;
                slots_[i].needs_validity |=
                    dependency.needs_validity || dependency.needs_values;
                return static_cast<column_index_t>(i);
            }
        }
        AssertInfo(slots_.size() < std::numeric_limits<column_index_t>::max(),
                   "too many scalar input slots");
        dependency.needs_validity |= dependency.needs_values;
        slots_.push_back(std::move(dependency));
        return static_cast<column_index_t>(slots_.size() - 1);
    }

    const std::vector<InputDependency>&
    slots() const {
        return slots_;
    }

    RowTypePtr
    schema() const {
        std::vector<std::string> names;
        std::vector<DataType> types;
        for (size_t i = 0; i < slots_.size(); ++i) {
            names.push_back("slot_" + std::to_string(i));
            types.push_back(slots_[i].type);
        }
        return std::make_shared<RowType>(std::move(names), std::move(types));
    }

 private:
    std::vector<InputDependency> slots_;
};

}  // namespace milvus::exec
