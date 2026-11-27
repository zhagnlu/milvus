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

#include "exec/plan/PreparedExecution.h"

namespace milvus::exec::plan {

// Internal batched projection. The caller supplies already selected physical
// positions (including visibility/PK semantics) and bounds each binding.
// This class does not scan an exclusion bitmap or collect segment offsets.
class ScalarProject {
 public:
    ScalarProject(PreparedExecution execution, std::vector<std::string> names);

    const RowTypePtr&
    output_type() const {
        return output_type_;
    }

    // Sparse selections compact in increasing batch-row order. Physical
    // offsets may be repeated/out of order; their order is retained. No result
    // escapes until every expression and the final checkpoint succeed.
    RowVectorPtr
    Eval(const ReadBatchBinding& binding,
         SelectionView rows,
         const RowVector* upstream = nullptr,
         std::function<void()> checkpoint = {}) const;

 private:
    PreparedExecution execution_;
    RowTypePtr output_type_;
};

}  // namespace milvus::exec::plan
