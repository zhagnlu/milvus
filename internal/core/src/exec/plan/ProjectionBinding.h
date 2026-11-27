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

#include "expr/ProjectionExpr.h"
#include "plan/ScalarProjectNode.h"

namespace milvus::exec::plan {
struct ProjectionField {
    std::string name;
    FieldId field;
    DataType type;
    bool nullable = false;
};
struct NamedProjection {
    std::string name;
    expr::TypedExprPtr expression;
    bool visible = true;
};
struct BoundProjectionItem {
    ResultColumnDescriptor output;
    expr::TypedExprPtr expression;
};
struct BoundProjection {
    std::vector<ResultColumnDescriptor> inputs;
    std::vector<BoundProjectionItem> items;
};

// Internal structured AS binder. Names resolve only against storage inputs and
// preceding items. No SQL text parser, function/CASE support or filter aliases.
BoundProjection
BindProjection(const std::vector<ProjectionField>& inputs,
               const std::vector<NamedProjection>& items);

// Validate even hand-built bound plans, then lower to existing scan and data
// batch Project nodes. Visible names are assigned only by the final projection;
// intermediate columns carry stable IDs via explicit upstream mappings.
::milvus::plan::PlanNodePtr
MakeProjectionPlan(const BoundProjection& projection,
                   std::shared_ptr<::milvus::plan::MvccNode> source,
                   const std::string& id_prefix = "projection");
}  // namespace milvus::exec::plan
