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

#include <map>
#include "common/Schema.h"
#include "exec/plan/CompiledExprPlan.h"
#include "expr/ITypeExpr.h"

namespace milvus::exec::plan {

struct CompileOptions {
    std::string relation;
    // Explicit field-ID -> upstream-column mapping. Missing/wrong mapped
    // columns are rejected, never silently rebound to storage by name.
    RowTypePtr upstream_schema;
    std::map<int64_t, column_index_t> upstream_columns;
    // Opt-in internal physical planning. No production entry switches here.
    // Compilation records alternatives; Prepare chooses from query resources.
    bool plan_indexes = false;
    // Immutable schema metadata for struct membership and MATCH row validity.
    std::shared_ptr<const Schema> schema;
    // MATCH lowers scalar array elements into a local row domain. Preserve
    // array arithmetic coercions when those columns become scalar-typed.
    bool array_element_arithmetic = false;
    std::map<ResultColumnId, column_index_t> upstream_results;
};

struct UnsupportedFeature {
    std::string location;  // root[i]/child[j], never a potentially huge literal
    std::string feature;
};

struct CompileResult {
    CompiledExprPlanPtr plan;
    std::vector<UnsupportedFeature> unsupported;

    bool
    supported() const {
        return plan != nullptr;
    }
};

// Validate struct membership and lower an explicit element row domain to the
// ordinary scalar tree. nullptr is a capability miss, without evaluating it.
expr::TypedExprPtr
LowerElementPredicate(const expr::TypedExprPtr& root,
                      const Schema& schema, const std::string& structure);

// Existing logical Expr tree -> new scalar calculation + automatic input slots.
// No TTL injection, QueryContext, resource binding, reading, or fallback.
// Any unsupported subtree rejects the WHOLE set (no partially usable plan).
CompileResult
CompileExpressions(const std::vector<expr::TypedExprPtr>& roots,
                   const CompileOptions& options = {});

// Filter-only entry: add an explicitly supplied TTL predicate exactly once.
// The caller builds TTL from the appropriate query clock/schema. Projection
// compilation above never invents or applies a filter.
CompileResult
CompileFilter(const expr::TypedExprPtr& predicate,
              const expr::TypedExprPtr& ttl_predicate = nullptr,
              const CompileOptions& options = {});

}  // namespace milvus::exec::plan
