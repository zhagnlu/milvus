// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific
// language governing permissions and limitations under the License.

#pragma once
#include "exec/expression/ScalarExpr.h"
#include "pb/plan.pb.h"

namespace milvus::exec::scalar {
ExprPtr
MakeGeometryPredicate(ExprPtr input,
                      proto::plan::GISFunctionFilterExpr_GISOp operation,
                      std::string wkt,
                      double distance);

// Preserve predicate-major order and lazy WKT errors while parsing each
// selected row once. Only adjacent direct same-slot inputs can be merged.
std::vector<ExprPtr>
FuseAdjacentGeometryPredicates(
    LogicalOp op,
    std::vector<ExprPtr> inputs,
    const std::function<ExprPtr(ExprPtr)>& resolve = {});

// Each input is a document-aligned ARRAY; the child is compiled over scalar
// element slots in the same order. No Segment/reader/implicit cursor here.
ExprPtr
MakeArrayMatch(ExprPtr anchor,
               std::vector<ExprPtr> arrays,
               RowTypePtr element_schema,
               ExprPtr child,
               proto::plan::MatchType match,
               int64_t threshold);
}  // namespace milvus::exec::scalar
