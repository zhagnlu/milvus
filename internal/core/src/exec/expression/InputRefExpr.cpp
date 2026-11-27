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

#include "exec/expression/InputRefExpr.h"

namespace milvus::exec::scalar {

void
InputRefExpr::CheckBinding(const EvalCtx& ctx) const {
    const auto& schema = ctx.input().schema();
    AssertInfo(static_cast<size_t>(slot_) < schema->column_count(),
               "scalar input slot out of bounds");
    AssertInfo(schema->column_type(slot_) == type(),
               "scalar input binding type mismatch");
}

ReadOnlyVectorView
InputRefExpr::EvalImpl(SelectionView rows, EvalCtx& ctx) const {
    CheckBinding(ctx);
    return ctx.input().GetColumn(slot_, rows);
}

ValidityResult
InputRefExpr::GetValidity(SelectionView rows, EvalCtx& ctx) const {
    AssertInfo(rows.row_count() == ctx.input().size(),
               "scalar validity selection length mismatch");
    if (rows.IsNone()) {
        return {OwnedSelection::None(rows.row_count()),
                OwnedSelection::None(rows.row_count())};
    }
    ctx.Checkpoint();
    CheckBinding(ctx);
    return ctx.input().GetValidity(slot_, rows);
}

}  // namespace milvus::exec::scalar
