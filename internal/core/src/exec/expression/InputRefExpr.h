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

#include "exec/expression/ScalarExpr.h"

namespace milvus::exec::scalar {

class InputRefExpr final : public Expr {
 public:
    InputRefExpr(column_index_t slot, DataType type) : Expr(type), slot_(slot) {
    }

    column_index_t
    slot() const {
        return slot_;
    }

    // IS NULL can use validity-only loaders without materializing payloads.
    ValidityResult
    GetValidity(SelectionView rows, EvalCtx& ctx) const;

 protected:
    ReadOnlyVectorView
    EvalImpl(SelectionView rows, EvalCtx& ctx) const override;

 private:
    void
    CheckBinding(const EvalCtx& ctx) const;

    column_index_t slot_;
};

}  // namespace milvus::exec::scalar
