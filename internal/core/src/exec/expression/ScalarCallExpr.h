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
#include "exec/expression/function/FunctionFactory.h"

namespace milvus::exec::scalar {

// Selection-native functions must inspect/compute only rows. The returned
// immutable result has batch length and at least the requested coverage.
using ScalarFunction = std::function<ReadOnlyVectorView(
    const std::vector<ReadOnlyVectorView>&, SelectionView, EvalCtx&)>;

ExprPtr
MakeCall(DataType result_type,
         std::vector<ExprPtr> inputs,
         ScalarFunction function);

// Explicit migration adapter, not an automatic fallback on failure. Only
// selected rows are gathered and materialized for legacy SimpleVector kernels;
// their result is scattered back into the original batch coordinate domain.
ExprPtr
MakeLegacyCall(DataType result_type,
               std::vector<ExprPtr> inputs,
               expression::FilterFunctionPtr function);

}  // namespace milvus::exec::scalar
