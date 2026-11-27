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

// Multiple typed roots share BatchInput, not mutable expression results.
// Evaluation returns only after all roots succeed. Successful input loads
// remain reusable after a failure; partial root outputs are not published.
class ExprSet {
 public:
    explicit ExprSet(std::vector<ExprPtr> roots);

    const std::vector<ExprPtr>&
    roots() const {
        return roots_;
    }

    std::vector<ReadOnlyVectorView>
    Eval(SelectionView rows, EvalCtx& ctx) const;

    // Each root gets its own selection, in the same batch coordinate domain.
    std::vector<ReadOnlyVectorView>
    Eval(const std::vector<SelectionView>& rows, EvalCtx& ctx) const;

 private:
    std::vector<ExprPtr> roots_;
};

}  // namespace milvus::exec::scalar
