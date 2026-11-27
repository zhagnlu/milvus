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

#include "exec/expression/ScalarExprSet.h"

namespace milvus::exec::scalar {

ExprSet::ExprSet(std::vector<ExprPtr> roots) : roots_(std::move(roots)) {
    for (const auto& root : roots_) {
        AssertInfo(root, "scalar expression set contains a null root");
    }
}

std::vector<ReadOnlyVectorView>
ExprSet::Eval(SelectionView rows, EvalCtx& ctx) const {
    AssertInfo(rows.row_count() == ctx.input().size(),
               "scalar expression set selection length mismatch");
    return Eval(std::vector<SelectionView>(roots_.size(), rows), ctx);
}

std::vector<ReadOnlyVectorView>
ExprSet::Eval(const std::vector<SelectionView>& rows, EvalCtx& ctx) const {
    AssertInfo(rows.size() == roots_.size(), "root selection count mismatch");
    // Check all row domains before any root can cause an input load.
    for (const auto& selection : rows) {
        AssertInfo(selection.row_count() == ctx.input().size(),
                   "root selection length mismatch");
    }
    std::vector<ReadOnlyVectorView> results;
    results.reserve(roots_.size());
    for (size_t i = 0; i < roots_.size(); ++i) {
        results.push_back(roots_[i]->Eval(rows[i], ctx));
    }
    return results;
}

}  // namespace milvus::exec::scalar
