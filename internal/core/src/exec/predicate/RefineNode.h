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
#include "exec/predicate/IndexNode.h"
#include "exec/expression/ScalarExpr.h"

namespace milvus::exec::predicate {

// Explicit exact subprogram, no TTL injection and no Reader -> Expr callback.
// Unproven outside-candidate rows still require exact NULL/FALSE resolution.
// Dedicated GIS/NGRAM extraction/fusion is not migrated by this generic node.
class RefineNode {
 public:
    RefineNode(std::shared_ptr<IndexNode> coarse, scalar::ExprPtr exact);
    ReadOnlyVectorView
    Eval(const ReadBatchBinding&, SelectionView, scalar::EvalCtx&);

 private:
    std::shared_ptr<IndexNode> coarse_;
    scalar::ExprPtr exact_;
    bool evaluating_ = false;
};
}  // namespace milvus::exec::predicate
