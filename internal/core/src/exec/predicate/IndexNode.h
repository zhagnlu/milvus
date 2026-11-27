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
#include "exec/predicate/PredicateResult.h"
#include "segcore/reader/IndexReader.h"

namespace milvus::exec::predicate {

// Query-local, single-flow state, NOT the global ExprResCache. Bind the same
// instance across this query's batches. Never reuse it across source generations.
// Successful whole-source results are immutable; failed reads publish nothing.
class IndexNode {
 public:
    IndexNode(std::shared_ptr<const segcore::reader::IndexReader> reader,
              std::shared_ptr<const segcore::reader::ReadContext> context,
              bool use_index = true);
    std::shared_ptr<const segcore::reader::IndexResult>
    Read(const ReadBatchBinding& binding, SelectionView rows);
    ReadOnlyVectorView
    Eval(const ReadBatchBinding& binding, SelectionView rows);
    OwnedSelection
    FilterTrue(const ReadBatchBinding& binding, SelectionView rows);

 private:
    std::shared_ptr<const segcore::reader::IndexReader> reader_;
    std::shared_ptr<const segcore::reader::ReadContext> context_;
    std::shared_ptr<const segcore::reader::IndexResult> cached_;
    bool loading_ = false;
};
}  // namespace milvus::exec::predicate
