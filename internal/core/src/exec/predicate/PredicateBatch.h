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
#include "exec/predicate/RefineNode.h"
#include "exec/input/BatchInput.h"

namespace milvus::exec::predicate {

using PredicateSource = std::function<ReadOnlyVectorView(
    const ReadBatchBinding&, SelectionView, scalar::EvalCtx&)>;

PredicateSource
BindIndex(std::shared_ptr<IndexNode> node);
PredicateSource
BindRefine(std::shared_ptr<RefineNode> node);

// One batch's predicate cache, physically separate from its raw-value slots.
// Raw BatchInput must outlive the provider and EvalCtx. Providers cannot be
// rebound to another raw object. Reset changes only the same workspace's batch.
// Single-flow only. Query-local IndexNodes may be shared by successive batches.
class PredicateBatch final : public scalar::PredicateInputs {
 public:
    PredicateBatch(BatchInput& raw,
                   ReadBatchBinding binding,
                   std::vector<PredicateSource> sources);
    PredicateBatch(const PredicateBatch&) = delete;
    PredicateBatch&
    operator=(const PredicateBatch&) = delete;
    PredicateBatch(PredicateBatch&&) = delete;
    PredicateBatch&
    operator=(PredicateBatch&&) = delete;

    void
    Reset(ReadBatchBinding binding);

    ReadOnlyVectorView
    Get(column_index_t, SelectionView, scalar::EvalCtx&) override;

 private:
    BatchInput& raw_;
    ReadBatchBinding binding_;
    std::vector<PredicateSource> sources_;
    // BOOL views already own validity/coverage. Avoid a second field-input
    // state machine and its duplicate masks for predicate results.
    std::vector<std::optional<ReadOnlyVectorView>> results_;
    std::vector<bool> loading_;
    scalar::EvalCtx* active_ = nullptr;
};
}  // namespace milvus::exec::predicate
