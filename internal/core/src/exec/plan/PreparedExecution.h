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

#include <functional>
#include "exec/input/BatchInput.h"
#include "exec/access/ReadBatchBinding.h"
#include "exec/plan/CompiledExprPlan.h"
#include "exec/predicate/PredicateBatch.h"

namespace milvus::exec::plan {

using ::milvus::exec::InputLoaders;

// Metadata-only preparation result. owner protects resources required for
// future reads. Factory and loaders must use the supplied immutable binding,
// never borrow an advancing batch cursor. Returned values own their payloads.
struct PreparedInput {
    DataType type = DataType::NONE;
    bool supports_values = false;
    // True promises a distinct validity-only loader. Value capability alone
    // also satisfies validity requests via BatchInput's cached value fallback.
    bool supports_validity = false;
    std::shared_ptr<const void> owner;
    std::function<InputLoaders(const ReadBatchBinding&)> make_loaders;
    std::function<BatchInput::GeometryLookup(const ReadBatchBinding&)>
        make_geometry_lookup;
    // Optional explicit-position access avoids reconstructing source closures
    // for each batch. Generic factories remain supported.
    std::function<InputValueResult(const ReadBatchBinding&, SelectionView)>
        read_values;
    std::function<ValidityResult(const ReadBatchBinding&, SelectionView)>
        read_validity;
};

enum class PredicateRoute { Raw, ExactIndex, Refine };
// Backend identity and result completeness are independent. A stats source
// may provide an exact predicate or only candidates requiring raw refinement.
enum class PredicateBackend { Raw, ScalarIndex, PkIndex, TextIndex, JsonStats };
enum class ExecutionGranularity { Batch, WholeSegment };

struct PreparedPredicates {
    bool supported = true;
    bool force_batch = false;
    std::vector<predicate::PredicateSource> sources;
    std::vector<PredicateRoute> routes;
    std::vector<InputUse> raw_inputs;
    std::vector<PredicateBackend> backends;
    std::shared_ptr<const scalar::ExprSet> expressions;
};

// Owns the batch and its predicate provider in destruction-safe order. A batch
// may outlive PreparedExecution; resources are retained by its bound sources.
class ExecutionBatch {
 public:
    ExecutionBatch(CompiledExprPlanPtr plan,
                   std::unique_ptr<BatchInput> input,
                   ReadBatchBinding binding,
                   std::vector<predicate::PredicateSource> sources,
                   std::shared_ptr<const scalar::ExprSet> expressions = {});
    std::vector<ReadOnlyVectorView>
    Eval(SelectionView rows, std::function<void()> checkpoint = {});
    std::vector<ReadOnlyVectorView>
    Eval(const std::vector<SelectionView>& rows,
         std::function<void()> checkpoint = {});
    // Available on workspaces created by MakeReusableBatch. Synchronous only;
    // old returned views remain valid. Failure poisons this workspace.
    void
    Reset(const ReadBatchBinding& binding);

    BatchInput&
    input() {
        return *input_;
    }

 private:
    friend class PreparedExecution;
    std::optional<std::vector<std::function<void(const ReadBatchBinding&)>>>
        resets_;
    bool evaluating_ = false;
    bool ready_ = true;
    CompiledExprPlanPtr plan_;
    std::unique_ptr<BatchInput> input_;
    std::shared_ptr<predicate::PredicateBatch> predicates_;
    std::shared_ptr<const scalar::ExprSet> expressions_;
};

class PreparedExecution {
 public:
    PreparedExecution(
        CompiledExprPlanPtr plan,
        std::vector<std::optional<PreparedInput>> inputs,
        std::optional<PreparedPredicates> predicates = std::nullopt);

    // Stored factories are deferred until a nonempty missing request. Optional
    // upstream_rows explicitly bounds a partial upstream snapshot; it does NOT
    // restrict other inputs or act as an implicit calculation selection.
    std::unique_ptr<BatchInput>
    MakeBatch(const ReadBatchBinding& binding,
              const RowVector* upstream = nullptr,
              std::optional<SelectionView> upstream_rows = std::nullopt) const;

    std::unique_ptr<ExecutionBatch>
    MakeExecutionBatch(
        const ReadBatchBinding& binding,
        const RowVector* upstream = nullptr,
        std::optional<SelectionView> upstream_rows = std::nullopt) const;

    // No upstream inputs: used by scan/candidate operators that consume one
    // batch before advancing. Independent MakeExecutionBatch remains available.
    std::unique_ptr<ExecutionBatch>
    MakeReusableBatch(const ReadBatchBinding& binding) const;

    const std::vector<PredicateRoute>&
    predicate_routes() const {
        return predicates_.routes;
    }

    const std::vector<PredicateBackend>&
    predicate_backends() const {
        return predicates_.backends;
    }
    ExecutionGranularity
    granularity() const {
        return !predicates_.force_batch && plan_->whole_segment_safe() &&
                       predicates_.raw_inputs.empty()
                   ? ExecutionGranularity::WholeSegment
                   : ExecutionGranularity::Batch;
    }

    const CompiledExprPlanPtr&
    plan() const {
        return plan_;
    }

 private:
    std::unique_ptr<BatchInput>
    MakeRawBatch(const ReadBatchBinding&,
                 const RowVector*,
                 std::optional<SelectionView>,
                 std::vector<std::function<void(const ReadBatchBinding&)>>*
                     resets = nullptr) const;
    CompiledExprPlanPtr plan_;
    std::vector<std::optional<PreparedInput>> inputs_;
    std::vector<std::optional<InputDependency>> requirements_;
    PreparedPredicates predicates_;
};

// Shared by preparation and constructor validation; stable slot numbers are
// retained even when an unselected branch's raw slot is left unbound.
std::vector<std::optional<InputDependency>>
RequiredInputs(const CompiledExprPlan&, const std::vector<InputUse>&);

}  // namespace milvus::exec::plan
