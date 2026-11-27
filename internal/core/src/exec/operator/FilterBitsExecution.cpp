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
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "exec/operator/FilterBitsExecution.h"

#include <algorithm>
#include "exec/access/ReaderInput.h"
#include "exec/plan/ExprCompiler.h"
#include "exec/plan/ExecutionPreparer.h"

namespace milvus::exec {
struct FilterBitsExecution::State {
    plan::PreparedExecution prepared;
    std::shared_ptr<const segcore::reader::ReadContext> context;
    int64_t active_count;
    int64_t batch_size;
    bool emitted = false;
};

FilterBitsExecution::FilterBitsExecution(std::unique_ptr<State> state)
    : state_(std::move(state)) {
}
FilterBitsExecution::~FilterBitsExecution() = default;

std::unique_ptr<FilterBitsExecution>
FilterBitsExecution::TryCreate(
    const expr::TypedExprPtr& predicate,
    const expr::TypedExprPtr& ttl_predicate,
    std::shared_ptr<const segcore::SegmentReadSnapshot> snapshot,
    int64_t active_count,
    int64_t batch_size,
    std::shared_ptr<const segcore::reader::ReadContext> context,
    std::shared_ptr<const Schema> schema) {
    plan::PhysicalPrepareOptions physical;
    physical.bind_index =
        access::MakeSealedIndexBinding(snapshot, active_count, context);
    auto* op_context = context->op_context.get();
    return TryCreateWithBinding(
        predicate,
        ttl_predicate,
        access::MakeSealedReaderBinding(
            std::move(snapshot), active_count, op_context),
        active_count,
        batch_size,
        std::move(context),
        physical,
        std::move(schema));
}

std::unique_ptr<FilterBitsExecution>
FilterBitsExecution::TryCreateWithBinding(
    const expr::TypedExprPtr& predicate,
    const expr::TypedExprPtr& ttl_predicate,
    const access::BindColumnReader& bind,
    int64_t active_count,
    int64_t batch_size,
    std::shared_ptr<const segcore::reader::ReadContext> context,
    const plan::PhysicalPrepareOptions& physical,
    std::shared_ptr<const Schema> schema,
    bool array_element_arithmetic) {
    AssertInfo(context && active_count >= 0 && batch_size > 0,
               "invalid FilterBits execution binding");
    segcore::reader::CheckRequestSize(static_cast<size_t>(active_count));
    segcore::reader::CheckRequestSize(static_cast<size_t>(batch_size));
    plan::CompileOptions options;
    options.plan_indexes = true;
    options.schema = std::move(schema);
    options.array_element_arithmetic = array_element_arithmetic;
    auto compiled = plan::CompileFilter(predicate, ttl_predicate, options);
    if (!compiled.supported())
        return nullptr;
    AssertInfo(predicate->type() == DataType::BOOL &&
                   (!ttl_predicate || ttl_predicate->type() == DataType::BOOL),
               "FilterBits requires BOOL predicates");

    auto selected = plan::AdjustExecution(*compiled.plan, physical);
    if (!selected.supported)
        return nullptr;
    auto requirements =
        plan::RequiredInputs(*compiled.plan, selected.raw_inputs);
    std::vector<std::optional<plan::PreparedInput>> inputs(requirements.size());
    for (size_t slot = 0; slot < requirements.size(); ++slot) {
        if (!requirements[slot])
            continue;
        const auto& dependency = *requirements[slot];
        if (!bind)
            return nullptr;
        auto reader = bind(dependency);
        if (!reader)
            return nullptr;
        AssertInfo(reader->row_count() == active_count &&
                       reader->type() == dependency.type,
                   "FilterBits reader has wrong type or query domain");
        inputs[slot] = access::PrepareReaderInput(
            std::move(reader), context, dependency.needs_values);
    }
    plan::PreparedExecution prepared(
        compiled.plan, std::move(inputs), std::move(selected));
    if (prepared.granularity() == plan::ExecutionGranularity::WholeSegment)
        batch_size = std::max<int64_t>(active_count, 1);
    return std::unique_ptr<FilterBitsExecution>(new FilterBitsExecution(
        std::make_unique<State>(State{std::move(prepared),
                                      std::move(context),
                                      active_count,
                                      batch_size})));
}

const plan::PreparedExecution&
FilterBitsExecution::prepared() const {
    return state_->prepared;
}

bool
FilterBitsExecution::IsFinished() const {
    return state_->emitted || state_->active_count == 0;
}

RowVectorPtr
FilterBitsExecution::GetOutput() {
    if (IsFinished())
        return nullptr;
    state_->context->Check();
    TargetBitmap excluded(state_->active_count, true);
    std::unique_ptr<plan::ExecutionBatch> batch;
    for (int64_t begin = 0; begin < state_->active_count;) {
        state_->context->Check();
        const auto count =
            std::min(state_->batch_size, state_->active_count - begin);
        const auto binding = ReadBatchBinding::Range(begin, count);
        if (batch)
            batch->Reset(binding);
        else
            batch = state_->prepared.MakeReusableBatch(binding);
        auto results =
            batch->Eval(SelectionView::All(count), state_->context->checkpoint);
        AssertInfo(results.size() == 1 &&
                       results[0].logical_type() == DataType::BOOL &&
                       results[0].size() == count &&
                       results[0].defined_rows().IsAll(),
                   "FilterBits received incomplete/non-BOOL output");
        results[0].CopyTrueTo(SelectionView::All(count), excluded, begin);
        excluded.view(begin, count).flip();
        begin += count;
    }
    auto output = std::make_shared<RowVector>(
        std::vector<VectorPtr>{std::make_shared<ColumnVector>(
            std::move(excluded), TargetBitmap(state_->active_count, true))});
    state_->context->Check();
    state_->emitted = true;
    return output;
}
}  // namespace milvus::exec
