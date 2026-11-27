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

#include "exec/operator/ScalarProjectNode.h"
#include "exec/plan/ExprCompiler.h"
#include "exec/plan/ExecutionPreparer.h"
#include "exec/plan/ScalarProject.h"
#include "exec/access/ReaderInput.h"
#include "exec/access/SegmentReaderInput.h"
#include "exec/expression/Utils.h"
#include "segcore/SegmentInterface.h"

namespace milvus::exec {

struct ScalarProjectState {
    exec::plan::ScalarProject project;
    RowTypePtr input_type;
};

PhyScalarProjectNode::~PhyScalarProjectNode() = default;

PhyScalarProjectNode::PhyScalarProjectNode(
    int32_t operator_id,
    DriverContext* ctx,
    const std::shared_ptr<const milvus::plan::ScalarProjectNode>& node)
    : Operator(
          ctx, node->output_type(), operator_id, node->id(), "ScalarProject") {
    exec::plan::CompileOptions options;
    options.upstream_schema = node->sources()[0]->output_type();
    options.upstream_columns = node->upstream_columns();
    options.upstream_results = node->upstream_results();
    auto compiled =
        exec::plan::CompileExpressions(node->expressions(), options);
    AssertInfo(compiled.supported(), "unsupported internal scalar projection");
    // Data-batch mode must have explicit upstream bindings. In particular,
    // a missing column must not silently read a same-ID field from storage.
    for (const auto& input : compiled.plan->inputs().slots())
        AssertInfo(
            input.upstream_column.has_value(),
            "scalar project operator requires an upstream column binding");
    std::vector<std::string> names;
    for (size_t i = 0; i < node->output_type()->column_count(); ++i)
        names.push_back(node->output_type()->column_name(i));
    state_ = std::make_unique<ScalarProjectState>(ScalarProjectState{
        exec::plan::ScalarProject(exec::plan::Prepare(compiled.plan),
                                  std::move(names)),
        options.upstream_schema});
}

void
PhyScalarProjectNode::AddInput(RowVectorPtr& input) {
    AssertInfo(NeedInput() && input, "scalar project cannot accept input");
    const auto& actual = input->row_type();
    const auto& expected = state_->input_type;
    AssertInfo(actual && actual->column_count() == expected->column_count(),
               "scalar project requires a typed data batch");
    for (size_t i = 0; i < expected->column_count(); ++i)
        AssertInfo(actual->column_name(i) == expected->column_name(i) &&
                       actual->column_type(i) == expected->column_type(i),
                   "scalar project input schema changed");
    input_ = std::move(input);
}

RowVectorPtr
PhyScalarProjectNode::GetOutput() {
    if (!input_)
        return nullptr;
    auto* context = operator_context_->get_exec_context()->get_query_context();
    auto result =
        state_->project.Eval(ReadBatchBinding::Range(0, input_->size()),
                             SelectionView::All(input_->size()),
                             input_.get(),
                             [context] { checkCancellation(context); });
    input_.reset();
    return result->size() ? result : nullptr;
}

struct ScalarProjectScanState {
    exec::plan::ScalarProject project;
    std::shared_ptr<segcore::reader::ReadContext> context;
    segcore::OffsetBatchIterator next;
    size_t batch_size;
};

PhyScalarProjectScanNode::~PhyScalarProjectScanNode() = default;

PhyScalarProjectScanNode::PhyScalarProjectScanNode(
    int32_t operator_id,
    DriverContext* ctx,
    const std::shared_ptr<const milvus::plan::ScalarProjectScanNode>& node)
    : Operator(ctx,
               node->output_type(),
               operator_id,
               node->id(),
               "ScalarProjectScan") {
    auto* query = operator_context_->get_exec_context()->get_query_context();
    AssertInfo(query->get_segment() && !query->bitset_is_element_level() &&
                   query->get_struct_name().empty(),
               "scalar scan requires a row-level segment query");
    const auto batch_size = query->query_config()->get_expr_batch_size();
    segcore::CheckOffsetBatchSize(batch_size);
    auto context = std::make_shared<segcore::reader::ReadContext>();
    if (auto* op = query->get_op_context())
        context->op_context = std::shared_ptr<OpContext>(op, [](OpContext*) {});
    context->checkpoint = [query] { checkCancellation(query); };
    auto compiled = exec::plan::CompileExpressions(node->expressions());
    AssertInfo(compiled.supported(),
               "unsupported internal scalar scan projection");
    // An empty physical segment has no loaded column/PK resources. Keep plan
    // validation above, but do not require readers for a domain with no rows.
    if (query->get_active_count() == 0) {
        context->Check();
        finished_ = true;
        return;
    }
    auto bind = access::MakeSegmentReaderBinding(*query->get_segment(),
                                                 query->get_read_snapshot(),
                                                 query->get_active_count(),
                                                 context->op_context.get());
    auto prepared =
        exec::plan::Prepare(compiled.plan, [&](const auto& dependency) {
            AssertInfo(bind, "scalar scan has no column reader binding");
            auto reader = bind(dependency);
            AssertInfo(reader &&
                           reader->row_count() == query->get_active_count() &&
                           reader->type() == dependency.type,
                       "scalar scan reader is unavailable or has the wrong "
                       "domain/type");
            return access::PrepareReaderInput(
                std::move(reader), context, dependency.needs_values);
        });
    std::vector<std::string> names;
    for (size_t i = 0; i < node->output_type()->column_count(); ++i)
        names.push_back(node->output_type()->column_name(i));
    state_ = std::make_unique<ScalarProjectScanState>(ScalarProjectScanState{
        exec::plan::ScalarProject(std::move(prepared), std::move(names)),
        std::move(context),
        {},
        static_cast<size_t>(batch_size)});
}

void
PhyScalarProjectScanNode::AddInput(RowVectorPtr& input) {
    AssertInfo(NeedInput() && input && input->childrens().size() == 1,
               "scalar scan requires one exclusion bitmap");
    auto* query = operator_context_->get_exec_context()->get_query_context();
    auto column = GetColumnVector(input);
    AssertInfo(
        column->IsBitmap() && column->size() == query->get_active_count(),
        "scalar scan bitmap has the wrong encoding or query domain");
    auto next = access::MakeSegmentOffsetIterator(
        *query->get_segment(),
        query->get_read_snapshot(),
        BitsetTypeView(column->GetRawData(), column->size()),
        *state_->context);
    AssertInfo(next, "scalar scan offset traversal is unavailable");
    state_->next = std::move(next);
    input_ = std::move(input);
}

RowVectorPtr
PhyScalarProjectScanNode::GetOutput() {
    if (finished_ || !input_)
        return nullptr;
    state_->context->Check();
    auto offsets = state_->next(state_->batch_size);
    if (offsets.empty()) {
        state_->next = {};
        input_.reset();
        finished_ = true;
        return nullptr;
    }
    // Preserve PK traversal order. Only an already consecutive batch can use
    // Scan's pinned views instead of materializing each value through Take.
    bool contiguous = true;
    for (size_t i = 1; i < offsets.size(); ++i)
        if (offsets[i] != offsets[i - 1] + 1) {
            contiguous = false;
            break;
        }
    auto binding =
        contiguous
            ? ReadBatchBinding::Range(offsets.front(), offsets.size())
            : ReadBatchBinding::Offsets(std::span<const int64_t>(offsets));
    return state_->project.Eval(binding,
                                SelectionView::All(binding.size()),
                                nullptr,
                                state_->context->checkpoint);
}

}  // namespace milvus::exec
