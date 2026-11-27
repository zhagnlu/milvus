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

#include "exec/operator/CandidateFilter.h"
#include "exec/plan/ExprCompiler.h"
#include "exec/QueryContext.h"
#include "common/Common.h"
#include "exec/operator/FilterBitsExecution.h"
#include "segcore/SegmentInterface.h"

namespace milvus::exec {
CandidateFilter::CandidateFilter(std::unique_ptr<FilterBitsExecution> execution,
                                 int64_t count,
                                 QueryContext* query)
    : execution_(std::move(execution)), count_(count), query_(query) {
}
CandidateFilter::~CandidateFilter() = default;

std::unique_ptr<CandidateFilter>
CandidateFilter::TryCreate(const expr::TypedExprPtr& predicate,
                           ExecContext* context,
                           const std::string& element_structure) {
    if (!predicate || !context->get_query_config()->enable_expr_execution_v2())
        return nullptr;
    auto* query = context->get_query_context();
    auto* segment = query->get_segment();
    auto count = query->get_active_count();
    const auto documents = count;
    auto snapshot = query->get_read_snapshot();
    // Direct scoring callers may not have installed the request snapshot yet.
    if (!snapshot) {
        snapshot = segment->CaptureReadSnapshot();
        query->set_read_snapshot(snapshot);
    }
    auto read = std::make_shared<segcore::reader::ReadContext>();
    if (auto* op = query->get_op_context())
        read->op_context = std::shared_ptr<OpContext>(op, [](OpContext*) {});
    read->checkpoint = [query] { checkCancellation(query); };
    plan::PhysicalPrepareOptions physical;
    physical.reorder = OPTIMIZE_EXPR_ENABLED.load();
    if (auto ns = segment->get_schema_snapshot()->get_namespace_field_id())
        physical.namespace_field = ns->get();
    physical.bind_index = access::MakeSegmentIndexBinding(
        *segment,
        snapshot,
        count,
        read,
        query->get_plan_options().expr_use_json_stats);
    auto root = predicate;
    auto schema = segment->get_schema_snapshot();
    auto bind = access::MakeSegmentReaderBinding(
        *segment, snapshot, documents, read->op_context.get());
    if (!element_structure.empty()) {
        root =
            plan::LowerElementPredicate(predicate, *schema, element_structure);
        if (!root)
            return nullptr;
        auto anchor =
            schema->GetFirstArrayFieldInStruct(element_structure).get_id();
        auto offsets = snapshot ? snapshot->GetStructElementOffsets(anchor)
                                : segment->GetStructElementOffsets(anchor);
        if (!offsets)
            return nullptr;
        AssertInfo(documents <= offsets->GetRowCount(),
                   "element mapping lacks query prefix");
        count = documents == 0
                    ? 0
                    : offsets->ElementIDRangeOfRow(documents - 1).second;
        physical.bind_index =
            access::MakeElementIndexBinding(snapshot, schema, offsets, read);
        bind = access::MakeElementReaderBinding(
            std::move(bind), schema, std::move(offsets), documents, count);
    }
    auto execution = FilterBitsExecution::TryCreateWithBinding(
        root,
        nullptr,
        bind,
        count,
        context->get_query_config()->get_expr_batch_size(),
        read,
        physical,
        schema,
        !element_structure.empty());
    if (!execution)
        return nullptr;
    return std::unique_ptr<CandidateFilter>(
        new CandidateFilter(std::move(execution), count, query));
}

TargetBitmap
CandidateFilter::Eval(const FixedVector<int32_t>& offsets) {
    checkCancellation(query_);
    for (auto offset : offsets) {
        AssertInfo(offset >= 0 && offset < count_,
                   "candidate offset outside prepared query domain");
    }
    TargetBitmap matches(offsets.size(), false);
    if (offsets.empty())
        return matches;
    auto batch =
        execution_->prepared().MakeExecutionBatch(ReadBatchBinding::Offsets(
            std::span<const int32_t>(offsets.data(), offsets.size())));
    auto results = batch->Eval(SelectionView::All(offsets.size()),
                               [this] { checkCancellation(query_); });
    AssertInfo(results.size() == 1 && results[0].size() == offsets.size() &&
                   results[0].logical_type() == DataType::BOOL &&
                   results[0].defined_rows().IsAll(),
               "candidate filter returned incomplete BOOL result");
    auto masks = results[0].CopyBoolMasks(SelectionView::All(offsets.size()));
    matches = std::move(masks.truth).TakeBitmap();
    checkCancellation(query_);
    return matches;
}
}  // namespace milvus::exec
