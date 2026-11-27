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

#include "exec/predicate/RefineNode.h"

namespace milvus::exec::predicate {
RefineNode::RefineNode(std::shared_ptr<IndexNode> coarse, scalar::ExprPtr exact)
    : coarse_(std::move(coarse)), exact_(std::move(exact)) {
    AssertInfo(coarse_ && exact_ && exact_->type() == DataType::BOOL,
               "invalid refine program");
}
ReadOnlyVectorView
RefineNode::Eval(const ReadBatchBinding& binding,
                 SelectionView rows,
                 scalar::EvalCtx& ctx) {
    AssertInfo(binding.size() == ctx.input().size() &&
                   rows.row_count() == binding.size(),
               "refine batch domain mismatch");
    if (rows.IsNone())
        return ReadOnlyVectorView::Nulls(DataType::BOOL, rows);
    AssertInfo(!evaluating_, "recursive refine dependency");
    struct Guard {
        bool& flag;
        ~Guard() {
            flag = false;
        }
    } guard{evaluating_};
    evaluating_ = true;
    ctx.Checkpoint();
    auto coarse = coarse_->Read(binding, rows);
    if (coarse->kind() != segcore::reader::IndexResultKind::Candidates)
        ThrowInfo(Unsupported,
                  "refine requires an explicit candidate/proof result");
    auto proven_false = OwnedSelection::None(binding.size());
    auto proven_null = OwnedSelection::None(binding.size());
    auto candidates = OwnedSelection::None(binding.size());
    if (binding.is_range()) {
        const auto begin = static_cast<size_t>(binding.range_begin());
        proven_false = OwnedSelection::Intersect(
            rows, coarse->false_rows().Slice(begin, binding.size()));
        proven_null = OwnedSelection::Intersect(
            rows, coarse->null_rows().Slice(begin, binding.size()));
        candidates = OwnedSelection::Intersect(
            rows, coarse->selected().Slice(begin, binding.size()));
    } else {
        rows.ForEachSelected([&](size_t row) {
            const auto offset = binding.offset(row);
            if (coarse->false_rows().IsSelected(offset))
                proven_false.SetSelected(row);
            if (coarse->null_rows().IsSelected(offset))
                proven_null.SetSelected(row);
            if (coarse->selected().IsSelected(offset))
                candidates.SetSelected(row);
        });
    }
    auto missing = OwnedSelection::Subtract(rows, proven_false.view());
    missing.Subtract(proven_null.view());
    auto truth = OwnedSelection::None(binding.size());
    auto valid = std::move(proven_false);
    if (!missing.view().IsNone()) {
        auto exact =
            exact_->Eval(missing.view(), ctx).CopyBoolMasks(missing.view());
        const auto omitted =
            OwnedSelection::Subtract(exact.truth.view(), candidates.view());
        AssertInfo(omitted.view().IsNone(), "coarse index omitted a true row");
        truth = std::move(exact.truth);
        valid.Union(exact.valid.view());
    }
    ctx.Checkpoint();
    return ReadOnlyVectorView::FromBooleanMasks(
        std::move(truth), std::move(valid), rows);
}
}  // namespace milvus::exec::predicate
