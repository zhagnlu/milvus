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

#include "exec/predicate/PredicateBatch.h"

namespace milvus::exec::predicate {
PredicateSource
BindIndex(std::shared_ptr<IndexNode> node) {
    AssertInfo(node, "null predicate index node");
    return [node = std::move(node)](const ReadBatchBinding& binding,
                                    SelectionView rows,
                                    scalar::EvalCtx&) {
        return node->Eval(binding, rows);
    };
}
PredicateSource
BindRefine(std::shared_ptr<RefineNode> node) {
    AssertInfo(node, "null predicate refine node");
    return [node = std::move(node)](const ReadBatchBinding& binding,
                                    SelectionView rows,
                                    scalar::EvalCtx& ctx) {
        return node->Eval(binding, rows, ctx);
    };
}
PredicateBatch::PredicateBatch(BatchInput& raw,
                               ReadBatchBinding binding,
                               std::vector<PredicateSource> sources)
    : raw_(raw), binding_(std::move(binding)), sources_(std::move(sources)) {
    AssertInfo(raw_.size() == binding_.size(),
               "predicate/raw batch domain mismatch");
    AssertInfo(sources_.size() <= std::numeric_limits<column_index_t>::max(),
               "too many predicate inputs");
    for (const auto& source : sources_)
        AssertInfo(source, "missing predicate source");
    results_.resize(sources_.size());
    loading_.resize(sources_.size(), false);
}
void
PredicateBatch::Reset(ReadBatchBinding binding) {
    AssertInfo(!active_ && raw_.size() == binding.size(),
               "predicate reset during evaluation or with a mismatched domain");
    for (auto& result : results_) result.reset();
    binding_ = std::move(binding);
}

ReadOnlyVectorView
PredicateBatch::Get(column_index_t slot,
                    SelectionView rows,
                    scalar::EvalCtx& ctx) {
    AssertInfo(&ctx.input() == &raw_,
               "predicate provider used with another raw batch");
    AssertInfo(!active_ || active_ == &ctx,
               "predicate context changed during evaluation");
    struct Guard {
        scalar::EvalCtx*& active;
        scalar::EvalCtx* previous;
        ~Guard() {
            active = previous;
        }
    } guard{active_, active_};
    active_ = &ctx;
    AssertInfo(slot < results_.size() && rows.row_count() == binding_.size(),
               "predicate selection/slot mismatch");
    auto& cached = results_[slot];
    if (rows.IsNone())
        return ReadOnlyVectorView::Nulls(DataType::BOOL, rows);
    if (cached && rows.IsSubsetOf(cached->defined_rows()))
        return *cached;
    AssertInfo(!loading_[slot], "recursive loading of the same predicate slot");
    struct LoadingGuard {
        std::vector<bool>& flags;
        size_t slot;
        ~LoadingGuard() {
            flags[slot] = false;
        }
    } loading_guard{loading_, slot};
    loading_[slot] = true;
    auto missing = cached
                       ? OwnedSelection::Subtract(rows, cached->defined_rows())
                       : OwnedSelection::None(0);
    const auto requested = cached ? missing.view() : rows;
    auto loaded = sources_[slot](binding_, requested, ctx);
    AssertInfo(loaded.logical_type() == DataType::BOOL &&
                   loaded.size() == binding_.size() &&
                   requested.IsSubsetOf(loaded.defined_rows()),
               "predicate returned wrong type/domain/coverage");
    if (cached) {
        auto overlap = OwnedSelection::Intersect(cached->defined_rows(),
                                                 loaded.defined_rows());
        if (!overlap.view().IsNone()) {
            const auto old_valid = cached->CopyValidity();
            const auto new_valid = loaded.CopyValidity();
            AssertInfo(
                old_valid.view().EqualsOn(new_valid.view(), overlap.view()),
                "predicate changed validity within an immutable batch");
        }
        auto added = OwnedSelection::Subtract(loaded.defined_rows(),
                                              cached->defined_rows());
        loaded = cached->Merge(loaded, added.view());
    }
    // Failed source/validation/merge never replaces the published view.
    cached = std::move(loaded);
    return *cached;
}
}  // namespace milvus::exec::predicate
