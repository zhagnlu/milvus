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

#include "exec/predicate/PredicateResult.h"

namespace milvus::exec::predicate {
void
ValidateBinding(const ReadBatchBinding& binding,
                SelectionView rows,
                size_t source_rows) {
    AssertInfo(binding.size() == rows.row_count(),
               "predicate batch selection mismatch");
    if (binding.is_range()) {
        const auto begin = static_cast<uint64_t>(binding.range_begin());
        AssertInfo(
            begin <= source_rows && binding.size() <= source_rows - begin,
            "predicate range outside source snapshot");
    } else {
        for (size_t i = 0; i < binding.size(); ++i)
            AssertInfo(static_cast<uint64_t>(binding.offset(i)) < source_rows,
                       "predicate offset outside source snapshot");
    }
}
ReadOnlyVectorView
MapPredicate(const segcore::reader::IndexResult& result,
             const ReadBatchBinding& binding,
             SelectionView rows) {
    ValidateBinding(binding, rows, result.size());
    if (result.kind() != segcore::reader::IndexResultKind::FullThreeValued)
        ThrowInfo(Unsupported,
                  "incomplete index truth cannot become a BOOL projection");
    if (binding.is_range()) {
        const auto begin = static_cast<size_t>(binding.range_begin());
        auto truth = OwnedSelection::Intersect(
            rows, result.selected().Slice(begin, binding.size()));
        auto valid = OwnedSelection::Intersect(
            rows, result.valid().Slice(begin, binding.size()));
        return ReadOnlyVectorView::FromBooleanMasks(
            std::move(truth), std::move(valid), rows);
    }
    BitmapResultBuilder output(binding.size());
    rows.ForEachSelected([&](size_t row) {
        const auto offset = binding.offset(row);
        if (result.valid().IsSelected(offset))
            output.Set<bool>(row, result.selected().IsSelected(offset));
    });
    return std::move(output).Finish(rows);
}
ReadOnlyVectorView
MapPredicate(std::shared_ptr<const segcore::reader::IndexResult> result,
             const ReadBatchBinding& binding,
             SelectionView rows) {
    AssertInfo(result, "missing index result");
    if (!binding.is_range())
        return MapPredicate(*result, binding, rows);
    ValidateBinding(binding, rows, result->size());
    if (result->kind() != segcore::reader::IndexResultKind::FullThreeValued)
        ThrowInfo(Unsupported,
                  "incomplete index truth cannot become a BOOL projection");
    const auto begin = static_cast<size_t>(binding.range_begin());
    const auto truth = result->selected().Slice(begin, binding.size());
    const auto valid = result->valid().Slice(begin, binding.size());
    return ReadOnlyVectorView::BorrowBooleanMasks(
               truth, valid, std::move(result))
        .Restrict(rows);
}
OwnedSelection
MapFilterTrue(const segcore::reader::IndexResult& result,
              const ReadBatchBinding& binding,
              SelectionView rows) {
    ValidateBinding(binding, rows, result.size());
    if (result.kind() == segcore::reader::IndexResultKind::Candidates)
        ThrowInfo(Unsupported, "candidate rows are not exact filter matches");
    if (binding.is_range())
        return OwnedSelection::Intersect(
            rows,
            result.selected().Slice(static_cast<size_t>(binding.range_begin()),
                                    binding.size()));
    auto selected = OwnedSelection::None(binding.size());
    rows.ForEachSelected([&](size_t row) {
        if (result.selected().IsSelected(binding.offset(row)))
            selected.SetSelected(row);
    });
    return selected;
}
}  // namespace milvus::exec::predicate
