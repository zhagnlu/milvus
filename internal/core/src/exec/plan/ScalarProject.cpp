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

#include "exec/plan/ScalarProject.h"

#include <unordered_set>

namespace milvus::exec::plan {

ScalarProject::ScalarProject(PreparedExecution execution,
                             std::vector<std::string> names)
    : execution_(std::move(execution)) {
    const auto& roots = execution_.plan()->expressions().roots();
    AssertInfo(names.size() == roots.size() && !names.empty(),
               "projection names and expressions must have equal nonzero size");
    std::unordered_set<std::string> unique;
    std::vector<DataType> types;
    for (size_t i = 0; i < roots.size(); ++i) {
        AssertInfo(!names[i].empty() && unique.insert(names[i]).second,
                   "projection output names must be nonempty and unique");
        types.push_back(roots[i]->type());
    }
    output_type_ =
        std::make_shared<RowType>(std::move(names), std::move(types));
}

RowVectorPtr
ScalarProject::Eval(const ReadBatchBinding& binding,
                    SelectionView rows,
                    const RowVector* upstream,
                    std::function<void()> checkpoint) const {
    AssertInfo(binding.size() == rows.row_count(),
               "projection selection and binding sizes differ");
    TargetBitmap valid(binding.size(), true);
    TargetBitmap active(binding.size(), false),
        null_rows(binding.size(), false);
    if (upstream) {
        AssertInfo(upstream->size() == binding.size(),
                   "projection upstream size differs from binding");
        rows.ForEachSelected([&](size_t row) {
            const bool present = upstream->IsValid(row);
            valid.set(row, present);
            active.set(row, present);
            null_rows.set(row, !present);
        });
    }
    const auto eval_rows = upstream ? SelectionView::FromBitmap(active) : rows;
    auto batch = execution_.MakeExecutionBatch(
        binding,
        upstream,
        upstream ? std::optional<SelectionView>(eval_rows) : std::nullopt);
    auto values = batch->Eval(eval_rows, checkpoint);
    const auto nulls = SelectionView::FromBitmap(null_rows);
    const bool has_null_rows = null_rows.count() != 0;
    std::vector<VectorPtr> children;
    for (auto& value : values) {
        if (has_null_rows)
            value = value.Merge(
                ReadOnlyVectorView::Nulls(value.logical_type(), nulls), nulls);
        children.push_back(value.AsVector());
    }
    auto result = std::make_shared<RowVector>(
        output_type_, binding.size(), std::move(children), std::move(valid));
    if (!rows.IsAll()) {
        std::vector<vector_size_t> indices;
        rows.ForEachSelected([&](size_t row) { indices.push_back(row); });
        result = SelectRows(*result,
                            std::make_shared<RowMapping>(std::move(indices)));
    }
    if (checkpoint)
        checkpoint();
    return result;
}

}  // namespace milvus::exec::plan
