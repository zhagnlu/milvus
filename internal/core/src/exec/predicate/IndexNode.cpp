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

#include "exec/predicate/IndexNode.h"

namespace milvus::exec::predicate {
IndexNode::IndexNode(
    std::shared_ptr<const segcore::reader::IndexReader> reader,
    std::shared_ptr<const segcore::reader::ReadContext> context,
    bool use_index)
    : reader_(std::move(reader)), context_(std::move(context)) {
    if (!use_index)
        ThrowInfo(Unsupported, "index source disabled by execution policy");
    AssertInfo(reader_ && context_ && reader_->row_count() >= 0,
               "invalid index node source");
}
std::shared_ptr<const segcore::reader::IndexResult>
IndexNode::Read(const ReadBatchBinding& binding, SelectionView rows) {
    ValidateBinding(binding, rows, reader_->row_count());
    if (rows.IsNone())
        return nullptr;
    context_->Check();
    if (cached_)
        return cached_;
    AssertInfo(!loading_, "recursive index query");
    struct Guard {
        bool& flag;
        ~Guard() {
            flag = false;
        }
    } guard{loading_};
    loading_ = true;
    auto candidate = std::make_shared<const segcore::reader::IndexResult>(
        reader_->Query(*context_));
    AssertInfo(candidate->size() == static_cast<size_t>(reader_->row_count()),
               "index query returned wrong row domain");
    context_->Check();
    cached_ = std::move(candidate);
    return cached_;
}
ReadOnlyVectorView
IndexNode::Eval(const ReadBatchBinding& binding, SelectionView rows) {
    auto result = Read(binding, rows);
    return result ? MapPredicate(std::move(result), binding, rows)
                  : ReadOnlyVectorView::Nulls(DataType::BOOL, rows);
}
OwnedSelection
IndexNode::FilterTrue(const ReadBatchBinding& binding, SelectionView rows) {
    auto result = Read(binding, rows);
    return result ? MapFilterTrue(*result, binding, rows)
                  : OwnedSelection::None(binding.size());
}
}  // namespace milvus::exec::predicate
