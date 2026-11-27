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

#include "exec/access/ScanNode.h"

namespace milvus::exec::access {
using namespace segcore::reader;

ScanNode::ScanNode(std::shared_ptr<const ColumnReader> reader,
                   ReadBatchBinding binding,
                   std::shared_ptr<const ReadContext> context)
    : reader_(std::move(reader)),
      binding_(std::move(binding)),
      context_(std::move(context)) {
    AssertInfo(reader_ && context_ && binding_.is_range(),
               "invalid ScanNode binding");
}

ReadResult
ScanNode::Read(SelectionView rows, ReadProjection projection) const {
    RangeRequest request(
        binding_.range_begin(), binding_.size(), projection, rows);
    if (rows.IsNone()) {
        ReadResult empty{std::nullopt,
                         OwnedSelection::None(rows.row_count()),
                         OwnedSelection::None(rows.row_count())};
        if (projection == ReadProjection::ValuesAndValidity)
            empty.values = ReadOnlyVectorView::Nulls(reader_->type(), rows);
        return empty;
    }
    auto result = reader_->ReadRange(request, *context_);
    result.Validate(reader_->type(), rows, projection);
    return result;
}
}  // namespace milvus::exec::access
