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
#include "exec/access/ReadBatchBinding.h"
#include "segcore/reader/ColumnReader.h"

namespace milvus::exec::access {

// Stateless with respect to execution progress. One explicit batch binding.
class ScanNode {
 public:
    ScanNode(std::shared_ptr<const segcore::reader::ColumnReader> reader,
             ReadBatchBinding binding,
             std::shared_ptr<const segcore::reader::ReadContext> context);
    segcore::reader::ReadResult
    Read(SelectionView rows, segcore::reader::ReadProjection projection) const;

 private:
    std::shared_ptr<const segcore::reader::ColumnReader> reader_;
    ReadBatchBinding binding_;
    std::shared_ptr<const segcore::reader::ReadContext> context_;
};
}  // namespace milvus::exec::access
