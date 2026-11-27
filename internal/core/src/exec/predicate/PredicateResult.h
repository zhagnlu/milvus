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
#include "common/VectorAccess.h"
#include "exec/access/ReadBatchBinding.h"
#include "segcore/reader/IndexResult.h"

namespace milvus::exec::predicate {

// Only full three-valued results may become general BOOL values.
// Exact TRUE sets have a separate null-rejecting filter API.
void
ValidateBinding(const ReadBatchBinding& binding,
                SelectionView rows,
                size_t source_rows);
ReadOnlyVectorView
MapPredicate(const segcore::reader::IndexResult& result,
             const ReadBatchBinding& binding,
             SelectionView rows);
// The shared immutable result can back a Range view without materialization.
ReadOnlyVectorView
MapPredicate(std::shared_ptr<const segcore::reader::IndexResult> result,
             const ReadBatchBinding& binding,
             SelectionView rows);
OwnedSelection
MapFilterTrue(const segcore::reader::IndexResult& result,
              const ReadBatchBinding& binding,
              SelectionView rows);
}  // namespace milvus::exec::predicate
