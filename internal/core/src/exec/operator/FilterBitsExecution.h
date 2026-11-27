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

#pragma once

#include "common/Vector.h"
#include "exec/access/SegmentReaderInput.h"
#include "expr/ITypeExpr.h"
#include "segcore/reader/ReadRequest.h"
#include "exec/plan/ExecutionPreparer.h"

namespace milvus::segcore {
class SegmentReadSnapshot;
}

namespace milvus::exec {

// FilterBits consumer of the parallel expression engine. Keep this header out
// of legacy operator headers: exec::plan and milvus::plan are distinct.
// No MVCC/delete filtering, global cache or implicit cursors.
class FilterBitsExecution {
 public:
    // nullptr is a metadata-only capability miss, before any payload reading.
    // Invalid bindings and compile errors propagate, as do all execution errors.
    // The snapshot must belong to this request; the caller retains any borrowed
    // OpContext until execution/destruction completes. No asynchronous reads.
    static std::unique_ptr<FilterBitsExecution>
    TryCreate(const expr::TypedExprPtr& predicate,
              const expr::TypedExprPtr& ttl_predicate,
              std::shared_ptr<const segcore::SegmentReadSnapshot> snapshot,
              int64_t active_count,
              int64_t batch_size,
              std::shared_ptr<const segcore::reader::ReadContext> context,
              std::shared_ptr<const Schema> schema = {});

    // Calls bind synchronously during preparation; no binder/segment borrow is
    // retained. A missing binder or unsupported dependency returns nullptr.
    static std::unique_ptr<FilterBitsExecution>
    TryCreateWithBinding(
        const expr::TypedExprPtr& predicate,
        const expr::TypedExprPtr& ttl_predicate,
        const access::BindColumnReader& bind,
        int64_t active_count,
        int64_t batch_size,
        std::shared_ptr<const segcore::reader::ReadContext> context,
        const plan::PhysicalPrepareOptions& physical = {},
        std::shared_ptr<const Schema> schema = {},
        bool array_element_arithmetic = false);

    ~FilterBitsExecution();
    FilterBitsExecution(const FilterBitsExecution&) = delete;
    FilterBitsExecution&
    operator=(const FilterBitsExecution&) = delete;

    // Emits one segment-wide exclusion bitmap: FALSE/UNKNOWN -> 1; TRUE -> 0.
    // Empty domains and subsequent calls return nullptr. All output validity
    // bits are true. Failure publishes neither output nor completed progress.
    RowVectorPtr
    GetOutput();

    bool
    IsFinished() const;

    const plan::PreparedExecution&
    prepared() const;

 private:
    struct State;
    explicit FilterBitsExecution(std::unique_ptr<State> state);
    std::unique_ptr<State> state_;
};
}  // namespace milvus::exec
