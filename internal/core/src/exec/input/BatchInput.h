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

#include "exec/input/InputSlot.h"

namespace milvus {
class Geometry;
}

namespace milvus::exec {

// Single-flow, batch-local state. No Segment, index, cursor or error translation.
// Loaders capture immutable bindings and retain owners needed for future reads.
// Copying/moving a live input could share callbacks bound to another batch.
// Only the execution workspace may reset state and rebind its source callbacks;
// published vectors retain independent immutable owners.
class BatchInput {
 public:
    using GeometryLookup = std::function<const Geometry*(size_t)>;
    void
    BindGeometryLookup(column_index_t slot, GeometryLookup lookup);
    const GeometryLookup&
    GetGeometryLookup(column_index_t slot) const;
    using ValueLoader = InputValueLoader;
    using ValidityLoader = InputValidityLoader;

    BatchInput(RowTypePtr schema, size_t size);
    explicit BatchInput(const RowVector& input);
    BatchInput(const BatchInput&) = delete;
    BatchInput&
    operator=(const BatchInput&) = delete;
    BatchInput(BatchInput&&) = delete;
    BatchInput&
    operator=(BatchInput&&) = delete;

    size_t
    size() const {
        return size_;
    }
    const RowTypePtr&
    schema() const {
        return schema_;
    }

    // Bind each kind once, in any order, before the first nonempty request for
    // ANY slot. Seeds may be partial; loaders can fill missing positions later.
    // Conflicting seed validity is rejected without changing existing state.
    void
    BindValues(column_index_t slot, ReadOnlyVectorView values);
    void
    BindValidity(column_index_t slot, ValidityResult validity);
    void
    BindLoader(column_index_t slot,
               ValueLoader values,
               ValidityLoader validity = {});

    // Snapshot only a referenced upstream column and its explicit coverage.
    // Row-level NULLs override children without reading their payloads. Encoded
    // immutable columns share owners; mutable legacy columns are detached.
    void
    BindColumn(column_index_t slot,
               const RowVector& upstream,
               column_index_t column,
               SelectionView defined);

    // Empty requests never load or freeze binding, and return empty coverage.
    // Nonempty successful requests may return additional cached coverage.
    ReadOnlyVectorView
    GetColumn(column_index_t slot, SelectionView rows);
    ValidityResult
    GetValidity(column_index_t slot, SelectionView rows);

 private:
    friend class BatchInputResetAccess;
    void
    Reset(size_t size);
    InputSlot&
    GetSlot(column_index_t slot, SelectionView rows);
    InputSlot&
    BindSlot(column_index_t slot);
    RowTypePtr schema_;
    size_t size_;
    std::vector<InputSlot> slots_;
    std::vector<GeometryLookup> geometry_;
    bool started_ = false;
};

}  // namespace milvus::exec
