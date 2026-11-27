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

#include "exec/input/InputLoader.h"

namespace milvus::exec {
class BatchInput;

// BatchInput owns the binding lifecycle; this internal state machine owns one
// column's values, validity, callbacks and transactional incremental loading.
// Slots are never copied between queries/batches or mutated concurrently.
class InputSlot {
 public:
    InputSlot(DataType type, size_t size);
    InputSlot(const InputSlot&) = delete;
    InputSlot&
    operator=(const InputSlot&) = delete;
    InputSlot(InputSlot&&) = default;
    InputSlot&
    operator=(InputSlot&&) = default;

 private:
    friend class BatchInput;
    struct State {
        ReadOnlyVectorView values;
        ValidityResult validity;
    };

    void
    Reset(size_t size);
    void
    BindValues(ReadOnlyVectorView values);
    void
    BindValidity(ValidityResult validity);
    void
    BindLoader(InputLoaders loaders);
    ReadOnlyVectorView
    GetColumn(SelectionView rows);
    ValidityResult
    GetValidity(SelectionView rows);
    State
    LoadValues(SelectionView rows);
    void
    Publish(State state) noexcept;
    void
    ValidateValues(const ReadOnlyVectorView& values,
                   SelectionView requested) const;
    void
    ValidateValidity(const ValidityResult& validity,
                     SelectionView requested) const;
    static ValidityResult
    ExtractValidity(const ReadOnlyVectorView& values);
    static ValidityResult
    MergeValidity(const ValidityResult& old, const ValidityResult& patch);

    // The empty immutable view can be reused for equal-sized batches.
    ReadOnlyVectorView empty_;
    State state_;
    InputLoaders loaders_;
    bool bound_values_ = false;
    bool bound_validity_ = false;
    bool bound_loader_ = false;
    bool loading_ = false;
};

}  // namespace milvus::exec
