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

#include <functional>
#include <utility>

#include "exec/input/BatchInput.h"

namespace milvus::exec::scalar {

class EvalCtx;

// Full three-valued BOOL inputs occupy a separate namespace from field slots.
// Calculation knows this protocol, never IndexNode, Segment or row offsets.
class PredicateInputs {
 public:
    virtual ~PredicateInputs() = default;
    virtual ReadOnlyVectorView
    Get(column_index_t slot, SelectionView rows, EvalCtx& ctx) = 0;
};

// A single batch's calculation resources. No segment, offsets, reader or
// mutable implicit selection. The input and checkpoint captures outlive Eval.
class EvalCtx {
 public:
    explicit EvalCtx(BatchInput& input,
                     std::function<void()> checkpoint = {},
                     std::shared_ptr<PredicateInputs> predicates = {})
        : input_(input),
          checkpoint_(std::move(checkpoint)),
          predicates_(std::move(predicates)) {
    }

    BatchInput&
    input() const {
        return input_;
    }

    void
    Checkpoint() const {
        if (checkpoint_) {
            checkpoint_();
        }
    }

    ReadOnlyVectorView
    GetPredicate(column_index_t slot, SelectionView rows) {
        AssertInfo(predicates_, "predicate input is not bound");
        return predicates_->Get(slot, rows, *this);
    }

 private:
    BatchInput& input_;
    std::function<void()> checkpoint_;
    std::shared_ptr<PredicateInputs> predicates_;
};

}  // namespace milvus::exec::scalar
