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
#include "common/VectorAccess.h"

namespace milvus::exec {

// Both masks own their storage and use batch-local positions. valid must be a
// subset of defined. Undefined is NOT NULL and must not be inspected as such.
struct ValidityResult {
    OwnedSelection defined;
    OwnedSelection valid;
};

namespace access {
class ReaderInputAccess;
}

// The Reader bridge can retain masks already checked by ScanNode/PositionNode.
// Other value producers keep returning a view; InputSlot derives its validity.
class InputValueResult {
 public:
    InputValueResult(ReadOnlyVectorView values) : values_(std::move(values)) {
    }

 private:
    friend class InputSlot;
    friend class access::ReaderInputAccess;
    InputValueResult(ReadOnlyVectorView values, ValidityResult validity)
        : values_(std::move(values)), validity_(std::move(validity)) {
    }
    ReadOnlyVectorView values_;
    std::optional<ValidityResult> validity_;
};

// Called synchronously for missing positions only. Successful results cover
// every requested row, may include extra valid coverage, and own their data.
// NULL counts as a defined value. These callbacks do not advance a batch.
using InputValueLoader = std::function<InputValueResult(SelectionView)>;
using InputValidityLoader = std::function<ValidityResult(SelectionView)>;

struct InputLoaders {
    InputValueLoader values;
    InputValidityLoader validity;
};

}  // namespace milvus::exec
