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

#include <optional>
#include "common/VectorAccess.h"
#include "segcore/reader/ReadRequest.h"

namespace milvus::segcore::reader {

// Immutable target values retain their payload owners independently of a cursor.
// Scan/Growing can borrow pinned data; generated/materialized values own buffers.
// This first adapter has no raw/transformed distinction: no extraction/cast.
// Future semantic readers must NOT label raw overread as target coverage.
struct ReadResult {
    std::optional<ReadOnlyVectorView> values;
    OwnedSelection defined;
    OwnedSelection valid;

    void
    Validate(DataType type,
             SelectionView requested,
             ReadProjection projection) const {
        AssertInfo(defined.view().row_count() == requested.row_count() &&
                       valid.view().row_count() == requested.row_count(),
                   "reader result domain mismatch");
        AssertInfo(requested.IsSubsetOf(defined.view()),
                   "reader result is missing requested validity");
        AssertInfo(valid.view().IsSubsetOf(defined.view()),
                   "reader valid row is undefined");
        if (projection == ReadProjection::ValuesAndValidity) {
            AssertInfo(values.has_value(), "reader did not return values");
        }
        if (values) {
            AssertInfo(values->size() == requested.row_count() &&
                           values->logical_type() == type,
                       "reader value shape/type mismatch");
            if (projection == ReadProjection::ValuesAndValidity) {
                AssertInfo(requested.IsSubsetOf(values->defined_rows()),
                           "reader result is missing requested value");
            }
            AssertInfo(values->defined_rows().IsSubsetOf(defined.view()),
                       "reader value/validity mismatch");
            const auto borrowed = values->TryBorrowValidity();
            const auto owned =
                borrowed ? OwnedSelection::None(0) : values->CopyValidity();
            const auto actual = borrowed ? *borrowed : owned.view();
            AssertInfo(actual.EqualsOn(valid.view(), values->defined_rows()),
                       "reader value/validity mismatch");
        }
    }
};

}  // namespace milvus::segcore::reader
