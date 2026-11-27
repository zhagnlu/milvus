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
#include "common/Selection.h"

namespace milvus::segcore::reader {

enum class IndexResultKind { FullThreeValued, ExactTrue, Candidates };

// Immutable, whole-source row domain. No excluded bitmap or field-value
// interpretation. Borrowed masks are copied; owned masks can be transferred.
// Neither form retains a writable alias into the index.
class IndexResult {
 public:
    static IndexResult
    Full(SelectionView matches, SelectionView valid);
    static IndexResult
    Full(OwnedSelection matches, OwnedSelection valid);
    static IndexResult
    TrueOnly(SelectionView matches);
    // candidates bounds possible TRUE rows. false_rows/null_rows are additional
    // exact proofs, not the complement of candidates. Unproven outside rows
    // can still be FALSE or UNKNOWN and require exact evaluation for projection.
    static IndexResult
    Candidates(SelectionView candidates,
               SelectionView false_rows,
               SelectionView null_rows);

    IndexResultKind
    kind() const {
        return kind_;
    }
    size_t
    size() const {
        return selected_.view().row_count();
    }
    SelectionView
    selected() const& {
        return selected_.view();
    }
    SelectionView
    valid() const& {
        return valid_.view();
    }
    SelectionView
    false_rows() const& {
        return false_.view();
    }
    SelectionView
    null_rows() const& {
        return null_.view();
    }
    SelectionView
    selected() const&& = delete;
    SelectionView
    valid() const&& = delete;
    SelectionView
    false_rows() const&& = delete;
    SelectionView
    null_rows() const&& = delete;

 private:
    IndexResult(IndexResultKind kind,
                SelectionView selected,
                SelectionView valid,
                SelectionView false_rows,
                SelectionView null_rows);
    IndexResult(OwnedSelection matches, OwnedSelection valid);
    IndexResultKind kind_;
    OwnedSelection selected_, valid_, false_, null_;
};

}  // namespace milvus::segcore::reader
