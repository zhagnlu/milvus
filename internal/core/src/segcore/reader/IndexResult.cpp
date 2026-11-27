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

#include "segcore/reader/IndexResult.h"

namespace milvus::segcore::reader {
IndexResult::IndexResult(IndexResultKind kind,
                         SelectionView selected,
                         SelectionView valid,
                         SelectionView false_rows,
                         SelectionView null_rows)
    : kind_(kind),
      selected_(OwnedSelection::CopyOf(selected)),
      valid_(OwnedSelection::CopyOf(valid)),
      false_(OwnedSelection::CopyOf(false_rows)),
      null_(OwnedSelection::CopyOf(null_rows)) {
    AssertInfo(valid.row_count() == size() &&
                   false_rows.row_count() == size() &&
                   null_rows.row_count() == size(),
               "index result domain mismatch");
}

IndexResult
IndexResult::Full(SelectionView matches, SelectionView valid) {
    AssertInfo(matches.row_count() == valid.row_count(),
               "index validity domain mismatch");
    AssertInfo(matches.IsSubsetOf(valid), "index TRUE row cannot be NULL");
    return IndexResult(OwnedSelection::CopyOf(matches),
                       OwnedSelection::CopyOf(valid));
}
IndexResult::IndexResult(OwnedSelection matches, OwnedSelection valid)
    : kind_(IndexResultKind::FullThreeValued),
      selected_(std::move(matches)),
      valid_(std::move(valid)),
      false_(OwnedSelection::None(size())),
      null_(OwnedSelection::None(size())) {
}
IndexResult
IndexResult::Full(OwnedSelection matches, OwnedSelection valid) {
    AssertInfo(matches.view().row_count() == valid.view().row_count(),
               "index validity domain mismatch");
    AssertInfo(matches.view().IsSubsetOf(valid.view()),
               "index TRUE row cannot be NULL");
    return IndexResult(std::move(matches), std::move(valid));
}
IndexResult
IndexResult::TrueOnly(SelectionView matches) {
    auto none = SelectionView::None(matches.row_count());
    return IndexResult(IndexResultKind::ExactTrue, matches, none, none, none);
}
IndexResult
IndexResult::Candidates(SelectionView candidates,
                        SelectionView false_rows,
                        SelectionView null_rows) {
    IndexResult result(IndexResultKind::Candidates,
                       candidates,
                       SelectionView::None(candidates.row_count()),
                       false_rows,
                       null_rows);
    const auto false_candidates =
        OwnedSelection::Intersect(false_rows, candidates);
    const auto false_null = OwnedSelection::Intersect(false_rows, null_rows);
    AssertInfo(false_candidates.view().IsNone() && false_null.view().IsNone(),
               "conflicting index candidate/FALSE/NULL proofs");
    const auto null_candidates =
        OwnedSelection::Intersect(null_rows, candidates);
    AssertInfo(null_candidates.view().IsNone(),
               "NULL proof overlaps candidates");
    return result;
}
}  // namespace milvus::segcore::reader
