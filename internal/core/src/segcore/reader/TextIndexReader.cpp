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
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific
// language governing permissions and limitations under the License.

#include "segcore/reader/TextIndexReader.h"
#include "index/contracts/query/ITextMatchReader.h"
#include "segcore/reader/ScalarQueryView.h"

namespace milvus::segcore::reader {
IndexResult
TextIndexReader::Query(const ReadContext& context) const {
    context.Check();
    AssertInfo(loader_ && count_ >= 0, "invalid prepared text index source");
    uint32_t argument = query_.op == proto::plan::TextMatch ? 1 : 0;
    if (query_.op == proto::plan::TextMatchFuzzy && query_.arguments.empty())
        ThrowInfo(InvalidParameter,
                  "max_edit_distance is required for text_match_fuzzy");
    if (!query_.arguments.empty()) {
        AssertInfo(query_.arguments[0].has_int64_val(),
                   "text query argument must be INT64");
        const auto value = query_.arguments[0].int64_val();
        if (query_.op == proto::plan::PhraseMatch &&
            (value < 0 || value > std::numeric_limits<uint32_t>::max()))
            ThrowInfo(InvalidParameter,
                      "phrase match slop must be within [0, UINT32_MAX]");
        if (query_.op == proto::plan::TextMatchFuzzy &&
            (value < 0 || value > 2))
            ThrowInfo(InvalidParameter,
                      "fuzzy match distance must be within [0, 2]");
        argument = static_cast<uint32_t>(value);
    }
    auto owner = loader_(context);
    auto* index = dynamic_cast<const index::ITextMatchReader*>(owner.get());
    AssertInfo(index, "prepared text index loader returned no resource");
    auto matches = [&] {
        switch (query_.op) {
            case proto::plan::TextMatch:
                return index->MatchQuery(query_.text, argument);
            case proto::plan::PhraseMatch:
                return index->PhraseMatchQuery(query_.text, argument);
            case proto::plan::TextMatchFuzzy:
                return index->FuzzyMatchQuery(query_.text, argument);
            default:
                ThrowInfo(UnexpectedError, "invalid compiled text query");
        }
    }();
    auto valid = NullReader(*owner).IsNotNull();
    // Count can grow between the two index calls. Never expose validity for a
    // row outside the match query's visible domain. A short tail is UNKNOWN.
    const auto visible =
        std::min({matches.size(),
                  valid.size(),
                  static_cast<size_t>(covered_ < 0 ? count_ : covered_)});
    matches.resize(visible);
    valid.resize(visible);
    matches.resize(count_, false);
    valid.resize(count_, false);
    matches &= valid;
    context.Check();
    return IndexResult::Full(SelectionView::FromBitmap(matches),
                             SelectionView::FromBitmap(valid));
}
}  // namespace milvus::segcore::reader
