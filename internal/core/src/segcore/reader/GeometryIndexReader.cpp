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

#include "segcore/reader/GeometryIndexReader.h"
#include "index/contracts/query/ISpatialReader.h"
#include "segcore/reader/ScalarQueryView.h"
#include "exec/expression/GISFunctionFilterExpr.h"
#include "common/Geometry.h"
namespace milvus::segcore::reader {
IndexResult
GeometryIndexReader::Query(const ReadContext& context) const {
    context.Check();
    AssertInfo(resource_ && count_ >= 0,
               "invalid prepared geometry index source");
    auto owner = resource_->CreateReader(context.op_context.get());
    auto* index = dynamic_cast<const index::ISpatialReader*>(owner.get());
    AssertInfo(index, "prepared geometry index type mismatch");
    Geometry geometry(GetThreadLocalGEOSContext(), query_.wkt.c_str());
    auto candidates =
        index->Candidates(exec::gis_detail::ToSpatialOp(query_.op), geometry);
    // Historical R-Trees can omit interior empty/corrupt rows. A short tree
    // cannot prove any negative, so refine the entire fixed row domain.
    if (candidates.size() < static_cast<size_t>(count_))
        candidates = TargetBitmap(count_, true);
    else
        candidates.resize(count_);
    auto valid = NullReader(*owner).IsNotNull();
    AssertInfo(valid.size() == static_cast<size_t>(count_),
               "geometry index validity domain mismatch");
    candidates &= valid;
    auto nulls = valid.clone();
    nulls.flip();
    auto false_rows = candidates.clone();
    false_rows.flip();
    false_rows &= valid;
    context.Check();
    return IndexResult::Candidates(SelectionView::FromBitmap(candidates),
                                   SelectionView::FromBitmap(false_rows),
                                   SelectionView::FromBitmap(nulls));
}
}  // namespace milvus::segcore::reader
