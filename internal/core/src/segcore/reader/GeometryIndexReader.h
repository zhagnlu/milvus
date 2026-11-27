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

#pragma once
#include "segcore/reader/IndexReader.h"
#include "index/contracts/IIndex.h"
#include "pb/plan.pb.h"
namespace milvus::segcore::reader {
struct GeometryIndexQuery {
    proto::plan::GISFunctionFilterExpr_GISOp op;
    std::string wkt;
};
class GeometryIndexReader final : public IndexReader {
 public:
    GeometryIndexReader(index::IndexPtr resource,
                        int64_t count,
                        GeometryIndexQuery query)
        : resource_(std::move(resource)),
          count_(count),
          query_(std::move(query)) {
    }
    int64_t
    row_count() const override {
        return count_;
    }
    IndexResult
    Query(const ReadContext&) const override;

 private:
    index::IndexPtr resource_;
    int64_t count_;
    GeometryIndexQuery query_;
};
}  // namespace milvus::segcore::reader
