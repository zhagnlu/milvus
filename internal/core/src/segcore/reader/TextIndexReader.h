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
#include <functional>
#include "segcore/reader/IndexReader.h"
#include "pb/plan.pb.h"

namespace milvus::index {
class IIndexReaderBase;
}
namespace milvus::segcore::reader {
struct TextIndexQuery {
    proto::plan::OpType op;
    std::string text;
    std::vector<proto::plan::GenericValue> arguments;
};
// A prepared resource owner, with lazy pinning. Results belong to the fixed
// query prefix even when the growing index has fewer/more visible documents.
class TextIndexReader final : public IndexReader {
 public:
    using Loader = std::function<std::shared_ptr<const index::IIndexReaderBase>(
        const ReadContext&)>;
    TextIndexReader(Loader loader,
                    int64_t count,
                    TextIndexQuery query,
                    int64_t covered = -1)
        : loader_(std::move(loader)),
          count_(count),
          query_(std::move(query)),
          covered_(covered) {
    }
    int64_t
    row_count() const override {
        return count_;
    }
    IndexResult
    Query(const ReadContext&) const override;

 private:
    Loader loader_;
    int64_t count_;
    TextIndexQuery query_;
    int64_t covered_;
};
}  // namespace milvus::segcore::reader
