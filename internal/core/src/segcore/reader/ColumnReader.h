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

#include "segcore/reader/ReadResult.h"

namespace milvus::exec {
class SimpleGeometryCache;
}

namespace milvus::segcore::reader {

// One immutable source generation; each call is finite and independently
// positioned. No Expr, batch cache, next-batch decision, or index predicate.
class ColumnReader {
 public:
    virtual ~ColumnReader() = default;
    virtual std::shared_ptr<exec::SimpleGeometryCache>
    geometry_cache() const {
        return nullptr;
    }
    virtual DataType
    type() const = 0;
    virtual int64_t
    row_count() const = 0;
    virtual ReadResult
    ReadRange(const RangeRequest&, const ReadContext&) const = 0;
    virtual ReadResult
    Take(const TakeRequest&, const ReadContext&) const = 0;
};

}  // namespace milvus::segcore::reader
