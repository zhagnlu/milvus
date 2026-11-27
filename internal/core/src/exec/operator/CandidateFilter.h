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
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <memory>
#include "common/Types.h"
#include "expr/ITypeExpr.h"

namespace milvus::exec {
class ExecContext;
class QueryContext;
class FilterBitsExecution;

// Query-scoped predicate preparation shared by candidate consumers. Eval returns
// TRUE bits in candidate order; FALSE/UNKNOWN are both rejected. No cursor.
// The caller keeps the QueryContext and its borrowed OpContext alive.
class CandidateFilter {
 public:
    static std::unique_ptr<CandidateFilter>
    TryCreate(const expr::TypedExprPtr& predicate,
              ExecContext* context,
              const std::string& element_structure = {});
    ~CandidateFilter();
    TargetBitmap
    Eval(const FixedVector<int32_t>& offsets);

 private:
    explicit CandidateFilter(std::unique_ptr<FilterBitsExecution> execution,
                             int64_t count,
                             QueryContext* query);
    std::unique_ptr<FilterBitsExecution> execution_;
    int64_t count_;
    QueryContext* query_;
};
}  // namespace milvus::exec
