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

#include "exec/expression/ScalarExpr.h"
#include "pb/plan.pb.h"

namespace milvus::exec::scalar {
ExprPtr
MakeJsonComparison(ExprPtr input,
                   std::vector<std::string> path,
                   proto::plan::OpType operation,
                   proto::plan::GenericValue bound);
ExprPtr
MakeJsonRange(ExprPtr input,
              std::vector<std::string> path,
              proto::plan::GenericValue lower,
              proto::plan::GenericValue upper,
              bool lower_inclusive,
              bool upper_inclusive);
ExprPtr
MakeJsonTerm(ExprPtr input,
             std::vector<std::string> path,
             std::vector<proto::plan::GenericValue> terms,
             bool in_field);
ExprPtr
MakeJsonContains(ExprPtr input,
                 std::vector<std::string> path,
                 std::vector<proto::plan::GenericValue> terms,
                 bool all,
                 bool same_type);
}  // namespace milvus::exec::scalar
