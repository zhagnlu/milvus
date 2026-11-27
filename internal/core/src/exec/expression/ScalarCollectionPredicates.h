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
MakeArrayElement(ExprPtr input, int64_t index, DataType type);
ExprPtr
MakeCollectionLength(ExprPtr input, std::vector<std::string> path = {});
ExprPtr
MakeArrayEquality(ExprPtr input, proto::plan::Array value, bool negate);
ExprPtr
MakeArrayContains(ExprPtr input,
                  DataType value_type,
                  std::vector<proto::plan::GenericValue> terms,
                  bool all);
}  // namespace milvus::exec::scalar
