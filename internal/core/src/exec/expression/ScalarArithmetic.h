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

// Value-producing +, -, *, /. Integral inputs promote to INT64; any floating
// input promotes to DOUBLE, matching the Go scalar parser. NULL propagates.
// Integer add/sub/mul wrap; divide-by-zero and INT64_MIN / -1 throw only on
// selected, non-NULL rows. No results are published on failure.
ExprPtr
MakeArithmetic(ExprPtr left, ExprPtr right, proto::plan::ArithOpType operation);

// Fuses field arithmetic and comparison. Signed integer operands/results use
// INT64, as in the legacy kernel; FLOAT/DOUBLE retain their own precision.
// truncate_floating selects the ordinary-field Mod/bitwise/shift convention:
// truncate both operands to INT64, operate, then compare at field precision.
// Array-element Mod uses floating remainder and leaves this flag false.
ExprPtr
MakeArithmeticCompare(ExprPtr input,
                      proto::plan::ArithOpType operation,
                      CompareOp comparison,
                      const VectorPtr& operand,
                      const VectorPtr& bound,
                      bool truncate_floating = false);

// Shared numeric kernel for raw JSON and stats. Callers validate constant
// divisors before evaluation; INT64_MIN / -1 remains a row-local error.
bool
MatchJsonArithmeticNumber(int64_t value,
                          proto::plan::ArithOpType operation,
                          CompareOp comparison,
                          const proto::plan::GenericValue& operand,
                          const proto::plan::GenericValue& bound);
bool
MatchJsonArithmeticNumber(double value,
                          proto::plan::ArithOpType operation,
                          CompareOp comparison,
                          const proto::plan::GenericValue& operand,
                          const proto::plan::GenericValue& bound);

// JSON numbers preserve INT64 arithmetic when both constants and payload are
// integral; uint64/double payloads use the legacy double arithmetic branch.
ExprPtr
MakeJsonArithmeticCompare(ExprPtr input,
                          std::vector<std::string> path,
                          proto::plan::ArithOpType operation,
                          CompareOp comparison,
                          proto::plan::GenericValue operand,
                          proto::plan::GenericValue bound);

}  // namespace milvus::exec::scalar
