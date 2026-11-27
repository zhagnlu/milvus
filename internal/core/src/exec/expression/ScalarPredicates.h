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

#include "exec/expression/ScalarExpr.h"

namespace milvus {
class RoaringMembership;
}

namespace milvus::exec::scalar {

// Constants are snapshotted when constructing the expression. NULL constants
// must have been resolved by the logical compiler, not treated as SQL NULL IN.
ExprPtr
MakeTerm(ExprPtr input, const VectorPtr& constants);

// Already published, type-checked constants can be shared with a compiled
// plan. Construct the path-specific lookup structure without another snapshot.
ExprPtr
MakeTerm(ExprPtr input, const ReadOnlyVectorView& constants);

enum class StringOp { Prefix, Suffix, Contains, Like, Regex };
ExprPtr
MakeStringMatch(ExprPtr input, StringOp op, std::string pattern);

// Fuse adjacent predicates over the same direct input slot, preserving their
// order and lazy pattern construction. Never cross another operand or NOT.
std::vector<ExprPtr>
FuseAdjacentStringMatches(LogicalOp op,
                          std::vector<ExprPtr> inputs,
                          const std::function<ExprPtr(ExprPtr)>& resolve = {});

// Uses the same Json::exist semantics as the legacy raw path: missing, NULL
// source, JSON null and empty objects are false, not UNKNOWN.
ExprPtr
MakeJsonExists(ExprPtr input, std::vector<std::string> path);

ExprPtr
MakeRoaringMembership(ExprPtr input,
                      std::shared_ptr<const RoaringMembership> membership);

// Retains the immutable MBF1 owner and reuses the production SBBF prober.
// JSON uses strict runtime types, not the numeric coercions of exact IN.
ExprPtr
MakeBloomMembership(ExprPtr input,
                    std::shared_ptr<const std::string> blob,
                    std::vector<std::string> path = {});

}  // namespace milvus::exec::scalar
