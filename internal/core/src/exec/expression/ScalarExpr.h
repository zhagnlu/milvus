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

#include <memory>
#include <vector>

#include "exec/expression/ScalarEvalCtx.h"

namespace milvus::exec::scalar {

// Migration boundary: separate from the legacy exec::Expr/SegmentExpr.
// Nodes contain calculation structure only and can be reused with a new batch.
// Results retain batch coordinates, are immutable, and define exactly rows.
class Expr {
 public:
    explicit Expr(DataType type) : type_(type) {
    }
    virtual ~Expr() = default;

    DataType
    type() const {
        return type_;
    }

    // Checks shape/coverage centrally. An empty selection never calls EvalImpl.
    // Child/loader exceptions propagate without translation or fallback.
    ReadOnlyVectorView
    Eval(SelectionView rows, EvalCtx& ctx) const;

 protected:
    virtual ReadOnlyVectorView
    EvalImpl(SelectionView rows, EvalCtx& ctx) const = 0;

 private:
    DataType type_;
};

using ExprPtr = std::shared_ptr<const Expr>;

enum class CompareOp {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Prefix
};
enum class LogicalOp { And, Or };

// Stable legacy tiers. Opaque/possibly throwing nodes are ordering barriers;
// a candidate bitmap is never classified as an exact BOOL expression here.
enum class ReorderTier {
    Namespace,
    Numeric,
    Indexed,
    String,
    Membership,
    LightConjunction,
    Other,
    Array,
    StringLike,
    ArrayLike,
    Json,
    JsonLike,
    HeavyConjunction,
    Compare
};
struct ReorderProperty {
    ReorderTier tier = ReorderTier::Other;
    bool safe = false;
};
using ClassifyExpression = std::function<ReorderProperty(const Expr*)>;

// Returns a query-local tree. Recurse only through AND/OR; NOT and unknown
// nodes remain intact. Flatten equal logical operators and stably sort safe
// runs, without moving an operand across an exception/side-effect barrier.
ExprPtr
AdjustLogicalExpression(
    ExprPtr root,
    const ClassifyExpression& classify,
    const std::function<ExprPtr(ExprPtr)>& resolve = {},
    const std::function<std::vector<ExprPtr>(LogicalOp, std::vector<ExprPtr>)>&
        physical = {});

// Snapshot a single typed scalar (including a typed NULL), then broadcast it
// to each batch. NONE/untyped NULL must be resolved by the future compiler.
ExprPtr
MakeLiteral(const VectorPtr& single_value);

// Numeric operands use their native C++ comparison promotions, as in the legacy
// column comparator. String operands compare string views; Prefix is string-only.
// No JSON extraction or string/numeric coercion.
ExprPtr
MakeComparison(CompareOp op, ExprPtr left, ExprPtr right);

// One input evaluation and one BOOL result for a two-sided range. Bounds are
// snapshotted, non-null single values of the input's type, not broadcast Exprs.
ExprPtr
MakeRange(ExprPtr input,
          const VectorPtr& lower,
          const VectorPtr& upper,
          bool lower_inclusive,
          bool upper_inclusive);

// Full three-valued semantics, including in projection. Only definite FALSE
// short-circuits AND and only definite TRUE short-circuits OR.
ExprPtr
MakeLogical(LogicalOp op, std::vector<ExprPtr> inputs);

ExprPtr
MakeNot(ExprPtr input);

ExprPtr
MakeIsNull(ExprPtr input, bool negate = false);

}  // namespace milvus::exec::scalar
