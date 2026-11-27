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

#include "exec/expression/ScalarExpr.h"
#include "exec/expression/ScalarBatchKernels.h"
#include "exec/expression/ScalarStringKernels.h"
#include "exec/expression/ScalarPredicates.h"
#include "exec/expression/ScalarComplexPredicates.h"

#include "exec/expression/InputRefExpr.h"

namespace milvus::exec::scalar {
namespace {

template <typename Fn>
decltype(auto)
DispatchComparable(DataType type, Fn&& fn) {
    switch (type) {
        case DataType::BOOL:
            return fn.template operator()<bool>();
        case DataType::INT8:
            return fn.template operator()<int8_t>();
        case DataType::INT16:
            return fn.template operator()<int16_t>();
        case DataType::INT32:
            return fn.template operator()<int32_t>();
        case DataType::INT64:
        case DataType::TIMESTAMPTZ:
            return fn.template operator()<int64_t>();
        case DataType::FLOAT:
            return fn.template operator()<float>();
        case DataType::DOUBLE:
            return fn.template operator()<double>();
        case DataType::STRING:
        case DataType::VARCHAR:
        case DataType::TEXT:
            return fn.template operator()<std::string_view>();
        default:
            // Internal implementation capability, not user-input validation.
            ThrowInfo(
                Unsupported, "scalar comparison does not support {}", type);
    }
}

template <typename T>
T
ReadValue(const ReadOnlyVectorView& values,
          size_t row,
          const T* flat = nullptr) {
    if constexpr (std::is_same_v<T, std::string_view>) {
        return values.StringViewAt(row);
    } else {
        return flat ? flat[row] : values.ValueAt<T>(row);
    }
}

template <typename T>
const T*
ComparableData(const ReadOnlyVectorView& values) {
    if constexpr (std::is_same_v<T, std::string_view>)
        return nullptr;
    else
        return values.FlatData<T>();
}

template <typename T>
std::optional<T>
ReadSelectedValue(const ReadOnlyVectorView& values, size_t row, const T* flat) {
    if constexpr (std::is_same_v<T, std::string_view>) {
        return values.TryStringViewAt(row);
    } else {
        if (!flat)
            return values.TryValueAt<T>(row);
        if (!values.IsValid(row))
            return std::nullopt;
        return flat[row];
    }
}

template <typename T, typename U>
bool
Compare(CompareOp op, const T& left, const U& right) {
    // Compare directly rather than using a three-way comparator: unordered
    // floating-point values must keep their native comparison semantics.
    switch (op) {
        case CompareOp::Equal:
            return left == right;
        case CompareOp::NotEqual:
            return left != right;
        case CompareOp::Less:
            return left < right;
        case CompareOp::LessEqual:
            return left <= right;
        case CompareOp::Greater:
            return left > right;
        case CompareOp::GreaterEqual:
            return left >= right;
        case CompareOp::Prefix:
            if constexpr (std::is_same_v<T, std::string_view> &&
                          std::is_same_v<U, std::string_view>)
                return left.starts_with(right);
            break;
    }
    ThrowInfo(UnexpectedError, "invalid scalar comparison operation");
}

class LiteralExpr final : public Expr {
 public:
    explicit LiteralExpr(ReadOnlyVectorView value)
        : Expr(value.logical_type()), value_(std::move(value)) {
        AssertInfo(value_.size() == 1, "literal needs exactly one typed value");
    }

 protected:
    ReadOnlyVectorView
    EvalImpl(SelectionView rows, EvalCtx&) const override {
        return value_.Broadcast(rows);
    }

 private:
    ReadOnlyVectorView value_;
};

class ComparisonExpr final : public Expr {
 public:
    ComparisonExpr(CompareOp op, ExprPtr left, ExprPtr right)
        : Expr(DataType::BOOL),
          op_(op),
          left_(std::move(left)),
          right_(std::move(right)) {
        AssertInfo(left_ && right_, "comparison requires two expressions");
        DispatchComparable(left_->type(), [&]<typename T>() {
            DispatchComparable(right_->type(), [&]<typename U>() {
                constexpr bool left_string =
                    std::is_same_v<T, std::string_view>;
                constexpr bool right_string =
                    std::is_same_v<U, std::string_view>;
                AssertInfo(left_string == right_string,
                           "comparison cannot mix string and numeric inputs");
                AssertInfo(op != CompareOp::Prefix || left_string,
                           "prefix comparison requires string inputs");
                if constexpr (left_string == right_string)
                    kernel_ = &EvaluateComparison<T, U>;
            });
        });
        AssertInfo(op >= CompareOp::Equal && op <= CompareOp::Prefix,
                   "invalid scalar comparison operation");
    }

 protected:
    ReadOnlyVectorView
    EvalImpl(SelectionView rows, EvalCtx& ctx) const override {
        const auto left = left_->Eval(rows, ctx);
        const auto right = right_->Eval(rows, ctx);
        if (left_->type() == DataType::BOOL &&
            right_->type() == DataType::BOOL) {
            auto a = left.CopyBoolMasks(rows);
            auto b = right.CopyBoolMasks(rows);
            auto valid =
                OwnedSelection::Intersect(a.valid.view(), b.valid.view());
            auto less =
                OwnedSelection::Subtract(b.truth.view(), a.truth.view());
            auto greater =
                OwnedSelection::Subtract(a.truth.view(), b.truth.view());
            auto truth = OwnedSelection::None(rows.row_count());
            switch (op_) {
                case CompareOp::Equal:
                case CompareOp::NotEqual:
                    truth = OwnedSelection::Union(less.view(), greater.view());
                    if (op_ == CompareOp::Equal)
                        truth = OwnedSelection::Subtract(valid.view(),
                                                         truth.view());
                    break;
                case CompareOp::Less:
                    truth = std::move(less);
                    break;
                case CompareOp::Greater:
                    truth = std::move(greater);
                    break;
                case CompareOp::LessEqual:
                    truth =
                        OwnedSelection::Subtract(valid.view(), greater.view());
                    break;
                case CompareOp::GreaterEqual:
                    truth = OwnedSelection::Subtract(valid.view(), less.view());
                    break;
                default:
                    ThrowInfo(UnexpectedError,
                              "invalid BOOL comparison operation");
            }
            truth.Intersect(valid.view());
            return ReadOnlyVectorView::FromBooleanMasks(
                std::move(truth), std::move(valid), rows);
        }
        BitmapResultBuilder result(rows.row_count());
        kernel_(left, right, rows, op_, result);
        return std::move(result).Finish(rows);
    }

 private:
    template <typename T, typename U>
    static void
    EvaluateComparison(const ReadOnlyVectorView& left,
                       const ReadOnlyVectorView& right,
                       SelectionView rows,
                       CompareOp op,
                       BitmapResultBuilder& result) {
        if constexpr (std::is_arithmetic_v<T> && std::is_arithmetic_v<U>) {
            if (ComparePrimitive<T, U>(
                    left, right, rows, op, result, [op](T a, U b) {
                        return Compare(op, a, b);
                    }))
                return;
        }
        if constexpr (std::is_same_v<T, std::string_view> &&
                      std::is_same_v<U, std::string_view>) {
            if (CompareStrings(left, right, rows, op, result))
                return;
        }
        const auto* left_data = ComparableData<T>(left);
        const auto* right_data = ComparableData<U>(right);
        rows.ForEachSelected([&](size_t row) {
            const auto lhs = ReadSelectedValue<T>(left, row, left_data);
            if (!lhs)
                return;
            const auto rhs = ReadSelectedValue<U>(right, row, right_data);
            if (rhs)
                result.Set<bool>(row, Compare(op, *lhs, *rhs));
        });
    }
    using Kernel = void (*)(const ReadOnlyVectorView&,
                            const ReadOnlyVectorView&,
                            SelectionView,
                            CompareOp,
                            BitmapResultBuilder&);
    Kernel kernel_ = nullptr;
    CompareOp op_;
    ExprPtr left_;
    ExprPtr right_;
};

class RangeExpr final : public Expr {
 public:
    RangeExpr(ExprPtr input,
              ReadOnlyVectorView lower,
              ReadOnlyVectorView upper,
              bool lower_inclusive,
              bool upper_inclusive)
        : Expr(DataType::BOOL),
          input_(std::move(input)),
          lower_(std::move(lower)),
          upper_(std::move(upper)),
          lower_inclusive_(lower_inclusive),
          upper_inclusive_(upper_inclusive) {
        AssertInfo(input_ && lower_.size() == 1 && upper_.size() == 1,
                   "range requires an input and two single-value bounds");
        AssertInfo(input_->type() == lower_.logical_type() &&
                       input_->type() == upper_.logical_type(),
                   "range requires equal logical input and bound types");
        AssertInfo(lower_.IsValid(0) && upper_.IsValid(0),
                   "range bounds must be non-null");
        DispatchComparable(input_->type(), []<typename T>() {});
    }

 protected:
    ReadOnlyVectorView
    EvalImpl(SelectionView rows, EvalCtx& ctx) const override {
        const auto values = input_->Eval(rows, ctx);
        if (input_->type() == DataType::BOOL) {
            auto masks = values.CopyBoolMasks(rows);
            const bool lower = lower_.ValueAt<bool>(0),
                       upper = upper_.ValueAt<bool>(0);
            auto accept = [&](bool value) {
                return (lower_inclusive_ ? value >= lower : value > lower) &&
                       (upper_inclusive_ ? value <= upper : value < upper);
            };
            auto truth =
                accept(false)
                    ? (accept(true)
                           ? OwnedSelection::CopyOf(masks.valid.view())
                           : OwnedSelection::Subtract(masks.valid.view(),
                                                      masks.truth.view()))
                    : (accept(true) ? std::move(masks.truth)
                                    : OwnedSelection::None(rows.row_count()));
            return ReadOnlyVectorView::FromBooleanMasks(
                std::move(truth), std::move(masks.valid), rows);
        }
        BitmapResultBuilder result(rows.row_count());
        DispatchComparable(input_->type(), [&]<typename T>() {
            const auto lower = ReadValue<T>(lower_, 0);
            const auto upper = ReadValue<T>(upper_, 0);
            if constexpr (std::is_arithmetic_v<T>) {
                if (RangePrimitive<T>(
                        values,
                        rows,
                        lower,
                        upper,
                        lower_inclusive_,
                        upper_inclusive_,
                        result,
                        [&](T value) {
                            return (lower_inclusive_ ? value >= lower
                                                     : value > lower) &&
                                   (upper_inclusive_ ? value <= upper
                                                     : value < upper);
                        }))
                    return;
            }
            if constexpr (std::is_same_v<T, std::string_view>) {
                StringInput(values).Match(
                    rows,
                    result,
                    [&](std::string_view value) {
                        return (lower_inclusive_ ? value >= lower
                                                 : value > lower) &&
                               (upper_inclusive_ ? value <= upper
                                                 : value < upper);
                    },
                    true,
                    SelectionKernelCost::StringComparison());
                return;
            }
            const auto* data = ComparableData<T>(values);
            rows.ForEachSelected([&](size_t row) {
                const auto selected = ReadSelectedValue<T>(values, row, data);
                if (!selected)
                    return;
                const auto value = *selected;
                // Direct comparisons retain unordered floating-point
                // semantics. Even an empty/reversed range must preserve NULL.
                result.Set<bool>(
                    row,
                    (lower_inclusive_ ? value >= lower : value > lower) &&
                        (upper_inclusive_ ? value <= upper : value < upper));
            });
        });
        return std::move(result).Finish(rows);
    }

 private:
    ExprPtr input_;
    ReadOnlyVectorView lower_;
    ReadOnlyVectorView upper_;
    bool lower_inclusive_;
    bool upper_inclusive_;
};

class LogicalExpr final : public Expr {
 public:
    LogicalOp
    op() const {
        return op_;
    }
    const std::vector<ExprPtr>&
    inputs() const {
        return inputs_;
    }
    LogicalExpr(LogicalOp op, std::vector<ExprPtr> inputs)
        : Expr(DataType::BOOL), op_(op), inputs_(std::move(inputs)) {
        AssertInfo(op == LogicalOp::And || op == LogicalOp::Or,
                   "invalid scalar logical operation");
        AssertInfo(!inputs_.empty(), "logical expression needs an input");
        for (const auto& input : inputs_) {
            AssertInfo(input && input->type() == DataType::BOOL,
                       "logical expression requires BOOL inputs");
        }
    }

 protected:
    ReadOnlyVectorView
    EvalImpl(SelectionView rows, EvalCtx& ctx) const override {
        const bool is_and = op_ == LogicalOp::And;
        auto active = OwnedSelection::CopyOf(rows);
        auto truth = is_and ? OwnedSelection::CopyOf(rows)
                            : OwnedSelection::None(rows.row_count());
        auto falsity = is_and ? OwnedSelection::None(rows.row_count())
                              : OwnedSelection::CopyOf(rows);
        for (const auto& input : inputs_) {
            if (active.view().IsNone())
                break;
            const auto values = input->Eval(active.view(), ctx);
            auto masks = values.CopyBoolMasks(active.view());
            // Consume the detached validity mask as the child's FALSE set.
            // AND: intersect TRUE, union FALSE. OR: union TRUE, intersect FALSE.
            // UNKNOWN belongs to neither set, so a later decisive result wins.
            masks.valid.Subtract(masks.truth.view());
            if (is_and) {
                truth.Intersect(masks.truth.view());
                falsity.Union(masks.valid.view());
                active.Subtract(masks.valid.view());
            } else {
                truth.Union(masks.truth.view());
                falsity.Intersect(masks.valid.view());
                active.Subtract(masks.truth.view());
            }
        }
        // Only TRUE and FALSE are valid. Reuse the FALSE buffer for validity.
        falsity.Union(truth.view());
        return ReadOnlyVectorView::FromBooleanMasks(
            std::move(truth), std::move(falsity), rows);
    }

 private:
    LogicalOp op_;
    std::vector<ExprPtr> inputs_;
};

class NotExpr final : public Expr {
 public:
    explicit NotExpr(ExprPtr input)
        : Expr(DataType::BOOL), input_(std::move(input)) {
        AssertInfo(input_ && input_->type() == DataType::BOOL,
                   "NOT requires a BOOL input");
    }

 protected:
    ReadOnlyVectorView
    EvalImpl(SelectionView rows, EvalCtx& ctx) const override {
        const auto values = input_->Eval(rows, ctx);
        auto masks = values.CopyBoolMasks(rows);
        auto truth =
            OwnedSelection::Subtract(masks.valid.view(), masks.truth.view());
        return ReadOnlyVectorView::FromBooleanMasks(
            std::move(truth), std::move(masks.valid), rows);
    }

 private:
    ExprPtr input_;
};

class IsNullExpr final : public Expr {
 public:
    IsNullExpr(ExprPtr input, bool negate)
        : Expr(DataType::BOOL), input_(std::move(input)), negate_(negate) {
        AssertInfo(input_, "IS NULL requires an input");
    }

 protected:
    ReadOnlyVectorView
    EvalImpl(SelectionView rows, EvalCtx& ctx) const override {
        auto valid = OwnedSelection::None(rows.row_count());
        if (const auto* ref = dynamic_cast<const InputRefExpr*>(input_.get())) {
            valid = ref->GetValidity(rows, ctx).valid;
        } else {
            valid = input_->Eval(rows, ctx).CopyValidity();
        }
        auto truth = negate_ ? OwnedSelection::Intersect(rows, valid.view())
                             : OwnedSelection::Subtract(rows, valid.view());
        return ReadOnlyVectorView::FromBooleanMasks(
            std::move(truth), OwnedSelection::CopyOf(rows), rows);
    }

 private:
    ExprPtr input_;
    bool negate_;
};

}  // namespace

ReadOnlyVectorView
Expr::Eval(SelectionView rows, EvalCtx& ctx) const {
    AssertInfo(rows.row_count() == ctx.input().size(),
               "scalar expression selection length mismatch");
    if (rows.IsNone()) {
        return ReadOnlyVectorView::Nulls(type(), rows);
    }
    ctx.Checkpoint();
    auto result = EvalImpl(rows, ctx);
    AssertInfo(
        result.size() == rows.row_count() && result.logical_type() == type(),
        "scalar expression returned wrong shape/type");
    // Restrict also rejects incomplete coverage; outside rows stay undefined.
    return result.Restrict(rows);
}

ExprPtr
MakeLiteral(const VectorPtr& single_value) {
    return std::make_shared<LiteralExpr>(
        ReadOnlyVectorView::CopyOf(single_value));
}

ExprPtr
MakeComparison(CompareOp op, ExprPtr left, ExprPtr right) {
    return std::make_shared<ComparisonExpr>(
        op, std::move(left), std::move(right));
}

ExprPtr
MakeRange(ExprPtr input,
          const VectorPtr& lower,
          const VectorPtr& upper,
          bool lower_inclusive,
          bool upper_inclusive) {
    AssertInfo(lower && upper, "range requires bound vectors");
    return std::make_shared<RangeExpr>(std::move(input),
                                       ReadOnlyVectorView::CopyOf(lower),
                                       ReadOnlyVectorView::CopyOf(upper),
                                       lower_inclusive,
                                       upper_inclusive);
}

ExprPtr
MakeLogical(LogicalOp op, std::vector<ExprPtr> inputs) {
    return std::make_shared<LogicalExpr>(op, std::move(inputs));
}

ExprPtr
MakeNot(ExprPtr input) {
    return std::make_shared<NotExpr>(std::move(input));
}

ExprPtr
AdjustLogicalExpression(
    ExprPtr root,
    const ClassifyExpression& classify,
    const std::function<ExprPtr(ExprPtr)>& resolve,
    const std::function<std::vector<ExprPtr>(LogicalOp, std::vector<ExprPtr>)>&
        physical) {
    struct Adjusted {
        ExprPtr expr;
        ReorderProperty property;
    };
    std::function<Adjusted(ExprPtr)> visit = [&](ExprPtr node) -> Adjusted {
        auto logical = dynamic_cast<const LogicalExpr*>(node.get());
        if (!logical)
            return {node, classify(node.get())};
        std::vector<Adjusted> inputs;
        std::function<void(const ExprPtr&)> append = [&](const ExprPtr& child) {
            auto nested = dynamic_cast<const LogicalExpr*>(child.get());
            if (nested && nested->op() == logical->op()) {
                for (const auto& grandchild : nested->inputs())
                    append(grandchild);
            } else {
                inputs.push_back(visit(child));
            }
        };
        for (const auto& child : logical->inputs()) append(child);
        bool safe = true, heavy = false;
        for (size_t begin = 0; begin < inputs.size();) {
            if (!inputs[begin].property.safe) {
                safe = false;
                ++begin;
                continue;
            }
            size_t end = begin;
            while (end < inputs.size() && inputs[end].property.safe) ++end;
            std::stable_sort(inputs.begin() + begin,
                             inputs.begin() + end,
                             [](const auto& left, const auto& right) {
                                 return left.property.tier <
                                        right.property.tier;
                             });
            begin = end;
        }
        std::vector<ExprPtr> children;
        for (auto& input : inputs) {
            heavy |= input.property.tier >= ReorderTier::StringLike;
            children.push_back(std::move(input.expr));
        }
        if (physical)
            children = physical(logical->op(), std::move(children));
        auto fused = FuseAdjacentStringMatches(
            logical->op(), std::move(children), resolve);
        fused = FuseAdjacentGeometryPredicates(
            logical->op(), std::move(fused), resolve);
        auto result = fused.size() == 1
                          ? std::move(fused.front())
                          : MakeLogical(logical->op(), std::move(fused));
        return {std::move(result),
                {heavy ? ReorderTier::HeavyConjunction
                       : ReorderTier::LightConjunction,
                 safe}};
    };
    return visit(std::move(root)).expr;
}

ExprPtr
MakeIsNull(ExprPtr input, bool negate) {
    return std::make_shared<IsNullExpr>(std::move(input), negate);
}

}  // namespace milvus::exec::scalar
