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

#include "exec/expression/ScalarBatchKernels.h"
#include "exec/expression/ScalarArithmetic.h"
#include "exec/expression/ScalarObjectKernels.h"

#include <bit>
#include <cmath>
#include "exec/expression/ScalarCallExpr.h"
#include <limits>

namespace milvus::exec::scalar {
namespace {
int64_t
TruncateInteger(double value) {
    if (!std::isfinite(value) || value < -0x1p63 || value >= 0x1p63)
        ThrowInfo(ExprInvalid, "arithmetic operand is outside INT64 range");
    return static_cast<int64_t>(value);
}

template <typename T>
T
Arithmetic(T left, T right, proto::plan::ArithOpType operation) {
    using namespace proto::plan;
    if constexpr (std::is_integral_v<T>) {
        // Match the legacy SIMD add/sub/multiply low 64 bits without signed
        // overflow UB in the scalar C++ evaluator.
        const auto a = static_cast<uint64_t>(left);
        const auto b = static_cast<uint64_t>(right);
        switch (operation) {
            case Add:
                return std::bit_cast<int64_t>(a + b);
            case Sub:
                return std::bit_cast<int64_t>(a - b);
            case Mul:
                return std::bit_cast<int64_t>(a * b);
            case BitAnd:
                return left & right;
            case BitOr:
                return left | right;
            case BitXor:
                return left ^ right;
            case Shl:
                return std::bit_cast<int64_t>(a << right);
            case Shr:
                return left >> right;
            case Div:
            case Mod:
                // No representable quotient exists; never execute C++ UB.
                if (left == std::numeric_limits<int64_t>::min() && right == -1)
                    ThrowInfo(ExprInvalid, "integer division overflow");
                return operation == Div ? left / right : left % right;
            default:
                break;
        }
    } else {
        switch (operation) {
            case Add:
                return left + right;
            case Sub:
                return left - right;
            case Mul:
                return left * right;
            case Div:
                return left / right;
            case Mod:
                return std::fmod(left, right);
            default:
                break;
        }
    }
    ThrowInfo(UnexpectedError, "invalid compiled arithmetic operation");
}

template <typename T>
T
NumericValue(const ReadOnlyVectorView& input, size_t row) {
    switch (input.logical_type()) {
        case DataType::INT8:
            return input.ValueAt<int8_t>(row);
        case DataType::INT16:
            return input.ValueAt<int16_t>(row);
        case DataType::INT32:
            return input.ValueAt<int32_t>(row);
        case DataType::INT64:
            return input.ValueAt<int64_t>(row);
        case DataType::FLOAT:
            if constexpr (std::is_floating_point_v<T>)
                return input.ValueAt<float>(row);
            break;
        case DataType::DOUBLE:
            if constexpr (std::is_floating_point_v<T>)
                return input.ValueAt<double>(row);
            break;
        default:
            break;
    }
    ThrowInfo(UnexpectedError, "invalid compiled numeric input type");
}

template <typename T>
bool
ProjectArithmetic(const ReadOnlyVectorView& left,
                  const ReadOnlyVectorView& right,
                  SelectionView rows,
                  proto::plan::ArithOpType operation,
                  VectorResultBuilder& output,
                  EvalCtx& ctx) {
    // The common promoted types need no per-row type/encoding dispatch.
    // Mixed-width inputs retain the checked generic path below.
    constexpr auto type =
        std::is_integral_v<T> ? DataType::INT64 : DataType::DOUBLE;
    if (left.logical_type() != type || right.logical_type() != type)
        return false;
    const auto lhs = left.DecodePrimitive(), rhs = right.DecodePrimitive();
    if (!lhs || !rhs)
        return false;
    size_t l = 0, r = 0;
    while (l < lhs->spans.size() && r < rhs->spans.size()) {
        ctx.Checkpoint();
        const auto& a = lhs->spans[l];
        const auto& b = rhs->spans[r];
        const auto begin = std::max(a.begin, b.begin);
        const auto end = std::min(a.begin + a.count, b.begin + b.count);
        if (begin < end) {
            auto active = OwnedSelection::Intersect(
                rows.Slice(begin, end - begin),
                a.valid.Slice(begin - a.begin, end - begin));
            active.Intersect(b.valid.Slice(begin - b.begin, end - begin));
            const auto* av = static_cast<const T*>(a.data);
            const auto* bv = static_cast<const T*>(b.data);
            auto evaluate = [&](auto op) {
                size_t processed = 0;
                auto value = [&](size_t i) -> std::optional<T> {
                    // Bound cancellation latency even for a large dense span.
                    if ((processed++ & 1023) == 0)
                        ctx.Checkpoint();
                    const auto x = av[a.constant ? 0 : begin - a.begin + i];
                    const auto y = bv[b.constant ? 0 : begin - b.begin + i];
                    if constexpr (decltype(op)::value == proto::plan::Div)
                        if (y == 0)
                            ThrowInfo(ExprInvalid, "cannot divide by zero");
                    return Arithmetic(x, y, decltype(op)::value);
                };
                // Strict selection: never read NULL or unrequested payloads,
                // including division operands that would raise an error.
                if (ChooseSelectionKernel(
                        active.view(),
                        false,
                        SelectionKernelCost::Arithmetic(false)) ==
                    SelectionKernelMode::Sparse)
                    output.EvaluateSelected<T>(begin, active.view(), value);
                else
                    active.view().ForEachSelectedRange(
                        [&](size_t offset, size_t count) {
                            output.EvaluateRange<T>(
                                begin + offset, count, [&](size_t i) {
                                    return value(offset + i);
                                });
                        });
            };
            switch (operation) {
                case proto::plan::Add:
                    evaluate(std::integral_constant<proto::plan::ArithOpType,
                                                    proto::plan::Add>{});
                    break;
                case proto::plan::Sub:
                    evaluate(std::integral_constant<proto::plan::ArithOpType,
                                                    proto::plan::Sub>{});
                    break;
                case proto::plan::Mul:
                    evaluate(std::integral_constant<proto::plan::ArithOpType,
                                                    proto::plan::Mul>{});
                    break;
                case proto::plan::Div:
                    evaluate(std::integral_constant<proto::plan::ArithOpType,
                                                    proto::plan::Div>{});
                    break;
                default:
                    ThrowInfo(UnexpectedError,
                              "invalid compiled arithmetic operation");
            }
        }
        if (a.begin + a.count <= b.begin + b.count)
            ++l;
        if (b.begin + b.count <= a.begin + a.count)
            ++r;
    }
    ctx.Checkpoint();
    return true;
}

template <typename T>
class ArithmeticExpr final : public Expr {
 public:
    ArithmeticExpr(ExprPtr left,
                   ExprPtr right,
                   proto::plan::ArithOpType operation)
        : Expr(std::is_integral_v<T> ? DataType::INT64 : DataType::DOUBLE),
          left_(std::move(left)),
          right_(std::move(right)),
          operation_(operation) {
    }

 protected:
    ReadOnlyVectorView
    EvalImpl(SelectionView rows, EvalCtx& ctx) const override {
        const auto left = left_->Eval(rows, ctx);
        const auto right = right_->Eval(rows, ctx);
        VectorResultBuilder result(type(), rows.row_count());
        if (ProjectArithmetic<T>(left, right, rows, operation_, result, ctx))
            return std::move(result).Finish(rows);
        rows.ForEachSelected([&](size_t row) {
            ctx.Checkpoint();
            if (!left.IsValid(row) || !right.IsValid(row)) {
                result.SetNull(row);
                return;
            }
            const auto a = NumericValue<T>(left, row);
            const auto b = NumericValue<T>(right, row);
            if (operation_ == proto::plan::Div && b == 0)
                ThrowInfo(ExprInvalid, "cannot divide by zero");
            result.Set<T>(row, Arithmetic(a, b, operation_));
        });
        return std::move(result).Finish(rows);
    }

 private:
    ExprPtr left_, right_;
    const proto::plan::ArithOpType operation_;
};

template <typename T>
bool
Compare(T value, T bound, CompareOp operation) {
    switch (operation) {
        case CompareOp::Equal:
            return value == bound;
        case CompareOp::NotEqual:
            return value != bound;
        case CompareOp::Less:
            return value < bound;
        case CompareOp::LessEqual:
            return value <= bound;
        case CompareOp::Greater:
            return value > bound;
        case CompareOp::GreaterEqual:
            return value >= bound;
        default:
            ThrowInfo(UnexpectedError, "invalid arithmetic comparison");
    }
}

template <proto::plan::ArithOpType Op, typename T>
T
ArithmeticValue(T left, T right) {
    using namespace proto::plan;
    if constexpr (std::is_integral_v<T>) {
        const auto a = static_cast<uint64_t>(left),
                   b = static_cast<uint64_t>(right);
        if constexpr (Op == Add)
            return std::bit_cast<int64_t>(a + b);
        else if constexpr (Op == Sub)
            return std::bit_cast<int64_t>(a - b);
        else if constexpr (Op == Mul)
            return std::bit_cast<int64_t>(a * b);
        else if constexpr (Op == BitAnd)
            return left & right;
        else if constexpr (Op == BitOr)
            return left | right;
        else if constexpr (Op == BitXor)
            return left ^ right;
        else if constexpr (Op == Shl)
            return std::bit_cast<int64_t>(a << right);
        else if constexpr (Op == Shr)
            return left >> right;
        else {
            if (left == std::numeric_limits<int64_t>::min() && right == -1)
                ThrowInfo(ExprInvalid, "integer division overflow");
            if constexpr (Op == Div)
                return left / right;
            else
                return left % right;
        }
    } else {
        if constexpr (Op == Add)
            return left + right;
        else if constexpr (Op == Sub)
            return left - right;
        else if constexpr (Op == Mul)
            return left * right;
        else if constexpr (Op == Div)
            return left / right;
        else if constexpr (Op == Mod)
            return std::fmod(left, right);
        else
            ThrowInfo(UnexpectedError, "invalid floating arithmetic kernel");
    }
}

template <typename Fn>
decltype(auto)
DispatchArithmetic(proto::plan::ArithOpType op, const Fn& fn) {
    using namespace proto::plan;
    switch (op) {
        case Add:
            return fn.template operator()<Add>();
        case Sub:
            return fn.template operator()<Sub>();
        case Mul:
            return fn.template operator()<Mul>();
        case Div:
            return fn.template operator()<Div>();
        case Mod:
            return fn.template operator()<Mod>();
        case BitAnd:
            return fn.template operator()<BitAnd>();
        case BitOr:
            return fn.template operator()<BitOr>();
        case BitXor:
            return fn.template operator()<BitXor>();
        case Shl:
            return fn.template operator()<Shl>();
        case Shr:
            return fn.template operator()<Shr>();
        default:
            ThrowInfo(UnexpectedError, "invalid compiled arithmetic operation");
    }
}

template <typename T>
class ArithmeticCompareExpr final : public Expr {
    using Result = std::conditional_t<std::is_integral_v<T>, int64_t, T>;

 public:
    ArithmeticCompareExpr(ExprPtr input,
                          proto::plan::ArithOpType operation,
                          CompareOp comparison,
                          Result operand,
                          Result bound,
                          bool truncate_floating)
        : Expr(DataType::BOOL),
          input_(std::move(input)),
          operation_(operation),
          comparison_(comparison),
          operand_(operand),
          bound_(bound),
          truncate_floating_(truncate_floating) {
    }

 protected:
    ReadOnlyVectorView
    EvalImpl(SelectionView rows, EvalCtx& ctx) const override {
        const auto input = input_->Eval(rows, ctx);
        if constexpr (std::is_floating_point_v<T>) {
            if (truncate_floating_) {
                const auto right = TruncateInteger(operand_);
                if (operation_ == proto::plan::Mod && right == 0)
                    ThrowInfo(ExprInvalid, "modulus divisor truncates to zero");
                if ((operation_ == proto::plan::Shl ||
                     operation_ == proto::plan::Shr) &&
                    (right < 0 || right >= 64))
                    ThrowInfo(ExprInvalid,
                              "shift amount must be in range [0, 64)");
                BitmapResultBuilder result(rows.row_count());
                DispatchArithmetic(
                    operation_, [&]<proto::plan::ArithOpType Op>() {
                        DispatchComparison(
                            comparison_, [&]<bitset::CompareOpType Cmp>() {
                                EvaluatePrimitive<T>(
                                    input, rows, result, [&](T raw) {
                                        ctx.Checkpoint();
                                        const auto value = ArithmeticValue<Op>(
                                            TruncateInteger(raw), right);
                                        return bitset::CompareOperator<
                                            Cmp>::compare(static_cast<T>(value),
                                                          bound_);
                                    });
                            });
                    });
                return std::move(result).Finish(rows);
            }
        }
        // Validate on execution, not compilation: a short-circuited branch
        // must not execute an invalid arithmetic operation.
        if ((operation_ == proto::plan::Div ||
             operation_ == proto::plan::Mod) &&
            operand_ == 0)
            ThrowInfo(ExprInvalid,
                      "division or modulus by zero in arithmetic expression");
        if constexpr (std::is_integral_v<T>) {
            if ((operation_ == proto::plan::Shl ||
                 operation_ == proto::plan::Shr) &&
                (operand_ < 0 || operand_ >= 64))
                ThrowInfo(ExprInvalid, "shift amount must be in range [0, 64)");
        }
        BitmapResultBuilder result(rows.row_count());
        DispatchArithmetic(operation_, [&]<proto::plan::ArithOpType Op>() {
            DispatchComparison(comparison_, [&]<bitset::CompareOpType Cmp>() {
                constexpr bool basic = Op == proto::plan::Add ||
                                       Op == proto::plan::Sub ||
                                       Op == proto::plan::Mul;
                constexpr bool total = std::is_integral_v<T> &&
                                       Op != proto::plan::Div &&
                                       Op != proto::plan::Mod;
                const auto test = [&](T raw) {
                    return bitset::CompareOperator<Cmp>::compare(
                        ArithmeticValue<Op>(static_cast<Result>(raw), operand_),
                        bound_);
                };
                EvaluatePrimitiveKernel<T>(
                    input,
                    rows,
                    result,
                    total || (std::is_floating_point_v<T> && basic),
                    SelectionKernelCost::Arithmetic(basic),
                    [&](size_t begin,
                        size_t count,
                        const T* data,
                        bool constant) {
                        if (constant)
                            result.SetRange(begin, count, test(*data));
                        else if constexpr (basic) {
                            constexpr auto arithmetic =
                                Op == proto::plan::Add
                                    ? bitset::ArithOpType::Add
                                : Op == proto::plan::Sub
                                    ? bitset::ArithOpType::Sub
                                    : bitset::ArithOpType::Mul;
                            result.ArithmeticCompare<T, arithmetic, Cmp>(
                                begin, count, data, operand_, bound_);
                        } else
                            result.EvaluateRange(begin, count, [&](size_t i) {
                                return test(data[i]);
                            });
                    },
                    test);
            });
        });
        return std::move(result).Finish(rows);
    }

 private:
    ExprPtr input_;
    proto::plan::ArithOpType operation_;
    CompareOp comparison_;
    Result operand_;
    Result bound_;
    bool truncate_floating_;
};
}  // namespace

ExprPtr
MakeArithmetic(ExprPtr left,
               ExprPtr right,
               proto::plan::ArithOpType operation) {
    AssertInfo(left && right && IsNumericDataType(left->type()) &&
                   IsNumericDataType(right->type()) &&
                   left->type() != DataType::BOOL &&
                   right->type() != DataType::BOOL,
               "arithmetic requires numeric inputs");
    AssertInfo(operation == proto::plan::Add || operation == proto::plan::Sub ||
                   operation == proto::plan::Mul ||
                   operation == proto::plan::Div,
               "unsupported scalar arithmetic operation");
    if (IsIntegerDataType(left->type()) && IsIntegerDataType(right->type()))
        return std::make_shared<ArithmeticExpr<int64_t>>(
            std::move(left), std::move(right), operation);
    return std::make_shared<ArithmeticExpr<double>>(
        std::move(left), std::move(right), operation);
}

ExprPtr
MakeArithmeticCompare(ExprPtr input,
                      proto::plan::ArithOpType operation,
                      CompareOp comparison,
                      const VectorPtr& operand,
                      const VectorPtr& bound,
                      bool truncate_floating) {
    AssertInfo(input && operand && bound,
               "arithmetic requires input and constants");
    const auto right = ReadOnlyVectorView::CopyOf(operand);
    const auto limit = ReadOnlyVectorView::CopyOf(bound);
    AssertInfo(right.size() == 1 && limit.size() == 1 && right.IsValid(0) &&
                   limit.IsValid(0),
               "arithmetic requires non-null scalar constants");
    AssertInfo(
        comparison >= CompareOp::Equal && comparison <= CompareOp::GreaterEqual,
        "invalid compiled arithmetic comparison");
    auto make = [&]<typename T>() -> ExprPtr {
        using Result = std::conditional_t<std::is_integral_v<T>, int64_t, T>;
        const auto type =
            std::is_integral_v<T> ? DataType::INT64 : input->type();
        AssertInfo(right.logical_type() == type && limit.logical_type() == type,
                   "arithmetic constant type mismatch");
        const bool basic =
            operation == proto::plan::Add || operation == proto::plan::Sub ||
            operation == proto::plan::Mul || operation == proto::plan::Div;
        const bool integer =
            operation == proto::plan::Mod || operation == proto::plan::BitAnd ||
            operation == proto::plan::BitOr ||
            operation == proto::plan::BitXor || operation == proto::plan::Shl ||
            operation == proto::plan::Shr;
        AssertInfo(
            (!truncate_floating || (std::is_floating_point_v<T> && integer)) &&
                (basic || operation == proto::plan::Mod ||
                 ((std::is_integral_v<T> || truncate_floating) && integer)),
            "invalid compiled arithmetic operation/type");
        return std::make_shared<ArithmeticCompareExpr<T>>(
            input,
            operation,
            comparison,
            right.ValueAt<Result>(0),
            limit.ValueAt<Result>(0),
            truncate_floating);
    };
    switch (input->type()) {
        case DataType::INT8:
            return make.template operator()<int8_t>();
        case DataType::INT16:
            return make.template operator()<int16_t>();
        case DataType::INT32:
            return make.template operator()<int32_t>();
        case DataType::INT64:
            return make.template operator()<int64_t>();
        case DataType::FLOAT:
            return make.template operator()<float>();
        case DataType::DOUBLE:
            return make.template operator()<double>();
        default:
            ThrowInfo(Unsupported, "arithmetic input type is not migrated");
    }
}
bool
MatchJsonArithmeticNumber(double value,
                          proto::plan::ArithOpType operation,
                          CompareOp comparison,
                          const proto::plan::GenericValue& operand,
                          const proto::plan::GenericValue& bound) {
    const bool integral = bound.has_int64_val();
    return Compare(Arithmetic<double>(value,
                                      integral ? double(operand.int64_val())
                                               : operand.float_val(),
                                      operation),
                   integral ? double(bound.int64_val()) : bound.float_val(),
                   comparison);
}

bool
MatchJsonArithmeticNumber(int64_t value,
                          proto::plan::ArithOpType operation,
                          CompareOp comparison,
                          const proto::plan::GenericValue& operand,
                          const proto::plan::GenericValue& bound) {
    if (!bound.has_int64_val())
        return MatchJsonArithmeticNumber(
            double(value), operation, comparison, operand, bound);
    return Compare(Arithmetic<int64_t>(value, operand.int64_val(), operation),
                   bound.int64_val(),
                   comparison);
}

ExprPtr
MakeJsonArithmeticCompare(ExprPtr input,
                          std::vector<std::string> path,
                          proto::plan::ArithOpType operation,
                          CompareOp comparison,
                          proto::plan::GenericValue operand,
                          proto::plan::GenericValue bound) {
    return MakeCall(
        DataType::BOOL,
        {std::move(input)},
        [pointer = Json::pointer(std::move(path)),
         operation,
         comparison,
         operand = std::move(operand),
         bound = std::move(bound)](
            const auto& inputs, SelectionView rows, EvalCtx& ctx) {
            const bool integral = bound.has_int64_val();
            const double divisor =
                integral ? static_cast<double>(operand.int64_val())
                         : operand.float_val();
            if ((operation == proto::plan::Div ||
                 operation == proto::plan::Mod) &&
                divisor == 0)
                ThrowInfo(ExprInvalid,
                          "division or modulus by zero in JSON field "
                          "arithmetic expression");
            const bool bitwise = operation >= proto::plan::BitAnd &&
                                 operation <= proto::plan::Shr;
            auto integer = [](double value) -> int64_t {
                if (!std::isfinite(value) || value < -0x1p63 || value >= 0x1p63)
                    ThrowInfo(ExprInvalid,
                              "JSON bitwise operand is outside INT64 range");
                return static_cast<int64_t>(value);
            };
            const int64_t bit_operand =
                bitwise ? (integral ? operand.int64_val() : integer(divisor))
                        : 0;
            if ((operation == proto::plan::Shl ||
                 operation == proto::plan::Shr) &&
                (bit_operand < 0 || bit_operand >= 64))
                ThrowInfo(ExprInvalid, "shift amount must be in range [0, 64)");
            BitmapResultBuilder output(rows.row_count());
            const ObjectInput<Json> objects(inputs[0]);
            DispatchArithmetic(operation, [&]<proto::plan::ArithOpType Op>() {
                DispatchComparison(
                    comparison, [&]<bitset::CompareOpType Cmp>() {
                        objects.MatchOptional(
                            rows,
                            output,
                            [&](const Json& json) -> std::optional<bool> {
                                ctx.Checkpoint();
                                if constexpr (Op >= proto::plan::BitAnd &&
                                              Op <= proto::plan::Shr) {
                                    auto number = json.at_numeric(pointer);
                                    if (number.error())
                                        return std::nullopt;
                                    const auto value = number.value();
                                    const int64_t left =
                                        integral && value.is_int64()
                                            ? value.get_int64()
                                            : integer(
                                                  value.is_int64()
                                                      ? static_cast<double>(
                                                            value.get_int64())
                                                  : value.is_uint64()
                                                      ? static_cast<double>(
                                                            value.get_uint64())
                                                      : value.get_double());
                                    const auto calculated =
                                        ArithmeticValue<Op>(left, bit_operand);
                                    return integral
                                               ? bitset::CompareOperator<Cmp>::
                                                     compare(calculated,
                                                             bound.int64_val())
                                               : bitset::CompareOperator<Cmp>::
                                                     compare(
                                                         static_cast<double>(
                                                             calculated),
                                                         bound.float_val());
                                } else {
                                    auto match_double = [&](double value) {
                                        return bitset::CompareOperator<Cmp>::
                                            compare(
                                                ArithmeticValue<Op>(
                                                    value,
                                                    integral
                                                        ? double(
                                                              operand
                                                                  .int64_val())
                                                        : operand.float_val()),
                                                integral
                                                    ? double(bound.int64_val())
                                                    : bound.float_val());
                                    };
                                    if (integral) {
                                        auto number = json.at_numeric(pointer);
                                        if (number.error())
                                            return std::nullopt;
                                        const auto value = number.value();
                                        if (value.is_int64())
                                            return bitset::
                                                CompareOperator<Cmp>::compare(
                                                    ArithmeticValue<Op>(
                                                        value.get_int64(),
                                                        operand.int64_val()),
                                                    bound.int64_val());
                                        return match_double(
                                            value.is_uint64()
                                                ? static_cast<double>(
                                                      value.get_uint64())
                                                : value.get_double());
                                    }
                                    auto value =
                                        json.template at<double>(pointer);
                                    if (value.error())
                                        return std::nullopt;
                                    return match_double(value.value());
                                }
                            });
                    });
            });
            return std::move(output).Finish(rows);
        });
}
}  // namespace milvus::exec::scalar
