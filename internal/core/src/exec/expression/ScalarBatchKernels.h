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

#include "common/VectorAccess.h"
#include "exec/expression/ScalarExpr.h"
#include "exec/expression/KernelSelection.h"

namespace milvus::exec::scalar {

inline bitset::CompareOpType
BitmapCompareOp(CompareOp op) {
    switch (op) {
        case CompareOp::Equal:
            return bitset::CompareOpType::EQ;
        case CompareOp::NotEqual:
            return bitset::CompareOpType::NE;
        case CompareOp::Less:
            return bitset::CompareOpType::LT;
        case CompareOp::LessEqual:
            return bitset::CompareOpType::LE;
        case CompareOp::Greater:
            return bitset::CompareOpType::GT;
        case CompareOp::GreaterEqual:
            return bitset::CompareOpType::GE;
        default:
            ThrowInfo(UnexpectedError, "non-numeric bulk comparison");
    }
}

inline bitset::CompareOpType
ReverseComparison(bitset::CompareOpType op) {
    using Op = bitset::CompareOpType;
    switch (op) {
        case Op::LT:
            return Op::GT;
        case Op::LE:
            return Op::GE;
        case Op::GT:
            return Op::LT;
        case Op::GE:
            return Op::LE;
        default:
            return op;
    }
}

// Bind comparison semantics once, before entering a row kernel.
template <typename Fn>
decltype(auto)
DispatchComparison(CompareOp op, Fn&& fn) {
    switch (op) {
        case CompareOp::Equal:
            return fn.template operator()<bitset::CompareOpType::EQ>();
        case CompareOp::NotEqual:
            return fn.template operator()<bitset::CompareOpType::NE>();
        case CompareOp::Less:
            return fn.template operator()<bitset::CompareOpType::LT>();
        case CompareOp::LessEqual:
            return fn.template operator()<bitset::CompareOpType::LE>();
        case CompareOp::Greater:
            return fn.template operator()<bitset::CompareOpType::GT>();
        case CompareOp::GreaterEqual:
            return fn.template operator()<bitset::CompareOpType::GE>();
        default:
            ThrowInfo(UnexpectedError, "invalid compiled batch comparison");
    }
}

// One shared dispatcher for typed predicate kernels. Input readability and
// audited operator eligibility must both be established by the caller.
template <typename Range, typename Test>
void
EvaluateSelection(size_t begin,
                  SelectionView active,
                  bool can_evaluate_all,
                  SelectionKernelCost cost,
                  BitmapResultBuilder& output,
                  const Range& range,
                  const Test& test,
                  SelectionKernelPreference preference =
                      SelectionKernelPreference::Adaptive) {
    if (active.IsNone())
        return;
    const auto mode =
        ChooseSelectionKernel(active, can_evaluate_all, cost, preference);
    if (mode == SelectionKernelMode::Dense) {
        range(0, active.row_count());
        if (!active.IsAll())
            output.ApplyValidity(begin, active);
    } else if (mode == SelectionKernelMode::Sparse) {
        output.EvaluateSelected(begin, active, test);
    } else {
        active.ForEachSelectedRange(range);
    }
}

template <typename Test>
void
EvaluateOptionalSelection(size_t begin,
                          SelectionView active,
                          SelectionKernelCost cost,
                          BitmapResultBuilder& output,
                          const Test& test,
                          SelectionKernelPreference preference =
                              SelectionKernelPreference::Adaptive) {
    if (ChooseSelectionKernel(active, false, cost, preference) ==
        SelectionKernelMode::Sparse)
        output.EvaluateOptionalSelected(begin, active, test);
    else
        active.ForEachSelectedRange([&](size_t offset, size_t count) {
            output.EvaluateOptionalRange(begin + offset, count, [&](size_t i) {
                return test(offset + i);
            });
        });
}

// Numeric membership is total on readable payloads. Sparse/partial inputs
// choose between packed probes and contiguous batch membership kernels.
template <typename T, typename Terms, typename Test>
bool
MatchPrimitive(const ReadOnlyVectorView& input,
               SelectionView rows,
               BitmapResultBuilder& output,
               const Terms& terms,
               const Test& test,
               SelectionKernelCost cost,
               SelectionKernelPreference preference =
                   SelectionKernelPreference::Adaptive) {
    static_assert(std::is_arithmetic_v<T> && !std::is_same_v<T, bool>);
    const auto decoded = input.DecodePrimitive();
    if (!decoded)
        return false;
    for (const auto& span : decoded->spans) {
        auto requested = rows.Slice(span.begin, span.count);
        auto active = OwnedSelection::Intersect(requested, span.valid);
        const auto* data = static_cast<const T*>(span.data);
        auto evaluate = [&](size_t offset, size_t count) {
            if (span.constant)
                output.SetRange(span.begin + offset, count, test(*data));
            else
                output.MatchTerms(
                    span.begin + offset, count, data + offset, terms);
        };
        EvaluateSelection(
            span.begin,
            active.view(),
            span.CanEvaluateAll(0, span.count),
            cost,
            output,
            evaluate,
            [&](size_t i) { return test(data[span.constant ? 0 : i]); },
            preference);
    }
    return true;
}

template <typename T, typename Test>
void
EvaluatePrimitive(const ReadOnlyVectorView& input,
                  SelectionView rows,
                  BitmapResultBuilder& output,
                  const Test& test) {
    if (const auto decoded = input.DecodePrimitive()) {
        for (const auto& span : decoded->spans) {
            auto active = OwnedSelection::Intersect(
                rows.Slice(span.begin, span.count), span.valid);
            const auto* data = static_cast<const T*>(span.data);
            auto probe = [&](size_t i) {
                return test(data[span.constant ? 0 : i]);
            };
            EvaluateSelection(
                span.begin,
                active.view(),
                false,
                SelectionKernelCost::SelectedCallback(),
                output,
                [&](size_t offset, size_t count) {
                    output.EvaluateRange(
                        span.begin + offset, count, [&](size_t i) {
                            return probe(offset + i);
                        });
                },
                probe);
        }
    } else {
        rows.ForEachSelected([&](size_t row) {
            if (const auto value = input.TryValueAt<T>(row))
                output.Set<bool>(row, test(*value));
        });
    }
}

template <typename T, typename Range, typename Test>
void
EvaluatePrimitiveKernel(const ReadOnlyVectorView& input,
                        SelectionView rows,
                        BitmapResultBuilder& output,
                        bool total,
                        SelectionKernelCost cost,
                        const Range& range,
                        const Test& test,
                        SelectionKernelPreference preference =
                            SelectionKernelPreference::Adaptive) {
    if (const auto decoded = input.DecodePrimitive()) {
        for (const auto& span : decoded->spans) {
            auto active = OwnedSelection::Intersect(
                rows.Slice(span.begin, span.count), span.valid);
            const auto* data = static_cast<const T*>(span.data);
            EvaluateSelection(
                span.begin,
                active.view(),
                total && span.CanEvaluateAll(0, span.count),
                cost,
                output,
                [&](size_t offset, size_t count) {
                    range(span.begin + offset,
                          count,
                          data + (span.constant ? 0 : offset),
                          span.constant);
                },
                [&](size_t i) { return test(data[span.constant ? 0 : i]); },
                preference);
        }
    } else {
        rows.ForEachSelected([&](size_t row) {
            if (const auto value = input.TryValueAt<T>(row))
                output.Set<bool>(row, test(*value));
        });
    }
}

// Checked numeric projections (for example calendar arithmetic) have no
// dense path: even physically readable unselected values may throw.
template <typename T, typename Result, typename Test>
void
ProjectPrimitive(const ReadOnlyVectorView& input,
                 SelectionView rows,
                 VectorResultBuilder& output,
                 const Test& test) {
    if (const auto decoded = input.DecodePrimitive()) {
        for (const auto& span : decoded->spans) {
            auto active = OwnedSelection::Intersect(
                rows.Slice(span.begin, span.count), span.valid);
            const auto* data = static_cast<const T*>(span.data);
            auto probe = [&](size_t i) -> std::optional<Result> {
                return test(data[span.constant ? 0 : i]);
            };
            if (ChooseSelectionKernel(
                    active.view(),
                    false,
                    SelectionKernelCost::SelectedCallback()) ==
                SelectionKernelMode::Sparse)
                output.EvaluateSelected<Result>(
                    span.begin, active.view(), probe);
            else
                active.view().ForEachSelectedRange(
                    [&](size_t offset, size_t count) {
                        output.EvaluateRange<Result>(
                            span.begin + offset, count, [&](size_t i) {
                                return probe(offset + i);
                            });
                    });
        }
    } else {
        rows.ForEachSelected([&](size_t row) {
            if (auto value = input.TryValueAt<T>(row)) {
                std::optional<Result> result = test(*value);
                if (result)
                    output.Set<Result>(row, std::move(*result));
            }
        });
    }
}

// Keep the decoded owner alive throughout all callbacks. These callbacks see
// only requested, defined and valid rows; no NULL payload is speculatively read.
template <typename T, typename Fn>
void
ForEachPrimitiveValue(const ReadOnlyVectorView& input,
                      SelectionView rows,
                      Fn&& fn) {
    if (auto decoded = input.DecodePrimitive()) {
        for (const auto& span : decoded->spans) {
            auto selected = OwnedSelection::Intersect(
                rows.Slice(span.begin, span.count), span.valid);
            const auto* data = static_cast<const T*>(span.data);
            selected.view().ForEachSelectedRange(
                [&](size_t begin, size_t count) {
                    if (span.constant) {
                        const T value = data[0];
                        for (size_t i = begin; i < begin + count; ++i)
                            fn(span.begin + i, value);
                    } else {
                        for (size_t i = begin; i < begin + count; ++i)
                            fn(span.begin + i, data[i]);
                    }
                });
        }
        return;
    }
    rows.ForEachSelected([&](size_t row) {
        if (const auto value = input.TryValueAt<T>(row))
            fn(row, *value);
    });
}

// Decode both operands before writing. A failed decode leaves output untouched
// so callers can use their encoding-aware fallback with identical semantics.
template <typename T, typename U, typename Test>
bool
ComparePrimitive(const ReadOnlyVectorView& left,
                 const ReadOnlyVectorView& right,
                 SelectionView rows,
                 CompareOp op,
                 BitmapResultBuilder& output,
                 const Test& test,
                 SelectionKernelPreference preference =
                     SelectionKernelPreference::Adaptive) {
    const auto lhs = left.DecodePrimitive();
    const auto rhs = right.DecodePrimitive();
    if (!lhs || !rhs)
        return false;
    const auto comparison = BitmapCompareOp(op);
    constexpr bool total = !std::is_same_v<T, bool> && !std::is_same_v<U, bool>;
    constexpr bool simd = std::is_same_v<T, U> && total;
    constexpr auto cost = simd ? SelectionKernelCost::NumericComparison()
                               : SelectionKernelCost::ScalarComparison();
    size_t l = 0, r = 0;
    while (l < lhs->spans.size() && r < rhs->spans.size()) {
        const auto& a = lhs->spans[l];
        const auto& b = rhs->spans[r];
        const size_t begin = std::max(a.begin, b.begin);
        const size_t end = std::min(a.begin + a.count, b.begin + b.count);
        if (begin < end) {
            auto active = OwnedSelection::Intersect(
                rows.Slice(begin, end - begin),
                a.valid.Slice(begin - a.begin, end - begin));
            active.Intersect(b.valid.Slice(begin - b.begin, end - begin));
            if (!active.view().IsNone()) {
                const auto* av = static_cast<const T*>(a.data) +
                                 (a.constant ? 0 : begin - a.begin);
                const auto* bv = static_cast<const U*>(b.data) +
                                 (b.constant ? 0 : begin - b.begin);
                const bool readable =
                    total && a.CanEvaluateAll(begin - a.begin, end - begin) &&
                    b.CanEvaluateAll(begin - b.begin, end - begin);
                const auto mode = ChooseSelectionKernel(
                    active.view(), readable, cost, preference);
                auto evaluate_range = [&](size_t offset, size_t count) {
                    const size_t row = begin + offset;
                    const auto* ap = av + (a.constant ? 0 : offset);
                    const auto* bp = bv + (b.constant ? 0 : offset);
                    if (a.constant && b.constant)
                        output.SetRange(row, count, test(*ap, *bp));
                    else if constexpr (simd) {
                        if (b.constant)
                            output.CompareConstant(
                                row, count, ap, *bp, comparison);
                        else if (a.constant)
                            output.CompareConstant(
                                row,
                                count,
                                bp,
                                *ap,
                                ReverseComparison(comparison));
                        else
                            output.CompareColumns(
                                row, count, ap, bp, comparison);
                    } else {
                        DispatchComparison(op, [&]<bitset::CompareOpType Op>() {
                            output.EvaluateRange(row, count, [&](size_t i) {
                                return bitset::CompareOperator<Op>::compare(
                                    ap[a.constant ? 0 : i],
                                    bp[b.constant ? 0 : i]);
                            });
                        });
                    }
                };
                if (mode == SelectionKernelMode::Dense) {
                    evaluate_range(0, end - begin);
                    if (!active.view().IsAll())
                        output.ApplyValidity(begin, active.view());
                } else if (mode == SelectionKernelMode::Sparse) {
                    DispatchComparison(op, [&]<bitset::CompareOpType Op>() {
                        output.EvaluateSelected(
                            begin, active.view(), [&](size_t i) {
                                return bitset::CompareOperator<Op>::compare(
                                    av[a.constant ? 0 : i],
                                    bv[b.constant ? 0 : i]);
                            });
                    });
                } else {
                    active.view().ForEachSelectedRange(evaluate_range);
                }
            }
        }
        if (a.begin + a.count <= b.begin + b.count)
            ++l;
        if (b.begin + b.count <= a.begin + a.count)
            ++r;
    }
    return true;
}

template <typename T, typename Test>
bool
RangePrimitive(const ReadOnlyVectorView& input,
               SelectionView rows,
               T lower,
               T upper,
               bool lower_inclusive,
               bool upper_inclusive,
               BitmapResultBuilder& output,
               const Test& test) {
    const auto decoded = input.DecodePrimitive();
    if (!decoded)
        return false;
    const auto op = lower_inclusive
                        ? (upper_inclusive ? bitset::RangeType::IncInc
                                           : bitset::RangeType::IncExc)
                        : (upper_inclusive ? bitset::RangeType::ExcInc
                                           : bitset::RangeType::ExcExc);
    for (const auto& span : decoded->spans) {
        auto active = OwnedSelection::Intersect(
            rows.Slice(span.begin, span.count), span.valid);
        if (active.view().IsNone())
            continue;
        const auto* data = static_cast<const T*>(span.data);
        const bool readable =
            !std::is_same_v<T, bool> && span.CanEvaluateAll(0, span.count);
        const auto mode = ChooseSelectionKernel(
            active.view(), readable, SelectionKernelCost::NumericComparison());
        auto evaluate_range = [&](size_t begin, size_t count) {
            if (span.constant)
                output.SetRange(span.begin + begin, count, test(data[0]));
            else if constexpr (!std::is_same_v<T, bool>)
                output.WithinRange(
                    span.begin + begin, count, data + begin, lower, upper, op);
            else
                output.EvaluateRange(span.begin + begin, count, [&](size_t i) {
                    return test(data[begin + i]);
                });
        };
        if (mode == SelectionKernelMode::Dense) {
            evaluate_range(0, span.count);
            if (!active.view().IsAll())
                output.ApplyValidity(span.begin, active.view());
        } else if (mode == SelectionKernelMode::Sparse) {
            output.EvaluateSelected(span.begin, active.view(), [&](size_t i) {
                return test(data[span.constant ? 0 : i]);
            });
        } else {
            active.view().ForEachSelectedRange(evaluate_range);
        }
    }
    return true;
}
}  // namespace milvus::exec::scalar
