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

#include "exec/expression/ScalarBatchKernels.h"

namespace milvus::exec::scalar {

// Physical representation is dispatched at the span boundary, never per row.
template <typename Fn>
void
VisitStringSpan(const ReadOnlyVectorView::DecodedSpan& span, const Fn& fn) {
    if (span.string_views)
        fn(static_cast<const std::string_view*>(span.data));
    else
        fn(static_cast<const std::string*>(span.data));
}

// Decode once even when several fused matchers consume the input. The source
// view and decoded owner outlive all callbacks. No NULL payload is examined.
class StringInput {
 public:
    explicit StringInput(const ReadOnlyVectorView& input)
        : input_(input), decoded_(input.DecodeStrings()) {
    }

    template <typename Fn>
    void
    ForEachRange(SelectionView rows, const Fn& fn) const {
        if (!decoded_) {
            rows.ForEachSelected([&](size_t row) {
                if (auto value = input_.TryStringViewAt(row))
                    fn(row, size_t{1}, &*value, true);
            });
            return;
        }
        for (const auto& span : decoded_->spans) {
            auto active = OwnedSelection::Intersect(
                rows.Slice(span.begin, span.count), span.valid);
            if (active.view().IsNone())
                continue;
            VisitStringSpan(span, [&](const auto* data) {
                active.view().ForEachSelectedRange(
                    [&](size_t offset, size_t count) {
                        fn(span.begin + offset,
                           count,
                           data + (span.constant ? 0 : offset),
                           span.constant);
                    });
            });
        }
    }

    template <typename Range, typename Value>
    void
    Visit(SelectionView rows,
          const Range& range,
          const Value& value,
          SelectionKernelCost cost =
              SelectionKernelCost::SelectedCallback()) const {
        if (!decoded_) {
            rows.ForEachSelected([&](size_t row) {
                if (auto text = input_.TryStringViewAt(row))
                    value(row, *text);
            });
            return;
        }
        for (const auto& span : decoded_->spans) {
            auto active = OwnedSelection::Intersect(
                rows.Slice(span.begin, span.count), span.valid);
            VisitStringSpan(span, [&](const auto* data) {
                VisitSelected(
                    active.view(),
                    cost,
                    [&](size_t offset, size_t count) {
                        range(span.begin + offset,
                              count,
                              data + (span.constant ? 0 : offset),
                              span.constant);
                    },
                    [&](size_t i) {
                        value(span.begin + i,
                              std::string_view(data[span.constant ? 0 : i]));
                    });
            });
        }
    }

    template <typename Test>
    void
    Match(SelectionView rows,
          BitmapResultBuilder& output,
          const Test& test,
          bool total = false,
          SelectionKernelCost cost =
              SelectionKernelCost::SelectedCallback()) const {
        MatchKernel(
            rows,
            output,
            [&](size_t begin, size_t count, const auto* data, bool constant) {
                output.EvaluateRange(begin, count, [&](size_t i) {
                    return test(std::string_view(data[constant ? 0 : i]));
                });
            },
            test,
            total,
            cost);
    }

    template <typename Range, typename Test>
    void
    MatchKernel(SelectionView rows,
                BitmapResultBuilder& output,
                const Range& range,
                const Test& test,
                bool total,
                SelectionKernelCost cost,
                SelectionKernelPreference preference =
                    SelectionKernelPreference::Adaptive) const {
        if (!decoded_) {
            rows.ForEachSelected([&](size_t row) {
                if (auto value = input_.TryStringViewAt(row))
                    output.Set<bool>(row, test(*value));
            });
            return;
        }
        for (const auto& span : decoded_->spans) {
            auto active = OwnedSelection::Intersect(
                rows.Slice(span.begin, span.count), span.valid);
            VisitStringSpan(span, [&](const auto* data) {
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
                    [&](size_t i) {
                        return test(
                            std::string_view(data[span.constant ? 0 : i]));
                    },
                    preference);
            });
        }
    }

 private:
    const ReadOnlyVectorView& input_;
    std::optional<ReadOnlyVectorView::DecodedView> decoded_;
};

// Decode both operands before writing so unsupported encodings can fall back
// without leaving partially evaluated output. Intersect unequal span boundaries.
inline bool
CompareStrings(const ReadOnlyVectorView& left,
               const ReadOnlyVectorView& right,
               SelectionView rows,
               CompareOp op,
               BitmapResultBuilder& output) {
    const auto lhs = left.DecodeStrings();
    const auto rhs = right.DecodeStrings();
    if (!lhs || !rhs)
        return false;
    const auto comparison = op == CompareOp::Prefix ? bitset::CompareOpType::EQ
                                                    : BitmapCompareOp(op);
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
                VisitStringSpan(a, [&](const auto* adata) {
                    VisitStringSpan(b, [&](const auto* bdata) {
                        auto evaluate_range = [&](size_t offset, size_t count) {
                            const size_t row = begin + offset;
                            const auto* av =
                                adata + (a.constant ? 0 : row - a.begin);
                            const auto* bv =
                                bdata + (b.constant ? 0 : row - b.begin);
                            if (op == CompareOp::Prefix) {
                                output.EvaluateRange(row, count, [&](size_t i) {
                                    return std::string_view(
                                               av[a.constant ? 0 : i])
                                        .starts_with(std::string_view(
                                            bv[b.constant ? 0 : i]));
                                });
                                return;
                            }
                            using A = std::remove_cv_t<
                                std::remove_pointer_t<decltype(av)>>;
                            using B = std::remove_cv_t<
                                std::remove_pointer_t<decltype(bv)>>;
                            // Convert a bound once per run; the column payload is
                            // never converted or copied to another array.
                            if constexpr (std::is_same_v<A, std::string_view>) {
                                if (!a.constant && b.constant) {
                                    output.CompareConstant(
                                        row,
                                        count,
                                        av,
                                        std::string_view(*bv),
                                        comparison);
                                    return;
                                }
                            }
                            if constexpr (std::is_same_v<B, std::string_view>) {
                                if (a.constant && !b.constant) {
                                    output.CompareConstant(
                                        row,
                                        count,
                                        bv,
                                        std::string_view(*av),
                                        ReverseComparison(comparison));
                                    return;
                                }
                            }
                            if constexpr (std::is_same_v<A, B>) {
                                if (!a.constant && !b.constant) {
                                    output.CompareColumns(
                                        row, count, av, bv, comparison);
                                    return;
                                }
                            }
                            // Owned strings, constant/constant and mixed storage
                            // use the same block bitmap writer with no view access.
                            auto evaluate = [&]<bitset::CompareOpType Op>() {
                                output.EvaluateRange(row, count, [&](size_t i) {
                                    return bitset::CompareOperator<Op>::compare(
                                        std::string_view(
                                            av[a.constant ? 0 : i]),
                                        std::string_view(
                                            bv[b.constant ? 0 : i]));
                                });
                            };
                            switch (comparison) {
                                case bitset::CompareOpType::EQ:
                                    evaluate.template
                                    operator()<bitset::CompareOpType::EQ>();
                                    break;
                                case bitset::CompareOpType::NE:
                                    evaluate.template
                                    operator()<bitset::CompareOpType::NE>();
                                    break;
                                case bitset::CompareOpType::LT:
                                    evaluate.template
                                    operator()<bitset::CompareOpType::LT>();
                                    break;
                                case bitset::CompareOpType::LE:
                                    evaluate.template
                                    operator()<bitset::CompareOpType::LE>();
                                    break;
                                case bitset::CompareOpType::GT:
                                    evaluate.template
                                    operator()<bitset::CompareOpType::GT>();
                                    break;
                                case bitset::CompareOpType::GE:
                                    evaluate.template
                                    operator()<bitset::CompareOpType::GE>();
                                    break;
                            }
                        };
                        const bool readable =
                            a.CanEvaluateAll(begin - a.begin, end - begin) &&
                            b.CanEvaluateAll(begin - b.begin, end - begin);
                        auto run = [&](const auto& test) {
                            EvaluateSelection(
                                begin,
                                active.view(),
                                readable,
                                SelectionKernelCost::StringComparison(),
                                output,
                                evaluate_range,
                                [&](size_t i) {
                                    return test(
                                        std::string_view(
                                            adata[a.constant
                                                      ? 0
                                                      : begin - a.begin + i]),
                                        std::string_view(
                                            bdata[b.constant
                                                      ? 0
                                                      : begin - b.begin + i]));
                                });
                        };
                        if (op == CompareOp::Prefix)
                            run([](std::string_view a, std::string_view b) {
                                return a.starts_with(b);
                            });
                        else
                            DispatchComparison(
                                op, [&]<bitset::CompareOpType Op>() {
                                    run([](std::string_view a,
                                           std::string_view b) {
                                        return bitset::CompareOperator<
                                            Op>::compare(a, b);
                                    });
                                });
                    });
                });
            }
        }
        if (a.begin + a.count <= b.begin + b.count)
            ++l;
        if (b.begin + b.count <= a.begin + a.count)
            ++r;
    }
    return true;
}

}  // namespace milvus::exec::scalar
