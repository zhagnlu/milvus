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

#include "exec/expression/ScalarPredicates.h"
#include "exec/expression/ScalarBatchKernels.h"
#include "exec/expression/ScalarStringKernels.h"
#include "exec/expression/ScalarObjectKernels.h"

#include <unordered_set>
#include <cmath>
#include "common/RegexQuery.h"
#include "common/RoaringMembership.h"
#include "exec/expression/BloomFilterExpr.h"
#include "exec/expression/Element.h"
#include "exec/expression/ScalarCallExpr.h"
#include "exec/expression/InputRefExpr.h"

namespace milvus::exec::scalar {
namespace {
template <typename Test>
ReadOnlyVectorView
EvaluatePredicate(const ReadOnlyVectorView& input,
                  SelectionView rows,
                  const Test& test,
                  bool null_is_false = false) {
    BitmapResultBuilder result(rows.row_count());
    rows.ForEachSelected([&](size_t row) {
        if (input.IsValid(row)) {
            result.Set<bool>(row, test(input, row));

        } else if (null_is_false) {
            result.Set<bool>(row, false);
        }
    });
    return std::move(result).Finish(rows);
}

template <typename Test>
ExprPtr
Predicate(ExprPtr input, Test test, bool null_is_false = false) {
    return MakeCall(DataType::BOOL,
                    {std::move(input)},
                    [test = std::move(test), null_is_false](
                        const auto& inputs, SelectionView rows, EvalCtx&) {
                        return EvaluatePredicate(
                            inputs[0], rows, test, null_is_false);
                    });
}

template <typename T, typename Test>
ExprPtr
PrimitivePredicate(ExprPtr input, Test test) {
    return MakeCall(DataType::BOOL,
                    {std::move(input)},
                    [test = std::move(test)](
                        const auto& inputs, SelectionView rows, EvalCtx&) {
                        BitmapResultBuilder result(rows.row_count());
                        EvaluatePrimitive<T>(inputs[0], rows, result, test);
                        return std::move(result).Finish(rows);
                    });
}

struct StringPredicate {
    StringOp op;
    std::string pattern;
};

template <typename Evaluate>
auto
WithStringMatcher(const StringPredicate& predicate, Evaluate&& evaluate) {
    if (predicate.op == StringOp::Like)
        return evaluate(LikePatternMatcher(predicate.pattern));
    if (predicate.op == StringOp::Regex)
        return evaluate(PartialRegexMatcher(predicate.pattern));
    const std::string_view pattern = predicate.pattern;
    if (predicate.op == StringOp::Prefix)
        return evaluate([pattern](std::string_view text) {
            return text.starts_with(pattern);
        });
    if (predicate.op == StringOp::Suffix)
        return evaluate([pattern](std::string_view text) {
            return text.ends_with(pattern);
        });
    return evaluate([pattern](std::string_view text) {
        return text.find(pattern) != std::string_view::npos;
    });
}

// Cheap literal matchers check cancellation between bounded chunks, outside the
// inner kernel. Keep per-row checks for potentially expensive LIKE/regex work.
template <typename Fn>
void
ForEachStringMatchRange(const StringInput& input,
                        SelectionView rows,
                        StringOp op,
                        EvalCtx& ctx,
                        const Fn& fn) {
    const size_t interval =
        op == StringOp::Like || op == StringOp::Regex ? 1 : 256;
    size_t sparse_since_check = 0;
    input.Visit(
        rows,
        [&](size_t begin, size_t count, const auto* data, bool constant) {
            for (size_t offset = 0; offset < count; offset += interval) {
                ctx.Checkpoint();
                fn(begin + offset,
                   std::min(interval, count - offset),
                   data + (constant ? 0 : offset),
                   constant);
            }
        },
        [&](size_t row, std::string_view text) {
            if (sparse_since_check++ % interval == 0)
                ctx.Checkpoint();
            // This callback is scalar bookkeeping (fused matcher state), not a
            // batch bitmap kernel. Single predicates use Match's packed writer.
            fn(row, size_t{1}, &text, true);
        });
}

class StringMatchExpr final : public Expr {
 public:
    StringMatchExpr(ExprPtr input,
                    LogicalOp op,
                    std::vector<StringPredicate> predicates)
        : Expr(DataType::BOOL),
          input(std::move(input)),
          op(op),
          predicates(std::move(predicates)) {
    }
    const ExprPtr input;
    const LogicalOp op;
    const std::vector<StringPredicate> predicates;

 protected:
    ReadOnlyVectorView
    EvalImpl(SelectionView rows, EvalCtx& ctx) const override {
        const auto values = input->Eval(rows, ctx);
        // Preserve lazy pattern construction even when decoding would fail.
        std::optional<StringInput> strings;
        auto get_strings = [&]() -> const StringInput& {
            if (!strings)
                strings.emplace(values);
            return *strings;
        };
        if (predicates.size() == 1) {
            return WithStringMatcher(
                predicates.front(), [&](const auto& matches) {
                    BitmapResultBuilder output(rows.row_count());
                    if constexpr (std::is_same_v<
                                      std::decay_t<decltype(matches)>,
                                      LikePatternMatcher> ||
                                  std::is_same_v<
                                      std::decay_t<decltype(matches)>,
                                      PartialRegexMatcher>) {
                        get_strings().Match(
                            rows, output, [&](std::string_view value) {
                                ctx.Checkpoint();
                                return matches(value);
                            });
                    } else {
                        size_t checked = 0;
                        get_strings().MatchKernel(
                            rows,
                            output,
                            [&](size_t begin,
                                size_t count,
                                const auto* data,
                                bool constant) {
                                for (size_t offset = 0; offset < count;
                                     offset += 256) {
                                    ctx.Checkpoint();
                                    output.EvaluateRange(
                                        begin + offset,
                                        std::min<size_t>(256, count - offset),
                                        [&](size_t i) {
                                            return matches(std::string_view(
                                                data[constant ? 0
                                                              : offset + i]));
                                        });
                                }
                            },
                            [&](std::string_view value) {
                                if (checked++ % 256 == 0)
                                    ctx.Checkpoint();
                                return matches(value);
                            },
                            false,
                            SelectionKernelCost::SelectedCallback());
                    }
                    return std::move(output).Finish(rows);
                });
        }
        const bool is_and = op == LogicalOp::And;
        auto active = OwnedSelection::CopyOf(rows);
        BitmapResultBuilder output(rows.row_count());
        for (const auto& predicate : predicates) {
            if (active.view().IsNone())
                break;
            ctx.Checkpoint();
            // Keep vector-major order: construct a matcher only when its
            // operand is reached, including when active inputs are all NULL.
            auto evaluate = [&](const auto& matches) {
                auto next = OwnedSelection::CopyOf(active.view());
                ForEachStringMatchRange(
                    get_strings(),
                    active.view(),
                    predicate.op,
                    ctx,
                    [&](size_t begin,
                        size_t count,
                        const auto* data,
                        bool constant) {
                        for (size_t i = 0; i < count; ++i) {
                            if (matches(std::string_view(
                                    data[constant ? 0 : i])) != is_and) {
                                output.Set<bool>(begin + i, !is_and);
                                next.SetSelected(begin + i, false);
                            }
                        }
                    });
                active = std::move(next);
            };
            WithStringMatcher(predicate, evaluate);
        }
        get_strings().ForEachRange(
            active.view(), [&](size_t begin, size_t count, const auto*, bool) {
                output.SetRange(begin, count, is_and);
            });
        return std::move(output).Finish(rows);
    }
};

template <typename T>
T
Value(const ReadOnlyVectorView& values, size_t row) {
    if constexpr (std::is_same_v<T, std::string>)
        return std::string(values.StringViewAt(row));
    else
        return values.ValueAt<T>(row);
}

template <typename T>
ExprPtr
Term(ExprPtr input, const ReadOnlyVectorView& constants) {
    if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>) {
        std::vector<T> values;
        values.reserve(constants.size());
        for (size_t i = 0; i < constants.size(); ++i) {
            AssertInfo(constants.IsValid(i), "unresolved NULL term constant");
            const auto value = Value<T>(constants, i);
            if constexpr (std::is_floating_point_v<T>) {
                // NaN cannot equal an IN term and cannot participate in sorting.
                if (std::isnan(value))
                    continue;
            }
            values.push_back(value);
        }
        // MakeTerm is also called directly, without the Go rewriter. The SIMD
        // head/tail uses binary search, so normalize constants here as well.
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        auto make = [&]<typename Terms>(Terms terms) {
            return MakeCall(
                DataType::BOOL,
                {std::move(input)},
                [terms = std::move(terms),
                 cost = SelectionKernelCost::NumericMembership(
                     values.size(),
                     std::is_same_v<Terms, SimdBatchElement<T>>)](
                    const auto& inputs, SelectionView rows, EvalCtx&) {
                    BitmapResultBuilder result(rows.row_count());
                    auto test = [&](T value) {
                        return terms.In(MultiElement::ValueType(
                            std::in_place_type<T>, value));
                    };
                    if (!MatchPrimitive<T>(
                            inputs[0], rows, result, terms, test, cost))
                        ForEachPrimitiveValue<T>(
                            inputs[0], rows, [&](size_t row, T value) {
                                result.Set<bool>(row, test(value));
                            });
                    return std::move(result).Finish(rows);
                });
        };
        if (values.size() <= static_cast<size_t>(simdLaneCount<T>()) * 8)
            return make(SimdBatchElement<T>(values));
        return make(SetElement<T>(values));
    }
    if constexpr (std::is_same_v<T, bool>) {
        bool accepts_false = false, accepts_true = false;
        for (size_t i = 0; i < constants.size(); ++i) {
            AssertInfo(constants.IsValid(i), "unresolved NULL term constant");
            if (constants.ValueAt<bool>(i))
                accepts_true = true;
            else
                accepts_false = true;
        }
        return MakeCall(
            DataType::BOOL,
            {std::move(input)},
            [accepts_false, accepts_true](
                const auto& inputs, SelectionView rows, EvalCtx&) {
                auto masks = inputs[0].CopyBoolMasks(rows);
                auto truth =
                    accepts_false
                        ? (accepts_true
                               ? OwnedSelection::CopyOf(masks.valid.view())
                               : OwnedSelection::Subtract(masks.valid.view(),
                                                          masks.truth.view()))
                        : (accepts_true
                               ? std::move(masks.truth)
                               : OwnedSelection::None(rows.row_count()));
                return ReadOnlyVectorView::FromBooleanMasks(
                    std::move(truth), std::move(masks.valid), rows);
            });
    }
    if constexpr (std::is_same_v<T, std::string>) {
        std::vector<std::string> values;
        values.reserve(constants.size());
        for (size_t i = 0; i < constants.size(); ++i) {
            AssertInfo(constants.IsValid(i), "unresolved NULL term constant");
            values.emplace_back(constants.StringViewAt(i));
        }
        // Reuse legacy hashing and bulk membership, borrowing candidate text.
        return MakeCall(
            DataType::BOOL,
            {std::move(input)},
            [terms = SetElement<std::string>(values)](
                const auto& inputs, SelectionView rows, EvalCtx&) {
                BitmapResultBuilder output(rows.row_count());
                StringInput(inputs[0]).MatchKernel(
                    rows,
                    output,
                    [&](size_t begin,
                        size_t count,
                        const auto* data,
                        bool constant) {
                        if (constant) {
                            output.SetRange(
                                begin,
                                count,
                                terms.In(MultiElement::ValueType(
                                    std::in_place_type<std::string_view>,
                                    std::string_view(*data))));
                        } else {
                            output.MatchTerms(begin, count, data, terms);
                        }
                    },
                    [&](std::string_view value) {
                        return terms.In(MultiElement::ValueType(
                            std::in_place_type<std::string_view>, value));
                    },
                    true,
                    SelectionKernelCost::StringMembership());
                return std::move(output).Finish(rows);
            });
    }
    std::unordered_set<T> terms;
    for (size_t i = 0; i < constants.size(); ++i) {
        AssertInfo(constants.IsValid(i), "unresolved NULL term constant");
        terms.insert(Value<T>(constants, i));
    }
    if constexpr (std::is_arithmetic_v<T>) {
        return PrimitivePredicate<T>(std::move(input),
                                     [terms = std::move(terms)](T value) {
                                         return terms.count(value) != 0;
                                     });
    }
    return Predicate(
        std::move(input),
        [terms = std::move(terms)](const auto& values, size_t row) {
            return terms.count(Value<T>(values, row)) != 0;
        });
}
}  // namespace

ExprPtr
MakeTerm(ExprPtr input, const VectorPtr& constants) {
    AssertInfo(input && constants, "term requires input and constants");
    return MakeTerm(std::move(input), ReadOnlyVectorView::CopyOf(constants));
}

ExprPtr
MakeTerm(ExprPtr input, const ReadOnlyVectorView& values) {
    AssertInfo(input, "term requires input");
    AssertInfo(input->type() == values.logical_type(),
               "term constant type mismatch");
    switch (input->type()) {
        case DataType::BOOL:
            return Term<bool>(std::move(input), values);
        case DataType::INT8:
            return Term<int8_t>(std::move(input), values);
        case DataType::INT16:
            return Term<int16_t>(std::move(input), values);
        case DataType::INT32:
            return Term<int32_t>(std::move(input), values);
        case DataType::INT64:
        case DataType::TIMESTAMPTZ:
            return Term<int64_t>(std::move(input), values);
        case DataType::FLOAT:
            return Term<float>(std::move(input), values);
        case DataType::DOUBLE:
            return Term<double>(std::move(input), values);
        case DataType::STRING:
        case DataType::VARCHAR:
        case DataType::TEXT:
            return Term<std::string>(std::move(input), values);
        default:
            ThrowInfo(Unsupported, "term input type is not migrated");
    }
}

ExprPtr
MakeStringMatch(ExprPtr input, StringOp op, std::string pattern) {
    AssertInfo(input && (input->type() == DataType::VARCHAR ||
                         input->type() == DataType::STRING ||
                         input->type() == DataType::TEXT),
               "string match requires a text input");
    AssertInfo(op == StringOp::Like || op == StringOp::Regex ||
                   op == StringOp::Prefix || op == StringOp::Suffix ||
                   op == StringOp::Contains,
               "invalid string match operation");
    return std::make_shared<StringMatchExpr>(
        std::move(input),
        LogicalOp::And,
        std::vector<StringPredicate>{{op, std::move(pattern)}});
}

std::vector<ExprPtr>
FuseAdjacentStringMatches(LogicalOp op,
                          std::vector<ExprPtr> inputs,
                          const std::function<ExprPtr(ExprPtr)>& resolve) {
    std::vector<ExprPtr> resolved;
    for (const auto& input : inputs)
        resolved.push_back(resolve ? resolve(input) : input);
    std::vector<ExprPtr> result;
    auto eligible = [op](const ExprPtr& node) -> const StringMatchExpr* {
        auto* match = dynamic_cast<const StringMatchExpr*>(node.get());
        return match && (match->predicates.size() == 1 || match->op == op) &&
                       dynamic_cast<const InputRefExpr*>(match->input.get())
                   ? match
                   : nullptr;
    };
    for (size_t begin = 0; begin < inputs.size();) {
        auto* first = eligible(resolved[begin]);
        if (!first) {
            result.push_back(inputs[begin++]);
            continue;
        }
        const auto* field =
            static_cast<const InputRefExpr*>(first->input.get());
        auto predicates = first->predicates;
        size_t end = begin + 1;
        for (; end < inputs.size(); ++end) {
            const auto* next = eligible(resolved[end]);
            if (!next)
                break;
            const auto* other =
                static_cast<const InputRefExpr*>(next->input.get());
            if (field->slot() != other->slot() ||
                field->type() != other->type())
                break;
            predicates.insert(predicates.end(),
                              next->predicates.begin(),
                              next->predicates.end());
        }
        result.push_back(end == begin + 1
                             ? inputs[begin]
                             : std::make_shared<StringMatchExpr>(
                                   first->input, op, std::move(predicates)));
        begin = end;
    }
    return result;
}

ExprPtr
MakeJsonExists(ExprPtr input, std::vector<std::string> path) {
    AssertInfo(input && input->type() == DataType::JSON,
               "JSON exists requires JSON input");
    return MakeCall(
        DataType::BOOL,
        {std::move(input)},
        [pointer = Json::pointer(std::move(path))](
            const auto& inputs, SelectionView rows, EvalCtx&) {
            BitmapResultBuilder result(rows.row_count());
            ObjectInput<Json>(inputs[0]).Match(
                rows, result, [&](const Json& json) {
                    return json.exist(pointer);
                });
            // EXISTS is definite FALSE even for SQL NULL. Set validity after
            // the sparse writer, which clears gaps between evaluated objects.
            result.ApplyValidity(0, rows);
            return std::move(result).Finish(rows);
        });
}

ExprPtr
MakeRoaringMembership(ExprPtr input,
                      std::shared_ptr<const RoaringMembership> membership) {
    AssertInfo(input && membership,
               "roaring membership requires decoded owner");
    auto make = [&]<typename T>() {
        return PrimitivePredicate<T>(input, [membership](T value) {
            return membership->Contains(static_cast<int64_t>(value));
        });
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
        default:
            ThrowInfo(Unsupported, "roaring requires a signed integer input");
    }
}
ExprPtr
MakeBloomMembership(ExprPtr input,
                    std::shared_ptr<const std::string> blob,
                    std::vector<std::string> path) {
    AssertInfo(input && blob, "bloom membership requires input and blob owner");
    AssertInfo(input->type() == DataType::JSON || path.empty(),
               "only JSON bloom inputs may have a path");
    // Like the legacy physical constructor, validate before evaluation, even
    // for empty/short-circuited branches. The view aliases the retained owner.
    const auto filter = SplitBlockBloomFilterView::Parse(*blob);
    auto make = [&]<typename T>() {
        return PrimitivePredicate<T>(input, [blob, filter](T value) {
            return filter.TestScalar(value);
        });
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
        case DataType::VARCHAR:
            return MakeCall(
                DataType::BOOL,
                {input},
                [blob, filter](
                    const auto& inputs, SelectionView rows, EvalCtx&) {
                    BitmapResultBuilder output(rows.row_count());
                    StringInput(inputs[0]).Match(
                        rows, output, [&](std::string_view text) {
                            return filter.TestScalar(text);
                        });
                    return std::move(output).Finish(rows);
                });
        case DataType::JSON:
            return MakeCall(
                DataType::BOOL,
                {std::move(input)},
                [blob, filter, pointer = Json::pointer(std::move(path))](
                    const auto& inputs, SelectionView rows, EvalCtx&) {
                    BitmapResultBuilder result(rows.row_count());
                    ObjectInput<Json>(inputs[0]).MatchOptional(
                        rows,
                        result,
                        [&](const Json& json) -> std::optional<bool> {
                            const auto value = json.at_string_or_int64(pointer);
                            switch (value.kind) {
                                case JsonStringOrInt64::Kind::String:
                                    return filter.TestScalar(
                                        value.string_value);
                                case JsonStringOrInt64::Kind::Int64:
                                    return filter.TestInt64(value.int64_value);
                                case JsonStringOrInt64::Kind::OtherNumber:
                                    return false;
                                case JsonStringOrInt64::Kind::NoProbeValue:
                                    return std::nullopt;
                            }
                            return std::nullopt;
                        });
                    return std::move(result).Finish(rows);
                });
        default:
            ThrowInfo(Unsupported, "bloom input type is not migrated");
    }
}
}  // namespace milvus::exec::scalar
