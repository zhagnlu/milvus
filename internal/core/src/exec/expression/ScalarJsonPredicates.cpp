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

#include "exec/expression/ScalarJsonPredicates.h"

#include <cmath>
#include <unordered_map>
#include "exec/expression/Element.h"
#include "common/RegexQuery.h"
#include "exec/expression/JsonArrayComparison.h"
#include "exec/expression/JsonNumberComparison.h"
#include "exec/expression/ScalarCallExpr.h"
#include "exec/expression/ScalarObjectKernels.h"

namespace milvus::exec::scalar {
namespace {
using Value = proto::plan::GenericValue;

template <typename T>
bool
Compare(const T& left, const T& right, proto::plan::OpType op) {
    switch (op) {
        case proto::plan::Equal:
            return left == right;
        case proto::plan::NotEqual:
            return left != right;
        case proto::plan::LessThan:
            return left < right;
        case proto::plan::LessEqual:
            return left <= right;
        case proto::plan::GreaterThan:
            return left > right;
        case proto::plan::GreaterEqual:
            return left >= right;
        default:
            ThrowInfo(UnexpectedError, "invalid JSON comparison");
    }
}

template <typename Test>
ExprPtr
Predicate(ExprPtr input, Test test) {
    AssertInfo(input && input->type() == DataType::JSON,
               "JSON predicate requires JSON input");
    return MakeCall(DataType::BOOL,
                    {std::move(input)},
                    [test = std::move(test)](
                        const auto& inputs, SelectionView rows, EvalCtx&) {
                        BitmapResultBuilder result(rows.row_count());
                        ObjectInput<Json>(inputs[0]).MatchOptional(
                            rows, result, test);
                        return std::move(result).Finish(rows);
                    });
}

bool
PreciseRangeBound(const Value& value) {
    // Same boundary as SegmentExpr::JsonNumericBoundRequiresPreciseInt64Comparison.
    if (value.has_int64_val())
        return value.int64_val() <= -(int64_t{1} << 53) ||
               value.int64_val() >= (int64_t{1} << 53);
    if (!value.has_float_val() || !std::isfinite(value.float_val()))
        return false;
    const auto magnitude = std::abs(value.float_val());
    return magnitude >= 0x1p53 && magnitude <= 0x1p63;
}

// Numeric coercion is deliberately specific to Contains. Ordinary JSON IN
// and scalar comparisons have different legacy conversion rules.
template <typename Element>
bool
ContainsValue(Element&& element, const Value& target, bool same_type) {
    switch (target.val_case()) {
        case Value::kBoolVal: {
            auto value = element.template get<bool>();
            return !value.error() && value.value() == target.bool_val();
        }
        case Value::kInt64Val: {
            auto value = element.template get<int64_t>();
            if (!value.error())
                return value.value() == target.int64_val();
            auto number = element.template get<double>();
            if (number.error())
                return false;
            if (same_type) {
                auto exact = DoubleToInt64Exact(number.value());
                return exact && *exact == target.int64_val();
            }
            return number.value() == target.int64_val();
        }
        case Value::kFloatVal: {
            auto value = element.template get<double>();
            return !value.error() && value.value() == target.float_val();
        }
        case Value::kStringVal: {
            auto value = element.template get<std::string_view>();
            return !value.error() && value.value() == target.string_val();
        }
        case Value::kArrayVal: {
            auto value = element.get_array();
            return !value.error() &&
                   CompareTwoJsonArray(value, target.array_val());
        }
        default:
            ThrowInfo(UnexpectedError,
                      "invalid compiled JSON contains constant");
    }
}

template <typename T>
std::optional<bool>
ScalarTerm(const Json& json,
           const std::string& pointer,
           const std::vector<Value>& terms) {
    auto value = json.template at<T>(pointer);
    if (value.error())
        return std::nullopt;
    for (const auto& term : terms) {
        if constexpr (std::is_same_v<T, bool>) {
            if (value.value() == term.bool_val())
                return true;
        } else if constexpr (std::is_same_v<T, double>) {
            if (value.value() == term.float_val())
                return true;
        } else {
            if (value.value() == term.string_val())
                return true;
        }
    }
    return false;
}

// Homogeneous contains parses each element once and probes prepared constants.
// The heterogeneous/nested-array path below retains its distinct coercion rules.
template <typename T>
ExprPtr
TypedContains(ExprPtr input,
              std::string pointer,
              const std::vector<Value>& terms,
              bool all) {
    using Hash = std::
        conditional_t<std::is_same_v<T, std::string>, StringHash, std::hash<T>>;
    std::unordered_map<T, size_t, Hash, std::equal_to<>> targets;
    for (const auto& term : terms) {
        T value;
        if constexpr (std::is_same_v<T, bool>)
            value = term.bool_val();
        else if constexpr (std::is_same_v<T, int64_t>)
            value = term.int64_val();
        else if constexpr (std::is_same_v<T, double>)
            value = term.float_val();
        else
            value = term.string_val();
        targets.try_emplace(std::move(value), targets.size());
    }
    return MakeCall(
        DataType::BOOL,
        {std::move(input)},
        [pointer = std::move(pointer), targets = std::move(targets), all](
            const auto& inputs, SelectionView rows, EvalCtx&) {
            BitmapResultBuilder output(rows.row_count());
            std::vector<uint8_t> found(all ? targets.size() : 0);
            ObjectInput<Json>(inputs[0]).MatchOptional(
                rows, output, [&](const Json& json) -> std::optional<bool> {
                    auto document = json.doc();
                    auto array = document.at_pointer(pointer).get_array();
                    if (array.error())
                        return std::nullopt;
                    if (all)
                        std::fill(found.begin(), found.end(), 0);
                    size_t remaining = targets.size();
                    for (auto&& element : array) {
                        using Probe =
                            std::conditional_t<std::is_same_v<T, std::string>,
                                               std::string_view,
                                               T>;
                        std::optional<Probe> probe;
                        auto value = element.template get<Probe>();
                        if (!value.error())
                            probe = value.value();
                        else if constexpr (std::is_same_v<T, int64_t>) {
                            auto floating = element.template get<double>();
                            if (!floating.error())
                                probe = DoubleToInt64Exact(floating.value());
                        }
                        if (!probe)
                            continue;
                        const auto hit = targets.find(*probe);
                        if (hit == targets.end())
                            continue;
                        if (!all)
                            return true;
                        if (!found[hit->second]) {
                            found[hit->second] = 1;
                            if (--remaining == 0)
                                return true;
                        }
                    }
                    return all && remaining == 0;
                });
            return std::move(output).Finish(rows);
        });
}
}  // namespace

ExprPtr
MakeJsonComparison(ExprPtr input,
                   std::vector<std::string> path,
                   proto::plan::OpType operation,
                   Value bound) {
    // Matcher construction remains execution-local, preserving skipped-pattern
    // errors. The string itself is extracted only on selected valid rows.
    const auto pointer = Json::pointer(std::move(path));
    if (bound.has_string_val() && (operation == proto::plan::Match ||
                                   operation == proto::plan::RegexMatch)) {
        return MakeCall(
            DataType::BOOL,
            {std::move(input)},
            [pointer, operation, pattern = bound.string_val()](
                const auto& inputs, SelectionView rows, EvalCtx&) {
                auto evaluate = [&](const auto& matcher) {
                    BitmapResultBuilder result(rows.row_count());
                    ObjectInput<Json>(inputs[0]).MatchOptional(
                        rows,
                        result,
                        [&](const Json& json) -> std::optional<bool> {
                            auto value =
                                json.template at<std::string_view>(pointer);
                            if (value.error())
                                return std::nullopt;
                            return matcher(value.value());
                        });
                    return std::move(result).Finish(rows);
                };
                if (operation == proto::plan::Match)
                    return evaluate(LikePatternMatcher(pattern));
                return evaluate(PartialRegexMatcher(pattern));
            });
    }
    return Predicate(
        std::move(input),
        [pointer, operation, bound = std::move(bound)](
            const Json& json) -> std::optional<bool> {
            switch (bound.val_case()) {
                case Value::kInt64Val: {
                    auto value = json.at_numeric(pointer);
                    if (value.error())
                        return std::nullopt;
                    const auto comparison =
                        CompareJsonNumberToBound(value.value(), bound);
                    return comparison &&
                           JsonNumberMatchesOp(*comparison, operation);
                }
                case Value::kFloatVal: {
                    auto value = json.at<double>(pointer);
                    if (value.error())
                        return std::nullopt;
                    return Compare(value.value(), bound.float_val(), operation);
                }
                case Value::kBoolVal: {
                    auto value = json.at<bool>(pointer);
                    if (value.error())
                        return std::nullopt;
                    return Compare(value.value(), bound.bool_val(), operation);
                }
                case Value::kStringVal: {
                    auto value = json.at<std::string_view>(pointer);
                    if (value.error())
                        return std::nullopt;
                    const std::string_view target(bound.string_val());
                    if (operation == proto::plan::PrefixMatch)
                        return value.value().starts_with(target);
                    if (operation == proto::plan::PostfixMatch)
                        return value.value().ends_with(target);
                    if (operation == proto::plan::InnerMatch)
                        return value.value().find(target) !=
                               std::string_view::npos;
                    return Compare(value.value(), target, operation);
                }
                case Value::kArrayVal: {
                    auto document = json.doc();
                    auto value = document.at_pointer(pointer).get_array();
                    if (value.error())
                        return std::nullopt;
                    const bool equal =
                        CompareTwoJsonArray(value, bound.array_val());
                    return operation == proto::plan::Equal ? equal : !equal;
                }
                default:
                    ThrowInfo(UnexpectedError,
                              "invalid compiled JSON comparison bound");
            }
        });
}

ExprPtr
MakeJsonRange(ExprPtr input,
              std::vector<std::string> path,
              Value lower,
              Value upper,
              bool lower_inclusive,
              bool upper_inclusive) {
    return Predicate(
        std::move(input),
        [pointer = Json::pointer(std::move(path)),
         lower,
         upper,
         lower_inclusive,
         upper_inclusive](const Json& json) -> std::optional<bool> {
            if (lower.has_string_val()) {
                auto value = json.at<std::string_view>(pointer);
                if (value.error())
                    return std::nullopt;
                return (lower_inclusive ? value.value() >= lower.string_val()
                                        : value.value() > lower.string_val()) &&
                       (upper_inclusive ? value.value() <= upper.string_val()
                                        : value.value() < upper.string_val());
            }
            auto value = json.at_numeric(pointer);
            if (value.error())
                return std::nullopt;
            if (PreciseRangeBound(lower) || PreciseRangeBound(upper)) {
                auto a = CompareJsonNumberToBoundWithUint64DoubleFallback(
                    value.value(), lower);
                auto b = CompareJsonNumberToBoundWithUint64DoubleFallback(
                    value.value(), upper);
                return a && b && (lower_inclusive ? *a >= 0 : *a > 0) &&
                       (upper_inclusive ? *b <= 0 : *b < 0);
            }
            const auto number = value.value();
            if (number.is_int64() && lower.has_int64_val() &&
                upper.has_int64_val()) {
                const auto n = number.get_int64();
                return (lower_inclusive ? n >= lower.int64_val()
                                        : n > lower.int64_val()) &&
                       (upper_inclusive ? n <= upper.int64_val()
                                        : n < upper.int64_val());
            }
            const double n =
                number.is_int64()    ? static_cast<double>(number.get_int64())
                : number.is_uint64() ? static_cast<double>(number.get_uint64())
                                     : number.get_double();
            const double a = lower.has_int64_val()
                                 ? static_cast<double>(lower.int64_val())
                                 : lower.float_val();
            const double b = upper.has_int64_val()
                                 ? static_cast<double>(upper.int64_val())
                                 : upper.float_val();
            return (lower_inclusive ? n >= a : n > a) &&
                   (upper_inclusive ? n <= b : n < b);
        });
}

ExprPtr
MakeJsonTerm(ExprPtr input,
             std::vector<std::string> path,
             std::vector<Value> terms,
             bool in_field) {
    AssertInfo(input && input->type() == DataType::JSON,
               "JSON term requires JSON input");
    AssertInfo(!in_field || terms.size() == 1,
               "JSON in-field term requires one constant");
    if (terms.empty())
        return MakeLiteral(
            std::make_shared<ConstantVector<bool>>(DataType::BOOL, 1, false));
    return Predicate(
        std::move(input),
        [pointer = Json::pointer(std::move(path)),
         terms = std::move(terms),
         in_field](const Json& json) -> std::optional<bool> {
            if (in_field) {
                auto document = json.doc();
                auto array = document.at_pointer(pointer).get_array();
                if (array.error())
                    return std::nullopt;
                const auto& target = terms[0];
                for (auto&& element : array) {
                    // in-field integer membership is strict: 1 does not match 1.0.
                    if (target.has_int64_val()) {
                        auto n = element.template get<int64_t>();
                        if (!n.error() && n.value() == target.int64_val())
                            return true;
                    } else if (ContainsValue(element, target, true))
                        return true;
                }
                return false;
            }
            if (terms[0].has_bool_val())
                return ScalarTerm<bool>(json, pointer, terms);
            if (terms[0].has_float_val())
                return ScalarTerm<double>(json, pointer, terms);
            if (terms[0].has_string_val())
                return ScalarTerm<std::string_view>(json, pointer, terms);
            auto value = json.at_numeric(pointer);
            if (value.error())
                return std::nullopt;
            const auto number = value.value();
            const auto integer =
                number.is_int64()
                    ? std::optional<int64_t>(number.get_int64())
                    : DoubleToInt64Exact(
                          number.is_uint64()
                              ? static_cast<double>(number.get_uint64())
                              : number.get_double());
            if (!integer)
                return false;
            for (const auto& term : terms)
                if (*integer == term.int64_val())
                    return true;
            return false;
        });
}

ExprPtr
MakeJsonContains(ExprPtr input,
                 std::vector<std::string> path,
                 std::vector<Value> terms,
                 bool all,
                 bool same_type) {
    AssertInfo(input && input->type() == DataType::JSON,
               "JSON contains requires JSON input");
    if (terms.empty())
        return MakeLiteral(
            std::make_shared<ConstantVector<bool>>(DataType::BOOL, 1, all));
    if (same_type &&
        std::all_of(terms.begin(), terms.end(), [&](const auto& term) {
            return term.val_case() == terms.front().val_case();
        })) {
        const auto pointer = Json::pointer(path);
        switch (terms.front().val_case()) {
            case Value::kBoolVal:
                return TypedContains<bool>(
                    std::move(input), pointer, terms, all);
            case Value::kInt64Val:
                return TypedContains<int64_t>(
                    std::move(input), pointer, terms, all);
            case Value::kFloatVal:
                return TypedContains<double>(
                    std::move(input), pointer, terms, all);
            case Value::kStringVal:
                return TypedContains<std::string>(
                    std::move(input), pointer, terms, all);
            default:
                break;
        }
    }
    return MakeCall(
        DataType::BOOL,
        {std::move(input)},
        [pointer = Json::pointer(std::move(path)),
         terms = std::move(terms),
         all,
         same_type](const auto& inputs, SelectionView rows, EvalCtx&) {
            BitmapResultBuilder output(rows.row_count());
            // Per-Eval scratch is safe for concurrent evaluations and reused
            // across documents. ANY has no ALL bookkeeping allocation.
            std::vector<uint8_t> found(all ? terms.size() : 0);
            ObjectInput<Json>(inputs[0]).MatchOptional(
                rows, output, [&](const Json& json) -> std::optional<bool> {
                    auto evaluate = [&](auto array) -> std::optional<bool> {
                        if (array.error())
                            return std::nullopt;
                        if (all)
                            std::fill(found.begin(), found.end(), 0);
                        size_t remaining = terms.size();
                        for (auto&& element : array) {
                            for (size_t i = 0; i < terms.size(); ++i) {
                                if ((!all || !found[i]) &&
                                    ContainsValue(
                                        element, terms[i], same_type)) {
                                    if (!all)
                                        return true;
                                    found[i] = 1;
                                    if (--remaining == 0)
                                        return true;
                                }
                            }
                        }
                        return all && remaining == 0;
                    };
                    if (!same_type || terms[0].has_array_val()) {
                        auto document = json.dom_doc();
                        return evaluate(
                            document.at_pointer(pointer).get_array());
                    }
                    auto document = json.doc();
                    return evaluate(document.at_pointer(pointer).get_array());
                });
            return std::move(output).Finish(rows);
        });
}
}  // namespace milvus::exec::scalar
