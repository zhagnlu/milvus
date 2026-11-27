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

#include "exec/expression/ScalarCollectionPredicates.h"
#include "exec/expression/ScalarCallExpr.h"
#include "exec/expression/Element.h"
#include "exec/expression/ScalarObjectKernels.h"

#include <array>

namespace milvus::exec::scalar {
namespace {
// Cheap ARRAY row/element work shares a bounded cancellation budget. A large
// single array must also poll; a row-only interval would not provide that bound.
class ArrayWorkCheckpoint {
 public:
    explicit ArrayWorkCheckpoint(EvalCtx& ctx) : ctx_(ctx) {
    }
    void
    operator()() {
        if (remaining_ == 0) {
            ctx_.Checkpoint();
            remaining_ = 256;
        }
        --remaining_;
    }

 private:
    EvalCtx& ctx_;
    size_t remaining_ = 0;
};

template <typename A>
void
CheckArrayPayload(const A& array, DataType target) {
    const auto stored = array.get_element_type();
    const bool integer = stored == DataType::INT8 ||
                         stored == DataType::INT16 ||
                         stored == DataType::INT32 || stored == DataType::INT64;
    const bool compatible =
        (target == DataType::BOOL && stored == DataType::BOOL) ||
        (target == DataType::INT64 && integer) ||
        ((target == DataType::FLOAT || target == DataType::DOUBLE) &&
         (integer || stored == DataType::FLOAT ||
          stored == DataType::DOUBLE)) ||
        (target == DataType::VARCHAR &&
         (stored == DataType::VARCHAR || stored == DataType::STRING));
    AssertInfo(array.length() == 0 || compatible,
               "ARRAY payload differs from compiled element type");
}
}  // namespace

ExprPtr
MakeArrayElement(ExprPtr input, int64_t index, DataType type) {
    return MakeCall(
        type,
        {std::move(input)},
        [index, type](const auto& inputs, SelectionView rows, EvalCtx& ctx) {
            VectorResultBuilder output(type, rows.row_count());
            ArrayWorkCheckpoint checkpoint(ctx);
            auto evaluate = [&]<typename T>() {
                ArrayInput(inputs[0]).Project<T>(
                    rows, output, [&](const auto& array) -> std::optional<T> {
                        if constexpr (std::is_same_v<T, std::string>)
                            ctx.Checkpoint();
                        else
                            checkpoint();
                        CheckArrayPayload(array, type);
                        if (index < 0 || index >= array.length() ||
                            !array.is_element_valid(index))
                            return std::nullopt;
                        return array.template get_data_unchecked<T>(index);
                    });
            };
            switch (type) {
                case DataType::BOOL:
                    evaluate.template operator()<bool>();
                    break;
                case DataType::INT64:
                    evaluate.template operator()<int64_t>();
                    break;
                case DataType::DOUBLE:
                    evaluate.template operator()<double>();
                    break;
                case DataType::VARCHAR:
                    evaluate.template operator()<std::string>();
                    break;
                default:
                    ThrowInfo(UnexpectedError,
                              "invalid compiled array element type");
            }
            return std::move(output).Finish(rows);
        });
}

ExprPtr
MakeCollectionLength(ExprPtr input, std::vector<std::string> path) {
    const auto type = input->type();
    return MakeCall(
        DataType::INT64,
        {std::move(input)},
        [type, pointer = Json::pointer(std::move(path))](
            const auto& inputs, SelectionView rows, EvalCtx& ctx) {
            VectorResultBuilder output(DataType::INT64, rows.row_count());
            ArrayWorkCheckpoint checkpoint(ctx);
            auto evaluate = [&]<typename T>(const auto& length) {
                ObjectInput<T>(inputs[0]).template Project<int64_t>(
                    rows,
                    output,
                    [&](const T& value) -> std::optional<int64_t> {
                        if constexpr (std::is_same_v<T, Json>)
                            ctx.Checkpoint();
                        else
                            checkpoint();
                        return length(value);
                    });
            };
            if (type == DataType::VECTOR_ARRAY) {
                evaluate.template operator()<VectorArray>(
                    [](const auto& array) { return array.length(); });
            } else if (type == DataType::JSON) {
                evaluate.template operator()<Json>(
                    [&](const Json& json) -> std::optional<int64_t> {
                        auto doc = json.doc();
                        auto array = doc.at_pointer(pointer).get_array();
                        if (array.error())
                            return std::nullopt;
                        return array.count_elements().value();
                    });
            } else if (inputs[0].template DecodeObjects<ArrayView>()) {
                evaluate.template operator()<ArrayView>(
                    [](const auto& array) { return array.length(); });
            } else if (inputs[0].template DecodeObjects<Array>()) {
                evaluate.template operator()<Array>(
                    [](const auto& array) { return array.length(); });
            } else if (inputs[0].template DecodeObjects<ArrayValue>()) {
                evaluate.template operator()<ArrayValue>(
                    [](const auto& array) { return array.size(); });
            } else {
                rows.ForEachSelectedRange([&](size_t begin, size_t count) {
                    output.EvaluateRange<int64_t>(
                        begin, count, [&](size_t i) -> std::optional<int64_t> {
                            checkpoint();
                            return inputs[0].TryArrayLengthAt(begin + i);
                        });
                });
            }
            return std::move(output).Finish(rows);
        });
}

ExprPtr
MakeArrayEquality(ExprPtr input, proto::plan::Array value, bool negate) {
    return MakeCall(
        DataType::BOOL,
        {std::move(input)},
        [value = std::move(value), negate](
            const auto& inputs, SelectionView rows, EvalCtx& ctx) {
            BitmapResultBuilder output(rows.row_count());
            ArrayInput(inputs[0]).Match(rows, output, [&](const auto& array) {
                ctx.Checkpoint();
                return ((array.length() == 0 && value.array_size() == 0) ||
                        array.is_same_array(value)) != negate;
            });
            return std::move(output).Finish(rows);
        });
}

namespace {
template <typename T>
ExprPtr
Contains(ExprPtr input,
         const std::vector<proto::plan::GenericValue>& terms,
         bool all) {
    using Hash = std::conditional_t<std::is_same_v<T, std::string>,
                                    StringHash,
                                    ankerl::unordered_dense::hash<T>>;
    ankerl::unordered_dense::map<T, size_t, Hash, std::equal_to<>> targets;
    auto insert = [&](T value) {
        targets.try_emplace(std::move(value), targets.size());
    };
    for (const auto& term : terms) {
        if constexpr (std::is_same_v<T, std::string>)
            insert(term.string_val());
        else if constexpr (std::is_same_v<T, bool>)
            insert(term.bool_val());
        else if constexpr (std::is_integral_v<T>)
            insert(term.int64_val());
        else
            insert(static_cast<T>(term.has_int64_val()
                                      ? static_cast<double>(term.int64_val())
                                      : term.float_val()));
    }
    return MakeCall(
        DataType::BOOL,
        {std::move(input)},
        [targets = std::move(targets), all](
            const auto& inputs, SelectionView rows, EvalCtx& ctx) {
            BitmapResultBuilder output(rows.row_count());
            const bool small = targets.size() <= 64;
            const uint64_t full_mask =
                targets.size() >= 64 ? UINT64_MAX
                                     : (uint64_t{1} << targets.size()) - 1;
            std::vector<uint64_t> found(
                all && !small ? (targets.size() + 63) / 64 : 0);
            ArrayWorkCheckpoint checkpoint(ctx);
            auto evaluate = [&](const auto& find) {
                ArrayInput(inputs[0]).Match(
                    rows, output, [&](const auto& array) {
                        checkpoint();
                        constexpr auto type =
                            std::is_same_v<T, bool>          ? DataType::BOOL
                            : std::is_same_v<T, std::string> ? DataType::VARCHAR
                            : std::is_integral_v<T>          ? DataType::INT64
                                                             : DataType::DOUBLE;
                        CheckArrayPayload(array, type);
                        // Distinct targets cannot all occur in fewer logical elements,
                        // even when those elements contain duplicates or NULLs.
                        if (all && static_cast<size_t>(array.length()) <
                                       targets.size())
                            return false;
                        if (all)
                            std::fill(found.begin(), found.end(), 0);
                        uint64_t small_found = 0;
                        size_t remaining = targets.size();
                        if (all && remaining == 0)
                            return true;
                        for (int i = 0; i < array.length(); ++i) {
                            checkpoint();
                            if (!array.is_element_valid(i))
                                continue;
                            using Probe = std::conditional_t<
                                std::is_same_v<T, std::string>,
                                std::string_view,
                                T>;
                            const auto value =
                                array.template get_data_unchecked<Probe>(i);
                            const auto target = find(value);
                            if (target == targets.size())
                                continue;
                            if (!all)
                                return true;
                            if (small) {
                                small_found |= uint64_t{1} << target;
                                if (small_found == full_mask)
                                    return true;
                                continue;
                            }
                            auto& word = found[target / 64];
                            const auto bit = uint64_t{1} << (target % 64);
                            if (!(word & bit)) {
                                word |= bit;
                                if (--remaining == 0)
                                    return true;
                            }
                        }
                        return false;
                    });
            };
            // For tiny numeric sets, direct comparisons avoid hashing each
            // array element. Resolve the lookup outside the row loop. Strings
            // keep the transparent hash lookup; their comparison cost differs.
            if constexpr (std::is_arithmetic_v<T>) {
                if (targets.size() <= 4) {
                    std::array<T, 4> values{};
                    for (const auto& [value, index] : targets)
                        values[index] = value;
                    evaluate([&](T value) -> size_t {
                        for (size_t i = 0; i < targets.size(); ++i)
                            if (value == values[i])
                                return i;
                        return targets.size();
                    });
                    return std::move(output).Finish(rows);
                }
            }
            evaluate([&](const auto& value) {
                const auto it = targets.find(value);
                return it == targets.end() ? targets.size() : it->second;
            });
            return std::move(output).Finish(rows);
        });
}
}  // namespace

ExprPtr
MakeArrayContains(ExprPtr input,
                  DataType type,
                  std::vector<proto::plan::GenericValue> terms,
                  bool all) {
    switch (type) {
        case DataType::BOOL:
            return Contains<bool>(std::move(input), terms, all);
        case DataType::INT64:
            return Contains<int64_t>(std::move(input), terms, all);
        case DataType::FLOAT:
            return Contains<float>(std::move(input), terms, all);
        case DataType::DOUBLE:
            return Contains<double>(std::move(input), terms, all);
        case DataType::VARCHAR:
            return Contains<std::string>(std::move(input), terms, all);
        default:
            ThrowInfo(UnexpectedError,
                      "invalid compiled array membership type");
    }
}
}  // namespace milvus::exec::scalar
