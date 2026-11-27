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

#include "segcore/reader/ScalarIndexReader.h"
#include <cmath>
#include <nlohmann/json.hpp>
#include "index/contracts/query/INgramReader.h"
#include "index/contracts/query/IJsonIndexReader.h"
#include "index/scalar/json/JsonValueProjection.h"
#include "segcore/reader/ScalarQueryView.h"
#include "common/RegexQuery.h"
#include "common/StructElementOffsets.h"
#include "segcore/reader/ColumnReader.h"
#include "segcore/json_stats/JsonKeyStats.h"
#include "exec/expression/JsonNumberComparison.h"
#include "exec/expression/Utils.h"
#include "exec/expression/Expr.h"
#include "exec/expression/ScalarArithmetic.h"

namespace milvus::segcore::reader {

namespace {
nlohmann::json
BsonJson(const bson::value_view& value, const ReadContext& context);
nlohmann::json
BsonArray(const bson::array_view& array, const ReadContext& context) {
    auto result = nlohmann::json::array();
    for (const auto& value : array)
        result.push_back(BsonJson(value.get_value(), context));
    return result;
}
nlohmann::json
BsonJson(const bson::value_view& value, const ReadContext& context) {
    context.Check();
    switch (value.type()) {
        case bson::type::k_null:
            return nullptr;
        case bson::type::k_bool:
            return value.get_bool().value;
        case bson::type::k_int32:
            return value.get_int32().value;
        case bson::type::k_int64:
            return value.get_int64().value;
        case bson::type::k_double:
            return value.get_double().value;
        case bson::type::k_string:
            return std::string(value.get_string().value);
        case bson::type::k_array:
            return BsonArray(value.get_array().value, context);
        case bson::type::k_document: {
            auto result = nlohmann::json::object();
            for (const auto& element : value.get_document().view())
                result[std::string(element.key())] =
                    BsonJson(element.get_value(), context);
            return result;
        }
        default:
            ThrowInfo(UnexpectedError,
                      "unsupported BSON type in JSON stats path");
    }
}
}  // namespace
JsonStatsPathResult
ReadJsonStatsPath(index::JsonKeyStats& stats,
                  int64_t count,
                  const JsonStatsPath& path,
                  SelectionView rows,
                  const std::function<int64_t(size_t)>& offset,
                  const ReadContext& context) {
    context.Check();
    AssertInfo(stats.Count() == count && !path.path.empty(),
               "invalid selected stats path/domain");
    VectorResultBuilder output(DataType::JSON, rows.row_count());
    auto uncertain = OwnedSelection::None(rows.row_count());
    if (rows.IsNone())
        return {std::move(output).Finish(rows), std::move(uncertain)};
    std::unordered_map<int64_t, std::vector<size_t>> positions;
    rows.ForEachSelected([&](size_t row) {
        auto source = offset(row);
        AssertInfo(source >= 0 && source < count,
                   "selected stats row outside snapshot");
        positions[source].push_back(row);
    });
    const auto pointer = Json::pointer(path.path);
    if (path.string_sensitive &&
        !stats.GetShreddingField(pointer, index::JSONType::STRING).empty())
        uncertain = OwnedSelection::CopyOf(rows);
    auto emit = [&](int64_t source, size_t prefix, nlohmann::json value) {
        auto found = positions.find(source);
        if (found == positions.end())
            return;
        // Explicit object wrapping preserves numeric object keys; a saved
        // array ancestor itself resolves array indexes in the ordinary kernel.
        for (size_t i = prefix; i > 0; --i) {
            auto object = nlohmann::json::object();
            object[path.path[i - 1]] = std::move(value);
            value = std::move(object);
        }
        auto text = value.dump();
        for (auto row : found->second) {
            output.Set<Json>(row, Json(simdjson::padded_string(text)));
            uncertain.SetSelected(row, false);
        }
    };
    std::vector<int32_t> offsets;
    offsets.reserve(positions.size());
    for (const auto& [row, batch_rows] : positions) {
        AssertInfo(row <= std::numeric_limits<int32_t>::max(),
                   "stats offset exceeds INT32 domain");
        offsets.push_back(static_cast<int32_t>(row));
    }
    std::sort(offsets.begin(), offsets.end());
    auto shred = [&]<typename T>(const std::string& pointer,
                                 index::JSONType type,
                                 size_t prefix) {
        auto field = stats.GetShreddingField(pointer, type);
        if (field.empty())
            return;
        auto column = stats.GetShreddingColumn(field);
        AssertInfo(column && column->NumRows() == count,
                   "selected stats column domain mismatch");
        auto values = column->Take(
            context.op_context.get(),
            TakeOptions{OffsetView::From(offsets.data(), offsets.size()),
                        TargetTypeOf<T>(),
                        {}});
        AssertInfo(values && values->Size() == offsets.size(),
                   "selected stats Take shape mismatch");
        for (size_t i = 0; i < offsets.size(); ++i) {
            context.Check();
            auto value = values->template Get<T>(i);
            AssertInfo(!value.data_skipped,
                       "stats projection cannot use skipped payload");
            if (!value.is_valid)
                continue;
            AssertInfo(value.value, "valid stats Take omitted its value");
            if constexpr (std::is_same_v<T, std::string_view>) {
                if (type == index::JSONType::ARRAY) {
                    BsonView bson(
                        reinterpret_cast<const uint8_t*>(value.value->data()),
                        value.value->size());
                    auto array = bson.ParseAsArrayAtOffset(0);
                    AssertInfo(array, "invalid shredded BSON array");
                    emit(offsets[i], prefix, BsonArray(*array, context));
                } else
                    emit(offsets[i], prefix, std::string(*value.value));
            } else
                emit(offsets[i], prefix, *value.value);
        }
    };
    shred.template operator()<bool>(
        pointer, index::JSONType::BOOL, path.path.size());
    shred.template operator()<int64_t>(
        pointer, index::JSONType::INT64, path.path.size());
    shred.template operator()<double>(
        pointer, index::JSONType::DOUBLE, path.path.size());
    shred.template operator()<std::string_view>(
        pointer, index::JSONType::STRING, path.path.size());
    PinWrapper<index::BsonInvertedIndex*> bson(nullptr);
    for (size_t prefix = path.path.size(); prefix > 0; --prefix) {
        const auto parent = Json::pointer(std::vector<std::string>(
            path.path.begin(), path.path.begin() + prefix));
        shred.template operator()<std::string_view>(
            parent, index::JSONType::ARRAY, prefix);
        stats.ExecuteForSharedData(
            context.op_context.get(),
            bson,
            parent,
            [&](BsonView document, uint32_t row, uint32_t offset) {
                context.Check();
                AssertInfo(row < count, "shared stats row outside snapshot");
                if (!positions.count(row))
                    return;
                if (auto array = document.ParseAsArrayAtOffset(offset))
                    emit(row, prefix, BsonArray(*array, context));
                else if (prefix == path.path.size()) {
                    if (auto v = document.ParseAsValueAtOffset<bool>(offset))
                        emit(row, prefix, *v);
                    else if (auto v =
                                 document.ParseAsValueAtOffset<int64_t>(offset))
                        emit(row, prefix, *v);
                    else if (auto v =
                                 document.ParseAsValueAtOffset<double>(offset))
                        emit(row, prefix, *v);
                    else if (auto v =
                                 document
                                     .ParseAsValueAtOffset<std::string_view>(
                                         offset))
                        emit(row, prefix, std::string(*v));
                }
            },
            [&](uint32_t row) {
                context.Check();
                return positions.count(row) != 0;
            });
    }
    context.Check();
    return {std::move(output).Finish(rows), std::move(uncertain)};
}

IndexResult
NgramIndexReader::Query(const ReadContext& context) const {
    context.Check();
    // Validate at evaluation, including all-NULL batches. Invalid patterns must
    // not disappear merely because the candidate phase returns no hits.
    if (query_.op == proto::plan::Match) {
        LikePatternMatcher validate(query_.pattern);
    } else if (query_.op == proto::plan::RegexMatch) {
        PartialRegexMatcher validate(query_.pattern);
    }
    auto owner = resource_->CreateReader(context.op_context.get());
    auto* index = dynamic_cast<const index::INgramReader*>(owner.get());
    AssertInfo(index && owner->Count() == count_,
               "ngram index snapshot domain mismatch");
    TargetBitmap candidates(count_, true);
    if (count_ &&
        index->CanHandle(query_.pattern, exec::ToIndexPatternOp(query_.op)))
        index->Candidates(
            query_.pattern, exec::ToIndexPatternOp(query_.op), candidates);
    auto valid = NullReader(*owner).IsNotNull();
    AssertInfo(valid.size() == count_, "ngram validity domain mismatch");
    candidates &= valid;
    auto nulls = valid.clone();
    nulls.flip();
    auto negatives = candidates.clone();
    negatives.flip();
    negatives &= valid;
    context.Check();
    return IndexResult::Candidates(SelectionView::FromBitmap(candidates),
                                   SelectionView::FromBitmap(negatives),
                                   SelectionView::FromBitmap(nulls));
}

bool
JsonStatsArithmetic::CanEvaluateWholeSource() const {
    const bool integral = operand.has_int64_val() && bound.has_int64_val();
    const bool floating = operand.has_float_val() && bound.has_float_val() &&
                          std::isfinite(operand.float_val()) &&
                          std::isfinite(bound.float_val());
    if (!integral && !floating)
        return false;
    if (operation == proto::plan::Add || operation == proto::plan::Sub ||
        operation == proto::plan::Mul)
        return true;
    if (operation != proto::plan::Div && operation != proto::plan::Mod)
        return false;
    // Row-specific INT64_MIN / -1 errors cannot be raised outside selection.
    return integral ? operand.int64_val() != 0 && operand.int64_val() != -1
                    : operand.float_val() != 0;
}

namespace {
template <typename T>
T
Value(const proto::plan::GenericValue& value) {
    if constexpr (std::is_same_v<T, std::string>)
        return value.string_val();
    else if constexpr (std::is_same_v<T, bool>)
        return value.bool_val();
    else if constexpr (std::is_integral_v<T>)
        return static_cast<T>(value.int64_val());
    else
        return static_cast<T>(value.has_int64_val()
                                  ? static_cast<double>(value.int64_val())
                                  : value.float_val());
}
template <typename T>
IndexResult
QueryScalar(const index::IIndexReaderBase& source,
            int64_t count,
            const TypedIndexQuery& q,
            const ReadContext& context) {
    ScalarQueryView<T> index(source);
    AssertInfo(source.Caps().predicate &&
                   (source.CoordDomain() == index::Domain::Element) ==
                       q.element_domain,
               "prepared index has incompatible backend or address domain");
    AssertInfo(source.Count() == count,
               "scalar index snapshot domain mismatch");
    using Op = ScalarIndexOp;
    auto matches = [&]() -> TargetBitmap {
        const T lower = Value<T>(q.lower), upper = Value<T>(q.upper);
        switch (q.op) {
            case Op::Equal:
                return index.In(1, &lower);
            case Op::NotEqual:
                return index.NotIn(1, &lower);
            case Op::Less:
                return index.Range(lower, OpType::LessThan);
            case Op::LessEqual:
                return index.Range(lower, OpType::LessEqual);
            case Op::Greater:
                return index.Range(lower, OpType::GreaterThan);
            case Op::GreaterEqual:
                return index.Range(lower, OpType::GreaterEqual);
            case Op::Between:
                return index.Range(
                    lower, q.lower_inclusive, upper, q.upper_inclusive);
            case Op::IsNull:
                return index.IsNull();
            case Op::IsNotNull:
                return index.IsNotNull();
            case Op::In:
            case Op::NotIn: {
                auto values = std::make_unique<T[]>(q.terms.size());
                for (size_t i = 0; i < q.terms.size(); ++i)
                    values[i] = Value<T>(q.terms[i]);
                if (q.op == Op::In)
                    return index.In(q.terms.size(), values.get());
                return index.NotIn(q.terms.size(), values.get());
            }
            default:
                ThrowInfo(UnexpectedError, "invalid prepared scalar operation");
        }
    }();
    auto valid = (q.op == Op::IsNull || q.op == Op::IsNotNull)
                     ? TargetBitmap(count, true)
                     : index.IsNotNull();
    AssertInfo(matches.size() == count && valid.size() == count,
               "scalar predicate returned incomplete domain");
    context.Check();
    return IndexResult::Full(OwnedSelection::OwnBitmap(std::move(matches)),
                             OwnedSelection::OwnBitmap(std::move(valid)));
}
}  // namespace

IndexResult
QueryTypedIndex(const index::IIndexReaderBase& source,
                int64_t count,
                const TypedIndexQuery& query,
                const ReadContext& context) {
    context.Check();
    if (query.op == ScalarIndexOp::Exists) {
        AssertInfo(!query.pointer.empty() && source.Caps().json_paths &&
                       source.CoordDomain() == index::Domain::Row &&
                       source.Count() == count,
                   "invalid prepared JSON existence source");
        // Exists tracks nonempty path presence, including cast failures. An
        // IsNotNull mask would incorrectly discard values of another type.
        const auto* json =
            dynamic_cast<const index::IJsonIndexReader*>(&source);
        AssertInfo(json && !json->CastTypesOf(query.pointer).empty(),
                   "invalid prepared JSON path");
        auto matches = json->Exists(query.pointer);
        AssertInfo(matches.size() == count,
                   "JSON existence index returned incomplete domain");
        context.Check();
        return IndexResult::Full(OwnedSelection::OwnBitmap(std::move(matches)),
                                 OwnedSelection::All(count));
    }
    index::JsonResolvedReader resolved;
    const index::IIndexReaderBase* scalar = &source;
    if (!query.pointer.empty()) {
        const auto* json =
            dynamic_cast<const index::IJsonIndexReader*>(&source);
        AssertInfo(json, "prepared JSON index does not expose path resolution");
        for (const auto& cast : json->CastTypesOf(query.pointer)) {
            if (index::json::IsDataTypeSupported(cast, query.type, false)) {
                resolved = json->Resolve(query.pointer, cast);
                if (resolved)
                    break;
            }
        }
        AssertInfo(resolved, "prepared JSON path is unavailable");
        scalar = resolved.get();
    }
    auto run = [&]<typename T>() -> IndexResult {
        return QueryScalar<T>(*scalar, count, query, context);
    };
    // Path DOUBLE casts accept integer literals only after preparation has
    // ruled out precision-sensitive bounds.
    if (!query.pointer.empty() && scalar->ValueType() == DataType::DOUBLE)
        return run.template operator()<double>();
    switch (query.type) {
        case DataType::BOOL:
            return run.template operator()<bool>();
        case DataType::INT8:
            return run.template operator()<int8_t>();
        case DataType::INT16:
            return run.template operator()<int16_t>();
        case DataType::INT32:
            return run.template operator()<int32_t>();
        case DataType::INT64:
            return run.template operator()<int64_t>();
        case DataType::FLOAT:
            return run.template operator()<float>();
        case DataType::DOUBLE:
            return run.template operator()<double>();
        case DataType::STRING:
        case DataType::VARCHAR:
        case DataType::TEXT:
            return run.template operator()<std::string>();
        default:
            ThrowInfo(UnexpectedError, "invalid prepared scalar type");
    }
}

IndexResult
QueryArrayIndex(const index::IIndexReaderBase& source,
                int64_t count,
                const TypedIndexQuery& query,
                const IStructElementOffsets* offsets,
                const ColumnReader* row_validity,
                const ReadContext& context) {
    context.Check();
    AssertInfo(count >= 0 && (source.CoordDomain() == index::Domain::Element) ==
                                 query.nested_array,
               "array index address domain mismatch");
    AssertInfo(source.Caps().predicate,
               "prepared array index lacks predicates");
    const int64_t source_count =
        query.nested_array ? (offsets ? offsets->GetTotalElementCount() : -1)
                           : count;
    AssertInfo(source.Count() == source_count &&
                   (!query.nested_array ||
                    (offsets && offsets->GetRowCount() == count)),
               "array index snapshot domain mismatch");
    TargetBitmap valid(count, true);
    if (query.nested_array && query.array_nullable) {
        AssertInfo(row_validity && row_validity->type() == DataType::ARRAY &&
                       row_validity->row_count() == count,
                   "nested array index requires parent row validity");
        valid.reset();
        // Empty arrays and NULL parents both contribute zero elements. Their
        // distinction cannot be recovered from the element index's validity.
        for (int64_t begin = 0; begin < count; begin += 8192) {
            context.Check();
            const size_t size = std::min<int64_t>(8192, count - begin);
            auto rows = SelectionView::All(size);
            auto result = row_validity->ReadRange(
                RangeRequest(begin, size, ReadProjection::ValidityOnly, rows),
                context);
            result.Validate(
                DataType::ARRAY, rows, ReadProjection::ValidityOnly);
            result.valid.view().ForEachSelected(
                [&](size_t row) { valid.set(begin + row); });
        }
    }
    auto run = [&]<typename T>() -> IndexResult {
        ScalarQueryView<T> typed_view(source);
        auto* typed = &typed_view;
        if (!query.nested_array) {
            valid = typed->IsNotNull();
            AssertInfo(valid.size() == count,
                       "array index validity domain mismatch");
        }
        auto reduce = [&](TargetBitmap hits) {
            context.Check();
            AssertInfo(hits.size() == source_count,
                       "array index returned incomplete domain");
            if (!query.nested_array)
                return hits;
            return offsets->ForEachRowElementRange(
                [&](int32_t begin, int32_t end) {
                    context.Check();
                    AssertInfo(
                        begin >= 0 && end >= begin && end <= source_count,
                        "array element range exceeds index domain");
                    for (int32_t element = begin; element < end; ++element) {
                        if ((element - begin) % 8192 == 0)
                            context.Check();
                        if (hits[element])
                            return true;
                    }
                    return false;
                },
                0,
                count);
        };
        TargetBitmap matches(count, false);
        switch (query.op) {
            case ScalarIndexOp::ContainsAny: {
                if (!query.terms.empty()) {
                    auto terms = std::make_unique<T[]>(query.terms.size());
                    for (size_t i = 0; i < query.terms.size(); ++i)
                        terms[i] = Value<T>(query.terms[i]);
                    matches =
                        reduce(typed->In(query.terms.size(), terms.get()));
                }
                break;
            }
            case ScalarIndexOp::ContainsAll:
                matches.set();
                for (const auto& term : query.terms) {
                    context.Check();
                    const T value = Value<T>(term);
                    // Different values may occur at different element IDs in
                    // one row. Intersect only after reducing each term to rows.
                    matches &= reduce(typed->In(1, &value));
                }
                break;
            case ScalarIndexOp::IsNull:
                matches = valid.clone();
                matches.flip();
                valid.set();
                break;
            case ScalarIndexOp::IsNotNull:
                matches = valid.clone();
                valid.set();
                break;
            default:
                ThrowInfo(UnexpectedError, "invalid prepared array operation");
        }
        matches &= valid;
        context.Check();
        return IndexResult::Full(OwnedSelection::OwnBitmap(std::move(matches)),
                                 OwnedSelection::OwnBitmap(std::move(valid)));
    };
    switch (query.type) {
        case DataType::BOOL:
            return run.template operator()<bool>();
        case DataType::INT8:
            return run.template operator()<int8_t>();
        case DataType::INT16:
            return run.template operator()<int16_t>();
        case DataType::INT32:
            return run.template operator()<int32_t>();
        case DataType::INT64:
            return run.template operator()<int64_t>();
        case DataType::FLOAT:
            return run.template operator()<float>();
        case DataType::DOUBLE:
            return run.template operator()<double>();
        case DataType::STRING:
        case DataType::VARCHAR:
            return run.template operator()<std::string>();
        default:
            ThrowInfo(UnexpectedError, "invalid prepared array element type");
    }
}

namespace {
bool
BsonContainsValue(const bson::value_view& value,
                  const proto::plan::GenericValue& term,
                  bool same_type) {
    using Value = proto::plan::GenericValue;
    switch (term.val_case()) {
        case Value::kBoolVal: {
            auto v = BsonView::GetValueFromBsonView<bool>(value);
            return v && *v == term.bool_val();
        }
        case Value::kInt64Val: {
            if (value.type() == bson::type::k_int32)
                return value.get_int32().value == term.int64_val();
            if (value.type() == bson::type::k_int64)
                return value.get_int64().value == term.int64_val();
            if (value.type() != bson::type::k_double)
                return false;
            const auto number = value.get_double().value;
            if (!same_type)
                return number == term.int64_val();
            auto exact = exec::DoubleToInt64Exact(number);
            return exact && *exact == term.int64_val();
        }
        case Value::kFloatVal:
            if (value.type() == bson::type::k_int32)
                return double(value.get_int32().value) == term.float_val();
            if (value.type() == bson::type::k_int64)
                return double(value.get_int64().value) == term.float_val();
            return value.type() == bson::type::k_double &&
                   value.get_double().value == term.float_val();
        case Value::kStringVal: {
            auto v = BsonView::GetValueFromBsonView<std::string_view>(value);
            return v && *v == term.string_val();
        }
        case Value::kArrayVal: {
            auto v = BsonView::GetValueFromBsonView<bson::array_view>(value);
            return v && exec::CompareTwoJsonArray(*v, term.array_val());
        }
        default:
            ThrowInfo(UnexpectedError, "invalid prepared JSON contains term");
    }
}

bool
BsonContains(const bson::array_view& array,
             const TypedIndexQuery& query,
             const ReadContext& context,
             std::vector<uint8_t>& found) {
    const bool all = query.op == ScalarIndexOp::ContainsAll;
    if (all)
        std::fill(found.begin(), found.end(), 0);
    size_t remaining = query.terms.size();
    for (const auto& element : array) {
        context.Check();
        for (size_t i = 0; i < query.terms.size(); ++i) {
            if ((!all || !found[i]) &&
                BsonContainsValue(
                    element.get_value(), query.terms[i], query.same_type)) {
                if (!all)
                    return true;
                found[i] = true;
                if (--remaining == 0)
                    return true;
            }
        }
    }
    return all && remaining == 0;
}

bool
MatchLength(int64_t length,
            const proto::plan::GenericValue& bound,
            proto::plan::OpType op) {
    auto compare = [&](auto target) {
        switch (op) {
            case proto::plan::Equal:
                return length == target;
            case proto::plan::NotEqual:
                return length != target;
            case proto::plan::LessThan:
                return length < target;
            case proto::plan::LessEqual:
                return length <= target;
            case proto::plan::GreaterThan:
                return length > target;
            case proto::plan::GreaterEqual:
                return length >= target;
            default:
                ThrowInfo(UnexpectedError, "invalid stats length comparison");
        }
    };
    return bound.has_int64_val() ? compare(bound.int64_val())
                                 : compare(bound.float_val());
}

IndexResult
QueryCollectionStats(index::JsonKeyStats& stats,
                     int64_t count,
                     const TypedIndexQuery& query,
                     const ReadContext& context,
                     const proto::plan::GenericValue* length_bound = nullptr,
                     proto::plan::OpType length_op = proto::plan::Equal) {
    // Shredding scan and shared-column visitors execute synchronously. Keep
    // ALL scratch local to this query invocation and reuse it across rows.
    std::vector<uint8_t> found(!length_bound &&
                                       query.op == ScalarIndexOp::ContainsAll
                                   ? query.terms.size()
                                   : 0);
    auto match_array = [&](const bson::array_view& array) {
        if (!length_bound)
            return BsonContains(array, query, context, found);
        int64_t length = 0;
        for (const auto& element : array) {
            context.Check();
            ++length;
        }
        return MatchLength(length, *length_bound, length_op);
    };
    const bool exists = query.op == ScalarIndexOp::Exists;
    TargetBitmap matches(count, false), valid(count, exists);
    // EXISTS examines descendant fields as well: an object is nonempty iff
    // one of its descendants is nonempty. Arrays need payload inspection;
    // a valid [] or [null, {}] is still empty under Json::exist semantics.
    auto fields = exists ? stats.GetShreddingFieldsWithPrefix(query.pointer)
                         : stats.GetShreddingFields(query.pointer,
                                                    {index::JSONType::ARRAY});
    for (const auto& field : fields) {
        context.Check();
        const auto type = stats.GetShreddingJsonType(field);
        TargetBitmap hits(count, false), known(count, false);
        if (type == index::JSONType::ARRAY) {
            const auto processed =
                stats.ExecutorForShreddingData<std::string_view>(
                    context.op_context.get(),
                    field,
                    [&](const std::string_view* values,
                        ValidityView validity,
                        size_t n,
                        TargetBitmapView output,
                        TargetBitmapView output_valid) {
                        for (size_t i = 0; i < n; ++i) {
                            context.Check();
                            if (validity && !validity[i])
                                continue;
                            BsonView bson(reinterpret_cast<const uint8_t*>(
                                              values[i].data()),
                                          values[i].size());
                            auto array = bson.ParseAsArrayAtOffset(0);
                            AssertInfo(array,
                                       "shredded array has no BSON array");
                            output_valid[i] = true;
                            output[i] =
                                exists ? !BsonView::IsBsonValueEmpty(*array)
                                       : match_array(*array);
                        }
                    },
                    nullptr,
                    TargetBitmapView(hits),
                    TargetBitmapView(known));
            AssertInfo(processed == count,
                       "shredded collection predicate domain mismatch");
        } else {
            AssertInfo(exists && (type == index::JSONType::BOOL ||
                                  type == index::JSONType::INT64 ||
                                  type == index::JSONType::DOUBLE ||
                                  type == index::JSONType::STRING),
                       "unsupported shredded existence type");
            // ApplyValidDataInChunk clears invalid positions in its input
            // bitmap; valid positions must start set.
            hits.set();
            const auto processed = stats.ExecutorForGettingValid(
                context.op_context.get(), field, TargetBitmapView(hits));
            AssertInfo(processed == count,
                       "shredded existence validity domain mismatch");
        }
        matches |= hits;
        if (!exists)
            valid |= known;
    }
    PinWrapper<index::BsonInvertedIndex*> bson(nullptr);
    stats.ExecuteForSharedData(
        context.op_context.get(),
        bson,
        query.pointer,
        [&](BsonView document, uint32_t row, uint32_t offset) {
            context.Check();
            AssertInfo(row < count, "stats shared row outside snapshot");
            if (exists) {
                // Shared BSON may contain the remaining children of an
                // already-matching shredded object. Never erase that match.
                if (!document.IsBsonValueEmpty(offset))
                    matches.set(row, true);
            } else if (auto array = document.ParseAsArrayAtOffset(offset)) {
                valid.set(row, true);
                matches.set(row, match_array(*array));
            }
        });
    context.Check();
    return IndexResult::Full(OwnedSelection::OwnBitmap(std::move(matches)),
                             OwnedSelection::OwnBitmap(std::move(valid)));
}

template <typename T>
std::optional<bool>
MatchStats(const T& value, const TypedIndexQuery& query) {
    using Op = ScalarIndexOp;
    auto precise_bound = [](const proto::plan::GenericValue& bound) {
        if (bound.has_int64_val())
            return bound.int64_val() <= -(int64_t{1} << 53) ||
                   bound.int64_val() >= (int64_t{1} << 53);
        return bound.has_float_val() && std::abs(bound.float_val()) >= 0x1p53 &&
               std::abs(bound.float_val()) <= 0x1p63;
    };
    const bool precise_range =
        query.op == Op::Between &&
        (precise_bound(query.lower) || precise_bound(query.upper));
    auto compare =
        [&](const proto::plan::GenericValue& bound) -> std::optional<int> {
        if constexpr (std::is_same_v<T, std::string_view>) {
            if (!bound.has_string_val())
                return std::nullopt;
            return value < bound.string_val()   ? -1
                   : value > bound.string_val() ? 1
                                                : 0;
        } else if constexpr (std::is_same_v<T, bool>) {
            if (!bound.has_bool_val())
                return std::nullopt;
            return value < bound.bool_val()   ? -1
                   : value > bound.bool_val() ? 1
                                              : 0;
        } else {
            if (bound.has_float_val() && !precise_range) {
                const double number = static_cast<double>(value);
                return number < bound.float_val()   ? -1
                       : number > bound.float_val() ? 1
                                                    : 0;
            }
            return exec::CompareJsonNumberToBound(value, bound);
        }
    };
    auto lower = compare(query.lower);
    if (!lower)
        return std::nullopt;
    switch (query.op) {
        case Op::Equal:
            return *lower == 0;
        case Op::NotEqual:
            return *lower != 0;
        case Op::Less:
            return *lower < 0;
        case Op::LessEqual:
            return *lower <= 0;
        case Op::Greater:
            return *lower > 0;
        case Op::GreaterEqual:
            return *lower >= 0;
        case Op::Between: {
            auto upper = compare(query.upper);
            if (!upper)
                return std::nullopt;
            return (query.lower_inclusive ? *lower >= 0 : *lower > 0) &&
                   (query.upper_inclusive ? *upper <= 0 : *upper < 0);
        }
        case Op::In:
            for (const auto& term : query.terms) {
                auto result = compare(term);
                if (result && *result == 0)
                    return true;
            }
            return false;
        default:
            ThrowInfo(UnexpectedError, "invalid prepared stats operation");
    }
}

IndexResult
QueryStats(
    index::JsonKeyStats& stats,
    int64_t count,
    const TypedIndexQuery& query,
    const ReadContext& context,
    const std::function<bool(std::string_view)>& string_match = {},
    const JsonStatsArithmetic* arithmetic = nullptr,
    exec::scalar::CompareOp comparison = exec::scalar::CompareOp::Equal) {
    if (query.op == ScalarIndexOp::Exists ||
        query.op == ScalarIndexOp::ContainsAny ||
        query.op == ScalarIndexOp::ContainsAll)
        return QueryCollectionStats(stats, count, query, context);
    TargetBitmap matches(count, false), valid(count, false);
    auto match = [&](auto value) -> std::optional<bool> {
        using T = decltype(value);
        if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, double>) {
            if (arithmetic)
                return exec::scalar::MatchJsonArithmeticNumber(
                    value,
                    arithmetic->operation,
                    comparison,
                    arithmetic->operand,
                    arithmetic->bound);
        }
        return MatchStats(value, query);
    };
    auto shred = [&]<typename T>(index::JSONType type) {
        auto field = stats.GetShreddingField(query.pointer, type);
        if (field.empty())
            return;
        TargetBitmap hits(count, false), known(count, false);
        stats.ExecutorForShreddingData<T>(
            context.op_context.get(),
            field,
            [&](const T* values,
                ValidityView validity,
                size_t n,
                TargetBitmapView output,
                TargetBitmapView output_valid) {
                context.Check();
                for (size_t i = 0; i < n; ++i) {
                    if (validity && !validity[i])
                        continue;
                    auto result = [&]() -> std::optional<bool> {
                        if constexpr (std::is_same_v<T, std::string_view>)
                            if (string_match)
                                return string_match(values[i]);
                        return match(values[i]);
                    }();
                    output_valid[i] = result.has_value();
                    output[i] = result.value_or(false);
                }
            },
            nullptr,
            TargetBitmapView(hits),
            TargetBitmapView(known));
        matches |= hits;
        valid |= known;
    };
    switch (query.type) {
        case DataType::BOOL:
            shred.template operator()<bool>(index::JSONType::BOOL);
            break;
        case DataType::INT64:
        case DataType::DOUBLE:
            shred.template operator()<int64_t>(index::JSONType::INT64);
            shred.template operator()<double>(index::JSONType::DOUBLE);
            break;
        case DataType::VARCHAR:
            shred.template operator()<std::string_view>(
                index::JSONType::STRING);
            break;
        default:
            ThrowInfo(UnexpectedError, "invalid prepared stats type");
    }
    PinWrapper<index::BsonInvertedIndex*> bson(nullptr);
    stats.ExecuteForSharedData(
        context.op_context.get(),
        bson,
        query.pointer,
        [&](BsonView document, uint32_t row, uint32_t offset) {
            context.Check();
            AssertInfo(row < count, "stats shared row outside snapshot");
            std::optional<bool> result;
            if (query.type == DataType::BOOL) {
                if (auto value = document.ParseAsValueAtOffset<bool>(offset))
                    result = MatchStats(*value, query);
            } else if (query.type == DataType::VARCHAR) {
                if (auto value =
                        document.ParseAsValueAtOffset<std::string_view>(offset))
                    result = string_match
                                 ? std::optional<bool>(string_match(*value))
                                 : MatchStats(*value, query);
            } else {
                if (auto integer =
                        document.ParseAsValueAtOffset<int64_t>(offset))
                    result = match(*integer);
                else if (auto floating =
                             document.ParseAsValueAtOffset<double>(offset))
                    result = match(*floating);
            }
            valid.set(row, result.has_value());
            matches.set(row, result.value_or(false));
        });
    context.Check();
    if (string_match) {
        // Historic shredded string columns conflate empty strings and NULL.
        // Keep all unknown rows as candidates so raw refinement can recover
        // the original validity. Only a known non-match proves FALSE.
        auto candidates = OwnedSelection::All(count);
        auto false_rows = OwnedSelection::None(count);
        for (size_t row = 0; row < count; ++row) {
            context.Check();
            if (valid[row] && !matches[row]) {
                candidates.SetSelected(row, false);
                false_rows.SetSelected(row);
            }
        }
        return IndexResult::Candidates(
            candidates.view(), false_rows.view(), SelectionView::None(count));
    }
    return IndexResult::Full(OwnedSelection::OwnBitmap(std::move(matches)),
                             OwnedSelection::OwnBitmap(std::move(valid)));
}
}  // namespace

IndexResult
JsonStatsReader::Query(const ReadContext& context) const {
    context.Check();
    AssertInfo(load_ && count_ >= 0 && !query_.pointer.empty(),
               "invalid JSON stats source binding");
    auto stats = load_(context);
    AssertInfo(stats && stats->Count() == count_,
               "JSON stats snapshot domain mismatch");
    if (query_.arithmetic) {
        AssertInfo(query_.arithmetic->CanEvaluateWholeSource(),
                   "unsafe whole-source stats arithmetic descriptor");
        using Compare = exec::scalar::CompareOp;
        const auto comparison = [&] {
            switch (query_.op) {
                case proto::plan::Equal:
                    return Compare::Equal;
                case proto::plan::NotEqual:
                    return Compare::NotEqual;
                case proto::plan::LessThan:
                    return Compare::Less;
                case proto::plan::LessEqual:
                    return Compare::LessEqual;
                case proto::plan::GreaterThan:
                    return Compare::Greater;
                case proto::plan::GreaterEqual:
                    return Compare::GreaterEqual;
                default:
                    ThrowInfo(UnexpectedError,
                              "invalid stats arithmetic comparison");
            }
        }();
        TypedIndexQuery query{};
        query.type = query_.arithmetic->bound.has_int64_val()
                         ? DataType::INT64
                         : DataType::DOUBLE;
        query.pointer = query_.pointer;
        return QueryStats(*stats,
                          count_,
                          query,
                          context,
                          {},
                          &*query_.arithmetic,
                          comparison);
    }
    if (query_.length_bound) {
        TypedIndexQuery query{};
        query.op = ScalarIndexOp::ContainsAny;
        query.pointer = query_.pointer;
        return QueryCollectionStats(
            *stats, count_, query, context, &*query_.length_bound, query_.op);
    }
    if (query_.string_pattern) {
        TypedIndexQuery query{};
        query.op = ScalarIndexOp::Equal;
        query.type = DataType::VARCHAR;
        query.pointer = query_.pointer;
        auto evaluate = [&](const auto& matcher) {
            return QueryStats(
                *stats, count_, query, context, [&](std::string_view value) {
                    return matcher(value);
                });
        };
        const auto& pattern = *query_.string_pattern;
        if (query_.op == proto::plan::Match)
            return evaluate(LikePatternMatcher(pattern));
        if (query_.op == proto::plan::RegexMatch)
            return evaluate(PartialRegexMatcher(pattern));
        return evaluate([&](std::string_view value) {
            if (query_.op == proto::plan::PrefixMatch)
                return value.starts_with(pattern);
            if (query_.op == proto::plan::PostfixMatch)
                return value.ends_with(pattern);
            AssertInfo(query_.op == proto::plan::InnerMatch,
                       "invalid stats string operation");
            return value.find(pattern) != std::string_view::npos;
        });
    }
    if (query_.predicate)
        return QueryStats(*stats, count_, *query_.predicate, context);
    TypedIndexQuery query{};
    query.pointer = query_.pointer;
    query.type = DataType::INT64;
    query.lower.set_int64_val(query_.bound);
    switch (query_.op) {
        case proto::plan::Equal:
            query.op = ScalarIndexOp::Equal;
            break;
        case proto::plan::NotEqual:
            query.op = ScalarIndexOp::NotEqual;
            break;
        case proto::plan::LessThan:
            query.op = ScalarIndexOp::Less;
            break;
        case proto::plan::LessEqual:
            query.op = ScalarIndexOp::LessEqual;
            break;
        case proto::plan::GreaterThan:
            query.op = ScalarIndexOp::Greater;
            break;
        case proto::plan::GreaterEqual:
            query.op = ScalarIndexOp::GreaterEqual;
            break;
        default:
            ThrowInfo(UnexpectedError,
                      "invalid prepared JSON stats comparison");
    }
    return QueryStats(*stats, count_, query, context);
}

template <typename T>
IntegerIndexReader<T>::IntegerIndexReader(
    std::shared_ptr<const index::IIndexReaderBase> index,
    int64_t row_count,
    Int64IndexQuery query)
    : index_(std::move(index)), row_count_(row_count), query_(query) {
    AssertInfo(index_ && row_count_ >= 0, "invalid scalar index binding");
    if (!index_->Caps().predicate ||
        index_->CoordDomain() != index::Domain::Row)
        ThrowInfo(Unsupported, "scalar index source capability not migrated");
    AssertInfo(index_->Count() == row_count_,
               "scalar index snapshot domain mismatch");
    auto fits = [](int64_t value) {
        return value >= std::numeric_limits<T>::min() &&
               value <= std::numeric_limits<T>::max();
    };
    AssertInfo(fits(query_.lower) && fits(query_.upper),
               "integer index bound is not representable");
    for (auto value : query_.terms)
        AssertInfo(fits(value), "integer index term is not representable");
    switch (query_.op) {
        case ScalarIndexOp::Equal:
        case ScalarIndexOp::NotEqual:
        case ScalarIndexOp::Less:
        case ScalarIndexOp::LessEqual:
        case ScalarIndexOp::Greater:
        case ScalarIndexOp::GreaterEqual:
        case ScalarIndexOp::Between:
        case ScalarIndexOp::IsNull:
        case ScalarIndexOp::IsNotNull:
        case ScalarIndexOp::In:
        case ScalarIndexOp::NotIn:
            break;
        default:
            ThrowInfo(UnexpectedError, "invalid scalar index query");
    }
}
template <typename T>
IndexResult
IntegerIndexReader<T>::Query(const ReadContext& context) const {
    context.Check();
    ScalarQueryView<T> typed(*index_);
    AssertInfo(index_->Count() == row_count_,
               "scalar index generation changed");
    auto matches = [&]() -> TargetBitmap {
        const T lower = static_cast<T>(query_.lower);
        const T upper = static_cast<T>(query_.upper);
        switch (query_.op) {
            case ScalarIndexOp::Equal:
                return typed.In(1, &lower);
            case ScalarIndexOp::NotEqual:
                return typed.NotIn(1, &lower);
            case ScalarIndexOp::Less:
                return typed.Range(lower, OpType::LessThan);
            case ScalarIndexOp::LessEqual:
                return typed.Range(lower, OpType::LessEqual);
            case ScalarIndexOp::Greater:
                return typed.Range(lower, OpType::GreaterThan);
            case ScalarIndexOp::GreaterEqual:
                return typed.Range(lower, OpType::GreaterEqual);
            case ScalarIndexOp::Between:
                return typed.Range(lower,
                                   query_.lower_inclusive,
                                   upper,
                                   query_.upper_inclusive);
            case ScalarIndexOp::IsNull:
                return typed.IsNull();
            case ScalarIndexOp::IsNotNull:
                return typed.IsNotNull();
            case ScalarIndexOp::In:
            case ScalarIndexOp::NotIn: {
                // bool needs a real contiguous array, not vector<bool> bits.
                auto terms = std::make_unique<T[]>(query_.terms.size());
                for (size_t i = 0; i < query_.terms.size(); ++i)
                    terms[i] = static_cast<T>(query_.terms[i]);
                if (query_.op == ScalarIndexOp::In)
                    return typed.In(query_.terms.size(), terms.get());
                return typed.NotIn(query_.terms.size(), terms.get());
            }
            default:
                ThrowInfo(UnexpectedError, "invalid scalar index query");
        }
    }();
    AssertInfo(matches.size() == static_cast<size_t>(row_count_),
               "scalar index returned incomplete row domain");
    context.Check();
    if (query_.op == ScalarIndexOp::IsNull ||
        query_.op == ScalarIndexOp::IsNotNull)
        return IndexResult::Full(OwnedSelection::OwnBitmap(std::move(matches)),
                                 OwnedSelection::All(row_count_));
    auto valid = typed.IsNotNull();
    AssertInfo(valid.size() == static_cast<size_t>(row_count_),
               "scalar index validity has incomplete row domain");
    context.Check();
    return IndexResult::Full(OwnedSelection::OwnBitmap(std::move(matches)),
                             OwnedSelection::OwnBitmap(std::move(valid)));
}
template class IntegerIndexReader<bool>;
template class IntegerIndexReader<int8_t>;
template class IntegerIndexReader<int16_t>;
template class IntegerIndexReader<int32_t>;
template class IntegerIndexReader<int64_t>;
}  // namespace milvus::segcore::reader
