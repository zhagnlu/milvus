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

#include "SqlProjection.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <ctime>
#include <limits>
#include <optional>
#include <unordered_map>
#include <variant>

#include "absl/time/time.h"
#include "re2/re2.h"
#include "common/Json.h"
#include "segcore/Utils.h"
#include "segcore/json_stats/JsonKeyStats.h"

namespace milvus::exec {
namespace {

using SqlValue =
    std::variant<std::monostate, bool, int64_t, double, std::string>;
using SqlExpr = proto::plan::SQLScalarExpr;

// Bind the wire representation once per projection invocation. The owned tree
// keeps child dispatch independent of protobuf strings in the per-row loop;
// source pointers refer to the immutable SQLProject for this invocation.
enum class SqlOp {
    Unknown,
    And,
    Or,
    Not,
    IsNull,
    IsNotNull,
    In,
    Like,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Column,
    Literal,
    JsonPath,
    Cast,
    ToTimestamp,
    Extract,
    UnaryPlus,
    UnaryMinus,
    TimestampDiff,
    Add,
    Subtract,
    Multiply,
    Divide,
};

struct PreparedJsonPath {
    int64_t field_id;
    std::string pointer;
    std::vector<std::string> numeric_parents;
    std::unique_ptr<index::JsonKeyStats::ShreddingReader> string_reader;
    std::unique_ptr<index::JsonKeyStats::ShreddingReader> integer_reader;
};

enum class ExtractUnit {
    Unknown,
    Epoch,
    Hour,
    Minute,
    Second,
    Day,
    Month,
    Year,
    Dow,
    Doy
};

struct PreparedSqlExpr {
    const SqlExpr* source;
    SqlOp op = SqlOp::Unknown;
    std::vector<PreparedSqlExpr> args;
    ExtractUnit unit = ExtractUnit::Unknown;
    mutable std::unique_ptr<PreparedJsonPath> path;
};

PreparedSqlExpr
PrepareSqlExpr(const SqlExpr& expr) {
    static constexpr std::pair<std::string_view, SqlOp> operators[] = {
        {"and", SqlOp::And},
        {"or", SqlOp::Or},
        {"not", SqlOp::Not},
        {"is_null", SqlOp::IsNull},
        {"is_not_null", SqlOp::IsNotNull},
        {"in", SqlOp::In},
        {"like", SqlOp::Like},
        {"=", SqlOp::Equal},
        {"<>", SqlOp::NotEqual},
        {"<", SqlOp::Less},
        {"<=", SqlOp::LessEqual},
        {">", SqlOp::Greater},
        {">=", SqlOp::GreaterEqual},
        {"column", SqlOp::Column},
        {"literal", SqlOp::Literal},
        {"json_path", SqlOp::JsonPath},
        {"cast", SqlOp::Cast},
        {"to_timestamp", SqlOp::ToTimestamp},
        {"extract", SqlOp::Extract},
        {"unary+", SqlOp::UnaryPlus},
        {"unary-", SqlOp::UnaryMinus},
        {"timestamp_diff", SqlOp::TimestampDiff},
        {"+", SqlOp::Add},
        {"-", SqlOp::Subtract},
        {"*", SqlOp::Multiply},
        {"/", SqlOp::Divide},
    };
    PreparedSqlExpr result{&expr};
    for (const auto& [name, op] : operators) {
        if (expr.op() == name) {
            result.op = op;
            break;
        }
    }
    if (result.op == SqlOp::Extract) {
        static constexpr std::pair<std::string_view, ExtractUnit> units[] = {
            {"epoch", ExtractUnit::Epoch},
            {"hour", ExtractUnit::Hour},
            {"minute", ExtractUnit::Minute},
            {"second", ExtractUnit::Second},
            {"day", ExtractUnit::Day},
            {"month", ExtractUnit::Month},
            {"year", ExtractUnit::Year},
            {"dow", ExtractUnit::Dow},
            {"doy", ExtractUnit::Doy}};
        for (const auto& [name, unit] : units) {
            if (expr.time_unit() == name) {
                result.unit = unit;
                break;
            }
        }
    }
    result.args.reserve(expr.args_size());
    for (const auto& arg : expr.args()) {
        result.args.emplace_back(PrepareSqlExpr(arg));
    }
    return result;
}

bool
IsNull(const SqlValue& value) {
    return std::holds_alternative<std::monostate>(value);
}

double
Number(const SqlValue& value) {
    if (const auto* integer = std::get_if<int64_t>(&value)) {
        return static_cast<double>(*integer);
    }
    if (const auto* number = std::get_if<double>(&value)) {
        return *number;
    }
    ThrowInfo(ExprInvalid, "SQL arithmetic requires a numeric value");
}

double
Finite(double value) {
    if (!std::isfinite(value)) {
        ThrowInfo(ExprInvalid, "SQL floating point overflow");
    }
    return value;
}

std::string_view
Trim(std::string_view value) {
    const auto begin = value.find_first_not_of(" \t\r\n\f\v");
    if (begin == std::string_view::npos) {
        return {};
    }
    const auto end = value.find_last_not_of(" \t\r\n\f\v");
    return value.substr(begin, end - begin + 1);
}

template <typename T>
T
ParseNumber(const std::string& value) {
    auto text = Trim(value);
    if (!text.empty() && text.front() == '+') {
        text.remove_prefix(1);
    }
    T number{};
    const auto result =
        std::from_chars(text.data(), text.data() + text.size(), number);
    if (text.empty() || result.ec != std::errc{} ||
        result.ptr != text.data() + text.size()) {
        ThrowInfo(ExprInvalid, "invalid SQL numeric cast");
    }
    return number;
}

int64_t
Integer(const SqlValue& value) {
    if (const auto* integer = std::get_if<int64_t>(&value)) {
        return *integer;
    }
    if (const auto* text = std::get_if<std::string>(&value)) {
        return ParseNumber<int64_t>(*text);
    }
    if (const auto* boolean = std::get_if<bool>(&value)) {
        return *boolean ? 1 : 0;
    }
    const auto number = std::nearbyint(Finite(Number(value)));
    // 2^63 is representable as double, INT64_MAX is not.
    if (number < -9223372036854775808.0 || number >= 9223372036854775808.0) {
        ThrowInfo(ExprInvalid, "SQL integer cast overflow");
    }
    return static_cast<int64_t>(number);
}

SqlValue
Cast(const SqlValue& value, DataType type) {
    if (IsNull(value)) {
        return {};
    }
    switch (type) {
        case DataType::INT8:
        case DataType::INT16:
        case DataType::INT32:
        case DataType::INT64:
        case DataType::TIMESTAMPTZ: {
            auto integer = Integer(value);
            if ((type == DataType::INT8 &&
                 (integer < INT8_MIN || integer > INT8_MAX)) ||
                (type == DataType::INT16 &&
                 (integer < INT16_MIN || integer > INT16_MAX)) ||
                (type == DataType::INT32 &&
                 (integer < INT32_MIN || integer > INT32_MAX))) {
                ThrowInfo(ExprInvalid, "SQL integer cast overflow");
            }
            return integer;
        }
        case DataType::FLOAT:
        case DataType::DOUBLE: {
            const auto* text = std::get_if<std::string>(&value);
            auto number =
                Finite(text ? ParseNumber<double>(*text) : Number(value));
            if (type == DataType::FLOAT) {
                number = Finite(static_cast<float>(number));
            }
            return number;
        }
        case DataType::VARCHAR:
        case DataType::STRING:
            if (const auto* text = std::get_if<std::string>(&value)) {
                return *text;
            }
            if (const auto* boolean = std::get_if<bool>(&value)) {
                return std::string(*boolean ? "true" : "false");
            }
            if (const auto* integer = std::get_if<int64_t>(&value)) {
                return std::to_string(*integer);
            }
            return fmt::format("{}", Finite(Number(value)));
        case DataType::BOOL:
            if (const auto* boolean = std::get_if<bool>(&value)) {
                return *boolean;
            }
            if (const auto* integer = std::get_if<int64_t>(&value)) {
                return *integer != 0;
            }
            if (const auto* text = std::get_if<std::string>(&value)) {
                auto trimmed = std::string(Trim(*text));
                std::transform(
                    trimmed.begin(),
                    trimmed.end(),
                    trimmed.begin(),
                    [](unsigned char ch) { return std::tolower(ch); });
                if (trimmed == "true" || trimmed == "t" || trimmed == "1") {
                    return true;
                }
                if (trimmed == "false" || trimmed == "f" || trimmed == "0") {
                    return false;
                }
            }
            ThrowInfo(ExprInvalid, "invalid SQL boolean cast");
        default:
            ThrowInfo(ExprInvalid, "unsupported SQL cast type {}", type);
    }
}

SqlValue
ReadValue(const ColumnVectorPtr& column, size_t row) {
    if (!column->ValidAt(row)) {
        return {};
    }
    switch (column->type()) {
        case DataType::BOOL:
            return column->ValueAt<bool>(row);
        case DataType::INT8:
            return int64_t(column->ValueAt<int8_t>(row));
        case DataType::INT16:
            return int64_t(column->ValueAt<int16_t>(row));
        case DataType::INT32:
            return int64_t(column->ValueAt<int32_t>(row));
        case DataType::INT64:
        case DataType::TIMESTAMPTZ:
            return column->ValueAt<int64_t>(row);
        case DataType::FLOAT:
            return double(column->ValueAt<float>(row));
        case DataType::DOUBLE:
            return column->ValueAt<double>(row);
        case DataType::VARCHAR:
        case DataType::STRING:
            return column->ValueAt<std::string>(row);
        case DataType::JSON:
            return std::string(column->RawAsValues<Json>()[row].data());
        default:
            ThrowInfo(
                ExprInvalid, "unsupported SQL source type {}", column->type());
    }
}

void
WriteValue(const ColumnVectorPtr& column, size_t row, SqlValue value) {
    if (IsNull(value)) {
        column->nullAt(row);
        return;
    }
    switch (column->type()) {
        case DataType::BOOL:
            column->SetValueAt<bool>(row, std::get<bool>(value));
            break;
        case DataType::INT8:
            column->SetValueAt<int8_t>(row, std::get<int64_t>(value));
            break;
        case DataType::INT16:
            column->SetValueAt<int16_t>(row, std::get<int64_t>(value));
            break;
        case DataType::INT32:
            column->SetValueAt<int32_t>(row, std::get<int64_t>(value));
            break;
        case DataType::INT64:
        case DataType::TIMESTAMPTZ:
            column->SetValueAt<int64_t>(row, std::get<int64_t>(value));
            break;
        case DataType::FLOAT:
            column->SetValueAt<float>(row, std::get<double>(value));
            break;
        case DataType::DOUBLE:
            column->SetValueAt<double>(row, std::get<double>(value));
            break;
        case DataType::VARCHAR:
        case DataType::STRING:
            // Output columns own these strings. Transfer temporary expression values
            // instead of allocating a second copy through SetValueAt(const T&).
            column->RawAsValues<std::string>()[row] =
                std::move(std::get<std::string>(value));
            break;
        default:
            ThrowInfo(ExprInvalid,
                      "unsupported SQL projection type {}",
                      column->type());
    }
}

class SqlProjectionEvaluator {
 public:
    SqlProjectionEvaluator(const std::vector<int64_t>& offsets,
                           const segcore::SegmentInternalInterface* segment,
                           OpContext* op_context,
                           const std::string& timezone,
                           bool use_json_stats)
        : offsets_(offsets),
          segment_(segment),
          schema_(segment->get_schema_snapshot()),
          op_context_(op_context),
          timezone_(timezone),
          use_json_stats_(use_json_stats) {
    }

    SqlValue
    Eval(const PreparedSqlExpr& prepared, size_t row) {
        const auto& expr = *prepared.source;
        const auto op = prepared.op;
        const auto type = static_cast<DataType>(expr.data_type());
        if (op == SqlOp::And || op == SqlOp::Or || op == SqlOp::Not ||
            op == SqlOp::IsNull || op == SqlOp::IsNotNull || op == SqlOp::In ||
            op == SqlOp::Like || op == SqlOp::Equal || op == SqlOp::NotEqual ||
            op == SqlOp::Less || op == SqlOp::LessEqual ||
            op == SqlOp::Greater || op == SqlOp::GreaterEqual) {
            return Predicate(prepared, row);
        }
        if (op == SqlOp::Column) {
            if (type == DataType::JSON) {
                auto value = JsonSource(expr.source_field_id(), row);
                return value ? SqlValue(std::string(value->data()))
                             : SqlValue{};
            }
            return ReadValue(Source(expr.source_field_id(),
                                    static_cast<DataType>(expr.data_type())),
                             row);
        }
        if (op == SqlOp::Literal) {
            if (!expr.has_literal()) {
                return {};
            }
            const auto& literal = expr.literal();
            switch (literal.val_case()) {
                case proto::plan::GenericValue::kInt64Val:
                    return literal.int64_val();
                case proto::plan::GenericValue::kFloatVal:
                    return Finite(literal.float_val());
                case proto::plan::GenericValue::kBoolVal:
                    return literal.bool_val();
                case proto::plan::GenericValue::kStringVal:
                    return literal.string_val();
                default:
                    ThrowInfo(ExprInvalid, "invalid SQL literal");
            }
        }
        if (op == SqlOp::JsonPath) {
            return JsonPath(prepared, row);
        }
        if (expr.args_size() < 1) {
            ThrowInfo(ExprInvalid, "missing SQL expression operand");
        }
        // Preserve ->> text semantics generally; a surrounding integer cast
        // lets an INT64 shredding value bypass text formatting and reparsing.
        if (op == SqlOp::Cast &&
            (type == DataType::INT8 || type == DataType::INT16 ||
             type == DataType::INT32 || type == DataType::INT64) &&
            prepared.args[0].op == SqlOp::JsonPath &&
            expr.args(0).data_type() == proto::schema::VarChar) {
            auto& path = PrepareJsonPath(prepared.args[0]);
            if (path.integer_reader) {
                auto value =
                    path.integer_reader->Get(op_context_, offsets_[row]);
                if (value && std::holds_alternative<int64_t>(*value)) {
                    return Cast(std::get<int64_t>(*value), type);
                }
            }
        }
        auto left = Eval(prepared.args[0], row);
        if (IsNull(left)) {
            return {};
        }
        if (op == SqlOp::Cast) {
            return Cast(left, type);
        }
        if (op == SqlOp::ToTimestamp) {
            auto micros = static_cast<long double>(Number(left)) * 1000000.0L;
            if (!std::isfinite(micros) || micros < INT64_MIN ||
                micros > INT64_MAX) {
                ThrowInfo(ExprInvalid, "SQL timestamp out of range");
            }
            return static_cast<int64_t>(std::round(micros));
        }
        if (op == SqlOp::Extract) {
            return Extract(prepared, left);
        }
        if (op == SqlOp::UnaryPlus) {
            return left;
        }
        if (op == SqlOp::UnaryMinus) {
            if (const auto* integer = std::get_if<int64_t>(&left)) {
                if (*integer == INT64_MIN) {
                    ThrowInfo(ExprInvalid, "SQL integer arithmetic overflow");
                }
                return -*integer;
            }
            return Finite(-Number(left));
        }
        if (expr.args_size() != 2) {
            ThrowInfo(ExprInvalid,
                      "SQL binary expression requires two operands");
        }
        auto right = Eval(prepared.args[1], row);
        if (IsNull(right)) {
            return {};
        }
        if (op == SqlOp::TimestampDiff) {
            return double(
                (static_cast<long double>(Integer(left)) - Integer(right)) /
                1000000.0L);
        }
        if (type == DataType::INT64) {
            auto a = Integer(left), b = Integer(right);
            int64_t result;
            bool overflow = false;
            if (op == SqlOp::Add)
                overflow = __builtin_add_overflow(a, b, &result);
            else if (op == SqlOp::Subtract)
                overflow = __builtin_sub_overflow(a, b, &result);
            else if (op == SqlOp::Multiply)
                overflow = __builtin_mul_overflow(a, b, &result);
            else if (op == SqlOp::Divide) {
                if (b == 0)
                    ThrowInfo(ExprInvalid, "SQL division by zero");
                overflow = a == INT64_MIN && b == -1;
                if (!overflow)
                    result = a / b;
            } else
                ThrowInfo(
                    ExprInvalid, "unsupported SQL operator {}", expr.op());
            if (overflow)
                ThrowInfo(ExprInvalid, "SQL integer arithmetic overflow");
            return result;
        }
        auto a = Number(left), b = Number(right);
        if (op == SqlOp::Add)
            return Finite(a + b);
        if (op == SqlOp::Subtract)
            return Finite(a - b);
        if (op == SqlOp::Multiply)
            return Finite(a * b);
        if (op == SqlOp::Divide) {
            if (b == 0)
                ThrowInfo(ExprInvalid, "SQL division by zero");
            return Finite(a / b);
        }
        ThrowInfo(ExprInvalid, "unsupported SQL operator {}", expr.op());
    }

    // A column kernel for direct text paths. Typed validity is not SQL NULL:
    // collect exactly the missing offsets and fetch their raw JSON in bounded
    // batches. Other expression shapes retain lazy scalar evaluation.
    bool
    ProjectJsonPath(const PreparedSqlExpr& prepared,
                    const std::vector<size_t>& selected,
                    const ColumnVectorPtr& column) {
        const auto& expr = *prepared.source;
        auto& path = PrepareJsonPath(prepared);
        if (!path.string_reader) {
            return false;
        }
        std::vector<int64_t> missing_offsets;
        std::vector<size_t> missing_rows;
        constexpr size_t batch_size = 1024;
        std::array<int64_t, batch_size> row_ids;
        std::array<std::optional<std::string_view>, batch_size> values;
        auto* output = column->RawAsValues<std::string>();
        for (size_t begin = 0; begin < selected.size(); begin += batch_size) {
            segcore::CheckCancellation(
                op_context_, segment_->get_segment_id(), "SQL JSON projection");
            const auto count = std::min(batch_size, selected.size() - begin);
            for (size_t i = 0; i < count; ++i) {
                row_ids[i] = offsets_[selected[begin + i]];
            }
            path.string_reader->GetStringViews(
                op_context_, {row_ids.data(), count}, {values.data(), count});
            for (size_t i = 0; i < count; ++i) {
                const auto row = begin + i;
                if (values[i]) {
                    // Copy into the owning result once; the reader keeps the
                    // borrowed input pinned through this batch and fallback.
                    output[row].assign(values[i]->data(), values[i]->size());
                } else {
                    missing_offsets.push_back(row_ids[i]);
                    missing_rows.push_back(row);
                }
            }
        }
        for (size_t begin = 0; begin < missing_offsets.size();
             begin += batch_size) {
            segcore::CheckCancellation(
                op_context_, segment_->get_segment_id(), "SQL JSON fallback");
            const auto count =
                std::min(batch_size, missing_offsets.size() - begin);
            auto raw = ReadSource(path.field_id,
                                  DataType::JSON,
                                  missing_offsets.data() + begin,
                                  count);
            for (size_t i = 0; i < count; ++i) {
                const Json* source =
                    raw->ValidAt(i) ? &raw->RawAsValues<Json>()[i] : nullptr;
                WriteValue(column,
                           missing_rows[begin + i],
                           ReadJsonPath(expr, path.pointer, source));
            }
        }
        return true;
    }

 private:
    int
    Compare(const SqlValue& left, const SqlValue& right) {
        if (const auto* a = std::get_if<int64_t>(&left)) {
            if (const auto* b = std::get_if<int64_t>(&right)) {
                return *a < *b ? -1 : (*a > *b ? 1 : 0);
            }
        }
        if (const auto* a = std::get_if<std::string>(&left)) {
            return a->compare(std::get<std::string>(right));
        }
        if (const auto* a = std::get_if<bool>(&left)) {
            const auto b = std::get<bool>(right);
            return *a == b ? 0 : (*a ? 1 : -1);
        }
        const auto a = Number(left), b = Number(right);
        if (std::isnan(a) || std::isnan(b))
            ThrowInfo(ExprInvalid, "NaN SQL comparisons are not supported");
        return a < b ? -1 : (a > b ? 1 : 0);
    }

    SqlValue
    Predicate(const PreparedSqlExpr& prepared, size_t row) {
        const auto& expr = *prepared.source;
        const auto op = prepared.op;
        if (op == SqlOp::And || op == SqlOp::Or) {
            bool unknown = false;
            for (const auto& arg : prepared.args) {
                auto value = Eval(arg, row);
                if (IsNull(value)) {
                    unknown = true;
                    continue;
                }
                if (op == SqlOp::And && !std::get<bool>(value))
                    return false;
                if (op == SqlOp::Or && std::get<bool>(value))
                    return true;
            }
            if (unknown)
                return {};
            return op == SqlOp::And;
        }
        if (expr.args_size() < 1)
            ThrowInfo(ExprInvalid, "missing SQL predicate operand");
        auto left = Eval(prepared.args[0], row);
        if (op == SqlOp::IsNull)
            return IsNull(left);
        if (op == SqlOp::IsNotNull)
            return !IsNull(left);
        if (IsNull(left))
            return {};
        if (op == SqlOp::Not)
            return !std::get<bool>(left);
        if (op == SqlOp::In) {
            bool unknown = false;
            for (int i = 1; i < expr.args_size(); ++i) {
                auto value = Eval(prepared.args[i], row);
                if (IsNull(value)) {
                    unknown = true;
                    continue;
                }
                if (Compare(left, value) == 0)
                    return true;
            }
            if (unknown)
                return {};
            return false;
        }
        if (expr.args_size() != 2)
            ThrowInfo(ExprInvalid, "SQL comparison requires two operands");
        auto right = Eval(prepared.args[1], row);
        if (IsNull(right))
            return {};
        if (op == SqlOp::Like) {
            const auto& pattern = std::get<std::string>(right);
            auto found = patterns_.find(pattern);
            if (found == patterns_.end()) {
                std::string regex = "(?s)";
                bool escaped = false;
                for (char ch : pattern) {
                    if (escaped) {
                        regex += RE2::QuoteMeta(std::string(1, ch));
                        escaped = false;
                    } else if (ch == '\\')
                        escaped = true;
                    else if (ch == '%')
                        regex += ".*";
                    else if (ch == '_')
                        regex += ".";
                    else
                        regex += RE2::QuoteMeta(std::string(1, ch));
                }
                if (escaped)
                    ThrowInfo(ExprInvalid, "LIKE pattern ends with escape");
                found = patterns_.emplace(pattern, std::make_unique<RE2>(regex))
                            .first;
                if (!found->second->ok())
                    ThrowInfo(ExprInvalid, "invalid SQL LIKE pattern");
            }
            return RE2::FullMatch(std::get<std::string>(left), *found->second);
        }
        const auto cmp = Compare(left, right);
        if (op == SqlOp::Equal)
            return cmp == 0;
        if (op == SqlOp::NotEqual)
            return cmp != 0;
        if (op == SqlOp::Less)
            return cmp < 0;
        if (op == SqlOp::LessEqual)
            return cmp <= 0;
        if (op == SqlOp::Greater)
            return cmp > 0;
        if (op == SqlOp::GreaterEqual)
            return cmp >= 0;
        ThrowInfo(ExprInvalid, "unsupported SQL predicate {}", expr.op());
    }

    ColumnVectorPtr
    Source(int64_t id, DataType type) {
        if (auto found = sources_.find(id); found != sources_.end()) {
            return found->second;
        }
        auto result = ReadSource(id, type, offsets_.data(), offsets_.size());
        sources_.emplace(id, result);
        return result;
    }

    ColumnVectorPtr
    ReadSource(int64_t id,
               DataType type,
               const int64_t* offsets,
               size_t count) {
        const auto field_id = FieldId(id);
        TargetBitmap valid(count, false);
        FieldDataPtr data;
        if (type == DataType::JSON) {
            FixedVector<Json> values(count);
            if (schema_->has_field(field_id)) {
                segment_->bulk_subscript(op_context_,
                                         field_id,
                                         type,
                                         offsets,
                                         count,
                                         values.data(),
                                         valid);
            }
            data = std::make_shared<FieldDataImpl<Json, true>>(
                1, type, false, std::move(values));
        } else if (schema_->has_field(field_id)) {
            data = bulk_script_field_data(op_context_,
                                          field_id,
                                          type,
                                          offsets,
                                          count,
                                          segment_,
                                          valid,
                                          true);
        } else {
            data = InitScalarFieldDataWithLength(type, count);
        }
        const auto null_count = count - valid.count();
        return std::make_shared<ColumnVector>(
            std::move(data), std::move(valid), null_count);
    }

    const Json*
    JsonSource(int64_t id, size_t row) {
        // A missing typed shredding cell can trigger raw fallback. Keep only a
        // bounded batch per JSON field, rather than loading every matched row.
        constexpr size_t batch_size = 1024;
        const auto begin = row / batch_size * batch_size;
        auto& [cached_begin, column] = json_sources_[id];
        if (!column || cached_begin != begin) {
            column = ReadSource(id,
                                DataType::JSON,
                                offsets_.data() + begin,
                                std::min(batch_size, offsets_.size() - begin));
            cached_begin = begin;
        }
        const auto local = row - begin;
        return column->ValidAt(local) ? &column->RawAsValues<Json>()[local]
                                      : nullptr;
    }

    PreparedJsonPath&
    PrepareJsonPath(const PreparedSqlExpr& prepared) {
        if (prepared.path) {
            return *prepared.path;
        }
        const auto& expr = *prepared.source;
        if (expr.args_size() != 1 || expr.args(0).op() != "column") {
            ThrowInfo(ExprInvalid, "JSON path requires a source column");
        }
        PreparedJsonPath path;
        path.field_id = expr.args(0).source_field_id();
        std::vector<std::string> components(expr.json_path().begin(),
                                            expr.json_path().end());
        path.pointer = Json::pointer(components);
        std::vector<std::string> parent;
        for (const auto& component : components) {
            if (!component.empty() &&
                std::all_of(
                    component.begin(), component.end(), [](unsigned char ch) {
                        return std::isdigit(ch);
                    })) {
                path.numeric_parents.push_back(Json::pointer(parent));
            }
            parent.push_back(component);
        }
        if (use_json_stats_ && path.numeric_parents.empty() &&
            !path.pointer.empty() &&
            expr.data_type() == proto::schema::VarChar &&
            schema_->has_field(FieldId(path.field_id))) {
            auto stats =
                segment_->GetJsonStats(op_context_, FieldId(path.field_id));
            if (stats) {
                path.string_reader =
                    stats->CreateShreddingReader(path.pointer,
                                                 index::JSONType::STRING,
                                                 segment_->get_row_count());
                path.integer_reader =
                    stats->CreateShreddingReader(path.pointer,
                                                 index::JSONType::INT64,
                                                 segment_->get_row_count());
            }
        }
        prepared.path = std::make_unique<PreparedJsonPath>(std::move(path));
        return *prepared.path;
    }

    SqlValue
    JsonPath(const PreparedSqlExpr& prepared, size_t row) {
        const auto& expr = *prepared.source;
        auto& path = PrepareJsonPath(prepared);
        const auto id = path.field_id;
        const auto& pointer = path.pointer;
        if (!schema_->has_field(FieldId(id))) {
            return {};
        }
        // SQL -> '0' addresses an object key, never an array position.
        for (const auto& parent : path.numeric_parents) {
            auto source = JsonSource(id, row);
            if (!source)
                return {};
            auto kind = source->type(parent);
            if (kind.error()) {
                if (kind.error() == simdjson::NO_SUCH_FIELD ||
                    kind.error() == simdjson::INCORRECT_TYPE ||
                    kind.error() == simdjson::INDEX_OUT_OF_BOUNDS)
                    return {};
                ThrowInfo(UnexpectedError, "invalid stored JSON path parent");
            }
            if (kind.value() != simdjson::ondemand::json_type::object)
                return {};
        }
        if (path.string_reader) {
            auto value = path.string_reader->Get(op_context_, offsets_[row]);
            if (value && std::holds_alternative<std::string>(*value)) {
                return std::move(std::get<std::string>(*value));
            }
        }
        return ReadJsonPath(expr, pointer, JsonSource(id, row));
    }

    SqlValue
    ReadJsonPath(const SqlExpr& expr,
                 const std::string& pointer,
                 const Json* source) {
        if (!source) {
            return {};
        }
        const auto& json = *source;
        auto kind = json.type(pointer);
        if (kind.error()) {
            if (kind.error() == simdjson::NO_SUCH_FIELD ||
                kind.error() == simdjson::INCORRECT_TYPE ||
                kind.error() == simdjson::INDEX_OUT_OF_BOUNDS) {
                return {};
            }
            ThrowInfo(UnexpectedError,
                      "invalid stored JSON during SQL projection");
        }
        if (expr.data_type() == proto::schema::VarChar &&
            kind.value() == simdjson::ondemand::json_type::null) {
            return {};
        }
        if (expr.data_type() == proto::schema::VarChar &&
            kind.value() == simdjson::ondemand::json_type::string) {
            auto text = json.at<std::string_view>(pointer);
            if (text.error())
                ThrowInfo(UnexpectedError, "failed to read stored JSON string");
            return std::string(text.value());
        }
        auto raw = json.at_string_any(pointer);
        if (raw.error())
            ThrowInfo(UnexpectedError, "failed to read stored JSON value");
        return raw.value();
    }

    double
    Extract(const PreparedSqlExpr& prepared, const SqlValue& value) {
        const auto& expr = *prepared.source;
        const auto unit = prepared.unit;
        if (unit == ExtractUnit::Epoch) {
            return prepared.args[0].op == SqlOp::TimestampDiff
                       ? Number(value)
                       : double(Integer(value)) / 1000000.0;
        }
        if (!timezone_value_) {
            absl::TimeZone timezone;
            if (!absl::LoadTimeZone(timezone_.empty() ? "UTC" : timezone_,
                                    &timezone)) {
                ThrowInfo(ExprInvalid, "invalid SQL timezone");
            }
            timezone_value_ = timezone;
            timezone_is_utc_ = timezone == absl::UTCTimeZone();
        }
        const auto& timezone = *timezone_value_;
        const auto micros = Integer(value);
        if (timezone_is_utc_ &&
            (unit == ExtractUnit::Hour || unit == ExtractUnit::Minute ||
             unit == ExtractUnit::Second)) {
            // Unix microseconds use fixed 86400-second UTC days. Normalize
            // negative remainders without negating INT64_MIN, and retain the
            // scalar second-plus-fraction rounding used by the calendar path.
            constexpr int64_t day_micros = 86400000000LL;
            auto within_day = micros % day_micros;
            if (within_day < 0) {
                within_day += day_micros;
            }
            const auto seconds = within_day / 1000000;
            if (unit == ExtractUnit::Hour)
                return seconds / 3600;
            if (unit == ExtractUnit::Minute)
                return (seconds / 60) % 60;
            return seconds % 60 + (within_day % 1000000) / 1000000.0;
        }
        auto remainder = micros % 1000000;
        if (remainder < 0) {
            remainder += 1000000;
        }
        const auto time = absl::FromUnixMicros(micros);
        const auto parts = timezone.At(time).cs;
        if (unit == ExtractUnit::Hour)
            return parts.hour();
        if (unit == ExtractUnit::Minute)
            return parts.minute();
        if (unit == ExtractUnit::Second)
            return parts.second() + remainder / 1000000.0;
        if (unit == ExtractUnit::Day)
            return parts.day();
        if (unit == ExtractUnit::Month)
            return parts.month();
        if (unit == ExtractUnit::Year)
            return parts.year();
        if (unit == ExtractUnit::Dow)
            return std::stoi(absl::FormatTime("%w", time, timezone));
        if (unit == ExtractUnit::Doy)
            return absl::CivilDay(parts) - absl::CivilDay(parts.year(), 1, 1) +
                   1;
        ThrowInfo(
            ExprInvalid, "unsupported SQL extract unit {}", expr.time_unit());
    }

    const std::vector<int64_t>& offsets_;
    const segcore::SegmentInternalInterface* segment_;
    // Hold one immutable schema for this projection. Segment read APIs and
    // shredding readers retain their own resource snapshots/pins. This cache
    // never survives an invocation, including schema changes between queries.
    SchemaPtr schema_;
    OpContext* op_context_;
    std::string timezone_;
    bool use_json_stats_;
    std::optional<absl::TimeZone> timezone_value_;
    bool timezone_is_utc_ = false;
    std::unordered_map<int64_t, ColumnVectorPtr> sources_;
    std::unordered_map<int64_t, std::pair<size_t, ColumnVectorPtr>>
        json_sources_;
    std::unordered_map<std::string, std::unique_ptr<RE2>> patterns_;
};

}  // namespace

RowVectorPtr
EvaluateSqlProjection(const proto::plan::SQLProject& project,
                      const std::vector<FieldId>& output_ids,
                      const std::vector<int64_t>& offsets,
                      const segcore::SegmentInternalInterface* segment,
                      OpContext* op_context,
                      bool use_json_stats) {
    if (project.version() != 1)
        ThrowInfo(ExprInvalid, "unsupported SQL projection version");
    SqlProjectionEvaluator evaluator(
        offsets, segment, op_context, project.timezone(), use_json_stats);
    std::optional<PreparedSqlExpr> predicate;
    if (project.has_predicate()) {
        predicate = PrepareSqlExpr(project.predicate());
    }
    std::vector<size_t> selected;
    selected.reserve(offsets.size());
    for (size_t row = 0; row < offsets.size(); ++row) {
        if ((row & 1023) == 0) {
            segcore::CheckCancellation(
                op_context, segment->get_segment_id(), "SQL predicate");
        }
        if (project.has_predicate()) {
            auto value = evaluator.Eval(*predicate, row);
            if (IsNull(value) || !std::get<bool>(value))
                continue;
        }
        selected.push_back(row);
    }
    if (selected.empty())
        return nullptr;
    std::unordered_map<int64_t, const SqlExpr*> expressions;
    for (const auto& column : project.columns()) {
        if (!expressions.emplace(column.id(), &column.expression()).second) {
            ThrowInfo(ExprInvalid, "duplicate SQL projection ID");
        }
    }
    std::vector<VectorPtr> columns;
    columns.reserve(output_ids.size());
    for (auto id : output_ids) {
        auto found = expressions.find(id.get());
        if (found == expressions.end())
            ThrowInfo(ExprInvalid, "missing SQL projection expression");
        const auto& expr = *found->second;
        const auto prepared = PrepareSqlExpr(expr);
        auto data = InitScalarFieldDataWithLength(
            static_cast<DataType>(expr.data_type()), selected.size());
        auto column = std::make_shared<ColumnVector>(
            std::move(data), TargetBitmap(selected.size(), true), 0);
        if (prepared.op != SqlOp::JsonPath ||
            !evaluator.ProjectJsonPath(prepared, selected, column)) {
            for (size_t row = 0; row < selected.size(); ++row) {
                if ((row & 1023) == 0) {
                    segcore::CheckCancellation(op_context,
                                               segment->get_segment_id(),
                                               "SQL projection");
                }
                WriteValue(
                    column, row, evaluator.Eval(prepared, selected[row]));
            }
        }
        columns.emplace_back(std::move(column));
    }
    auto result = std::make_shared<RowVector>(std::move(columns));
    result->resize(selected.size());
    return result;
}

}  // namespace milvus::exec
