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

#include "exec/plan/ExprCompiler.h"
#include "expr/ProjectionExpr.h"

#include <cmath>
#include <charconv>
#include <limits>
#include "exec/expression/InputRefExpr.h"
#include "exec/expression/PredicateRefExpr.h"
#include "exec/expression/ScalarCallExpr.h"
#include "exec/expression/ScalarPredicates.h"
#include "exec/expression/ScalarArithmetic.h"
#include "exec/expression/ScalarBatchKernels.h"
#include "exec/expression/ScalarJsonPredicates.h"
#include "exec/expression/ScalarComplexPredicates.h"
#include "exec/expression/ScalarCollectionPredicates.h"
#include "exec/expression/TimestampArithmetic.h"

namespace milvus::exec::plan {
namespace {
using GenericValue = proto::plan::GenericValue;

bool
IsString(DataType type) {
    return type == DataType::STRING || type == DataType::VARCHAR ||
           type == DataType::TEXT;
}

bool
IsInteger(DataType type) {
    return type == DataType::INT8 || type == DataType::INT16 ||
           type == DataType::INT32 || type == DataType::INT64;
}

bool
JsonConstant(const GenericValue& value, bool allow_array = true) {
    switch (value.val_case()) {
        case GenericValue::kBoolVal:
        case GenericValue::kInt64Val:
        case GenericValue::kStringVal:
            return true;
        case GenericValue::kFloatVal:
            return std::isfinite(value.float_val());
        case GenericValue::kArrayVal:
            if (!allow_array)
                return false;
            for (const auto& item : value.array_val().array())
                if (!JsonConstant(item, false))
                    return false;
            return true;
        default:
            return false;
    }
}

bool
IsScalar(DataType type) {
    switch (type) {
        case DataType::BOOL:
        case DataType::INT8:
        case DataType::INT16:
        case DataType::INT32:
        case DataType::INT64:
        case DataType::TIMESTAMPTZ:
        case DataType::FLOAT:
        case DataType::DOUBLE:
        case DataType::STRING:
        case DataType::VARCHAR:
        case DataType::TEXT:
            return true;
        default:
            return false;
    }
}

// ARRAY leaves are decoded at the legacy comparison width. Integer arrays
// must not truncate an INT64 predicate literal to INT8/INT16/INT32.
DataType
ArrayValueType(DataType type) {
    if (IsInteger(type))
        return DataType::INT64;
    if (type == DataType::FLOAT || type == DataType::DOUBLE)
        return DataType::DOUBLE;
    if (type == DataType::STRING || type == DataType::VARCHAR)
        return DataType::VARCHAR;
    return type == DataType::BOOL ? type : DataType::NONE;
}

bool
ArrayTermCompatible(DataType type, const GenericValue& value) {
    if (type == DataType::FLOAT && value.has_float_val() &&
        std::abs(value.float_val()) > std::numeric_limits<float>::max())
        return false;
    if (type == DataType::DOUBLE || type == DataType::FLOAT)
        return value.has_int64_val() ||
               (value.has_float_val() && std::isfinite(value.float_val()));
    if (type == DataType::INT64)
        return value.has_int64_val();
    if (type == DataType::BOOL)
        return value.has_bool_val();
    if (type == DataType::VARCHAR)
        return value.has_string_val();
    return false;
}

std::optional<scalar::CompareOp>
Comparison(proto::plan::OpType op) {
    switch (op) {
        case proto::plan::Equal:
            return scalar::CompareOp::Equal;
        case proto::plan::NotEqual:
            return scalar::CompareOp::NotEqual;
        case proto::plan::LessThan:
            return scalar::CompareOp::Less;
        case proto::plan::LessEqual:
            return scalar::CompareOp::LessEqual;
        case proto::plan::GreaterThan:
            return scalar::CompareOp::Greater;
        case proto::plan::GreaterEqual:
            return scalar::CompareOp::GreaterEqual;
        default:
            return std::nullopt;
    }
}

template <typename T>
VectorPtr
Constant(DataType type, T value) {
    return std::make_shared<ConstantVector<T>>(type, 1, value);
}

template <typename T>
VectorPtr
IntegerLiteral(DataType type, const GenericValue& value) {
    if (value.val_case() != GenericValue::kInt64Val ||
        value.int64_val() < std::numeric_limits<T>::min() ||
        value.int64_val() > std::numeric_limits<T>::max()) {
        return nullptr;
    }
    return Constant<T>(type, static_cast<T>(value.int64_val()));
}

// Eligibility checks must not allocate a vector merely to inspect its type.
bool
LiteralFits(DataType type, const GenericValue& value) {
    auto integer = [&]<typename T>() {
        return value.has_int64_val() &&
               value.int64_val() >= std::numeric_limits<T>::min() &&
               value.int64_val() <= std::numeric_limits<T>::max();
    };
    switch (type) {
        case DataType::NONE:
            return value.val_case() == GenericValue::VAL_NOT_SET;
        case DataType::BOOL:
            return value.has_bool_val();
        case DataType::INT8:
            return integer.template operator()<int8_t>();
        case DataType::INT16:
            return integer.template operator()<int16_t>();
        case DataType::INT32:
            return integer.template operator()<int32_t>();
        case DataType::INT64:
        case DataType::TIMESTAMPTZ:
            return value.has_int64_val();
        case DataType::FLOAT:
            return value.has_float_val() && std::isfinite(value.float_val()) &&
                   std::abs(value.float_val()) <=
                       std::numeric_limits<float>::max();
        case DataType::DOUBLE:
            return value.has_float_val() && std::isfinite(value.float_val());
        case DataType::STRING:
        case DataType::VARCHAR:
        case DataType::TEXT:
            return value.has_string_val();
        default:
            return false;
    }
}

VectorPtr
Literal(DataType type, const GenericValue& value) {
    if (!LiteralFits(type, value))
        return nullptr;
    switch (type) {
        case DataType::NONE:
            return std::make_shared<ConstantVector<bool>>(type, 1, false, 1);
        case DataType::BOOL:
            return Constant(type, value.bool_val());
        case DataType::INT8:
            return IntegerLiteral<int8_t>(type, value);
        case DataType::INT16:
            return IntegerLiteral<int16_t>(type, value);
        case DataType::INT32:
            return IntegerLiteral<int32_t>(type, value);
        case DataType::INT64:
        case DataType::TIMESTAMPTZ:
            return IntegerLiteral<int64_t>(type, value);
        case DataType::FLOAT:
            return Constant(type, static_cast<float>(value.float_val()));
        case DataType::DOUBLE:
            return Constant(type, value.float_val());
        case DataType::STRING:
        case DataType::VARCHAR:
        case DataType::TEXT:
            return Constant(type, value.string_val());
        default:
            return nullptr;
    }
}

std::string
UnmigratedNode(const expr::ITypeExpr& node) {
    if (dynamic_cast<const expr::TermFilterExpr*>(&node))
        return "TermFilterExpr";
    if (dynamic_cast<const expr::BinaryArithOpEvalRangeExpr*>(&node))
        return "BinaryArithOpEvalRangeExpr";
    if (dynamic_cast<const expr::TimestamptzArithCompareExpr*>(&node))
        return "TimestamptzArithCompareExpr";
    if (dynamic_cast<const expr::ExistsExpr*>(&node))
        return "ExistsExpr";
    if (dynamic_cast<const expr::JsonContainsExpr*>(&node))
        return "JsonContainsExpr";
    if (dynamic_cast<const expr::GISFunctionFilterExpr*>(&node))
        return "GISFunctionFilterExpr";
    if (dynamic_cast<const expr::MatchExpr*>(&node))
        return "MatchExpr";
    if (dynamic_cast<const expr::BloomFilterExpr*>(&node))
        return "BloomFilterExpr";
    if (dynamic_cast<const expr::RoaringFilterExpr*>(&node))
        return "RoaringFilterExpr";
    if (dynamic_cast<const expr::FieldAccessTypeExpr*>(&node))
        return "FieldAccessTypeExpr";
    if (dynamic_cast<const expr::InputTypeExpr*>(&node))
        return "InputTypeExpr";
    return "unknown logical expression";
}

// Compile the semantic constants once, independently of index availability.
// Narrow integer overflow is dropped; malformed types reject the whole plan.
std::optional<ReadOnlyVectorView>
TermConstants(DataType type, const std::vector<GenericValue>& constants) {
    size_t count = 0;
    for (const auto& value : constants) {
        if (LiteralFits(type, value)) {
            ++count;
        } else if (!(value.has_int64_val() &&
                     (type == DataType::INT8 || type == DataType::INT16 ||
                      type == DataType::INT32))) {
            return std::nullopt;
        }
    }
    VectorResultBuilder output(type, count);
    auto fill = [&]<typename T>() {
        size_t row = 0;
        for (const auto& value : constants) {
            if (!LiteralFits(type, value))
                continue;
            if constexpr (std::is_same_v<T, bool>)
                output.Set<T>(row++, value.bool_val());
            else if constexpr (std::is_integral_v<T>)
                output.Set<T>(row++, static_cast<T>(value.int64_val()));
            else if constexpr (std::is_floating_point_v<T>)
                output.Set<T>(row++, static_cast<T>(value.float_val()));
            else
                output.Set<T>(row++, value.string_val());
        }
    };
    switch (type) {
        case DataType::BOOL:
            fill.template operator()<bool>();
            break;
        case DataType::INT8:
            fill.template operator()<int8_t>();
            break;
        case DataType::INT16:
            fill.template operator()<int16_t>();
            break;
        case DataType::INT32:
            fill.template operator()<int32_t>();
            break;
        case DataType::INT64:
        case DataType::TIMESTAMPTZ:
            fill.template operator()<int64_t>();
            break;
        case DataType::FLOAT:
            fill.template operator()<float>();
            break;
        case DataType::DOUBLE:
            fill.template operator()<double>();
            break;
        case DataType::STRING:
        case DataType::VARCHAR:
        case DataType::TEXT:
            fill.template operator()<std::string>();
            break;
        default:
            return std::nullopt;
    }
    return std::move(output).Finish(SelectionView::All(count));
}

// Rewrite only the element address domain; the ordinary scalar compiler
// remains the single implementation of each predicate and its type checks.
expr::TypedExprPtr
ElementPredicate(const expr::TypedExprPtr& node,
                 const Schema& schema,
                 const std::string& structure,
                 size_t depth = 0) {
    if (!node || depth > 256)
        return nullptr;
    auto column =
        [&](const expr::ColumnInfo& value) -> std::optional<expr::ColumnInfo> {
        if (!value.element_level_)
            return std::nullopt;
        const auto& meta = schema[value.field_id_];
        auto name = GetStructNameForArrayField(meta);
        const auto type = meta.get_element_type();
        if (name && *name == structure && meta.is_nested_array() &&
            type == value.element_type_) {
            const auto& child_type =
                meta.get_array_type_schema().array_element();
            return expr::ColumnInfo(
                value.field_id_,
                DataType::ARRAY,
                child_type.array_element().has_array_element()
                    ? DataType::ARRAY
                    : static_cast<DataType>(
                          child_type.array_element().leaf_type()),
                value.nested_path_,
                true);
        }
        if (!name || *name != structure || !value.nested_path_.empty() ||
            meta.get_data_type() != DataType::ARRAY ||
            type != value.element_type_ ||
            !(type == DataType::BOOL || IsInteger(type) ||
              type == DataType::FLOAT || type == DataType::DOUBLE ||
              type == DataType::VARCHAR || type == DataType::STRING))
            return std::nullopt;
        return expr::ColumnInfo(value.field_id_, type, {}, true);
    };
    auto child = [&](const expr::TypedExprPtr& value) {
        return ElementPredicate(value, schema, structure, depth + 1);
    };
    if (auto logical =
            dynamic_cast<const expr::LogicalBinaryExpr*>(node.get())) {
        if (node->inputs().size() != 2)
            return nullptr;
        auto left = child(node->inputs()[0]), right = child(node->inputs()[1]);
        return left && right ? std::make_shared<expr::LogicalBinaryExpr>(
                                   logical->op_type_, left, right)
                             : nullptr;
    }
    if (auto logical =
            dynamic_cast<const expr::LogicalUnaryExpr*>(node.get())) {
        if (node->inputs().size() != 1)
            return nullptr;
        auto input = child(node->inputs()[0]);
        return input ? std::make_shared<expr::LogicalUnaryExpr>(
                           logical->op_type_, input)
                     : nullptr;
    }
    if (auto value = dynamic_cast<const expr::ColumnExpr*>(node.get())) {
        auto field = column(value->GetColumn());
        return field ? std::make_shared<expr::ColumnExpr>(*field) : nullptr;
    }
    if (auto call = dynamic_cast<const expr::CallExpr*>(node.get())) {
        std::vector<expr::TypedExprPtr> inputs;
        for (const auto& arg : node->inputs()) {
            auto converted = child(arg);
            if (!converted)
                return nullptr;
            inputs.push_back(std::move(converted));
        }
        return std::make_shared<expr::CallExpr>(
            call->fun_name(), inputs, call->function_ptr());
    }
    if (auto compare = dynamic_cast<const expr::CompareExpr*>(node.get())) {
        auto element =
            [&](FieldId id,
                DataType declared) -> std::optional<expr::ColumnInfo> {
            const auto& meta = schema[id];
            if (declared != DataType::ARRAY &&
                declared != meta.get_element_type())
                return std::nullopt;
            expr::ColumnInfo field(
                id, DataType::ARRAY, meta.get_element_type(), {}, true);
            field.element_level_ = true;
            return column(field);
        };
        auto left = element(compare->left_field_id_, compare->left_data_type_);
        auto right =
            element(compare->right_field_id_, compare->right_data_type_);
        return left && right
                   ? std::make_shared<expr::CompareExpr>(left->field_id_,
                                                         right->field_id_,
                                                         left->data_type_,
                                                         right->data_type_,
                                                         compare->op_type_)
                   : nullptr;
    }
    if (auto unary =
            dynamic_cast<const expr::UnaryRangeFilterExpr*>(node.get())) {
        auto field = column(unary->column_);
        return field ? std::make_shared<expr::UnaryRangeFilterExpr>(
                           *field,
                           unary->op_type_,
                           unary->val_,
                           unary->extra_values_)
                     : nullptr;
    }
    if (auto range =
            dynamic_cast<const expr::BinaryRangeFilterExpr*>(node.get())) {
        auto field = column(range->column_);
        return field ? std::make_shared<expr::BinaryRangeFilterExpr>(
                           *field,
                           range->lower_val_,
                           range->upper_val_,
                           range->lower_inclusive_,
                           range->upper_inclusive_)
                     : nullptr;
    }
    if (auto term = dynamic_cast<const expr::TermFilterExpr*>(node.get())) {
        auto field = column(term->column_);
        return field ? std::make_shared<expr::TermFilterExpr>(
                           *field, term->vals_, term->is_in_field_)
                     : nullptr;
    }
    if (auto null = dynamic_cast<const expr::NullExpr*>(node.get())) {
        auto field = column(null->column_);
        return field ? std::make_shared<expr::NullExpr>(*field, null->op_)
                     : nullptr;
    }
    if (auto contains =
            dynamic_cast<const expr::JsonContainsExpr*>(node.get())) {
        auto field = column(contains->column_);
        return field ? std::make_shared<expr::JsonContainsExpr>(
                           *field,
                           contains->op_,
                           contains->same_type_,
                           contains->vals_)
                     : nullptr;
    }
    if (auto arithmetic =
            dynamic_cast<const expr::BinaryArithOpEvalRangeExpr*>(node.get())) {
        auto field = column(arithmetic->column_);
        return field ? std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
                           *field,
                           arithmetic->op_type_,
                           arithmetic->arith_op_type_,
                           arithmetic->value_,
                           arithmetic->right_operand_)
                     : nullptr;
    }
    if (auto bloom = dynamic_cast<const expr::BloomFilterExpr*>(node.get())) {
        auto field = column(bloom->column_);
        return field ? std::make_shared<expr::BloomFilterExpr>(
                           *field, bloom->filter_blob_)
                     : nullptr;
    }
    if (auto roaring =
            dynamic_cast<const expr::RoaringFilterExpr*>(node.get())) {
        auto field = column(roaring->column_);
        return field ? std::make_shared<expr::RoaringFilterExpr>(
                           *field, roaring->membership_)
                     : nullptr;
    }
    if (dynamic_cast<const expr::ValueExpr*>(node.get()) ||
        dynamic_cast<const expr::AlwaysTrueExpr*>(node.get()))
        return node;
    return nullptr;
}

class Compiler {
 public:
    explicit Compiler(const CompileOptions& options) : options_(options) {
        AssertInfo(options.upstream_columns.empty() || options.upstream_schema,
                   "upstream bindings require a schema");
        for (const auto& [field, column] : options.upstream_columns) {
            AssertInfo(column < options.upstream_schema->column_count(),
                       "upstream column binding out of bounds");
        }
        AssertInfo(options.upstream_results.empty() || options.upstream_schema,
                   "result bindings require an upstream schema");
        for (const auto& [id, column] : options.upstream_results)
            AssertInfo(
                id.value && column < options.upstream_schema->column_count(),
                "invalid result column binding");
    }

    CompileResult
    Run(const std::vector<expr::TypedExprPtr>& sources) {
        std::vector<scalar::ExprPtr> roots;
        for (size_t i = 0; i < sources.size(); ++i) {
            roots.push_back(
                Build(sources[i], "root[" + std::to_string(i) + "]", 0));
        }
        if (!unsupported_.empty()) {
            return {nullptr, std::move(unsupported_)};
        }
        return {
            std::make_shared<const CompiledExprPlan>(std::move(layout_),
                                                     std::move(roots),
                                                     options_.upstream_schema,
                                                     std::move(direct_inputs_),
                                                     std::move(predicates_),
                                                     std::move(properties_),
                                                     !has_opaque_calls_),
            {}};
    }

 private:
    scalar::ExprPtr
    UnsupportedAt(const std::string& location, const std::string& feature) {
        unsupported_.push_back({location, feature});
        return nullptr;
    }

    scalar::ExprPtr
    Field(const expr::ColumnInfo& column,
          bool values,
          const std::string& location,
          bool allow_nested = false) {
        if (column.data_type_ == DataType::ARRAY &&
            !column.nested_path_.empty() && !column.element_level_) {
            int64_t index = -1;
            const auto& path = column.nested_path_[0];
            const auto parsed =
                std::from_chars(path.data(), path.data() + path.size(), index);
            auto type = ArrayValueType(column.element_type_);
            if (column.nested_path_.size() != 1 || parsed.ec != std::errc{} ||
                parsed.ptr != path.data() + path.size() || index < 0 ||
                type == DataType::NONE)
                return UnsupportedAt(location, "ARRAY subscript");
            auto raw = column;
            raw.nested_path_.clear();
            auto input = Field(raw, true, location);
            return input ? scalar::MakeArrayElement(input, index, type)
                         : nullptr;
        }
        if (column.data_type_ == DataType::ARRAY &&
            !((allow_nested || !values) &&
              column.element_type_ == DataType::ARRAY) &&
            !(column.element_type_ == DataType::BOOL ||
              IsInteger(column.element_type_) ||
              column.element_type_ == DataType::FLOAT ||
              column.element_type_ == DataType::DOUBLE ||
              column.element_type_ == DataType::VARCHAR ||
              column.element_type_ == DataType::STRING))
            return UnsupportedAt(location, "nested/unsupported ARRAY input");
        if ((!IsScalar(column.data_type_) &&
             column.data_type_ != DataType::JSON &&
             column.data_type_ != DataType::GEOMETRY &&
             column.data_type_ != DataType::ARRAY &&
             !(!values && IsOrdinaryVectorDataType(column.data_type_)) &&
             column.data_type_ != DataType::VECTOR_ARRAY) ||
            !column.nested_path_.empty() || column.element_level_) {
            return UnsupportedAt(location, "complex/path/element column input");
        }
        InputDependency dependency{
            options_.relation, column.field_id_.get(), column.data_type_};
        dependency.element_type = column.element_type_;
        dependency.path = column.nested_path_;
        dependency.element_level = column.element_level_;
        dependency.nullable = column.nullable_;
        dependency.needs_values = values;
        dependency.needs_validity = true;
        const auto found = options_.upstream_columns.find(dependency.field_id);
        if (found != options_.upstream_columns.end()) {
            dependency.upstream_column = found->second;
            AssertInfo(options_.upstream_schema->column_type(found->second) ==
                           column.data_type_,
                       "upstream binding type differs from logical column");
        }
        auto slot = layout_.Intern(std::move(dependency));
        (exact_inputs_ ? *exact_inputs_ : direct_inputs_)
            .push_back({slot, values});
        return std::make_shared<scalar::InputRefExpr>(slot, column.data_type_);
    }

    // Only audited, raw-only leaf programs may become alternatives. Their
    // exact program is compiled with substitution disabled, so it cannot
    // request itself or another predicate slot. No TTL reinjection occurs.
    std::optional<segcore::reader::Int64IndexQuery>
    IndexQuery(const expr::TypedExprPtr& node, const expr::ColumnInfo*& field) {
        using Op = segcore::reader::ScalarIndexOp;
        using Query = segcore::reader::Int64IndexQuery;
        std::optional<Query> query;
        auto integer = [](const GenericValue& value) -> std::optional<int64_t> {
            if (value.val_case() == GenericValue::kInt64Val)
                return value.int64_val();
            if (value.val_case() == GenericValue::kBoolVal)
                return value.bool_val() ? 1 : 0;
            return std::nullopt;
        };
        if (auto unary =
                dynamic_cast<const expr::UnaryRangeFilterExpr*>(node.get())) {
            field = &unary->column_;
            if (!unary->extra_values_.empty() || !integer(unary->val_))
                return std::nullopt;
            std::optional<Op> op;
            switch (unary->op_type_) {
                case proto::plan::Equal:
                    op = Op::Equal;
                    break;
                case proto::plan::NotEqual:
                    op = Op::NotEqual;
                    break;
                case proto::plan::LessThan:
                    op = Op::Less;
                    break;
                case proto::plan::LessEqual:
                    op = Op::LessEqual;
                    break;
                case proto::plan::GreaterThan:
                    op = Op::Greater;
                    break;
                case proto::plan::GreaterEqual:
                    op = Op::GreaterEqual;
                    break;
                default:
                    return std::nullopt;
            }
            query = Query{*op, *integer(unary->val_)};
        } else if (auto range =
                       dynamic_cast<const expr::BinaryRangeFilterExpr*>(
                           node.get())) {
            field = &range->column_;
            if (!integer(range->lower_val_) || !integer(range->upper_val_))
                return std::nullopt;
            query = Query{Op::Between,
                          *integer(range->lower_val_),
                          *integer(range->upper_val_),
                          range->lower_inclusive_,
                          range->upper_inclusive_};
        } else if (auto term =
                       dynamic_cast<const expr::TermFilterExpr*>(node.get())) {
            field = &term->column_;
            if (term->is_in_field_)
                return std::nullopt;
            query = Query{Op::In};
            for (const auto& value : term->vals_) {
                auto number = integer(value);
                if (!number)
                    return std::nullopt;
                // Use precisely the raw compiler's representability filter.
                if (LiteralFits(field->data_type_, value))
                    query->terms.push_back(*number);
            }
        } else if (auto null =
                       dynamic_cast<const expr::NullExpr*>(node.get())) {
            field = &null->column_;
            if (null->op_ != proto::plan::NullExpr_NullOp_IsNull &&
                null->op_ != proto::plan::NullExpr_NullOp_IsNotNull)
                return std::nullopt;
            query = Query{null->op_ == proto::plan::NullExpr_NullOp_IsNull
                              ? Op::IsNull
                              : Op::IsNotNull};
        }
        if (!query ||
            (field->data_type_ != DataType::INT64 &&
             field->data_type_ != DataType::TIMESTAMPTZ &&
             field->data_type_ != DataType::INT32 &&
             field->data_type_ != DataType::INT16 &&
             field->data_type_ != DataType::INT8 &&
             field->data_type_ != DataType::BOOL) ||
            !field->nested_path_.empty() || field->element_level_ ||
            options_.upstream_columns.count(field->field_id_.get()))
            return std::nullopt;
        // The typed index must never receive a truncated narrow bound. Such
        // predicates still compile to exact raw comparisons below.
        if (field->data_type_ == DataType::INT8 ||
            field->data_type_ == DataType::INT16 ||
            field->data_type_ == DataType::INT32) {
            GenericValue lower, upper;
            lower.set_int64_val(query->lower);
            upper.set_int64_val(query->upper);
            if (!LiteralFits(field->data_type_, lower) ||
                !LiteralFits(field->data_type_, upper))
                return std::nullopt;
        }
        return query;
    }

    std::optional<segcore::reader::TypedIndexQuery>
    ArrayQuery(const expr::TypedExprPtr& node) {
        using Op = segcore::reader::ScalarIndexOp;
        const expr::ColumnInfo* column = nullptr;
        segcore::reader::TypedIndexQuery query{};
        bool in_field = false;
        if (auto contains =
                dynamic_cast<const expr::JsonContainsExpr*>(node.get())) {
            column = &contains->column_;
            if (contains->op_ !=
                    proto::plan::JSONContainsExpr_JSONOp_Contains &&
                contains->op_ !=
                    proto::plan::JSONContainsExpr_JSONOp_ContainsAny &&
                contains->op_ !=
                    proto::plan::JSONContainsExpr_JSONOp_ContainsAll)
                return std::nullopt;
            query.op = contains->op_ ==
                               proto::plan::JSONContainsExpr_JSONOp_ContainsAll
                           ? Op::ContainsAll
                           : Op::ContainsAny;
            query.terms = contains->vals_;
        } else if (auto term =
                       dynamic_cast<const expr::TermFilterExpr*>(node.get())) {
            if (!term->is_in_field_ || term->vals_.size() != 1)
                return std::nullopt;
            column = &term->column_;
            query.op = Op::ContainsAny;
            query.terms = term->vals_;
            in_field = true;
        } else if (auto null =
                       dynamic_cast<const expr::NullExpr*>(node.get())) {
            column = &null->column_;
            if (null->op_ != proto::plan::NullExpr_NullOp_IsNull &&
                null->op_ != proto::plan::NullExpr_NullOp_IsNotNull)
                return std::nullopt;
            query.op = null->op_ == proto::plan::NullExpr_NullOp_IsNull
                           ? Op::IsNull
                           : Op::IsNotNull;
        }
        if (!column || column->data_type_ != DataType::ARRAY ||
            column->element_level_ || !column->nested_path_.empty() ||
            options_.upstream_columns.count(column->field_id_.get()) ||
            !options_.schema)
            return std::nullopt;
        const auto& meta = (*options_.schema)[column->field_id_];
        // Existing array indexes read element payloads without checking their
        // null bitmap. They cannot prove membership for nullable elements.
        if (meta.get_data_type() != DataType::ARRAY || meta.is_nested_array() ||
            meta.is_element_nullable() ||
            meta.get_element_type() != column->element_type_)
            return std::nullopt;
        query.type = meta.get_element_type();
        if (!(query.type == DataType::BOOL || IsInteger(query.type) ||
              query.type == DataType::FLOAT || query.type == DataType::DOUBLE ||
              query.type == DataType::STRING ||
              query.type == DataType::VARCHAR))
            return std::nullopt;
        for (const auto& term : query.terms) {
            if (query.type == DataType::FLOAT ||
                query.type == DataType::DOUBLE) {
                if (!ArrayTermCompatible(query.type, term))
                    return std::nullopt;
                const double value = term.has_int64_val()
                                         ? double(term.int64_val())
                                         : term.float_val();
                // IN-field keeps double comparison precision for FLOAT arrays,
                // whereas contains deliberately rounds its constant to FLOAT.
                if (in_field && query.type == DataType::FLOAT &&
                    double(float(value)) != value)
                    return std::nullopt;
            } else if (!Literal(query.type, term)) {
                return std::nullopt;
            }
        }
        query.nested_array = GetStructNameForArrayField(meta).has_value();
        query.array_nullable = meta.is_nullable();
        return query;
    }

    std::optional<segcore::reader::TypedIndexQuery>
    TypedQuery(const expr::TypedExprPtr& node) {
        using Op = segcore::reader::ScalarIndexOp;
        using Query = segcore::reader::TypedIndexQuery;
        const expr::ColumnInfo* field = nullptr;
        Query q{};
        if (auto unary =
                dynamic_cast<const expr::UnaryRangeFilterExpr*>(node.get())) {
            field = &unary->column_;
            if (!unary->extra_values_.empty())
                return std::nullopt;
            switch (unary->op_type_) {
                case proto::plan::Equal:
                    q.op = Op::Equal;
                    break;
                case proto::plan::NotEqual:
                    q.op = Op::NotEqual;
                    break;
                case proto::plan::LessThan:
                    q.op = Op::Less;
                    break;
                case proto::plan::LessEqual:
                    q.op = Op::LessEqual;
                    break;
                case proto::plan::GreaterThan:
                    q.op = Op::Greater;
                    break;
                case proto::plan::GreaterEqual:
                    q.op = Op::GreaterEqual;
                    break;
                default:
                    return std::nullopt;
            }
            q.lower = unary->val_;
        } else if (auto range =
                       dynamic_cast<const expr::BinaryRangeFilterExpr*>(
                           node.get())) {
            field = &range->column_;
            q.op = Op::Between;
            q.lower = range->lower_val_;
            q.upper = range->upper_val_;
            q.lower_inclusive = range->lower_inclusive_;
            q.upper_inclusive = range->upper_inclusive_;
            if (q.lower.val_case() != q.upper.val_case())
                return std::nullopt;
        } else if (auto term =
                       dynamic_cast<const expr::TermFilterExpr*>(node.get())) {
            field = &term->column_;
            if (term->is_in_field_ ||
                (term->vals_.empty() && field->data_type_ == DataType::JSON))
                return std::nullopt;
            q.op = Op::In;
            if (!term->vals_.empty())
                q.lower = term->vals_.front();
            for (const auto& value : term->vals_) {
                if (value.val_case() != q.lower.val_case())
                    return std::nullopt;
                q.terms.push_back(value);
            }
        } else if (auto null =
                       dynamic_cast<const expr::NullExpr*>(node.get())) {
            field = &null->column_;
            if (field->data_type_ == DataType::JSON)
                return std::nullopt;
            q.op = null->op_ == proto::plan::NullExpr_NullOp_IsNull
                       ? Op::IsNull
                       : Op::IsNotNull;
        } else if (auto exists =
                       dynamic_cast<const expr::ExistsExpr*>(node.get())) {
            field = &exists->column_;
            if (field->data_type_ != DataType::JSON)
                return std::nullopt;
            q.op = Op::Exists;
        } else {
            return std::nullopt;
        }
        if (field->element_level_ ||
            options_.upstream_columns.count(field->field_id_.get()))
            return std::nullopt;
        if (field->data_type_ == DataType::JSON) {
            if (field->nested_path_.empty())
                return std::nullopt;
            for (const auto& part : field->nested_path_) {
                // Numeric components can denote array addressing; flat indexes
                // also cannot distinguish dots/escaped pointer components.
                if (part.empty() ||
                    part.find_first_of("./~") != std::string::npos ||
                    std::all_of(part.begin(), part.end(), [](char c) {
                        return c >= '0' && c <= '9';
                    }))
                    return std::nullopt;
            }
            q.pointer = Json::pointer(field->nested_path_);
            // Existence is independent of the path index's cast value type.
            if (q.op == Op::Exists)
                return q;
            switch (q.lower.val_case()) {
                case GenericValue::kBoolVal:
                    q.type = DataType::BOOL;
                    break;
                case GenericValue::kInt64Val:
                    q.type = DataType::INT64;
                    break;
                case GenericValue::kFloatVal:
                    q.type = DataType::DOUBLE;
                    break;
                case GenericValue::kStringVal:
                    q.type = DataType::VARCHAR;
                    break;
                default:
                    return std::nullopt;
            }
        } else {
            if (!field->nested_path_.empty())
                return std::nullopt;
            q.type = field->data_type_;
            if (q.type != DataType::FLOAT && q.type != DataType::DOUBLE &&
                !IsString(q.type))
                return std::
                    nullopt;  // Integral fields use the existing descriptor.
        }
        if (q.op != Op::IsNull && q.op != Op::IsNotNull &&
            !(q.op == Op::In && q.terms.empty())) {
            if (!LiteralFits(q.type, q.lower) ||
                (q.op == Op::Between && !LiteralFits(q.type, q.upper)))
                return std::nullopt;
            for (const auto& term : q.terms)
                if (!LiteralFits(q.type, term))
                    return std::nullopt;
        }
        return q;
    }

    VectorPtr
    BoundValue(DataType type,
               const GenericValue& value,
               const std::string& location) {
        auto literal = Literal(type, value);
        if (!literal)
            UnsupportedAt(
                location,
                "literal type/conversion/overflow/nonfinite semantics");
        return literal;
    }

    scalar::ExprPtr
    Bound(DataType type,
          const GenericValue& value,
          const std::string& location) {
        auto literal = BoundValue(type, value, location);
        return literal ? scalar::MakeLiteral(literal) : nullptr;
    }

    scalar::ExprPtr
    JsonField(const expr::ColumnInfo& column, const std::string& location) {
        auto raw = column;
        raw.nested_path_.clear();
        return Field(raw, true, location);
    }

    scalar::ExprPtr
    Build(const expr::TypedExprPtr& node,
          const std::string& location,
          size_t depth) {
        auto result = BuildImpl(node, location, depth);
        has_opaque_calls_ |=
            dynamic_cast<const expr::CallExpr*>(node.get()) != nullptr;
        if (!result)
            return result;
        using Tier = scalar::ReorderTier;
        ExpressionProperty property;
        const expr::ColumnInfo* field = nullptr;
        bool like = false, membership = false, may_throw = false;
        if (auto unary =
                dynamic_cast<const expr::UnaryRangeFilterExpr*>(node.get())) {
            field = &unary->column_;
            property.namespace_candidate = field->field_id_.get();
            like = !Comparison(unary->op_type_).has_value();
            may_throw = unary->op_type_ == proto::plan::TextMatch ||
                        unary->op_type_ == proto::plan::PhraseMatch ||
                        unary->op_type_ == proto::plan::TextMatchFuzzy ||
                        unary->op_type_ == proto::plan::Match ||
                        unary->op_type_ == proto::plan::RegexMatch;
        } else if (auto range =
                       dynamic_cast<const expr::BinaryRangeFilterExpr*>(
                           node.get())) {
            field = &range->column_;
        } else if (auto term =
                       dynamic_cast<const expr::TermFilterExpr*>(node.get())) {
            field = &term->column_;
        } else if (auto null =
                       dynamic_cast<const expr::NullExpr*>(node.get())) {
            field = &null->column_;
        } else if (auto bloom =
                       dynamic_cast<const expr::BloomFilterExpr*>(node.get())) {
            field = &bloom->column_;
            membership = true;
        } else if (auto roaring = dynamic_cast<const expr::RoaringFilterExpr*>(
                       node.get())) {
            field = &roaring->column_;
            membership = true;
        } else if (dynamic_cast<const expr::CompareExpr*>(node.get())) {
            property.reorder = {Tier::Compare, true};
        }
        if (field) {
            // Arithmetic, calls, JSON parsing and unreviewed expression kinds
            // remain barriers. Storage failures still propagate; they are not
            // predicate side effects or a reason to silently change routes.
            property.reorder.safe = !field->element_level_ &&
                                    field->nested_path_.empty() &&
                                    IsScalar(field->data_type_) && !may_throw;
            if (field->data_type_ == DataType::JSON)
                property.reorder.tier = Tier::Json;
            else if (membership)
                property.reorder.tier = Tier::Membership;
            else if (IsNumericDataType(field->data_type_))
                property.reorder.tier = Tier::Numeric;
            else if (IsString(field->data_type_))
                property.reorder.tier = like ? Tier::StringLike : Tier::String;
        }
        if (auto predicate =
                dynamic_cast<const scalar::PredicateRefExpr*>(result.get()))
            property.predicate_slot = predicate->slot();
        properties_.emplace(result.get(), property);
        return result;
    }

    scalar::ExprPtr
    BuildImpl(const expr::TypedExprPtr& node,
              const std::string& location,
              size_t depth) {
        AssertInfo(node, "null logical scalar expression");
        if (depth > 256) {
            return UnsupportedAt(location, "scalar compilation depth limit");
        }
        if (auto unary =
                dynamic_cast<const expr::UnaryRangeFilterExpr*>(node.get());
            unary && (unary->op_type_ == proto::plan::TextMatch ||
                      unary->op_type_ == proto::plan::PhraseMatch ||
                      unary->op_type_ == proto::plan::TextMatchFuzzy)) {
            const auto& field = unary->column_;
            if (!options_.plan_indexes || !IsString(field.data_type_) ||
                field.element_level_ || !field.nested_path_.empty() ||
                !unary->val_.has_string_val() ||
                options_.upstream_columns.count(field.field_id_.get()))
                return UnsupportedAt(location, "text index predicate domain");
            for (const auto& argument : unary->extra_values_)
                if (!argument.has_int64_val())
                    return UnsupportedAt(location, "text query argument type");
            PredicatePlan predicate;
            predicate.field.relation = options_.relation;
            predicate.field.field_id = field.field_id_.get();
            predicate.field.type = field.data_type_;
            predicate.field.nullable = field.nullable_;
            predicate.text_query =
                segcore::reader::TextIndexQuery{unary->op_type_,
                                                unary->val_.string_val(),
                                                unary->extra_values_};
            auto slot = static_cast<column_index_t>(predicates_.size());
            predicates_.push_back(std::move(predicate));
            return std::make_shared<scalar::PredicateRefExpr>(slot);
        }
        if (options_.plan_indexes && !exact_inputs_) {
            const expr::ColumnInfo* field = nullptr;
            auto query = IndexQuery(node, field);
            auto typed = query ? std::nullopt : ArrayQuery(node);
            if (!query && !typed)
                typed = TypedQuery(node);
            std::optional<segcore::reader::JsonStatsQuery> stats;
            if (auto unary =
                    dynamic_cast<const expr::UnaryRangeFilterExpr*>(node.get());
                unary && unary->column_.data_type_ == DataType::JSON &&
                !unary->column_.element_level_ && unary->val_.has_int64_val() &&
                Comparison(unary->op_type_) && unary->extra_values_.empty() &&
                !unary->column_.nested_path_.empty() &&
                !options_.upstream_columns.count(
                    unary->column_.field_id_.get())) {
                const auto& path = unary->column_.nested_path_;
                const bool numeric_component =
                    std::any_of(path.begin(), path.end(), [](const auto& part) {
                        return !part.empty() &&
                               std::all_of(
                                   part.begin(), part.end(), [](char c) {
                                       return c >= '0' && c <= '9';
                                   });
                    });
                if (!numeric_component)
                    stats = segcore::reader::JsonStatsQuery{
                        Json::pointer(path),
                        unary->op_type_,
                        unary->val_.int64_val()};
            }
            if (typed && !typed->pointer.empty()) {
                stats = segcore::reader::JsonStatsQuery{
                    typed->pointer, proto::plan::Equal, 0, *typed};
            }
            auto stats_path = [&](const expr::ColumnInfo& column) {
                if (column.data_type_ != DataType::JSON ||
                    column.element_level_ || column.nested_path_.empty() ||
                    options_.upstream_columns.count(column.field_id_.get()))
                    return false;
                // Shredding does not descend into array elements. Numeric
                // components may mean array addressing, so retain Raw there.
                return std::none_of(
                    column.nested_path_.begin(),
                    column.nested_path_.end(),
                    [](const std::string& part) {
                        return !part.empty() &&
                               std::all_of(
                                   part.begin(), part.end(), [](char c) {
                                       return c >= '0' && c <= '9';
                                   });
                    });
            };
            if (auto arithmetic =
                    dynamic_cast<const expr::BinaryArithOpEvalRangeExpr*>(
                        node.get());
                arithmetic && stats_path(arithmetic->column_) &&
                Comparison(arithmetic->op_type_)) {
                segcore::reader::JsonStatsArithmetic query{
                    arithmetic->arith_op_type_,
                    arithmetic->right_operand_,
                    arithmetic->value_};
                // Only non-throwing row calculations may run outside selection.
                if (query.CanEvaluateWholeSource()) {
                    stats = segcore::reader::JsonStatsQuery{
                        Json::pointer(arithmetic->column_.nested_path_),
                        arithmetic->op_type_,
                        0};
                    stats->arithmetic = std::move(query);
                }
            }
            if (auto arithmetic =
                    dynamic_cast<const expr::BinaryArithOpEvalRangeExpr*>(
                        node.get());
                arithmetic && stats_path(arithmetic->column_) &&
                arithmetic->arith_op_type_ == proto::plan::ArrayLength &&
                Comparison(arithmetic->op_type_)) {
                stats = segcore::reader::JsonStatsQuery{
                    Json::pointer(arithmetic->column_.nested_path_),
                    arithmetic->op_type_,
                    0};
                stats->length_bound = arithmetic->value_;
            }
            if (auto unary =
                    dynamic_cast<const expr::UnaryRangeFilterExpr*>(node.get());
                unary && stats_path(unary->column_) &&
                unary->val_.has_string_val() && unary->extra_values_.empty() &&
                (unary->op_type_ == proto::plan::PrefixMatch ||
                 unary->op_type_ == proto::plan::PostfixMatch ||
                 unary->op_type_ == proto::plan::InnerMatch ||
                 unary->op_type_ == proto::plan::Match ||
                 unary->op_type_ == proto::plan::RegexMatch)) {
                stats = segcore::reader::JsonStatsQuery{
                    Json::pointer(unary->column_.nested_path_),
                    unary->op_type_,
                    0};
                stats->string_pattern = unary->val_.string_val();
            }
            if (auto exists = dynamic_cast<const expr::ExistsExpr*>(node.get());
                exists && stats_path(exists->column_)) {
                segcore::reader::TypedIndexQuery q{};
                q.op = segcore::reader::ScalarIndexOp::Exists;
                q.pointer = Json::pointer(exists->column_.nested_path_);
                stats = segcore::reader::JsonStatsQuery{
                    q.pointer, proto::plan::Equal, 0, std::move(q)};
            }
            if (auto contains =
                    dynamic_cast<const expr::JsonContainsExpr*>(node.get());
                contains && stats_path(contains->column_) &&
                !contains->vals_.empty()) {
                segcore::reader::TypedIndexQuery q{};
                q.op = contains->op_ ==
                               proto::plan::JSONContainsExpr_JSONOp_ContainsAll
                           ? segcore::reader::ScalarIndexOp::ContainsAll
                           : segcore::reader::ScalarIndexOp::ContainsAny;
                q.type = DataType::ARRAY;
                q.pointer = Json::pointer(contains->column_.nested_path_);
                q.terms = contains->vals_;
                q.same_type = contains->same_type_;
                // The recursive Raw compilation below validates the operation
                // and constants before publishing any physical alternative.
                stats = segcore::reader::JsonStatsQuery{
                    q.pointer, proto::plan::Equal, 0, std::move(q)};
            }
            auto* gis =
                dynamic_cast<const expr::GISFunctionFilterExpr*>(node.get());
            std::optional<segcore::reader::GeometryIndexQuery> geometry;
            if (gis &&
                gis->op_ >= proto::plan::GISFunctionFilterExpr_GISOp_Equals &&
                gis->op_ <= proto::plan::GISFunctionFilterExpr_GISOp_Within &&
                gis->column_.nested_path_.empty() &&
                !gis->column_.element_level_ &&
                !options_.upstream_columns.count(gis->column_.field_id_.get()))
                geometry = segcore::reader::GeometryIndexQuery{
                    gis->op_, gis->geometry_wkt_};
            std::optional<segcore::reader::NgramQuery> ngram;
            if (auto unary =
                    dynamic_cast<const expr::UnaryRangeFilterExpr*>(node.get());
                unary && !unary->column_.element_level_ &&
                unary->val_.has_string_val() && unary->extra_values_.empty() &&
                !options_.upstream_columns.count(
                    unary->column_.field_id_.get()) &&
                ((IsString(unary->column_.data_type_) &&
                  unary->column_.nested_path_.empty()) ||
                 stats_path(unary->column_)) &&
                (unary->op_type_ == proto::plan::Match ||
                 unary->op_type_ == proto::plan::RegexMatch ||
                 unary->op_type_ == proto::plan::PrefixMatch ||
                 unary->op_type_ == proto::plan::PostfixMatch ||
                 unary->op_type_ == proto::plan::InnerMatch)) {
                ngram = segcore::reader::NgramQuery{
                    unary->op_type_,
                    unary->val_.string_val(),
                    unary->column_.data_type_ == DataType::JSON
                        ? Json::pointer(unary->column_.nested_path_)
                        : ""};
            }
            std::optional<segcore::reader::JsonStatsPath> selected_stats;
            const expr::ColumnInfo* selected_column = nullptr;
            bool string_sensitive = false, row_sensitive = false;
            if (auto p = dynamic_cast<const expr::UnaryRangeFilterExpr*>(
                    node.get())) {
                selected_column = &p->column_;
                string_sensitive = p->val_.has_string_val();
            } else if (auto p =
                           dynamic_cast<const expr::BinaryRangeFilterExpr*>(
                               node.get())) {
                selected_column = &p->column_;
                string_sensitive = p->lower_val_.has_string_val();
            } else if (auto p = dynamic_cast<const expr::TermFilterExpr*>(
                           node.get())) {
                if (!p->vals_.empty()) {
                    selected_column = &p->column_;
                    string_sensitive = p->vals_[0].has_string_val();
                }
            } else if (auto p = dynamic_cast<const expr::JsonContainsExpr*>(
                           node.get())) {
                if (!p->vals_.empty())
                    selected_column = &p->column_;
            } else if (auto p = dynamic_cast<
                           const expr::BinaryArithOpEvalRangeExpr*>(
                           node.get())) {
                selected_column = &p->column_;
                row_sensitive =
                    p->arith_op_type_ != proto::plan::ArrayLength &&
                    !segcore::reader::JsonStatsArithmetic{
                        p->arith_op_type_, p->right_operand_, p->value_}
                         .CanEvaluateWholeSource();
            }
            if (selected_column &&
                selected_column->data_type_ == DataType::JSON &&
                !selected_column->element_level_ &&
                !selected_column->nested_path_.empty() &&
                !options_.upstream_columns.count(
                    selected_column->field_id_.get())) {
                const auto& path = selected_column->nested_path_;
                bool array_address = std::any_of(
                    path.begin(), path.end(), [](const std::string& part) {
                        return !part.empty() &&
                               std::all_of(
                                   part.begin(), part.end(), [](char c) {
                                       return c >= '0' && c <= '9';
                                   });
                    });
                if (array_address || row_sensitive)
                    selected_stats =
                        segcore::reader::JsonStatsPath{path, string_sensitive};
            }
            if (query || stats || typed || geometry || ngram ||
                selected_stats) {
                std::vector<InputUse> uses;
                struct Guard {
                    std::vector<InputUse>*& active;
                    ~Guard() {
                        active = nullptr;
                    }
                } guard{exact_inputs_};
                exact_inputs_ = &uses;
                scalar::ExprPtr exact;
                std::function<scalar::ExprPtr()> prepare_exact;
                if (auto term =
                        dynamic_cast<const expr::TermFilterExpr*>(node.get());
                    term && !term->is_in_field_ &&
                    IsScalar(term->column_.data_type_) &&
                    term->column_.nested_path_.empty() &&
                    !term->column_.element_level_) {
                    auto input = Field(term->column_, true, location);
                    auto constants =
                        TermConstants(term->column_.data_type_, term->vals_);
                    if (!constants)
                        return UnsupportedAt(location,
                                             "term literal conversion");
                    if (!input)
                        return nullptr;
                    prepare_exact = [input = std::move(input),
                                     constants = std::move(*constants)] {
                        return scalar::MakeTerm(input, constants);
                    };
                } else {
                    exact = Build(node, location, depth);
                    if (!exact)
                        return nullptr;
                }
                AssertInfo(!uses.empty(),
                           "index alternative has no raw dependency");
                AssertInfo(predicates_.size() <
                               std::numeric_limits<column_index_t>::max(),
                           "too many predicate slots");
                auto slot = static_cast<column_index_t>(predicates_.size());
                predicates_.push_back(
                    {layout_.slots()[uses.front().slot],
                     query.value_or(segcore::reader::Int64IndexQuery{}),
                     std::move(exact),
                     std::move(uses),
                     std::move(stats),
                     std::move(typed),
                     std::nullopt,
                     std::move(geometry),
                     std::move(ngram),
                     std::move(selected_stats),
                     std::nullopt,
                     std::move(prepare_exact)});
                return std::make_shared<scalar::PredicateRefExpr>(slot);
            }
        }
        auto children = [&] {
            std::vector<scalar::ExprPtr> compiled;
            for (size_t i = 0; i < node->inputs().size(); ++i) {
                compiled.push_back(
                    Build(node->inputs()[i],
                          location + "/child[" + std::to_string(i) + "]",
                          depth + 1));
            }
            return compiled;
        };
        auto complete = [](const auto& inputs) {
            return std::all_of(
                inputs.begin(), inputs.end(), [](const auto& input) {
                    return input != nullptr;
                });
        };
        if (auto gis =
                dynamic_cast<const expr::GISFunctionFilterExpr*>(node.get())) {
            if (gis->column_.data_type_ != DataType::GEOMETRY ||
                gis->op_ < proto::plan::GISFunctionFilterExpr_GISOp_Equals ||
                gis->op_ > proto::plan::GISFunctionFilterExpr_GISOp_STIsValid)
                return UnsupportedAt(location, "GISFunctionFilterExpr");
            auto input = Field(gis->column_, true, location);
            return input ? scalar::MakeGeometryPredicate(input,
                                                         gis->op_,
                                                         gis->geometry_wkt_,
                                                         gis->distance_)
                         : nullptr;
        }
        if (auto match = dynamic_cast<const expr::MatchExpr*>(node.get())) {
            if (options_.plan_indexes && !exact_inputs_ && options_.schema &&
                node->inputs().size() == 1) {
                auto child = ElementPredicate(node->inputs()[0],
                                              *options_.schema,
                                              match->get_struct_name());
                if (!child)
                    return UnsupportedAt(location, "MATCH element predicate");
                CompileOptions child_options;
                child_options.plan_indexes = true;
                child_options.array_element_arithmetic = true;
                auto indexed = CompileExpressions({child}, child_options);
                if (!indexed.supported())
                    return UnsupportedAt(location, "MATCH child capability");
                std::vector<InputUse> uses;
                struct Guard {
                    std::vector<InputUse>*& active;
                    ~Guard() {
                        active = nullptr;
                    }
                } guard{exact_inputs_};
                exact_inputs_ = &uses;
                auto exact = Build(node, location, depth);
                if (!exact)
                    return nullptr;
                const auto& anchor =
                    options_.schema->GetFirstArrayFieldInStruct(
                        match->get_struct_name());
                PredicatePlan predicate;
                predicate.field = layout_.slots()[uses.front().slot];
                predicate.exact = std::move(exact);
                predicate.raw_inputs = std::move(uses);
                predicate.match = MatchPlan{indexed.plan,
                                            options_.schema,
                                            anchor.get_id().get(),
                                            match->get_match_type(),
                                            match->get_count()};
                auto slot = predicates_.size();
                predicates_.push_back(std::move(predicate));
                return std::make_shared<scalar::PredicateRefExpr>(slot);
            }
            if (!options_.schema || node->inputs().size() != 1 ||
                match->get_match_type() < proto::plan::MatchAll ||
                match->get_match_type() > proto::plan::MatchExact)
                return UnsupportedAt(location, "MatchExpr");
            auto child = ElementPredicate(
                node->inputs()[0], *options_.schema, match->get_struct_name());
            if (!child)
                return UnsupportedAt(location, "MATCH element predicate");
            CompileOptions child_options;
            child_options.array_element_arithmetic = true;
            auto compiled = CompileExpressions({child}, child_options);
            if (!compiled.supported())
                return UnsupportedAt(location, "MATCH child capability");
            const auto& anchor_meta =
                options_.schema->GetFirstArrayFieldInStruct(
                    match->get_struct_name());
            if (anchor_meta.get_data_type() != DataType::VECTOR_ARRAY &&
                (anchor_meta.get_data_type() != DataType::ARRAY ||
                 (!IsScalar(anchor_meta.get_element_type()) &&
                  !anchor_meta.is_nested_array())))
                return UnsupportedAt(location, "MATCH anchor type");
            auto array_input = [&](FieldId id) {
                const auto& meta = (*options_.schema)[id];
                return Field(expr::ColumnInfo(id,
                                              meta.get_data_type(),
                                              meta.get_element_type(),
                                              {},
                                              meta.is_nullable()),
                             true,
                             location,
                             true);
            };
            auto anchor = array_input(anchor_meta.get_id());
            std::vector<scalar::ExprPtr> arrays;
            for (const auto& dep : compiled.plan->inputs().slots()) {
                auto input = array_input(FieldId(dep.field_id));
                if (!input)
                    return nullptr;
                arrays.push_back(std::move(input));
            }
            return anchor ? scalar::MakeArrayMatch(
                                anchor,
                                std::move(arrays),
                                compiled.plan->inputs().schema(),
                                compiled.plan->expressions().roots()[0],
                                match->get_match_type(),
                                match->get_count())
                          : nullptr;
        }
        if (auto result =
                dynamic_cast<const expr::ResultColumnExpr*>(node.get())) {
            const auto found = options_.upstream_results.find(result->id());
            if (found == options_.upstream_results.end())
                return UnsupportedAt(location, "unbound result column");
            if (options_.upstream_schema->column_type(found->second) !=
                result->type())
                return UnsupportedAt(location, "result column type mismatch");
            InputDependency dependency{options_.relation, 0, result->type()};
            dependency.result_column = result->id();
            dependency.upstream_column = found->second;
            dependency.nullable = result->nullable();
            dependency.needs_values = true;
            dependency.needs_validity = true;
            const auto slot = layout_.Intern(std::move(dependency));
            (exact_inputs_ ? *exact_inputs_ : direct_inputs_)
                .push_back({slot, true});
            return std::make_shared<scalar::InputRefExpr>(slot, result->type());
        }
        if (auto arithmetic =
                dynamic_cast<const expr::BinaryArithExpr*>(node.get())) {
            const auto op = arithmetic->op();
            if (op != proto::plan::Add && op != proto::plan::Sub &&
                op != proto::plan::Mul && op != proto::plan::Div)
                return UnsupportedAt(location, "scalar arithmetic operation");
            auto left =
                Build(node->inputs()[0], location + "/child[0]", depth + 1);
            auto right =
                Build(node->inputs()[1], location + "/child[1]", depth + 1);
            return left && right ? scalar::MakeArithmetic(left, right, op)
                                 : nullptr;
        }
        if (auto value = dynamic_cast<const expr::ValueExpr*>(node.get())) {
            return Bound(value->type(), value->GetGenericValue(), location);
        }
        if (dynamic_cast<const expr::AlwaysTrueExpr*>(node.get())) {
            return scalar::MakeLiteral(Constant(DataType::BOOL, true));
        }
        if (auto column = dynamic_cast<const expr::ColumnExpr*>(node.get())) {
            if (!column->GetColumn().nested_path_.empty())
                return UnsupportedAt(location, "path projection logical type");
            return Field(column->GetColumn(), true, location);
        }
        if (auto unary =
                dynamic_cast<const expr::UnaryRangeFilterExpr*>(node.get())) {
            auto op = Comparison(unary->op_type_);
            if (unary->column_.data_type_ == DataType::ARRAY &&
                unary->column_.nested_path_.empty() &&
                !unary->column_.element_level_ && unary->val_.has_array_val() &&
                unary->extra_values_.empty() &&
                (unary->op_type_ == proto::plan::Equal ||
                 unary->op_type_ == proto::plan::NotEqual) &&
                JsonConstant(unary->val_)) {
                auto input = Field(unary->column_, true, location);
                return input ? scalar::MakeArrayEquality(
                                   input,
                                   unary->val_.array_val(),
                                   unary->op_type_ == proto::plan::NotEqual)
                             : nullptr;
            }
            if (unary->column_.data_type_ == DataType::JSON) {
                const bool pattern =
                    unary->val_.has_string_val() &&
                    (unary->op_type_ == proto::plan::PrefixMatch ||
                     unary->op_type_ == proto::plan::PostfixMatch ||
                     unary->op_type_ == proto::plan::InnerMatch ||
                     unary->op_type_ == proto::plan::Match ||
                     unary->op_type_ == proto::plan::RegexMatch);
                if ((!op && !pattern) || !unary->extra_values_.empty() ||
                    !JsonConstant(unary->val_) ||
                    (unary->val_.has_array_val() &&
                     unary->op_type_ != proto::plan::Equal &&
                     unary->op_type_ != proto::plan::NotEqual))
                    return UnsupportedAt(location,
                                         "JSON unary operation/constants");
                auto input = JsonField(unary->column_, location);
                return input ? scalar::MakeJsonComparison(
                                   input,
                                   unary->column_.nested_path_,
                                   unary->op_type_,
                                   unary->val_)
                             : nullptr;
            }
            if (!op && unary->extra_values_.empty() &&
                (unary->column_.data_type_ == DataType::VARCHAR ||
                 unary->column_.data_type_ == DataType::STRING ||
                 unary->column_.data_type_ == DataType::TEXT ||
                 (unary->column_.data_type_ == DataType::ARRAY &&
                  !unary->column_.nested_path_.empty() &&
                  IsString(unary->column_.element_type_))) &&
                unary->val_.val_case() == GenericValue::kStringVal) {
                std::optional<scalar::StringOp> pattern;
                switch (unary->op_type_) {
                    case proto::plan::PrefixMatch:
                        pattern = scalar::StringOp::Prefix;
                        break;
                    case proto::plan::PostfixMatch:
                        pattern = scalar::StringOp::Suffix;
                        break;
                    case proto::plan::InnerMatch:
                        pattern = scalar::StringOp::Contains;
                        break;
                    case proto::plan::Match:
                        pattern = scalar::StringOp::Like;
                        break;
                    case proto::plan::RegexMatch:
                        pattern = scalar::StringOp::Regex;
                        break;
                    default:
                        break;
                }
                if (pattern) {
                    auto input = Field(unary->column_, true, location);
                    return input
                               ? scalar::MakeStringMatch(
                                     input, *pattern, unary->val_.string_val())
                               : nullptr;
                }
            }
            if (!op || !unary->extra_values_.empty()) {
                return UnsupportedAt(location,
                                     "unary pattern/text/special operation");
            }
            auto input = Field(unary->column_, true, location);
            if (input && IsInteger(input->type()) &&
                unary->val_.has_int64_val() &&
                !Literal(input->type(), unary->val_))
                return scalar::MakeComparison(
                    *op, input, Bound(DataType::INT64, unary->val_, location));
            auto literal =
                Bound(input ? input->type() : unary->column_.data_type_,
                      unary->val_,
                      location);
            return input && literal
                       ? scalar::MakeComparison(*op, input, literal)
                       : nullptr;
        }
        if (auto range =
                dynamic_cast<const expr::BinaryRangeFilterExpr*>(node.get())) {
            if (range->column_.data_type_ == DataType::JSON) {
                const auto numeric = [](const auto& value) {
                    return value.has_int64_val() || value.has_float_val();
                };
                if (!JsonConstant(range->lower_val_, false) ||
                    !JsonConstant(range->upper_val_, false) ||
                    !((numeric(range->lower_val_) &&
                       numeric(range->upper_val_)) ||
                      (range->lower_val_.has_string_val() &&
                       range->upper_val_.has_string_val())))
                    return UnsupportedAt(location, "JSON range bound types");
                auto input = JsonField(range->column_, location);
                return input
                           ? scalar::MakeJsonRange(input,
                                                   range->column_.nested_path_,
                                                   range->lower_val_,
                                                   range->upper_val_,
                                                   range->lower_inclusive_,
                                                   range->upper_inclusive_)
                           : nullptr;
            }
            auto input = Field(range->column_, true, location);
            if (input && IsInteger(input->type()) &&
                range->lower_val_.has_int64_val() &&
                range->upper_val_.has_int64_val() &&
                (!Literal(input->type(), range->lower_val_) ||
                 !Literal(input->type(), range->upper_val_))) {
                return scalar::MakeLogical(
                    scalar::LogicalOp::And,
                    {scalar::MakeComparison(
                         range->lower_inclusive_
                             ? scalar::CompareOp::GreaterEqual
                             : scalar::CompareOp::Greater,
                         input,
                         Bound(DataType::INT64, range->lower_val_, location)),
                     scalar::MakeComparison(
                         range->upper_inclusive_ ? scalar::CompareOp::LessEqual
                                                 : scalar::CompareOp::Less,
                         input,
                         Bound(DataType::INT64, range->upper_val_, location))});
            }
            auto lower =
                BoundValue(input ? input->type() : range->column_.data_type_,
                           range->lower_val_,
                           location);
            auto upper =
                BoundValue(input ? input->type() : range->column_.data_type_,
                           range->upper_val_,
                           location);
            if (!input || !lower || !upper)
                return nullptr;
            return scalar::MakeRange(input,
                                     lower,
                                     upper,
                                     range->lower_inclusive_,
                                     range->upper_inclusive_);
        }
        if (auto compare = dynamic_cast<const expr::CompareExpr*>(node.get())) {
            auto op = Comparison(compare->op_type_);
            const bool strings = IsString(compare->left_data_type_) &&
                                 IsString(compare->right_data_type_);
            if (strings && compare->op_type_ == proto::plan::PrefixMatch)
                op = scalar::CompareOp::Prefix;
            if (!op || !IsScalar(compare->left_data_type_) ||
                !IsScalar(compare->right_data_type_) ||
                IsString(compare->left_data_type_) !=
                    IsString(compare->right_data_type_)) {
                return UnsupportedAt(location,
                                     "mixed-type/special column comparison");
            }
            auto left = Field(expr::ColumnInfo(compare->left_field_id_,
                                               compare->left_data_type_),
                              true,
                              location);
            auto right = Field(expr::ColumnInfo(compare->right_field_id_,
                                                compare->right_data_type_),
                               true,
                               location);
            return left && right ? scalar::MakeComparison(*op, left, right)
                                 : nullptr;
        }
        if (auto null = dynamic_cast<const expr::NullExpr*>(node.get())) {
            if (!null->column_.nested_path_.empty())
                return UnsupportedAt(location, "path NULL predicate");
            if (null->op_ != proto::plan::NullExpr_NullOp_IsNull &&
                null->op_ != proto::plan::NullExpr_NullOp_IsNotNull) {
                return UnsupportedAt(location, "unknown NULL operation");
            }
            auto input = Field(null->column_, false, location);
            return input ? scalar::MakeIsNull(
                               input,
                               null->op_ ==
                                   proto::plan::NullExpr_NullOp_IsNotNull)
                         : nullptr;
        }
        if (auto arithmetic =
                dynamic_cast<const expr::BinaryArithOpEvalRangeExpr*>(
                    node.get())) {
            const auto source_type = arithmetic->column_.data_type_;
            const auto type =
                source_type == DataType::ARRAY &&
                        !arithmetic->column_.nested_path_.empty()
                    ? ArrayValueType(arithmetic->column_.element_type_)
                    : source_type;
            const auto operation = arithmetic->arith_op_type_;
            if (operation == proto::plan::ArrayLength &&
                (source_type == DataType::JSON ||
                 source_type == DataType::ARRAY ||
                 source_type == DataType::VECTOR_ARRAY) &&
                !arithmetic->column_.element_level_ &&
                (source_type == DataType::JSON ||
                 arithmetic->column_.nested_path_.empty())) {
                auto comparison = Comparison(arithmetic->op_type_);
                if (!comparison ||
                    !(arithmetic->value_.has_int64_val() ||
                      (arithmetic->value_.has_float_val() &&
                       std::isfinite(arithmetic->value_.float_val()))))
                    return UnsupportedAt(location, "collection length bound");
                auto input =
                    source_type == DataType::JSON
                        ? JsonField(arithmetic->column_, location)
                        : Field(arithmetic->column_, true, location, true);
                auto bound =
                    Bound(arithmetic->value_.has_int64_val() ? DataType::INT64
                                                             : DataType::DOUBLE,
                          arithmetic->value_,
                          location);
                return input && bound
                           ? scalar::MakeComparison(
                                 *comparison,
                                 scalar::MakeCollectionLength(
                                     input, arithmetic->column_.nested_path_),
                                 bound)
                           : nullptr;
            }
            if (source_type == DataType::JSON &&
                !arithmetic->column_.element_level_) {
                auto comparison = Comparison(arithmetic->op_type_);
                const auto& value = arithmetic->value_;
                const auto& operand = arithmetic->right_operand_;
                const bool numeric =
                    (value.has_int64_val() && operand.has_int64_val()) ||
                    (value.has_float_val() && operand.has_float_val() &&
                     std::isfinite(value.float_val()) &&
                     std::isfinite(operand.float_val()));
                if (!comparison || !numeric || operation < proto::plan::Add ||
                    operation > proto::plan::Shr ||
                    operation == proto::plan::ArrayLength)
                    return UnsupportedAt(location,
                                         "JSON arithmetic operation/constants");
                auto input = JsonField(arithmetic->column_, location);
                return input ? scalar::MakeJsonArithmeticCompare(
                                   input,
                                   arithmetic->column_.nested_path_,
                                   operation,
                                   *comparison,
                                   operand,
                                   value)
                             : nullptr;
            }
            const bool basic = operation == proto::plan::Add ||
                               operation == proto::plan::Sub ||
                               operation == proto::plan::Mul ||
                               operation == proto::plan::Div;
            const bool integer = operation == proto::plan::Mod ||
                                 operation == proto::plan::BitAnd ||
                                 operation == proto::plan::BitOr ||
                                 operation == proto::plan::BitXor ||
                                 operation == proto::plan::Shl ||
                                 operation == proto::plan::Shr;
            const bool array_arithmetic = source_type == DataType::ARRAY ||
                                          options_.array_element_arithmetic;
            auto comparison = Comparison(arithmetic->op_type_);
            if (!comparison ||
                !(IsInteger(type) || type == DataType::FLOAT ||
                  type == DataType::DOUBLE) ||
                !(basic ||
                  ((IsInteger(type) || !array_arithmetic) && integer) ||
                  (array_arithmetic && operation == proto::plan::Mod)))
                return UnsupportedAt(location, "BinaryArithOpEvalRangeExpr");
            auto input = Field(arithmetic->column_, true, location);
            const auto constant_type = IsInteger(type) ? DataType::INT64 : type;
            auto operand =
                BoundValue(constant_type, arithmetic->right_operand_, location);
            auto bound =
                BoundValue(constant_type, arithmetic->value_, location);
            return input && operand && bound
                       ? scalar::MakeArithmeticCompare(
                             input,
                             operation,
                             *comparison,
                             operand,
                             bound,
                             !IsInteger(type) && !array_arithmetic && integer)
                       : nullptr;
        }
        if (auto timestamp =
                dynamic_cast<const expr::TimestamptzArithCompareExpr*>(
                    node.get())) {
            auto comparison = Comparison(timestamp->compare_op_);
            if (!comparison ||
                timestamp->column_.data_type_ != DataType::TIMESTAMPTZ ||
                (timestamp->arith_op_ != proto::plan::Add &&
                 timestamp->arith_op_ != proto::plan::Sub &&
                 timestamp->arith_op_ != proto::plan::Unknown))
                return UnsupportedAt(location, "TimestamptzArithCompareExpr");
            auto input = Field(timestamp->column_, true, location);
            auto bound = Bound(
                DataType::TIMESTAMPTZ, timestamp->compare_value_, location);
            if (!input || !bound)
                return nullptr;
            auto calculated = scalar::MakeCall(
                DataType::TIMESTAMPTZ,
                {input},
                [operation = timestamp->arith_op_,
                 interval = timestamp->interval_](
                    const auto& inputs, SelectionView rows, scalar::EvalCtx&) {
                    VectorResultBuilder output(DataType::TIMESTAMPTZ,
                                               rows.row_count());
                    scalar::ProjectPrimitive<int64_t, int64_t>(
                        inputs[0], rows, output, [&](int64_t value) {
                            return scalar::ApplyTimestampInterval(
                                value, operation, interval);
                        });
                    return std::move(output).Finish(rows);
                });
            return scalar::MakeComparison(*comparison, calculated, bound);
        }
        if (auto term = dynamic_cast<const expr::TermFilterExpr*>(node.get())) {
            if (term->column_.data_type_ == DataType::ARRAY &&
                term->is_in_field_ && !term->column_.element_level_ &&
                term->column_.nested_path_.empty()) {
                const auto type = ArrayValueType(term->column_.element_type_);
                if (term->vals_.size() != 1 ||
                    !ArrayTermCompatible(type, term->vals_[0]))
                    return UnsupportedAt(location, "ARRAY in-field constant");
                auto input = Field(term->column_, true, location);
                return input ? scalar::MakeArrayContains(
                                   input, type, term->vals_, false)
                             : nullptr;
            }
            if (term->column_.data_type_ == DataType::ARRAY &&
                !term->is_in_field_ && !term->column_.element_level_ &&
                term->vals_.empty())
                return scalar::MakeLiteral(Constant(DataType::BOOL, false));
            if (term->column_.data_type_ == DataType::JSON &&
                !term->column_.element_level_) {
                if (term->is_in_field_ && term->vals_.size() != 1)
                    return UnsupportedAt(location,
                                         "JSON in-field requires one constant");
                for (const auto& value : term->vals_)
                    if (!JsonConstant(value, false) ||
                        value.val_case() != term->vals_[0].val_case())
                        return UnsupportedAt(location,
                                             "JSON term constant types");
                // The legacy empty JSON term branch produces valid FALSE,
                // including on SQL NULL or a missing path.
                if (term->vals_.empty())
                    return scalar::MakeLiteral(Constant(DataType::BOOL, false));
                auto input = JsonField(term->column_, location);
                return input ? scalar::MakeJsonTerm(input,
                                                    term->column_.nested_path_,
                                                    term->vals_,
                                                    term->is_in_field_)
                             : nullptr;
            }
            if (term->is_in_field_ ||
                (!IsScalar(term->column_.data_type_) &&
                 !(term->column_.data_type_ == DataType::ARRAY &&
                   !term->column_.nested_path_.empty())))
                return UnsupportedAt(location,
                                     "TermFilterExpr complex/in-field");
            auto input = Field(term->column_, true, location);
            if (!input)
                return nullptr;
            auto constants = TermConstants(input->type(), term->vals_);
            if (!constants)
                return UnsupportedAt(location, "term literal conversion");
            return scalar::MakeTerm(input, *constants);
        }
        if (auto exists = dynamic_cast<const expr::ExistsExpr*>(node.get())) {
            if (exists->column_.data_type_ != DataType::JSON ||
                exists->column_.element_level_)
                return UnsupportedAt(location, "ExistsExpr non-JSON/element");
            auto raw = exists->column_;
            raw.nested_path_.clear();
            auto input = Field(raw, true, location);
            return input ? scalar::MakeJsonExists(input,
                                                  exists->column_.nested_path_)
                         : nullptr;
        }
        if (auto contains =
                dynamic_cast<const expr::JsonContainsExpr*>(node.get())) {
            const bool all = contains->op_ ==
                             proto::plan::JSONContainsExpr_JSONOp_ContainsAll;
            if (contains->column_.data_type_ == DataType::ARRAY &&
                !contains->column_.element_level_ &&
                contains->column_.nested_path_.empty() &&
                (all ||
                 contains->op_ ==
                     proto::plan::JSONContainsExpr_JSONOp_Contains ||
                 contains->op_ ==
                     proto::plan::JSONContainsExpr_JSONOp_ContainsAny)) {
                auto type =
                    contains->column_.element_type_ == DataType::FLOAT
                        ? DataType::FLOAT
                        : ArrayValueType(contains->column_.element_type_);
                for (const auto& value : contains->vals_)
                    if (!ArrayTermCompatible(type, value))
                        return UnsupportedAt(location,
                                             "ARRAY contains constants");
                auto input = Field(contains->column_, true, location);
                return input ? scalar::MakeArrayContains(
                                   input, type, contains->vals_, all)
                             : nullptr;
            }
            if (contains->column_.data_type_ != DataType::JSON ||
                contains->column_.element_level_ ||
                !(all ||
                  contains->op_ ==
                      proto::plan::JSONContainsExpr_JSONOp_Contains ||
                  contains->op_ ==
                      proto::plan::JSONContainsExpr_JSONOp_ContainsAny))
                return UnsupportedAt(location, "JsonContainsExpr");
            for (const auto& value : contains->vals_)
                if (!JsonConstant(value) ||
                    (contains->same_type_ &&
                     value.val_case() != contains->vals_[0].val_case()))
                    return UnsupportedAt(location,
                                         "JSON contains constant types");
            if (contains->vals_.empty())
                return scalar::MakeLiteral(Constant(DataType::BOOL, all));
            auto input = JsonField(contains->column_, location);
            return input ? scalar::MakeJsonContains(
                               input,
                               contains->column_.nested_path_,
                               contains->vals_,
                               all,
                               contains->same_type_)
                         : nullptr;
        }
        if (auto bloom =
                dynamic_cast<const expr::BloomFilterExpr*>(node.get())) {
            const auto type = bloom->column_.data_type_;
            if (type != DataType::INT8 && type != DataType::INT16 &&
                type != DataType::INT32 && type != DataType::INT64 &&
                type != DataType::VARCHAR && type != DataType::JSON)
                return UnsupportedAt(location, "BloomFilterExpr input type");
            auto raw = bloom->column_;
            if (type == DataType::JSON)
                raw.nested_path_.clear();
            auto input = Field(raw, true, location);
            return input ? scalar::MakeBloomMembership(
                               input,
                               bloom->filter_blob_,
                               bloom->column_.nested_path_)
                         : nullptr;
        }
        if (auto roaring =
                dynamic_cast<const expr::RoaringFilterExpr*>(node.get())) {
            if (roaring->column_.data_type_ != DataType::INT64 &&
                roaring->column_.data_type_ != DataType::INT32 &&
                roaring->column_.data_type_ != DataType::INT16 &&
                roaring->column_.data_type_ != DataType::INT8)
                return UnsupportedAt(location, "RoaringFilterExpr non-integer");
            auto input = Field(roaring->column_, true, location);
            return input ? scalar::MakeRoaringMembership(input,
                                                         roaring->membership_)
                         : nullptr;
        }
        if (auto logical =
                dynamic_cast<const expr::LogicalBinaryExpr*>(node.get())) {
            AssertInfo(node->inputs().size() == 2,
                       "logical binary arity mismatch");
            auto inputs = children();
            if (logical->op_type_ != expr::LogicalBinaryExpr::OpType::And &&
                logical->op_type_ != expr::LogicalBinaryExpr::OpType::Or) {
                return UnsupportedAt(location,
                                     "unknown logical binary operation");
            }
            return complete(inputs)
                       ? scalar::MakeLogical(
                             logical->op_type_ ==
                                     expr::LogicalBinaryExpr::OpType::And
                                 ? scalar::LogicalOp::And
                                 : scalar::LogicalOp::Or,
                             std::move(inputs))
                       : nullptr;
        }
        if (auto logical =
                dynamic_cast<const expr::LogicalUnaryExpr*>(node.get())) {
            AssertInfo(node->inputs().size() == 1,
                       "logical unary arity mismatch");
            auto inputs = children();
            if (logical->op_type_ !=
                expr::LogicalUnaryExpr::OpType::LogicalNot) {
                return UnsupportedAt(location,
                                     "unknown logical unary operation");
            }
            return complete(inputs) ? scalar::MakeNot(inputs[0]) : nullptr;
        }
        if (auto call = dynamic_cast<const expr::CallExpr*>(node.get())) {
            auto inputs = children();
            if (!call->function_ptr()) {
                return UnsupportedAt(location, "unbound legacy function");
            }
            return complete(inputs)
                       ? scalar::MakeLegacyCall(call->type(),
                                                std::move(inputs),
                                                call->function_ptr())
                       : nullptr;
        }
        UnsupportedAt(location, UnmigratedNode(*node));
        children();  // collect nested unsupported features too, without evaluating
        return nullptr;
    }

    const CompileOptions& options_;
    InputLayout layout_;
    std::vector<InputUse> direct_inputs_;
    std::vector<InputUse>* exact_inputs_ = nullptr;
    std::vector<PredicatePlan> predicates_;
    ExpressionProperties properties_;
    bool has_opaque_calls_ = false;
    std::vector<UnsupportedFeature> unsupported_;
};

}  // namespace

CompileResult
CompileExpressions(const std::vector<expr::TypedExprPtr>& roots,
                   const CompileOptions& options) {
    return Compiler(options).Run(roots);
}

expr::TypedExprPtr
LowerElementPredicate(const expr::TypedExprPtr& root,
                      const Schema& schema,
                      const std::string& structure) {
    return ElementPredicate(root, schema, structure);
}

CompileResult
CompileFilter(const expr::TypedExprPtr& predicate,
              const expr::TypedExprPtr& ttl_predicate,
              const CompileOptions& options) {
    AssertInfo(predicate && predicate->type() == DataType::BOOL,
               "filter requires a BOOL predicate");
    if (ttl_predicate) {
        AssertInfo(ttl_predicate->type() == DataType::BOOL,
                   "TTL requires a BOOL predicate");
        return CompileExpressions({std::make_shared<expr::LogicalBinaryExpr>(
                                      expr::LogicalBinaryExpr::OpType::And,
                                      predicate,
                                      ttl_predicate)},
                                  options);
    }
    return CompileExpressions({predicate}, options);
}

}  // namespace milvus::exec::plan
