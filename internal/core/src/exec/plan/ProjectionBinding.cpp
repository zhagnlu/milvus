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

#include "exec/plan/ProjectionBinding.h"
#include "exec/plan/ExprCompiler.h"

#include <map>
#include <algorithm>
#include <set>

namespace milvus::exec::plan {
namespace {
using Descriptor = ResultColumnDescriptor;
using Symbols = std::map<std::string, Descriptor>;
using Available = std::map<ResultColumnId, Descriptor>;

bool
ScalarType(DataType type) {
    return IsNumericDataType(type) || type == DataType::BOOL ||
           type == DataType::VARCHAR || type == DataType::STRING ||
           type == DataType::TEXT || type == DataType::TIMESTAMPTZ;
}
expr::TypedExprPtr
Ref(const Descriptor& column) {
    return std::make_shared<expr::ResultColumnExpr>(
        column.id, column.type, column.nullable);
}
std::string
InternalName(ResultColumnId id) {
    return "__projection_" + std::to_string(id.value);
}
void
Require(bool ok, size_t item, const char* reason) {
    if (!ok)
        ThrowInfo(ExprInvalid, "projection item[{}]: {}", item, reason);
}
struct BoundValue {
    expr::TypedExprPtr expression;
    bool nullable;
};
BoundValue
Bind(const expr::TypedExprPtr& value,
     const Symbols& symbols,
     size_t item,
     size_t depth) {
    Require(
        value && depth < 128, item, "missing expression or excessive nesting");
    if (auto name =
            dynamic_cast<const expr::ProjectionNameExpr*>(value.get())) {
        auto found = symbols.find(name->name());
        if (found == symbols.end())
            ThrowInfo(ExprInvalid,
                      "projection item[{}]: unknown or forward reference '{}'",
                      item,
                      name->name().substr(0, 64));
        return {Ref(found->second), found->second.nullable};
    }
    if (auto column = dynamic_cast<const expr::ColumnExpr*>(value.get())) {
        const auto& info = column->GetColumn();
        Require(info.nested_path_.empty() && !info.element_level_,
                item,
                "path/element references are not supported");
        for (const auto& [name, d] : symbols)
            if (d.source_field && *d.source_field == info.field_id_) {
                Require(d.type == info.data_type_,
                        item,
                        "storage column type mismatch");
                return {Ref(d), d.nullable};
            }
        Require(false, item, "unknown storage column");
    }
    if (auto literal = dynamic_cast<const expr::ValueExpr*>(value.get())) {
        Require(ScalarType(literal->type()), item, "unsupported literal type");
        // Explicit DOUBLE preserves the protocol literal's precision.
        return {
            std::make_shared<expr::ValueExpr>(
                literal->GetGenericValue(),
                literal->GetGenericValue().has_float_val() ? DataType::DOUBLE
                                                           : literal->type()),
            false};
    }
    auto unbound =
        dynamic_cast<const expr::ProjectionArithmeticExpr*>(value.get());
    auto typed = dynamic_cast<const expr::BinaryArithExpr*>(value.get());
    Require(
        unbound || typed,
        item,
        "unsupported projection expression (functions/CASE are not enabled)");
    auto op = unbound ? unbound->op() : typed->op();
    Require(op == proto::plan::Add || op == proto::plan::Sub ||
                op == proto::plan::Mul || op == proto::plan::Div,
            item,
            "unsupported arithmetic operation");
    Require(
        value->inputs().size() == 2, item, "arithmetic requires two operands");
    auto a = Bind(value->inputs()[0], symbols, item, depth + 1);
    auto b = Bind(value->inputs()[1], symbols, item, depth + 1);
    Require(IsNumericDataType(a.expression->type()) &&
                IsNumericDataType(b.expression->type()) &&
                a.expression->type() != DataType::BOOL &&
                b.expression->type() != DataType::BOOL,
            item,
            "arithmetic requires numeric operands");
    return {
        std::make_shared<expr::BinaryArithExpr>(op, a.expression, b.expression),
        a.nullable || b.nullable};
}

bool
ValidateExpr(const expr::TypedExprPtr& expression,
             const Available& available,
             std::set<ResultColumnId>& used,
             size_t depth = 0) {
    AssertInfo(expression && depth < 128,
               "invalid bound projection expression");
    if (auto ref =
            dynamic_cast<const expr::ResultColumnExpr*>(expression.get())) {
        auto found = available.find(ref->id());
        AssertInfo(found != available.end() &&
                       found->second.type == ref->type() &&
                       found->second.nullable == ref->nullable(),
                   "bound projection references missing/forward result or "
                   "inconsistent type/nullability");
        used.insert(ref->id());
        return ref->nullable();
    }
    if (auto literal = dynamic_cast<const expr::ValueExpr*>(expression.get())) {
        AssertInfo(ScalarType(literal->type()),
                   "unsupported bound projection literal");
        return false;
    }
    auto arithmetic =
        dynamic_cast<const expr::BinaryArithExpr*>(expression.get());
    AssertInfo(arithmetic && expression->inputs().size() == 2,
               "unsupported bound projection expression");
    AssertInfo(arithmetic->op() == proto::plan::Add ||
                   arithmetic->op() == proto::plan::Sub ||
                   arithmetic->op() == proto::plan::Mul ||
                   arithmetic->op() == proto::plan::Div,
               "unsupported bound projection arithmetic");
    const bool a =
        ValidateExpr(expression->inputs()[0], available, used, depth + 1);
    const bool b =
        ValidateExpr(expression->inputs()[1], available, used, depth + 1);
    return a || b;
}
}  // namespace

BoundProjection
BindProjection(const std::vector<ProjectionField>& inputs,
               const std::vector<NamedProjection>& items) {
    Require(!items.empty(), 0, "empty projection");
    BoundProjection result;
    Symbols symbols;
    std::set<FieldId> fields;
    uint64_t next = 1;
    for (const auto& input : inputs) {
        Require(!input.name.empty() && ScalarType(input.type) &&
                    fields.insert(input.field).second,
                0,
                "invalid or duplicate storage input");
        Descriptor d{{next++},
                     input.name,
                     input.type,
                     input.nullable,
                     input.field,
                     false};
        Require(
            symbols.emplace(input.name, d).second, 0, "duplicate input name");
        result.inputs.push_back(std::move(d));
    }
    std::set<std::string> aliases;
    for (size_t i = 0; i < items.size(); ++i) {
        const auto& item = items[i];
        Require(!item.name.empty() && aliases.insert(item.name).second,
                i,
                "empty or duplicate alias");
        auto bound = Bind(item.expression, symbols, i, 0);
        const auto existing = symbols.find(item.name);
        auto ref =
            dynamic_cast<const expr::ResultColumnExpr*>(bound.expression.get());
        const bool passthrough = existing != symbols.end() &&
                                 existing->second.source_field && ref &&
                                 ref->id() == existing->second.id;
        Require(existing == symbols.end() || passthrough,
                i,
                "alias conflicts with an input column");
        Require(
            item.name.front() != '_' || passthrough, i, "reserved alias name");
        Descriptor d{{next++},
                     item.name,
                     bound.expression->type(),
                     bound.nullable,
                     passthrough ? existing->second.source_field : std::nullopt,
                     item.visible};
        result.items.push_back({d, std::move(bound.expression)});
        symbols.insert_or_assign(item.name, std::move(d));
    }
    Require(std::any_of(result.items.begin(),
                        result.items.end(),
                        [](const auto& item) { return item.output.visible; }),
            0,
            "no visible output");
    return result;
}

::milvus::plan::PlanNodePtr
MakeProjectionPlan(const BoundProjection& projection,
                   std::shared_ptr<::milvus::plan::MvccNode> source,
                   const std::string& prefix) {
    namespace logical = ::milvus::plan;
    AssertInfo(source && !projection.items.empty(),
               "projection requires a source and outputs");
    Available available;
    std::set<ResultColumnId> used;
    std::set<std::string> input_names, output_names;
    std::set<FieldId> fields;
    for (const auto& input : projection.inputs) {
        AssertInfo(input.id.value && !input.name.empty() &&
                       ScalarType(input.type) && input.source_field &&
                       input_names.insert(input.name).second &&
                       fields.insert(*input.source_field).second &&
                       available.emplace(input.id, input).second,
                   "invalid bound projection inputs");
    }
    for (const auto& item : projection.items) {
        const auto& d = item.output;
        AssertInfo(d.id.value && !d.name.empty() &&
                       output_names.insert(d.name).second &&
                       !available.count(d.id) && item.expression &&
                       d.type == item.expression->type(),
                   "invalid bound projection output descriptor");
        const bool nullable = ValidateExpr(item.expression, available, used);
        AssertInfo(d.nullable == nullable,
                   "bound projection nullability mismatch");
        if (d.source_field) {
            auto ref = dynamic_cast<const expr::ResultColumnExpr*>(
                item.expression.get());
            AssertInfo(
                ref && available.at(ref->id()).source_field == d.source_field,
                "bound projection storage provenance mismatch");
        }
        // Repeat collision checks for directly constructed internal plans.
        for (const auto& input : projection.inputs)
            if (d.name == input.name) {
                auto ref = dynamic_cast<const expr::ResultColumnExpr*>(
                    item.expression.get());
                AssertInfo(ref && ref->id() == input.id,
                           "bound projection alias conflicts with input");
            }
        AssertInfo(d.name.front() != '_' || input_names.count(d.name),
                   "reserved bound projection alias");
        available.emplace(d.id, d);
    }
    std::vector<Descriptor> current;
    std::vector<expr::TypedExprPtr> roots;
    std::vector<std::string> names;
    for (const auto& input : projection.inputs)
        if (used.count(input.id)) {
            current.push_back(input);
            roots.push_back(std::make_shared<expr::ColumnExpr>(expr::ColumnInfo(
                *input.source_field, input.type, {}, input.nullable)));
            names.push_back(InternalName(input.id));
        }
    if (roots.empty()) {
        proto::plan::GenericValue zero;
        zero.set_int64_val(0);
        roots.push_back(std::make_shared<expr::ValueExpr>(zero));
        names.push_back("__projection_seed");
    }
    logical::PlanNodePtr node =
        std::make_shared<logical::ScalarProjectScanNode>(
            prefix + "_scan", roots, names, source);
    size_t stage = 0, next_item = 0;
    while (next_item < projection.items.size()) {
        roots.clear();
        names.clear();
        std::map<ResultColumnId, column_index_t> bindings;
        for (size_t i = 0; i < current.size(); ++i) {
            bindings.emplace(current[i].id, i);
            roots.push_back(Ref(current[i]));
            names.push_back(InternalName(current[i].id));
        }
        const auto first_item = next_item;
        // Adjacent independent outputs share one ExprSet/input batch. A
        // dependency on a newly produced alias starts the next Project stage.
        for (; next_item < projection.items.size(); ++next_item) {
            const auto& item = projection.items[next_item];
            std::set<ResultColumnId> dependencies;
            ValidateExpr(item.expression, available, dependencies);
            if (std::any_of(dependencies.begin(),
                            dependencies.end(),
                            [&](auto id) { return !bindings.count(id); }))
                break;
            roots.push_back(item.expression);
            names.push_back(InternalName(item.output.id));
        }
        AssertInfo(next_item > first_item,
                   "projection stage has an unavailable dependency");
        node = std::make_shared<logical::ScalarProjectNode>(
            prefix + "_stage_" + std::to_string(stage++),
            roots,
            names,
            std::map<int64_t, column_index_t>{},
            node,
            bindings);
        for (size_t i = first_item; i < next_item; ++i)
            current.push_back(projection.items[i].output);
    }
    roots.clear();
    names.clear();
    std::map<ResultColumnId, column_index_t> bindings;
    for (size_t i = 0; i < current.size(); ++i)
        bindings.emplace(current[i].id, i);
    for (const auto& item : projection.items)
        if (item.output.visible) {
            roots.push_back(Ref(item.output));
            names.push_back(item.output.name);
        }
    AssertInfo(!roots.empty(), "bound projection has no visible output");
    return std::make_shared<logical::ScalarProjectNode>(
        prefix + "_output",
        roots,
        names,
        std::map<int64_t, column_index_t>{},
        node,
        bindings);
}
}  // namespace milvus::exec::plan
