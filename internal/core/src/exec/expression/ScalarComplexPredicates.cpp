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

#include "exec/expression/ScalarComplexPredicates.h"
#include "exec/expression/ScalarStringKernels.h"
#include "exec/expression/ScalarCallExpr.h"
#include "common/Geometry.h"
#include "exec/expression/InputRefExpr.h"

namespace milvus::exec::scalar {
namespace {
struct GeometryPredicate {
    proto::plan::GISFunctionFilterExpr_GISOp operation;
    std::string wkt;
    double distance;
};

bool
MatchGeometry(const Geometry& left,
              const Geometry* right,
              const GeometryPredicate& predicate,
              GEOSContextHandle_t geos) {
    using namespace proto::plan;
    switch (predicate.operation) {
        case GISFunctionFilterExpr_GISOp_Equals:
            return left.equals(*right, geos);
        case GISFunctionFilterExpr_GISOp_Touches:
            return left.touches(*right, geos);
        case GISFunctionFilterExpr_GISOp_Overlaps:
            return left.overlaps(*right, geos);
        case GISFunctionFilterExpr_GISOp_Crosses:
            return left.crosses(*right, geos);
        case GISFunctionFilterExpr_GISOp_Contains:
            return left.contains(*right, geos);
        case GISFunctionFilterExpr_GISOp_Intersects:
            return left.intersects(*right, geos);
        case GISFunctionFilterExpr_GISOp_Within:
            return left.within(*right, geos);
        case GISFunctionFilterExpr_GISOp_DWithin:
            return left.dwithin(*right, predicate.distance, geos);
        case GISFunctionFilterExpr_GISOp_STIsValid:
            return left.is_valid(geos);
        default:
            ThrowInfo(UnexpectedError, "invalid compiled GIS operation");
    }
}

class GeometryPredicateExpr final : public Expr {
 public:
    GeometryPredicateExpr(ExprPtr input,
                          LogicalOp op,
                          std::vector<GeometryPredicate> predicates)
        : Expr(DataType::BOOL),
          input(std::move(input)),
          op(op),
          predicates(std::move(predicates)) {
    }
    const ExprPtr input;
    const LogicalOp op;
    const std::vector<GeometryPredicate> predicates;

 protected:
    ReadOnlyVectorView
    EvalImpl(SelectionView rows, EvalCtx& ctx) const override {
        const auto* field = dynamic_cast<const InputRefExpr*>(input.get());
        const auto* lookup =
            field ? &ctx.input().GetGeometryLookup(field->slot()) : nullptr;
        std::optional<ReadOnlyVectorView> values;
        std::optional<ValidityResult> validity;
        std::vector<const Geometry*> cached;
        if (lookup && *lookup) {
            validity = ctx.input().GetValidity(field->slot(), rows);
            cached.resize(rows.row_count(), nullptr);
            auto missing = OwnedSelection::None(rows.row_count());
            rows.ForEachSelected([&](size_t row) {
                ctx.Checkpoint();
                if (validity->valid.view().IsSelected(row)) {
                    cached[row] = (*lookup)(row);
                    // An unpublished cache entry is a miss, not a FALSE proof.
                    if (!cached[row])
                        missing.SetSelected(row);
                }
            });
            if (!missing.view().IsNone())
                values = input->Eval(missing.view(), ctx);
        } else {
            values = input->Eval(rows, ctx);
        }
        auto valid = [&](size_t row) {
            return validity ? validity->valid.view().IsSelected(row)
                            : values->IsValid(row);
        };
        auto cached_at = [&](size_t row) -> const Geometry* {
            return cached.empty() ? nullptr : cached[row];
        };
        auto geos = GetThreadLocalGEOSContext();
        if (predicates.size() == 1) {
            const auto& predicate = predicates.front();
            std::unique_ptr<Geometry> right;
            if (predicate.operation !=
                proto::plan::GISFunctionFilterExpr_GISOp_STIsValid)
                right = std::make_unique<Geometry>(geos, predicate.wkt.c_str());
            BitmapResultBuilder output(rows.row_count());
            if (!(lookup && *lookup)) {
                StringInput(*values).Match(
                    rows, output, [&](std::string_view bytes) {
                        ctx.Checkpoint();
                        Geometry parsed;
                        parsed.TryParseFromWkb(
                            geos, bytes.data(), bytes.size());
                        return parsed.IsValid() &&
                               MatchGeometry(
                                   parsed, right.get(), predicate, geos);
                    });
                return std::move(output).Finish(rows);
            }
            rows.ForEachSelected([&](size_t row) {
                ctx.Checkpoint();
                if (!valid(row))
                    return;
                Geometry parsed;
                const Geometry* left = cached_at(row);
                if (!left) {
                    const auto bytes = values->StringViewAt(row);
                    parsed.TryParseFromWkb(geos, bytes.data(), bytes.size());
                    left = &parsed;
                }
                output.Set<bool>(
                    row,
                    left->IsValid() &&
                        MatchGeometry(*left, right.get(), predicate, geos));
            });
            return std::move(output).Finish(rows);
        }
        // GEOS objects stay on this evaluation thread and never escape Eval.
        // Single predicates keep their streaming one-row allocation behavior.
        std::vector<Geometry> parsed(rows.row_count());
        auto defined = OwnedSelection::None(rows.row_count());
        auto active = OwnedSelection::CopyOf(rows);
        const bool is_and = op == LogicalOp::And;
        BitmapResultBuilder output(rows.row_count());
        for (const auto& predicate : predicates) {
            if (active.view().IsNone())
                break;
            ctx.Checkpoint();
            std::unique_ptr<Geometry> right;
            if (predicate.operation !=
                proto::plan::GISFunctionFilterExpr_GISOp_STIsValid)
                right = std::make_unique<Geometry>(geos, predicate.wkt.c_str());
            auto next = OwnedSelection::CopyOf(active.view());
            active.view().ForEachSelected([&](size_t row) {
                ctx.Checkpoint();
                if (!valid(row))
                    return;  // UNKNOWN does not short-circuit either AND or OR.
                auto& left = parsed[row];
                if (!cached_at(row) && !defined.view().IsSelected(row)) {
                    const auto bytes = values->StringViewAt(row);
                    left.TryParseFromWkb(geos, bytes.data(), bytes.size());
                    defined.SetSelected(row);
                }
                const Geometry* geometry =
                    cached_at(row) ? cached_at(row) : &left;
                const bool value =
                    geometry->IsValid() &&
                    MatchGeometry(*geometry, right.get(), predicate, geos);
                if (value != is_and) {
                    output.Set<bool>(row, value);
                    next.SetSelected(row, false);
                }
            });
            active = std::move(next);
        }
        active.view().ForEachSelected([&](size_t row) {
            if (valid(row))
                output.Set<bool>(row, is_and);
        });
        return std::move(output).Finish(rows);
    }
};
}  // namespace

ExprPtr
MakeGeometryPredicate(ExprPtr input,
                      proto::plan::GISFunctionFilterExpr_GISOp operation,
                      std::string wkt,
                      double distance) {
    return std::make_shared<GeometryPredicateExpr>(
        std::move(input),
        LogicalOp::And,
        std::vector<GeometryPredicate>{{operation, std::move(wkt), distance}});
}

std::vector<ExprPtr>
FuseAdjacentGeometryPredicates(LogicalOp op,
                               std::vector<ExprPtr> inputs,
                               const std::function<ExprPtr(ExprPtr)>& resolve) {
    std::vector<ExprPtr> resolved;
    for (const auto& input : inputs)
        resolved.push_back(resolve ? resolve(input) : input);
    auto eligible = [op](const ExprPtr& node) -> const GeometryPredicateExpr* {
        auto* geometry = dynamic_cast<const GeometryPredicateExpr*>(node.get());
        return geometry &&
                       (geometry->predicates.size() == 1 ||
                        geometry->op == op) &&
                       dynamic_cast<const InputRefExpr*>(geometry->input.get())
                   ? geometry
                   : nullptr;
    };
    std::vector<ExprPtr> result;
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
            if (other->slot() != field->slot() ||
                other->type() != field->type())
                break;
            predicates.insert(predicates.end(),
                              next->predicates.begin(),
                              next->predicates.end());
        }
        result.push_back(end == begin + 1
                             ? inputs[begin]
                             : std::make_shared<GeometryPredicateExpr>(
                                   first->input, op, std::move(predicates)));
        begin = end;
    }
    return result;
}

namespace {
ReadOnlyVectorView
Elements(const ArrayView& array, DataType type, EvalCtx& ctx) {
    const auto stored = array.get_element_type();
    AssertInfo(
        array.length() == 0 || stored == type ||
            (stored == DataType::INT32 &&
             (type == DataType::INT8 || type == DataType::INT16)) ||
            ((stored == DataType::STRING || stored == DataType::VARCHAR) &&
             (type == DataType::STRING || type == DataType::VARCHAR)),
        "MATCH array payload type differs from schema");
    VectorResultBuilder output(type, array.length());
    auto copy = [&]<typename T>() {
        for (int i = 0; i < array.length(); ++i) {
            ctx.Checkpoint();
            if (!array.is_element_valid(i))
                output.SetNull(i);
            else
                output.Set<T>(i, array.get_data_unchecked<T>(i));
        }
    };
    switch (type) {
        case DataType::BOOL:
            copy.template operator()<bool>();
            break;
        case DataType::INT8:
            copy.template operator()<int8_t>();
            break;
        case DataType::INT16:
            copy.template operator()<int16_t>();
            break;
        case DataType::INT32:
            copy.template operator()<int32_t>();
            break;
        case DataType::INT64:
            copy.template operator()<int64_t>();
            break;
        case DataType::FLOAT:
            copy.template operator()<float>();
            break;
        case DataType::DOUBLE:
            copy.template operator()<double>();
            break;
        case DataType::VARCHAR:
        case DataType::STRING:
            copy.template operator()<std::string>();
            break;
        default:
            ThrowInfo(UnexpectedError, "invalid compiled MATCH element type");
    }
    return std::move(output).Finish(SelectionView::All(array.length()));
}
}  // namespace

ExprPtr
MakeArrayMatch(ExprPtr anchor,
               std::vector<ExprPtr> arrays,
               RowTypePtr element_schema,
               ExprPtr child,
               proto::plan::MatchType match,
               int64_t threshold) {
    AssertInfo(anchor && element_schema && child &&
                   child->type() == DataType::BOOL &&
                   arrays.size() == element_schema->column_count(),
               "MATCH requires aligned array inputs");
    arrays.insert(arrays.begin(), std::move(anchor));
    return MakeCall(
        DataType::BOOL,
        std::move(arrays),
        [element_schema = std::move(element_schema),
         child = std::move(child),
         match,
         threshold](const auto& inputs, SelectionView rows, EvalCtx& ctx) {
            VectorResultBuilder output(DataType::BOOL, rows.row_count());
            rows.ForEachSelected([&](size_t row) {
                ctx.Checkpoint();
                const bool valid = inputs[0].IsValid(row);
                for (const auto& input : inputs)
                    AssertInfo(input.IsValid(row) == valid,
                               "MATCH struct fields disagree on row validity");
                if (!valid) {
                    output.SetNull(row);
                    return;
                }
                const auto count =
                    inputs[0].logical_type() == DataType::VECTOR_ARRAY
                        ? inputs[0].VectorArrayAt(row).length()
                        : inputs[0].ArrayLengthAt(row);
                BatchInput elements(element_schema, count);
                for (size_t slot = 0; slot < element_schema->column_count();
                     ++slot) {
                    const auto& input = inputs[slot + 1];
                    AssertInfo(input.ArrayLengthAt(row) == count,
                               "MATCH struct fields disagree on element count");
                    if (element_schema->column_type(slot) == DataType::ARRAY) {
                        const auto view = input.ArrayValueAt(row).View();
                        if (view.type()
                                .array_element()
                                .array_element()
                                .has_array_element()) {
                            auto owned =
                                std::make_shared<std::vector<ArrayValue>>(
                                    count);
                            auto valid = OwnedSelection::None(count);
                            for (size_t i = 0; i < count; ++i) {
                                ctx.Checkpoint();
                                const auto child = view.array_at(i);
                                if (!child.is_null()) {
                                    (*owned)[i] = ArrayValue(
                                        child.output_data(), child.type());
                                    valid.SetSelected(i);
                                }
                            }
                            elements.BindValues(
                                slot,
                                ReadOnlyVectorView::BorrowImmutable<ArrayValue>(
                                    DataType::ARRAY,
                                    owned->data(),
                                    owned,
                                    SelectionView::All(count),
                                    valid.view()));
                            continue;
                        }
                        VectorResultBuilder values(DataType::ARRAY, count);
                        for (size_t i = 0; i < count; ++i) {
                            ctx.Checkpoint();
                            const auto child = view.array_at(i);
                            if (child.is_null())
                                values.SetNull(i);
                            else
                                // ColumnarArrayChunk has validity for array
                                // nodes; its scalar leaf chunks are non-nullable.
                                values.Set<Array>(i, Array(child.output_data()));
                        }
                        elements.BindValues(slot,
                                            std::move(values).Finish(
                                                SelectionView::All(count)));
                        continue;
                    }
                    const auto& array = input.ArrayViewAt(row);
                    // Per-row bounded element batches avoid any global element
                    // cursor and bound temporary storage by one document.
                    elements.BindValues(
                        slot,
                        Elements(
                            array, element_schema->column_type(slot), ctx));
                }
                int64_t hits = 0, known = 0;
                if (count != 0) {
                    EvalCtx element_ctx(elements, [&ctx] { ctx.Checkpoint(); });
                    auto result =
                        child->Eval(SelectionView::All(count), element_ctx);
                    auto masks =
                        result.CopyBoolMasks(SelectionView::All(count));
                    known = masks.valid.view().selected_count();
                    hits = masks.truth.view().selected_count();
                }
                bool value;
                switch (match) {
                    case proto::plan::MatchAny:
                        value = hits != 0;
                        break;
                    case proto::plan::MatchAll:
                        value = hits == known;
                        break;
                    case proto::plan::MatchLeast:
                        value = hits >= threshold;
                        break;
                    case proto::plan::MatchMost:
                        value = hits <= threshold;
                        break;
                    case proto::plan::MatchExact:
                        value = hits == threshold;
                        break;
                    default:
                        ThrowInfo(UnexpectedError,
                                  "invalid compiled MATCH operation");
                }
                output.Set<bool>(row, value);
            });
            return std::move(output).Finish(rows);
        });
}
}  // namespace milvus::exec::scalar
