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

#include "exec/plan/ExecutionPreparer.h"
#include "segcore/reader/ScalarQueryView.h"
#include "exec/expression/ScalarComplexPredicates.h"
#include "exec/expression/ScalarPredicates.h"
#include "exec/expression/PredicateRefExpr.h"
#include "common/RegexQuery.h"
#include "common/Geometry.h"

namespace milvus::exec::plan {

std::optional<PreparedIndex>
PrepareInt64Index(std::shared_ptr<const index::IIndexReaderBase> index,
                  int64_t row_count,
                  segcore::reader::Int64IndexQuery query,
                  std::shared_ptr<const segcore::reader::ReadContext> context) {
    return PrepareIntegerIndex<int64_t>(
        std::move(index), row_count, std::move(query), std::move(context));
}

PreparedPredicates
AdjustExecution(const CompiledExprPlan& plan,
                const PhysicalPrepareOptions& physical) {
    PreparedPredicates selected;
    selected.raw_inputs = plan.direct_inputs();
    std::vector<std::shared_ptr<predicate::IndexNode>> coarse_nodes;
    std::vector<bool> fusion_safe;
    std::vector<scalar::ExprPtr> raw_programs;
    for (const auto& p : plan.predicates()) {
        bool safe = false;
        if (p.geometry_query) {
            // Constant eligibility check only. Invalid WKT stays an evaluation
            // barrier and is reported only if that expression is reached.
            auto geos = GetThreadLocalGEOSContext();
            auto reader = GEOSWKTReader_create_r(geos);
            if (!reader)
                ThrowInfo(MemAllocateFailed,
                          "cannot allocate geometry eligibility reader");
            auto value = GEOSWKTReader_read_r(
                geos, reader, p.geometry_query->wkt.c_str());
            GEOSWKTReader_destroy_r(geos, reader);
            safe = value != nullptr;
            if (value)
                GEOSGeom_destroy_r(geos, value);
        } else if (p.ngram_query) {
            safe = true;
            if (p.ngram_query->op == proto::plan::Match) {
                size_t escapes = 0;
                for (auto i = p.ngram_query->pattern.rbegin();
                     i != p.ngram_query->pattern.rend() && *i == '\\';
                     ++i)
                    ++escapes;
                safe = escapes % 2 == 0;
            }
            if (p.ngram_query->op == proto::plan::RegexMatch) {
                RE2::Options options;
                options.set_dot_nl(true);
                options.set_log_errors(false);
                RE2 regex(p.ngram_query->pattern, options);
                safe = regex.ok();
            }
        }
        fusion_safe.push_back(safe);
    }
    for (const auto& predicate : plan.predicates()) {
        AssertInfo(
            !predicate.exact || predicate.exact->type() == DataType::BOOL,
            "invalid exact predicate program");
        auto source = physical.use_index && physical.bind_index
                          ? physical.bind_index(predicate)
                          : std::nullopt;
        scalar::ExprPtr raw;
        auto prepare_raw = [&]() {
            if (!raw)
                raw = predicate.PrepareRaw();
            return raw;
        };
        auto route = PredicateRoute::Raw;
        coarse_nodes.push_back(nullptr);
        if (source)
            selected.force_batch |= source->force_batch;
        if (source && source->selected_source) {
            selected.sources.push_back(source->selected_source);
            route = source->selected_needs_raw ? PredicateRoute::Refine
                                               : source->selected_route;
        } else if (source) {
            AssertInfo(source->reader && source->context,
                       "index capability omitted its reader/context");
            using Kind = segcore::reader::IndexResultKind;
            if (source->kind == Kind::FullThreeValued ||
                (source->kind == Kind::Candidates && predicate.HasRaw())) {
                auto node = std::make_shared<predicate::IndexNode>(
                    source->reader, source->context);
                if (source->kind == Kind::FullThreeValued) {
                    selected.sources.push_back(
                        predicate::BindIndex(std::move(node)));
                    route = PredicateRoute::ExactIndex;
                } else {
                    coarse_nodes.back() = node;
                    selected.sources.push_back(predicate::BindRefine(
                        std::make_shared<predicate::RefineNode>(
                            std::move(node), prepare_raw())));
                    route = PredicateRoute::Refine;
                }
            } else {
                AssertInfo(source->kind == Kind::ExactTrue ||
                               source->kind == Kind::Candidates,
                           "unknown index result capability");
                // All compiled predicate slots are full BOOL consumers. A TRUE
                // set alone is insufficient even when this root is a filter.
            }
        }
        if (route == PredicateRoute::Raw && !predicate.HasRaw()) {
            selected.supported = false;
            return selected;
        }
        if (route == PredicateRoute::Raw) {
            selected.sources.push_back(
                [exact = prepare_raw()](const ReadBatchBinding&,
                                        SelectionView rows,
                                        scalar::EvalCtx& ctx) {
                    return exact->Eval(rows, ctx);
                });
        }
        if (route != PredicateRoute::ExactIndex &&
            (!source || !source->selected_source || source->selected_needs_raw))
            selected.raw_inputs.insert(selected.raw_inputs.end(),
                                       predicate.raw_inputs.begin(),
                                       predicate.raw_inputs.end());
        raw_programs.push_back(std::move(raw));
        selected.routes.push_back(route);
        selected.backends.push_back(route == PredicateRoute::Raw
                                        ? PredicateBackend::Raw
                                        : source->backend);
    }
    auto classify = [&](const scalar::Expr* expr) {
        auto it = plan.properties().find(expr);
        if (it == plan.properties().end())
            return scalar::ReorderProperty{};
        const auto& metadata = it->second;
        auto property = metadata.reorder;
        bool exact = false;
        if (metadata.predicate_slot) {
            AssertInfo(*metadata.predicate_slot < selected.routes.size(),
                       "reorder metadata references an invalid predicate slot");
            exact = selected.routes[*metadata.predicate_slot] ==
                    PredicateRoute::ExactIndex;
        }
        if (!metadata.predicate_slot ||
            (!plan.predicates()[*metadata.predicate_slot].text_query &&
             !plan.predicates()[*metadata.predicate_slot].selected_stats &&
             !plan.predicates()[*metadata.predicate_slot].match))
            property.safe |= exact;
        using Tier = scalar::ReorderTier;
        if (metadata.predicate_slot && fusion_safe[*metadata.predicate_slot]) {
            property.safe = true;
            property.tier = Tier::HeavyConjunction;
        }
        if (physical.namespace_field &&
            metadata.namespace_candidate == physical.namespace_field)
            property.tier = Tier::Namespace;
        else if (exact && property.tier != Tier::Numeric)
            property.tier = Tier::Indexed;
        return property;
    };
    auto resolve = [&](scalar::ExprPtr node) {
        auto property = plan.properties().find(node.get());
        if (property != plan.properties().end() &&
            property->second.predicate_slot) {
            const auto slot = *property->second.predicate_slot;
            // Expose selected raw GIS kernels to same-field fusion. Indexed
            // coarse/refine nodes retain their original execution boundary.
            if (selected.routes[slot] == PredicateRoute::Raw &&
                (plan.predicates()[slot].geometry_query ||
                 plan.predicates()[slot].ngram_query))
                return raw_programs[slot];
        }
        return node;
    };
    auto slot_of = [&](const scalar::ExprPtr& expr) -> std::optional<size_t> {
        auto* ref = dynamic_cast<const scalar::PredicateRefExpr*>(expr.get());
        if (ref && ref->slot() < plan.predicates().size() &&
            coarse_nodes[ref->slot()] && fusion_safe[ref->slot()])
            return ref->slot();
        return std::nullopt;
    };
    auto add_source = [&](predicate::PredicateSource source,
                          PredicateBackend backend) -> scalar::ExprPtr {
        auto slot = selected.sources.size();
        AssertInfo(slot < std::numeric_limits<column_index_t>::max(),
                   "too many prepared predicate slots");
        selected.sources.push_back(std::move(source));
        selected.routes.push_back(PredicateRoute::Refine);
        selected.backends.push_back(backend);
        return std::make_shared<scalar::PredicateRefExpr>(slot);
    };
    auto optimize = [&](scalar::LogicalOp op,
                        std::vector<scalar::ExprPtr> inputs) {
        if (op != scalar::LogicalOp::And)
            return inputs;
        std::vector<scalar::ExprPtr> output;
        for (size_t begin = 0; begin < inputs.size();) {
            if (!classify(inputs[begin].get()).safe) {
                output.push_back(inputs[begin++]);
                continue;
            }
            size_t end = begin;
            while (end < inputs.size() && classify(inputs[end].get()).safe)
                ++end;
            // TRUE means "not proven false", never "the predicate matches".
            // These guards may precede other safe conditions; exact refinement
            // remains present and produces the final three-valued result.
            for (size_t i = begin; i < end; ++i)
                if (auto slot = slot_of(inputs[i])) {
                    output.push_back(add_source(
                        [coarse = coarse_nodes[*slot]](
                            const ReadBatchBinding& binding,
                            SelectionView rows,
                            scalar::EvalCtx& ctx) {
                            auto result = coarse->Read(binding, rows);
                            BitmapResultBuilder bits(binding.size());
                            rows.ForEachSelected([&](size_t row) {
                                ctx.Checkpoint();
                                bits.Set<bool>(row,
                                               !result->false_rows().IsSelected(
                                                   binding.offset(row)));
                            });
                            return std::move(bits).Finish(rows);
                        },
                        selected.backends[*slot]));
                }
            for (size_t i = begin; i < end;) {
                auto first = slot_of(inputs[i]);
                if (!first) {
                    output.push_back(inputs[i++]);
                    continue;
                }
                const auto& p = plan.predicates()[*first];
                size_t j = i + 1;
                std::vector<std::shared_ptr<predicate::IndexNode>> nodes{
                    coarse_nodes[*first]};
                std::vector<scalar::ExprPtr> exacts{raw_programs[*first]};
                for (; j < end; ++j) {
                    auto next = slot_of(inputs[j]);
                    if (!next)
                        break;
                    const auto& q = plan.predicates()[*next];
                    if (p.field.field_id != q.field.field_id ||
                        p.geometry_query.has_value() !=
                            q.geometry_query.has_value() ||
                        p.ngram_query.has_value() != q.ngram_query.has_value())
                        break;
                    nodes.push_back(coarse_nodes[*next]);
                    exacts.push_back(raw_programs[*next]);
                }
                if (j == i + 1) {
                    output.push_back(inputs[i++]);
                    continue;
                }
                exacts =
                    scalar::FuseAdjacentStringMatches(op, std::move(exacts));
                exacts = scalar::FuseAdjacentGeometryPredicates(
                    op, std::move(exacts));
                auto exact = exacts.size() == 1
                                 ? exacts.front()
                                 : scalar::MakeLogical(op, std::move(exacts));
                output.push_back(add_source(
                    [nodes = std::move(nodes), exact](
                        const ReadBatchBinding& binding,
                        SelectionView rows,
                        scalar::EvalCtx& ctx) {
                        std::vector<
                            std::shared_ptr<const segcore::reader::IndexResult>>
                            proofs;
                        for (const auto& node : nodes)
                            proofs.push_back(node->Read(binding, rows));
                        BitmapResultBuilder output(binding.size());
                        auto missing = OwnedSelection::None(binding.size());
                        rows.ForEachSelected([&](size_t row) {
                            ctx.Checkpoint();
                            bool negative = false, null = true;
                            for (const auto& proof : proofs) {
                                negative |= proof->false_rows().IsSelected(
                                    binding.offset(row));
                                null &= proof->null_rows().IsSelected(
                                    binding.offset(row));
                            }
                            if (negative)
                                output.Set<bool>(row, false);
                            else if (!null)
                                missing.SetSelected(row);
                        });
                        if (!missing.view().IsNone()) {
                            auto values = exact->Eval(missing.view(), ctx);
                            missing.view().ForEachSelected([&](size_t row) {
                                if (values.IsValid(row)) {
                                    bool value = values.ValueAt<bool>(row);
                                    for (const auto& proof : proofs)
                                        AssertInfo(
                                            !value ||
                                                proof->selected().IsSelected(
                                                    binding.offset(row)),
                                            "fused coarse index omitted a true "
                                            "row");
                                    output.Set<bool>(row, value);
                                }
                            });
                        }
                        ctx.Checkpoint();
                        return std::move(output).Finish(rows);
                    },
                    selected.backends[*first]));
                i = j;
            }
            begin = end;
        }
        return output;
    };
    std::vector<scalar::ExprPtr> roots;
    for (const auto& root : plan.expressions().roots())
        roots.push_back(physical.reorder
                            ? scalar::AdjustLogicalExpression(
                                  root, classify, resolve, optimize)
                            : root);
    selected.expressions =
        std::make_shared<const scalar::ExprSet>(std::move(roots));
    return selected;
}

PreparedExecution
Prepare(CompiledExprPlanPtr plan,
        const BindInput& bind,
        const PhysicalPrepareOptions& physical) {
    AssertInfo(plan, "cannot prepare an unsupported/null scalar plan");
    auto selected = AdjustExecution(*plan, physical);
    AssertInfo(selected.supported, "required predicate source is unavailable");
    auto requirements = RequiredInputs(*plan, selected.raw_inputs);
    std::vector<std::optional<PreparedInput>> inputs;
    for (const auto& dependency : requirements) {
        AssertInfo(!dependency || !dependency->result_column ||
                       dependency->upstream_column,
                   "result dependency must not bind storage");
        if (!dependency || dependency->upstream_column) {
            inputs.emplace_back(std::nullopt);
        } else {
            AssertInfo(bind, "stored scalar input requires a source binder");
            // Do not catch binding errors and turn them into capability fallback.
            inputs.emplace_back(bind(*dependency));
        }
    }
    return PreparedExecution(
        std::move(plan), std::move(inputs), std::move(selected));
}

}  // namespace milvus::exec::plan
