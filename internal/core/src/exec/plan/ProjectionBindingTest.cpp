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

#include <gtest/gtest.h>
#include "exec/plan/ProjectionBinding.h"
#include "exec/plan/ExprCompiler.h"
#include "exec/plan/ExecutionPreparer.h"
#include "exec/plan/ScalarProject.h"
#include "exec/QueryContext.h"
#include "query/ExecPlanNodeVisitor.h"
#include "test_utils/storage_test_utils.h"

namespace milvus::exec::plan {
namespace {
namespace logical = ::milvus::plan;
expr::TypedExprPtr
Name(std::string name) {
    return std::make_shared<expr::ProjectionNameExpr>(std::move(name));
}
expr::TypedExprPtr
Number(int64_t n) {
    proto::plan::GenericValue v;
    v.set_int64_val(n);
    return std::make_shared<expr::ValueExpr>(v);
}
expr::TypedExprPtr
Op(expr::TypedExprPtr a, proto::plan::ArithOpType op, expr::TypedExprPtr b) {
    return std::make_shared<expr::ProjectionArithmeticExpr>(op, a, b);
}
std::vector<NamedProjection>
Chain() {
    return {{"adjusted", Op(Name("a"), proto::plan::Add, Number(1))},
            {"score", Op(Name("adjusted"), proto::plan::Mul, Number(2))},
            {"extra", Op(Name("adjusted"), proto::plan::Add, Number(3))}};
}
BoundProjection
Bound() {
    return BindProjection({{"a", FieldId(100), DataType::INT64, true}},
                          Chain());
}
std::vector<logical::PlanNodePtr>
Nodes(const BoundProjection& bound) {
    std::vector<logical::PlanNodePtr> nodes;
    auto node =
        MakeProjectionPlan(bound, std::make_shared<logical::MvccNode>("mvcc"));
    while (node && !std::dynamic_pointer_cast<const logical::MvccNode>(node)) {
        nodes.push_back(node);
        node = node->sources().empty() ? nullptr : node->sources()[0];
    }
    std::reverse(nodes.begin(), nodes.end());
    return nodes;
}
std::vector<std::string>
Names(RowTypePtr type) {
    std::vector<std::string> names;
    for (size_t i = 0; i < type->column_count(); ++i)
        names.push_back(type->column_name(i));
    return names;
}
struct Counts {
    size_t calls = 0, rows = 0;
};
class CountedExpr final : public scalar::Expr {
 public:
    CountedExpr(scalar::ExprPtr expression, std::shared_ptr<Counts> counts)
        : Expr(expression->type()),
          expression_(std::move(expression)),
          counts_(std::move(counts)) {
    }

 protected:
    ReadOnlyVectorView
    EvalImpl(SelectionView rows, scalar::EvalCtx& ctx) const override {
        ++counts_->calls;
        counts_->rows += rows.selected_count();
        return expression_->Eval(rows, ctx);
    }

 private:
    scalar::ExprPtr expression_;
    std::shared_ptr<Counts> counts_;
};
}  // namespace

TEST(ProjectionBindingTest, OrderedAliasesResolveToDistinctResultIdentities) {
    auto bound = Bound();
    ASSERT_EQ(bound.items.size(), 3);
    EXPECT_EQ(bound.items[0].output.type, DataType::INT64);
    EXPECT_TRUE(bound.items[0].output.nullable);
    for (size_t i : {1, 2}) {
        auto ref = dynamic_cast<const expr::ResultColumnExpr*>(
            bound.items[i].expression->inputs()[0].get());
        ASSERT_NE(ref, nullptr);
        EXPECT_EQ(ref->id(), bound.items[0].output.id);
        EXPECT_FALSE(bound.items[i].output.source_field.has_value());
    }
    auto passthrough = BindProjection({{"a", FieldId(100), DataType::INT64}},
                                      {{"a", Name("a")}});
    EXPECT_EQ(passthrough.items[0].output.name, "a");
    EXPECT_NE(passthrough.items[0].output.id, passthrough.inputs[0].id);
    auto nodes = Nodes(bound);
    EXPECT_EQ(nodes.back()->output_type()->column_name(1), "score");
}

TEST(ProjectionBindingTest,
     InvalidAliasesAndUnsupportedExpressionsFailDuringBinding) {
    const std::vector<ProjectionField> fields{
        {"a", FieldId(100), DataType::INT64}};
    for (const auto& items : std::vector<std::vector<NamedProjection>>{
             {{"x", Name("missing")}},
             {{"x", Name("x")}},
             {{"x", Name("y")}, {"y", Name("x")}},
             {{"x", Number(1)}, {"x", Number(2)}},
             {{"a", Op(Name("a"), proto::plan::Add, Number(1))}},
             {{"_score", Name("a")}},
             {{"", Name("a")}},
             {{"x", Op(Name("a"), proto::plan::Mod, Number(2))}},
             {{"hidden", Name("a"), false}}}) {
        try {
            BindProjection(fields, items);
            FAIL() << "expected binder rejection";
        } catch (const SegcoreError& error) {
            EXPECT_EQ(error.get_error_code(), ExprInvalid);
        }
    }
    EXPECT_THROW(BindProjection(
                     {{"text", FieldId(101), DataType::VARCHAR}},
                     {{"bad", Op(Name("text"), proto::plan::Add, Number(1))}}),
                 SegcoreError);
    EXPECT_FALSE(CompileExpressions({Name("a")}).supported());
}

TEST(ProjectionBindingTest, DirectPlansRejectForwardCyclesTypesAndMetadata) {
    auto mvcc = std::make_shared<logical::MvccNode>("mvcc");
    for (int kind = 0; kind < 5; ++kind) {
        auto bound = Bound();
        if (kind == 0)
            bound.items[0].expression =
                std::make_shared<expr::ResultColumnExpr>(
                    bound.items[0].output.id, DataType::INT64, true);
        if (kind == 1)
            bound.items[0].expression =
                std::make_shared<expr::ResultColumnExpr>(
                    bound.items[2].output.id, DataType::INT64, true);
        if (kind == 2)
            bound.items[1].output.id = bound.items[0].output.id;
        if (kind == 3)
            bound.items[1].output.type = DataType::DOUBLE;
        if (kind == 4)
            bound.items[1].output.nullable = false;
        EXPECT_THROW(MakeProjectionPlan(bound, mvcc), SegcoreError);
    }
}

TEST(ProjectionBindingTest,
     ResultReferencesNeverFallbackToSameNumberedStorageFields) {
    auto result = std::make_shared<expr::ResultColumnExpr>(ResultColumnId{100},
                                                           DataType::INT64);
    auto field = std::make_shared<expr::ColumnExpr>(
        expr::ColumnInfo(FieldId(100), DataType::INT64));
    CompileOptions options;
    options.upstream_schema = std::make_shared<RowType>(
        std::vector<std::string>{"x"}, std::vector<DataType>{DataType::INT64});
    options.upstream_columns = {{100, 0}};
    EXPECT_FALSE(CompileExpressions({result}, options).supported());
    options.upstream_columns.clear();
    options.upstream_results = {{ResultColumnId{100}, 0}};
    auto compiled = CompileExpressions({result, field}, options);
    ASSERT_TRUE(compiled.supported());
    ASSERT_EQ(compiled.plan->inputs().slots().size(), 2);
    EXPECT_TRUE(compiled.plan->inputs().slots()[0].result_column.has_value());
    EXPECT_FALSE(
        compiled.plan->inputs().slots()[1].upstream_column.has_value());
    int storage_bindings = 0;
    auto upstream_only = CompileExpressions({result}, options);
    auto prepared =
        Prepare(upstream_only.plan, [&](const auto&) -> PreparedInput {
            ++storage_bindings;
            throw std::runtime_error("must not bind storage");
        });
    EXPECT_EQ(storage_bindings, 0);
    EXPECT_THROW(prepared.MakeExecutionBatch(ReadBatchBinding::Range(0, 1)),
                 SegcoreError);
    auto wrong = std::make_shared<expr::ResultColumnExpr>(ResultColumnId{100},
                                                          DataType::DOUBLE);
    EXPECT_FALSE(CompileExpressions({wrong}, options).supported());
    InputLayout malformed;
    auto dependency = upstream_only.plan->inputs().slots()[0];
    dependency.upstream_column.reset();
    malformed.Intern(dependency);
    auto broken = std::make_shared<CompiledExprPlan>(
        std::move(malformed),
        upstream_only.plan->expressions().roots(),
        options.upstream_schema);
    EXPECT_THROW(Prepare(broken,
                         [&](const auto&) -> PreparedInput {
                             ++storage_bindings;
                             throw std::runtime_error(
                                 "result dependency reached storage");
                         }),
                 SegcoreError);
    EXPECT_EQ(storage_bindings, 0);
}

TEST(ProjectionBindingTest,
     ProducerRunsOnceAndViewsSurviveAcrossSelectedBatches) {
    auto nodes = Nodes(Bound());
    auto source =
        std::dynamic_pointer_cast<const logical::ScalarProjectScanNode>(
            nodes[0]);
    ASSERT_NE(source, nullptr);
    auto reads = std::make_shared<std::vector<std::vector<int64_t>>>();
    auto scan = CompileExpressions(source->expressions());
    ScalarProject read(
        Prepare(scan.plan,
                [&](const auto&) {
                    return PreparedInput{
                        DataType::INT64,
                        true,
                        false,
                        reads,
                        [reads](const ReadBatchBinding& binding) {
                            return InputLoaders{
                                [reads, binding](SelectionView rows) {
                                    VectorResultBuilder values(DataType::INT64,
                                                               binding.size());
                                    std::vector<int64_t> offsets;
                                    rows.ForEachSelected([&](size_t row) {
                                        offsets.push_back(binding.offset(row));
                                        if (binding.offset(row) != 2)
                                            values.Set<int64_t>(
                                                row, binding.offset(row) * 10);
                                    });
                                    reads->push_back(offsets);
                                    return std::move(values).Finish(rows);
                                },
                                {}};
                        }};
                }),
        Names(source->output_type()));
    auto counts = std::make_shared<Counts>();
    std::vector<std::unique_ptr<ScalarProject>> stages;
    for (size_t i = 1; i < nodes.size(); ++i) {
        auto node = std::dynamic_pointer_cast<const logical::ScalarProjectNode>(
            nodes[i]);
        ASSERT_NE(node, nullptr);
        CompileOptions options;
        options.upstream_schema = node->sources()[0]->output_type();
        options.upstream_results = node->upstream_results();
        auto compiled = CompileExpressions(node->expressions(), options);
        ASSERT_TRUE(compiled.supported());
        auto roots = compiled.plan->expressions().roots();
        if (i == 1)
            roots.back() = std::make_shared<CountedExpr>(roots.back(), counts);
        auto observed = std::make_shared<CompiledExprPlan>(
            compiled.plan->inputs(), roots, options.upstream_schema);
        stages.push_back(std::make_unique<ScalarProject>(
            Prepare(observed), Names(node->output_type())));
    }
    auto run = [&](ReadBatchBinding binding, SelectionView rows) {
        auto output = read.Eval(binding, rows);
        for (const auto& stage : stages)
            output = stage->Eval(ReadBatchBinding::Range(0, output->size()),
                                 SelectionView::All(output->size()),
                                 output.get());
        return output;
    };
    auto selected = OwnedSelection::FromIndices(4, {0, 1, 3});
    auto first = run(ReadBatchBinding::Offsets({7, 2, 99, 7}), selected.view());
    auto second = run(ReadBatchBinding::Offsets({9}), SelectionView::All(1));
    EXPECT_EQ(counts->calls, 2);
    EXPECT_EQ(counts->rows, 4);
    EXPECT_EQ(*reads, (std::vector<std::vector<int64_t>>{{7, 2, 7}, {9}}));
    stages.clear();
    for (size_t col = 0; col < 3; ++col) {
        auto values = ReadOnlyVectorView::CopyOf(first->child(col));
        const int64_t expected[] = {71, 142, 74};
        EXPECT_EQ(values.ValueAt<int64_t>(0), expected[col]);
        EXPECT_FALSE(values.IsValid(1));
        EXPECT_EQ(values.ValueAt<int64_t>(2), expected[col]);
    }
    EXPECT_EQ(ReadOnlyVectorView::CopyOf(second->child(1)).ValueAt<int64_t>(0),
              182);
}

TEST(ProjectionBindingTest,
     RealDriverChainsAliasesSortsAndPropagatesLateFailure) {
    auto schema = std::make_shared<Schema>();
    auto pk = schema->AddDebugField("pk", DataType::INT64);
    schema->set_primary_field_id(pk);
    auto a = schema->AddDebugField("a", DataType::INT64, true);
    auto data = segcore::DataGen(schema, 11);
    for (auto& field : *data.raw_->mutable_fields_data()) {
        FixedVector<int64_t> values(11);
        FixedVector<bool> valid(11, true);
        for (size_t i = 0; i < 11; ++i) {
            values[i] = i;
            valid[i] = field.field_id() != a.get() || i != 4;
        }
        auto replacement =
            segcore::CreateDataArrayFrom(values.data(),
                                         valid.data(),
                                         11,
                                         (*schema)[FieldId(field.field_id())]);
        field.Swap(replacement.get());
    }
    for (bool growing : {false, true}) {
        std::unique_ptr<segcore::SegmentSealed> sealed;
        segcore::SegmentGrowingPtr live;
        if (growing) {
            live = segcore::CreateGrowingSegment(schema, empty_index_meta);
            live->PreInsert(11);
            live->Insert(0,
                         11,
                         data.row_ids_.data(),
                         data.timestamps_.data(),
                         data.SharedRaw());
        } else
            sealed = CreateSealedWithFieldDataLoaded(schema, data);
        auto segment =
            growing
                ? static_cast<segcore::SegmentInternalInterface*>(live.get())
                : static_cast<segcore::SegmentInternalInterface*>(sealed.get());
        auto context = [&] {
            auto ctx = std::make_shared<QueryContext>(
                "alias-test",
                segment,
                11,
                MAX_TIMESTAMP,
                0,
                0,
                query::PlanOptions{},
                std::make_shared<QueryConfig>(
                    std::unordered_map<std::string, std::string>{
                        {QueryConfig::kExprEvalBatchSize, "3"}}));
            ctx->set_read_snapshot(segment->CaptureReadSnapshot());
            return ctx;
        };
        auto bound = BindProjection({{"a", a, DataType::INT64, true}}, Chain());
        auto node = MakeProjectionPlan(
            bound, std::make_shared<logical::MvccNode>("mvcc"));
        auto order = std::make_shared<logical::OrderByNode>(
            "sort",
            std::vector<expr::FieldAccessTypeExprPtr>{
                std::make_shared<expr::FieldAccessTypeExpr>(
                    DataType::INT64, "score", FieldId(0))},
            std::vector<logical::SortOrder>{{false, false}},
            3,
            std::vector<logical::PlanNodePtr>{node});
        logical::PlanFragment fragment(order);
        auto output =
            query::ExecPlanNodeVisitor::ExecuteTask(fragment, context());
        ASSERT_EQ(output->size(), 3);
        EXPECT_EQ(output->row_type()->column_name(1), "score");
        auto score = ReadOnlyVectorView::CopyOf(output->child(1));
        EXPECT_EQ(score.ValueAt<int64_t>(0), 22);
        EXPECT_EQ(score.ValueAt<int64_t>(2), 18);
        auto constant = MakeProjectionPlan(
            BindProjection(
                {},
                {{"five", Number(5)},
                 {"ten", Op(Name("five"), proto::plan::Mul, Number(2))}}),
            std::make_shared<logical::MvccNode>("constant_mvcc"));
        logical::PlanFragment constant_fragment(constant);
        auto constant_output = query::ExecPlanNodeVisitor::ExecuteTask(
            constant_fragment, context());
        ASSERT_EQ(constant_output->size(), 11);
        EXPECT_EQ(ReadOnlyVectorView::CopyOf(constant_output->child(1))
                      .ValueAt<int64_t>(10),
                  10);
        auto items = Chain();
        items.push_back({"quotient",
                         Op(Name("score"),
                            proto::plan::Div,
                            Op(Number(8), proto::plan::Sub, Name("a")))});
        auto failing = MakeProjectionPlan(
            BindProjection({{"a", a, DataType::INT64, true}}, items),
            std::make_shared<logical::MvccNode>("mvcc"));
        logical::PlanFragment failure(failing);
        RowVectorPtr partial;
        try {
            partial =
                query::ExecPlanNodeVisitor::ExecuteTask(failure, context());
            FAIL() << "expected late division failure";
        } catch (const SegcoreError& error) {
            EXPECT_EQ(error.get_error_code(), ExprInvalid);
        }
        EXPECT_EQ(partial, nullptr);
        folly::CancellationSource cancellation;
        OpContext op(cancellation.getToken());
        auto cancelled = context();
        cancelled->set_op_context(&op);
        cancellation.requestCancellation();
        EXPECT_THROW(
            query::ExecPlanNodeVisitor::ExecuteTask(fragment, cancelled),
            SegcoreError);
    }
}

TEST(ProjectionBindingTest,
     HiddenIntermediateAndConstantOnlyPlanKeepOutputSchema) {
    auto items = Chain();
    items[0].visible = false;
    auto nodes = Nodes(
        BindProjection({{"a", FieldId(100), DataType::INT64, true}}, items));
    EXPECT_EQ(Names(nodes.back()->output_type()),
              (std::vector<std::string>{"score", "extra"}));
    auto constants = Nodes(BindProjection(
        {},
        {{"base", Number(4)},
         {"score", Op(Name("base"), proto::plan::Mul, Number(2))}}));
    EXPECT_EQ(Names(constants.back()->output_type()),
              (std::vector<std::string>{"base", "score"}));
    auto scan = std::dynamic_pointer_cast<const logical::ScalarProjectScanNode>(
        constants.front());
    EXPECT_TRUE(
        CompileExpressions(scan->expressions()).plan->inputs().slots().empty());
}
TEST(ProjectionBindingTest, StringAliasesRetainOwnersAndEmptyResultsKeepNames) {
    auto nodes = Nodes(
        BindProjection({{"text", FieldId(100), DataType::VARCHAR, true}},
                       {{"label", Name("text")}, {"copy", Name("label")}}));
    VectorResultBuilder values(DataType::VARCHAR, 3);
    values.Set<std::string>(0, std::string(4096, 'x'));
    values.Set<std::string>(2, "tail");
    RowVectorPtr output = std::make_shared<RowVector>(
        nodes[0]->output_type(),
        3,
        std::vector<VectorPtr>{
            std::move(values).Finish(SelectionView::All(3)).AsVector()});
    for (size_t i = 1; i < nodes.size(); ++i) {
        auto node = std::dynamic_pointer_cast<const logical::ScalarProjectNode>(
            nodes[i]);
        CompileOptions options;
        options.upstream_schema = node->sources()[0]->output_type();
        options.upstream_results = node->upstream_results();
        auto compiled = CompileExpressions(node->expressions(), options);
        ASSERT_TRUE(compiled.supported());
        ScalarProject project(Prepare(compiled.plan),
                              Names(node->output_type()));
        output = project.Eval(
            ReadBatchBinding::Range(0, 3), SelectionView::All(3), output.get());
    }
    for (size_t i = 0; i < 2; ++i) {
        auto result = ReadOnlyVectorView::CopyOf(output->child(i));
        EXPECT_EQ(result.StringViewAt(0), std::string(4096, 'x'));
        EXPECT_FALSE(result.IsValid(1));
        EXPECT_EQ(result.StringViewAt(2), "tail");
    }
    auto schema = std::make_shared<Schema>();
    auto pk = schema->AddDebugField("pk", DataType::INT64);
    schema->set_primary_field_id(pk);
    auto segment = segcore::CreateSealedSegment(schema, empty_index_meta);
    query::RetrievePlanNode retrieve;
    retrieve.limit_ = segcore::Unlimited;
    retrieve.plannodes_ = MakeProjectionPlan(
        BindProjection(
            {},
            {{"base", Number(4)},
             {"score", Op(Name("base"), proto::plan::Mul, Number(2))}}),
        std::make_shared<logical::MvccNode>("empty_mvcc"));
    query::ExecPlanNodeVisitor visitor(*segment, MAX_TIMESTAMP);
    auto empty = visitor.get_retrieve_result(retrieve);
    ASSERT_EQ(empty.field_data_.size(), 2);
    EXPECT_EQ(empty.field_data_[1].field_name(), "score");
    EXPECT_EQ(empty.field_data_[1].type(), proto::schema::Int64);
    EXPECT_EQ(empty.field_data_[1].scalars().long_data().data_size(), 0);
}
}  // namespace milvus::exec::plan
