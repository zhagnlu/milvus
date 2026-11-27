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
#include "exec/plan/ExprCompiler.h"
#include "exec/plan/ExecutionPreparer.h"
#include "segcore/reader/ReaderTestUtils.h"

namespace milvus::exec::plan {
namespace {
using namespace segcore::reader;
proto::plan::GenericValue
Int(int64_t n) {
    proto::plan::GenericValue v;
    v.set_int64_val(n);
    return v;
}
expr::ColumnInfo
Field(int64_t id = 100) {
    return expr::ColumnInfo(FieldId(id), DataType::INT64, {}, true);
}
expr::TypedExprPtr
Compare(proto::plan::OpType op = proto::plan::GreaterThan, int64_t n = 4) {
    return std::make_shared<expr::UnaryRangeFilterExpr>(Field(), op, Int(n));
}
expr::TypedExprPtr
Project() {
    return std::make_shared<expr::ColumnExpr>(Field());
}
expr::TypedExprPtr
Null(bool negate = false) {
    return std::make_shared<expr::NullExpr>(
        Field(),
        negate ? proto::plan::NullExpr_NullOp_IsNotNull
               : proto::plan::NullExpr_NullOp_IsNull);
}
CompiledExprPlanPtr
Compile(std::vector<expr::TypedExprPtr> roots) {
    CompileOptions options;
    options.relation = "snapshot-A";
    options.plan_indexes = true;
    auto result = CompileExpressions(roots, options);
    EXPECT_TRUE(result.supported());
    return result.plan;
}
void
Expect(const ReadOnlyVectorView& v, const std::vector<int>& expected) {
    ASSERT_EQ(v.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_EQ(v.IsValid(i) ? int(v.ValueAt<bool>(i)) : -1, expected[i])
            << i;
}
struct Data {
    const int64_t values[6] = {1, 0, 7, 9, 0, 3};
    const bool valid[6] = {true, false, true, true, false, true};
    int binds = 0, factories = 0, reads = 0, queries = 0, index_binds = 0;
    bool fail_read = false, fail_query = false, fail_bind = false;
    std::vector<std::vector<int64_t>> requests;
};
PreparedInput
Raw(std::shared_ptr<Data> data, const InputDependency& dependency) {
    ++data->binds;
    EXPECT_EQ(dependency.relation, "snapshot-A");
    EXPECT_EQ(dependency.field_id, 100);
    return {DataType::INT64,
            true,
            true,
            data,
            [data](const ReadBatchBinding& binding) {
                ++data->factories;
                auto values = [data, binding](SelectionView rows) {
                    ++data->reads;
                    if (data->fail_read)
                        ThrowInfo(FileReadFailed, "raw source failed");
                    auto column = std::make_shared<ColumnVector>(
                        DataType::INT64, binding.size());
                    std::vector<int64_t> positions;
                    rows.ForEachSelected([&](size_t i) {
                        auto offset = binding.offset(i);
                        AssertInfo(offset < 6, "fixture offset out of range");
                        positions.push_back(offset);
                        column->SetValueAt<int64_t>(i, data->values[offset]);
                        if (!data->valid[offset])
                            column->nullAt(i);
                    });
                    data->requests.push_back(std::move(positions));
                    return ReadOnlyVectorView::CopyOf(column, rows);
                };
                return InputLoaders{
                    values, [data, binding](SelectionView rows) {
                        ++data->reads;
                        auto valid = OwnedSelection::None(binding.size());
                        rows.ForEachSelected([&](size_t i) {
                            if (data->valid[binding.offset(i)])
                                valid.SetSelected(i);
                        });
                        return ValidityResult{OwnedSelection::CopyOf(rows),
                                              std::move(valid)};
                    }};
            }};
}
class CountingReader : public IndexReader {
 public:
    CountingReader(std::shared_ptr<const IndexReader> inner,
                   std::shared_ptr<Data> data)
        : inner_(std::move(inner)), data_(std::move(data)) {
    }
    int64_t
    row_count() const override {
        return inner_->row_count();
    }
    IndexResult
    Query(const ReadContext& ctx) const override {
        ++data_->queries;
        if (data_->fail_query)
            ThrowInfo(FileReadFailed, "index source failed");
        return inner_->Query(ctx);
    }

 private:
    std::shared_ptr<const IndexReader> inner_;
    std::shared_ptr<Data> data_;
};
PhysicalPrepareOptions
Indexed(std::shared_ptr<Data> data) {
    auto sorted =
        std::make_shared<milvus::segcore::reader::TestScalarIndex<int64_t>>();
    sorted->Build(6, data->values, data->valid);
    return {
        true, [data, sorted](const PredicatePlan& predicate)
                  -> std::optional<PreparedIndex> {
            // This fixture only owns the INT64 index on field 100.
            if (predicate.field.field_id != 100 || predicate.typed_query)
                return std::nullopt;
            ++data->index_binds;
            if (data->fail_bind)
                ThrowInfo(FileReadFailed, "index bind failed");
            EXPECT_EQ(predicate.field.relation, "snapshot-A");
            EXPECT_EQ(predicate.field.field_id, 100);
            auto result = PrepareInt64Index(
                sorted, 6, predicate.query, std::make_shared<ReadContext>());
            result->reader =
                std::make_shared<CountingReader>(result->reader, data);
            return result;
        }};
}
class CoarseReader : public IndexReader {
 public:
    std::shared_ptr<Data> data;
    IndexResultKind actual = IndexResultKind::Candidates;
    explicit CoarseReader(std::shared_ptr<Data> data) : data(std::move(data)) {
    }
    int64_t
    row_count() const override {
        return 6;
    }
    IndexResult
    Query(const ReadContext&) const override {
        ++data->queries;
        auto yes = OwnedSelection::FromIndices(6, {2, 3});
        if (actual == IndexResultKind::ExactTrue)
            return IndexResult::TrueOnly(yes.view());
        auto candidates = OwnedSelection::FromIndices(6, {0, 2, 3});
        auto no = OwnedSelection::FromIndices(6, {5});
        auto nulls = OwnedSelection::FromIndices(6, {1});
        return IndexResult::Candidates(
            candidates.view(), no.view(), nulls.view());
    }
};
}  // namespace

TEST(PhysicalPlanTest, CompilerRecordsResourceFreeAlternativesAndDirectUses) {
    auto plan = Compile({Compare(), Project(), Null()});
    ASSERT_EQ(plan->predicates().size(), 2);
    ASSERT_EQ(plan->inputs().slots().size(), 1);
    ASSERT_EQ(plan->direct_inputs().size(), 1);
    EXPECT_TRUE(plan->direct_inputs()[0].values);
    EXPECT_EQ(plan->predicates()[0].query.op, ScalarIndexOp::Greater);
    EXPECT_EQ(plan->predicates()[0].query.lower, 4);
    EXPECT_TRUE(plan->predicates()[0].raw_inputs[0].values);
    EXPECT_FALSE(plan->predicates()[1].raw_inputs[0].values);
}

TEST(PhysicalPlanTest, ExactIndexBindsNoRawSourceAndQueryIsLazyAcrossBatches) {
    auto data = std::make_shared<Data>();
    auto prepared = Prepare(Compile({Compare()}), {}, Indexed(data));
    EXPECT_EQ(prepared.predicate_routes(),
              (std::vector<PredicateRoute>{PredicateRoute::ExactIndex}));
    EXPECT_EQ(data->queries, 0);
    auto first = prepared.MakeExecutionBatch(ReadBatchBinding::Range(3, 3));
    first->Eval(SelectionView::None(3));
    EXPECT_EQ(data->queries, 0);
    Expect(first->Eval(SelectionView::All(3))[0], {1, -1, 0});
    auto earlier =
        prepared.MakeExecutionBatch(ReadBatchBinding::Offsets({2, 1, 2, 0}));
    Expect(earlier->Eval(SelectionView::All(4))[0], {1, -1, 1, 0});
    EXPECT_EQ(data->queries, 1);
    EXPECT_THROW(prepared.MakeBatch(ReadBatchBinding::Range(0, 1)),
                 SegcoreError);
}

TEST(PhysicalPlanTest, ExactIndexAndProjectionShareBatchWithoutEagerRawReads) {
    auto data = std::make_shared<Data>();
    auto prepared = Prepare(
        Compile({Compare(), Project()}),
        [data](const auto& dep) { return Raw(data, dep); },
        Indexed(data));
    EXPECT_EQ(data->binds, 1);
    auto batch =
        prepared.MakeExecutionBatch(ReadBatchBinding::Offsets({3, 1, 3, 5}));
    EXPECT_EQ(data->factories, 0);
    auto result = batch->Eval(std::vector<SelectionView>{
        SelectionView::All(4), SelectionView::None(4)});
    Expect(result[0], {1, -1, 1, 0});
    EXPECT_EQ(data->reads, 0);
    auto rows = OwnedSelection::FromIndices(4, {0, 2});
    result = batch->Eval(
        std::vector<SelectionView>{SelectionView::None(4), rows.view()});
    EXPECT_EQ(result[1].ValueAt<int64_t>(0), 9);
    EXPECT_EQ(result[1].ValueAt<int64_t>(2), 9);
    EXPECT_EQ(data->requests, (std::vector<std::vector<int64_t>>{{3, 3}}));
    EXPECT_EQ(data->queries, 1);
}

TEST(PhysicalPlanTest,
     DisabledPolicySkipsIndexBinderAndSelectsRawBeforeExecution) {
    auto data = std::make_shared<Data>();
    auto options = Indexed(data);
    options.use_index = false;
    auto prepared = Prepare(
        Compile({Compare()}),
        [data](const auto& dep) { return Raw(data, dep); },
        options);
    EXPECT_EQ(prepared.predicate_routes()[0], PredicateRoute::Raw);
    EXPECT_EQ(data->index_binds, 0);
    EXPECT_EQ(data->reads, 0);
    auto batch = prepared.MakeExecutionBatch(ReadBatchBinding::Range(0, 6));
    Expect(batch->Eval(SelectionView::All(6))[0], {0, -1, 1, 1, -1, 0});
    EXPECT_EQ(data->reads, 1);
}

TEST(PhysicalPlanTest, MissingOrTrueOnlyCapabilityUsesRawWithoutQuerying) {
    for (bool true_only : {false, true}) {
        auto data = std::make_shared<Data>();
        auto reader = std::make_shared<CoarseReader>(data);
        reader->actual = IndexResultKind::ExactTrue;
        PhysicalPrepareOptions options{
            true, [=](const auto&) -> std::optional<PreparedIndex> {
                if (!true_only)
                    return std::nullopt;
                return PreparedIndex{IndexResultKind::ExactTrue,
                                     reader,
                                     std::make_shared<ReadContext>()};
            }};
        auto prepared = Prepare(
            Compile({Compare()}),
            [data](const auto& dep) { return Raw(data, dep); },
            options);
        EXPECT_EQ(prepared.predicate_routes()[0], PredicateRoute::Raw);
        auto batch = prepared.MakeExecutionBatch(ReadBatchBinding::Range(0, 6));
        Expect(batch->Eval(SelectionView::All(6))[0], {0, -1, 1, 1, -1, 0});
        EXPECT_EQ(data->queries, 0);
    }
}

TEST(PhysicalPlanTest,
     CandidateRouteAutomaticallyRefinesAndSharesRawProjection) {
    auto data = std::make_shared<Data>();
    auto reader = std::make_shared<CoarseReader>(data);
    PhysicalPrepareOptions options{true, [=](const auto&) {
                                       return PreparedIndex{
                                           IndexResultKind::Candidates,
                                           reader,
                                           std::make_shared<ReadContext>()};
                                   }};
    auto prepared = Prepare(
        Compile({Compare(), Project()}),
        [data](const auto& dep) { return Raw(data, dep); },
        options);
    EXPECT_EQ(prepared.predicate_routes()[0], PredicateRoute::Refine);
    auto batch = prepared.MakeExecutionBatch(ReadBatchBinding::Range(0, 6));
    auto result = batch->Eval(std::vector<SelectionView>{
        SelectionView::All(6), SelectionView::None(6)});
    Expect(result[0], {0, -1, 1, 1, -1, 0});
    EXPECT_EQ(data->requests,
              (std::vector<std::vector<int64_t>>{{0, 2, 3, 4}}));
    result = batch->Eval(std::vector<SelectionView>{SelectionView::None(6),
                                                    SelectionView::All(6)});
    EXPECT_FALSE(result[1].IsValid(1));
    EXPECT_EQ(result[1].ValueAt<int64_t>(5), 3);
    EXPECT_EQ(data->requests,
              (std::vector<std::vector<int64_t>>{{0, 2, 3, 4}, {1, 5}}));
    EXPECT_EQ(data->queries, 1);
}

TEST(PhysicalPlanTest,
     PrunedComparisonDoesNotUpgradeAnotherRootValidityRequirement) {
    auto data = std::make_shared<Data>();
    auto index_options = Indexed(data);
    PhysicalPrepareOptions options{true, [=](const PredicatePlan& p) {
                                       return p.query.op ==
                                                      ScalarIndexOp::IsNull
                                                  ? std::nullopt
                                                  : index_options.bind_index(p);
                                   }};
    auto prepared = Prepare(
        Compile({Compare(), Null()}),
        [&](const InputDependency& dep) {
            EXPECT_FALSE(dep.needs_values);
            EXPECT_TRUE(dep.needs_validity);
            auto source = Raw(data, dep);
            source.supports_values =
                false;  // only validity, comparison's raw alternative is pruned
            return source;
        },
        options);
    auto batch = prepared.MakeExecutionBatch(ReadBatchBinding::Range(0, 6));
    auto result = batch->Eval(SelectionView::All(6));
    Expect(result[0], {0, -1, 1, 1, -1, 0});
    Expect(result[1], {0, 1, 0, 0, 1, 0});
    EXPECT_EQ(data->queries, 1);
    EXPECT_EQ(data->reads, 1);
}

TEST(PhysicalPlanTest, EveryPlannedOperationMatchesRawCompilation) {
    std::vector<expr::TypedExprPtr> roots;
    for (auto op : {proto::plan::Equal,
                    proto::plan::NotEqual,
                    proto::plan::LessThan,
                    proto::plan::LessEqual,
                    proto::plan::GreaterThan,
                    proto::plan::GreaterEqual})
        roots.push_back(Compare(op, 7));
    roots.push_back(Null());
    roots.push_back(Null(true));
    roots.push_back(std::make_shared<expr::BinaryRangeFilterExpr>(
        Field(), Int(3), Int(9), true, false));
    auto data = std::make_shared<Data>();
    auto physical = Prepare(Compile(roots), {}, Indexed(data));
    auto raw = Prepare(Compile(roots),
                       [data](const auto& dep) { return Raw(data, dep); });
    auto binding = ReadBatchBinding::Offsets({4, 3, 0, 2, 5, 1, 2});
    auto actual =
        physical.MakeExecutionBatch(binding)->Eval(SelectionView::All(7));
    auto expected =
        raw.MakeExecutionBatch(binding)->Eval(SelectionView::All(7));
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t root = 0; root < actual.size(); ++root)
        for (size_t row = 0; row < 7; ++row) {
            EXPECT_EQ(actual[root].IsValid(row), expected[root].IsValid(row));
            if (actual[root].IsValid(row))
                EXPECT_EQ(actual[root].ValueAt<bool>(row),
                          expected[root].ValueAt<bool>(row));
        }
}

TEST(PhysicalPlanTest, QueryAndBindingErrorsNeverFallBackToRaw) {
    auto data = std::make_shared<Data>();
    auto plan = Compile({Compare()});
    auto options = Indexed(data);
    data->fail_bind = true;
    try {
        Prepare(
            plan, [data](const auto& dep) { return Raw(data, dep); }, options);
        FAIL();
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), FileReadFailed);
    }
    EXPECT_EQ(data->binds, 0);
    data->fail_bind = false;
    auto prepared = Prepare(
        plan, [data](const auto& dep) { return Raw(data, dep); }, options);
    auto batch = prepared.MakeExecutionBatch(ReadBatchBinding::Range(0, 6));
    data->fail_query = true;
    try {
        batch->Eval(SelectionView::All(6));
        FAIL();
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), FileReadFailed);
    }
    EXPECT_EQ(data->queries, 1);
    EXPECT_EQ(data->binds, 0);
    EXPECT_EQ(data->reads, 0);
    data->fail_query = false;
    Expect(batch->Eval(SelectionView::All(6))[0], {0, -1, 1, 1, -1, 0});
    EXPECT_EQ(data->queries, 2);
}

TEST(PhysicalPlanTest, LyingCapabilityFailsInsteadOfPublishingOrReadingRaw) {
    auto data = std::make_shared<Data>();
    auto reader = std::make_shared<CoarseReader>(data);
    reader->actual = IndexResultKind::ExactTrue;
    PhysicalPrepareOptions options{true, [=](const auto&) {
                                       return PreparedIndex{
                                           IndexResultKind::FullThreeValued,
                                           reader,
                                           std::make_shared<ReadContext>()};
                                   }};
    auto prepared = Prepare(Compile({Compare()}), {}, options);
    auto batch = prepared.MakeExecutionBatch(ReadBatchBinding::Range(0, 6));
    EXPECT_THROW(batch->Eval(SelectionView::All(6)), SegcoreError);
    EXPECT_EQ(data->queries, 1);
    EXPECT_EQ(data->reads, 0);
}

TEST(PhysicalPlanTest,
     CandidateRawFailureDoesNotPublishAndCoarseRemainsReusable) {
    auto data = std::make_shared<Data>();
    PhysicalPrepareOptions options{true, [=](const auto&) {
                                       return PreparedIndex{
                                           IndexResultKind::Candidates,
                                           std::make_shared<CoarseReader>(data),
                                           std::make_shared<ReadContext>()};
                                   }};
    auto prepared = Prepare(
        Compile({Compare()}),
        [data](const auto& dep) { return Raw(data, dep); },
        options);
    auto batch = prepared.MakeExecutionBatch(ReadBatchBinding::Range(0, 6));
    data->fail_read = true;
    EXPECT_THROW(batch->Eval(SelectionView::All(6)), SegcoreError);
    data->fail_read = false;
    Expect(batch->Eval(SelectionView::All(6))[0], {0, -1, 1, 1, -1, 0});
    EXPECT_EQ(data->reads, 2);
    EXPECT_EQ(data->queries, 1);
}

TEST(PhysicalPlanTest,
     RepreparedPlanHasIndependentQueryResultsAndRetainedBatchOwners) {
    auto plan = Compile({Compare()});
    auto first = std::make_shared<Data>(), second = std::make_shared<Data>();
    std::unique_ptr<ExecutionBatch> retained;
    {
        auto prepared = Prepare(plan, {}, Indexed(first));
        retained = prepared.MakeExecutionBatch(ReadBatchBinding::Range(0, 6));
    }
    auto prepared = Prepare(plan, {}, Indexed(second));
    Expect(retained->Eval(SelectionView::All(6))[0], {0, -1, 1, 1, -1, 0});
    Expect(prepared.MakeExecutionBatch(ReadBatchBinding::Range(0, 6))
               ->Eval(SelectionView::All(6))[0],
           {0, -1, 1, 1, -1, 0});
    EXPECT_EQ(first->queries, 1);
    EXPECT_EQ(second->queries, 1);
}

TEST(PhysicalPlanTest, LogicalShortCircuitAndCancellationDoNotQuery) {
    proto::plan::GenericValue false_value;
    false_value.set_bool_val(false);
    auto root = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::And,
        std::make_shared<expr::ValueExpr>(false_value),
        Compare());
    auto data = std::make_shared<Data>();
    auto prepared = Prepare(Compile({root}), {}, Indexed(data));
    auto batch = prepared.MakeExecutionBatch(ReadBatchBinding::Range(0, 6));
    EXPECT_THROW(batch->Eval(SelectionView::All(6),
                             [] { ThrowInfo(FollyCancel, "cancel"); }),
                 SegcoreError);
    Expect(batch->Eval(SelectionView::All(6))[0], {0, 0, 0, 0, 0, 0});
    EXPECT_EQ(data->queries, 0);
}

TEST(PhysicalPlanTest, UpstreamFieldsNeverProbeStoredIndexes) {
    CompileOptions options;
    options.plan_indexes = true;
    options.upstream_columns[100] = 0;
    options.upstream_schema = std::make_shared<RowType>(
        std::vector<std::string>{"a"}, std::vector<DataType>{DataType::INT64});
    auto compiled = CompileExpressions({Compare()}, options);
    ASSERT_TRUE(compiled.supported());
    EXPECT_TRUE(compiled.plan->predicates().empty());
    auto data = std::make_shared<Data>();
    auto prepared = Prepare(compiled.plan, {}, Indexed(data));
    EXPECT_EQ(data->index_binds, 0);
}

TEST(PhysicalPlanTest, IndexHelperRejectsUnsupportedBackendAndInvalidBindings) {
    auto context = std::make_shared<ReadContext>();
    EXPECT_FALSE(
        PrepareInt64Index(nullptr, 6, {ScalarIndexOp::Equal, 1}, context));
    auto nested =
        std::make_shared<milvus::segcore::reader::TestScalarIndex<int64_t>>(
            storage::FileManagerContext(), true);
    EXPECT_FALSE(
        PrepareInt64Index(nested, 6, {ScalarIndexOp::Equal, 1}, context));
    auto sorted =
        std::make_shared<milvus::segcore::reader::TestScalarIndex<int64_t>>();
    const int64_t values[] = {1, 2};
    sorted->Build(2, values);
    EXPECT_THROW(
        PrepareInt64Index(sorted, 3, {ScalarIndexOp::Equal, 1}, context),
        SegcoreError);
    EXPECT_THROW(
        PrepareInt64Index(sorted, 2, {ScalarIndexOp::Equal, 1}, nullptr),
        SegcoreError);
}

TEST(PhysicalPlanTest, UnsupportedRootStillRejectsWholePhysicalPlan) {
    auto term = std::make_shared<expr::TermFilterExpr>(
        Field(), std::vector<proto::plan::GenericValue>{Int(1)}, true);
    CompileOptions options;
    options.plan_indexes = true;
    auto result = CompileExpressions({Compare(), term}, options);
    EXPECT_FALSE(result.supported());
    EXPECT_FALSE(result.unsupported.empty());
}

TEST(PhysicalPlanTest, PlannedNotAndOrPreserveUnknown) {
    auto negate = std::make_shared<expr::LogicalUnaryExpr>(
        expr::LogicalUnaryExpr::OpType::LogicalNot, Compare());
    auto either = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::Or,
        negate,
        Compare(proto::plan::Equal, 9));
    auto data = std::make_shared<Data>();
    auto prepared = Prepare(Compile({either}), {}, Indexed(data));
    auto batch = prepared.MakeExecutionBatch(ReadBatchBinding::Range(0, 6));
    Expect(batch->Eval(SelectionView::All(6))[0], {1, -1, 0, 1, -1, 1});
    EXPECT_EQ(data->queries, 2);
}

TEST(PhysicalPlanTest,
     FilterTtlIsCompiledOnceAndNeverInjectedIntoExactProgram) {
    CompileOptions options;
    options.plan_indexes = true;
    options.relation = "snapshot-A";
    auto compiled =
        CompileFilter(Compare(), Compare(proto::plan::LessThan, 9), options);
    ASSERT_TRUE(compiled.supported());
    ASSERT_EQ(compiled.plan->predicates().size(), 2);
    auto data = std::make_shared<Data>();
    auto prepared = Prepare(compiled.plan, {}, Indexed(data));
    auto result = prepared.MakeExecutionBatch(ReadBatchBinding::Range(0, 6))
                      ->Eval(SelectionView::All(6));
    Expect(result[0], {0, -1, 1, 0, -1, 0});
    EXPECT_EQ(data->queries, 2);
}

TEST(PhysicalPlanTest,
     ExactProgramsAreExecutableWithRawInputsAndNoPredicateProvider) {
    auto plan = Compile({Compare(), Null()});
    auto data = std::make_shared<Data>();
    // Turning off physical planning compiles the same ordinary field layout.
    auto ordinary = CompileExpressions({Project()});
    auto prepared = Prepare(ordinary.plan, [data](const auto& dep) {
        auto copy = dep;
        copy.relation = "snapshot-A";
        return Raw(data, copy);
    });
    auto raw = prepared.MakeBatch(ReadBatchBinding::Range(0, 6));
    scalar::EvalCtx context(*raw);  // deliberately no PredicateInputs
    Expect(plan->predicates()[0].exact->Eval(SelectionView::All(6), context),
           {0, -1, 1, 1, -1, 0});
    Expect(plan->predicates()[1].exact->Eval(SelectionView::All(6), context),
           {0, 1, 0, 0, 1, 0});
}

TEST(PhysicalPlanTest,
     InvalidPreparedMetadataAndMissingRawCapabilitiesFailBeforeQuery) {
    auto data = std::make_shared<Data>();
    auto compiled = Compile({Compare()});
    EXPECT_THROW(Prepare(compiled), SegcoreError);
    PhysicalPrepareOptions malformed{
        true, [](const auto&) {
            return PreparedIndex{
                IndexResultKind::FullThreeValued, nullptr, nullptr};
        }};
    EXPECT_THROW(Prepare(compiled, {}, malformed), SegcoreError);
    EXPECT_THROW(Prepare(compiled,
                         [&](const auto& dep) {
                             auto source = Raw(data, dep);
                             source.supports_values = false;
                             return source;
                         }),
                 SegcoreError);
    EXPECT_EQ(data->queries, 0);
    EXPECT_EQ(data->reads, 0);
}
TEST(PhysicalPlanTest, PreparedOrderIsStableAndDoesNotMutateSharedPlan) {
    proto::plan::GenericValue text;
    text.set_string_val("x");
    auto string = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(FieldId(101), DataType::VARCHAR, {}, false),
        proto::plan::Equal,
        text);
    auto root = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::And, string, Compare());
    auto compiled = Compile({root});
    for (bool reorder : {true, false, true}) {
        auto data = std::make_shared<Data>();
        std::vector<size_t> string_rows;
        auto physical = Indexed(data);
        physical.reorder = reorder;
        auto prepared = Prepare(
            compiled,
            [&](const InputDependency& dep) {
                EXPECT_EQ(dep.field_id, 101);
                return PreparedInput{
                    DataType::VARCHAR,
                    true,
                    false,
                    data,
                    [&](const ReadBatchBinding& binding) {
                        return InputLoaders{
                            [&, size = binding.size()](SelectionView rows) {
                                auto column = std::make_shared<ColumnVector>(
                                    DataType::VARCHAR, size);
                                rows.ForEachSelected([&](size_t row) {
                                    string_rows.push_back(row);
                                    column->SetValueAt<std::string>(row, "x");
                                });
                                return ReadOnlyVectorView::CopyOf(column, rows);
                            },
                            {}};
                    }};
            },
            physical);
        EXPECT_EQ(prepared.granularity(), ExecutionGranularity::Batch);
        auto batch = prepared.MakeExecutionBatch(ReadBatchBinding::Range(0, 6));
        Expect(batch->Eval(SelectionView::All(6))[0], {0, -1, 1, 1, -1, 0});
        EXPECT_EQ(string_rows,
                  reorder ? std::vector<size_t>({1, 2, 3, 4})
                          : std::vector<size_t>({0, 1, 2, 3, 4, 5}));
        EXPECT_EQ(data->queries, 1);
        EXPECT_EQ(data->binds, 0);
    }
}

TEST(PhysicalPlanTest,
     NumericTierIsNotOverriddenByIndexAndNamespaceTakesPriority) {
    auto second = std::make_shared<expr::UnaryRangeFilterExpr>(
        Field(101), proto::plan::GreaterThan, Int(4));
    auto compiled = Compile({std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::And, Compare(), second)});
    for (bool namespace_first : {false, true}) {
        auto data = std::make_shared<Data>();
        auto indexed = Indexed(data);
        PhysicalPrepareOptions options;
        if (namespace_first)
            options.namespace_field = 101;
        options.bind_index = [&](const PredicatePlan& predicate)
            -> std::optional<PreparedIndex> {
            if (predicate.field.field_id == 100)
                return std::nullopt;
            // An exact FALSE source allows observing whether raw evaluated.
            auto p = predicate;
            p.field.field_id = 100;
            p.query.lower = 100;
            return indexed.bind_index(p);
        };
        auto prepared = Prepare(
            compiled, [&](const auto& dep) { return Raw(data, dep); }, options);
        auto batch = prepared.MakeExecutionBatch(ReadBatchBinding::Range(0, 1));
        Expect(batch->Eval(SelectionView::All(1))[0], {0});
        EXPECT_EQ(data->reads, namespace_first ? 0 : 1);
        EXPECT_EQ(data->queries, namespace_first ? 1 : 0);
    }
}

TEST(PhysicalPlanTest, BackendIdentityDoesNotSubstituteForResultCompleteness) {
    auto compiled = Compile({Compare()});
    for (auto backend : {PredicateBackend::ScalarIndex,
                         PredicateBackend::PkIndex,
                         PredicateBackend::TextIndex,
                         PredicateBackend::JsonStats}) {
        auto data = std::make_shared<Data>();
        auto indexed = Indexed(data);
        auto binder = indexed.bind_index;
        indexed.bind_index = [binder, backend](const PredicatePlan& p) {
            auto result = binder(p);
            result->backend = backend;
            return result;
        };
        auto prepared = Prepare(compiled, {}, indexed);
        EXPECT_EQ(prepared.granularity(), ExecutionGranularity::WholeSegment);
        EXPECT_EQ(prepared.predicate_backends()[0], backend);
        auto batch = prepared.MakeExecutionBatch(
            ReadBatchBinding::Offsets({3, 1, 3, 0}));
        Expect(batch->Eval(SelectionView::All(4))[0], {1, -1, 1, 0});
    }
    auto data = std::make_shared<Data>();
    auto coarse = std::make_shared<CoarseReader>(data);
    PhysicalPrepareOptions options{true, [coarse](const PredicatePlan&) {
                                       return PreparedIndex{
                                           IndexResultKind::Candidates,
                                           coarse,
                                           std::make_shared<ReadContext>(),
                                           PredicateBackend::JsonStats};
                                   }};
    auto prepared = Prepare(
        compiled, [&](const auto& dep) { return Raw(data, dep); }, options);
    EXPECT_EQ(prepared.granularity(), ExecutionGranularity::Batch);
    EXPECT_EQ(prepared.predicate_routes()[0], PredicateRoute::Refine);
}
TEST(PhysicalPlanTest, WholeSegmentPlanStillSkipsUnneededIndexQueries) {
    auto full = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::Or, Null(), Null(true));
    auto root = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::Or, full, Compare());
    auto data = std::make_shared<Data>();
    auto prepared = Prepare(Compile({root}), {}, Indexed(data));
    EXPECT_EQ(prepared.granularity(), ExecutionGranularity::WholeSegment);
    EXPECT_EQ(data->queries, 0);
    auto batch = prepared.MakeExecutionBatch(ReadBatchBinding::Range(0, 6));
    Expect(batch->Eval(SelectionView::All(6))[0], {1, 1, 1, 1, 1, 1});
    EXPECT_EQ(data->queries, 2);
    EXPECT_EQ(data->binds, 0);
}

TEST(PhysicalPlanTest, PatternCompilationRecordsExceptionBarrier) {
    proto::plan::GenericValue pattern;
    pattern.set_string_val("[");
    auto expr = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(FieldId(101), DataType::VARCHAR),
        proto::plan::RegexMatch,
        pattern);
    auto compiled = Compile({expr});
    ASSERT_TRUE(compiled);
    const auto& property =
        compiled->properties().at(compiled->expressions().roots()[0].get());
    EXPECT_FALSE(property.reorder.safe);
}
TEST(PhysicalPlanTest, JsonStatsPlanningClassifiesWholeAndSelectedPaths) {
    for (auto path : {std::vector<std::string>{"n"},
                      std::vector<std::string>{"arr", "0"},
                      std::vector<std::string>{}}) {
        for (bool integer : {true, false}) {
            proto::plan::GenericValue bound;
            if (integer)
                bound.set_int64_val(10);
            else
                bound.set_float_val(10.0);
            auto predicate = std::make_shared<expr::UnaryRangeFilterExpr>(
                expr::ColumnInfo(FieldId(100), DataType::JSON, path),
                proto::plan::GreaterThan,
                bound);
            auto compiled = Compile({predicate});
            ASSERT_TRUE(compiled);
            ASSERT_EQ(compiled->predicates().size(), path.empty() ? 0 : 1);
            if (path == std::vector<std::string>{"n"})
                EXPECT_TRUE(compiled->predicates()[0].json_stats);
            else if (!path.empty())
                EXPECT_TRUE(compiled->predicates()[0].selected_stats);
        }
    }
}

TEST(PhysicalPlanTest, ReusableWorkspaceRebindsRowsAndRetainsPublishedViews) {
    auto data = std::make_shared<Data>();
    auto prepared = Prepare(Compile({Compare(), Project()}),
                            [data](const auto& dep) { return Raw(data, dep); });
    auto batch = prepared.MakeReusableBatch(ReadBatchBinding::Range(0, 3));
    auto first = batch->Eval(SelectionView::All(3));
    Expect(first[0], {0, -1, 1});
    EXPECT_EQ(data->factories, 1);
    EXPECT_EQ(data->reads, 1);
    batch->Reset(ReadBatchBinding::Range(3, 3));
    EXPECT_EQ(data->factories, 1);  // still lazy
    auto second = batch->Eval(SelectionView::All(3));
    Expect(second[0], {1, -1, 0});
    Expect(first[0], {0, -1, 1});
    EXPECT_EQ(first[1].ValueAt<int64_t>(0), 1);
    EXPECT_EQ(second[1].ValueAt<int64_t>(0), 9);
    EXPECT_EQ(data->factories, 2);
    EXPECT_EQ(data->reads, 2);
    batch->Reset(ReadBatchBinding::Offsets({2, 1, 2, 0}));
    auto sparse = OwnedSelection::FromIndices(4, {0, 1});
    auto partial = batch->Eval(sparse.view());
    EXPECT_TRUE(partial[0].ValueAt<bool>(0));
    EXPECT_FALSE(partial[0].IsValid(1));
    auto third = batch->Eval(SelectionView::All(4));
    Expect(third[0], {1, -1, 1, 0});
    EXPECT_EQ(data->requests.back(), (std::vector<int64_t>{2, 0}));
    EXPECT_THROW(partial[0].IsValid(2), SegcoreError);
    batch->Reset(ReadBatchBinding::Range(0, 0));
    EXPECT_TRUE(batch->Eval(SelectionView::All(0))[0].defined_rows().IsNone());
    batch->Reset(ReadBatchBinding::Range(5, 1));
    Expect(batch->Eval(SelectionView::All(1))[0], {0});
    Expect(second[0], {1, -1, 0});
}

TEST(PhysicalPlanTest,
     ReusableWorkspaceGuardsResetAndRecoversAfterReadFailure) {
    auto data = std::make_shared<Data>();
    auto prepared = Prepare(Compile({Compare()}),
                            [data](const auto& dep) { return Raw(data, dep); });
    auto batch = prepared.MakeReusableBatch(ReadBatchBinding::Range(0, 3));
    auto first = batch->Eval(SelectionView::All(3));
    batch->Reset(ReadBatchBinding::Range(3, 3));
    data->fail_read = true;
    EXPECT_THROW(batch->Eval(SelectionView::All(3)), SegcoreError);
    data->fail_read = false;
    Expect(batch->Eval(SelectionView::All(3))[0], {1, -1, 0});
    EXPECT_THROW(
        batch->Eval(SelectionView::All(3),
                    [&] { batch->Reset(ReadBatchBinding::Range(0, 3)); }),
        SegcoreError);
    Expect(batch->Eval(SelectionView::All(3))[0], {1, -1, 0});
    Expect(first[0], {0, -1, 1});
    auto independent =
        prepared.MakeExecutionBatch(ReadBatchBinding::Range(0, 3));
    EXPECT_THROW(independent->Reset(ReadBatchBinding::Range(3, 3)),
                 SegcoreError);
}

TEST(PhysicalPlanTest, ReusableWorkspaceKeepsQueryIndexCacheAndSourceOwners) {
    auto data = std::make_shared<Data>();
    std::unique_ptr<ExecutionBatch> batch;
    {
        auto prepared = Prepare(
            Compile({Compare(), Project()}),
            [data](const auto& dep) { return Raw(data, dep); },
            Indexed(data));
        batch = prepared.MakeReusableBatch(ReadBatchBinding::Range(0, 3));
    }
    auto first = batch->Eval(SelectionView::All(3));
    batch->Reset(ReadBatchBinding::Offsets({3, 1, 3, 5}));
    auto second = batch->Eval(SelectionView::All(4));
    Expect(second[0], {1, -1, 1, 0});
    Expect(first[0], {0, -1, 1});
    EXPECT_EQ(data->queries, 1);
    EXPECT_EQ(second[1].ValueAt<int64_t>(2), 9);
}

TEST(PhysicalPlanTest, FailedWorkspaceResetCannotPublishOrResumeEvaluation) {
    auto fail = std::make_shared<bool>(false);
    auto logical = std::make_shared<expr::ColumnExpr>(
        expr::ColumnInfo(FieldId(100), DataType::GEOMETRY, {}, true));
    auto prepared = Prepare(Compile({logical}), [fail](const auto&) {
        PreparedInput source;
        source.type = DataType::GEOMETRY;
        source.supports_values = true;
        source.owner = fail;
        source.make_loaders = [](const ReadBatchBinding&) {
            return InputLoaders{[](SelectionView rows) {
                                    return ReadOnlyVectorView::Nulls(
                                        DataType::GEOMETRY, rows);
                                },
                                {}};
        };
        source.make_geometry_lookup = [fail](const ReadBatchBinding&) {
            if (*fail)
                ThrowInfo(FileReadFailed, "injected rebinding failure");
            return [](size_t) -> const Geometry* { return nullptr; };
        };
        return source;
    });
    auto batch = prepared.MakeReusableBatch(ReadBatchBinding::Range(0, 3));
    auto saved = batch->Eval(SelectionView::All(3))[0];
    *fail = true;
    EXPECT_THROW(batch->Reset(ReadBatchBinding::Range(3, 2)), SegcoreError);
    *fail = false;
    EXPECT_THROW(batch->Eval(SelectionView::All(2)), SegcoreError);
    EXPECT_THROW(batch->Reset(ReadBatchBinding::Range(0, 3)), SegcoreError);
    EXPECT_EQ(saved.size(), 3);
    EXPECT_FALSE(saved.IsValid(0));
    EXPECT_TRUE(saved.defined_rows().IsAll());
}

TEST(PhysicalPlanTest, InLookupIsPreparedOnlyForTheSelectedPath) {
    auto logical = std::make_shared<expr::TermFilterExpr>(
        Field(), std::vector{Int(7), Int(9), Int(7)});
    auto compiled = Compile({logical});
    ASSERT_EQ(compiled->predicates().size(), 1);
    ASSERT_FALSE(compiled->predicates()[0].exact);
    ASSERT_TRUE(compiled->predicates()[0].prepare_exact);
    EXPECT_TRUE(compiled->direct_inputs().empty());
    ASSERT_EQ(compiled->predicates()[0].raw_inputs.size(), 1);
    EXPECT_TRUE(compiled->predicates()[0].raw_inputs[0].values);
    // The shared plan owns its constants, including the deferred raw path.
    logical.reset();
    auto builds = std::make_shared<int>(0);
    auto predicates = compiled->predicates();
    auto factory = predicates[0].prepare_exact;
    predicates[0].prepare_exact = [builds, factory] {
        ++*builds;
        return factory();
    };
    auto plan = std::make_shared<const CompiledExprPlan>(
        compiled->inputs(),
        compiled->expressions().roots(),
        compiled->upstream_schema(),
        compiled->direct_inputs(),
        std::move(predicates),
        compiled->properties(),
        compiled->whole_segment_safe());
    // Reuse one immutable compiled plan across independent query preparations.
    for (int route = 0; route < 4; ++route) {
        auto data = std::make_shared<Data>();
        auto options = Indexed(data);
        if (route != 0) {
            auto reader = std::make_shared<CoarseReader>(data);
            reader->actual = route == 2 ? IndexResultKind::ExactTrue
                                        : IndexResultKind::Candidates;
            options.bind_index =
                [route, reader](const auto&) -> std::optional<PreparedIndex> {
                if (route == 1)
                    return std::nullopt;
                return PreparedIndex{
                    reader->actual, reader, std::make_shared<ReadContext>()};
            };
        }
        const int before = *builds;
        auto bind = options.bind_index;
        options.bind_index = [=](const auto& predicate) {
            EXPECT_EQ(*builds,
                      before);  // No raw structure before physical selection.
            return bind(predicate);
        };
        auto prepared = Prepare(
            plan, [data](const auto& dep) { return Raw(data, dep); }, options);
        EXPECT_EQ(*builds, before + (route != 0));
        EXPECT_EQ(prepared.predicate_routes()[0],
                  route == 0   ? PredicateRoute::ExactIndex
                  : route == 3 ? PredicateRoute::Refine
                               : PredicateRoute::Raw);
        EXPECT_EQ(data->queries, 0);
        EXPECT_EQ(data->reads, 0);
        auto batch = prepared.MakeReusableBatch(ReadBatchBinding::Range(0, 6));
        auto partial = OwnedSelection::FromIndices(6, {0, 2});
        auto result = batch->Eval(partial.view())[0];
        EXPECT_FALSE(result.ValueAt<bool>(0));
        EXPECT_TRUE(result.ValueAt<bool>(2));
        Expect(batch->Eval(SelectionView::All(6))[0], {0, -1, 1, 1, -1, 0});
        batch->Reset(ReadBatchBinding::Offsets({3, 1, 3, 0}));
        Expect(batch->Eval(SelectionView::All(4))[0], {1, -1, 1, 0});
        EXPECT_EQ(*builds, before + (route != 0));
        if (route == 0) {
            EXPECT_EQ(data->binds, 0);
            EXPECT_EQ(data->queries, 1);
        }
    }
}

TEST(PhysicalPlanTest,
     InPublicCompilationRetainsNarrowOverflowAndNullSemantics) {
    auto check = []<typename T>(DataType type) {
        const T values[] = {1, 0, 7, 9, 0, 3};
        const bool valid[] = {true, false, true, true, false, true};
        auto index =
            std::make_shared<milvus::segcore::reader::TestScalarIndex<T>>();
        index->Build(6, values, valid);
        const int64_t outside = int64_t(std::numeric_limits<T>::max()) + 1;
        for (auto terms :
             {std::vector<proto::plan::GenericValue>{},
              std::vector{Int(outside), Int(-outside - 1)},
              std::vector{Int(outside), Int(7), Int(7), Int(-outside - 1)}}) {
            auto term = std::make_shared<expr::TermFilterExpr>(
                expr::ColumnInfo(FieldId(100), type, {}, true), terms);
            auto negate = std::make_shared<expr::LogicalUnaryExpr>(
                expr::LogicalUnaryExpr::OpType::LogicalNot, term);
            auto compiled = Compile({term, negate});
            ASSERT_TRUE(compiled);
            const bool has_seven = terms.size() == 4;
            EXPECT_EQ(compiled->predicates()[0].query.terms.size(),
                      has_seven ? 2 : 0);
            VectorResultBuilder input(type, 6);
            for (size_t row = 0; row < 6; ++row)
                if (valid[row])
                    input.Set<T>(row, values[row]);
            auto view = std::move(input).Finish(SelectionView::All(6));
            for (bool indexed : {false, true}) {
                PhysicalPrepareOptions options{
                    indexed, [=](const PredicatePlan& p) {
                        return PrepareIntegerIndex<T>(
                            index, 6, p.query, std::make_shared<ReadContext>());
                    }};
                auto prepared = Prepare(
                    compiled,
                    [=](const InputDependency&) {
                        return PreparedInput{
                            type,
                            true,
                            false,
                            std::make_shared<ReadOnlyVectorView>(view),
                            [=](const ReadBatchBinding&) {
                                return InputLoaders{
                                    [=](SelectionView rows) {
                                        return view.Restrict(rows);
                                    },
                                    {}};
                            }};
                    },
                    options);
                auto results =
                    prepared.MakeExecutionBatch(ReadBatchBinding::Range(0, 6))
                        ->Eval(SelectionView::All(6));
                Expect(results[0], {0, -1, int(has_seven), 0, -1, 0});
                Expect(results[1], {1, -1, int(!has_seven), 1, -1, 1});
            }
        }
    };
    check.template operator()<int8_t>(DataType::INT8);
    check.template operator()<int16_t>(DataType::INT16);
    check.template operator()<int32_t>(DataType::INT32);
}

TEST(PhysicalPlanTest, InDeferredRawPreservesTypedConstantsAndNulls) {
    auto check = []<typename T>(
                     DataType type, T a, T b, proto::plan::GenericValue term) {
        auto compiled = Compile({std::make_shared<expr::TermFilterExpr>(
            expr::ColumnInfo(FieldId(100), type, {}, true),
            std::vector{term, term})});
        ASSERT_TRUE(compiled);
        ASSERT_EQ(compiled->predicates().size(), 1);
        EXPECT_FALSE(compiled->predicates()[0].exact);
        EXPECT_TRUE(compiled->predicates()[0].prepare_exact);
        VectorResultBuilder values(type, 3);
        values.Set<T>(0, a);
        values.Set<T>(1, b);
        auto view = std::move(values).Finish(SelectionView::All(3));
        auto prepared = Prepare(compiled, [=](const auto&) {
            return PreparedInput{
                type,
                true,
                false,
                std::make_shared<ReadOnlyVectorView>(view),
                [=](const ReadBatchBinding&) {
                    return InputLoaders{
                        [=](SelectionView rows) { return view.Restrict(rows); },
                        {}};
                }};
        });
        Expect(prepared.MakeExecutionBatch(ReadBatchBinding::Range(0, 3))
                   ->Eval(SelectionView::All(3))[0],
               {1, 0, -1});
    };
    proto::plan::GenericValue boolean, floating, text;
    boolean.set_bool_val(true);
    floating.set_float_val(1.5);
    text.set_string_val("北京");
    check(DataType::BOOL, true, false, boolean);
    check(DataType::FLOAT, 1.5f, 2.f, floating);
    check(DataType::DOUBLE, 1.5, 2., floating);
    check(DataType::VARCHAR, std::string("北京"), std::string("上海"), text);
}

TEST(PhysicalPlanTest, InMalformedConstantsRejectBeforePhysicalBinding) {
    proto::plan::GenericValue wrong, nonfinite;
    wrong.set_string_val("wrong type");
    nonfinite.set_float_val(std::numeric_limits<double>::infinity());
    for (bool plan_indexes : {false, true}) {
        CompileOptions options;
        options.plan_indexes = plan_indexes;
        for (auto type : {DataType::INT8,
                          DataType::INT64,
                          DataType::BOOL,
                          DataType::FLOAT,
                          DataType::DOUBLE}) {
            for (auto value : {wrong, nonfinite, proto::plan::GenericValue{}}) {
                auto term = std::make_shared<expr::TermFilterExpr>(
                    expr::ColumnInfo(FieldId(100), type, {}, true),
                    std::vector{value});
                auto compiled = CompileExpressions({term}, options);
                EXPECT_FALSE(compiled.supported());
                EXPECT_FALSE(compiled.unsupported.empty());
            }
        }
    }
}

TEST(PhysicalPlanTest, InIndexFailureDoesNotConstructOrFallBackToRaw) {
    auto plan = Compile(
        {std::make_shared<expr::TermFilterExpr>(Field(), std::vector{Int(7)})});
    auto data = std::make_shared<Data>();
    data->fail_bind = true;
    EXPECT_THROW(Prepare(
                     plan,
                     [data](const auto& dep) { return Raw(data, dep); },
                     Indexed(data)),
                 SegcoreError);
    EXPECT_EQ(data->binds, 0);
    data->fail_bind = false;
    auto prepared = Prepare(plan, {}, Indexed(data));
    data->fail_query = true;
    auto batch = prepared.MakeExecutionBatch(ReadBatchBinding::Range(0, 6));
    EXPECT_THROW(batch->Eval(SelectionView::All(6)), SegcoreError);
    EXPECT_EQ(data->binds, 0);
    EXPECT_EQ(data->reads, 0);
    data->fail_query = false;
    Expect(batch->Eval(SelectionView::All(6))[0], {0, -1, 1, 0, -1, 0});
}

}  // namespace milvus::exec::plan
