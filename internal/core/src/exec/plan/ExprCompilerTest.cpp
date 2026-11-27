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
#include "exec/expression/function/impl/StringFunctions.h"

namespace milvus::exec::plan {
namespace {

proto::plan::GenericValue
Int(int64_t value) {
    proto::plan::GenericValue result;
    result.set_int64_val(value);
    return result;
}

proto::plan::GenericValue
Text(const std::string& value) {
    proto::plan::GenericValue result;
    result.set_string_val(value);
    return result;
}

expr::ColumnInfo
Column(int64_t id, DataType type = DataType::INT64, bool nullable = false) {
    return expr::ColumnInfo(FieldId(id), type, {}, nullable);
}

expr::TypedExprPtr
Greater(int64_t id, int64_t value) {
    return std::make_shared<expr::UnaryRangeFilterExpr>(
        Column(id), proto::plan::GreaterThan, Int(value));
}

expr::TypedExprPtr
And(expr::TypedExprPtr left, expr::TypedExprPtr right) {
    return std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::And, left, right);
}

RowTypePtr
Schema(std::vector<std::string> names, std::vector<DataType> types) {
    return std::make_shared<RowType>(std::move(names), std::move(types));
}

ReadOnlyVectorView
Integers(std::vector<int64_t> values) {
    auto column =
        std::make_shared<ColumnVector>(DataType::INT64, values.size());
    for (size_t i = 0; i < values.size(); ++i)
        column->SetValueAt<int64_t>(i, values[i]);
    return ReadOnlyVectorView::CopyOf(column);
}

struct SourceState {
    int factories = 0;
    int loads = 0;
    bool fail_factory = false;
    bool fail_load = false;
    std::vector<std::vector<int64_t>> positions;
};

PreparedInput
NumericSource(std::shared_ptr<SourceState> state, int64_t shift = 0) {
    return {DataType::INT64,
            true,
            false,
            state,
            [state, shift](const ReadBatchBinding& binding) {
                ++state->factories;
                if (std::exchange(state->fail_factory, false))
                    ThrowInfo(FileReadFailed, "injected factory failure");
                return InputLoaders{
                    [state, shift, binding](SelectionView rows) {
                        ++state->loads;
                        if (std::exchange(state->fail_load, false))
                            ThrowInfo(FileReadFailed, "injected value failure");
                        auto column = std::make_shared<ColumnVector>(
                            DataType::INT64, binding.size());
                        std::vector<int64_t> positions;
                        rows.ForEachSelected([&](size_t row) {
                            positions.push_back(binding.offset(row));
                            column->SetValueAt<int64_t>(
                                row, binding.offset(row) + shift);
                        });
                        state->positions.push_back(std::move(positions));
                        return ReadOnlyVectorView::CopyOf(column, rows);
                    },
                    {}};
            }};
}

std::vector<ReadOnlyVectorView>
Eval(const PreparedExecution& prepared, BatchInput& input, SelectionView rows) {
    scalar::EvalCtx ctx(input);
    return prepared.plan()->expressions().Eval(rows, ctx);
}

int
BoolAt(const ReadOnlyVectorView& values, size_t row) {
    return values.IsValid(row) ? values.ValueAt<bool>(row) : -1;
}

}  // namespace

TEST(ExprCompilerTest, EmbeddedColumnDependenciesAreInternedAcrossRoots) {
    auto less = std::make_shared<expr::UnaryRangeFilterExpr>(
        Column(100), proto::plan::LessThan, Int(10));
    auto compiled =
        CompileExpressions({And(Greater(100, 4), less),
                            std::make_shared<expr::ColumnExpr>(Column(100))});
    ASSERT_TRUE(compiled.supported());
    ASSERT_EQ(compiled.plan->inputs().slots().size(), 1);
    EXPECT_EQ(compiled.plan->inputs().slots()[0].field_id, 100);
    EXPECT_TRUE(compiled.plan->inputs().slots()[0].needs_values);
    auto state = std::make_shared<SourceState>();
    int binds = 0;
    auto prepared = Prepare(compiled.plan, [&](const InputDependency& dep) {
        ++binds;
        EXPECT_EQ(dep.field_id, 100);
        return NumericSource(state);
    });
    auto batch = prepared.MakeBatch(ReadBatchBinding::Offsets({3, 5, 11}));
    EXPECT_EQ(binds, 1);
    EXPECT_EQ(state->factories, 0);
    EXPECT_EQ(state->loads, 0);
    auto output = Eval(prepared, *batch, SelectionView::All(3));
    EXPECT_EQ(BoolAt(output[0], 0), 0);
    EXPECT_EQ(BoolAt(output[0], 1), 1);
    EXPECT_EQ(BoolAt(output[0], 2), 0);
    EXPECT_EQ(output[1].ValueAt<int64_t>(1), 5);
    EXPECT_EQ(state->loads, 1);
}

TEST(ExprCompilerTest, ShortCircuitFirstBatchDoesNotAdvanceUnusedSecondField) {
    auto compiled = CompileExpressions({And(Greater(100, 4), Greater(101, 4))});
    ASSERT_TRUE(compiled.supported());
    auto first = std::make_shared<SourceState>(),
         second = std::make_shared<SourceState>();
    auto prepared = Prepare(compiled.plan, [&](const InputDependency& dep) {
        return NumericSource(dep.field_id == 100 ? first : second);
    });
    auto a = prepared.MakeBatch(ReadBatchBinding::Range(0, 3));
    auto b = prepared.MakeBatch(ReadBatchBinding::Range(10, 2));
    Eval(prepared, *a, SelectionView::All(3));
    EXPECT_EQ(second->factories, 0);
    EXPECT_EQ(second->loads, 0);
    auto out = Eval(prepared, *b, SelectionView::All(2));
    EXPECT_EQ(BoolAt(out[0], 0), 1);
    EXPECT_EQ(second->factories, 1);
    EXPECT_EQ(second->positions[0], (std::vector<int64_t>{10, 11}));
    // Repeated or backward batch evaluation does not consume a shared cursor.
    Eval(prepared, *a, SelectionView::All(3));
    EXPECT_EQ(first->loads, 2);
}

TEST(ExprCompilerTest,
     OffsetsOwnOrderAndDuplicatesAndEmptyRowsNeverInitialize) {
    auto compiled =
        CompileExpressions({std::make_shared<expr::ColumnExpr>(Column(100))});
    ASSERT_TRUE(compiled.supported());
    auto state = std::make_shared<SourceState>();
    auto prepared = Prepare(compiled.plan,
                            [&](const auto&) { return NumericSource(state); });
    std::vector<int64_t> offsets{130, 3, 130};
    auto binding = ReadBatchBinding::Offsets(offsets);
    auto batch = prepared.MakeBatch(binding);
    offsets.assign(3, 999);
    Eval(prepared, *batch, SelectionView::None(3));
    EXPECT_EQ(state->factories, 0);
    auto rows = OwnedSelection::FromIndices(3, {0, 2});
    auto out = Eval(prepared, *batch, rows.view());
    EXPECT_EQ(out[0].ValueAt<int64_t>(0), 130);
    EXPECT_EQ(out[0].ValueAt<int64_t>(2), 130);
    EXPECT_EQ(state->positions[0], (std::vector<int64_t>{130, 130}));
    auto all = Eval(prepared, *batch, SelectionView::All(3));
    EXPECT_EQ(state->positions[1], (std::vector<int64_t>{3}));
    EXPECT_EQ(all[0].ValueAt<int64_t>(1), 3);
    EXPECT_FALSE(out[0].defined_rows().IsSelected(1));
}

TEST(ExprCompilerTest, SamePlanHasIndependentQueryAndBatchState) {
    auto compiled =
        CompileExpressions({std::make_shared<expr::ColumnExpr>(Column(100))});
    auto one = std::make_shared<SourceState>(),
         two = std::make_shared<SourceState>();
    auto p1 = Prepare(compiled.plan,
                      [&](const auto&) { return NumericSource(one, 10); });
    auto p2 = Prepare(compiled.plan,
                      [&](const auto&) { return NumericSource(two, 20); });
    auto b1 = p1.MakeBatch(ReadBatchBinding::Range(0, 2));
    auto b2 = p2.MakeBatch(ReadBatchBinding::Range(0, 2));
    EXPECT_EQ(Eval(p1, *b1, SelectionView::All(2))[0].ValueAt<int64_t>(0), 10);
    EXPECT_EQ(two->loads, 0);
    EXPECT_EQ(Eval(p2, *b2, SelectionView::All(2))[0].ValueAt<int64_t>(0), 20);
    EXPECT_EQ(one->loads, 1);
    EXPECT_EQ(two->loads, 1);
}

TEST(ExprCompilerTest, BatchRetainsQueryOwnerAfterPreparedExecutionDies) {
    auto compiled =
        CompileExpressions({std::make_shared<expr::ColumnExpr>(Column(100))});
    std::weak_ptr<SourceState> weak;
    std::unique_ptr<BatchInput> batch;
    {
        auto state = std::make_shared<SourceState>();
        weak = state;
        auto prepared = Prepare(
            compiled.plan, [&](const auto&) { return NumericSource(state); });
        batch = prepared.MakeBatch(ReadBatchBinding::Range(5, 2));
    }
    EXPECT_FALSE(weak.expired());
    scalar::EvalCtx ctx(*batch);
    auto result = compiled.plan->expressions().Eval(SelectionView::All(2), ctx);
    batch.reset();
    EXPECT_TRUE(weak.expired());
    EXPECT_EQ(result[0].ValueAt<int64_t>(1), 6);
}

TEST(ExprCompilerTest, ValidityOnlyRequirementsMergeWithValueRequirements) {
    auto null =
        std::make_shared<expr::NullExpr>(Column(100, DataType::INT64, true),
                                         proto::plan::NullExpr_NullOp_IsNull);
    auto compiled = CompileExpressions({null});
    ASSERT_TRUE(compiled.supported());
    EXPECT_FALSE(compiled.plan->inputs().slots()[0].needs_values);
    auto owner = std::make_shared<int>(0);
    auto binder = [&](const auto&) {
        return PreparedInput{
            DataType::INT64,
            false,
            true,
            owner,
            [owner](const ReadBatchBinding& binding) {
                ++*owner;
                return InputLoaders{
                    {}, [binding](SelectionView rows) {
                        auto valid = OwnedSelection::None(binding.size());
                        rows.ForEachSelected([&](size_t row) {
                            if (binding.offset(row) % 2 == 0)
                                valid.SetSelected(row);
                        });
                        return ValidityResult{OwnedSelection::CopyOf(rows),
                                              std::move(valid)};
                    }};
            }};
    };
    auto prepared = Prepare(compiled.plan, binder);
    auto batch = prepared.MakeBatch(ReadBatchBinding::Range(0, 3));
    EXPECT_EQ(*owner, 0);
    auto out = Eval(prepared, *batch, SelectionView::All(3));
    EXPECT_EQ(BoolAt(out[0], 0), 0);
    EXPECT_EQ(BoolAt(out[0], 1), 1);
    EXPECT_EQ(BoolAt(out[0], 2), 0);

    auto mixed = CompileExpressions({null, Greater(100, 1)});
    ASSERT_TRUE(mixed.supported());
    ASSERT_EQ(mixed.plan->inputs().slots().size(), 1);
    EXPECT_TRUE(mixed.plan->inputs().slots()[0].needs_values);
    try {
        Prepare(mixed.plan, binder);
        FAIL();
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), Unsupported);
    }
    EXPECT_EQ(*owner, 1);  // rejection did not initialize another loader
}

TEST(ExprCompilerTest, ValueOnlySourceServesValidityThroughBatchInputCache) {
    auto compiled = CompileExpressions(
        {std::make_shared<expr::NullExpr>(Column(100),
                                          proto::plan::NullExpr_NullOp_IsNull),
         std::make_shared<expr::ColumnExpr>(Column(100))});
    auto state = std::make_shared<SourceState>();
    auto prepared = Prepare(compiled.plan,
                            [&](const auto&) { return NumericSource(state); });
    auto batch = prepared.MakeBatch(ReadBatchBinding::Range(0, 2));
    auto out = Eval(prepared, *batch, SelectionView::All(2));
    EXPECT_EQ(BoolAt(out[0], 0), 0);
    EXPECT_EQ(out[1].ValueAt<int64_t>(1), 1);
    EXPECT_EQ(state->loads, 1);
}

TEST(ExprCompilerTest, RangePreparedReaderHandlesOffsetsAndLoadFailure) {
    auto range = std::make_shared<expr::BinaryRangeFilterExpr>(
        Column(100), Int(10), Int(100), false, false);
    auto compiled = CompileFilter(range);
    ASSERT_TRUE(compiled.supported());
    auto state = std::make_shared<SourceState>();
    auto prepared = Prepare(compiled.plan,
                            [&](const auto&) { return NumericSource(state); });
    auto batch =
        prepared.MakeBatch(ReadBatchBinding::Offsets({100, 11, 10, 99, 11}));
    state->fail_load = true;
    try {
        Eval(prepared, *batch, SelectionView::All(5));
        FAIL() << "reader failure must propagate";
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), FileReadFailed);
    }
    auto output = Eval(prepared, *batch, SelectionView::All(5));
    const std::vector<int> expected{0, 1, 0, 1, 1};
    for (size_t row = 0; row < expected.size(); ++row)
        EXPECT_EQ(BoolAt(output[0], row), expected[row]);
    EXPECT_EQ(state->positions,
              (std::vector<std::vector<int64_t>>{{100, 11, 10, 99, 11}}));
    EXPECT_EQ(state->loads, 2);
    EXPECT_TRUE(
        CompileFilter(
            std::make_shared<expr::BinaryRangeFilterExpr>(
                Column(100, DataType::INT8), Int(10), Int(128), false, false))
            .supported());
}

TEST(ExprCompilerTest, MixedColumnComparisonsUseLegacyNativePromotions) {
    CompileOptions options;
    options.upstream_schema =
        Schema({"a", "b"}, {DataType::INT64, DataType::DOUBLE});
    options.upstream_columns = {{100, 0}, {101, 1}};
    std::vector<expr::TypedExprPtr> roots;
    for (auto op : {proto::plan::Equal,
                    proto::plan::NotEqual,
                    proto::plan::LessThan,
                    proto::plan::LessEqual,
                    proto::plan::GreaterThan,
                    proto::plan::GreaterEqual})
        roots.push_back(std::make_shared<expr::CompareExpr>(
            FieldId(100), FieldId(101), DataType::INT64, DataType::DOUBLE, op));
    auto compiled = CompileExpressions(roots, options);
    ASSERT_TRUE(compiled.supported());
    auto a = MakeWritableCopy(Integers({1, 2, 9007199254740993LL, 0}));
    auto b = std::make_shared<ColumnVector>(DataType::DOUBLE, 4);
    const std::vector<double> values{1.5, 1, 9007199254740992.0, 0};
    for (size_t i = 0; i < values.size(); ++i)
        b->SetValueAt<double>(i, values[i]);
    b->nullAt(3);
    RowVector upstream(options.upstream_schema, 4, {a.values, b});
    auto prepared = Prepare(compiled.plan);
    auto batch = prepared.MakeBatch(ReadBatchBinding::Range(0, 4), &upstream);
    auto output = Eval(prepared, *batch, SelectionView::All(4));
    const std::vector<std::vector<int>> expected{{0, 0, 1, -1},
                                                 {1, 1, 0, -1},
                                                 {1, 0, 0, -1},
                                                 {1, 0, 1, -1},
                                                 {0, 1, 0, -1},
                                                 {0, 1, 1, -1}};
    for (size_t root = 0; root < expected.size(); ++root)
        for (size_t row = 0; row < 4; ++row)
            EXPECT_EQ(BoolAt(output[root], row), expected[root][row]);
}

TEST(ExprCompilerTest, ColumnPrefixComparisonAcceptsStringRepresentations) {
    CompileOptions options;
    options.upstream_schema =
        Schema({"a", "b"}, {DataType::VARCHAR, DataType::STRING});
    options.upstream_columns = {{100, 0}, {101, 1}};
    auto logical =
        std::make_shared<expr::CompareExpr>(FieldId(100),
                                            FieldId(101),
                                            DataType::VARCHAR,
                                            DataType::STRING,
                                            proto::plan::PrefixMatch);
    auto compiled = CompileExpressions({logical}, options);
    ASSERT_TRUE(compiled.supported());
    auto a = std::make_shared<ColumnVector>(DataType::VARCHAR, 3);
    auto b = std::make_shared<ColumnVector>(DataType::STRING, 3);
    for (size_t i = 0; i < 3; ++i) {
        a->SetValueAt<std::string>(i, i == 0 ? "abc" : "zzz");
        b->SetValueAt<std::string>(i, "ab");
    }
    b->nullAt(2);
    RowVector upstream(options.upstream_schema, 3, {a, b});
    auto prepared = Prepare(compiled.plan);
    auto batch = prepared.MakeBatch(ReadBatchBinding::Range(0, 3), &upstream);
    auto output = Eval(prepared, *batch, SelectionView::All(3));
    EXPECT_EQ(BoolAt(output[0], 0), 1);
    EXPECT_EQ(BoolAt(output[0], 1), 0);
    EXPECT_EQ(BoolAt(output[0], 2), -1);
}

TEST(ExprCompilerTest,
     ExistingBinaryRangeCompareAndNotCompileWithNullSemantics) {
    CompileOptions options;
    options.upstream_schema =
        Schema({"a", "b"}, {DataType::INT64, DataType::INT64});
    options.upstream_columns = {{100, 0}, {101, 1}};
    auto range = std::make_shared<expr::BinaryRangeFilterExpr>(
        Column(100), Int(2), Int(4), true, false);
    auto comparison =
        std::make_shared<expr::CompareExpr>(FieldId(100),
                                            FieldId(101),
                                            DataType::INT64,
                                            DataType::INT64,
                                            proto::plan::LessThan);
    auto not_expr = std::make_shared<expr::LogicalUnaryExpr>(
        expr::LogicalUnaryExpr::OpType::LogicalNot, range);
    auto compiled = CompileExpressions({range, comparison, not_expr}, options);
    ASSERT_TRUE(compiled.supported());
    ASSERT_EQ(compiled.plan->inputs().slots().size(), 2);
    auto a = MakeWritableCopy(Integers({1, 2, 3, 4}));
    a.values->nullAt(2);
    RowVector upstream(options.upstream_schema,
                       4,
                       {a.values, Integers({2, 2, 5, 3}).AsVector()});
    auto prepared = Prepare(compiled.plan);
    auto batch = prepared.MakeBatch(ReadBatchBinding::Range(0, 4), &upstream);
    auto out = Eval(prepared, *batch, SelectionView::All(4));
    const std::vector<int> expected_range{0, 1, -1, 0},
        expected_compare{1, 0, -1, 0};
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(BoolAt(out[0], i), expected_range[i]);
        EXPECT_EQ(BoolAt(out[1], i), expected_compare[i]);
        EXPECT_EQ(BoolAt(out[2], i),
                  expected_range[i] < 0 ? -1 : !expected_range[i]);
    }
}

TEST(ExprCompilerTest,
     UpstreamSchemaMissingAndTypeMismatchNeverFallBackToStorage) {
    CompileOptions options;
    options.upstream_schema = Schema({"a"}, {DataType::INT64});
    options.upstream_columns = {{100, 0}};
    auto compiled = CompileExpressions({Greater(100, 1)}, options);
    int binds = 0;
    auto prepared = Prepare(compiled.plan, [&](const auto&) {
        ++binds;
        return NumericSource(std::make_shared<SourceState>());
    });
    EXPECT_EQ(binds, 0);
    EXPECT_THROW(prepared.MakeBatch(ReadBatchBinding::Range(0, 2)),
                 SegcoreError);
    RowVector renamed(
        Schema({"other"}, {DataType::INT64}), 2, {Integers({1, 2}).AsVector()});
    EXPECT_THROW(prepared.MakeBatch(ReadBatchBinding::Range(0, 2), &renamed),
                 SegcoreError);
    RowVector wrong_size(
        options.upstream_schema, 1, {Integers({1}).AsVector()});
    EXPECT_THROW(prepared.MakeBatch(ReadBatchBinding::Range(0, 2), &wrong_size),
                 SegcoreError);
    options.upstream_columns[100] = 1;
    EXPECT_THROW(CompileExpressions({Greater(100, 1)}, options), SegcoreError);
    options.upstream_columns[100] = 0;
    options.upstream_schema = Schema({"a"}, {DataType::DOUBLE});
    EXPECT_THROW(CompileExpressions({Greater(100, 1)}, options), SegcoreError);
    EXPECT_EQ(binds, 0);
}

TEST(ExprCompilerTest,
     MixedUpstreamAndStoredInputsAlignWithoutReadingNullRowPayload) {
    CompileOptions options;
    options.upstream_schema = Schema({"a"}, {DataType::INT64});
    options.upstream_columns = {{100, 0}};
    auto compiled =
        CompileExpressions({And(Greater(100, 4), Greater(101, 4))}, options);
    auto state = std::make_shared<SourceState>();
    auto prepared =
        Prepare(compiled.plan, [&](const InputDependency& dependency) {
            EXPECT_EQ(dependency.field_id, 101);
            return NumericSource(state);
        });
    RowVector upstream(
        options.upstream_schema, 3, {Integers({1, 8, 10}).AsVector()});
    upstream.SetRowValid(2, false);
    auto batch =
        prepared.MakeBatch(ReadBatchBinding::Offsets({0, 9, 10}), &upstream);
    auto out = Eval(prepared, *batch, SelectionView::All(3));
    EXPECT_EQ(BoolAt(out[0], 0), 0);
    EXPECT_EQ(BoolAt(out[0], 1), 1);
    EXPECT_EQ(BoolAt(out[0], 2), -1);
    EXPECT_EQ(state->positions[0], (std::vector<int64_t>{9, 10}));
}

TEST(ExprCompilerTest, ExistingFunctionCallCompilesThroughLegacyAdapter) {
    CompileOptions options;
    options.upstream_schema = Schema({"name"}, {DataType::VARCHAR});
    options.upstream_columns = {{100, 0}};
    auto call = std::make_shared<expr::CallExpr>(
        "starts_with",
        std::vector<expr::TypedExprPtr>{
            std::make_shared<expr::ColumnExpr>(Column(100, DataType::VARCHAR)),
            std::make_shared<expr::ValueExpr>(Text("a"))},
        expression::function::StartsWithVarchar);
    auto compiled = CompileExpressions({call}, options);
    ASSERT_TRUE(compiled.supported());
    EXPECT_FALSE(compiled.plan->whole_segment_safe());
    auto strings = std::make_shared<ColumnVector>(DataType::VARCHAR, 3);
    strings->SetValueAt<std::string>(0, "apple");
    strings->SetValueAt<std::string>(1, "banana");
    strings->nullAt(2);
    RowVector upstream(options.upstream_schema, 3, {strings});
    auto prepared = Prepare(compiled.plan);
    auto batch = prepared.MakeBatch(ReadBatchBinding::Range(0, 3), &upstream);
    auto out = Eval(prepared, *batch, SelectionView::All(3));
    EXPECT_EQ(BoolAt(out[0], 0), 1);
    EXPECT_EQ(BoolAt(out[0], 1), 0);
    EXPECT_EQ(BoolAt(out[0], 2), -1);
}

TEST(ExprCompilerTest, UnsupportedChildRejectsWholeSetBeforePreparation) {
    auto unsupported = std::make_shared<expr::TermFilterExpr>(
        Column(100), std::vector{Int(1)}, true);
    auto root = And(Greater(101, 0), unsupported);
    auto compiled =
        CompileExpressions({std::make_shared<expr::AlwaysTrueExpr>(), root});
    EXPECT_FALSE(compiled.supported());
    ASSERT_EQ(compiled.unsupported.size(), 1);
    EXPECT_EQ(compiled.unsupported[0].feature,
              "TermFilterExpr complex/in-field");
    EXPECT_EQ(compiled.unsupported[0].location, "root[1]/child[1]");
    int binds = 0;
    EXPECT_THROW(
        Prepare(compiled.plan,
                [&](const auto&) {
                    ++binds;
                    return NumericSource(std::make_shared<SourceState>());
                }),
        SegcoreError);
    EXPECT_EQ(binds, 0);
}

TEST(ExprCompilerTest, AllUnmigratedLegacyNodeFamiliesAreReportedExplicitly) {
    std::vector<expr::TypedExprPtr> roots{
        std::make_shared<expr::TermFilterExpr>(
            Column(100), std::vector{Int(1)}, true),
        std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
            Column(100),
            proto::plan::Equal,
            static_cast<proto::plan::ArithOpType>(0),
            Int(1),
            Int(2)),
        std::make_shared<expr::TimestamptzArithCompareExpr>(
            Column(100, DataType::TIMESTAMPTZ),
            proto::plan::Mod,
            proto::plan::Interval{},
            proto::plan::Equal,
            Int(1)),
        std::make_shared<expr::ExistsExpr>(Column(100, DataType::ARRAY)),
        std::make_shared<expr::JsonContainsExpr>(Column(100, DataType::JSON),
                                                 static_cast<ContainsType>(0),
                                                 true,
                                                 std::vector{Int(1)}),
        std::make_shared<expr::GISFunctionFilterExpr>(
            Column(100, DataType::GEOMETRY),
            static_cast<GISFunctionType>(0),
            "POINT(0 0)"),
        std::make_shared<expr::MatchExpr>(
            "items",
            static_cast<expr::MatchType>(0),
            1,
            std::make_shared<expr::AlwaysTrueExpr>()),
        std::make_shared<expr::BloomFilterExpr>(Column(100, DataType::DOUBLE),
                                                std::string{}),
        std::make_shared<expr::RoaringFilterExpr>(Column(100, DataType::DOUBLE),
                                                  nullptr),
        std::make_shared<expr::FieldAccessTypeExpr>(DataType::INT64,
                                                    FieldId(100)),
        std::make_shared<expr::InputTypeExpr>(DataType::ROW)};
    auto compiled = CompileExpressions(roots);
    EXPECT_FALSE(compiled.supported());
    ASSERT_EQ(compiled.unsupported.size(), roots.size());
    const std::vector<std::string> names{"TermFilterExpr complex/in-field",
                                         "BinaryArithOpEvalRangeExpr",
                                         "TimestamptzArithCompareExpr",
                                         "ExistsExpr non-JSON/element",
                                         "JsonContainsExpr",
                                         "GISFunctionFilterExpr",
                                         "MatchExpr",
                                         "BloomFilterExpr input type",
                                         "RoaringFilterExpr non-integer",
                                         "FieldAccessTypeExpr",
                                         "InputTypeExpr"};
    for (size_t i = 0; i < roots.size(); ++i)
        EXPECT_EQ(compiled.unsupported[i].feature, names[i]);
}

TEST(ExprCompilerTest, UnsupportedSubfeaturesCannotSilentlyChangeSemantics) {
    auto path = Column(100, DataType::INT64);
    path.nested_path_ = {"x"};
    auto element = Column(101, DataType::INT64);
    element.element_level_ = true;
    std::vector<expr::TypedExprPtr> roots{
        std::make_shared<expr::ColumnExpr>(path),
        std::make_shared<expr::ColumnExpr>(element),
        std::make_shared<expr::UnaryRangeFilterExpr>(
            Column(102, DataType::VARCHAR), proto::plan::TextMatch, Text("a")),
        std::make_shared<expr::CompareExpr>(FieldId(104),
                                            FieldId(105),
                                            DataType::INT64,
                                            DataType::VARCHAR,
                                            proto::plan::Equal)};
    auto compiled = CompileExpressions(roots);
    EXPECT_FALSE(compiled.supported());
    EXPECT_EQ(compiled.unsupported.size(), roots.size());
}

TEST(ExprCompilerTest, UntypedNullLiteralProjectsAndPassesToLegacyFunction) {
    auto null = std::make_shared<expr::ValueExpr>(proto::plan::GenericValue{});
    auto call = std::make_shared<expr::CallExpr>(
        "null_argument",
        std::vector<expr::TypedExprPtr>{null},
        +[](const RowVector& args, VectorPtr& output) {
            EXPECT_EQ(args.size(), 2);
            const auto* value =
                dynamic_cast<const ConstantVector<bool>*>(args.child(0).get());
            ASSERT_NE(value, nullptr);
            EXPECT_EQ(value->type(), DataType::NONE);
            EXPECT_EQ(value->size(), args.size());
            EXPECT_TRUE(value->IsNull());
            output = std::make_shared<ConstantVector<bool>>(
                DataType::BOOL, args.size(), true);
        });
    auto compiled = CompileExpressions({null, call});
    ASSERT_TRUE(compiled.supported());
    auto prepared = Prepare(compiled.plan);
    auto batch = prepared.MakeBatch(ReadBatchBinding::Offsets({7, 2, 7, 1}));
    auto rows = OwnedSelection::FromIndices(4, {1, 3});
    auto results = Eval(prepared, *batch, rows.view());
    ASSERT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].logical_type(), DataType::NONE);
    EXPECT_EQ(results[0].size(), 4);
    for (size_t row : {1, 3}) {
        EXPECT_FALSE(results[0].IsValid(row));
        EXPECT_EQ(BoolAt(results[1], row), 1);
    }
    EXPECT_THROW(results[0].IsValid(0), SegcoreError);
    EXPECT_THROW(results[1].IsValid(2), SegcoreError);
    auto published = ReadOnlyVectorView::CopyOf(results[0].AsVector());
    EXPECT_FALSE(published.IsValid(3));
    auto empty = prepared.MakeBatch(ReadBatchBinding::Range(0, 0));
    EXPECT_EQ(Eval(prepared, *empty, SelectionView::All(0))[0].size(), 0);
}

TEST(ExprCompilerTest, FilterTTLIsExplicitAndProjectionDoesNotInjectIt) {
    auto value = std::make_shared<expr::ValueExpr>(Int(42));
    auto projection = CompileExpressions({value});
    ASSERT_TRUE(projection.supported());
    EXPECT_TRUE(projection.plan->inputs().slots().empty());
    auto prepared_projection = Prepare(projection.plan);
    auto constants =
        prepared_projection.MakeBatch(ReadBatchBinding::Range(0, 3));
    EXPECT_EQ(Eval(prepared_projection, *constants, SelectionView::All(3))[0]
                  .ValueAt<int64_t>(2),
              42);

    auto null_ttl = std::make_shared<expr::NullExpr>(
        Column(101, DataType::TIMESTAMPTZ, true),
        proto::plan::NullExpr_NullOp_IsNull);
    auto future_ttl = std::make_shared<expr::UnaryRangeFilterExpr>(
        Column(101, DataType::TIMESTAMPTZ, true),
        proto::plan::GreaterThan,
        Int(100));
    auto ttl = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::Or, null_ttl, future_ttl);
    CompileOptions options;
    options.upstream_schema =
        Schema({"a", "ttl"}, {DataType::INT64, DataType::TIMESTAMPTZ});
    options.upstream_columns = {{100, 0}, {101, 1}};
    auto filtered = CompileFilter(Greater(100, 1), ttl, options);
    ASSERT_TRUE(filtered.supported());
    ASSERT_EQ(filtered.plan->inputs().slots().size(), 2);
    auto ts = std::make_shared<ColumnVector>(DataType::TIMESTAMPTZ, 4);
    ts->SetValueAt<int64_t>(0, 200);
    ts->SetValueAt<int64_t>(1, 100);
    ts->SetValueAt<int64_t>(2, 101);
    ts->nullAt(3);
    RowVector upstream(
        options.upstream_schema, 4, {Integers({0, 2, 2, 2}).AsVector(), ts});
    auto prepared = Prepare(filtered.plan);
    auto batch = prepared.MakeBatch(ReadBatchBinding::Range(0, 4), &upstream);
    auto out = Eval(prepared, *batch, SelectionView::All(4));
    for (size_t i = 0; i < 4; ++i) EXPECT_EQ(BoolAt(out[0], i), i >= 2);
    EXPECT_THROW(CompileFilter(value), SegcoreError);
}

TEST(ExprCompilerTest, FactoryAndLoadErrorsKeepCodeAndDoNotPublishCoverage) {
    auto compiled = CompileExpressions({Greater(100, 0)});
    auto state = std::make_shared<SourceState>();
    state->fail_factory = true;
    auto prepared = Prepare(compiled.plan,
                            [&](const auto&) { return NumericSource(state); });
    auto batch = prepared.MakeBatch(ReadBatchBinding::Range(1, 3));
    auto run_failure = [&] {
        try {
            Eval(prepared, *batch, SelectionView::All(3));
            FAIL();
        } catch (const SegcoreError& error) {
            EXPECT_EQ(error.get_error_code(), FileReadFailed);
        }
    };
    run_failure();
    EXPECT_EQ(state->loads, 0);
    state->fail_load = true;
    run_failure();
    EXPECT_EQ(state->factories, 2);
    auto out = Eval(prepared, *batch, SelectionView::All(3));
    EXPECT_EQ(BoolAt(out[0], 2), 1);
    EXPECT_EQ(state->factories, 2);
    EXPECT_EQ(state->loads, 2);
}

TEST(ExprCompilerTest, BinderErrorsAndBrokenFactoryContractsAreNotFallback) {
    auto compiled = CompileExpressions({Greater(100, 0)});
    try {
        Prepare(compiled.plan, [](const auto&) -> PreparedInput {
            ThrowInfo(InsufficientResource, "injected binder failure");
        });
        FAIL();
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), InsufficientResource);
    }
    auto owner = std::make_shared<int>(0);
    auto prepared = Prepare(compiled.plan, [&](const auto&) {
        return PreparedInput{
            DataType::INT64, true, false, owner, [owner](const auto& binding) {
                if (++*owner == 1)
                    return InputLoaders{};
                return InputLoaders{
                    [size = binding.size()](SelectionView rows) {
                        return Integers(std::vector<int64_t>(size, 9))
                            .Restrict(rows);
                    },
                    {}};
            }};
    });
    auto batch = prepared.MakeBatch(ReadBatchBinding::Range(0, 2));
    EXPECT_THROW(Eval(prepared, *batch, SelectionView::All(2)), SegcoreError);
    auto out = Eval(prepared, *batch, SelectionView::All(2));
    EXPECT_EQ(BoolAt(out[0], 0), 1);
    EXPECT_EQ(*owner, 2);
    EXPECT_THROW(Prepare(compiled.plan), SegcoreError);
}

TEST(ExprCompilerTest, SourceIdentityDoesNotUseIncompleteColumnInfoEquality) {
    InputLayout layout;
    InputDependency a{"relation", 100, DataType::INT64};
    a.needs_validity = true;
    EXPECT_EQ(layout.Intern(a), 0);
    a.needs_values = true;
    a.nullable = true;
    EXPECT_EQ(layout.Intern(a), 0);
    EXPECT_TRUE(layout.slots()[0].needs_values);
    EXPECT_TRUE(layout.slots()[0].nullable);
    auto other = a;
    other.path = {"x"};
    EXPECT_EQ(layout.Intern(other), 1);
    other.path = {"x", "y"};
    EXPECT_EQ(layout.Intern(other), 2);
    other = a;
    other.relation = "other";
    EXPECT_EQ(layout.Intern(other), 3);
    other = a;
    other.upstream_column = 0;
    EXPECT_EQ(layout.Intern(other), 4);
    other = a;
    other.type = DataType::DOUBLE;
    EXPECT_EQ(layout.Intern(other), 5);
    other = a;
    other.element_level = true;
    EXPECT_EQ(layout.Intern(other), 6);
}

TEST(ExprCompilerTest, TypedBoundsPreserveLegacyScalarExtraction) {
    auto check = []<typename T>(
                     DataType type, T value, proto::plan::GenericValue bound) {
        CompileOptions options;
        options.upstream_schema = Schema({"value"}, {type});
        options.upstream_columns = {{100, 0}};
        auto node = std::make_shared<expr::UnaryRangeFilterExpr>(
            Column(100, type), proto::plan::Equal, bound);
        auto compiled = CompileExpressions({node}, options);
        ASSERT_TRUE(compiled.supported());
        RowVector upstream(
            options.upstream_schema,
            2,
            {std::make_shared<ConstantVector<T>>(type, 2, value)});
        auto prepared = Prepare(compiled.plan);
        auto batch =
            prepared.MakeBatch(ReadBatchBinding::Range(0, 2), &upstream);
        auto result = Eval(prepared, *batch, SelectionView::All(2));
        EXPECT_EQ(BoolAt(result[0], 0), 1);
        EXPECT_EQ(BoolAt(result[0], 1), 1);
    };
    check(DataType::INT8, int8_t{-128}, Int(-128));
    check(DataType::INT16, int16_t{32767}, Int(32767));
    check(DataType::INT32, int32_t{-2147483647}, Int(-2147483647));
    check(DataType::INT64,
          std::numeric_limits<int64_t>::min(),
          Int(std::numeric_limits<int64_t>::min()));
    proto::plan::GenericValue decimal;
    decimal.set_float_val(0.1);
    check(DataType::FLOAT, static_cast<float>(0.1), decimal);
    check(DataType::DOUBLE, 0.1, decimal);
    proto::plan::GenericValue boolean;
    boolean.set_bool_val(true);
    check(DataType::BOOL, true, boolean);
    for (auto type : {DataType::STRING, DataType::VARCHAR, DataType::TEXT}) {
        check(type, std::string("北京"), Text("北京"));
    }
}

TEST(ExprCompilerTest, NonfiniteLiteralsAndExcessiveDepthRejectWholePlan) {
    for (double value : {std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::max()}) {
        proto::plan::GenericValue bound;
        bound.set_float_val(value);
        auto node = std::make_shared<expr::UnaryRangeFilterExpr>(
            Column(100, DataType::FLOAT), proto::plan::GreaterThan, bound);
        auto compiled = CompileExpressions({Greater(101, 0), node});
        EXPECT_FALSE(compiled.supported());
        ASSERT_EQ(compiled.unsupported.size(), 1);
        EXPECT_EQ(compiled.unsupported[0].location, "root[1]");
    }
    expr::TypedExprPtr deep = Greater(100, 0);
    for (size_t i = 0; i < 260; ++i) {
        deep = std::make_shared<expr::LogicalUnaryExpr>(
            expr::LogicalUnaryExpr::OpType::LogicalNot, deep);
    }
    auto compiled = CompileExpressions({deep});
    EXPECT_FALSE(compiled.supported());
    ASSERT_EQ(compiled.unsupported.size(), 1);
    EXPECT_EQ(compiled.unsupported[0].feature,
              "scalar compilation depth limit");
}

TEST(ExprCompilerTest, PartialUpstreamDoesNotBecomeImplicitRootSelection) {
    CompileOptions options;
    options.upstream_schema = Schema({"a"}, {DataType::INT64});
    options.upstream_columns = {{100, 0}};
    auto compiled =
        CompileExpressions({Greater(100, 5), Greater(101, 0)}, options);
    ASSERT_TRUE(compiled.supported());
    auto rows = OwnedSelection::FromIndices(4, {1, 3});
    RowVector upstream(
        options.upstream_schema,
        4,
        {Integers({0, 7, 0, 9}).Restrict(rows.view()).AsVector()});
    auto source = std::make_shared<SourceState>();
    auto prepared = Prepare(compiled.plan,
                            [&](const auto&) { return NumericSource(source); });
    auto batch = prepared.MakeBatch(
        ReadBatchBinding::Offsets({4, 3, 2, 1}), &upstream, rows.view());
    EXPECT_EQ(source->factories, 0);
    scalar::EvalCtx ctx(*batch);
    auto output = compiled.plan->expressions().Eval(
        std::vector<SelectionView>{rows.view(), SelectionView::All(4)}, ctx);
    EXPECT_EQ(BoolAt(output[0], 1), 1);
    EXPECT_EQ(BoolAt(output[0], 3), 1);
    EXPECT_THROW(output[0].IsValid(0), SegcoreError);
    EXPECT_EQ(source->positions,
              (std::vector<std::vector<int64_t>>{{4, 3, 2, 1}}));
    EXPECT_THROW(Eval(prepared, *batch, SelectionView::All(4)), SegcoreError);
    EXPECT_EQ(source->loads,
              1);  // missing upstream is not fetched from storage
}

TEST(ExprCompilerTest,
     PartialProjectionFeedsNextCompiledBatchWithoutCompaction) {
    auto first =
        CompileExpressions({std::make_shared<expr::ColumnExpr>(Column(100))});
    auto source = std::make_shared<SourceState>();
    auto prepared_first =
        Prepare(first.plan, [&](const auto&) { return NumericSource(source); });
    auto batch_first = prepared_first.MakeBatch(ReadBatchBinding::Range(10, 4));
    auto rows = OwnedSelection::FromIndices(4, {1, 3});
    auto values = Eval(prepared_first, *batch_first, rows.view());
    CompileOptions options;
    options.upstream_schema = Schema({"projection"}, {DataType::INT64});
    options.upstream_columns = {{200, 0}};
    auto next = CompileExpressions({Greater(200, 11)}, options);
    RowVector upstream(options.upstream_schema, 4, {values[0].AsVector()});
    auto prepared_next = Prepare(next.plan);
    auto batch_next = prepared_next.MakeBatch(
        ReadBatchBinding::Range(10, 4), &upstream, rows.view());
    batch_first.reset();
    values.clear();
    rows.ResetAll(4);  // the snapshot's coverage must not expand with this mask
    auto selected = OwnedSelection::FromIndices(4, {1, 3});
    auto result = Eval(prepared_next, *batch_next, selected.view());
    EXPECT_EQ(BoolAt(result[0], 1), 0);
    EXPECT_EQ(BoolAt(result[0], 3), 1);
    EXPECT_EQ(source->loads, 1);
    EXPECT_THROW(Eval(prepared_next, *batch_next, SelectionView::All(4)),
                 SegcoreError);
}

TEST(ExprCompilerTest, InvalidPartialUpstreamFailsBeforeAnyStoredFactoryRuns) {
    CompileOptions options;
    options.upstream_schema = Schema({"a"}, {DataType::INT64});
    options.upstream_columns = {{100, 0}};
    auto compiled =
        CompileExpressions({Greater(100, 0), Greater(101, 0)}, options);
    auto source = std::make_shared<SourceState>();
    auto prepared = Prepare(compiled.plan,
                            [&](const auto&) { return NumericSource(source); });
    RowVector upstream(
        options.upstream_schema, 2, {Integers({1, 2}).AsVector()});
    EXPECT_THROW(
        prepared.MakeBatch(
            ReadBatchBinding::Range(0, 2), nullptr, SelectionView::All(2)),
        SegcoreError);
    EXPECT_THROW(
        prepared.MakeBatch(
            ReadBatchBinding::Range(0, 2), &upstream, SelectionView::All(3)),
        SegcoreError);
    EXPECT_EQ(source->factories, 0);
    EXPECT_EQ(source->loads, 0);
}

TEST(ExprCompilerTest, InvalidBindingsAndZeroColumnBatches) {
    EXPECT_THROW(ReadBatchBinding::Range(-1, 3), SegcoreError);
    EXPECT_THROW(
        ReadBatchBinding::Range(std::numeric_limits<int64_t>::max(), 1),
        SegcoreError);
    EXPECT_THROW(ReadBatchBinding::Offsets({1, -1}), SegcoreError);
    EXPECT_THROW(ReadBatchBinding::Range(0, 0).offset(0), SegcoreError);
    auto compiled = CompileExpressions({});
    ASSERT_TRUE(compiled.supported());
    auto prepared = Prepare(compiled.plan);
    auto batch = prepared.MakeBatch(ReadBatchBinding::Range(0, 5));
    EXPECT_EQ(batch->size(), 5);
    EXPECT_TRUE(Eval(prepared, *batch, SelectionView::All(5)).empty());
    EXPECT_THROW(CompileExpressions({nullptr}), SegcoreError);
}

}  // namespace milvus::exec::plan
