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
#include <google/protobuf/text_format.h>
#include <limits>

#include "exec/operator/ScalarProjectNode.h"
#include "exec/operator/QueryOrderByNode.h"
#include "exec/plan/ExecutionPreparer.h"
#include "exec/plan/ExprCompiler.h"
#include "exec/plan/ScalarProject.h"
#include "query/PlanProto.h"

namespace milvus::exec::plan {
namespace {
expr::TypedExprPtr
Literal(int64_t n) {
    proto::plan::GenericValue value;
    value.set_int64_val(n);
    return std::make_shared<expr::ValueExpr>(value);
}

expr::TypedExprPtr
Column(int64_t id, DataType type = DataType::INT64) {
    return std::make_shared<expr::ColumnExpr>(
        expr::ColumnInfo(FieldId(id), type));
}

expr::TypedExprPtr
Arithmetic(expr::TypedExprPtr left,
           proto::plan::ArithOpType op,
           expr::TypedExprPtr right) {
    return std::make_shared<expr::BinaryArithExpr>(
        op, std::move(left), std::move(right));
}

RowTypePtr
InputType() {
    return std::make_shared<RowType>(
        std::vector<std::string>{"a", "b"},
        std::vector<DataType>{DataType::INT64, DataType::INT64});
}

RowVectorPtr
Input(std::vector<int64_t> a, std::vector<int64_t> b) {
    std::vector<VectorPtr> children;
    for (const auto& values : {a, b}) {
        auto column =
            std::make_shared<ColumnVector>(DataType::INT64, values.size());
        for (size_t i = 0; i < values.size(); ++i)
            column->SetValueAt<int64_t>(i, values[i]);
        children.push_back(column);
    }
    return std::make_shared<RowVector>(
        InputType(), a.size(), std::move(children));
}

ScalarProject
Project(std::vector<expr::TypedExprPtr> roots,
        std::vector<std::string> names,
        bool upstream = true) {
    CompileOptions options;
    if (upstream) {
        options.upstream_schema = InputType();
        options.upstream_columns = {{100, 0}, {101, 1}};
    }
    auto compiled = CompileExpressions(roots, options);
    EXPECT_TRUE(compiled.supported());
    return ScalarProject(Prepare(compiled.plan), std::move(names));
}

ReadOnlyVectorView
View(const RowVectorPtr& rows, size_t column = 0) {
    return ReadOnlyVectorView::CopyOf(rows->child(column));
}
}  // namespace

TEST(ScalarProjectTest,
     ArithmeticAndBooleanOutputsKeepTheirTypesAcrossBatches) {
    proto::plan::GenericValue bound;
    bound.set_int64_val(10);
    auto condition = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(FieldId(100), DataType::INT64),
        proto::plan::GreaterThan,
        bound);
    auto project =
        Project({condition,
                 Column(100),
                 Arithmetic(Column(100), proto::plan::Add, Literal(1)),
                 Literal(42)},
                {"matched", "value", "adjusted", "constant"});
    auto input = Input({3, 11, -4}, {1, 2, 3});
    auto first = project.Eval(
        ReadBatchBinding::Range(0, 3), SelectionView::All(3), input.get());
    EXPECT_EQ(first->row_type()->column_type(0), DataType::BOOL);
    EXPECT_EQ(first->row_type()->column_name(2), "adjusted");
    EXPECT_FALSE(View(first).ValueAt<bool>(0));
    EXPECT_TRUE(View(first).ValueAt<bool>(1));
    EXPECT_EQ(View(first, 2).ValueAt<int64_t>(2), -3);
    auto flat = MaterializeScalarColumn(View(first));
    EXPECT_FALSE(flat->IsBitmap());
    EXPECT_EQ(flat->type(), DataType::BOOL);
    input = Input({90}, {4});
    auto second = project.Eval(
        ReadBatchBinding::Range(0, 1), SelectionView::All(1), input.get());
    EXPECT_EQ(View(second, 2).ValueAt<int64_t>(0), 91);
    EXPECT_EQ(View(first, 2).ValueAt<int64_t>(0), 4);
}

TEST(ScalarProjectTest,
     SelectionSkipsInvalidDivisionAndFailurePublishesNothing) {
    auto project = Project(
        {Literal(42), Arithmetic(Column(100), proto::plan::Div, Column(101))},
        {"first", "quotient"});
    auto input = Input({10, 20, 30, 40}, {2, 0, 5, 0});
    const auto selected = OwnedSelection::FromIndices(4, {0, 2});
    auto output = project.Eval(
        ReadBatchBinding::Range(0, 4), selected.view(), input.get());
    ASSERT_EQ(output->size(), 2);
    EXPECT_EQ(View(output, 1).ValueAt<int64_t>(0), 5);
    EXPECT_EQ(View(output, 1).ValueAt<int64_t>(1), 6);
    RowVectorPtr failed;
    try {
        failed = project.Eval(
            ReadBatchBinding::Range(0, 4), SelectionView::All(4), input.get());
        FAIL() << "expected division failure";
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), ErrorCode::ExprInvalid);
    }
    EXPECT_EQ(failed, nullptr);
    auto empty = project.Eval(
        ReadBatchBinding::Range(0, 4), SelectionView::None(4), input.get());
    EXPECT_EQ(empty->size(), 0);
    EXPECT_EQ(empty->row_type()->column_count(), 2);
    EXPECT_EQ(View(output, 1).ValueAt<int64_t>(0), 5);
}

TEST(ScalarProjectTest, NumericPromotionWrappingAndDivisionOverflow) {
    const auto max = std::numeric_limits<int64_t>::max();
    const auto min = std::numeric_limits<int64_t>::min();
    auto project =
        Project({Arithmetic(Literal(max), proto::plan::Add, Literal(1)),
                 Arithmetic(Literal(min), proto::plan::Sub, Literal(1)),
                 Arithmetic(Literal(max), proto::plan::Mul, Literal(2)),
                 Arithmetic(Literal(-7), proto::plan::Div, Literal(2))},
                {"add", "sub", "mul", "div"},
                false);
    auto output =
        project.Eval(ReadBatchBinding::Range(0, 1), SelectionView::All(1));
    EXPECT_EQ(View(output, 0).ValueAt<int64_t>(0), min);
    EXPECT_EQ(View(output, 1).ValueAt<int64_t>(0), max);
    EXPECT_EQ(View(output, 2).ValueAt<int64_t>(0), -2);
    EXPECT_EQ(View(output, 3).ValueAt<int64_t>(0), -3);
    auto invalid =
        Project({Arithmetic(Literal(min), proto::plan::Div, Literal(-1))},
                {"invalid"},
                false);
    EXPECT_THROW(
        invalid.Eval(ReadBatchBinding::Range(0, 1), SelectionView::All(1)),
        SegcoreError);
    EXPECT_EQ(
        invalid.Eval(ReadBatchBinding::Range(0, 1), SelectionView::None(1))
            ->size(),
        0);
    auto unsupported = CompileExpressions(
        {Arithmetic(Literal(4), proto::plan::Mod, Literal(2))});
    EXPECT_FALSE(unsupported.supported());
}

TEST(ScalarProjectTest, MixedFieldArithmeticPromotesToDouble) {
    auto type = std::make_shared<RowType>(
        std::vector<std::string>{"small", "fraction"},
        std::vector<DataType>{DataType::INT8, DataType::FLOAT});
    auto small = std::make_shared<ColumnVector>(DataType::INT8, 2);
    small->SetValueAt<int8_t>(0, 7);
    small->SetValueAt<int8_t>(1, -9);
    auto fraction = std::make_shared<ColumnVector>(DataType::FLOAT, 2);
    fraction->SetValueAt<float>(0, 1.5f);
    fraction->SetValueAt<float>(1, -2.25f);
    RowVector input(type, 2, std::vector<VectorPtr>{small, fraction});
    CompileOptions options;
    options.upstream_schema = type;
    options.upstream_columns = {{100, 0}, {101, 1}};
    std::vector<expr::TypedExprPtr> roots;
    for (auto op : {proto::plan::Add,
                    proto::plan::Sub,
                    proto::plan::Mul,
                    proto::plan::Div})
        roots.push_back(Arithmetic(
            Column(100, DataType::INT8), op, Column(101, DataType::FLOAT)));
    auto compiled = CompileExpressions(roots, options);
    ASSERT_TRUE(compiled.supported());
    ScalarProject project(Prepare(compiled.plan), {"add", "sub", "mul", "div"});
    auto output = project.Eval(
        ReadBatchBinding::Range(0, 2), SelectionView::All(2), &input);
    for (size_t i = 0; i < 4; ++i)
        EXPECT_EQ(output->row_type()->column_type(i), DataType::DOUBLE);
    EXPECT_DOUBLE_EQ(View(output, 0).ValueAt<double>(0), 8.5);
    EXPECT_DOUBLE_EQ(View(output, 1).ValueAt<double>(1), -6.75);
    EXPECT_DOUBLE_EQ(View(output, 2).ValueAt<double>(1), 20.25);
    EXPECT_DOUBLE_EQ(View(output, 3).ValueAt<double>(1), 4.0);
}

TEST(ScalarProjectTest,
     ProtoScalarEntryRetainsDoublePrecisionAndFilterTypeGuard) {
    proto::plan::Expr wire;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
        R"(binary_arith_expr {op: Add left {value_expr {value {float_val: 1.0000000001}}} right {value_expr {value {int64_val: 0}}}})",
        &wire));
    query::ProtoParser parser(std::make_shared<milvus::Schema>());
    auto root = parser.ParseScalarExprs(wire);
    ASSERT_EQ(root->type(), DataType::DOUBLE);
    auto project = Project({root}, {"precise"}, false);
    auto output =
        project.Eval(ReadBatchBinding::Range(0, 1), SelectionView::All(1));
    EXPECT_DOUBLE_EQ(View(output).ValueAt<double>(0), 1.0000000001);
    EXPECT_THROW(parser.ParseExprs(wire), SegcoreError);
    proto::plan::Expr literal;
    literal.mutable_value_expr()->mutable_value()->set_float_val(1.0000000001);
    EXPECT_EQ(parser.ParseScalarExprs(literal)->type(), DataType::DOUBLE);
    // Legacy scalar arguments retain their old declared FLOAT type.
    EXPECT_EQ(parser.ParseExprs(literal, query::ProtoParser::TypeIsAny)->type(),
              DataType::FLOAT);
}

TEST(ScalarProjectTest, NullColumnsRowsAndCancellation) {
    auto input = Input({10, 20}, {0, 2});
    std::dynamic_pointer_cast<ColumnVector>(input->child(0))->nullAt(0);
    auto project = Project(
        {Arithmetic(Column(100), proto::plan::Div, Column(101))}, {"quotient"});
    auto output = project.Eval(
        ReadBatchBinding::Range(0, 2), SelectionView::All(2), input.get());
    EXPECT_FALSE(View(output).IsValid(0));
    EXPECT_EQ(View(output).ValueAt<int64_t>(1), 10);
    auto null_rows = std::make_shared<RowVector>(
        InputType(), 2, input->childrens(), TargetBitmap(2, false));
    auto constant_failure = Project(
        {Arithmetic(Literal(1), proto::plan::Div, Literal(0))}, {"constant"});
    output = constant_failure.Eval(
        ReadBatchBinding::Range(0, 2), SelectionView::All(2), null_rows.get());
    EXPECT_FALSE(output->IsValid(0));
    EXPECT_FALSE(View(output).IsValid(1));
    RowVectorPtr cancelled;
    EXPECT_THROW(
        cancelled = project.Eval(ReadBatchBinding::Range(0, 2),
                                 SelectionView::All(2),
                                 input.get(),
                                 [] { throw folly::FutureCancellation(); }),
        folly::FutureCancellation);
    EXPECT_EQ(cancelled, nullptr);
}

TEST(ScalarProjectTest, ReaderPreservesOffsetOrderAndReadsOncePerBatch) {
    auto compiled = CompileExpressions(
        {Column(100), Arithmetic(Column(100), proto::plan::Add, Literal(1))});
    ASSERT_TRUE(compiled.supported());
    auto requests = std::make_shared<std::vector<std::vector<int64_t>>>();
    auto prepared = Prepare(compiled.plan, [&](const auto&) {
        return PreparedInput{
            DataType::INT64,
            true,
            false,
            requests,
            [requests](const ReadBatchBinding& binding) {
                return InputLoaders{
                    [requests, binding](SelectionView rows) {
                        VectorResultBuilder values(DataType::INT64,
                                                   binding.size());
                        std::vector<int64_t> offsets;
                        rows.ForEachSelected([&](size_t row) {
                            offsets.push_back(binding.offset(row));
                            values.Set<int64_t>(row, binding.offset(row) * 10);
                        });
                        requests->push_back(offsets);
                        return std::move(values).Finish(rows);
                    },
                    {}};
            }};
    });
    ScalarProject project(std::move(prepared), {"value", "increment"});
    auto first = project.Eval(ReadBatchBinding::Offsets({7, 2, 7}),
                              SelectionView::All(3));
    EXPECT_EQ(*requests, (std::vector<std::vector<int64_t>>{{7, 2, 7}}));
    EXPECT_EQ(View(first, 1).ValueAt<int64_t>(1), 21);
    EXPECT_EQ(View(first, 1).ValueAt<int64_t>(2), 71);
    auto second =
        project.Eval(ReadBatchBinding::Offsets({9}), SelectionView::All(1));
    EXPECT_EQ(requests->size(), 2);
    EXPECT_EQ(View(first, 1).ValueAt<int64_t>(0), 71);
    EXPECT_EQ(View(second, 1).ValueAt<int64_t>(0), 91);
}

TEST(ScalarProjectTest,
     PhysicalOperatorDrainsMultipleBatchesAndHandlesEmptyInput) {
    auto source = std::make_shared<milvus::plan::ProjectNode>(
        "source",
        std::vector<FieldId>{FieldId(100), FieldId(101)},
        std::vector<std::string>{"a", "b"},
        std::vector<DataType>{DataType::INT64, DataType::INT64});
    auto logical = std::make_shared<milvus::plan::ScalarProjectNode>(
        "projection",
        std::vector<expr::TypedExprPtr>{
            Arithmetic(Column(100), proto::plan::Add, Literal(1))},
        std::vector<std::string>{"adjusted"},
        std::map<int64_t, column_index_t>{{100, 0}, {101, 1}},
        source);
    auto context =
        std::make_shared<QueryContext>("scalar-project",
                                       nullptr,
                                       0,
                                       MAX_TIMESTAMP,
                                       0,
                                       0,
                                       query::PlanOptions{false},
                                       std::make_shared<QueryConfig>());
    auto task = Task::Create(
        "scalar-project-task", milvus::plan::PlanFragment(logical), 0, context);
    DriverContext driver(task, 0, 0, 0, 0);
    PhyScalarProjectNode op(0, &driver, logical);
    EXPECT_TRUE(op.NeedInput());
    EXPECT_EQ(op.GetOutput(), nullptr);
    auto input = Input({1, 2}, {3, 4});
    op.AddInput(input);
    EXPECT_EQ(input, nullptr);
    EXPECT_FALSE(op.NeedInput());
    auto first = op.GetOutput();
    EXPECT_TRUE(op.NeedInput());
    input = Input({}, {});
    op.AddInput(input);
    EXPECT_EQ(op.GetOutput(), nullptr);
    input = Input({5}, {6});
    op.AddInput(input);
    op.NoMoreInput();
    EXPECT_FALSE(op.IsFinished());
    auto last = op.GetOutput();
    EXPECT_EQ(View(last).ValueAt<int64_t>(0), 6);
    EXPECT_EQ(View(first).ValueAt<int64_t>(0), 2);
    EXPECT_TRUE(op.IsFinished());
    EXPECT_FALSE(op.NeedInput());
    EXPECT_EQ(op.GetOutput(), nullptr);
    PhyScalarProjectNode empty(1, &driver, logical);
    empty.NoMoreInput();
    EXPECT_TRUE(empty.IsFinished());

    // Sort consumes immutable projected values across batches and orders the
    // complete input before applying its segment-local limit.
    auto order = std::make_shared<milvus::plan::OrderByNode>(
        "order",
        std::vector<expr::FieldAccessTypeExprPtr>{
            std::make_shared<expr::FieldAccessTypeExpr>(
                DataType::INT64, "adjusted", FieldId(100))},
        std::vector<milvus::plan::SortOrder>{{false, false}},
        2,
        std::vector<milvus::plan::PlanNodePtr>{logical});
    PhyQueryOrderByNode sort(2, &driver, order);
    sort.AddInput(first);
    sort.AddInput(last);
    sort.NoMoreInput();
    auto sorted = sort.GetOutput();
    ASSERT_EQ(sorted->size(), 2);
    EXPECT_EQ(View(sorted).ValueAt<int64_t>(0), 6);
    EXPECT_EQ(View(sorted).ValueAt<int64_t>(1), 3);
}

TEST(ScalarProjectTest, StringsAndConstantsMaterializeAfterOwnersAreReleased) {
    auto type =
        std::make_shared<RowType>(std::vector<std::string>{"text"},
                                  std::vector<DataType>{DataType::VARCHAR});
    CompileOptions options;
    options.upstream_schema = type;
    options.upstream_columns = {{100, 0}};
    proto::plan::GenericValue literal;
    literal.set_string_val("constant");
    auto compiled =
        CompileExpressions({Column(100, DataType::VARCHAR),
                            std::make_shared<expr::ValueExpr>(literal)},
                           options);
    ASSERT_TRUE(compiled.supported());
    ScalarProject project(Prepare(compiled.plan), {"text", "label"});
    auto column = std::make_shared<ColumnVector>(DataType::VARCHAR, 3);
    column->SetValueAt<std::string>(0, std::string(4096, 'x'));
    column->nullAt(1);
    column->SetValueAt<std::string>(2, "tail");
    auto input =
        std::make_shared<RowVector>(type, 3, std::vector<VectorPtr>{column});
    auto dictionary = SelectRows(
        *input,
        std::make_shared<RowMapping>(std::vector<vector_size_t>{2, 1, 0, 2}));
    auto selected = OwnedSelection::FromIndices(4, {0, 1, 2});
    auto output = project.Eval(
        ReadBatchBinding::Range(0, 4), selected.view(), dictionary.get());
    input.reset();
    column.reset();
    dictionary.reset();
    auto materialized = MaterializeScalarColumn(View(output));
    auto constants = MaterializeScalarColumn(View(output, 1));
    output.reset();
    auto values = ReadOnlyVectorView::CopyOf(materialized);
    EXPECT_EQ(values.StringViewAt(0), "tail");
    EXPECT_FALSE(values.IsValid(1));
    EXPECT_EQ(values.StringViewAt(2), std::string(4096, 'x'));
    EXPECT_EQ(ReadOnlyVectorView::CopyOf(constants).ValueAt<std::string>(2),
              "constant");
}

TEST(ScalarProjectTest, ReaderFailureAndFinalCheckpointPreserveOriginalErrors) {
    auto compiled = CompileExpressions({Literal(1), Column(100)});
    ASSERT_TRUE(compiled.supported());
    auto prepared = Prepare(compiled.plan, [](const auto&) {
        return PreparedInput{
            DataType::INT64,
            true,
            false,
            std::make_shared<int>(0),
            [](const ReadBatchBinding&) {
                return InputLoaders{[](SelectionView) -> InputValueResult {
                                        ThrowInfo(
                                            FileReadFailed,
                                            "injected projection read failure");
                                    },
                                    {}};
            }};
    });
    ScalarProject reader(std::move(prepared), {"constant", "read"});
    RowVectorPtr result;
    try {
        result =
            reader.Eval(ReadBatchBinding::Range(0, 1), SelectionView::All(1));
        FAIL() << "expected reader failure";
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), ErrorCode::FileReadFailed);
    }
    EXPECT_EQ(result, nullptr);
    EXPECT_EQ(
        reader.Eval(ReadBatchBinding::Range(0, 1), SelectionView::None(1))
            ->size(),
        0);
    auto project = Project({Literal(1)}, {"constant"}, false);
    // Empty Eval never executes an expression checkpoint; cancellation must
    // still be observed before publishing even this empty result.
    EXPECT_THROW(
        result = project.Eval(ReadBatchBinding::Range(0, 0),
                              SelectionView::All(0),
                              nullptr,
                              [] { throw folly::FutureCancellation(); }),
        folly::FutureCancellation);
    EXPECT_EQ(result, nullptr);
}

}  // namespace milvus::exec::plan

namespace milvus::exec::plan {
TEST(ScalarProjectTest, ArithmeticAcrossDecodedSpansAndDictionarySelections) {
    auto check = []<typename T>(DataType type) {
        constexpr size_t n = 4099;
        auto make = [&](bool right) {
            VectorResultBuilder values(type, n);
            for (size_t i = 0; i < n; ++i)
                if (i % (right ? 13 : 11))
                    values.Set<T>(i,
                                  right ? T(i % 7 + 1) : T(int(i % 101) - 50));
            auto view = std::move(values).Finish(SelectionView::All(n));
            const size_t split = right ? 1025 : 641;
            return ReadOnlyVectorView::Segments(
                type,
                SelectionView::All(n),
                {{0, view.Slice(0, split)},
                 {split, view.Slice(split, n - split)}});
        };
        auto row_type =
            std::make_shared<RowType>(std::vector<std::string>{"a", "b"},
                                      std::vector<DataType>{type, type});
        RowVector input(
            row_type, n, {make(false).AsVector(), make(true).AsVector()});
        CompileOptions options;
        options.upstream_schema = row_type;
        options.upstream_columns = {{100, 0}, {101, 1}};
        std::vector<expr::TypedExprPtr> roots;
        for (auto op : {proto::plan::Add,
                        proto::plan::Sub,
                        proto::plan::Mul,
                        proto::plan::Div})
            roots.push_back(
                Arithmetic(Column(100, type), op, Column(101, type)));
        auto compiled = CompileExpressions(roots, options);
        ASSERT_TRUE(compiled.supported());
        ScalarProject project(Prepare(compiled.plan),
                              {"add", "sub", "mul", "div"});
        std::vector<vector_size_t> mapping(n);
        for (size_t i = 0; i < n; ++i) mapping[i] = (n - i - 1) / 2;
        auto dictionary =
            SelectRows(input, std::make_shared<RowMapping>(mapping));
        for (bool mapped : {false, true})
            for (bool sparse : {false, true}) {
                auto rows = OwnedSelection::None(n);
                for (size_t i = 0; i < n; ++i)
                    if (!sparse || i % 3 == 0)
                        rows.SetSelected(i);
                auto output = project.Eval(ReadBatchBinding::Range(0, n),
                                           rows.view(),
                                           mapped ? dictionary.get() : &input);
                size_t dest = 0;
                for (size_t i = 0; i < n; ++i) {
                    if (sparse && i % 3)
                        continue;
                    const auto row = mapped ? size_t(mapping[i]) : i;
                    const bool valid = row % 11 && row % 13;
                    const T a = T(int(row % 101) - 50), b = T(row % 7 + 1);
                    const T expected[] = {
                        T(a + b), T(a - b), T(a * b), T(a / b)};
                    for (size_t col = 0; col < 4; ++col) {
                        auto result = View(output, col);
                        ASSERT_EQ(result.IsValid(dest), valid);
                        if (valid)
                            EXPECT_EQ(result.ValueAt<T>(dest), expected[col]);
                    }
                    ++dest;
                }
                EXPECT_EQ(output->size(), dest);
            }
    };
    check.template operator()<int64_t>(DataType::INT64);
    check.template operator()<double>(DataType::DOUBLE);
}
}  // namespace milvus::exec::plan
