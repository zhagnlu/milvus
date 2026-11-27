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
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "common/GeometryCache.h"
#include <filesystem>
#include <fstream>
#include <roaring/roaring64map.hh>
#include "common/RoaringMembership.h"
#include "exec/plan/ExprCompiler.h"
#include "exec/plan/ExecutionPreparer.h"
#include "exec/expression/ScalarPredicates.h"
#include "exec/expression/InputRefExpr.h"
#include "exec/expression/ScalarComplexPredicates.h"
#include "common/Geometry.h"
#include "exec/expression/TimestampArithmetic.h"

namespace milvus::exec::plan {
namespace {
proto::plan::GenericValue
Int(int64_t n) {
    proto::plan::GenericValue v;
    v.set_int64_val(n);
    return v;
}
proto::plan::GenericValue
Text(std::string s) {
    proto::plan::GenericValue v;
    v.set_string_val(std::move(s));
    return v;
}
proto::plan::GenericValue
Float(double n) {
    proto::plan::GenericValue v;
    v.set_float_val(n);
    return v;
}
proto::plan::GenericValue
Array(std::vector<proto::plan::GenericValue> values) {
    proto::plan::GenericValue v;
    for (const auto& item : values) *v.mutable_array_val()->add_array() = item;
    return v;
}
expr::ColumnInfo
Field(DataType type, std::vector<std::string> path = {}) {
    return {FieldId(100), type, std::move(path), true};
}

template <typename T>
std::shared_ptr<ColumnVector>
Column(DataType type, std::vector<T> values, int null_row = -1) {
    auto result = std::make_shared<ColumnVector>(type, values.size());
    for (size_t i = 0; i < values.size(); ++i)
        result->template SetValueAt<T>(i, values[i]);
    if (null_row >= 0)
        result->nullAt(null_row);
    return result;
}
std::vector<ReadOnlyVectorView>
EvaluateColumn(const std::vector<expr::TypedExprPtr>& roots,
               std::shared_ptr<ColumnVector> column,
               SelectionView rows,
               bool indexes = false) {
    auto type = ReadOnlyVectorView::CopyOf(column).logical_type();
    auto schema = std::make_shared<RowType>(std::vector<std::string>{"x"},
                                            std::vector<DataType>{type});
    CompileOptions options;
    options.upstream_columns = {{100, 0}};
    options.upstream_schema = schema;
    options.plan_indexes = indexes;
    auto result = CompileExpressions(roots, options);
    EXPECT_TRUE(result.supported());
    if (!result.supported())
        return {};
    RowVector input(schema, rows.row_count(), {column});
    auto batch = Prepare(result.plan)
                     .MakeExecutionBatch(
                         ReadBatchBinding::Range(0, rows.row_count()), &input);
    return batch->Eval(rows);
}
void
Expect(const ReadOnlyVectorView& values, std::vector<int> expected) {
    ASSERT_EQ(values.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_EQ(values.IsValid(i) ? int(values.ValueAt<bool>(i)) : -1,
                  expected[i])
            << i;
}
std::shared_ptr<const RoaringMembership>
Membership() {
    roaring::Roaring64Map bitmap;
    bitmap.add(uint64_t(-1));
    bitmap.add(uint64_t(7));
    std::string body(bitmap.getSizeInBytes(true), '\0');
    bitmap.write(body.data(), true);
    std::string blob(RoaringMembership::kHeaderSize, '\0');
    blob.replace(0, 4, "MRB1");
    auto write = [&](size_t offset, uint64_t value, size_t bytes) {
        for (size_t i = 0; i < bytes; ++i)
            blob[offset + i] = char(value >> (8 * i));
    };
    write(4, RoaringMembership::kVersion, 2);
    write(6, RoaringMembership::kFormatPortableRoaring64, 2);
    write(8, bitmap.cardinality(), 8);
    write(16, body.size(), 8);
    return RoaringMembership::Parse(blob + body);
}

std::string
Unhex(const std::string& hex) {
    std::string result;
    for (size_t i = 0; i < hex.size(); i += 2)
        result.push_back(
            static_cast<char>(std::stoul(hex.substr(i, 2), nullptr, 16)));
    return result;
}

std::string
IntBloom() {
    // Shared Go/C++ golden vector: min int64, -1, 0, 1, 2, 42, 1000000007, max int64.
    return Unhex(
        "4d424631010001000800000000000000fca9f1d24d62503f0100000001000000"
        "0117100a804845062834804200440612100448a40680089c4040542838410046");
}
}  // namespace

TEST(ScalarMigrationTest, BloomCompilerMatchesSharedGoldenVectors) {
    auto path = std::filesystem::path(__FILE__).parent_path() /
                "../../../unittest/testdata/bloom/golden_vectors.json";
    std::ifstream stream(path);
    ASSERT_TRUE(stream.good()) << path;
    nlohmann::json fixture;
    stream >> fixture;
    for (const auto& test : fixture.at("cases")) {
        SCOPED_TRACE(test.at("name").get<std::string>());
        auto blob = Unhex(test.at("blob_hex").get<std::string>());
        for (const auto& probe : test.at("probes")) {
            const bool integer = probe.at("kind") == "int64";
            const auto type = integer ? DataType::INT64 : DataType::VARCHAR;
            auto logical =
                std::make_shared<expr::BloomFilterExpr>(Field(type), blob);
            auto column =
                integer
                    ? Column<int64_t>(
                          type,
                          {std::stoll(probe.at("int64").get<std::string>())})
                    : Column<std::string>(
                          type, {probe.at("string").get<std::string>()});
            auto out =
                EvaluateColumn({logical}, column, SelectionView::All(1), true);
            ASSERT_EQ(out.size(), 1);
            EXPECT_TRUE(out[0].IsValid(0));
            EXPECT_EQ(out[0].ValueAt<bool>(0), probe.at("expect").get<bool>());
            auto json_text = integer ? probe.at("int64").get<std::string>()
                                     : probe.at("string").dump(-1, ' ', true);
            auto json_logical = std::make_shared<expr::BloomFilterExpr>(
                Field(DataType::JSON), blob);
            auto json_out = EvaluateColumn(
                {json_logical},
                Column<Json>(DataType::JSON,
                             {Json(simdjson::padded_string(json_text))}),
                SelectionView::All(1));
            ASSERT_EQ(json_out.size(), 1);
            EXPECT_TRUE(json_out[0].IsValid(0));
            EXPECT_EQ(json_out[0].ValueAt<bool>(0),
                      probe.at("expect").get<bool>());
        }
    }
}

TEST(ScalarMigrationTest, BloomNarrowSignExtensionNullAndNot) {
    auto run = []<typename T>(DataType type) {
        auto logical =
            std::make_shared<expr::BloomFilterExpr>(Field(type), IntBloom());
        auto negated = std::make_shared<expr::LogicalUnaryExpr>(
            expr::LogicalUnaryExpr::OpType::LogicalNot, logical);
        auto out =
            EvaluateColumn({logical, negated},
                           Column<T>(type, {T(-1), T(42), T(7), T(0)}, 3),
                           SelectionView::All(4));
        ASSERT_EQ(out.size(), 2);
        Expect(out[0], {1, 1, 0, -1});
        Expect(out[1], {0, 0, 1, -1});
    };
    run.template operator()<int8_t>(DataType::INT8);
    run.template operator()<int16_t>(DataType::INT16);
    run.template operator()<int32_t>(DataType::INT32);
    run.template operator()<int64_t>(DataType::INT64);
}

TEST(ScalarMigrationTest, BloomJsonStrictTypesPathsAndSharedRawSlot) {
    auto logical = std::make_shared<expr::BloomFilterExpr>(
        Field(DataType::JSON, {"a/b", "~"}), IntBloom());
    auto negated = std::make_shared<expr::LogicalUnaryExpr>(
        expr::LogicalUnaryExpr::OpType::LogicalNot, logical);
    auto other = std::make_shared<expr::BloomFilterExpr>(
        Field(DataType::JSON, {"other"}), IntBloom());
    auto project = std::make_shared<expr::ColumnExpr>(Field(DataType::JSON));
    auto compiled = CompileExpressions({logical, negated, other, project});
    ASSERT_TRUE(compiled.supported());
    ASSERT_EQ(compiled.plan->inputs().slots().size(), 1);
    EXPECT_TRUE(compiled.plan->inputs().slots()[0].path.empty());
    // JSON numbers outside int64 are definite non-members; no-probe types
    // are UNKNOWN. Numeric strings never alias the integer domain.
    std::vector<Json> values;
    for (const auto& value : {"42",
                              "42.0",
                              "18446744073709551615",
                              "\"42\"",
                              "true",
                              "null",
                              "{}",
                              "[]",
                              "7",
                              "-1"})
        values.emplace_back(simdjson::padded_string(
            std::string("{\"a/b\":{\"~\":") + value + "}}"));
    values.emplace_back(simdjson::padded_string(std::string("{}")));
    values.emplace_back(simdjson::padded_string(std::string("{}")));
    auto out = EvaluateColumn({logical, negated, other, project},
                              Column<Json>(DataType::JSON, values, 11),
                              SelectionView::All(12));
    ASSERT_EQ(out.size(), 4);
    Expect(out[0], {1, 0, 0, 0, -1, -1, -1, -1, 0, 1, -1, -1});
    Expect(out[1], {0, 1, 1, 1, -1, -1, -1, -1, 1, 0, -1, -1});
    Expect(out[2], std::vector<int>(12, -1));
    EXPECT_TRUE(out[3].IsValid(10));
    EXPECT_FALSE(out[3].IsValid(11));
}

TEST(ScalarMigrationTest, BloomJsonRootStringsAndAbsentDomain) {
    // A saturated integer-only filter must still reject strings, including
    // strings whose bytes could be an int64 LE representation.
    auto blob = IntBloom();
    std::fill(blob.begin() + 32, blob.end(), char(0xff));
    std::vector<Json> values;
    for (auto text : {"\"ABCDEFGH\"", "42", "42.0", "null", "true"})
        values.emplace_back(simdjson::padded_string(std::string(text)));
    auto logical =
        std::make_shared<expr::BloomFilterExpr>(Field(DataType::JSON), blob);
    auto out = EvaluateColumn(
        {logical}, Column<Json>(DataType::JSON, values), SelectionView::All(5));
    ASSERT_EQ(out.size(), 1);
    Expect(out[0], {0, 1, 0, -1, -1});
    blob[28] = 2;  // UTF-8 only; numerical probes must now fail.
    logical =
        std::make_shared<expr::BloomFilterExpr>(Field(DataType::JSON), blob);
    out = EvaluateColumn(
        {logical}, Column<Json>(DataType::JSON, values), SelectionView::All(5));
    Expect(out[0], {1, 0, 0, -1, -1});
    blob[28] = 0;  // Empty domain, even if all bits happen to be set.
    logical =
        std::make_shared<expr::BloomFilterExpr>(Field(DataType::JSON), blob);
    out = EvaluateColumn(
        {logical}, Column<Json>(DataType::JSON, values), SelectionView::All(5));
    Expect(out[0], {0, 0, 0, -1, -1});
}

TEST(ScalarMigrationTest, BloomRetainsBlobAfterLogicalPlanIsDestroyed) {
    auto owner = std::make_shared<const std::string>(IntBloom());
    std::weak_ptr<const std::string> weak = owner;
    auto logical =
        std::make_shared<expr::BloomFilterExpr>(Field(DataType::INT64), owner);
    auto compiled = CompileExpressions({logical});
    ASSERT_TRUE(compiled.supported());
    logical.reset();
    owner.reset();
    EXPECT_FALSE(weak.expired());
    auto batch = Prepare(compiled.plan, [](const auto&) {
                     PreparedInput input;
                     input.type = DataType::INT64;
                     input.supports_values = true;
                     auto column = Column<int64_t>(DataType::INT64, {-1, 7});
                     input.owner = column;
                     input.make_loaders = [column](const auto&) {
                         InputLoaders loaders;
                         loaders.values = [column](SelectionView rows) {
                             return ReadOnlyVectorView::CopyOf(column, rows);
                         };
                         return loaders;
                     };
                     return input;
                 }).MakeExecutionBatch(ReadBatchBinding::Range(0, 2));
    compiled.plan.reset();
    Expect(batch->Eval(SelectionView::All(2))[0], {1, 0});
    batch.reset();
    EXPECT_TRUE(weak.expired());
}

TEST(ScalarMigrationTest, BloomMalformedEnvelopePreservesInputErrorAtCompile) {
    std::vector<std::string> bad{
        "", IntBloom().substr(0, 31), IntBloom() + "x"};
    for (auto [offset, byte] : std::vector<std::pair<size_t, char>>{
             {0, 'X'}, {4, 2}, {6, 2}, {24, 0}, {24, 3}, {28, 4}, {29, 1}}) {
        auto blob = IntBloom();
        blob[offset] = byte;
        bad.push_back(std::move(blob));
    }
    for (const auto& blob : bad) {
        auto logical = std::make_shared<expr::BloomFilterExpr>(
            Field(DataType::INT64), blob);
        try {
            CompileExpressions({logical});
            FAIL() << "invalid envelope accepted";
        } catch (const SegcoreError& e) {
            EXPECT_EQ(e.get_error_code(), ExprInvalid);
        }
    }
}

TEST(ScalarMigrationTest, BloomEmptyShortCircuitAndCancellationDoNotRead) {
    auto bloom = scalar::MakeBloomMembership(
        std::make_shared<scalar::InputRefExpr>(0, DataType::INT64),
        std::make_shared<const std::string>(IntBloom()));
    auto no = scalar::MakeLiteral(Column<bool>(DataType::BOOL, {false}));
    auto conjunction = scalar::MakeLogical(scalar::LogicalOp::And, {no, bloom});
    BatchInput input(
        std::make_shared<RowType>(std::vector<std::string>{"x"},
                                  std::vector<DataType>{DataType::INT64}),
        2);
    int reads = 0;
    input.BindLoader(0, [&](SelectionView rows) {
        ++reads;
        return ReadOnlyVectorView::CopyOf(
            Column<int64_t>(DataType::INT64, {-1, 7}), rows);
    });
    scalar::EvalCtx ctx(input);
    Expect(conjunction->Eval(SelectionView::All(2), ctx), {0, 0});
    bloom->Eval(SelectionView::None(2), ctx);
    EXPECT_EQ(reads, 0);
    scalar::EvalCtx cancelled(input,
                              [] { ThrowInfo(FollyCancel, "cancel bloom"); });
    try {
        bloom->Eval(SelectionView::All(2), cancelled);
        FAIL();
    } catch (const SegcoreError& e) {
        EXPECT_EQ(e.get_error_code(), FollyCancel);
    }
    EXPECT_EQ(reads, 0);
    Expect(bloom->Eval(SelectionView::All(2), ctx), {1, 0});
    EXPECT_EQ(reads, 1);

    // Bad envelopes are rejected during construction, even below FALSE AND,
    // matching the legacy physical constructor rather than deferred parsing.
    proto::plan::GenericValue no_value;
    no_value.set_bool_val(false);
    auto invalid = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::And,
        std::make_shared<expr::ValueExpr>(no_value),
        std::make_shared<expr::BloomFilterExpr>(Field(DataType::INT64), ""));
    try {
        CompileExpressions({invalid});
        FAIL();
    } catch (const SegcoreError& e) {
        EXPECT_EQ(e.get_error_code(), ExprInvalid);
    }
}

TEST(ScalarMigrationTest, BloomJsonSkipsUnselectedCorruptionAndKeepsErrorCode) {
    auto logical = std::make_shared<expr::BloomFilterExpr>(
        Field(DataType::JSON, {"x"}), IntBloom());
    auto column =
        Column<Json>(DataType::JSON,
                     {Json(simdjson::padded_string(std::string("{\"x\":42}"))),
                      Json(simdjson::padded_string(std::string("{\"x\":\"") +
                                                   char(0xff) + "\"}"))});
    auto rows = OwnedSelection::None(2);
    rows.SetSelected(0);
    auto out = EvaluateColumn({logical}, column, rows.view());
    ASSERT_EQ(out.size(), 1);
    EXPECT_TRUE(out[0].ValueAt<bool>(0));
    EXPECT_FALSE(out[0].defined_rows().IsSelected(1));
    EvaluateColumn({logical}, column, SelectionView::None(2));
    try {
        EvaluateColumn({logical}, column, SelectionView::All(2));
        FAIL();
    } catch (const SegcoreError& e) {
        EXPECT_EQ(e.get_error_code(), DataFormatBroken);
    }
}

TEST(ScalarMigrationTest, TermCompilerOverflowDuplicatesEmptyAndNot) {
    auto column = Column<int8_t>(DataType::INT8, {-1, 7, 8, 0}, 3);
    auto term = std::make_shared<expr::TermFilterExpr>(
        Field(DataType::INT8),
        std::vector{Int(-1), Int(7), Int(7), Int(128), Int(-129)});
    auto not_term = std::make_shared<expr::LogicalUnaryExpr>(
        expr::LogicalUnaryExpr::OpType::LogicalNot, term);
    auto empty = std::make_shared<expr::TermFilterExpr>(
        Field(DataType::INT8), std::vector<proto::plan::GenericValue>{});
    auto results = EvaluateColumn(
        {term, not_term, empty}, column, SelectionView::All(4), true);
    ASSERT_EQ(results.size(), 3);
    Expect(results[0], {1, 1, 0, -1});
    Expect(results[1], {0, 0, 1, -1});
    Expect(results[2], {0, 0, 0, -1});
}

TEST(ScalarMigrationTest, ArithmeticIntegerOperationsAndNullsAcrossWidths) {
    struct Case {
        proto::plan::ArithOpType operation;
        int64_t operand;
        int64_t bound;
        std::vector<int> expected;
    };
    const std::vector<Case> cases{{proto::plan::Add, 2, 8, {1, 0, 0, 0, -1}},
                                  {proto::plan::Sub, 2, 4, {1, 0, 0, 0, -1}},
                                  {proto::plan::Mul, 2, 12, {1, 0, 0, 0, -1}},
                                  {proto::plan::Div, 2, 3, {1, 0, 0, 0, -1}},
                                  {proto::plan::Mod, 4, 2, {1, 0, 0, 0, -1}},
                                  {proto::plan::BitAnd, 3, 2, {1, 1, 0, 0, -1}},
                                  {proto::plan::BitOr, 3, 7, {1, 0, 0, 0, -1}},
                                  {proto::plan::BitXor, 3, 5, {1, 0, 0, 0, -1}},
                                  {proto::plan::Shl, 1, 12, {1, 0, 0, 0, -1}},
                                  {proto::plan::Shr, 1, 3, {1, 0, 0, 0, -1}}};
    auto check = [&]<typename T>(DataType type) {
        for (const auto& test : cases) {
            auto logical = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
                Field(type),
                proto::plan::Equal,
                test.operation,
                Int(test.bound),
                Int(test.operand));
            auto negated = std::make_shared<expr::LogicalUnaryExpr>(
                expr::LogicalUnaryExpr::OpType::LogicalNot, logical);
            auto output = EvaluateColumn({logical, negated},
                                         Column<T>(type, {6, -6, 0, 9, 0}, 4),
                                         SelectionView::All(5));
            ASSERT_EQ(output.size(), 2);
            Expect(output[0], test.expected);
            auto inverse = test.expected;
            for (auto& value : inverse)
                if (value >= 0)
                    value = !value;
            Expect(output[1], inverse);
        }
    };
    check.template operator()<int8_t>(DataType::INT8);
    check.template operator()<int16_t>(DataType::INT16);
    check.template operator()<int32_t>(DataType::INT32);
    check.template operator()<int64_t>(DataType::INT64);
    // Constants/intermediate results are not narrowed to the INT8 field.
    auto promoted = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
        Field(DataType::INT8),
        proto::plan::Equal,
        proto::plan::Add,
        Int(128),
        Int(1));
    auto out = EvaluateColumn({promoted},
                              Column<int8_t>(DataType::INT8, {127}),
                              SelectionView::All(1));
    ASSERT_EQ(out.size(), 1);
    Expect(out[0], {1});
}

TEST(ScalarMigrationTest, ArithmeticFloatingAndDefinedIntegerOverflow) {
    auto check = []<typename T>(DataType type) {
        auto logical = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
            Field(type),
            proto::plan::LessThan,
            proto::plan::Mul,
            Float(3),
            Float(0.5));
        auto output = EvaluateColumn(
            {logical},
            Column<T>(type, {5, 6, -6, std::numeric_limits<T>::quiet_NaN()}),
            SelectionView::All(4));
        ASSERT_EQ(output.size(), 1);
        Expect(output[0], {1, 0, 1, 0});
    };
    check.template operator()<float>(DataType::FLOAT);
    check.template operator()<double>(DataType::DOUBLE);
    auto wrapped = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
        Field(DataType::INT64),
        proto::plan::Equal,
        proto::plan::Add,
        Int(std::numeric_limits<int64_t>::min()),
        Int(1));
    auto output = EvaluateColumn(
        {wrapped},
        Column<int64_t>(DataType::INT64,
                        {std::numeric_limits<int64_t>::max(), 0}),
        SelectionView::All(2));
    ASSERT_EQ(output.size(), 1);
    Expect(output[0], {1, 0});
}

TEST(ScalarMigrationTest, ArithmeticErrorsAreLazyAndKeepCodes) {
    auto column = Column<int64_t>(DataType::INT64,
                                  {std::numeric_limits<int64_t>::min(), 6});
    for (const auto& [operation, operand] :
         std::vector<std::pair<proto::plan::ArithOpType, int64_t>>{
             {proto::plan::Div, 0},
             {proto::plan::Mod, 0},
             {proto::plan::Shl, -1},
             {proto::plan::Shr, 64},
             {proto::plan::Div, -1}}) {
        auto logical = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
            Field(DataType::INT64),
            proto::plan::Equal,
            operation,
            Int(0),
            Int(operand));
        auto skipped = std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::Or,
            std::make_shared<expr::AlwaysTrueExpr>(),
            logical);
        auto result = EvaluateColumn({skipped}, column, SelectionView::All(2));
        ASSERT_EQ(result.size(), 1);
        Expect(result[0], {1, 1});
        EXPECT_NO_THROW(
            EvaluateColumn({logical}, column, SelectionView::None(2)));
        try {
            EvaluateColumn({logical}, column, SelectionView::All(2));
            FAIL() << "invalid arithmetic must not execute undefined behavior";
        } catch (const SegcoreError& error) {
            EXPECT_EQ(error.get_error_code(), ExprInvalid);
        }
        if (operation == proto::plan::Div && operand == -1) {
            const auto selected = OwnedSelection::FromIndices(2, {1});
            auto output = EvaluateColumn({logical}, column, selected.view());
            ASSERT_EQ(output.size(), 1);
            EXPECT_FALSE(output[0].ValueAt<bool>(1));
        }
    }
}

TEST(ScalarMigrationTest,
     FloatingDivisionCompilesWithRoundedQuotientSemantics) {
    // Multiplying the bound loses the rounded quotient semantics.
    // Both new execution and the legacy sequential path now use division.
    volatile double numerator = 1;
    volatile double divisor = 49;
    const double quotient = numerator / divisor;
    EXPECT_NE(numerator, divisor * quotient);
    for (const auto type : {DataType::FLOAT, DataType::DOUBLE}) {
        auto expression = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
            Field(type),
            proto::plan::Equal,
            proto::plan::Div,
            Float(quotient),
            Float(49));
        EXPECT_TRUE(CompileExpressions({expression}).supported());
    }
}

TEST(ScalarMigrationTest, JsonNumericComparisonRangeAndTermPreservePrecision) {
    const auto field = Field(DataType::JSON, {"x"});
    auto equal = std::make_shared<expr::UnaryRangeFilterExpr>(
        field, proto::plan::Equal, Int(9007199254740993LL));
    auto range = std::make_shared<expr::BinaryRangeFilterExpr>(
        field, Int(9007199254740992LL), Int(9007199254740994LL), false, false);
    auto term = std::make_shared<expr::TermFilterExpr>(
        field, std::vector{Int(9007199254740993LL)}, false);
    auto negate = std::make_shared<expr::LogicalUnaryExpr>(
        expr::LogicalUnaryExpr::OpType::LogicalNot, equal);
    auto project = std::make_shared<expr::ColumnExpr>(Field(DataType::JSON));
    auto compiled = CompileExpressions({equal, range, term, project});
    ASSERT_TRUE(compiled.supported());
    EXPECT_EQ(compiled.plan->inputs().slots().size(), 1);
    std::vector<Json> values;
    for (const auto* text : {"{\"x\":9007199254740993}",
                             "{\"x\":9007199254740992}",
                             "{\"x\":9007199254740993.0}",
                             "{\"x\":18446744073709551615}",
                             "{\"x\":1}",
                             "{\"x\":\"1\"}",
                             "{}",
                             "{\"x\":null}",
                             "{}"})
        values.emplace_back(simdjson::padded_string(std::string(text)));
    auto output = EvaluateColumn({equal, range, term, negate},
                                 Column<Json>(DataType::JSON, values, 8),
                                 SelectionView::All(values.size()));
    ASSERT_EQ(output.size(), 4);
    for (size_t i = 0; i < 3; ++i)
        Expect(output[i], {1, 0, 0, 0, 0, -1, -1, -1, -1});
    Expect(output[3], {0, 1, 1, 1, 1, -1, -1, -1, -1});
}

TEST(ScalarMigrationTest, JsonPatternsAndArrayEqualityRetainValidity) {
    const auto field = Field(DataType::JSON, {"x"});
    auto prefix = std::make_shared<expr::UnaryRangeFilterExpr>(
        field, proto::plan::PrefixMatch, Text("a"));
    auto like = std::make_shared<expr::UnaryRangeFilterExpr>(
        field, proto::plan::Match, Text("a%"));
    auto array = std::make_shared<expr::UnaryRangeFilterExpr>(
        field, proto::plan::Equal, Array({Int(1), Text("s")}));
    std::vector<Json> values;
    for (const auto* text : {"{\"x\":\"abc\"}",
                             "{\"x\":\"b\"}",
                             "{\"x\":[1,\"s\"]}",
                             "{\"x\":[1.0,\"s\"]}",
                             "{}",
                             "{\"x\":null}"})
        values.emplace_back(simdjson::padded_string(std::string(text)));
    auto output = EvaluateColumn({prefix, like, array},
                                 Column<Json>(DataType::JSON, values),
                                 SelectionView::All(values.size()));
    ASSERT_EQ(output.size(), 3);
    Expect(output[0], {1, 0, -1, -1, -1, -1});
    Expect(output[1], {1, 0, -1, -1, -1, -1});
    Expect(output[2], {-1, -1, 1, 0, -1, -1});
}

TEST(ScalarMigrationTest, JsonContainsAllAnyMixedDuplicatesAndEmpty) {
    const auto field = Field(DataType::JSON, {"x"});
    auto contains =
        [&](bool all, bool same, std::vector<proto::plan::GenericValue> terms) {
            return std::make_shared<expr::JsonContainsExpr>(
                field,
                all ? proto::plan::JSONContainsExpr_JSONOp_ContainsAll
                    : proto::plan::JSONContainsExpr_JSONOp_ContainsAny,
                same,
                terms);
        };
    std::vector<Json> values;
    for (const auto* text : {"{\"x\":[1.0,\"s\",[1,\"x\"]]}",
                             "{\"x\":[1,2,1]}",
                             "{\"x\":[]}",
                             "{\"x\":1}",
                             "{}",
                             "{\"x\":null}",
                             "{}"})
        values.emplace_back(simdjson::padded_string(std::string(text)));
    auto output = EvaluateColumn(
        {contains(true, false, {Int(1), Text("s"), Array({Int(1), Text("x")})}),
         contains(false, true, {Int(1), Int(2)}),
         contains(true, true, {Int(1), Int(1)}),
         contains(true, false, {}),
         contains(false, false, {})},
        Column<Json>(DataType::JSON, values, 6),
        SelectionView::All(values.size()));
    ASSERT_EQ(output.size(), 5);
    Expect(output[0], {1, 0, 0, -1, -1, -1, -1});
    Expect(output[1], {1, 1, 0, -1, -1, -1, -1});
    Expect(output[2], {1, 1, 0, -1, -1, -1, -1});
    Expect(output[3], {1, 1, 1, 1, 1, 1, 1});
    Expect(output[4], {0, 0, 0, 0, 0, 0, 0});
}

TEST(ScalarMigrationTest, JsonInFieldIsStrictAndEmptyInIsValidFalse) {
    const auto field = Field(DataType::JSON, {"x"});
    auto in_field = std::make_shared<expr::TermFilterExpr>(
        field, std::vector{Int(1)}, true);
    auto empty = std::make_shared<expr::TermFilterExpr>(
        field, std::vector<proto::plan::GenericValue>{}, false);
    auto negated = std::make_shared<expr::LogicalUnaryExpr>(
        expr::LogicalUnaryExpr::OpType::LogicalNot, in_field);
    std::vector<Json> values;
    for (const auto* text :
         {"{\"x\":[1]}", "{\"x\":[1.0]}", "{\"x\":[]}", "{}", "{}"})
        values.emplace_back(simdjson::padded_string(std::string(text)));
    auto output = EvaluateColumn({in_field, empty, negated},
                                 Column<Json>(DataType::JSON, values, 4),
                                 SelectionView::All(values.size()));
    ASSERT_EQ(output.size(), 3);
    Expect(output[0], {1, 0, 0, -1, -1});
    Expect(output[1], {0, 0, 0, 0, 0});
    Expect(output[2], {0, 1, 1, -1, -1});
}

TEST(ScalarMigrationTest, JsonContainsTraversesNonmatchingElementsAndStrings) {
    auto contains = [](std::vector<proto::plan::GenericValue> terms) {
        return std::make_shared<expr::JsonContainsExpr>(
            Field(DataType::JSON, {"x"}),
            proto::plan::JSONContainsExpr_JSONOp_ContainsAll,
            true,
            terms);
    };
    std::vector<Json> values;
    for (const auto* text : {"{\"x\":[null,{},[],false,\"other\",\"a\",\"b\"]}",
                             "{\"x\":[null,{},[],false,9,1.0,2]}",
                             "{\"x\":[\"a\"]}",
                             "{\"x\":[]}"})
        values.emplace_back(simdjson::padded_string(std::string(text)));
    auto output = EvaluateColumn({contains({Text("a"), Text("b"), Text("a")}),
                                  contains({Int(1), Int(2)})},
                                 Column<Json>(DataType::JSON, values),
                                 SelectionView::All(values.size()));
    ASSERT_EQ(output.size(), 2);
    Expect(output[0], {1, 0, 0, 0});
    Expect(output[1], {0, 1, 0, 0});
}

TEST(ScalarMigrationTest, TypedJsonContainsResetsScratchAndPreservesCoercion) {
    std::vector<Json> documents;
    for (const auto* text :
         {R"({"x":[true,false,1,2.5,"a","b",9007199254740993]})",
          R"({"x":[true,1.0,"a",9007199254740992.0]})",
          R"({"x":[null,{},[],false,2.5,"b"]})",
          R"({"x":[]})",
          R"({})"})
        documents.emplace_back(simdjson::padded_string(std::string(text)));
    auto boolean = [](bool b) {
        proto::plan::GenericValue value;
        value.set_bool_val(b);
        return value;
    };
    auto contains = [](std::vector<proto::plan::GenericValue> terms) {
        return std::make_shared<expr::JsonContainsExpr>(
            Field(DataType::JSON, {"x"}),
            proto::plan::JSONContainsExpr_JSONOp_ContainsAll,
            true,
            terms);
    };
    auto result = EvaluateColumn(
        {contains({boolean(true), boolean(false), boolean(true)}),
         contains({Float(1), Float(2.5), Float(1)}),
         contains({Text("a"), Text("b"), Text("a")}),
         contains({Int(9007199254740993LL)})},
        Column<Json>(DataType::JSON, documents),
        SelectionView::All(documents.size()));
    ASSERT_EQ(result.size(), 4);
    for (const auto& output : result) Expect(output, {1, 0, 0, 0, -1});
}

TEST(ScalarMigrationTest, TermNativeTypesAndConstantOwnership) {
    auto run = []<typename T>(DataType type, T a, T b) {
        auto values = Column<T>(type, {a, b, a}, 2);
        auto constants = Column<T>(type, {a});
        auto expr = scalar::MakeTerm(
            std::make_shared<scalar::InputRefExpr>(0, type), constants);
        constants->template SetValueAt<T>(0, b);
        BatchInput input(
            std::make_shared<RowType>(std::vector<std::string>{"x"},
                                      std::vector<DataType>{type}),
            3);
        input.BindValues(0, ReadOnlyVectorView::CopyOf(values));
        scalar::EvalCtx ctx(input);
        Expect(expr->Eval(SelectionView::All(3), ctx), {1, 0, -1});
    };
    run(DataType::BOOL, false, true);
    run(DataType::INT16, int16_t(-1), int16_t(2));
    run(DataType::INT32, int32_t(-1), int32_t(2));
    run(DataType::INT64, int64_t(-1), int64_t(2));
    run(DataType::TIMESTAMPTZ, int64_t(-1), int64_t(2));
    run(DataType::FLOAT, 1.5f, 2.f);
    run(DataType::DOUBLE, 1.5, 2.);
    run(DataType::VARCHAR, std::string("北京"), std::string("上海"));
}

TEST(ScalarMigrationTest, StringOperationsUseExistingUnicodeAndRegexSemantics) {
    auto column = Column<std::string>(
        DataType::VARCHAR, {"北京", "北X京", "a%b", "x\n北京", ""}, 4);
    std::vector<expr::TypedExprPtr> roots;
    for (auto [op, text] :
         std::vector<std::pair<proto::plan::OpType, std::string>>{
             {proto::plan::PrefixMatch, "北"},
             {proto::plan::PostfixMatch, "京"},
             {proto::plan::InnerMatch, "北京"},
             {proto::plan::Match, "北_"},
             {proto::plan::Match, "a\\%b"},
             {proto::plan::RegexMatch, "北.京"}})
        roots.push_back(std::make_shared<expr::UnaryRangeFilterExpr>(
            Field(DataType::VARCHAR), op, Text(text)));
    auto out = EvaluateColumn(roots, column, SelectionView::All(5));
    ASSERT_EQ(out.size(), 6);
    Expect(out[0], {1, 1, 0, 0, -1});
    Expect(out[1], {1, 1, 0, 1, -1});
    Expect(out[2], {1, 0, 0, 1, -1});
    Expect(out[3], {1, 0, 0, 0, -1});
    Expect(out[4], {0, 0, 1, 0, -1});
    Expect(out[5], {0, 1, 0, 0, -1});
}

TEST(ScalarMigrationTest, AdjacentStringFusionPreservesThreeValuesAndBarriers) {
    using namespace scalar;
    auto column = Column<std::string>(
        DataType::VARCHAR, {"北京", "北X京", "a%b", "x\n北京", ""}, 4);
    auto field = std::make_shared<InputRefExpr>(0, DataType::VARCHAR);
    auto first = MakeStringMatch(field, StringOp::Like, "北%");
    auto second =
        MakeStringMatch(std::make_shared<InputRefExpr>(0, DataType::VARCHAR),
                        StringOp::Regex,
                        "京$");
    auto schema =
        std::make_shared<RowType>(std::vector<std::string>{"s"},
                                  std::vector<DataType>{DataType::VARCHAR});
    RowVector upstream(schema, 5, {column});
    for (auto op : {LogicalOp::And, LogicalOp::Or}) {
        auto fused = FuseAdjacentStringMatches(op, {first, second});
        ASSERT_EQ(fused.size(), 1);
        auto adjusted = AdjustLogicalExpression(
            MakeLogical(op, {first, second}),
            [](const auto*) { return ReorderProperty{}; });
        // Prepare's adjustment emits the fused node directly, without a
        // redundant one-child logical result allocation.
        EXPECT_EQ(FuseAdjacentStringMatches(op, {adjusted, first}).size(), 1);
        BatchInput input(upstream);
        EvalCtx ctx(input);
        auto values = fused[0]->Eval(SelectionView::All(5), ctx);
        Expect(values,
               op == LogicalOp::And ? std::vector<int>{1, 1, 0, 0, -1}
                                    : std::vector<int>{1, 1, 0, 1, -1});
        auto sparse = OwnedSelection::FromIndices(5, {1, 3, 4});
        auto partial = fused[0]->Eval(sparse.view(), ctx);
        EXPECT_FALSE(partial.defined_rows().IsSelected(0));
        EXPECT_FALSE(partial.IsValid(4));
        EXPECT_EQ(FuseAdjacentStringMatches(op, {first, MakeNot(second), first})
                      .size(),
                  3);
        auto other = MakeStringMatch(
            std::make_shared<InputRefExpr>(1, DataType::VARCHAR),
            StringOp::Like,
            "%");
        EXPECT_EQ(FuseAdjacentStringMatches(op, {first, other, first}).size(),
                  3);
        auto logical = std::make_shared<expr::LogicalBinaryExpr>(
            op == LogicalOp::And ? expr::LogicalBinaryExpr::OpType::And
                                 : expr::LogicalBinaryExpr::OpType::Or,
            std::make_shared<expr::UnaryRangeFilterExpr>(
                Field(DataType::VARCHAR), proto::plan::Match, Text("北%")),
            std::make_shared<expr::UnaryRangeFilterExpr>(
                Field(DataType::VARCHAR),
                proto::plan::RegexMatch,
                Text("京$")));
        auto prepared =
            EvaluateColumn({logical}, column, SelectionView::All(5));
        ASSERT_EQ(prepared.size(), 1);
        Expect(prepared[0],
               op == LogicalOp::And ? std::vector<int>{1, 1, 0, 0, -1}
                                    : std::vector<int>{1, 1, 0, 1, -1});
    }
}

TEST(ScalarMigrationTest, StringFusionKeepsPatternConstructionLazyAndOrdered) {
    using namespace scalar;
    auto field = std::make_shared<InputRefExpr>(0, DataType::VARCHAR);
    for (auto op : {LogicalOp::And, LogicalOp::Or}) {
        auto first = MakeStringMatch(
            field, StringOp::Prefix, op == LogicalOp::And ? "z" : "");
        auto bad = MakeStringMatch(field, StringOp::Regex, "[");
        auto fused = FuseAdjacentStringMatches(op, {first, bad});
        ASSERT_EQ(fused.size(), 1);
        auto schema =
            std::make_shared<RowType>(std::vector<std::string>{"s"},
                                      std::vector<DataType>{DataType::VARCHAR});
        for (bool null : {false, true}) {
            auto column =
                Column<std::string>(DataType::VARCHAR, {"a"}, null ? 0 : -1);
            RowVector upstream(schema, 1, {column});
            BatchInput input(upstream);
            EvalCtx ctx(input);
            EXPECT_NO_THROW(fused[0]->Eval(SelectionView::None(1), ctx));
            if (!null) {
                auto result = fused[0]->Eval(SelectionView::All(1), ctx);
                EXPECT_EQ(result.ValueAt<bool>(0), op == LogicalOp::Or);
            } else {
                try {
                    fused[0]->Eval(SelectionView::All(1), ctx);
                    FAIL();
                } catch (const SegcoreError& error) {
                    EXPECT_EQ(error.get_error_code(), InvalidParameter);
                }
            }
            auto reversed = FuseAdjacentStringMatches(op, {bad, first});
            EXPECT_THROW(reversed[0]->Eval(SelectionView::All(1), ctx),
                         SegcoreError);
        }
    }
}

TEST(ScalarMigrationTest, GeometryFusionPreservesOrderNullsAndDistances) {
    using namespace scalar;
    auto geos = GetThreadLocalGEOSContext();
    auto point = [&](const char* wkt) {
        return Geometry(geos, wkt).to_wkb_string();
    };
    auto column = Column<std::string>(DataType::GEOMETRY,
                                      {point("POINT(0 0)"),
                                       point("POINT(3 4)"),
                                       "bad WKB",
                                       "",
                                       point("POINT EMPTY")},
                                      3);
    auto schema =
        std::make_shared<RowType>(std::vector<std::string>{"g"},
                                  std::vector<DataType>{DataType::GEOMETRY});
    RowVector upstream(schema, 5, {column});
    auto input = std::make_shared<InputRefExpr>(0, DataType::GEOMETRY);
    auto near =
        MakeGeometryPredicate(input,
                              proto::plan::GISFunctionFilterExpr_GISOp_DWithin,
                              "POINT(0 0)",
                              600000.0);
    auto equal =
        MakeGeometryPredicate(input,
                              proto::plan::GISFunctionFilterExpr_GISOp_Equals,
                              "POINT(0 0)",
                              0);
    auto valid = MakeGeometryPredicate(
        input, proto::plan::GISFunctionFilterExpr_GISOp_STIsValid, "", 0);
    for (auto op : {LogicalOp::And, LogicalOp::Or}) {
        auto fused = FuseAdjacentGeometryPredicates(op, {equal, near, valid});
        ASSERT_EQ(fused.size(), 1);
        BatchInput batch(upstream);
        EvalCtx ctx(batch);
        Expect(fused[0]->Eval(SelectionView::All(5), ctx),
               op == LogicalOp::And ? std::vector<int>{1, 0, 0, -1, 0}
                                    : std::vector<int>{1, 1, 0, -1, 1});
        auto sparse = OwnedSelection::FromIndices(5, {1, 3});
        auto partial = fused[0]->Eval(sparse.view(), ctx);
        EXPECT_FALSE(partial.defined_rows().IsSelected(0));
        EXPECT_FALSE(partial.IsValid(3));
        EXPECT_EQ(
            FuseAdjacentGeometryPredicates(op, {near, MakeNot(equal), valid})
                .size(),
            3);
        auto other = MakeGeometryPredicate(
            std::make_shared<InputRefExpr>(1, DataType::GEOMETRY),
            proto::plan::GISFunctionFilterExpr_GISOp_STIsValid,
            "",
            0);
        EXPECT_EQ(
            FuseAdjacentGeometryPredicates(op, {near, other, valid}).size(), 3);
        auto logical = std::make_shared<expr::LogicalBinaryExpr>(
            op == LogicalOp::And ? expr::LogicalBinaryExpr::OpType::And
                                 : expr::LogicalBinaryExpr::OpType::Or,
            std::make_shared<expr::GISFunctionFilterExpr>(
                Field(DataType::GEOMETRY),
                proto::plan::GISFunctionFilterExpr_GISOp_Equals,
                "POINT(0 0)",
                0),
            std::make_shared<expr::GISFunctionFilterExpr>(
                Field(DataType::GEOMETRY),
                proto::plan::GISFunctionFilterExpr_GISOp_DWithin,
                "POINT(0 0)",
                600000.0));
        CompileOptions options;
        options.plan_indexes = true;
        auto compiled = CompileExpressions({logical}, options);
        ASSERT_TRUE(compiled.supported());
        auto adjusted = AdjustExecution(*compiled.plan, {});
        ASSERT_TRUE(adjusted.supported);
        EXPECT_EQ(FuseAdjacentGeometryPredicates(
                      op, {adjusted.expressions->roots()[0], near})
                      .size(),
                  1);
        Expect(
            EvaluateColumn({logical}, column, SelectionView::All(5), true)[0],
            op == LogicalOp::And ? std::vector<int>{1, 0, 0, -1, 0}
                                 : std::vector<int>{1, 1, 0, -1, 1});
    }
    auto bad = MakeGeometryPredicate(
        input,
        proto::plan::GISFunctionFilterExpr_GISOp_Intersects,
        "bad WKT",
        0);
    BatchInput batch(upstream);
    EvalCtx ctx(batch);
    auto rows12 = OwnedSelection::FromIndices(5, {1, 2});
    auto rows3 = OwnedSelection::FromIndices(5, {3});
    auto rows01 = OwnedSelection::FromIndices(5, {0, 1});
    auto rows1 = OwnedSelection::FromIndices(5, {1});

    auto fused =
        FuseAdjacentGeometryPredicates(LogicalOp::And, {equal, bad})[0];
    EXPECT_NO_THROW(fused->Eval(SelectionView::None(5), ctx));
    EXPECT_NO_THROW(fused->Eval(rows12.view(), ctx));
    EXPECT_THROW(fused->Eval(rows3.view(), ctx), SegcoreError);
    auto disjunction =
        FuseAdjacentGeometryPredicates(LogicalOp::Or, {near, bad})[0];
    EXPECT_NO_THROW(disjunction->Eval(rows01.view(), ctx));
    EXPECT_THROW(
        FuseAdjacentGeometryPredicates(LogicalOp::And, {bad, equal})[0]->Eval(
            rows1.view(), ctx),
        SegcoreError);
    int checkpoints = 0;
    EvalCtx cancelled(batch, [&] {
        if (++checkpoints == 6)
            ThrowInfo(FollyCancel, "cancel during GIS fusion");
    });
    EXPECT_THROW(
        FuseAdjacentGeometryPredicates(LogicalOp::Or, {equal, near})[0]->Eval(
            SelectionView::All(5), cancelled),
        SegcoreError);
    // A later evaluation starts fresh on its own thread-local GEOS context.
    Expect(
        FuseAdjacentGeometryPredicates(LogicalOp::Or, {equal, near})[0]->Eval(
            SelectionView::All(5), ctx),
        {1, 1, 0, -1, 1});
}

TEST(ScalarMigrationTest, PatternErrorsKeepCodesAndRespectShortCircuit) {
    for (auto [op, text, code] :
         std::vector<std::tuple<proto::plan::OpType, std::string, ErrorCode>>{
             {proto::plan::Match, "abc\\", ExprInvalid},
             {proto::plan::RegexMatch, "[", InvalidParameter}}) {
        auto logical = std::make_shared<expr::UnaryRangeFilterExpr>(
            Field(DataType::VARCHAR), op, Text(text));
        EXPECT_TRUE(CompileExpressions({logical}).supported());
        EvaluateColumn({logical},
                       Column<std::string>(DataType::VARCHAR, {"a"}),
                       SelectionView::None(1));
        proto::plan::GenericValue no;
        no.set_bool_val(false);
        auto conjunction = std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And,
            std::make_shared<expr::ValueExpr>(no),
            logical);
        auto skipped =
            EvaluateColumn({conjunction},
                           Column<std::string>(DataType::VARCHAR, {"a"}),
                           SelectionView::All(1));
        ASSERT_EQ(skipped.size(), 1);
        EXPECT_FALSE(skipped[0].ValueAt<bool>(0));
        try {
            EvaluateColumn({logical},
                           Column<std::string>(DataType::VARCHAR, {"a"}),
                           SelectionView::All(1));
            FAIL();
        } catch (const SegcoreError& e) {
            EXPECT_EQ(e.get_error_code(), code);
        }
    }
}

TEST(ScalarMigrationTest, JsonExistsSharesRawSlotAndKeepsLegacyFalseSemantics) {
    auto exists =
        std::make_shared<expr::ExistsExpr>(Field(DataType::JSON, {"a/b", "~"}));
    auto project = std::make_shared<expr::ColumnExpr>(Field(DataType::JSON));
    auto compiled = CompileExpressions({exists, project});
    ASSERT_TRUE(compiled.supported());
    EXPECT_EQ(compiled.plan->inputs().slots().size(), 1);
    EXPECT_TRUE(compiled.plan->inputs().slots()[0].path.empty());
    std::vector<Json> values;
    for (auto text : {"{\"a/b\":{\"~\":7}}",
                      "{\"a/b\":{\"~\":null}}",
                      "{}",
                      "{\"a/b\":{\"~\":{}}}",
                      "{}"})
        values.emplace_back(simdjson::padded_string(std::string(text)));
    auto out = EvaluateColumn({exists, project},
                              Column<Json>(DataType::JSON, values, 4),
                              SelectionView::All(5));
    ASSERT_EQ(out.size(), 2);
    Expect(out[0], {1, 0, 0, 0, 0});
    EXPECT_FALSE(out[1].IsValid(4));
}

TEST(ScalarMigrationTest, JsonParsingOnlySelectedRowsAndOriginalFailureCode) {
    auto values =
        Column<Json>(DataType::JSON,
                     {Json(simdjson::padded_string(std::string("{\"x\":1}"))),
                      Json(simdjson::padded_string(std::string("{\"x\":\"") +
                                                   char(0xff) + "\"}"))});
    auto exists =
        std::make_shared<expr::ExistsExpr>(Field(DataType::JSON, {"x"}));
    auto rows = OwnedSelection::None(2);
    rows.SetSelected(0);
    auto out = EvaluateColumn({exists}, values, rows.view());
    ASSERT_EQ(out.size(), 1);
    EXPECT_TRUE(out[0].ValueAt<bool>(0));
    EXPECT_FALSE(out[0].defined_rows().IsSelected(1));
    try {
        EvaluateColumn({exists}, values, SelectionView::All(2));
        FAIL();
    } catch (const SegcoreError& e) {
        EXPECT_EQ(e.get_error_code(), DataFormatBroken);
    }
}

TEST(ScalarMigrationTest, JsonPredicateFailuresPreserveSelectionAndCodes) {
    const auto field = Field(DataType::JSON, {"x"});
    const std::vector<expr::TypedExprPtr> roots{
        std::make_shared<expr::UnaryRangeFilterExpr>(
            field, proto::plan::Equal, Int(1)),
        std::make_shared<expr::BinaryRangeFilterExpr>(
            field, Int(0), Int(2), false, false),
        std::make_shared<expr::TermFilterExpr>(
            field, std::vector{Int(1)}, false),
        std::make_shared<expr::JsonContainsExpr>(
            field,
            proto::plan::JSONContainsExpr_JSONOp_Contains,
            true,
            std::vector{Int(1)})};
    auto values =
        Column<Json>(DataType::JSON,
                     {Json(simdjson::padded_string(std::string("{\"x\":1}"))),
                      Json(simdjson::padded_string(std::string("{\"x\":\"") +
                                                   char(0xff) + "\"}"))});
    const auto rows = OwnedSelection::FromIndices(2, {0});
    for (const auto& root : roots) {
        auto output = EvaluateColumn({root}, values, rows.view());
        ASSERT_EQ(output.size(), 1);
        EXPECT_FALSE(output[0].defined_rows().IsSelected(1));
        try {
            EvaluateColumn({root}, values, SelectionView::All(2));
            FAIL() << "corrupt selected JSON must not be converted to FALSE";
        } catch (const SegcoreError& error) {
            EXPECT_EQ(error.get_error_code(), DataFormatBroken);
        }
    }
    for (auto [operation, pattern, code] :
         std::vector<std::tuple<proto::plan::OpType, std::string, ErrorCode>>{
             {proto::plan::Match, "abc\\", ExprInvalid},
             {proto::plan::RegexMatch, "[", InvalidParameter}}) {
        auto bad = std::make_shared<expr::UnaryRangeFilterExpr>(
            field, operation, Text(pattern));
        auto skipped = std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::Or,
            std::make_shared<expr::AlwaysTrueExpr>(),
            bad);
        auto output = EvaluateColumn({skipped}, values, SelectionView::All(2));
        ASSERT_EQ(output.size(), 1);
        Expect(output[0], {1, 1});
        try {
            EvaluateColumn({bad}, values, rows.view());
            FAIL();
        } catch (const SegcoreError& error) {
            EXPECT_EQ(error.get_error_code(), code);
        }
    }
}

TEST(ScalarMigrationTest, RoaringOwnsDecodedBitmapAndSignExtendsNarrowValues) {
    auto membership = Membership();
    auto logical = std::make_shared<expr::RoaringFilterExpr>(
        Field(DataType::INT8), membership);
    membership.reset();
    auto out = EvaluateColumn({logical},
                              Column<int8_t>(DataType::INT8, {-1, 7, 8, 0}, 3),
                              SelectionView::All(4));
    ASSERT_EQ(out.size(), 1);
    Expect(out[0], {1, 1, 0, -1});
    EXPECT_THROW(CompileExpressions({std::make_shared<expr::RoaringFilterExpr>(
                     Field(DataType::INT64), nullptr)}),
                 SegcoreError);
}

TEST(ScalarMigrationTest, TimestampCalendarAndPreEpochSemantics) {
    proto::plan::Interval second;
    second.set_seconds(1);
    EXPECT_EQ(scalar::ApplyTimestampInterval(-1, proto::plan::Add, second),
              999999);
    EXPECT_EQ(scalar::ApplyTimestampInterval(999999, proto::plan::Sub, second),
              -1);
    proto::plan::Interval month;
    month.set_months(1);
    // timegm normalization, not end-of-month clamping: 2024-01-31 + 1 month = March 2.
    EXPECT_EQ(scalar::ApplyTimestampInterval(
                  1706659200000000LL, proto::plan::Add, month),
              1709337600000000LL);
    auto logical = std::make_shared<expr::TimestamptzArithCompareExpr>(
        Field(DataType::TIMESTAMPTZ),
        proto::plan::Add,
        second,
        proto::plan::Equal,
        Int(999999));
    auto out =
        EvaluateColumn({logical},
                       Column<int64_t>(DataType::TIMESTAMPTZ, {-1, 0, 1}, 2),
                       SelectionView::All(3));
    ASSERT_EQ(out.size(), 1);
    Expect(out[0], {1, 0, -1});
}

TEST(ScalarMigrationTest, TimestampOverflowIsNotEvaluatedOutsideSelection) {
    proto::plan::Interval huge;
    huge.set_years(INT64_MAX);
    auto logical = std::make_shared<expr::TimestamptzArithCompareExpr>(
        Field(DataType::TIMESTAMPTZ),
        proto::plan::Add,
        huge,
        proto::plan::Equal,
        Int(0));
    auto column = Column<int64_t>(DataType::TIMESTAMPTZ, {0, 1}, 0);
    auto rows = OwnedSelection::None(2);
    rows.SetSelected(0);
    auto out = EvaluateColumn({logical}, column, rows.view());
    ASSERT_EQ(out.size(), 1);
    EXPECT_FALSE(out[0].IsValid(0));
    try {
        EvaluateColumn({logical}, column, SelectionView::All(2));
        FAIL();
    } catch (const SegcoreError& e) {
        EXPECT_EQ(e.get_error_code(), UnexpectedError);
    }
}

TEST(ScalarMigrationTest, TimestampAdaptiveProjectionSkipsOverflowingGaps) {
    constexpr size_t n = 1024;
    proto::plan::Interval second;
    second.set_seconds(1);
    auto logical = std::make_shared<expr::TimestamptzArithCompareExpr>(
        Field(DataType::TIMESTAMPTZ),
        proto::plan::Add,
        second,
        proto::plan::Equal,
        Int(1000000));
    for (bool clustered : {false, true}) {
        auto rows = OwnedSelection::None(n);
        std::vector<int64_t> values(n, INT64_MAX);
        for (size_t i = 0; i < n; ++i)
            if (clustered ? i < 100 : i % 10 == 0) {
                rows.SetSelected(i);
                values[i] = 0;
            }
        auto out =
            EvaluateColumn({logical},
                           Column<int64_t>(DataType::TIMESTAMPTZ, values, 0),
                           rows.view());
        ASSERT_EQ(out.size(), 1);
        rows.view().ForEachSelected([&](size_t i) {
            EXPECT_EQ(out[0].IsValid(i), i != 0);
            if (i)
                EXPECT_TRUE(out[0].ValueAt<bool>(i));
        });
        EXPECT_TRUE(out[0].defined_rows().Equals(rows.view()));
    }
}

TEST(ScalarMigrationTest, NewKernelsRespectShortCircuitAndEmptyRequests) {
    auto term = scalar::MakeTerm(
        std::make_shared<scalar::InputRefExpr>(0, DataType::INT64),
        Column<int64_t>(DataType::INT64, {1}));
    auto no = scalar::MakeLiteral(Column<bool>(DataType::BOOL, {false}));
    auto conjunction = scalar::MakeLogical(scalar::LogicalOp::And, {no, term});
    BatchInput input(
        std::make_shared<RowType>(std::vector<std::string>{"x"},
                                  std::vector<DataType>{DataType::INT64}),
        3);
    int reads = 0;
    input.BindLoader(0, [&](SelectionView) -> ReadOnlyVectorView {
        ++reads;
        ThrowInfo(FileReadFailed, "injected read failure");
    });
    scalar::EvalCtx ctx(input);
    Expect(conjunction->Eval(SelectionView::All(3), ctx), {0, 0, 0});
    term->Eval(SelectionView::None(3), ctx);
    EXPECT_EQ(reads, 0);
    try {
        term->Eval(SelectionView::All(3), ctx);
        FAIL();
    } catch (const SegcoreError& e) {
        EXPECT_EQ(e.get_error_code(), FileReadFailed);
    }
}
}  // namespace milvus::exec::plan

TEST(ScalarGeometryCacheMigrationTest, HitsMissesDuplicatesAndOwnerLifetime) {
    using namespace milvus;
    using namespace milvus::exec;
    namespace scalar = milvus::exec::scalar;
    auto schema = std::make_shared<const RowType>(
        std::vector<std::string>{"g"},
        std::vector<DataType>{DataType::GEOMETRY});
    auto cache = std::make_shared<SimpleGeometryCache>();
    auto point =
        Geometry(GetThreadLocalGEOSContext(), "POINT(0 0)").to_wkb_string();
    cache->AppendDataAt(7, point.data(), point.size());
    std::weak_ptr<SimpleGeometryCache> weak = cache;
    BatchInput input(schema, 4);
    input.BindGeometryLookup(0, [cache](size_t row) {
        return cache->GetByOffset(row == 0 || row == 2 ? 7 : row);
    });
    cache.reset();
    ASSERT_FALSE(weak.expired());
    std::vector<size_t> payloads;
    input.BindLoader(
        0,
        [&](SelectionView rows) {
            VectorResultBuilder values(DataType::GEOMETRY, 4);
            rows.ForEachSelected([&](size_t row) {
                payloads.push_back(row);
                values.Set<std::string>(row, point);
            });
            return std::move(values).Finish(rows);
        },
        [&](SelectionView rows) {
            auto valid = OwnedSelection::CopyOf(rows);
            valid.SetSelected(3, false);
            return ValidityResult{OwnedSelection::CopyOf(rows),
                                  std::move(valid)};
        });
    auto field = std::make_shared<scalar::InputRefExpr>(0, DataType::GEOMETRY);
    auto expression = scalar::MakeGeometryPredicate(
        field,
        proto::plan::GISFunctionFilterExpr_GISOp_Equals,
        "POINT(0 0)",
        0);
    scalar::EvalCtx ctx(input);
    auto result = expression->Eval(SelectionView::All(4), ctx);
    EXPECT_EQ(payloads, (std::vector<size_t>{1}));
    for (size_t row : {0, 1, 2}) EXPECT_TRUE(result.ValueAt<bool>(row));
    EXPECT_FALSE(result.IsValid(3));
}
