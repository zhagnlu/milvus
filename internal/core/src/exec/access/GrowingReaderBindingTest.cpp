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
#include "exec/access/SegmentReaderInput.h"
#include "exec/access/ReaderInput.h"
#include "exec/operator/FilterBitsExecution.h"
#include "exec/plan/ExprCompiler.h"
#include "exec/plan/ExecutionPreparer.h"
#include "segcore/InsertRecord.h"

namespace milvus::exec::access {
namespace {
using namespace segcore::reader;

struct GrowingSource {
    SchemaPtr schema = std::make_shared<Schema>();
    FieldId field{0};
    std::unique_ptr<segcore::InsertRecordGrowing> record;

    explicit GrowingSource(DataType type) {
        auto pk = schema->AddDebugField("pk", DataType::INT64);
        schema->set_primary_field_id(pk);
        field = schema->AddDebugField("value", type, true);
        record = std::make_unique<segcore::InsertRecordGrowing>(*schema, 3);
        DataArray validity;
        for (bool valid : {true, false, true, true, false})
            validity.add_valid_data(valid);
        record->get_valid_data(field)->set_data_raw(
            0, 5, &validity, (*schema)[field]);
    }

    template <typename T>
    void
    Fill(const T* values) {
        record->get_data<T>(field)->set_data_raw(0, values, 5);
        record->ack_responder_.AddSegment(0, 5);
    }

    InputDependency
    Dependency() const {
        return {"", field.get(), (*schema)[field].get_data_type()};
    }
};
}  // namespace

TEST(GrowingReaderBindingTest, EveryOrdinaryScalarSourceRetainsBoundPrefix) {
    auto check = []<typename T>(DataType type) {
        GrowingSource source(type);
        const T raw[] = {T(0), T(1), T(1), T(0), T(1)};
        source.Fill(raw);
        auto bind = MakeGrowingReaderBinding(*source.record, source.schema, 4);
        auto reader = bind(source.Dependency());
        ASSERT_NE(reader, nullptr);
        EXPECT_EQ(reader->row_count(), 4);
        EXPECT_EQ(reader->type(), type);
        bind = {};
        source.record.reset();
        source.schema.reset();
        auto result =
            reader->Take(TakeRequest({3, 1, 2, 3},
                                     ReadProjection::ValuesAndValidity,
                                     SelectionView::All(4)),
                         {});
        EXPECT_EQ(result.values->ValueAt<T>(0), raw[3]);
        EXPECT_FALSE(result.values->IsValid(1));
        EXPECT_EQ(result.values->ValueAt<T>(2), raw[2]);
        EXPECT_EQ(result.values->ValueAt<T>(3), raw[3]);
        EXPECT_THROW(reader->Take(TakeRequest({4},
                                              ReadProjection::ValidityOnly,
                                              SelectionView::All(1)),
                                  {}),
                     SegcoreError);
    };
    check.template operator()<bool>(DataType::BOOL);
    check.template operator()<int8_t>(DataType::INT8);
    check.template operator()<int16_t>(DataType::INT16);
    check.template operator()<int32_t>(DataType::INT32);
    check.template operator()<int64_t>(DataType::INT64);
    check.template operator()<int64_t>(DataType::TIMESTAMPTZ);
    check.template operator()<float>(DataType::FLOAT);
    check.template operator()<double>(DataType::DOUBLE);
}

TEST(GrowingReaderBindingTest,
     StringsAndJsonRetainPayloadAfterRecordDestruction) {
    GrowingSource strings(DataType::VARCHAR), json(DataType::JSON);
    const std::string raw[] = {
        "北京", "unused", std::string(4096, 'a'), "tail", "null"};
    const Json documents[] = {Json(simdjson::padded_string("{\"x\":1}", 7)),
                              Json{},
                              Json{},
                              Json{},
                              Json{}};
    strings.Fill(raw);
    json.Fill(documents);
    auto string_reader = MakeGrowingReaderBinding(
        *strings.record, strings.schema, 5)(strings.Dependency());
    auto json_reader = MakeGrowingReaderBinding(
        *json.record, json.schema, 5)(json.Dependency());
    strings.record.reset();
    json.record.reset();
    auto rows = OwnedSelection::FromIndices(5, {0, 2});
    auto result = string_reader->ReadRange(
        RangeRequest(0, 5, ReadProjection::ValuesAndValidity, rows.view()), {});
    EXPECT_EQ(result.values->StringViewAt(0), raw[0]);
    EXPECT_EQ(result.values->StringViewAt(2), raw[2]);
    auto selected = OwnedSelection::FromIndices(5, {0});
    auto doc = json_reader->ReadRange(
        RangeRequest(0, 5, ReadProjection::ValuesAndValidity, selected.view()),
        {});
    EXPECT_TRUE(doc.values->ValueAt<Json>(0).exist("/x"));
}

TEST(GrowingReaderBindingTest, AckAndPublishedSchemaAreRequired) {
    GrowingSource source(DataType::INT64);
    const int64_t raw[] = {0, 1, 2, 3, 4};
    // Allocated/filled capacity does not imply acknowledged rows.
    source.record->get_data<int64_t>(source.field)->set_data_raw(0, raw, 5);
    EXPECT_THROW(MakeGrowingReaderBinding(*source.record, source.schema, 5),
                 SegcoreError);
    source.record->ack_responder_.AddSegment(0, 5);
    auto bind = MakeGrowingReaderBinding(*source.record, source.schema, 5);
    auto wrong = source.Dependency();
    wrong.type = DataType::DOUBLE;
    EXPECT_THROW(bind(wrong), SegcoreError);
    auto unpublished = std::make_shared<Schema>();
    EXPECT_THROW(MakeGrowingReaderBinding(
                     *source.record, unpublished, 5)(source.Dependency()),
                 SegcoreError);
    auto unsupported = source.Dependency();
    unsupported.element_level = true;
    EXPECT_EQ(bind(unsupported), nullptr);
    unsupported = source.Dependency();
    unsupported.path = {"nested"};
    EXPECT_EQ(bind(unsupported), nullptr);
    unsupported = source.Dependency();
    unsupported.relation = "another";
    EXPECT_EQ(bind(unsupported), nullptr);
}

TEST(GrowingReaderBindingTest, FilterAndProjectionShareOneSlotAndMissingRows) {
    GrowingSource source(DataType::INT64);
    const int64_t raw[] = {0, 1, 2, 3, 4};
    source.Fill(raw);
    expr::ColumnInfo field(source.field, DataType::INT64, {}, true);
    proto::plan::GenericValue bound;
    bound.set_int64_val(2);
    auto predicate = std::make_shared<expr::UnaryRangeFilterExpr>(
        field, proto::plan::GreaterEqual, bound);
    auto projection = std::make_shared<expr::ColumnExpr>(field);
    auto compiled = plan::CompileExpressions({predicate, projection});
    ASSERT_TRUE(compiled.supported());
    ASSERT_EQ(compiled.plan->inputs().slots().size(), 1);
    auto bind = MakeGrowingReaderBinding(*source.record, source.schema, 5);
    int bindings = 0, checkpoints = 0;
    auto context = std::make_shared<ReadContext>();
    context->checkpoint = [&] { ++checkpoints; };
    auto prepared = plan::Prepare(compiled.plan, [&](const auto& dependency) {
        ++bindings;
        return PrepareReaderInput(bind(dependency), context);
    });
    EXPECT_EQ(bindings, 1);
    EXPECT_EQ(checkpoints, 0);
    bind = {};
    source.record.reset();
    auto batch = prepared.MakeExecutionBatch(ReadBatchBinding::Range(0, 5));
    auto selected = OwnedSelection::FromIndices(5, {0, 2, 3});
    auto results = batch->Eval(selected.view());
    EXPECT_FALSE(results[0].ValueAt<bool>(0));
    EXPECT_TRUE(results[0].ValueAt<bool>(2));
    EXPECT_EQ(results[1].ValueAt<int64_t>(3), 3);
    const auto reads = checkpoints;
    batch->Eval(selected.view());
    EXPECT_EQ(checkpoints, reads);  // Both roots reuse loaded values.
    auto all = batch->Eval(SelectionView::All(5));
    EXPECT_FALSE(all[0].IsValid(1));
    EXPECT_FALSE(all[1].IsValid(4));
    EXPECT_GT(checkpoints, reads);
}
TEST(GrowingReaderBindingTest, ArithmeticAndJsonPredicatesUseOwnedReaders) {
    auto evaluate = [](GrowingSource& source,
                       const std::vector<expr::TypedExprPtr>& roots) {
        auto compiled = plan::CompileExpressions(roots);
        EXPECT_TRUE(compiled.supported());
        if (!compiled.supported())
            return std::vector<ReadOnlyVectorView>{};
        auto bind = MakeGrowingReaderBinding(*source.record, source.schema, 5);
        auto prepared =
            plan::Prepare(compiled.plan, [&](const auto& dependency) {
                return PrepareReaderInput(bind(dependency),
                                          std::make_shared<ReadContext>());
            });
        bind = {};
        source.record.reset();
        return prepared
            .MakeExecutionBatch(ReadBatchBinding::Offsets({3, 1, 0, 2, 0}))
            ->Eval(SelectionView::All(5));
    };
    auto integer = [](int64_t n) {
        proto::plan::GenericValue value;
        value.set_int64_val(n);
        return value;
    };
    GrowingSource numbers(DataType::INT8);
    const int8_t raw[] = {127, 0, 6, -6, 0};
    numbers.Fill(raw);
    auto arithmetic = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
        expr::ColumnInfo(numbers.field, DataType::INT8, {}, true),
        proto::plan::Equal,
        proto::plan::Add,
        integer(128),
        integer(1));
    auto numeric = evaluate(numbers, {arithmetic});
    ASSERT_EQ(numeric.size(), 1);
    const std::vector<int> numeric_expected{0, -1, 1, 0, 1};
    for (size_t row = 0; row < 5; ++row)
        EXPECT_EQ(
            numeric[0].IsValid(row) ? int(numeric[0].ValueAt<bool>(row)) : -1,
            numeric_expected[row]);

    GrowingSource jsons(DataType::JSON);
    std::vector<Json> raw_json;
    for (const auto* text : {"{\"x\":11,\"tags\":[1]}",
                             "{}",
                             "{\"x\":100,\"tags\":[]}",
                             "{\"x\":50,\"tags\":[1.0]}",
                             "{}"})
        raw_json.emplace_back(simdjson::padded_string(std::string(text)));
    jsons.Fill(raw_json.data());
    auto range = std::make_shared<expr::BinaryRangeFilterExpr>(
        expr::ColumnInfo(jsons.field, DataType::JSON, {"x"}, true),
        integer(10),
        integer(100),
        false,
        false);
    auto contains = std::make_shared<expr::JsonContainsExpr>(
        expr::ColumnInfo(jsons.field, DataType::JSON, {"tags"}, true),
        proto::plan::JSONContainsExpr_JSONOp_Contains,
        true,
        std::vector{integer(1)});
    auto json = evaluate(jsons, {range, contains});
    ASSERT_EQ(json.size(), 2);
    const std::vector<int> expected{1, -1, 1, 0, 1};
    for (const auto& root : json)
        for (size_t row = 0; row < 5; ++row)
            EXPECT_EQ(root.IsValid(row) ? int(root.ValueAt<bool>(row)) : -1,
                      expected[row]);
}
}  // namespace milvus::exec::access

namespace milvus::exec::access {
TEST(GrowingReaderBindingTest, TextSourceRetainsPrefixAndNulls) {
    GrowingSource source(DataType::TEXT);
    const std::string raw[] = {
        "北京", "unused", std::string(8192, 'x'), "tail", "null"};
    source.Fill(raw);
    auto bind = MakeGrowingReaderBinding(*source.record, source.schema, 4);
    auto reader = bind(source.Dependency());
    ASSERT_TRUE(reader);
    bind = {};
    source.record.reset();
    auto result =
        reader->Take(segcore::reader::TakeRequest(
                         {3, 1, 2, 3},
                         segcore::reader::ReadProjection::ValuesAndValidity,
                         SelectionView::All(4)),
                     {});
    EXPECT_EQ(result.values->logical_type(), DataType::TEXT);
    EXPECT_FALSE(result.values->IsValid(1));
    EXPECT_EQ(result.values->StringViewAt(2), raw[2]);
    EXPECT_EQ(result.values->StringViewAt(3), raw[3]);
}
}  // namespace milvus::exec::access
