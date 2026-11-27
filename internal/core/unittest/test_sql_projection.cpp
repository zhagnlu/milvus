// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <tuple>

#include "absl/time/time.h"
#include "exec/expression/function/FunctionFactory.h"
#include "exec/operator/SqlProjection.h"
#include "query/PlanProto.h"
#include "segcore/SegmentGrowing.h"
#include "test_utils/DataGen.h"
#include "test_utils/storage_test_utils.h"

using namespace milvus;
using namespace milvus::segcore;

class SQLProjectionTest : public testing::TestWithParam<bool> {};

TEST_P(SQLProjectionTest, FilterBeforeComputedGrouping) {
    exec::expression::FunctionFactory::Instance().Initialize();
    auto schema = std::make_shared<Schema>();
    schema->AddDebugField(
        "vec", DataType::VECTOR_FLOAT, 4, knowhere::metric::L2);
    auto pk = schema->AddDebugField("pk", DataType::INT64);
    auto json = schema->AddDebugField("data", DataType::JSON);
    schema->set_primary_field_id(pk);
    constexpr int64_t row_count = 1030;  // crosses the raw JSON batch boundary
    auto data = DataGen(schema, row_count);
    for (auto& field : *data.raw_->mutable_fields_data()) {
        if (field.field_id() == pk.get()) {
            auto* values = field.mutable_scalars()->mutable_long_data();
            values->clear_data();
            for (int64_t i = 1; i <= row_count; ++i) values->add_data(i);
        }
        if (field.field_id() == json.get()) {
            auto* values = field.mutable_scalars()->mutable_json_data();
            values->clear_data();
            const char* inputs[] = {R"({"x":"9007199254740993"})",
                                    R"({"x":9007199254740993})",
                                    R"({"x":"9007199254740994"})",
                                    R"({"other":1})"};
            for (int64_t i = 0; i < row_count; ++i) {
                values->add_data(inputs[i % 4]);
            }
        }
    }
    SegmentInternalInterface* segment;
    SegmentGrowingPtr growing;
    SegmentSealedUPtr sealed;
    if (GetParam()) {
        growing = CreateGrowingSegment(schema, empty_index_meta);
        growing->PreInsert(row_count);
        growing->Insert(0,
                        row_count,
                        data.row_ids_.data(),
                        data.timestamps_.data(),
                        data.SharedRaw());
        segment = growing.get();
    } else {
        sealed = CreateSealedWithFieldDataLoaded(schema, data);
        segment = sealed.get();
    }
    proto::plan::PlanNode wire;
    auto* query = wire.mutable_query();
    query->set_limit(-1);
    query->mutable_predicates()->mutable_always_true_expr();
    auto* project = query->mutable_sql_project();
    project->set_version(1);
    auto* column = project->add_columns();
    column->set_id(10000);
    auto* cast = column->mutable_expression();
    cast->set_op("cast");
    cast->set_data_type(proto::schema::Int64);
    auto* path = cast->add_args();
    path->set_op("json_path");
    path->set_data_type(proto::schema::VarChar);
    path->add_json_path("x");
    auto* source = path->add_args();
    source->set_op("column");
    source->set_data_type(proto::schema::JSON);
    source->set_source_field_id(json.get());
    auto* predicate = project->mutable_predicate();
    predicate->set_op("is_not_null");
    predicate->set_data_type(proto::schema::Bool);
    *predicate->add_args() = *path;
    query->add_group_by_field_ids(10000);
    auto* count = query->add_aggregates();
    count->set_field_id(10000);
    count->set_op(proto::plan::count);
    query::ProtoParser parser(schema);
    auto plan = parser.CreateRetrievePlan(wire);
    auto result = segment->Retrieve(
        nullptr, plan.get(), MAX_TIMESTAMP, DEFAULT_MAX_OUTPUT_SIZE, false);
    ASSERT_EQ(result->fields_data_size(), 2);
    std::map<int64_t, int64_t> actual;
    const auto& keys = result->fields_data(0).scalars().long_data();
    const auto& counts = result->fields_data(1).scalars().long_data();
    for (int i = 0; i < keys.data_size(); ++i) {
        actual[keys.data(i)] = counts.data(i);
    }
    EXPECT_EQ(actual,
              (std::map<int64_t, int64_t>{{9007199254740993LL, 516},
                                          {9007199254740994LL, 257}}));

    OpContext cancelled;
    folly::CancellationSource cancellation;
    cancelled.cancellation_token = cancellation.getToken();
    cancellation.requestCancellation();
    try {
        exec::EvaluateSqlProjection(
            *project, {FieldId(10000)}, {0}, segment, &cancelled);
        FAIL() << "SQL projection ignored cancellation";
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), ErrorCode::FollyCancel);
    }

    proto::plan::SQLProject invalid;
    invalid.set_version(1);
    auto* invalid_column = invalid.add_columns();
    invalid_column->set_id(10000);
    auto* divide = invalid_column->mutable_expression();
    divide->set_op("/");
    divide->set_data_type(proto::schema::Int64);
    for (int64_t value : {1, 0}) {
        auto* literal = divide->add_args();
        literal->set_op("literal");
        literal->set_data_type(proto::schema::Int64);
        literal->mutable_literal()->set_int64_val(value);
    }
    try {
        exec::EvaluateSqlProjection(
            invalid, {FieldId(10000)}, {0}, segment, nullptr);
        FAIL() << "SQL projection accepted division by zero";
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), ErrorCode::ExprInvalid);
    }
}

// Binding must not eagerly evaluate unreachable operands or cache row values.
TEST_P(SQLProjectionTest, PreparedOperatorsPreserveLazyEvaluation) {
    auto schema = std::make_shared<Schema>();
    schema->AddDebugField(
        "vec", DataType::VECTOR_FLOAT, 4, knowhere::metric::L2);
    auto pk = schema->AddDebugField("pk", DataType::INT64);
    schema->set_primary_field_id(pk);
    auto data = DataGen(schema, 2);
    SegmentGrowingPtr growing;
    SegmentSealedUPtr sealed;
    SegmentInternalInterface* segment;
    if (GetParam()) {
        growing = CreateGrowingSegment(schema, empty_index_meta);
        growing->PreInsert(2);
        growing->Insert(0,
                        2,
                        data.row_ids_.data(),
                        data.timestamps_.data(),
                        data.SharedRaw());
        segment = growing.get();
    } else {
        sealed = CreateSealedWithFieldDataLoaded(schema, data);
        segment = sealed.get();
    }
    auto integer = [](int64_t value) {
        proto::plan::SQLScalarExpr expr;
        expr.set_op("literal");
        expr.set_data_type(proto::schema::Int64);
        expr.mutable_literal()->set_int64_val(value);
        return expr;
    };
    auto evaluate = [&](const proto::plan::SQLScalarExpr& expr,
                        const std::string& timezone = "UTC") {
        proto::plan::SQLProject project;
        project.set_version(1);
        project.set_timezone(timezone);
        auto* column = project.add_columns();
        column->set_id(10000);
        *column->mutable_expression() = expr;
        return exec::EvaluateSqlProjection(
            project, {FieldId(10000)}, {0, 1}, segment, nullptr);
    };
    for (const auto& [op, expected] :
         std::vector<std::pair<std::string, int64_t>>{
             {"+", 10}, {"-", 6}, {"*", 16}, {"/", 4}}) {
        proto::plan::SQLScalarExpr expr;
        expr.set_op(op);
        expr.set_data_type(proto::schema::Int64);
        *expr.add_args() = integer(8);
        *expr.add_args() = integer(2);
        auto result = evaluate(expr);
        auto column = std::dynamic_pointer_cast<ColumnVector>(result->child(0));
        EXPECT_EQ(column->RawAsValues<int64_t>()[0], expected) << op;
        EXPECT_EQ(column->RawAsValues<int64_t>()[1], expected) << op;
    }
    for (const auto& op : {"and", "or"}) {
        proto::plan::SQLScalarExpr expr;
        expr.set_op(op);
        expr.set_data_type(proto::schema::Bool);
        auto* left = expr.add_args();
        left->set_op("literal");
        left->set_data_type(proto::schema::Bool);
        const bool expected = std::string(op) == "or";
        left->mutable_literal()->set_bool_val(expected);
        auto* unreachable = expr.add_args();
        unreachable->set_op("unsupported");
        unreachable->set_data_type(proto::schema::Bool);
        auto result = evaluate(expr);
        auto column = std::dynamic_pointer_cast<ColumnVector>(result->child(0));
        EXPECT_EQ(column->RawAsValues<bool>()[0], expected);
        EXPECT_EQ(column->RawAsValues<bool>()[1], expected);
    }
    proto::plan::SQLScalarExpr extract;
    extract.set_op("extract");
    extract.set_data_type(proto::schema::Double);
    *extract.add_args() = integer(-500000);
    extract.mutable_args(0)->set_data_type(proto::schema::Timestamptz);
    for (const auto& [unit, timezone, expected] :
         std::vector<std::tuple<std::string, std::string, double>>{
             {"hour", "UTC", 23},
             {"hour", "Asia/Shanghai", 7},
             {"second", "UTC", 59.5},
             {"epoch", "Invalid/Zone", -0.5}}) {
        extract.set_time_unit(unit);
        auto result = evaluate(extract, timezone);
        auto column = std::dynamic_pointer_cast<ColumnVector>(result->child(0));
        EXPECT_DOUBLE_EQ(column->RawAsValues<double>()[0], expected);
        EXPECT_DOUBLE_EQ(column->RawAsValues<double>()[1], expected);
    }
    // Compare with the independent civil-time API across UTC, aliases and DST
    // zones. Exercise day/subsecond boundaries, negative epochs and the entire
    // INT64 microsecond range, rather than only current positive timestamps.
    std::vector<int64_t> instants{
        std::numeric_limits<int64_t>::min(),
        std::numeric_limits<int64_t>::max(),
        -86400000001LL,
        -86400000000LL,
        -86399999999LL,
        -60000001,
        -1000001,
        -1000000,
        -999999,
        -1,
        0,
        1,
        999999,
        1000000,
        59999999,
        60000000,
        86399999999LL,
        86400000000LL,
        1710053999999999LL,  // US spring DST transition
        1710054000000000LL,
        1730613599999999LL,  // US fall DST transition
        1730613600000000LL};
    uint64_t seed = 1;
    for (int i = 0; i < 64; ++i) {
        seed = seed * 6364136223846793005ULL + 1;
        const auto magnitude =
            static_cast<int64_t>(seed & uint64_t(INT64_MAX));
        instants.push_back(i % 2 ? magnitude : -magnitude);
    }
    for (const auto& name : {"UTC", "", "Etc/UTC", "Asia/Shanghai",
                             "America/New_York", "Europe/Paris"}) {
        absl::TimeZone zone;
        ASSERT_TRUE(absl::LoadTimeZone(*name ? name : "UTC", &zone));
        for (const auto micros : instants) {
            extract.mutable_args(0)->mutable_literal()->set_int64_val(micros);
            const auto parts = zone.At(absl::FromUnixMicros(micros)).cs;
            auto remainder = micros % 1000000;
            if (remainder < 0)
                remainder += 1000000;
            for (const auto& [unit, expected] :
                 std::vector<std::pair<std::string, double>>{
                     {"hour", double(parts.hour())},
                     {"minute", double(parts.minute())},
                     {"second", parts.second() + remainder / 1000000.0}}) {
                extract.set_time_unit(unit);
                auto result = evaluate(extract, name);
                auto column =
                    std::dynamic_pointer_cast<ColumnVector>(result->child(0));
                for (int row = 0; row < 2; ++row) {
                    EXPECT_DOUBLE_EQ(column->RawAsValues<double>()[row],
                                     expected)
                        << name << " " << unit << " at " << micros;
                }
            }
        }
    }
    extract.set_time_unit("hour");
    try {
        evaluate(extract, "Invalid/Zone");
        FAIL() << "SQL extract accepted invalid timezone";
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), ErrorCode::ExprInvalid);
    }
    extract.mutable_args(0)->clear_literal();
    auto null_extract = evaluate(extract, "Invalid/Zone");
    auto null_extract_column =
        std::dynamic_pointer_cast<ColumnVector>(null_extract->child(0));
    EXPECT_FALSE(null_extract_column->ValidAt(0));
    auto null = integer(0);
    null.clear_literal();
    auto result = evaluate(null);
    auto column = std::dynamic_pointer_cast<ColumnVector>(result->child(0));
    EXPECT_FALSE(column->ValidAt(0));
    EXPECT_FALSE(column->ValidAt(1));
}

INSTANTIATE_TEST_SUITE_P(Segments, SQLProjectionTest, testing::Bool());
