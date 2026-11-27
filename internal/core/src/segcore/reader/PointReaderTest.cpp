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
#include <thread>
#include "exec/access/ReaderInput.h"
#include "exec/operator/FilterBitsExecution.h"
#include "exec/plan/ExprCompiler.h"
#include "exec/plan/ExecutionPreparer.h"
#include "segcore/reader/ReaderTestUtils.h"
#include "segcore/reader/GrowingColumnReader.h"
#include "segcore/reader/IndexColumnReader.h"

namespace milvus::segcore::reader {
namespace {
using namespace milvus::exec;
constexpr auto values = ReadProjection::ValuesAndValidity;
constexpr auto validity = ReadProjection::ValidityOnly;
template <typename T>
std::shared_ptr<ConcurrentVector<T>>
Growing(const std::vector<T>& data) {
    auto column = std::make_shared<ConcurrentVector<T>>(3);
    column->set_data_raw(0, data.data(), data.size());
    return column;
}
template <typename T>
class ObservedIndex : public milvus::segcore::reader::TestScalarIndex<T> {
 public:
    mutable int lookups = 0;
    mutable int validity_reads = 0;
    bool fail = false;
    bool fail_validity = false;
    std::optional<T>
    Lookup(int64_t row) const override {
        ++lookups;
        if (fail)
            ThrowInfo(FileReadFailed, "injected reverse lookup failure");
        return milvus::segcore::reader::TestScalarIndex<T>::Lookup(row);
    }
    TargetBitmap
    IsNotNull() const override {
        ++validity_reads;
        if (fail_validity)
            ThrowInfo(FileReadFailed, "injected index validity failure");
        return milvus::segcore::reader::TestScalarIndex<T>::IsNotNull();
    }
};
proto::plan::GenericValue
Int(int64_t value) {
    proto::plan::GenericValue v;
    v.set_int64_val(value);
    return v;
}
}  // namespace

TEST(PointReaderTest, GrowingHandleOutlivesUniqueColumnAndValidityOwner) {
    auto valid = std::make_shared<ThreadSafeValidData>(3);
    DataArray data;
    for (bool value : {true, false, true, true}) data.add_valid_data(value);
    FieldMeta meta(
        FieldName("x"), FieldId(100), DataType::INT64, true, std::nullopt);
    valid->set_data_raw(0, 4, &data, meta);
    std::weak_ptr<const ThreadSafeValidData> weak = valid;
    auto column =
        std::make_unique<ConcurrentVector<int64_t>>(3, nullptr, valid);
    int64_t raw[] = {10, 20, 30, 40};
    column->set_data_raw(0, raw, 4);
    auto reader = std::make_unique<GrowingColumnReader<int64_t>>(
        *column, DataType::INT64, 4);
    column.reset();
    valid.reset();
    EXPECT_FALSE(weak.expired());
    auto result = reader->Take(
        TakeRequest({3, 1, 3, 0}, values, SelectionView::All(4)), {});
    EXPECT_EQ(result.values->ValueAt<int64_t>(0), 40);
    EXPECT_FALSE(result.values->IsValid(1));
    EXPECT_EQ(result.values->ValueAt<int64_t>(2), 40);
    reader.reset();
    EXPECT_FALSE(weak.expired());
    EXPECT_EQ(result.values->ValueAt<int64_t>(3), 10);
    result.values.reset();
    EXPECT_TRUE(weak.expired());
}

TEST(PointReaderTest, GrowingHandlePinsMmapStringsUntilLastViewRelease) {
    auto manager = storage::MmapManager::GetInstance().GetMmapChunkManager();
    auto descriptor = manager->Register();
    std::weak_ptr<storage::MmapChunkDescriptor> weak = descriptor;
    auto column =
        std::make_unique<ConcurrentVector<std::string>>(3, descriptor);
    const std::string raw[] = {"北京", std::string(4096, 'x'), "tail", "last"};
    column->set_data_raw(0, raw, 4);
    auto reader = std::make_unique<GrowingColumnReader<std::string>>(
        *column, DataType::VARCHAR, 4);
    column.reset();
    descriptor.reset();
    ASSERT_FALSE(weak.expired());
    auto result = reader->Take(
        TakeRequest({3, 0, 1, 0}, values, SelectionView::All(4)), {});
    EXPECT_EQ(result.values->StringViewAt(0), "last");
    EXPECT_EQ(result.values->StringViewAt(1), raw[0]);
    EXPECT_EQ(result.values->StringViewAt(2), raw[1]);
    reader.reset();
    EXPECT_FALSE(weak.expired());
    EXPECT_EQ(result.values->StringViewAt(3), raw[0]);
    result.values.reset();
    EXPECT_TRUE(weak.expired());
}

TEST(PointReaderTest, GrowingTruncatedValidityIsFailureNotNull) {
    auto valid = std::make_shared<ThreadSafeValidData>(3);
    auto column =
        std::make_unique<ConcurrentVector<int64_t>>(3, nullptr, valid);
    int64_t raw[] = {1, 2, 3};
    column->set_data_raw(0, raw, 3);
    GrowingColumnReader<int64_t> reader(*column, DataType::INT64, 3);
    try {
        reader.ReadRange(RangeRequest(0, 3, validity, SelectionView::All(3)),
                         {});
        FAIL() << "unpublished validity became NULL";
    } catch (const SegcoreError& e) {
        EXPECT_EQ(e.get_error_code(), UnexpectedError);
    }
}

TEST(PointReaderTest, GrowingBoundFilterRetriesWithoutRetainingBinder) {
    auto column = Growing<int64_t>({0, 1, 2, 3, 4, 5});
    int binds = 0;
    access::BindColumnReader bind = [&](const InputDependency& input) {
        ++binds;
        EXPECT_EQ(input.field_id, 100);
        return std::make_shared<GrowingColumnReader<int64_t>>(
            *column, input.type, 6);
    };
    auto predicate = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(FieldId(100), DataType::INT64),
        proto::plan::GreaterEqual,
        Int(3));
    auto context = std::make_shared<ReadContext>();
    int checks = 0;
    bool fail = true;
    context->checkpoint = [&] {
        if (++checks == 12 && fail)
            ThrowInfo(FileReadFailed, "injected growing checkpoint failure");
    };
    auto execution = FilterBitsExecution::TryCreateWithBinding(
        predicate, nullptr, bind, 6, 2, context);
    ASSERT_NE(execution, nullptr);
    EXPECT_EQ(binds, 1);
    EXPECT_EQ(checks, 0);  // Preparation captures metadata, never reads values.
    bind = {};
    column.reset();
    try {
        execution->GetOutput();
        FAIL() << "execution error became fallback";
    } catch (const SegcoreError& e) {
        EXPECT_EQ(e.get_error_code(), FileReadFailed);
    }
    EXPECT_FALSE(execution->IsFinished());
    fail = false;
    auto output = ReadOnlyVectorView::CopyOf(execution->GetOutput()->child(0));
    for (size_t row = 0; row < 6; ++row) {
        EXPECT_TRUE(output.IsValid(row));
        EXPECT_EQ(output.ValueAt<bool>(row), row < 3);
    }
    EXPECT_EQ(execution->GetOutput(), nullptr);
}

TEST(PointReaderTest, GrowingRangeTakeDuplicatesAndNullablePrefix) {
    auto valid = std::make_shared<ThreadSafeValidData>(3);
    DataArray data;
    for (bool value : {true, false, true, true, true, false})
        data.add_valid_data(value);
    FieldMeta meta(
        FieldName("x"), FieldId(100), DataType::INT64, true, std::nullopt);
    valid->set_data_raw(0, 6, &data, meta);
    auto column =
        std::make_shared<ConcurrentVector<int64_t>>(3, nullptr, valid);
    int64_t raw[] = {1, 2, 3, 4, 5, 6};
    column->set_data_raw(0, raw, 6);
    GrowingColumnReader<int64_t> reader(column, DataType::INT64, 6);
    auto result = reader.Take(
        TakeRequest({4, 1, 4, 2}, values, SelectionView::All(4)), {});
    result.Validate(DataType::INT64, SelectionView::All(4), values);
    EXPECT_EQ(result.values->ValueAt<int64_t>(0), 5);
    EXPECT_FALSE(result.values->IsValid(1));
    EXPECT_EQ(result.values->ValueAt<int64_t>(2), 5);
    EXPECT_EQ(result.values->ValueAt<int64_t>(3), 3);
    auto range =
        reader.ReadRange(RangeRequest(2, 4, values, SelectionView::All(4)), {});
    EXPECT_EQ(range.values->ValueAt<int64_t>(0), 3);
    EXPECT_FALSE(range.values->IsValid(3));
    auto only = reader.ReadRange(
        RangeRequest(0, 6, validity, SelectionView::All(6)), {});
    EXPECT_FALSE(only.values.has_value());
    EXPECT_FALSE(only.valid.view().IsSelected(1));
}

TEST(PointReaderTest, GrowingPinsGenerationAcrossAppendAndClear) {
    auto column = Growing<int64_t>({0, 1, 2, 3, 4, 5});
    GrowingColumnReader<int64_t> reader(column, DataType::INT64, 6);
    std::thread writer([&] {
        for (int64_t i = 6; i < 100; ++i) column->set_data_raw(i, &i, 1);
    });
    for (int i = 0; i < 20; ++i) {
        auto out = reader.Take(
            TakeRequest({5, 0, 5}, values, SelectionView::All(3)), {});
        EXPECT_EQ(out.values->ValueAt<int64_t>(0), 5);
        EXPECT_EQ(out.values->ValueAt<int64_t>(1), 0);
    }
    writer.join();
    EXPECT_THROW(
        reader.Take(TakeRequest({6}, values, SelectionView::All(1)), {}),
        SegcoreError);
    column->clear();
    int64_t replacement[] = {100, 101, 102};
    column->set_data_raw(0, replacement, 3);
    auto retained =
        reader.ReadRange(RangeRequest(0, 6, values, SelectionView::All(6)), {});
    EXPECT_EQ(retained.values->ValueAt<int64_t>(5), 5);
    column.reset();
    EXPECT_EQ(retained.values->ValueAt<int64_t>(1), 1);
}

TEST(PointReaderTest, GrowingStringsJsonAndResultOwnership) {
    auto strings =
        Growing<std::string>({"北京", std::string(1000, 'x'), "上海", "tail"});
    auto reader = std::make_unique<GrowingColumnReader<std::string>>(
        strings, DataType::VARCHAR, 4);
    auto out =
        reader->Take(TakeRequest({1, 0, 1}, values, SelectionView::All(3)), {});
    strings->clear();
    strings.reset();
    reader.reset();
    EXPECT_EQ(out.values->StringViewAt(1), "北京");
    EXPECT_EQ(out.values->StringViewAt(2).size(), 1000);
    auto json = Growing<Json>(
        {Json(simdjson::padded_string(std::string("{\"x\":1}")))});
    auto json_reader =
        std::make_unique<GrowingColumnReader<Json>>(json, DataType::JSON, 1);
    auto parsed = json_reader->ReadRange(
        RangeRequest(0, 1, values, SelectionView::All(1)), {});
    json->clear();
    json.reset();
    json_reader.reset();
    EXPECT_TRUE(parsed.values->ValueAt<Json>(0).exist("/x"));
}

TEST(PointReaderTest, GrowingScalarPhysicalRepresentations) {
    auto run = []<typename T>(DataType type) {
        auto column = std::make_shared<ConcurrentVector<T>>(2);
        T raw[] = {T(0), T(1), T(1), T(0)};
        column->set_data_raw(0, raw, 4);
        GrowingColumnReader<T> reader(column, type, 4);
        auto out = reader.Take(
            TakeRequest({2, 0, 2}, values, SelectionView::All(3)), {});
        EXPECT_EQ(out.values->template ValueAt<T>(0), T(1));
        EXPECT_EQ(out.values->template ValueAt<T>(1), T(0));
        EXPECT_EQ(out.values->template ValueAt<T>(2), T(1));
    };
    run.template operator()<bool>(DataType::BOOL);
    run.template operator()<int8_t>(DataType::INT8);
    run.template operator()<int16_t>(DataType::INT16);
    run.template operator()<int32_t>(DataType::INT32);
    run.template operator()<int64_t>(DataType::INT64);
    run.template operator()<int64_t>(DataType::TIMESTAMPTZ);
    run.template operator()<float>(DataType::FLOAT);
    run.template operator()<double>(DataType::DOUBLE);
}

TEST(PointReaderTest, GrowingBoundsEmptyAndCancellation) {
    auto column = Growing<int64_t>({1, 2, 3});
    EXPECT_THROW((GrowingColumnReader<int64_t>(column, DataType::FLOAT, 3)),
                 SegcoreError);
    EXPECT_THROW((GrowingColumnReader<int64_t>(column, DataType::INT64, 4)),
                 SegcoreError);
    GrowingColumnReader<int64_t> reader(column, DataType::INT64, 3);
    int checks = 0;
    ReadContext context;
    context.checkpoint = [&] {
        ++checks;
        ThrowInfo(FollyCancel, "injected checkpoint cancellation");
    };
    reader.ReadRange(RangeRequest(0, 3, values, SelectionView::None(3)),
                     context);
    EXPECT_EQ(checks, 0);
    EXPECT_THROW(
        reader.Take(TakeRequest({3}, values, SelectionView::None(1)), context),
        SegcoreError);
    EXPECT_EQ(checks, 0);
    try {
        reader.ReadRange(RangeRequest(0, 3, values, SelectionView::All(3)),
                         context);
        FAIL();
    } catch (const SegcoreError& e) {
        EXPECT_EQ(e.get_error_code(), FollyCancel);
    }
}

TEST(PointReaderTest, IndexReverseLookupValidityOnlyAndIncrementalInput) {
    auto index = std::make_shared<ObservedIndex<int64_t>>();
    int64_t raw[] = {1, 2, 3, 4};
    bool valid[] = {true, false, true, true};
    index->Build(4, raw, valid);
    auto reader =
        std::make_shared<IndexColumnReader<int64_t>>(index, DataType::INT64, 4);
    auto context = std::make_shared<ReadContext>();
    auto source = access::PrepareReaderInput(reader, context);
    auto loaders = source.make_loaders(ReadBatchBinding::Offsets({3, 1, 3, 0}));
    BatchInput input(
        std::make_shared<RowType>(std::vector<std::string>{"x"},
                                  std::vector<DataType>{DataType::INT64}),
        4);
    input.BindLoader(0, loaders.values, loaders.validity);
    input.GetValidity(0, SelectionView::All(4));
    EXPECT_EQ(index->lookups, 0);
    auto first = OwnedSelection::None(4);
    first.SetSelected(0);
    first.SetSelected(1);
    auto before = input.GetColumn(0, first.view());
    EXPECT_EQ(index->lookups, 1);
    auto all = input.GetColumn(0, SelectionView::All(4));
    EXPECT_EQ(index->lookups, 3);
    EXPECT_EQ(before.ValueAt<int64_t>(0), 4);
    EXPECT_FALSE(all.IsValid(1));
    EXPECT_EQ(all.ValueAt<int64_t>(2), 4);
    EXPECT_EQ(all.ValueAt<int64_t>(3), 1);
    EXPECT_EQ(index->validity_reads, 1);
}

TEST(PointReaderTest,
     ReverseLookupEmptyValidityFailureAndFinalCancelAreNotCached) {
    auto index = std::make_shared<ObservedIndex<int64_t>>();
    int64_t raw[] = {1, 2};
    index->Build(2, raw);
    IndexColumnReader<int64_t> reader(index, DataType::INT64, 2);
    ReadContext context;
    int checks = 0;
    context.checkpoint = [&] { ++checks; };
    reader.ReadRange(RangeRequest(0, 2, values, SelectionView::None(2)),
                     context);
    EXPECT_EQ(checks, 0);
    EXPECT_EQ(index->validity_reads, 0);
    index->fail_validity = true;
    try {
        reader.ReadRange(RangeRequest(0, 2, validity, SelectionView::All(2)),
                         context);
        FAIL();
    } catch (const SegcoreError& e) {
        EXPECT_EQ(e.get_error_code(), FileReadFailed);
    }
    index->fail_validity = false;
    checks = 0;
    context.checkpoint = [&] {
        if (++checks == 2)
            ThrowInfo(FollyCancel, "cancel validity before publishing");
    };
    try {
        reader.ReadRange(RangeRequest(0, 2, validity, SelectionView::All(2)),
                         context);
        FAIL();
    } catch (const SegcoreError& e) {
        EXPECT_EQ(e.get_error_code(), FollyCancel);
    }
    context.checkpoint = {};
    reader.ReadRange(RangeRequest(0, 2, validity, SelectionView::All(2)),
                     context);
    EXPECT_EQ(index->validity_reads, 3);
    EXPECT_EQ(index->lookups, 0);
}

TEST(PointReaderTest, IntegerIndexBoundsAndEmptyTermsPreserveNulls) {
    auto index =
        std::make_shared<milvus::segcore::reader::TestScalarIndex<int8_t>>();
    int8_t raw[] = {-128, 0, 127};
    bool valid[] = {true, false, true};
    index->Build(3, raw, valid);
    EXPECT_THROW(
        (IntegerIndexReader<int8_t>(index, 3, {ScalarIndexOp::Equal, 128})),
        SegcoreError);
    EXPECT_THROW(
        (IntegerIndexReader<int8_t>(index, 2, {ScalarIndexOp::Equal, 0})),
        SegcoreError);
    for (auto op : {ScalarIndexOp::In, ScalarIndexOp::NotIn}) {
        auto result = IntegerIndexReader<int8_t>(index, 3, {op}).Query({});
        auto out = predicate::MapPredicate(
            result, ReadBatchBinding::Range(0, 3), SelectionView::All(3));
        EXPECT_EQ(out.ValueAt<bool>(0), op == ScalarIndexOp::NotIn);
        EXPECT_FALSE(out.IsValid(1));
        EXPECT_EQ(out.ValueAt<bool>(2), op == ScalarIndexOp::NotIn);
    }
}

TEST(PointReaderTest, ReverseLookupFailureDoesNotPublishInputCoverage) {
    auto index = std::make_shared<ObservedIndex<int64_t>>();
    int64_t raw[] = {1, 2};
    index->Build(2, raw);
    auto reader =
        std::make_shared<IndexColumnReader<int64_t>>(index, DataType::INT64, 2);
    auto source =
        access::PrepareReaderInput(reader, std::make_shared<ReadContext>());
    auto loaders = source.make_loaders(ReadBatchBinding::Range(0, 2));
    BatchInput input(
        std::make_shared<RowType>(std::vector<std::string>{"x"},
                                  std::vector<DataType>{DataType::INT64}),
        2);
    input.BindLoader(0, loaders.values, loaders.validity);
    index->fail = true;
    try {
        input.GetColumn(0, SelectionView::All(2));
        FAIL();
    } catch (const SegcoreError& e) {
        EXPECT_EQ(e.get_error_code(), FileReadFailed);
    }
    index->fail = false;
    auto result = input.GetColumn(0, SelectionView::All(2));
    EXPECT_EQ(index->lookups, 3);
    EXPECT_EQ(result.ValueAt<int64_t>(1), 2);
}

TEST(PointReaderTest, IntegerIndexTypesAndInNotInFullThreeValues) {
    auto run = []<typename T>(DataType type) {
        auto index =
            std::make_shared<milvus::segcore::reader::TestScalarIndex<T>>();
        T raw[] = {T(0), T(1), T(1), T(0)};
        bool valid[] = {true, true, false, true};
        index->Build(4, raw, valid);
        for (auto op : {ScalarIndexOp::In, ScalarIndexOp::NotIn}) {
            Int64IndexQuery query{op};
            query.terms = {1, 1};
            IntegerIndexReader<T> reader(index, 4, query);
            auto result = reader.Query({});
            auto mapped =
                predicate::MapPredicate(result,
                                        ReadBatchBinding::Offsets({3, 2, 1, 3}),
                                        SelectionView::All(4));
            EXPECT_EQ(mapped.template ValueAt<bool>(0),
                      op == ScalarIndexOp::NotIn);
            EXPECT_FALSE(mapped.IsValid(1));
            EXPECT_EQ(mapped.template ValueAt<bool>(2),
                      op == ScalarIndexOp::In);
            EXPECT_EQ(mapped.template ValueAt<bool>(3),
                      op == ScalarIndexOp::NotIn);
        }
        IndexColumnReader<T> raw_reader(index, type, 4);
        auto result = raw_reader.Take(
            TakeRequest({1, 2, 0}, values, SelectionView::All(3)), {});
        EXPECT_EQ(result.values->template ValueAt<T>(0), T(1));
        EXPECT_FALSE(result.values->IsValid(1));
    };
    run.template operator()<bool>(DataType::BOOL);
    run.template operator()<int8_t>(DataType::INT8);
    run.template operator()<int16_t>(DataType::INT16);
    run.template operator()<int32_t>(DataType::INT32);
    run.template operator()<int64_t>(DataType::INT64);
}

TEST(PointReaderTest, IntegerComparisonsMatchIndependentNullableOracle) {
    auto run = []<typename T>() {
        auto index =
            std::make_shared<milvus::segcore::reader::TestScalarIndex<T>>();
        T raw[] = {std::numeric_limits<T>::min(),
                   T(0),
                   T(1),
                   std::numeric_limits<T>::max(),
                   T(0)};
        bool valid[] = {true, true, true, true, false};
        index->Build(5, raw, valid);
        std::vector<int64_t> offsets{3, 4, 0, 2, 1, 3};
        for (auto op : {ScalarIndexOp::Equal,
                        ScalarIndexOp::NotEqual,
                        ScalarIndexOp::Less,
                        ScalarIndexOp::LessEqual,
                        ScalarIndexOp::Greater,
                        ScalarIndexOp::GreaterEqual,
                        ScalarIndexOp::Between,
                        ScalarIndexOp::IsNull,
                        ScalarIndexOp::IsNotNull}) {
            for (bool inclusive : {false, true}) {
                Int64IndexQuery query{op, 0, 1, inclusive, inclusive};
                auto result = IntegerIndexReader<T>(index, 5, query).Query({});
                auto out =
                    predicate::MapPredicate(result,
                                            ReadBatchBinding::Offsets(offsets),
                                            SelectionView::All(offsets.size()));
                for (size_t i = 0; i < offsets.size(); ++i) {
                    const auto row = offsets[i];
                    const bool null_test = op == ScalarIndexOp::IsNull ||
                                           op == ScalarIndexOp::IsNotNull;
                    EXPECT_EQ(out.IsValid(i), null_test || valid[row]);
                    if (!out.IsValid(i))
                        continue;
                    const auto value = static_cast<int64_t>(raw[row]);
                    bool expected = false;
                    switch (op) {
                        case ScalarIndexOp::Equal:
                            expected = value == 0;
                            break;
                        case ScalarIndexOp::NotEqual:
                            expected = value != 0;
                            break;
                        case ScalarIndexOp::Less:
                            expected = value < 0;
                            break;
                        case ScalarIndexOp::LessEqual:
                            expected = value <= 0;
                            break;
                        case ScalarIndexOp::Greater:
                            expected = value > 0;
                            break;
                        case ScalarIndexOp::GreaterEqual:
                            expected = value >= 0;
                            break;
                        case ScalarIndexOp::Between:
                            expected = inclusive ? (value >= 0 && value <= 1)
                                                 : (value > 0 && value < 1);
                            break;
                        case ScalarIndexOp::IsNull:
                            expected = !valid[row];
                            break;
                        case ScalarIndexOp::IsNotNull:
                            expected = valid[row];
                            break;
                        default:
                            FAIL();
                    }
                    EXPECT_EQ(out.template ValueAt<bool>(i), expected);
                }
            }
        }
    };
    run.template operator()<bool>();
    run.template operator()<int8_t>();
    run.template operator()<int16_t>();
    run.template operator()<int32_t>();
    run.template operator()<int64_t>();
}

TEST(PointReaderTest, NarrowIndexTermPlanMatchesGrowingAndSharesProjection) {
    auto column = Growing<int8_t>({-1, 2, 3, 4, 2, 7});
    auto reader = std::make_shared<GrowingColumnReader<int8_t>>(
        column, DataType::INT8, 6);
    auto index =
        std::make_shared<milvus::segcore::reader::TestScalarIndex<int8_t>>();
    int8_t raw[] = {-1, 2, 3, 4, 2, 7};
    index->Build(6, raw);
    expr::ColumnInfo field(FieldId(100), DataType::INT8);
    auto term = std::make_shared<expr::TermFilterExpr>(
        field, std::vector{Int(-1), Int(2), Int(128)});
    plan::CompileOptions options;
    options.plan_indexes = true;
    auto compiled = plan::CompileExpressions(
        {term, std::make_shared<expr::ColumnExpr>(field)}, options);
    ASSERT_TRUE(compiled.supported());
    ASSERT_EQ(compiled.plan->predicates().size(), 1);
    EXPECT_EQ(compiled.plan->predicates()[0].query.terms,
              (std::vector<int64_t>{-1, 2}));
    auto context = std::make_shared<ReadContext>();
    std::vector<ReadOnlyVectorView> expected;
    for (bool use_index : {false, true}) {
        auto prepared = plan::Prepare(
            compiled.plan,
            [&](const auto&) {
                return access::PrepareReaderInput(reader, context);
            },
            {use_index, [&](const auto& predicate) {
                 return plan::PrepareIntegerIndex<int8_t>(
                     index, 6, predicate.query, context);
             }});
        auto batch = prepared.MakeExecutionBatch(
            ReadBatchBinding::Offsets({4, 0, 3, 4}));
        auto out = batch->Eval(SelectionView::All(4));
        for (size_t i = 0; i < 4; ++i) {
            EXPECT_EQ(out[0].ValueAt<bool>(i), i != 2);
            EXPECT_EQ(out[1].ValueAt<int8_t>(i),
                      (raw[std::vector<int>{4, 0, 3, 4}[i]]));
        }
    }
}
TEST(PointReaderTest,
     BloomGrowingAndIndexReverseLookupShareSelectedProjection) {
    const std::string hex =
        "4d424631010001000800000000000000fca9f1d24d62503f0100000001000000"
        "0117100a804845062834804200440612100448a40680089c4040542838410046";
    std::string blob;
    for (size_t i = 0; i < hex.size(); i += 2)
        blob.push_back(char(std::stoul(hex.substr(i, 2), nullptr, 16)));
    int64_t raw[] = {-1, 7, 42, 0, 3, 2};
    bool valid[] = {true, true, true, false, true, true};
    auto validity_data = std::make_shared<ThreadSafeValidData>(3);
    DataArray data;
    for (bool v : valid) data.add_valid_data(v);
    FieldMeta meta(
        FieldName("x"), FieldId(100), DataType::INT64, true, std::nullopt);
    validity_data->set_data_raw(0, 6, &data, meta);
    auto growing =
        std::make_shared<ConcurrentVector<int64_t>>(3, nullptr, validity_data);
    growing->set_data_raw(0, raw, 6);
    auto index = std::make_shared<ObservedIndex<int64_t>>();
    index->Build(6, raw, valid);
    std::vector<std::shared_ptr<ColumnReader>> readers{
        std::make_shared<GrowingColumnReader<int64_t>>(
            growing, DataType::INT64, 6),
        std::make_shared<IndexColumnReader<int64_t>>(
            index, DataType::INT64, 6)};
    expr::ColumnInfo field(FieldId(100), DataType::INT64, {}, true);
    auto bloom = std::make_shared<expr::BloomFilterExpr>(field, blob);
    plan::CompileOptions options;
    options.plan_indexes = true;
    auto compiled = plan::CompileExpressions(
        {bloom, std::make_shared<expr::ColumnExpr>(field)}, options);
    ASSERT_TRUE(compiled.supported());
    // Reverse lookup recovers values; it is not a native Bloom index query.
    EXPECT_TRUE(compiled.plan->predicates().empty());
    auto context = std::make_shared<ReadContext>();
    for (const auto& reader : readers) {
        auto batch =
            plan::Prepare(compiled.plan, [&](const auto&) {
                return access::PrepareReaderInput(reader, context);
            }).MakeExecutionBatch(ReadBatchBinding::Offsets({2, 3, 1, 2, 0}));
        auto rows = OwnedSelection::None(5);
        rows.SetSelected(0);
        rows.SetSelected(1);
        rows.SetSelected(3);
        auto out = batch->Eval(rows.view());
        EXPECT_TRUE(out[0].ValueAt<bool>(0));
        EXPECT_FALSE(out[0].IsValid(1));
        EXPECT_TRUE(out[0].ValueAt<bool>(3));
        EXPECT_FALSE(out[0].defined_rows().IsSelected(2));
        EXPECT_EQ(out[1].ValueAt<int64_t>(3), 42);
        auto all = batch->Eval(SelectionView::All(5));
        EXPECT_FALSE(all[0].ValueAt<bool>(2));
        EXPECT_TRUE(all[0].ValueAt<bool>(4));
        EXPECT_EQ(all[1].ValueAt<int64_t>(4), -1);
    }
    // Two duplicate positions in the first batch, then two newly requested
    // positions. Projection and repeat evaluation do not fetch them again.
    EXPECT_EQ(index->lookups, 4);
    EXPECT_EQ(index->validity_reads, 1);

    auto reader = readers[1];
    auto batch = plan::Prepare(compiled.plan, [&](const auto&) {
                     return access::PrepareReaderInput(reader, context);
                 }).MakeExecutionBatch(ReadBatchBinding::Range(0, 6));
    index->fail = true;
    EXPECT_NO_THROW(batch->Eval(SelectionView::None(6)));
    try {
        batch->Eval(SelectionView::All(6));
        FAIL();
    } catch (const SegcoreError& e) {
        EXPECT_EQ(e.get_error_code(), FileReadFailed);
    }
    const auto before_retry = index->lookups;
    index->fail = false;
    auto recovered = batch->Eval(SelectionView::All(6));
    EXPECT_EQ(index->lookups - before_retry, 5);
    EXPECT_TRUE(recovered[0].ValueAt<bool>(0));
    EXPECT_FALSE(recovered[0].IsValid(3));
}

TEST(PointReaderTest, BloomGrowingJsonDuplicateOffsetsAndEscapedStrings) {
    // Saturated UTF-8-only MBF1: distinguishes the runtime domains without
    // relying on an independently reimplemented hash builder in this test.
    std::string blob(64, '\0');
    blob.replace(0, 4, "MBF1");
    blob[4] = blob[6] = blob[24] = 1;
    blob[28] = 2;
    std::fill(blob.begin() + 32, blob.end(), char(0xff));
    auto column = Growing<Json>(
        {Json(simdjson::padded_string(
             std::string("{\"x\":\"\\u5317\\u4eac\"}"))),
         Json(simdjson::padded_string(std::string("{\"x\":42}"))),
         Json(simdjson::padded_string(std::string("{\"x\":null}")))});
    auto reader =
        std::make_shared<GrowingColumnReader<Json>>(column, DataType::JSON, 3);
    expr::ColumnInfo field(FieldId(100), DataType::JSON, {"x"}, true);
    auto logical = std::make_shared<expr::BloomFilterExpr>(field, blob);
    auto compiled = plan::CompileExpressions({logical});
    ASSERT_TRUE(compiled.supported());
    auto context = std::make_shared<ReadContext>();
    auto batch = plan::Prepare(compiled.plan, [&](const auto& dependency) {
                     EXPECT_TRUE(dependency.path.empty());
                     return access::PrepareReaderInput(reader, context);
                 }).MakeExecutionBatch(ReadBatchBinding::Offsets({2, 0, 1, 0}));
    column->clear();  // Reader pins the original generation.
    auto out = batch->Eval(SelectionView::All(4));
    EXPECT_FALSE(out[0].IsValid(0));
    EXPECT_TRUE(out[0].ValueAt<bool>(1));
    EXPECT_FALSE(out[0].ValueAt<bool>(2));
    EXPECT_TRUE(out[0].ValueAt<bool>(3));
}

TEST(PointReaderTest, GrowingRangeBorrowsNumericAndVariablePayloads) {
    auto column = Growing<int64_t>({10, 20, 30, 40});
    auto snapshot = column->acquire_chunks();
    const auto* raw =
        static_cast<const int64_t*>(column->get_span_base(snapshot, 0).data());
    auto reader = std::make_unique<GrowingColumnReader<int64_t>>(
        column, DataType::INT64, 4);
    auto out = reader->ReadRange(
        RangeRequest(1, 2, values, SelectionView::All(2)), {});
    EXPECT_EQ(out.values->FlatData<int64_t>(), raw + 1);
    auto selected = OwnedSelection::FromIndices(4, {0, 2});
    // Unselected Take offsets need not follow the selected rows' affine map.
    auto sparse =
        reader->Take(TakeRequest({0, 3, 2, 0}, values, selected.view()), {});
    EXPECT_EQ(sparse.values->ValueAt<int64_t>(0), 10);
    EXPECT_EQ(sparse.values->ValueAt<int64_t>(2), 30);
    EXPECT_FALSE(sparse.values->defined_rows().IsSelected(1));
    auto validity = reader->Take(
        TakeRequest(
            {3, 0, 3, 1}, ReadProjection::ValidityOnly, selected.view()),
        {});
    EXPECT_FALSE(validity.values.has_value());
    EXPECT_EQ(validity.valid.view().selected_count(), 2);
    reader.reset();
    column->clear();
    column.reset();
    EXPECT_EQ(out.values->ValueAt<int64_t>(1), 30);

    auto strings =
        Growing<std::string>({std::string(4096, 'z'), "b", "c", "d"});
    auto spans = strings->acquire_chunks();
    const auto* text = static_cast<const std::string*>(
                           strings->get_span_base(spans, 0).data())[0]
                           .data();
    auto sr = std::make_unique<GrowingColumnReader<std::string>>(
        strings, DataType::VARCHAR, 4);
    auto taken =
        sr->Take(TakeRequest({3, 0, 0, 2}, values, SelectionView::All(4)), {});
    EXPECT_EQ(taken.values->StringViewAt(1).data(), text);
    EXPECT_EQ(taken.values->StringViewAt(2).data(), text);
    sr.reset();
    strings->clear();
    strings.reset();
    EXPECT_EQ(taken.values->StringViewAt(1), std::string(4096, 'z'));

    auto jsons = Growing<Json>(
        {Json(simdjson::padded_string(std::string_view("{\"v\":7}")))});
    auto js = jsons->acquire_chunks();
    const auto* jdata =
        static_cast<const Json*>(jsons->get_span_base(js, 0).data())[0]
            .data()
            .data();
    GrowingColumnReader<Json> jr(jsons, DataType::JSON, 1);
    const auto j =
        jr.ReadRange(RangeRequest(0, 1, values, SelectionView::All(1)), {});
    EXPECT_EQ(j.values->JsonViewAt(0).data().data(), jdata);
}

TEST(PointReaderTest, VectorValidityReaderPreservesPositionsAndFailures) {
    std::vector<int64_t> visited;
    bool fail = true;
    ValidityColumnReader reader(
        DataType::VECTOR_FLOAT, 8, [&](int64_t offset, const ReadContext&) {
            visited.push_back(offset);
            if (fail && offset == 2)
                ThrowInfo(FileReadFailed, "injected vector validity failure");
            return offset != 2;
        });
    auto rows = OwnedSelection::FromIndices(4, {0, 2, 3});
    TakeRequest request({7, 0, 7, 2}, validity, rows.view());
    try {
        reader.Take(request, {});
        FAIL();
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), FileReadFailed);
    }
    EXPECT_EQ(visited, (std::vector<int64_t>{7, 7, 2}));
    visited.clear();
    fail = false;
    auto result = reader.Take(request, {});
    EXPECT_FALSE(result.values.has_value());
    EXPECT_EQ(result.defined.view().selected_count(), 3);
    EXPECT_TRUE(result.valid.view().IsSelected(0));
    EXPECT_TRUE(result.valid.view().IsSelected(2));
    EXPECT_FALSE(result.valid.view().IsSelected(3));
    EXPECT_FALSE(result.defined.view().IsSelected(1));
    EXPECT_EQ(visited, (std::vector<int64_t>{7, 7, 2}));
    visited.clear();
    ReadContext cancelled;
    cancelled.checkpoint = [] { ThrowInfo(FollyCancel, "cancel validity"); };
    try {
        reader.Take(request, cancelled);
        FAIL();
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), FollyCancel);
    }
    EXPECT_TRUE(visited.empty());
    EXPECT_NO_THROW(reader.ReadRange(
        RangeRequest(0, 4, validity, SelectionView::None(4)), cancelled));
    EXPECT_THROW(
        reader.Take(TakeRequest({0}, values, SelectionView::All(1)), {}),
        SegcoreError);
}

TEST(PointReaderTest, DeferredReverseLookupRetriesAndKeepsQueryPrefix) {
    auto index = std::make_shared<ObservedIndex<int64_t>>();
    int64_t raw[] = {1, 2, 3, 4};
    bool valid[] = {true, false, true, true};
    index->Build(4, raw, valid);
    int loads = 0;
    IndexColumnReader<int64_t> reader(
        DataType::INT64,
        3,
        4,
        [&](const ReadContext&)
            -> std::shared_ptr<const index::IIndexReaderBase> {
            if (++loads == 1)
                ThrowInfo(FileReadFailed,
                          "injected reverse source load failure");
            return index;
        });
    reader.ReadRange(RangeRequest(0, 3, values, SelectionView::None(3)), {});
    EXPECT_EQ(loads, 0);
    try {
        reader.Take(TakeRequest({2, 1, 2, 0}, values, SelectionView::All(4)),
                    {});
        FAIL();
    } catch (const SegcoreError& e) {
        EXPECT_EQ(e.get_error_code(), FileReadFailed);
    }
    EXPECT_EQ(index->lookups, 0);
    auto result = reader.Take(
        TakeRequest({2, 1, 2, 0}, values, SelectionView::All(4)), {});
    EXPECT_EQ(loads, 2);
    EXPECT_EQ(index->lookups, 3);
    EXPECT_FALSE(result.values->IsValid(1));
    EXPECT_EQ(result.values->ValueAt<int64_t>(0), 3);
    EXPECT_EQ(result.values->ValueAt<int64_t>(2), 3);
    reader.Take(TakeRequest({1}, validity, SelectionView::All(1)), {});
    EXPECT_EQ(loads, 2);
    EXPECT_EQ(index->lookups, 3);
    EXPECT_THROW(
        reader.Take(TakeRequest({3}, values, SelectionView::All(1)), {}),
        SegcoreError);
}

}  // namespace milvus::segcore::reader
