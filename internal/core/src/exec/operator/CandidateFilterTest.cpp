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

#include <limits>
#include "common/VirtualPK.h"
#include <gtest/gtest.h>
#include "exec/QueryContext.h"
#include "segcore/SegmentGrowingImpl.h"
#include "segcore/boost_score_c.h"
#include "common/init_c.h"
#include "query/PlanImpl.h"
#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include "exec/operator/CandidateFilter.h"
#include "rescores/BoostScoreRunner.h"
#include "exec/predicate/IndexNode.h"
#include "segcore/reader/PointColumnReader.h"
#include "test_utils/cachinglayer_test_utils.h"
#include "test_utils/storage_test_utils.h"
#include "exec/operator/FilterBitsExecution.h"
#include "exec/plan/ExprCompiler.h"
#include "segcore/reader/ReaderTestUtils.h"
#include "index/scalar/json/JsonValueProjection.h"

namespace milvus::exec {
namespace {
using namespace segcore::reader;
template <typename T>
void
CheckOrdinaryTypedIndex(const std::string& family,
                        DataType type,
                        const std::vector<T>& values,
                        proto::plan::GenericValue lower,
                        proto::plan::GenericValue upper) {
    auto schema = std::make_shared<Schema>();
    auto pk = schema->AddDebugField("pk", DataType::INT64);
    auto field = schema->AddDebugField("value", type, true);
    schema->set_primary_field_id(pk);
    auto data = segcore::DataGen(schema, values.size());
    const bool valid[] = {true, false, true, true, true};
    for (auto& column : *data.raw_->mutable_fields_data()) {
        if (column.field_id() != field.get())
            continue;
        auto replacement = segcore::CreateDataArrayFrom(
            values.data(), valid, values.size(), (*schema)[field]);
        column.Swap(replacement.get());
    }
    auto segment = CreateSealedWithFieldDataLoaded(schema, data);
    storage::FileManagerContext index_context(
        storage::RemoteChunkManagerSingleton::GetInstance()
            .GetRemoteChunkManager());
    index_context.fieldDataMeta.field_schema.set_data_type(
        static_cast<proto::schema::DataType>(type));
    index_context.fieldDataMeta.field_schema.set_nullable(true);
    index_context.fieldDataMeta.field_schema.set_fieldid(field.get());
    index_context.fieldDataMeta.field_id = field.get();
    auto source =
        BuildReaderIndex<T>(family, values.size(), values.data(), valid);
    const bool cheap_lookup = source->Caps().cheap_value_lookup;
    segcore::LoadIndexInfo info;
    info.field_id = field.get();
    info.index_params = GenIndexParams(source.get(), source.family);
    SetTestIndexMetadata(info, *source, source.family);
    info.index = source.index;
    segment->LoadIndex(info);
    auto snapshot = segment->CaptureReadSnapshot();
    auto read = std::make_shared<segcore::reader::ReadContext>();
    expr::ColumnInfo column(field, type, {}, true);
    std::vector<expr::TypedExprPtr> predicates;
    for (auto op : {proto::plan::Equal,
                    proto::plan::NotEqual,
                    proto::plan::LessThan,
                    proto::plan::LessEqual,
                    proto::plan::GreaterThan,
                    proto::plan::GreaterEqual})
        predicates.push_back(
            std::make_shared<expr::UnaryRangeFilterExpr>(column, op, lower));
    predicates.push_back(std::make_shared<expr::BinaryRangeFilterExpr>(
        column, lower, upper, true, false));
    predicates.push_back(std::make_shared<expr::TermFilterExpr>(
        column, std::vector{lower, upper}));
    predicates.push_back(std::make_shared<expr::NullExpr>(
        column, proto::plan::NullExpr_NullOp_IsNull));
    predicates.push_back(std::make_shared<expr::TermFilterExpr>(
        column, std::vector<proto::plan::GenericValue>{}));
    for (const auto& predicate : predicates) {
        SCOPED_TRACE(predicate->ToString());
        auto indexed = FilterBitsExecution::TryCreate(
            predicate, nullptr, snapshot, values.size(), 2, read);
        auto raw = FilterBitsExecution::TryCreateWithBinding(
            predicate,
            nullptr,
            access::MakeSealedReaderBinding(snapshot, values.size()),
            values.size(),
            2,
            read);
        ASSERT_TRUE(indexed && raw);
        ASSERT_EQ(indexed->prepared().predicate_routes()[0],
                  plan::PredicateRoute::ExactIndex);
        auto binding = ReadBatchBinding::Offsets({4, 1, 3, 0, 3, 2});
        auto actual_batch = indexed->prepared().MakeExecutionBatch(binding);
        auto expected_batch = raw->prepared().MakeExecutionBatch(binding);
        auto actual = actual_batch->Eval(SelectionView::All(6))[0];
        auto expected = expected_batch->Eval(SelectionView::All(6))[0];
        for (size_t row = 0; row < 6; ++row) {
            EXPECT_EQ(actual.IsValid(row), expected.IsValid(row));
            if (actual.IsValid(row) && expected.IsValid(row))
                EXPECT_EQ(actual.template ValueAt<bool>(row),
                          expected.template ValueAt<bool>(row));
        }
    }
    segment->DropFieldData(field);
    auto no_raw = FilterBitsExecution::TryCreate(predicates[0],
                                                 nullptr,
                                                 segment->CaptureReadSnapshot(),
                                                 values.size(),
                                                 2,
                                                 read);
    ASSERT_TRUE(no_raw);
    EXPECT_EQ(no_raw->prepared().granularity(),
              plan::ExecutionGranularity::WholeSegment);
    EXPECT_TRUE(no_raw->GetOutput());
    // Force an original-value calculation after dropping the column. These
    // predicates cannot be answered by the new exact-index descriptors.
    expr::TypedExprPtr value_predicate;
    if constexpr (std::is_same_v<T, std::string>)
        value_predicate = std::make_shared<expr::UnaryRangeFilterExpr>(
            column, proto::plan::PrefixMatch, lower);
    else
        value_predicate = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
            column, proto::plan::LessThan, proto::plan::Add, upper, lower);
    for (int64_t prefix :
         {int64_t(values.size()), int64_t(values.size() - 1)}) {
        auto current = segment->CaptureReadSnapshot();
        auto reverse = FilterBitsExecution::TryCreateWithBinding(
            value_predicate,
            nullptr,
            access::MakeSealedReaderBinding(current, prefix),
            prefix,
            2,
            read);
        if (!cheap_lookup) {
            EXPECT_FALSE(reverse);
        } else {
            ASSERT_TRUE(reverse);
            auto batch = reverse->prepared().MakeExecutionBatch(
                ReadBatchBinding::Offsets({3, 1, 3, 0}));
            batch->Eval(SelectionView::None(4));
            auto actual = batch->Eval(SelectionView::All(4))[0];
            const int offsets[] = {3, 1, 3, 0};
            for (int i = 0; i < 4; ++i) {
                EXPECT_EQ(actual.IsValid(i), valid[offsets[i]]);
                if (!valid[offsets[i]])
                    continue;
                if constexpr (std::is_same_v<T, std::string>)
                    EXPECT_EQ(
                        actual.template ValueAt<bool>(i),
                        values[offsets[i]].starts_with(lower.string_val()));
                else
                    EXPECT_EQ(
                        actual.template ValueAt<bool>(i),
                        T(values[offsets[i]] + (lower.has_int64_val()
                                                    ? T(lower.int64_val())
                                                    : T(lower.float_val()))) <
                            (upper.has_int64_val() ? T(upper.int64_val())
                                                   : T(upper.float_val())));
            }
        }
    }
}

void
CheckArrayIndex(DataType element,
                const std::string& backend,
                bool nested,
                bool element_nullable = false,
                bool nullable = true) {
    SCOPED_TRACE(fmt::format("{} {} {}", int(element), backend, nested));
    const int64_t count = 6;
    const std::string name = nested ? "items[a]" : "a";
    auto schema = std::make_shared<Schema>();
    auto pk = schema->AddDebugField("pk", DataType::INT64);
    schema->set_primary_field_id(pk);
    auto field =
        schema->AddDebugArrayField(name, element, nullable, element_nullable);
    const bool valid[] = {!nullable, true, true, !nullable, true, true};
    std::vector<ScalarFieldProto> rows(count);
    const std::vector<std::vector<int>> values = {
        {}, {1, 2}, {}, {}, {2, 3}, {1}};
    auto term = [&](int v) {
        proto::plan::GenericValue value;
        if (element == DataType::BOOL)
            value.set_bool_val(v % 2);
        else if (element == DataType::VARCHAR)
            value.set_string_val(std::to_string(v));
        else if (element == DataType::FLOAT || element == DataType::DOUBLE)
            value.set_float_val(v * 0.1);
        else
            value.set_int64_val(v);
        return value;
    };
    for (size_t row = 0; row < rows.size(); ++row) {
        // Initialize empty arrays with their actual physical element type too.
        if (element == DataType::BOOL)
            rows[row].mutable_bool_data();
        else if (element == DataType::VARCHAR)
            rows[row].mutable_string_data();
        else if (element == DataType::FLOAT)
            rows[row].mutable_float_data();
        else if (element == DataType::DOUBLE)
            rows[row].mutable_double_data();
        else if (element == DataType::INT64)
            rows[row].mutable_long_data();
        else
            rows[row].mutable_int_data();
        for (int v : values[row]) {
            if (element == DataType::BOOL)
                rows[row].mutable_bool_data()->add_data(v % 2);
            else if (element == DataType::VARCHAR)
                rows[row].mutable_string_data()->add_data(std::to_string(v));
            else if (element == DataType::FLOAT)
                rows[row].mutable_float_data()->add_data(v * 0.1);
            else if (element == DataType::DOUBLE)
                rows[row].mutable_double_data()->add_data(v * 0.1);
            else if (element == DataType::INT64)
                rows[row].mutable_long_data()->add_data(v);
            else
                rows[row].mutable_int_data()->add_data(v);
        }
    }
    auto data = segcore::DataGen(schema, count);
    for (auto& column : *data.raw_->mutable_fields_data()) {
        if (column.field_id() != field.get())
            continue;
        column.clear_valid_data();
        auto* scalars = column.mutable_scalars();
        scalars->clear_valid_data();
        scalars->mutable_array_data()->clear_data();
        for (size_t i = 0; i < rows.size(); ++i) {
            if (nullable)
                scalars->add_valid_data(valid[i]);
            *scalars->mutable_array_data()->add_data() = rows[i];
        }
    }
    auto segment = CreateSealedWithFieldDataLoaded(schema, data);
    auto field_data =
        storage::CreateFieldData(DataType::ARRAY, DataType::NONE, nullable);
    std::vector<Array> arrays;
    for (const auto& row : rows) arrays.emplace_back(row);
    const uint8_t validity = 0b00110110;
    if (nullable)
        field_data->FillFieldData(arrays.data(), &validity, count, 0);
    else
        field_data->FillFieldData(arrays.data(), count);
    storage::FileManagerContext index_context(
        storage::RemoteChunkManagerSingleton::GetInstance()
            .GetRemoteChunkManager());
    auto& meta = index_context.fieldDataMeta;
    meta.field_id = field.get();
    meta.field_schema.set_fieldid(field.get());
    meta.field_schema.set_name(name);
    meta.field_schema.set_data_type(proto::schema::DataType::Array);
    meta.field_schema.set_element_type(
        static_cast<proto::schema::DataType>(element));
    meta.field_schema.set_nullable(nullable);
    const auto family = ReaderFamily(backend);
    Config params{{"field_type", int(DataType::ARRAY)},
                  {"value_type", int(element)},
                  {"array_element_type", int(element)},
                  {"nested", nested}};
    auto build = [&]<typename T>() -> BuiltReaderIndex {
        if (nested) {
            FixedVector<T> flattened(5);
            size_t pos = 0;
            for (size_t row = 0; row < arrays.size(); ++row) {
                if (!valid[row])
                    continue;
                for (int i = 0; i < arrays[row].length(); ++i)
                    flattened[pos++] = arrays[row].get_data_unchecked<T>(i);
            }
            return BuildReaderIndex(
                family, pos, flattened.data(), nullptr, params);
        }
        std::vector<ArrayView> views;
        for (auto& array : arrays)
            views.emplace_back(const_cast<char*>(array.data()),
                               array.length(),
                               array.byte_size(),
                               array.get_element_type(),
                               array.get_offsets_data());
        return BuildReaderIndex(family,
                                views.size(),
                                views.data(),
                                nullable ? valid : nullptr,
                                params);
    };
    auto source = [&]() -> BuiltReaderIndex {
        switch (element) {
            case DataType::BOOL:
                return build.template operator()<bool>();
            case DataType::INT8:
                return build.template operator()<int8_t>();
            case DataType::INT16:
                return build.template operator()<int16_t>();
            case DataType::INT32:
                return build.template operator()<int32_t>();
            case DataType::INT64:
                return build.template operator()<int64_t>();
            case DataType::FLOAT:
                return build.template operator()<float>();
            case DataType::DOUBLE:
                return build.template operator()<double>();
            default:
                return build.template operator()<std::string>();
        }
    }();
    ASSERT_EQ(source->Caps().nested, nested);
    ASSERT_EQ(source->Count(), nested ? 5 : count);
    segcore::LoadIndexInfo info;
    info.field_id = field.get();
    info.field_type = DataType::ARRAY;
    info.element_type = element;
    info.index_params = GenIndexParams(source.get(), source.family);
    SetTestIndexMetadata(info, *source, source.family);
    info.index = source.index;
    segment->LoadIndex(info);
    auto snapshot = segment->CaptureReadSnapshot();
    auto read = std::make_shared<segcore::reader::ReadContext>();
    expr::ColumnInfo column(field, DataType::ARRAY, {}, nullable);
    column.element_type_ = element;
    std::vector<expr::TypedExprPtr> predicates;
    std::vector<bool> expect_index;
    for (auto op : {proto::plan::JSONContainsExpr_JSONOp_Contains,
                    proto::plan::JSONContainsExpr_JSONOp_ContainsAny,
                    proto::plan::JSONContainsExpr_JSONOp_ContainsAll}) {
        for (auto terms : {std::vector<proto::plan::GenericValue>{},
                           std::vector{term(1)},
                           std::vector{term(1), term(2)},
                           std::vector{term(2), term(2)},
                           std::vector{term(1), term(3)}}) {
            if (op == proto::plan::JSONContainsExpr_JSONOp_Contains &&
                terms.size() != 1)
                continue;
            predicates.push_back(std::make_shared<expr::JsonContainsExpr>(
                column, op, true, terms));
            expect_index.push_back(!element_nullable);
        }
    }
    for (auto op : {proto::plan::NullExpr_NullOp_IsNull,
                    proto::plan::NullExpr_NullOp_IsNotNull}) {
        predicates.push_back(std::make_shared<expr::NullExpr>(column, op));
        expect_index.push_back(!element_nullable);
    }
    predicates.push_back(std::make_shared<expr::TermFilterExpr>(
        column, std::vector{term(1)}, true));
    expect_index.push_back(!element_nullable && element != DataType::FLOAT);
    if (element == DataType::INT8) {
        proto::plan::GenericValue overflow;
        overflow.set_int64_val(257);
        predicates.push_back(std::make_shared<expr::JsonContainsExpr>(
            column,
            proto::plan::JSONContainsExpr_JSONOp_Contains,
            true,
            std::vector{overflow}));
        expect_index.push_back(false);
    }
    const size_t leaf_count = predicates.size();
    for (size_t i = 0; i < leaf_count; ++i) {
        predicates.push_back(std::make_shared<expr::LogicalUnaryExpr>(
            expr::LogicalUnaryExpr::OpType::LogicalNot, predicates[i]));
        expect_index.push_back(expect_index[i]);
    }
    std::vector<std::unique_ptr<FilterBitsExecution>> prepared;
    std::vector<ReadOnlyVectorView> expected_results;
    for (size_t p = 0; p < predicates.size(); ++p) {
        SCOPED_TRACE(predicates[p]->ToString());
        auto indexed = FilterBitsExecution::TryCreate(
            predicates[p], nullptr, snapshot, count - 1, 2, read, schema);
        auto raw = FilterBitsExecution::TryCreateWithBinding(
            predicates[p],
            nullptr,
            access::MakeSealedReaderBinding(snapshot, count - 1),
            count - 1,
            2,
            read,
            {},
            schema);
        ASSERT_TRUE(indexed && raw);
        EXPECT_EQ(indexed->prepared().granularity(),
                  expect_index[p] ? plan::ExecutionGranularity::WholeSegment
                                  : plan::ExecutionGranularity::Batch);
        const std::vector<int32_t> offsets{4, 0, 2, 1, 4, 3, 1};
        for (auto binding :
             {ReadBatchBinding::Offsets(std::span<const int32_t>(offsets)),
              ReadBatchBinding::Range(0, count - 1)}) {
            auto actual_batch = indexed->prepared().MakeExecutionBatch(binding);
            auto raw_batch = raw->prepared().MakeExecutionBatch(binding);
            auto size = binding.size();
            actual_batch->Eval(SelectionView::None(size));
            auto sparse_bits = TargetBitmap(size, false);
            sparse_bits.set(1);
            sparse_bits.set(3);
            for (auto selection : {SelectionView::FromBitmap(sparse_bits),
                                   SelectionView::All(size)}) {
                auto actual = actual_batch->Eval(selection)[0];
                auto expected = raw_batch->Eval(selection)[0];
                selection.ForEachSelected([&](size_t row) {
                    EXPECT_EQ(actual.IsValid(row), expected.IsValid(row))
                        << row;
                    if (actual.IsValid(row) && expected.IsValid(row))
                        EXPECT_EQ(actual.ValueAt<bool>(row),
                                  expected.ValueAt<bool>(row))
                            << row;
                });
            }
        }
        auto batch = raw->prepared().MakeExecutionBatch(
            ReadBatchBinding::Range(0, count - 1));
        expected_results.push_back(
            batch->Eval(SelectionView::All(count - 1))[0]);
        prepared.push_back(std::move(indexed));
    }
    if (nested && !element_nullable) {
        auto element_column = column;
        element_column.element_level_ = true;
        auto child = std::make_shared<expr::UnaryRangeFilterExpr>(
            element_column, proto::plan::Equal, term(1));
        for (auto operation : {proto::plan::MatchAny,
                               proto::plan::MatchAll,
                               proto::plan::MatchLeast,
                               proto::plan::MatchMost,
                               proto::plan::MatchExact}) {
            auto match =
                std::make_shared<expr::MatchExpr>("items", operation, 1, child);
            auto indexed = FilterBitsExecution::TryCreate(
                match, nullptr, snapshot, count - 1, 2, read, schema);
            auto raw = FilterBitsExecution::TryCreateWithBinding(
                match,
                nullptr,
                access::MakeSealedReaderBinding(snapshot, count - 1),
                count - 1,
                2,
                read,
                {},
                schema);
            ASSERT_TRUE(indexed && raw);
            ASSERT_EQ(indexed->prepared().predicate_routes(),
                      std::vector{plan::PredicateRoute::ExactIndex});
            auto binding = ReadBatchBinding::Offsets({4, 0, 2, 1, 4, 3, 1});
            auto a = indexed->prepared().MakeExecutionBatch(binding);
            auto b = raw->prepared().MakeExecutionBatch(binding);
            auto sparse =
                OwnedSelection::FromIndices(binding.size(), {0, 1, 3, 6});
            for (auto rows :
                 {sparse.view(), SelectionView::All(binding.size())}) {
                auto actual = a->Eval(rows)[0], expected = b->Eval(rows)[0];
                rows.ForEachSelected([&](size_t row) {
                    ASSERT_EQ(actual.IsValid(row), expected.IsValid(row));
                    if (expected.IsValid(row))
                        EXPECT_EQ(actual.ValueAt<bool>(row),
                                  expected.ValueAt<bool>(row));
                });
            }
        }
        auto lowered = plan::LowerElementPredicate(child, *schema, "items");
        auto offsets = snapshot->GetStructElementOffsets(field);
        ASSERT_TRUE(lowered && offsets);
        auto elements = offsets->ElementIDRangeOfRow(count - 2).second;
        auto bind = access::MakeElementReaderBinding(
            access::MakeSealedReaderBinding(snapshot, count - 1),
            schema,
            offsets,
            count - 1,
            elements);
        plan::PhysicalPrepareOptions physical;
        physical.bind_index =
            access::MakeElementIndexBinding(snapshot, schema, offsets, read);
        auto indexed = FilterBitsExecution::TryCreateWithBinding(
            lowered, nullptr, bind, elements, 2, read, physical, schema, true);
        auto raw = FilterBitsExecution::TryCreateWithBinding(
            lowered, nullptr, bind, elements, 2, read, {}, schema, true);
        ASSERT_TRUE(indexed && raw);
        ASSERT_EQ(indexed->prepared().predicate_routes(),
                  std::vector{plan::PredicateRoute::ExactIndex});
        auto binding = ReadBatchBinding::Offsets({3, 0, 1, 0, 2});
        auto a = indexed->prepared().MakeExecutionBatch(binding)->Eval(
            SelectionView::All(5))[0];
        auto b = raw->prepared().MakeExecutionBatch(binding)->Eval(
            SelectionView::All(5))[0];
        for (size_t i = 0; i < 5; ++i) {
            ASSERT_EQ(a.IsValid(i), b.IsValid(i));
            if (b.IsValid(i))
                EXPECT_EQ(a.ValueAt<bool>(i), b.ValueAt<bool>(i));
        }
    }
    segment->DropFieldData(field);
    auto config = std::make_shared<QueryConfig>(
        std::unordered_map<std::string, std::string>{
            {QueryConfig::kEnableExprExecutionV2, "true"},
            {QueryConfig::kExprEvalBatchSize, "2"}});
    QueryContext candidate_query("array-index",
                                 segment.get(),
                                 count - 1,
                                 MAX_TIMESTAMP,
                                 0,
                                 0,
                                 query::PlanOptions{},
                                 config);
    candidate_query.set_read_snapshot(segment->CaptureReadSnapshot());
    ExecContext candidate_context(&candidate_query);
    for (size_t p = 0; p < predicates.size(); ++p) {
        auto fresh =
            FilterBitsExecution::TryCreate(predicates[p],
                                           nullptr,
                                           segment->CaptureReadSnapshot(),
                                           count - 1,
                                           2,
                                           read,
                                           schema);
        EXPECT_EQ(bool(fresh), expect_index[p] && !nested);
        auto candidate =
            CandidateFilter::TryCreate(predicates[p], &candidate_context);
        EXPECT_EQ(bool(candidate), expect_index[p] && !nested);
        if (candidate) {
            FixedVector<int32_t> offsets{4, 0, 1, 4, 2, 3};
            auto matches = candidate->Eval(offsets);
            for (size_t row = 0; row < offsets.size(); ++row) {
                const auto& expected = expected_results[p];
                EXPECT_EQ(matches[row],
                          expected.IsValid(offsets[row]) &&
                              expected.ValueAt<bool>(offsets[row]));
            }
        }
        // Already prepared snapshots retain both the array mapping and parent validity.
        auto& execution = fresh ? fresh : prepared[p];
        auto batch = execution->prepared().MakeExecutionBatch(
            ReadBatchBinding::Range(0, count - 1));
        auto actual = batch->Eval(SelectionView::All(count - 1))[0];
        for (size_t row = 0; row < count - 1; ++row) {
            EXPECT_EQ(actual.IsValid(row), expected_results[p].IsValid(row));
            if (actual.IsValid(row) && expected_results[p].IsValid(row))
                EXPECT_EQ(actual.ValueAt<bool>(row),
                          expected_results[p].ValueAt<bool>(row));
        }
    }
}
TEST(CandidateTypedIndexTest, ArrayBitmapTypes) {
    for (auto type : {DataType::BOOL,
                      DataType::INT8,
                      DataType::INT16,
                      DataType::INT32,
                      DataType::INT64,
                      DataType::FLOAT,
                      DataType::DOUBLE,
                      DataType::VARCHAR})
        CheckArrayIndex(type, "BITMAP", false);
}
TEST(CandidateTypedIndexTest, NestedArrayBitmapTypes) {
    for (auto type : {DataType::BOOL,
                      DataType::INT8,
                      DataType::INT16,
                      DataType::INT32,
                      DataType::INT64,
                      DataType::VARCHAR})
        CheckArrayIndex(type, "BITMAP", true);
}
TEST(CandidateTypedIndexTest, ArrayOtherBackends) {
    for (bool nested : {false, true})
        for (const auto* backend : {"INVERTED", "HYBRID", "STL_SORT"}) {
            if (!nested && std::string_view(backend) == "STL_SORT")
                continue;
            for (auto type : {DataType::BOOL,
                              DataType::INT64,
                              DataType::FLOAT,
                              DataType::DOUBLE,
                              DataType::VARCHAR})
                CheckArrayIndex(type, backend, nested);
        }
}
TEST(CandidateTypedIndexTest, NestedArrayNonNullable) {
    CheckArrayIndex(DataType::INT64, "BITMAP", true, false, false);
}
TEST(CandidateTypedIndexTest, ArrayNullableElementsKeepRaw) {
    CheckArrayIndex(DataType::INT64, "BITMAP", false, true);
}

TEST(CandidateTypedIndexTest, ArrayIndexFailuresDoNotPublishResults) {
    using namespace segcore::reader;
    storage::FileManagerContext index_context(
        storage::RemoteChunkManagerSingleton::GetInstance()
            .GetRemoteChunkManager());
    index_context.fieldDataMeta.field_schema.set_data_type(
        proto::schema::Array);
    index_context.fieldDataMeta.field_schema.set_element_type(
        proto::schema::Int64);
    index_context.fieldDataMeta.field_schema.set_nullable(true);
    std::vector<ScalarFieldProto> rows(3);
    rows[0].mutable_long_data();
    rows[1].mutable_long_data()->add_data(1);
    rows[1].mutable_long_data()->add_data(2);
    rows[2].mutable_long_data();
    std::vector<Array> arrays;
    for (auto& row : rows) arrays.emplace_back(row);
    const uint8_t valid = 0b110;
    auto data = storage::CreateFieldData(DataType::ARRAY, DataType::NONE, true);
    data->FillFieldData(arrays.data(), &valid, 3, 0);
    const int64_t flattened[] = {1, 2};
    auto source =
        BuildReaderIndex<int64_t>(index::families::kSort,
                                  2,
                                  flattened,
                                  nullptr,
                                  {{"field_type", int(DataType::ARRAY)},
                                   {"value_type", int(DataType::INT64)},
                                   {"array_element_type", int(DataType::INT64)},
                                   {"nested", true}});
    StructElementOffsetsSealed offsets({0, 0, 2, 2});
    TypedIndexQuery query{};
    query.type = DataType::INT64;
    query.op = ScalarIndexOp::ContainsAll;
    query.nested_array = true;
    query.array_nullable = true;
    for (int64_t term : {1, 2}) {
        proto::plan::GenericValue v;
        v.set_int64_val(term);
        query.terms.push_back(v);
    }
    int validity_reads = 0;
    bool fail = true, cancel = false;
    auto context = std::make_shared<ReadContext>();
    ValidityColumnReader validity(
        DataType::ARRAY,
        3,
        ValidityColumnReader::ResolveValidity{
            [&](const auto&) -> ValidityColumnReader::IsValid {
                ++validity_reads;
                if (fail)
                    throw std::runtime_error(
                        "injected parent validity failure");
                return [&](int64_t row, const auto& ctx) {
                    if (cancel && row == 1)
                        throw std::runtime_error(
                            "injected cancellation during validity read");
                    ctx.Check();
                    return row != 0;
                };
            }});
    class Reader final : public IndexReader {
     public:
        std::function<IndexResult(const ReadContext&)> query;
        mutable int calls = 0;
        int64_t
        row_count() const override {
            return 3;
        }
        IndexResult
        Query(const ReadContext& context) const override {
            ++calls;
            return query(context);
        }
    };
    auto reader = std::make_shared<Reader>();
    reader->query = [&](const auto& ctx) {
        return QueryArrayIndex(*source, 3, query, &offsets, &validity, ctx);
    };
    predicate::IndexNode node(reader, context);
    auto binding = ReadBatchBinding::Offsets({2, 0, 1, 1});
    node.Eval(binding, SelectionView::None(4));
    EXPECT_EQ(reader->calls, 0);
    EXPECT_THROW(node.Eval(binding, SelectionView::All(4)), std::runtime_error);
    fail = false;
    cancel = true;
    EXPECT_THROW(node.Eval(binding, SelectionView::All(4)), std::runtime_error);
    cancel = false;
    auto actual = node.Eval(binding, SelectionView::All(4));
    EXPECT_EQ(reader->calls, 3);
    EXPECT_EQ(validity_reads, 3);
    EXPECT_TRUE(actual.IsValid(0));
    EXPECT_FALSE(actual.ValueAt<bool>(0));
    EXPECT_FALSE(actual.IsValid(1));
    EXPECT_TRUE(actual.ValueAt<bool>(2));
    EXPECT_TRUE(actual.ValueAt<bool>(3));
    node.Eval(binding, SelectionView::All(4));
    EXPECT_EQ(reader->calls, 3);
    EXPECT_THROW(
        QueryArrayIndex(*source, 4, query, &offsets, &validity, *context),
        SegcoreError);
    EXPECT_THROW(
        QueryArrayIndex(*source, 3, query, nullptr, &validity, *context),
        SegcoreError);
    EXPECT_THROW(
        QueryArrayIndex(*source, 3, query, &offsets, nullptr, *context),
        SegcoreError);
    query.nested_array = false;
    EXPECT_THROW(
        QueryArrayIndex(*source, 3, query, nullptr, &validity, *context),
        SegcoreError);
}
TEST(CandidateTypedIndexTest, SortedFloat) {
    proto::plan::GenericValue lower, upper;
    lower.set_float_val(1.25);
    upper.set_float_val(3.5);
    CheckOrdinaryTypedIndex<float>(index::families::kSort,
                                   DataType::FLOAT,
                                   {-2.f, 0.f, 1.25f, 2.5f, 4.f},
                                   lower,
                                   upper);
}
TEST(CandidateTypedIndexTest, SortedInt64OriginalValue) {
    proto::plan::GenericValue lower, upper;
    lower.set_int64_val(1);
    upper.set_int64_val(10);
    CheckOrdinaryTypedIndex<int64_t>(index::families::kSort,
                                     DataType::INT64,
                                     {-2, 0, 1, 8, 11},
                                     lower,
                                     upper);
}
TEST(CandidateTypedIndexTest, HybridDouble) {
    proto::plan::GenericValue lower, upper;
    lower.set_float_val(1.25);
    upper.set_float_val(3.5);
    CheckOrdinaryTypedIndex<double>(index::families::kHybrid,
                                    DataType::DOUBLE,
                                    {-2., 0., 1.25, 2.5, 4.},
                                    lower,
                                    upper);
}
TEST(CandidateTypedIndexTest, BitmapString) {
    proto::plan::GenericValue lower, upper;
    lower.set_string_val("b");
    upper.set_string_val("d");
    CheckOrdinaryTypedIndex<std::string>(index::families::kBitmap,
                                         DataType::VARCHAR,
                                         {"a", "b", "c", "d", "e"},
                                         lower,
                                         upper);
}

TEST(CandidateTypedIndexTest, TrieString) {
    proto::plan::GenericValue lower, upper;
    lower.set_string_val("b");
    upper.set_string_val("d");
    CheckOrdinaryTypedIndex<std::string>(index::families::kMarisa,
                                         DataType::VARCHAR,
                                         {"a", "b", "c", "d", "e"},
                                         lower,
                                         upper);
}

class JsonCandidateTest : public ::testing::Test {
 protected:
    void
    SetUp() override {
        schema = std::make_shared<Schema>();
        auto pk = schema->AddDebugField("pk", DataType::INT64);
        field = schema->AddDebugField("json", DataType::JSON, true);
        vector_field = schema->AddDebugField(
            "vec", DataType::VECTOR_FLOAT, 4, knowhere::metric::L2);
        schema->set_primary_field_id(pk);
        documents = {R"({"n": 1, "s": "a", "flag": true})",
                     R"({"n": 12, "s": "b", "flag": false})",
                     R"({"n": 12.5, "s": "c", "flag": true})",
                     R"({"n": "12", "s": 12, "flag": "true"})",
                     R"({"n": null, "s": null})",
                     R"({})",
                     R"({"n": 99, "s": "d", "flag": true})",
                     R"({"n": [12], "s": ["b"], "flag": [true]})",
                     R"({"n": -3, "s": "e"})",
                     R"({"n": 9007199254740993, "s": "f"})"};
        auto data = segcore::DataGen(schema, documents.size());
        FixedVector<bool> valid(documents.size(), true);
        valid[6] = false;
        for (auto& column : *data.raw_->mutable_fields_data()) {
            if (column.field_id() == field.get()) {
                auto replacement =
                    segcore::CreateDataArrayFrom(documents.data(),
                                                 valid.data(),
                                                 documents.size(),
                                                 (*schema)[field]);
                column.Swap(replacement.get());
            }
        }
        segment = CreateSealedWithFieldDataLoaded(schema, data);
        raw = std::make_shared<FieldData<Json>>(DataType::JSON, true);
        std::vector<Json> json;
        for (const auto& document : documents)
            json.emplace_back(simdjson::padded_string(document));
        raw->add_json_data(json);
        std::fill(raw->ValidData(),
                  raw->ValidData() + raw->ValidDataSize(),
                  uint8_t(255));
        raw->ValidData()[0] &= ~(1 << 6);
    }
    template <typename T>
    void
    LoadPath(const std::string& path,
             const std::string& cast_type,
             const std::string& backend,
             const std::string& cast_function = "unknown") {
        storage::FileManagerContext context(
            storage::RemoteChunkManagerSingleton::GetInstance()
                .GetRemoteChunkManager());
        auto& proto = context.fieldDataMeta.field_schema;
        proto.set_data_type(proto::schema::JSON);
        proto.set_fieldid(field.get());
        proto.set_nullable(true);
        context.fieldDataMeta.field_id = field.get();
        auto cast = JsonCastType::FromString(cast_type);
        auto projected = index::ConvertJsonToTypedFieldData<T>(
            {raw},
            proto,
            path,
            cast,
            JsonCastFunction::FromString(cast_function));
        const auto count = documents.size();
        FixedVector<bool> valid(count);
        for (size_t i = 0; i < count; ++i)
            valid[i] =
                (projected.field_data->ValidData()[i / 8] >> (i % 8)) & 1;
        auto source = BuildReaderIndex<T>(
            ReaderFamily(backend),
            count,
            static_cast<const T*>(projected.field_data->Data()),
            valid.data(),
            {{"field_type", int(DataType::JSON)},
             {"value_type", int(cast.ToMilvusDataType())},
             {JSON_PATH, path},
             {JSON_CAST_TYPE, cast_type}},
            [&](storage::ArtifactPtr artifact) {
                return std::make_unique<index::JsonProjectedIndexArtifact>(
                    std::move(artifact),
                    path,
                    cast,
                    count,
                    projected.non_exist_offsets,
                    true);
            });
        segcore::LoadIndexInfo info;
        info.field_id = field.get();
        info.field_type = DataType::JSON;
        info.index_params = {{"index_type", backend},
                             {JSON_PATH, path},
                             {JSON_CAST_TYPE, cast_type}};
        if (cast_function != "unknown")
            info.index_params[JSON_CAST_FUNCTION] = cast_function;
        SetTestIndexMetadata(info, *source, source.family);
        info.index = source.index;
        segment->LoadIndex(info);
    }
    std::shared_ptr<QueryContext>
    Context() {
        auto config = std::make_shared<QueryConfig>(
            std::unordered_map<std::string, std::string>{
                {QueryConfig::kEnableExprExecutionV2, "true"},
                {QueryConfig::kExprEvalBatchSize, "2"}});
        auto result = std::make_shared<QueryContext>("json-candidate",
                                                     segment.get(),
                                                     documents.size(),
                                                     Timestamp(42) << 18,
                                                     0,
                                                     0,
                                                     query::PlanOptions{},
                                                     config);
        result->set_read_snapshot(segment->CaptureReadSnapshot());
        return result;
    }
    void
    Check(const std::string& path,
          proto::plan::GenericValue lower,
          proto::plan::GenericValue upper) {
        auto snapshot = segment->CaptureReadSnapshot();
        auto read = std::make_shared<segcore::reader::ReadContext>();
        expr::ColumnInfo column(field, DataType::JSON, {path}, true);
        std::vector<expr::TypedExprPtr> predicates;
        for (auto op : {proto::plan::Equal,
                        proto::plan::NotEqual,
                        proto::plan::LessThan,
                        proto::plan::LessEqual,
                        proto::plan::GreaterThan,
                        proto::plan::GreaterEqual})
            predicates.push_back(std::make_shared<expr::UnaryRangeFilterExpr>(
                column, op, lower));
        if (!lower.has_bool_val())
            predicates.push_back(std::make_shared<expr::BinaryRangeFilterExpr>(
                column, lower, upper, true, false));
        predicates.push_back(std::make_shared<expr::TermFilterExpr>(
            column, std::vector{lower, upper}));
        predicates.push_back(std::make_shared<expr::ExistsExpr>(column));
        predicates.push_back(std::make_shared<expr::LogicalUnaryExpr>(
            expr::LogicalUnaryExpr::OpType::LogicalNot,
            std::make_shared<expr::ExistsExpr>(column)));
        predicates.push_back(std::make_shared<expr::LogicalUnaryExpr>(
            expr::LogicalUnaryExpr::OpType::LogicalNot,
            std::make_shared<expr::LogicalBinaryExpr>(
                expr::LogicalBinaryExpr::OpType::Or,
                predicates[0],
                predicates[4])));
        predicates.push_back(std::make_shared<expr::TermFilterExpr>(
            column, std::vector<proto::plan::GenericValue>{}));
        std::vector<ReadOnlyVectorView> expected;
        for (const auto& predicate : predicates) {
            auto execution = FilterBitsExecution::TryCreateWithBinding(
                predicate,
                nullptr,
                access::MakeSealedReaderBinding(snapshot, documents.size()),
                documents.size(),
                2,
                read);
            ASSERT_TRUE(execution);
            auto batch = execution->prepared().MakeExecutionBatch(
                ReadBatchBinding::Range(0, documents.size()));
            expected.push_back(
                batch->Eval(SelectionView::All(documents.size()))[0]);
        }
        segment->DropFieldData(field);
        auto query = Context();
        ExecContext context(query.get());
        FixedVector<int32_t> offsets{9, 3, 1, 6, 7, 4, 5, 2, 1, 8, 0};
        for (size_t p = 0; p < predicates.size(); ++p) {
            auto candidate =
                CandidateFilter::TryCreate(predicates[p], &context);
            ASSERT_TRUE(candidate) << p;
            auto matches = candidate->Eval(offsets);
            for (size_t i = 0; i < offsets.size(); ++i)
                EXPECT_EQ(matches[i],
                          expected[p].IsValid(offsets[i]) &&
                              expected[p].ValueAt<bool>(offsets[i]))
                    << p << " " << i;
            // Full predicate validity is tested separately from null-rejecting consumers.
            auto full =
                FilterBitsExecution::TryCreate(predicates[p],
                                               nullptr,
                                               query->get_read_snapshot(),
                                               documents.size(),
                                               2,
                                               read);
            ASSERT_TRUE(full);
            EXPECT_EQ(full->prepared().granularity(),
                      plan::ExecutionGranularity::WholeSegment);
            auto batch = full->prepared().MakeExecutionBatch(
                ReadBatchBinding::Range(0, documents.size()));
            auto actual = batch->Eval(SelectionView::All(documents.size()))[0];
            for (size_t row = 0; row < documents.size(); ++row)
                EXPECT_EQ(actual.IsValid(row), expected[p].IsValid(row))
                    << p << " " << row;
        }
        proto::plan::GenericValue precise;
        precise.set_int64_val(9007199254740993);
        auto unsupported = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(field, DataType::JSON, {"n"}, true),
            proto::plan::Equal,
            precise);
        EXPECT_FALSE(CandidateFilter::TryCreate(unsupported, &context));
    }
    void
    CheckBoostWithoutRawJson(const std::string& path,
                             proto::plan::GenericValue bound,
                             const std::vector<bool>& expected) {
        struct RestoreDefault {
            bool previous = EXPR_EXECUTION_V2_ENABLED.load();
            ~RestoreDefault() {
                SetDefaultExprExecutionV2Enabled(previous);
            }
        } restore;
        SetDefaultExprExecutionV2Enabled(true);
        segment->DropFieldData(field);
        ASSERT_FALSE(segment->HasRawData(field.get()));
        ASSERT_EQ(
            segment->CaptureReadSnapshot()->GetDataScanResources(field).first,
            nullptr);

        proto::plan::PlanNode node;
        auto* anns = node.mutable_vector_anns();
        anns->set_vector_type(proto::plan::VectorType::FloatVector);
        anns->set_field_id(vector_field.get());
        anns->set_placeholder_tag("$0");
        auto* info = anns->mutable_query_info();
        info->set_topk(10);
        info->set_metric_type(knowhere::metric::L2);
        info->set_search_params("{}");
        auto plan = query::CreateSearchPlanFromPlanNode(schema, node);
        proto::plan::ScoreFunction scorer;
        scorer.set_type(proto::plan::FunctionType::FunctionTypeWeight);
        scorer.set_weight(2.0F);
        auto* unary = scorer.mutable_filter()->mutable_unary_range_expr();
        unary->set_op(proto::plan::LessThan);
        *unary->mutable_value() = bound;
        auto* col = unary->mutable_column_info();
        col->set_field_id(field.get());
        col->set_data_type(proto::schema::JSON);
        col->set_nullable(true);
        col->add_nested_path(path);
        const auto blob = scorer.SerializeAsString();
        const std::vector<std::vector<int64_t>> chunks{
            {1, 3, 1, 6}, {}, {9, 5, 2, 0, 7, 8, 4}};
        struct ArrowChunks {
            ArrowArray arrays[3]{};
            ArrowSchema schemas[3]{};
            ~ArrowChunks() {
                for (auto& a : arrays)
                    if (a.release)
                        a.release(&a);
                for (auto& s : schemas)
                    if (s.release)
                        s.release(&s);
            }
        } exported;
        std::vector<std::vector<float>> scores(3);
        std::unique_ptr<bool[]> valid[3];
        float* outputs[3];
        bool* validity[3];
        for (size_t c = 0; c < chunks.size(); ++c) {
            arrow::Int64Builder builder;
            ASSERT_TRUE(builder.AppendValues(chunks[c]).ok());
            std::shared_ptr<arrow::Array> array;
            ASSERT_TRUE(builder.Finish(&array).ok());
            ASSERT_TRUE(arrow::ExportArray(
                            *array, &exported.arrays[c], &exported.schemas[c])
                            .ok());
            scores[c].resize(chunks[c].size(), -1.0F);
            valid[c] = std::make_unique<bool[]>(chunks[c].size());
            outputs[c] = scores[c].data();
            validity[c] = valid[c].get();
        }
        auto status = ComputeScorerScoresOnOffsetChunks(
            static_cast<CSegmentInterface>(segment.get()),
            static_cast<CSearchPlan>(plan.get()),
            blob.data(),
            blob.size(),
            exported.arrays,
            exported.schemas,
            3,
            MAX_TIMESTAMP,
            0,
            0,
            0,
            outputs,
            validity);
        const std::string message = status.error_msg ? status.error_msg : "";
        if (status.error_msg && status.error_msg[0])
            free(const_cast<char*>(status.error_msg));
        ASSERT_EQ(status.error_code, Success) << message;
        for (size_t c = 0; c < chunks.size(); ++c)
            for (size_t i = 0; i < chunks[c].size(); ++i) {
                const bool hit = expected[chunks[c][i]];
                EXPECT_EQ(validity[c][i], hit) << c << ":" << i;
                if (hit)
                    EXPECT_FLOAT_EQ(outputs[c][i], 2.0F);
            }
    }
    SchemaPtr schema;
    FieldId field{0};
    FieldId vector_field{0};
    std::unique_ptr<segcore::SegmentSealed> segment;
    std::shared_ptr<FieldData<Json>> raw;
    std::vector<std::string> documents;
};
TEST_F(JsonCandidateTest, BoostCapiNumericPathWithoutRawJson) {
    LoadPath<double>("/n", "DOUBLE", "STL_SORT");
    proto::plan::GenericValue bound;
    bound.set_float_val(100.0);
    CheckBoostWithoutRawJson(
        "n",
        bound,
        {true, true, true, false, false, false, false, false, true, false});
}
TEST_F(JsonCandidateTest, BoostCapiStringPathWithoutRawJson) {
    LoadPath<std::string>("/s", "VARCHAR", "BITMAP");
    proto::plan::GenericValue bound;
    bound.set_string_val("z");
    CheckBoostWithoutRawJson(
        "s",
        bound,
        {true, true, true, false, false, false, false, false, true, true});
}

TEST_F(JsonCandidateTest, SortedBoolPathWithoutRawJson) {
    LoadPath<bool>("/flag", "BOOL", "STL_SORT");
    proto::plan::GenericValue lower, upper;
    lower.set_bool_val(false);
    upper.set_bool_val(true);
    Check("flag", lower, upper);
}
TEST_F(JsonCandidateTest, SortedNumericPathWithoutRawJson) {
    LoadPath<double>("/n", "DOUBLE", "STL_SORT");
    proto::plan::GenericValue lower, upper;
    lower.set_int64_val(12);
    upper.set_int64_val(20);
    Check("n", lower, upper);
}
TEST_F(JsonCandidateTest, BitmapStringPathWithoutRawJson) {
    LoadPath<std::string>("/s", "VARCHAR", "BITMAP");
    proto::plan::GenericValue lower, upper;
    lower.set_string_val("b");
    upper.set_string_val("e");
    Check("s", lower, upper);
}
TEST_F(JsonCandidateTest, CastFunctionIndexPreservesRawJsonComparison) {
    LoadPath<double>("/n", "DOUBLE", "STL_SORT", "STRING_TO_DOUBLE");
    auto snapshot = segment->CaptureReadSnapshot();
    EXPECT_FALSE(
        snapshot->GetJsonPredicateResource(field, "/n", DataType::INT64, true));
    proto::plan::GenericValue bound;
    bound.set_int64_val(12);
    auto root = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(field, DataType::JSON, {"n"}, true),
        proto::plan::Equal,
        bound);
    auto read = std::make_shared<segcore::reader::ReadContext>();
    auto execution = FilterBitsExecution::TryCreate(
        root, nullptr, snapshot, documents.size(), 2, read);
    auto raw_execution = FilterBitsExecution::TryCreateWithBinding(
        root,
        nullptr,
        access::MakeSealedReaderBinding(snapshot, documents.size()),
        documents.size(),
        2,
        read);
    ASSERT_TRUE(execution && raw_execution);
    auto binding = ReadBatchBinding::Range(0, documents.size());
    auto actual_batch = execution->prepared().MakeExecutionBatch(binding);
    auto expected_batch = raw_execution->prepared().MakeExecutionBatch(binding);
    auto actual = actual_batch->Eval(SelectionView::All(documents.size()))[0];
    auto expected =
        expected_batch->Eval(SelectionView::All(documents.size()))[0];
    for (size_t row = 0; row < documents.size(); ++row) {
        EXPECT_EQ(actual.IsValid(row), expected.IsValid(row));
        if (actual.IsValid(row) && expected.IsValid(row))
            EXPECT_EQ(actual.ValueAt<bool>(row), expected.ValueAt<bool>(row));
    }
    EXPECT_TRUE(actual.IsValid(1) && actual.ValueAt<bool>(1));
    EXPECT_FALSE(actual.IsValid(3) && actual.ValueAt<bool>(3));
}
TEST_F(JsonCandidateTest, InvertedNumericPathWithoutRawJson) {
    LoadPath<double>("/n", "DOUBLE", "INVERTED");
    proto::plan::GenericValue lower, upper;
    lower.set_float_val(12.5);
    upper.set_float_val(20.5);
    Check("n", lower, upper);
}
}  // namespace
}  // namespace milvus::exec

namespace milvus::exec {
namespace {
struct PrimaryKeyLoadProbe {
    int loads = 0;
    bool fail = false;
};
class ObservedPrimaryKeySnapshot final : public segcore::SegmentReadSnapshot {
 public:
    ObservedPrimaryKeySnapshot(
        std::shared_ptr<const segcore::SegmentReadSnapshot> source,
        std::shared_ptr<PrimaryKeyLoadProbe> probe)
        : source_(std::move(source)), probe_(std::move(probe)) {
    }
    int64_t
    chunk_size(FieldId f,
               int64_t c,
               OpContext* op_ctx = nullptr) const override {
        return source_->chunk_size(f, c, op_ctx);
    }
    int64_t
    num_rows_until_chunk(FieldId f,
                         int64_t c,
                         OpContext* op_ctx = nullptr) const override {
        return source_->num_rows_until_chunk(f, c, op_ctx);
    }
    std::pair<int64_t, int64_t>
    get_chunk_by_offset(FieldId f,
                        int64_t o,
                        OpContext* op_ctx = nullptr) const override {
        return source_->get_chunk_by_offset(f, o, op_ctx);
    }
    int64_t
    num_chunk_data(FieldId f, OpContext* op_ctx = nullptr) const override {
        return source_->num_chunk_data(f, op_ctx);
    }
    int64_t
    get_row_count() const override {
        return source_->get_row_count();
    }
    std::pair<std::shared_ptr<ChunkedColumnInterface>, FieldSkipMetricsView>
    GetDataScanResources(FieldId f) const override {
        return source_->GetDataScanResources(f);
    }
    segcore::PrimaryKeyLookupLoader
    GetPrimaryKeyLookup(FieldId f, DataType t) const override {
        auto load = source_->GetPrimaryKeyLookup(f, t);
        if (!load)
            return {};
        return [load = std::move(load), probe = probe_](OpContext* context) {
            ++probe->loads;
            if (probe->fail)
                ThrowInfo(FieldNotLoaded, "injected prepared PK load failure");
            return load(context);
        };
    }

 private:
    std::shared_ptr<const segcore::SegmentReadSnapshot> source_;
    std::shared_ptr<PrimaryKeyLoadProbe> probe_;
};

template <typename T>
void
CheckPrimaryKey(const std::vector<T>& values,
                DataType type,
                bool sealed = false,
                bool sorted = false,
                std::optional<int64_t> virtual_segment = std::nullopt) {
    auto schema = std::make_shared<Schema>();
    const auto pk = schema->AddDebugField("pk", type);
    schema->set_primary_field_id(pk);
    auto growing =
        sealed ? nullptr : segcore::CreateGrowingSegment(schema, nullptr);
    segcore::SegmentSealedUPtr sealed_segment;
    auto make_data = [&](const std::vector<T>& input) {
        auto data = segcore::DataGen(schema, input.size());
        for (auto& column : *data.raw_->mutable_fields_data()) {
            if (column.field_id() != pk.get())
                continue;
            auto replacement = segcore::CreateDataArrayFrom(
                input.data(), nullptr, input.size(), (*schema)[pk]);
            column.Swap(replacement.get());
        }
        return data;
    };
    auto append = [&](const std::vector<T>& input, int64_t start) {
        auto data = make_data(input);
        for (size_t i = 0; i < input.size(); ++i) {
            data.row_ids_[i] = start + i;
            data.timestamps_[i] = 10 + start + i;
        }
        const auto offset = growing->PreInsert(input.size());
        ASSERT_EQ(offset, start);
        growing->Insert(offset,
                        input.size(),
                        data.row_ids_.data(),
                        data.timestamps_.data(),
                        data.SharedRaw());
    };
    if (sealed) {
        std::vector<T> first(values.begin(), values.begin() + 3);
        std::vector<T> second(values.begin() + 3, values.end());
        // Query a strict prefix of the physical column, including duplicates
        // spanning the two chunks. The suffix must never enter the result.
        second.push_back(values.back());
        second.push_back(values.back());
        sealed_segment = CreateTwoChunkSealed(
            schema, make_data(first), make_data(second), sorted);
        if (virtual_segment) {
            auto* impl = dynamic_cast<segcore::ChunkedSegmentSealedImpl*>(
                sealed_segment.get());
            auto runtime = impl->TestCloneMutableRuntimeResourceState();
            runtime->virtual_pk2offset =
                std::make_shared<const segcore::VirtualPKOffsetMap>(
                    *virtual_segment, values.size() + 2);
            runtime->pk_index_slot.reset();
            runtime->fields.erase(pk);
            impl->TestPublishRuntimeResourceState(std::move(runtime));
        }
    } else {
        append(values, 0);
    }
    auto* segment =
        sealed ? static_cast<segcore::SegmentInternalInterface*>(
                     sealed_segment.get())
               : static_cast<segcore::SegmentInternalInterface*>(growing.get());
    auto snapshot = segment->CaptureReadSnapshot();
    if (virtual_segment)
        ASSERT_FALSE(snapshot->GetDataScanResources(pk).first);
    else if (sealed)
        ASSERT_EQ(snapshot->num_chunk_data(pk), 2);
    auto probe = std::make_shared<PrimaryKeyLoadProbe>();
    if (sealed)
        snapshot =
            std::make_shared<ObservedPrimaryKeySnapshot>(snapshot, probe);
    auto literal = [](const T& value) {
        proto::plan::GenericValue result;
        if constexpr (std::is_same_v<T, int64_t>)
            result.set_int64_val(value);
        else
            result.set_string_val(value);
        return result;
    };
    expr::ColumnInfo column(pk, type);
    std::vector<std::pair<expr::TypedExprPtr, std::vector<bool>>> cases;
    auto add = [&](expr::TypedExprPtr expression, auto truth) {
        std::vector<bool> expected;
        for (const auto& v : values) expected.push_back(truth(v));
        cases.emplace_back(std::move(expression), std::move(expected));
    };
    const auto lower = values[1], upper = values[4];
    const auto extremes = [&]() -> std::vector<T> {
        if constexpr (std::is_same_v<T, int64_t>)
            return {std::numeric_limits<int64_t>::min(),
                    lower,
                    std::numeric_limits<int64_t>::max()};
        else
            return {"", lower, "zzzz"};
    }();
    for (const auto& target : extremes) {
        for (auto op : {proto::plan::Equal,
                        proto::plan::NotEqual,
                        proto::plan::LessThan,
                        proto::plan::LessEqual,
                        proto::plan::GreaterThan,
                        proto::plan::GreaterEqual}) {
            add(std::make_shared<expr::UnaryRangeFilterExpr>(
                    column, op, literal(target)),
                [=](const T& v) {
                    switch (op) {
                        case proto::plan::Equal:
                            return v == target;
                        case proto::plan::NotEqual:
                            return v != target;
                        case proto::plan::LessThan:
                            return v < target;
                        case proto::plan::LessEqual:
                            return v <= target;
                        case proto::plan::GreaterThan:
                            return v > target;
                        default:
                            return v >= target;
                    }
                });
        }
    }
    for (bool li : {false, true})
        for (bool ui : {false, true}) {
            add(std::make_shared<expr::BinaryRangeFilterExpr>(
                    column, literal(lower), literal(upper), li, ui),
                [=](const T& v) {
                    return (li ? v >= lower : v > lower) &&
                           (ui ? v <= upper : v < upper);
                });
        }
    add(std::make_shared<expr::BinaryRangeFilterExpr>(
            column, literal(upper), literal(lower), true, true),
        [](const T&) { return false; });
    auto in = std::make_shared<expr::TermFilterExpr>(
        column, std::vector{literal(lower), literal(upper), literal(lower)});
    add(in, [=](const T& v) { return v == lower || v == upper; });
    add(std::make_shared<expr::LogicalUnaryExpr>(
            expr::LogicalUnaryExpr::OpType::LogicalNot, in),
        [=](const T& v) { return v != lower && v != upper; });
    add(std::make_shared<expr::TermFilterExpr>(
            column, std::vector<proto::plan::GenericValue>{}),
        [](const T&) { return false; });

    auto read = std::make_shared<segcore::reader::ReadContext>();
    plan::PhysicalPrepareOptions physical;
    physical.bind_index = access::MakeSegmentIndexBinding(
        *segment, snapshot, values.size(), read);
    if (type == DataType::VARCHAR) {
        // A text predicate also has a default-initialized integer descriptor.
        // It must never be mistaken for PK equality on the empty string.
        plan::PredicatePlan text{};
        text.field.field_id = pk.get();
        text.field.type = type;
        text.text_query.emplace();
        EXPECT_FALSE(physical.bind_index(text));
    }
    std::vector<std::unique_ptr<FilterBitsExecution>> executions;
    for (const auto& [expression, expected] : cases) {
        auto execution = FilterBitsExecution::TryCreateWithBinding(
            expression,
            nullptr,
            [](const InputDependency&)
                -> std::shared_ptr<const segcore::reader::ColumnReader> {
                ADD_FAILURE() << "PK index plan must not bind raw inputs";
                return nullptr;
            },
            values.size(),
            2,
            read,
            physical,
            schema);
        ASSERT_TRUE(execution);
        EXPECT_EQ(execution->prepared().granularity(),
                  plan::ExecutionGranularity::WholeSegment);
        for (auto backend : execution->prepared().predicate_backends())
            EXPECT_EQ(backend, plan::PredicateBackend::PkIndex);
        executions.push_back(std::move(execution));
    }
    EXPECT_EQ(probe->loads, 0);
    // Drop the preparation callback too: only published readers may retain
    // the map after segment release, not this test's temporary binder.
    physical.bind_index = {};
    // A lookup after append must still use the original prefix. Duplicate PKs
    // are physical rows, not deduplicated IDs; release must preserve the map.
    if (!sealed)
        append({lower, upper}, values.size());
    snapshot.reset();
    growing.reset();
    if (sealed_segment)
        dynamic_cast<segcore::ChunkedSegmentSealedImpl*>(sealed_segment.get())
            ->TestClearPublishedState();
    sealed_segment.reset();
    const std::vector<int32_t> offsets{5, 1, 0, 1, 4, 2, 3};
    for (size_t c = 0; c < cases.size(); ++c) {
        auto batch = executions[c]->prepared().MakeExecutionBatch(
            ReadBatchBinding::Offsets(offsets));
        auto empty = batch->Eval(SelectionView::None(offsets.size()))[0];
        EXPECT_TRUE(empty.defined_rows().IsNone());
        if (c == 0 && sealed) {
            EXPECT_EQ(probe->loads, 0);
            probe->fail = true;
            try {
                batch->Eval(SelectionView::All(offsets.size()));
                FAIL() << "PK load failure must propagate without Raw fallback";
            } catch (const SegcoreError& error) {
                EXPECT_EQ(error.get_error_code(), FieldNotLoaded);
            }
            EXPECT_EQ(probe->loads, 1);
            probe->fail = false;
        }
        if (c == 0) {
            int checks = 0;
            // Fail after the PK map has been queried, before publication.
            read->checkpoint = [&] {
                if (++checks == 4)
                    throw folly::FutureCancellation();
            };
            EXPECT_THROW(batch->Eval(SelectionView::All(offsets.size())),
                         folly::FutureCancellation);
            EXPECT_EQ(checks, 4);
            read->checkpoint = {};
        }
        auto result = batch->Eval(SelectionView::All(offsets.size()))[0];
        for (size_t i = 0; i < offsets.size(); ++i) {
            EXPECT_TRUE(result.IsValid(i));
            EXPECT_EQ(result.ValueAt<bool>(i), cases[c].second[offsets[i]])
                << c << ":" << i;
        }
        const auto loaded = probe->loads;
        auto all = executions[c]->prepared().MakeExecutionBatch(
            ReadBatchBinding::Range(0, values.size()));
        result = all->Eval(SelectionView::All(values.size()))[0];
        EXPECT_EQ(probe->loads, loaded);
        for (size_t i = 0; i < values.size(); ++i)
            EXPECT_EQ(result.ValueAt<bool>(i), cases[c].second[i]);
    }
}
}  // namespace

TEST(GrowingPrimaryKeyMigrationTest, Int64PrefixDuplicatesAndLifetime) {
    CheckPrimaryKey<int64_t>({10, 20, 10, 30, 40, 20}, DataType::INT64);
}
TEST(GrowingPrimaryKeyMigrationTest, StringPrefixDuplicatesAndLifetime) {
    CheckPrimaryKey<std::string>({"a", "b", "a", "c", "d", "b"},
                                 DataType::VARCHAR);
}
TEST(SealedPrimaryKeyMigrationTest, Int64UnsortedPrefixAndLifetime) {
    CheckPrimaryKey<int64_t>({10, 20, 10, 30, 40, 20}, DataType::INT64, true);
}
TEST(SealedPrimaryKeyMigrationTest, StringUnsortedPrefixAndLifetime) {
    CheckPrimaryKey<std::string>(
        {"a", "b", "a", "c", "d", "b"}, DataType::VARCHAR, true);
}
TEST(SealedPrimaryKeyMigrationTest, Int64SortedCrossChunkDuplicates) {
    CheckPrimaryKey<int64_t>(
        {10, 20, 20, 20, 40, 50}, DataType::INT64, true, true);
}
TEST(SealedPrimaryKeyMigrationTest, StringSortedCrossChunkDuplicates) {
    CheckPrimaryKey<std::string>(
        {"a", "b", "b", "b", "d", "e"}, DataType::VARCHAR, true, true);
}
TEST(SealedPrimaryKeyMigrationTest,
     VirtualPrefixLifetimeAndExtremeComparisons) {
    for (int64_t segment_id : {int64_t{123}, int64_t{0x9abcdef0}}) {
        std::vector<int64_t> values;
        for (int64_t i = 0; i < 6; ++i)
            values.push_back(GetVirtualPK(segment_id, i));
        CheckPrimaryKey(values, DataType::INT64, true, false, segment_id);
    }
}
TEST(GrowingPrimaryKeyMigrationTest, ClearPreservesPublishedMap) {
    for (auto type : {DataType::INT64, DataType::VARCHAR}) {
        Schema schema;
        const auto pk = schema.AddDebugField("pk", type);
        schema.set_primary_field_id(pk);
        segcore::InsertRecord<false> record(schema, 16);
        PkType key = type == DataType::INT64 ? PkType(int64_t{42})
                                             : PkType(std::string("key"));
        record.pk2offset_->insert(key, 0);
        auto published = record.pk2offset_;
        record.clear();
        EXPECT_TRUE(record.pk2offset_->empty());
        EXPECT_EQ(published->find(key), std::vector<int64_t>{0});
        record.pk2offset_->insert(key, 1);
        EXPECT_EQ(published->find(key), std::vector<int64_t>{0});
    }
}
}  // namespace milvus::exec
