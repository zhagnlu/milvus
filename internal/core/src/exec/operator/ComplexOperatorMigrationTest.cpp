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
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific
// language governing permissions and limitations under the License.

#include <gtest/gtest.h>
#include "segcore/reader/ReaderTestUtils.h"
#include "index/contracts/query/ISpatialReader.h"
#include <chrono>
#include <filesystem>
#include "common/Geometry.h"
#include "rescores/BoostScoreRunner.h"
#include "test_utils/cachinglayer_test_utils.h"
#include "exec/QueryContext.h"
#include "exec/expression/Expr.h"
#include "exec/expression/EvalCtx.h"
#include "exec/expression/ScalarComplexPredicates.h"
#include "exec/expression/ScalarArithmetic.h"
#include "exec/expression/InputRefExpr.h"
#include "exec/operator/CandidateFilter.h"
#include "exec/operator/ElementFilterBitsNode.h"
#include "exec/operator/IterativeElementFilterNode.h"
#include "exec/Task.h"
#include "common/ElementFilterIterator.h"
#include "exec/operator/FilterBitsExecution.h"
#include "exec/plan/ExprCompiler.h"
#include "exec/plan/ExecutionPreparer.h"
#include "segcore/SegmentGrowingImpl.h"
#include "segcore/segment_c.h"
#include "segcore/reader/GrowingColumnReader.h"
#include "test_utils/storage_test_utils.h"

namespace milvus::exec {
namespace {
proto::plan::GenericValue
Int(int64_t value) {
    proto::plan::GenericValue result;
    result.set_int64_val(value);
    return result;
}

// Compare complete three-valued output and the null-rejecting consumer.
// Use legacy execution for supported semantics, or an explicit truth table
// for nullable elements that legacy flattening does not preserve.
void
CheckExecution(segcore::SegmentInternalInterface& segment,
               const expr::TypedExprPtr& predicate,
               int64_t count,
               const std::vector<int>& truth = {}) {
    SCOPED_TRACE(::testing::Message()
                 << "segment=" << static_cast<int>(segment.type())
                 << ", explicit_truth=" << !truth.empty() << ", "
                 << predicate->ToString());
    VectorPtr expected_column;
    if (!truth.empty()) {
        ASSERT_EQ(truth.size(), count);
        auto values = std::make_shared<ColumnVector>(DataType::BOOL, count);
        for (int64_t row = 0; row < count; ++row) {
            if (truth[row] < 0)
                values->nullAt(row);
            else
                values->SetValueAt<bool>(row, truth[row] != 0);
        }
        expected_column = std::move(values);
    } else {
        auto old_config = std::make_shared<QueryConfig>(
            std::unordered_map<std::string, std::string>{
                {QueryConfig::kExprEvalBatchSize, std::to_string(count)}});
        QueryContext old_query("old-complex",
                               &segment,
                               count,
                               MAX_TIMESTAMP,
                               0,
                               0,
                               query::PlanOptions{},
                               old_config);
        old_query.set_read_snapshot(segment.CaptureReadSnapshot());
        ExecContext old_context(&old_query);
        ExprSet legacy({predicate}, &old_context);
        EvalCtx legacy_ctx(&old_context);
        std::vector<VectorPtr> results;
        legacy.Eval(0, 1, true, legacy_ctx, results);
        ASSERT_EQ(results.size(), 1);
        expected_column = results[0];
    }
    auto expected = ReadOnlyVectorView::CopyOf(expected_column);
    ASSERT_EQ(expected.size(), count);
    auto snapshot = segment.CaptureReadSnapshot();
    auto read = std::make_shared<segcore::reader::ReadContext>();
    plan::PhysicalPrepareOptions physical;
    physical.bind_index =
        access::MakeSegmentIndexBinding(segment, snapshot, count, read);
    auto execution = FilterBitsExecution::TryCreateWithBinding(
        predicate,
        nullptr,
        access::MakeSegmentReaderBinding(segment, snapshot, count),
        count,
        2,
        read,
        physical,
        segment.get_schema_snapshot());
    ASSERT_TRUE(execution);
    if (dynamic_cast<const expr::GISFunctionFilterExpr*>(predicate.get()) ||
        dynamic_cast<const expr::MatchExpr*>(predicate.get()))
        EXPECT_EQ(execution->prepared().granularity(),
                  plan::ExecutionGranularity::Batch);
    const std::vector<int64_t> offsets{count - 1, 1, 0, 3, 1, 2};
    auto batch = execution->prepared().MakeExecutionBatch(
        ReadBatchBinding::Offsets(offsets));
    auto selected = OwnedSelection::FromIndices(offsets.size(), {0, 1, 3, 4});
    auto actual = batch->Eval(selected.view())[0];
    for (size_t i = 0; i < offsets.size(); ++i) {
        EXPECT_EQ(actual.defined_rows().IsSelected(i),
                  selected.view().IsSelected(i));
        if (!selected.view().IsSelected(i))
            continue;
        EXPECT_EQ(actual.IsValid(i), expected.IsValid(offsets[i]));
        if (actual.IsValid(i) && expected.IsValid(offsets[i]))
            EXPECT_EQ(actual.ValueAt<bool>(i),
                      expected.ValueAt<bool>(offsets[i]));
    }
    auto full = batch->Eval(SelectionView::All(offsets.size()))[0];
    for (size_t i = 0; i < offsets.size(); ++i) {
        EXPECT_EQ(full.IsValid(i), expected.IsValid(offsets[i]));
        if (full.IsValid(i) && expected.IsValid(offsets[i]))
            EXPECT_EQ(full.ValueAt<bool>(i),
                      expected.ValueAt<bool>(offsets[i]));
    }
    auto filtered = execution->GetOutput();
    ASSERT_TRUE(filtered);
    auto excluded = ReadOnlyVectorView::CopyOf(filtered->child(0));
    for (int64_t i = 0; i < count; ++i)
        EXPECT_EQ(excluded.ValueAt<bool>(i),
                  !(expected.IsValid(i) && expected.ValueAt<bool>(i)));
    auto config = std::make_shared<QueryConfig>(
        std::unordered_map<std::string, std::string>{
            {QueryConfig::kEnableExprExecutionV2, "true"}});
    QueryContext query_context("new-complex",
                               &segment,
                               count,
                               MAX_TIMESTAMP,
                               0,
                               0,
                               query::PlanOptions{},
                               config);
    ExecContext context(&query_context);
    auto candidate = CandidateFilter::TryCreate(predicate, &context);
    ASSERT_TRUE(candidate);
    auto matches = candidate->Eval(
        FixedVector<int32_t>{int32_t(count - 1), 1, 0, 3, 1, 2});
    for (size_t i = 0; i < offsets.size(); ++i)
        EXPECT_EQ(
            matches[i],
            expected.IsValid(offsets[i]) && expected.ValueAt<bool>(offsets[i]));
}

TEST(ComplexOperatorMigrationTest, RTreeCandidatesRefineAndPreserveNulls) {
    auto schema = std::make_shared<Schema>();
    auto pk = schema->AddDebugField("pk", DataType::INT64);
    schema->set_primary_field_id(pk);
    auto field = schema->AddDebugField("geometry", DataType::GEOMETRY, true);
    auto data = segcore::DataGen(schema, 6);
    std::vector<std::string> wkb;
    for (const auto* text : {"POINT(0 0)",
                             "POINT(20 20)",
                             "LINESTRING(-1 0, 1 0)",
                             "POINT EMPTY",
                             "POINT(0 0)",
                             "POLYGON((-2 -2,2 -2,2 2,-2 2,-2 -2))"})
        wkb.push_back(
            Geometry(GetThreadLocalGEOSContext(), text).to_wkb_string());
    for (auto& col : *data.raw_->mutable_fields_data()) {
        if (col.field_id() != field.get())
            continue;
        col.clear_valid_data();
        auto* scalar = col.mutable_scalars();
        scalar->clear_valid_data();
        scalar->mutable_geometry_data()->clear_data();
        for (size_t i = 0; i < wkb.size(); ++i) {
            scalar->mutable_geometry_data()->add_data(wkb[i]);
            scalar->add_valid_data(i != 4);
        }
    }
    auto segment = CreateSealedWithFieldDataLoaded(schema, data);
    storage::FileManagerContext context(
        storage::RemoteChunkManagerSingleton::GetInstance()
            .GetRemoteChunkManager());
    context.fieldDataMeta.field_schema.set_data_type(proto::schema::Geometry);
    context.fieldDataMeta.field_schema.set_nullable(true);
    context.fieldDataMeta.field_schema.set_fieldid(field.get());
    context.fieldDataMeta.field_id = field.get();
    const bool geometry_valid[] = {true, true, true, true, false, true};
    auto source = BuildTestScalarIndex<std::string>(
        index::families::kRTree,
        wkb.size(),
        wkb.data(),
        geometry_valid,
        {{"field_type", int(DataType::GEOMETRY)}});
    auto resource = source.index;
    segcore::LoadIndexInfo info;
    info.field_id = field.get();
    info.field_type = DataType::GEOMETRY;
    info.index_params["index_type"] = "RTREE";
    SetTestIndexMetadata(info, *source, index::families::kRTree);
    info.index = resource;
    segment->LoadIndex(info);
    for (int operation = proto::plan::GISFunctionFilterExpr_GISOp_Equals;
         operation <= proto::plan::GISFunctionFilterExpr_GISOp_Within;
         ++operation) {
        auto root = std::make_shared<expr::GISFunctionFilterExpr>(
            expr::ColumnInfo(field, DataType::GEOMETRY, {}, true),
            static_cast<GISFunctionType>(operation),
            "POINT(0 0)",
            0);
        CheckExecution(*segment, root, 6);
        CheckExecution(*segment,
                       std::make_shared<expr::LogicalUnaryExpr>(
                           expr::LogicalUnaryExpr::OpType::LogicalNot, root),
                       6);
        auto read = std::make_shared<segcore::reader::ReadContext>();
        auto execution = FilterBitsExecution::TryCreate(
            root, nullptr, segment->CaptureReadSnapshot(), 6, 2, read);
        ASSERT_TRUE(execution);
        EXPECT_EQ(execution->prepared().predicate_routes(),
                  std::vector{plan::PredicateRoute::Refine});
        EXPECT_EQ(execution->prepared().granularity(),
                  plan::ExecutionGranularity::Batch);
    }
    auto point_predicate = std::make_shared<expr::GISFunctionFilterExpr>(
        expr::ColumnInfo(field, DataType::GEOMETRY, {}, true),
        proto::plan::GISFunctionFilterExpr_GISOp_Equals,
        "POINT(0 0)",
        0);
    auto area_predicate = std::make_shared<expr::GISFunctionFilterExpr>(
        expr::ColumnInfo(field, DataType::GEOMETRY, {}, true),
        proto::plan::GISFunctionFilterExpr_GISOp_Intersects,
        "POLYGON((-3 -3,3 -3,3 3,-3 3,-3 -3))",
        0);
    for (auto operation : {expr::LogicalBinaryExpr::OpType::And,
                           expr::LogicalBinaryExpr::OpType::Or}) {
        auto root = std::make_shared<expr::LogicalBinaryExpr>(
            operation, point_predicate, area_predicate);
        auto truth = operation == expr::LogicalBinaryExpr::OpType::And
                         ? std::vector<int>{1, 0, 0, 0, -1, 0}
                         : std::vector<int>{1, 0, 1, 0, -1, 1};
        CheckExecution(*segment, root, 6, truth);
        for (auto& value : truth)
            if (value >= 0)
                value = !value;
        CheckExecution(*segment,
                       std::make_shared<expr::LogicalUnaryExpr>(
                           expr::LogicalUnaryExpr::OpType::LogicalNot, root),
                       6,
                       truth);
    }
    class ShortIndex : public index::IIndexReaderBase,
                       public index::ISpatialReader,
                       public index::INullReader {
     public:
        explicit ShortIndex(index::IIndexReaderBasePtr base)
            : base_(std::move(base)) {
        }
        index::ReaderCaps
        Caps() const override {
            return base_->Caps();
        }
        index::Domain
        CoordDomain() const override {
            return base_->CoordDomain();
        }
        int64_t
        Count() const override {
            return base_->Count();
        }
        DataType
        ValueType() const override {
            return base_->ValueType();
        }
        TargetBitmap
        Candidates(index::SpatialOp, const Geometry&) const override {
            return TargetBitmap(2, false);
        }
        TargetBitmap
        IsNull() const override {
            return segcore::reader::NullReader(*base_).IsNull();
        }
        TargetBitmap
        IsNotNull() const override {
            return segcore::reader::NullReader(*base_).IsNotNull();
        }

     private:
        index::IIndexReaderBasePtr base_;
    };
    auto short_resource = std::make_shared<segcore::reader::TestReaderIndex>(
        resource->Metadata(), [resource](OpContext* ctx) {
            return std::make_unique<ShortIndex>(resource->CreateReader(ctx));
        });
    segcore::reader::GeometryIndexReader short_reader(
        short_resource,
        6,
        {proto::plan::GISFunctionFilterExpr_GISOp_Intersects, "POINT(0 0)"});
    auto promoted = short_reader.Query({});
    EXPECT_EQ(promoted.selected().selected_count(), 5);
    EXPECT_TRUE(promoted.null_rows().IsSelected(4));
    EXPECT_TRUE(promoted.false_rows().IsNone());
}

TEST(ComplexOperatorMigrationTest, RecursiveArrayLengthAndMatch) {
    for (auto [leaf, deeper] : {std::pair{DataType::INT64, false},
                                std::pair{DataType::VARCHAR, false},
                                std::pair{DataType::INT64, true},
                                std::pair{DataType::VARCHAR, true}}) {
        SCOPED_TRACE(::testing::Message()
                     << static_cast<int>(leaf) << ":" << deeper);
        auto schema = std::make_shared<Schema>();
        auto pk = schema->AddDebugField("pk", DataType::INT64);
        schema->set_primary_field_id(pk);
        auto data = segcore::DataGen(schema, 6);
        FieldId field(pk.get() + 1);
        proto::schema::TypeSchema type;
        type.set_nullable(true);
        auto* child = type.mutable_array_element();
        child->set_nullable(true);
        auto* leaf_type = child->mutable_array_element();
        if (deeper) {
            leaf_type->set_nullable(true);
            leaf_type = leaf_type->mutable_array_element();
        }
        leaf_type->set_leaf_type(static_cast<proto::schema::DataType>(leaf));
        leaf_type->set_nullable(false);
        schema->AddField(FieldMeta(FieldName("items[values]"),
                                   field,
                                   DataType::ARRAY,
                                   DataType::ARRAY,
                                   true,
                                   true,
                                   std::nullopt,
                                   std::string{},
                                   LOCAL_FORMAT_RAW,
                                   type));
        const std::vector<std::vector<std::vector<int>>> arrays{
            {}, {{1, 2}, {3}}, {{}, {4, 5}}, {{}}, {}, {{1, 2}, {1, 2}}};
        auto* col = data.raw_->add_fields_data();
        col->set_field_id(field.get());
        col->set_type(proto::schema::Array);
        auto* scalars = col->mutable_scalars();
        auto* rows = scalars->mutable_array_data();
        rows->set_element_type(proto::schema::Array);
        for (size_t i = 0; i < arrays.size(); ++i) {
            scalars->add_valid_data(i != 4);
            auto* row = rows->add_data();
            auto* children = row->mutable_array_data();
            children->set_element_type(
                deeper ? proto::schema::Array
                       : static_cast<proto::schema::DataType>(leaf));
            for (size_t j = 0; j < arrays[i].size(); ++j) {
                auto* element = children->add_data();
                // Recursive ARRAY encodes NULL child arrays as DATA_NOT_SET.
                if (i == 2 && j == 0)
                    continue;
                if (deeper)
                    element->mutable_array_data()->set_element_type(
                        static_cast<proto::schema::DataType>(leaf));
                else if (leaf == DataType::INT64)
                    element->mutable_long_data();
                else
                    element->mutable_string_data();
                for (int value : arrays[i][j]) {
                    element->add_valid_data(true);
                    auto* value_row =
                        deeper ? element->mutable_array_data()->add_data()
                               : element;
                    if (deeper)
                        value_row->add_valid_data(true);
                    if (leaf == DataType::INT64)
                        value_row->mutable_long_data()->add_data(value);
                    else
                        value_row->mutable_string_data()->add_data(
                            std::to_string(value));
                }
            }
        }
        auto growing = segcore::CreateGrowingSegment(schema, empty_index_meta);
        growing->PreInsert(6);
        growing->Insert(0,
                        6,
                        data.row_ids_.data(),
                        data.timestamps_.data(),
                        data.SharedRaw());
        // Recursive ARRAY storage is V2/V3-only. Flush through the real V3
        // writer instead of feeding it through the legacy V1 test loader.
        const auto path =
            (std::filesystem::temp_directory_path() /
             ("scalar-recursive-array-" +
              std::to_string(
                  std::chrono::steady_clock::now().time_since_epoch().count())))
                .string();
        auto cleanup =
            folly::makeGuard([&] { std::filesystem::remove_all(path); });
        auto& config = segcore::SegcoreConfig::default_config();
        const auto lazy = config.get_lazy_column_group_enabled();
        config.set_lazy_column_group_enabled(false);
        auto restore_lazy = folly::makeGuard(
            [&] { config.set_lazy_column_group_enabled(lazy); });
        auto flush_schema = schema->ToProto();
        for (auto [id, name] : {std::pair{RowFieldID, "RowID"},
                                std::pair{TimestampFieldID, "Timestamp"}}) {
            auto* system = flush_schema.add_fields();
            system->set_fieldid(id.get());
            system->set_name(name);
            system->set_data_type(proto::schema::Int64);
        }
        const auto schema_blob = flush_schema.SerializeAsString();
        CFlushConfig flush_config{};
        flush_config.segment_path = path.c_str();
        flush_config.read_version = -1;
        flush_config.retry_limit = 3;
        flush_config.schema_blob = schema_blob.data();
        flush_config.schema_length = schema_blob.size();
        flush_config.schema_based_pattern = "0|1|100,101";
        CFlushResult flush{};
        auto free_flush = folly::makeGuard([&] { FreeFlushResult(&flush); });
        auto status =
            FlushGrowingSegmentData(growing.get(), 0, 6, &flush_config, &flush);
        ASSERT_EQ(status.error_code, Success) << status.error_msg;
        proto::segcore::SegmentLoadInfo load;
        load.set_collectionid(1);
        load.set_partitionid(2);
        load.set_segmentid(3);
        load.set_storageversion(STORAGE_V3);
        load.set_num_of_rows(6);
        load.set_manifest_path("{\"base_path\":\"" + path + "\",\"ver\":" +
                               std::to_string(flush.committed_version) + "}");
        load.set_insert_channel("scalar-recursive-array-test");
        auto sealed = segcore::CreateSealedSegment(
            Schema::ParseFrom(flush_schema), empty_index_meta, 3);
        sealed->SetLoadInfo(load);
        milvus::tracer::TraceContext trace;
        sealed->Load(trace, nullptr);
        expr::ColumnInfo outer(
            field, DataType::ARRAY, DataType::ARRAY, {}, true);
        auto root_length = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
            outer,
            proto::plan::Equal,
            proto::plan::ArrayLength,
            Int(2),
            Int(0));
        auto null = std::make_shared<expr::NullExpr>(
            outer, proto::plan::NullExpr_NullOp_IsNull);
        outer.element_level_ = true;
        auto length = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
            outer,
            proto::plan::Equal,
            proto::plan::ArrayLength,
            Int(2),
            Int(0));
        auto literal = Int(2);
        if (leaf == DataType::VARCHAR)
            literal.set_string_val("2");
        auto contains = std::make_shared<expr::JsonContainsExpr>(
            outer,
            proto::plan::JSONContainsExpr_JSONOp_Contains,
            true,
            std::vector<proto::plan::GenericValue>{literal});
        const auto check = [&](auto& segment) {
            CheckExecution(segment, root_length, 6, {0, 1, 1, 0, -1, 1});
            CheckExecution(segment, null, 6, {0, 0, 0, 0, 1, 0});
            CheckExecution(segment,
                           std::make_shared<expr::MatchExpr>(
                               "items",
                               proto::plan::MatchAny,
                               1,
                               std::make_shared<expr::NullExpr>(
                                   outer, proto::plan::NullExpr_NullOp_IsNull)),
                           6,
                           {0, 0, 1, 0, -1, 0});
            CheckExecution(segment,
                           std::make_shared<expr::MatchExpr>(
                               "items", proto::plan::MatchAny, 1, length),
                           6,
                           {0, 1, 1, 0, -1, 1});
            CheckExecution(segment,
                           std::make_shared<expr::MatchExpr>(
                               "items", proto::plan::MatchAll, 1, length),
                           6,
                           {1, 0, 1, 0, -1, 1});
            CheckExecution(segment,
                           std::make_shared<expr::MatchExpr>(
                               "items", proto::plan::MatchExact, 2, length),
                           6,
                           {0, 0, 0, 0, -1, 1});
            if (!deeper)
                CheckExecution(segment,
                               std::make_shared<expr::MatchExpr>(
                                   "items", proto::plan::MatchAny, 1, contains),
                               6,
                               {0, 1, 0, 0, -1, 1});
        };
        check(*growing);
        check(*sealed);
        auto& mmap = storage::MmapManager::GetInstance().GetMmapConfig();
        const auto saved_mmap = mmap.GetEnableGrowingMmap();
        mmap.growing_enable_mmap = true;
        auto restore_mmap =
            folly::makeGuard([&] { mmap.growing_enable_mmap = saved_mmap; });
        auto mapped = segcore::CreateGrowingSegment(schema, empty_index_meta);
        mapped->PreInsert(6);
        mapped->Insert(0,
                       6,
                       data.row_ids_.data(),
                       data.timestamps_.data(),
                       data.SharedRaw());
        auto* impl = dynamic_cast<segcore::SegmentGrowingImpl*>(mapped.get());
        ASSERT_TRUE(
            impl->get_insert_record().get_data<ArrayValue>(field)->is_mmap());
        check(*mapped);
    }
}

TEST(ComplexOperatorMigrationTest, MatchFloatingModuloKeepsArraySemantics) {
    for (auto type : {DataType::FLOAT, DataType::DOUBLE}) {
        auto schema = std::make_shared<Schema>();
        auto pk = schema->AddDebugField("pk", DataType::INT64);
        schema->set_primary_field_id(pk);
        auto field = schema->AddDebugArrayField("items[value]", type, true);
        auto data = segcore::DataGen(schema, 6);
        const std::vector<double> values{9.75, 7.5, 0, 5, 0, 9.75};
        for (auto& col : *data.raw_->mutable_fields_data()) {
            if (col.field_id() != field.get())
                continue;
            auto* scalar = col.mutable_scalars();
            scalar->clear_valid_data();
            auto* arrays = scalar->mutable_array_data();
            arrays->clear_data();
            for (size_t i = 0; i < values.size(); ++i) {
                scalar->add_valid_data(i != 4);
                auto* row = arrays->add_data();
                if (type == DataType::FLOAT)
                    row->mutable_float_data()->add_data(values[i]);
                else
                    row->mutable_double_data()->add_data(values[i]);
            }
        }
        auto growing = segcore::CreateGrowingSegment(schema, empty_index_meta);
        growing->PreInsert(6);
        growing->Insert(0,
                        6,
                        data.row_ids_.data(),
                        data.timestamps_.data(),
                        data.SharedRaw());
        auto sealed = CreateSealedWithFieldDataLoaded(schema, data);
        expr::ColumnInfo column(field, DataType::ARRAY, type, {}, true);
        column.element_level_ = true;
        proto::plan::GenericValue operand, bound;
        operand.set_float_val(2.5);
        bound.set_float_val(2.25);
        auto predicate = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
            column, proto::plan::Equal, proto::plan::Mod, bound, operand);
        auto root = std::make_shared<expr::MatchExpr>(
            "items", proto::plan::MatchAny, 1, predicate);
        CheckExecution(*growing, root, 6, {1, 0, 0, 0, -1, 1});
        CheckExecution(*sealed, root, 6, {1, 0, 0, 0, -1, 1});
        for (auto* segment :
             {static_cast<segcore::SegmentInternalInterface*>(growing.get()),
              static_cast<segcore::SegmentInternalInterface*>(sealed.get())}) {
            auto config = std::make_shared<QueryConfig>(
                std::unordered_map<std::string, std::string>{
                    {QueryConfig::kEnableExprExecutionV2, "true"}});
            QueryContext query("element-modulo",
                               segment,
                               6,
                               MAX_TIMESTAMP,
                               0,
                               0,
                               query::PlanOptions{},
                               config);
            ExecContext context(&query);
            auto filter =
                CandidateFilter::TryCreate(predicate, &context, "items");
            ASSERT_TRUE(filter);
            auto last = segment->GetStructElementOffsets(field)
                            ->ElementIDRangeOfRow(5)
                            .first;
            auto matches = filter->Eval(FixedVector<int32_t>{last, 0, 1, last});
            EXPECT_TRUE(matches[0]);
            EXPECT_TRUE(matches[1]);
            EXPECT_FALSE(matches[2]);
            EXPECT_TRUE(matches[3]);
        }
    }
}

TEST(ComplexOperatorMigrationTest, OrdinaryVectorNullPredicates) {
    for (auto type : {DataType::VECTOR_FLOAT,
                      DataType::VECTOR_BINARY,
                      DataType::VECTOR_FLOAT16,
                      DataType::VECTOR_BFLOAT16,
                      DataType::VECTOR_INT8,
                      DataType::VECTOR_SPARSE_U32_F32}) {
        for (bool nullable : {false, true}) {
            auto schema = std::make_shared<Schema>();
            auto pk = schema->AddDebugField("pk", DataType::INT64);
            schema->set_primary_field_id(pk);
            auto field = schema->AddDebugField(
                "vector", type, 16, knowhere::metric::L2, nullable);
            auto data = segcore::DataGen(schema, 6);
            auto growing =
                segcore::CreateGrowingSegment(schema, empty_index_meta);
            growing->PreInsert(6);
            growing->Insert(0,
                            6,
                            data.row_ids_.data(),
                            data.timestamps_.data(),
                            data.SharedRaw());
            auto sealed = CreateSealedWithFieldDataLoaded(schema, data);
            for (auto op : {proto::plan::NullExpr_NullOp_IsNull,
                            proto::plan::NullExpr_NullOp_IsNotNull}) {
                auto root = std::make_shared<expr::NullExpr>(
                    expr::ColumnInfo(field, type, {}, nullable), op);
                CheckExecution(*growing, root, 6);
                CheckExecution(*sealed, root, 6);
            }
        }
    }
}

// Real Knowhere IdMap, including compact vector IDs and logical NULL rows.
BuiltTestIndex
VectorIndexWithValidity(const std::vector<bool>& valid, bool include_bitmap) {
    FixedVector<bool> bits(valid.size());
    std::vector<float> data;
    for (size_t i = 0; i < valid.size(); ++i) {
        bits[i] = valid[i];
        if (!include_bitmap || valid[i])
            data.insert(data.end(), 16, static_cast<float>(i));
    }
    Config params{{"field_type", int(DataType::VECTOR_FLOAT)},
                  {"value_type", int(DataType::VECTOR_FLOAT)},
                  {index::METRIC_TYPE, knowhere::metric::L2},
                  {DIM_KEY, 16},
                  {INDEX_NUM_ROWS_KEY, valid.size()}};
    auto adapted = index::AdaptIndexType(
        {.index_type = knowhere::IndexEnum::INDEX_FAISS_IDMAP,
         .field_type = DataType::VECTOR_FLOAT,
         .element_type = DataType::NONE,
         .index_engine_version =
             knowhere::Version::GetCurrentVersion().VersionNumber(),
         .params = std::move(params)});
    params = std::move(adapted.params);
    index::VectorBuildInput<float> input{
        .physical_values = data,
        .logical_rows = int64_t(valid.size()),
        .physical_rows = int64_t(data.size() / 16),
        .dim = 16,
        .parent_validity = include_bitmap
                               ? ValidityView::FromExpanded(bits.data())
                               : ValidityView{}};
    index::VectorMemBuilder<float> builder(
        DataType::NONE,
        knowhere::IndexEnum::INDEX_FAISS_IDMAP,
        knowhere::metric::L2,
        knowhere::Version::GetCurrentVersion().VersionNumber(),
        16,
        params,
        true);
    return test_index_detail::PersistAndLoad(
        std::move(builder).Build(input), index::families::kVectorMem, params);
}

class VectorValiditySnapshot final : public segcore::SegmentReadSnapshot {
 public:
    explicit VectorValiditySnapshot(
        std::shared_ptr<const segcore::SegmentReadSnapshot> base)
        : base_(std::move(base)) {
    }
    int64_t
    chunk_size(FieldId f,
               int64_t c,
               OpContext* op_ctx = nullptr) const override {
        return base_->chunk_size(f, c, op_ctx);
    }
    int64_t
    num_rows_until_chunk(FieldId f,
                         int64_t c,
                         OpContext* op_ctx = nullptr) const override {
        return base_->num_rows_until_chunk(f, c, op_ctx);
    }
    std::pair<int64_t, int64_t>
    get_chunk_by_offset(FieldId f,
                        int64_t o,
                        OpContext* op_ctx = nullptr) const override {
        return base_->get_chunk_by_offset(f, o, op_ctx);
    }
    int64_t
    num_chunk_data(FieldId f, OpContext* op_ctx = nullptr) const override {
        return base_->num_chunk_data(f, op_ctx);
    }
    int64_t
    get_row_count() const override {
        return base_->get_row_count();
    }
    index::IndexPtr
    GetVectorIndexResource(FieldId f,
                           OpContext* op_ctx = nullptr) const override {
        return index_override ? index_override
                              : base_->GetVectorIndexResource(f, op_ctx);
    }
    std::pair<std::shared_ptr<ChunkedColumnInterface>, FieldSkipMetricsView>
    GetDataScanResources(FieldId f) const override {
        ++raw_lookups;
        if (fail_raw)
            ThrowInfo(FieldNotLoaded, "unexpected raw vector validity read");
        return base_->GetDataScanResources(f);
    }
    index::IndexPtr index_override;
    bool fail_raw = false;
    mutable int raw_lookups = 0;

 private:
    std::shared_ptr<const segcore::SegmentReadSnapshot> base_;
};

TEST(ComplexOperatorMigrationTest,
     VectorNullPrefersIndexAndFallsBackOnlyWithoutValidity) {
    for (bool bitmap : {true, false}) {
        for (bool all_null : {false, true}) {
            const std::vector<bool> valid =
                all_null
                    ? std::vector<bool>(6, false)
                    : std::vector<bool>{true, false, true, false, false, true};
            auto schema = std::make_shared<Schema>();
            auto pk = schema->AddDebugField("pk", DataType::INT64);
            schema->set_primary_field_id(pk);
            auto field = schema->AddDebugField("vector",
                                               DataType::VECTOR_FLOAT,
                                               16,
                                               knowhere::metric::L2,
                                               true);
            auto data = segcore::DataGen(schema, 6);
            for (auto& column : *data.raw_->mutable_fields_data()) {
                if (column.field_id() != field.get())
                    continue;
                column.clear_valid_data();
                auto* vectors = column.mutable_vectors();
                vectors->clear_valid_data();
                vectors->mutable_float_vector()->clear_data();
                for (size_t i = 0; i < valid.size(); ++i) {
                    vectors->add_valid_data(valid[i]);
                    if (valid[i])
                        for (int dim = 0; dim < 16; ++dim)
                            vectors->mutable_float_vector()->add_data(i);
                }
            }
            auto sealed = CreateSealedWithFieldDataLoaded(schema, data);
            auto index = VectorIndexWithValidity(valid, bitmap);
            segcore::LoadIndexInfo info;
            info.field_id = field.get();
            info.field_type = DataType::VECTOR_FLOAT;
            info.index_params =
                GenIndexParams(index.get(), index::families::kVectorMem);
            SetTestIndexMetadata(info, *index, index::families::kVectorMem);
            info.index = CreateTestIndex("scalar-vector-validity-" +
                                             std::to_string(bitmap) +
                                             std::to_string(all_null),
                                         std::move(index));
            sealed->LoadIndex(info);
            auto snapshot = std::make_shared<VectorValiditySnapshot>(
                sealed->CaptureReadSnapshot());
            snapshot->fail_raw = bitmap;
            auto root = std::make_shared<expr::NullExpr>(
                expr::ColumnInfo(field, DataType::VECTOR_FLOAT, {}, true),
                proto::plan::NullExpr_NullOp_IsNull);
            auto context = std::make_shared<segcore::reader::ReadContext>();
            auto execution = FilterBitsExecution::TryCreate(
                root, nullptr, snapshot, 6, 2, context);
            ASSERT_TRUE(execution);
            EXPECT_EQ(snapshot->raw_lookups, 0);
            const std::vector<int32_t> offsets{5, 1, 0, 5, 4, 2};
            auto batch = execution->prepared().MakeExecutionBatch(
                ReadBatchBinding::Offsets(offsets));
            batch->Eval(SelectionView::None(6));
            EXPECT_EQ(snapshot->raw_lookups, 0);
            auto rows = OwnedSelection::FromIndices(6, {0, 1, 3});
            auto partial = batch->Eval(rows.view())[0];
            for (size_t i : {0, 1, 3})
                EXPECT_EQ(partial.ValueAt<bool>(i), !valid[offsets[i]]);
            EXPECT_THROW(partial.IsValid(2), SegcoreError);
            auto full = batch->Eval(SelectionView::All(6))[0];
            for (size_t i = 0; i < offsets.size(); ++i)
                EXPECT_EQ(full.ValueAt<bool>(i), !valid[offsets[i]]);
            EXPECT_EQ(snapshot->raw_lookups == 0, bitmap);
            for (auto op : {proto::plan::NullExpr_NullOp_IsNull,
                            proto::plan::NullExpr_NullOp_IsNotNull}) {
                std::vector<int> truth;
                for (bool value : valid)
                    truth.push_back(op == proto::plan::NullExpr_NullOp_IsNull
                                        ? !value
                                        : value);
                CheckExecution(*sealed,
                               std::make_shared<expr::NullExpr>(
                                   expr::ColumnInfo(
                                       field, DataType::VECTOR_FLOAT, {}, true),
                                   op),
                               6,
                               truth);
            }
            sealed->DropFieldData(field);
            if (bitmap) {
                std::vector<int> truth;
                for (bool value : valid) truth.push_back(!value);
                CheckExecution(*sealed, root, 6, truth);
            } else {
                auto missing = FilterBitsExecution::TryCreate(
                    root,
                    nullptr,
                    sealed->CaptureReadSnapshot(),
                    6,
                    2,
                    context);
                ASSERT_TRUE(missing);
                EXPECT_THROW(missing->GetOutput(), SegcoreError);
            }
            // Existing readers retain the original generation, including the
            // raw fallback, after data is dropped and the live segment released.
            snapshot.reset();
            info.index.reset();
            sealed.reset();
            auto retained = execution->prepared().MakeExecutionBatch(
                ReadBatchBinding::Range(0, 6));
            auto result = retained->Eval(SelectionView::All(6))[0];
            for (size_t i = 0; i < valid.size(); ++i)
                EXPECT_EQ(result.ValueAt<bool>(i), !valid[i]);
        }
    }
}

TEST(ComplexOperatorMigrationTest, VectorValidityPinFailureDoesNotReadRaw) {
    auto schema = std::make_shared<Schema>();
    auto pk = schema->AddDebugField("pk", DataType::INT64);
    schema->set_primary_field_id(pk);
    auto field = schema->AddDebugField(
        "vector", DataType::VECTOR_FLOAT, 16, knowhere::metric::L2, true);
    auto sealed =
        CreateSealedWithFieldDataLoaded(schema, segcore::DataGen(schema, 6));
    auto snapshot =
        std::make_shared<VectorValiditySnapshot>(sealed->CaptureReadSnapshot());
    snapshot->fail_raw = true;
    auto calls = std::make_shared<int>(0);
    auto owner =
        VectorIndexWithValidity({true, false, true, false, true, false}, true)
            .index;
    snapshot->index_override =
        std::make_shared<segcore::reader::TestReaderIndex>(
            owner->Metadata(), [owner, calls](OpContext* ctx) {
                if (++*calls == 1)
                    ThrowInfo(FileReadFailed,
                              "injected vector index read failure");
                return owner->CreateReader(ctx);
            });
    auto root = std::make_shared<expr::NullExpr>(
        expr::ColumnInfo(field, DataType::VECTOR_FLOAT, {}, true),
        proto::plan::NullExpr_NullOp_IsNull);
    auto context = std::make_shared<segcore::reader::ReadContext>();
    bool cancel = false;
    context->checkpoint = [&] {
        if (cancel)
            ThrowInfo(FollyCancel, "cancel vector validity");
    };
    auto execution =
        FilterBitsExecution::TryCreate(root, nullptr, snapshot, 6, 2, context);
    ASSERT_TRUE(execution);
    auto batch = execution->prepared().MakeExecutionBatch(
        ReadBatchBinding::Offsets({1, 0, 1}));
    batch->Eval(SelectionView::None(3));
    EXPECT_EQ(*calls, 0);
    cancel = true;
    try {
        batch->Eval(SelectionView::All(3));
        FAIL();
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), FollyCancel);
    }
    EXPECT_EQ(*calls, 0);
    cancel = false;
    try {
        batch->Eval(SelectionView::All(3));
        FAIL();
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), FileReadFailed);
    }
    EXPECT_EQ(snapshot->raw_lookups, 0);
    auto result = batch->Eval(SelectionView::All(3))[0];
    EXPECT_TRUE(result.ValueAt<bool>(0));
    EXPECT_FALSE(result.ValueAt<bool>(1));
    EXPECT_TRUE(result.ValueAt<bool>(2));
    EXPECT_EQ(snapshot->raw_lookups, 0);
    EXPECT_EQ(*calls, 2);

    // A malformed bitmap is an error, not a reason to use raw data.
    snapshot->index_override =
        CreateTestIndex("scalar-vector-short-validity",
                        VectorIndexWithValidity({true, false}, true));
    auto short_index =
        FilterBitsExecution::TryCreate(root, nullptr, snapshot, 6, 2, context);
    ASSERT_TRUE(short_index);
    EXPECT_THROW(short_index->GetOutput(), SegcoreError);
    EXPECT_EQ(snapshot->raw_lookups, 0);
}

TEST(ComplexOperatorMigrationTest, GrowingVectorValidityRetainsIndexPrefix) {
    auto schema = std::make_shared<Schema>();
    auto pk = schema->AddDebugField("pk", DataType::INT64);
    schema->set_primary_field_id(pk);
    auto field = schema->AddDebugField(
        "vector", DataType::VECTOR_FLOAT, 16, knowhere::metric::L2, true);
    FieldIndexMeta field_meta(
        field,
        {{"index_type", knowhere::IndexEnum::INDEX_FAISS_IVFFLAT},
         {"metric_type", knowhere::metric::L2},
         {"nlist", "1"}},
        {{"dim", "16"}});
    auto index_meta = std::make_shared<CollectionIndexMeta>(
        80, std::map<FieldId, FieldIndexMeta>{{field, field_meta}});
    segcore::SegcoreConfig config;
    config.set_chunk_rows(16);
    config.set_nlist(1);
    config.set_nprobe(1);
    config.set_build_ratio(0.1F);
    config.set_enable_interim_segment_index(true);
    config.set_dense_vector_intermin_index_type(
        knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC);
    auto segment = segcore::CreateGrowingSegment(schema, index_meta, 1, config);
    // IVF requires at least 39 valid training rows per cluster. Keep one NULL
    // logical row while building the index before capturing the reader.
    auto first =
        segcore::DataGen(schema, 40, 42, 0, 1, 10, 1, false, true, false, 1);
    segment->PreInsert(80);
    segment->Insert(0,
                    40,
                    first.row_ids_.data(),
                    first.timestamps_.data(),
                    first.SharedRaw());
    auto* growing = dynamic_cast<segcore::SegmentGrowingImpl*>(segment.get());
    ASSERT_GE(growing->PinGrowingIndex(nullptr, field).CoveredRowEnd(), 40);
    auto binder = access::MakeSegmentReaderBinding(*segment, nullptr, 40);
    InputDependency dependency{"", field.get(), DataType::VECTOR_FLOAT};
    dependency.needs_values = false;
    dependency.needs_validity = true;
    dependency.nullable = true;
    auto reader = binder(dependency);
    ASSERT_TRUE(reader);
    binder = {};
    std::weak_ptr<segcore::ThreadSafeValidData> raw =
        growing->get_insert_record().get_valid_data(field);
    auto second =
        segcore::DataGen(schema, 40, 43, 40, 1, 10, 1, false, true, false, 20);
    segment->Insert(40,
                    40,
                    second.row_ids_.data(),
                    second.timestamps_.data(),
                    second.SharedRaw());
    segment.reset();
    EXPECT_TRUE(raw.expired());
    auto result =
        reader->Take(segcore::reader::TakeRequest(
                         {39, 0, 20, 0},
                         segcore::reader::ReadProjection::ValidityOnly,
                         SelectionView::All(4)),
                     {});
    EXPECT_FALSE(result.values);
    EXPECT_TRUE(result.valid.view().IsSelected(0));
    EXPECT_FALSE(result.valid.view().IsSelected(1));
    EXPECT_TRUE(result.valid.view().IsSelected(2));
    EXPECT_FALSE(result.valid.view().IsSelected(3));
    EXPECT_THROW(reader->Take(segcore::reader::TakeRequest(
                                  {40},
                                  segcore::reader::ReadProjection::ValidityOnly,
                                  SelectionView::All(1)),
                              {}),
                 SegcoreError);
}

TEST(ComplexOperatorMigrationTest, FloatingDivisionAgreesAcrossAccessModes) {
    for (auto type : {DataType::FLOAT, DataType::DOUBLE}) {
        auto schema = std::make_shared<Schema>();
        auto pk = schema->AddDebugField("pk", DataType::INT64);
        schema->set_primary_field_id(pk);
        auto field = schema->AddDebugField("number", type, true);
        auto data = segcore::DataGen(schema, 6);
        const std::vector<double> values{1.0, -1.0, 0.0, 49.0, 0.0, 2.0};
        for (auto& col : *data.raw_->mutable_fields_data()) {
            if (col.field_id() != field.get())
                continue;
            col.clear_valid_data();
            auto* scalar = col.mutable_scalars();
            scalar->clear_valid_data();
            scalar->mutable_float_data()->clear_data();
            scalar->mutable_double_data()->clear_data();
            for (size_t i = 0; i < values.size(); ++i) {
                scalar->add_valid_data(i != 4);
                if (type == DataType::FLOAT)
                    scalar->mutable_float_data()->add_data(values[i]);
                else
                    scalar->mutable_double_data()->add_data(values[i]);
            }
        }
        auto growing = segcore::CreateGrowingSegment(schema, empty_index_meta);
        growing->PreInsert(6);
        growing->Insert(0,
                        6,
                        data.row_ids_.data(),
                        data.timestamps_.data(),
                        data.SharedRaw());
        auto sealed = CreateSealedWithFieldDataLoaded(schema, data);
        for (double divisor : {49.0, -49.0}) {
            const double bound =
                type == DataType::FLOAT
                    ? static_cast<double>(1.0f / static_cast<float>(divisor))
                    : 1.0 / divisor;
            proto::plan::GenericValue operand, limit;
            operand.set_float_val(divisor);
            limit.set_float_val(bound);
            for (auto op : {proto::plan::Equal,
                            proto::plan::NotEqual,
                            proto::plan::LessThan,
                            proto::plan::LessEqual,
                            proto::plan::GreaterThan,
                            proto::plan::GreaterEqual}) {
                auto root = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
                    expr::ColumnInfo(field, type, {}, true),
                    op,
                    proto::plan::Div,
                    limit,
                    operand);
                CheckExecution(*growing, root, 6);
                CheckExecution(*sealed, root, 6);
                // The boundary equality is true for a rounded quotient, even
                // though multiplying it by 49 need not reproduce the input.
                if (op == proto::plan::Equal)
                    CheckExecution(*sealed, root, 6, {1, 0, 0, 0, -1, 0});
            }
        }
    }
}

TEST(ComplexOperatorMigrationTest, FloatingIntegerOperationsAcrossAccessModes) {
    for (auto type : {DataType::FLOAT, DataType::DOUBLE}) {
        auto schema = std::make_shared<Schema>();
        auto pk = schema->AddDebugField("pk", DataType::INT64);
        schema->set_primary_field_id(pk);
        auto field = schema->AddDebugField("number", type, true);
        for (auto operation : {proto::plan::Mod,
                               proto::plan::BitAnd,
                               proto::plan::BitOr,
                               proto::plan::BitXor,
                               proto::plan::Shl,
                               proto::plan::Shr}) {
            SCOPED_TRACE(::testing::Message()
                         << static_cast<int>(type) << ", op=" << operation);
            auto data = segcore::DataGen(schema, 6);
            // Avoid signed left-shift UB in the legacy oracle.
            const std::vector<double> values{
                9.75,
                operation == proto::plan::Shl ? 7.5 : -7.5,
                3.25,
                0.75,
                0.0,
                17.5};
            for (auto& col : *data.raw_->mutable_fields_data()) {
                if (col.field_id() != field.get())
                    continue;
                col.clear_valid_data();
                auto* scalar = col.mutable_scalars();
                scalar->clear_valid_data();
                if (type == DataType::FLOAT)
                    scalar->mutable_float_data()->clear_data();
                else
                    scalar->mutable_double_data()->clear_data();
                for (size_t i = 0; i < values.size(); ++i) {
                    scalar->add_valid_data(i != 4);
                    if (type == DataType::FLOAT)
                        scalar->mutable_float_data()->add_data(values[i]);
                    else
                        scalar->mutable_double_data()->add_data(values[i]);
                }
            }
            auto growing =
                segcore::CreateGrowingSegment(schema, empty_index_meta);
            growing->PreInsert(6);
            growing->Insert(0,
                            6,
                            data.row_ids_.data(),
                            data.timestamps_.data(),
                            data.SharedRaw());
            auto sealed = CreateSealedWithFieldDataLoaded(schema, data);
            proto::plan::GenericValue operand, limit;
            operand.set_float_val(2.5);
            for (double boundary : {1.0, 1.5}) {
                limit.set_float_val(boundary);
                for (auto comparison : {proto::plan::Equal,
                                        proto::plan::NotEqual,
                                        proto::plan::LessThan,
                                        proto::plan::LessEqual,
                                        proto::plan::GreaterThan,
                                        proto::plan::GreaterEqual}) {
                    auto root =
                        std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
                            expr::ColumnInfo(field, type, {}, true),
                            comparison,
                            operation,
                            limit,
                            operand);
                    CheckExecution(*growing, root, 6);
                    CheckExecution(*sealed, root, 6);
                }
            }
            if (operation == proto::plan::Mod) {
                limit.set_float_val(1.0);
                auto root = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
                    expr::ColumnInfo(field, type, {}, true),
                    proto::plan::Equal,
                    operation,
                    limit,
                    operand);
                // Integer remainder: 9 % 2 == 1; fmod(9.75, 2.5) != 1.
                CheckExecution(*growing, root, 6, {1, 0, 1, 0, -1, 1});
                CheckExecution(*sealed, root, 6, {1, 0, 1, 0, -1, 1});
            }
        }
    }
}

TEST(ComplexOperatorMigrationTest, FloatingIntegerErrorsRespectSelection) {
    const auto check = []<typename T>(DataType type) {
        const auto constant = [type](T value) {
            return std::make_shared<ConstantVector<T>>(type, 1, value);
        };
        auto column = std::make_shared<ColumnVector>(type, 7);
        column->SetValueAt<T>(0, T{9.75});
        column->SetValueAt<T>(1, std::numeric_limits<T>::infinity());
        column->SetValueAt<T>(2, std::numeric_limits<T>::quiet_NaN());
        column->SetValueAt<T>(3, T{0x1p63});
        column->SetValueAt<T>(4, -std::numeric_limits<T>::infinity());
        column->SetValueAt<T>(5, T{-0x1p63});
        column->nullAt(6);
        auto schema = std::make_shared<RowType>(
            std::vector<std::string>{"number"}, std::vector<DataType>{type});
        RowVector row(schema, 7, {column});
        BatchInput input(row);
        scalar::EvalCtx ctx(input);
        auto ref = std::make_shared<scalar::InputRefExpr>(0, type);
        const auto make = [&](proto::plan::ArithOpType operation, T operand) {
            return scalar::MakeArithmeticCompare(ref,
                                                 operation,
                                                 scalar::CompareOp::Equal,
                                                 constant(operand),
                                                 constant(T{1}),
                                                 true);
        };
        const auto expect_invalid = [&](const scalar::ExprPtr& expr,
                                        SelectionView rows) {
            try {
                expr->Eval(rows, ctx);
                FAIL() << "expected ExprInvalid";
            } catch (const SegcoreError& error) {
                EXPECT_EQ(error.get_error_code(), ExprInvalid);
            }
        };
        auto good = make(proto::plan::Mod, T{2.5});
        auto selected = OwnedSelection::FromIndices(7, {0, 6});
        auto result = good->Eval(selected.view(), ctx);
        EXPECT_TRUE(result.template ValueAt<bool>(0));
        EXPECT_FALSE(result.IsValid(6));
        for (size_t index : {1, 2, 3, 4}) {
            auto invalid_row = OwnedSelection::FromIndices(7, {index});
            expect_invalid(good, invalid_row.view());
        }
        // The lower bound is representable; only its division by -1 overflows.
        auto minimum_row = OwnedSelection::FromIndices(7, {5});
        EXPECT_NO_THROW(good->Eval(minimum_row.view(), ctx));
        expect_invalid(make(proto::plan::Mod, T{-1}), minimum_row.view());
        auto first_row = OwnedSelection::FromIndices(7, {0});
        for (auto bad :
             {make(proto::plan::Mod, T{0.5}),
              make(proto::plan::Shl, T{-1}),
              make(proto::plan::Shr, T{64}),
              make(proto::plan::BitAnd, T{0x1p63}),
              make(proto::plan::BitOr, std::numeric_limits<T>::infinity())}) {
            EXPECT_NO_THROW(bad->Eval(SelectionView::None(7), ctx));
            expect_invalid(bad, first_row.view());
            auto decisive =
                scalar::MakeLiteral(std::make_shared<ConstantVector<bool>>(
                    DataType::BOOL, 1, false));
            auto conjunction =
                scalar::MakeLogical(scalar::LogicalOp::And, {decisive, bad});
            auto short_circuited =
                conjunction->Eval(SelectionView::All(7), ctx);
            EXPECT_FALSE(short_circuited.template ValueAt<bool>(0));
        }
        scalar::EvalCtx cancelled(input, [] {
            ThrowInfo(FollyCancel, "cancel floating arithmetic");
        });
        try {
            good->Eval(selected.view(), cancelled);
            FAIL();
        } catch (const SegcoreError& error) {
            EXPECT_EQ(error.get_error_code(), FollyCancel);
        }
    };
    check.template operator()<float>(DataType::FLOAT);
    check.template operator()<double>(DataType::DOUBLE);
}

TEST(ComplexOperatorMigrationTest, JsonBitwiseGrowingAndSealed) {
    auto schema = std::make_shared<Schema>();
    auto pk = schema->AddDebugField("pk", DataType::INT64);
    schema->set_primary_field_id(pk);
    auto field = schema->AddDebugField("json", DataType::JSON, true);
    auto data = segcore::DataGen(schema, 6);
    std::vector<std::string> json{R"({"x":9})",
                                  R"({"x":7.5})",
                                  R"({"x":-3})",
                                  R"({"x":0})",
                                  R"({"x":"bad"})",
                                  "{}"};
    for (auto& col : *data.raw_->mutable_fields_data()) {
        if (col.field_id() != field.get())
            continue;
        col.clear_valid_data();
        col.mutable_scalars()->clear_valid_data();
        col.mutable_scalars()->mutable_json_data()->clear_data();
        for (size_t i = 0; i < json.size(); ++i) {
            col.mutable_scalars()->mutable_json_data()->add_data(json[i]);
            col.mutable_scalars()->add_valid_data(i != 3);
        }
    }
    auto growing = segcore::CreateGrowingSegment(schema, empty_index_meta);
    growing->PreInsert(6);
    growing->Insert(
        0, 6, data.row_ids_.data(), data.timestamps_.data(), data.SharedRaw());
    auto sealed = CreateSealedWithFieldDataLoaded(schema, data);
    for (int op = proto::plan::BitAnd; op <= proto::plan::Shr; ++op) {
        for (bool floating : {false, true}) {
            int64_t bounds[] = {1, 11, 10, 18, 4};
            auto bound = Int(bounds[op - proto::plan::BitAnd]);
            auto operand = Int(op >= proto::plan::Shl ? 1 : 3);
            if (floating) {
                auto b = bound.int64_val();
                auto v = operand.int64_val();
                bound.set_float_val(b);
                operand.set_float_val(v);
            }
            auto root = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
                expr::ColumnInfo(field, DataType::JSON, {"x"}, true),
                proto::plan::Equal,
                static_cast<proto::plan::ArithOpType>(op),
                bound,
                operand);
            std::vector<int> truth{
                1, 0, op == proto::plan::BitAnd ? 1 : 0, -1, -1, -1};
            CheckExecution(*growing, root, 6, truth);
            CheckExecution(*sealed, root, 6, truth);
        }
    }
}

TEST(ComplexOperatorMigrationTest,
     ElementCandidateAndFilterUseFixedElementDomain) {
    for (bool sealed_mode : {false, true}) {
        auto schema = std::make_shared<Schema>();
        auto pk = schema->AddDebugField("pk", DataType::INT64);
        schema->set_primary_field_id(pk);
        // The existing sealed ArrayChunk format has no element null bitmap.
        auto field = schema->AddDebugArrayField(
            "items[a]", DataType::INT64, true, !sealed_mode);
        auto data = segcore::DataGen(schema, 6);
        const std::vector<std::vector<int64_t>> arrays{
            {}, {1, 5}, {8, 2}, {4}, {}, {7}};
        for (auto& col : *data.raw_->mutable_fields_data()) {
            if (col.field_id() != field.get())
                continue;
            col.clear_valid_data();
            auto* scalars = col.mutable_scalars();
            scalars->clear_valid_data();
            scalars->mutable_array_data()->clear_data();
            for (size_t row = 0; row < arrays.size(); ++row) {
                scalars->add_valid_data(row != 4);
                auto* array = scalars->mutable_array_data()->add_data();
                array->mutable_long_data();
                for (auto value : arrays[row]) {
                    array->mutable_long_data()->add_data(value);
                    if (!sealed_mode)
                        array->add_valid_data(value != 8);
                }
            }
        }
        auto growing = segcore::CreateGrowingSegment(schema, empty_index_meta);
        if (!sealed_mode) {
            growing->PreInsert(6);
            growing->Insert(0,
                            6,
                            data.row_ids_.data(),
                            data.timestamps_.data(),
                            data.SharedRaw());
        }
        auto sealed = sealed_mode
                          ? CreateSealedWithFieldDataLoaded(schema, data)
                          : nullptr;
        auto* segment =
            sealed_mode
                ? static_cast<segcore::SegmentInternalInterface*>(sealed.get())
                : static_cast<segcore::SegmentInternalInterface*>(
                      growing.get());
        auto config = std::make_shared<QueryConfig>(
            std::unordered_map<std::string, std::string>{
                {QueryConfig::kEnableExprExecutionV2, "true"},
                {QueryConfig::kExprEvalBatchSize, "2"}});
        auto query = std::make_shared<QueryContext>("element-migration",
                                                    segment,
                                                    6,
                                                    MAX_TIMESTAMP,
                                                    0,
                                                    0,
                                                    query::PlanOptions{},
                                                    config);
        ExecContext context(query.get());
        expr::ColumnInfo element(
            field, DataType::ARRAY, DataType::INT64, {}, true);
        element.element_level_ = true;
        auto root = std::make_shared<expr::UnaryRangeFilterExpr>(
            element, proto::plan::GreaterThan, Int(3));
        std::shared_ptr<CandidateFilter> filter =
            CandidateFilter::TryCreate(root, &context, "items");
        ASSERT_TRUE(filter);
        if (!sealed_mode) {
            growing->PreInsert(6);
            growing->Insert(6,
                            6,
                            data.row_ids_.data(),
                            data.timestamps_.data(),
                            data.SharedRaw());
        }
        FixedVector<int32_t> offsets{5, 1, 0, 5, 2, 4};
        auto matches = filter->Eval(offsets);
        const bool expected[] = {true, true, false, true, sealed_mode, true};
        for (size_t i = 0; i < offsets.size(); ++i)
            EXPECT_EQ(matches[i], expected[i]);
        EXPECT_THROW(filter->Eval(FixedVector<int32_t>{6}), SegcoreError);
        EXPECT_EQ(filter->Eval(FixedVector<int32_t>{}).size(), 0);
        auto null_filter = CandidateFilter::TryCreate(
            std::make_shared<expr::NullExpr>(
                element, proto::plan::NullExpr_NullOp_IsNull),
            &context,
            "items");
        ASSERT_TRUE(null_filter);
        auto nulls = null_filter->Eval(offsets);
        for (size_t i = 0; i < offsets.size(); ++i)
            EXPECT_EQ(nulls[i], !sealed_mode && offsets[i] == 2);

        auto task = Task::Create(
            "element-test", ::milvus::plan::PlanFragment(), 0, query);
        DriverContext driver(task, 0, 0, 0, 0);
        PhyElementFilterBitsNode node(
            0,
            &driver,
            std::make_shared<::milvus::plan::ElementFilterBitsNode>(
                "element", root, "items"));
        TargetBitmap excluded(6, false);
        excluded[3] = true;
        RowVectorPtr input = std::make_shared<RowVector>(
            std::vector<VectorPtr>{std::make_shared<ColumnVector>(
                std::move(excluded), TargetBitmap(6, true))});
        auto original_input = input;
        node.AddInput(input);
        folly::CancellationSource cancel;
        OpContext cancelled_op(cancel.getToken());
        query->set_op_context(&cancelled_op);
        cancel.requestCancellation();
        EXPECT_THROW(node.GetOutput(), folly::FutureCancellation);
        EXPECT_FALSE(node.IsFinished());
        query->set_op_context(nullptr);
        auto output = node.GetOutput();
        ASSERT_TRUE(output);
        auto original_bits =
            ReadOnlyVectorView::CopyOf(original_input->child(0));
        for (int i = 0; i < 6; ++i)
            EXPECT_EQ(original_bits.ValueAt<bool>(i), i == 3);
        auto bits = ReadOnlyVectorView::CopyOf(output->child(0));
        ASSERT_EQ(bits.size(), 6);
        for (int i = 0; i < 6; ++i)
            EXPECT_EQ(bits.ValueAt<bool>(i),
                      !(i == 1 || i == 5 || (i == 2 && sealed_mode)));

        class Candidates final : public VectorIterator {
         public:
            bool
            HasNext() override {
                return position < 6;
            }
            std::optional<std::pair<int64_t, float>>
            Next() override {
                if (!HasNext())
                    return std::nullopt;
                const int64_t ids[] = {5, 1, 0, 5, 2, 4};
                auto p = position++;
                return std::make_pair(ids[p], float(p));
            }
            size_t position = 0;
        };
        ElementFilterIterator iterator(
            std::make_shared<Candidates>(), &context, nullptr, filter);
        filter.reset();
        for (int i = 0; i < 6; ++i) {
            if (!expected[i])
                continue;
            auto item = iterator.Next();
            ASSERT_TRUE(item);
            EXPECT_EQ(item->first, offsets[i]);
            EXPECT_EQ(item->second, float(i));
        }
        EXPECT_FALSE(iterator.HasNext());

        // Exercise the production operator's iterator handoff as well.
        SearchResult search;
        search.element_level_ = true;
        search.vector_iterators_ = std::vector<std::shared_ptr<VectorIterator>>{
            std::make_shared<Candidates>()};
        query->set_search_result(std::move(search));
        PhyIterativeElementFilterNode iterative(
            1,
            &driver,
            std::make_shared<::milvus::plan::IterativeElementFilterNode>(
                "iterative-element",
                root,
                "items",
                std::vector<::milvus::plan::PlanNodePtr>{},
                true));
        iterative.AddInput(output);
        iterative.NoMoreInput();
        ASSERT_TRUE(iterative.GetOutput());
        auto result = query->get_search_result();
        ASSERT_TRUE(result.vector_iterators_);
        ASSERT_EQ(result.vector_iterators_->size(), 1);
        auto& wrapped = result.vector_iterators_->front();
        for (int i = 0; i < 6; ++i) {
            if (!expected[i])
                continue;
            auto item = wrapped->Next();
            ASSERT_TRUE(item);
            EXPECT_EQ(item->first, offsets[i]);
            EXPECT_EQ(item->second, float(i));
        }
        EXPECT_FALSE(wrapped->HasNext());
    }
}

TEST(ComplexOperatorMigrationTest, MatchTwoColumnsAndFunctionChild) {
    auto schema = std::make_shared<Schema>();
    auto pk = schema->AddDebugField("pk", DataType::INT64);
    schema->set_primary_field_id(pk);
    auto left =
        schema->AddDebugArrayField("items[a]", DataType::INT64, true, true);
    auto right =
        schema->AddDebugArrayField("items[b]", DataType::INT64, true, true);
    auto data = segcore::DataGen(schema, 6);
    std::vector<std::vector<int64_t>> a{{}, {1, 5}, {8, 2}, {4}, {}, {7}};
    std::vector<std::vector<int64_t>> b{{}, {2, 3}, {6, 4}, {5}, {}, {1}};
    for (auto& col : *data.raw_->mutable_fields_data()) {
        if (col.field_id() != left.get() && col.field_id() != right.get())
            continue;
        auto& rows = col.field_id() == left.get() ? a : b;
        col.clear_valid_data();
        auto* scalars = col.mutable_scalars();
        scalars->clear_valid_data();
        scalars->mutable_array_data()->clear_data();
        for (size_t row = 0; row < rows.size(); ++row) {
            scalars->add_valid_data(row != 4);
            auto* arr = scalars->mutable_array_data()->add_data();
            arr->mutable_long_data();
            for (auto value : rows[row]) {
                arr->mutable_long_data()->add_data(value);
                arr->add_valid_data(
                    !(row == 2 && col.field_id() == left.get()));
            }
        }
    }
    auto growing = segcore::CreateGrowingSegment(schema, empty_index_meta);
    growing->PreInsert(6);
    growing->Insert(
        0, 6, data.row_ids_.data(), data.timestamps_.data(), data.SharedRaw());
    auto compare =
        std::make_shared<expr::CompareExpr>(left,
                                            right,
                                            DataType::ARRAY,
                                            DataType::ARRAY,
                                            proto::plan::GreaterThan);
    auto root = std::make_shared<expr::MatchExpr>(
        "items", proto::plan::MatchAny, 0, compare);
    CheckExecution(*growing, root, 6, {0, 1, 0, 0, -1, 1});
    expr::ColumnInfo element(left, DataType::ARRAY, DataType::INT64, {}, true);
    element.element_level_ = true;
    auto call = std::make_shared<expr::CallExpr>(
        "greater_than_three",
        std::vector<expr::TypedExprPtr>{
            std::make_shared<expr::ColumnExpr>(element)},
        +[](const RowVector& args, VectorPtr& result) {
            auto values = ReadOnlyVectorView::CopyOf(args.child(0));
            auto out =
                std::make_shared<ColumnVector>(DataType::BOOL, args.size());
            for (size_t i = 0; i < args.size(); ++i) {
                if (!values.IsValid(i))
                    out->nullAt(i);
                else
                    out->SetValueAt<bool>(i, values.ValueAt<int64_t>(i) > 3);
            }
            result = out;
        });
    auto function_root = std::make_shared<expr::MatchExpr>(
        "items", proto::plan::MatchAny, 0, call);
    CheckExecution(*growing, function_root, 6, {0, 1, 0, 1, -1, 1});
}

TEST(ComplexOperatorMigrationTest, TextPredicatesGrowingAndSealed) {
    auto schema = std::make_shared<Schema>();
    schema->AddField(FieldMeta(
        FieldName("pk"), FieldId(100), DataType::INT64, false, std::nullopt));
    schema->set_primary_field_id(FieldId(100));
    std::map<std::string, std::string> analyzer_parameters;
    schema->AddField(FieldMeta(FieldName("text"),
                               FieldId(101),
                               DataType::VARCHAR,
                               65536,
                               true,
                               true,
                               true,
                               analyzer_parameters,
                               std::nullopt));
    auto data = segcore::DataGen(schema, 6);
    const std::vector<std::string> texts{"red apple",
                                         "red green apple",
                                         "blue apple",
                                         "",
                                         "ignored",
                                         "red pear"};
    for (auto& col : *data.raw_->mutable_fields_data()) {
        if (col.field_id() != 101)
            continue;
        col.clear_valid_data();
        auto* scalar = col.mutable_scalars();
        scalar->clear_valid_data();
        scalar->mutable_string_data()->clear_data();
        for (size_t i = 0; i < texts.size(); ++i) {
            scalar->mutable_string_data()->add_data(texts[i]);
            scalar->add_valid_data(i != 4);
        }
    }
    auto growing = segcore::CreateGrowingSegment(schema, empty_index_meta);
    growing->PreInsert(6);
    growing->Insert(
        0, 6, data.row_ids_.data(), data.timestamps_.data(), data.SharedRaw());
    auto pin = growing->PinGrowingIndex(nullptr, FieldId(101));
    ASSERT_TRUE(pin);
    ASSERT_EQ(pin.CoveredRowEnd(), 6);
    auto sealed = CreateSealedWithFieldDataLoaded(schema, data);
    sealed->CreateTextIndex(FieldId(101));
    expr::ColumnInfo field(FieldId(101), DataType::VARCHAR, {}, true);
    for (auto op : {proto::plan::TextMatch,
                    proto::plan::PhraseMatch,
                    proto::plan::TextMatchFuzzy}) {
        for (int argument : {0, 1, 2}) {
            proto::plan::GenericValue text;
            text.set_string_val(
                op == proto::plan::TextMatchFuzzy ? "appl" : "red apple");
            auto root = std::make_shared<expr::UnaryRangeFilterExpr>(
                field, op, text, std::vector{Int(argument)});
            CheckExecution(*growing, root, 6);
            CheckExecution(*sealed, root, 6);
            auto negated = std::make_shared<expr::LogicalUnaryExpr>(
                expr::LogicalUnaryExpr::OpType::LogicalNot, root);
            CheckExecution(*growing, negated, 6);
            CheckExecution(*sealed, negated, 6);
        }
    }
    proto::plan::GenericValue text;
    text.set_string_val("red");
    auto root = std::make_shared<expr::UnaryRangeFilterExpr>(
        field, proto::plan::TextMatch, text);
    auto snapshot = sealed->CaptureReadSnapshot();
    auto context = std::make_shared<segcore::reader::ReadContext>();
    auto prepared =
        FilterBitsExecution::TryCreate(root, nullptr, snapshot, 6, 2, context);
    ASSERT_TRUE(prepared);
    EXPECT_EQ(prepared->prepared().granularity(),
              plan::ExecutionGranularity::WholeSegment);
    EXPECT_EQ(prepared->prepared().predicate_backends(),
              std::vector{plan::PredicateBackend::TextIndex});
    EXPECT_TRUE(prepared->prepared().plan()->inputs().slots().empty());
    // Missing mandatory text source is a capability miss, never a raw-column load.
    int raw_loads = 0;
    EXPECT_FALSE(FilterBitsExecution::TryCreateWithBinding(
        root,
        nullptr,
        [&](const InputDependency&)
            -> std::shared_ptr<const segcore::reader::ColumnReader> {
            ++raw_loads;
            return nullptr;
        },
        6,
        2,
        context));
    EXPECT_EQ(raw_loads, 0);
    auto invalid = std::make_shared<expr::UnaryRangeFilterExpr>(
        field, proto::plan::TextMatchFuzzy, text, std::vector{Int(3)});
    proto::plan::GenericValue no;
    no.set_bool_val(false);
    auto shorted = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::And,
        std::make_shared<expr::ValueExpr>(no),
        invalid);
    auto lazy = FilterBitsExecution::TryCreate(
        shorted, nullptr, snapshot, 6, 2, context);
    ASSERT_TRUE(lazy);
    EXPECT_NO_THROW(lazy->GetOutput());
    auto failing = FilterBitsExecution::TryCreate(
        invalid, nullptr, snapshot, 6, 2, context);
    ASSERT_TRUE(failing);
    try {
        failing->GetOutput();
        FAIL();
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), InvalidParameter);
    }

    auto config = std::make_shared<QueryConfig>(
        std::unordered_map<std::string, std::string>{
            {QueryConfig::kEnableExprExecutionV2, "true"}});
    QueryContext query_context("text-boost",
                               sealed.get(),
                               6,
                               MAX_TIMESTAMP,
                               0,
                               0,
                               query::PlanOptions{},
                               config);
    query_context.set_read_snapshot(snapshot);
    ExecContext execution_context(&query_context);
    auto scorer = std::make_shared<rescores::WeightScorer>(root, 2.0f);
    std::unique_ptr<ExprSet> legacy;
    std::unique_ptr<CandidateFilter> candidate;
    EXPECT_FALSE(rescores::ComputeNonNativeFilterBitset(
        &execution_context, scorer, &legacy, &candidate));
    ASSERT_TRUE(candidate);
    EXPECT_FALSE(legacy);
    for (int chunk = 0; chunk < 2; ++chunk) {
        FixedVector<int32_t> offsets{5, 2, 0, 4, 5, 3};
        std::vector<std::optional<float>> scores(offsets.size());
        rescores::ComputeScorerScores(&execution_context,
                                      nullptr,
                                      sealed.get(),
                                      scorer,
                                      offsets,
                                      scores,
                                      nullptr,
                                      nullptr,
                                      candidate.get());
        for (size_t i = 0; i < scores.size(); ++i) {
            EXPECT_EQ(scores[i].has_value(), i == 0 || i == 2 || i == 4);
            if (scores[i])
                EXPECT_EQ(*scores[i], 2.0f);
        }
    }
    candidate.reset();
    failing.reset();
    lazy.reset();
    query_context.set_read_snapshot(nullptr);
    // A prepared snapshot survives releasing the source segment and all caller snapshots.
    sealed.reset();
    snapshot.reset();
    ASSERT_TRUE(prepared->GetOutput());
}

TEST(ComplexOperatorMigrationTest, TextSourceIsLazyCachedAndPreservesFailures) {
    const std::string texts[] = {"apple", "", "pear"};
    const bool text_valid[] = {true, false, true};
    auto built = segcore::reader::BuildReaderIndex<std::string>(
        index::families::kText,
        3,
        texts,
        text_valid,
        {{"analyzer_name", "milvus_tokenizer"},
         {"analyzer_params", "{}"},
         {"tantivy_index_version", 7}});
    std::shared_ptr<const index::IIndexReaderBase> source =
        std::move(built.reader);
    int loads = 0;
    auto ctx = std::make_shared<segcore::reader::ReadContext>();
    segcore::reader::TextIndexQuery query{proto::plan::TextMatch, "apple", {}};
    auto reader = std::make_shared<segcore::reader::TextIndexReader>(
        [&](const auto&) {
            if (++loads == 1)
                ThrowInfo(FileReadFailed, "injected text pin failure");
            return source;
        },
        5,
        query);
    predicate::IndexNode node(reader, ctx);
    auto binding = ReadBatchBinding::Offsets({2, 0, 0, 1, 4});
    auto empty = node.Eval(binding, SelectionView::None(5));
    EXPECT_TRUE(empty.defined_rows().IsNone());
    EXPECT_EQ(loads, 0);
    try {
        node.Eval(binding, SelectionView::All(5));
        FAIL();
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), FileReadFailed);
    }
    auto result = node.Eval(binding, SelectionView::All(5));
    EXPECT_FALSE(result.ValueAt<bool>(0));
    EXPECT_TRUE(result.ValueAt<bool>(1));
    EXPECT_TRUE(result.ValueAt<bool>(2));
    EXPECT_FALSE(result.IsValid(3));
    EXPECT_FALSE(result.IsValid(4));
    node.Eval(ReadBatchBinding::Range(0, 2), SelectionView::All(2));
    EXPECT_EQ(loads, 2);
    segcore::reader::TextIndexReader shorter(
        [source](const auto&) { return source; }, 1, query);
    EXPECT_EQ(shorter.Query(*ctx).size(), 1);
    auto cancel = std::make_shared<segcore::reader::ReadContext>();
    cancel->checkpoint = [] { ThrowInfo(FollyCancel, "cancel text query"); };
    try {
        reader->Query(*cancel);
        FAIL();
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), FollyCancel);
    }
    for (auto op : {proto::plan::PhraseMatch, proto::plan::TextMatchFuzzy}) {
        segcore::reader::TextIndexReader invalid(
            [source](const auto&) { return source; },
            3,
            {op, "apple", {Int(-1)}});
        try {
            invalid.Query(*ctx);
            FAIL();
        } catch (const SegcoreError& error) {
            EXPECT_EQ(error.get_error_code(), InvalidParameter);
        }
    }
}

TEST(ComplexOperatorMigrationTest, VectorArrayLengthGrowingAndSealed) {
    for (auto element : {DataType::VECTOR_FLOAT,
                         DataType::VECTOR_FLOAT16,
                         DataType::VECTOR_BFLOAT16,
                         DataType::VECTOR_INT8,
                         DataType::VECTOR_BINARY}) {
        SCOPED_TRACE(static_cast<int>(element));
        auto schema = std::make_shared<Schema>();
        auto pk = schema->AddDebugField("pk", DataType::INT64);
        schema->set_primary_field_id(pk);
        auto field = schema->AddDebugVectorArrayField(
            "items[vectors]", element, 16, knowhere::metric::L2, true, true);
        auto data = segcore::DataGen(schema, 6);
        std::vector<int> counts{0, 2, 3, 1, 0, 2};
        for (auto& col : *data.raw_->mutable_fields_data()) {
            if (col.field_id() != field.get())
                continue;
            col.clear_valid_data();
            col.mutable_vectors()->clear_valid_data();
            auto* arrays = col.mutable_vectors()->mutable_vector_array();
            arrays->clear_data();
            for (size_t row = 0; row < counts.size(); ++row) {
                col.mutable_vectors()->add_valid_data(row != 4);
                auto* vector = arrays->add_data();
                vector->set_dim(16);
                int valid_count = 0;
                for (int i = 0; i < counts[row]; ++i) {
                    vector->add_valid_data(i != 1);
                    valid_count += i != 1;
                }
                size_t bytes =
                    valid_count * vector_bytes_per_element(element, 16);
                if (element == DataType::VECTOR_FLOAT)
                    for (int i = 0; i < valid_count * 16; ++i)
                        vector->mutable_float_vector()->add_data(1.0f);
                else if (element == DataType::VECTOR_FLOAT16)
                    vector->set_float16_vector(std::string(bytes, '\0'));
                else if (element == DataType::VECTOR_BFLOAT16)
                    vector->set_bfloat16_vector(std::string(bytes, '\0'));
                else if (element == DataType::VECTOR_INT8)
                    vector->set_int8_vector(std::string(bytes, '\0'));
                else
                    vector->set_binary_vector(std::string(bytes, '\0'));
                if (element == DataType::VECTOR_FLOAT && valid_count == 0)
                    vector->mutable_float_vector();
            }
        }
        auto growing = segcore::CreateGrowingSegment(schema, empty_index_meta);
        growing->PreInsert(6);
        growing->Insert(0,
                        6,
                        data.row_ids_.data(),
                        data.timestamps_.data(),
                        data.SharedRaw());
        // The current sealed chunk format has no per-element validity. Its
        // fixture uses full payloads with all-valid elements; growing below
        // separately verifies compact payloads with NULL elements.
        for (auto& col : *data.raw_->mutable_fields_data()) {
            if (col.field_id() != field.get())
                continue;
            auto* arrays = col.mutable_vectors()->mutable_vector_array();
            for (int row = 0; row < arrays->data_size(); ++row) {
                auto* vector = arrays->mutable_data(row);
                vector->Clear();
                vector->set_dim(16);
                for (int i = 0; i < counts[row]; ++i)
                    vector->add_valid_data(true);
                const auto bytes =
                    counts[row] * vector_bytes_per_element(element, 16);
                if (element == DataType::VECTOR_FLOAT) {
                    auto* values = vector->mutable_float_vector();
                    for (int i = 0; i < counts[row] * 16; ++i)
                        values->add_data(1.0f);
                } else if (element == DataType::VECTOR_FLOAT16)
                    vector->set_float16_vector(std::string(bytes, '\0'));
                else if (element == DataType::VECTOR_BFLOAT16)
                    vector->set_bfloat16_vector(std::string(bytes, '\0'));
                else if (element == DataType::VECTOR_INT8)
                    vector->set_int8_vector(std::string(bytes, '\0'));
                else
                    vector->set_binary_vector(std::string(bytes, '\0'));
            }
        }
        auto sealed = CreateSealedWithFieldDataLoaded(schema, data);
        expr::ColumnInfo column(
            field, DataType::VECTOR_ARRAY, element, {}, true);
        auto length = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
            column,
            proto::plan::GreaterEqual,
            proto::plan::ArrayLength,
            Int(2),
            Int(0));
        CheckExecution(*growing, length, 6, {0, 1, 1, 0, -1, 1});
        CheckExecution(*sealed, length, 6, {0, 1, 1, 0, -1, 1});
        CheckExecution(*growing, length, 6);
        CheckExecution(*sealed, length, 6);
        auto match = std::make_shared<expr::MatchExpr>(
            "items",
            proto::plan::MatchLeast,
            2,
            std::make_shared<expr::AlwaysTrueExpr>());
        CheckExecution(*growing, match, 6, {0, 1, 1, 0, -1, 1});
        CheckExecution(*sealed, match, 6, {0, 1, 1, 0, -1, 1});
        for (auto* segment :
             {static_cast<segcore::SegmentInternalInterface*>(growing.get()),
              static_cast<segcore::SegmentInternalInterface*>(sealed.get())}) {
            InputDependency dependency;
            dependency.field_id = field.get();
            dependency.type = DataType::VECTOR_ARRAY;
            dependency.element_type = element;
            dependency.needs_values = true;
            auto reader = access::MakeSegmentReaderBinding(
                *segment, segment->CaptureReadSnapshot(), 6)(dependency);
            ASSERT_TRUE(reader);
            auto loaded = reader->Take(
                segcore::reader::TakeRequest(
                    {2, 1, 2},
                    segcore::reader::ReadProjection::ValuesAndValidity,
                    SelectionView::All(3)),
                {});
            ASSERT_TRUE(loaded.values);
            auto detached = ReadOnlyVectorView::CopyOf(
                MakeWritableCopy(*loaded.values).values);
            loaded.values.reset();
            reader.reset();
            const auto& array = detached.VectorArrayAt(0);
            EXPECT_EQ(array.length(), 3);
            EXPECT_EQ(array.physical_length(),
                      segment->type() == SegmentType::Growing ? 2 : 3);
            EXPECT_EQ(array.dim(), 16);
            EXPECT_EQ(array.get_element_type(), element);
            auto payload = array.output_data();
            if (segment->type() == SegmentType::Growing) {
                ASSERT_EQ(payload.valid_data_size(), 3);
                EXPECT_TRUE(payload.valid_data(0));
                EXPECT_FALSE(payload.valid_data(1));
                EXPECT_TRUE(payload.valid_data(2));
            } else {
                EXPECT_EQ(payload.valid_data_size(), 0);
            }
        }
        auto is_null = std::make_shared<expr::NullExpr>(
            column, proto::plan::NullExpr_NullOp_IsNull);
        CheckExecution(*growing, is_null, 6, {0, 0, 0, 0, 1, 0});
        CheckExecution(*sealed, is_null, 6, {0, 0, 0, 0, 1, 0});

        // GetOwn must detach compact payloads and element validity, including
        // repeated offsets and NULL rows, before releasing the column pin.
        auto snapshot = sealed->CaptureReadSnapshot();
        auto source = snapshot->GetDataScanResources(field).first;
        std::vector<int64_t> offsets{2, 4, 1, 2};
        auto take =
            source->Take(nullptr,
                         {OffsetView::From(offsets.data(), offsets.size()),
                          ChunkedColumnInterface::TargetType::VectorArrayView,
                          {}});
        ASSERT_TRUE(take);
        auto owned = take->GetOwn();
        take.reset();
        source.reset();
        snapshot.reset();
        sealed.reset();
        auto* views = owned.values.data_as<VectorArrayView>();
        EXPECT_EQ(views[0].length(), 3);
        EXPECT_EQ(views[2].length(), 2);
        EXPECT_EQ(views[3].length(), 3);
        EXPECT_FALSE(owned.validity[1]);
        EXPECT_EQ(views[0].output_data().valid_data_size(), 0);

        InputDependency prefix_dependency;
        prefix_dependency.field_id = field.get();
        prefix_dependency.type = DataType::VECTOR_ARRAY;
        prefix_dependency.element_type = element;
        prefix_dependency.needs_values = true;
        auto prefix = access::MakeSegmentReaderBinding(
            *growing, nullptr, 5)(prefix_dependency);
        ASSERT_TRUE(prefix);
        auto begin = growing->PreInsert(6);
        growing->Insert(begin,
                        6,
                        data.row_ids_.data(),
                        data.timestamps_.data(),
                        data.SharedRaw());
        growing.reset();
        auto pinned = prefix->ReadRange(
            {0,
             5,
             segcore::reader::ReadProjection::ValuesAndValidity,
             SelectionView::All(5)},
            {});
        prefix.reset();
        ASSERT_TRUE(pinned.values);
        EXPECT_EQ(pinned.values->size(), 5);
        EXPECT_EQ(pinned.values->VectorArrayAt(2).length(), 3);
        EXPECT_FALSE(pinned.values->IsValid(4));
    }
}

TEST(ComplexOperatorMigrationTest,
     NullableArrayElementsNeverReadPlaceholderValues) {
    auto schema = std::make_shared<Schema>();
    auto pk = schema->AddDebugField("pk", DataType::INT64);
    schema->set_primary_field_id(pk);
    auto id = schema->AddDebugArrayField("values", DataType::INT64, true, true);
    auto data = segcore::DataGen(schema, 6);
    const std::vector<std::vector<int>> rows{
        {}, {1, 99}, {4, 99}, {3}, {}, {8, 2}};
    for (auto& col : *data.raw_->mutable_fields_data()) {
        if (col.field_id() != id.get())
            continue;
        col.clear_valid_data();
        col.mutable_scalars()->clear_valid_data();
        auto* arrays = col.mutable_scalars()->mutable_array_data();
        arrays->clear_data();
        arrays->set_element_type(proto::schema::Int64);
        for (size_t row = 0; row < rows.size(); ++row) {
            col.mutable_scalars()->add_valid_data(row != 4);
            auto* array = arrays->add_data();
            for (int value : rows[row]) {
                array->mutable_long_data()->add_data(value);
                array->add_valid_data(value != 99);
            }
        }
    }
    auto growing = segcore::CreateGrowingSegment(schema, empty_index_meta);
    growing->PreInsert(6);
    growing->Insert(
        0, 6, data.row_ids_.data(), data.timestamps_.data(), data.SharedRaw());
    expr::ColumnInfo field(id, DataType::ARRAY, DataType::INT64, {}, true);
    auto element = field;
    element.nested_path_ = {"1"};
    auto equal = std::make_shared<expr::UnaryRangeFilterExpr>(
        element, proto::plan::Equal, Int(99));
    CheckExecution(*growing, equal, 6, {-1, -1, -1, -1, -1, 0});
    auto contains = std::make_shared<expr::TermFilterExpr>(
        field, std::vector<proto::plan::GenericValue>{Int(99)}, true);
    CheckExecution(*growing, contains, 6, {0, 0, 0, 0, -1, 0});
    auto all = std::make_shared<expr::JsonContainsExpr>(
        field,
        proto::plan::JSONContainsExpr_JSONOp_ContainsAll,
        true,
        std::vector<proto::plan::GenericValue>{Int(1), Int(99)});
    CheckExecution(*growing, all, 6, {0, 0, 0, 0, -1, 0});
    auto length = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
        field,
        proto::plan::GreaterEqual,
        proto::plan::ArrayLength,
        Int(2),
        Int(0));
    CheckExecution(*growing, length, 6, {0, 1, 1, 0, -1, 1});
}

TEST(ComplexOperatorMigrationTest, JsonArithmeticErrorsStayLazyAndKeepCodes) {
    auto column = std::make_shared<ColumnVector>(DataType::JSON, 3);
    column->SetValueAt<Json>(0, Json(std::string_view(R"({"x":8})")));
    column->SetValueAt<Json>(1, Json(std::string_view(R"({"x":"bad"})")));
    column->nullAt(2);
    auto schema =
        std::make_shared<RowType>(std::vector<std::string>{"json"},
                                  std::vector<DataType>{DataType::JSON});
    RowVector row(schema, 3, {column});
    BatchInput input(row);
    scalar::EvalCtx ctx(input);
    auto ref = std::make_shared<scalar::InputRefExpr>(0, DataType::JSON);
    auto bad = scalar::MakeJsonArithmeticCompare(
        ref, {"x"}, proto::plan::Div, scalar::CompareOp::Equal, Int(0), Int(1));
    EXPECT_NO_THROW(bad->Eval(SelectionView::None(3), ctx));
    try {
        bad->Eval(SelectionView::All(3), ctx);
        FAIL();
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), ExprInvalid);
    }
    auto good = scalar::MakeJsonArithmeticCompare(
        ref, {"x"}, proto::plan::Div, scalar::CompareOp::Equal, Int(2), Int(4));
    auto result = good->Eval(SelectionView::All(3), ctx);
    EXPECT_TRUE(result.ValueAt<bool>(0));
    EXPECT_FALSE(result.IsValid(1));
    EXPECT_FALSE(result.IsValid(2));
    scalar::EvalCtx cancelled(
        input, [] { ThrowInfo(FollyCancel, "cancel JSON arithmetic"); });
    try {
        good->Eval(SelectionView::All(3), cancelled);
        FAIL();
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), FollyCancel);
    }
    for (auto shift : {proto::plan::Shl, proto::plan::Shr}) {
        auto bad_shift = scalar::MakeJsonArithmeticCompare(
            ref, {"x"}, shift, scalar::CompareOp::Equal, Int(64), Int(0));
        EXPECT_NO_THROW(bad_shift->Eval(SelectionView::None(3), ctx));
        try {
            bad_shift->Eval(SelectionView::All(3), ctx);
            FAIL();
        } catch (const SegcoreError& error) {
            EXPECT_EQ(error.get_error_code(), ExprInvalid);
        }
    }
    column->SetValueAt<Json>(0, Json(std::string_view(R"({"x":1e100})")));
    BatchInput oversized(row);
    scalar::EvalCtx oversized_context(oversized);
    auto bad_cast = scalar::MakeJsonArithmeticCompare(ref,
                                                      {"x"},
                                                      proto::plan::BitAnd,
                                                      scalar::CompareOp::Equal,
                                                      Int(1),
                                                      Int(0));
    try {
        bad_cast->Eval(SelectionView::All(3), oversized_context);
        FAIL();
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), ExprInvalid);
    }
}

TEST(ComplexOperatorMigrationTest, ArrayPredicatesShareIncrementalInput) {
    expr::ColumnInfo field(
        FieldId(101), DataType::ARRAY, DataType::INT64, {}, true);
    auto first = field;
    first.nested_path_ = {"0"};
    auto second = field;
    second.nested_path_ = {"1"};
    auto compiled = plan::CompileExpressions(
        {std::make_shared<expr::UnaryRangeFilterExpr>(
             first, proto::plan::Equal, Int(3)),
         std::make_shared<expr::UnaryRangeFilterExpr>(
             second, proto::plan::Equal, Int(7)),
         std::make_shared<expr::TermFilterExpr>(
             field, std::vector<proto::plan::GenericValue>{Int(7)}, true)});
    ASSERT_TRUE(compiled.supported());
    ASSERT_EQ(compiled.plan->inputs().slots().size(), 1);
    proto::schema::ScalarField values;
    values.mutable_long_data()->add_data(3);
    values.mutable_long_data()->add_data(7);
    auto column = std::make_shared<ColumnVector>(DataType::ARRAY, 3);
    column->SetValueAt<Array>(0, Array(values));
    column->nullAt(1);
    column->SetValueAt<Array>(2, Array(values));
    auto view = ReadOnlyVectorView::CopyOf(column);
    BatchInput input(compiled.plan->inputs().schema(), 3);
    std::vector<std::vector<size_t>> requests;
    input.BindLoader(0, [&](SelectionView rows) {
        requests.emplace_back();
        rows.ForEachSelected(
            [&](size_t row) { requests.back().push_back(row); });
        return view.Restrict(rows);
    });
    scalar::EvalCtx ctx(input);
    auto selected = OwnedSelection::FromIndices(3, {0, 1});
    auto partial = compiled.plan->expressions().Eval(selected.view(), ctx);
    auto full = compiled.plan->expressions().Eval(SelectionView::All(3), ctx);
    EXPECT_EQ(requests, (std::vector<std::vector<size_t>>{{0, 1}, {2}}));
    for (size_t root = 0; root < full.size(); ++root) {
        EXPECT_TRUE(partial[root].ValueAt<bool>(0));
        EXPECT_FALSE(partial[root].IsValid(1));
        EXPECT_FALSE(partial[root].defined_rows().IsSelected(2));
        EXPECT_TRUE(full[root].ValueAt<bool>(0));
        EXPECT_FALSE(full[root].IsValid(1));
        EXPECT_TRUE(full[root].ValueAt<bool>(2));
    }
}

TEST(ComplexOperatorMigrationTest, NarrowIntegerOutOfDomainComparisons) {
    for (auto type : {DataType::INT8, DataType::INT16, DataType::INT32}) {
        SCOPED_TRACE(static_cast<int>(type));
        const int bits = type == DataType::INT8    ? 8
                         : type == DataType::INT16 ? 16
                                                   : 32;
        const int64_t low = -(int64_t{1} << (bits - 1));
        const int64_t high = -low - 1;
        auto schema = std::make_shared<Schema>();
        auto pk = schema->AddDebugField("pk", DataType::INT64);
        schema->set_primary_field_id(pk);
        auto id = schema->AddDebugField("small", type, true);
        auto data = segcore::DataGen(schema, 6);
        for (auto& col : *data.raw_->mutable_fields_data()) {
            if (col.field_id() != id.get())
                continue;
            col.clear_valid_data();
            col.mutable_scalars()->clear_valid_data();
            col.mutable_scalars()->mutable_int_data()->clear_data();
            int row = 0;
            for (int64_t value :
                 {low, int64_t{-1}, int64_t{0}, high, int64_t{0}, int64_t{9}}) {
                col.mutable_scalars()->mutable_int_data()->add_data(value);
                col.mutable_scalars()->add_valid_data(row++ != 4);
            }
        }
        auto sealed = CreateSealedWithFieldDataLoaded(schema, data);
        auto growing = segcore::CreateGrowingSegment(schema, empty_index_meta);
        growing->PreInsert(6);
        growing->Insert(0,
                        6,
                        data.row_ids_.data(),
                        data.timestamps_.data(),
                        data.SharedRaw());
        expr::ColumnInfo field(id, type, {}, true);
        std::vector<expr::TypedExprPtr> predicates;
        for (auto op : {proto::plan::Equal,
                        proto::plan::NotEqual,
                        proto::plan::LessThan,
                        proto::plan::LessEqual,
                        proto::plan::GreaterThan,
                        proto::plan::GreaterEqual})
            for (int64_t bound : {low - 1, high + 1, INT64_MIN, INT64_MAX})
                predicates.push_back(
                    std::make_shared<expr::UnaryRangeFilterExpr>(
                        field, op, Int(bound)));
        for (auto bounds : {std::pair{low - 1, high + 1},
                            std::pair{high + 1, low - 1},
                            std::pair{low - 1, int64_t{0}},
                            std::pair{int64_t{0}, high + 1}})
            for (bool inclusive : {false, true})
                predicates.push_back(
                    std::make_shared<expr::BinaryRangeFilterExpr>(
                        field,
                        Int(bounds.first),
                        Int(bounds.second),
                        inclusive,
                        inclusive));
        for (size_t i = 0; i < predicates.size(); ++i) {
            SCOPED_TRACE(i);
            plan::CompileOptions options;
            options.plan_indexes = true;
            auto compiled = plan::CompileExpressions({predicates[i]}, options);
            ASSERT_TRUE(compiled.supported());
            EXPECT_TRUE(compiled.plan->predicates().empty());
            CheckExecution(*sealed, predicates[i], 6);
            CheckExecution(*growing, predicates[i], 6);
        }
    }
}

TEST(ComplexOperatorMigrationTest,
     SealedArrayViewsRetainPinsAndDetachOnRequest) {
    auto schema = std::make_shared<Schema>();
    auto pk = schema->AddDebugField("pk", DataType::INT64);
    schema->set_primary_field_id(pk);
    auto field = schema->AddDebugArrayField("a", DataType::INT64, true);
    auto data = segcore::DataGen(schema, 3);
    for (auto& column : *data.raw_->mutable_fields_data()) {
        if (column.field_id() != field.get())
            continue;
        column.clear_valid_data();
        auto* scalar = column.mutable_scalars();
        scalar->clear_valid_data();
        auto* arrays = scalar->mutable_array_data();
        arrays->clear_data();
        arrays->set_element_type(proto::schema::Int64);
        for (int i = 0; i < 3; ++i) {
            scalar->add_valid_data(i != 1);
            auto* values = arrays->add_data()->mutable_long_data();
            values->add_data(10 + i);
            values->add_data(20 + i);
        }
    }
    auto segment = CreateSealedWithFieldDataLoaded(schema, data);
    InputDependency dependency;
    dependency.field_id = field.get();
    dependency.type = DataType::ARRAY;
    dependency.element_type = DataType::INT64;
    dependency.needs_values = true;
    auto reader = access::MakeSegmentReaderBinding(
        *segment, segment->CaptureReadSnapshot(), 3)(dependency);
    ASSERT_TRUE(reader);
    auto loaded = reader->ReadRange(
        segcore::reader::RangeRequest(
            0,
            3,
            segcore::reader::ReadProjection::ValuesAndValidity,
            SelectionView::All(3)),
        {});
    ASSERT_TRUE(loaded.values);
    auto decoded = loaded.values->DecodeObjects<ArrayView>();
    ASSERT_TRUE(decoded);
    ASSERT_FALSE(loaded.values->DecodeObjects<Array>());
    auto mapped = loaded.values->Dictionary(
        std::make_shared<RowMapping>(std::vector<vector_size_t>{2, 0, 0, 1}));
    const auto* payload = loaded.values->ArrayViewAt(0).data();
    EXPECT_EQ(mapped.ArrayViewAt(1).data(), payload);
    auto detached = ReadOnlyVectorView::CopyOf(MakeWritableCopy(mapped).values);
    EXPECT_NE(detached.ArrayAt(1).data(), payload);
    loaded.values.reset();
    decoded.reset();
    reader.reset();
    segment.reset();
    EXPECT_EQ(mapped.ArrayViewAt(0).get_data_unchecked<int64_t>(0), 12);
    EXPECT_EQ(mapped.ArrayViewAt(1).data(), mapped.ArrayViewAt(2).data());
    EXPECT_FALSE(mapped.TryArrayViewAt(3));
    EXPECT_EQ(detached.ArrayAt(1).get_data_unchecked<int64_t>(1), 20);
}

TEST(ComplexOperatorMigrationTest, OrdinaryArrayOperationsGrowingAndSealed) {
    const std::vector<std::vector<int>> rows{
        {}, {1, 3, 6}, {2, 5}, {9, 10}, {}, {7}};
    for (auto type : {DataType::INT8,
                      DataType::INT64,
                      DataType::FLOAT,
                      DataType::DOUBLE,
                      DataType::BOOL,
                      DataType::VARCHAR}) {
        SCOPED_TRACE(static_cast<int>(type));
        auto schema = std::make_shared<Schema>();
        auto pk = schema->AddDebugField("pk", DataType::INT64);
        schema->set_primary_field_id(pk);
        auto id = schema->AddDebugArrayField("values", type, true);
        auto data = segcore::DataGen(schema, rows.size());
        auto literal = [type](int value) {
            proto::plan::GenericValue result;
            if (type == DataType::FLOAT || type == DataType::DOUBLE)
                result.set_float_val(value);
            else if (type == DataType::BOOL)
                result.set_bool_val(value % 2 != 0);
            else if (type == DataType::VARCHAR)
                result.set_string_val("v" + std::to_string(value));
            else
                result.set_int64_val(value);
            return result;
        };
        for (auto& col : *data.raw_->mutable_fields_data()) {
            if (col.field_id() != id.get())
                continue;
            col.clear_valid_data();
            col.mutable_scalars()->clear_valid_data();
            auto* arrays = col.mutable_scalars()->mutable_array_data();
            arrays->clear_data();
            arrays->set_element_type(
                static_cast<proto::schema::DataType>(type));
            for (size_t row = 0; row < rows.size(); ++row) {
                col.mutable_scalars()->add_valid_data(row != 4);
                auto* array = arrays->add_data();
                for (int value : rows[row]) {
                    switch (type) {
                        case DataType::INT8:
                            array->mutable_int_data()->add_data(value);
                            break;
                        case DataType::INT64:
                            array->mutable_long_data()->add_data(value);
                            break;
                        case DataType::FLOAT:
                            array->mutable_float_data()->add_data(value);
                            break;
                        case DataType::DOUBLE:
                            array->mutable_double_data()->add_data(value);
                            break;
                        case DataType::BOOL:
                            array->mutable_bool_data()->add_data(value % 2 !=
                                                                 0);
                            break;
                        case DataType::VARCHAR:
                            array->mutable_string_data()->add_data(
                                "v" + std::to_string(value));
                            break;
                        default:
                            FAIL();
                    }
                }
            }
        }
        auto sealed = CreateSealedWithFieldDataLoaded(schema, data);
        auto growing = segcore::CreateGrowingSegment(schema, empty_index_meta);
        growing->PreInsert(rows.size());
        growing->Insert(0,
                        rows.size(),
                        data.row_ids_.data(),
                        data.timestamps_.data(),
                        data.SharedRaw());
        expr::ColumnInfo field(id, DataType::ARRAY, type, {}, true);
        auto element = field;
        element.nested_path_ = {"1"};
        std::vector<expr::TypedExprPtr> predicates;
        proto::plan::GenericValue array_value;
        array_value.mutable_array_val()->set_same_type(true);
        *array_value.mutable_array_val()->add_array() = literal(9);
        *array_value.mutable_array_val()->add_array() = literal(10);
        for (auto op : {proto::plan::Equal, proto::plan::NotEqual}) {
            predicates.push_back(std::make_shared<expr::UnaryRangeFilterExpr>(
                field, op, array_value));
            predicates.push_back(std::make_shared<expr::UnaryRangeFilterExpr>(
                element, op, literal(3)));
        }
        predicates.push_back(std::make_shared<expr::TermFilterExpr>(
            field, std::vector<proto::plan::GenericValue>{literal(3)}, true));
        predicates.push_back(std::make_shared<expr::TermFilterExpr>(
            element,
            std::vector<proto::plan::GenericValue>{literal(3), literal(10)},
            false));
        predicates.push_back(std::make_shared<expr::TermFilterExpr>(
            element, std::vector<proto::plan::GenericValue>{}, false));
        for (auto op : {proto::plan::JSONContainsExpr_JSONOp_Contains,
                        proto::plan::JSONContainsExpr_JSONOp_ContainsAny,
                        proto::plan::JSONContainsExpr_JSONOp_ContainsAll}) {
            for (const auto& terms : {std::vector<proto::plan::GenericValue>{},
                                      std::vector<proto::plan::GenericValue>{
                                          literal(9), literal(10), literal(9)}})
                predicates.push_back(std::make_shared<expr::JsonContainsExpr>(
                    field, op, true, terms));
        }
        predicates.push_back(std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
            field,
            proto::plan::GreaterEqual,
            proto::plan::ArrayLength,
            Int(2),
            Int(0)));
        if (type != DataType::BOOL && type != DataType::VARCHAR) {
            predicates.push_back(std::make_shared<expr::BinaryRangeFilterExpr>(
                element, literal(2), literal(10), true, false));
            for (auto op : {proto::plan::Add,
                            proto::plan::Sub,
                            proto::plan::Mul,
                            proto::plan::Div,
                            proto::plan::Mod})
                predicates.push_back(
                    std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
                        element,
                        proto::plan::GreaterThan,
                        op,
                        literal(1),
                        literal(2)));
        }
        if (type == DataType::INT8 || type == DataType::INT64) {
            for (auto op : {proto::plan::BitAnd,
                            proto::plan::BitOr,
                            proto::plan::BitXor,
                            proto::plan::Shl,
                            proto::plan::Shr})
                predicates.push_back(
                    std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
                        element, proto::plan::GreaterThan, op, Int(1), Int(2)));
        }
        if (type == DataType::FLOAT) {
            proto::plan::GenericValue fractional;
            fractional.set_float_val(3.0000001);
            predicates.push_back(std::make_shared<expr::TermFilterExpr>(
                field,
                std::vector<proto::plan::GenericValue>{fractional},
                true));
            predicates.push_back(std::make_shared<expr::JsonContainsExpr>(
                field,
                proto::plan::JSONContainsExpr_JSONOp_ContainsAll,
                true,
                std::vector<proto::plan::GenericValue>{fractional}));
        }
        if (type == DataType::VARCHAR) {
            proto::plan::GenericValue pattern;
            pattern.set_string_val("v%");
            predicates.push_back(std::make_shared<expr::UnaryRangeFilterExpr>(
                element, proto::plan::Match, pattern));
        }
        for (size_t i = 0; i < predicates.size(); ++i) {
            SCOPED_TRACE(i);
            CheckExecution(*sealed, predicates[i], rows.size());
            CheckExecution(*growing, predicates[i], rows.size());
        }
    }
}

TEST(ComplexOperatorMigrationTest, JsonLengthAndArithmeticGrowingAndSealed) {
    auto schema = std::make_shared<Schema>();
    auto pk = schema->AddDebugField("pk", DataType::INT64);
    schema->set_primary_field_id(pk);
    auto id = schema->AddDebugField("json", DataType::JSON, true);
    auto data = segcore::DataGen(schema, 8);
    const std::vector<std::string> values{R"({"x":9007199254740993,"a":[]})",
                                          R"({"x":7,"a":[1,2]})",
                                          R"({"x":7.5,"a":[null,3,4]})",
                                          R"({"x":18446744073709551615,"a":5})",
                                          R"({"x":2,"a":[1]})",
                                          R"({"x":null})",
                                          R"({"x":"7","a":null})",
                                          R"({})"};
    for (auto& col : *data.raw_->mutable_fields_data()) {
        if (col.field_id() != id.get())
            continue;
        col.clear_valid_data();
        col.mutable_scalars()->clear_valid_data();
        col.mutable_scalars()->mutable_json_data()->clear_data();
        for (size_t row = 0; row < values.size(); ++row) {
            col.mutable_scalars()->mutable_json_data()->add_data(values[row]);
            col.mutable_scalars()->add_valid_data(row != 4);
        }
    }
    auto sealed = CreateSealedWithFieldDataLoaded(schema, data);
    auto growing = segcore::CreateGrowingSegment(schema, empty_index_meta);
    growing->PreInsert(values.size());
    growing->Insert(0,
                    values.size(),
                    data.row_ids_.data(),
                    data.timestamps_.data(),
                    data.SharedRaw());
    expr::ColumnInfo field(id, DataType::JSON, {"x"}, true);
    for (auto op : {proto::plan::Add,
                    proto::plan::Sub,
                    proto::plan::Mul,
                    proto::plan::Div,
                    proto::plan::Mod,
                    proto::plan::ArrayLength}) {
        for (bool floating : {false, true}) {
            SCOPED_TRACE(::testing::Message() << op << ":" << floating);
            auto input = field;
            if (op == proto::plan::ArrayLength)
                input.nested_path_ = {"a"};
            auto bound = Int(3), operand = Int(2);
            if (floating) {
                bound.set_float_val(3.5);
                operand.set_float_val(2.5);
            }
            auto predicate = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
                input, proto::plan::GreaterThan, op, bound, operand);
            CheckExecution(*sealed, predicate, values.size());
            CheckExecution(*growing, predicate, values.size());
        }
    }
    auto precise = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
        field,
        proto::plan::Equal,
        proto::plan::Sub,
        Int(9007199254740992),
        Int(1));
    CheckExecution(*sealed, precise, values.size());
    CheckExecution(*growing, precise, values.size());
}

TEST(ComplexOperatorMigrationTest, GeometryAllOperationsGrowingAndSealed) {
    auto schema = std::make_shared<Schema>();
    auto pk = schema->AddDebugField("pk", DataType::INT64);
    schema->set_primary_field_id(pk);
    auto field = schema->AddDebugField("g", DataType::GEOMETRY, true);
    auto data = segcore::DataGen(schema, 6);
    const std::vector<std::string> wkts{
        "POINT(0 0)",
        "POINT(3 4)",
        "LINESTRING(-2 0, 2 0)",
        "POLYGON((-1 -1, 1 -1, 1 1, -1 1, -1 -1))",
        "POINT(0 0)",
        "POINT EMPTY"};
    for (auto& col : *data.raw_->mutable_fields_data()) {
        if (col.field_id() != field.get())
            continue;
        col.clear_valid_data();
        col.mutable_scalars()->clear_valid_data();
        col.mutable_scalars()->mutable_geometry_data()->clear_data();
        for (size_t i = 0; i < wkts.size(); ++i) {
            Geometry geometry(GetThreadLocalGEOSContext(), wkts[i].c_str());
            col.mutable_scalars()->mutable_geometry_data()->add_data(
                geometry.to_wkb_string());
            col.mutable_scalars()->add_valid_data(i != 4);
        }
    }
    auto sealed = CreateSealedWithFieldDataLoaded(schema, data);
    auto growing = segcore::CreateGrowingSegment(schema, empty_index_meta);
    growing->PreInsert(6);
    growing->Insert(
        0, 6, data.row_ids_.data(), data.timestamps_.data(), data.SharedRaw());
    for (int operation = 1; operation <= 9; ++operation) {
        SCOPED_TRACE(operation);
        auto predicate = std::make_shared<expr::GISFunctionFilterExpr>(
            expr::ColumnInfo(field, DataType::GEOMETRY, {}, true),
            static_cast<GISFunctionType>(operation),
            "POINT(0 0)",
            5.0);
        CheckExecution(*sealed, predicate, 6);
        CheckExecution(*growing, predicate, 6);
        auto negated = std::make_shared<expr::LogicalUnaryExpr>(
            expr::LogicalUnaryExpr::OpType::LogicalNot, predicate);
        CheckExecution(*sealed, negated, 6);
        auto other = std::make_shared<expr::GISFunctionFilterExpr>(
            expr::ColumnInfo(field, DataType::GEOMETRY, {}, true),
            proto::plan::GISFunctionFilterExpr_GISOp_DWithin,
            "POINT(0 0)",
            1.0);
        for (auto op : {expr::LogicalBinaryExpr::OpType::And,
                        expr::LogicalBinaryExpr::OpType::Or}) {
            auto group =
                std::make_shared<expr::LogicalBinaryExpr>(op, predicate, other);
            CheckExecution(*sealed, group, 6);
            CheckExecution(*growing, group, 6);
        }
    }
}

TEST(ComplexOperatorMigrationTest, MatchAllQuantifiersGrowingAndSealed) {
    auto schema = std::make_shared<Schema>();
    auto pk = schema->AddDebugField("pk", DataType::INT64);
    schema->set_primary_field_id(pk);
    // Anchor is deliberately absent from the child predicate.
    auto anchor = schema->AddDebugArrayField(
        "items[anchor]", DataType::INT64, true, true);
    auto field =
        schema->AddDebugArrayField("items[value]", DataType::INT64, true, true);
    auto data = segcore::DataGen(schema, 6);
    const std::vector<std::vector<int64_t>> values{
        {}, {1, 4, 6}, {2, 5}, {9, 10}, {}, {7}};
    for (auto& col : *data.raw_->mutable_fields_data()) {
        if (col.field_id() != field.get() && col.field_id() != anchor.get())
            continue;
        col.clear_valid_data();
        col.mutable_scalars()->clear_valid_data();
        auto* arrays = col.mutable_scalars()->mutable_array_data();
        arrays->clear_data();
        arrays->set_element_type(proto::schema::Int64);
        for (size_t row = 0; row < values.size(); ++row) {
            col.mutable_scalars()->add_valid_data(row != 4);
            auto* array = arrays->add_data();
            if (row == 4) {
                // A row-NULL placeholder must not be decoded as a nullable
                // element array: its element bitmap is intentionally absent.
                array->mutable_long_data()->add_data(999);
            }
            for (size_t i = 0; i < values[row].size(); ++i) {
                array->mutable_long_data()->add_data(values[row][i]);
                array->add_valid_data(!(row == 2 || (row == 1 && i == 1)));
            }
        }
    }
    auto growing = segcore::CreateGrowingSegment(schema, empty_index_meta);
    growing->PreInsert(6);
    growing->Insert(
        0, 6, data.row_ids_.data(), data.timestamps_.data(), data.SharedRaw());
    // The legacy sealed ArrayChunk layout has no element-validity bitmap.
    // Exercise its supported representation separately; Growing above and the
    // kernel tests retain the nullable-element cases, including all-NULL rows.
    for (auto& col : *data.raw_->mutable_fields_data()) {
        if (col.field_id() != field.get() && col.field_id() != anchor.get())
            continue;
        for (auto& row :
             *col.mutable_scalars()->mutable_array_data()->mutable_data())
            row.clear_valid_data();
    }
    auto sealed = CreateSealedWithFieldDataLoaded(schema, data);
    for (auto& col : *data.raw_->mutable_fields_data()) {
        if (col.field_id() != field.get() && col.field_id() != anchor.get())
            continue;
        for (auto& row :
             *col.mutable_scalars()->mutable_array_data()->mutable_data())
            for (int i = 0; i < row.long_data().data_size(); ++i)
                row.add_valid_data(true);
    }
    auto plain_growing =
        segcore::CreateGrowingSegment(schema, empty_index_meta);
    plain_growing->PreInsert(6);
    plain_growing->Insert(
        0, 6, data.row_ids_.data(), data.timestamps_.data(), data.SharedRaw());
    expr::ColumnInfo element(field, DataType::ARRAY, DataType::INT64, {}, true);
    element.element_level_ = true;
    auto child = std::make_shared<expr::UnaryRangeFilterExpr>(
        element, proto::plan::GreaterThan, Int(3));
    for (int operation = 0; operation <= 4; ++operation) {
        for (int threshold : {0, 1, 2, 3}) {
            SCOPED_TRACE(::testing::Message() << operation << ":" << threshold);
            auto predicate = std::make_shared<expr::MatchExpr>(
                "items",
                static_cast<proto::plan::MatchType>(operation),
                threshold,
                child);
            CheckExecution(*sealed, predicate, 6);
            CheckExecution(*plain_growing, predicate, 6);
            // Rows contain [], [1,NULL,6], [NULL,NULL], [9,10], NULL, [7].
            // Explicit per-quantifier truth tables keep the NULL oracle
            // independent from both old flattening and the new reduction.
            std::vector<int> truth;
            switch (operation) {
                case proto::plan::MatchAll:
                    truth = {1, 0, 1, 1, -1, 1};
                    break;
                case proto::plan::MatchAny:
                    truth = {0, 1, 0, 1, -1, 1};
                    break;
                case proto::plan::MatchLeast:
                    truth = {threshold == 0,
                             threshold <= 1,
                             threshold == 0,
                             threshold <= 2,
                             -1,
                             threshold <= 1};
                    break;
                case proto::plan::MatchMost:
                    truth = {1,
                             threshold >= 1,
                             1,
                             threshold >= 2,
                             -1,
                             threshold >= 1};
                    break;
                case proto::plan::MatchExact:
                    truth = {threshold == 0,
                             threshold == 1,
                             threshold == 0,
                             threshold == 2,
                             -1,
                             threshold == 1};
                    break;
            }
            CheckExecution(*growing, predicate, 6, truth);
        }
    }
    auto constant = std::make_shared<expr::MatchExpr>(
        "items",
        proto::plan::MatchAll,
        0,
        std::make_shared<expr::AlwaysTrueExpr>());
    // Constant children still need the schema anchor for NULL/empty rows.
    auto compiled = plan::CompileExpressions(
        {constant}, plan::CompileOptions{"", {}, {}, false, schema});
    ASSERT_TRUE(compiled.supported());
    auto execution = FilterBitsExecution::TryCreate(
        constant,
        nullptr,
        sealed->CaptureReadSnapshot(),
        6,
        2,
        std::make_shared<segcore::reader::ReadContext>(),
        schema);
    ASSERT_TRUE(execution);
    auto output = execution->GetOutput();
    auto excluded = ReadOnlyVectorView::CopyOf(output->child(0));
    for (size_t row = 0; row < 6; ++row)
        EXPECT_EQ(excluded.ValueAt<bool>(row), row == 4);
    auto second_column = element;
    second_column.field_id_ = anchor;
    auto both = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::And,
        child,
        std::make_shared<expr::UnaryRangeFilterExpr>(
            second_column, proto::plan::LessThan, Int(10)));
    auto combined = std::make_shared<expr::MatchExpr>(
        "items", proto::plan::MatchAny, 0, both);
    CheckExecution(*sealed, combined, 6);
    CheckExecution(*plain_growing, combined, 6);
    CheckExecution(*growing, combined, 6, {0, 1, 0, 1, -1, 1});
}

TEST(ComplexOperatorMigrationTest, MatchSkipsEmptyNullAndUnselectedErrors) {
    expr::ColumnInfo field(FieldId(100), DataType::INT64, {}, true);
    auto arithmetic = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
        field, proto::plan::Equal, proto::plan::Div, Int(1), Int(0));
    auto child = plan::CompileExpressions({arithmetic});
    ASSERT_TRUE(child.supported());
    proto::schema::ScalarField values;
    values.mutable_long_data()->add_data(1);
    auto column = std::make_shared<ColumnVector>(DataType::ARRAY, 3);
    column->SetValueAt<Array>(0, Array{});
    column->nullAt(1);
    column->SetValueAt<Array>(2, Array(values));
    auto schema =
        std::make_shared<RowType>(std::vector<std::string>{"array"},
                                  std::vector<DataType>{DataType::ARRAY});
    RowVector row(schema, 3, {column});
    BatchInput input(row);
    scalar::EvalCtx ctx(input);
    auto ref = std::make_shared<scalar::InputRefExpr>(0, DataType::ARRAY);
    auto match = scalar::MakeArrayMatch(ref,
                                        {ref},
                                        child.plan->inputs().schema(),
                                        child.plan->expressions().roots()[0],
                                        proto::plan::MatchAll,
                                        0);
    auto selected = OwnedSelection::FromIndices(3, {0, 1});
    auto result = match->Eval(selected.view(), ctx);
    EXPECT_TRUE(result.ValueAt<bool>(0));
    EXPECT_FALSE(result.IsValid(1));
    try {
        match->Eval(SelectionView::All(3), ctx);
        FAIL();
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), ExprInvalid);
    }
    auto after = match->Eval(selected.view(), ctx);
    EXPECT_TRUE(after.ValueAt<bool>(0));
    EXPECT_FALSE(after.defined_rows().IsSelected(2));
}

TEST(ComplexOperatorMigrationTest,
     ArrayViewsDetachAndPreserveNullableElements) {
    proto::schema::ScalarField values;
    values.mutable_long_data()->add_data(3);
    values.mutable_long_data()->add_data(8);
    values.add_valid_data(true);
    values.add_valid_data(false);
    Array original(values, true);
    auto column = std::make_shared<ColumnVector>(DataType::ARRAY, 2);
    column->SetValueAt<Array>(0, original);
    column->nullAt(1);
    auto view = ReadOnlyVectorView::CopyOf(column);
    column->SetValueAt<Array>(0, Array{});
    auto mapped = view.Dictionary(
        std::make_shared<RowMapping>(std::vector<vector_size_t>{1, 0, 0}));
    EXPECT_FALSE(mapped.IsValid(0));
    EXPECT_EQ(mapped.ArrayAt(1).length(), 2);
    EXPECT_EQ(mapped.ArrayAt(2).get_data_unchecked<int64_t>(0), 3);
    EXPECT_FALSE(mapped.ArrayAt(2).is_element_valid(1));
    auto copy = MakeWritableCopy(mapped);
    auto detached = ReadOnlyVectorView::CopyOf(copy.values);
    EXPECT_EQ(detached.ArrayAt(1).length(), 2);
}

TEST(ComplexOperatorMigrationTest, MmapArrayReaderDetachesNullablePayloads) {
    auto manager = storage::MmapManager::GetInstance().GetMmapChunkManager();
    auto descriptor = manager->Register();
    auto column =
        std::make_unique<segcore::ConcurrentVector<Array>>(2, descriptor);
    proto::schema::ScalarField proto;
    for (int i = 0; i < 70; ++i) {
        proto.mutable_string_data()->add_data(i == 0 ? "first" : "x");
        proto.add_valid_data(i != 1 && i != 64 && i != 69);
    }
    std::vector<Array> arrays{Array(proto, true), Array{}, Array(proto, true)};
    column->set_data_raw(0, arrays.data(), 3);
    arrays.clear();
    auto reader = std::make_unique<segcore::reader::GrowingColumnReader<Array>>(
        *column, DataType::ARRAY, 3);
    column.reset();
    descriptor.reset();
    auto selected = OwnedSelection::FromIndices(4, {0, 2, 3});
    auto result =
        reader->Take(segcore::reader::TakeRequest(
                         {2, 1, 0, 2},
                         segcore::reader::ReadProjection::ValuesAndValidity,
                         selected.view()),
                     {});
    auto range = reader->ReadRange(
        segcore::reader::RangeRequest(
            0,
            3,
            segcore::reader::ReadProjection::ValuesAndValidity,
            SelectionView::All(3)),
        {});
    reader.reset();
    ASSERT_TRUE(result.values);
    EXPECT_FALSE(result.values->defined_rows().IsSelected(1));
    for (size_t row : {0, 2, 3}) {
        EXPECT_EQ(
            result.values->ArrayAt(row).get_data_unchecked<std::string>(0),
            "first");
        EXPECT_FALSE(result.values->ArrayAt(row).is_element_valid(1));
        EXPECT_TRUE(result.values->ArrayAt(row).is_element_valid(63));
        EXPECT_FALSE(result.values->ArrayAt(row).is_element_valid(64));
        EXPECT_FALSE(result.values->ArrayAt(row).is_element_valid(69));
    }
    EXPECT_EQ(range.values->ArrayAt(1).length(), 0);
    EXPECT_FALSE(range.values->ArrayAt(2).is_element_valid(1));
}

TEST(ComplexOperatorMigrationTest, GeometryInvalidRowsAndLazyErrors) {
    auto column = std::make_shared<ColumnVector>(DataType::GEOMETRY, 3);
    column->SetValueAt<std::string>(0, "broken-WKB");
    column->nullAt(1);
    Geometry geometry(GetThreadLocalGEOSContext(), "POINT(0 0)");
    column->SetValueAt<std::string>(2, geometry.to_wkb_string());
    auto schema =
        std::make_shared<RowType>(std::vector<std::string>{"g"},
                                  std::vector<DataType>{DataType::GEOMETRY});
    RowVector row(schema, 3, {column});
    BatchInput input(row);
    scalar::EvalCtx ctx(input);
    auto ref = std::make_shared<scalar::InputRefExpr>(0, DataType::GEOMETRY);
    auto valid = scalar::MakeGeometryPredicate(
        ref, proto::plan::GISFunctionFilterExpr_GISOp_STIsValid, "", 0);
    auto output = valid->Eval(SelectionView::All(3), ctx);
    EXPECT_FALSE(output.ValueAt<bool>(0));
    EXPECT_FALSE(output.IsValid(1));
    EXPECT_TRUE(output.ValueAt<bool>(2));
    auto bad = scalar::MakeGeometryPredicate(
        ref,
        proto::plan::GISFunctionFilterExpr_GISOp_Intersects,
        "invalid WKT",
        0);
    auto empty = bad->Eval(SelectionView::None(3), ctx);
    EXPECT_TRUE(empty.defined_rows().IsNone());
    EXPECT_THROW(bad->Eval(SelectionView::All(3), ctx), SegcoreError);
    auto cancel = [&] {
        ThrowInfo(FollyCancel, "injected complex cancellation");
    };
    scalar::EvalCtx cancelled(input, cancel);
    try {
        valid->Eval(SelectionView::All(3), cancelled);
        FAIL();
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), FollyCancel);
    }
}
}  // namespace
}  // namespace milvus::exec

namespace milvus::exec {
TEST(ComplexOperatorMigrationTest, NgramCandidatesFuseBeforeRawRefinement) {
    auto cm = storage::RemoteChunkManagerSingleton::GetInstance()
                  .GetRemoteChunkManager();
    storage::FileManagerContext context(cm);
    context.fieldDataMeta.field_id = 181;
    context.fieldDataMeta.field_schema.set_data_type(proto::schema::VarChar);
    context.fieldDataMeta.field_schema.set_nullable(true);
    context.indexMeta.build_id = 961029;
    std::vector<std::string> strings{
        "alpha beta", "alpha only", "beta only", "other", "", "null"};
    const bool valid[] = {true, true, true, true, true, false};
    auto source =
        BuildTestScalarIndex<std::string>(index::families::kNgram,
                                          strings.size(),
                                          strings.data(),
                                          valid,
                                          {{"min_gram", 2}, {"max_gram", 4}});
    auto resource = source.index;
    auto unary = [&](std::string text) {
        proto::plan::GenericValue value;
        value.set_string_val(std::move(text));
        return std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(FieldId(181), DataType::VARCHAR, {}, true),
            proto::plan::Match,
            value);
    };
    auto root = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::And,
        unary("%alpha%"),
        unary("%beta%"));
    plan::CompileOptions options;
    options.plan_indexes = true;
    auto compiled = plan::CompileExpressions({root}, options);
    ASSERT_TRUE(compiled.supported());
    auto context_read = std::make_shared<segcore::reader::ReadContext>();
    plan::PhysicalPrepareOptions physical;
    physical.bind_index = [&](const plan::PredicatePlan& p)
        -> std::optional<plan::PreparedIndex> {
        EXPECT_TRUE(p.ngram_query);
        return plan::PreparedIndex{
            segcore::reader::IndexResultKind::Candidates,
            std::make_shared<segcore::reader::NgramIndexReader>(
                resource, strings.size(), *p.ngram_query),
            context_read};
    };
    std::vector<int64_t> requested;
    auto prepared = plan::Prepare(
        compiled.plan,
        [&](const InputDependency&) {
            plan::PreparedInput input;
            input.type = DataType::VARCHAR;
            input.supports_values = true;
            input.owner = resource;
            input.make_loaders = [&](const ReadBatchBinding& binding) {
                return InputLoaders{[&, binding](SelectionView rows) {
                                        VectorResultBuilder values(
                                            DataType::VARCHAR, binding.size());
                                        rows.ForEachSelected([&](size_t row) {
                                            auto offset = binding.offset(row);
                                            requested.push_back(offset);
                                            if (offset != 5)
                                                values.Set<std::string>(
                                                    row, strings[offset]);
                                        });
                                        return std::move(values).Finish(rows);
                                    },
                                    {}};
            };
            return input;
        },
        physical);
    ASSERT_GT(prepared.predicate_routes().size(), 2);
    auto batch = prepared.MakeExecutionBatch(
        ReadBatchBinding::Offsets({2, 0, 1, 5, 0, 3, 4}));
    auto result = batch->Eval(SelectionView::All(7))[0];
    EXPECT_EQ(requested, (std::vector<int64_t>{0, 0}));
    for (size_t row : {0, 2, 5, 6}) {
        EXPECT_TRUE(result.IsValid(row));
        EXPECT_FALSE(result.ValueAt<bool>(row));
    }
    for (size_t row : {1, 4}) EXPECT_TRUE(result.ValueAt<bool>(row));
    EXPECT_FALSE(result.IsValid(3));
    // Unsupported short literals keep all valid rows as candidates, and an
    // invalid reached pattern cannot be hidden by empty candidates.
    segcore::reader::NgramIndexReader short_reader(
        resource, strings.size(), {proto::plan::Match, "%a%", ""});
    auto short_result = short_reader.Query({});
    EXPECT_EQ(short_result.selected().selected_count(), 5);
    segcore::reader::NgramIndexReader invalid_reader(
        resource, strings.size(), {proto::plan::Match, "abc\\", ""});
    EXPECT_THROW(invalid_reader.Query({}), SegcoreError);
}
}  // namespace milvus::exec
