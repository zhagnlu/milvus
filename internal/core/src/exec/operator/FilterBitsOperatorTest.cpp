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
#include <thread>

#include "common/VectorAccess.h"
#include "exec/operator/FilterBitsNode.h"
#include "exec/operator/MvccNode.h"
#include "segcore/reader/GrowingColumnReader.h"
#include "test_utils/cachinglayer_test_utils.h"
#include "test_utils/storage_test_utils.h"
#include "exec/operator/CandidateFilter.h"
#include "exec/operator/IterativeFilterNode.h"
#include "rescores/BoostScoreRunner.h"
#include "exec/operator/FilterBitsExecution.h"
#include "segcore/reader/ReaderTestUtils.h"

namespace milvus::exec {

class FilterBitsNodeTestAccess {
 public:
    static void
    ExpectWholeIndex(const PhyFilterBitsNode& node) {
        ASSERT_TRUE(node.scalar_filter_);
        EXPECT_EQ(node.scalar_filter_->prepared().granularity(),
                  exec::plan::ExecutionGranularity::WholeSegment);
        for (auto route : node.scalar_filter_->prepared().predicate_routes())
            EXPECT_EQ(route, exec::plan::PredicateRoute::ExactIndex);
    }
    static void
    ExpectRoute(const PhyFilterBitsNode& node, bool scalar) {
        EXPECT_EQ(node.scalar_filter_ != nullptr, scalar);
        EXPECT_EQ(node.exprs_ != nullptr, !scalar);
        if (scalar)
            EXPECT_FALSE(node.enable_expr_cache_);
    }

    static int64_t
    ProcessedRows(const PhyFilterBitsNode& node) {
        return node.num_processed_rows_;
    }
};

namespace {

// Delegate scans to the real loaded column; intercept each actual read. A
// metadata-only slot supplies the ordinary-column capability and row geometry;
// reading that slot would fail (it has no payload chunks).
// A failure on read 2 happens after one batch was evaluated, even when
// successive batches reuse a physical cursor.
class ObservedFilterColumn : public ChunkedColumn {
 public:
    ObservedFilterColumn(std::shared_ptr<ChunkedColumn> column,
                         const FieldMeta& field)
        : ChunkedColumn(MetadataSlot(*column), field),
          column_(std::move(column)) {
    }

    ScanResult
    Scan(OpContext* context, const ScanOptions& options) const override {
        return std::make_unique<ObservedCursor>(column_->Scan(context, options),
                                                *this);
    }

 private:
    class ObservedCursor : public ScanCursor {
     public:
        ObservedCursor(ScanResult cursor, const ObservedFilterColumn& observer)
            : cursor_(std::move(cursor)), observer_(observer) {
        }
        int64_t
        Position() const override {
            return cursor_->Position();
        }
        void
        Seek(int64_t position) override {
            cursor_->Seek(position);
        }
        bool
        Next(int64_t length, ScanReadMode mode, ScanBatch* out) override {
            observer_.starts.push_back(Position());
            if (observer_.before_read)
                observer_.before_read(observer_.starts.size());
            return cursor_->Next(length, mode, out);
        }

     private:
        ScanResult cursor_;
        const ObservedFilterColumn& observer_;
    };

 public:
    mutable std::vector<int64_t> starts;
    std::function<void(size_t)> before_read;

 private:
    static std::shared_ptr<cachinglayer::CacheSlot<Chunk>>
    MetadataSlot(const ChunkedColumn& column) {
        std::vector<int64_t> sizes;
        for (int64_t chunk = 0; chunk < column.num_chunks(); ++chunk)
            sizes.push_back(column.chunk_row_nums(chunk));
        return cachinglayer::Manager::GetInstance().CreateCacheSlot<Chunk>(
            std::make_unique<TestChunkTranslator>(
                sizes,
                "filter-bits-observed-column",
                std::vector<std::unique_ptr<Chunk>>{}),
            nullptr);
    }

    std::shared_ptr<ChunkedColumn> column_;
};

class ObservedFilterSnapshot : public segcore::SegmentReadSnapshot {
 public:
    ObservedFilterSnapshot(
        std::shared_ptr<const segcore::SegmentReadSnapshot> snapshot,
        FieldId field,
        std::shared_ptr<ObservedFilterColumn> column)
        : snapshot_(std::move(snapshot)),
          field_(field),
          column_(std::move(column)) {
    }

    int64_t
    chunk_size(FieldId field,
               int64_t chunk,
               OpContext* op_ctx = nullptr) const override {
        return snapshot_->chunk_size(field, chunk);
    }
    int64_t
    num_rows_until_chunk(FieldId field,
                         int64_t chunk,
                         OpContext* op_ctx = nullptr) const override {
        return snapshot_->num_rows_until_chunk(field, chunk);
    }
    std::pair<int64_t, int64_t>
    get_chunk_by_offset(FieldId field,
                        int64_t offset,
                        OpContext* op_ctx = nullptr) const override {
        return snapshot_->get_chunk_by_offset(field, offset);
    }
    int64_t
    num_chunk_data(FieldId field, OpContext* op_ctx = nullptr) const override {
        return snapshot_->num_chunk_data(field);
    }
    int64_t
    get_row_count() const override {
        return snapshot_->get_row_count();
    }
    std::pair<std::shared_ptr<ChunkedColumnInterface>, FieldSkipMetricsView>
    GetDataScanResources(FieldId field) const override {
        if (fail_lookup)
            ThrowInfo(FieldNotLoaded, "injected FilterBits snapshot failure");
        auto resources = snapshot_->GetDataScanResources(field);
        if (field == field_)
            resources.first = column_;
        return resources;
    }

    bool fail_lookup = false;

 private:
    std::shared_ptr<const segcore::SegmentReadSnapshot> snapshot_;
    FieldId field_;
    std::shared_ptr<ObservedFilterColumn> column_;
};

// Only drive the real FilterBits operator. Task/DriverContext supply its normal
// context; no Driver pipeline, MVCC or VectorSearch operator is constructed.
struct FilterRun {
    std::shared_ptr<QueryContext> query;
    std::shared_ptr<Task> task;
    DriverContext driver;
    PhyFilterBitsNode node;

    FilterRun(std::shared_ptr<QueryContext> context,
              const expr::TypedExprPtr& predicate)
        : query(std::move(context)),
          task(Task::Create(
              "filter-bits-test", ::milvus::plan::PlanFragment(), 0, query)),
          driver(task, 0, 0, 0, 0),
          node(0,
               &driver,
               std::make_shared<::milvus::plan::FilterBitsNode>("filter",
                                                                predicate)) {
    }
};

class FilterBitsOperatorTest : public ::testing::Test {
 protected:
    void
    LoadSortedIndex() {
        const int64_t values[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        const bool valid[] = {
            true, false, true, true, true, false, true, true, true, false};
        auto sorted = BuildTestScalarIndex<int64_t>(
            index::families::kSort, 10, values, valid);
        segcore::LoadIndexInfo info;
        info.field_id = value_.get();
        info.index_params = GenIndexParams(sorted.get());
        SetTestIndexMetadata(info, *sorted, index::families::kSort);
        info.index = CreateTestIndex("filter-bits-sorted", std::move(sorted));
        segment_->LoadIndex(info);
    }
    void
    Load(bool entity_ttl = false) {
        schema_ = std::make_shared<Schema>();
        auto pk = schema_->AddDebugField("pk", DataType::INT64);
        value_ = schema_->AddDebugField("value", DataType::INT64, true);
        ttl_ = schema_->AddDebugField("expires", DataType::INT64, true);
        schema_->set_primary_field_id(pk);
        if (entity_ttl)
            schema_->set_ttl_field_id(ttl_);

        // Use the standard test loader (local ChunkManager initialized by
        // init_gtest), but explicit values/validity give an independent oracle.
        auto data = segcore::DataGen(schema_, 10);
        for (auto& field : *data.raw_->mutable_fields_data()) {
            const FieldId id(field.field_id());
            if (id != value_ && id != ttl_)
                continue;
            FixedVector<int64_t> values(10);
            FixedVector<bool> valid(10, true);
            const int64_t expiry[] = {99, 0, 100, 101, 99, 0, 101, 100, 101, 0};
            for (size_t row = 0; row < 10; ++row) {
                values[row] = id == value_ ? row : expiry[row];
                valid[row] = row % 4 != 1;
            }
            auto replacement = segcore::CreateDataArrayFrom(
                values.data(), valid.data(), 10, (*schema_)[id]);
            field.Swap(replacement.get());
        }
        segment_ = CreateSealedWithFieldDataLoaded(schema_, data);
    }

    expr::TypedExprPtr
    AtLeast(int64_t threshold) const {
        proto::plan::GenericValue value;
        value.set_int64_val(threshold);
        return std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(value_, DataType::INT64, {}, true),
            proto::plan::GreaterEqual,
            value);
    }

    std::shared_ptr<QueryContext>
    Context(bool scalar,
            int64_t batch = 3,
            int64_t rows = 10,
            bool snapshot = true,
            int64_t physical_us = 100) const {
        std::unordered_map<std::string, std::string> config{
            {QueryConfig::kExprEvalBatchSize, std::to_string(batch)}};
        // Omitting the flag exercises default-off, not just explicit false.
        if (scalar)
            config.emplace(QueryConfig::kEnableExprExecutionV2, "true");
        auto query = std::make_shared<QueryContext>(
            "filter-bits-test",
            segment_.get(),
            rows,
            Timestamp(42) << 18,
            0,
            0,
            query::PlanOptions{},
            std::make_shared<QueryConfig>(config),
            nullptr,
            std::unordered_map<std::string, std::shared_ptr<BaseConfig>>{},
            physical_us);
        if (snapshot)
            query->set_read_snapshot(segment_->CaptureReadSnapshot());
        return query;
    }

    static void
    ExpectOutput(FilterRun& run, const std::vector<bool>& excluded) {
        // Both hooks must also work when no legacy ExprSet was constructed.
        auto pool = std::make_shared<folly::CPUThreadPoolExecutor>(1);
        run.node.PrefetchAsync(pool);
        run.node.WaitPrefetch();
        auto output = run.node.GetOutput();
        ASSERT_NE(output, nullptr);
        ASSERT_EQ(output->childrens().size(), 1);
        auto column = ReadOnlyVectorView::CopyOf(output->child(0));
        ASSERT_EQ(column.size(), excluded.size());
        for (size_t row = 0; row < excluded.size(); ++row) {
            EXPECT_TRUE(column.IsValid(row)) << row;
            EXPECT_EQ(column.ValueAt<bool>(row), excluded[row]) << row;
        }
        EXPECT_EQ(FilterBitsNodeTestAccess::ProcessedRows(run.node),
                  excluded.size());
        EXPECT_TRUE(run.node.IsFinished());
        EXPECT_EQ(run.node.GetOutput(), nullptr);
        run.node.Close();
        EXPECT_TRUE(run.node.IsFinished());
    }

    std::shared_ptr<ObservedFilterColumn>
    Observe(const std::shared_ptr<QueryContext>& context) const {
        auto snapshot = context->get_read_snapshot();
        auto column = std::dynamic_pointer_cast<ChunkedColumn>(
            snapshot->GetDataScanResources(value_).first);
        EXPECT_NE(column, nullptr);
        if (!column)
            return nullptr;
        auto observed =
            std::make_shared<ObservedFilterColumn>(column, (*schema_)[value_]);
        context->set_read_snapshot(std::make_shared<ObservedFilterSnapshot>(
            snapshot, value_, observed));
        return observed;
    }

    SchemaPtr schema_;
    FieldId value_{0};
    FieldId ttl_{0};
    std::unique_ptr<segcore::SegmentSealed> segment_;
};

class GrowingFilterBitsOperatorTest : public FilterBitsOperatorTest {
 protected:
    void
    LoadGrowing(bool mmap = false, bool entity_ttl = false) {
        schema_ = std::make_shared<Schema>();
        pk_ = schema_->AddDebugField("pk", DataType::INT64);
        schema_->set_primary_field_id(pk_);
        value_ = schema_->AddDebugField("value", DataType::INT64, true);
        ttl_ = schema_->AddDebugField("expiry", DataType::INT64, true);
        text_ = schema_->AddDebugField("str", DataType::VARCHAR, true);
        json_ = schema_->AddDebugField("json", DataType::JSON, true);
        if (entity_ttl)
            schema_->set_ttl_field_id(ttl_);
        auto& mmap_config = storage::MmapManager::GetInstance().GetMmapConfig();
        if (!previous_mmap_)
            previous_mmap_ = mmap_config.growing_enable_mmap;
        mmap_config.growing_enable_mmap = mmap;
        auto config = segcore::SegcoreConfig::default_config();
        config.set_chunk_rows(3);
        growing_ = segcore::CreateGrowingSegment(
            schema_, empty_index_meta, 91, config);
        Append(0, 10);
    }

    void
    Append(int64_t begin, int64_t count) {
        auto data = segcore::DataGen(schema_, count);
        for (auto& field : *data.raw_->mutable_fields_data()) {
            const FieldId id(field.field_id());
            FixedVector<bool> valid(count, true);
            FixedVector<int64_t> numbers(count);
            std::vector<std::string> strings;
            std::vector<std::string> jsons;
            for (int64_t i = 0; i < count; ++i) {
                const auto row = begin + i;
                valid[i] = id == pk_ || row % 4 != 1;
                numbers[i] = id == ttl_ ? 99 + row % 3 : row;
                strings.push_back(row % 2 ? "北京" : "hello");
                // CreateDataArrayFrom's JSON input is UTF-8 strings, not Json objects.
                jsons.emplace_back(row % 2 ? "{\"x\":1}" : "{}");
            }
            const void* raw =
                id == text_   ? static_cast<const void*>(strings.data())
                : id == json_ ? static_cast<const void*>(jsons.data())
                              : static_cast<const void*>(numbers.data());
            auto replacement = segcore::CreateDataArrayFrom(
                raw, valid.data(), count, (*schema_)[id]);
            field.Swap(replacement.get());
        }
        for (int64_t i = 0; i < count; ++i) {
            data.row_ids_[i] = begin + i;
            data.timestamps_[i] = 10 + begin + i;
        }
        auto offset = growing_->PreInsert(count);
        ASSERT_EQ(offset, begin);
        growing_->Insert(offset,
                         count,
                         data.row_ids_.data(),
                         data.timestamps_.data(),
                         data.SharedRaw());
    }

    std::shared_ptr<QueryContext>
    GrowingContext(bool scalar,
                   int64_t batch = 3,
                   Timestamp ts = 19,
                   Timestamp collection_ttl = 0) const {
        std::unordered_map<std::string, std::string> config{
            {QueryConfig::kExprEvalBatchSize, std::to_string(batch)}};
        if (scalar)
            config.emplace(QueryConfig::kEnableExprExecutionV2, "true");
        return std::make_shared<QueryContext>(
            "growing-filter",
            growing_.get(),
            growing_->get_active_count(ts),
            ts,
            collection_ttl,
            0,
            query::PlanOptions{},
            std::make_shared<QueryConfig>(config),
            nullptr,
            std::unordered_map<std::string, std::shared_ptr<BaseConfig>>{},
            100);
    }

    void
    TearDown() override {
        growing_.reset();
        if (previous_mmap_)
            storage::MmapManager::GetInstance()
                .GetMmapConfig()
                .growing_enable_mmap = *previous_mmap_;
    }
    std::optional<bool> previous_mmap_;
    FieldId pk_{0}, text_{0}, json_{0};
    segcore::SegmentGrowingPtr growing_;
};

TEST_F(GrowingFilterBitsOperatorTest,
       RealInsertNullablePrefixAndTailMatchLegacy) {
    for (bool mmap : {false, true}) {
        LoadGrowing(mmap);
        for (Timestamp ts : {Timestamp(9), Timestamp(16), Timestamp(19)}) {
            for (int64_t batch : {1, 4, 16}) {
                for (bool scalar : {false, true}) {
                    auto context = GrowingContext(scalar, batch, ts);
                    EXPECT_EQ(context->get_read_snapshot(), nullptr);
                    FilterRun run(context, AtLeast(3));
                    FilterBitsNodeTestAccess::ExpectRoute(run.node, scalar);
                    const auto rows = context->get_active_count();
                    if (!rows) {
                        EXPECT_EQ(run.node.GetOutput(), nullptr);
                        EXPECT_TRUE(run.node.IsFinished());
                    } else {
                        std::vector<bool> expected(rows);
                        for (int64_t i = 0; i < rows; ++i)
                            expected[i] = i < 3 || i % 4 == 1;
                        ExpectOutput(run, expected);
                    }
                }
            }
        }
    }
}

TEST_F(GrowingFilterBitsOperatorTest, BoundPrefixDoesNotGrowDuringAppend) {
    LoadGrowing(true);
    // Even inserts whose timestamp fits a later query cannot enlarge this
    // already prepared domain. ConcurrentVector may publish new chunks.
    auto context = GrowingContext(true, 2, 1000);
    FilterRun run(context, AtLeast(3));
    FilterBitsNodeTestAccess::ExpectRoute(run.node, true);
    std::thread writer([&] { Append(10, 20); });
    auto output = run.node.GetOutput();
    writer.join();
    auto result = ReadOnlyVectorView::CopyOf(output->child(0));
    ASSERT_EQ(result.size(), 10);
    for (size_t i = 0; i < result.size(); ++i)
        EXPECT_EQ(result.ValueAt<bool>(i), i < 3 || i % 4 == 1);
    EXPECT_EQ(growing_->get_active_count(1000), 30);
    EXPECT_EQ(FilterBitsNodeTestAccess::ProcessedRows(run.node), 10);
}

TEST_F(GrowingFilterBitsOperatorTest, StringJsonAndValidityMatchLegacy) {
    for (bool mmap : {false, true}) {
        LoadGrowing(mmap);
        proto::plan::GenericValue pattern;
        pattern.set_string_val("hello");
        auto string_test = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(text_, DataType::VARCHAR, {}, true),
            proto::plan::Equal,
            pattern);
        auto exists = std::make_shared<expr::ExistsExpr>(
            expr::ColumnInfo(json_, DataType::JSON, {"x"}, true));
        auto is_null = std::make_shared<expr::NullExpr>(
            expr::ColumnInfo(value_, DataType::INT64, {}, true),
            proto::plan::NullExpr_NullOp_IsNull);
        for (auto predicate :
             std::vector<expr::TypedExprPtr>{string_test, exists, is_null}) {
            for (bool scalar : {false, true}) {
                SCOPED_TRACE(fmt::format("mmap={} scalar={} predicate={}",
                                         mmap,
                                         scalar,
                                         predicate == string_test ? "string"
                                         : predicate == exists    ? "exists"
                                                                  : "is_null"));
                FilterRun run(GrowingContext(scalar), predicate);
                FilterBitsNodeTestAccess::ExpectRoute(run.node, scalar);
                std::vector<bool> expected(10);
                for (int i = 0; i < 10; ++i) {
                    const bool valid = i % 4 != 1;
                    const bool matches =
                        predicate == string_test ? valid && i % 2 == 0
                        : predicate == exists    ? valid && i % 2 == 1
                                                 : !valid;
                    expected[i] = !matches;
                }
                ExpectOutput(run, expected);
            }
        }
    }
}

TEST_F(GrowingFilterBitsOperatorTest, BoundMmapReadersOutliveSegment) {
    LoadGrowing(true);
    auto bind = access::MakeSegmentReaderBinding(*growing_, nullptr, 10);
    auto text = bind(InputDependency{"", text_.get(), DataType::VARCHAR});
    auto json = bind(InputDependency{"", json_.get(), DataType::JSON});
    ASSERT_NE(text, nullptr);
    ASSERT_NE(json, nullptr);
    bind = {};
    growing_.reset();
    using namespace segcore::reader;
    auto request = RangeRequest(
        0, 10, ReadProjection::ValuesAndValidity, SelectionView::All(10));
    auto strings = text->ReadRange(request, {});
    auto jsons = json->ReadRange(request, {});
    EXPECT_EQ(strings.values->StringViewAt(0), "hello");
    EXPECT_FALSE(strings.values->IsValid(1));
    EXPECT_TRUE(jsons.values->ValueAt<Json>(3).exist("/x"));
    text.reset();
    json.reset();
    EXPECT_EQ(strings.values->StringViewAt(3), "北京");
}

TEST_F(GrowingFilterBitsOperatorTest, CancellationDoesNotPublishOrFallback) {
    LoadGrowing();
    folly::CancellationSource source;
    OpContext op(source.getToken());
    auto context = GrowingContext(true);
    context->set_op_context(&op);
    FilterRun run(context, AtLeast(3));
    source.requestCancellation();
    RowVectorPtr output;
    EXPECT_THROW(output = run.node.GetOutput(), folly::FutureCancellation);
    EXPECT_EQ(output, nullptr);
    FilterBitsNodeTestAccess::ExpectRoute(run.node, true);
    EXPECT_EQ(FilterBitsNodeTestAccess::ProcessedRows(run.node), 0);
    EXPECT_FALSE(run.node.IsFinished());
}

TEST_F(GrowingFilterBitsOperatorTest,
       EntityTTLAndDownstreamMvccDeleteMatchLegacy) {
    LoadGrowing(false, true);
    proto::schema::IDs ids;
    ids.mutable_int_id()->add_data(5);
    Timestamp deleted_at = 18;
    ASSERT_TRUE(growing_->Delete(1, &ids, &deleted_at).ok());
    for (bool scalar : {false, true}) {
        auto context = GrowingContext(scalar, 3, 19, 12);
        FilterRun run(context, std::make_shared<expr::AlwaysTrueExpr>());
        FilterBitsNodeTestAccess::ExpectRoute(run.node, scalar);
        auto filtered = run.node.GetOutput();
        auto filter_plan = std::make_shared<::milvus::plan::FilterBitsNode>(
            "filter", std::make_shared<expr::AlwaysTrueExpr>());
        PhyMvccNode mvcc(
            1,
            &run.driver,
            std::make_shared<::milvus::plan::MvccNode>(
                "mvcc", std::vector<::milvus::plan::PlanNodePtr>{filter_plan}));
        mvcc.AddInput(filtered);
        auto result = ReadOnlyVectorView::CopyOf(mvcc.GetOutput()->child(0));
        for (size_t row = 0; row < 10; ++row) {
            const bool expired = row % 4 != 1 && 99 + row % 3 <= 100;
            EXPECT_EQ(result.ValueAt<bool>(row),
                      expired || row <= 2 || row == 5);
        }
    }
}

TEST_F(GrowingFilterBitsOperatorTest,
       BindingRejectsUnknownFieldAndInvalidDomain) {
    LoadGrowing();
    EXPECT_THROW(access::MakeSegmentReaderBinding(*growing_, nullptr, 11),
                 SegcoreError);
    auto bind = access::MakeSegmentReaderBinding(*growing_, nullptr, 10);
    EXPECT_THROW(bind(InputDependency{"", 987654, DataType::INT64}),
                 SegcoreError);
    EXPECT_THROW(bind(InputDependency{"", value_.get(), DataType::DOUBLE}),
                 SegcoreError);
    auto dependency = InputDependency{"", value_.get(), DataType::INT64};
    dependency.element_level = true;
    EXPECT_EQ(bind(dependency), nullptr);
    dependency.element_level = false;
    dependency.relation = "another-segment";
    EXPECT_EQ(bind(dependency), nullptr);
}

TEST_F(FilterBitsOperatorTest, CandidateOffsetsAndBoostSharePreparedIndex) {
    Load();
    LoadSortedIndex();
    segment_->DropFieldData(value_);
    auto query = Context(true);
    ExecContext context(query.get());
    auto predicate = AtLeast(4);
    auto candidate = [&] {
        ExecContext temporary(query.get());
        return CandidateFilter::TryCreate(predicate, &temporary);
    }();
    ASSERT_TRUE(candidate);
    FixedVector<int32_t> offsets{8, 1, 4, 8, 5, 0};
    auto matches = candidate->Eval(offsets);
    for (size_t i = 0; i < offsets.size(); ++i)
        EXPECT_EQ(matches[i], i == 0 || i == 2 || i == 3);
    EXPECT_EQ(candidate->Eval(FixedVector<int32_t>{}).size(), 0);
    EXPECT_ANY_THROW(candidate->Eval(FixedVector<int32_t>{10}));
    EXPECT_ANY_THROW(candidate->Eval(FixedVector<int32_t>{-1}));
    EXPECT_EQ(candidate->Eval(FixedVector<int32_t>{7, 3}).count(), 1);

    auto scorer = std::make_shared<rescores::WeightScorer>(predicate, 2.0f);
    std::unique_ptr<ExprSet> legacy;
    std::unique_ptr<CandidateFilter> prepared;
    auto full = rescores::ComputeNonNativeFilterBitset(
        &context, scorer, &legacy, &prepared);
    EXPECT_FALSE(full);
    EXPECT_FALSE(legacy);
    ASSERT_TRUE(prepared);
    for (int batch = 0; batch < 2; ++batch) {
        std::vector<std::optional<float>> scores(offsets.size());
        rescores::ComputeScorerScores(&context,
                                      nullptr,
                                      segment_.get(),
                                      scorer,
                                      offsets,
                                      scores,
                                      nullptr,
                                      nullptr,
                                      prepared.get());
        for (size_t i = 0; i < offsets.size(); ++i) {
            EXPECT_EQ(scores[i].has_value(), bool(matches[i]));
            if (scores[i])
                EXPECT_EQ(*scores[i], 2.0f);
        }
    }
}

class CandidateTestIterator : public VectorIterator {
 public:
    explicit CandidateTestIterator(std::vector<std::pair<int64_t, float>> rows)
        : rows_(std::move(rows)) {
    }
    bool
    HasNext() override {
        return next_ < rows_.size();
    }
    std::optional<std::pair<int64_t, float>>
    Next() override {
        return HasNext() ? std::optional(rows_[next_++]) : std::nullopt;
    }

 private:
    std::vector<std::pair<int64_t, float>> rows_;
    size_t next_ = 0;
};

TEST_F(FilterBitsOperatorTest,
       IterativeIndexOnlySkipsEmptyBatchesAndKeepsDistances) {
    Load();
    LoadSortedIndex();
    segment_->DropFieldData(value_);
    auto query = Context(true);
    SearchInfo search;
    search.metric_type_ = "L2";
    query->set_search_info(search);
    SearchResult result;
    result.total_nq_ = 2;
    result.unity_topK_ = 2;
    result.vector_iterators_ = std::vector<std::shared_ptr<VectorIterator>>{
        std::make_shared<CandidateTestIterator>(
            std::vector<std::pair<int64_t, float>>{
                {1, .1f}, {3, .2f}, {8, .3f}, {5, .4f}, {4, .5f}}),
        std::make_shared<CandidateTestIterator>(
            std::vector<std::pair<int64_t, float>>{
                {7, .6f}, {0, .7f}, {6, .8f}})};
    query->set_search_result(std::move(result));
    auto task = Task::Create(
        "candidate-test", ::milvus::plan::PlanFragment(), 0, query);
    DriverContext driver(task, 0, 0, 0, 0);
    PhyIterativeFilterNode node(
        0,
        &driver,
        std::make_shared<::milvus::plan::IterativeFilterNode>(
            "filter", AtLeast(4), std::vector<::milvus::plan::PlanNodePtr>{}));
    auto input = std::make_shared<RowVector>(std::vector<VectorPtr>{});
    node.AddInput(input);
    node.NoMoreInput();
    node.GetOutput();
    const auto& actual = query->get_search_result();
    const int64_t expected[] = {8, 4, 7, 6};
    const float distances[] = {.3f, .5f, .6f, .8f};
    ASSERT_EQ(actual.seg_offsets_.size(), 4);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(actual.seg_offsets_[i], expected[i]);
        EXPECT_FLOAT_EQ(actual.distances_[i], distances[i]);
    }
    node.Close();
}

class FailingCandidateIndex : public segcore::reader::TestScalarIndex<int64_t> {
 public:
    FailingCandidateIndex(index::IIndexReaderBasePtr reader,
                          std::shared_ptr<int> calls)
        : TestScalarIndex(std::move(reader)), calls_(std::move(calls)) {
    }
    TargetBitmap
    Range(const int64_t& value, index::CompareOp op) const override {
        if (++*calls_ == 1)
            ThrowInfo(FollyCancel, "injected candidate index cancellation");
        return TestScalarIndex::Range(value, op);
    }

 private:
    std::shared_ptr<int> calls_;
};
TEST_F(FilterBitsOperatorTest,
       CandidateIndexFailurePropagatesAndDoesNotCachePartialResult) {
    Load();
    auto calls = std::make_shared<int>(0);
    const int64_t values[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto source =
        BuildTestScalarIndex<int64_t>(index::families::kSort, 10, values);
    segcore::LoadIndexInfo info;
    info.field_id = value_.get();
    info.index_params = GenIndexParams(source.get());
    SetTestIndexMetadata(info, *source, index::families::kSort);
    auto owner = source.index;
    info.index = std::make_shared<segcore::reader::TestReaderIndex>(
        owner->Metadata(), [owner, calls](OpContext* ctx) {
            return std::make_unique<FailingCandidateIndex>(
                owner->CreateReader(ctx), calls);
        });
    segment_->LoadIndex(info);
    segment_->DropFieldData(value_);
    auto query = Context(true);
    ExecContext context(query.get());
    auto candidate = CandidateFilter::TryCreate(AtLeast(4), &context);
    ASSERT_TRUE(candidate);
    EXPECT_EQ(*calls, 0);
    try {
        candidate->Eval(FixedVector<int32_t>{8, 2});
        FAIL() << "expected index failure";
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), FollyCancel);
    }
    auto result = candidate->Eval(FixedVector<int32_t>{2, 8});
    EXPECT_FALSE(result[0]);
    EXPECT_TRUE(result[1]);
    EXPECT_EQ(*calls, 2);
    EXPECT_EQ(candidate->Eval(FixedVector<int32_t>{7}).count(), 1);
    EXPECT_EQ(*calls, 2);
}

TEST_F(FilterBitsOperatorTest, BitmapIntegerIndexNeedsNoRawData) {
    Load();
    storage::FileManagerContext index_context(
        storage::RemoteChunkManagerSingleton::GetInstance()
            .GetRemoteChunkManager());
    const int64_t values[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const bool valid[] = {
        true, false, true, true, true, false, true, true, true, false};
    auto loaded = BuildTestScalarIndex<int64_t>(
        index::families::kBitmap, 10, values, valid);
    segcore::LoadIndexInfo info;
    info.field_id = value_.get();
    info.index_params = GenIndexParams(loaded.get(), index::families::kBitmap);
    SetTestIndexMetadata(info, *loaded, index::families::kBitmap);
    info.index = CreateTestIndex("candidate-bitmap", std::move(loaded));
    segment_->LoadIndex(info);
    segment_->DropFieldData(value_);
    auto query = Context(true);
    ExecContext context(query.get());
    auto candidate = CandidateFilter::TryCreate(AtLeast(4), &context);
    ASSERT_TRUE(candidate);
    auto values_out = candidate->Eval(FixedVector<int32_t>{8, 5, 3, 8});
    EXPECT_TRUE(values_out[0]);
    EXPECT_FALSE(values_out[1]);
    EXPECT_FALSE(values_out[2]);
    EXPECT_TRUE(values_out[3]);
}

TEST_F(FilterBitsOperatorTest,
       IndexOnlyDoesNotRequireRawColumnAndRunsWholeSegment) {
    Load();
    LoadSortedIndex();
    segment_->DropFieldData(value_);
    for (int64_t count : {int64_t(10), int64_t(7)}) {
        FilterRun run(Context(true, 2, count), AtLeast(3));
        FilterBitsNodeTestAccess::ExpectRoute(run.node, true);
        FilterBitsNodeTestAccess::ExpectWholeIndex(run.node);
        std::vector<bool> expected{
            true, true, true, false, false, true, false, false, false, true};
        expected.resize(count);
        ExpectOutput(run, expected);
    }
}

TEST_F(FilterBitsOperatorTest, FullIndexNotOrKeepsNullSemantics) {
    Load();
    LoadSortedIndex();
    segment_->DropFieldData(value_);
    auto predicate = std::make_shared<expr::LogicalUnaryExpr>(
        expr::LogicalUnaryExpr::OpType::LogicalNot,
        std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::Or, AtLeast(3), AtLeast(8)));
    FilterRun run(Context(true, 2), predicate);
    FilterBitsNodeTestAccess::ExpectWholeIndex(run.node);
    ExpectOutput(
        run, {false, true, false, true, true, true, true, true, true, true});
}

TEST_F(FilterBitsOperatorTest, CapturedIndexResourceOutlivesSegment) {
    Load();
    LoadSortedIndex();
    segment_->DropFieldData(value_);
    auto snapshot = segment_->CaptureReadSnapshot();
    auto execution = FilterBitsExecution::TryCreate(
        AtLeast(3),
        nullptr,
        snapshot,
        10,
        2,
        std::make_shared<segcore::reader::ReadContext>());
    ASSERT_TRUE(execution);
    snapshot.reset();
    segment_.reset();
    auto output = execution->GetOutput();
    auto values = ReadOnlyVectorView::CopyOf(output->child(0));
    const std::vector<bool> excluded{
        true, true, true, false, false, true, false, false, false, true};
    for (size_t i = 0; i < excluded.size(); ++i)
        EXPECT_EQ(values.ValueAt<bool>(i), excluded[i]);
}

TEST_F(FilterBitsOperatorTest, RawNullableTailAndVisiblePrefixMatchLegacy) {
    Load();
    for (int64_t rows : {7, 10}) {
        for (int64_t batch : {1, 3, 4, 16}) {
            std::vector<bool> expected(rows);
            for (int64_t row = 0; row < rows; ++row)
                expected[row] = row < 3 || row % 4 == 1;
            for (bool scalar : {false, true}) {
                SCOPED_TRACE(::testing::Message()
                             << rows << "/" << batch << "/" << scalar);
                FilterRun run(Context(scalar, batch, rows), AtLeast(3));
                FilterBitsNodeTestAccess::ExpectRoute(run.node, scalar);
                ExpectOutput(run, expected);
            }
        }
    }
}

TEST_F(FilterBitsOperatorTest, EntityTTLUsesPhysicalClockAndNullNeverExpires) {
    Load(true);
    const int64_t expiry[] = {99, 0, 100, 101, 99, 0, 101, 100, 101, 0};
    for (int64_t now : {99, 100, 101}) {
        std::vector<bool> expected(10);
        for (size_t row = 0; row < expected.size(); ++row)
            expected[row] = row % 4 != 1 && expiry[row] <= now;
        for (bool scalar : {false, true}) {
            // A constant predicate isolates TTL (including NULL rows) while
            // exercising the real CreateTTLFieldFilterExpression factory.
            FilterRun run(Context(scalar, 3, 10, true, now),
                          std::make_shared<expr::AlwaysTrueExpr>());
            FilterBitsNodeTestAccess::ExpectRoute(run.node, scalar);
            ExpectOutput(run, expected);
        }
    }
}

TEST_F(FilterBitsOperatorTest, EmptyDomainEmitsNothing) {
    Load();
    for (bool scalar : {false, true}) {
        FilterRun run(Context(scalar, 3, 0), AtLeast(3));
        FilterBitsNodeTestAccess::ExpectRoute(run.node, scalar);
        EXPECT_TRUE(run.node.IsFinished());
        EXPECT_EQ(run.node.GetOutput(), nullptr);
        EXPECT_EQ(FilterBitsNodeTestAccess::ProcessedRows(run.node), 0);
        run.node.Close();
    }
}

TEST_F(FilterBitsOperatorTest, CompoundThreeValuedPredicateMatchesLegacy) {
    Load();
    auto below_seven = std::make_shared<expr::LogicalUnaryExpr>(
        expr::LogicalUnaryExpr::OpType::LogicalNot, AtLeast(7));
    auto between = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::And, AtLeast(3), below_seven);
    auto is_null = std::make_shared<expr::NullExpr>(
        expr::ColumnInfo(value_, DataType::INT64, {}, true),
        proto::plan::NullExpr_NullOp_IsNull);
    auto predicate = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::Or, between, is_null);
    for (bool scalar : {false, true}) {
        FilterRun run(Context(scalar), predicate);
        FilterBitsNodeTestAccess::ExpectRoute(run.node, scalar);
        ExpectOutput(
            run,
            {true, false, true, false, false, false, false, true, true, false});
    }
}

TEST_F(FilterBitsOperatorTest, EntityTTLAndUserPredicateAreBothApplied) {
    Load(true);
    for (bool scalar : {false, true}) {
        FilterRun run(Context(scalar), AtLeast(3));
        FilterBitsNodeTestAccess::ExpectRoute(run.node, scalar);
        ExpectOutput(
            run,
            {true, true, true, false, true, true, false, true, false, true});
    }
}

TEST_F(FilterBitsOperatorTest, ConstantPredicatesMatchLegacy) {
    Load();
    for (bool value : {false, true}) {
        expr::TypedExprPtr predicate = std::make_shared<expr::AlwaysTrueExpr>();
        if (!value)
            predicate = std::make_shared<expr::LogicalUnaryExpr>(
                expr::LogicalUnaryExpr::OpType::LogicalNot, predicate);
        for (bool scalar : {false, true}) {
            FilterRun run(Context(scalar), predicate);
            FilterBitsNodeTestAccess::ExpectRoute(run.node, scalar);
            ExpectOutput(run, std::vector<bool>(10, !value));
        }
    }
}

TEST_F(FilterBitsOperatorTest, MissingSnapshotSelectsLegacyBeforeExecution) {
    Load();
    FilterRun run(Context(true, 3, 10, false), AtLeast(3));
    FilterBitsNodeTestAccess::ExpectRoute(run.node, false);
    ExpectOutput(
        run, {true, true, true, false, false, true, false, false, false, true});
}

TEST_F(FilterBitsOperatorTest, ScalarRouteDoesNotUseLegacyResultCache) {
    Load();
    auto context = Context(true);
    context->set_enable_expr_cache(true);
    FilterRun run(context, AtLeast(3));
    FilterBitsNodeTestAccess::ExpectRoute(run.node, true);
    ExpectOutput(
        run, {true, true, true, false, false, true, false, false, false, true});
}

TEST_F(FilterBitsOperatorTest, RequestCancellationDoesNotPublishOrFallback) {
    Load();
    for (bool scalar : {false, true}) {
        folly::CancellationSource source;
        OpContext op(source.getToken());
        auto context = Context(scalar);
        context->set_op_context(&op);
        FilterRun run(context, AtLeast(3));
        FilterBitsNodeTestAccess::ExpectRoute(run.node, scalar);
        source.requestCancellation();
        RowVectorPtr output;
        EXPECT_THROW(output = run.node.GetOutput(), folly::FutureCancellation);
        EXPECT_EQ(output, nullptr);
        EXPECT_EQ(FilterBitsNodeTestAccess::ProcessedRows(run.node), 0);
        EXPECT_FALSE(run.node.IsFinished());
        FilterBitsNodeTestAccess::ExpectRoute(run.node, scalar);
        run.node.Close();
    }
}

TEST_F(FilterBitsOperatorTest, CloseBeforeExecutionReleasesScalarAndFinishes) {
    Load();
    FilterRun run(Context(true), AtLeast(3));
    FilterBitsNodeTestAccess::ExpectRoute(run.node, true);
    run.node.Close();
    EXPECT_TRUE(run.node.IsFinished());
    EXPECT_EQ(run.node.GetOutput(), nullptr);
    EXPECT_NO_THROW(run.node.Close());
}

TEST_F(FilterBitsOperatorTest, ReadFailurePreservesCodeAndDoesNotFallback) {
    Load();
    for (auto code : {FileReadFailed, DataFormatBroken}) {
        auto context = Context(true);
        auto column = Observe(context);
        ASSERT_NE(column, nullptr);
        column->before_read = [code](size_t reads) {
            if (reads == 2)
                ThrowInfo(code, "injected FilterBits scan failure");
        };
        FilterRun run(context, AtLeast(3));
        FilterBitsNodeTestAccess::ExpectRoute(run.node, true);
        RowVectorPtr output;
        try {
            output = run.node.GetOutput();
            FAIL() << "scan error was swallowed";
        } catch (const SegcoreError& error) {
            EXPECT_EQ(error.get_error_code(), code);
        }
        EXPECT_EQ(column->starts, (std::vector<int64_t>{0, 3}));
        EXPECT_EQ(output, nullptr);
        EXPECT_EQ(FilterBitsNodeTestAccess::ProcessedRows(run.node), 0);
        EXPECT_FALSE(run.node.IsFinished());
        FilterBitsNodeTestAccess::ExpectRoute(run.node, true);

        // Explicit caller retry, not automatic legacy fallback; start at zero.
        column->before_read = {};
        ExpectOutput(
            run,
            {true, true, true, false, false, true, false, false, false, true});
        EXPECT_EQ(column->starts, (std::vector<int64_t>{0, 3, 0, 3, 6, 9}));
    }
}

TEST_F(FilterBitsOperatorTest, MidExecutionCancellationDoesNotPublish) {
    Load();
    folly::CancellationSource source;
    OpContext op(source.getToken());
    auto context = Context(true);
    context->set_op_context(&op);
    auto column = Observe(context);
    ASSERT_NE(column, nullptr);
    column->before_read = [&](size_t reads) {
        if (reads == 2)
            source.requestCancellation();
    };
    FilterRun run(context, AtLeast(3));
    FilterBitsNodeTestAccess::ExpectRoute(run.node, true);
    RowVectorPtr output;
    EXPECT_THROW(output = run.node.GetOutput(), folly::FutureCancellation);
    EXPECT_EQ(column->starts, (std::vector<int64_t>{0, 3}));
    EXPECT_EQ(output, nullptr);
    EXPECT_EQ(FilterBitsNodeTestAccess::ProcessedRows(run.node), 0);
    EXPECT_FALSE(run.node.IsFinished());
    FilterBitsNodeTestAccess::ExpectRoute(run.node, true);
    run.node.Close();
    EXPECT_TRUE(run.node.IsFinished());
}

TEST_F(FilterBitsOperatorTest, MidExecutionAllocationFailureDoesNotPublish) {
    Load();
    auto context = Context(true);
    auto column = Observe(context);
    ASSERT_NE(column, nullptr);
    column->before_read = [](size_t reads) {
        if (reads == 2)
            throw std::bad_alloc();
    };
    FilterRun run(context, AtLeast(3));
    RowVectorPtr output;
    EXPECT_THROW(output = run.node.GetOutput(), std::bad_alloc);
    EXPECT_EQ(column->starts, (std::vector<int64_t>{0, 3}));
    EXPECT_EQ(output, nullptr);
    EXPECT_EQ(FilterBitsNodeTestAccess::ProcessedRows(run.node), 0);
    EXPECT_FALSE(run.node.IsFinished());
    FilterBitsNodeTestAccess::ExpectRoute(run.node, true);
    run.node.Close();
}

TEST_F(FilterBitsOperatorTest, SnapshotFailureIsNotACapabilityMiss) {
    Load();
    auto context = Context(true);
    auto column = Observe(context);
    ASSERT_NE(column, nullptr);
    auto snapshot = std::dynamic_pointer_cast<const ObservedFilterSnapshot>(
        context->get_read_snapshot());
    ASSERT_NE(snapshot, nullptr);
    // The fixture owns a mutable object; QueryContext intentionally exposes a
    // const view. Set the injection before constructing the real operator.
    std::const_pointer_cast<ObservedFilterSnapshot>(snapshot)->fail_lookup =
        true;
    try {
        FilterRun run(context, AtLeast(3));
        FAIL() << "snapshot error was treated as a fallback";
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), FieldNotLoaded);
    }
    EXPECT_TRUE(column->starts.empty());
}

}  // namespace
}  // namespace milvus::exec
