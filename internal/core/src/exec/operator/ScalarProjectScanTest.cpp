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
#include "exec/operator/ScalarProjectNode.h"
#include "exec/operator/MvccNode.h"
#include "common/VectorAccess.h"
#include "query/ExecPlanNodeVisitor.h"
#include "test_utils/storage_test_utils.h"
#include "test_utils/cachinglayer_test_utils.h"

namespace milvus::exec {
namespace {
namespace logical = ::milvus::plan;

// Observe real column reads; no replacement expression evaluator.
class ObservedTakeColumn : public ChunkedColumn {
 public:
    ObservedTakeColumn(std::shared_ptr<ChunkedColumn> column,
                       const FieldMeta& field)
        : ChunkedColumn(MetadataSlot(*column), field),
          column_(std::move(column)) {
    }
    TakeResultPtr
    Take(OpContext* ctx, TakeOptions options) const override {
        sizes.push_back(options.offsets.size);
        if (before_read)
            before_read(sizes.size());
        return column_->Take(ctx, std::move(options));
    }
    ScanResult
    Scan(OpContext* ctx, const ScanOptions& options) const override {
        class Cursor : public ScanCursor {
         public:
            Cursor(ScanResult base, const ObservedTakeColumn* observed)
                : base_(std::move(base)), observed_(observed) {
            }
            int64_t
            Position() const override {
                return base_->Position();
            }
            void
            Seek(int64_t position) override {
                base_->Seek(position);
            }
            bool
            Next(int64_t length, ScanReadMode mode, ScanBatch* out) override {
                observed_->sizes.push_back(length);
                ++observed_->scan_calls;
                if (observed_->before_read)
                    observed_->before_read(observed_->sizes.size());
                return base_->Next(length, mode, out);
            }

         private:
            ScanResult base_;
            const ObservedTakeColumn* observed_;
        };
        return std::make_unique<Cursor>(column_->Scan(ctx, options), this);
    }
    mutable std::vector<size_t> sizes;
    mutable size_t scan_calls = 0;
    std::function<void(size_t)> before_read;

 private:
    static std::shared_ptr<cachinglayer::CacheSlot<Chunk>>
    MetadataSlot(const ChunkedColumn& column) {
        std::vector<int64_t> sizes;
        for (int64_t i = 0; i < column.num_chunks(); ++i)
            sizes.push_back(column.chunk_row_nums(i));
        return cachinglayer::Manager::GetInstance().CreateCacheSlot<Chunk>(
            std::make_unique<TestChunkTranslator>(
                sizes,
                "projection-observer",
                std::vector<std::unique_ptr<Chunk>>{}),
            nullptr);
    }
    std::shared_ptr<ChunkedColumn> column_;
};

class ObservedSnapshot : public segcore::SegmentReadSnapshot {
 public:
    ObservedSnapshot(std::shared_ptr<const segcore::SegmentReadSnapshot> base,
                     FieldId field,
                     std::shared_ptr<ObservedTakeColumn> column)
        : base_(std::move(base)), field_(field), column_(std::move(column)) {
    }
    int64_t
    chunk_size(FieldId f,
               int64_t c,
               OpContext* op_ctx = nullptr) const override {
        return base_->chunk_size(f, c);
    }
    int64_t
    num_rows_until_chunk(FieldId f,
                         int64_t c,
                         OpContext* op_ctx = nullptr) const override {
        return base_->num_rows_until_chunk(f, c);
    }
    std::pair<int64_t, int64_t>
    get_chunk_by_offset(FieldId f,
                        int64_t o,
                        OpContext* op_ctx = nullptr) const override {
        return base_->get_chunk_by_offset(f, o);
    }
    int64_t
    num_chunk_data(FieldId f, OpContext* op_ctx = nullptr) const override {
        return base_->num_chunk_data(f);
    }
    int64_t
    get_row_count() const override {
        return base_->get_row_count();
    }
    segcore::OffsetBatchIterator
    MakeOffsetIterator(BitsetTypeView excluded,
                       OpContext* ctx,
                       std::function<void()> check) const override {
        return base_->MakeOffsetIterator(excluded, ctx, std::move(check));
    }
    std::pair<std::shared_ptr<ChunkedColumnInterface>, FieldSkipMetricsView>
    GetDataScanResources(FieldId f) const override {
        auto resources = base_->GetDataScanResources(f);
        if (f == field_)
            resources.first = column_;
        return resources;
    }

 private:
    std::shared_ptr<const segcore::SegmentReadSnapshot> base_;
    FieldId field_;
    std::shared_ptr<ObservedTakeColumn> column_;
};

class ScalarProjectScanTest : public ::testing::Test {
 protected:
    SchemaPtr schema = std::make_shared<Schema>();
    FieldId pk{0}, value{0}, text{0};
    segcore::SegmentGrowingPtr growing;
    std::unique_ptr<segcore::SegmentSealed> sealed;
    const std::vector<int64_t> keys{3, 1, 2, 1, 4, 2, 5, 3};
    void
    SetUp() override {
        pk = schema->AddDebugField("pk", DataType::INT64);
        schema->set_primary_field_id(pk);
        value = schema->AddDebugField("value", DataType::INT64, true);
        text = schema->AddDebugField("text", DataType::VARCHAR, true);
    }
    segcore::GeneratedData
    Data(int64_t begin = 0, int64_t count = 8) {
        auto data = segcore::DataGen(schema, count);
        for (auto& field : *data.raw_->mutable_fields_data()) {
            FieldId id(field.field_id());
            FixedVector<int64_t> numbers(count);
            FixedVector<bool> valid(count, true);
            std::vector<std::string> strings;
            for (int64_t i = 0; i < count; ++i) {
                auto row = begin + i;
                numbers[i] = id == pk ? keys[row % keys.size()] : row;
                valid[i] = id == pk || row != 4;
                strings.push_back(std::string(4096, 'a' + row));
            }
            auto replacement = segcore::CreateDataArrayFrom(
                id == text ? static_cast<const void*>(strings.data())
                           : static_cast<const void*>(numbers.data()),
                valid.data(),
                count,
                (*schema)[id]);
            field.Swap(replacement.get());
        }
        for (int64_t i = 0; i < count; ++i) {
            data.row_ids_[i] = begin + i;
            data.timestamps_[i] = 10 + begin + i;
        }
        return data;
    }
    void
    Load(bool use_growing) {
        auto data = Data();
        if (use_growing) {
            auto config = segcore::SegcoreConfig::default_config();
            config.set_chunk_rows(3);
            growing = segcore::CreateGrowingSegment(
                schema, empty_index_meta, 91, config);
            growing->PreInsert(8);
            growing->Insert(0,
                            8,
                            data.row_ids_.data(),
                            data.timestamps_.data(),
                            data.SharedRaw());
        } else
            sealed = CreateSealedWithFieldDataLoaded(schema, data);
    }
    const segcore::SegmentInternalInterface*
    Segment() {
        if (growing)
            return growing.get();
        return sealed.get();
    }
    std::shared_ptr<QueryContext>
    Context(size_t batch, Timestamp ts = 17, Timestamp ttl = 0) {
        auto* segment = Segment();
        auto context = std::make_shared<QueryContext>(
            "projection-scan",
            segment,
            segment->get_active_count(ts),
            ts,
            ttl,
            0,
            query::PlanOptions{},
            std::make_shared<QueryConfig>(
                std::unordered_map<std::string, std::string>{
                    {QueryConfig::kExprEvalBatchSize, std::to_string(batch)}}));
        context->set_read_snapshot(segment->CaptureReadSnapshot());
        return context;
    }
    std::shared_ptr<logical::ScalarProjectScanNode>
    Plan(bool constant_only = false) {
        proto::plan::GenericValue one;
        one.set_int64_val(1);
        std::vector<expr::TypedExprPtr> expressions;
        std::vector<std::string> names;
        if (constant_only) {
            expressions.push_back(std::make_shared<expr::ValueExpr>(one));
            names = {"one"};
        } else {
            auto column = std::make_shared<expr::ColumnExpr>(
                expr::ColumnInfo(value, DataType::INT64, {}, true));
            expressions = {column,
                           std::make_shared<expr::BinaryArithExpr>(
                               proto::plan::Add,
                               column,
                               std::make_shared<expr::ValueExpr>(one)),
                           std::make_shared<expr::ColumnExpr>(expr::ColumnInfo(
                               text, DataType::VARCHAR, {}, true))};
            names = {"value", "adjusted", "text"};
        }
        return std::make_shared<logical::ScalarProjectScanNode>(
            "scan",
            expressions,
            names,
            std::make_shared<logical::MvccNode>("mvcc"));
    }
    RowVectorPtr
    Execute(std::shared_ptr<QueryContext> context, bool constant_only = false) {
        logical::PlanFragment fragment(Plan(constant_only));
        return query::ExecPlanNodeVisitor::ExecuteTask(fragment, context);
    }
};

TEST_F(ScalarProjectScanTest, GrowingMvccDeleteAndFilteredVersionSelection) {
    Load(true);
    proto::schema::IDs ids;
    ids.mutable_int_id()->add_data(2);
    Timestamp deletion = 16;
    ASSERT_TRUE(growing->Delete(1, &ids, &deletion).ok());
    for (const auto& [ts, expected] :
         std::vector<std::pair<Timestamp, std::vector<int64_t>>>{
             {13, {3, 2, 0}}, {16, {3, 0, 4, 6}}, {17, {3, 7, 4, 6}}}) {
        for (size_t batch : {1, 2, 16}) {
            auto context = Context(batch, ts);
            auto output = Execute(context);
            ASSERT_EQ(output->size(), expected.size());
            auto values = ReadOnlyVectorView::CopyOf(output->child(0));
            for (size_t i = 0; i < expected.size(); ++i) {
                EXPECT_EQ(values.IsValid(i), expected[i] != 4);
                if (expected[i] != 4)
                    EXPECT_EQ(values.ValueAt<int64_t>(i), expected[i]);
            }
            auto constants = Execute(Context(batch, ts), true);
            ASSERT_EQ(
                constants->size(),
                expected.size());  // PK choice also applies without columns
        }
    }
    auto context = Context(1, 13);
    auto task = Task::Create("scan-test", logical::PlanFragment(), 0, context);
    DriverContext driver(task, 0, 0, 0, 0);
    PhyMvccNode mvcc(0, &driver, std::make_shared<logical::MvccNode>("mvcc"));
    auto bits = mvcc.GetOutput();
    auto col = std::dynamic_pointer_cast<ColumnVector>(bits->child(0));
    BitsetTypeView excluded(col->GetRawData(), col->size());
    excluded.set(3, true);
    PhyScalarProjectScanNode scan(1, &driver, Plan());
    scan.AddInput(bits);
    scan.NoMoreInput();
    // New insertions between batches cannot escape the original query prefix.
    auto first = scan.GetOutput();
    EXPECT_EQ(ReadOnlyVectorView::CopyOf(first->child(0)).ValueAt<int64_t>(0),
              1);
    auto tail = Data(8, 2);
    growing->PreInsert(2);
    growing->Insert(
        8, 2, tail.row_ids_.data(), tail.timestamps_.data(), tail.SharedRaw());
    EXPECT_EQ(ReadOnlyVectorView::CopyOf(scan.GetOutput()->child(0))
                  .ValueAt<int64_t>(0),
              2);
    EXPECT_EQ(ReadOnlyVectorView::CopyOf(scan.GetOutput()->child(0))
                  .ValueAt<int64_t>(0),
              0);
    EXPECT_EQ(scan.GetOutput(), nullptr);
    EXPECT_TRUE(scan.IsFinished());
    auto ttl_output = Execute(Context(2, 17, 12));
    ASSERT_EQ(ttl_output->size(), 4);
    auto empty = Execute(Context(2, 9));
    EXPECT_EQ(empty, nullptr);
}

TEST_F(ScalarProjectScanTest,
       SealedReadsAreBoundedAndRetainedBatchesOwnStrings) {
    Load(false);
    auto context = Context(2);
    auto snapshot = context->get_read_snapshot();
    auto raw = std::dynamic_pointer_cast<ChunkedColumn>(
        snapshot->GetDataScanResources(value).first);
    ASSERT_NE(raw, nullptr);
    auto observed = std::make_shared<ObservedTakeColumn>(raw, (*schema)[value]);
    context->set_read_snapshot(
        std::make_shared<ObservedSnapshot>(snapshot, value, observed));
    auto task = Task::Create("scan-test", logical::PlanFragment(), 0, context);
    DriverContext driver(task, 0, 0, 0, 0);
    PhyMvccNode mvcc(0, &driver, std::make_shared<logical::MvccNode>("mvcc"));
    auto input = mvcc.GetOutput();
    PhyScalarProjectScanNode scan(1, &driver, Plan());
    EXPECT_TRUE(observed->sizes.empty());  // binding must not read payloads
    scan.AddInput(input);
    scan.NoMoreInput();
    std::vector<RowVectorPtr> outputs;
    while (auto output = scan.GetOutput()) {
        EXPECT_LE(output->size(), 2);
        outputs.push_back(output);
    }
    ASSERT_EQ(observed->sizes, (std::vector<size_t>{2, 2, 2, 2}));
    ASSERT_EQ(outputs.size(), 4);
    const std::vector<int64_t> expected{
        1, 3, 2, 5, 0, 7, 4, 6};  // sealed preserves duplicate PK rows
    for (size_t i = 0; i < expected.size(); ++i) {
        auto values = ReadOnlyVectorView::CopyOf(outputs[i / 2]->child(1));
        auto strings = ReadOnlyVectorView::CopyOf(outputs[i / 2]->child(2));
        const auto row = expected[i];
        EXPECT_EQ(values.IsValid(i % 2), row != 4);
        if (row != 4) {
            EXPECT_EQ(values.ValueAt<int64_t>(i % 2), row + 1);
            EXPECT_EQ(strings.StringViewAt(i % 2),
                      std::string(4096, 'a' + row));
        }
    }
    EXPECT_TRUE(scan.IsFinished());
    EXPECT_FALSE(scan.NeedInput());
}

TEST_F(ScalarProjectScanTest,
       LaterReadFailureAndCancellationDoNotReturnPartialDriverResult) {
    Load(false);
    // Single-row batches take the Range/Scan route even in nonphysical PK
    // order; two-row batches exercise the original positional Take route.
    for (size_t batch : {1, 2})
        for (bool cancel : {false, true}) {
            auto context = Context(batch);
            folly::CancellationSource cancellation;
            OpContext op(cancellation.getToken());
            context->set_op_context(&op);
            auto snapshot = context->get_read_snapshot();
            auto raw = std::dynamic_pointer_cast<ChunkedColumn>(
                snapshot->GetDataScanResources(value).first);
            auto observed =
                std::make_shared<ObservedTakeColumn>(raw, (*schema)[value]);
            observed->before_read = [&](size_t n) {
                if (n == 2) {
                    if (cancel)
                        cancellation.requestCancellation();
                    else
                        ThrowInfo(FileReadFailed,
                                  "injected later projection read failure");
                }
            };
            context->set_read_snapshot(
                std::make_shared<ObservedSnapshot>(snapshot, value, observed));
            RowVectorPtr output;
            try {
                output = Execute(context);
                FAIL() << "expected later-batch read/cancellation failure";
            } catch (const SegcoreError& error) {
                EXPECT_EQ(error.get_error_code(),
                          cancel ? FollyCancel : FileReadFailed);
            }
            EXPECT_EQ(output, nullptr);
            EXPECT_EQ(observed->sizes.size(), 2);
            EXPECT_EQ(observed->scan_calls, batch == 1 ? 2 : 0);
        }
}

TEST_F(ScalarProjectScanTest,
       RealFilterMvccPipelineChoosesLatestMatchingVersion) {
    Load(true);
    proto::plan::GenericValue bound;
    bound.set_int64_val(3);
    auto predicate = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(value, DataType::INT64, {}, true),
        proto::plan::LessThan,
        bound);
    auto filter =
        std::make_shared<logical::FilterBitsNode>("filter", predicate);
    auto mvcc = std::make_shared<logical::MvccNode>(
        "mvcc", std::vector<logical::PlanNodePtr>{filter});
    auto scan = std::make_shared<logical::ScalarProjectScanNode>(
        "scan",
        Plan()->expressions(),
        std::vector<std::string>{"value", "adjusted", "text"},
        mvcc);
    logical::PlanFragment fragment(scan);
    auto result = query::ExecPlanNodeVisitor::ExecuteTask(fragment, Context(1));
    ASSERT_EQ(result->size(), 3);
    auto values = ReadOnlyVectorView::CopyOf(result->child(0));
    EXPECT_EQ(values.ValueAt<int64_t>(0), 1);
    EXPECT_EQ(values.ValueAt<int64_t>(1), 2);
    EXPECT_EQ(values.ValueAt<int64_t>(2), 0);
}

TEST_F(ScalarProjectScanTest,
       SortedSealedCrossChunkDuplicatesAndTimestampMask) {
    auto first = Data(0, 3), second = Data(3, 5);
    auto set_sorted_keys = [&](segcore::GeneratedData& data, size_t begin) {
        for (auto& field : *data.raw_->mutable_fields_data()) {
            if (field.field_id() != pk.get())
                continue;
            auto* keys = field.mutable_scalars()->mutable_long_data();
            keys->clear_data();
            for (size_t i = 0; i < data.row_ids_.size(); ++i)
                keys->add_data((begin + i) / 2);
        }
    };
    set_sorted_keys(first, 0);
    set_sorted_keys(second, 3);
    sealed = CreateTwoChunkSealed(schema, first, second, true);
    for (size_t batch : {1, 2, 5}) {
        auto output = Execute(Context(batch, 16));
        ASSERT_EQ(output->size(), 7);
        auto values = ReadOnlyVectorView::CopyOf(output->child(0));
        for (size_t i = 0; i < 7; ++i) {
            EXPECT_EQ(values.IsValid(i), i != 4);
            if (i != 4)
                EXPECT_EQ(values.ValueAt<int64_t>(i), i);
        }
    }
}

TEST_F(ScalarProjectScanTest, AllFilteredReadsNothingAndEmptyUpstreamFinishes) {
    Load(false);
    auto context = Context(2);
    auto task = Task::Create("scan-test", logical::PlanFragment(), 0, context);
    DriverContext driver(task, 0, 0, 0, 0);
    PhyScalarProjectScanNode scan(1, &driver, Plan());
    RowVectorPtr input = std::make_shared<RowVector>(
        std::vector<VectorPtr>{std::make_shared<ColumnVector>(
            TargetBitmap(8, true), TargetBitmap(8, true))});
    scan.AddInput(input);
    scan.NoMoreInput();
    EXPECT_EQ(scan.GetOutput(), nullptr);
    EXPECT_TRUE(scan.IsFinished());
    PhyScalarProjectScanNode empty(2, &driver, Plan());
    empty.NoMoreInput();
    EXPECT_TRUE(empty.IsFinished());
    EXPECT_EQ(empty.GetOutput(), nullptr);
}

TEST_F(ScalarProjectScanTest,
       RetrieveAssemblyPreservesNamedSchemaForEmptyAndNonemptyResults) {
    for (bool use_growing : {false, true}) {
        growing.reset();
        sealed.reset();
        Load(use_growing);
        for (int kind :
             {0, 1, 2}) {  // nonempty, all filtered, no visible input
            auto mvcc = std::make_shared<logical::MvccNode>("mvcc");
            if (kind == 1) {
                proto::plan::GenericValue bound;
                bound.set_int64_val(-1);
                auto predicate = std::make_shared<expr::UnaryRangeFilterExpr>(
                    expr::ColumnInfo(value, DataType::INT64, {}, true),
                    proto::plan::LessThan,
                    bound);
                auto filter = std::make_shared<logical::FilterBitsNode>(
                    "filter", predicate);
                mvcc = std::make_shared<logical::MvccNode>(
                    "mvcc", std::vector<logical::PlanNodePtr>{filter});
            }
            proto::plan::GenericValue zero;
            zero.set_int64_val(0);
            auto roots = Plan()->expressions();
            roots.insert(roots.begin(),
                         std::make_shared<expr::UnaryRangeFilterExpr>(
                             expr::ColumnInfo(value, DataType::INT64, {}, true),
                             proto::plan::GreaterThan,
                             zero));
            auto scan = std::make_shared<logical::ScalarProjectScanNode>(
                "scan",
                roots,
                std::vector<std::string>{
                    "positive", "value", "adjusted", "text"},
                mvcc);
            for (bool ordered : {false, true}) {
                query::RetrievePlanNode retrieve;
                retrieve.limit_ = segcore::Unlimited;
                retrieve.plannodes_ = scan;
                if (ordered) {
                    retrieve.plannodes_ =
                        std::make_shared<logical::OrderByNode>(
                            "order",
                            std::vector<expr::FieldAccessTypeExprPtr>{
                                std::make_shared<expr::FieldAccessTypeExpr>(
                                    DataType::INT64, "adjusted", value)},
                            std::vector<logical::SortOrder>{{true, false}},
                            100,
                            std::vector<logical::PlanNodePtr>{scan});
                }
                query::ExecPlanNodeVisitor visitor(
                    *Segment(), kind == 2 ? Timestamp(9) : Timestamp(17));
                auto result = visitor.get_retrieve_result(retrieve);
                ASSERT_EQ(result.field_data_.size(), 4);
                EXPECT_EQ(result.total_data_cnt_,
                          kind == 0 ? (use_growing ? 5 : 8) : 0);
                for (size_t i = 0; i < 4; ++i) {
                    const auto& field = result.field_data_[i];
                    EXPECT_EQ(field.field_name(),
                              scan->output_type()->column_name(i));
                    EXPECT_EQ(field.type(),
                              static_cast<proto::schema::DataType>(
                                  scan->output_type()->column_type(i)));
                }
                EXPECT_TRUE(result.field_data_[0].scalars().has_bool_data());
                EXPECT_TRUE(result.field_data_[1].scalars().has_long_data());
                EXPECT_TRUE(result.field_data_[2].scalars().has_long_data());
                EXPECT_TRUE(result.field_data_[3].scalars().has_string_data());
                const auto rows = result.total_data_cnt_;
                EXPECT_EQ(
                    result.field_data_[0].scalars().bool_data().data_size(),
                    rows);
                EXPECT_EQ(
                    result.field_data_[1].scalars().long_data().data_size(),
                    rows);
                EXPECT_EQ(
                    result.field_data_[2].scalars().long_data().data_size(),
                    rows);
                EXPECT_EQ(
                    result.field_data_[3].scalars().string_data().data_size(),
                    rows);
                for (const auto& field : result.field_data_) {
                    EXPECT_EQ(field.scalars().valid_data_size(), rows);
                    if (rows)
                        EXPECT_EQ(
                            std::count(field.scalars().valid_data().begin(),
                                       field.scalars().valid_data().end(),
                                       false),
                            1);
                }
            }
        }
    }
}

TEST_F(ScalarProjectScanTest, PhysicallyEmptySegmentsNeedNoReaders) {
    for (bool use_growing : {false, true}) {
        growing.reset();
        sealed.reset();
        if (use_growing)
            growing = segcore::CreateGrowingSegment(schema, empty_index_meta);
        else
            sealed = segcore::CreateSealedSegment(schema, empty_index_meta);
        query::RetrievePlanNode retrieve;
        retrieve.limit_ = segcore::Unlimited;
        retrieve.plannodes_ = Plan();
        query::ExecPlanNodeVisitor visitor(*Segment(), Timestamp(17));
        auto result = visitor.get_retrieve_result(retrieve);
        EXPECT_EQ(result.total_data_cnt_, 0);
        ASSERT_EQ(result.field_data_.size(), 3);
        EXPECT_EQ(result.field_data_[0].field_name(), "value");
        EXPECT_EQ(result.field_data_[1].field_name(), "adjusted");
        EXPECT_EQ(result.field_data_[2].field_name(), "text");
        EXPECT_EQ(result.field_data_[0].type(), proto::schema::Int64);
        EXPECT_EQ(result.field_data_[2].type(), proto::schema::VarChar);
        EXPECT_EQ(result.field_data_[0].scalars().long_data().data_size(), 0);
        EXPECT_EQ(result.field_data_[2].scalars().string_data().data_size(), 0);
    }
}

}  // namespace
}  // namespace milvus::exec
