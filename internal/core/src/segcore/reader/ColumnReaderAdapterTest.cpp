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
#include <cstring>
#include "common/init_c.h"
#include "segcore/SegmentInterface.h"
#include "exec/QueryContext.h"
#include "exec/operator/FilterBitsExecution.h"
#include "exec/access/ReaderInput.h"
#include "exec/access/ScanNode.h"
#include "exec/access/PositionNode.h"
#include "exec/plan/ExprCompiler.h"
#include "exec/plan/ExecutionPreparer.h"
#include "segcore/reader/ReaderTestUtils.h"
#include "mmap/ChunkedColumn.h"
#include "mmap/VirtualPKChunkedColumn.h"
#include "segcore/reader/ColumnReaderAdapter.h"

namespace milvus::segcore::reader {
namespace {
using namespace milvus::exec;
constexpr auto kValues = ReadProjection::ValuesAndValidity;
constexpr auto kValidity = ReadProjection::ValidityOnly;

// Minimal in-memory Cell source; the real CacheSlot, ChunkedColumn,
// RawScanCursor and RawTakeResult remain on the exercised path.
class ReaderTestTranslator final : public cachinglayer::Translator<Chunk> {
 public:
    ReaderTestTranslator(
        const std::vector<int64_t>& sizes,
        std::vector<std::unique_ptr<Chunk>> chunks,
        std::function<void(cachinglayer::cid_t)> before_load = {})
        : chunks_(std::move(chunks)),
          before_load_(std::move(before_load)),
          meta_(cachinglayer::StorageType::MEMORY,
                cachinglayer::CellIdMappingMode::IDENTICAL,
                cachinglayer::CellDataType::SCALAR_FIELD,
                CacheWarmupPolicy::CacheWarmupPolicy_Disable,
                false) {
        meta_.num_rows_until_chunk_.push_back(0);
        for (auto size : sizes)
            meta_.num_rows_until_chunk_.push_back(
                meta_.num_rows_until_chunk_.back() + size);
        storagev1translator::virtual_chunk_config(
            meta_.num_rows_until_chunk_.back(),
            sizes.size(),
            meta_.num_rows_until_chunk_,
            meta_.virt_chunk_order_,
            meta_.vcid_to_cid_arr_);
    }
    size_t
    num_cells() const override {
        return chunks_.size();
    }
    cachinglayer::cid_t
    cell_id_of(cachinglayer::uid_t id) const override {
        return id;
    }
    std::pair<cachinglayer::ResourceUsage, cachinglayer::ResourceUsage>
    estimated_byte_size_of_cell(cachinglayer::cid_t) const override {
        return {{0, 0}, {0, 0}};
    }
    int64_t
    cells_storage_bytes(
        const std::vector<cachinglayer::cid_t>&) const override {
        return 0;
    }
    const std::string&
    key() const override {
        return key_;
    }
    cachinglayer::Meta*
    meta() override {
        return &meta_;
    }
    std::vector<std::pair<cachinglayer::cid_t, std::unique_ptr<Chunk>>>
    get_cells(OpContext*,
              const std::vector<cachinglayer::cid_t>& ids) override {
        std::vector<std::pair<cachinglayer::cid_t, std::unique_ptr<Chunk>>>
            result;
        // Fail before consuming fixture payload, allowing explicit retry.
        for (auto id : ids)
            if (before_load_)
                before_load_(id);
        for (auto id : ids) {
            AssertInfo(id < chunks_.size() && chunks_[id],
                       "reader fixture Cell already consumed");
            result.emplace_back(id, std::move(chunks_[id]));
        }
        return result;
    }

 private:
    std::vector<std::unique_ptr<Chunk>> chunks_;
    std::function<void(cachinglayer::cid_t)> before_load_;
    storagev1translator::CTMeta meta_;
    std::string key_ = "expr_reader_test";
};

struct Probe {
    int scans = 0;
    int takes = 0;
    int nexts = 0;
    int pins = 0;
    int fail_after = -1;
    int fault = 0;  // 1 EOF, 2 zero progress, 3 wrong start, 4 skip, 5 oversize
    bool no_scan = false;
    bool no_take = false;
    bool fail_take_pin = false;
    bool fail_load = false;
    std::vector<int64_t> offsets;
    std::vector<const void*> take_offset_buffers;
    std::vector<std::weak_ptr<void>> scan_owners;
    std::vector<std::pair<int64_t, const void*>> scan_payloads;
};

class ObservedCursor final : public ScanCursor {
 public:
    ObservedCursor(ScanResult inner, std::shared_ptr<Probe> probe)
        : inner_(std::move(inner)), probe_(std::move(probe)) {
    }
    int64_t
    Position() const override {
        return inner_->Position();
    }
    void
    Seek(int64_t position) override {
        inner_->Seek(position);
    }
    bool
    Next(int64_t length, ScanReadMode mode, ScanBatch* out) override {
        if (probe_->fail_after >= 0 && probe_->nexts == probe_->fail_after)
            ThrowInfo(DataFormatBroken, "injected truncated Cell");
        ++probe_->nexts;
        if (probe_->fault == 1)
            return false;
        auto present = inner_->Next(length, mode, out);
        if (present && mode == ScanReadMode::DataAndValidity) {
            probe_->scan_owners.push_back(out->owner);
            for (int64_t i = 0; i < out->size; ++i) {
                if (out->validity && !out->validity[i])
                    continue;
                const void* address = nullptr;
                switch (out->values.target_type) {
                    case TargetType::Int64:
                        address = out->values.data_as<int64_t>() + i;
                        break;
                    case TargetType::StringView:
                        address =
                            out->values.data_as<std::string_view>()[i].data();
                        break;
                    case TargetType::Json:
                        address = out->values.data_as<Json>()[i].data().data();
                        break;
                    default:
                        break;
                }
                probe_->scan_payloads.emplace_back(out->row_id_start + i,
                                                   address);
            }
        }
        if (probe_->fault == 6)
            out->owner.reset();
        if (probe_->fault == 2)
            out->size = 0;
        if (probe_->fault == 3)
            ++out->row_id_start;
        if (probe_->fault == 4)
            out->data_skipped = true;
        if (probe_->fault == 5)
            out->size = length + 1;
        return present;
    }

 private:
    ScanResult inner_;
    std::shared_ptr<Probe> probe_;
};

template <typename Base>
class ObservedColumn final : public Base {
 public:
    ObservedColumn(std::shared_ptr<CacheSlot<Chunk>> slot,
                   const FieldMeta& meta,
                   std::shared_ptr<Probe> probe)
        : Base(std::move(slot), meta), probe_(std::move(probe)) {
    }
    ScanResult
    Scan(OpContext* ctx, const ScanOptions& options) const override {
        ++probe_->scans;
        EXPECT_FALSE(options.prefetch);
        EXPECT_EQ(options.filter, nullptr);
        if (probe_->no_scan)
            return nullptr;
        return std::make_unique<ObservedCursor>(Base::Scan(ctx, options),
                                                probe_);
    }
    TakeResultPtr
    Take(OpContext* ctx, TakeOptions options) const override {
        ++probe_->takes;
        EXPECT_EQ(options.filter, nullptr);
        EXPECT_EQ(options.offsets.element_type, OffsetElementType::Int32);
        probe_->take_offset_buffers.push_back(options.offsets.data);
        probe_->offsets.clear();
        for (int64_t i = 0; i < options.offsets.size; ++i)
            probe_->offsets.push_back(options.offsets[i]);
        if (probe_->no_take)
            return nullptr;
        return Base::Take(ctx, options);
    }
    PinWrapper<Chunk*>
    GetChunk(OpContext* ctx, int64_t id) const override {
        ++probe_->pins;
        return Base::GetChunk(ctx, id);
    }

 protected:
    typename Base::TakeCellPin
    MakeTakeCellPin(OpContext* ctx) const override {
        auto pin = Base::MakeTakeCellPin(ctx);
        return [pin = std::move(pin), probe = probe_](int64_t id) {
            if (probe->fail_take_pin && id == 1)
                ThrowInfo(FileReadFailed, "injected Take Cell read failure");
            return pin(id);
        };
    }

 private:
    std::shared_ptr<Probe> probe_;
};

struct Fixture {
    // Buffers outlive Column/cache pins, including a reader retained by a batch.
    std::vector<std::vector<char>> buffers;
    std::shared_ptr<Probe> probe = std::make_shared<Probe>();
    std::shared_ptr<ChunkedColumnInterface> column;
};

std::shared_ptr<ColumnReaderAdapter>
MakeReader(DataType type = DataType::INT64,
           bool nullable = true,
           std::vector<int64_t> sizes = {3, 4, 3},
           std::shared_ptr<Probe>* probe = nullptr,
           int64_t visible = 10,
           std::shared_ptr<ChunkedColumnInterface>* retained_column = nullptr) {
    auto fixture = std::make_shared<Fixture>();
    if (probe)
        *probe = fixture->probe;
    std::vector<std::unique_ptr<Chunk>> chunks;
    fixture->buffers.resize(sizes.size());
    int64_t start = 0;
    for (size_t c = 0; c < sizes.size(); ++c) {
        const auto count = sizes[c];
        const size_t bitmap = nullable ? (count + 7) / 8 : 0;
        auto& buffer = fixture->buffers[c];
        if (type == DataType::VARCHAR || type == DataType::TEXT ||
            type == DataType::JSON) {
            std::vector<std::string> values;
            size_t bytes = bitmap + sizeof(uint32_t) * (count + 1);
            for (int64_t i = 0; i < count; ++i) {
                values.push_back(type == DataType::JSON
                                     ? "{\"city\":\"北京\",\"n\":" +
                                           std::to_string(start + i) + "}"
                                     : "北京-" + std::to_string(start + i) +
                                           std::string(100, 'x'));
                bytes += values.back().size();
            }
            buffer.resize(bytes + 64, 0);
            size_t offset = bitmap + sizeof(uint32_t) * (count + 1);
            for (int64_t i = 0; i <= count; ++i) {
                uint32_t encoded = offset;
                std::memcpy(buffer.data() + bitmap + i * sizeof(uint32_t),
                            &encoded,
                            sizeof(encoded));
                if (i < count) {
                    std::memcpy(buffer.data() + offset,
                                values[i].data(),
                                values[i].size());
                    offset += values[i].size();
                }
            }
        } else {
            size_t width = type == DataType::BOOL || type == DataType::INT8 ? 1
                           : type == DataType::INT16                        ? 2
                           : type == DataType::INT32 || type == DataType::FLOAT
                               ? 4
                               : 8;
            buffer.resize(bitmap + count * width, 0);
            for (int64_t i = 0; i < count; ++i) {
                const int64_t value = start + i;
                auto* dest = buffer.data() + bitmap + i * width;
                if (type == DataType::FLOAT) {
                    float v = value + .5;
                    std::memcpy(dest, &v, width);
                } else if (type == DataType::DOUBLE) {
                    double v = value + .5;
                    std::memcpy(dest, &v, width);
                } else if (type == DataType::BOOL) {
                    bool v = value % 2;
                    std::memcpy(dest, &v, width);
                } else
                    std::memcpy(dest, &value, width);
            }
        }
        if (nullable) {
            for (int64_t i = 0; i < count; ++i)
                if ((start + i) % 4 != 1)
                    buffer[i / 8] |= 1 << (i % 8);
        }
        auto guard = std::make_shared<ChunkMmapGuard>(nullptr, 0, "");
        if (type == DataType::VARCHAR || type == DataType::TEXT ||
            type == DataType::JSON)
            chunks.push_back(std::make_unique<StringChunk>(
                count, buffer.data(), buffer.size(), nullable, guard));
        else {
            size_t width = (buffer.size() - bitmap) / count;
            chunks.push_back(std::make_unique<FixedWidthChunk>(count,
                                                               1,
                                                               buffer.data(),
                                                               buffer.size(),
                                                               width,
                                                               nullable,
                                                               guard));
        }
        start += count;
    }
    auto translator = std::make_unique<ReaderTestTranslator>(
        sizes,
        std::move(chunks),
        [probe = fixture->probe](cachinglayer::cid_t id) {
            if (probe->fail_load && id == 1)
                ThrowInfo(FileReadFailed, "injected translator read failure");
        });
    auto slot = cachinglayer::Manager::GetInstance().CreateCacheSlot<Chunk>(
        std::move(translator), nullptr);
    FieldMeta meta(
        FieldName("field"), FieldId(100), type, nullable, std::nullopt);
    if (type == DataType::VARCHAR || type == DataType::TEXT)
        fixture->column = std::make_shared<
            ObservedColumn<ChunkedVariableColumn<std::string>>>(
            slot, meta, fixture->probe);
    else if (type == DataType::JSON)
        fixture->column =
            std::make_shared<ObservedColumn<ChunkedVariableColumn<Json>>>(
                slot, meta, fixture->probe);
    else
        fixture->column = std::make_shared<ObservedColumn<ChunkedColumn>>(
            slot, meta, fixture->probe);
    // Aliasing ownership pins generation AND test backing payload.
    std::shared_ptr<const ChunkedColumnInterface> column(fixture,
                                                         fixture->column.get());
    if (retained_column)
        *retained_column = std::shared_ptr<ChunkedColumnInterface>(
            fixture, fixture->column.get());
    return std::make_shared<ColumnReaderAdapter>(
        std::move(column), type, visible);
}

std::shared_ptr<ReadContext>
Context() {
    return std::make_shared<ReadContext>();
}

std::unique_ptr<BatchInput>
Batch(std::shared_ptr<const ColumnReader> reader,
      ReadBatchBinding binding,
      std::shared_ptr<ReadContext> context = Context()) {
    auto source = access::PrepareReaderInput(reader, context);
    auto loaders = source.make_loaders(binding);
    auto input = std::make_unique<BatchInput>(
        std::make_shared<RowType>(std::vector<std::string>{"x"},
                                  std::vector<DataType>{reader->type()}),
        binding.size());
    input->BindLoader(
        0, std::move(loaders.values), std::move(loaders.validity));
    return input;
}

// Only the snapshot lookup is a fixture. Reads below still use the actual
// ChunkedColumn, CacheSlot, cursor, Reader adapter and compiled expression.
class FilterSnapshot final : public SegmentReadSnapshot {
 public:
    std::shared_ptr<ChunkedColumnInterface> column;
    mutable int lookups = 0;
    bool fail_lookup = false;
    bool ready = true;
    int64_t rows = 10;
    int64_t
    chunk_size(FieldId, int64_t c, OpContext* op_ctx = nullptr) const override {
        return column->chunk_row_nums(c);
    }
    int64_t
    num_rows_until_chunk(FieldId,
                         int64_t c,
                         OpContext* op_ctx = nullptr) const override {
        return column->GetNumRowsUntilChunk(c);
    }
    std::pair<int64_t, int64_t>
    get_chunk_by_offset(FieldId,
                        int64_t offset,
                        OpContext* op_ctx = nullptr) const override {
        return column->GetChunkIDByOffset(offset);
    }
    int64_t
    num_chunk_data(FieldId, OpContext* op_ctx = nullptr) const override {
        return ready && column ? column->num_chunks() : 0;
    }
    int64_t
    get_row_count() const override {
        return rows;
    }
    std::pair<std::shared_ptr<ChunkedColumnInterface>, FieldSkipMetricsView>
    GetDataScanResources(FieldId) const override {
        ++lookups;
        if (fail_lookup)
            ThrowInfo(FieldNotLoaded, "injected snapshot lookup failure");
        return {column, {}};
    }
};

expr::TypedExprPtr
AtLeast(int64_t threshold) {
    proto::plan::GenericValue value;
    value.set_int64_val(threshold);
    return std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(FieldId(100), DataType::INT64, {}, true),
        proto::plan::GreaterEqual,
        value);
}

std::shared_ptr<FilterSnapshot>
Snapshot(std::shared_ptr<Probe>* probe = nullptr) {
    auto snapshot = std::make_shared<FilterSnapshot>();
    MakeReader(DataType::INT64, true, {3, 4, 3}, probe, 10, &snapshot->column);
    return snapshot;
}

void
ExpectExcluded(const RowVectorPtr& output, const std::vector<bool>& expected) {
    ASSERT_TRUE(output);
    ASSERT_EQ(output->childrens().size(), 1);
    auto values = ReadOnlyVectorView::CopyOf(output->child(0));
    ASSERT_EQ(values.size(), expected.size());
    for (size_t row = 0; row < expected.size(); ++row) {
        EXPECT_TRUE(values.IsValid(row));
        EXPECT_EQ(values.ValueAt<bool>(row), expected[row]) << row;
    }
}
}  // namespace

TEST(FilterBitsExecutionTest, QueryConfigIsExplicitlyOptIn) {
    QueryConfig defaults;
    EXPECT_FALSE(defaults.enable_expr_execution_v2());
    QueryConfig enabled(std::unordered_map<std::string, std::string>{
        {QueryConfig::kEnableExprExecutionV2, "true"},
        {QueryConfig::kExprEvalBatchSize, "3"}});
    EXPECT_TRUE(enabled.enable_expr_execution_v2());
    EXPECT_EQ(enabled.get_expr_batch_size(), 3);
}

TEST(FilterBitsExecutionTest, ServiceDefaultIsCapturedAndCanBeOverridden) {
    struct RestoreDefault {
        bool saved = EXPR_EXECUTION_V2_ENABLED.load();
        ~RestoreDefault() {
            SetDefaultExprExecutionV2Enabled(saved);
        }
    } restore;

    SetDefaultExprExecutionV2Enabled(false);
    QueryConfig before;
    EXPECT_FALSE(before.enable_expr_execution_v2());
    SetDefaultExprExecutionV2Enabled(true);
    QueryConfig enabled;
    QueryConfig with_batch_size(std::unordered_map<std::string, std::string>{
        {QueryConfig::kExprEvalBatchSize, "17"}});
    QueryConfig explicit_off(std::unordered_map<std::string, std::string>{
        {QueryConfig::kEnableExprExecutionV2, "false"}});
    EXPECT_TRUE(enabled.enable_expr_execution_v2());
    EXPECT_TRUE(with_batch_size.enable_expr_execution_v2());
    EXPECT_FALSE(explicit_off.enable_expr_execution_v2());
    EXPECT_FALSE(before.enable_expr_execution_v2());

    SetDefaultExprExecutionV2Enabled(false);
    EXPECT_TRUE(enabled.enable_expr_execution_v2());
    EXPECT_FALSE(QueryConfig().enable_expr_execution_v2());
    QueryConfig explicit_on(std::unordered_map<std::string, std::string>{
        {QueryConfig::kEnableExprExecutionV2, "true"}});
    EXPECT_TRUE(explicit_on.enable_expr_execution_v2());
}

TEST(FilterBitsExecutionTest, ArithmeticAndJsonRangeThroughSealedReaders) {
    proto::plan::GenericValue two, five, eight;
    two.set_int64_val(2);
    five.set_int64_val(5);
    eight.set_int64_val(8);
    auto snapshot = Snapshot();
    auto arithmetic = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
        expr::ColumnInfo(FieldId(100), DataType::INT64, {}, true),
        proto::plan::LessThan,
        proto::plan::Add,
        five,
        two);
    auto execution = FilterBitsExecution::TryCreate(
        arithmetic, nullptr, snapshot, 10, 3, Context());
    ASSERT_TRUE(execution);
    ExpectExcluded(
        execution->GetOutput(),
        {false, true, false, true, true, true, true, true, true, true});
    auto json_snapshot = std::make_shared<FilterSnapshot>();
    MakeReader(
        DataType::JSON, true, {3, 4, 3}, nullptr, 10, &json_snapshot->column);
    auto range = std::make_shared<expr::BinaryRangeFilterExpr>(
        expr::ColumnInfo(FieldId(100), DataType::JSON, {"n"}, true),
        two,
        eight,
        false,
        false);
    auto negated = std::make_shared<expr::LogicalUnaryExpr>(
        expr::LogicalUnaryExpr::OpType::LogicalNot, range);
    execution = FilterBitsExecution::TryCreate(
        negated, nullptr, json_snapshot, 10, 4, Context());
    ASSERT_TRUE(execution);
    ExpectExcluded(
        execution->GetOutput(),
        {false, true, false, true, true, true, true, true, false, true});
    EXPECT_EQ(json_snapshot->lookups, 1);
}

TEST(FilterBitsExecutionTest, ConstantPredicatesDoNotResolveColumns) {
    auto snapshot = Snapshot();
    for (bool value : {false, true}) {
        proto::plan::GenericValue literal;
        literal.set_bool_val(value);
        auto execution = FilterBitsExecution::TryCreate(
            std::make_shared<expr::ValueExpr>(literal),
            nullptr,
            snapshot,
            10,
            3,
            Context());
        ASSERT_TRUE(execution);
        EXPECT_EQ(snapshot->lookups, 0);
        ExpectExcluded(execution->GetOutput(), std::vector<bool>(10, !value));
    }
}

TEST(FilterBitsExecutionTest, BatchesEmitExcludedFalseAndNullWithFullValidity) {
    std::shared_ptr<Probe> probe;
    auto snapshot = Snapshot(&probe);
    auto execution = FilterBitsExecution::TryCreate(
        AtLeast(3), nullptr, snapshot, 10, 3, Context());
    ASSERT_TRUE(execution);
    EXPECT_FALSE(execution->IsFinished());
    EXPECT_EQ(snapshot->lookups, 1);
    EXPECT_EQ(probe->scans, 0);
    EXPECT_EQ(probe->pins, 0);
    ExpectExcluded(
        execution->GetOutput(),
        {true, true, true, false, false, true, false, false, false, true});
    EXPECT_EQ(probe->scans,
              1);  // One physical cursor, explicit positions per batch.
    EXPECT_EQ(snapshot->lookups,
              1);  // Never re-resolve the live segment per batch.
    EXPECT_TRUE(execution->IsFinished());
    EXPECT_EQ(execution->GetOutput(), nullptr);
}

TEST(FilterBitsExecutionTest, EmptyDomainAndVisiblePrefix) {
    std::shared_ptr<Probe> probe;
    auto snapshot = Snapshot(&probe);
    auto empty = FilterBitsExecution::TryCreate(
        AtLeast(3), nullptr, snapshot, 0, 3, Context());
    ASSERT_TRUE(empty);
    EXPECT_TRUE(empty->IsFinished());
    EXPECT_EQ(empty->GetOutput(), nullptr);
    EXPECT_EQ(probe->scans, 0);
    auto prefix = FilterBitsExecution::TryCreate(
        AtLeast(3), nullptr, snapshot, 7, 3, Context());
    ASSERT_TRUE(prefix);
    ExpectExcluded(prefix->GetOutput(),
                   {true, true, true, false, false, true, false});
    EXPECT_EQ(probe->scans, 1);
}

TEST(FilterBitsExecutionTest, ExplicitTtlIsAppliedAndNullExpiryPasses) {
    auto snapshot = Snapshot();
    auto ttl = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::Or,
        std::make_shared<expr::NullExpr>(
            expr::ColumnInfo(FieldId(100), DataType::INT64, {}, true),
            proto::plan::NullExpr_NullOp_IsNull),
        AtLeast(6));
    auto execution =
        FilterBitsExecution::TryCreate(std::make_shared<expr::AlwaysTrueExpr>(),
                                       ttl,
                                       snapshot,
                                       10,
                                       4,
                                       Context());
    ASSERT_TRUE(execution);
    EXPECT_EQ(snapshot->lookups, 1);
    ExpectExcluded(
        execution->GetOutput(),
        {true, false, true, true, true, false, false, false, false, false});
}

TEST(FilterBitsExecutionTest, SnapshotAndPayloadSurviveOriginalOwnerRelease) {
    auto snapshot = Snapshot();
    std::weak_ptr<FilterSnapshot> weak = snapshot;
    auto execution = FilterBitsExecution::TryCreate(
        AtLeast(0), nullptr, snapshot, 10, 4, Context());
    ASSERT_TRUE(execution);
    snapshot.reset();
    EXPECT_FALSE(weak.expired());
    auto output = execution->GetOutput();
    execution.reset();
    EXPECT_TRUE(weak.expired());
    ExpectExcluded(
        output,
        {false, true, false, false, false, true, false, false, false, true});
}

TEST(FilterBitsExecutionTest,
     CapabilityMissesDoNotReadAndFailuresDoNotFallback) {
    std::shared_ptr<Probe> probe;
    auto snapshot = Snapshot(&probe);
    auto unsupported = std::make_shared<expr::JsonContainsExpr>(
        expr::ColumnInfo(FieldId(100), DataType::JSON),
        static_cast<ContainsType>(0),
        true,
        std::vector<proto::plan::GenericValue>{});
    EXPECT_EQ(FilterBitsExecution::TryCreate(
                  unsupported, nullptr, snapshot, 10, 4, Context()),
              nullptr);
    EXPECT_EQ(snapshot->lookups, 0);
    EXPECT_EQ(FilterBitsExecution::TryCreate(
                  AtLeast(0), nullptr, nullptr, 10, 4, Context()),
              nullptr);
    snapshot->ready = false;
    EXPECT_EQ(FilterBitsExecution::TryCreate(
                  AtLeast(0), nullptr, snapshot, 10, 4, Context()),
              nullptr);
    snapshot->ready = true;
    snapshot->column = std::make_shared<VirtualPKChunkedColumn>(1, 10);
    EXPECT_EQ(FilterBitsExecution::TryCreate(
                  AtLeast(0), nullptr, snapshot, 10, 4, Context()),
              nullptr);
    snapshot->column.reset();
    EXPECT_EQ(FilterBitsExecution::TryCreate(
                  AtLeast(0), nullptr, snapshot, 10, 4, Context()),
              nullptr);
    snapshot->fail_lookup = true;
    try {
        FilterBitsExecution::TryCreate(
            AtLeast(0), nullptr, snapshot, 10, 4, Context());
        FAIL();
    } catch (const SegcoreError& e) {
        EXPECT_EQ(e.get_error_code(), FieldNotLoaded);
    }
    EXPECT_EQ(probe->scans, 0);
    EXPECT_EQ(probe->pins, 0);
}

TEST(FilterBitsExecutionTest, InvalidBindingsAreNotCapabilityMisses) {
    auto snapshot = Snapshot();
    for (auto [count, batch] : std::vector<std::pair<int64_t, int64_t>>{
             {-1, 4}, {11, 4}, {10, 0}, {10, -1}}) {
        try {
            FilterBitsExecution::TryCreate(
                AtLeast(0), nullptr, snapshot, count, batch, Context());
            FAIL();
        } catch (const SegcoreError& e) {
            EXPECT_EQ(e.get_error_code(), UnexpectedError);
        }
    }
}

TEST(FilterBitsExecutionTest,
     MidExecutionFailurePublishesNoProgressAndCanRetry) {
    std::shared_ptr<Probe> probe;
    auto snapshot = Snapshot(&probe);
    auto execution = FilterBitsExecution::TryCreate(
        AtLeast(3), nullptr, snapshot, 10, 3, Context());
    ASSERT_TRUE(execution);
    probe->fail_after = 2;
    try {
        execution->GetOutput();
        FAIL();
    } catch (const SegcoreError& e) {
        EXPECT_EQ(e.get_error_code(), DataFormatBroken);
    }
    EXPECT_FALSE(execution->IsFinished());
    const auto scans_before_retry = probe->scans;
    probe->fail_after = -1;
    ExpectExcluded(
        execution->GetOutput(),
        {true, true, true, false, false, true, false, false, false, true});
    EXPECT_EQ(probe->scans - scans_before_retry,
              1);  // Failed cursor discarded; restart at row zero.
}

TEST(FilterBitsExecutionTest, CancellationBeforeReadAndBeforeCommitKeepsState) {
    std::shared_ptr<Probe> probe;
    auto snapshot = Snapshot(&probe);
    auto context = Context();
    int checkpoints = 0;
    int cancel_at = 1;
    context->checkpoint = [&] {
        if (++checkpoints == cancel_at)
            ThrowInfo(FollyCancel, "injected FilterBits cancellation");
    };
    auto execution = FilterBitsExecution::TryCreate(
        AtLeast(3), nullptr, snapshot, 10, 3, context);
    ASSERT_TRUE(execution);
    EXPECT_THROW(execution->GetOutput(), SegcoreError);
    EXPECT_FALSE(execution->IsFinished());
    EXPECT_EQ(probe->scans, 0);
    cancel_at = 0;
    checkpoints = 0;
    ASSERT_TRUE(execution->GetOutput());
    const auto total_checkpoints = checkpoints;
    execution = FilterBitsExecution::TryCreate(
        AtLeast(3), nullptr, snapshot, 10, 3, context);
    cancel_at = total_checkpoints;
    checkpoints = 0;
    try {
        execution->GetOutput();
        FAIL();
    } catch (const SegcoreError& e) {
        EXPECT_EQ(e.get_error_code(), FollyCancel);
    }
    EXPECT_FALSE(execution->IsFinished());
    cancel_at = 0;
    ExpectExcluded(
        execution->GetOutput(),
        {true, true, true, false, false, true, false, false, false, true});
}

TEST(ColumnReaderAdapterTest,
     RangeCrossesPhysicalCellsAndKeepsLocalCoordinates) {
    std::shared_ptr<Probe> probe;
    auto reader = MakeReader(DataType::INT64, true, {3, 4, 3}, &probe);
    auto result = reader->ReadRange(
        RangeRequest(2, 6, kValues, SelectionView::All(6)), *Context());
    result.Validate(DataType::INT64, SelectionView::All(6), kValues);
    EXPECT_EQ(probe->scans, 1);
    EXPECT_EQ(probe->nexts, 3);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(result.values->IsValid(i), (i + 2) % 4 != 1);
        if (result.values->IsValid(i))
            EXPECT_EQ(result.values->ValueAt<int64_t>(i), i + 2);
    }
}

TEST(ColumnReaderAdapterTest, SparseRangeDoesNotPublishUnselectedRows) {
    auto reader = MakeReader();
    auto rows = OwnedSelection::FromIndices(7, {1, 4, 6});
    auto result =
        reader->ReadRange(RangeRequest(2, 7, kValues, rows.view()), *Context());
    EXPECT_EQ(result.values->ValueAt<int64_t>(1), 3);
    EXPECT_EQ(result.values->ValueAt<int64_t>(4), 6);
    EXPECT_EQ(result.values->ValueAt<int64_t>(6), 8);
    EXPECT_THROW(result.values->IsValid(0), SegcoreError);
    EXPECT_EQ(result.defined.view().selected_count(), 3);
}

TEST(ColumnReaderAdapterTest,
     TakeCompactsSelectionPreservingOrderAndDuplicates) {
    std::shared_ptr<Probe> probe;
    auto reader = MakeReader(DataType::INT64, true, {3, 4, 3}, &probe);
    auto binding = ReadBatchBinding::Offsets({8, 1, 8, 3, 5});
    auto batch = Batch(reader, binding);
    auto rows = OwnedSelection::FromIndices(5, {0, 2, 4});
    auto first = batch->GetColumn(0, rows.view());
    EXPECT_EQ(probe->offsets, (std::vector<int64_t>{8, 8, 5}));
    EXPECT_NE(probe->take_offset_buffers.back(), binding.offsets().data());
    EXPECT_EQ(first.ValueAt<int64_t>(0), 8);
    EXPECT_EQ(first.ValueAt<int64_t>(2), 8);
    EXPECT_FALSE(first.IsValid(4));
    auto all = batch->GetColumn(0, SelectionView::All(5));
    EXPECT_EQ(probe->offsets, (std::vector<int64_t>{1, 3}));
    EXPECT_FALSE(all.IsValid(1));
    EXPECT_EQ(all.ValueAt<int64_t>(3), 3);
    EXPECT_THROW(first.IsValid(3), SegcoreError);
}

TEST(ColumnReaderAdapterTest, OffsetSnapshotSharedThroughMultipleFieldReads) {
    std::shared_ptr<Probe> probe;
    auto reader = MakeReader(DataType::INT64, true, {3, 4, 3}, &probe);
    FixedVector<int32_t> candidate_offsets{8, 1, 8, 3};
    auto binding = ReadBatchBinding::Offsets(std::span<const int32_t>(
        candidate_offsets.data(), candidate_offsets.size()));
    const auto* snapshot_data = binding.offsets().data();
    EXPECT_NE(snapshot_data, candidate_offsets.data());
    auto compiled = exec::plan::CompileExpressions(
        {std::make_shared<expr::ColumnExpr>(
             expr::ColumnInfo(FieldId(100), DataType::INT64, {}, true)),
         std::make_shared<expr::ColumnExpr>(
             expr::ColumnInfo(FieldId(101), DataType::INT64, {}, true))});
    ASSERT_TRUE(compiled.supported());
    auto prepared = exec::plan::Prepare(compiled.plan, [&](const auto&) {
        return access::PrepareReaderInput(reader, Context());
    });
    auto batch = prepared.MakeExecutionBatch(binding);
    // Both field loaders must retain the one immutable snapshot even after
    // the original binding and the caller's candidate buffer are reused.
    binding = ReadBatchBinding::Range(0, 0);
    candidate_offsets.assign(4, 0);
    auto output = batch->Eval(SelectionView::All(4));
    ASSERT_EQ(output.size(), 2);
    EXPECT_EQ(probe->take_offset_buffers,
              (std::vector<const void*>{snapshot_data, snapshot_data}));
    for (const auto& values : output) {
        EXPECT_EQ(values.ValueAt<int64_t>(0), 8);
        EXPECT_FALSE(values.IsValid(1));
        EXPECT_EQ(values.ValueAt<int64_t>(2), 8);
        EXPECT_EQ(values.ValueAt<int64_t>(3), 3);
    }
    batch.reset();
    // Published values do not borrow the offsets or the short-lived request.
    EXPECT_EQ(output[0].ValueAt<int64_t>(2), 8);
}

TEST(ColumnReaderAdapterTest, TakeRequestsShareOwnedOffsetsAndDetachMasks) {
    std::shared_ptr<Probe> probe;
    auto reader = MakeReader(DataType::INT64, true, {3, 4, 3}, &probe);
    auto rows = OwnedSelection::FromIndices(4, {0, 2});
    auto request = [&] {
        std::vector<int32_t> caller{8, 1, 8, 3};
        auto binding = ReadBatchBinding::Offsets(caller);
        TakeRequest result(binding.offsets(), kValues, rows.view());
        EXPECT_EQ(result.offsets.data(), binding.offsets().data());
        caller.assign(4, 0);
        return result;
    }();
    rows.ResetAll(4);
    auto sparse = reader->Take(request, *Context());
    EXPECT_EQ(probe->offsets, (std::vector<int64_t>{8, 8}));
    EXPECT_EQ(sparse.values->ValueAt<int64_t>(2), 8);
    EXPECT_THROW(sparse.values->IsValid(1), SegcoreError);
    TakeRequest all(request.offsets, kValues, SelectionView::All(4));
    EXPECT_EQ(all.offsets.data(), request.offsets.data());
    auto full = reader->Take(all, *Context());
    EXPECT_EQ(probe->take_offset_buffers.back(), all.offsets.data());
    EXPECT_FALSE(full.values->IsValid(1));
    EXPECT_EQ(full.values->ValueAt<int64_t>(3), 3);
}

TEST(ColumnReaderAdapterTest, Int32OffsetsCheckWiderInputsBeforeNarrowing) {
    const int64_t largest = std::numeric_limits<int32_t>::max();
    std::vector<int64_t> wide{largest, 0, largest};
    auto binding = ReadBatchBinding::Offsets(wide);
    EXPECT_EQ(binding.offset(0), largest);
    EXPECT_EQ(binding.offset(2), largest);
    EXPECT_EQ(sizeof(*binding.offsets().data()), sizeof(int32_t));
    TakeRequest request(wide, kValues, SelectionView::All(3));
    wide.assign(3, 7);
    EXPECT_EQ(request.offsets[0], largest);
    for (auto invalid : {-1L,
                         largest + 1,
                         std::numeric_limits<int64_t>::max(),
                         std::numeric_limits<int64_t>::min()}) {
        std::vector<int64_t> bad{0, invalid};
        EXPECT_THROW(ReadBatchBinding::Offsets(bad), SegcoreError);
        EXPECT_THROW(TakeRequest(bad, kValues, SelectionView::All(2)),
                     SegcoreError);
    }
    EXPECT_THROW(ReadBatchBinding::Offsets({-1}), SegcoreError);
    auto empty = ReadBatchBinding::Offsets({});
    EXPECT_FALSE(empty.is_range());
    EXPECT_EQ(empty.offsets().size(), 0);
    TakeRequest empty_request(empty.offsets(), kValues, SelectionView::All(0));
    EXPECT_EQ(empty_request.offsets.data(), nullptr);
    // Range remains arithmetic addressing, not an expanded INT32 array.
    auto range = ReadBatchBinding::Range(largest + 1, 2);
    EXPECT_TRUE(range.is_range());
    EXPECT_EQ(range.offset(1), largest + 2);
    EXPECT_THROW(range.offsets(), SegcoreError);
}

TEST(ColumnReaderAdapterTest, ValidityOnlyAndKnownNullAvoidValueLoads) {
    std::shared_ptr<Probe> probe;
    auto reader = MakeReader(DataType::INT64, true, {3, 4, 3}, &probe);
    auto batch = Batch(reader, ReadBatchBinding::Offsets({1, 2, 5, 7}));
    auto valid = batch->GetValidity(0, SelectionView::All(4));
    EXPECT_EQ(valid.valid.view().selected_count(), 2);
    auto nulls = OwnedSelection::FromIndices(4, {0, 2});
    auto result = batch->GetColumn(0, nulls.view());
    EXPECT_FALSE(result.IsValid(0));
    EXPECT_EQ(probe->takes, 1);
    auto all = batch->GetColumn(0, SelectionView::All(4));
    EXPECT_EQ(probe->offsets, (std::vector<int64_t>{2, 7}));
    EXPECT_EQ(all.ValueAt<int64_t>(3), 7);
}

TEST(ColumnReaderAdapterTest, NonNullableValidityNeedsNoColumnIO) {
    std::shared_ptr<Probe> probe;
    auto reader = MakeReader(DataType::INT64, false, {3, 4, 3}, &probe);
    auto range = reader->ReadRange(
        RangeRequest(3, 4, kValidity, SelectionView::All(4)), *Context());
    auto take = reader->Take(
        TakeRequest({8, 2}, kValidity, SelectionView::All(2)), *Context());
    EXPECT_FALSE(range.values);
    EXPECT_TRUE(range.valid.view().IsAll());
    EXPECT_TRUE(take.valid.view().IsAll());
    EXPECT_EQ(probe->scans + probe->takes + probe->pins, 0);
}

TEST(ColumnReaderAdapterTest, EmptyRequestsDoNotOpenOrCheckContext) {
    std::shared_ptr<Probe> probe;
    auto reader = MakeReader(DataType::INT64, true, {3, 4, 3}, &probe);
    auto context = Context();
    context->checkpoint = [] { FAIL() << "empty read checked context"; };
    access::ScanNode scan(reader, ReadBatchBinding::Range(2, 4), context);
    auto r = scan.Read(SelectionView::None(4), kValues);
    access::PositionNode take(
        reader, ReadBatchBinding::Offsets({3, 2}), context);
    auto t = take.Read(SelectionView::None(2), kValidity);
    reader->ReadRange(RangeRequest(10, 0, kValues, SelectionView::All(0)),
                      *context);
    reader->Take(TakeRequest({}, kValues, SelectionView::All(0)), *context);
    EXPECT_TRUE(r.values->defined_rows().IsNone());
    EXPECT_TRUE(t.defined.view().IsNone());
    EXPECT_EQ(probe->scans + probe->takes, 0);
}

TEST(ColumnReaderAdapterTest, BoundsUseReadableSnapshotAndRequestsOwnInputs) {
    auto reader = MakeReader(DataType::INT64, true, {3, 4, 3}, nullptr, 8);
    EXPECT_THROW(
        reader->ReadRange(RangeRequest(7, 2, kValues, SelectionView::All(2)),
                          *Context()),
        SegcoreError);
    EXPECT_THROW(reader->Take(TakeRequest({8}, kValues, SelectionView::All(1)),
                              *Context()),
                 SegcoreError);
    EXPECT_THROW(RangeRequest(INT64_MAX, 1, kValues, SelectionView::All(1)),
                 SegcoreError);
    EXPECT_THROW(TakeRequest({-1}, kValues, SelectionView::All(1)),
                 SegcoreError);
    EXPECT_THROW(TakeRequest({1}, kValues, SelectionView::All(2)),
                 SegcoreError);
    std::vector<int64_t> offsets{6, 3};
    auto rows = OwnedSelection::FromIndices(2, {0});
    TakeRequest req(offsets, kValues, rows.view());
    offsets[0] = 999;
    rows.ResetAll(2);
    auto result = reader->Take(req, *Context());
    EXPECT_EQ(result.values->ValueAt<int64_t>(0), 6);
    EXPECT_THROW(result.values->IsValid(1), SegcoreError);
}

TEST(ColumnReaderAdapterTest, IndependentBatchesCanReadBackwardAndInterleave) {
    auto reader = MakeReader();
    auto later = Batch(reader, ReadBatchBinding::Range(6, 3));
    auto earlier = Batch(reader, ReadBatchBinding::Range(0, 3));
    auto r = later->GetColumn(0, SelectionView::All(3));
    auto l = earlier->GetColumn(0, SelectionView::All(3));
    EXPECT_EQ(r.ValueAt<int64_t>(0), 6);
    EXPECT_EQ(l.ValueAt<int64_t>(0), 0);
    EXPECT_EQ(later->GetColumn(0, SelectionView::All(3)).ValueAt<int64_t>(2),
              8);
}

TEST(ColumnReaderAdapterTest, StringsSurviveCellSwitchAndReaderDestruction) {
    auto reader = MakeReader(DataType::VARCHAR);
    auto context = Context();
    auto batch =
        Batch(reader, ReadBatchBinding::Offsets({8, 0, 8, 1, 4}), context);
    auto values = batch->GetColumn(0, SelectionView::All(5));
    auto range = reader->ReadRange(
        RangeRequest(2, 6, kValues, SelectionView::All(6)), *context);
    reader.reset();
    batch.reset();
    context.reset();
    EXPECT_EQ(values.StringViewAt(0), "北京-8" + std::string(100, 'x'));
    EXPECT_EQ(values.StringViewAt(1), "北京-0" + std::string(100, 'x'));
    EXPECT_FALSE(values.IsValid(3));
    EXPECT_EQ(range.values->StringViewAt(5), "北京-7" + std::string(100, 'x'));
}

TEST(ColumnReaderAdapterTest, ScalarRepresentationsUseCorrectTargetWidths) {
    for (auto type : {DataType::BOOL,
                      DataType::INT8,
                      DataType::INT16,
                      DataType::INT32,
                      DataType::INT64,
                      DataType::TIMESTAMPTZ,
                      DataType::FLOAT,
                      DataType::DOUBLE}) {
        auto reader = MakeReader(type);
        auto scan = reader->ReadRange(
            RangeRequest(2, 2, kValues, SelectionView::All(2)), *Context());
        auto take = reader->Take(
            TakeRequest({2, 3}, kValues, SelectionView::All(2)), *Context());
        for (const auto* result : {&scan, &take}) {
            EXPECT_EQ(result->values->logical_type(), type);
            switch (type) {
                case DataType::BOOL:
                    EXPECT_TRUE(result->values->ValueAt<bool>(1));
                    break;
                case DataType::INT8:
                    EXPECT_EQ(result->values->ValueAt<int8_t>(1), 3);
                    break;
                case DataType::INT16:
                    EXPECT_EQ(result->values->ValueAt<int16_t>(1), 3);
                    break;
                case DataType::INT32:
                    EXPECT_EQ(result->values->ValueAt<int32_t>(1), 3);
                    break;
                case DataType::FLOAT:
                    EXPECT_EQ(result->values->ValueAt<float>(1), 3.5);
                    break;
                case DataType::DOUBLE:
                    EXPECT_EQ(result->values->ValueAt<double>(1), 3.5);
                    break;
                default:
                    EXPECT_EQ(result->values->ValueAt<int64_t>(1), 3);
                    break;
            }
        }
    }
}

TEST(ColumnReaderAdapterTest, FailedSecondCellDoesNotPublishPartialCoverage) {
    std::shared_ptr<Probe> probe;
    auto reader = MakeReader(DataType::INT64, true, {3, 4, 3}, &probe);
    auto batch = Batch(reader, ReadBatchBinding::Range(0, 8));
    probe->fail_after = 1;
    try {
        batch->GetColumn(0, SelectionView::All(8));
        FAIL();
    } catch (const SegcoreError& e) {
        EXPECT_EQ(e.get_error_code(), DataFormatBroken);
    }
    EXPECT_EQ(probe->scans, 1);  // no automatic retry
    probe->fail_after = -1;
    const auto before = probe->nexts;
    auto result = batch->GetColumn(0, SelectionView::All(8));
    EXPECT_EQ(probe->nexts - before, 3);  // rereads first Cell too
    EXPECT_EQ(result.ValueAt<int64_t>(7), 7);
}

TEST(ColumnReaderAdapterTest, InvalidScanResultsAreErrorsNotNullOrEOFSuccess) {
    for (int fault : {1, 2, 3, 4, 5, 6}) {
        std::shared_ptr<Probe> probe;
        auto reader = MakeReader(DataType::INT64, true, {3, 4, 3}, &probe);
        probe->fault = fault;
        auto batch = Batch(reader, ReadBatchBinding::Range(0, 4));
        EXPECT_THROW(batch->GetColumn(0, SelectionView::All(4)), SegcoreError);
        probe->fault = 0;
        EXPECT_EQ(
            batch->GetColumn(0, SelectionView::All(4)).ValueAt<int64_t>(3), 3);
    }
}

TEST(ColumnReaderAdapterTest, MissingBackendCapabilityDoesNotFallback) {
    std::shared_ptr<Probe> probe;
    auto reader = MakeReader(DataType::INT64, true, {3, 4, 3}, &probe);
    probe->no_scan = true;
    probe->no_take = true;
    for (auto binding :
         {ReadBatchBinding::Range(0, 2), ReadBatchBinding::Offsets({0, 2})}) {
        auto batch = Batch(reader, binding);
        try {
            batch->GetColumn(0, SelectionView::All(2));
            FAIL();
        } catch (const SegcoreError& e) {
            EXPECT_EQ(e.get_error_code(), Unsupported);
        }
    }
    EXPECT_EQ(probe->scans, 1);
    EXPECT_EQ(probe->takes, 1);
}

TEST(ColumnReaderAdapterTest,
     CancellationAndAllocationFailureRemainTransactional) {
    for (bool oom : {false, true}) {
        std::shared_ptr<Probe> probe;
        auto reader = MakeReader(DataType::INT64, true, {3, 4, 3}, &probe);
        auto context = Context();
        context->checkpoint = [probe, oom] {
            if (probe->nexts >= 1) {
                if (oom)
                    throw std::bad_alloc();
                ThrowInfo(FollyCancel, "injected reader cancellation");
            }
        };
        auto batch = Batch(reader, ReadBatchBinding::Range(0, 8), context);
        if (oom)
            EXPECT_THROW(batch->GetColumn(0, SelectionView::All(8)),
                         std::bad_alloc);
        else {
            try {
                batch->GetColumn(0, SelectionView::All(8));
                FAIL();
            } catch (const SegcoreError& e) {
                EXPECT_EQ(e.get_error_code(), FollyCancel);
            }
        }
        for (const auto& owner : probe->scan_owners)
            EXPECT_TRUE(owner.expired());
        context->checkpoint = {};
        EXPECT_EQ(
            batch->GetColumn(0, SelectionView::All(8)).ValueAt<int64_t>(7), 7);
        EXPECT_EQ(probe->scans, 2);
    }
}

TEST(ColumnReaderAdapterTest,
     CompiledRootsReadRealColumnsWithDifferentCellBoundaries) {
    auto left = MakeReader(DataType::INT64, true, {3, 4, 3});
    auto right = MakeReader(DataType::INT64, true, {2, 6, 2});
    auto c1 = expr::ColumnInfo(FieldId(100), DataType::INT64, {}, true);
    auto c2 = expr::ColumnInfo(FieldId(101), DataType::INT64, {}, true);
    auto compare = std::make_shared<expr::CompareExpr>(FieldId(100),
                                                       FieldId(101),
                                                       DataType::INT64,
                                                       DataType::INT64,
                                                       proto::plan::Equal);
    auto compiled = exec::plan::CompileExpressions(
        {compare, std::make_shared<expr::ColumnExpr>(c1)});
    ASSERT_TRUE(compiled.supported());
    auto context = Context();
    auto prepared =
        exec::plan::Prepare(compiled.plan, [&](const InputDependency& dep) {
            return access::PrepareReaderInput(
                dep.field_id == 100 ? left : right, context);
        });
    auto batch = prepared.MakeBatch(ReadBatchBinding::Range(2, 6));
    left.reset();
    right.reset();
    scalar::EvalCtx ctx(*batch);
    auto results =
        prepared.plan()->expressions().Eval(SelectionView::All(6), ctx);
    for (size_t i = 0; i < 6; ++i) {
        if ((i + 2) % 4 == 1)
            EXPECT_FALSE(results[0].IsValid(i));
        else {
            EXPECT_TRUE(results[0].ValueAt<bool>(i));
            EXPECT_EQ(results[1].ValueAt<int64_t>(i), i + 2);
        }
    }
}

TEST(ColumnReaderAdapterTest, NullableValidityScanCrossesCellsWithoutValues) {
    auto reader = MakeReader();
    auto result = reader->ReadRange(
        RangeRequest(2, 6, kValidity, SelectionView::All(6)), *Context());
    EXPECT_FALSE(result.values);
    EXPECT_TRUE(result.defined.view().IsAll());
    for (size_t i = 0; i < 6; ++i)
        EXPECT_EQ(result.valid.view().IsSelected(i), (i + 2) % 4 != 1);
}

TEST(ColumnReaderAdapterTest, LazyTakeFailureLeavesBothCoveragesUnpublished) {
    for (bool validity : {false, true}) {
        std::shared_ptr<Probe> probe;
        auto reader = MakeReader(DataType::INT64, true, {3, 4, 3}, &probe);
        auto batch = Batch(reader, ReadBatchBinding::Offsets({2, 4, 8}));
        probe->fail_take_pin = true;
        try {
            if (validity)
                batch->GetValidity(0, SelectionView::All(3));
            else
                batch->GetColumn(0, SelectionView::All(3));
            FAIL();
        } catch (const SegcoreError& e) {
            EXPECT_EQ(e.get_error_code(), FileReadFailed);
        }
        probe->fail_take_pin = false;
        auto result = batch->GetColumn(0, SelectionView::All(3));
        EXPECT_EQ(probe->offsets, (std::vector<int64_t>{2, 4, 8}));
        EXPECT_EQ(probe->takes, 2);
        EXPECT_EQ(result.ValueAt<int64_t>(0), 2);
    }
}

TEST(ColumnReaderAdapterTest, JsonScanTakeRetainsOwnedPaddedPayload) {
    auto reader = MakeReader(DataType::JSON);
    auto range = reader->ReadRange(
        RangeRequest(2, 6, kValues, SelectionView::All(6)), *Context());
    auto taken = reader->Take(
        TakeRequest({8, 1, 2, 8}, kValues, SelectionView::All(4)), *Context());
    range.Validate(DataType::JSON, SelectionView::All(6), kValues);
    taken.Validate(DataType::JSON, SelectionView::All(4), kValues);
    reader.reset();
    EXPECT_FALSE(taken.values->IsValid(1));
    EXPECT_TRUE(taken.values->ValueAt<Json>(0).exist("/city"));
    EXPECT_EQ(static_cast<std::string_view>(taken.values->ValueAt<Json>(3)),
              "{\"city\":\"北京\",\"n\":8}");
    EXPECT_EQ(static_cast<std::string_view>(range.values->ValueAt<Json>(0)),
              "{\"city\":\"北京\",\"n\":2}");
}

TEST(ColumnReaderAdapterTest, TextScanTakeAndValidityKeepLogicalType) {
    auto reader = MakeReader(DataType::TEXT);
    auto range = reader->ReadRange(
        RangeRequest(2, 6, kValues, SelectionView::All(6)), *Context());
    auto taken = reader->Take(
        TakeRequest({8, 1, 2, 8}, kValues, SelectionView::All(4)), *Context());
    auto valid = reader->Take(
        TakeRequest({8, 1, 2, 8}, kValidity, SelectionView::All(4)),
        *Context());
    reader.reset();
    range.Validate(DataType::TEXT, SelectionView::All(6), kValues);
    taken.Validate(DataType::TEXT, SelectionView::All(4), kValues);
    EXPECT_FALSE(taken.values->IsValid(1));
    EXPECT_FALSE(valid.valid.view().IsSelected(1));
    EXPECT_EQ(taken.values->StringViewAt(0), taken.values->StringViewAt(3));
    EXPECT_EQ(range.values->StringViewAt(0), taken.values->StringViewAt(2));
}

TEST(ColumnReaderAdapterTest, UnmigratedPayloadTypesAreRejectedExplicitly) {
    for (auto type : {DataType::VECTOR_FLOAT}) {
        auto fixture = std::make_shared<Fixture>();
        // Reuse a supported Column's base via a small standalone fixture;
        // rejection must happen at metadata preparation, before reading.
        auto translator = std::make_unique<ReaderTestTranslator>(
            std::vector<int64_t>{0}, std::vector<std::unique_ptr<Chunk>>(1));
        auto slot = cachinglayer::Manager::GetInstance().CreateCacheSlot<Chunk>(
            std::move(translator), nullptr);
        FieldMeta meta(
            FieldName("x"), FieldId(100), DataType::INT64, false, std::nullopt);
        fixture->column = std::make_shared<ChunkedColumn>(slot, meta);
        try {
            ColumnReaderAdapter adapter(fixture->column, type, 0);
            FAIL();
        } catch (const SegcoreError& e) {
            EXPECT_EQ(e.get_error_code(), Unsupported);
        }
    }
}

TEST(ColumnReaderAdapterTest,
     ResultProtocolRejectsMissingOrConflictingCoverage) {
    auto all = SelectionView::All(2);
    auto none = SelectionView::None(2);
    ReadResult missing{
        std::nullopt, OwnedSelection::All(2), OwnedSelection::All(2)};
    EXPECT_THROW(missing.Validate(DataType::INT64, all, kValues), SegcoreError);
    missing.Validate(DataType::INT64, all, kValidity);
    missing.values = ReadOnlyVectorView::Nulls(DataType::INT64, all);
    EXPECT_THROW(missing.Validate(DataType::INT64, all, kValues), SegcoreError);
    missing.valid.ResetNone(2);
    missing.Validate(DataType::INT64, all, kValues);
    EXPECT_THROW(missing.Validate(DataType::VARCHAR, all, kValues),
                 SegcoreError);
    missing.values = ReadOnlyVectorView::Nulls(DataType::INT64, none);
    EXPECT_THROW(missing.Validate(DataType::INT64, all, kValues), SegcoreError);
    missing.defined.ResetNone(2);
    EXPECT_THROW(missing.Validate(DataType::INT64, all, kValidity),
                 SegcoreError);
}

TEST(ColumnReaderAdapterTest, TranslatorErrorSurvivesCacheSlotAndReaderLayers) {
    for (bool take : {false, true}) {
        for (bool validity : {false, true}) {
            std::shared_ptr<Probe> probe;
            auto reader = MakeReader(DataType::INT64, true, {3, 4, 3}, &probe);
            auto batch = Batch(reader,
                               take ? ReadBatchBinding::Offsets({2, 4, 8})
                                    : ReadBatchBinding::Range(2, 7));
            const auto rows = SelectionView::All(batch->size());
            probe->fail_load = true;
            try {
                if (validity)
                    batch->GetValidity(0, rows);
                else
                    batch->GetColumn(0, rows);
                FAIL();
            } catch (const SegcoreError& e) {
                EXPECT_EQ(e.get_error_code(), FileReadFailed);
            }
            probe->fail_load = false;
            const auto result = batch->GetColumn(0, rows);
            EXPECT_EQ(result.ValueAt<int64_t>(0), 2);
            EXPECT_EQ(result.ValueAt<int64_t>(batch->size() - 1), 8);
        }
    }
}

TEST(ColumnReaderAdapterTest, UnalignedStringOffsetsWorkForEveryViewEntry) {
    // One-byte nullable bitmap, three uint32 offsets, four payload bytes.
    std::vector<char> buffer(18, 0);
    buffer[0] = 3;
    const uint32_t offsets[] = {13, 15, 17};
    std::memcpy(buffer.data() + 1, offsets, sizeof(offsets));
    std::memcpy(buffer.data() + 13, "abcd", 4);
    StringChunk chunk(2,
                      buffer.data(),
                      buffer.size(),
                      true,
                      std::make_shared<ChunkMmapGuard>(nullptr, 0, ""));
    EXPECT_EQ(chunk[0], "ab");
    EXPECT_EQ(chunk[1], "cd");
    auto range = chunk.StringViews(std::make_pair(1, 1));
    EXPECT_EQ(range.first[0], "cd");
    EXPECT_TRUE(range.second[0]);
    auto take = chunk.ViewsByOffsets(FixedVector<int32_t>{1, 0, 1});
    EXPECT_EQ(take.first, (std::vector<std::string_view>{"cd", "ab", "cd"}));
    EXPECT_TRUE(take.second[0]);
    EXPECT_EQ(chunk.binary_search_string("cd"), 1);
    EXPECT_EQ(chunk.lower_bound_string("ab"), 0);
    EXPECT_EQ(chunk.upper_bound_string("cd"), 2);
}

TEST(ColumnReaderAdapterTest, PhysicalPlanSelectsIndexAndRealColumnProjection) {
    std::shared_ptr<Probe> probe;
    auto reader = MakeReader(DataType::INT64, true, {3, 4, 3}, &probe);
    auto context = std::make_shared<ReadContext>();
    auto sorted =
        std::make_shared<milvus::segcore::reader::TestScalarIndex<int64_t>>();
    int64_t values[10];
    bool valid[10];
    for (int i = 0; i < 10; ++i) {
        values[i] = i;
        valid[i] = i % 4 != 1;
    }
    sorted->Build(10, values, valid);
    expr::ColumnInfo field(FieldId(100), DataType::INT64, {}, true);
    proto::plan::GenericValue bound;
    bound.set_int64_val(4);
    exec::plan::CompileOptions options;
    options.plan_indexes = true;
    auto compiled = exec::plan::CompileExpressions(
        {std::make_shared<expr::UnaryRangeFilterExpr>(
             field, proto::plan::GreaterThan, bound),
         std::make_shared<expr::ColumnExpr>(field)},
        options);
    ASSERT_TRUE(compiled.supported());
    for (bool use_index : {true, false}) {
        probe->takes = 0;
        auto prepared = exec::plan::Prepare(
            compiled.plan,
            [&](const auto&) {
                return access::PrepareReaderInput(reader, context);
            },
            {use_index, [&](const exec::plan::PredicatePlan& predicate) {
                 return exec::plan::PrepareInt64Index(
                     sorted, 10, predicate.query, context);
             }});
        auto batch = prepared.MakeExecutionBatch(
            ReadBatchBinding::Offsets({8, 1, 6, 8}));
        auto filtered = batch->Eval(std::vector<SelectionView>{
            SelectionView::All(4), SelectionView::None(4)});
        EXPECT_TRUE(filtered[0].ValueAt<bool>(0));
        EXPECT_FALSE(filtered[0].IsValid(1));
        EXPECT_TRUE(filtered[0].ValueAt<bool>(2));
        EXPECT_EQ(probe->takes, use_index ? 0 : 1);
        auto selected = OwnedSelection::FromIndices(4, {0, 2, 3});
        auto projected = batch->Eval(std::vector<SelectionView>{
            SelectionView::None(4), selected.view()});
        EXPECT_EQ(projected[1].ValueAt<int64_t>(0), 8);
        EXPECT_EQ(projected[1].ValueAt<int64_t>(2), 6);
        EXPECT_EQ(projected[1].ValueAt<int64_t>(3), 8);
        // Index route defers Take until projection. Raw route reuses its values.
        EXPECT_EQ(probe->takes, 1);
    }
}

TEST(ColumnReaderAdapterTest,
     NumericScanSharesPayloadAndReleasesPinWithLastView) {
    std::shared_ptr<Probe> probe;
    auto reader = MakeReader(DataType::INT64, false, {8, 8}, &probe);
    auto result = reader->ReadRange(
        RangeRequest(2, 4, kValues, SelectionView::All(4)), *Context());
    ASSERT_EQ(probe->scan_owners.size(), 1);
    ASSERT_EQ(probe->scan_payloads.size(), 4);
    EXPECT_EQ(result.values->FlatData<int64_t>(),
              probe->scan_payloads[0].second);
    auto retained = result.values->Slice(1, 2);
    result.values.reset();
    reader.reset();
    EXPECT_FALSE(probe->scan_owners[0].expired());
    EXPECT_EQ(retained.ValueAt<int64_t>(1), 4);
    retained =
        ReadOnlyVectorView::Nulls(DataType::INT64, SelectionView::None(2));
    EXPECT_TRUE(probe->scan_owners[0].expired());
}

TEST(ColumnReaderAdapterTest,
     IncrementalStringAndJsonScansKeepPayloadAddresses) {
    for (auto type : {DataType::VARCHAR, DataType::JSON}) {
        std::shared_ptr<Probe> probe;
        auto reader = MakeReader(type, false, {3, 4, 3}, &probe);
        auto batch = Batch(reader, ReadBatchBinding::Range(0, 10));
        const auto selected = OwnedSelection::FromIndices(10, {0, 4, 8});
        auto early = batch->GetColumn(0, selected.view());
        auto all = batch->GetColumn(0, SelectionView::All(10));
        reader.reset();
        batch.reset();
        for (const auto& [row, address] : probe->scan_payloads) {
            const void* actual =
                type == DataType::JSON
                    ? static_cast<const void*>(
                          all.JsonViewAt(row).data().data())
                    : static_cast<const void*>(all.StringViewAt(row).data());
            EXPECT_EQ(actual, address);
        }
        EXPECT_THROW(early.IsValid(1), SegcoreError);
        all = ReadOnlyVectorView::Nulls(type, SelectionView::None(10));
        for (size_t i = 0; i < probe->scan_owners.size(); ++i)
            EXPECT_EQ(probe->scan_owners[i].expired(), i >= 3);
        early = ReadOnlyVectorView::Nulls(type, SelectionView::None(10));
        for (const auto& owner : probe->scan_owners)
            EXPECT_TRUE(owner.expired());
    }
}

TEST(ColumnReaderAdapterTest, AlreadyOwnedTakeTransfersBackendBuffer) {
    class ObservedVirtual : public VirtualPKChunkedColumn {
     public:
        ObservedVirtual() : VirtualPKChunkedColumn(7, 10) {
        }
        TakeResultPtr
        Take(OpContext* ctx, TakeOptions options) const override {
            auto result = VirtualPKChunkedColumn::Take(ctx, options);
            auto owned = result->GetOwn();
            owner = owned.owner;
            address = owned.values.data_as<int64_t>();
            return result;
        }
        mutable std::weak_ptr<void> owner;
        mutable const int64_t* address = nullptr;
    };
    auto column = std::make_shared<ObservedVirtual>();
    auto reader =
        std::make_unique<ColumnReaderAdapter>(column, DataType::INT64, 10);
    auto out = reader->Take(
        TakeRequest({4, 1, 4}, kValues, SelectionView::All(3)), *Context());
    EXPECT_EQ(out.values->FlatData<int64_t>(), column->address);
    EXPECT_EQ(out.values->ValueAt<int64_t>(0), out.values->ValueAt<int64_t>(2));
    auto owner = column->owner;
    auto rows = OwnedSelection::FromIndices(4, {1, 3});
    auto sparse = reader->Take(TakeRequest({0, 4, 0, 1}, kValues, rows.view()),
                               *Context());
    EXPECT_EQ(sparse.values->ValueAt<int64_t>(1),
              out.values->ValueAt<int64_t>(0));
    EXPECT_EQ(sparse.values->ValueAt<int64_t>(3),
              out.values->ValueAt<int64_t>(1));
    EXPECT_THROW(sparse.values->IsValid(0), SegcoreError);
    reader.reset();
    column.reset();
    EXPECT_FALSE(owner.expired());
    out.values.reset();
    EXPECT_TRUE(owner.expired());
}

TEST(ColumnReaderAdapterTest, BulkValidationRejectsMaskMismatchAcrossWords) {
    constexpr size_t n = 131;
    BitmapResultBuilder bits(n);
    for (size_t i = 0; i < n; ++i)
        if (i != 64)
            bits.Set<bool>(i, true);
    auto values = std::move(bits).Finish(SelectionView::All(n));
    auto valid = OwnedSelection::All(n);
    valid.SetSelected(64, false);
    ReadResult result{values, OwnedSelection::All(n), valid};
    ASSERT_NO_THROW(
        result.Validate(DataType::BOOL, SelectionView::All(n), kValues));
    // Both directions of disagreement, including the second mask word.
    result.valid.SetSelected(64);
    EXPECT_THROW(
        result.Validate(DataType::BOOL, SelectionView::All(n), kValues),
        SegcoreError);
    result.valid = valid;
    result.valid.SetSelected(129, false);
    EXPECT_THROW(
        result.Validate(DataType::BOOL, SelectionView::All(n), kValues),
        SegcoreError);
    result.valid = valid;
    result.defined.SetSelected(129, false);
    EXPECT_THROW(
        result.Validate(DataType::BOOL, SelectionView::All(n), kValues),
        SegcoreError);
    result.defined = OwnedSelection::All(n);
    auto subset = OwnedSelection::FromIndices(n, {1, 64, 129});
    result.values = values.Restrict(subset.view());
    ASSERT_NO_THROW(result.Validate(DataType::BOOL, subset.view(), kValues));
    EXPECT_THROW(
        result.Validate(DataType::BOOL, SelectionView::All(n), kValues),
        SegcoreError);
    // Extra validity outside value coverage is allowed; mismatches inside are not.
    result.valid.SetSelected(129, false);
    EXPECT_THROW(result.Validate(DataType::BOOL, subset.view(), kValidity),
                 SegcoreError);
}

TEST(ColumnReaderAdapterTest, ReaderValidityMetadataSurvivesLoaderBridge) {
    class MetadataReader final : public ColumnReader {
     public:
        mutable int calls = 0;
        bool disagree = false;
        bool change_known = false;
        DataType
        type() const override {
            return DataType::INT64;
        }
        int64_t
        row_count() const override {
            return 4;
        }
        ReadResult
        Read(SelectionView rows) const {
            ++calls;
            VectorResultBuilder builder(DataType::INT64, 4);
            for (size_t i = 0; i < 3; ++i) builder.Set<int64_t>(i, i + 10);
            auto values = std::move(builder).Finish(SelectionView::All(4));
            auto valid = OwnedSelection::FromIndices(4, {0, 1, 2});
            if (disagree || change_known)
                valid.SetSelected(0, false);
            return {values.Restrict(rows),
                    OwnedSelection::All(4),
                    std::move(valid)};
        }
        ReadResult
        ReadRange(const RangeRequest& r, const ReadContext&) const override {
            return Read(r.eval_rows.view());
        }
        ReadResult
        Take(const TakeRequest& r, const ReadContext&) const override {
            return Read(r.eval_rows.view());
        }
    };
    for (bool offsets : {false, true}) {
        auto reader = std::make_shared<MetadataReader>();
        auto input = Batch(reader,
                           offsets ? ReadBatchBinding::Offsets({0, 1, 2, 3})
                                   : ReadBatchBinding::Range(0, 4));
        auto first = OwnedSelection::FromIndices(4, {0});
        reader->disagree = true;
        EXPECT_THROW(input->GetColumn(0, first.view()), SegcoreError);
        reader->disagree = false;
        EXPECT_EQ(input->GetColumn(0, first.view()).ValueAt<int64_t>(0), 10);
        const auto reads = reader->calls;
        auto validity = input->GetValidity(0, SelectionView::All(4));
        EXPECT_EQ(reader->calls, reads);  // Preserve extra validity coverage.
        EXPECT_EQ(validity.valid.view().selected_count(), 3);
        // A later read must not overwrite previously published validity, even
        // when that row is outside the returned values' coverage.
        reader->change_known = true;
        EXPECT_THROW(input->GetColumn(0, SelectionView::All(4)), SegcoreError);
        reader->change_known = false;
        auto all = input->GetColumn(0, SelectionView::All(4));
        EXPECT_EQ(all.ValueAt<int64_t>(2), 12);
        EXPECT_FALSE(all.IsValid(3));
        EXPECT_EQ(input->GetColumn(0, first.view()).ValueAt<int64_t>(0), 10);
    }
}

TEST(ColumnReaderAdapterTest, BulkValidityAcrossUnalignedCellsAndSparseRanges) {
    for (bool nullable : {false, true}) {
        auto reader =
            MakeReader(DataType::INT64, nullable, {67, 69, 137}, nullptr, 273);
        for (auto projection : {kValues, kValidity}) {
            for (bool sparse : {false, true}) {
                constexpr size_t size = 263;
                auto rows = OwnedSelection::All(size);
                if (sparse)
                    for (size_t i = 0; i < size; ++i)
                        if (i % 3 == 0)
                            rows.SetSelected(i, false);
                auto result = reader->ReadRange(
                    RangeRequest(3, size, projection, rows.view()), *Context());
                result.Validate(DataType::INT64, rows.view(), projection);
                for (size_t i = 0; i < size; ++i) {
                    const bool valid = !nullable || (i + 3) % 4 != 1;
                    EXPECT_EQ(result.defined.view().IsSelected(i),
                              rows.view().IsSelected(i));
                    EXPECT_EQ(result.valid.view().IsSelected(i),
                              rows.view().IsSelected(i) && valid);
                    if (result.values && rows.view().IsSelected(i) && valid)
                        EXPECT_EQ(result.values->ValueAt<int64_t>(i), i + 3);
                }
            }
        }
    }
}

TEST(ColumnReaderAdapterTest, DenseNumericCapabilityComesFromPhysicalScan) {
    for (auto type : {DataType::INT8,
                      DataType::INT16,
                      DataType::INT32,
                      DataType::INT64,
                      DataType::FLOAT,
                      DataType::DOUBLE}) {
        auto reader = MakeReader(type, true);
        auto result = reader->ReadRange(
            RangeRequest(0, 3, kValues, SelectionView::All(3)), *Context());
        auto decoded = result.values->DecodePrimitive();
        // Some small chunks have an unaligned payload after their null bitmap;
        // the existing typed-access fallback must still reject those addresses.
        if (decoded) {
            EXPECT_EQ(decoded->spans[0].payload_access,
                      PrimitivePayloadAccess::AllDefinedRows);
            EXPECT_FALSE(decoded->spans[0].valid.IsSelected(1));
        }
    }
    for (auto [type, target] :
         {std::pair{DataType::FLOAT, TargetType::Float},
          std::pair{DataType::DOUBLE, TargetType::Double}}) {
        std::shared_ptr<ChunkedColumnInterface> physical_column;
        auto reader =
            MakeReader(type, true, {128}, nullptr, 128, &physical_column);
        auto cursor =
            physical_column->Scan(nullptr, ScanOptions::ForData(0, target));
        ScanBatch physical;
        ASSERT_TRUE(cursor->Next(64, ScanReadMode::DataAndValidity, &physical));
        EXPECT_TRUE(physical.floating_null_payload_readable);
        EXPECT_FALSE(physical.integer_null_payload_readable);
        ASSERT_TRUE(cursor->Next(64, ScanReadMode::ValidityOnly, &physical));
        EXPECT_FALSE(physical.floating_null_payload_readable);
        auto result = reader->ReadRange(
            RangeRequest(0, 128, kValues, SelectionView::All(128)), *Context());
        auto decoded = result.values->DecodePrimitive();
        ASSERT_TRUE(decoded);
        EXPECT_TRUE(decoded->spans[0].CanReadAll(0, 128));
    }
    std::shared_ptr<ChunkedColumnInterface> column;
    auto reader =
        MakeReader(DataType::INT64, true, {128}, nullptr, 128, &column);
    auto cursor =
        column->Scan(nullptr, ScanOptions::ForData(0, TargetType::Int64));
    ScanBatch physical;
    ASSERT_TRUE(cursor->Next(64, ScanReadMode::DataAndValidity, &physical));
    EXPECT_TRUE(physical.integer_null_payload_readable);
    ASSERT_TRUE(cursor->Next(64, ScanReadMode::ValidityOnly, &physical));
    EXPECT_FALSE(physical.integer_null_payload_readable);
    auto batch = Batch(reader, ReadBatchBinding::Range(0, 128));
    auto values = batch->GetColumn(0, SelectionView::All(128));
    auto decoded = values.DecodePrimitive();
    ASSERT_TRUE(decoded);
    EXPECT_TRUE(decoded->spans[0].CanReadAll(0, 128));
    auto rows = OwnedSelection::FromIndices(128, {1, 64});
    auto sparse = reader->ReadRange(RangeRequest(0, 128, kValues, rows.view()),
                                    *Context());
    auto partial = sparse.values->DecodePrimitive();
    ASSERT_TRUE(partial);
    EXPECT_FALSE(partial->spans[0].CanReadAll(0, partial->spans[0].count));
}

TEST(ColumnReaderAdapterTest,
     ValidatedReaderResultStillChecksDestinationSlotType) {
    auto reader = MakeReader(DataType::INT64, true, {4, 6});
    auto source = access::PrepareReaderInput(reader, Context());
    for (bool direct : {false, true}) {
        BatchInput input(
            std::make_shared<RowType>(std::vector<std::string>{"bool_slot"},
                                      std::vector<DataType>{DataType::BOOL}),
            4);
        input.BindLoader(0, [source, direct](SelectionView rows) {
            auto binding = ReadBatchBinding::Range(0, 4);
            return direct ? source.read_values(binding, rows)
                          : source.make_loaders(binding).values(rows);
        });
        EXPECT_THROW(input.GetColumn(0, SelectionView::All(4)), SegcoreError);
        EXPECT_THROW(input.GetColumn(0, SelectionView::All(4)), SegcoreError);
    }
}

TEST(ColumnReaderAdapterTest, ReusedCursorSeeksBackwardAndDiscardsFailedState) {
    std::shared_ptr<Probe> probe;
    auto reader = MakeReader(DataType::INT64, true, {3, 4, 3}, &probe);
    auto context = Context();
    auto initial_validity = reader->ReadRange(
        RangeRequest(0, 3, kValidity, SelectionView::All(3)), *context);
    EXPECT_FALSE(initial_validity.valid.view().IsSelected(1));
    auto later = reader->ReadRange(
        RangeRequest(6, 4, kValues, SelectionView::All(4)), *context);
    auto earlier = reader->ReadRange(
        RangeRequest(0, 4, kValues, SelectionView::All(4)), *context);
    EXPECT_EQ(probe->scans,
              2);  // Backward request recreates a forward-only cursor.
    EXPECT_EQ(later.values->ValueAt<int64_t>(0), 6);
    EXPECT_FALSE(later.values->IsValid(3));
    EXPECT_EQ(earlier.values->ValueAt<int64_t>(3), 3);
    auto validity = reader->ReadRange(
        RangeRequest(4, 4, kValidity, SelectionView::All(4)), *context);
    EXPECT_FALSE(validity.valid.view().IsSelected(1));
    EXPECT_EQ(probe->scans, 2);
    auto take = reader->Take(
        TakeRequest({2, 0, 2}, kValues, SelectionView::All(3)), *context);
    EXPECT_EQ(take.values->ValueAt<int64_t>(2), 2);
    probe->fail_after = probe->nexts;
    EXPECT_THROW(
        reader->ReadRange(RangeRequest(8, 2, kValues, SelectionView::All(2)),
                          *context),
        SegcoreError);
    probe->fail_after = -1;
    auto retried = reader->ReadRange(
        RangeRequest(2, 6, kValues, SelectionView::All(6)), *context);
    EXPECT_EQ(probe->scans, 3);
    EXPECT_EQ(retried.values->ValueAt<int64_t>(5), 7);
    context->op_context = std::make_shared<OpContext>();
    reader->ReadRange(RangeRequest(0, 3, kValues, SelectionView::All(3)),
                      *context);
    EXPECT_EQ(probe->scans, 4);  // Rebind the cursor's operation context.
    context->op_context = std::make_shared<OpContext>();
    reader->ReadRange(RangeRequest(3, 3, kValues, SelectionView::All(3)),
                      *context);
    EXPECT_EQ(probe->scans, 5);
    reader.reset();
    EXPECT_EQ(later.values->ValueAt<int64_t>(0), 6);
    EXPECT_EQ(earlier.values->ValueAt<int64_t>(3), 3);
}

TEST(ColumnReaderAdapterTest,
     DenseStringCapabilityComesFromPinnedViewsAndResets) {
    std::shared_ptr<ChunkedColumnInterface> column;
    auto reader =
        MakeReader(DataType::VARCHAR, true, {3, 3, 3}, nullptr, 9, &column);
    auto result = reader->ReadRange(
        RangeRequest(0, 3, kValues, SelectionView::All(3)), *Context());
    auto decoded = result.values->DecodeStrings();
    ASSERT_TRUE(decoded);
    ASSERT_EQ(decoded->spans.size(), 1);
    EXPECT_TRUE(decoded->spans[0].string_views);
    EXPECT_TRUE(decoded->spans[0].CanReadAll(0, 3));
    EXPECT_FALSE(decoded->spans[0].valid.IsSelected(1));
    // Exercise the actual NULL payload, rather than checking only the flag.
    const auto null_payload =
        static_cast<const std::string_view*>(decoded->spans[0].data)[1];
    EXPECT_EQ(null_payload, "北京-1" + std::string(100, 'x'));
    auto scan =
        column->Scan(nullptr, ScanOptions::ForData(0, TargetType::StringView));
    ScanBatch physical;
    ASSERT_TRUE(scan->Next(3, ScanReadMode::DataAndValidity, &physical));
    EXPECT_TRUE(physical.string_null_payload_readable);
    ASSERT_TRUE(scan->Next(3, ScanReadMode::ValidityOnly, &physical));
    EXPECT_FALSE(physical.string_null_payload_readable);
}

}  // namespace milvus::segcore::reader
