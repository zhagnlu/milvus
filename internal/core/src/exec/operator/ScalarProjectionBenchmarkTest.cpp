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

// Explicitly selected microbenchmark of projection operators. Normal wildcard
// UT runs skip TimingAndMemoryMatrix. No latency/RSS threshold is asserted.
#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#ifdef __GLIBC__
#include <malloc.h>
#endif
#include "exec/operator/ProjectNode.h"
#include "exec/operator/ScalarProjectNode.h"
#include "common/VectorAccess.h"
#include "test_utils/storage_test_utils.h"

namespace milvus::exec {
namespace {
namespace logical = ::milvus::plan;
using Clock = std::chrono::steady_clock;
using Report = nlohmann::json;
constexpr size_t kBatch = 8192;

uint64_t
ProcKiB(const std::string& key) {
    std::ifstream in("/proc/self/status");
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream field(line);
        std::string name;
        uint64_t n;
        if ((field >> name >> n) && name == key + ":")
            return n;
    }
    return 0;
}

std::string
Allocator() {
    Dl_info info{};
    return dladdr(reinterpret_cast<void*>(&malloc), &info) && info.dli_fname
               ? info.dli_fname
               : "unknown";
}

std::optional<size_t>
HeapBytes() {
#ifdef __GLIBC__
#if __GLIBC_PREREQ(2, 33)
    if (Allocator().find("libc.so") != std::string::npos) {
        auto m = mallinfo2();
        return m.uordblks + m.hblkhd;
    }
#endif
#endif
    return std::nullopt;
}

bool
TrimHeap() {
#ifdef __GLIBC__
    if (Allocator().find("libc.so") != std::string::npos) {
        malloc_trim(0);
        return true;
    }
#endif
    return false;
}

bool
ResetRssPeak() {
    // Linux clear_refs=5 resets VmHWM to current RSS, without dropping pages.
    // Never substitute lifetime process peak or zero on an unsupported host.
    std::ofstream reset("/proc/self/clear_refs");
    reset << "5";
    reset.close();
    return !reset.fail();
}

// Optional runtime capability: neither a jemalloc link dependency nor a zero
// estimate on allocators/builds without statistics. Counters cover this thread
// only; peak.read is approximate and may not exist in the installed version.
class ThreadMemory {
 public:
    ThreadMemory() {
        if (Allocator().find("jemalloc") == std::string::npos)
            return;
        ctl_ = reinterpret_cast<Control>(dlsym(RTLD_DEFAULT, "mallctl"));
        if (!ctl_)
            return;
        size_t size = sizeof(uint64_t*);
        if (ctl_("thread.allocatedp", &allocated_, &size, nullptr, 0) != 0)
            allocated_ = nullptr;
        size = sizeof(uint64_t*);
        if (ctl_("thread.deallocatedp", &deallocated_, &size, nullptr, 0) != 0)
            deallocated_ = nullptr;
    }
    bool
    available() const {
        return allocated_ && deallocated_;
    }
    void
    Begin() {
        if (!available())
            return;
        peak_reset_ =
            ctl_("thread.peak.reset", nullptr, nullptr, nullptr, 0) == 0;
        allocated_base_ = *allocated_;
        deallocated_base_ = *deallocated_;
    }
    uint64_t
    Allocated() const {
        return *allocated_ - allocated_base_;
    }
    int64_t
    LiveDelta() const {
        return int64_t(Allocated()) -
               int64_t(*deallocated_ - deallocated_base_);
    }
    std::optional<uint64_t>
    Peak() const {
        uint64_t peak = 0;
        size_t size = sizeof(peak);
        if (peak_reset_ &&
            ctl_("thread.peak.read", &peak, &size, nullptr, 0) == 0)
            return peak;
        return std::nullopt;
    }

 private:
    using Control = int (*)(const char*, void*, size_t*, void*, size_t);
    Control ctl_ = nullptr;
    uint64_t *allocated_ = nullptr, *deallocated_ = nullptr;
    uint64_t allocated_base_ = 0, deallocated_base_ = 0;
    bool peak_reset_ = false;
};

double
Us(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

struct Sample {
    double prepare_us = 0, eval_us = 0, cleanup_us = 0;
    size_t rows = 0, batches = 0, max_rows = 0;
    std::optional<size_t> heap_base, heap_output_max;
    uint64_t rss_base_kib = 0, rss_peak_kib = 0;
    bool rss_reset = false, trimmed = false;
    std::optional<uint64_t> thread_allocated, thread_peak;
    std::optional<int64_t> thread_output_live_max;
    double
    Total() const {
        return prepare_us + eval_us + cleanup_us;
    }
};

struct Fixture {
    SchemaPtr schema = std::make_shared<Schema>();
    FieldId pk{0}, a{0}, b{0};
    size_t count;
    std::unique_ptr<segcore::SegmentSealed> segment;
    explicit Fixture(size_t rows) : count(rows) {
        pk = schema->AddDebugField("pk", DataType::INT64);
        schema->set_primary_field_id(pk);
        a = schema->AddDebugField("a", DataType::INT64, true);
        b = schema->AddDebugField("b", DataType::INT64, true);
        auto data = segcore::DataGen(schema, rows);
        for (auto& field : *data.raw_->mutable_fields_data()) {
            FieldId id(field.field_id());
            FixedVector<int64_t> numbers(rows);
            FixedVector<bool> valid(rows, true);
            for (size_t row = 0; row < rows; ++row) {
                numbers[row] = id == pk ? row : id == a ? A(row) : B(row);
                valid[row] = id == pk || (id == a ? ValidA(row) : ValidB(row));
            }
            auto replacement = segcore::CreateDataArrayFrom(
                numbers.data(), valid.data(), rows, (*schema)[id]);
            field.Swap(replacement.get());
        }
        segment = CreateSealedWithFieldDataLoaded(schema, data);
    }
    static int64_t
    A(size_t row) {
        return row * 17 % 1000;
    }
    static int64_t
    B(size_t row) {
        return int64_t(row * 37 % 1000) - 500;
    }
    static bool
    ValidA(size_t row) {
        return row % 97 != 0;
    }
    static bool
    ValidB(size_t row) {
        return row % 89 != 0;
    }

    std::vector<expr::TypedExprPtr>
    Roots(bool arithmetic) {
        auto ca = std::make_shared<expr::ColumnExpr>(
            expr::ColumnInfo(a, DataType::INT64, {}, true));
        auto cb = std::make_shared<expr::ColumnExpr>(
            expr::ColumnInfo(b, DataType::INT64, {}, true));
        if (!arithmetic)
            return {ca, cb};
        auto literal = [](int64_t n) {
            proto::plan::GenericValue v;
            v.set_int64_val(n);
            return std::make_shared<expr::ValueExpr>(v);
        };
        return {std::make_shared<expr::BinaryArithExpr>(
                    proto::plan::Add, ca, literal(1)),
                std::make_shared<expr::BinaryArithExpr>(
                    proto::plan::Add,
                    ca,
                    std::make_shared<expr::BinaryArithExpr>(
                        proto::plan::Mul, cb, literal(2)))};
    }

    void
    Run(bool bounded,
        bool arithmetic,
        size_t stride,
        bool memory,
        Sample* sample) {
        auto roots = Roots(arithmetic);
        auto mvcc = std::make_shared<logical::MvccNode>("visibility");
        auto fields = std::make_shared<logical::ProjectNode>(
            "fields",
            std::vector<FieldId>{a, b},
            std::vector<std::string>{"x", "y"},
            std::vector<DataType>{DataType::INT64, DataType::INT64},
            std::vector<logical::PlanNodePtr>{mvcc});
        auto scan = std::make_shared<logical::ScalarProjectScanNode>(
            "scan", roots, std::vector<std::string>{"x", "y"}, mvcc);
        auto project = std::make_shared<logical::ScalarProjectNode>(
            "compute",
            roots,
            std::vector<std::string>{"x", "y"},
            std::map<int64_t, column_index_t>{{a.get(), 0}, {b.get(), 1}},
            fields);
        auto context = std::make_shared<QueryContext>(
            "projection-benchmark",
            segment.get(),
            count,
            MAX_TIMESTAMP,
            0,
            0,
            query::PlanOptions{},
            std::make_shared<QueryConfig>(
                std::unordered_map<std::string, std::string>{
                    {QueryConfig::kExprEvalBatchSize,
                     std::to_string(kBatch)}}));
        context->set_read_snapshot(segment->CaptureReadSnapshot());
        OpContext op;
        context->set_op_context(&op);
        auto task = Task::Create(
            "projection-benchmark", logical::PlanFragment(), 0, context);
        DriverContext driver(task, 0, 0, 0, 0);
        TargetBitmap excluded(count, true);
        for (size_t row = 0; row < count; row += stride)
            excluded.set(row, false);
        RowVectorPtr input = std::make_shared<RowVector>(
            std::vector<VectorPtr>{std::make_shared<ColumnVector>(
                std::move(excluded), TargetBitmap(count, true))});
        ThreadMemory thread_memory;
        if (memory) {
            sample->trimmed = TrimHeap();
            sample->heap_base = HeapBytes();
            sample->heap_output_max = sample->heap_base;
            sample->rss_reset = ResetRssPeak();
            sample->rss_base_kib = ProcKiB("VmRSS");
            thread_memory.Begin();
            if (thread_memory.available())
                sample->thread_output_live_max = 0;
        }
        std::unique_ptr<PhyProjectNode> fetch;
        std::unique_ptr<PhyScalarProjectNode> compute;
        std::unique_ptr<PhyScalarProjectScanNode> source;
        auto begin = Clock::now();
        if (bounded) {
            source =
                std::make_unique<PhyScalarProjectScanNode>(0, &driver, scan);
            source->AddInput(input);
            source->NoMoreInput();
        } else {
            fetch = std::make_unique<PhyProjectNode>(0, &driver, fields);
            fetch->AddInput(input);
            if (arithmetic)
                compute =
                    std::make_unique<PhyScalarProjectNode>(1, &driver, project);
        }
        sample->prepare_us = Us(begin, Clock::now());
        size_t physical_row = 0;
        while (true) {
            begin = Clock::now();
            RowVectorPtr output;
            if (bounded)
                output = source->GetOutput();
            else {
                output = fetch->GetOutput();
                if (output && compute) {
                    compute->AddInput(output);
                    output = compute->GetOutput();
                }
            }
            sample->eval_us += Us(begin, Clock::now());
            if (!output)
                break;
            if (memory && sample->heap_base)
                if (auto heap = HeapBytes())
                    sample->heap_output_max =
                        std::max(*sample->heap_output_max, *heap);
            sample->rows += output->size();
            ++sample->batches;
            sample->max_rows =
                std::max<size_t>(sample->max_rows, output->size());
            if (memory) {
                if (thread_memory.available())
                    sample->thread_output_live_max =
                        std::max(*sample->thread_output_live_max,
                                 thread_memory.LiveDelta());
                // Timed runs already validate every value. Exclude oracle
                // copies/assertion bookkeeping from the allocation sample.
                continue;
            }
            ASSERT_EQ(output->childrens().size(), 2);
            auto x = ReadOnlyVectorView::CopyOf(output->child(0));
            auto y = ReadOnlyVectorView::CopyOf(output->child(1));
            // Independent full oracle is outside the timed operator calls.
            for (size_t i = 0; i < output->size();
                 ++i, physical_row += stride) {
                ASSERT_LT(physical_row, count);
                const bool va = ValidA(physical_row), vb = ValidB(physical_row);
                ASSERT_EQ(x.IsValid(i), va);
                ASSERT_EQ(y.IsValid(i), arithmetic ? va && vb : vb);
                if (va)
                    ASSERT_EQ(x.ValueAt<int64_t>(i),
                              A(physical_row) + (arithmetic ? 1 : 0));
                if (arithmetic ? va && vb : vb)
                    ASSERT_EQ(y.ValueAt<int64_t>(i),
                              arithmetic ? A(physical_row) + 2 * B(physical_row)
                                         : B(physical_row));
            }
        }
        begin = Clock::now();
        source.reset();
        compute.reset();
        fetch.reset();
        sample->cleanup_us = Us(begin, Clock::now());
        if (memory) {
            if (thread_memory.available()) {
                sample->thread_allocated = thread_memory.Allocated();
                sample->thread_peak = thread_memory.Peak();
            }
            sample->rss_peak_kib = ProcKiB("VmHWM");
        }
        ASSERT_EQ(sample->rows, (count + stride - 1) / stride);
        ASSERT_EQ(sample->batches,
                  bounded ? (sample->rows + kBatch - 1) / kBatch : 1);
        ASSERT_LE(sample->max_rows, bounded ? kBatch : count);
    }
};

Report
Times(std::vector<double> samples) {
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    return {{"median_us", sorted[sorted.size() / 2]}, {"samples_us", samples}};
}

Report
Memory(const Sample& s) {
    Report out = {{"rss_reset", s.rss_reset},
                  {"glibc_trim_invoked", s.trimmed},
                  {"source_rows", s.rows},
                  {"source_batches", s.batches},
                  {"max_source_batch_rows", s.max_rows}};
    out["thread_allocated_bytes"] =
        s.thread_allocated ? Report(*s.thread_allocated) : Report(nullptr);
    out["thread_live_at_output_max_delta_bytes"] =
        s.thread_output_live_max ? Report(*s.thread_output_live_max)
                                 : Report(nullptr);
    out["thread_approx_peak_delta_bytes"] =
        s.thread_peak ? Report(*s.thread_peak) : Report(nullptr);
    if (s.rss_reset && s.rss_base_kib && s.rss_peak_kib) {
        out["rss_baseline_kib"] = s.rss_base_kib;
        out["rss_peak_kib"] = s.rss_peak_kib;
        out["rss_peak_delta_kib"] = s.rss_peak_kib > s.rss_base_kib
                                        ? s.rss_peak_kib - s.rss_base_kib
                                        : 0;
    } else
        out["rss_peak_delta_kib"] = nullptr;
    if (s.heap_base && s.heap_output_max) {
        out["heap_baseline_bytes"] = *s.heap_base;
        out["heap_live_at_output_max_delta_bytes"] =
            *s.heap_output_max > *s.heap_base
                ? *s.heap_output_max - *s.heap_base
                : 0;
    } else
        out["heap_live_at_output_max_delta_bytes"] = nullptr;
    return out;
}

TEST(ScalarProjectionBenchmarkTest, SmallOracle) {
    Fixture fixture(2051);
    for (bool bounded : {false, true})
        for (bool arithmetic : {false, true})
            for (size_t stride : {1, 100}) {
                Sample sample;
                ASSERT_NO_FATAL_FAILURE(
                    fixture.Run(bounded, arithmetic, stride, false, &sample));
            }
}

TEST(ScalarProjectionBenchmarkTest, TimingAndMemoryMatrix) {
    if (::testing::GTEST_FLAG(filter).find("TimingAndMemoryMatrix") ==
        std::string::npos)
        GTEST_SKIP() << "Select TimingAndMemoryMatrix explicitly to benchmark";
    std::string cpu = "unknown", line;
    std::ifstream info("/proc/cpuinfo");
    while (std::getline(info, line))
        if (line.rfind("model name", 0) == 0) {
            cpu = line;
            break;
        }
    std::cout << "PROJECTION_BENCH_ENV "
              << Report{{"compiler", __VERSION__},
                        {"cpu", cpu},
                        {"allocator", Allocator()},
                        {"batch", kBatch},
                        {"warmups_per_mode", 2},
                        {"samples_per_mode", 7},
#ifdef __OPTIMIZE__
                        {"optimized_test", true}
#else
                        {"optimized_test", false}
#endif
                 }
                     .dump()
              << std::endl;
    for (size_t rows : {16384, 131072, 524288}) {
        Fixture fixture(rows);
        for (bool arithmetic : {false, true})
            for (size_t stride : {1, 100}) {
                for (int round = 0; round < 2; ++round)
                    for (bool bounded : {false, true}) {
                        Sample sample;
                        ASSERT_NO_FATAL_FAILURE(fixture.Run(
                            bounded, arithmetic, stride, false, &sample));
                    }
                std::vector<double> times[2], prepare[2], eval[2];
                for (int round = 0; round < 7; ++round)
                    for (int turn = 0; turn < 2; ++turn) {
                        int mode = (round + turn) % 2;
                        Sample sample;
                        ASSERT_NO_FATAL_FAILURE(fixture.Run(
                            mode, arithmetic, stride, false, &sample));
                        times[mode].push_back(sample.Total());
                        prepare[mode].push_back(sample.prepare_us);
                        eval[mode].push_back(sample.eval_us);
                    }
                Report report = {
                    {"rows", rows},
                    {"stride", stride},
                    {"shape", arithmetic ? "arithmetic" : "passthrough"},
                    {"batch", kBatch},
                    {"input_bitmap_bytes", 2 * ((rows + 7) / 8)}};
                for (int mode = 0; mode < 2; ++mode) {
                    Sample memory;
                    ASSERT_NO_FATAL_FAILURE(
                        fixture.Run(mode, arithmetic, stride, true, &memory));
                    report[mode ? "bounded" : "legacy"] = {
                        {"total", Times(times[mode])},
                        {"prepare", Times(prepare[mode])},
                        {"eval", Times(eval[mode])},
                        {"memory", Memory(memory)}};
                }
                std::cout << "PROJECTION_BENCH " << report.dump() << std::endl;
            }
    }
}

}  // namespace
}  // namespace milvus::exec
