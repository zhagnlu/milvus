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

// Compare the real FilterBits operator, including query-local preparation.
// No timing threshold is asserted: correctness/routing are CI gates, timings
// are opt-in observations. See unittest/scalar_execution_perf.md.
#include <gtest/gtest.h>

#include <arrow/builder.h>
#include <algorithm>
#include <array>
#include "exec/expression/ScalarBatchKernels.h"
#include "exec/expression/ScalarStringKernels.h"
#include "exec/expression/ScalarObjectKernels.h"
#include "exec/expression/Element.h"
#include "exec/expression/function/impl/StringFunctions.h"
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <roaring/roaring64map.hh>
#include "common/RoaringMembership.h"
#include <ctime>
#include <map>
#include <nlohmann/json.hpp>

#include "common/Common.h"
#include "exec/operator/FilterBitsNode.h"
#include "exec/expression/JsonContainsExpr.h"
// Include the legacy operator before introducing exec::plan.
#include "exec/operator/FilterBitsExecution.h"
#include "segcore/json_stats/JsonKeyStats.h"
#include "segcore/storagev2translator/JsonStatsTranslator.h"
#include "storage/InsertData.h"
#include "storage/PayloadReader.h"
#include "test_utils/Constants.h"
#include "test_utils/cachinglayer_test_utils.h"
#include "test_utils/storage_test_utils.h"

namespace milvus::exec {
class ScalarExecutionPerformanceAccess {
 public:
    static nlohmann::json
    Describe(const PhyFilterBitsNode& node, bool scalar) {
        nlohmann::json out;
        if (scalar) {
            const auto& p = node.scalar_filter_->prepared();
            out["granularity"] =
                p.granularity() == plan::ExecutionGranularity::WholeSegment
                    ? "whole_segment"
                    : "batch";
            out["backends"] = nlohmann::json::array();
            for (auto backend : p.predicate_backends()) {
                out["backends"].push_back(
                    backend == plan::PredicateBackend::Raw ? "raw"
                    : backend == plan::PredicateBackend::JsonStats
                        ? "json_stats"
                        : "scalar_index");
            }
        } else {
            out["granularity"] =
                node.exprs_->CanExecuteAllAtOnce() ? "whole_segment" : "batch";
            out["backends"] = nlohmann::json::array();
            std::function<void(const std::shared_ptr<Expr>&)> visit =
                [&](const auto& expr) {
                    if (auto segment =
                            std::dynamic_pointer_cast<SegmentExpr>(expr)) {
                        const auto path = segment->GetExecPath();
                        out["backends"].push_back(
                            path == ExprExecPath::RawData     ? "raw"
                            : path == ExprExecPath::JsonStats ? "json_stats"
                                                              : "scalar_index");
                    }
                    for (const auto& child : expr->GetInputsRef()) visit(child);
                };
            for (const auto& expr : node.exprs_->exprs()) visit(expr);
        }
        return out;
    }

    static void
    CheckSingle(const PhyFilterBitsNode& node, bool scalar, int layout) {
        ASSERT_EQ(bool(node.scalar_filter_), scalar) << "unexpected fallback";
        ASSERT_EQ(bool(node.exprs_), !scalar);
        EXPECT_FALSE(node.enable_expr_cache_);
        const auto expected = layout == 3   ? "json_stats"
                              : layout == 2 ? "scalar_index"
                                            : "raw";
        const auto actual = Describe(node, scalar);
        EXPECT_EQ(actual.at("granularity"),
                  layout >= 2 ? "whole_segment" : "batch");
        if (layout >= 2)
            EXPECT_FALSE(actual.at("backends").empty());
        for (const auto& backend : actual.at("backends"))
            EXPECT_EQ(backend, expected);
        if (scalar) {
            for (auto route :
                 node.scalar_filter_->prepared().predicate_routes())
                EXPECT_EQ(route,
                          layout >= 2 ? plan::PredicateRoute::ExactIndex
                                      : plan::PredicateRoute::Raw);
        }
    }

    static void
    Check(const PhyFilterBitsNode& node,
          bool scalar,
          int indexed_fields,
          bool compound,
          bool direct_raw = false) {
        ASSERT_EQ(bool(node.scalar_filter_), scalar) << "unexpected fallback";
        ASSERT_EQ(bool(node.exprs_), !scalar);
        EXPECT_FALSE(node.enable_expr_cache_);
        if (!scalar) {
            EXPECT_EQ(node.exprs_->CanExecuteAllAtOnce(), indexed_fields == 2);
            return;
        }
        const auto& prepared = node.scalar_filter_->prepared();
        const auto& routes = prepared.predicate_routes();
        const auto predicates = direct_raw ? 0 : (compound ? 3 : 1);
        ASSERT_EQ(routes.size(), predicates);
        const auto exact = std::count(
            routes.begin(), routes.end(), plan::PredicateRoute::ExactIndex);
        EXPECT_EQ(
            exact,
            indexed_fields == 2 ? predicates : (compound ? indexed_fields : 0));
        EXPECT_EQ(
            std::count(routes.begin(), routes.end(), plan::PredicateRoute::Raw),
            predicates - exact);
        EXPECT_EQ(prepared.granularity(),
                  indexed_fields == 2 ? plan::ExecutionGranularity::WholeSegment
                                      : plan::ExecutionGranularity::Batch);
    }
};

namespace {
// Wall latency remains the default. Optional thread CPU time excludes periods
// when this synchronous query thread is descheduled; it does not measure I/O
// wait, worker-thread CPU, or remove frequency/cache interference.
struct Clock {
    using time_point = std::chrono::steady_clock::time_point;
    static inline bool thread_cpu = false;
    static time_point
    now() {
        if (!thread_cpu)
            return std::chrono::steady_clock::now();
        timespec value{};
        if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) != 0)
            std::
                abort();  // A benchmark cannot publish timestamps from failure.
        return time_point(std::chrono::nanoseconds(
            value.tv_sec * int64_t{1000000000} + value.tv_nsec));
    }
};
using Json = nlohmann::json;

struct Sample {
    double prepare_us;
    double eval_us;
    double total_us;
};

double
Micros(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

// ASSERT at the call site, rather than silently accepting malformed settings.
int64_t
Setting(const char* name, int64_t fallback) {
    const char* value = std::getenv(name);
    if (!value)
        return fallback;
    int64_t parsed = -1;
    const auto end = value + std::char_traits<char>::length(value);
    const auto result = std::from_chars(value, end, parsed);
    return result.ec == std::errc{} && result.ptr == end ? parsed : -1;
}

Json
Summarize(std::vector<double> values) {
    auto sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const size_t n = sorted.size();
    const double median =
        n % 2 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
    return {{"median_us", median},
            {"p95_us", sorted[static_cast<size_t>(std::ceil(n * .95)) - 1]},
            {"min_us", sorted.front()},
            {"max_us", sorted.back()},
            {"samples_us", values}};
}

Json
Summarize(const std::vector<Sample>& samples) {
    std::vector<double> prepare, eval, total, cleanup;
    for (const auto& s : samples) {
        prepare.push_back(s.prepare_us);
        eval.push_back(s.eval_us);
        total.push_back(s.total_us);
        cleanup.push_back(s.total_us - s.prepare_us - s.eval_us);
    }
    return {{"prepare", Summarize(prepare)},
            {"eval", Summarize(eval)},
            {"total", Summarize(total)},
            {"cleanup", Summarize(cleanup)}};
}

const std::vector<std::string>&
PerformanceShapes() {
    static const std::vector<std::string> shapes{"float_in",
                                                 "double_in",
                                                 "bool_in",
                                                 "mixed_compare",
                                                 "float_arith",
                                                 "double_arith",
                                                 "roaring",
                                                 "bloom_int",
                                                 "bloom_string",
                                                 "array_element",
                                                 "array_length",
                                                 "array_any",
                                                 "array_all",
                                                 "array_eq",
                                                 "legacy_call",
                                                 "legacy_starts_with",
                                                 "timestamp_arith",
                                                 "compound",
                                                 "unary",
                                                 "eq",
                                                 "ne",
                                                 "le",
                                                 "gt",
                                                 "ge",
                                                 "range",
                                                 "in",
                                                 "arith",
                                                 "arith_sub",
                                                 "arith_mul",
                                                 "arith_div",
                                                 "arith_mod",
                                                 "compare",
                                                 "in64",
                                                 "in1024",
                                                 "is_null",
                                                 "string_in",
                                                 "string_in_long",
                                                 "string_range",
                                                 "string_eq",
                                                 "string_prefix",
                                                 "string_like",
                                                 "json_unary",
                                                 "json_range",
                                                 "json_in",
                                                 "json_contains",
                                                 "json_contains_all",
                                                 "json_contains_any",
                                                 "json_exists",
                                                 "json_arith",
                                                 "json_rare"};
    return shapes;
}

class ScalarExecutionPerformanceTest : public ::testing::TestWithParam<int> {
 protected:
    void
    Measure(const std::string& shape, bool emit = false);
    void
    SetUp() override {
        const char* clock = std::getenv("MILVUS_SCALAR_PERF_CLOCK");
        const std::string clock_name = clock ? clock : "wall";
        ASSERT_TRUE(clock_name == "wall" || clock_name == "thread_cpu");
        Clock::thread_cpu = clock_name == "thread_cpu";
        previous_reorder_ = OPTIMIZE_EXPR_ENABLED.exchange(true);
    }
    void
    TearDown() override {
        OPTIMIZE_EXPR_ENABLED.store(previous_reorder_);
    }

    bool
    IsStructured() const {
        return shape_.starts_with("json_") || shape_.starts_with("string_");
    }

    bool
    IsTyped() const {
        return shape_.starts_with("float_") || shape_.starts_with("double_") ||
               shape_ == "bool_in" || shape_ == "mixed_compare" ||
               shape_ == "timestamp_arith" || shape_.starts_with("array_") ||
               shape_ == "roaring" || shape_.starts_with("bloom_") ||
               shape_ == "legacy_call" || shape_ == "legacy_starts_with";
    }

    void
    LoadTyped(int64_t count) {
        ASSERT_EQ(GetParam(), 0);
        rows_ = count;
        const bool array = shape_.starts_with("array_");
        const bool bloom = shape_.starts_with("bloom_");
        const auto type =
            array                           ? DataType::ARRAY
            : shape_.starts_with("float_")  ? DataType::FLOAT
            : shape_.starts_with("double_") ? DataType::DOUBLE
            : shape_ == "timestamp_arith"   ? DataType::TIMESTAMPTZ
            : shape_ == "bool_in"           ? DataType::BOOL
            : shape_ == "bloom_string" || shape_ == "legacy_starts_with"
                ? DataType::VARCHAR
                : DataType::INT64;
        Json probes;
        std::string blob;
        if (bloom) {
            const auto path =
                std::filesystem::path(__FILE__).parent_path() /
                "../../../unittest/testdata/bloom/golden_vectors.json";
            std::ifstream stream(path);
            ASSERT_TRUE(stream.good());
            Json fixture;
            stream >> fixture;
            for (const auto& test : fixture.at("cases")) {
                if (test.at("name") != (type == DataType::VARCHAR
                                            ? "small_string_set"
                                            : "small_int64_set"))
                    continue;
                const auto hex = test.at("blob_hex").get<std::string>();
                for (size_t i = 0; i < hex.size(); i += 2)
                    blob.push_back(static_cast<char>(
                        std::stoul(hex.substr(i, 2), nullptr, 16)));
                for (const auto& probe : test.at("probes"))
                    if (probe.at("kind") ==
                        (type == DataType::VARCHAR ? "string" : "int64"))
                        probes.push_back(probe);
            }
            ASSERT_FALSE(probes.empty());
        }
        schema_ = std::make_shared<Schema>();
        auto pk = schema_->AddDebugField("pk", DataType::INT64);
        schema_->set_primary_field_id(pk);
        a_ = array
                 ? schema_->AddDebugArrayField("a", DataType::INT64, nullable_)
                 : schema_->AddDebugField("a", type, nullable_);
        b_ = schema_->AddDebugField("b", DataType::DOUBLE, nullable_);
        auto data = segcore::DataGen(schema_, count);
        auto number = [&](int64_t i) { return double(Value(i, false)) + .25; };
        expected_ = TargetBitmap(count, true);
        for (int64_t i = 0; i < count; ++i) {
            const auto n = Value(i, false);
            bool hit = false;
            if (bloom)
                hit = probes[i % probes.size()].at("expect").get<bool>();
            else if (shape_ == "bool_in")
                hit = n % 2 != 0;
            else if (shape_ == "mixed_compare")
                hit = Valid(i, true) && n < double(Value(i, true)) + .25;
            else if (shape_ == "array_length")
                hit = n % 13 == 0;
            else if (shape_ == "array_element")
                hit = n % 13 != 0 && n < 10;
            else if (shape_ == "array_any")
                hit = n % 13 != 0 &&
                      ((n == 1 || n == 5 || n == 10 || n == 90) ||
                       ((n + 1) % 100 == 1 || (n + 1) % 100 == 5 ||
                        (n + 1) % 100 == 10 || (n + 1) % 100 == 90));
            else if ((shape_ == "array_all" || shape_ == "array_eq"))
                hit = n == 10;
            else if (shape_ == "legacy_starts_with")
                hit = ("value_" + std::to_string(n)).starts_with("value_1");
            else if (shape_ == "legacy_call" || shape_ == "timestamp_arith")
                hit = n < 10;
            else if (shape_ == "float_arith" || shape_ == "double_arith")
                hit = number(i) + 2.5 < 12.5;
            else
                hit = n == 1 || n == 5 || n == 10 || n == 90;
            expected_.set(i, !(Valid(i, false) && hit));
        }
        for (auto& field : *data.raw_->mutable_fields_data()) {
            const FieldId id(field.field_id());
            if (id != a_ && id != b_)
                continue;
            FixedVector<bool> valid(count);
            for (int64_t i = 0; i < count; ++i) valid[i] = Valid(i, id == b_);
            auto fill = [&]<typename T>(const auto& value) {
                FixedVector<T> values(count);
                for (int64_t i = 0; i < count; ++i) values[i] = value(i);
                auto replacement = segcore::CreateDataArrayFrom(
                    values.data(), valid.data(), count, (*schema_)[id]);
                field.Swap(replacement.get());
            };
            if (id == b_)
                fill.template operator()<double>(
                    [&](int64_t i) { return double(Value(i, true)) + .25; });
            else if (array) {
                // Keep the old/new performance oracle in their shared domain.
                // Element NULL semantics have independent migration UT coverage.
                field.clear_valid_data();
                auto* scalar = field.mutable_scalars();
                scalar->clear_valid_data();
                auto* arrays = scalar->mutable_array_data();
                arrays->clear_data();
                for (int64_t i = 0; i < count; ++i) {
                    scalar->add_valid_data(valid[i]);
                    auto* row = arrays->add_data();
                    auto* values = row->mutable_long_data();
                    const auto n = Value(i, false);
                    if (n % 13 != 0) {
                        values->add_data(n);
                        values->add_data((n + 1) % 100);
                    }
                }
            } else if (type == DataType::FLOAT)
                fill.template operator()<float>(number);
            else if (type == DataType::DOUBLE)
                fill.template operator()<double>(number);
            else if (type == DataType::BOOL)
                fill.template operator()<bool>(
                    [&](int64_t i) { return Value(i, false) % 2 != 0; });
            else if (type == DataType::VARCHAR)
                fill.template operator()<std::string>([&](int64_t i) {
                    if (shape_ == "legacy_starts_with")
                        return "value_" + std::to_string(Value(i, false));
                    return probes[i % probes.size()]
                        .at("string")
                        .get<std::string>();
                });
            else
                fill.template operator()<int64_t>([&](int64_t i) {
                    return bloom ? std::stoll(probes[i % probes.size()]
                                                  .at("int64")
                                                  .get<std::string>())
                                 : Value(i, false);
                });
        }
        segment_ = CreateSealedWithFieldDataLoaded(schema_, data);
        auto integer = [](int64_t n) {
            proto::plan::GenericValue v;
            v.set_int64_val(n);
            return v;
        };
        auto floating = [](double n) {
            proto::plan::GenericValue v;
            v.set_float_val(n);
            return v;
        };
        const expr::ColumnInfo column =
            array ? expr::ColumnInfo(a_, type, DataType::INT64, {}, nullable_)
                  : expr::ColumnInfo(a_, type, {}, nullable_);
        if (bloom)
            predicate_ = std::make_shared<expr::BloomFilterExpr>(column, blob);
        else if (shape_ == "roaring") {
            roaring::Roaring64Map bitmap;
            for (auto n : {1, 5, 10, 90}) bitmap.add(uint64_t(n));
            std::string body(bitmap.getSizeInBytes(true), '\0');
            bitmap.write(body.data(), true);
            std::string envelope(RoaringMembership::kHeaderSize, '\0');
            envelope.replace(0, 4, "MRB1");
            auto put = [&](size_t off, uint64_t value, size_t bytes) {
                for (size_t i = 0; i < bytes; ++i)
                    envelope[off + i] = char(value >> (8 * i));
            };
            put(4, RoaringMembership::kVersion, 2);
            put(6, RoaringMembership::kFormatPortableRoaring64, 2);
            put(8, bitmap.cardinality(), 8);
            put(16, body.size(), 8);
            predicate_ = std::make_shared<expr::RoaringFilterExpr>(
                column, RoaringMembership::Parse(envelope + body));
        } else if (shape_ == "mixed_compare")
            predicate_ =
                std::make_shared<expr::CompareExpr>(a_,
                                                    b_,
                                                    DataType::INT64,
                                                    DataType::DOUBLE,
                                                    proto::plan::LessThan);
        else if (shape_ == "timestamp_arith") {
            proto::plan::Interval second;
            second.set_seconds(1);
            predicate_ = std::make_shared<expr::TimestamptzArithCompareExpr>(
                column,
                proto::plan::Add,
                second,
                proto::plan::LessThan,
                integer(1000010));
        } else if (shape_ == "array_length")
            predicate_ = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
                column,
                proto::plan::LessThan,
                proto::plan::ArrayLength,
                integer(2),
                integer(0));
        else if (shape_ == "array_element") {
            auto selected = column;
            selected.nested_path_ = {"0"};
            predicate_ = std::make_shared<expr::UnaryRangeFilterExpr>(
                selected, proto::plan::LessThan, integer(10));
        } else if (shape_ == "array_eq") {
            proto::plan::GenericValue value;
            value.mutable_array_val()->set_same_type(true);
            *value.mutable_array_val()->add_array() = integer(10);
            *value.mutable_array_val()->add_array() = integer(11);
            predicate_ = std::make_shared<expr::UnaryRangeFilterExpr>(
                column, proto::plan::Equal, value);
        } else if (array)
            predicate_ = std::make_shared<expr::JsonContainsExpr>(
                column,
                shape_ == "array_all"
                    ? proto::plan::JSONContainsExpr_JSONOp_ContainsAll
                    : proto::plan::JSONContainsExpr_JSONOp_ContainsAny,
                true,
                shape_ == "array_all"
                    ? std::vector<proto::plan::GenericValue>{integer(10),
                                                             integer(11)}
                    : std::vector<proto::plan::GenericValue>{
                          integer(1), integer(5), integer(10), integer(90)});
        else if (shape_ == "legacy_starts_with") {
            proto::plan::GenericValue prefix;
            prefix.set_string_val("value_1");
            predicate_ = std::make_shared<expr::CallExpr>(
                "starts_with",
                std::vector<expr::TypedExprPtr>{
                    std::make_shared<expr::ColumnExpr>(column),
                    std::make_shared<expr::ValueExpr>(prefix)},
                expression::function::StartsWithVarchar);
        } else if (shape_ == "legacy_call")
            predicate_ = std::make_shared<expr::CallExpr>(
                "less_than_ten",
                std::vector<expr::TypedExprPtr>{
                    std::make_shared<expr::ColumnExpr>(column)},
                +[](const RowVector& args, VectorPtr& out) {
                    auto values = ReadOnlyVectorView::CopyOf(args.child(0));
                    TargetBitmap truth(args.size(), false),
                        valid(args.size(), false);
                    for (size_t i = 0; i < args.size(); ++i)
                        if (auto value = values.TryValueAt<int64_t>(i)) {
                            valid.set(i);
                            truth.set(i, *value < 10);
                        }
                    out = std::make_shared<ColumnVector>(std::move(truth),
                                                         std::move(valid));
                });
        else if (shape_.ends_with("arith"))
            predicate_ = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
                column,
                proto::plan::LessThan,
                proto::plan::Add,
                floating(12.5),
                floating(2.5));
        else {
            auto terms =
                std::vector<proto::plan::GenericValue>{floating(1.25),
                                                       floating(5.25),
                                                       floating(10.25),
                                                       floating(90.25)};
            if (type == DataType::BOOL) {
                proto::plan::GenericValue v;
                v.set_bool_val(true);
                terms = {v};
            }
            predicate_ = std::make_shared<expr::TermFilterExpr>(column, terms);
        }
    }

    void
    AttachStats(const std::vector<std::string>& documents) {
        // Build/upload/load real JSON stats once, outside every timed query.
        // Both execution modes see the same immutable generation and warm cells.
        arrow::BinaryBuilder builder;
        for (size_t i = 0; i < documents.size(); ++i) {
            ASSERT_TRUE((Valid(i, false) ? builder.Append(documents[i])
                                         : builder.AppendNull())
                            .ok());
        }
        std::shared_ptr<arrow::Array> array;
        ASSERT_TRUE(builder.Finish(&array).ok());
        auto field_data = std::make_shared<FieldData<milvus::Json>>(
            DataType::JSON, nullable_);
        field_data->FillFieldData(array);
        auto payload = std::make_shared<storage::PayloadReader>(field_data);
        storage::InsertData insert(payload);
        proto::schema::FieldSchema schema;
        schema.set_data_type(proto::schema::DataType::JSON);
        schema.set_fieldid(a_.get());
        schema.set_nullable(nullable_);
        const auto uid =
            static_cast<int64_t>(segcore::NextSegmentInstanceUid());
        storage::FieldDataMeta field_meta{7101, 7102, uid, a_.get(), schema};
        storage::IndexMeta index_meta{uid, a_.get(), uid, 1};
        insert.SetFieldDataMeta(field_meta);
        insert.SetTimestamps(0, 100);
        auto bytes = insert.Serialize(storage::Remote);
        const auto root_path =
            TestLocalPath + "/scalar-perf-stats-" + std::to_string(uid);
        auto config = std::make_shared<milvus_storage::api::Properties>();
        (*config)[PROPERTY_FS_STORAGE_TYPE] = std::string("local");
        (*config)[PROPERTY_FS_ROOT_PATH] = root_path;
        auto cm = storage::CreateChunkManager(config);
        auto fs = storage::InitArrowFileSystem(config);
        const auto log = root_path + "/insert";
        cm->Write(log, bytes.data(), bytes.size());
        storage::FileManagerContext context(field_meta, index_meta, cm, fs);
        Config build_config;
        build_config[INSERT_FILES_KEY] = std::vector<std::string>{log};
        index::JsonKeyStats stats(context, false);
        stats.Build(build_config);
        const auto uploaded = stats.Upload(build_config);
        std::vector<std::string> files;
        for (const auto& file : uploaded.Files())
            files.push_back(file.file_name);
        auto info = std::make_shared<proto::indexcgo::LoadJsonKeyIndexInfo>();
        info->set_collectionid(7101);
        info->set_partitionid(7102);
        info->set_fieldid(a_.get());
        info->set_buildid(uid);
        info->set_version(1);
        *info->mutable_schema() = schema;
        for (const auto& file : files) info->add_files(file);
        info->set_load_priority(proto::common::LoadPriority::HIGH);
        info->set_base_path(storage::GenRemoteJsonStatsPathPrefix(
            cm, uid, 1, 7101, 7102, uid, a_.get()));
        info->set_warmup_policy("disable");
        auto translator =
            std::make_unique<segcore::storagev2translator::JsonStatsTranslator>(
                segcore::storagev2translator::JsonStatsLoadInfo{
                    uid, uid, "scalar-perf-stats"},
                info,
                cm,
                fs);
        auto slot =
            cachinglayer::Manager::GetInstance()
                .CreateCacheSlot<index::JsonKeyStats>(std::move(translator));
        auto* sealed =
            dynamic_cast<segcore::ChunkedSegmentSealedImpl*>(segment_.get());
        ASSERT_NE(sealed, nullptr);
        sealed->SetJsonStatsForTesting(a_, std::move(slot));
    }

    void
    LoadStructured(int64_t count) {
        rows_ = count;
        const bool json = shape_.starts_with("json_");
        schema_ = std::make_shared<Schema>();
        const auto pk = schema_->AddDebugField("pk", DataType::INT64);
        a_ = schema_->AddDebugField(json ? "j" : "s",
                                    json ? DataType::JSON : DataType::VARCHAR,
                                    nullable_);
        schema_->set_primary_field_id(pk);
        auto data = segcore::DataGen(schema_, rows_);
        std::vector<std::string> values;
        FixedVector<bool> valid(rows_);
        expected_ = TargetBitmap(rows_, true);
        for (int64_t i = 0; i < rows_; ++i) {
            const auto n = Value(i, false);
            const auto text =
                (shape_ == "string_in_long" ? std::string(256, 'x')
                                            : std::string{}) +
                "value_" + std::to_string(n);
            bool matches = false;
            valid[i] = Valid(i, false);
            if (!json) {
                values.push_back(text);
                matches = shape_.starts_with("string_in")
                              ? (n == 1 || n == 5 || n == 10 || n == 90)
                          : shape_ == "string_range"
                              ? (text >= "value_10" && text < "value_20")
                          : shape_ == "string_eq" ? n == 10
                          : shape_ == "string_prefix"
                              ? text.starts_with("value_1")
                              : text.find('1') != std::string::npos;
            } else {
                const bool exists = i % 23 != 0;
                const bool typed = exists && i % 29 != 0 && i % 31 != 0;
                Json doc = Json::object();
                if (exists) {
                    if (i % 29 == 0) {
                        doc["n"] = nullptr;
                        doc["arr"] = nullptr;
                    } else if (i % 31 == 0) {
                        doc["n"] = "wrong_type";
                        doc["arr"] = false;
                    } else {
                        doc["n"] = n;
                        doc["arr"] = Json::array({n, (n + 1) % 100});
                    }
                }
                if (i % 101 == 0)
                    doc["rare"] = n;
                values.push_back(doc.dump());
                const auto contains = [&](int64_t x) {
                    return n == x || (n + 1) % 100 == x;
                };
                if (shape_ == "json_unary" || shape_ == "json_arith")
                    matches = typed && n < 10;
                else if (shape_ == "json_range")
                    matches = typed && n >= 10 && n < 20;
                else if (shape_ == "json_in")
                    matches = typed && (n == 1 || n == 5 || n == 10 || n == 90);
                else if (shape_ == "json_contains")
                    matches = typed && contains(10);
                else if (shape_ == "json_contains_all")
                    matches = typed && contains(10) && contains(11);
                else if (shape_ == "json_contains_any")
                    matches = typed && (contains(1) || contains(5) ||
                                        contains(10) || contains(90));
                else if (shape_ == "json_exists")
                    matches = exists && i % 29 != 0;
                else if (shape_ == "json_rare")
                    matches = i % 101 == 0 && n < 10;
            }
            expected_.set(i, !(valid[i] && matches));
        }
        for (auto& field : *data.raw_->mutable_fields_data()) {
            if (field.field_id() != a_.get())
                continue;
            auto replacement = segcore::CreateDataArrayFrom(
                values.data(), valid.data(), rows_, (*schema_)[a_]);
            field.Swap(replacement.get());
        }
        segment_ = CreateSealedWithFieldDataLoaded(schema_, data);
        if (GetParam() == 3)
            ASSERT_NO_FATAL_FAILURE(AttachStats(values));
        const auto integer = [](int64_t n) {
            proto::plan::GenericValue v;
            v.set_int64_val(n);
            return v;
        };
        if (!json) {
            const expr::ColumnInfo column(a_, DataType::VARCHAR, {}, nullable_);
            if (shape_.starts_with("string_in")) {
                std::vector<proto::plan::GenericValue> terms;
                for (int n : {1, 5, 10, 90}) {
                    proto::plan::GenericValue value;
                    value.set_string_val((shape_ == "string_in_long"
                                              ? std::string(256, 'x')
                                              : std::string{}) +
                                         "value_" + std::to_string(n));
                    terms.push_back(value);
                }
                predicate_ =
                    std::make_shared<expr::TermFilterExpr>(column, terms);
                return;
            }
            if (shape_ == "string_range") {
                proto::plan::GenericValue lower, upper;
                lower.set_string_val("value_10");
                upper.set_string_val("value_20");
                predicate_ = std::make_shared<expr::BinaryRangeFilterExpr>(
                    column, lower, upper, true, false);
                return;
            }
            proto::plan::GenericValue value;
            value.set_string_val(shape_ == "string_eq"       ? "value_10"
                                 : shape_ == "string_prefix" ? "value_1"
                                                             : "%1%");
            predicate_ = std::make_shared<expr::UnaryRangeFilterExpr>(
                expr::ColumnInfo(a_, DataType::VARCHAR, {}, nullable_),
                shape_ == "string_eq"       ? proto::plan::Equal
                : shape_ == "string_prefix" ? proto::plan::PrefixMatch
                                            : proto::plan::Match,
                value);
            return;
        }
        const expr::ColumnInfo column(
            a_,
            DataType::JSON,
            {shape_.starts_with("json_contains") ? "arr"
             : shape_ == "json_rare"             ? "rare"
                                                 : "n"},
            nullable_);
        if (shape_.starts_with("json_contains")) {
            std::vector<proto::plan::GenericValue> terms{integer(10)};
            auto op = proto::plan::JSONContainsExpr_JSONOp_Contains;
            if (shape_ == "json_contains_all") {
                op = proto::plan::JSONContainsExpr_JSONOp_ContainsAll;
                terms.push_back(integer(11));
            }
            if (shape_ == "json_contains_any") {
                op = proto::plan::JSONContainsExpr_JSONOp_ContainsAny;
                terms = {integer(1), integer(5), integer(10), integer(90)};
            }
            predicate_ = std::make_shared<expr::JsonContainsExpr>(
                column, op, true, terms);
        } else if (shape_ == "json_range")
            predicate_ = std::make_shared<expr::BinaryRangeFilterExpr>(
                column, integer(10), integer(20), true, false);
        else if (shape_ == "json_in")
            predicate_ = std::make_shared<expr::TermFilterExpr>(
                column,
                std::vector<proto::plan::GenericValue>{
                    integer(1), integer(5), integer(10), integer(90)});
        else if (shape_ == "json_exists")
            predicate_ = std::make_shared<expr::ExistsExpr>(column);
        else if (shape_ == "json_arith")
            predicate_ = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
                column,
                proto::plan::LessThan,
                proto::plan::Add,
                integer(12),
                integer(2));
        else
            predicate_ = std::make_shared<expr::UnaryRangeFilterExpr>(
                column, proto::plan::LessThan, integer(10));
    }

    void
    Load(int64_t count) {
        if (IsTyped()) {
            ASSERT_NO_FATAL_FAILURE(LoadTyped(count));
            return;
        }
        if (IsStructured()) {
            ASSERT_NO_FATAL_FAILURE(LoadStructured(count));
            return;
        }
        rows_ = count;
        schema_ = std::make_shared<Schema>();
        const auto pk = schema_->AddDebugField("pk", DataType::INT64);
        a_ = schema_->AddDebugField("a", DataType::INT64, nullable_);
        b_ = schema_->AddDebugField("b", DataType::INT64, nullable_);
        schema_->set_primary_field_id(pk);
        auto data = segcore::DataGen(schema_, rows_);
        expected_ = TargetBitmap(rows_, true);
        for (int64_t i = 0; i < rows_; ++i) {
            const auto a = Value(i, false), b = Value(i, true);
            bool value;
            if (shape_ == "eq")
                value = a == 10;
            else if (shape_ == "ne")
                value = a != 10;
            else if (shape_ == "le")
                value = a <= 10;
            else if (shape_ == "gt")
                value = a > 10;
            else if (shape_ == "ge")
                value = a >= 10;
            else if (shape_ == "range")
                value = a >= 10 && a < 20;
            else if (shape_ == "in")
                value = a == 1 || a == 5 || a == 10 || a == 90;
            else if (shape_ == "in64" || shape_ == "in1024")
                value = a < (shape_ == "in64" ? 64 : 1024);
            else if (shape_ == "is_null")
                value = !Valid(i, false);
            else if (shape_ == "arith")
                value = a + 2 < 12;
            else if (shape_ == "arith_sub")
                value = a - 2 < 12;
            else if (shape_ == "arith_mul")
                value = a * 2 < 12;
            else if (shape_ == "arith_div")
                value = a / 2 < 12;
            else if (shape_ == "arith_mod")
                value = a % 7 < 3;
            else if (shape_ == "compare")
                value = Valid(i, true) && a < b;
            else
                value = a < 10 &&
                        (!compound_ || (Valid(i, true) && b > 10 && a != 8));
            const bool matches =
                (shape_ == "is_null" || Valid(i, false)) && value;
            expected_.set(i, !matches);
        }
        for (auto& field : *data.raw_->mutable_fields_data()) {
            const FieldId id(field.field_id());
            if (id != a_ && id != b_)
                continue;
            FixedVector<int64_t> values(rows_);
            FixedVector<bool> valid(rows_);
            for (int64_t i = 0; i < rows_; ++i) {
                values[i] = Value(i, id == b_);
                valid[i] = Valid(i, id == b_);
            }
            auto replacement = segcore::CreateDataArrayFrom(
                values.data(), valid.data(), rows_, (*schema_)[id]);
            field.Swap(replacement.get());
        }
        segment_ = CreateSealedWithFieldDataLoaded(schema_, data);
        if (GetParam() >= 1)
            LoadIndex(b_);
        if (GetParam() == 2)
            LoadIndex(a_);
        auto unary = [this](FieldId field, auto op, int64_t value) {
            proto::plan::GenericValue v;
            v.set_int64_val(value);
            return std::make_shared<expr::UnaryRangeFilterExpr>(
                expr::ColumnInfo(field, DataType::INT64, {}, nullable_), op, v);
        };
        if (!compound_) {
            auto v = [](int64_t n) {
                proto::plan::GenericValue x;
                x.set_int64_val(n);
                return x;
            };
            const expr::ColumnInfo column(a_, DataType::INT64, {}, nullable_);
            if (shape_ == "range")
                predicate_ = std::make_shared<expr::BinaryRangeFilterExpr>(
                    column, v(10), v(20), true, false);
            else if (shape_ == "in")
                predicate_ = std::make_shared<expr::TermFilterExpr>(
                    column,
                    std::vector<proto::plan::GenericValue>{
                        v(1), v(5), v(10), v(90)});
            else if (shape_ == "in64" || shape_ == "in1024") {
                std::vector<proto::plan::GenericValue> terms;
                for (int64_t i = 0; i < (shape_ == "in64" ? 64 : 1024); ++i)
                    terms.push_back(v(i));
                predicate_ =
                    std::make_shared<expr::TermFilterExpr>(column, terms);
            } else if (shape_ == "is_null")
                predicate_ = std::make_shared<expr::NullExpr>(
                    column, proto::plan::NullExpr_NullOp_IsNull);
            else if (shape_.starts_with("arith"))
                predicate_ = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
                    column,
                    proto::plan::LessThan,
                    shape_ == "arith_sub"   ? proto::plan::Sub
                    : shape_ == "arith_mul" ? proto::plan::Mul
                    : shape_ == "arith_div" ? proto::plan::Div
                    : shape_ == "arith_mod" ? proto::plan::Mod
                                            : proto::plan::Add,
                    v(shape_ == "arith_mod" ? 3 : 12),
                    v(shape_ == "arith_mod" ? 7 : 2));
            else if (shape_ == "compare")
                predicate_ =
                    std::make_shared<expr::CompareExpr>(a_,
                                                        b_,
                                                        DataType::INT64,
                                                        DataType::INT64,
                                                        proto::plan::LessThan);
            else {
                auto op = proto::plan::LessThan;
                if (shape_ == "eq")
                    op = proto::plan::Equal;
                else if (shape_ == "ne")
                    op = proto::plan::NotEqual;
                else if (shape_ == "le")
                    op = proto::plan::LessEqual;
                else if (shape_ == "gt")
                    op = proto::plan::GreaterThan;
                else if (shape_ == "ge")
                    op = proto::plan::GreaterEqual;
                predicate_ = unary(a_, op, 10);
            }
            return;
        }
        auto first = std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And,
            unary(a_, proto::plan::LessThan, 10),
            unary(b_, proto::plan::GreaterThan, 10));
        predicate_ = std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And,
            first,
            unary(a_, proto::plan::NotEqual, 8));
    }

    int64_t
    Value(int64_t row, bool b) const {
        if (shape_ == "in64" || shape_ == "in1024")
            return row * 17 % 8192;
        return b ? ((row / 100) * 37 + row * 13) % 100 : (row * 17) % 100;
    }
    bool
    Valid(int64_t row, bool b) const {
        return !nullable_ || row % (b ? 89 : 97) != 0;
    }

    void
    LoadIndex(FieldId field) {
        FixedVector<int64_t> values(rows_);
        FixedVector<bool> valid(rows_);
        for (int64_t i = 0; i < rows_; ++i) {
            values[i] = Value(i, field == b_);
            valid[i] = Valid(i, field == b_);
        }
        const auto family = index_type_ == "bitmap" ? index::families::kBitmap
                                                    : index::families::kSort;
        auto index = BuildTestScalarIndex<int64_t>(
            family, rows_, values.data(), valid.data());
        segcore::LoadIndexInfo info;
        info.field_id = field.get();
        info.index_params = GenIndexParams(index.get(), family);
        SetTestIndexMetadata(info, *index, family);
        info.index = CreateTestIndex(
            "scalar-perf-" + std::to_string(field.get()), std::move(index));
        segment_->LoadIndex(info);
    }

    // Output remains alive until validation, which is outside every timer.
    void
    Run(bool scalar, int64_t batch, Sample* sample) {
        RowVectorPtr output;
        Clock::time_point prepared, evaluated;
        const auto start = Clock::now();
        {
            auto config = std::make_shared<QueryConfig>(
                std::unordered_map<std::string, std::string>{
                    {QueryConfig::kEnableExprExecutionV2,
                     scalar ? "true" : "false"},
                    {QueryConfig::kExprEvalBatchSize, std::to_string(batch)}});
            auto query = std::make_shared<QueryContext>("scalar-perf",
                                                        segment_.get(),
                                                        rows_,
                                                        Timestamp(42) << 18,
                                                        0,
                                                        0,
                                                        query::PlanOptions{},
                                                        std::move(config));
            query->set_enable_expr_cache(false);
            query->set_read_snapshot(segment_->CaptureReadSnapshot());
            auto task = Task::Create(
                "scalar-perf", ::milvus::plan::PlanFragment(), 0, query);
            DriverContext driver(task, 0, 0, 0, 0);
            PhyFilterBitsNode node(
                0,
                &driver,
                std::make_shared<::milvus::plan::FilterBitsNode>("filter",
                                                                 predicate_));
            prepared = Clock::now();
            output = node.GetOutput();
            evaluated = Clock::now();
            node.Close();
        }
        const auto end = Clock::now();
        *sample = {Micros(start, prepared),
                   Micros(prepared, evaluated),
                   Micros(start, end)};
        CheckOutput(output);
    }

    void
    CheckOutput(const RowVectorPtr& output) {
        ASSERT_NE(output, nullptr);
        ASSERT_EQ(output->childrens().size(), 1);
        auto col = std::dynamic_pointer_cast<ColumnVector>(output->child(0));
        ASSERT_TRUE(col && col->IsBitmap());
        ASSERT_EQ(col->size(), rows_);
        TargetBitmapView bits(col->GetRawData(), rows_);
        TargetBitmapView valid(col->GetValidRawData(), rows_);
        ASSERT_EQ(valid.count(), rows_);
        ASSERT_TRUE(bits == expected_) << "incorrect exclusion bitmap";
    }

    // Diagnostic control only: no production switch disables cancellation.
    // Both variants use the same new executor and normal Raw batch size.
    void
    RunCheckpointControl(bool checked,
                         int64_t batch,
                         Sample* sample,
                         size_t* calls = nullptr) {
        ASSERT_EQ(GetParam(), 0);
        RowVectorPtr output;
        Clock::time_point prepared, evaluated;
        const auto start = Clock::now();
        {
            auto query =
                std::make_shared<QueryContext>("scalar-checkpoint-control",
                                               segment_.get(),
                                               rows_,
                                               Timestamp(42) << 18,
                                               0,
                                               0,
                                               query::PlanOptions{});
            auto context = std::make_shared<segcore::reader::ReadContext>();
            if (calls) {
                context->checkpoint = [query = query.get(), calls] {
                    ++*calls;
                    checkCancellation(query);
                };
            } else if (checked) {
                context->checkpoint = [query = query.get()] {
                    checkCancellation(query);
                };
            }
            auto execution =
                FilterBitsExecution::TryCreate(predicate_,
                                               nullptr,
                                               segment_->CaptureReadSnapshot(),
                                               rows_,
                                               batch,
                                               context,
                                               schema_);
            ASSERT_TRUE(execution);
            ASSERT_EQ(execution->prepared().granularity(),
                      plan::ExecutionGranularity::Batch);
            prepared = Clock::now();
            output = execution->GetOutput();
            evaluated = Clock::now();
        }
        *sample = {Micros(start, prepared),
                   Micros(prepared, evaluated),
                   Micros(start, Clock::now())};
        CheckOutput(output);
    }

    void
    CheckRoute(bool scalar, int64_t batch) {
        auto query = std::make_shared<QueryContext>(
            "scalar-perf-route",
            segment_.get(),
            rows_,
            Timestamp(42) << 18,
            0,
            0,
            query::PlanOptions{},
            std::make_shared<QueryConfig>(
                std::unordered_map<std::string, std::string>{
                    {QueryConfig::kEnableExprExecutionV2,
                     scalar ? "true" : "false"},
                    {QueryConfig::kExprEvalBatchSize, std::to_string(batch)}}));
        query->set_enable_expr_cache(false);
        query->set_read_snapshot(segment_->CaptureReadSnapshot());
        auto task = Task::Create(
            "scalar-perf-route", ::milvus::plan::PlanFragment(), 0, query);
        DriverContext driver(task, 0, 0, 0, 0);
        PhyFilterBitsNode node(0,
                               &driver,
                               std::make_shared<::milvus::plan::FilterBitsNode>(
                                   "filter", predicate_));
        if (IsStructured() || IsTyped()) {
            ScalarExecutionPerformanceAccess::CheckSingle(
                node,
                scalar,
                !scalar && shape_ == "json_arith" && GetParam() == 3
                    ? 0
                    : GetParam());
        } else
            ScalarExecutionPerformanceAccess::Check(
                node,
                scalar,
                GetParam(),
                compound_,
                shape_.starts_with("arith") || shape_ == "compare");
        auto actual = ScalarExecutionPerformanceAccess::Describe(node, scalar);
        actual["effective_batches"] =
            actual.at("granularity") == "whole_segment"
                ? 1
                : (rows_ + batch - 1) / batch;
        routes_[scalar ? "new" : "old"] = std::move(actual);
        node.Close();
    }

    std::string index_type_ = "sorted";
    Json routes_;
    std::string shape_ = "compound";
    bool compound_ = true;
    bool nullable_ = true;
    bool previous_reorder_ = false;
    int64_t rows_ = 0;
    SchemaPtr schema_;
    FieldId a_{0}, b_{0};
    std::unique_ptr<segcore::SegmentSealed> segment_;
    expr::TypedExprPtr predicate_;
    TargetBitmap expected_;
};

TEST_P(ScalarExecutionPerformanceTest, CorrectnessAndRouting) {
    if (const char* expression = std::getenv("MILVUS_SCALAR_PERF_EXPRESSION"))
        shape_ = expression;
    else if (GetParam() == 3)
        shape_ = "json_contains";
    compound_ = shape_ == "compound";
    if (const char* index = std::getenv("MILVUS_SCALAR_PERF_INDEX_TYPE"))
        index_type_ = index;
    nullable_ = Setting("MILVUS_SCALAR_PERF_NULLABLE", 1) != 0;
    ASSERT_NO_FATAL_FAILURE(Load(3073));  // partial last batch and NULLs
    for (bool scalar : {false, true}) {
        ASSERT_NO_FATAL_FAILURE(CheckRoute(scalar, 1024));
        Sample sample;
        ASSERT_NO_FATAL_FAILURE(Run(scalar, 1024, &sample));
    }
}

TEST_P(ScalarExecutionPerformanceTest, DISABLED_CompareModes) {
    const char* expression = std::getenv("MILVUS_SCALAR_PERF_EXPRESSION");
    Measure(expression        ? expression
            : GetParam() == 3 ? "json_contains"
                              : "compound");
}

// Dev CLI forwards a gtest filter but no benchmark environment/disabled flag.
// An explicit name is required: normal wildcard UT runs skip timing work.
TEST_P(ScalarExecutionPerformanceTest, StringBatchMatrix) {
    if (GetParam() != 0 || ::testing::GTEST_FLAG(filter).find(
                               "StringBatchMatrix") == std::string::npos)
        GTEST_SKIP() << "Select StringBatchMatrix/Raw explicitly to measure";
    for (int pass = 0; pass < 3; ++pass) {
        std::vector<std::string> shapes{"string_eq",
                                        "string_range",
                                        "string_prefix",
                                        "string_like",
                                        "string_in",
                                        "string_in_long"};
        if (pass % 2)
            std::reverse(shapes.begin(), shapes.end());
        for (const auto& shape : shapes) {
            SCOPED_TRACE(shape + " pass=" + std::to_string(pass));
            ASSERT_NO_FATAL_FAILURE(Measure(shape, true));
        }
    }
}

// Explicit opt-in regression workload for path preparation. Raw uses the
// production-sized batch default; only a selected all-index plan is whole-source.
TEST_P(ScalarExecutionPerformanceTest, InPrepareMatrix) {
    if (::testing::GTEST_FLAG(filter).find("InPrepareMatrix") ==
            std::string::npos ||
        (GetParam() != 0 && GetParam() != 2))
        GTEST_SKIP();
    std::vector<std::string> shapes{"in", "in64", "in1024"};
    for (int pass = 0; pass < 3; ++pass) {
        if (pass == 1)
            std::reverse(shapes.begin(), shapes.end());
        for (const auto& shape : shapes) {
            SCOPED_TRACE(shape);
            ASSERT_NO_FATAL_FAILURE(Measure(shape, true));
        }
    }
}

// Compare complete decode/decision/kernel/publication costs. Validation occurs
// outside the timer; no absolute speed threshold is a CI correctness gate.
TEST(ScalarSelectionPerformanceTest, ModeMatrix) {
    if (::testing::GTEST_FLAG(filter).find("ModeMatrix") == std::string::npos)
        GTEST_SKIP();
    using namespace scalar;
    constexpr size_t n = 8192;
    auto data = std::make_shared<std::vector<int64_t>>(n);
    for (size_t i = 0; i < n; ++i) (*data)[i] = (i * 17) % 100;
    for (const std::string pattern : {"scattered", "clustered"}) {
        for (int percent : {1, 5, 10, 25, 50, 90, 100}) {
            auto selected = OwnedSelection::None(n);
            for (size_t i = 0; i < n; ++i)
                if (pattern == "scattered"
                        ? int((i * 17) % 100) < percent
                        : int(i % 1024) < percent * 1024 / 100)
                    selected.SetSelected(i);
            for (bool complete : {true, false}) {
                auto declared =
                    complete ? SelectionView::All(n) : selected.view();
                auto source = ReadOnlyVectorView::BorrowImmutable<int64_t>(
                    DataType::INT64,
                    data->data(),
                    data,
                    declared,
                    declared,
                    PrimitivePayloadAccess::AllDefinedRows);
                auto input = source.Restrict(selected.view());
                auto constant = ReadOnlyVectorView::CopyOf(
                                    std::make_shared<ConstantVector<int64_t>>(
                                        DataType::INT64, 1, 10))
                                    .Broadcast(selected.view());
                std::array<std::vector<double>, 4> timings;
                const std::array<SelectionKernelPreference, 4> modes{
                    SelectionKernelPreference::Ranges,
                    SelectionKernelPreference::Sparse,
                    SelectionKernelPreference::Dense,
                    SelectionKernelPreference::Adaptive};
                const std::array<std::string, 4> names{
                    "ranges", "sparse", "dense", "adaptive"};
                for (int round = -5; round < 31; ++round) {
                    for (size_t order = 0; order < 4; ++order) {
                        const size_t m = (order + size_t(round + 5)) % 4;
                        std::optional<ReadOnlyVectorView> result;
                        const auto start = Clock::now();
                        for (int repeat = 0; repeat < 16; ++repeat) {
                            BitmapResultBuilder out(n);
                            const bool decoded =
                                ComparePrimitive<int64_t, int64_t>(
                                    input,
                                    constant,
                                    selected.view(),
                                    CompareOp::Less,
                                    out,
                                    [](auto a, auto b) { return a < b; },
                                    modes[m]);
                            if (!decoded)
                                std::abort();
                            result = std::move(out).Finish(selected.view());
                        }
                        const double elapsed = Micros(start, Clock::now()) / 16;
                        ASSERT_TRUE(
                            result->defined_rows().Equals(selected.view()));
                        selected.view().ForEachSelected([&](size_t row) {
                            ASSERT_TRUE(result->IsValid(row));
                            ASSERT_EQ(result->ValueAt<bool>(row),
                                      (*data)[row] < 10);
                        });
                        if (round >= 0)
                            timings[m].push_back(elapsed);
                    }
                }
                Json report{{"rows", n},
                            {"pattern", pattern},
                            {"percent", percent},
                            {"full_payload", complete},
                            {"rounds", 31},
                            {"repeats", 16}};
                for (size_t m = 0; m < 4; ++m) {
                    auto sorted = timings[m];
                    std::sort(sorted.begin(), sorted.end());
                    report[names[m]] = sorted[sorted.size() / 2];
                }
                const auto span = input.DecodePrimitive()->spans[0];
                const bool can_expand = span.CanEvaluateAll(0, n);
                // Match the kernel's intersection/canonicalization, including
                // an all-selected bitmap becoming the uniform All encoding.
                auto active =
                    OwnedSelection::Intersect(selected.view(), span.valid);
                active.Intersect(constant.DecodePrimitive()->spans[0].valid);
                const auto chosen = ChooseSelectionKernel(
                    active.view(),
                    can_expand,
                    SelectionKernelCost::NumericComparison());
                report["dense_eligible"] = can_expand;
                report["chosen"] =
                    chosen == SelectionKernelMode::Dense    ? "dense"
                    : chosen == SelectionKernelMode::Sparse ? "sparse"
                                                            : "ranges";
                std::cout << "SCALAR_SELECTION_PERF " << report.dump()
                          << std::endl;
            }
        }
    }
}

TEST_P(ScalarExecutionPerformanceTest, AdaptiveBatchMatrix) {
    if (::testing::GTEST_FLAG(filter).find("AdaptiveBatchMatrix") ==
            std::string::npos ||
        GetParam() > 1)
        GTEST_SKIP();
    std::vector<std::string> shapes =
        GetParam() == 0 ? std::vector<std::string>{"unary",
                                                   "eq",
                                                   "range",
                                                   "compound",
                                                   "mixed_compare"}
                        : std::vector<std::string>{"compound"};
    for (int pass = 0; pass < 3; ++pass) {
        if (pass == 1)
            std::reverse(shapes.begin(), shapes.end());
        for (const auto& shape : shapes)
            ASSERT_NO_FATAL_FAILURE(Measure(shape, true));
    }
}

TEST(ScalarSelectionPerformanceTest, RemainingModeMatrix) {
    if (::testing::GTEST_FLAG(filter).find("RemainingModeMatrix") ==
        std::string::npos)
        GTEST_SKIP();
    using namespace scalar;
    constexpr size_t n = 8192;
    auto numbers = std::make_shared<std::vector<int64_t>>(n);
    auto strings = std::make_shared<std::vector<std::string>>(n);
    auto objects = std::make_shared<std::vector<::milvus::Json>>();
    for (size_t i = 0; i < n; ++i) {
        (*numbers)[i] = (i * 17) % 1500;
        (*strings)[i] = "value_" + std::to_string((*numbers)[i]);
        objects->emplace_back(simdjson::padded_string(
            "{\"x\":" + std::to_string((*numbers)[i]) + "}"));
    }
    SimdBatchElement<int64_t> small(std::vector<int64_t>{1, 5, 10, 90});
    std::vector<int64_t> many;
    for (int64_t i = 0; i < 1024; ++i) many.push_back(i * 2);
    SetElement<int64_t> large(many);
    SetElement<std::string> texts(
        std::vector<std::string>{"value_1", "value_5", "value_10", "value_90"});
    for (const std::string pattern : {"scattered", "clustered"}) {
        for (int percent : {10, 50, 90, 99}) {
            auto selected = OwnedSelection::None(n);
            for (size_t i = 0; i < n; ++i)
                if (pattern == "scattered"
                        ? (i * 17) % 100 < percent
                        : i % 1024 < size_t(percent * 1024 / 100))
                    selected.SetSelected(i);
            for (bool full : {true, false}) {
                auto declared = full ? SelectionView::All(n) : selected.view();
                auto input = ReadOnlyVectorView::BorrowImmutable<int64_t>(
                                 DataType::INT64,
                                 numbers->data(),
                                 numbers,
                                 declared,
                                 declared,
                                 PrimitivePayloadAccess::AllDefinedRows)
                                 .Restrict(selected.view());
                auto text = ReadOnlyVectorView::BorrowImmutable<std::string>(
                                DataType::VARCHAR,
                                strings->data(),
                                strings,
                                declared,
                                declared,
                                PrimitivePayloadAccess::AllDefinedRows)
                                .Restrict(selected.view());
                auto json = ReadOnlyVectorView::BorrowImmutable<::milvus::Json>(
                                DataType::JSON,
                                objects->data(),
                                objects,
                                declared,
                                declared)
                                .Restrict(selected.view());
                auto measure = [&](const char* kernel,
                                   SelectionKernelCost cost,
                                   bool dense_eligible,
                                   const auto& run,
                                   const auto& oracle) {
                    const std::array<SelectionKernelPreference, 4> modes{
                        SelectionKernelPreference::Ranges,
                        SelectionKernelPreference::Sparse,
                        SelectionKernelPreference::Dense,
                        SelectionKernelPreference::Adaptive};
                    const std::array<std::string, 4> names{
                        "ranges", "sparse", "dense", "adaptive"};
                    std::array<std::vector<double>, 4> timings;
                    for (int round = -5; round < 31; ++round) {
                        for (size_t order = 0; order < 4; ++order) {
                            const auto m = (order + size_t(round + 5)) % 4;
                            std::optional<ReadOnlyVectorView> result;
                            const auto start = Clock::now();
                            for (int repeat = 0; repeat < 4; ++repeat) {
                                BitmapResultBuilder out(n);
                                run(out, modes[m]);
                                result = std::move(out).Finish(selected.view());
                            }
                            const double elapsed =
                                Micros(start, Clock::now()) / 4;
                            ASSERT_TRUE(
                                result->defined_rows().Equals(selected.view()));
                            selected.view().ForEachSelected([&](size_t i) {
                                ASSERT_TRUE(result->IsValid(i));
                                ASSERT_EQ(result->ValueAt<bool>(i), oracle(i));
                            });
                            if (round >= 0)
                                timings[m].push_back(elapsed);
                        }
                    }
                    Json report{{"kernel", kernel},
                                {"pattern", pattern},
                                {"percent", percent},
                                {"full_payload", full},
                                {"rows", n},
                                {"rounds", 31},
                                {"repeats", 4}};
                    report["dense_eligible"] = dense_eligible;
                    const auto chosen = ChooseSelectionKernel(
                        selected.view(), dense_eligible, cost);
                    report["chosen"] =
                        chosen == SelectionKernelMode::Dense    ? "dense"
                        : chosen == SelectionKernelMode::Sparse ? "sparse"
                                                                : "ranges";
                    for (size_t m = 0; m < 4; ++m)
                        report[names[m]] =
                            Summarize(timings[m]).at("median_us");
                    std::cout << "SCALAR_REMAINING_SELECTION_PERF "
                              << report.dump() << std::endl;
                };
                auto membership =
                    [&](const auto& terms, bool simd, const char* name) {
                        auto probe = [&](int64_t x) {
                            return terms.In(MultiElement::ValueType(
                                std::in_place_type<int64_t>, x));
                        };
                        measure(
                            name,
                            SelectionKernelCost::NumericMembership(terms.Size(),
                                                                   simd),
                            full,
                            [&](BitmapResultBuilder& out,
                                SelectionKernelPreference mode) {
                                if (!MatchPrimitive<int64_t>(
                                        input,
                                        selected.view(),
                                        out,
                                        terms,
                                        probe,
                                        SelectionKernelCost::NumericMembership(
                                            terms.Size(), simd),
                                        mode))
                                    std::abort();
                            },
                            [&](size_t i) { return probe((*numbers)[i]); });
                    };
                membership(small, true, "in4");
                membership(large, false, "in1024");
                for (bool bulk : {false, true}) {
                    const auto cost = SelectionKernelCost::Arithmetic(bulk);
                    auto probe = [&](int64_t x) {
                        return bulk ? x + 2 < 10 : x / 3 < 10;
                    };
                    measure(
                        bulk ? "arith_add" : "arith_div",
                        cost,
                        full && bulk,
                        [&](BitmapResultBuilder& out,
                            SelectionKernelPreference mode) {
                            EvaluatePrimitiveKernel<int64_t>(
                                input,
                                selected.view(),
                                out,
                                bulk,
                                cost,
                                [&](size_t begin,
                                    size_t count,
                                    const int64_t* data,
                                    bool constant) {
                                    if (bulk && !constant)
                                        out.ArithmeticCompare<
                                            int64_t,
                                            bitset::ArithOpType::Add,
                                            bitset::CompareOpType::LT>(
                                            begin, count, data, 2, 10);
                                    else
                                        out.EvaluateRange(
                                            begin, count, [&](size_t i) {
                                                return probe(
                                                    data[constant ? 0 : i]);
                                            });
                                },
                                probe,
                                mode);
                        },
                        [&](size_t i) { return probe((*numbers)[i]); });
                }
                auto text_probe = [&](std::string_view x) {
                    return texts.In(MultiElement::ValueType(
                        std::in_place_type<std::string_view>, x));
                };
                measure(
                    "string_in",
                    SelectionKernelCost::StringMembership(),
                    full,
                    [&](BitmapResultBuilder& out,
                        SelectionKernelPreference mode) {
                        StringInput(text).MatchKernel(
                            selected.view(),
                            out,
                            [&](size_t begin,
                                size_t count,
                                const auto* data,
                                bool) {
                                out.MatchTerms(begin, count, data, texts);
                            },
                            text_probe,
                            true,
                            SelectionKernelCost::StringMembership(),
                            mode);
                    },
                    [&](size_t i) { return text_probe((*strings)[i]); });
                measure(
                    "json_unary",
                    SelectionKernelCost::ObjectCallback(),
                    false,
                    [&](BitmapResultBuilder& out,
                        SelectionKernelPreference mode) {
                        ObjectInput<::milvus::Json>(json).MatchOptional(
                            selected.view(),
                            out,
                            [&](const ::milvus::Json& value)
                                -> std::optional<bool> {
                                auto number = value.at<int64_t>("/x");
                                if (number.error())
                                    return std::nullopt;
                                return number.value() < 10;
                            },
                            mode);
                    },
                    [&](size_t i) { return (*numbers)[i] < 10; });
            }
        }
    }
}

TEST_P(ScalarExecutionPerformanceTest, RemainingAdaptiveBatchMatrix) {
    if (GetParam() != 0 ||
        ::testing::GTEST_FLAG(filter).find("RemainingAdaptiveBatchMatrix") ==
            std::string::npos)
        GTEST_SKIP();
    std::vector<std::string> shapes{"in",
                                    "in64",
                                    "in1024",
                                    "float_in",
                                    "double_in",
                                    "arith",
                                    "arith_div",
                                    "arith_mod",
                                    "float_arith",
                                    "double_arith",
                                    "string_eq",
                                    "string_range",
                                    "string_prefix",
                                    "string_like",
                                    "string_in",
                                    "string_in_long",
                                    "json_unary",
                                    "json_in",
                                    "json_contains_any",
                                    "json_contains_all",
                                    "json_arith",
                                    "json_exists",
                                    "array_element",
                                    "array_length",
                                    "array_any",
                                    "array_all",
                                    "array_eq",
                                    "roaring",
                                    "bloom_int",
                                    "bloom_string",
                                    "legacy_call",
                                    "timestamp_arith"};
    for (int pass = 0; pass < 3; ++pass) {
        if (pass == 1)
            std::reverse(shapes.begin(), shapes.end());
        for (const auto& shape : shapes)
            ASSERT_NO_FATAL_FAILURE(Measure(shape, true));
    }
}

TEST_P(ScalarExecutionPerformanceTest, LegacyAdapterMatrix) {
    if (GetParam() != 0 || ::testing::GTEST_FLAG(filter).find(
                               "LegacyAdapterMatrix") == std::string::npos)
        GTEST_SKIP();
    for (int pass = 0; pass < 3; ++pass)
        for (const auto& shape : {"legacy_starts_with", "legacy_call"})
            ASSERT_NO_FATAL_FAILURE(Measure(shape, true));
}

// Broad, same-revision audit: only explicit filters opt into timings.
TEST_P(ScalarExecutionPerformanceTest, RegressionAuditMatrix) {
    if (::testing::GTEST_FLAG(filter).find("RegressionAuditMatrix") ==
        std::string::npos)
        GTEST_SKIP();
    std::vector<std::string> shapes =
        GetParam() == 0   ? PerformanceShapes()
        : GetParam() == 1 ? std::vector<std::string>{"compound"}
        : GetParam() == 2 ? std::vector<std::string>{"unary",
                                                     "eq",
                                                     "ne",
                                                     "le",
                                                     "gt",
                                                     "ge",
                                                     "range",
                                                     "in",
                                                     "in64",
                                                     "in1024",
                                                     "is_null",
                                                     "compound"}
                          : std::vector<std::string>{"json_unary",
                                                     "json_range",
                                                     "json_in",
                                                     "json_contains",
                                                     "json_contains_all",
                                                     "json_contains_any",
                                                     "json_exists",
                                                     "json_arith",
                                                     "json_rare"};
    const std::vector<std::string> indexes =
        GetParam() == 1 || GetParam() == 2
            ? std::vector<std::string>{"sorted", "bitmap"}
            : std::vector<std::string>{"sorted"};
    ASSERT_EQ(std::getenv("MILVUS_SCALAR_PERF_INDEX_TYPE"), nullptr)
        << "Audit explicitly covers both index implementations";
    for (const auto& index : indexes) {
        index_type_ = index;
        for (int pass = 0; pass < 3; ++pass) {
            if (pass == 1)
                std::reverse(shapes.begin(), shapes.end());
            for (const auto& shape : shapes) {
                SCOPED_TRACE(shape + " / " + index +
                             " pass=" + std::to_string(pass));
                ASSERT_NO_FATAL_FAILURE(Measure(shape, true));
            }
        }
    }
}

// Attribute the normal 32-batch ARRAY path without changing Reader selection.
// Loading the single input explicitly lets Eval measure computation on cached
// input. These instrumented stages are diagnostic, not operator latency.
TEST_P(ScalarExecutionPerformanceTest, ArrayStageDiagnostic) {
    if (::testing::GTEST_FLAG(filter).find("ArrayStageDiagnostic") ==
            std::string::npos ||
        GetParam() != 0)
        GTEST_SKIP();
    constexpr int64_t batch_size = 8192, rows = 262144;
    for (const auto& shape : {"array_any",
                              "array_all",
                              "array_element",
                              "array_length",
                              "array_eq"}) {
        SCOPED_TRACE(shape);
        shape_ = shape;
        compound_ = false;
        nullable_ = true;
        ASSERT_NO_FATAL_FAILURE(Load(rows));
        ASSERT_NO_FATAL_FAILURE(CheckRoute(true, batch_size));
        for (int pass = 0; pass < 3; ++pass) {
            std::array<std::vector<double>, 4> timings;
            for (int round = -5; round < 31; ++round) {
                auto query =
                    std::make_shared<QueryContext>("scalar-array-stages",
                                                   segment_.get(),
                                                   rows_,
                                                   Timestamp(42) << 18,
                                                   0,
                                                   0,
                                                   query::PlanOptions{});
                auto context = std::make_shared<segcore::reader::ReadContext>();
                context->checkpoint = [query] {
                    checkCancellation(query.get());
                };
                auto execution = FilterBitsExecution::TryCreate(
                    predicate_,
                    nullptr,
                    segment_->CaptureReadSnapshot(),
                    rows_,
                    batch_size,
                    context,
                    schema_);
                ASSERT_TRUE(execution);
                ASSERT_EQ(execution->prepared().plan()->inputs().slots().size(),
                          1);
                auto batch = execution->prepared().MakeReusableBatch(
                    ReadBatchBinding::Range(0, batch_size));
                TargetBitmap excluded(rows_, true);
                std::array<double, 4> elapsed{};
                for (int64_t begin = 0; begin < rows_; begin += batch_size) {
                    const auto all = SelectionView::All(batch_size);
                    auto start = Clock::now();
                    batch->Reset(ReadBatchBinding::Range(begin, batch_size));
                    auto reset = Clock::now();
                    auto input = batch->input().GetColumn(0, all);
                    auto loaded = Clock::now();
                    auto result = batch->Eval(all, context->checkpoint);
                    auto evaluated = Clock::now();
                    result.at(0).CopyTrueTo(all, excluded, begin);
                    excluded.view(begin, batch_size).flip();
                    auto copied = Clock::now();
                    elapsed[0] += Micros(start, reset);
                    elapsed[1] += Micros(reset, loaded);
                    elapsed[2] += Micros(loaded, evaluated);
                    elapsed[3] += Micros(evaluated, copied);
                    ASSERT_TRUE(input.defined_rows().IsAll());
                }
                ASSERT_TRUE(excluded == expected_);
                if (round >= 0)
                    for (size_t i = 0; i < elapsed.size(); ++i)
                        timings[i].push_back(elapsed[i]);
            }
            Json report{{"expression", shape},
                        {"pass", pass},
                        {"rows", rows},
                        {"batch_size", batch_size},
                        {"diagnostic_only", true}};
            const std::array<std::string, 4> names{
                "reset", "load", "cached_eval", "output"};
            for (size_t i = 0; i < names.size(); ++i)
                report[names[i]] = Summarize(timings[i]);
            std::cout << "SCALAR_ARRAY_STAGE_DIAGNOSTIC " << report.dump()
                      << std::endl;
        }
    }
}

// Test-only attribution controls. The legacy scan is the real public
// ProcessDataChunks path with a fixture-specific copy of its INT64 small-set
// callback, NOT an instrumented production Eval. Compare its total with the
// separately measured real operators before interpreting the split. Cached
// controls share the exact new Reader input and validate every output outside
// timers. Their differences include code-generation effects, not only branches.
TEST_P(ScalarExecutionPerformanceTest, ArrayCostDiagnostic) {
    if (::testing::GTEST_FLAG(filter).find("ArrayCostDiagnostic") ==
            std::string::npos ||
        GetParam() != 0)
        GTEST_SKIP();
    constexpr int64_t batch_size = 8192, rows = 262144;
    auto run_shape = [&]<bool All>() {
        shape_ = All ? "array_all" : "array_any";
        compound_ = false;
        const std::vector<int64_t> terms =
            All ? std::vector<int64_t>{10, 11}
                : std::vector<int64_t>{1, 5, 10, 90};
        ankerl::unordered_dense::set<int64_t> any_targets;
        any_targets.max_load_factor(0.5f);
        ankerl::unordered_dense::map<int64_t, uint32_t> all_targets;
        for (size_t i = 0; i < terms.size(); ++i) {
            any_targets.insert(terms[i]);
            all_targets[terms[i]] = i;
        }
        const uint64_t full = (uint64_t{1} << terms.size()) - 1;
        auto hash_probe = [&](const ArrayView& array) {
            if constexpr (All)
                if (array.length() < terms.size())
                    return false;
            uint64_t found = 0;
            for (int j = 0; j < array.length(); ++j) {
                const auto value = array.get_data_unchecked<int64_t>(j);
                if constexpr (All) {
                    auto it = all_targets.find(value);
                    if (it != all_targets.end()) {
                        found |= uint64_t{1} << it->second;
                        if (found == full)
                            return true;
                    }
                } else if (any_targets.find(value) != any_targets.end()) {
                    return true;
                }
            }
            return All && found == full;
        };
        for (bool nullable : {false, true}) {
            nullable_ = nullable;
            ASSERT_NO_FATAL_FAILURE(Load(rows));
            ASSERT_NO_FATAL_FAILURE(CheckRoute(false, batch_size));
            ASSERT_NO_FATAL_FAILURE(CheckRoute(true, batch_size));
            for (int pass = 0; pass < 3; ++pass) {
                std::map<std::string, std::vector<double>> timings;
                for (int round = -5; round < 31; ++round) {
                    std::map<std::string, double> elapsed;
                    for (bool scalar : {round % 2 == 0, round % 2 != 0}) {
                        Sample sample;
                        ASSERT_NO_FATAL_FAILURE(
                            Run(scalar, batch_size, &sample));
                        elapsed[scalar ? "operator_new" : "operator_old"] =
                            sample.total_us;
                    }
                    auto query =
                        std::make_shared<QueryContext>("scalar-array-cost",
                                                       segment_.get(),
                                                       rows_,
                                                       Timestamp(42) << 18,
                                                       0,
                                                       0,
                                                       query::PlanOptions{});
                    auto context =
                        std::make_shared<segcore::reader::ReadContext>();
                    context->checkpoint = [query] {
                        checkCancellation(query.get());
                    };
                    auto execution = FilterBitsExecution::TryCreate(
                        predicate_,
                        nullptr,
                        segment_->CaptureReadSnapshot(),
                        rows_,
                        batch_size,
                        context,
                        schema_);
                    ASSERT_TRUE(execution);
                    auto batch = execution->prepared().MakeReusableBatch(
                        ReadBatchBinding::Range(0, batch_size));
                    auto logical =
                        std::dynamic_pointer_cast<const expr::JsonContainsExpr>(
                            predicate_);
                    ASSERT_TRUE(logical);
                    PhyJsonContainsFilterExpr legacy({},
                                                     logical,
                                                     "array-cost",
                                                     nullptr,
                                                     segment_.get(),
                                                     rows_,
                                                     batch_size,
                                                     0);
                    legacy.DetermineExecPath();
                    ASSERT_EQ(legacy.GetExecPath(), ExprExecPath::RawData);
                    TargetBitmap legacy_truth(rows_, false),
                        legacy_valid(rows_, true);
                    TargetBitmap new_excluded(rows_, true);
                    size_t spans = 0, selected_ranges = 0, ranges_spans = 0,
                           sparse_spans = 0;
                    for (int64_t begin = 0; begin < rows_;
                         begin += batch_size) {
                        const auto all = SelectionView::All(batch_size);
                        auto measure_old = [&] {
                            double compute = 0;
                            auto callback = [&](const ArrayView* data,
                                                ValidityView valid,
                                                const int32_t*,
                                                int size,
                                                TargetBitmapView truth,
                                                TargetBitmapView validity) {
                                auto start = Clock::now();
                                if (data) {
                                    for (int i = 0; i < size; ++i) {
                                        if (valid && !valid[i]) {
                                            truth[i] = validity[i] = false;
                                            continue;
                                        }
                                        truth[i] = hash_probe(data[i]);
                                    }
                                }
                                compute += Micros(start, Clock::now());
                            };
                            auto start = Clock::now();
                            const auto processed =
                                legacy.ProcessDataChunks<ArrayView>(
                                    callback,
                                    std::nullptr_t{},
                                    legacy_truth.view(begin, batch_size),
                                    legacy_valid.view(begin, batch_size));
                            const double total = Micros(start, Clock::now());
                            elapsed["legacy_scan_total"] += total;
                            elapsed["legacy_callback"] += compute;
                            elapsed["legacy_scan_other"] += total - compute;
                            ASSERT_EQ(processed, batch_size);
                        };
                        if (round % 2 == 0)
                            measure_old();
                        auto start = Clock::now();
                        batch->Reset(
                            ReadBatchBinding::Range(begin, batch_size));
                        auto reset = Clock::now();
                        auto input = batch->input().GetColumn(0, all);
                        auto loaded = Clock::now();
                        auto result = batch->Eval(all, context->checkpoint);
                        auto evaluated = Clock::now();
                        result.at(0).CopyTrueTo(all, new_excluded, begin);
                        new_excluded.view(begin, batch_size).flip();
                        auto copied = Clock::now();
                        elapsed["new_reset"] += Micros(start, reset);
                        elapsed["new_load"] += Micros(reset, loaded);
                        elapsed["new_cached_eval"] += Micros(loaded, evaluated);
                        elapsed["new_output"] += Micros(evaluated, copied);
                        if (round % 2 != 0)
                            measure_old();
                        auto decoded = input.DecodeObjects<ArrayView>();
                        ASSERT_TRUE(decoded);
                        // Establish the domain of the deliberately unchecked controls.
                        for (const auto& span : decoded->spans) {
                            ++spans;
                            const auto active = OwnedSelection::Intersect(
                                all.Slice(span.begin, span.count), span.valid);
                            if (scalar::ChooseSelectionKernel(
                                    active.view(),
                                    false,
                                    scalar::SelectionKernelCost::
                                        ObjectCallback()) ==
                                scalar::SelectionKernelMode::Ranges)
                                ++ranges_spans;
                            else
                                ++sparse_spans;
                            span.valid.ForEachSelectedRange(
                                [&](size_t, size_t) { ++selected_ranges; });
                            const auto* data =
                                static_cast<const ArrayView*>(span.data);
                            span.valid.ForEachSelected([&](size_t i) {
                                const auto& array = data[span.constant ? 0 : i];
                                ASSERT_EQ(array.get_element_type(),
                                          DataType::INT64);
                                for (int j = 0; j < array.length(); ++j)
                                    ASSERT_TRUE(array.is_element_valid(j));
                            });
                        }
                        const std::array<std::string, 7> names{
                            "cached_direct_hash",
                            "cached_object_hash",
                            "cached_object_tiny",
                            "cached_object_checked",
                            "cached_object_typed",
                            "cached_checked_ranges",
                            "cached_checked_sparse"};
                        // Rotate controls to avoid consistently favoring one cache order.
                        for (size_t step = 0; step < names.size(); ++step) {
                            const size_t mode =
                                (step + round + 5) % names.size();
                            TargetBitmap truth(batch_size, false),
                                valid(batch_size, false);
                            std::optional<ReadOnlyVectorView> output;
                            const auto t0 = Clock::now();
                            if (mode == 0) {
                                // Includes DecodeObjects, just as ObjectInput does.
                                auto local = input.DecodeObjects<ArrayView>();
                                for (const auto& span : local->spans) {
                                    const auto* data =
                                        static_cast<const ArrayView*>(
                                            span.data);
                                    for (size_t i = 0; i < span.count; ++i) {
                                        if (!span.valid.IsSelected(i))
                                            continue;
                                        valid[span.begin + i] = true;
                                        truth[span.begin + i] = hash_probe(
                                            data[span.constant ? 0 : i]);
                                    }
                                }
                            } else {
                                BitmapResultBuilder builder(batch_size);
                                auto match = [&](const auto& probe) {
                                    if (mode < 5) {
                                        scalar::ObjectInput<ArrayView>(input)
                                            .Match(all, builder, probe);
                                        return;
                                    }
                                    // Same decoded traversal as ObjectInput::Match,
                                    // changing only the selection preference.
                                    auto local =
                                        input.DecodeObjects<ArrayView>();
                                    for (const auto& span : local->spans) {
                                        auto active = OwnedSelection::Intersect(
                                            all.Slice(span.begin, span.count),
                                            span.valid);
                                        const auto* data =
                                            static_cast<const ArrayView*>(
                                                span.data);
                                        auto row = [&](size_t i) {
                                            return probe(
                                                data[span.constant ? 0 : i]);
                                        };
                                        scalar::EvaluateSelection(
                                            span.begin,
                                            active.view(),
                                            false,
                                            scalar::SelectionKernelCost::
                                                ObjectCallback(),
                                            builder,
                                            [&](size_t offset, size_t count) {
                                                builder.EvaluateRange(
                                                    span.begin + offset,
                                                    count,
                                                    [&](size_t i) {
                                                        return row(offset + i);
                                                    });
                                            },
                                            row,
                                            mode == 5
                                                ? scalar::
                                                      SelectionKernelPreference::
                                                          Ranges
                                                : scalar::
                                                      SelectionKernelPreference::
                                                          Sparse);
                                    }
                                };
                                if (mode == 1)
                                    match(hash_probe);
                                else {
                                    auto tiny = [&]<bool Checked,
                                                    bool Typed>() {
                                        size_t remaining = 0;
                                        auto checkpoint = [&] {
                                            if (remaining == 0) {
                                                context->checkpoint();
                                                remaining = 256;
                                            }
                                            --remaining;
                                        };
                                        match([&](const ArrayView& array) {
                                            if constexpr (Checked) {
                                                checkpoint();
                                                const auto stored =
                                                    array.get_element_type();
                                                AssertInfo(
                                                    array.length() == 0 ||
                                                        stored ==
                                                            DataType::INT8 ||
                                                        stored ==
                                                            DataType::INT16 ||
                                                        stored ==
                                                            DataType::INT32 ||
                                                        stored ==
                                                            DataType::INT64,
                                                    "diagnostic ARRAY "
                                                    "payload mismatch");
                                            }
                                            if constexpr (All)
                                                if (array.length() <
                                                    terms.size())
                                                    return false;
                                            uint64_t found = 0;
                                            for (int j = 0; j < array.length();
                                                 ++j) {
                                                if constexpr (Checked) {
                                                    checkpoint();
                                                    if (!array.is_element_valid(
                                                            j))
                                                        continue;
                                                }
                                                const int64_t value =
                                                    Typed
                                                        ? reinterpret_cast<
                                                              const int64_t*>(
                                                              array.data())[j]
                                                        : array
                                                              .get_data_unchecked<
                                                                  int64_t>(j);
                                                for (size_t k = 0;
                                                     k < terms.size();
                                                     ++k) {
                                                    if (value != terms[k])
                                                        continue;
                                                    if constexpr (!All)
                                                        return true;
                                                    found |= uint64_t{1} << k;
                                                    if (found == full)
                                                        return true;
                                                    break;
                                                }
                                            }
                                            return false;
                                        });
                                    };
                                    if (mode == 2)
                                        tiny.template
                                        operator()<false, false>();
                                    else if (mode == 4)
                                        tiny.template operator()<false, true>();
                                    else
                                        tiny.template operator()<true, false>();
                                }
                                output = std::move(builder).Finish(all);
                            }
                            elapsed[names[mode]] += Micros(t0, Clock::now());
                            if (output) {
                                output->CopyTrueTo(all, truth);
                                const auto output_validity =
                                    output->CopyValidity();
                                output_validity.view().CopyTo(valid);
                            }
                            const auto input_validity = input.CopyValidity();
                            ASSERT_TRUE(SelectionView::FromBitmap(valid).Equals(
                                input_validity.view()));
                            truth.flip();
                            ASSERT_TRUE(truth ==
                                        expected_.view(begin, batch_size))
                                << names[mode];
                        }
                    }
                    legacy_truth.flip();
                    ASSERT_TRUE(legacy_truth == expected_);
                    ASSERT_TRUE(new_excluded == expected_);
                    if (round >= 0)
                        for (const auto& [name, value] : elapsed)
                            timings[name].push_back(value);
                    if (round == 30) {
                        Json report{{"expression", shape_},
                                    {"nullable", nullable},
                                    {"pass", pass},
                                    {"rows", rows},
                                    {"batch_size", batch_size},
                                    {"spans", spans},
                                    {"valid_ranges", selected_ranges},
                                    {"ranges_spans", ranges_spans},
                                    {"sparse_spans", sparse_spans},
                                    {"diagnostic_only", true},
                                    {"legacy_callback_is_control", true}};
                        for (const auto& [name, values] : timings)
                            report[name] = Summarize(values);
                        std::cout << "SCALAR_ARRAY_COST_DIAGNOSTIC "
                                  << report.dump() << std::endl;
                    }
                }
            }
        }
    };
    run_shape.template operator()<false>();
    run_shape.template operator()<true>();
}

TEST_P(ScalarExecutionPerformanceTest, CheckpointDiagnostic) {
    if (::testing::GTEST_FLAG(filter).find("CheckpointDiagnostic") ==
            std::string::npos ||
        GetParam() != 0)
        GTEST_SKIP();
    constexpr int64_t batch = 8192, rows = 262144;
    for (const auto& shape : {"array_any",
                              "array_all",
                              "array_element",
                              "array_length",
                              "string_like",
                              "legacy_call",
                              "legacy_starts_with"}) {
        SCOPED_TRACE(shape);
        shape_ = shape;
        compound_ = false;
        nullable_ = true;
        ASSERT_NO_FATAL_FAILURE(Load(rows));
        ASSERT_NO_FATAL_FAILURE(CheckRoute(true, batch));
        Sample untimed;
        size_t checkpoints = 0;
        ASSERT_NO_FATAL_FAILURE(
            RunCheckpointControl(true, batch, &untimed, &checkpoints));
        for (int pass = 0; pass < 3; ++pass) {
            std::vector<Sample> checked, unbound;
            for (int round = -5; round < 31; ++round) {
                for (bool enabled : {round % 2 == 0, round % 2 != 0}) {
                    Sample sample;
                    ASSERT_NO_FATAL_FAILURE(
                        RunCheckpointControl(enabled, batch, &sample));
                    if (round >= 0)
                        (enabled ? checked : unbound).push_back(sample);
                }
            }
            Json report{{"expression", shape},
                        {"pass", pass},
                        {"rows", rows},
                        {"batch_size", batch},
                        {"checkpoint_calls", checkpoints},
                        {"checked", Summarize(checked)},
                        {"unbound", Summarize(unbound)},
                        {"diagnostic_only", true},
                        {"op_context_present", false}};
            std::cout << "SCALAR_CHECKPOINT_DIAGNOSTIC " << report.dump()
                      << std::endl;
        }
    }
}

// Isolate the actual adapters around the real StartsWithVarchar function.
// Prebuilt inputs are warm; this excludes Reader/planning and is not a service
// latency estimate. Keep full-operator measurements in RegressionAuditMatrix.
TEST(ScalarRegressionDiagnostics, LegacyAdapterStages) {
    if (::testing::GTEST_FLAG(filter).find("LegacyAdapterStages") ==
        std::string::npos)
        GTEST_SKIP();
    constexpr size_t n = 8192;
    struct Backing {
        std::vector<std::string> strings;
        std::vector<std::string_view> views;
    };
    auto backing = std::make_shared<Backing>();
    for (size_t i = 0; i < n; ++i)
        backing->strings.push_back("value_" + std::to_string(i * 17 % 100));
    for (const auto& value : backing->strings) backing->views.push_back(value);
    auto valid = OwnedSelection::All(n);
    for (size_t i = 0; i < n; i += 97) valid.SetSelected(i, false);
    const auto values = ReadOnlyVectorView::BorrowImmutable<std::string_view>(
        DataType::VARCHAR,
        backing->views.data(),
        backing,
        SelectionView::All(n),
        valid.view());
    const auto prefix = ReadOnlyVectorView::CopyOf(
        std::make_shared<ConstantVector<std::string>>(
            DataType::VARCHAR, n, std::string("value_1")));
    RowVector args(n,
                   std::vector<VectorPtr>{MakeWritableCopy(values).values,
                                          prefix.CopyForLegacyFunction()});
    VectorPtr expected;
    expression::function::StartsWithVarchar(args, expected);
    VectorPtr value_copy, constant_copy, preserved_constant, kernel_result;
    std::optional<ReadOnlyVectorView> result_copy, adapter_result;
    const std::array<std::string, 6> names{"input_copy",
                                           "constant_expand",
                                           "constant_preserve",
                                           "function",
                                           "result_copy",
                                           "adapter_total"};
    const std::array<std::function<void()>, 6> run{
        [&] { value_copy = MakeWritableCopy(values).values; },
        [&] { constant_copy = MakeWritableCopy(prefix).values; },
        [&] { preserved_constant = prefix.CopyForLegacyFunction(); },
        [&] { expression::function::StartsWithVarchar(args, kernel_result); },
        [&] { result_copy = ReadOnlyVectorView::CopyOf(expected); },
        [&] {
            RowVector local(
                n,
                std::vector<VectorPtr>{values.CopyForLegacyFunction(),
                                       prefix.CopyForLegacyFunction()});
            VectorPtr output;
            expression::function::StartsWithVarchar(local, output);
            adapter_result = ReadOnlyVectorView::CopyOf(output);
        }};
    auto validate = [&](size_t stage) {
        if (stage < 3) {
            auto column = std::dynamic_pointer_cast<SimpleVector>(
                stage == 0   ? value_copy
                : stage == 1 ? constant_copy
                             : preserved_constant);
            ASSERT_TRUE(column);
            ASSERT_EQ(column->size(), n);
            for (size_t i = 0; i < n; ++i) {
                const bool known = stage != 0 || valid.view().IsSelected(i);
                ASSERT_EQ(column->ValidAt(i), known);
                if (known)
                    ASSERT_EQ(*static_cast<const std::string*>(
                                  column->RawValueAt(i, sizeof(std::string))),
                              stage != 0 ? "value_1" : backing->strings[i]);
            }
            return;
        }
        const auto result = stage == 3
                                ? ReadOnlyVectorView::CopyOf(kernel_result)
                            : stage == 4 ? *result_copy
                                         : *adapter_result;
        for (size_t i = 0; i < n; ++i) {
            ASSERT_EQ(result.IsValid(i), valid.view().IsSelected(i));
            if (result.IsValid(i))
                ASSERT_EQ(result.ValueAt<bool>(i),
                          backing->strings[i].starts_with("value_1"));
        }
    };
    for (int pass = 0; pass < 3; ++pass) {
        std::array<std::vector<double>, 6> timings;
        for (int round = -5; round < 31; ++round) {
            for (size_t order = 0; order < run.size(); ++order) {
                const auto stage = (order + size_t(round + 5)) % run.size();
                const auto start = Clock::now();
                for (int repeat = 0; repeat < 4; ++repeat) run[stage]();
                const auto elapsed = Micros(start, Clock::now()) / 4;
                ASSERT_NO_FATAL_FAILURE(validate(stage));
                if (round >= 0)
                    timings[stage].push_back(elapsed);
            }
        }
        Json report{{"rows", n},
                    {"pass", pass},
                    {"diagnostic_only", true},
                    {"rounds", 31},
                    {"repeats", 4},
                    {"reader_included", false}};
        for (size_t i = 0; i < names.size(); ++i)
            report[names[i]] = Summarize(timings[i]);
        std::cout << "SCALAR_LEGACY_ADAPTER_DIAGNOSTIC " << report.dump()
                  << std::endl;
    }
}

// Isolate bitmap packaging from backend query cost. Both alternatives consume
// exactly the same complete three-valued result; these are not operator times.
TEST(ScalarRegressionDiagnostics, IndexResultStages) {
    if (::testing::GTEST_FLAG(filter).find("IndexResultStages") ==
        std::string::npos)
        GTEST_SKIP();
    constexpr size_t n = 262144;
    TargetBitmap truth(n, false), valid(n, true);
    for (size_t row = 0; row < n; ++row) {
        valid.set(row, row % 97 != 0);
        truth.set(row, row % 97 != 0 && row % 100 < 10);
    }
    auto source = std::make_shared<const segcore::reader::IndexResult>(
        segcore::reader::IndexResult::Full(SelectionView::FromBitmap(truth),
                                           SelectionView::FromBitmap(valid)));
    const auto binding = ReadBatchBinding::Range(0, n);
    const auto rows = SelectionView::All(n);
    TargetBitmap output(n, false);
    std::optional<OwnedSelection> invalid;
    bool subset = false;
    const std::array<std::string, 4> names{"validate_subtract",
                                           "validate_subset",
                                           "map_materialized",
                                           "map_shared"};
    const std::array<std::function<void()>, 4> run{
        [&] {
            invalid =
                OwnedSelection::Subtract(source->selected(), source->valid());
        },
        [&] { subset = source->selected().IsSubsetOf(source->valid()); },
        [&] {
            predicate::MapPredicate(*source, binding, rows)
                .CopyTrueTo(rows, output);
        },
        [&] {
            predicate::MapPredicate(source, binding, rows)
                .CopyTrueTo(rows, output);
        }};
    for (int pass = 0; pass < 3; ++pass) {
        std::array<std::vector<double>, 4> timings;
        for (int round = -5; round < 31; ++round) {
            for (size_t order = 0; order < run.size(); ++order) {
                const auto stage = (order + size_t(round + 5)) % run.size();
                const auto start = Clock::now();
                for (int repeat = 0; repeat < 16; ++repeat) run[stage]();
                const auto elapsed = Micros(start, Clock::now()) / 16;
                if (stage == 0)
                    ASSERT_TRUE(invalid->view().IsNone());
                else if (stage == 1)
                    ASSERT_TRUE(subset);
                else
                    ASSERT_TRUE(SelectionView::FromBitmap(output).Equals(
                        source->selected()));
                if (round >= 0)
                    timings[stage].push_back(elapsed);
            }
        }
        Json report{{"rows", n},
                    {"pass", pass},
                    {"rounds", 31},
                    {"repeats", 16},
                    {"diagnostic_only", true},
                    {"index_query_included", false}};
        for (size_t i = 0; i < names.size(); ++i)
            report[names[i]] = Summarize(timings[i]);
        std::cout << "SCALAR_INDEX_RESULT_DIAGNOSTIC " << report.dump()
                  << std::endl;
    }
}

TEST_P(ScalarExecutionPerformanceTest, KernelBatchMatrix) {
    if (::testing::GTEST_FLAG(filter).find("KernelBatchMatrix") ==
        std::string::npos)
        GTEST_SKIP() << "Select KernelBatchMatrix explicitly to measure";
    std::vector<std::string> shapes =
        GetParam() == 1   ? std::vector<std::string>{"compound"}
        : GetParam() == 2 ? std::vector<std::string>{"unary", "in", "in1024"}
        : GetParam() == 3
            ? std::vector<std::string>{"json_unary",
                                       "json_in",
                                       "json_contains_any",
                                       "json_contains_all"}
            : std::vector<std::string>{
                  "arith",        "arith_sub",         "arith_mul",
                  "arith_div",    "arith_mod",         "float_arith",
                  "double_arith", "float_in",          "double_in",
                  "bool_in",      "mixed_compare",     "roaring",
                  "bloom_int",    "bloom_string",      "array_element",
                  "array_length", "array_any",         "array_all",
                  "array_eq",     "legacy_call",       "json_unary",
                  "json_in",      "json_contains_any", "json_contains_all",
                  "json_arith",   "compound",          "string_eq",
                  "string_in"};
    for (int pass = 0; pass < 3; ++pass) {
        if (pass == 1)
            std::reverse(shapes.begin(), shapes.end());
        for (const auto& shape : shapes) {
            SCOPED_TRACE(shape);
            ASSERT_NO_FATAL_FAILURE(Measure(shape, true));
        }
    }
}

TEST_P(ScalarExecutionPerformanceTest, TypedKernelCorrectness) {
    if (GetParam() != 0)
        GTEST_SKIP();
    for (const auto& shape : {"float_in",
                              "double_in",
                              "bool_in",
                              "mixed_compare",
                              "float_arith",
                              "double_arith",
                              "roaring",
                              "bloom_int",
                              "bloom_string",
                              "array_element",
                              "array_length",
                              "array_any",
                              "array_all",
                              "array_eq",
                              "legacy_call"}) {
        SCOPED_TRACE(shape);
        shape_ = shape;
        compound_ = false;
        ASSERT_NO_FATAL_FAILURE(Load(3073));
        for (bool modern : {false, true}) {
            ASSERT_NO_FATAL_FAILURE(CheckRoute(modern, 8192));
            Sample sample;
            ASSERT_NO_FATAL_FAILURE(Run(modern, 8192, &sample));
        }
    }
}

void
ScalarExecutionPerformanceTest::Measure(const std::string& shape, bool emit) {
    const auto& shapes = PerformanceShapes();
    ASSERT_NE(std::find(shapes.begin(), shapes.end(), shape), shapes.end());
    if (shape.starts_with("arith") || shape == "compare")
        ASSERT_EQ(GetParam(), 0);
    shape_ = shape;
    if (const char* index = std::getenv("MILVUS_SCALAR_PERF_INDEX_TYPE"))
        index_type_ = index;
    ASSERT_TRUE(index_type_ == "sorted" || index_type_ == "bitmap");
    if (IsStructured())
        ASSERT_TRUE(GetParam() == 0 ||
                    (GetParam() == 3 && shape_.starts_with("json_")));
    else
        ASSERT_NE(GetParam(), 3);
    compound_ = shape == "compound";
    const auto nullable = Setting("MILVUS_SCALAR_PERF_NULLABLE", 1);
    ASSERT_TRUE(nullable == 0 || nullable == 1);
    nullable_ = nullable == 1;
    const auto rows = Setting("MILVUS_SCALAR_PERF_ROWS", 262144);
    const auto rounds = Setting("MILVUS_SCALAR_PERF_ROUNDS", 31);
    const auto warmup = Setting("MILVUS_SCALAR_PERF_WARMUP", 5);
    const auto batch = Setting("MILVUS_SCALAR_PERF_BATCH", 8192);
    ASSERT_GE(rows, 1);
    ASSERT_LE(rows, 10000000);
    ASSERT_GE(rounds, 3);
    ASSERT_LE(rounds, 1000);
    ASSERT_GE(warmup, 1);
    ASSERT_LE(warmup, 100);
    ASSERT_GE(batch, 1);
    ASSERT_LE(batch, rows);
    ASSERT_NO_FATAL_FAILURE(Load(rows));
    ASSERT_NO_FATAL_FAILURE(CheckRoute(false, batch));
    ASSERT_NO_FATAL_FAILURE(CheckRoute(true, batch));
    std::vector<Sample> legacy, scalar;
    for (int64_t round = -warmup; round < rounds; ++round) {
        // Alternate which engine goes first to reduce ordering/thermal bias.
        for (bool modern : {round % 2 == 0, round % 2 != 0}) {
            Sample sample;
            ASSERT_NO_FATAL_FAILURE(Run(modern, batch, &sample));
            if (round >= 0)
                (modern ? scalar : legacy).push_back(sample);
        }
    }
    const auto old_stats = Summarize(legacy), new_stats = Summarize(scalar);
    const double old_us = old_stats["total"]["median_us"];
    const double new_us = new_stats["total"]["median_us"];
    const Json report{
        {"clock", Clock::thread_cpu ? "thread_cpu" : "wall"},
        {"expression_kind", shape_},
        {"expression",
         std::map<std::string, std::string>{
             {"float_in", "float_in"},
             {"double_in", "double_in"},
             {"bool_in", "bool_in"},
             {"mixed_compare", "mixed_compare"},
             {"float_arith", "float_arith"},
             {"double_arith", "double_arith"},
             {"roaring", "roaring"},
             {"bloom_int", "bloom_int"},
             {"bloom_string", "bloom_string"},
             {"array_element", "array_element"},
             {"array_length", "array_length"},
             {"array_any", "array_any"},
             {"array_all", "array_all"},
             {"array_eq", "array_eq"},
             {"legacy_call", "legacy_call"},
             {"legacy_starts_with",
              "starts_with(s, value_1) [legacy function]"},
             {"timestamp_arith", "ts + INTERVAL 1 SECOND < 1000010"},
             {"compound", "a < 10 && b > 10 && a != 8"},
             {"unary", "a < 10"},
             {"eq", "a == 10"},
             {"ne", "a != 10"},
             {"le", "a <= 10"},
             {"gt", "a > 10"},
             {"ge", "a >= 10"},
             {"range", "10 <= a < 20"},
             {"in", "a IN [1, 5, 10, 90]"},
             {"arith", "a + 2 < 12"},
             {"arith_sub", "a - 2 < 12"},
             {"arith_mul", "a * 2 < 12"},
             {"arith_div", "a / 2 < 12"},
             {"arith_mod", "a % 7 < 3"},
             {"compare", "a < b"},
             {"in64", "a IN [0..63]"},
             {"in1024", "a IN [0..1023]"},
             {"is_null", "a IS NULL"},
             {"string_in", "s IN [value_1,value_5,value_10,value_90]"},
             {"string_in_long", "s IN four strings with 256-byte prefix"},
             {"string_range", "value_10 <= s < value_20"},
             {"string_eq", "s == 'value_10'"},
             {"string_prefix", "s LIKE 'value_1%'"},
             {"string_like", "s LIKE '%1%'"},
             {"json_unary", "j['n'] < 10"},
             {"json_range", "10 <= j['n'] < 20"},
             {"json_in", "j['n'] IN [1,5,10,90]"},
             {"json_contains", "JSON_CONTAINS(j['arr'], 10)"},
             {"json_contains_all", "JSON_CONTAINS_ALL(j['arr'], [10,11])"},
             {"json_contains_any", "JSON_CONTAINS_ANY(j['arr'], [1,5,10,90])"},
             {"json_exists", "EXISTS j['n']"},
             {"json_arith", "j['n'] + 2 < 12"},
             {"json_rare", "j['rare'] < 10"}}
             .at(shape_)},
        {"layout",
         GetParam() == 0   ? "raw"
         : GetParam() == 1 ? "b_index"
         : GetParam() == 2 ? "all_index"
                           : "json_stats"},
        {"rows", rows},
        {"batch_size", batch},
        {"rounds", rounds},
        {"warmup_per_mode", warmup},
        {"matches", rows - expected_.count()},
        {"index_type",
         GetParam() == 3           ? "JSON_STATS"
         : index_type_ == "bitmap" ? "BITMAP"
                                   : "STL_SORT"},
        {"physical_plans", routes_},
        {"comparison_kind",
         shape_ == "json_arith" && GetParam() == 3 ? "path_change"
                                                   : "same_backend"},
        {"nullable", nullable_},
        {"reorder", true},
        {"result_cache", false},
        {"storage_cache", "warm"},
        {"old", old_stats},
        {"new", new_stats},
        {"new_over_old", new_us / old_us},
        {"old_over_new_speedup", old_us / new_us},
        {"new_rows_per_second", rows * 1e6 / new_us},
        {"old_rows_per_second", rows * 1e6 / old_us}};
    RecordProperty("scalar_performance", report.dump());
    if (emit)
        std::cout << (std::string(::testing::UnitTest::GetInstance()
                                      ->current_test_info()
                                      ->name())
                              .starts_with("StringBatchMatrix")
                          ? "SCALAR_STRING_PERF "
                          : "SCALAR_KERNEL_PERF ")
                  << report.dump() << std::endl;
}

INSTANTIATE_TEST_SUITE_P(Paths,
                         ScalarExecutionPerformanceTest,
                         ::testing::Values(0, 1, 2, 3),
                         [](const ::testing::TestParamInfo<int>& p) {
                             return p.param == 0   ? "Raw"
                                    : p.param == 1 ? "Mixed"
                                    : p.param == 2 ? "AllIndex"
                                                   : "Stats";
                         });
}  // namespace
}  // namespace milvus::exec
