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

#pragma once
#include "test_utils/index_test_utils.h"
#include "index/PackedIndexLoad.h"
#include "index/scalar/json/JsonProjectedIndexLoad.h"
#include "index/scalar/json/JsonProjectedIndexArtifact.h"
#include <arrow/io/memory.h>
#include "storage/IndexEntryDirectStreamWriter.h"
#include "storage/RemoteInputStream.h"
#include "storage/RemoteOutputStream.h"
#include "segcore/reader/ScalarQueryView.h"

namespace milvus::segcore::reader {
// Test-only index factory for request-boundary failures and reader decorators.
class TestReaderIndex : public index::IIndex {
 public:
    using Factory = std::function<index::IIndexReaderBasePtr(OpContext*)>;
    TestReaderIndex(index::IndexMetadata metadata, Factory factory)
        : metadata_(metadata), factory_(std::move(factory)) {
    }
    index::IndexMetadata
    Metadata() const override {
        return metadata_;
    }
    index::IIndexReaderBasePtr
    CreateReader(OpContext* ctx) const override {
        return factory_(ctx);
    }
    void
    CancelWarmup() const override {
    }

 private:
    index::IndexMetadata metadata_;
    Factory factory_;
};

struct BuiltReaderIndex : BuiltTestIndex {
    std::string family;
    BuiltReaderIndex(index::IndexPtr owner, std::string f)
        : BuiltTestIndex(std::move(owner)), family(std::move(f)) {
    }
};

// Use the packed artifact selector for hybrid and JSON fixtures as production does.
inline BuiltReaderIndex
LoadReaderArtifact(storage::ArtifactPtr artifact,
                   const std::string& family,
                   Config params) {
    auto buffer = arrow::io::BufferOutputStream::Create().ValueOrDie();
    auto output = std::make_shared<storage::RemoteOutputStream>(buffer);
    storage::IndexEntryDirectStreamWriter writer(output, 4096);
    artifact->Serialize(writer);
    writer.Finish();
    auto input = std::make_shared<storage::RemoteInputStream>(
        std::make_shared<arrow::io::BufferReader>(
            buffer->Finish().ValueOrDie()));
    auto source = storage::IndexEntryReader::Open(input, input->Size());
    auto resolved =
        index::ResolvePackedLoadFamily(family, source->IndexMeta(), params);
    if (params.contains(JSON_PATH))
        params = index::AnnotateJsonProjectionCompleteness(
            std::move(params), source->Directory(), source->IndexMeta());
    storage::LoadOptions options;
    options.params = std::move(params);
    options.mmap_dir_path = TestRemotePath;
    auto owner = index::LoaderRegistry::Instance().Lookup(resolved).Load(
        {index::OpenedIndexSource{index::PackedIndexSource{
             std::shared_ptr<storage::IndexEntryReader>(std::move(source))}},
         options,
         index::test::TestIndexCacheConfig()});
    return {std::move(owner), std::move(resolved)};
}

template <typename T>
BuiltReaderIndex
BuildReaderIndex(
    const std::string& family,
    size_t n,
    const T* values,
    const bool* valid = nullptr,
    Config params = Config::object(),
    std::function<storage::ArtifactPtr(storage::ArtifactPtr)> wrap = {}) {
    if constexpr (std::is_same_v<T, std::string>) {
        std::vector<std::string_view> views(values, values + n);
        return BuildReaderIndex(
            family, n, views.data(), valid, std::move(params), std::move(wrap));
    } else {
        if (!params.contains("field_type"))
            params["field_type"] = int(index::CppDataType<T>());
        if (!params.contains("value_type"))
            params["value_type"] = int(index::CppDataType<T>());
        params["nullable"] = valid != nullptr;
        params["num_rows"] = n;
        if (!params.contains("nested"))
            params["nested"] = false;
        params[index::FIELD_ID] = 100;
        params["local_dir"] = TestRemotePath;
        auto build_params = params;
        if (wrap)
            build_params["field_type"] = build_params["value_type"];
        auto builder =
            index::BuilderRegistry<index::ScalarBuildInput<T>>::Instance()
                .Create(family, build_params);
        AssertInfo(builder, "missing test builder");
        index::ScalarBuildBatch<T> batch{{values, n},
                                         ValidityView::FromExpanded(valid)};
        auto artifact =
            std::move(*builder).Build(index::ScalarBuildInput<T>{{&batch, 1}});
        if (wrap)
            artifact = wrap(std::move(artifact));
        return LoadReaderArtifact(
            std::move(artifact), family, std::move(params));
    }
}

inline std::string
ReaderFamily(const std::string& legacy) {
    if (legacy == "STL_SORT")
        return index::families::kSort;
    if (legacy == "BITMAP")
        return index::families::kBitmap;
    if (legacy == "INVERTED")
        return index::families::kInverted;
    if (legacy == "HYBRID")
        return index::families::kHybrid;
    if (legacy == "Trie")
        return index::families::kMarisa;
    return legacy;
}

// Real persisted sorted reader with overridable query methods for fault injection.
// The fixture owns the loaded Index and its request reader separately.
template <typename T>
class TestScalarIndex : public index::IIndexReaderBase,
                        public index::IScalarPredicateReader<QueryValue<T>>,
                        public index::INullReader,
                        public index::IScalarValueReader<QueryValue<T>> {
 public:
    explicit TestScalarIndex(const storage::FileManagerContext& = {},
                             bool nested = false)
        : nested_(nested) {
    }
    explicit TestScalarIndex(index::IIndexReaderBasePtr reader)
        : nested_(reader->Caps().nested), reader_(std::move(reader)) {
    }
    void
    Build(size_t n, const T* values, const bool* valid = nullptr) {
        auto built = BuildTestScalarIndex<T>(
            index::families::kSort,
            n,
            values,
            valid,
            Config{{"nested", nested_},
                   {"field_type",
                    int(nested_ ? DataType::ARRAY
                                : index::CppDataType<QueryValue<T>>())},
                   {"array_element_type",
                    int(index::CppDataType<QueryValue<T>>())}});
        index_ = std::move(built.index);
        reader_ = std::move(built.reader);
    }
    index::ReaderCaps
    Caps() const override {
        if (reader_)
            return reader_->Caps();
        index::ReaderCaps caps;
        caps.predicate = true;
        caps.nested = nested_;
        caps.value_lookup = caps.cheap_value_lookup = !nested_;
        return caps;
    }
    index::Domain
    CoordDomain() const override {
        return nested_ ? index::Domain::Element : index::Domain::Row;
    }
    int64_t
    Count() const override {
        return reader_ ? reader_->Count() : 0;
    }
    DataType
    ValueType() const override {
        return index::CppDataType<QueryValue<T>>();
    }
    TargetBitmap
    In(size_t n, const QueryValue<T>* values) const override {
        return Predicate().In(n, values);
    }
    TargetBitmap
    NotIn(size_t n, const QueryValue<T>* values) const override {
        return Predicate().NotIn(n, values);
    }
    TargetBitmap
    Range(const QueryValue<T>& v, index::CompareOp op) const override {
        return Predicate().Range(v, op);
    }
    TargetBitmap
    Range(const QueryValue<T>& lo,
          bool li,
          const QueryValue<T>& hi,
          bool ui) const override {
        return Predicate().Range(lo, li, hi, ui);
    }
    TargetBitmap
    IsNull() const override {
        return NullReader(*reader_).IsNull();
    }
    TargetBitmap
    IsNotNull() const override {
        return NullReader(*reader_).IsNotNull();
    }
    std::optional<index::owned_t<QueryValue<T>>>
    Lookup(int64_t row) const override {
        return Values().Lookup(row);
    }
    void
    Gather(const int64_t* offsets,
           int64_t count,
           const std::function<void(int64_t, const QueryValue<T>*, bool)>& out)
        const override {
        Values().Gather(offsets, count, out);
    }

 private:
    const index::IScalarPredicateReader<QueryValue<T>>&
    Predicate() const {
        return dynamic_cast<
            const index::IScalarPredicateReader<QueryValue<T>>&>(*reader_);
    }
    const index::IScalarValueReader<QueryValue<T>>&
    Values() const {
        return dynamic_cast<const index::IScalarValueReader<QueryValue<T>>&>(
            *reader_);
    }
    bool nested_;
    index::IndexPtr index_;
    index::IIndexReaderBasePtr reader_;
};
}  // namespace milvus::segcore::reader
