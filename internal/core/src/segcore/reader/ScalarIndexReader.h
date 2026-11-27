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
#include "segcore/reader/ScalarQueryView.h"
#include "index/contracts/IIndex.h"
#include "common/VectorAccess.h"
#include "segcore/reader/IndexReader.h"

namespace milvus::index {
class JsonKeyStats;

}  // namespace milvus::index
namespace milvus {
class IStructElementOffsets;
}

namespace milvus::segcore::reader {

enum class ScalarIndexOp {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Between,
    IsNull,
    IsNotNull,
    In,
    NotIn,
    Exists,
    ContainsAny,
    ContainsAll
};
struct Int64IndexQuery {
    ScalarIndexOp op;
    int64_t lower = 0;
    int64_t upper = 0;
    bool lower_inclusive = true;
    bool upper_inclusive = true;
    std::vector<int64_t> terms;
};

// Typed scalar/path query. JSON paths identify predicate inputs, never a request
// to reverse-lookup a complete JSON document.
struct TypedIndexQuery {
    ScalarIndexOp op;
    DataType type = DataType::NONE;
    proto::plan::GenericValue lower;
    proto::plan::GenericValue upper;
    bool lower_inclusive = true;
    bool upper_inclusive = true;
    std::vector<proto::plan::GenericValue> terms;
    std::string pointer;
    // JSON contains has operation-specific numeric coercion rules.
    bool same_type = true;
    // ARRAY membership is a document predicate even when its index stores
    // element IDs. These properties are proven from the compilation schema.
    bool nested_array = false;
    bool array_nullable = false;
    bool element_domain = false;
};

// Execute an audited row-domain scalar backend with an owned, pinned resource.
// JSON cast indexes supply a path-specific extraction validity mask.
IndexResult
QueryTypedIndex(const index::IIndexReaderBase& index,
                int64_t count,
                const TypedIndexQuery& query,
                const ReadContext& context);

struct NgramQuery {
    proto::plan::OpType op;
    std::string pattern;
    std::string pointer;
};
class NgramIndexReader final : public IndexReader {
 public:
    NgramIndexReader(index::IndexPtr resource, int64_t count, NgramQuery query)
        : resource_(std::move(resource)),
          count_(count),
          query_(std::move(query)) {
    }
    int64_t
    row_count() const override {
        return count_;
    }
    IndexResult
    Query(const ReadContext&) const override;

 private:
    index::IndexPtr resource_;
    int64_t count_;
    NgramQuery query_;
};

class ColumnReader;
IndexResult
QueryArrayIndex(const index::IIndexReaderBase& index,
                int64_t row_count,
                const TypedIndexQuery& query,
                const IStructElementOffsets* offsets,
                const ColumnReader* row_validity,
                const ReadContext& context);

// The typed descriptor handles scalar comparisons, ranges and IN. The legacy
// INT64 unary descriptor remains source-compatible with existing callers.
// Shredding and shared BSON both participate; missing/null/wrong-type is UNKNOWN.
struct JsonStatsArithmetic {
    proto::plan::ArithOpType operation;
    proto::plan::GenericValue operand;
    proto::plan::GenericValue bound;
    bool
    CanEvaluateWholeSource() const;
};
struct JsonStatsPath {
    std::vector<std::string> path;
    bool string_sensitive = false;
};
struct JsonStatsPathResult {
    ReadOnlyVectorView documents;
    OwnedSelection uncertain;
};
// Reconstruct only the addressed leaf/array ancestor into a batch-local JSON
// input. Numeric path components are resolved by the ordinary JSON kernel.
JsonStatsPathResult
ReadJsonStatsPath(index::JsonKeyStats& stats,
                  int64_t count,
                  const JsonStatsPath& path,
                  SelectionView rows,
                  const std::function<int64_t(size_t)>& offset,
                  const ReadContext& context);

struct JsonStatsQuery {
    std::string pointer;
    proto::plan::OpType op;
    int64_t bound;
    std::optional<TypedIndexQuery> predicate;
    // These alternatives are stats-only; never send them to a scalar index.
    std::optional<proto::plan::GenericValue> length_bound;
    std::optional<std::string> string_pattern;
    std::optional<JsonStatsArithmetic> arithmetic;
};
class JsonStatsReader final : public IndexReader {
 public:
    using Load =
        std::function<std::shared_ptr<index::JsonKeyStats>(const ReadContext&)>;
    JsonStatsReader(Load load, int64_t count, JsonStatsQuery query)
        : load_(std::move(load)), count_(count), query_(std::move(query)) {
    }
    int64_t
    row_count() const override {
        return count_;
    }
    IndexResult
    Query(const ReadContext&) const override;

 private:
    Load load_;
    int64_t count_;
    JsonStatsQuery query_;
};

// Ordinary row-level BOOL/signed-integer scalar indexes, exact Count() domain.
// Caller retains the immutable index generation through the shared owner.
// Nested/ARRAY indexes and unknown tails are NOT silently treated as row indexes.
// Bounds/terms retain int64 precision and are checked before narrowing.
// TypedIndexQuery handles floating-point/string and JSON path predicates.
template <typename T>
class IntegerIndexReader final : public IndexReader {
    static_assert(std::is_integral_v<T> &&
                  (std::is_signed_v<T> || std::is_same_v<T, bool>));

 public:
    IntegerIndexReader(std::shared_ptr<const index::IIndexReaderBase> index,
                       int64_t row_count,
                       Int64IndexQuery query);
    int64_t
    row_count() const override {
        return row_count_;
    }
    IndexResult
    Query(const ReadContext&) const override;

 private:
    std::shared_ptr<const index::IIndexReaderBase> index_;
    int64_t row_count_;
    const Int64IndexQuery query_;
};
using ScalarIndexReader = IntegerIndexReader<int64_t>;
}  // namespace milvus::segcore::reader
