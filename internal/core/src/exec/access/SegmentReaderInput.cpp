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

#include "exec/access/SegmentReaderInput.h"

#include "mmap/ChunkedColumn.h"
#include "mmap/ChunkedColumnGroup.h"
#include "segcore/SegmentGrowingImpl.h"
#include "segcore/reader/ColumnReaderAdapter.h"
#include "segcore/reader/GrowingColumnReader.h"
#include "segcore/json_stats/JsonKeyStats.h"
#include "index/contracts/query/IVectorReader.h"
#include "exec/access/ReaderInput.h"
#include "segcore/reader/IndexColumnReader.h"
#include <algorithm>
#include <map>

namespace milvus::exec::access {
namespace {
bool
SupportsRawColumn(const InputDependency& dependency) {
    if (!dependency.relation.empty() || dependency.upstream_column ||
        dependency.element_level || !dependency.path.empty() ||
        (dependency.element_type != DataType::NONE &&
         dependency.type != DataType::ARRAY &&
         dependency.type != DataType::VECTOR_ARRAY))
        return false;
    if (!dependency.needs_values && IsOrdinaryVectorDataType(dependency.type))
        return true;
    switch (dependency.type) {
        case DataType::BOOL:
        case DataType::INT8:
        case DataType::INT16:
        case DataType::INT32:
        case DataType::INT64:
        case DataType::TIMESTAMPTZ:
        case DataType::FLOAT:
        case DataType::DOUBLE:
        case DataType::VARCHAR:
        case DataType::TEXT:
        case DataType::STRING:
        case DataType::JSON:
        case DataType::GEOMETRY:
        case DataType::ARRAY:
        case DataType::VECTOR_ARRAY:
            return true;
        default:
            return false;
    }
}

// PK sources retain their resource generation and load only on first Eval.
// Results contain all physical versions; MVCC remains the consumer's job.
class PrimaryKeyReader final : public segcore::reader::IndexReader {
 public:
    PrimaryKeyReader(segcore::PrimaryKeyLookupLoader load,
                     int64_t count,
                     segcore::reader::TypedIndexQuery query)
        : load_(std::move(load)), count_(count), query_(std::move(query)) {
    }

    int64_t
    row_count() const override {
        return count_;
    }

    segcore::reader::IndexResult
    Query(const segcore::reader::ReadContext& context) const override {
        context.Check();
        auto source = load_(context.op_context.get());
        AssertInfo(source, "prepared PK loader returned no lookup");
        BitsetType matches(count_, false);
        BitsetTypeView view(matches);
        auto lookup = [&](const proto::plan::GenericValue& value,
                          proto::plan::OpType op,
                          BitsetTypeView& output) {
            context.Check();
            PkType pk = query_.type == DataType::VARCHAR
                            ? PkType(value.string_val())
                            : PkType(value.int64_val());
            source(pk, op, output);
        };
        using Op = segcore::reader::ScalarIndexOp;
        switch (query_.op) {
            case Op::In:
            case Op::NotIn:
                for (const auto& value : query_.terms)
                    lookup(value, proto::plan::Equal, view);
                if (query_.op == Op::NotIn)
                    matches.flip();
                break;
            case Op::Between: {
                lookup(query_.lower,
                       query_.lower_inclusive ? proto::plan::GreaterEqual
                                              : proto::plan::GreaterThan,
                       view);
                BitsetType upper(count_, false);
                BitsetTypeView upper_view(upper);
                lookup(query_.upper,
                       query_.upper_inclusive ? proto::plan::LessEqual
                                              : proto::plan::LessThan,
                       upper_view);
                matches &= upper;
                break;
            }
            case Op::IsNull:
                break;
            case Op::IsNotNull:
                matches.flip();
                break;
            default: {
                auto op = proto::plan::Equal;
                if (query_.op == Op::Less)
                    op = proto::plan::LessThan;
                if (query_.op == Op::LessEqual)
                    op = proto::plan::LessEqual;
                if (query_.op == Op::Greater)
                    op = proto::plan::GreaterThan;
                if (query_.op == Op::GreaterEqual)
                    op = proto::plan::GreaterEqual;
                lookup(query_.lower, op, view);
                if (query_.op == Op::NotEqual)
                    matches.flip();
                break;
            }
        }
        context.Check();
        return segcore::reader::IndexResult::Full(
            SelectionView::FromBitmap(matches), SelectionView::All(count_));
    }

 private:
    const segcore::PrimaryKeyLookupLoader load_;
    const int64_t count_;
    const segcore::reader::TypedIndexQuery query_;
};

std::optional<segcore::reader::TypedIndexQuery>
PrimaryKeyQuery(const plan::PredicatePlan& predicate) {
    const auto type = predicate.field.type;
    if (predicate.text_query || predicate.geometry_query ||
        predicate.ngram_query ||
        (type != DataType::INT64 && type != DataType::VARCHAR))
        return std::nullopt;
    segcore::reader::TypedIndexQuery q{};
    if (predicate.typed_query) {
        q = *predicate.typed_query;
    } else {
        if (type != DataType::INT64)
            return std::nullopt;
        const auto& source = predicate.query;
        q.type = type;
        q.op = source.op;
        q.lower.set_int64_val(source.lower);
        q.upper.set_int64_val(source.upper);
        q.lower_inclusive = source.lower_inclusive;
        q.upper_inclusive = source.upper_inclusive;
        for (auto term : source.terms)
            q.terms.emplace_back().set_int64_val(term);
    }
    if (!q.pointer.empty() || q.type != type)
        return std::nullopt;
    auto valid = [type](const proto::plan::GenericValue& v) {
        return type == DataType::INT64 ? v.has_int64_val() : v.has_string_val();
    };
    using Op = segcore::reader::ScalarIndexOp;
    switch (q.op) {
        case Op::IsNull:
        case Op::IsNotNull:
            return q;
        case Op::In:
        case Op::NotIn:
            if (std::all_of(q.terms.begin(), q.terms.end(), valid))
                return q;
            break;
        case Op::Between:
            if (valid(q.lower) && valid(q.upper))
                return q;
            break;
        case Op::Equal:
        case Op::NotEqual:
        case Op::Less:
        case Op::LessEqual:
        case Op::Greater:
        case Op::GreaterEqual:
            if (valid(q.lower))
                return q;
            break;
        default:
            break;
    }
    return std::nullopt;
}

struct PinnedSealedColumn {
    std::shared_ptr<const segcore::SegmentReadSnapshot> snapshot;
    std::shared_ptr<ChunkedColumnInterface> column;
};

using VectorIndexLoader =
    std::function<std::shared_ptr<const index::IIndexReaderBase>(
        const segcore::reader::ReadContext&)>;
using ValidityReader = segcore::reader::ValidityColumnReader;

template <typename T>
class ElementColumnReader final : public segcore::reader::PointColumnReader {
 public:
    ElementColumnReader(
        DataType type,
        int64_t count,
        int64_t documents,
        std::shared_ptr<const segcore::reader::ColumnReader> parent,
        std::shared_ptr<const IStructElementOffsets> offsets)
        : PointColumnReader(type, count),
          documents_(documents),
          parent_(std::move(parent)),
          offsets_(std::move(offsets)) {
    }

 protected:
    segcore::reader::ReadResult
    ReadPoints(SelectionView rows,
               segcore::reader::ReadProjection projection,
               const Offset& offset,
               const segcore::reader::ReadContext& ctx) const override {
        if (rows.IsNone())
            return ReadScalars<T>(
                rows,
                projection,
                [](size_t) { return false; },
                [](size_t) { return T{}; },
                ctx);
        ctx.Check();
        std::map<int32_t, size_t> positions;
        std::vector<int32_t> documents;
        std::vector<std::pair<size_t, int32_t>> mapping(rows.row_count());
        rows.ForEachSelected([&](size_t row) {
            ctx.Check();
            const auto [doc, element] = offsets_->ElementIDToRowID(offset(row));
            AssertInfo(doc >= 0 && doc < documents_ && element >= 0,
                       "element mapping exceeds prepared document prefix");
            auto [it, inserted] = positions.emplace(doc, documents.size());
            if (inserted)
                documents.push_back(doc);
            mapping[row] = {it->second, element};
        });
        auto arrays = parent_->Take(
            segcore::reader::TakeRequest(
                std::span<const int32_t>(documents),
                segcore::reader::ReadProjection::ValuesAndValidity,
                SelectionView::All(documents.size())),
            ctx);
        arrays.Validate(DataType::ARRAY,
                        SelectionView::All(documents.size()),
                        segcore::reader::ReadProjection::ValuesAndValidity);
        const auto& values = *arrays.values;
        return ReadScalars<T>(
            rows,
            projection,
            [&](size_t row) {
                auto [doc, element] = mapping[row];
                if (!values.IsValid(doc))
                    return false;
                const auto& array = values.ArrayViewAt(doc);
                const auto [begin, end] =
                    offsets_->ElementIDRangeOfRow(documents[doc]);
                AssertInfo(
                    array.length() == end - begin && element < array.length(),
                    "struct array disagrees with element mapping");
                return array.is_element_valid(element);
            },
            [&](size_t row) -> T {
                auto [doc, element] = mapping[row];
                return values.ArrayViewAt(doc).template get_data_unchecked<T>(
                    element);
            },
            ctx);
    }

 private:
    int64_t documents_;
    std::shared_ptr<const segcore::reader::ColumnReader> parent_;
    std::shared_ptr<const IStructElementOffsets> offsets_;
};

std::shared_ptr<const segcore::reader::ColumnReader>
VectorValidity(DataType type,
               int64_t count,
               VectorIndexLoader load,
               ValidityReader::ResolveValidity fallback) {
    return std::make_shared<ValidityReader>(
        type,
        count,
        ValidityReader::ResolveValidity{
            [load = std::move(load), fallback = std::move(fallback), count](
                const auto& ctx) -> ValidityReader::IsValid {
                if (load) {
                    auto owner = load(ctx);
                    auto* index =
                        dynamic_cast<const index::IVectorReader*>(owner.get());
                    AssertInfo(index, "missing prepared vector index");
                    if (index->HasValidData()) {
                        AssertInfo(owner->Count() >= static_cast<size_t>(count),
                                   "vector validity does not cover query rows");
                        return [owner = std::move(owner), index](int64_t row,
                                                                 const auto&) {
                            return index->IsRowValid(row);
                        };
                    }
                }
                // Only absent validity uses raw data. Pin/read failures and
                // malformed index domains propagate without changing sources.
                return fallback(ctx);
            }});
}

// Retain the immutable resource generation at Prepare, pin only on first
// predicate demand. A failed pin/query propagates and is never a raw fallback.
template <typename T>
class DeferredIntegerIndex final : public segcore::reader::IndexReader {
 public:
    DeferredIntegerIndex(index::IndexPtr resource,
                         int64_t count,
                         segcore::reader::Int64IndexQuery query)
        : resource_(std::move(resource)),
          count_(count),
          query_(std::move(query)) {
    }
    int64_t
    row_count() const override {
        return count_;
    }
    segcore::reader::IndexResult
    Query(const segcore::reader::ReadContext& context) const override {
        context.Check();
        auto owner = resource_->CreateReader(context.op_context.get());
        return segcore::reader::IntegerIndexReader<T>(
                   std::shared_ptr<const index::IIndexReaderBase>(
                       std::move(owner)),
                   count_,
                   query_)
            .Query(context);
    }

 private:
    index::IndexPtr resource_;
    int64_t count_;
    segcore::reader::Int64IndexQuery query_;
};
class DeferredTypedIndex final : public segcore::reader::IndexReader {
 public:
    DeferredTypedIndex(index::IndexPtr resource,
                       int64_t count,
                       segcore::reader::TypedIndexQuery query)
        : resource_(std::move(resource)),
          count_(count),
          query_(std::move(query)) {
    }
    int64_t
    row_count() const override {
        return count_;
    }
    segcore::reader::IndexResult
    Query(const segcore::reader::ReadContext& context) const override {
        context.Check();
        auto owner = resource_->CreateReader(context.op_context.get());
        return segcore::reader::QueryTypedIndex(
            *owner, count_, query_, context);
    }

 private:
    index::IndexPtr resource_;
    int64_t count_;
    segcore::reader::TypedIndexQuery query_;
};

class DeferredArrayIndex final : public segcore::reader::IndexReader {
 public:
    DeferredArrayIndex(
        index::IndexPtr resource,
        int64_t count,
        segcore::reader::TypedIndexQuery query,
        std::shared_ptr<const IStructElementOffsets> offsets,
        std::shared_ptr<const segcore::reader::ColumnReader> validity)
        : resource_(std::move(resource)),
          count_(count),
          query_(std::move(query)),
          offsets_(std::move(offsets)),
          validity_(std::move(validity)) {
    }
    int64_t
    row_count() const override {
        return count_;
    }
    segcore::reader::IndexResult
    Query(const segcore::reader::ReadContext& context) const override {
        context.Check();
        auto owner = resource_->CreateReader(context.op_context.get());
        return segcore::reader::QueryArrayIndex(
            *owner, count_, query_, offsets_.get(), validity_.get(), context);
    }

 private:
    index::IndexPtr resource_;
    int64_t count_;
    segcore::reader::TypedIndexQuery query_;
    std::shared_ptr<const IStructElementOffsets> offsets_;
    std::shared_ptr<const segcore::reader::ColumnReader> validity_;
};

}  // namespace

BindColumnReader
MakeElementReaderBinding(BindColumnReader parents,
                         std::shared_ptr<const Schema> schema,
                         std::shared_ptr<const IStructElementOffsets> offsets,
                         int64_t document_count,
                         int64_t element_count) {
    AssertInfo(parents && schema && offsets && document_count >= 0 &&
                   document_count <= offsets->GetRowCount(),
               "invalid element reader domain");
    return [parents = std::move(parents),
            schema = std::move(schema),
            offsets = std::move(offsets),
            document_count,
            element_count](const InputDependency& dependency)
               -> std::shared_ptr<const segcore::reader::ColumnReader> {
        const auto& field = (*schema)[FieldId(dependency.field_id)];
        if (field.get_data_type() != DataType::ARRAY ||
            field.is_nested_array() ||
            field.get_element_type() != dependency.type)
            return nullptr;
        auto raw = dependency;
        raw.type = DataType::ARRAY;
        raw.element_type = dependency.type;
        raw.nullable = field.is_nullable();
        raw.needs_values = true;
        auto parent = parents(raw);
        if (!parent)
            return nullptr;
        auto bind = [&]<typename T>()
            -> std::shared_ptr<const segcore::reader::ColumnReader> {
            return std::make_shared<ElementColumnReader<T>>(dependency.type,
                                                            element_count,
                                                            document_count,
                                                            parent,
                                                            offsets);
        };
        switch (dependency.type) {
            case DataType::BOOL:
                return bind.template operator()<bool>();
            case DataType::INT8:
                return bind.template operator()<int8_t>();
            case DataType::INT16:
                return bind.template operator()<int16_t>();
            case DataType::INT32:
                return bind.template operator()<int32_t>();
            case DataType::INT64:
                return bind.template operator()<int64_t>();
            case DataType::FLOAT:
                return bind.template operator()<float>();
            case DataType::DOUBLE:
                return bind.template operator()<double>();
            case DataType::VARCHAR:
            case DataType::STRING:
                return bind.template operator()<std::string>();
            default:
                return nullptr;
        }
    };
}

plan::BindIndex
MakeElementIndexBinding(
    std::shared_ptr<const segcore::SegmentReadSnapshot> snapshot,
    std::shared_ptr<const Schema> schema,
    std::shared_ptr<const IStructElementOffsets> offsets,
    std::shared_ptr<const segcore::reader::ReadContext> context) {
    if (!snapshot || !schema || !offsets)
        return {};
    return [snapshot, schema, offsets, context](
               const plan::PredicatePlan& predicate)
               -> std::optional<plan::PreparedIndex> {
        const auto& field = predicate.field;
        const auto& meta = (*schema)[FieldId(field.field_id)];
        if (meta.get_data_type() != DataType::ARRAY || meta.is_nested_array() ||
            !GetStructNameForArrayField(meta) || meta.is_element_nullable() ||
            meta.get_element_type() != field.type || field.upstream_column ||
            field.element_level || !field.path.empty() ||
            !field.relation.empty() || predicate.ngram_query ||
            predicate.selected_stats || predicate.match ||
            predicate.geometry_query || predicate.text_query)
            return std::nullopt;
        auto own_offsets =
            snapshot->GetStructElementOffsets(FieldId(field.field_id));
        if (!own_offsets)
            return std::nullopt;
        AssertInfo(own_offsets->GetRowCount() == offsets->GetRowCount() &&
                       own_offsets->GetTotalElementCount() ==
                           offsets->GetTotalElementCount(),
                   "nested index and element binding have different domains");
        for (int64_t row = 0; row < offsets->GetRowCount(); ++row) {
            context->Check();
            AssertInfo(own_offsets->ElementIDRangeOfRow(row) ==
                           offsets->ElementIDRangeOfRow(row),
                       "struct fields disagree on element positions");
        }
        segcore::reader::TypedIndexQuery query{};
        if (predicate.typed_query)
            query = *predicate.typed_query;
        else {
            const auto& q = predicate.query;
            query.op = q.op;
            query.type = field.type;
            query.lower.set_int64_val(q.lower);
            query.upper.set_int64_val(q.upper);
            query.lower_inclusive = q.lower_inclusive;
            query.upper_inclusive = q.upper_inclusive;
            for (auto term : q.terms)
                query.terms.emplace_back().set_int64_val(term);
            if (field.type == DataType::BOOL) {
                query.lower.set_bool_val(q.lower != 0);
                query.upper.set_bool_val(q.upper != 0);
                for (auto& term : query.terms) {
                    auto v = term.int64_val();
                    term.set_bool_val(v != 0);
                }
            }
            auto fits = [&](int64_t v) {
                switch (field.type) {
                    case DataType::INT8:
                        return v >= INT8_MIN && v <= INT8_MAX;
                    case DataType::INT16:
                        return v >= INT16_MIN && v <= INT16_MAX;
                    case DataType::INT32:
                        return v >= INT32_MIN && v <= INT32_MAX;
                    default:
                        return true;
                }
            };
            if (!fits(q.lower) || !fits(q.upper))
                return std::nullopt;
            for (auto term : q.terms)
                if (!fits(term))
                    return std::nullopt;
        }
        if (!query.pointer.empty())
            return std::nullopt;
        query.element_domain = true;
        index::IndexPtr resource;
        for (auto type : {"INVERTED", "BITMAP", "HYBRID", "STL_SORT"}) {
            resource = snapshot->GetScalarIndexResource(
                FieldId(field.field_id), type, context->op_context.get());
            if (resource)
                break;
        }
        if (!resource)
            return std::nullopt;
        return plan::PreparedIndex{
            segcore::reader::IndexResultKind::FullThreeValued,
            std::make_shared<DeferredTypedIndex>(
                resource, offsets->GetTotalElementCount(), query),
            context};
    };
}

plan::BindIndex
MakeSealedIndexBinding(
    std::shared_ptr<const segcore::SegmentReadSnapshot> snapshot,
    int64_t active_count,
    std::shared_ptr<const segcore::reader::ReadContext> context,
    bool use_json_stats) {
    if (!snapshot)
        return {};
    AssertInfo(context && active_count >= 0 &&
                   active_count <= snapshot->get_row_count(),
               "invalid scalar index snapshot domain");
    return [snapshot = std::move(snapshot),
            context = std::move(context),
            use_json_stats,
            active_count](const plan::PredicatePlan& predicate)
               -> std::optional<plan::PreparedIndex> {
        const auto& field = predicate.field;
        if (!field.relation.empty() || field.upstream_column ||
            field.element_level || !field.path.empty())
            return std::nullopt;
        if (predicate.match) {
            const auto& match = *predicate.match;
            auto offsets =
                snapshot->GetStructElementOffsets(FieldId(match.anchor));
            if (!offsets || active_count > offsets->GetRowCount() ||
                offsets->GetTotalElementCount() >
                    std::numeric_limits<vector_size_t>::max())
                return std::nullopt;
            const auto elements =
                active_count == 0
                    ? 0
                    : offsets->ElementIDRangeOfRow(active_count - 1).second;
            plan::PhysicalPrepareOptions physical;
            physical.bind_index = MakeElementIndexBinding(
                snapshot, match.schema, offsets, context);
            auto selected = plan::AdjustExecution(*match.child, physical);
            if (!selected.supported ||
                std::none_of(selected.routes.begin(),
                             selected.routes.end(),
                             [](auto route) {
                                 return route ==
                                        plan::PredicateRoute::ExactIndex;
                             }))
                return std::nullopt;
            auto parents = MakeSealedReaderBinding(
                snapshot, active_count, context->op_context.get());
            auto bind = MakeElementReaderBinding(
                parents, match.schema, offsets, active_count, elements);
            auto requirements =
                plan::RequiredInputs(*match.child, selected.raw_inputs);
            std::vector<std::optional<plan::PreparedInput>> inputs(
                requirements.size());
            for (size_t i = 0; i < requirements.size(); ++i)
                if (requirements[i]) {
                    auto reader = bind(*requirements[i]);
                    if (!reader)
                        return std::nullopt;
                    inputs[i] = PrepareReaderInput(
                        reader, context, requirements[i]->needs_values);
                }
            auto child = std::make_shared<plan::PreparedExecution>(
                match.child, std::move(inputs), std::move(selected));
            auto dependency = predicate.field;
            dependency.field_id = match.anchor;
            dependency.needs_values = false;
            dependency.needs_validity = true;
            std::shared_ptr<const segcore::reader::ColumnReader> validity;
            if (!dependency.nullable)
                validity = std::make_shared<ValidityReader>(
                    dependency.type, active_count, [](int64_t, const auto&) {
                        return true;
                    });
            else
                validity = parents(dependency);
            if (!validity)
                return std::nullopt;
            plan::PreparedIndex result{
                segcore::reader::IndexResultKind::FullThreeValued,
                nullptr,
                context};
            // Nullable parent validity is an input even for an all-index child.
            result.force_batch = dependency.nullable ||
                                 child->granularity() !=
                                     plan::ExecutionGranularity::WholeSegment;
            result.selected_route =
                child->granularity() == plan::ExecutionGranularity::WholeSegment
                    ? plan::PredicateRoute::ExactIndex
                    : plan::PredicateRoute::Refine;
            result.selected_source = [child, validity, offsets, context, match](
                                         const ReadBatchBinding& binding,
                                         SelectionView rows,
                                         scalar::EvalCtx& ctx) {
                using namespace segcore::reader;
                auto parent =
                    binding.is_range()
                        ? validity->ReadRange(
                              RangeRequest(binding.range_begin(),
                                           binding.size(),
                                           ReadProjection::ValidityOnly,
                                           rows),
                              *context)
                        : validity->Take(
                              TakeRequest(binding.offsets(),
                                          ReadProjection::ValidityOnly,
                                          rows),
                              *context);
                parent.Validate(
                    validity->type(), rows, ReadProjection::ValidityOnly);
                BitmapResultBuilder output(binding.size());
                rows.ForEachSelected([&](size_t row) {
                    ctx.Checkpoint();
                    if (!parent.valid.view().IsSelected(row))
                        return;
                    auto [begin, end] =
                        offsets->ElementIDRangeOfRow(binding.offset(row));
                    int64_t hits = 0, known = 0;
                    if (end != begin) {
                        auto batch = child->MakeExecutionBatch(
                            ReadBatchBinding::Range(begin, end - begin));
                        auto values =
                            batch->Eval(SelectionView::All(end - begin),
                                        [&] { ctx.Checkpoint(); })[0];
                        for (size_t i = 0; i < values.size(); ++i) {
                            if (i % 8192 == 0)
                                ctx.Checkpoint();
                            if (values.IsValid(i)) {
                                ++known;
                                hits += values.ValueAt<bool>(i);
                            }
                        }
                    }
                    bool value = false;
                    switch (match.operation) {
                        case proto::plan::MatchAny:
                            value = hits != 0;
                            break;
                        case proto::plan::MatchAll:
                            value = hits == known;
                            break;
                        case proto::plan::MatchLeast:
                            value = hits >= match.threshold;
                            break;
                        case proto::plan::MatchMost:
                            value = hits <= match.threshold;
                            break;
                        case proto::plan::MatchExact:
                            value = hits == match.threshold;
                            break;
                        default:
                            ThrowInfo(UnexpectedError,
                                      "invalid prepared MATCH operation");
                    }
                    output.Set<bool>(row, value);
                });
                ctx.Checkpoint();
                return std::move(output).Finish(rows);
            };
            return result;
        }
        if (auto query = PrimaryKeyQuery(predicate)) {
            auto load = snapshot->GetPrimaryKeyLookup(FieldId(field.field_id),
                                                      field.type);
            if (load)
                return plan::PreparedIndex{
                    segcore::reader::IndexResultKind::FullThreeValued,
                    std::make_shared<PrimaryKeyReader>(
                        std::move(load), active_count, std::move(*query)),
                    context,
                    plan::PredicateBackend::PkIndex};
        }
        auto ordinary = [&]() -> index::IndexPtr {
            for (const auto* type :
                 {"STL_SORT", "INVERTED", "BITMAP", "HYBRID", "Trie", "TRIE"}) {
                auto resource = snapshot->GetScalarIndexResource(
                    FieldId(field.field_id), type, context->op_context.get());
                if (resource)
                    return resource;
            }
            return nullptr;
        };
        if (predicate.ngram_query) {
            auto resource =
                snapshot->GetNgramResource(FieldId(field.field_id),
                                           predicate.ngram_query->pointer,
                                           context->op_context.get());
            if (resource)
                return plan::PreparedIndex{
                    segcore::reader::IndexResultKind::Candidates,
                    std::make_shared<segcore::reader::NgramIndexReader>(
                        std::move(resource),
                        snapshot->get_row_count(),
                        *predicate.ngram_query),
                    context,
                    plan::PredicateBackend::ScalarIndex};
            if (!predicate.json_stats)
                return std::nullopt;
        }
        if (predicate.geometry_query) {
            auto resource = snapshot->GetScalarIndexResource(
                FieldId(field.field_id), "RTREE", context->op_context.get());
            if (!resource)
                return std::nullopt;
            return plan::PreparedIndex{
                segcore::reader::IndexResultKind::Candidates,
                std::make_shared<segcore::reader::GeometryIndexReader>(
                    std::move(resource),
                    snapshot->get_row_count(),
                    *predicate.geometry_query),
                context,
                plan::PredicateBackend::ScalarIndex};
        }
        if (predicate.text_query) {
            auto load = snapshot->GetTextIndexResource(
                FieldId(field.field_id), context->op_context.get());
            if (!load)
                return std::nullopt;
            return plan::PreparedIndex{
                segcore::reader::IndexResultKind::FullThreeValued,
                std::make_shared<segcore::reader::TextIndexReader>(
                    [load = std::move(load)](const auto& ctx) {
                        return load(ctx.op_context.get());
                    },
                    snapshot->get_row_count(),
                    *predicate.text_query),
                context,
                plan::PredicateBackend::TextIndex};
        }
        if (field.type == DataType::ARRAY && predicate.typed_query) {
            const auto& query = *predicate.typed_query;
            index::IndexPtr resource;
            for (const auto* type :
                 {"BITMAP", "INVERTED", "HYBRID", "STL_SORT"}) {
                if (std::string_view(type) == "STL_SORT" && !query.nested_array)
                    continue;
                resource = snapshot->GetScalarIndexResource(
                    FieldId(field.field_id), type, context->op_context.get());
                if (resource)
                    break;
            }
            if (!resource)
                return std::nullopt;
            std::shared_ptr<const IStructElementOffsets> offsets;
            std::shared_ptr<const segcore::reader::ColumnReader> validity;
            if (query.nested_array) {
                offsets =
                    snapshot->GetStructElementOffsets(FieldId(field.field_id));
                if (!offsets)
                    return std::nullopt;
                if (query.array_nullable) {
                    auto dependency = field;
                    dependency.needs_values = false;
                    dependency.needs_validity = true;
                    validity = MakeSealedReaderBinding(
                        snapshot,
                        snapshot->get_row_count(),
                        context->op_context.get())(dependency);
                    if (!validity)
                        return std::nullopt;
                }
            }
            return plan::PreparedIndex{
                segcore::reader::IndexResultKind::FullThreeValued,
                std::make_shared<DeferredArrayIndex>(resource,
                                                     snapshot->get_row_count(),
                                                     query,
                                                     std::move(offsets),
                                                     std::move(validity)),
                context,
                plan::PredicateBackend::ScalarIndex};
        }
        if (predicate.typed_query) {
            const auto& query = *predicate.typed_query;
            index::IndexPtr resource;
            if (!query.pointer.empty()) {
                auto safe_cast = [&](const proto::plan::GenericValue& value) {
                    if (query.op == segcore::reader::ScalarIndexOp::Between &&
                        value.has_float_val() &&
                        std::abs(value.float_val()) >= 0x1p53)
                        return false;
                    return !value.has_int64_val() ||
                           (value.int64_val() > -(int64_t{1} << 53) &&
                            value.int64_val() < (int64_t{1} << 53));
                };
                bool allow_cast =
                    safe_cast(query.lower) && safe_cast(query.upper);
                for (const auto& term : query.terms)
                    allow_cast &= safe_cast(term);
                resource = snapshot->GetJsonPredicateResource(
                    FieldId(field.field_id),
                    query.pointer,
                    query.type,
                    allow_cast,
                    context->op_context.get());
            } else {
                resource = ordinary();
            }
            if (resource) {
                return plan::PreparedIndex{
                    segcore::reader::IndexResultKind::FullThreeValued,
                    std::make_shared<DeferredTypedIndex>(
                        resource, snapshot->get_row_count(), query),
                    context,
                    plan::PredicateBackend::ScalarIndex};
            }
            if (!predicate.json_stats)
                return std::nullopt;
        }
        if (predicate.selected_stats) {
            if (!use_json_stats)
                return std::nullopt;
            auto resource =
                snapshot->GetJsonStatsResource(FieldId(field.field_id));
            if (!resource)
                return std::nullopt;
            const auto descriptor = *predicate.selected_stats;
            const auto slot = predicate.raw_inputs.front().slot;
            plan::PreparedIndex prepared{
                segcore::reader::IndexResultKind::FullThreeValued,
                nullptr,
                context,
                plan::PredicateBackend::JsonStats};
            prepared.selected_needs_raw = descriptor.string_sensitive;
            // This source materializes projected values for the requested rows;
            // unlike IndexNode's bitmaps, its memory must stay batch-bounded.
            prepared.force_batch = true;
            prepared.selected_source = [resource,
                                        descriptor,
                                        slot,
                                        exact = predicate.exact,
                                        count = snapshot->get_row_count(),
                                        context](
                                           const ReadBatchBinding& binding,
                                           SelectionView rows,
                                           scalar::EvalCtx& ctx) {
                auto pin = SemiInlineGet(
                    resource->PinCells(context->op_context.get(), {0}));
                auto* stats = pin->get_cell_of(0);
                AssertInfo(stats && stats->Count() == count,
                           "selected stats snapshot domain mismatch");
                auto data = segcore::reader::ReadJsonStatsPath(
                    *stats,
                    count,
                    descriptor,
                    rows,
                    [&](size_t row) { return binding.offset(row); },
                    *context);
                // Repair only ambiguous historic shredded strings. Reading an
                // empty string as NULL would change BOOL projection and NOT.
                if (!data.uncertain.view().IsNone()) {
                    auto raw =
                        ctx.input().GetColumn(slot, data.uncertain.view());
                    VectorResultBuilder repaired(DataType::JSON,
                                                 binding.size());
                    rows.ForEachSelected([&](size_t row) {
                        const auto& source =
                            data.uncertain.view().IsSelected(row)
                                ? raw
                                : data.documents;
                        if (source.IsValid(row))
                            repaired.Set<Json>(row, source.JsonViewAt(row));
                    });
                    data.documents = std::move(repaired).Finish(rows);
                }
                BatchInput input(ctx.input().schema(), binding.size());
                input.BindValues(slot, std::move(data.documents));
                scalar::EvalCtx local(input, [&] {
                    ctx.Checkpoint();
                    context->Check();
                });
                return exact->Eval(rows, local);
            };
            return prepared;
        }
        if (predicate.json_stats) {
            if (!use_json_stats)
                return std::nullopt;
            auto stats =
                snapshot->GetJsonStatsResource(FieldId(field.field_id));
            if (!stats)
                return std::nullopt;
            auto load = [stats = std::move(stats)](
                            const segcore::reader::ReadContext& ctx) {
                auto pin =
                    SemiInlineGet(stats->PinCells(ctx.op_context.get(), {0}));
                auto* value = pin->get_cell_of(0);
                return std::shared_ptr<index::JsonKeyStats>(std::move(pin),
                                                            value);
            };
            return plan::PreparedIndex{
                // Existing shredded string stats encode empty strings as
                // NULL. They can prove FALSE, but require raw refinement for
                // every other row; never advertise full three-valued results.
                predicate.json_stats->string_pattern
                    ? segcore::reader::IndexResultKind::Candidates
                    : segcore::reader::IndexResultKind::FullThreeValued,
                std::make_shared<segcore::reader::JsonStatsReader>(
                    std::move(load),
                    snapshot->get_row_count(),
                    *predicate.json_stats),
                context,
                plan::PredicateBackend::JsonStats};
        }
        auto resource = ordinary();
        if (!resource)
            return std::nullopt;
        auto bind = [&]<typename T>() -> std::optional<plan::PreparedIndex> {
            return plan::PreparedIndex{
                segcore::reader::IndexResultKind::FullThreeValued,
                std::make_shared<DeferredIntegerIndex<T>>(
                    resource, snapshot->get_row_count(), predicate.query),
                context};
        };
        switch (field.type) {
            case DataType::BOOL:
                return bind.template operator()<bool>();
            case DataType::INT8:
                return bind.template operator()<int8_t>();
            case DataType::INT16:
                return bind.template operator()<int16_t>();
            case DataType::INT32:
                return bind.template operator()<int32_t>();
            case DataType::INT64:
            case DataType::TIMESTAMPTZ:
                return bind.template operator()<int64_t>();
            default:
                return std::nullopt;
        }
    };
}

plan::BindIndex
MakeSegmentIndexBinding(
    const segcore::SegmentInternalInterface& segment,
    std::shared_ptr<const segcore::SegmentReadSnapshot> snapshot,
    int64_t active_count,
    std::shared_ptr<const segcore::reader::ReadContext> context,
    bool use_json_stats) {
    if (segment.type() == SegmentType::Sealed)
        return MakeSealedIndexBinding(std::move(snapshot),
                                      active_count,
                                      std::move(context),
                                      use_json_stats);
    auto schema = segment.get_schema_snapshot();
    const auto* growing =
        dynamic_cast<const segcore::SegmentGrowingImpl*>(&segment);
    auto pk_offsets = growing ? growing->get_insert_record().pk2offset_
                              : std::shared_ptr<segcore::OffsetMap>{};
    return [&segment,
            active_count,
            schema = std::move(schema),
            pk_offsets = std::move(pk_offsets),
            context = std::move(context)](const plan::PredicatePlan& predicate)
               -> std::optional<plan::PreparedIndex> {
        const auto& field = predicate.field;
        if (!field.relation.empty() || field.upstream_column ||
            field.element_level || !field.path.empty())
            return std::nullopt;
        if (pk_offsets &&
            schema->get_primary_field_id() == FieldId(field.field_id) &&
            schema->operator[](FieldId(field.field_id)).get_data_type() ==
                field.type &&
            !schema->operator[](FieldId(field.field_id)).is_nullable()) {
            if (auto query = PrimaryKeyQuery(predicate)) {
                segcore::PrimaryKeyLookupLoader load =
                    [pk_offsets](OpContext*) -> segcore::PrimaryKeyLookup {
                    return [pk_offsets](const PkType& pk, auto op, auto& bits) {
                        pk_offsets->find_range(
                            pk, op, bits, [&](int64_t offset) {
                                return offset >= 0 && offset < bits.size();
                            });
                    };
                };
                return plan::PreparedIndex{
                    segcore::reader::IndexResultKind::FullThreeValued,
                    std::make_shared<PrimaryKeyReader>(
                        std::move(load), active_count, std::move(*query)),
                    context,
                    plan::PredicateBackend::PkIndex};
            }
        }
        if (!predicate.text_query)
            return std::nullopt;
        auto pin = segment.PinGrowingIndex(context->op_context.get(),
                                           FieldId(predicate.field.field_id));
        if (!pin || !pin.Reader().Caps().text_match)
            return std::nullopt;
        auto owner =
            std::make_shared<index::GrowingIndexSnapshotPin>(std::move(pin));
        return plan::PreparedIndex{
            segcore::reader::IndexResultKind::FullThreeValued,
            std::make_shared<segcore::reader::TextIndexReader>(
                [owner](const auto&) {
                    return std::shared_ptr<const index::IIndexReaderBase>(
                        owner, &owner->Reader());
                },
                active_count,
                *predicate.text_query,
                owner->CoveredRowEnd()),
            context,
            plan::PredicateBackend::TextIndex};
    };
}

namespace {
class CachedGeometryReader final : public segcore::reader::ColumnReader {
 public:
    CachedGeometryReader(std::shared_ptr<const ColumnReader> reader,
                         std::shared_ptr<SimpleGeometryCache> cache)
        : reader_(std::move(reader)), cache_(std::move(cache)) {
    }
    DataType
    type() const override {
        return reader_->type();
    }
    int64_t
    row_count() const override {
        return reader_->row_count();
    }
    std::shared_ptr<SimpleGeometryCache>
    geometry_cache() const override {
        return cache_;
    }
    segcore::reader::ReadResult
    ReadRange(const segcore::reader::RangeRequest& r,
              const segcore::reader::ReadContext& c) const override {
        return reader_->ReadRange(r, c);
    }
    segcore::reader::ReadResult
    Take(const segcore::reader::TakeRequest& r,
         const segcore::reader::ReadContext& c) const override {
        return reader_->Take(r, c);
    }

 private:
    std::shared_ptr<const ColumnReader> reader_;
    std::shared_ptr<SimpleGeometryCache> cache_;
};
}  // namespace
BindColumnReader
MakeSealedReaderBinding(
    std::shared_ptr<const segcore::SegmentReadSnapshot> snapshot,
    int64_t active_count,
    OpContext* op_context) {
    AssertInfo(active_count >= 0, "negative reader domain");
    if (!snapshot)
        return {};
    AssertInfo(active_count <= snapshot->get_row_count(),
               "FilterBits active domain exceeds its snapshot");
    return [snapshot = std::move(snapshot), active_count, op_context](
               const InputDependency& dependency)
               -> std::shared_ptr<const segcore::reader::ColumnReader> {
        if (!SupportsRawColumn(dependency))
            return nullptr;
        const FieldId field(dependency.field_id);
        if (IsOrdinaryVectorDataType(dependency.type)) {
            if (!dependency.nullable)
                return std::make_shared<ValidityReader>(
                    dependency.type, active_count, [](int64_t, const auto&) {
                        return true;
                    });
            auto resource = snapshot->GetVectorIndexResource(field, op_context);
            // When an index exists, do not even resolve its raw-column
            // fallback until the index reports that validity is absent.
            if (!resource && !snapshot->GetDataScanResources(field).first)
                return nullptr;
            VectorIndexLoader load;
            if (resource)
                load = [resource = std::move(resource)](const auto& ctx) {
                    return std::shared_ptr<const index::IIndexReaderBase>(
                        resource->CreateReader(ctx.op_context.get()));
                };
            return VectorValidity(
                dependency.type,
                active_count,
                std::move(load),
                [snapshot, field, active_count](
                    const auto&) -> ValidityReader::IsValid {
                    auto column = snapshot->GetDataScanResources(field).first;
                    AssertInfo(column && column->NumRows() >= active_count,
                               "vector validity is absent from both index and "
                               "raw column for field {}",
                               field.get());
                    return [snapshot, column = std::move(column)](
                               int64_t row, const auto& ctx) {
                        return !column->IsNullable() ||
                               column->IsValid(ctx.op_context.get(), row);
                    };
                });
        }
        auto column = snapshot->GetDataScanResources(field).first;
        if (!column) {
            index::IndexPtr resource;
            for (const auto* backend : {"STL_SORT", "BITMAP", "Trie", "TRIE"}) {
                resource = snapshot->GetScalarIndexResource(
                    field, backend, op_context);
                if (resource)
                    break;
            }
            if (!resource)
                return nullptr;
            const auto metadata = resource->Metadata();
            if (!metadata.caps.value_lookup ||
                !metadata.caps.cheap_value_lookup || metadata.caps.json_paths ||
                metadata.domain != index::Domain::Row)
                return nullptr;
            auto bind = [&]<typename T>()
                -> std::shared_ptr<const segcore::reader::ColumnReader> {
                return std::make_shared<segcore::reader::IndexColumnReader<T>>(
                    dependency.type,
                    active_count,
                    snapshot->get_row_count(),
                    [resource](const auto& ctx) {
                        return std::shared_ptr<const index::IIndexReaderBase>(
                            resource->CreateReader(ctx.op_context.get()));
                    });
            };
            switch (dependency.type) {
                case DataType::BOOL:
                    return bind.template operator()<bool>();
                case DataType::INT8:
                    return bind.template operator()<int8_t>();
                case DataType::INT16:
                    return bind.template operator()<int16_t>();
                case DataType::INT32:
                    return bind.template operator()<int32_t>();
                case DataType::INT64:
                case DataType::TIMESTAMPTZ:
                    return bind.template operator()<int64_t>();
                case DataType::FLOAT:
                    return bind.template operator()<float>();
                case DataType::DOUBLE:
                    return bind.template operator()<double>();
                case DataType::STRING:
                case DataType::VARCHAR:
                case DataType::TEXT:
                    return bind.template operator()<std::string>();
                default:
                    return nullptr;
            }
        }
        if (!column ||
            (!dynamic_cast<const ChunkedColumnBase*>(column.get()) &&
             !dynamic_cast<const ProxyChunkColumn*>(column.get())) ||
            (active_count > 0 &&
             snapshot->num_chunk_data(field, op_context) == 0))
            return nullptr;
        auto owner = std::make_shared<PinnedSealedColumn>(
            PinnedSealedColumn{snapshot, std::move(column)});
        auto pinned = std::shared_ptr<const ChunkedColumnInterface>(
            owner, owner->column.get());
        auto reader = std::make_shared<segcore::reader::ColumnReaderAdapter>(
            std::move(pinned),
            dependency.type,
            active_count,
            dependency.type == DataType::ARRAY &&
                dependency.element_type == DataType::ARRAY);
        if (dependency.type == DataType::GEOMETRY) {
            if (auto cache = snapshot->GetGeometryCache(field))
                return std::make_shared<CachedGeometryReader>(std::move(reader),
                                                              std::move(cache));
        }
        return reader;
    };
}

segcore::OffsetBatchIterator
MakeSegmentOffsetIterator(
    const segcore::SegmentInternalInterface& segment,
    std::shared_ptr<const segcore::SegmentReadSnapshot> snapshot,
    BitsetTypeView excluded,
    const segcore::reader::ReadContext& context) {
    context.Check();
    if (segment.type() == SegmentType::Sealed) {
        AssertInfo(snapshot, "sealed projection requires a request snapshot");
        return snapshot->MakeOffsetIterator(
            excluded, context.op_context.get(), context.checkpoint);
    }
    const auto* growing =
        dynamic_cast<const segcore::SegmentGrowingImpl*>(&segment);
    AssertInfo(
        growing && excluded.size() <= growing->get_insert_record().row_count(),
        "offset iterator exceeds acknowledged growing prefix");
    auto owner = growing->get_insert_record().pk2offset_;
    AssertInfo(owner, "growing primary key map is not ready");
    auto next = owner->make_offset_iterator(excluded, context.checkpoint);
    return [owner = std::move(owner), next = std::move(next)](
               size_t max_rows) mutable { return next(max_rows); };
}

BindColumnReader
MakeSegmentReaderBinding(
    const segcore::SegmentInternalInterface& segment,
    std::shared_ptr<const segcore::SegmentReadSnapshot> snapshot,
    int64_t active_count,
    OpContext* op_context) {
    BindColumnReader base;
    if (segment.type() == SegmentType::Sealed) {
        base = MakeSealedReaderBinding(snapshot, active_count, op_context);
    } else {
        const auto* growing =
            dynamic_cast<const segcore::SegmentGrowingImpl*>(&segment);
        if (!growing)
            return {};
        base = MakeGrowingReaderBinding(growing->get_insert_record(),
                                        growing->get_schema_snapshot(),
                                        active_count);
    }
    if (!base)
        return {};
    return [base = std::move(base),
            snapshot = std::move(snapshot),
            &segment,
            active_count,
            op_context](const InputDependency& dependency)
               -> std::shared_ptr<const segcore::reader::ColumnReader> {
        if (!snapshot && SupportsRawColumn(dependency) && dependency.nullable &&
            IsOrdinaryVectorDataType(dependency.type)) {
            auto pin = segment.PinGrowingIndex(op_context,
                                               FieldId(dependency.field_id));
            if (pin && pin.CoveredRowEnd() >= active_count) {
                auto* vector =
                    dynamic_cast<const index::IVectorReader*>(&pin.Reader());
                if (vector && vector->HasValidData()) {
                    AssertInfo(
                        pin.Reader().Count() >= active_count,
                        "growing vector validity does not cover query prefix");
                    auto owner =
                        std::make_shared<index::GrowingIndexSnapshotPin>(
                            std::move(pin));
                    return std::make_shared<ValidityReader>(
                        dependency.type,
                        active_count,
                        [owner, vector](int64_t row, const auto&) {
                            return vector->IsRowValid(row);
                        });
                }
            }
        }
        auto reader = base(dependency);
        if (!reader || dependency.type != DataType::GEOMETRY ||
            reader->geometry_cache())
            return reader;
        auto cache =
            snapshot ? snapshot->GetGeometryCache(FieldId(dependency.field_id))
                     : segment.GetGeometryCache(FieldId(dependency.field_id));
        if (!cache)
            return reader;
        return std::make_shared<CachedGeometryReader>(std::move(reader),
                                                      std::move(cache));
    };
}

BindColumnReader
MakeGrowingReaderBinding(const segcore::InsertRecordGrowing& record,
                         std::shared_ptr<const Schema> schema,
                         int64_t active_count) {
    AssertInfo(
        schema && active_count >= 0 && active_count <= record.row_count(),
        "growing query prefix exceeds acknowledged rows");
    return [&record, schema = std::move(schema), active_count](
               const InputDependency& dependency)
               -> std::shared_ptr<const segcore::reader::ColumnReader> {
        if (!SupportsRawColumn(dependency))
            return nullptr;
        const FieldId field(dependency.field_id);
        // Resolve only published fields: Reopen fills new columns before
        // publishing schema, so binding an unpublished column could race fill.
        const auto& meta = (*schema)[field];
        AssertInfo(meta.get_data_type() == dependency.type,
                   "growing field type differs from compiled dependency");
        if (IsOrdinaryVectorDataType(dependency.type)) {
            auto validity =
                meta.is_nullable() ? record.get_valid_data(field) : nullptr;
            return std::make_shared<ValidityReader>(
                dependency.type,
                active_count,
                [validity = std::move(validity)](int64_t row, const auto&) {
                    return !validity || validity->is_valid(row);
                });
        }
        auto* column = record.get_data_base(field);
        if (column->is_mapping_storage() &&
            dependency.type != DataType::VECTOR_ARRAY)
            return nullptr;
        auto bind = [&]<typename T>()
            -> std::shared_ptr<const segcore::reader::ColumnReader> {
            const auto* typed =
                dynamic_cast<const segcore::ConcurrentVector<T>*>(column);
            AssertInfo(typed, "growing column physical type mismatch");
            return std::make_shared<segcore::reader::GrowingColumnReader<T>>(
                *typed, dependency.type, active_count);
        };
        switch (dependency.type) {
            case DataType::BOOL:
                return bind.template operator()<bool>();
            case DataType::INT8:
                return bind.template operator()<int8_t>();
            case DataType::INT16:
                return bind.template operator()<int16_t>();
            case DataType::INT32:
                return bind.template operator()<int32_t>();
            case DataType::INT64:
            case DataType::TIMESTAMPTZ:
                return bind.template operator()<int64_t>();
            case DataType::FLOAT:
                return bind.template operator()<float>();
            case DataType::DOUBLE:
                return bind.template operator()<double>();
            case DataType::GEOMETRY:
            case DataType::VARCHAR:
            case DataType::TEXT:
            case DataType::STRING:
                return bind.template operator()<std::string>();
            case DataType::JSON:
                return bind.template operator()<Json>();
            case DataType::ARRAY:
                if (meta.is_nested_array())
                    return bind.template operator()<ArrayValue>();
                return bind.template operator()<Array>();
            case DataType::VECTOR_ARRAY:
                return bind.template operator()<VectorArray>();
            default:
                ThrowInfo(UnexpectedError, "invalid growing scalar source");
        }
    };
}
}  // namespace milvus::exec::access
