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

#include "segcore/reader/ColumnReaderAdapter.h"

#include <cstring>
#include "mmap/ChunkedColumnInterface.h"

#include <mutex>

namespace milvus::segcore::reader {
namespace {

TargetType
Target(DataType type, bool recursive_array = false) {
    switch (type) {
        case DataType::BOOL:
            return TargetType::Bool;
        case DataType::INT8:
            return TargetType::Int8;
        case DataType::INT16:
            return TargetType::Int16;
        case DataType::INT32:
            return TargetType::Int32;
        case DataType::INT64:
        case DataType::TIMESTAMPTZ:
            return TargetType::Int64;
        case DataType::FLOAT:
            return TargetType::Float;
        case DataType::DOUBLE:
            return TargetType::Double;
        case DataType::GEOMETRY:
        case DataType::STRING:
        case DataType::VARCHAR:
        case DataType::TEXT:
            return TargetType::StringView;
        case DataType::JSON:
            return TargetType::Json;
        case DataType::ARRAY:
            return recursive_array ? TargetType::ArrayValueView
                                   : TargetType::ArrayView;
        case DataType::VECTOR_ARRAY:
            return TargetType::VectorArrayView;
        default:
            ThrowInfo(
                Unsupported, "column reader type {} is not migrated", type);
    }
}

// Publish only after the finite request succeeds. Scan keeps pinned fragments;
// lazy Take materializes once before moving to a different Cell.
class ResultBuilder {
 public:
    ResultBuilder(DataType type,
                  SelectionView rows,
                  bool recursive_array = false)
        : type_(type),
          recursive_array_(recursive_array),
          rows_(rows),
          valid_(rows.row_count(), false) {
    }

    void
    SetValidity(size_t row, bool valid) {
        if (valid)
            valid_.set(row);
        else if (values_)
            values_->SetNull(row);
    }

    OwnedSelection
    AddValidity(size_t begin, size_t count, ValidityView validity = {}) {
        auto valid =
            OwnedSelection::FromValidity(validity, rows_.Slice(begin, count));
        valid.view().CopyTo(valid_, begin);
        return valid;
    }

    void
    Materialize() {
        if (recursive_array_) {
            recursive_values_ =
                std::make_shared<std::vector<ArrayValue>>(rows_.row_count());
            return;
        }
        values_ =
            std::make_unique<VectorResultBuilder>(type_, rows_.row_count());
    }

    template <typename T>
    void
    Set(size_t row, T value) {
        if constexpr (std::is_same_v<T, ArrayValueView>) {
            AssertInfo(recursive_values_,
                       "recursive Take output is not initialized");
            (*recursive_values_)[row] =
                ArrayValue(value.output_data(), value.type());
        } else if constexpr (std::is_same_v<T, std::string_view>)
            values_->Set<std::string>(row, std::string(value));
        else if constexpr (std::is_same_v<T, VectorArrayView>) {
            values_->Set<VectorArray>(row, value.ToOwned());
        } else if constexpr (std::is_same_v<T, ArrayView>) {
            Array owned;
            value.output_data(owned);
            values_->Set<Array>(row, std::move(owned));
        } else if constexpr (std::is_same_v<T, Json>)
            values_->Set<Json>(row, std::move(value));
        else
            values_->Set<T>(row, value);
    }

    void
    AddBatch(size_t begin,
             const ScanBatch& batch,
             std::shared_ptr<const ChunkedColumnInterface> column) {
        const auto rows = rows_.Slice(begin, batch.size);
        auto valid = AddValidity(begin, batch.size, batch.validity);
        AssertInfo(batch.owner, "Scan view requires a result-owned pin");
        struct Owner {
            std::shared_ptr<const ChunkedColumnInterface> generation;
            std::shared_ptr<void> pin;
        };
        auto owner =
            std::make_shared<Owner>(Owner{std::move(column), batch.owner});
        auto borrow = [&]<typename T>() {
            AssertInfo(batch.values.byte_width == sizeof(T),
                       "Scan scalar width mismatch");
            return ReadOnlyVectorView::BorrowImmutable<T>(
                type_,
                batch.values.data_as<T>(),
                owner,
                rows,
                valid.view(),
                (batch.integer_null_payload_readable ||
                 (std::is_floating_point_v<T> &&
                  batch.floating_null_payload_readable) ||
                 (std::is_same_v<T, std::string_view> &&
                  (type_ == DataType::STRING || type_ == DataType::VARCHAR ||
                   type_ == DataType::TEXT) &&
                  batch.string_null_payload_readable))
                    ? PrimitivePayloadAccess::AllDefinedRows
                    : PrimitivePayloadAccess::ValidRowsOnly);
        };
        auto view = [&]() {
            switch (Target(type_, recursive_array_)) {
                case TargetType::ArrayValueView: {
                    AssertInfo(
                        batch.values.byte_width == sizeof(ArrayValueView),
                        "Scan recursive array width mismatch");
                    auto owned =
                        std::make_shared<std::vector<ArrayValue>>(batch.size);
                    rows.ForEachSelected([&](size_t row) {
                        if (valid.view().IsSelected(row)) {
                            const auto& value =
                                batch.values.data_as<ArrayValueView>()[row];
                            (*owned)[row] =
                                ArrayValue(value.output_data(), value.type());
                        }
                    });
                    return ReadOnlyVectorView::BorrowImmutable<ArrayValue>(
                        DataType::ARRAY,
                        owned->data(),
                        owned,
                        rows,
                        valid.view());
                }
                case TargetType::Bool:
                    return borrow.template operator()<bool>();
                case TargetType::Int8:
                    return borrow.template operator()<int8_t>();
                case TargetType::Int16:
                    return borrow.template operator()<int16_t>();
                case TargetType::Int32:
                    return borrow.template operator()<int32_t>();
                case TargetType::Int64:
                    return borrow.template operator()<int64_t>();
                case TargetType::Float:
                    return borrow.template operator()<float>();
                case TargetType::Double:
                    return borrow.template operator()<double>();
                case TargetType::StringView:
                    return borrow.template operator()<std::string_view>();
                case TargetType::Json:
                    return borrow.template operator()<Json>();
                case TargetType::VectorArrayView: {
                    AssertInfo(
                        batch.values.byte_width == sizeof(VectorArrayView),
                        "Scan vector array view width mismatch");
                    VectorResultBuilder output(DataType::VECTOR_ARRAY,
                                               batch.size);
                    rows.ForEachSelected([&](size_t row) {
                        if (!valid.view().IsSelected(row))
                            output.SetNull(row);
                        else
                            output.Set<VectorArray>(
                                row,
                                batch.values.data_as<VectorArrayView>()[row]
                                    .ToOwned());
                    });
                    return std::move(output).Finish(rows);
                }
                case TargetType::ArrayView:
                    return borrow.template operator()<ArrayView>();
                default:
                    ThrowInfo(UnexpectedError, "invalid scalar reader target");
            }
        }();
        parts_.emplace_back(begin, std::move(view));
    }

    void
    TakeValues(const TakeResult& take, const ReadContext& context) {
        auto numeric = [&]<typename T>() {
            int64_t index = 0;
            auto probe = [&](size_t row) -> std::optional<T> {
                context.Check();
                const auto item = take.Get<T>(index++);
                AssertInfo(!item.data_skipped,
                           "predicate skip is not a field value");
                valid_.set(row, item.is_valid);
                if (!item.is_valid)
                    return std::nullopt;
                AssertInfo(item.value.has_value(),
                           "valid take item has no value");
                return *item.value;
            };
            if (rows_.IsAll())
                values_->EvaluateRange<T>(0, rows_.row_count(), probe);
            else
                values_->EvaluateSelected<T>(0, rows_, probe);
        };
        switch (Target(type_, recursive_array_)) {
            case TargetType::Bool:
                return numeric.template operator()<bool>();
            case TargetType::Int8:
                return numeric.template operator()<int8_t>();
            case TargetType::Int16:
                return numeric.template operator()<int16_t>();
            case TargetType::Int32:
                return numeric.template operator()<int32_t>();
            case TargetType::Int64:
                return numeric.template operator()<int64_t>();
            case TargetType::Float:
                return numeric.template operator()<float>();
            case TargetType::Double:
                return numeric.template operator()<double>();
            default:
                break;
        }
        int64_t index = 0;
        rows_.ForEachSelected([&](size_t row) {
            context.Check();
            TakeValue(row, take, index++);
        });
    }

    template <typename T>
    void
    TakeValue(size_t row, const TakeResult& take, int64_t index) {
        auto item = take.Get<T>(index);
        AssertInfo(!item.data_skipped, "predicate skip is not a field value");
        SetValidity(row, item.is_valid);
        if (item.is_valid) {
            AssertInfo(item.value.has_value(), "valid take item has no value");
            Set(row, *item.value);
        }
    }

    void
    TakeValue(size_t row, const TakeResult& take, int64_t index) {
        switch (Target(type_, recursive_array_)) {
            case TargetType::ArrayValueView:
                TakeValue<ArrayValueView>(row, take, index);
                break;
            case TargetType::Bool:
                TakeValue<bool>(row, take, index);
                break;
            case TargetType::Int8:
                TakeValue<int8_t>(row, take, index);
                break;
            case TargetType::Int16:
                TakeValue<int16_t>(row, take, index);
                break;
            case TargetType::Int32:
                TakeValue<int32_t>(row, take, index);
                break;
            case TargetType::Int64:
                TakeValue<int64_t>(row, take, index);
                break;
            case TargetType::Float:
                TakeValue<float>(row, take, index);
                break;
            case TargetType::Double:
                TakeValue<double>(row, take, index);
                break;
            case TargetType::StringView:
                TakeValue<std::string_view>(row, take, index);
                break;
            case TargetType::VectorArrayView:
                TakeValue<VectorArrayView>(row, take, index);
                break;
            case TargetType::ArrayView:
                TakeValue<ArrayView>(row, take, index);
                break;
            case TargetType::Json:
                TakeValue<Json>(row, take, index);
                break;
            default:
                ThrowInfo(UnexpectedError, "invalid scalar reader target");
        }
    }

    ReadResult
    Finish(ReadProjection projection) {
        std::optional<ReadOnlyVectorView> values;
        if (projection == ReadProjection::ValuesAndValidity &&
            recursive_values_) {
            values = ReadOnlyVectorView::BorrowImmutable<ArrayValue>(
                type_,
                recursive_values_->data(),
                recursive_values_,
                rows_,
                SelectionView::FromBitmap(valid_));
            return {std::move(values),
                    OwnedSelection::CopyOf(rows_),
                    OwnedSelection::OwnBitmap(std::move(valid_))};
        }
        if (projection == ReadProjection::ValuesAndValidity)
            values = values_ ? std::move(*values_).Finish(rows_)
                             : (rows_.IsNone()
                                    ? ReadOnlyVectorView::Nulls(type_, rows_)
                                    : ReadOnlyVectorView::Segments(
                                          type_, rows_, parts_));
        return {std::move(values),
                OwnedSelection::CopyOf(rows_),
                OwnedSelection::OwnBitmap(std::move(valid_))};
    }

 private:
    DataType type_;
    bool recursive_array_;
    std::shared_ptr<std::vector<ArrayValue>> recursive_values_;
    SelectionView rows_;
    CustomBitset valid_;
    std::unique_ptr<VectorResultBuilder> values_;
    std::vector<std::pair<size_t, ReadOnlyVectorView>> parts_;
};

}  // namespace

// Reuse physical scan machinery, never its position as logical execution state.
// Seek serves forward requests; backward requests recreate the forward-only
// physical cursor. No request inherits logical progress from a prior batch.
struct ColumnReaderAdapter::RangeSession {
    std::mutex mutex;
    std::shared_ptr<OpContext> context;
    ScanResult cursor;
};

ColumnReaderAdapter::~ColumnReaderAdapter() = default;

ColumnReaderAdapter::ColumnReaderAdapter(
    std::shared_ptr<const ChunkedColumnInterface> column,
    DataType type,
    int64_t readable_rows,
    bool recursive_array)
    : column_(std::move(column)),
      type_(type),
      readable_rows_(readable_rows),
      recursive_array_(recursive_array),
      range_session_(std::make_unique<RangeSession>()) {
    AssertInfo(column_ != nullptr && readable_rows >= 0 &&
                   static_cast<uint64_t>(readable_rows) <= column_->NumRows(),
               "invalid column reader snapshot");
    AssertInfo(!recursive_array_ || type_ == DataType::ARRAY,
               "recursive layout requires ARRAY type");
    Target(type_, recursive_array_);
}

ReadResult
ColumnReaderAdapter::ReadRange(const RangeRequest& request,
                               const ReadContext& context) const {
    AssertInfo(request.begin <= readable_rows_ &&
                   request.count <=
                       static_cast<size_t>(readable_rows_ - request.begin),
               "reader range outside snapshot");
    auto rows = request.eval_rows.view();
    if (!rows.IsNone())
        context.Check();
    ResultBuilder result(type_, rows, recursive_array_);
    if (rows.IsNone())
        return result.Finish(request.projection);
    context.Check();
    const bool validity_only =
        request.projection == ReadProjection::ValidityOnly;
    if (validity_only && !column_->IsNullable()) {
        result.AddValidity(0, rows.row_count());
        context.Check();
        return result.Finish(request.projection);
    }
    std::unique_lock lock(range_session_->mutex);
    struct DiscardOnFailure {
        RangeSession& session;
        bool success = false;
        ~DiscardOnFailure() {
            if (!success)
                session.cursor.reset();
        }
    } discard{*range_session_};
    // No speculative prefetch, implicit progress or cross-context reuse.
    if (!range_session_->cursor ||
        range_session_->context != context.op_context ||
        request.begin < range_session_->cursor->Position()) {
        range_session_->cursor.reset();
        auto cursor =
            column_->Scan(context.op_context.get(),
                          ScanOptions::ForData(request.begin,
                                               Target(type_, recursive_array_),
                                               ScanPinPolicy::ResultOwned));
        if (!cursor)
            ThrowInfo(Unsupported, "column does not support scalar Scan");
        range_session_->context = context.op_context;
        range_session_->cursor = std::move(cursor);
    }
    auto* cursor = range_session_->cursor.get();

    size_t next = 0;
    size_t end = request.count;
    while (end && !rows.IsSelected(end - 1)) --end;
    while (next < end) {
        while (next < end && !rows.IsSelected(next)) ++next;
        if (next == end)
            break;
        context.Check();
        const auto position = request.begin + next;
        cursor->Seek(position);
        AssertInfo(cursor->Position() == position,
                   "reader cursor position mismatch");
        ScanBatch batch;
        const bool present =
            cursor->Next(end - next,
                         validity_only ? ScanReadMode::ValidityOnly
                                       : ScanReadMode::DataAndValidity,
                         &batch);
        AssertInfo(present && batch.size > 0 &&
                       batch.size <= static_cast<int64_t>(end - next) &&
                       batch.row_id_start == position &&
                       cursor->Position() == position + batch.size,
                   "reader Scan ended early or returned an invalid range");
        AssertInfo(!batch.data_skipped, "predicate skip is not a field value");
        // One borrowed physical batch per Cell, even for sparse selections.
        // Raw values/descriptors can cover gaps, but only requested rows become
        // defined. Cells containing no selected rows are skipped by Seek.
        if (validity_only) {
            result.AddValidity(next, batch.size, batch.validity);
        } else {
            result.AddBatch(next, batch, column_);
        }
        next += batch.size;
    }
    context.Check();
    auto output = result.Finish(request.projection);
    discard.success = true;
    return output;
}

ReadResult
ColumnReaderAdapter::Take(const TakeRequest& request,
                          const ReadContext& context) const {
    for (auto offset : request.offsets)
        AssertInfo(offset < readable_rows_, "reader offset outside snapshot");
    auto rows = request.eval_rows.view();
    if (!rows.IsNone())
        context.Check();
    ResultBuilder result(type_, rows, recursive_array_);
    if (rows.IsNone())
        return result.Finish(request.projection);
    context.Check();
    const bool validity_only =
        request.projection == ReadProjection::ValidityOnly;
    if (validity_only && !column_->IsNullable()) {
        result.AddValidity(0, rows.row_count());
        context.Check();
        return result.Finish(request.projection);
    }
    std::vector<int32_t> selected_offsets;
    std::span<const int32_t> offsets(request.offsets.data(),
                                     request.offsets.size());
    if (!rows.IsAll()) {
        selected_offsets.reserve(rows.selected_count());
        rows.ForEachSelected([&](size_t row) {
            selected_offsets.push_back(request.offsets[row]);
        });
        offsets = selected_offsets;
    }
    // Offsets and OpContext outlive every lazy Take access. Copy each payload
    // immediately; GetOwn() would unnecessarily materialize the whole result.
    auto take = column_->Take(
        context.op_context.get(),
        TakeOptions{OffsetView::From(offsets.data(), offsets.size()),
                    Target(type_, recursive_array_),
                    {}});
    if (!take)
        ThrowInfo(Unsupported, "column does not support scalar Take");
    AssertInfo(take->Size() == static_cast<int64_t>(offsets.size()) &&
                   take->GetDataType() == type_ &&
                   take->GetTargetType() == Target(type_, recursive_array_),
               "reader Take shape/type mismatch");
    if (!validity_only && take->IsOwned()) {
        // The backend already generated/materialized these values. Adopt its
        // owner instead of copying them through another temporary vector.
        const auto owned = take->GetOwn();
        AssertInfo(owned.size == static_cast<int64_t>(offsets.size()),
                   "owned Take shape mismatch");
        auto compact_rows = SelectionView::All(offsets.size());
        ResultBuilder compact(type_, compact_rows, recursive_array_);
        ScanBatch batch;
        batch.values = owned.values;
        batch.validity = owned.validity;
        batch.owner = owned.owner;
        batch.size = owned.size;
        for (int64_t i = 0; i < owned.size; ++i) {
            context.Check();
            AssertInfo(!owned.data_skipped || !owned.data_skipped[i],
                       "predicate skip is not a field value");
        }
        compact.AddBatch(0, batch, column_);
        auto loaded = compact.Finish(request.projection);
        if (rows.IsAll()) {
            context.Check();
            return loaded;
        }
        std::vector<vector_size_t> mapping(rows.row_count(), 0);
        auto valid = OwnedSelection::None(rows.row_count());
        size_t index = 0;
        rows.ForEachSelected([&](size_t row) {
            mapping[row] = index;
            if (loaded.valid.view().IsSelected(index))
                valid.SetSelected(row);
            ++index;
        });
        auto output =
            loaded.values
                ->Dictionary(std::make_shared<RowMapping>(std::move(mapping)))
                .Restrict(rows);
        context.Check();
        return {
            std::move(output), OwnedSelection::CopyOf(rows), std::move(valid)};
    }
    if (!validity_only) {
        result.Materialize();
        result.TakeValues(*take, context);
    } else {
        int64_t index = 0;
        rows.ForEachSelected([&](size_t row) {
            context.Check();
            result.SetValidity(row, take->IsValid(index));
            ++index;
        });
    }
    context.Check();
    return result.Finish(request.projection);
}

}  // namespace milvus::segcore::reader
