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

#pragma once

#include "segcore/ConcurrentVector.h"
#include "segcore/reader/PointColumnReader.h"

namespace milvus::segcore::reader {

// Captures metadata and owners, without reading payloads. The source column
// need only live during construction. Each span belongs to the retained chunk
// generation, including mmap descriptors and variable-length payload storage.
// The caller supplies an acknowledged, immutable prefix; later appends are
// allowed, rewriting published values/validity is not.
template <typename T>
class GrowingReadHandle {
 public:
    GrowingReadHandle(const ConcurrentVector<T>& column, int64_t readable_rows)
        : rows_(readable_rows),
          chunk_size_(column.get_size_per_chunk()),
          validity_(column.validity_owner()),
          snapshot_(column.acquire_chunks()) {
        AssertInfo(rows_ >= 0 && chunk_size_ > 0,
                   "invalid growing read domain");
        mapped_ = column.is_mapping_storage();
        int64_t physical_rows = rows_;
        if (mapped_) {
            // Nullable vector arrays compact whole NULL rows. Capture the
            // logical prefix's mapping, never the live total valid count.
            physical_rows = 0;
            physical_offsets_.reserve(rows_);
            for (int64_t row = 0; row < rows_; ++row) {
                const auto physical = column.get_physical_offset(row);
                AssertInfo(physical >= -1 &&
                               (physical < 0 || physical == physical_rows),
                           "growing compacted mapping is not a fixed prefix");
                physical_offsets_.push_back(physical);
                physical_rows += physical >= 0;
            }
        }
        const auto count =
            physical_rows / chunk_size_ + (physical_rows % chunk_size_ != 0);
        AssertInfo(count <= snapshot_.count,
                   "growing snapshot shorter than acknowledged bound");
        for (int64_t chunk = 0; chunk < count; ++chunk) {
            auto span = column.get_span_base(snapshot_, chunk);
            const auto needed =
                std::min(chunk_size_, physical_rows - chunk * chunk_size_);
            AssertInfo(span.data() && span.row_count() >= needed,
                       "growing chunk shorter than acknowledged bound");
            const bool string_views =
                std::is_same_v<T, std::string> && column.is_mmap();
            AssertInfo(
                span.element_sizeof() ==
                    (string_views                     ? sizeof(std::string_view)
                     : std::is_same_v<T, Array>       ? sizeof(ArrayView)
                     : std::is_same_v<T, ArrayValue>  ? sizeof(ArrayValueView)
                     : std::is_same_v<T, VectorArray> ? sizeof(VectorArrayView)
                                                      : sizeof(T)),
                "growing physical scalar layout mismatch");
            spans_.push_back(span);
        }
        string_views_ = std::is_same_v<T, std::string> && column.is_mmap();
        array_views_ =
            (std::is_same_v<T, Array> || std::is_same_v<T, ArrayValue> ||
             std::is_same_v<T, VectorArray>);
    }

    int64_t
    size() const {
        return rows_;
    }

    bool
    IsValid(int64_t row) const {
        CheckRow(row);
        if (mapped_)
            return physical_offsets_[row] >= 0;
        if (!validity_)
            return true;
        // is_valid checks bounds; a truncated validity prefix is an internal
        // failure, not an invented NULL (bulk_is_valid_range pads with false).
        return validity_->is_valid(row);
    }

    int64_t
    ContiguousCount(int64_t row) const {
        CheckRow(row);
        return std::min(chunk_size_ - row % chunk_size_, rows_ - row);
    }

    const void*
    AddressAt(int64_t row) const {
        CheckRow(row);
        if (mapped_) {
            row = physical_offsets_[row];
            AssertInfo(row >= 0, "cannot address a compacted NULL row");
        }
        const auto& span = spans_[row / chunk_size_];
        return static_cast<const char*>(span.data()) +
               (row % chunk_size_) * span.element_sizeof();
    }
    bool
    UsesArrayViews() const {
        return array_views_;
    }

    bool
    UsesMappingStorage() const {
        return mapped_;
    }

    bool
    UsesStringViews() const {
        return string_views_;
    }

    ReadOnlyVectorView
    Borrow(int64_t begin,
           SelectionView rows,
           SelectionView valid,
           std::shared_ptr<const void> owner,
           DataType type) const {
        AssertInfo(valid.row_count() <= ContiguousCount(begin),
                   "growing borrowed range crosses a chunk");
        const auto& span = spans_[begin / chunk_size_];
        const auto in_chunk = begin % chunk_size_;
        if constexpr (std::is_same_v<T, std::string>) {
            if (string_views_)
                return ReadOnlyVectorView::BorrowImmutable<std::string_view>(
                    type,
                    static_cast<const std::string_view*>(span.data()) +
                        in_chunk,
                    std::move(owner),
                    rows,
                    valid);
        }
        return ReadOnlyVectorView::BorrowImmutable<T>(
            type,
            static_cast<const T*>(span.data()) + in_chunk,
            std::move(owner),
            rows,
            valid);
    }

 private:
    void
    CheckRow(int64_t row) const {
        AssertInfo(row >= 0 && row < rows_, "growing row outside fixed prefix");
    }

    int64_t rows_;
    int64_t chunk_size_;
    std::shared_ptr<const ThreadSafeValidData> validity_;
    ChunkSnapshot snapshot_;
    std::vector<SpanBase> spans_;
    std::vector<int64_t> physical_offsets_;
    bool mapped_ = false;
    bool string_views_ = false;
    bool array_views_ = false;
};

// MVCC/deletes remain the operator's responsibility. No live column or segment
// pointer survives construction; a prepared Reader can outlive its source.
template <typename T>
class GrowingColumnReader final : public PointColumnReader {
 public:
    GrowingColumnReader(std::shared_ptr<const ConcurrentVector<T>> column,
                        DataType type,
                        int64_t readable_rows)
        : GrowingColumnReader(CheckedColumn(column), type, readable_rows) {
    }

    GrowingColumnReader(const ConcurrentVector<T>& column,
                        DataType type,
                        int64_t readable_rows)
        : PointColumnReader(type, readable_rows),
          handle_(
              std::make_shared<GrowingReadHandle<T>>(column, readable_rows)) {
        AssertInfo(IsPointScalarType<T>(type),
                   "invalid growing scalar binding");
    }

 protected:
    ReadResult
    ReadPoints(SelectionView rows,
               ReadProjection projection,
               const Offset& offset,
               const ReadContext& ctx) const override {
        auto valid = OwnedSelection::None(rows.row_count());
        std::vector<std::pair<size_t, ReadOnlyVectorView>> parts;
        const bool values = projection == ReadProjection::ValuesAndValidity;
        if (!rows.IsNone()) {
            ctx.Check();
            if (!values) {
                rows.ForEachSelected([&](size_t row) {
                    ctx.Check();
                    if (handle_->IsValid(offset(row)))
                        valid.SetSelected(row);
                });
                ctx.Check();
                return {std::nullopt,
                        OwnedSelection::CopyOf(rows),
                        std::move(valid)};
            }
            if constexpr (std::is_same_v<T, ArrayValue>) {
                if (handle_->UsesArrayViews()) {
                    auto owned = std::make_shared<std::vector<ArrayValue>>(
                        rows.row_count());
                    rows.ForEachSelected([&](size_t row) {
                        ctx.Check();
                        const auto source = offset(row);
                        if (!handle_->IsValid(source))
                            return;
                        const auto& view = *static_cast<const ArrayValueView*>(
                            handle_->AddressAt(source));
                        (*owned)[row] =
                            ArrayValue(view.output_data(), view.type());
                        valid.SetSelected(row);
                    });
                    auto result =
                        ReadOnlyVectorView::BorrowImmutable<ArrayValue>(
                            type(), owned->data(), owned, rows, valid.view());
                    ctx.Check();
                    return {std::move(result),
                            OwnedSelection::CopyOf(rows),
                            std::move(valid)};
                }
            }
            if constexpr (std::is_same_v<T, Array> ||
                          std::is_same_v<T, VectorArray>) {
                if (handle_->UsesArrayViews()) {
                    VectorResultBuilder output(type(), rows.row_count());
                    rows.ForEachSelected([&](size_t row) {
                        ctx.Check();
                        const auto source = offset(row);
                        if (!handle_->IsValid(source)) {
                            output.SetNull(row);
                            return;
                        }
                        valid.SetSelected(row);
                        if constexpr (std::is_same_v<T, Array>) {
                            Array array;
                            static_cast<const ArrayView*>(
                                handle_->AddressAt(source))
                                ->output_data(array);
                            output.Set<Array>(row, std::move(array));
                        } else {
                            output.Set<VectorArray>(
                                row,
                                static_cast<const VectorArrayView*>(
                                    handle_->AddressAt(source))
                                    ->ToOwned());
                        }
                    });
                    ctx.Check();
                    return {std::move(output).Finish(rows),
                            OwnedSelection::CopyOf(rows),
                            std::move(valid)};
                }
            }
            // Reordered/duplicate Take offsets use one immutable pointer table,
            // rather than allocating a view and masks for each individual row.
            std::optional<int64_t> origin;
            bool consecutive = true;
            rows.ForEachSelected([&](size_t row) {
                const auto current = offset(row) - static_cast<int64_t>(row);
                if (!origin)
                    origin = current;
                else if (*origin != current)
                    consecutive = false;
            });
            if (!consecutive || handle_->UsesMappingStorage()) {
                struct Owner {
                    std::shared_ptr<const GrowingReadHandle<T>> handle;
                    std::vector<const void*> addresses;
                };
                auto owner = std::make_shared<Owner>(
                    Owner{handle_,
                          std::vector<const void*>(rows.row_count(), nullptr)});
                rows.ForEachSelected([&](size_t row) {
                    ctx.Check();
                    const auto source = offset(row);
                    if (handle_->IsValid(source)) {
                        valid.SetSelected(row);
                        owner->addresses[row] = handle_->AddressAt(source);
                    }
                });
                auto borrow = [&]<typename Value>() {
                    return ReadOnlyVectorView::BorrowIndirect<Value>(
                        type(),
                        owner->addresses.data(),
                        owner,
                        rows,
                        valid.view());
                };
                auto output = [&]() {
                    if constexpr (std::is_same_v<T, std::string>) {
                        if (handle_->UsesStringViews())
                            return borrow
                                .template operator()<std::string_view>();
                    }
                    return borrow.template operator()<T>();
                }();
                ctx.Check();
                return {std::move(output),
                        OwnedSelection::CopyOf(rows),
                        std::move(valid)};
            }
            // All selected offsets share the same affine mapping. Unselected
            // offsets need not follow it: borrow across their gaps without
            // inspecting their offsets or values, stopping at chunk boundaries.
            size_t row = 0;
            while (row < rows.row_count()) {
                if (!rows.IsSelected(row)) {
                    ++row;
                    continue;
                }
                const auto begin = row;
                const auto source = offset(row);
                row += std::min<size_t>(handle_->ContiguousCount(source),
                                        rows.row_count() - begin);
                auto local_valid = OwnedSelection::None(row - begin);
                const auto selected = rows.Slice(begin, row - begin);
                selected.ForEachSelected([&](size_t i) {
                    ctx.Check();
                    if (handle_->IsValid(source + i)) {
                        valid.SetSelected(begin + i);
                        local_valid.SetSelected(i);
                    }
                });
                parts.emplace_back(
                    begin,
                    handle_->Borrow(
                        source, selected, local_valid.view(), handle_, type()));
            }
            ctx.Check();
        }
        std::optional<ReadOnlyVectorView> output;
        if (values)
            output = rows.IsNone()
                         ? ReadOnlyVectorView::Nulls(type(), rows)
                         : ReadOnlyVectorView::Segments(type(), rows, parts);
        return {
            std::move(output), OwnedSelection::CopyOf(rows), std::move(valid)};
    }

 private:
    static const ConcurrentVector<T>&
    CheckedColumn(const std::shared_ptr<const ConcurrentVector<T>>& column) {
        AssertInfo(column, "missing growing column");
        return *column;
    }

    std::shared_ptr<const GrowingReadHandle<T>> handle_;
};
}  // namespace milvus::segcore::reader
