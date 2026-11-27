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

#include "segcore/reader/ColumnReader.h"

namespace milvus::segcore::reader {

// Range/Take protocol for sources with stable point access, not a substitute
// for Column's bulk Scan/Take fast path. No expression callbacks or cursor.
class PointColumnReader : public ColumnReader {
 public:
    PointColumnReader(DataType type, int64_t rows) : type_(type), rows_(rows) {
        AssertInfo(rows >= 0, "negative point reader domain");
    }
    DataType
    type() const final {
        return type_;
    }
    int64_t
    row_count() const final {
        return rows_;
    }

    ReadResult
    ReadRange(const RangeRequest& request, const ReadContext& ctx) const final {
        AssertInfo(
            request.begin <= rows_ &&
                request.count <= static_cast<uint64_t>(rows_ - request.begin),
            "point range outside snapshot");
        return ReadPoints(
            request.eval_rows.view(),
            request.projection,
            [&](size_t row) { return request.begin + row; },
            ctx);
    }
    ReadResult
    Take(const TakeRequest& request, const ReadContext& ctx) const final {
        for (auto offset : request.offsets)
            AssertInfo(offset < rows_, "point offset outside snapshot");
        return ReadPoints(
            request.eval_rows.view(),
            request.projection,
            [&](size_t row) { return request.offsets[row]; },
            ctx);
    }

 protected:
    using Offset = std::function<int64_t(size_t)>;
    virtual ReadResult
    ReadPoints(SelectionView,
               ReadProjection,
               const Offset&,
               const ReadContext&) const = 0;

    // Both callbacks are storage primitives. Validity-only requests NEVER
    // invoke value_at; NULL payloads are never accessed. Publish only after
    // all rows and the final cancellation/budget check succeed.
    template <typename T, typename ValidAt, typename ValueAt>
    ReadResult
    ReadScalars(SelectionView rows,
                ReadProjection projection,
                ValidAt valid_at,
                ValueAt value_at,
                const ReadContext& ctx) const {
        auto valid = OwnedSelection::None(rows.row_count());
        std::unique_ptr<VectorResultBuilder> values;
        const bool want_values =
            projection == ReadProjection::ValuesAndValidity;
        if (!rows.IsNone()) {
            ctx.Check();
            if (want_values)
                values = std::make_unique<VectorResultBuilder>(
                    type_, rows.row_count());
            rows.ForEachSelected([&](size_t row) {
                ctx.Check();
                if (!valid_at(row)) {
                    if (values)
                        values->SetNull(row);
                    return;
                }
                valid.SetSelected(row);
                if (values) {
                    values->Set<T>(row, value_at(row));
                }
            });
            ctx.Check();
        }
        std::optional<ReadOnlyVectorView> output;
        if (want_values)
            output = values ? std::move(*values).Finish(rows)
                            : ReadOnlyVectorView::Nulls(type_, rows);
        return {
            std::move(output), OwnedSelection::CopyOf(rows), std::move(valid)};
    }

 private:
    DataType type_;
    int64_t rows_;
};

// A validity source has no payload dependency, including ordinary vectors.
// The callback owns its fixed-generation resource and must not evaluate Exprs.
class ValidityColumnReader final : public PointColumnReader {
 public:
    using IsValid = std::function<bool(int64_t, const ReadContext&)>;
    // Resolve/pin once per nonempty request; the returned accessor owns its pin
    // until the request finishes. Empty selections never resolve a source.
    using ResolveValidity = std::function<IsValid(const ReadContext&)>;
    ValidityColumnReader(DataType type, int64_t rows, IsValid valid)
        : PointColumnReader(type, rows), valid_(std::move(valid)) {
        AssertInfo(valid_, "missing field validity source");
    }
    ValidityColumnReader(DataType type, int64_t rows, ResolveValidity resolve)
        : PointColumnReader(type, rows), resolve_(std::move(resolve)) {
        AssertInfo(resolve_, "missing field validity resolver");
    }

 protected:
    ReadResult
    ReadPoints(SelectionView rows,
               ReadProjection projection,
               const Offset& offset,
               const ReadContext& ctx) const override {
        AssertInfo(projection == ReadProjection::ValidityOnly,
                   "validity reader cannot produce field values");
        auto valid = OwnedSelection::None(rows.row_count());
        if (!rows.IsNone()) {
            ctx.Check();
            auto valid_at = resolve_ ? resolve_(ctx) : valid_;
            AssertInfo(valid_at, "validity resolver returned no accessor");
            rows.ForEachSelected([&](size_t row) {
                ctx.Check();
                if (valid_at(offset(row), ctx))
                    valid.SetSelected(row);
            });
            ctx.Check();
        }
        return {std::nullopt, OwnedSelection::CopyOf(rows), std::move(valid)};
    }

 private:
    IsValid valid_;
    ResolveValidity resolve_;
};

template <typename T>
bool
IsPointScalarType(DataType type) {
    if constexpr (std::is_same_v<T, bool>)
        return type == DataType::BOOL;
    if constexpr (std::is_same_v<T, int8_t>)
        return type == DataType::INT8;
    if constexpr (std::is_same_v<T, int16_t>)
        return type == DataType::INT16;
    if constexpr (std::is_same_v<T, int32_t>)
        return type == DataType::INT32;
    if constexpr (std::is_same_v<T, int64_t>)
        return type == DataType::INT64 || type == DataType::TIMESTAMPTZ;
    if constexpr (std::is_same_v<T, float>)
        return type == DataType::FLOAT;
    if constexpr (std::is_same_v<T, double>)
        return type == DataType::DOUBLE;
    if constexpr (std::is_same_v<T, std::string>)
        return type == DataType::STRING || type == DataType::VARCHAR ||
               type == DataType::TEXT || type == DataType::GEOMETRY;
    if constexpr (std::is_same_v<T, Array> || std::is_same_v<T, ArrayValue>)
        return type == DataType::ARRAY;
    if constexpr (std::is_same_v<T, VectorArray>)
        return type == DataType::VECTOR_ARRAY;
    if constexpr (std::is_same_v<T, Json>)
        return type == DataType::JSON;
    return false;
}
}  // namespace milvus::segcore::reader
