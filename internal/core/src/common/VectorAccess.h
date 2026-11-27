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

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>

#include "common/Selection.h"
#include "common/ArrayValue.h"
#include "common/Vector.h"

namespace milvus {

enum class ValueEncoding { Flat, Bitmap, Constant, Dictionary, Segmented };

// An ordered row mapping, unlike a selection set: duplicates and reordering
// are meaningful. Ownership transfer requires no retained writable aliases.
class RowMapping {
 public:
    explicit RowMapping(std::vector<vector_size_t> indices)
        : indices_(std::move(indices)) {
    }
    const std::vector<vector_size_t>&
    indices() const {
        return indices_;
    }

 private:
    std::vector<vector_size_t> indices_;
};
using RowMappingPtr = std::shared_ptr<const RowMapping>;

// CopyOf is deliberately a snapshot boundary, not a borrowed wrapper around a
// legacy VectorPtr. Legacy const methods leak mutable pointers, and exclusive
// ownership of a VectorPtr does not prove exclusive ownership of its buffers.
// Only defined_rows are read/copied; NULL payloads are never accessed. The view
// retains immutable values, validity and its defined-row mask. That mask describes
// this result, NOT BatchInput's mutable loading/coverage state.
//
// Supported scalar payloads: BOOL, signed integers, TIMESTAMPTZ, FLOAT, DOUBLE,
// STRING/VARCHAR/TEXT/GEOMETRY string storage, JSON, and owned scalar ARRAYs.
// Legacy flat/bitmap and
// constant vectors are detached; published views retain their immutable owners.
// An explicit producer guarantee, independent of SQL validity. Only defined
// slots are covered; the producer must supply initialized, readable objects.
enum class PrimitivePayloadAccess { ValidRowsOnly, AllDefinedRows };

class ReadOnlyVectorView {
 public:
    static ReadOnlyVectorView
    CopyOf(const VectorPtr& source, SelectionView defined_rows);

    static ReadOnlyVectorView
    CopyOf(const VectorPtr& source);

    // Wrap already immutable storage without copying payloads. The owner must
    // keep the typed elements (including string payloads) alive AND immutable
    // for all derived views. A shared_ptr<const T> alone does not prove this.
    // Coverage and validity are copied; no unselected/NULL payload is read.
    template <typename T>
    static ReadOnlyVectorView
    BorrowImmutable(
        DataType type,
        const T* values,
        std::shared_ptr<const void> owner,
        SelectionView defined,
        SelectionView valid,
        PrimitivePayloadAccess access = PrimitivePayloadAccess::ValidRowsOnly) {
        auto result = BorrowBuffer(type,
                                   values,
                                   std::move(owner),
                                   defined,
                                   valid,
                                   sizeof(T),
                                   std::is_same_v<T, std::string_view>,
                                   false,
                                   std::is_same_v<T, ArrayValue>,
                                   access,
                                   std::is_same_v<T, ArrayView>);
        if constexpr (std::is_same_v<T, std::string_view>)
            result.template CheckType<std::string>();
        else
            result.template CheckType<T>();
        return result;
    }

    // Gather from pinned, discontiguous typed objects. Only the pointer table
    // is new; neither primitive values nor variable-length bodies are copied.
    // The owner keeps both the immutable table and every pointed-to value alive.
    template <typename T>
    static ReadOnlyVectorView
    BorrowIndirect(DataType type,
                   const void* const* values,
                   std::shared_ptr<const void> owner,
                   SelectionView defined,
                   SelectionView valid) {
        auto result = BorrowBuffer(type,
                                   values,
                                   std::move(owner),
                                   defined,
                                   valid,
                                   sizeof(T),
                                   std::is_same_v<T, std::string_view>,
                                   true,
                                   std::is_same_v<T, ArrayValue>,
                                   PrimitivePayloadAccess::ValidRowsOnly,
                                   std::is_same_v<T, ArrayView>);
        if constexpr (std::is_same_v<T, std::string_view>)
            result.template CheckType<std::string>();
        else
            result.template CheckType<T>();
        return result;
    }

    // Ordered non-overlapping fragments in the output's coordinate domain.
    // Gaps must be undefined. Payloads and their pins are shared, not copied.
    static ReadOnlyVectorView
    Segments(DataType type,
             SelectionView rows,
             const std::vector<std::pair<size_t, ReadOnlyVectorView>>& parts);

    // These operations share immutable payload storage. They never return
    // pointers into temporary scratch buffers. Dictionary mappings compose.
    ReadOnlyVectorView
    Dictionary(RowMappingPtr mapping) const;

    ReadOnlyVectorView
    Slice(size_t begin, size_t count) const;

    ReadOnlyVectorView
    Restrict(SelectionView rows) const;

    // Construct defined NULLs without touching any value payload, and merge
    // immutable patches without invalidating previously returned views.
    static ReadOnlyVectorView
    Nulls(DataType type, SelectionView rows);
    ReadOnlyVectorView
    Merge(const ReadOnlyVectorView& patch, SelectionView rows) const;

    // Publish this immutable column through the existing VectorPtr/RowVector
    // operator interface. Consumers decode it using CopyOf, which shares it.
    VectorPtr
    AsVector() const;

    // Detached arguments for legacy SimpleVector functions. Requires complete
    // coverage; a constant remains a separately owned ConstantVector. Other
    // encodings materialize into writable columns. No input alias is exposed.
    VectorPtr
    CopyForLegacyFunction() const;

    // The view must outlive the string_view (including its payload owner).
    std::string_view
    StringViewAt(size_t row) const&;
    std::string_view
    StringViewAt(size_t row) const&& = delete;

    // Resolve a mapped position once for validity and string payload.
    std::optional<std::string_view>
    TryStringViewAt(size_t row) const&;
    std::optional<std::string_view>
    TryStringViewAt(size_t row) const&& = delete;

    // Borrow padded JSON bytes; retain this view until the returned Json and
    // any parser results have been consumed. No owning Json copy is made.
    Json
    JsonViewAt(size_t row) const&;
    Json
    JsonViewAt(size_t row) const&& = delete;

    size_t
    size() const;

    DataType
    logical_type() const;

    ValueEncoding
    encoding() const;

    SelectionView
    defined_rows() const&;

    SelectionView
    defined_rows() const&& = delete;

    // Reject undefined positions; they are not NULL.
    bool
    IsValid(size_t row) const;

    // Detached batch-coordinate validity, always a subset of defined_rows().
    // Flat/bitmap/borrowed storage and contiguous slices copy mask words;
    // mapped/fragmented encodings retain the encoding-aware fallback.
    // Never reads value payloads or exposes mutable source storage.
    OwnedSelection
    CopyValidity() const;

    // Borrow a contiguous logical validity mask when representable without
    // allocation. Retain this view for the lifetime of the borrowed selection.
    // The result is clipped to defined rows; mapped/fragmented cases may miss.
    std::optional<SelectionView>
    TryBorrowValidity() const&;
    std::optional<SelectionView>
    TryBorrowValidity() const&& = delete;

    // Copy TRUE bits directly to an output range. FALSE/NULL/undefined are 0.
    void
    CopyTrueTo(SelectionView rows,
               TargetBitmap& output,
               size_t begin = 0) const;

    struct BooleanMasks {
        OwnedSelection truth;
        OwnedSelection valid;
    };
    // Detached TRUE and validity masks, clipped to requested defined rows.
    // Bitmap/constant slices use bulk operations; flat BOOL spans pack valid
    // runs directly. Mapped and indirect encodings can gather.
    BooleanMasks
    CopyBoolMasks(SelectionView rows) const;

    // Publish owned BOOL masks without copying their buffers. Enforces
    // truth <= valid <= defined, retaining FALSE/NULL/undefined distinctions.
    static ReadOnlyVectorView
    FromBooleanMasks(OwnedSelection truth,
                     OwnedSelection valid,
                     SelectionView defined);

    // Share immutable, canonical TRUE/validity masks. The owner must keep the
    // masks alive and immutable, including their bitmap objects, for every
    // derived view. Unlike CopyOf, this is an explicit producer guarantee.
    static ReadOnlyVectorView
    BorrowBooleanMasks(SelectionView truth,
                       SelectionView valid,
                       std::shared_ptr<const void> owner);

    // Borrowed primitive or string fragments. The decoded owner keeps every data/mask
    // pointer alive. Masks are fragment-local. Only a total kernel may read
    // NULL slots with an explicit producer guarantee. CanEvaluateAll also
    // preserves prior physical coverage through Restrict; it never changes
    // logical coverage. Other kernels must intersect selection and validity.
    struct DecodedSpan {
        size_t begin;
        size_t count;
        const void* data;
        bool constant;
        SelectionView valid;
        SelectionView defined;
        PrimitivePayloadAccess payload_access;
        bool string_views = false;
        // Physical coverage can outlive a logical Restrict. It never grants
        // validity or defined output outside the caller's selection.
        bool full_payload_defined = false;

        bool
        CanReadAll(size_t offset, size_t size) const {
            return data && defined.Slice(offset, size).IsAll() &&
                   (payload_access == PrimitivePayloadAccess::AllDefinedRows ||
                    valid.Slice(offset, size).IsAll());
        }

        // For explicitly total kernels only. No storage read is performed to
        // obtain this capability; NULL payloads still need producer approval.
        bool
        CanEvaluateAll(size_t offset, size_t size) const {
            return data &&
                   (full_payload_defined ||
                    defined.Slice(offset, size).IsAll()) &&
                   (payload_access == PrimitivePayloadAccess::AllDefinedRows ||
                    valid.Slice(offset, size).IsAll());
        }
    };
    struct DecodedView {
        std::shared_ptr<const void> owner;
        folly::small_vector<DecodedSpan, 1> spans;
    };
    using PrimitiveSpan = DecodedSpan;
    using PrimitiveView = DecodedView;

    // Numeric/BOOL arrays and constants only. Arbitrary mappings, indirect or
    // unaligned storage and BOOL bitmaps return nullopt for scalar fallback.
    std::optional<PrimitiveView>
    DecodePrimitive() const;

    // String/string_view arrays and constants, retaining the source owner.
    // Representation is selected once per span; no string payload is copied.
    std::optional<DecodedView>
    DecodeStrings() const;

    // Typed complex objects retain their immutable source owner. Recursive
    // ARRAY and legacy ARRAY storage are distinct physical representations.
    template <typename T>
    std::optional<DecodedView>
    DecodeObjects() const {
        static_assert(std::is_same_v<T, Json> || std::is_same_v<T, Array> ||
                      std::is_same_v<T, ArrayView> ||
                      std::is_same_v<T, ArrayValue> ||
                      std::is_same_v<T, VectorArray>);
        CheckType<T>();
        return DecodeFlat(false,
                          sizeof(T),
                          alignof(T),
                          std::is_same_v<T, ArrayValue>,
                          std::is_same_v<T, ArrayView>);
    }
    template <typename T>
    const T*
    TryObjectAt(size_t row) const& {
        static_assert(std::is_same_v<T, Json> || std::is_same_v<T, Array> ||
                      std::is_same_v<T, ArrayView> ||
                      std::is_same_v<T, ArrayValue> ||
                      std::is_same_v<T, VectorArray>);
        CheckType<T>();
        return static_cast<const T*>(TryObjectAddress(
            row, std::is_same_v<T, ArrayValue>, std::is_same_v<T, ArrayView>));
    }
    template <typename T>
    const T*
    TryObjectAt(size_t row) const&& = delete;

    // O(1) broadcast for owned constants; preserves their immutable owner.
    ReadOnlyVectorView
    Broadcast(SelectionView rows) const;

    size_t
    CountNulls(SelectionView rows) const;

    // Optional aligned contiguous primitive fast path. The caller must still
    // honor defined/validity masks. nullptr means use the encoding-aware API.
    template <typename T>
    const T*
    FlatData() const& {
        static_assert(std::is_arithmetic_v<T>);
        CheckType<T>();
        const auto* data = FlatDataAddress();
        return reinterpret_cast<uintptr_t>(data) % alignof(T) == 0
                   ? static_cast<const T*>(data)
                   : nullptr;
    }
    template <typename T>
    const T*
    FlatData() const&& = delete;

    // Resolve encoding/row mapping once for both validity and the primitive
    // value. In particular, fragmented inputs need not be looked up twice.
    template <typename T>
    std::optional<T>
    TryValueAt(size_t row) const {
        static_assert(std::is_arithmetic_v<T>);
        CheckType<T>();
        T value;
        if (!ReadPrimitive(row, &value, sizeof(T)))
            return std::nullopt;
        return value;
    }

    // ArrayAt borrows owned Array storage; ArrayViewAt handles both physical
    // representations without copying payload. Keep this view alive while using it.
    std::optional<ArrayView>
    TryArrayViewAt(size_t row) const&;
    std::optional<ArrayView>
    TryArrayViewAt(size_t row) const&& = delete;
    ArrayView
    ArrayViewAt(size_t row) const&;
    ArrayView
    ArrayViewAt(size_t row) const&& = delete;

    const Array&
    ArrayAt(size_t row) const& {
        CheckType<Array>();
        AssertInfo(IsValid(row), "cannot read a NULL array payload");
        CheckArrayStorage(row, false);
        return *static_cast<const Array*>(ValueAddress(row));
    }
    const Array&
    ArrayAt(size_t row) const&& = delete;

    const ArrayValue&
    ArrayValueAt(size_t row) const& {
        CheckType<ArrayValue>();
        AssertInfo(IsValid(row), "cannot read a NULL recursive array payload");
        CheckArrayStorage(row, true);
        return *static_cast<const ArrayValue*>(ValueAddress(row));
    }
    const ArrayValue&
    ArrayValueAt(size_t row) const&& = delete;

    size_t
    ArrayLengthAt(size_t row) const;
    std::optional<size_t>
    TryArrayLengthAt(size_t row) const;

    const VectorArray&
    VectorArrayAt(size_t row) const& {
        CheckType<VectorArray>();
        AssertInfo(IsValid(row), "cannot read a NULL vector array payload");
        return *static_cast<const VectorArray*>(ValueAddress(row));
    }
    const VectorArray&
    VectorArrayAt(size_t row) const&& = delete;

    // Owning value API. Use StringViewAt/JsonViewAt to borrow variable payloads.
    template <typename T>
    T
    ValueAt(size_t row) const {
        static_assert(!std::is_same_v<T, ArrayView>,
                      "use ref-qualified ArrayViewAt for borrowed descriptors");
        CheckType<T>();
        AssertInfo(IsValid(row), "cannot read a NULL vector payload");
        if constexpr (std::is_same_v<T, bool>) {
            return BoolValueAt(row);
        } else if constexpr (std::is_same_v<T, Json>) {
            const auto text = static_cast<std::string_view>(JsonViewAt(row));
            return Json(simdjson::padded_string(text));
        } else if constexpr (std::is_same_v<T, std::string>) {
            return std::string(StringViewAt(row));
        } else if constexpr (std::is_same_v<T, VectorArray>) {
            return VectorArray(VectorArrayAt(row));
        } else if constexpr (std::is_same_v<T, Array>) {
            Array value;
            ArrayViewAt(row).output_data(value);
            return value;
        } else if constexpr (std::is_same_v<T, ArrayValue>) {
            return ArrayValueAt(row);
        } else {
            // Stored nullable payloads can be unaligned. This is one scalar
            // load, not a materialized copy of the input column.
            T value;
            std::memcpy(&value, ValueAddress(row), sizeof(T));
            return value;
        }
    }

 private:
    static ReadOnlyVectorView
    BorrowBuffer(
        DataType type,
        const void* values,
        std::shared_ptr<const void> owner,
        SelectionView defined,
        SelectionView valid,
        size_t width,
        bool string_views = false,
        bool indirect = false,
        bool recursive_array = false,
        PrimitivePayloadAccess access = PrimitivePayloadAccess::ValidRowsOnly,
        bool array_views = false);
    void
    CheckArrayStorage(size_t row, bool recursive) const;
    struct State;
    friend class VectorResultBuilder;
    friend class BitmapResultBuilder;
    static ReadOnlyVectorView
    AdoptPrivate(ColumnVectorPtr values, SelectionView rows);
    explicit ReadOnlyVectorView(std::shared_ptr<const State> state);

    const void*
    ValueAddress(size_t row) const;

    const void*
    FlatDataAddress() const;

    bool
    ReadPrimitive(size_t row, void* value, size_t width) const;

    std::optional<DecodedView>
    DecodeFlat(bool strings,
               size_t object_width = 0,
               size_t object_alignment = 0,
               bool recursive_array = false,
               bool array_views = false) const;
    const void*
    TryObjectAddress(size_t row, bool recursive_array, bool array_views) const;

    bool
    BoolValueAt(size_t row) const;

    template <typename T>
    void
    CheckType() const {
        CheckTypeFor<T>(logical_type());
    }

    template <typename T>
    static void
    CheckTypeFor(DataType type) {
        bool matches = false;
        if constexpr (std::is_same_v<T, bool>) {
            matches = type == DataType::BOOL;
        } else if constexpr (std::is_same_v<T, int8_t>) {
            matches = type == DataType::INT8;
        } else if constexpr (std::is_same_v<T, int16_t>) {
            matches = type == DataType::INT16;
        } else if constexpr (std::is_same_v<T, int32_t>) {
            matches = type == DataType::INT32;
        } else if constexpr (std::is_same_v<T, int64_t>) {
            matches = type == DataType::INT64 || type == DataType::TIMESTAMPTZ;
        } else if constexpr (std::is_same_v<T, float>) {
            matches = type == DataType::FLOAT;
        } else if constexpr (std::is_same_v<T, double>) {
            matches = type == DataType::DOUBLE;
        } else if constexpr (std::is_same_v<T, std::string>) {
            matches = type == DataType::STRING || type == DataType::VARCHAR ||
                      type == DataType::TEXT || type == DataType::GEOMETRY;
        } else if constexpr (std::is_same_v<T, Json>) {
            matches = type == DataType::JSON;
        } else if constexpr (std::is_same_v<T, ArrayView> ||
                             std::is_same_v<T, Array> ||
                             std::is_same_v<T, ArrayValue>) {
            matches = type == DataType::ARRAY;
        } else if constexpr (std::is_same_v<T, VectorArray>) {
            matches = type == DataType::VECTOR_ARRAY;
        } else {
            static_assert(!std::is_same_v<T, T>, "unsupported vector C++ type");
        }
        AssertInfo(matches,
                   "vector logical type does not match requested C++ type");
    }

    std::shared_ptr<const State> state_;
};

// Private writable storage: no mutable vector/pointer escapes. Finish consumes
// the buffer; publishing a result never snapshots it a second time.
class VectorResultBuilder {
 public:
    VectorResultBuilder(DataType type, size_t size, bool bitmap = false);
    VectorResultBuilder(const VectorResultBuilder&) = delete;
    VectorResultBuilder&
    operator=(const VectorResultBuilder&) = delete;

    template <typename T>
    void
    Set(size_t row, T value) {
        static_assert(
            !std::is_same_v<T, ArrayView>,
            "materialize borrowed arrays before publishing owned results");
        AssertInfo(values_ && row < static_cast<size_t>(values_->size()),
                   "invalid result builder write");
        ReadOnlyVectorView::CheckTypeFor<T>(
            values_->IsBitmap() ? DataType::BOOL : values_->type());
        if constexpr (std::is_same_v<T, Json>) {
            static_cast<Json*>(values_->GetRawData())[row] = Json(
                simdjson::padded_string(static_cast<std::string_view>(value)));
        } else if constexpr (std::is_same_v<T, bool>) {
            if (values_->IsBitmap()) {
                TargetBitmapView(values_->GetRawData(), values_->size())
                    .set(row, value);
            } else {
                static_cast<T*>(values_->GetRawData())[row] = value;
            }
        } else {
            static_cast<T*>(values_->GetRawData())[row] = std::move(value);
        }
        values_->clearNullAt(row);
    }

    // Bind destination type/address once for a whole run. A missing optional
    // keeps SQL NULL; callback exceptions cannot publish this private output.
    template <typename T, typename Fn>
    void
    EvaluateRange(size_t begin, size_t count, const Fn& fn) {
        static_assert(std::is_arithmetic_v<T> ||
                      std::is_same_v<T, std::string>);
        AssertInfo(values_ && !values_->IsBitmap() &&
                       begin <= values_->size() &&
                       count <= values_->size() - begin,
                   "invalid typed result range");
        ReadOnlyVectorView::CheckTypeFor<T>(values_->type());
        if (count == 0)
            return;
        auto* data = static_cast<T*>(values_->GetRawData()) + begin;
        for (size_t i = 0; i < count; ++i) {
            auto value = fn(i);
            if (value) {
                data[i] = std::move(*value);
                values_->clearNullAt(begin + i);
            } else
                values_->nullAt(begin + i);
        }
    }

    template <typename T, typename Fn>
    void
    EvaluateSelected(size_t begin, SelectionView selected, const Fn& fn) {
        static_assert(std::is_arithmetic_v<T> ||
                      std::is_same_v<T, std::string>);
        AssertInfo(values_ && !values_->IsBitmap() &&
                       begin <= values_->size() &&
                       selected.row_count() <= values_->size() - begin,
                   "invalid typed result selection");
        ReadOnlyVectorView::CheckTypeFor<T>(values_->type());
        auto* data = static_cast<T*>(values_->GetRawData());
        selected.ForEachSelected([&](size_t i) {
            auto value = fn(i);
            if (value) {
                data[begin + i] = std::move(*value);
                values_->clearNullAt(begin + i);
            } else {
                values_->nullAt(begin + i);
            }
        });
    }

    void
    SetNull(size_t row);
    ReadOnlyVectorView
    Finish(SelectionView rows) &&;

 private:
    ColumnVectorPtr values_;
};

// Bitmap kernels write directly into private bits, without per-row legacy
// vector/FieldData dispatch. Publication transfers both buffers once.
class BitmapResultBuilder {
 public:
    explicit BitmapResultBuilder(size_t size)
        : values_(size, false), valid_(size, false) {
    }
    BitmapResultBuilder(const BitmapResultBuilder&) = delete;
    BitmapResultBuilder&
    operator=(const BitmapResultBuilder&) = delete;
    template <typename T>
    void
    Set(size_t row, T value) {
        static_assert(std::is_same_v<T, bool>);
        AssertInfo(!finished_ && row < values_.size(),
                   "invalid bitmap builder write");
        values_.set(row, value);
        valid_.set(row);
    }
    // Bulk writes never expose a mutable buffer outside the builder.
    template <typename T>
    void
    CompareConstant(size_t begin,
                    size_t count,
                    const T* data,
                    T constant,
                    bitset::CompareOpType op) {
        CheckRange(begin, count);
        values_.view(begin, count)
            .inplace_compare_val(data, count, constant, op);
        valid_.view(begin, count).set();
    }
    template <typename T, typename U>
    void
    CompareColumns(size_t begin,
                   size_t count,
                   const T* left,
                   const U* right,
                   bitset::CompareOpType op) {
        CheckRange(begin, count);
        values_.view(begin, count)
            .inplace_compare_column(left, right, count, op);
        valid_.view(begin, count).set();
    }
    template <typename T>
    void
    WithinRange(size_t begin,
                size_t count,
                const T* data,
                T lower,
                T upper,
                bitset::RangeType op) {
        CheckRange(begin, count);
        values_.view(begin, count)
            .inplace_within_range_val(lower, upper, data, count, op);
        valid_.view(begin, count).set();
    }
    template <typename T,
              bitset::ArithOpType Arithmetic,
              bitset::CompareOpType Comparison>
    void
    ArithmeticCompare(size_t begin,
                      size_t count,
                      const T* data,
                      bitset::ArithHighPrecisionType<T> operand,
                      bitset::ArithHighPrecisionType<T> bound) {
        static_assert(Arithmetic == bitset::ArithOpType::Add ||
                      Arithmetic == bitset::ArithOpType::Sub ||
                      Arithmetic == bitset::ArithOpType::Mul);
        CheckRange(begin, count);
        values_.view(begin, count)
            .template inplace_arith_compare<T, Arithmetic, Comparison>(
                data, operand, bound, count);
        valid_.view(begin, count).set();
    }
    void
    SetRange(size_t begin, size_t count, bool value) {
        CheckRange(begin, count);
        auto bits = values_.view(begin, count);
        if (value)
            bits.set();
        else
            bits.reset();
        valid_.view(begin, count).set();
    }
    // Membership kernels OR matches into private bits. Clear this range first
    // and publish validity in bulk; the kernel must not retain the bitmap view.
    template <typename T, typename Terms>
    void
    MatchTerms(size_t begin, size_t count, const T* data, const Terms& terms) {
        CheckRange(begin, count);
        auto bits = values_.view(begin, count);
        bits.reset();
        // Legacy kernels accept an int length; views use size_t.
        for (size_t offset = 0; offset < count;) {
            const auto size =
                std::min(count - offset,
                         static_cast<size_t>(std::numeric_limits<int>::max()));
            terms.FilterChunk(data + offset,
                              static_cast<int>(size),
                              values_.view(begin + offset, size));
            offset += size;
        }
        valid_.view(begin, count).set();
    }
    // A typed kernel supplies values for an already validated contiguous run.
    // Exceptions leave a private, unpublished builder; validity is set last.
    template <typename Test>
    void
    EvaluateRange(size_t begin, size_t count, const Test& test) {
        CheckRange(begin, count);
        bitset::detail::ElementWiseBitsetPolicy<TargetBitmap::data_type>::op_func(
            values_.data(), begin, count, test);
        valid_.view(begin, count).set();
    }

    // Sparse scalar kernel: dispatch/type/address checks happen outside this
    // loop. Read only selected payloads and pack output once per bitmap word,
    // instead of invoking a batch kernel or checked Set() for every tiny run.
    template <typename Test>
    void
    EvaluateSelected(size_t begin, SelectionView selected, const Test& test) {
        CheckRange(begin, selected.row_count());
        using Word = TargetBitmap::data_type;
        using Policy = bitset::detail::ElementWiseBitsetPolicy<Word>;
        for (size_t offset = 0; offset < selected.row_count(); offset += 64) {
            const auto count =
                std::min<size_t>(64, selected.row_count() - offset);
            const Word valid = selected.ReadWord(offset, count);
            Word remaining = valid, truth = 0;
            while (remaining) {
                const auto bit =
                    static_cast<size_t>(__builtin_ctzll(remaining));
                if (test(offset + bit))
                    truth |= Word{1} << bit;
                remaining &= remaining - 1;
            }
            Policy::op_write(values_.data(), begin + offset, count, truth);
            Policy::op_write(valid_.data(), begin + offset, count, valid);
        }
    }
    // Sparse three-valued predicates preserve UNKNOWN without probing gaps.
    template <typename Test>
    void
    EvaluateOptionalSelected(size_t begin,
                             SelectionView selected,
                             const Test& test) {
        CheckRange(begin, selected.row_count());
        using Word = TargetBitmap::data_type;
        using Policy = bitset::detail::ElementWiseBitsetPolicy<Word>;
        for (size_t offset = 0; offset < selected.row_count(); offset += 64) {
            const auto count =
                std::min<size_t>(64, selected.row_count() - offset);
            Word remaining = selected.ReadWord(offset, count), truth = 0,
                 valid = 0;
            while (remaining) {
                const auto bit =
                    static_cast<size_t>(__builtin_ctzll(remaining));
                const auto value = test(offset + bit);
                if (value) {
                    valid |= Word{1} << bit;
                    if (*value)
                        truth |= Word{1} << bit;
                }
                remaining &= remaining - 1;
            }
            Policy::op_write(values_.data(), begin + offset, count, truth);
            Policy::op_write(valid_.data(), begin + offset, count, valid);
        }
    }
    // Three-valued callbacks pack truth and validity together; missing optional
    // means UNKNOWN. Unlike Set(), validation and bit writes are per word.
    template <typename Test>
    void
    EvaluateOptionalRange(size_t begin, size_t count, const Test& test) {
        CheckRange(begin, count);
        using Word = TargetBitmap::data_type;
        using Policy = bitset::detail::ElementWiseBitsetPolicy<Word>;
        constexpr size_t width = sizeof(Word) * 8;
        for (size_t offset = 0; offset < count; offset += width) {
            const size_t size = std::min(width, count - offset);
            Word truth = 0, valid = 0;
            for (size_t i = 0; i < size; ++i) {
                const auto value = test(offset + i);
                if (value) {
                    valid |= Word{1} << i;
                    if (*value)
                        truth |= Word{1} << i;
                }
            }
            Policy::op_write(values_.data(), begin + offset, size, truth);
            Policy::op_write(valid_.data(), begin + offset, size, valid);
        }
    }
    // Restore SQL validity after a total dense kernel. Both result masks must
    // be updated: clearing truth alone would turn NULL into FALSE under NOT.
    void
    ApplyValidity(size_t begin, SelectionView valid) {
        CheckRange(begin, valid.row_count());
        valid.CopyTo(valid_, begin);
        values_.view(begin, valid.row_count())
            .inplace_and(valid_.view(begin, valid.row_count()),
                         valid.row_count());
    }
    ReadOnlyVectorView
    Finish(SelectionView rows) &&;

 private:
    void
    CheckRange(size_t begin, size_t count) const {
        AssertInfo(!finished_ && begin <= values_.size() &&
                       count <= values_.size() - begin,
                   "invalid bitmap builder range write");
    }
    TargetBitmap values_;
    TargetBitmap valid_;
    bool finished_ = false;
};

// Values keep the original logical length/positions; only defined_rows may be
// consumed. Unselected storage is initialized but has NO semantic value. Never
// pass a partial copy to a legacy function that unconditionally reads all rows.
struct WritableVectorCopy {
    ColumnVectorPtr values;
    OwnedSelection defined_rows;
};

// Always detach, including validity and string payloads. Constant inputs are
// materialized; bitmap BOOL remains bitmap. No shared_ptr uniqueness heuristic.
WritableVectorCopy
MakeWritableCopy(const ReadOnlyVectorView& source);

// Copy only selected positions, keeping the original logical length.
WritableVectorCopy
MakeWritableCopy(const ReadOnlyVectorView& source, SelectionView rows);

// Legacy row consumers require a fully defined, flat column. In particular a
// projected BOOL is data, not the exclusion bitmap used by filter operators.
ColumnVectorPtr
MaterializeScalarColumn(const ReadOnlyVectorView& source);

// Transactional scatter: source[i] goes to destination[mapping[i]]. Repeated
// destinations are rejected. On failure destination and its coverage survive.
void
CopySelected(const ReadOnlyVectorView& source,
             SelectionView source_rows,
             const RowMapping& mapping,
             WritableVectorCopy& destination);

// Encoded columns preserve their immutable owners through RowVectorPtr. Legacy
// mutable inputs cross the snapshot boundary once; subsequent selections share
// payloads. Row-level NULLs are remapped separately from child NULLs.
RowVectorPtr
SelectRows(const RowVector& source, RowMappingPtr mapping);

RowVectorPtr
SliceRows(const RowVector& source, size_t begin, size_t count);

RowVectorPtr
MakeWritableCopy(const RowVector& source);

// Replaces the handle with a detached writable value. It deliberately does not
// infer buffer exclusivity from VectorPtr::use_count().
void
EnsureWritable(VectorPtr& vector);

// Bounded, single-execution-flow scratch pool. Release consumes ownership;
// callers must stop using all raw borrowed pointers before releasing. Published
// immutable snapshots detach from legacy writable vectors and remain valid.
class VectorPool {
 public:
    explicit VectorPool(size_t max_vectors = 8, size_t max_rows = 8192)
        : max_vectors_(max_vectors), max_rows_(max_rows) {
    }
    VectorPool(const VectorPool&) = delete;
    VectorPool&
    operator=(const VectorPool&) = delete;
    VectorPool(VectorPool&&) = default;
    VectorPool&
    operator=(VectorPool&&) = default;
    ColumnVectorPtr
    Acquire(DataType type, size_t size);
    void
    Release(ColumnVectorPtr vector);
    size_t
    cached_vectors() const {
        return vectors_.size();
    }

 private:
    size_t max_vectors_;
    size_t max_rows_;
    std::vector<ColumnVectorPtr> vectors_;
};

}  // namespace milvus
