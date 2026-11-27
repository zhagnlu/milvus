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

#include "common/VectorAccess.h"

#include <utility>
#include <numeric>
#include <unordered_map>

namespace milvus {
namespace {

template <typename T>
T
DetachedValue(const T& value) {
    if constexpr (std::is_same_v<T, Json>) {
        return Json(
            simdjson::padded_string(static_cast<std::string_view>(value)));
    } else {
        return T(value);
    }
}

template <typename Fn>
decltype(auto)
DispatchScalar(DataType type, Fn&& fn) {
    switch (type) {
        case DataType::BOOL:
            return fn.template operator()<bool>();
        case DataType::INT8:
            return fn.template operator()<int8_t>();
        case DataType::INT16:
            return fn.template operator()<int16_t>();
        case DataType::INT32:
            return fn.template operator()<int32_t>();
        case DataType::INT64:
        case DataType::TIMESTAMPTZ:
            return fn.template operator()<int64_t>();
        case DataType::FLOAT:
            return fn.template operator()<float>();
        case DataType::DOUBLE:
            return fn.template operator()<double>();
        case DataType::STRING:
        case DataType::VARCHAR:
        case DataType::TEXT:
        case DataType::GEOMETRY:
            return fn.template operator()<std::string>();
        case DataType::JSON:
            return fn.template operator()<Json>();
        case DataType::ARRAY:
            return fn.template operator()<Array>();
        case DataType::VECTOR_ARRAY:
            return fn.template operator()<VectorArray>();
        default:
            // This is an internal adapter capability, not user validation.
            ThrowInfo(
                Unsupported, "vector access does not support type {}", type);
    }
}

template <typename T, typename ValidFn, typename ValueFn>
ColumnVectorPtr
CopyFlat(DataType type,
         SelectionView rows,
         ValidFn&& is_valid,
         ValueFn&& value_at) {
    FixedVector<T> values(rows.row_count());
    TargetBitmap validity(rows.row_count(), false);
    size_t valid_count = 0;
    rows.ForEachSelected([&](size_t row) {
        if (is_valid(row)) {
            values[row] = DetachedValue<T>(value_at(row));
            validity.set(row);
            ++valid_count;
        }
    });
    FieldDataPtr data;
    if constexpr (std::is_same_v<T, VectorArray>) {
        data = std::make_shared<FieldDataVectorArrayImpl>(type, false);
        data->FillFieldData(values.data(), values.size());
    } else if constexpr (std::is_same_v<T, Array>) {
        data = std::make_shared<FieldData<Array>>(type, false);
        data->FillFieldData(values.data(), values.size());
    } else {
        data = std::make_shared<FieldData<T>>(type, false, std::move(values));
    }
    return std::make_shared<ColumnVector>(
        std::move(data), std::move(validity), rows.row_count() - valid_count);
}

template <typename T>
ColumnVectorPtr
CopyDecoded(DataType type,
            SelectionView rows,
            const ReadOnlyVectorView::DecodedView& decoded) {
    FixedVector<T> values(rows.row_count());
    TargetBitmap validity(rows.row_count(), false);
    size_t valid_count = 0;
    for (const auto& span : decoded.spans) {
        auto selected = OwnedSelection::Intersect(
            rows.Slice(span.begin, span.count), span.valid);
        selected.view().ForEachSelectedRange([&](size_t begin, size_t count) {
            auto* destination = values.data() + span.begin + begin;
            const auto source_begin = span.constant ? 0 : begin;
            if constexpr (std::is_same_v<T, std::string>) {
                if (span.string_views) {
                    const auto* source =
                        static_cast<const std::string_view*>(span.data);
                    for (size_t i = 0; i < count; ++i)
                        destination[i] = source[span.constant ? 0 : begin + i];
                } else {
                    const auto* source = static_cast<const T*>(span.data);
                    if (span.constant)
                        std::fill_n(destination, count, source[0]);
                    else
                        std::copy_n(source + source_begin, count, destination);
                }
            } else {
                const auto* source = static_cast<const T*>(span.data);
                if (span.constant)
                    std::fill_n(destination, count, source[0]);
                else
                    std::copy_n(source + source_begin, count, destination);
            }
            validity.set(span.begin + begin, count, true);
            valid_count += count;
        });
    }
    FieldDataPtr data =
        std::make_shared<FieldData<T>>(type, false, std::move(values));
    return std::make_shared<ColumnVector>(
        std::move(data), std::move(validity), rows.row_count() - valid_count);
}

}  // namespace

struct ReadOnlyVectorView::State : std::enable_shared_from_this<State> {
    State(VectorPtr owned_values,
          OwnedSelection rows,
          DataType type,
          ValueEncoding value_encoding,
          size_t width)
        : values(std::move(owned_values)),
          defined(std::move(rows)),
          logical_type(type),
          encoding(value_encoding),
          element_width(width) {
    }

    // Never expose this legacy mutable pointer; the public view shares only the
    // immutable State. It is created from detached values, never a caller's ptr.
    VectorPtr values;
    struct Part {
        size_t begin;
        size_t end;
        size_t source_begin;
        std::shared_ptr<const State> source;
    };
    std::vector<Part> parts;
    struct RowRef {
        size_t source;
        size_t row;
    };
    std::vector<RowRef> routes;
    std::vector<std::shared_ptr<const State>> sources;
    bool string_views = false;
    bool all_valid = false;
    bool indirect = false;
    bool recursive_array = false;
    bool array_views = false;
    bool full_payload_defined = false;
    PrimitivePayloadAccess payload_access =
        PrimitivePayloadAccess::ValidRowsOnly;
    const void*
    BorrowedAddress(size_t row) const {
        return indirect ? static_cast<const void* const*>(borrowed_values)[row]
                        : static_cast<const char*>(borrowed_values) +
                              row * element_width;
    }

    const State*
    Resolve(size_t& row) const {
        const State* state = this;
        while (true) {
            if (state->base) {
                row = state->mapping ? state->mapping->indices()[row]
                                     : row + state->offset;
                state = state->base.get();
            } else if (!state->routes.empty()) {
                const auto ref = state->routes[row];
                row = ref.row;
                state = state->sources[ref.source].get();
            } else if (!state->parts.empty()) {
                const auto it =
                    std::upper_bound(state->parts.begin(),
                                     state->parts.end(),
                                     row,
                                     [](size_t row, const Part& part) {
                                         return row < part.end;
                                     });
                AssertInfo(it != state->parts.end() && row >= it->begin,
                           "missing vector fragment");
                row = it->source_begin + row - it->begin;
                state = it->source.get();
            } else {
                return state;
            }
        }
    }
    std::shared_ptr<const State> base;
    RowMappingPtr mapping;
    size_t offset = 0;
    const void* borrowed_values = nullptr;
    std::shared_ptr<const void> owner;
    std::optional<OwnedSelection> validity;
    std::optional<SelectionView> bitmap_truth;
    std::optional<SelectionView> bitmap_validity;
    OwnedSelection defined;
    DataType logical_type;
    ValueEncoding encoding;
    size_t element_width;
};

namespace {
// This wrapper can travel through the existing operator interface, but does not
// implement legacy mutable SimpleVector accessors. Decode with CopyOf().
class PublishedVector final : public BaseVector {
 public:
    explicit PublishedVector(ReadOnlyVectorView view)
        : BaseVector(view.logical_type(), view.size()), view_(std::move(view)) {
        null_count_ = view_.CountNulls(view_.defined_rows());
    }
    const ReadOnlyVectorView&
    view() const {
        return view_;
    }
    void
    resize(vector_size_t size, bool = true) override {
        AssertInfo(
            size >= 0 && static_cast<size_t>(size) == length_,
            "immutable vector cannot be resized; obtain a writable copy");
    }
    void
    prepareForReuse() override {
        ThrowInfo(Unsupported, "immutable vector cannot be reused for writes");
    }
    VectorPtr
    cloneEmpty(vector_size_t size) const override {
        AssertInfo(size >= 0, "negative vector size");
        return std::make_shared<ColumnVector>(
            InitScalarFieldDataWithLength(type(), size),
            TargetBitmap(size, true),
            0);
    }

 private:
    ReadOnlyVectorView view_;
};
}  // namespace

ReadOnlyVectorView::ReadOnlyVectorView(std::shared_ptr<const State> state)
    : state_(std::move(state)) {
}

ReadOnlyVectorView
ReadOnlyVectorView::CopyOf(const VectorPtr& source) {
    AssertInfo(source && source->size() >= 0, "invalid source vector");
    if (const auto* published =
            dynamic_cast<const PublishedVector*>(source.get())) {
        return published->view();
    }
    return CopyOf(source, SelectionView::All(source->size()));
}

ReadOnlyVectorView
ReadOnlyVectorView::CopyOf(const VectorPtr& source, SelectionView rows) {
    AssertInfo(source && source->size() >= 0, "invalid source vector");
    AssertInfo(static_cast<size_t>(source->size()) == rows.row_count(),
               "vector and selection lengths differ");
    if (const auto* published =
            dynamic_cast<const PublishedVector*>(source.get())) {
        return published->view().Restrict(rows);
    }
    if (source->type() == DataType::NONE) {
        const auto* null =
            dynamic_cast<const ConstantVector<bool>*>(source.get());
        AssertInfo(null && null->IsNull(),
                   "untyped literal must be a NULL constant");
        return Nulls(DataType::NONE, rows);
    }
    const auto* column = dynamic_cast<const ColumnVector*>(source.get());
    if (column) {
        AssertInfo(column->Validity().size() == rows.row_count(),
                   "vector validity length does not match its logical length");
        if (column->IsBitmap()) {
            const TargetBitmapView bits(column->GetRawData(), rows.row_count());
            auto valid = column->Validity().clone();
            rows.IntersectInto(valid);
            TargetBitmap truth(bits);
            truth &= valid;
            auto copy = std::make_shared<ColumnVector>(std::move(truth),
                                                       std::move(valid));
            return ReadOnlyVectorView(
                std::make_shared<State>(std::move(copy),
                                        OwnedSelection::CopyOf(rows),
                                        DataType::BOOL,
                                        ValueEncoding::Bitmap,
                                        0));
        }
    }

    return DispatchScalar(source->type(), [&]<typename T>() {
        VectorPtr copy;
        auto encoding = ValueEncoding::Flat;
        if (column) {
            copy = CopyFlat<T>(
                source->type(),
                rows,
                [&](size_t i) { return column->Validity()[i]; },
                [&](size_t i) {
                    if constexpr (std::is_same_v<T, VectorArray>)
                        return VectorArray(static_cast<const VectorArray*>(
                            column->GetRawData())[i]);
                    else
                        return column->ValueAt<T>(i);
                });
        } else {
            const auto* constant =
                dynamic_cast<const ConstantVector<T>*>(source.get());
            AssertInfo(constant != nullptr,
                       "unsupported vector representation or mismatched "
                       "constant type");
            // Do not read a NULL or unselected constant payload.
            T value{};
            if (!rows.IsNone() && !constant->IsNull()) {
                value = DetachedValue<T>(constant->GetValue());
            }
            copy = std::make_shared<ConstantVector<T>>(
                source->type(),
                rows.row_count(),
                value,
                constant->IsNull() ? std::optional<size_t>(1) : std::nullopt);
            encoding = ValueEncoding::Constant;
        }
        return ReadOnlyVectorView(
            std::make_shared<State>(std::move(copy),
                                    OwnedSelection::CopyOf(rows),
                                    source->type(),
                                    encoding,
                                    sizeof(T)));
    });
}

size_t
ReadOnlyVectorView::size() const {
    return state_->defined.view().row_count();
}

DataType
ReadOnlyVectorView::logical_type() const {
    return state_->logical_type;
}

ValueEncoding
ReadOnlyVectorView::encoding() const {
    return state_->encoding;
}

SelectionView
ReadOnlyVectorView::defined_rows() const& {
    return state_->defined.view();
}

bool
ReadOnlyVectorView::IsValid(size_t row) const {
    AssertInfo(defined_rows().IsSelected(row),
               "cannot read an undefined vector position {}",
               row);
    if (state_->all_valid)
        return true;
    const State* state = state_->Resolve(row);
    if (state->all_valid)
        return true;
    if (state->bitmap_validity)
        return state->bitmap_validity->IsSelected(row);
    if (state->validity) {
        return state->validity->view().IsSelected(row);
    }
    // The concrete vector is private and detached; ValidAt's legacy non-const
    // signature cannot leak a mutable pointer through this interface.
    return static_cast<SimpleVector*>(state->values.get())->ValidAt(row);
}

std::optional<SelectionView>
ReadOnlyVectorView::TryBorrowValidity() const& {
    const auto rows = defined_rows();
    const State* state = state_.get();
    size_t offset = 0;
    while (true) {
        if (state->all_valid)
            return rows;
        if (!state->base || state->mapping)
            break;
        offset += state->offset;
        state = state->base.get();
    }
    if (state->base || !state->parts.empty() || !state->routes.empty())
        return std::nullopt;
    if (!state->validity && state->encoding == ValueEncoding::Constant)
        return static_cast<SimpleVector*>(state->values.get())->ValidAt(0)
                   ? rows
                   : SelectionView::None(size());
    const auto valid =
        state->bitmap_validity ? state->bitmap_validity->Slice(offset, size())
        : state->validity
            ? state->validity->view().Slice(offset, size())
            : SelectionView::FromBitmap(
                  static_cast<const ColumnVector*>(state->values.get())
                      ->Validity())
                  .Slice(offset, size());
    if (valid.IsSubsetOf(rows))
        return valid;
    return std::nullopt;
}

OwnedSelection
ReadOnlyVectorView::CopyValidity() const {
    const auto rows = defined_rows();
    if (rows.IsNone())
        return OwnedSelection::None(size());

    const State* state = state_.get();
    size_t offset = 0;
    while (true) {
        if (state->all_valid)
            return OwnedSelection::CopyOf(rows);
        if (!state->base)
            break;
        if (state->mapping)
            break;  // Dictionary positions need the original mapping.
        offset += state->offset;
        state = state->base.get();
    }
    if (!state->base && state->parts.empty() && state->routes.empty()) {
        if (state->bitmap_validity)
            return OwnedSelection::Intersect(
                rows, state->bitmap_validity->Slice(offset, size()));
        if (state->validity)
            return OwnedSelection::Intersect(
                rows, state->validity->view().Slice(offset, size()));
        if (state->encoding == ValueEncoding::Constant)
            return static_cast<SimpleVector*>(state->values.get())->ValidAt(0)
                       ? OwnedSelection::CopyOf(rows)
                       : OwnedSelection::None(size());
        const auto& valid =
            static_cast<const ColumnVector*>(state->values.get())->Validity();
        return OwnedSelection::Intersect(
            rows, SelectionView::FromBitmap(valid).Slice(offset, size()));
    }

    // Resolve through this view (including any outer slice/restriction), never
    // through the partially traversed state. Undefined rows stay unobserved.
    auto valid = OwnedSelection::None(size());
    rows.ForEachSelected([&](size_t row) {
        if (IsValid(row))
            valid.SetSelected(row);
    });
    return valid;
}

void
ReadOnlyVectorView::CopyTrueTo(SelectionView rows,
                               TargetBitmap& output,
                               size_t begin) const {
    AssertInfo(logical_type() == DataType::BOOL && rows.row_count() == size() &&
                   begin <= output.size() && size() <= output.size() - begin,
               "BOOL output has wrong type/domain");
    AssertInfo(rows.IsSubsetOf(defined_rows()),
               "BOOL output includes undefined rows");
    const State* state = state_.get();
    size_t offset = 0;
    while (state->base && !state->mapping) {
        offset += state->offset;
        state = state->base.get();
    }
    if (!state->base && state->parts.empty() && state->routes.empty() &&
        state->encoding == ValueEncoding::Bitmap) {
        if (state->bitmap_truth) {
            // BorrowBooleanMasks proves TRUE is already a subset of validity.
            state->bitmap_truth->Slice(offset, size()).CopyTo(output, begin);
            rows.IntersectInto(output, begin);
            return;
        }
        if (auto valid = TryBorrowValidity()) {
            const auto* column =
                static_cast<const ColumnVector*>(state->values.get());
            TargetBitmapView bits(column->GetRawData(), column->size());
            TargetBitmap::policy_type::op_copy(
                bits.data(), offset, output.data(), begin, size());
            valid->IntersectInto(output, begin);
            rows.IntersectInto(output, begin);
            return;
        }
    }
    auto masks = CopyBoolMasks(rows);
    masks.truth.view().CopyTo(output, begin);
}

ReadOnlyVectorView::BooleanMasks
ReadOnlyVectorView::CopyBoolMasks(SelectionView rows) const {
    AssertInfo(logical_type() == DataType::BOOL && rows.row_count() == size(),
               "BOOL mask request has wrong type/domain");
    AssertInfo(rows.IsSubsetOf(defined_rows()),
               "BOOL mask request includes undefined rows");
    const State* state = state_.get();
    size_t offset = 0;
    while (state->base && !state->mapping) {
        offset += state->offset;
        state = state->base.get();
    }
    auto truth = OwnedSelection::None(size());
    if (!state->base && state->parts.empty() && state->routes.empty() &&
        state->bitmap_truth) {
        // Both masks are canonical. Clip each once to the requested coverage;
        // no normalization scan or second validity intersection is necessary.
        auto valid = OwnedSelection::None(size());
        valid.Assign(state->bitmap_validity->Slice(offset, size()));
        valid.Intersect(rows);
        truth.Assign(state->bitmap_truth->Slice(offset, size()));
        truth.Intersect(rows);
        return {std::move(truth), std::move(valid)};
    }
    auto valid = CopyValidity();
    valid.Intersect(rows);
    if (valid.view().IsNone())
        return {std::move(truth), std::move(valid)};
    if (!state->base && state->parts.empty() && state->routes.empty()) {
        if (state->encoding == ValueEncoding::Bitmap) {
            const auto* column =
                static_cast<const ColumnVector*>(state->values.get());
            TargetBitmapView bits(column->GetRawData(), column->size());
            truth = OwnedSelection::OwnBitmap(
                TargetBitmap(bits.view(offset, size())));
            truth.Intersect(valid.view());
            return {std::move(truth), std::move(valid)};
        }
        if (state->encoding == ValueEncoding::Constant) {
            if (*static_cast<const bool*>(
                    static_cast<SimpleVector*>(state->values.get())
                        ->RawValueAt(0, sizeof(bool))))
                truth = OwnedSelection::CopyOf(valid.view());
            return {std::move(truth), std::move(valid)};
        }
    }
    if (auto decoded = DecodePrimitive()) {
        TargetBitmap bits(size(), false);
        for (const auto& span : decoded->spans) {
            const auto* data = static_cast<const bool*>(span.data);
            valid.view()
                .Slice(span.begin, span.count)
                .ForEachSelectedRange([&](size_t offset, size_t count) {
                    bitset::detail::ElementWiseBitsetPolicy<
                        TargetBitmap::data_type>::
                        op_func(bits.data(),
                                span.begin + offset,
                                count,
                                [&](size_t i) {
                                    return data[span.constant ? 0 : offset + i];
                                });
                });
        }
        return {OwnedSelection::OwnBitmap(std::move(bits)), std::move(valid)};
    }
    // Dictionary, indirect and incrementally merged views retain a fallback.
    // Only valid requested positions may have their payload examined.
    valid.view().ForEachSelected([&](size_t row) {
        if (BoolValueAt(row))
            truth.SetSelected(row);
    });
    return {std::move(truth), std::move(valid)};
}

ReadOnlyVectorView
ReadOnlyVectorView::BorrowBooleanMasks(SelectionView truth,
                                       SelectionView valid,
                                       std::shared_ptr<const void> owner) {
    AssertInfo(owner && truth.row_count() == valid.row_count(),
               "borrowed BOOL masks have wrong owner/domain");
    AssertInfo(truth.IsSubsetOf(valid),
               "borrowed BOOL TRUE row cannot be NULL");
    auto state = std::make_shared<State>(nullptr,
                                         OwnedSelection::All(truth.row_count()),
                                         DataType::BOOL,
                                         ValueEncoding::Bitmap,
                                         sizeof(bool));
    state->owner = std::move(owner);
    state->bitmap_truth = truth;
    state->bitmap_validity = valid;
    return ReadOnlyVectorView(std::move(state));
}

ReadOnlyVectorView
ReadOnlyVectorView::FromBooleanMasks(OwnedSelection truth,
                                     OwnedSelection valid,
                                     SelectionView defined) {
    AssertInfo(truth.view().row_count() == defined.row_count() &&
                   valid.view().row_count() == defined.row_count(),
               "BOOL masks have wrong row domain");
    AssertInfo(valid.view().IsSubsetOf(defined) &&
                   truth.view().IsSubsetOf(valid.view()),
               "BOOL masks violate truth/validity/coverage");
    auto bits = std::move(truth).TakeBitmap();
    auto validity = std::move(valid).TakeBitmap();
    return AdoptPrivate(
        std::make_shared<ColumnVector>(std::move(bits), std::move(validity)),
        defined);
}

std::optional<ReadOnlyVectorView::PrimitiveView>
ReadOnlyVectorView::DecodePrimitive() const {
    return DecodeFlat(false);
}

std::optional<ReadOnlyVectorView::DecodedView>
ReadOnlyVectorView::DecodeStrings() const {
    return DecodeFlat(true);
}

std::optional<ReadOnlyVectorView::DecodedView>
ReadOnlyVectorView::DecodeFlat(bool strings,
                               size_t object_width,
                               size_t object_alignment,
                               bool recursive_array,
                               bool array_views) const {
    const bool string_type = logical_type() == DataType::STRING ||
                             logical_type() == DataType::VARCHAR ||
                             logical_type() == DataType::TEXT ||
                             logical_type() == DataType::GEOMETRY;
    if (strings != string_type)
        return std::nullopt;
    size_t width = 0;
    switch (logical_type()) {
        case DataType::STRING:
        case DataType::VARCHAR:
        case DataType::TEXT:
        case DataType::GEOMETRY:
            break;
        case DataType::JSON:
        case DataType::ARRAY:
        case DataType::VECTOR_ARRAY:
            if (!object_width)
                return std::nullopt;
            width = object_width;
            break;
        case DataType::BOOL:
        case DataType::INT8:
            width = 1;
            break;
        case DataType::INT16:
            width = 2;
            break;
        case DataType::INT32:
        case DataType::FLOAT:
            width = 4;
            break;
        case DataType::INT64:
        case DataType::TIMESTAMPTZ:
        case DataType::DOUBLE:
            width = 8;
            break;
        default:
            return std::nullopt;
    }
    PrimitiveView output{state_, {}};
    auto visit = [&](auto&& self,
                     const State* state,
                     size_t start,
                     size_t count,
                     size_t begin) -> bool {
        if (!count)
            return true;
        while (state->base) {
            if (state->mapping)
                return false;
            start += state->offset;
            state = state->base.get();
        }
        if (!state->routes.empty())
            return false;
        if (!state->parts.empty()) {
            const size_t end = start + count;
            for (const auto& part : state->parts) {
                const size_t lo = std::max(start, part.begin);
                const size_t hi = std::min(end, part.end);
                if (lo < hi && !self(self,
                                     part.source.get(),
                                     part.source_begin + lo - part.begin,
                                     hi - lo,
                                     begin + lo - start))
                    return false;
            }
            return true;
        }
        const size_t leaf_width =
            strings ? (state->string_views ? sizeof(std::string_view)
                                           : sizeof(std::string))
                    : width;
        const size_t alignment =
            strings ? (state->string_views ? alignof(std::string_view)
                                           : alignof(std::string))
                    : (object_alignment ? object_alignment : width);
        const bool constant = state->encoding == ValueEncoding::Constant;
        if ((!constant && state->encoding != ValueEncoding::Flat) ||
            state->indirect ||
            (object_width && logical_type() == DataType::ARRAY &&
             (state->recursive_array != recursive_array ||
              state->array_views != array_views)) ||
            (state->element_width && state->element_width != leaf_width))
            return false;
        auto valid = SelectionView::All(count);
        if (state->validity) {
            valid = state->validity->view().Slice(start, count);
        } else if (constant) {
            if (!static_cast<SimpleVector*>(state->values.get())->ValidAt(0))
                valid = SelectionView::None(count);
        } else {
            valid = SelectionView::FromBitmap(
                        static_cast<const ColumnVector*>(state->values.get())
                            ->Validity())
                        .Slice(start, count);
        }
        // A NULL constant/vector need not have addressable payload storage.
        const void* data = nullptr;
        if (!valid.IsNone()) {
            data = state->owner
                       ? state->BorrowedAddress(constant ? 0 : start)
                       : static_cast<SimpleVector*>(state->values.get())
                             ->RawValueAt(constant ? 0 : start, leaf_width);
            if (!data || reinterpret_cast<uintptr_t>(data) % alignment)
                return false;
        }
        output.spans.push_back(
            {begin,
             count,
             data,
             constant,
             valid,
             state_->defined.view().Slice(begin, count),
             state->payload_access,
             state->string_views,
             state->full_payload_defined ||
                 state->defined.view().Slice(start, count).IsAll()});
        return true;
    };
    if (!visit(visit, state_.get(), 0, size(), 0))
        return std::nullopt;
    return output;
}

ReadOnlyVectorView
ReadOnlyVectorView::Broadcast(SelectionView rows) const {
    AssertInfo(size() == 1, "broadcast requires a single value");
    AssertInfo(rows.IsNone() || defined_rows().IsSelected(0),
               "cannot broadcast an undefined value");
    if (state_->encoding == ValueEncoding::Constant && !state_->base) {
        auto state = std::make_shared<State>(*state_);
        const bool valid = defined_rows().IsSelected(0) && IsValid(0);
        state->defined = OwnedSelection::CopyOf(rows);
        state->validity = valid ? OwnedSelection::All(rows.row_count())
                                : OwnedSelection::None(rows.row_count());
        state->all_valid = valid;
        state->full_payload_defined = valid;
        return ReadOnlyVectorView(std::move(state));
    }
    return Dictionary(std::make_shared<RowMapping>(
                          std::vector<vector_size_t>(rows.row_count(), 0)))
        .Restrict(rows);
}

size_t
ReadOnlyVectorView::CountNulls(SelectionView rows) const {
    AssertInfo(rows.row_count() == size(),
               "null count selection length mismatch");
    size_t count = 0;
    rows.ForEachSelected([&](size_t i) { count += !IsValid(i); });
    return count;
}

void
ReadOnlyVectorView::CheckArrayStorage(size_t row, bool recursive) const {
    const auto* leaf = state_->Resolve(row);
    if (leaf->recursive_array != recursive || leaf->array_views)
        ThrowInfo(Unsupported, "ARRAY physical representation mismatch");
}

size_t
ReadOnlyVectorView::ArrayLengthAt(size_t row) const {
    CheckType<Array>();
    AssertInfo(IsValid(row), "cannot read a NULL array length");
    auto physical = row;
    return state_->Resolve(physical)->recursive_array
               ? ArrayValueAt(row).size()
               : ArrayViewAt(row).length();
}

std::optional<size_t>
ReadOnlyVectorView::TryArrayLengthAt(size_t row) const {
    CheckType<Array>();
    AssertInfo(defined_rows().IsSelected(row),
               "cannot read an undefined array position");
    const auto* leaf = state_->Resolve(row);
    const bool valid =
        leaf->all_valid ||
        (leaf->validity
             ? leaf->validity->view().IsSelected(row)
             : static_cast<SimpleVector*>(leaf->values.get())->ValidAt(row));
    if (!valid)
        return std::nullopt;
    const auto offset = leaf->encoding == ValueEncoding::Constant ? 0 : row;
    const auto* data = leaf->owner
                           ? leaf->BorrowedAddress(offset)
                           : static_cast<SimpleVector*>(leaf->values.get())
                                 ->RawValueAt(offset, leaf->element_width);
    return leaf->recursive_array ? static_cast<const ArrayValue*>(data)->size()
           : leaf->array_views   ? static_cast<const ArrayView*>(data)->length()
                                 : static_cast<const Array*>(data)->length();
}

std::optional<ArrayView>
ReadOnlyVectorView::TryArrayViewAt(size_t row) const& {
    CheckType<Array>();
    AssertInfo(defined_rows().IsSelected(row),
               "cannot read an undefined array position");
    const auto* leaf = state_->Resolve(row);
    const bool valid =
        leaf->all_valid ||
        (leaf->validity
             ? leaf->validity->view().IsSelected(row)
             : static_cast<SimpleVector*>(leaf->values.get())->ValidAt(row));
    if (!valid)
        return std::nullopt;
    if (leaf->recursive_array)
        ThrowInfo(Unsupported, "ARRAY physical representation mismatch");
    const auto offset = leaf->encoding == ValueEncoding::Constant ? 0 : row;
    const auto* data = leaf->owner
                           ? leaf->BorrowedAddress(offset)
                           : static_cast<SimpleVector*>(leaf->values.get())
                                 ->RawValueAt(offset, leaf->element_width);
    if (leaf->array_views)
        return *static_cast<const ArrayView*>(data);
    const auto& array = *static_cast<const Array*>(data);
    return ArrayView(const_cast<char*>(array.data()),
                     array.length(),
                     array.byte_size(),
                     array.get_element_type(),
                     array.get_offsets_data(),
                     array.get_element_valid_data(),
                     array.is_element_nullable(),
                     array.has_invalid_element());
}

ArrayView
ReadOnlyVectorView::ArrayViewAt(size_t row) const& {
    auto value = TryArrayViewAt(row);
    AssertInfo(value.has_value(), "cannot read a NULL array payload");
    return *value;
}

const void*
ReadOnlyVectorView::ValueAddress(size_t row) const {
    const State* state = state_->Resolve(row);
    if (state->owner) {
        return state->BorrowedAddress(row);
    }
    const auto offset = state->encoding == ValueEncoding::Constant ? 0 : row;
    return static_cast<SimpleVector*>(state->values.get())
        ->RawValueAt(offset, state->element_width);
}

bool
ReadOnlyVectorView::BoolValueAt(size_t row) const {
    auto base_row = row;
    const State* state = state_->Resolve(base_row);
    if (state->encoding == ValueEncoding::Bitmap) {
        if (state->bitmap_truth)
            return state->bitmap_truth->IsSelected(base_row);
        const auto* column = static_cast<ColumnVector*>(state->values.get());
        const TargetBitmapView bits(column->GetRawData(), column->size());
        return bits[base_row];
    }
    return *static_cast<const bool*>(ValueAddress(row));
}

const void*
ReadOnlyVectorView::FlatDataAddress() const {
    const State* state = state_.get();
    while (state->base) {
        if (state->mapping)
            return nullptr;
        state = state->base.get();
    }
    if (size() == 0 || !state->parts.empty() || !state->routes.empty() ||
        state->encoding != ValueEncoding::Flat ||
        (state->owner && !state->borrowed_values))
        return nullptr;
    return ValueAddress(0);
}

bool
ReadOnlyVectorView::ReadPrimitive(size_t row, void* value, size_t width) const {
    AssertInfo(defined_rows().IsSelected(row),
               "cannot read an undefined vector position {}",
               row);
    const auto* state = state_->Resolve(row);
    const bool valid =
        state->all_valid ||
        (state->bitmap_validity ? state->bitmap_validity->IsSelected(row)
         : state->owner
             ? state->validity->view().IsSelected(row)
             : static_cast<SimpleVector*>(state->values.get())->ValidAt(row));
    if (!valid)
        return false;
    if (state->encoding == ValueEncoding::Bitmap) {
        if (state->bitmap_truth) {
            *static_cast<bool*>(value) = state->bitmap_truth->IsSelected(row);
            return true;
        }
        const auto* column = static_cast<ColumnVector*>(state->values.get());
        *static_cast<bool*>(value) =
            TargetBitmapView(column->GetRawData(), column->size())[row];
    } else {
        const auto* address =
            state->owner
                ? state->BorrowedAddress(row)
                : static_cast<SimpleVector*>(state->values.get())
                      ->RawValueAt(
                          state->encoding == ValueEncoding::Constant ? 0 : row,
                          state->element_width);
        std::memcpy(value, address, width);
    }
    return true;
}

WritableVectorCopy
MakeWritableCopy(const ReadOnlyVectorView& source) {
    const auto rows = source.defined_rows();
    ColumnVectorPtr values;
    if (source.encoding() == ValueEncoding::Bitmap) {
        auto masks = source.CopyBoolMasks(rows);
        values =
            std::make_shared<ColumnVector>(std::move(masks.truth).TakeBitmap(),
                                           std::move(masks.valid).TakeBitmap());
    } else {
        values = DispatchScalar(source.logical_type(), [&]<typename T>() {
            if constexpr (std::is_arithmetic_v<T>) {
                if (auto decoded = source.DecodePrimitive())
                    return CopyDecoded<T>(
                        source.logical_type(), rows, *decoded);
            } else if constexpr (std::is_same_v<T, std::string>) {
                if (auto decoded = source.DecodeStrings())
                    return CopyDecoded<T>(
                        source.logical_type(), rows, *decoded);
            }
            return CopyFlat<T>(
                source.logical_type(),
                rows,
                [&](size_t i) { return source.IsValid(i); },
                [&](size_t i) { return source.ValueAt<T>(i); });
        });
    }
    return {std::move(values), OwnedSelection::CopyOf(rows)};
}

VectorPtr
ReadOnlyVectorView::CopyForLegacyFunction() const {
    AssertInfo(defined_rows().IsAll(),
               "legacy function arguments require complete coverage");
    if (logical_type() == DataType::NONE)
        return std::make_shared<ConstantVector<bool>>(
            DataType::NONE, size(), false, 1);
    const State* state = state_.get();
    // Any mapping of one constant leaf is still constant. Fragmented/merged
    // views must retain their general copying path.
    while (state->base) state = state->base.get();
    if (state->parts.empty() && state->routes.empty() &&
        state->encoding == ValueEncoding::Constant) {
        return DispatchScalar(logical_type(), [&]<typename T>() -> VectorPtr {
            const bool valid = size() && IsValid(0);
            T value{};
            if (valid)
                value = DetachedValue<T>(ValueAt<T>(0));
            return std::make_shared<ConstantVector<T>>(
                logical_type(),
                size(),
                value,
                valid ? std::nullopt : std::optional<size_t>{1});
        });
    }
    return MakeWritableCopy(*this).values;
}

ColumnVectorPtr
MaterializeScalarColumn(const ReadOnlyVectorView& source) {
    AssertInfo(source.defined_rows().IsAll(),
               "row materialization requires complete coverage");
    if (source.logical_type() == DataType::BOOL) {
        return CopyFlat<bool>(
            DataType::BOOL,
            source.defined_rows(),
            [&](size_t row) { return source.IsValid(row); },
            [&](size_t row) { return source.ValueAt<bool>(row); });
    }
    return MakeWritableCopy(source).values;
}

ReadOnlyVectorView
ReadOnlyVectorView::BorrowBuffer(DataType type,
                                 const void* values,
                                 std::shared_ptr<const void> owner,
                                 SelectionView defined,
                                 SelectionView valid,
                                 size_t width,
                                 bool string_views,
                                 bool indirect,
                                 bool recursive_array,
                                 PrimitivePayloadAccess access,
                                 bool array_views) {
    AssertInfo(owner, "borrowed values require an immutable lifetime owner");
    AssertInfo(defined.row_count() == valid.row_count(),
               "borrowed validity length mismatch");
    AssertInfo(defined.row_count() <=
                   static_cast<size_t>(std::numeric_limits<int64_t>::max()),
               "borrowed vector exceeds logical size range");
    AssertInfo(values || defined.IsNone() || valid.IsNone(),
               "missing borrowed payload");
    const auto storage_width = indirect ? sizeof(const void*) : width;
    AssertInfo(defined.row_count() <=
                   std::numeric_limits<size_t>::max() / storage_width,
               "borrowed vector byte size overflow");
    auto state = std::make_shared<State>(nullptr,
                                         OwnedSelection::CopyOf(defined),
                                         type,
                                         ValueEncoding::Flat,
                                         width);
    state->string_views = string_views;
    state->indirect = indirect;
    state->recursive_array = recursive_array;
    state->array_views = array_views;
    AssertInfo(
        access == PrimitivePayloadAccess::ValidRowsOnly ||
            (!indirect &&
             (type == DataType::INT8 || type == DataType::INT16 ||
              type == DataType::INT32 || type == DataType::INT64 ||
              type == DataType::TIMESTAMPTZ || type == DataType::FLOAT ||
              type == DataType::DOUBLE || type == DataType::STRING ||
              type == DataType::VARCHAR || type == DataType::TEXT)),
        "dense NULL payload access requires direct numeric or string storage");
    state->payload_access = access;
    if (indirect)
        state->encoding = ValueEncoding::Segmented;
    state->all_valid = valid.selected_count() == valid.row_count();
    state->borrowed_values = values;
    state->owner = std::move(owner);
    state->validity.emplace(OwnedSelection::CopyOf(valid));
    return ReadOnlyVectorView(std::move(state));
}

ReadOnlyVectorView
ReadOnlyVectorView::Dictionary(RowMappingPtr mapping) const {
    AssertInfo(mapping, "dictionary mapping cannot be null");
    auto defined = OwnedSelection::All(mapping->indices().size());
    for (size_t i = 0; i < mapping->indices().size(); ++i) {
        const auto index = mapping->indices()[i];
        AssertInfo(index >= 0 && static_cast<size_t>(index) < size(),
                   "dictionary index out of bounds");
        if (!defined_rows().IsSelected(index)) {
            defined.SetSelected(i, false);
        }
    }
    auto state = std::make_shared<State>(nullptr,
                                         std::move(defined),
                                         logical_type(),
                                         ValueEncoding::Dictionary,
                                         state_->element_width);
    state->base = state_;
    state->all_valid = state_->all_valid;
    // Collapse nested dictionaries/slices once, so per-row reads do not walk
    // an ever-growing chain. Flat input keeps and shares the original mapping.
    if (state_->base) {
        auto indices = mapping->indices();
        for (auto& index : indices) {
            size_t row = index;
            auto base = state_;
            while (base->base) {
                row = base->mapping ? base->mapping->indices()[row]
                                    : row + base->offset;
                base = base->base;
            }
            AssertInfo(row <= static_cast<size_t>(
                                  std::numeric_limits<vector_size_t>::max()),
                       "dictionary index exceeds vector index range");
            index = row;
            state->base = std::move(base);
        }
        // An empty dictionary still retains the immutable source owner.
        while (state->base->base) {
            state->base = state->base->base;
        }
        mapping = std::make_shared<RowMapping>(std::move(indices));
    }
    state->mapping = std::move(mapping);
    return ReadOnlyVectorView(std::move(state));
}

ReadOnlyVectorView
ReadOnlyVectorView::Slice(size_t begin, size_t count) const {
    AssertInfo(begin <= size() && count <= size() - begin,
               "vector slice out of bounds");
    if (begin == 0 && count == size()) {
        return *this;
    }
    auto state = std::make_shared<State>(
        nullptr,
        OwnedSelection::CopyOf(defined_rows().Slice(begin, count)),
        logical_type(),
        ValueEncoding::Dictionary,
        state_->element_width);
    state->base = state_;
    state->all_valid = state_->all_valid;
    state->offset = begin;
    if (state_->base && !state_->mapping) {
        state->base = state_->base;
        state->offset += state_->offset;
    }
    return ReadOnlyVectorView(std::move(state));
}

ReadOnlyVectorView
ReadOnlyVectorView::Restrict(SelectionView rows) const {
    AssertInfo(rows.row_count() == size(), "vector selection length mismatch");
    AssertInfo(rows.IsSubsetOf(defined_rows()),
               "requested vector rows are undefined");
    // The subset check plus equal cardinality proves identical coverage.
    // Repeated sparse reads can share State instead of cloning its routing.
    if (rows.IsAll() ||
        rows.selected_count() == defined_rows().selected_count()) {
        return *this;
    }
    auto state = std::make_shared<State>(*state_);
    state->full_payload_defined |= state_->defined.view().IsAll();
    state->defined.Assign(rows);
    return ReadOnlyVectorView(std::move(state));
}

VectorPtr
ReadOnlyVectorView::AsVector() const {
    return std::make_shared<PublishedVector>(*this);
}

std::optional<std::string_view>
ReadOnlyVectorView::TryStringViewAt(size_t row) const& {
    CheckType<std::string>();
    AssertInfo(defined_rows().IsSelected(row),
               "cannot read an undefined vector position {}",
               row);
    const auto* state = state_->Resolve(row);
    if (!state->all_valid) {
        const bool valid =
            state->validity
                ? state->validity->view().IsSelected(row)
                : static_cast<SimpleVector*>(state->values.get())->ValidAt(row);
        if (!valid)
            return std::nullopt;
    }
    const auto offset = state->encoding == ValueEncoding::Constant ? 0 : row;
    const void* address = state->owner
                              ? state->BorrowedAddress(offset)
                              : static_cast<SimpleVector*>(state->values.get())
                                    ->RawValueAt(offset, state->element_width);
    if (state->string_views)
        return *static_cast<const std::string_view*>(address);
    return std::string_view(*static_cast<const std::string*>(address));
}

std::string_view
ReadOnlyVectorView::StringViewAt(size_t row) const& {
    CheckType<std::string>();
    AssertInfo(IsValid(row), "cannot read a NULL string");
    auto physical_row = row;
    const auto* state = state_->Resolve(physical_row);
    if (state->string_views)
        return *static_cast<const std::string_view*>(ValueAddress(row));
    return *static_cast<const std::string*>(ValueAddress(row));
}

WritableVectorCopy
MakeWritableCopy(const ReadOnlyVectorView& source, SelectionView rows) {
    return MakeWritableCopy(source.Restrict(rows));
}

void
CopySelected(const ReadOnlyVectorView& source,
             SelectionView rows,
             const RowMapping& mapping,
             WritableVectorCopy& destination) {
    AssertInfo(destination.values, "missing scatter destination");
    AssertInfo(rows.row_count() == source.size() &&
                   mapping.indices().size() == source.size(),
               "scatter source shape mismatch");
    AssertInfo(
        destination.values->size() ==
            static_cast<int64_t>(destination.defined_rows.view().row_count()),
        "scatter destination shape mismatch");
    const auto target_type = destination.values->IsBitmap()
                                 ? DataType::BOOL
                                 : destination.values->type();
    AssertInfo(target_type == source.logical_type(), "scatter type mismatch");
    auto written = OwnedSelection::None(destination.values->size());
    rows.ForEachSelected([&](size_t i) {
        AssertInfo(source.defined_rows().IsSelected(i),
                   "scatter source row is undefined");
        const auto target = mapping.indices()[i];
        AssertInfo(target >= 0 && target < destination.values->size(),
                   "scatter target out of bounds");
        AssertInfo(!written.view().IsSelected(target),
                   "duplicate scatter target");
        written.SetSelected(target);
    });
    // Work on a private candidate: failures in value access/allocation cannot
    // publish partial values or coverage, nor modify another buffer owner.
    auto candidate = MakeWritableCopy(ReadOnlyVectorView::CopyOf(
        destination.values, destination.defined_rows.view()));
    DispatchScalar(source.logical_type(), [&]<typename T>() {
        rows.ForEachSelected([&](size_t i) {
            const auto target = mapping.indices()[i];
            if (!source.IsValid(i)) {
                candidate.values->nullAt(target);
            } else {
                if constexpr (std::is_same_v<T, bool>) {
                    if (candidate.values->IsBitmap()) {
                        TargetBitmapView bits(candidate.values->GetRawData(),
                                              candidate.values->size());
                        bits.set(target, source.ValueAt<bool>(i));
                    } else {
                        candidate.values->SetValueAt<T>(target,
                                                        source.ValueAt<T>(i));
                    }
                } else {
                    candidate.values->SetValueAt<T>(target,
                                                    source.ValueAt<T>(i));
                }
                candidate.values->clearNullAt(target);
            }
        });
    });
    candidate.defined_rows.Union(written.view());
    destination = std::move(candidate);
}

namespace {
RowVectorPtr
SelectRowsImpl(const RowVector& source,
               RowMappingPtr mapping,
               SelectionView parent_valid) {
    AssertInfo(mapping, "row mapping cannot be null");
    TargetBitmap validity(mapping->indices().size(), true);
    for (size_t i = 0; i < mapping->indices().size(); ++i) {
        const auto row = mapping->indices()[i];
        AssertInfo(row >= 0 && row < source.size(),
                   "row mapping index out of bounds");
        validity.set(i, parent_valid.IsSelected(i) && source.IsValid(row));
    }
    auto requested = OwnedSelection::None(source.size());
    for (size_t i = 0; i < mapping->indices().size(); ++i) {
        if (validity[i]) {
            requested.SetSelected(mapping->indices()[i]);
        }
    }
    std::vector<VectorPtr> children;
    for (const auto& child : source.childrens()) {
        AssertInfo(child->size() == source.size(),
                   "input child length mismatch");
        if (const auto* nested = dynamic_cast<const RowVector*>(child.get())) {
            children.push_back(SelectRowsImpl(
                *nested, mapping, SelectionView::FromBitmap(validity)));
        } else {
            // Snapshot only requested source slots; copying unselected payloads
            // can read undefined values or trigger unnecessary I/O/errors.
            children.push_back(
                ReadOnlyVectorView::CopyOf(child, requested.view())
                    .Dictionary(mapping)
                    .AsVector());
        }
    }
    return std::make_shared<RowVector>(source.row_type(),
                                       mapping->indices().size(),
                                       std::move(children),
                                       std::move(validity));
}
}  // namespace

RowVectorPtr
SelectRows(const RowVector& source, RowMappingPtr mapping) {
    AssertInfo(mapping, "row mapping cannot be null");
    return SelectRowsImpl(
        source, mapping, SelectionView::All(mapping->indices().size()));
}

RowVectorPtr
SliceRows(const RowVector& source, size_t begin, size_t count) {
    AssertInfo(begin <= static_cast<size_t>(source.size()) &&
                   count <= source.size() - begin,
               "row slice out of bounds");
    AssertInfo(count == 0 || begin + count - 1 <=
                                 static_cast<size_t>(
                                     std::numeric_limits<vector_size_t>::max()),
               "row mapping exceeds index range");
    std::vector<vector_size_t> indices(count);
    std::iota(indices.begin(), indices.end(), begin);
    return SelectRows(source, std::make_shared<RowMapping>(std::move(indices)));
}

namespace {
RowVectorPtr
CopyRow(const RowVector& source, SelectionView parent_valid) {
    TargetBitmap validity(source.size(), false);
    parent_valid.ForEachSelected(
        [&](size_t i) { validity.set(i, source.IsValid(i)); });
    auto rows = SelectionView::FromBitmap(validity);
    std::vector<VectorPtr> children;
    for (const auto& child : source.childrens()) {
        AssertInfo(child->size() == source.size(),
                   "input child length mismatch");
        if (const auto* nested = dynamic_cast<const RowVector*>(child.get())) {
            children.push_back(CopyRow(*nested, rows));
        } else {
            children.push_back(
                MakeWritableCopy(ReadOnlyVectorView::CopyOf(child, rows))
                    .values);
        }
    }
    return std::make_shared<RowVector>(source.row_type(),
                                       source.size(),
                                       std::move(children),
                                       std::move(validity));
}
}  // namespace

RowVectorPtr
MakeWritableCopy(const RowVector& source) {
    return CopyRow(source, SelectionView::All(source.size()));
}

void
EnsureWritable(VectorPtr& vector) {
    AssertInfo(vector, "missing writable vector");
    if (const auto* row = dynamic_cast<const RowVector*>(vector.get())) {
        vector = MakeWritableCopy(*row);
    } else {
        const auto view = ReadOnlyVectorView::CopyOf(vector);
        AssertInfo(
            view.defined_rows().IsAll(),
            "partial vectors require MakeWritableCopy with explicit coverage");
        vector = MakeWritableCopy(view).values;
    }
}

ColumnVectorPtr
VectorPool::Acquire(DataType type, size_t size) {
    AssertInfo(
        size <= static_cast<size_t>(std::numeric_limits<vector_size_t>::max()),
        "scratch vector exceeds size range");
    ColumnVectorPtr vector;
    for (auto it = vectors_.begin(); it != vectors_.end(); ++it) {
        if ((*it)->type() == type) {
            vector = std::move(*it);
            vectors_.erase(it);
            vector->resize(size, false);
            break;
        }
    }
    if (!vector) {
        vector = std::make_shared<ColumnVector>(
            InitScalarFieldDataWithLength(type, size),
            TargetBitmap(size, false),
            size);
    }
    vector->valid_values_.reset();
    vector->null_count_ = size;
    return vector;
}

void
VectorPool::Release(ColumnVectorPtr vector) {
    // Variable-size values may retain arbitrarily large payloads. Keep those
    // out of this small scalar scratch pool; they need byte-budget accounting.
    if (!vector || vector.use_count() != 1 ||
        vector->values_.use_count() != 1 || vector->IsBitmap() ||
        vector->size() > static_cast<int64_t>(max_rows_) ||
        vectors_.size() >= max_vectors_ ||
        !(IsNumericDataType(vector->type()) ||
          vector->type() == DataType::TIMESTAMPTZ)) {
        return;
    }
    vectors_.push_back(std::move(vector));
}

ReadOnlyVectorView
ReadOnlyVectorView::Nulls(DataType type, SelectionView rows) {
    AssertInfo(rows.row_count() <=
                   static_cast<size_t>(std::numeric_limits<int64_t>::max()),
               "NULL vector exceeds logical size range");
    if (type == DataType::NONE || IsOrdinaryVectorDataType(type)) {
        auto state = std::make_shared<State>(nullptr,
                                             OwnedSelection::CopyOf(rows),
                                             type,
                                             ValueEncoding::Constant,
                                             0);
        state->validity.emplace(OwnedSelection::None(rows.row_count()));
        return ReadOnlyVectorView(std::move(state));
    }
    return DispatchScalar(type, [&]<typename T>() {
        VectorPtr values =
            std::make_shared<ConstantVector<T>>(type, rows.row_count(), T{}, 1);
        return ReadOnlyVectorView(
            std::make_shared<State>(std::move(values),
                                    OwnedSelection::CopyOf(rows),
                                    type,
                                    ValueEncoding::Constant,
                                    sizeof(T)));
    });
}

ReadOnlyVectorView
ReadOnlyVectorView::Merge(const ReadOnlyVectorView& patch,
                          SelectionView rows) const {
    AssertInfo(size() == patch.size() && rows.row_count() == size() &&
                   logical_type() == patch.logical_type(),
               "vector patch shape/type mismatch");
    AssertInfo(rows.IsSubsetOf(patch.defined_rows()),
               "undefined vector patch rows");
    if (rows.IsNone()) {
        return *this;
    }
    if (rows.IsAll()) {
        return patch;
    }
    if (defined_rows().IsNone()) {
        // The first selective load can retain the loader's immutable storage
        // directly; later merges build routing metadata without copying values.
        return patch.Restrict(rows);
    }
    auto defined = OwnedSelection::Union(defined_rows(), rows);
    auto state = std::make_shared<State>(nullptr,
                                         std::move(defined),
                                         logical_type(),
                                         ValueEncoding::Segmented,
                                         state_->element_width);
    // Direct row routing avoids a binary search or one allocation per sparse
    // run. Owners are deduplicated and point at leaves, never earlier merges.
    state->all_valid = state_->all_valid && patch.state_->all_valid;
    state->routes.resize(size());
    std::unordered_map<const State*, size_t> sources;
    state->defined.view().ForEachSelected([&](size_t row) {
        const auto& source = rows.IsSelected(row) ? patch : *this;
        auto physical = row;
        const auto* leaf = source.state_->Resolve(physical);
        auto [it, inserted] = sources.emplace(leaf, sources.size());
        if (inserted)
            state->sources.push_back(leaf->shared_from_this());
        state->routes[row] = {it->second, physical};
    });
    return ReadOnlyVectorView(std::move(state));
}

ReadOnlyVectorView
ReadOnlyVectorView::Segments(
    DataType type,
    SelectionView rows,
    const std::vector<std::pair<size_t, ReadOnlyVectorView>>& parts) {
    for (const auto& [begin, part] : parts)
        AssertInfo(part.logical_type() == type,
                   "vector fragment type mismatch");
    if (parts.size() == 1 && parts[0].first == 0 &&
        parts[0].second.size() == rows.row_count())
        return parts[0].second.Restrict(rows);
    auto state = std::make_shared<State>(nullptr,
                                         OwnedSelection::CopyOf(rows),
                                         type,
                                         ValueEncoding::Segmented,
                                         0);
    auto coverage = OwnedSelection::None(rows.row_count());
    state->all_valid = true;
    size_t end = 0;
    for (const auto& [begin, part] : parts) {
        AssertInfo(part.logical_type() == type && begin >= end &&
                       begin <= rows.row_count() &&
                       part.size() <= rows.row_count() - begin,
                   "invalid vector fragments");
        end = begin + part.size();
        state->all_valid = state->all_valid && part.state_->all_valid;
        if (!part.size())
            continue;
        part.defined_rows().ForEachSelected(
            [&](size_t row) { coverage.SetSelected(begin + row); });
        state->parts.push_back({begin, end, 0, part.state_});
    }
    AssertInfo(rows.IsSubsetOf(coverage.view()),
               "vector fragments do not cover requested rows");
    return ReadOnlyVectorView(std::move(state));
}

const void*
ReadOnlyVectorView::TryObjectAddress(size_t row,
                                     bool recursive_array,
                                     bool array_views) const {
    AssertInfo(defined_rows().IsSelected(row),
               "cannot read an undefined object position");
    const auto* leaf = state_->Resolve(row);
    const bool valid =
        leaf->all_valid ||
        (leaf->validity
             ? leaf->validity->view().IsSelected(row)
             : static_cast<SimpleVector*>(leaf->values.get())->ValidAt(row));
    if (!valid)
        return nullptr;
    if (logical_type() == DataType::ARRAY)
        AssertInfo(leaf->recursive_array == recursive_array &&
                       leaf->array_views == array_views,
                   "ARRAY physical representation mismatch");
    const auto offset = leaf->encoding == ValueEncoding::Constant ? 0 : row;
    return leaf->owner ? leaf->BorrowedAddress(offset)
                       : static_cast<SimpleVector*>(leaf->values.get())
                             ->RawValueAt(offset, leaf->element_width);
}

Json
ReadOnlyVectorView::JsonViewAt(size_t row) const& {
    CheckType<Json>();
    AssertInfo(IsValid(row), "cannot read a NULL JSON");
    const auto& json = *static_cast<const Json*>(ValueAddress(row));
    // Every stored Json already guarantees SIMD padding, including borrowed
    // reader payloads. Reconstruct a non-owning descriptor even for owned Json.
    return Json(static_cast<std::string_view>(json));
}

ReadOnlyVectorView
ReadOnlyVectorView::AdoptPrivate(ColumnVectorPtr values, SelectionView rows) {
    AssertInfo(values && values->size() == rows.row_count(),
               "result builder shape mismatch");
    const auto type = values->IsBitmap() ? DataType::BOOL : values->type();
    return DispatchScalar(type, [&]<typename T>() {
        const auto encoding =
            values->IsBitmap() ? ValueEncoding::Bitmap : ValueEncoding::Flat;
        return ReadOnlyVectorView(
            std::make_shared<State>(std::move(values),
                                    OwnedSelection::CopyOf(rows),
                                    type,
                                    encoding,
                                    sizeof(T)));
    });
}

VectorResultBuilder::VectorResultBuilder(DataType type,
                                         size_t size,
                                         bool bitmap) {
    AssertInfo(!bitmap || type == DataType::BOOL,
               "bitmap result must have BOOL type");
    if (bitmap) {
        values_ = std::make_shared<ColumnVector>(TargetBitmap(size, false),
                                                 TargetBitmap(size, false));
    } else {
        values_ = std::make_shared<ColumnVector>(
            InitScalarFieldDataWithLength(type, size),
            TargetBitmap(size, false),
            size);
    }
}

void
VectorResultBuilder::SetNull(size_t row) {
    AssertInfo(values_ && row < static_cast<size_t>(values_->size()),
               "invalid result builder null write");
    values_->nullAt(row);
}

ReadOnlyVectorView
VectorResultBuilder::Finish(SelectionView rows) && {
    return ReadOnlyVectorView::AdoptPrivate(std::move(values_), rows);
}

ReadOnlyVectorView
BitmapResultBuilder::Finish(SelectionView rows) && {
    AssertInfo(!finished_, "bitmap builder already published");
    finished_ = true;
    return ReadOnlyVectorView::AdoptPrivate(
        std::make_shared<ColumnVector>(std::move(values_), std::move(valid_)),
        rows);
}

}  // namespace milvus
