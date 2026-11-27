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

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <optional>
#include <utility>

#include "common/CustomBitset.h"
#include "common/EasyAssert.h"
#include "common/ValidityView.h"

namespace milvus {

// Positions are batch-local slots, NOT segment offsets. A borrowed bitmap must
// outlive this view and must not be modified, moved or resized while it is used.
// All/None views have no backing storage. An empty domain is both All and None.
class SelectionView {
 public:
    static SelectionView
    All(size_t row_count) {
        return SelectionView(row_count, true, nullptr, 0);
    }

    static SelectionView
    None(size_t row_count) {
        return SelectionView(row_count, false, nullptr, 0);
    }

    static SelectionView
    FromBitmap(const CustomBitset& bitmap) {
        return SelectionView(bitmap.size(), false, &bitmap, 0);
    }

    static SelectionView
    FromBitmap(const CustomBitset&&) = delete;

    size_t
    row_count() const {
        return row_count_;
    }

    // Representation checks and bounded word reads for kernel scheduling.
    // Unlike IsAll()/selected_count(), these never scan the whole bitmap.
    bool
    HasBitmap() const {
        return bitmap_ != nullptr;
    }

    uint64_t
    ReadWord(size_t begin, size_t count) const {
        AssertInfo(
            count <= 64 && begin <= row_count_ && count <= row_count_ - begin,
            "selection word outside domain");
        return count ? Word(begin, count) : uint64_t{0};
    }

    bool
    IsSelected(size_t row) const {
        AssertInfo(
            row < row_count_, "selection row {} out of {}", row, row_count_);
        return bitmap_ ? (*bitmap_)[offset_ + row] : all_;
    }

    size_t
    selected_count() const {
        if (!bitmap_) {
            return all_ ? row_count_ : 0;
        }
        return BitmapView().count();
    }

    bool
    IsAll() const {
        return bitmap_ ? BitmapView().all() : (all_ || row_count_ == 0);
    }

    bool
    IsNone() const {
        return bitmap_ ? BitmapView().none() : (!all_ || row_count_ == 0);
    }

    // Allocation-free comparisons, including unaligned bitmap slices.
    bool
    IsSubsetOf(SelectionView other) const {
        AssertInfo(row_count_ == other.row_count_, "selection domain mismatch");
        if (!row_count_ || (!bitmap_ && !all_) ||
            (!other.bitmap_ && other.all_) ||
            (bitmap_ && bitmap_ == other.bitmap_ && offset_ == other.offset_))
            return true;
        if (!other.bitmap_)
            return IsNone();
        if (!bitmap_)
            return other.IsAll();
        // Reduce aligned words directly so the compiler can vectorize long
        // masks. Reconstructing checked slices per word penalizes whole-segment
        // index plans even though it avoids allocation for small batches.
        if (offset_ % 64 == 0 && other.offset_ % 64 == 0) {
            const auto* a = bitmap_->data() + offset_ / 64;
            const auto* b = other.bitmap_->data() + other.offset_ / 64;
            uint64_t difference = 0;
            const size_t words = row_count_ / 64;
            for (size_t i = 0; i < words; ++i) difference |= a[i] & ~b[i];
            const auto tail = row_count_ % 64;
            if (tail)
                difference |=
                    (a[words] & ~b[words]) & (UINT64_MAX >> (64 - tail));
            return difference == 0;
        }
        for (size_t base = 0; base < row_count_; base += 64) {
            const auto count = std::min<size_t>(64, row_count_ - base);
            if (Word(base, count) & ~other.Word(base, count))
                return false;
        }
        return true;
    }

    bool
    EqualsOn(SelectionView other, SelectionView rows) const {
        AssertInfo(
            row_count_ == other.row_count_ && row_count_ == rows.row_count_,
            "selection domain mismatch");
        if ((!bitmap_ && !other.bitmap_ && all_ == other.all_) ||
            (bitmap_ && bitmap_ == other.bitmap_ && offset_ == other.offset_))
            return true;
        if (!rows.bitmap_ && !rows.all_)
            return true;
        if (!bitmap_ && !other.bitmap_)
            return rows.IsNone();
        if (bitmap_ && other.bitmap_ && offset_ % 64 == 0 &&
            other.offset_ % 64 == 0 &&
            (!rows.bitmap_ || rows.offset_ % 64 == 0)) {
            const auto* a = bitmap_->data() + offset_ / 64;
            const auto* b = other.bitmap_->data() + other.offset_ / 64;
            const size_t words = row_count_ / 64;
            uint64_t difference = 0;
            if (rows.bitmap_) {
                const auto* mask = rows.bitmap_->data() + rows.offset_ / 64;
                for (size_t i = 0; i < words; ++i)
                    difference |= (a[i] ^ b[i]) & mask[i];
            } else {
                for (size_t i = 0; i < words; ++i) difference |= a[i] ^ b[i];
            }
            if (const auto tail = row_count_ % 64)
                difference |=
                    (Word(words * 64, tail) ^ other.Word(words * 64, tail)) &
                    rows.Word(words * 64, tail);
            return difference == 0;
        }
        for (size_t base = 0; base < row_count_; base += 64) {
            const auto count = std::min<size_t>(64, row_count_ - base);
            if ((Word(base, count) ^ other.Word(base, count)) &
                rows.Word(base, count))
                return false;
        }
        return true;
    }

    bool
    Equals(SelectionView other) const {
        return EqualsOn(other, All(row_count_));
    }

    // Rebase [begin, begin + count) onto [0, count), including unaligned bits.
    SelectionView
    Slice(size_t begin, size_t count) const {
        AssertInfo(begin <= row_count_ && count <= row_count_ - begin,
                   "selection slice out of bounds");
        return SelectionView(count, all_, bitmap_, offset_ + begin);
    }

    // Detached bitmap materialization. Never exposes a mutable alias.
    CustomBitset
    ToBitmap() const {
        return bitmap_ ? CustomBitset(BitmapView())
                       : CustomBitset(row_count_, all_);
    }

    // Copy into a caller-owned output range, retaining surrounding bits.
    // Snapshot aliases first, including overlapping unaligned source slices.
    void
    CopyTo(CustomBitset& destination, size_t begin = 0) const {
        AssertInfo(begin <= destination.size() &&
                       row_count_ <= destination.size() - begin,
                   "selection copy outside destination");
        if (!row_count_)
            return;
        if (!bitmap_) {
            destination.set(begin, row_count_, all_);
        } else if (bitmap_ == &destination) {
            auto detached = ToBitmap();
            CustomBitset::policy_type::op_copy(
                detached.data(), 0, destination.data(), begin, row_count_);
        } else {
            CustomBitset::policy_type::op_copy(bitmap_->data(),
                                               offset_,
                                               destination.data(),
                                               begin,
                                               row_count_);
        }
    }

    // AND into caller-owned output without a temporary bitmap. Snapshot aliases
    // because overlapping slices can overwrite bits needed later.
    void
    IntersectInto(CustomBitset& destination, size_t begin = 0) const {
        AssertInfo(begin <= destination.size() &&
                       row_count_ <= destination.size() - begin,
                   "selection intersection outside destination");
        if (!row_count_ || (!bitmap_ && all_))
            return;
        if (!bitmap_) {
            destination.set(begin, row_count_, false);
        } else if (bitmap_ == &destination) {
            const auto detached = ToBitmap();
            destination.view(begin, row_count_) &= detached.view();
        } else {
            destination.view(begin, row_count_) &= BitmapView();
        }
    }

    // Maximal contiguous selected runs, including runs spanning bitmap words.
    // Kernels can validate once and operate on arrays rather than row accessors.
    template <typename Fn>
    void
    ForEachSelectedRange(Fn&& fn) const {
        if (!bitmap_) {
            if (all_ && row_count_)
                fn(size_t{0}, row_count_);
            return;
        }
        const auto bits = BitmapView();
        constexpr size_t width = sizeof(CustomBitset::data_type) * 8;
        size_t run_begin = 0, run_end = 0;
        auto append = [&](size_t begin, size_t end) {
            if (run_end != begin && run_end > run_begin) {
                fn(run_begin, run_end - run_begin);
                run_begin = begin;
            } else if (run_end == run_begin) {
                run_begin = begin;
            }
            run_end = end;
        };
        for (size_t base = 0; base < row_count_; base += width) {
            const size_t count = std::min(width, row_count_ - base);
            uint64_t word = bits.read(base, count);
            size_t offset = 0;
            while (word) {
                const size_t skip = __builtin_ctzll(word);
                offset += skip;
                word >>= skip;
                const size_t length =
                    word == UINT64_MAX ? 64 : __builtin_ctzll(~word);
                append(base + offset, base + offset + length);
                offset += length;
                word = length == 64 ? 0 : word >> length;
            }
        }
        if (run_end > run_begin)
            fn(run_begin, run_end - run_begin);
    }

    template <typename Fn>
    void
    ForEachSelected(Fn&& fn) const {
        if (!bitmap_) {
            if (all_) {
                for (size_t i = 0; i < row_count_; ++i) {
                    fn(i);
                }
            }
            return;
        }
        const auto bits = BitmapView();
        const auto selected = bits.count();
        if (selected == 0) {
            return;
        }
        if (selected == row_count_) {
            for (size_t i = 0; i < row_count_; ++i) {
                fn(i);
            }
            return;
        }
        // Exploit local all/none runs even in a mixed selection. Retain the
        // density heuristic for mixed words only.
        const bool dense = selected > row_count_ / 2;
        constexpr size_t word_bits = sizeof(CustomBitset::data_type) * 8;
        for (size_t base = 0; base < row_count_;) {
            const auto width = std::min(word_bits, row_count_ - base);
            auto word = bits.read(base, width);
            // width is in [1, word_bits], so the shift is never word_bits.
            // read() rebases unaligned slices and masks unused tail bits.
            const auto all_set =
                std::numeric_limits<CustomBitset::data_type>::max() >>
                (word_bits - width);
            if (word == 0) {
                // No callback for this word.
            } else if (word == all_set) {
                for (size_t i = 0; i < width; ++i) {
                    fn(base + i);
                }
            } else if (dense) {
                for (size_t i = 0; i < width; ++i) {
                    if ((word >> i) & 1) {
                        fn(base + i);
                    }
                }
            } else {
                while (word != 0) {
                    fn(base + static_cast<size_t>(__builtin_ctzll(word)));
                    word &= word - 1;
                }
            }
            base += width;
        }
    }

 private:
    friend class OwnedSelection;

    uint64_t
    Word(size_t begin, size_t count) const {
        return bitmap_ ? CustomBitset::policy_type::op_read(
                             bitmap_->data(), offset_ + begin, count)
                       : (all_ ? (UINT64_MAX >> (64 - count)) : uint64_t{0});
    }

    // Keep the mutable bitset view private: SelectionView only exposes reads.
    CustomBitsetView
    BitmapView() const {
        return bitmap_->view(offset_, row_count_);
    }

    SelectionView(size_t row_count,
                  bool all,
                  const CustomBitset* bitmap,
                  size_t offset)
        : row_count_(row_count), all_(all), bitmap_(bitmap), offset_(offset) {
    }

    size_t row_count_;
    bool all_;
    const CustomBitset* bitmap_;
    size_t offset_;
};

// Owned, mutable selection. Copying detaches storage; reads borrow a view.
// Mutation/move/destruction invalidates borrowed views. A view passed to a
// mutator is consumed during that call, including a view of this object's bits.
// All/None need no bitmap; resets retain any previously allocated buffer for
// reuse. Use a const owner when a consumer needs an immutable snapshot.
class OwnedSelection {
 public:
    OwnedSelection(const OwnedSelection& other)
        : OwnedSelection(CopyOf(other.view())) {
    }

    OwnedSelection(OwnedSelection&& other) noexcept
        : row_count_(other.row_count_),
          all_(other.all_),
          uses_bitmap_(other.uses_bitmap_),
          bitmap_(std::move(other.bitmap_)) {
        other.ResetNone(0);
        other.bitmap_.reset();
    }

    OwnedSelection&
    operator=(const OwnedSelection& other) {
        if (this != &other) {
            Assign(other.view());
        }
        return *this;
    }

    OwnedSelection&
    operator=(OwnedSelection&& other) noexcept {
        if (this != &other) {
            bitmap_ = std::move(other.bitmap_);
            row_count_ = other.row_count_;
            all_ = other.all_;
            uses_bitmap_ = other.uses_bitmap_;
            other.ResetNone(0);
            other.bitmap_.reset();
        }
        return *this;
    }

    static OwnedSelection
    All(size_t row_count) {
        return OwnedSelection(row_count, true);
    }

    static OwnedSelection
    None(size_t row_count) {
        return OwnedSelection(row_count, false);
    }

    static OwnedSelection
    CopyOf(SelectionView rows) {
        const auto count = rows.selected_count();
        if (count == 0) {
            return None(rows.row_count());
        }
        if (count == rows.row_count()) {
            return All(rows.row_count());
        }
        auto result = None(rows.row_count());
        result.bitmap_.emplace(rows.BitmapView());
        result.uses_bitmap_ = true;
        return result;
    }

    // Convert physical validity once per span, then mask batch coverage. Read
    // packed bytes explicitly: buffers need not be word aligned or padded.
    static OwnedSelection
    FromValidity(ValidityView validity, SelectionView rows) {
        if (!validity || rows.IsNone())
            return CopyOf(rows);
        CustomBitset bitmap(rows.row_count(), false);
        const auto* expanded = validity.expanded_data();
        for (size_t base = 0; base < rows.row_count(); base += 64) {
            const size_t count = std::min<size_t>(64, rows.row_count() - base);
            uint64_t word = 0;
            if (expanded) {
                if (count == 64) {
                    for (size_t k = 0; k < 64; ++k)
                        word |= uint64_t(expanded[base + k]) << k;
                } else {
                    for (size_t k = 0; k < count; ++k)
                        word |= uint64_t(expanded[base + k]) << k;
                }
            } else {
                const auto offset = validity.bit_offset() + base;
                const auto shift = offset % 8;
                const auto* bytes = validity.packed_data() + offset / 8;
                const auto length = (shift + count + 7) / 8;
                for (size_t k = 0; k < std::min<size_t>(8, length); ++k)
                    word |= uint64_t(bytes[k]) << (8 * k);
                word >>= shift;
                if (length > 8)
                    word |= uint64_t(bytes[8]) << (64 - shift);
                if (count < 64)
                    word &= (uint64_t{1} << count) - 1;
            }
            CustomBitset::policy_type::op_copy(
                &word, 0, bitmap.data(), base, count);
        }
        auto result = OwnBitmap(std::move(bitmap));
        result.Intersect(rows);
        return result;
    }

    // Transfer an owned bitmap into a selection, normalizing All/None once.
    static OwnedSelection
    OwnBitmap(CustomBitset bitmap) {
        return FromBitmap(std::move(bitmap));
    }

    // Consume private storage when publishing a result; no second bitmap copy.
    CustomBitset
    TakeBitmap() && {
        auto result =
            uses_bitmap_ ? std::move(*bitmap_) : CustomBitset(row_count_, all_);
        ResetNone(0);
        bitmap_.reset();
        return result;
    }

    static OwnedSelection
    FromIndices(size_t row_count, std::initializer_list<size_t> indices) {
        if (indices.size() == 0) {
            return None(row_count);
        }
        CustomBitset bitmap(row_count, false);
        for (auto i : indices) {
            AssertInfo(
                i < row_count, "selection row {} out of {}", i, row_count);
            bitmap.set(i);
        }
        return FromBitmap(std::move(bitmap));
    }

    SelectionView
    view() const& {
        return uses_bitmap_ ? SelectionView::FromBitmap(*bitmap_)
                            : (all_ ? SelectionView::All(row_count_)
                                    : SelectionView::None(row_count_));
    }

    SelectionView
    view() const&& = delete;

    // Reuse storage without a count/normalization scan. Also supports assigning
    // a slice of this selection, rebasing it onto [0, rows.row_count()).
    void
    Assign(SelectionView rows) {
        if (!rows.bitmap_ || rows.row_count_ == 0) {
            if (rows.all_) {
                ResetAll(rows.row_count_);
            } else {
                ResetNone(rows.row_count_);
            }
            return;
        }
        if (bitmap_ && rows.bitmap_ == &*bitmap_) {
            // The destination starts at zero, so copying forwards cannot
            // overwrite unread source bits. Read each word before writing it;
            // op_copy requires non-overlapping storage. Shrink only afterwards.
            if (rows.offset_ != 0) {
                const auto source = rows.BitmapView();
                constexpr size_t word_bits =
                    sizeof(CustomBitset::data_type) * 8;
                for (size_t base = 0; base < rows.row_count_;) {
                    const auto width =
                        std::min(word_bits, rows.row_count_ - base);
                    const auto word = source.read(base, width);
                    // Destination words are aligned; unused tail bits are
                    // outside the new domain and need not be preserved.
                    bitmap_->data()[base / word_bits] = word;
                    base += width;
                }
            }
            bitmap_->resize(rows.row_count_);
        } else {
            ResizeBuffer(rows.row_count_);
            CustomBitset::policy_type::op_copy(rows.bitmap_->data(),
                                               rows.offset_,
                                               bitmap_->data(),
                                               0,
                                               rows.row_count_);
        }
        row_count_ = rows.row_count_;
        uses_bitmap_ = true;
    }

    void
    ResetAll(size_t row_count) {
        row_count_ = row_count;
        all_ = true;
        uses_bitmap_ = false;
    }

    void
    ResetNone(size_t row_count) {
        row_count_ = row_count;
        all_ = false;
        uses_bitmap_ = false;
    }

    void
    SetSelected(size_t row, bool selected = true) {
        AssertInfo(
            row < row_count_, "selection row {} out of {}", row, row_count_);
        if (!uses_bitmap_ && all_ == selected) {
            return;
        }
        Materialize();
        bitmap_->set(row, selected);
    }

    // Single-operand overloads mutate in place. Two-operand static overloads
    // below retain their detached-result semantics. No normalization scans are
    // needed on the mutable path; the view computes All/None/count on demand.
    void
    Intersect(SelectionView rows) {
        CheckDomains(view(), rows);
        if (row_count_ == 0 || (!uses_bitmap_ && !all_) ||
            (!rows.bitmap_ && rows.all_)) {
            return;
        }
        if (!rows.bitmap_) {
            ResetNone(row_count_);
        } else if (!uses_bitmap_) {
            Assign(rows);
        } else if (rows.bitmap_ != &*bitmap_) {
            *bitmap_ &= rows.BitmapView();
        }
    }

    void
    Union(SelectionView rows) {
        CheckDomains(view(), rows);
        if (row_count_ == 0 || (!uses_bitmap_ && all_) ||
            (!rows.bitmap_ && !rows.all_)) {
            return;
        }
        if (!rows.bitmap_) {
            ResetAll(row_count_);
        } else if (!uses_bitmap_) {
            Assign(rows);
        } else if (rows.bitmap_ != &*bitmap_) {
            *bitmap_ |= rows.BitmapView();
        }
    }

    void
    Subtract(SelectionView rows) {
        CheckDomains(view(), rows);
        if (row_count_ == 0 || (!uses_bitmap_ && !all_) ||
            (!rows.bitmap_ && !rows.all_)) {
            return;
        }
        if (!rows.bitmap_ || (bitmap_ && rows.bitmap_ == &*bitmap_)) {
            ResetNone(row_count_);
            return;
        }
        Materialize();
        *bitmap_ -= rows.BitmapView();
    }

    static OwnedSelection
    Intersect(SelectionView left, SelectionView right) {
        CheckDomains(left, right);
        // Inspect storage-free All/None tags without scanning either bitmap.
        if (left.row_count_ == 0 || (!left.bitmap_ && !left.all_) ||
            (!right.bitmap_ && !right.all_)) {
            return None(left.row_count());
        }
        if (!left.bitmap_) {
            return CopyOf(right);
        }
        if (!right.bitmap_) {
            return CopyOf(left);
        }
        return Combine(left, right, [](auto& l, const auto& r) { l &= r; });
    }

    static OwnedSelection
    Union(SelectionView left, SelectionView right) {
        CheckDomains(left, right);
        if (left.row_count_ == 0 || (!left.bitmap_ && left.all_) ||
            (!right.bitmap_ && right.all_)) {
            return All(left.row_count());
        }
        if (!left.bitmap_) {
            return CopyOf(right);
        }
        if (!right.bitmap_) {
            return CopyOf(left);
        }
        return Combine(left, right, [](auto& l, const auto& r) { l |= r; });
    }

    static OwnedSelection
    Subtract(SelectionView left, SelectionView right) {
        CheckDomains(left, right);
        if (left.row_count_ == 0 || (!left.bitmap_ && !left.all_) ||
            (!right.bitmap_ && right.all_)) {
            return None(left.row_count());
        }
        if (!right.bitmap_) {
            return CopyOf(left);
        }
        return Combine(left, right, [](auto& l, const auto& r) { l -= r; });
    }

 private:
    void
    ResizeBuffer(size_t row_count) {
        constexpr size_t word_bits = sizeof(CustomBitset::data_type) * 8;
        AssertInfo(
            row_count <= std::numeric_limits<size_t>::max() - (word_bits - 1),
            "selection bitmap size overflows word rounding");
        if (bitmap_) {
            bitmap_->resize(row_count);
        } else {
            bitmap_.emplace(row_count);
        }
    }

    void
    Materialize() {
        if (!uses_bitmap_) {
            ResizeBuffer(row_count_);
            bitmap_->set(0, row_count_, all_);
            uses_bitmap_ = true;
        }
    }

    static void
    CheckDomains(SelectionView left, SelectionView right) {
        AssertInfo(left.row_count() == right.row_count(),
                   "selection domains must have equal lengths");
    }

    OwnedSelection(size_t row_count, bool all)
        : row_count_(row_count), all_(all) {
    }

    // Normalize once, then transfer storage instead of copying selected bits.
    static OwnedSelection
    FromBitmap(CustomBitset bitmap) {
        const auto count = bitmap.count();
        if (count == 0) {
            return None(bitmap.size());
        }
        if (count == bitmap.size()) {
            return All(bitmap.size());
        }
        auto result = None(bitmap.size());
        result.bitmap_.emplace(std::move(bitmap));
        result.uses_bitmap_ = true;
        return result;
    }

    template <typename Op>
    static OwnedSelection
    Combine(SelectionView left, SelectionView right, Op op) {
        // The caller handles a storage-free right operand. A storage-free
        // left operand is only possible for All - bitmap.
        CustomBitset bitmap = left.bitmap_
                                  ? CustomBitset(left.BitmapView())
                                  : CustomBitset(left.row_count_, left.all_);
        op(bitmap, right.BitmapView());
        return FromBitmap(std::move(bitmap));
    }

    size_t row_count_;
    bool all_;
    bool uses_bitmap_ = false;
    std::optional<CustomBitset> bitmap_;
};

}  // namespace milvus
