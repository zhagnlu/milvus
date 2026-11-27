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

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <vector>

#include "common/Selection.h"

namespace milvus {

namespace {

void
ExpectSelection(SelectionView rows, const std::vector<bool>& expected) {
    ASSERT_EQ(rows.row_count(), expected.size());
    std::vector<size_t> selected;
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(rows.IsSelected(i), expected[i]) << "row " << i;
        if (expected[i]) {
            selected.push_back(i);
        }
    }
    EXPECT_EQ(rows.selected_count(), selected.size());
    EXPECT_EQ(rows.IsAll(), selected.size() == expected.size());
    EXPECT_EQ(rows.IsNone(), selected.empty());
    std::vector<size_t> actual;
    rows.ForEachSelected([&](size_t i) { actual.push_back(i); });
    EXPECT_EQ(actual, selected);
    std::vector<size_t> by_range;
    size_t last_end = 0;
    bool first = true;
    rows.ForEachSelectedRange([&](size_t begin, size_t count) {
        EXPECT_GT(count, 0);
        if (!first)
            EXPECT_GT(begin, last_end);  // Adjacent runs must be coalesced.
        first = false;
        last_end = begin + count;
        for (size_t i = begin; i < last_end; ++i) by_range.push_back(i);
    });
    EXPECT_EQ(by_range, selected);
}

}  // namespace

TEST(SelectionTest, AllNoneAndZeroDomain) {
    for (size_t n : {0, 1, 7, 8, 63, 64, 65, 129}) {
        auto all = OwnedSelection::All(n);
        auto none = OwnedSelection::None(n);
        EXPECT_EQ(all.view().row_count(), n);
        EXPECT_EQ(none.view().row_count(), n);
        EXPECT_EQ(all.view().selected_count(), n);
        EXPECT_EQ(none.view().selected_count(), 0);
        EXPECT_TRUE(all.view().IsAll());
        EXPECT_TRUE(none.view().IsNone());
        EXPECT_EQ(all.view().IsNone(), n == 0);
        EXPECT_EQ(none.view().IsAll(), n == 0);
        size_t calls = 0;
        none.view().ForEachSelected([&](size_t) { ++calls; });
        EXPECT_EQ(calls, 0);
        all.view().ForEachSelected([&](size_t i) { EXPECT_EQ(i, calls++); });
        EXPECT_EQ(calls, n);
    }
    // All/None must not allocate row_count bits.
    auto huge = OwnedSelection::All(std::numeric_limits<size_t>::max());
    EXPECT_EQ(huge.view().selected_count(), std::numeric_limits<size_t>::max());
    const auto empty = SelectionView::None(huge.view().row_count());
    auto intersection = OwnedSelection::Intersect(huge.view(), huge.view());
    auto difference = OwnedSelection::Subtract(huge.view(), huge.view());
    auto combined = OwnedSelection::Union(huge.view(), empty);
    EXPECT_TRUE(intersection.view().IsAll());
    EXPECT_TRUE(difference.view().IsNone());
    EXPECT_TRUE(combined.view().IsAll());
}

TEST(SelectionTest, BitmapAndUnalignedSlices) {
    CustomBitset bits(140, false);
    for (size_t i = 0; i < bits.size(); ++i) {
        bits.set(i, i % 3 == 1);
    }
    const auto view = SelectionView::FromBitmap(bits);
    for (size_t begin : {0, 1, 7, 8, 63, 64, 65, 140}) {
        for (size_t count = 0; count <= bits.size() - begin; ++count) {
            const auto slice = view.Slice(begin, count);
            size_t selected = 0;
            for (size_t i = 0; i < count; ++i) {
                EXPECT_EQ(slice.IsSelected(i), (begin + i) % 3 == 1);
                selected += (begin + i) % 3 == 1;
            }
            EXPECT_EQ(slice.selected_count(), selected);
        }
    }
    auto nested = view.Slice(3, 80).Slice(7, 65);
    EXPECT_EQ(nested.row_count(), 65);
    for (size_t i = 0; i < 65; ++i) {
        EXPECT_EQ(nested.IsSelected(i), bits[10 + i]);
    }
}

TEST(SelectionTest, CopyDetachesAndPreservesSlots) {
    auto copy = [] {
        CustomBitset bits(8, false);
        bits.set(3);
        bits.set(5);
        auto owned =
            OwnedSelection::CopyOf(SelectionView::FromBitmap(bits).Slice(2, 4));
        bits.reset();
        return owned;
    }();
    std::vector<size_t> actual;
    copy.view().ForEachSelected([&](size_t i) { actual.push_back(i); });
    EXPECT_EQ(actual, (std::vector<size_t>{1, 3}));
    auto second = copy;
    copy = OwnedSelection::None(4);
    EXPECT_EQ(second.view().selected_count(), 2);
    copy = second;
    second = OwnedSelection::None(4);
    EXPECT_EQ(copy.view().selected_count(), 2);
    auto duplicates = OwnedSelection::FromIndices(4, {1, 1, 3});
    EXPECT_EQ(duplicates.view().selected_count(), 2);
}

TEST(SelectionTest, BitmapWordBoundariesAndEnumeration) {
    for (int pattern = 0; pattern < 4; ++pattern) {
        // End exactly at the allocation boundary to catch tail overreads.
        CustomBitset bits(1024, pattern == 1);
        if (pattern >= 2) {
            for (size_t i = 0; i < bits.size(); ++i) {
                bits.set(i, pattern == 2 ? i % 127 == 0 : i % 7 != 0);
            }
        }
        const auto view = SelectionView::FromBitmap(bits);
        for (size_t begin = 0; begin <= 65; ++begin) {
            for (size_t n : {0, 1, 7, 63, 64, 65, 127, 128, 513}) {
                SCOPED_TRACE(::testing::Message()
                             << pattern << ":" << begin << ":" << n);
                std::vector<bool> expected(n);
                for (size_t i = 0; i < n; ++i) {
                    expected[i] = bits[begin + i];
                }
                const auto slice = view.Slice(begin, n);
                ExpectSelection(slice, expected);
                auto copy = OwnedSelection::CopyOf(slice);
                ExpectSelection(copy.view(), expected);
            }
            const auto tail = view.Slice(bits.size() - begin, begin);
            std::vector<bool> expected(begin);
            for (size_t i = 0; i < begin; ++i) {
                expected[i] = bits[bits.size() - begin + i];
            }
            ExpectSelection(tail, expected);
        }
    }
    CustomBitset empty(0, false);
    ExpectSelection(SelectionView::FromBitmap(empty), {});
}

TEST(SelectionTest, LocalWordRunsAndPartialTails) {
    constexpr size_t word_bits = sizeof(CustomBitset::data_type) * 8;
    for (bool dense : {false, true}) {
        for (bool tail_selected : {false, true}) {
            for (size_t offset : {0, 1, 7, 31, 63}) {
                for (size_t tail = 0; tail <= word_bits; ++tail) {
                    SCOPED_TRACE(::testing::Message()
                                 << dense << ":" << tail_selected << ":"
                                 << offset << ":" << tail);
                    const size_t count = 4 * word_bits + tail;
                    // End at the allocation boundary. Prefix bits must not
                    // leak into the rebased, potentially unaligned selection.
                    CustomBitset bits(offset + count, true);
                    std::vector<bool> expected(count);
                    for (size_t row = 0; row < count; ++row) {
                        if (row < word_bits) {
                            expected[row] = true;
                        } else if (row < 2 * word_bits) {
                            expected[row] = false;
                        } else if (row < 3 * word_bits) {
                            expected[row] = dense ? row % word_bits != 3
                                                  : row % word_bits == 3;
                        } else if (row < 4 * word_bits) {
                            expected[row] = dense;
                        } else {
                            expected[row] = tail_selected;
                        }
                        bits.set(offset + row, expected[row]);
                    }
                    const auto rows =
                        SelectionView::FromBitmap(bits).Slice(offset, count);
                    // Verify exact callback positions and order, not merely
                    // the selected count, against an independent row oracle.
                    ExpectSelection(rows, expected);
                }
            }
        }
    }
}

TEST(SelectionTest, SetOperationsOnBorrowedSlicesAndTags) {
    for (size_t n : {0, 1, 63, 64, 65, 127, 129, 513, 1025}) {
        for (size_t offset : {0, 1, 7, 63, 64, 65, 128}) {
            SCOPED_TRACE(::testing::Message() << n << ":" << offset);
            const size_t right_offset = 129 - offset;
            // Opposite values outside the slices expose unmasked edge bits.
            CustomBitset l(offset + n + 1, true);
            CustomBitset r(right_offset + n + 1, false);
            for (size_t i = 0; i < n; ++i) {
                l.set(offset + i, i % 3 == 0);
                r.set(right_offset + i, i % 5 != 0);
            }
            CustomBitset zeros(offset + n, false);
            CustomBitset ones(right_offset + n, true);
            const std::vector<SelectionView> views = {
                SelectionView::All(n),
                SelectionView::None(n),
                SelectionView::FromBitmap(l).Slice(offset, n),
                SelectionView::FromBitmap(r).Slice(right_offset, n),
                SelectionView::FromBitmap(zeros).Slice(offset, n),
                SelectionView::FromBitmap(ones).Slice(right_offset, n)};
            for (const auto left : views) {
                for (const auto right : views) {
                    std::vector<bool> both(n), either(n), difference(n);
                    for (size_t i = 0; i < n; ++i) {
                        const bool a = left.IsSelected(i);
                        const bool b = right.IsSelected(i);
                        both[i] = a && b;
                        either[i] = a || b;
                        difference[i] = a && !b;
                    }
                    auto intersection = OwnedSelection::Intersect(left, right);
                    auto combined = OwnedSelection::Union(left, right);
                    auto remaining = OwnedSelection::Subtract(left, right);
                    ExpectSelection(intersection.view(), both);
                    ExpectSelection(combined.view(), either);
                    ExpectSelection(remaining.view(), difference);

                    // Exercise both tags and materialized all/none inputs on
                    // the mutable path, including unaligned right operands.
                    auto scratch = OwnedSelection::None(0);
                    scratch.Assign(left);
                    scratch.Intersect(right);
                    ExpectSelection(scratch.view(), both);
                    scratch.Assign(left);
                    scratch.Union(right);
                    ExpectSelection(scratch.view(), either);
                    scratch.Assign(left);
                    scratch.Subtract(right);
                    ExpectSelection(scratch.view(), difference);
                }
            }
            // Operations must not mutate either borrowed input.
            EXPECT_TRUE(l[0]);
            EXPECT_FALSE(r[right_offset + n]);
            for (size_t i = 0; i < n; ++i) {
                EXPECT_EQ(l[offset + i], i % 3 == 0);
                EXPECT_EQ(r[right_offset + i], i % 5 != 0);
            }
        }
    }
}

TEST(SelectionTest, SetOperationsAgainstScalarOracle) {
    constexpr size_t n = 6;
    for (unsigned left = 0; left < (1U << n); ++left) {
        for (unsigned right = 0; right < (1U << n); ++right) {
            CustomBitset l(n, false), r(n, false);
            for (size_t i = 0; i < n; ++i) {
                l.set(i, bool((left >> i) & 1));
                r.set(i, bool((right >> i) & 1));
            }
            auto lhs = OwnedSelection::CopyOf(SelectionView::FromBitmap(l));
            auto rhs = OwnedSelection::CopyOf(SelectionView::FromBitmap(r));
            auto both = OwnedSelection::Intersect(lhs.view(), rhs.view());
            auto either = OwnedSelection::Union(lhs.view(), rhs.view());
            auto difference = OwnedSelection::Subtract(lhs.view(), rhs.view());
            for (size_t i = 0; i < n; ++i) {
                EXPECT_EQ(both.view().IsSelected(i),
                          bool(((left & right) >> i) & 1));
                EXPECT_EQ(either.view().IsSelected(i),
                          bool(((left | right) >> i) & 1));
                EXPECT_EQ(difference.view().IsSelected(i),
                          bool(((left & ~right) >> i) & 1));
            }
        }
    }
}

TEST(SelectionTest, InvalidCoordinatesAreInternalErrors) {
    const auto rows = SelectionView::All(4);
    EXPECT_THROW(rows.IsSelected(4), SegcoreError);
    EXPECT_THROW(rows.Slice(5, 0), SegcoreError);
    EXPECT_THROW(rows.Slice(1, std::numeric_limits<size_t>::max()),
                 SegcoreError);
    EXPECT_THROW(OwnedSelection::FromIndices(4, {4}), SegcoreError);
    EXPECT_THROW(OwnedSelection::Union(rows, SelectionView::None(3)),
                 SegcoreError);
    EXPECT_THROW(OwnedSelection::Intersect(rows, SelectionView::All(3)),
                 SegcoreError);
    EXPECT_THROW(OwnedSelection::Subtract(rows, SelectionView::All(3)),
                 SegcoreError);
    EXPECT_TRUE(rows.Slice(4, 0).IsNone());
    try {
        rows.IsSelected(4);
        FAIL();
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), UnexpectedError);
    }
}

TEST(SelectionTest, AssignOwnSliceAndReuseAcrossSizes) {
    CustomBitset source(1024, false);
    for (size_t i = 0; i < source.size(); ++i) {
        source.set(i, i % 7 < 3);
    }
    auto rows = OwnedSelection::None(0);
    for (size_t offset = 0; offset <= 65; ++offset) {
        for (size_t n : {0, 1, 63, 64, 65, 129, 513}) {
            SCOPED_TRACE(::testing::Message() << offset << ":" << n);
            rows.Assign(SelectionView::FromBitmap(source));
            std::vector<bool> expected(n);
            for (size_t i = 0; i < n; ++i) {
                expected[i] = source[offset + i];
            }
            rows.Assign(rows.view().Slice(offset, n));
            ExpectSelection(rows.view(), expected);
            rows.Assign(rows.view());
            ExpectSelection(rows.view(), expected);
            // Assign an external unaligned view into previously used storage.
            rows.Assign(SelectionView::FromBitmap(source).Slice(offset, n));
            ExpectSelection(rows.view(), expected);
        }
    }
    for (size_t i = 0; i < source.size(); ++i) {
        EXPECT_EQ(source[i], i % 7 < 3);
    }
}

TEST(SelectionTest, ResetAndMaterializeDoNotExposeOldBits) {
    auto rows = OwnedSelection::None(0);
    for (size_t n : {129, 1, 65, 0, 513, 63, 1025, 7}) {
        for (bool selected : {false, true}) {
            if (selected) {
                rows.ResetAll(n);
            } else {
                rows.ResetNone(n);
            }
            std::vector<bool> expected(n, selected);
            ExpectSelection(rows.view(), expected);
            if (n != 0) {
                // A no-op must not materialize tags; a real change initializes
                // every logical bit, including old capacity and partial words.
                rows.SetSelected(n - 1, selected);
                rows.SetSelected(n / 2, !selected);
                expected[n / 2] = !selected;
                ExpectSelection(rows.view(), expected);
                rows.SetSelected(n / 2, selected);
                ExpectSelection(rows.view(), std::vector<bool>(n, selected));
            }
        }
    }
}

TEST(SelectionTest, InPlaceSelfOperationsAndDetachedCopies) {
    auto rows = OwnedSelection::FromIndices(129, {0, 3, 64, 128});
    auto snapshot = rows;
    rows.Intersect(rows.view());
    rows.Union(rows.view());
    EXPECT_EQ(rows.view().selected_count(), 4);
    rows.SetSelected(3, false);
    EXPECT_TRUE(snapshot.view().IsSelected(3));
    rows.Subtract(rows.view());
    EXPECT_TRUE(rows.view().IsNone());
    EXPECT_EQ(snapshot.view().selected_count(), 4);
    rows = snapshot;
    rows.SetSelected(64, false);
    EXPECT_TRUE(snapshot.view().IsSelected(64));

    auto moved = std::move(rows);
    EXPECT_EQ(moved.view().selected_count(), 3);
    EXPECT_EQ(rows.view().row_count(), 0);
    rows.ResetAll(65);
    rows.SetSelected(64, false);
    EXPECT_EQ(rows.view().selected_count(), 64);
    moved = std::move(rows);
    EXPECT_EQ(moved.view().selected_count(), 64);
    EXPECT_EQ(rows.view().row_count(), 0);
    rows.Assign(snapshot.view());
    EXPECT_EQ(rows.view().selected_count(), 4);
}

TEST(SelectionTest, InPlaceErrorsPreserveSelection) {
    auto rows = OwnedSelection::FromIndices(65, {3, 64});
    EXPECT_THROW(rows.SetSelected(65), SegcoreError);
    EXPECT_THROW(rows.Intersect(SelectionView::None(64)), SegcoreError);
    EXPECT_THROW(rows.Union(SelectionView::All(66)), SegcoreError);
    EXPECT_THROW(rows.Subtract(rows.view().Slice(1, 64)), SegcoreError);
    EXPECT_EQ(rows.view().selected_count(), 2);
    EXPECT_TRUE(rows.view().IsSelected(3));
    EXPECT_TRUE(rows.view().IsSelected(64));
}

TEST(SelectionTest, InPlaceTagsDoNotMaterializeHugeDomains) {
    const auto n = std::numeric_limits<size_t>::max();
    auto rows = OwnedSelection::All(n);
    rows.SetSelected(n - 1);
    rows.Intersect(SelectionView::All(n));
    rows.Subtract(SelectionView::None(n));
    EXPECT_EQ(rows.view().selected_count(), n);
    EXPECT_THROW(rows.SetSelected(0, false), SegcoreError);
    EXPECT_TRUE(rows.view().IsAll());
    rows.Intersect(SelectionView::None(n));
    rows.SetSelected(n - 1, false);
    EXPECT_TRUE(rows.view().IsNone());
    rows.Union(SelectionView::All(n));
    EXPECT_TRUE(rows.view().IsAll());
    rows.Subtract(rows.view());
    EXPECT_TRUE(rows.view().IsNone());
}

TEST(SelectionTest, CopyAndTransferBitmapsPreserveUnalignedRangesAndAliases) {
    for (size_t n : {size_t(0),
                     size_t(1),
                     size_t(63),
                     size_t(64),
                     size_t(65),
                     size_t(259)}) {
        CustomBitset source(n + 19, false);
        for (size_t j = 0; j < source.size(); ++j) source.set(j, j % 7 < 3);
        auto selection = SelectionView::FromBitmap(source).Slice(3, n);
        auto copy = selection.ToBitmap();
        for (size_t j = 0; j < n; ++j) EXPECT_EQ(copy[j], source[j + 3]);
        CustomBitset destination(n + 17, true);
        selection.CopyTo(destination, 9);
        for (size_t j = 0; j < destination.size(); ++j)
            EXPECT_EQ(destination[j],
                      j < 9 || j >= n + 9 ? true : source[j - 6]);
        auto before = source.clone();
        selection.CopyTo(source, 7);  // Source and destination overlap.
        for (size_t j = 0; j < source.size(); ++j)
            EXPECT_EQ(source[j],
                      j < 7 || j >= n + 7 ? before[j] : before[j - 4]);
        auto owned = OwnedSelection::OwnBitmap(std::move(copy));
        auto detached = std::move(owned).TakeBitmap();
        EXPECT_EQ(owned.view().row_count(), 0);
        for (size_t j = 0; j < n; ++j) EXPECT_EQ(detached[j], before[j + 3]);
        SelectionView::None(n).CopyTo(destination, 2);
        for (size_t j = 2; j < n + 2; ++j) EXPECT_FALSE(destination[j]);
        SelectionView::All(n).CopyTo(destination, 2);
        for (size_t j = 2; j < n + 2; ++j) EXPECT_TRUE(destination[j]);
    }
    CustomBitset small(3, false);
    EXPECT_THROW(SelectionView::All(4).CopyTo(small), SegcoreError);
    EXPECT_TRUE(small.none());
}

TEST(SelectionTest, PhysicalValidityMatchesScalarAcrossOffsetsAndTails) {
    for (size_t size : {0, 1, 7, 8, 63, 64, 65, 127, 131, 1024}) {
        for (size_t offset = 0; offset < 16; ++offset) {
            for (int pattern = 0; pattern < 3; ++pattern) {
                auto expanded = std::make_unique<bool[]>(size + offset);
                std::vector<uint8_t> packed((size + offset + 7) / 8, 0);
                for (size_t i = 0; i < size + offset; ++i) {
                    expanded[i] = pattern == 0 || (pattern == 2 && i % 7 != 1);
                    if (expanded[i])
                        packed[i / 8] |= uint8_t{1} << (i % 8);
                }
                for (int selection = 0; selection < 3; ++selection) {
                    auto rows = selection == 0 ? OwnedSelection::All(size)
                                               : OwnedSelection::None(size);
                    if (selection == 2)
                        for (size_t i = 0; i < size; ++i)
                            if (i % 3 != 0)
                                rows.SetSelected(i);
                    for (auto validity :
                         {ValidityView{},
                          ValidityView::FromExpanded(expanded.get())
                              .Subview(offset),
                          ValidityView::FromPacked(packed.data())
                              .Subview(offset)}) {
                        auto actual =
                            OwnedSelection::FromValidity(validity, rows.view());
                        std::vector<bool> expected(size);
                        for (size_t i = 0; i < size; ++i)
                            expected[i] = rows.view().IsSelected(i) &&
                                          (!validity || expanded[i + offset]);
                        ExpectSelection(actual.view(), expected);
                    }
                }
            }
        }
    }
}

TEST(SelectionTest, AllocationFreeComparisonsAndIntersectionRespectSlices) {
    CustomBitset a(333, false), b(333, false), mask(333, false);
    for (size_t i = 0; i < a.size(); ++i) {
        a.set(i, i % 3 == 0);
        b.set(i, i % 5 != 0);
        mask.set(i, i % 7 < 3);
    }
    for (size_t n : {0, 1, 7, 63, 64, 65, 129, 259})
        for (size_t shift : {0, 3, 64}) {
            auto x = SelectionView::FromBitmap(a).Slice(shift, n);
            auto y = SelectionView::FromBitmap(b).Slice(shift == 3 ? 9 : 0, n);
            auto m =
                SelectionView::FromBitmap(mask).Slice(shift == 3 ? 1 : 64, n);
            for (auto left :
                 {x, SelectionView::All(n), SelectionView::None(n)}) {
                for (auto right :
                     {y, SelectionView::All(n), SelectionView::None(n)}) {
                    bool subset = true, equal = true, equal_on = true;
                    for (size_t i = 0; i < n; ++i) {
                        subset &= !left.IsSelected(i) || right.IsSelected(i);
                        equal &= left.IsSelected(i) == right.IsSelected(i);
                        equal_on &= !m.IsSelected(i) ||
                                    left.IsSelected(i) == right.IsSelected(i);
                    }
                    EXPECT_EQ(left.IsSubsetOf(right), subset);
                    EXPECT_EQ(left.Equals(right), equal);
                    EXPECT_EQ(left.EqualsOn(right, m), equal_on);
                }
            }
            auto output = CustomBitset(b.view());
            x.IntersectInto(output, 5);
            for (size_t i = 0; i < output.size(); ++i)
                EXPECT_EQ(output[i],
                          b[i] && (i < 5 || i >= 5 + n || x.IsSelected(i - 5)));
            auto alias = CustomBitset(a.view());
            auto snapshot =
                SelectionView::FromBitmap(alias).Slice(3, n).ToBitmap();
            SelectionView::FromBitmap(alias).Slice(3, n).IntersectInto(alias,
                                                                       9);
            for (size_t i = 0; i < alias.size(); ++i)
                EXPECT_EQ(alias[i],
                          a[i] && (i < 9 || i >= 9 + n || snapshot[i - 9]));
        }
    EXPECT_THROW(SelectionView::All(1).IsSubsetOf(SelectionView::All(2)),
                 SegcoreError);
}

TEST(SelectionTest, BoundedWordReadsRespectSlicesAndUniformSelections) {
    auto rows = OwnedSelection::None(131);
    for (size_t i = 0; i < 131; i += 3) rows.SetSelected(i);
    for (auto source :
         {rows.view(), SelectionView::All(131), SelectionView::None(131)}) {
        auto slice = source.Slice(3, 127);
        for (size_t begin = 0; begin < 127; ++begin) {
            const auto count = std::min<size_t>(64, 127 - begin);
            const auto word = slice.ReadWord(begin, count);
            for (size_t bit = 0; bit < count; ++bit)
                EXPECT_EQ(bool((word >> bit) & 1),
                          source.IsSelected(begin + bit + 3));
        }
    }
    EXPECT_EQ(rows.view().ReadWord(131, 0), 0);
    EXPECT_THROW(rows.view().ReadWord(130, 2), SegcoreError);
    EXPECT_THROW(rows.view().ReadWord(0, 65), SegcoreError);
}

}  // namespace milvus
