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

#include <random>
#include "exec/input/BatchInput.h"

namespace milvus::exec {
namespace {

static_assert(!std::is_copy_constructible_v<BatchInput>);
static_assert(!std::is_move_constructible_v<BatchInput>);

RowTypePtr
InputSchema(std::vector<DataType> types = {DataType::INT64}) {
    std::vector<std::string> names;
    for (size_t i = 0; i < types.size(); ++i)
        names.push_back("c" + std::to_string(i));
    return std::make_shared<RowType>(std::move(names), std::move(types));
}

ReadOnlyVectorView
Numbers(size_t size, int64_t base = 10) {
    auto column = std::make_shared<ColumnVector>(DataType::INT64, size);
    for (size_t i = 0; i < size; ++i) column->SetValueAt<int64_t>(i, base + i);
    return ReadOnlyVectorView::CopyOf(column);
}

std::vector<size_t>
Rows(SelectionView rows) {
    std::vector<size_t> result;
    rows.ForEachSelected([&](size_t row) { result.push_back(row); });
    return result;
}

ValidityResult
Known(size_t size,
      std::initializer_list<size_t> defined,
      std::initializer_list<size_t> valid) {
    return {OwnedSelection::FromIndices(size, defined),
            OwnedSelection::FromIndices(size, valid)};
}

}  // namespace

TEST(BatchInputTest, SeedValuesAndValidityMergeInEitherBindingOrder) {
    for (bool validity_first : {false, true}) {
        BatchInput input(InputSchema(), 4);
        auto values = Numbers(4);
        auto first = OwnedSelection::FromIndices(4, {0});
        if (validity_first)
            input.BindValidity(0, Known(4, {0, 1, 2}, {0, 2}));
        input.BindValues(0, values.Restrict(first.view()));
        if (!validity_first)
            input.BindValidity(0, Known(4, {0, 1, 2}, {0, 2}));
        std::vector<size_t> loaded;
        input.BindLoader(0, [&](SelectionView rows) {
            loaded = Rows(rows);
            return values.Restrict(rows);
        });
        auto result = input.GetColumn(0, SelectionView::All(4));
        EXPECT_EQ(loaded, (std::vector<size_t>{2, 3}));
        EXPECT_EQ(result.ValueAt<int64_t>(0), 10);
        EXPECT_FALSE(result.IsValid(1));
        EXPECT_EQ(result.ValueAt<int64_t>(2), 12);
        const auto validity = input.GetValidity(0, SelectionView::All(4));
        EXPECT_TRUE(validity.defined.view().IsAll());
    }
}

TEST(BatchInputTest, InvalidOrConflictingSeedsAreTransactional) {
    BatchInput input(InputSchema(), 3);
    auto one = OwnedSelection::FromIndices(3, {0});
    input.BindValues(0, Numbers(3).Restrict(one.view()));
    EXPECT_THROW(input.BindValidity(0, Known(3, {1}, {0})), SegcoreError);
    EXPECT_THROW(input.BindValidity(0, Known(2, {0}, {0})), SegcoreError);
    EXPECT_THROW(input.BindValidity(0, Known(3, {0, 1}, {1})), SegcoreError);
    input.BindValidity(0, Known(3, {0, 1, 2}, {0, 2}));
    EXPECT_THROW(input.BindValidity(0, Known(3, {}, {})), SegcoreError);
    auto null = OwnedSelection::FromIndices(3, {1});
    EXPECT_FALSE(input.GetColumn(0, null.view()).IsValid(1));
    EXPECT_EQ(input.GetColumn(0, one.view()).ValueAt<int64_t>(0), 10);

    BatchInput reversed(InputSchema(), 3);
    reversed.BindValidity(0, Known(3, {0}, {}));
    EXPECT_THROW(reversed.BindValues(0, Numbers(3)), SegcoreError);
    reversed.BindValues(0,
                        ReadOnlyVectorView::Nulls(DataType::INT64, one.view()));
    EXPECT_FALSE(reversed.GetColumn(0, one.view()).IsValid(0));
}

TEST(BatchInputTest, FirstNonemptyRequestFreezesEverySlotEvenOnCacheHit) {
    for (bool validity : {false, true}) {
        BatchInput input(InputSchema({DataType::INT64, DataType::INT64}), 2);
        input.BindValues(0, Numbers(2));
        if (validity)
            input.GetValidity(0, SelectionView::All(2));
        else
            input.GetColumn(0, SelectionView::All(2));
        EXPECT_THROW(input.BindValues(1, Numbers(2)), SegcoreError);
        EXPECT_THROW(input.BindValidity(1, Known(2, {0}, {0})), SegcoreError);
        EXPECT_THROW(
            input.BindLoader(
                1, [](auto rows) { return Numbers(2).Restrict(rows); }),
            SegcoreError);
    }
}

TEST(BatchInputTest, FailedEvaluationDoesNotPermitChangingSources) {
    BatchInput input(InputSchema(), 2);
    try {
        input.GetColumn(0, SelectionView::All(2));
        FAIL();
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), UnexpectedError);
    }
    EXPECT_THROW(input.BindValues(0, Numbers(2)), SegcoreError);
    EXPECT_THROW(input.BindLoader(
                     0, [](auto rows) { return Numbers(2).Restrict(rows); }),
                 SegcoreError);
}

TEST(BatchInputTest, EmptyAndInvalidRequestsDoNotFreezeOrRead) {
    BatchInput input(InputSchema(), 3);
    EXPECT_THROW(input.GetColumn(1, SelectionView::All(3)), SegcoreError);
    EXPECT_THROW(input.GetValidity(0, SelectionView::All(2)), SegcoreError);
    const auto initial_values = input.GetColumn(0, SelectionView::None(3));
    const auto initial_validity = input.GetValidity(0, SelectionView::None(3));
    EXPECT_TRUE(initial_values.defined_rows().IsNone());
    EXPECT_TRUE(initial_validity.defined.view().IsNone());
    int loads = 0;
    input.BindLoader(0, [&](auto rows) {
        ++loads;
        return Numbers(3).Restrict(rows);
    });
    input.GetColumn(0, SelectionView::All(3));
    EXPECT_EQ(loads, 1);
    const auto empty = input.GetColumn(0, SelectionView::None(3));
    EXPECT_TRUE(empty.defined_rows().IsNone());
    EXPECT_THROW(empty.IsValid(0), SegcoreError);
    auto validity = input.GetValidity(0, SelectionView::None(3));
    EXPECT_TRUE(validity.defined.view().IsNone());
    EXPECT_TRUE(validity.valid.view().IsNone());
    EXPECT_EQ(loads, 1);
}

TEST(BatchInputTest, ValueFallbackLoadsOnlyMissingValidityNotKnownPayloads) {
    BatchInput input(InputSchema(), 4);
    input.BindValidity(0, Known(4, {0, 1}, {0}));
    std::vector<std::vector<size_t>> requests;
    input.BindLoader(0, [&](auto rows) {
        requests.push_back(Rows(rows));
        return Numbers(4).Restrict(rows);
    });
    const auto validity = input.GetValidity(0, SelectionView::All(4));
    EXPECT_EQ(requests, (std::vector<std::vector<size_t>>{{2, 3}}));
    EXPECT_FALSE(validity.valid.view().IsSelected(1));
    const auto values = input.GetColumn(0, SelectionView::All(4));
    EXPECT_EQ(requests, (std::vector<std::vector<size_t>>{{2, 3}, {0}}));
    EXPECT_EQ(values.ValueAt<int64_t>(0), 10);
    EXPECT_FALSE(values.IsValid(1));
}

TEST(BatchInputTest, ExtraCoverageNeverReplacesPreviouslyPublishedPayloads) {
    BatchInput input(InputSchema(), 4);
    const auto zero = OwnedSelection::FromIndices(4, {0});
    input.BindValues(0, Numbers(4, 100).Restrict(zero.view()));
    int calls = 0;
    input.BindLoader(0, [&](auto rows) {
        ++calls;
        EXPECT_EQ(Rows(rows), (std::vector<size_t>{1}));
        return Numbers(4,
                       200);  // extra coverage includes the already cached row
    });
    auto old = input.GetColumn(0, zero.view());
    const auto one = OwnedSelection::FromIndices(4, {1});
    input.GetColumn(0, one.view());
    auto all = input.GetColumn(0, SelectionView::All(4));
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(all.ValueAt<int64_t>(0), 100);
    EXPECT_EQ(all.ValueAt<int64_t>(3), 203);
    EXPECT_EQ(old.ValueAt<int64_t>(0), 100);
    EXPECT_THROW(old.IsValid(1), SegcoreError);
}

TEST(BatchInputTest, BadValidityLoadCannotInvalidatePublishedValues) {
    BatchInput input(InputSchema(), 3);
    auto one = OwnedSelection::FromIndices(3, {0});
    input.BindValues(0, Numbers(3).Restrict(one.view()));
    int calls = 0;
    input.BindLoader(0, {}, [&](auto rows) {
        EXPECT_EQ(Rows(rows), (std::vector<size_t>{1, 2}));
        if (++calls == 1)
            return Known(3, {0, 1, 2}, {1, 2});
        return Known(3, {0, 1, 2}, {0, 2});
    });
    EXPECT_THROW(input.GetValidity(0, SelectionView::All(3)), SegcoreError);
    auto old = input.GetValidity(0, one.view());
    EXPECT_EQ(Rows(old.defined.view()), (std::vector<size_t>{0}));
    auto result = input.GetValidity(0, SelectionView::All(3));
    EXPECT_EQ(calls, 2);
    EXPECT_FALSE(result.valid.view().IsSelected(1));
    EXPECT_EQ(input.GetColumn(0, one.view()).ValueAt<int64_t>(0), 10);
    result.valid.SetSelected(0, false);  // returned validity is detached
    const auto retained = input.GetValidity(0, one.view());
    EXPECT_TRUE(retained.valid.view().IsSelected(0));
}

TEST(BatchInputTest, CyclicLoadersFailButCanReadAlreadyPublishedDependencies) {
    BatchInput input(InputSchema({DataType::INT64, DataType::INT64}), 2);
    bool cycle = true;
    input.BindLoader(0, [&](auto rows) { return input.GetColumn(1, rows); });
    input.BindLoader(1, [&](auto rows) {
        if (std::exchange(cycle, false))
            return input.GetColumn(0, rows);
        return Numbers(2).Restrict(rows);
    });
    EXPECT_THROW(input.GetColumn(0, SelectionView::All(2)), SegcoreError);
    EXPECT_EQ(input.GetColumn(0, SelectionView::All(2)).ValueAt<int64_t>(1),
              11);

    BatchInput seeded(InputSchema(), 2);
    auto zero = OwnedSelection::FromIndices(2, {0});
    seeded.BindValues(0, Numbers(2).Restrict(zero.view()));
    seeded.BindLoader(0, [&](auto rows) {
        EXPECT_EQ(seeded.GetColumn(0, zero.view()).ValueAt<int64_t>(0), 10);
        return Numbers(2).Restrict(rows);
    });
    EXPECT_EQ(seeded.GetColumn(0, SelectionView::All(2)).ValueAt<int64_t>(1),
              11);
}

TEST(BatchInputTest, CrossProjectionRecursionDoesNotPublishCoverage) {
    BatchInput input(InputSchema(), 2);
    bool recurse = true;
    input.BindLoader(
        0,
        [&](auto rows) {
            if (std::exchange(recurse, false))
                input.GetValidity(0, rows);
            return Numbers(2).Restrict(rows);
        },
        [&](auto rows) {
            input.GetColumn(0, rows);
            return Known(2, {0, 1}, {0, 1});
        });
    EXPECT_THROW(input.GetColumn(0, SelectionView::All(2)), SegcoreError);
    auto result = input.GetColumn(0, SelectionView::All(2));
    EXPECT_EQ(result.ValueAt<int64_t>(1), 11);
}

TEST(BatchInputTest, SourceErrorsPreserveTypeCodeAndPreviouslyKnownNulls) {
    for (bool request_validity : {false, true}) {
        for (auto code : {FileReadFailed, MemAllocateFailed, FollyCancel}) {
            BatchInput input(InputSchema(), 3);
            input.BindValidity(0, Known(3, {0}, {}));
            int calls = 0;
            input.BindLoader(0, [&](auto rows) -> ReadOnlyVectorView {
                EXPECT_EQ(Rows(rows), (std::vector<size_t>{1, 2}));
                if (++calls == 1)
                    ThrowInfo(code, "injected input source failure");
                if (calls == 2)
                    throw std::bad_alloc();
                return Numbers(3).Restrict(rows);
            });
            auto run = [&] {
                if (request_validity)
                    input.GetValidity(0, SelectionView::All(3));
                else
                    input.GetColumn(0, SelectionView::All(3));
            };
            try {
                run();
                FAIL();
            } catch (const SegcoreError& e) {
                EXPECT_EQ(e.get_error_code(), code);
            }
            EXPECT_THROW(run(), std::bad_alloc);
            auto zero = OwnedSelection::FromIndices(3, {0});
            auto known = input.GetValidity(0, zero.view());
            EXPECT_EQ(Rows(known.defined.view()), (std::vector<size_t>{0}));
            run();
            EXPECT_EQ(calls, 3);
            const auto values = input.GetColumn(0, SelectionView::All(3));
            EXPECT_FALSE(values.IsValid(0));
            EXPECT_EQ(values.ValueAt<int64_t>(2), 12);
        }
    }
}

TEST(BatchInputTest, PartialUpstreamSnapshotHonorsRowNullAndCoverage) {
    auto schema = InputSchema();
    auto two = OwnedSelection::FromIndices(4, {2});
    auto selected = OwnedSelection::FromIndices(4, {1, 2});
    RowVector upstream(schema, 4, {Numbers(4).Restrict(two.view()).AsVector()});
    upstream.SetRowValid(1,
                         false);  // child's undefined payload must not be read
    BatchInput input(schema, 4);
    input.BindColumn(0, upstream, 0, selected.view());
    selected.ResetAll(4);  // bound coverage is owned, not borrowed
    auto needed = OwnedSelection::FromIndices(4, {1, 2});
    auto result = input.GetColumn(0, needed.view());
    EXPECT_FALSE(result.IsValid(1));
    EXPECT_EQ(result.ValueAt<int64_t>(2), 12);
    EXPECT_THROW(result.IsValid(0), SegcoreError);
    EXPECT_THROW(input.GetColumn(0, SelectionView::All(4)), SegcoreError);
    EXPECT_EQ(input.GetColumn(0, two.view()).ValueAt<int64_t>(2), 12);
}

TEST(BatchInputTest, UpstreamSnapshotDetachesMutableValuesAndRowValidity) {
    auto schema = InputSchema();
    auto column = std::make_shared<ColumnVector>(DataType::INT64, 2);
    column->SetValueAt<int64_t>(0, 7);
    column->SetValueAt<int64_t>(1, 8);
    RowVector upstream(schema, 2, {column});
    BatchInput input(schema, 2);
    input.BindColumn(0, upstream, 0, SelectionView::All(2));
    column->SetValueAt<int64_t>(0, 99);
    upstream.SetRowValid(1, false);
    auto result = input.GetColumn(0, SelectionView::All(2));
    EXPECT_EQ(result.ValueAt<int64_t>(0), 7);
    EXPECT_EQ(result.ValueAt<int64_t>(1), 8);
}

TEST(BatchInputTest,
     InvalidUpstreamBindingLeavesSlotAvailableForCorrectBinding) {
    auto schema = InputSchema();
    RowVector upstream(schema, 2, {Numbers(2).AsVector()});
    BatchInput input(schema, 2);
    EXPECT_THROW(input.BindColumn(0, upstream, 0, SelectionView::All(3)),
                 SegcoreError);
    EXPECT_THROW(input.BindColumn(0, upstream, 1, SelectionView::All(2)),
                 SegcoreError);
    auto boolean_schema = InputSchema({DataType::BOOL});
    RowVector boolean(
        boolean_schema,
        2,
        {std::make_shared<ConstantVector<bool>>(DataType::BOOL, 2, true)});
    EXPECT_THROW(input.BindColumn(0, boolean, 0, SelectionView::All(2)),
                 SegcoreError);
    input.BindColumn(0, upstream, 0, SelectionView::All(2));
    EXPECT_EQ(input.GetColumn(0, SelectionView::All(2)).ValueAt<int64_t>(1),
              11);
}

TEST(BatchInputTest,
     DictionaryDuplicatesRetainStringOwnerAfterBatchDestruction) {
    auto owner = std::make_shared<const std::vector<std::string>>(
        std::vector<std::string>{"北京", "上海"});
    std::weak_ptr<const void> weak = owner;
    auto source = ReadOnlyVectorView::BorrowImmutable(DataType::VARCHAR,
                                                      owner->data(),
                                                      owner,
                                                      SelectionView::All(2),
                                                      SelectionView::All(2));
    auto mapped = source.Dictionary(
        std::make_shared<RowMapping>(std::vector<vector_size_t>{1, 0, 1}));
    auto output = [&] {
        BatchInput input(InputSchema({DataType::VARCHAR}), 3);
        input.BindValues(0, mapped);
        auto result = input.GetColumn(0, SelectionView::All(3));
        EXPECT_EQ(result.StringViewAt(0), "上海");
        return result;
    }();
    source =
        ReadOnlyVectorView::Nulls(DataType::VARCHAR, SelectionView::None(2));
    mapped =
        ReadOnlyVectorView::Nulls(DataType::VARCHAR, SelectionView::None(3));
    owner.reset();
    EXPECT_FALSE(weak.expired());
    EXPECT_EQ(output.StringViewAt(1), "北京");
    EXPECT_EQ(output.StringViewAt(2), "上海");
}

TEST(BatchInputTest, AllNullAndZeroRowInputsNeverInvokeValueLoaders) {
    BatchInput input(InputSchema(), 3);
    input.BindValidity(0, Known(3, {0, 1, 2}, {}));
    input.BindLoader(0, [](auto) -> ReadOnlyVectorView {
        ADD_FAILURE() << "known NULLs requested payload";
        return Numbers(3);
    });
    const auto values = input.GetColumn(0, SelectionView::All(3));
    EXPECT_EQ(values.CountNulls(SelectionView::All(3)), 3);

    BatchInput empty(InputSchema(), 0);
    empty.BindLoader(0, [](auto) -> ReadOnlyVectorView {
        ADD_FAILURE() << "zero-row input initialized its loader";
        return Numbers(0);
    });
    const auto empty_values = empty.GetColumn(0, SelectionView::All(0));
    const auto empty_validity = empty.GetValidity(0, SelectionView::All(0));
    EXPECT_TRUE(empty_values.defined_rows().IsNone());
    EXPECT_TRUE(empty_validity.defined.view().IsNone());
}

TEST(BatchInputTest, LoaderCannotBindPreviouslyUnusedSlotDuringEvaluation) {
    BatchInput input(InputSchema({DataType::INT64, DataType::INT64}), 2);
    bool mutate = true;
    input.BindLoader(0, [&](auto rows) {
        if (std::exchange(mutate, false))
            input.BindValues(1, Numbers(2));
        return Numbers(2).Restrict(rows);
    });
    EXPECT_THROW(input.GetColumn(0, SelectionView::All(2)), SegcoreError);
    EXPECT_EQ(input.GetColumn(0, SelectionView::All(2)).ValueAt<int64_t>(0),
              10);
    EXPECT_THROW(input.GetColumn(1, SelectionView::All(2)), SegcoreError);
}

TEST(BatchInputTest, RandomIncrementalLoadsMatchIndependentCoverageOracle) {
    constexpr size_t size = 131;
    BatchInput input(InputSchema(), size);
    std::vector<bool> known_values(size), known_validity(size);
    auto column = std::make_shared<ColumnVector>(DataType::INT64, size);
    for (size_t row = 0; row < size; ++row) {
        if (row % 5 == 0)
            column->nullAt(row);
        else
            column->SetValueAt<int64_t>(row, row * 3);
    }
    const auto source = ReadOnlyVectorView::CopyOf(column);
    std::vector<size_t> expected;
    int value_calls = 0, validity_calls = 0;
    input.BindLoader(
        0,
        [&](auto rows) {
            ++value_calls;
            EXPECT_EQ(Rows(rows), expected);
            return source.Restrict(rows);
        },
        [&](auto rows) {
            ++validity_calls;
            EXPECT_EQ(Rows(rows), expected);
            auto valid = OwnedSelection::None(size);
            rows.ForEachSelected([&](size_t row) {
                if (row % 5)
                    valid.SetSelected(row);
            });
            return ValidityResult{OwnedSelection::CopyOf(rows),
                                  std::move(valid)};
        });
    std::mt19937 random(12345);
    for (int step = 0; step < 200; ++step) {
        auto rows = OwnedSelection::None(size);
        for (size_t row = 0; row < size; ++row)
            if (random() % 11 == 0)
                rows.SetSelected(row);
        bool validity_only = random() % 2;
        expected.clear();
        rows.view().ForEachSelected([&](size_t row) {
            if (validity_only ? !known_validity[row]
                              : (!known_values[row] &&
                                 !(known_validity[row] && row % 5 == 0))) {
                expected.push_back(row);
            }
        });
        auto before = validity_only ? validity_calls : value_calls;
        if (validity_only) {
            auto result = input.GetValidity(0, rows.view());
            rows.view().ForEachSelected([&](size_t row) {
                EXPECT_EQ(result.valid.view().IsSelected(row), row % 5 != 0);
                known_validity[row] = true;
            });
        } else {
            auto result = input.GetColumn(0, rows.view());
            rows.view().ForEachSelected([&](size_t row) {
                EXPECT_EQ(result.IsValid(row), row % 5 != 0);
                if (row % 5)
                    EXPECT_EQ(result.ValueAt<int64_t>(row), row * 3);
                known_values[row] = known_validity[row] = true;
            });
        }
        EXPECT_EQ((validity_only ? validity_calls : value_calls) - before,
                  expected.empty() ? 0 : 1);
    }
}

}  // namespace milvus::exec
