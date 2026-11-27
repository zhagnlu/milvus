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
#include "common/VectorAccess.h"
#include "exec/BatchInput.h"

namespace milvus {
namespace {
RowTypePtr
MakeRowSchema(std::vector<std::string> names, std::vector<DataType> types) {
    return std::make_shared<RowType>(std::move(names), std::move(types));
}

RowMappingPtr
Mapping(std::initializer_list<vector_size_t> rows) {
    return std::make_shared<RowMapping>(std::vector<vector_size_t>(rows));
}

template <typename T>
ReadOnlyVectorView
Values(DataType type, std::vector<T> values) {
    const auto n = values.size();
    auto owner = std::make_shared<const std::vector<T>>(std::move(values));
    return ReadOnlyVectorView::BorrowImmutable(type,
                                               owner->data(),
                                               owner,
                                               SelectionView::All(n),
                                               SelectionView::All(n));
}

class ThrowingData : public FieldData<int64_t> {
 public:
    ThrowingData()
        : FieldData<int64_t>(DataType::INT64, false, FixedVector<int64_t>(3)) {
    }
    void*
    Data() override {
        ThrowInfo(FileReadFailed, "injected payload failure");
    }
};
}  // namespace

TEST(VectorEngineTest, SchemaAndRowNullsSurviveCopiesAndSelection) {
    auto schema =
        MakeRowSchema({"a", "b"}, {DataType::INT64, DataType::VARCHAR});
    auto a = Values<int64_t>(DataType::INT64, {10, 20, 30});
    auto b =
        Values<std::string>(DataType::VARCHAR, {"ten", "twenty", "thirty"});
    RowVector batch(schema, 3, {a.AsVector(), b.AsVector()});
    batch.SetRowValid(1, false);
    EXPECT_EQ(batch.nullCount(), 1);
    EXPECT_EQ(batch.row_type()->column_name(0), "a");
    EXPECT_EQ(batch.child("b"), batch.child(1));
    EXPECT_THROW(batch.child(-1), SegcoreError);
    EXPECT_THROW(batch.child("missing"), SegcoreError);
    EXPECT_THROW(batch.SetRowValid(3, true), SegcoreError);

    auto selected = SelectRows(batch, Mapping({2, 1, 2, 0}));
    EXPECT_EQ(selected->row_type(), schema);
    EXPECT_EQ(selected->size(), 4);
    EXPECT_FALSE(selected->IsValid(1));
    EXPECT_EQ(selected->nullCount(), 1);
    auto text = ReadOnlyVectorView::CopyOf(selected->child("b"));
    EXPECT_EQ(text.ValueAt<std::string>(0), "thirty");
    EXPECT_EQ(text.StringViewAt(0).data(), b.StringViewAt(2).data());
    EXPECT_EQ(text.StringViewAt(2).data(), b.StringViewAt(2).data());
    auto writable = MakeWritableCopy(*selected);
    std::static_pointer_cast<ColumnVector>(writable->child(1))
        ->SetValueAt<std::string>(0, "new");
    EXPECT_EQ(text.ValueAt<std::string>(0), "thirty");
    EXPECT_FALSE(writable->IsValid(1));
    auto empty = std::dynamic_pointer_cast<RowVector>(batch.cloneEmpty(0));
    EXPECT_EQ(empty->row_type(), schema);
    EXPECT_EQ(empty->size(), 0);
    EXPECT_THROW((RowVector(schema, 2, {a.AsVector(), b.AsVector()})),
                 SegcoreError);
    EXPECT_THROW(
        (RowVector(
            MakeRowSchema({"a"}, {DataType::DOUBLE}), 3, {a.AsVector()})),
        SegcoreError);
    EXPECT_THROW(
        (RowVector(
            schema, 3, {a.AsVector(), b.AsVector()}, TargetBitmap(2, true))),
        SegcoreError);
}

TEST(VectorEngineTest, DictionaryCompositionLifetimeCoverageAndStrings) {
    std::weak_ptr<const std::vector<std::string>> weak;
    auto original = [&] {
        auto owner = std::make_shared<const std::vector<std::string>>(
            std::vector<std::string>{"a", std::string(4096, 'b'), "c", "d"});
        weak = owner;
        auto defined = OwnedSelection::FromIndices(4, {0, 1, 3});
        return ReadOnlyVectorView::BorrowImmutable(DataType::VARCHAR,
                                                   owner->data(),
                                                   owner,
                                                   defined.view(),
                                                   SelectionView::All(4));
    }();
    EXPECT_FALSE(weak.expired());
    auto first = original.Dictionary(Mapping({3, 1, 1, 2}));
    EXPECT_EQ(first.encoding(), ValueEncoding::Dictionary);
    EXPECT_EQ(first.StringViewAt(1).data(), original.StringViewAt(1).data());
    EXPECT_FALSE(first.defined_rows().IsSelected(3));
    EXPECT_THROW(first.IsValid(3), SegcoreError);
    auto nested = first.Slice(1, 2).Dictionary(Mapping({1, 0, 1}));
    EXPECT_EQ(nested.size(), 3);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(nested.StringViewAt(i).data(),
                  original.StringViewAt(1).data());
    }
    auto published = nested.AsVector();
    auto round_trip = ReadOnlyVectorView::CopyOf(published);
    EXPECT_EQ(round_trip.StringViewAt(2).data(), nested.StringViewAt(2).data());
    EXPECT_THROW(published->resize(4), SegcoreError);
    EXPECT_THROW(published->prepareForReuse(), SegcoreError);
    EXPECT_THROW(original.Dictionary(Mapping({-1})), SegcoreError);
    EXPECT_THROW(original.Dictionary(Mapping({4})), SegcoreError);
    EXPECT_THROW(original.Slice(3, 2), SegcoreError);
    EXPECT_EQ(original.Slice(4, 0).size(), 0);
    EXPECT_EQ(original.Dictionary(Mapping({})).Dictionary(Mapping({})).size(),
              0);
}

TEST(VectorEngineTest, BitmapAndConstantDictionaryDecoding) {
    auto bits = std::make_shared<ColumnVector>(TargetBitmap(65, true),
                                               TargetBitmap(65, true));
    TargetBitmapView(bits->GetRawData(), 65).reset(63);
    bits->nullAt(64);
    auto mapped =
        ReadOnlyVectorView::CopyOf(bits).Dictionary(Mapping({64, 63, 0, 63}));
    EXPECT_FALSE(mapped.IsValid(0));
    EXPECT_FALSE(mapped.ValueAt<bool>(1));
    EXPECT_TRUE(mapped.ValueAt<bool>(2));
    EXPECT_EQ(mapped.CountNulls(SelectionView::All(4)), 1);
    auto writable = MakeWritableCopy(mapped);
    EXPECT_EQ(writable.values->type(), DataType::BOOL);
    EXPECT_FALSE(writable.values->ValueAt<bool>(1));
    auto constant =
        std::make_shared<ConstantVector<int64_t>>(DataType::INT64, 100, 42);
    auto cm = ReadOnlyVectorView::CopyOf(constant).Slice(60, 10).Dictionary(
        Mapping({9, 0, 9}));
    EXPECT_EQ(cm.ValueAt<int64_t>(2), 42);
    bits->resize(129, false);
    EXPECT_FALSE(bits->ValidAt(128));
    EXPECT_FALSE(TargetBitmapView(bits->GetRawData(), 129)[63]);
    EXPECT_FALSE(mapped.IsValid(0));
    EXPECT_THROW(bits->resize(-1), SegcoreError);
}

TEST(VectorEngineTest, RowResizeDetachesAndFailureDoesNotPartiallyCommit) {
    auto values = std::make_shared<ColumnVector>(DataType::INT64, 3);
    values->SetValueAt<int64_t>(0, 10);
    RowVector first(3, {values});
    RowVector second(3, {values});
    first.SetRowValid(1, false);
    first.resize(5, false);
    EXPECT_EQ(values->size(), 3);
    EXPECT_EQ(second.child(0)->size(), 3);
    EXPECT_EQ(first.child(0)->size(), 5);
    EXPECT_EQ(first.nullCount(), 3);
    EXPECT_FALSE(first.IsValid(4));
    EXPECT_EQ(ReadOnlyVectorView::CopyOf(first.child(0)).ValueAt<int64_t>(0),
              10);
    first.resize(1);
    EXPECT_EQ(first.nullCount(), 0);
    FieldDataPtr bad = std::make_shared<ThrowingData>();
    auto failing = std::make_shared<ColumnVector>(
        std::move(bad), TargetBitmap(3, true), 0);
    RowVector batch(3, {values, failing});
    try {
        batch.resize(5);
        FAIL();
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), FileReadFailed);
    }
    EXPECT_EQ(batch.size(), 3);
    EXPECT_EQ(batch.child(0), values);
    EXPECT_EQ(values->size(), 3);
}

TEST(VectorEngineTest, NullRowsNeverReadChildPayloadIncludingNestedRows) {
    FieldDataPtr bad = std::make_shared<ThrowingData>();
    auto failing = std::make_shared<ColumnVector>(
        std::move(bad), TargetBitmap(3, true), 0);
    auto inner =
        std::make_shared<RowVector>(3, std::vector<VectorPtr>{failing});
    RowVector outer(3, {inner});
    for (size_t i = 0; i < 3; ++i) {
        outer.SetRowValid(i, false);
    }
    auto selected = SelectRows(outer, Mapping({2, 1}));
    EXPECT_EQ(selected->nullCount(), 2);
    auto nested = std::dynamic_pointer_cast<RowVector>(selected->child(0));
    EXPECT_EQ(nested->nullCount(), 2);
    auto copy = MakeWritableCopy(outer);
    EXPECT_EQ(copy->nullCount(), 3);
    RowVector scalar(3, {failing});
    for (size_t i = 0; i < 3; ++i) {
        scalar.SetRowValid(i, false);
    }
    exec::BatchInput input(scalar);
    EXPECT_EQ(input.GetColumn(0, SelectionView::All(3))
                  .CountNulls(SelectionView::All(3)),
              3);
}

TEST(VectorEngineTest, ScatterPreservesUnselectedRowsAndIsTransactional) {
    auto initial = Values<int64_t>(DataType::INT64, {10, 20, 30, 40});
    auto destination = MakeWritableCopy(initial);
    auto source = Values<int64_t>(DataType::INT64, {100, 200, 300});
    auto rows = OwnedSelection::FromIndices(3, {0, 2});
    CopySelected(source, rows.view(), RowMapping({3, -1, 1}), destination);
    auto result = ReadOnlyVectorView::CopyOf(destination.values);
    EXPECT_EQ(result.ValueAt<int64_t>(0), 10);
    EXPECT_EQ(result.ValueAt<int64_t>(1), 300);
    EXPECT_EQ(result.ValueAt<int64_t>(2), 30);
    EXPECT_EQ(result.ValueAt<int64_t>(3), 100);
    auto old = destination.values;
    EXPECT_THROW(
        CopySelected(source, rows.view(), RowMapping({1, 0, 1}), destination),
        SegcoreError);
    EXPECT_EQ(destination.values, old);
    EXPECT_THROW(
        CopySelected(source, rows.view(), RowMapping({1, 0, 9}), destination),
        SegcoreError);
    EXPECT_EQ(destination.values, old);
    auto nulls =
        ReadOnlyVectorView::Nulls(DataType::INT64, SelectionView::All(1));
    CopySelected(nulls, SelectionView::All(1), RowMapping({0}), destination);
    EXPECT_FALSE(destination.values->ValidAt(0));
    EXPECT_TRUE(destination.defined_rows.view().IsAll());
    EXPECT_EQ(result.ValueAt<int64_t>(0), 10);
}

TEST(VectorEngineTest, WritableBoundaryAndScratchReusePreservePublishedValues) {
    VectorPool pool(1, 16);
    auto scratch = pool.Acquire(DataType::INT64, 4);
    EXPECT_EQ(scratch->nullCount(), 4);
    scratch->SetValueAt<int64_t>(0, 77);
    scratch->clearNullAt(0);
    auto snapshot = ReadOnlyVectorView::CopyOf(scratch);
    const auto* object = scratch.get();
    const auto* buffer = scratch->GetRawData();
    pool.Release(std::move(scratch));
    EXPECT_EQ(pool.cached_vectors(), 1);
    auto reused = pool.Acquire(DataType::INT64, 4);
    EXPECT_EQ(reused.get(), object);
    EXPECT_EQ(reused->GetRawData(), buffer);
    EXPECT_EQ(reused->nullCount(), 4);
    reused->SetValueAt<int64_t>(0, 99);
    reused->clearNullAt(0);
    EXPECT_EQ(snapshot.ValueAt<int64_t>(0), 77);
    VectorPtr published = snapshot.AsVector();
    auto retained = published;
    EnsureWritable(published);
    std::dynamic_pointer_cast<ColumnVector>(published)->SetValueAt<int64_t>(0,
                                                                            88);
    EXPECT_EQ(ReadOnlyVectorView::CopyOf(retained).ValueAt<int64_t>(0), 77);
    auto alias = reused;
    pool.Release(std::move(reused));
    EXPECT_EQ(pool.cached_vectors(), 0);
    EXPECT_EQ(alias->ValueAt<int64_t>(0), 99);
    pool.Release(pool.Acquire(DataType::VARCHAR, 4));
    EXPECT_EQ(pool.cached_vectors(), 0);
}

TEST(VectorEngineTest, BatchInputLoadsOnlyMissingRowsAndRetainsOldViews) {
    exec::BatchInput input(MakeRowSchema({"a"}, {DataType::INT64}), 4);
    auto source = Values<int64_t>(DataType::INT64, {10, 20, 30, 40});
    std::vector<std::vector<size_t>> calls;
    input.BindLoader(0, [&](SelectionView rows) {
        calls.emplace_back();
        rows.ForEachSelected([&](size_t i) { calls.back().push_back(i); });
        return source.Restrict(rows);
    });
    const auto empty = input.GetColumn(0, SelectionView::None(4));
    EXPECT_TRUE(empty.defined_rows().IsNone());
    EXPECT_TRUE(calls.empty());
    auto requested = OwnedSelection::FromIndices(4, {1, 3});
    auto first = input.GetColumn(0, requested.view());
    EXPECT_EQ(calls, (std::vector<std::vector<size_t>>{{1, 3}}));
    EXPECT_THROW(first.ValueAt<int64_t>(0), SegcoreError);
    auto second = input.GetColumn(0, SelectionView::All(4));
    EXPECT_EQ(calls, (std::vector<std::vector<size_t>>{{1, 3}, {0, 2}}));
    EXPECT_EQ(second.ValueAt<int64_t>(2), 30);
    EXPECT_EQ(first.ValueAt<int64_t>(1), 20);
    EXPECT_THROW(first.ValueAt<int64_t>(0), SegcoreError);
    input.GetColumn(0, SelectionView::All(4));
    EXPECT_EQ(calls.size(), 2);
    const auto validity = input.GetValidity(0, SelectionView::All(4));
    EXPECT_TRUE(validity.valid.view().IsAll());
}

TEST(VectorEngineTest, ValidityOnlyAndKnownNullsAvoidValueLoading) {
    exec::BatchInput input(MakeRowSchema({"a"}, {DataType::INT64}), 3);
    auto source = Values<int64_t>(DataType::INT64, {10, 20, 30});
    int value_calls = 0, validity_calls = 0;
    input.BindLoader(
        0,
        [&](SelectionView rows) {
            ++value_calls;
            EXPECT_FALSE(rows.IsSelected(1));
            return source.Restrict(rows);
        },
        [&](SelectionView rows) {
            ++validity_calls;
            EXPECT_TRUE(rows.IsAll());
            return exec::ValidityResult{OwnedSelection::All(3),
                                        OwnedSelection::FromIndices(3, {0, 2})};
        });
    auto validity = input.GetValidity(0, SelectionView::All(3));
    EXPECT_FALSE(validity.valid.view().IsSelected(1));
    EXPECT_EQ(value_calls, 0);
    auto rows = OwnedSelection::FromIndices(3, {1});
    auto null = input.GetColumn(0, rows.view());
    EXPECT_FALSE(null.IsValid(1));
    EXPECT_EQ(value_calls, 0);
    auto all = input.GetColumn(0, SelectionView::All(3));
    EXPECT_EQ(value_calls, 1);
    EXPECT_EQ(validity_calls, 1);
    EXPECT_EQ(all.ValueAt<int64_t>(2), 30);
    EXPECT_FALSE(all.IsValid(1));
}

TEST(VectorEngineTest, FailedAndIncompleteLoadsDoNotPublishAndCanRetry) {
    exec::BatchInput input(MakeRowSchema({"a"}, {DataType::INT64}), 3);
    auto source = Values<int64_t>(DataType::INT64, {10, 20, 30});
    auto first_rows = OwnedSelection::FromIndices(3, {0});
    input.BindValues(0, source.Restrict(first_rows.view()));
    int attempt = 0;
    input.BindLoader(0, [&](SelectionView rows) -> ReadOnlyVectorView {
        EXPECT_FALSE(rows.IsSelected(0));
        ++attempt;
        if (attempt == 1) {
            ThrowInfo(FileReadFailed, "injected loader failure");
        }
        if (attempt == 2) {
            return source.Restrict(SelectionView::None(3));
        }
        return source.Restrict(rows);
    });
    try {
        input.GetColumn(0, SelectionView::All(3));
        FAIL();
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), FileReadFailed);
    }
    auto first = input.GetColumn(0, first_rows.view());
    EXPECT_EQ(first.defined_rows().selected_count(), 1);
    EXPECT_EQ(first.ValueAt<int64_t>(0), 10);
    EXPECT_THROW(input.GetColumn(0, SelectionView::All(3)), SegcoreError);
    const auto unchanged = input.GetColumn(0, first_rows.view());
    EXPECT_EQ(unchanged.defined_rows().selected_count(), 1);
    EXPECT_EQ(input.GetColumn(0, SelectionView::All(3)).ValueAt<int64_t>(2),
              30);
    EXPECT_EQ(attempt, 3);
}

TEST(VectorEngineTest, LoaderShapeValidityAndReentrancyChecks) {
    exec::BatchInput wrong(MakeRowSchema({"a"}, {DataType::INT64}), 2);
    wrong.BindLoader(0, [&](SelectionView) {
        return Values<double>(DataType::DOUBLE, {1, 2});
    });
    EXPECT_THROW(wrong.GetColumn(0, SelectionView::All(2)), SegcoreError);
    const auto empty = wrong.GetColumn(0, SelectionView::None(2));
    EXPECT_TRUE(empty.defined_rows().IsNone());
    EXPECT_THROW(wrong.GetColumn(1, SelectionView::All(2)), SegcoreError);
    EXPECT_THROW(wrong.GetColumn(0, SelectionView::All(3)), SegcoreError);
    exec::BatchInput recursive(MakeRowSchema({"a"}, {DataType::INT64}), 2);
    int calls = 0;
    recursive.BindLoader(0, [&](SelectionView rows) {
        if (++calls == 1) {
            return recursive.GetColumn(0, rows);
        }
        return Values<int64_t>(DataType::INT64, {1, 2});
    });
    EXPECT_THROW(recursive.GetColumn(0, SelectionView::All(2)), SegcoreError);
    EXPECT_EQ(recursive.GetColumn(0, SelectionView::All(2)).ValueAt<int64_t>(1),
              2);
    exec::BatchInput invalid(MakeRowSchema({"a"}, {DataType::INT64}), 2);
    invalid.BindLoader(0, {}, [](SelectionView) {
        return exec::ValidityResult{OwnedSelection::None(2),
                                    OwnedSelection::All(2)};
    });
    EXPECT_THROW(invalid.GetValidity(0, SelectionView::All(2)), SegcoreError);
    const auto empty_validity = invalid.GetValidity(0, SelectionView::None(2));
    EXPECT_TRUE(empty_validity.defined.view().IsNone());
}

TEST(VectorEngineTest, EncodedBatchFeedsProjectionAndNextFilter) {
    auto source = Values<int64_t>(DataType::INT64, {10, 20, 30, 40});
    RowVector batch(
        MakeRowSchema({"price"}, {DataType::INT64}), 4, {source.AsVector()});
    auto filtered = SelectRows(batch, Mapping({3, 1, 3}));
    exec::BatchInput input(*filtered);
    auto price = input.GetColumn(0, SelectionView::All(3));
    auto result = std::make_shared<ColumnVector>(DataType::INT64, input.size());
    SelectionView::All(input.size()).ForEachSelected([&](size_t row) {
        result->SetValueAt<int64_t>(row, price.ValueAt<int64_t>(row) * 2);
    });
    RowVector projected(MakeRowSchema({"doubled"}, {DataType::INT64}),
                        3,
                        {ReadOnlyVectorView::CopyOf(result).AsVector()});
    auto next = SelectRows(projected, Mapping({0, 2}));
    result->SetValueAt<int64_t>(0, -1);
    filtered.reset();
    EXPECT_EQ(next->size(), 2);
    auto values = ReadOnlyVectorView::CopyOf(next->child("doubled"));
    EXPECT_EQ(values.ValueAt<int64_t>(0), 80);
    EXPECT_EQ(values.ValueAt<int64_t>(1), 80);
    RowVector no_columns(9, {});
    auto selected = SelectRows(no_columns, Mapping({8, 0, 8}));
    EXPECT_EQ(selected->size(), 3);
    EXPECT_EQ(exec::BatchInput(*selected).size(), 3);
}

TEST(VectorEngineTest, BorrowedOwnersAndJsonSnapshotsKeepPayloadAlive) {
    std::weak_ptr<const std::vector<std::string>> weak;
    VectorPtr published;
    {
        auto owner = std::make_shared<const std::vector<std::string>>(
            std::vector<std::string>{std::string(2048, 'x'), "tail"});
        weak = owner;
        auto view = ReadOnlyVectorView::BorrowImmutable(DataType::TEXT,
                                                        owner->data(),
                                                        owner,
                                                        SelectionView::All(2),
                                                        SelectionView::All(2));
        published = view.Dictionary(Mapping({1, 0})).AsVector();
    }
    EXPECT_FALSE(weak.expired());
    EXPECT_EQ(ReadOnlyVectorView::CopyOf(published).ValueAt<std::string>(0),
              "tail");
    published.reset();
    EXPECT_TRUE(weak.expired());

    auto json_snapshot = [] {
        simdjson::padded_string bytes(std::string_view("{\"v\":42}"));
        FixedVector<Json> values(1);
        values[0] =
            Json(bytes.data(), bytes.size());  // non-owning Json payload
        FieldDataPtr data = std::make_shared<FieldData<Json>>(
            DataType::JSON, false, std::move(values));
        auto vector = std::make_shared<ColumnVector>(
            std::move(data), TargetBitmap(1, true), 0);
        auto copy = ReadOnlyVectorView::CopyOf(vector);
        bytes.data()[5] = '9';
        return copy;
    }();
    auto json = json_snapshot.Dictionary(Mapping({0, 0})).ValueAt<Json>(1);
    EXPECT_EQ(static_cast<std::string_view>(json), "{\"v\":42}");
    auto writable = MakeWritableCopy(json_snapshot);
    EXPECT_EQ(static_cast<std::string_view>(writable.values->ValueAt<Json>(0)),
              "{\"v\":42}");
}

TEST(VectorEngineTest, ReuseAndBitmapCloneRetainSchemaWithoutMutatingOwners) {
    auto bitmap = std::make_shared<ColumnVector>(TargetBitmap(65, true),
                                                 TargetBitmap(65, true));
    RowVector batch(65, {bitmap});
    batch.SetRowValid(0, false);
    auto schema = batch.row_type();
    EXPECT_EQ(schema->column_type(0), DataType::BOOL);
    auto clone = std::dynamic_pointer_cast<RowVector>(batch.cloneEmpty(3));
    EXPECT_EQ(clone->row_type(), schema);
    EXPECT_TRUE(
        std::dynamic_pointer_cast<ColumnVector>(clone->child(0))->IsBitmap());
    batch.prepareForReuse();
    EXPECT_EQ(batch.size(), 0);
    EXPECT_EQ(batch.child(0)->size(), 0);
    EXPECT_EQ(batch.nullCount(), 0);
    EXPECT_EQ(bitmap->size(), 65);
    batch.resize(2, false);
    EXPECT_EQ(batch.nullCount(), 2);
    batch.SetRowValid(0, true);
    EXPECT_EQ(batch.nullCount(), 1);
}

TEST(VectorEngineTest, LoaderErrorsPreserveCodesAndKnownValidityOnRetry) {
    for (auto code : {FileReadFailed, MemAllocateFailed, FollyCancel}) {
        exec::BatchInput input(MakeRowSchema({"a"}, {DataType::INT64}), 2);
        int calls = 0;
        input.BindLoader(0, [&](SelectionView rows) {
            if (++calls == 1) {
                ThrowInfo(code, "injected source error");
            }
            return Values<int64_t>(DataType::INT64, {10, 20}).Restrict(rows);
        });
        try {
            input.GetColumn(0, SelectionView::All(2));
            FAIL();
        } catch (const SegcoreError& error) {
            EXPECT_EQ(error.get_error_code(), code);
        }
        const auto empty = input.GetColumn(0, SelectionView::None(2));
        EXPECT_TRUE(empty.defined_rows().IsNone());
        EXPECT_EQ(input.GetColumn(0, SelectionView::All(2)).ValueAt<int64_t>(1),
                  20);
    }
    exec::BatchInput input(MakeRowSchema({"a"}, {DataType::INT64}), 2);
    int calls = 0;
    input.BindLoader(
        0,
        [&](SelectionView rows) {
            if (++calls == 1) {
                throw std::bad_alloc();
            }
            auto values = Values<int64_t>(DataType::INT64, {10, 20});
            // On the second attempt the loader wrongly claims the known NULL
            // position is valid. This must not replace the previously known facts.
            return calls == 2 ? values : values.Restrict(rows);
        },
        [](SelectionView) {
            return exec::ValidityResult{OwnedSelection::All(2),
                                        OwnedSelection::FromIndices(2, {0})};
        });
    input.GetValidity(0, SelectionView::All(2));
    EXPECT_THROW(input.GetColumn(0, SelectionView::All(2)), std::bad_alloc);
    EXPECT_THROW(input.GetColumn(0, SelectionView::All(2)), SegcoreError);
    const auto empty = input.GetColumn(0, SelectionView::None(2));
    EXPECT_TRUE(empty.defined_rows().IsNone());
    auto result = input.GetColumn(0, SelectionView::All(2));
    EXPECT_EQ(result.ValueAt<int64_t>(0), 10);
    EXPECT_FALSE(result.IsValid(1));
}

}  // namespace milvus
