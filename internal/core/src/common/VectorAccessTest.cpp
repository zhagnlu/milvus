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
#include <string>
#include <vector>

#include "common/VectorAccess.h"

namespace milvus {
namespace {

template <typename T>
ColumnVectorPtr
MakeColumn(DataType type, std::initializer_list<T> input) {
    FixedVector<T> values(input.begin(), input.end());
    FieldDataPtr data =
        std::make_shared<FieldData<T>>(type, false, std::move(values));
    return std::make_shared<ColumnVector>(
        std::move(data), TargetBitmap(input.size(), true), 0);
}

template <typename T>
void
CheckScalar(DataType type, T a, T b) {
    auto input = MakeColumn<T>(type, {a, b});
    const ReadOnlyVectorView view = ReadOnlyVectorView::CopyOf(input);
    EXPECT_EQ(view.logical_type(), type);
    EXPECT_EQ(view.encoding(), ValueEncoding::Flat);
    EXPECT_EQ(view.ValueAt<T>(0), a);
    EXPECT_EQ(view.ValueAt<T>(1), b);
    WritableVectorCopy output = MakeWritableCopy(view);
    output.values->SetValueAt<T>(0, b);
    output.values->nullAt(1);
    EXPECT_EQ(view.ValueAt<T>(0), a);
    EXPECT_TRUE(view.IsValid(1));
    EXPECT_EQ(output.values->ValueAt<T>(0), b);
}

// Fault injection at the actual source of payload access, not in the adapter.
class FailingPayload : public FieldData<int64_t> {
 public:
    FailingPayload()
        : FieldData<int64_t>(DataType::INT64, false, FixedVector<int64_t>(4)) {
    }

    void*
    Data() override {
        ThrowInfo(FileReadFailed, "injected payload failure");
    }
};

}  // namespace

TEST(VectorAccessTest, FixedWidthAndOwnedStrings) {
    CheckScalar<bool>(DataType::BOOL, false, true);
    CheckScalar<int8_t>(DataType::INT8, -128, 127);
    CheckScalar<int16_t>(DataType::INT16, -30000, 30000);
    CheckScalar<int32_t>(DataType::INT32, -100000, 100000);
    CheckScalar<int64_t>(DataType::INT64,
                         std::numeric_limits<int64_t>::min(),
                         std::numeric_limits<int64_t>::max());
    CheckScalar<int64_t>(DataType::TIMESTAMPTZ, -1, 123456);
    CheckScalar<float>(DataType::FLOAT, -1.5f, 2.25f);
    CheckScalar<double>(DataType::DOUBLE, -1.5, 2.25);
    CheckScalar<std::string>(DataType::STRING, "北京", std::string(1024, 'x'));
    CheckScalar<std::string>(DataType::VARCHAR, "", "上海");
}

TEST(VectorAccessTest, SnapshotDetachesValuesValidityAndPayload) {
    auto source = MakeColumn<std::string>(DataType::VARCHAR,
                                          {"北京", std::string(4096, 'x')});
    const auto view = ReadOnlyVectorView::CopyOf(source);
    source->SetValueAt<std::string>(0, "changed");
    source->nullAt(1);
    source->resize(0);
    source.reset();
    EXPECT_EQ(view.ValueAt<std::string>(0), "北京");
    EXPECT_EQ(view.ValueAt<std::string>(1), std::string(4096, 'x'));
    auto retained = view;
    EXPECT_EQ(retained.ValueAt<std::string>(0), "北京");
}

TEST(VectorAccessTest, UniqueVectorDoesNotImplyUniqueBuffer) {
    auto shared_data = InitScalarFieldDataWithLength(DataType::INT64, 2);
    auto* raw = static_cast<int64_t*>(shared_data->Data());
    raw[0] = 10;
    raw[1] = 20;
    auto retained_data = shared_data;
    auto source = std::make_shared<ColumnVector>(
        std::move(shared_data), TargetBitmap(2, true), 0);
    ASSERT_EQ(source.use_count(), 1);
    auto view = ReadOnlyVectorView::CopyOf(source);
    static_cast<int64_t*>(retained_data->Data())[0] = 99;
    source.reset();
    retained_data.reset();
    EXPECT_EQ(view.ValueAt<int64_t>(0), 10);
    EXPECT_EQ(view.ValueAt<int64_t>(1), 20);
}

TEST(VectorAccessTest, PartialValuesAreNotNullOrCompacted) {
    auto input = MakeColumn<int64_t>(DataType::INT64, {10, 20, 30, 40});
    input->nullAt(3);
    auto rows = OwnedSelection::FromIndices(4, {1, 3});
    const auto first = ReadOnlyVectorView::CopyOf(input, rows.view());
    rows.SetSelected(1, false);
    rows.SetSelected(0);
    rows.ResetNone(4);
    EXPECT_EQ(first.size(), 4);
    EXPECT_EQ(first.defined_rows().selected_count(), 2);
    EXPECT_EQ(first.ValueAt<int64_t>(1), 20);
    EXPECT_FALSE(first.IsValid(3));
    EXPECT_THROW(first.ValueAt<int64_t>(3), SegcoreError);
    EXPECT_THROW(first.IsValid(0), SegcoreError);
    EXPECT_THROW(first.ValueAt<int64_t>(2), SegcoreError);
    EXPECT_THROW(first.IsValid(4), SegcoreError);
    EXPECT_EQ(first.CountNulls(first.defined_rows()), 1);
    EXPECT_EQ(first.CountNulls(SelectionView::None(4)), 0);
    EXPECT_THROW(first.CountNulls(SelectionView::All(4)), SegcoreError);
    EXPECT_THROW(first.CountNulls(SelectionView::None(3)), SegcoreError);

    auto output = MakeWritableCopy(first);
    EXPECT_EQ(output.values->size(), 4);
    EXPECT_FALSE(output.defined_rows.view().IsSelected(0));
    output.defined_rows.SetSelected(0);
    EXPECT_FALSE(first.defined_rows().IsSelected(0));
    output.values->SetValueAt<int64_t>(1, 99);
    EXPECT_EQ(first.ValueAt<int64_t>(1), 20);
    // A later, independent snapshot must not invalidate an earlier result.
    auto missing = OwnedSelection::FromIndices(4, {0, 2});
    const auto later = ReadOnlyVectorView::CopyOf(input, missing.view());
    input.reset();
    EXPECT_EQ(later.ValueAt<int64_t>(2), 30);
    EXPECT_EQ(first.ValueAt<int64_t>(1), 20);
}

TEST(VectorAccessTest, BitmapLogicalBoolAndBitBoundaries) {
    for (size_t n : {0, 1, 7, 8, 63, 64, 65, 129}) {
        TargetBitmap bits(n, false), valid(n, true);
        for (size_t i = 0; i < n; ++i) {
            bits.set(i, i % 3 == 0);
            valid.set(i, i % 5 != 0);
        }
        auto source =
            std::make_shared<ColumnVector>(std::move(bits), std::move(valid));
        auto view = ReadOnlyVectorView::CopyOf(source);
        EXPECT_EQ(source->type(), DataType::INT8);  // legacy contract unchanged
        EXPECT_EQ(view.logical_type(), DataType::BOOL);
        EXPECT_EQ(view.encoding(), ValueEncoding::Bitmap);
        size_t nulls = 0;
        for (size_t i = 0; i < n; ++i) {
            EXPECT_EQ(view.IsValid(i), i % 5 != 0);
            if (i % 5 != 0) {
                EXPECT_EQ(view.ValueAt<bool>(i), i % 3 == 0);
            } else {
                ++nulls;
            }
        }
        EXPECT_EQ(view.CountNulls(SelectionView::All(n)), nulls);
        auto output = MakeWritableCopy(view);
        EXPECT_TRUE(output.values->IsBitmap());
        const auto copied = ReadOnlyVectorView::CopyOf(output.values);
        EXPECT_EQ(copied.CountNulls(SelectionView::All(n)), nulls);
        if (n > 1) {
            TargetBitmapView writable(output.values->GetRawData(), n);
            writable.set(1, true);
            EXPECT_FALSE(view.ValueAt<bool>(1));
            EXPECT_THROW(view.ValueAt<int8_t>(1), SegcoreError);
        }
    }
}

TEST(VectorAccessTest, ConstantsKeepLengthEncodingAndNullness) {
    auto source =
        std::make_shared<ConstantVector<int64_t>>(DataType::INT64, 65, 42);
    auto rows = OwnedSelection::FromIndices(65, {0, 64});
    auto view = ReadOnlyVectorView::CopyOf(source, rows.view());
    *static_cast<int64_t*>(source->RawValueAt(0, sizeof(int64_t))) = 99;
    EXPECT_EQ(view.encoding(), ValueEncoding::Constant);
    EXPECT_EQ(view.size(), 65);
    EXPECT_EQ(view.ValueAt<int64_t>(64), 42);
    EXPECT_THROW(view.IsValid(1), SegcoreError);
    auto output = MakeWritableCopy(view);
    EXPECT_FALSE(output.values->IsBitmap());
    EXPECT_EQ(output.values->ValueAt<int64_t>(64), 42);
    EXPECT_EQ(output.defined_rows.view().selected_count(), 2);

    auto null = std::make_shared<ConstantVector<std::string>>(
        DataType::VARCHAR, 4, "unused", 4);
    auto null_view = ReadOnlyVectorView::CopyOf(null);
    EXPECT_EQ(null_view.CountNulls(SelectionView::All(4)), 4);
    EXPECT_THROW(null_view.ValueAt<std::string>(2), SegcoreError);
    auto null_copy = MakeWritableCopy(null_view);
    EXPECT_EQ(null_copy.values->nullCount(), 4);
    EXPECT_FALSE(null_copy.values->ValidAt(2));

    auto boolean =
        std::make_shared<ConstantVector<bool>>(DataType::BOOL, 4, true);
    EXPECT_TRUE(ReadOnlyVectorView::CopyOf(boolean).ValueAt<bool>(3));
    auto text = std::make_shared<ConstantVector<std::string>>(
        DataType::VARCHAR, 4, "北京");
    EXPECT_EQ(ReadOnlyVectorView::CopyOf(text).ValueAt<std::string>(3), "北京");
}

TEST(VectorAccessTest, EmptyAndNullSelectionsNeverReadPayload) {
    FieldDataPtr data = std::make_shared<FailingPayload>();
    auto source = std::make_shared<ColumnVector>(
        std::move(data), TargetBitmap(4, false), 4);
    auto empty = ReadOnlyVectorView::CopyOf(source, SelectionView::None(4));
    EXPECT_EQ(empty.size(), 4);
    EXPECT_TRUE(empty.defined_rows().IsNone());
    EXPECT_THROW(empty.IsValid(0), SegcoreError);
    auto nulls = ReadOnlyVectorView::CopyOf(source);
    EXPECT_EQ(nulls.CountNulls(SelectionView::All(4)), 4);
    source->clearNullAt(1);
    auto selected = OwnedSelection::FromIndices(4, {1});
    // Source error survives the adapter unchanged; no partial result is returned.
    try {
        ReadOnlyVectorView::CopyOf(source, selected.view());
        FAIL();
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), FileReadFailed);
    }
    EXPECT_EQ(nulls.CountNulls(SelectionView::All(4)), 4);
}

TEST(VectorAccessTest, RejectsInvalidShapesTypesAndUnsupportedSources) {
    EXPECT_THROW(ReadOnlyVectorView::CopyOf(nullptr), SegcoreError);
    auto source = MakeColumn<int32_t>(DataType::INT32, {1, 2});
    EXPECT_THROW(ReadOnlyVectorView::CopyOf(source, SelectionView::All(3)),
                 SegcoreError);
    EXPECT_THROW(ReadOnlyVectorView::CopyOf(source).ValueAt<int64_t>(0),
                 SegcoreError);
    auto bad_constant =
        std::make_shared<ConstantVector<int64_t>>(DataType::INT32, 2, 42);
    EXPECT_THROW(ReadOnlyVectorView::CopyOf(bad_constant), SegcoreError);
    auto malformed = std::make_shared<ColumnVector>(TargetBitmap(4, true),
                                                    TargetBitmap(3, true));
    EXPECT_THROW(ReadOnlyVectorView::CopyOf(malformed), SegcoreError);
    auto row = std::make_shared<RowVector>(2, std::vector<VectorPtr>{source});
    try {
        ReadOnlyVectorView::CopyOf(row);
        FAIL();
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), Unsupported);
    }
    auto zero = MakeColumn<int64_t>(DataType::INT64, {});
    auto zero_view = ReadOnlyVectorView::CopyOf(zero);
    EXPECT_EQ(zero_view.size(), 0);
    EXPECT_TRUE(zero_view.defined_rows().IsNone());
    EXPECT_EQ(MakeWritableCopy(zero_view).values->size(), 0);
}

TEST(VectorAccessTest, BuilderPublishesWithoutCopyAndRejectsReuse) {
    VectorResultBuilder builder(DataType::VARCHAR, 3);
    std::string text(4096, 's');
    const char* payload = text.data();
    builder.Set<std::string>(0, std::move(text));
    auto rows = OwnedSelection::FromIndices(3, {0, 1});
    const auto out = std::move(builder).Finish(rows.view());
    EXPECT_EQ(out.StringViewAt(0).data(), payload);
    EXPECT_FALSE(out.IsValid(1));
    EXPECT_THROW(out.IsValid(2), SegcoreError);
    EXPECT_THROW(builder.Set<std::string>(0, "bad"), SegcoreError);
    EXPECT_THROW(std::move(builder).Finish(rows.view()), SegcoreError);
    VectorResultBuilder numeric(DataType::INT64, 1);
    EXPECT_THROW(numeric.Set<double>(0, 1), SegcoreError);
    numeric.Set<int64_t>(0, 17);
    auto number = std::move(numeric).Finish(SelectionView::All(1));
    ASSERT_NE(number.FlatData<int64_t>(), nullptr);
    EXPECT_EQ(number.FlatData<int64_t>()[0], 17);
}

TEST(VectorAccessTest, BorrowedDescriptorsAndJsonKeepOriginalPayload) {
    auto texts = std::make_shared<std::vector<std::string>>(
        std::initializer_list<std::string>{std::string(1024, 'a'), "tail"});
    struct Owner {
        std::shared_ptr<std::vector<std::string>> texts;
        std::vector<std::string_view> views;
    };
    auto owner =
        std::make_shared<Owner>(Owner{texts, {(*texts)[0], (*texts)[1]}});
    auto view = ReadOnlyVectorView::BorrowImmutable<std::string_view>(
        DataType::VARCHAR,
        owner->views.data(),
        owner,
        SelectionView::All(2),
        SelectionView::All(2));
    const auto* address = (*texts)[0].data();
    auto mapped = view.Dictionary(
        std::make_shared<RowMapping>(std::vector<vector_size_t>{1, 0, 0}));
    texts.reset();
    owner.reset();
    EXPECT_EQ(mapped.StringViewAt(1).data(), address);
    EXPECT_EQ(mapped.StringViewAt(2).data(), address);
    EXPECT_EQ(mapped.ValueAt<std::string>(0), "tail");
    auto json = std::make_shared<Json>(
        simdjson::padded_string(std::string_view("{\"x\":42}")));
    auto j = ReadOnlyVectorView::BorrowImmutable<Json>(DataType::JSON,
                                                       json.get(),
                                                       json,
                                                       SelectionView::All(1),
                                                       SelectionView::All(1));
    EXPECT_EQ(j.JsonViewAt(0).data().data(), json->data().data());
    EXPECT_NE(j.ValueAt<Json>(0).data().data(), json->data().data());
    json.reset();
    EXPECT_EQ(j.JsonViewAt(0).at<int64_t>("/x").value(), 42);
}

TEST(VectorAccessTest, SegmentsMergeMapsAndOwnersAreIndependent) {
    auto a = std::make_shared<std::vector<int64_t>>(
        std::initializer_list<int64_t>{10, 11});
    auto b = std::make_shared<std::vector<int64_t>>(
        std::initializer_list<int64_t>{20, 21});
    std::weak_ptr<void> weak_a = a, weak_b = b;
    auto av =
        ReadOnlyVectorView::BorrowImmutable<int64_t>(DataType::INT64,
                                                     a->data(),
                                                     a,
                                                     SelectionView::All(2),
                                                     SelectionView::All(2));
    auto bv =
        ReadOnlyVectorView::BorrowImmutable<int64_t>(DataType::INT64,
                                                     b->data(),
                                                     b,
                                                     SelectionView::All(2),
                                                     SelectionView::All(2));
    auto first_rows = OwnedSelection::FromIndices(6, {1, 2});
    auto old = ReadOnlyVectorView::Segments(
        DataType::INT64, first_rows.view(), {{1, av}});
    auto patch_rows = OwnedSelection::FromIndices(6, {4, 5});
    auto patch = ReadOnlyVectorView::Segments(
        DataType::INT64, patch_rows.view(), {{4, bv}});
    auto merged = old.Merge(patch, patch_rows.view());
    auto mapping =
        std::make_shared<RowMapping>(std::vector<vector_size_t>{5, 1, 5, 2});
    const auto mapped = merged.Dictionary(mapping).Slice(1, 3);
    EXPECT_EQ(mapped.ValueAt<int64_t>(0), 10);
    EXPECT_EQ(mapped.ValueAt<int64_t>(1), 21);
    EXPECT_EQ(mapped.ValueAt<int64_t>(2), 11);
    EXPECT_THROW(old.IsValid(4), SegcoreError);
    EXPECT_EQ(merged.ValueAt<int64_t>(4), 20);
    EXPECT_EQ(merged.FlatData<int64_t>(), nullptr);
    auto null_rows = OwnedSelection::FromIndices(6, {0, 3});
    merged = merged.Merge(
        ReadOnlyVectorView::Nulls(DataType::INT64, null_rows.view()),
        null_rows.view());
    EXPECT_TRUE(merged.defined_rows().IsAll());
    EXPECT_FALSE(merged.IsValid(0));
    EXPECT_FALSE(merged.IsValid(3));
    EXPECT_EQ(MakeWritableCopy(merged).values->ValueAt<int64_t>(5), 21);
    a.reset();
    b.reset();
    EXPECT_FALSE(weak_a.expired());
    EXPECT_FALSE(weak_b.expired());
    EXPECT_THROW(ReadOnlyVectorView::Segments(
                     DataType::INT64, SelectionView::All(6), {{1, av}}),
                 SegcoreError);
    EXPECT_THROW(
        ReadOnlyVectorView::Segments(
            DataType::INT64, SelectionView::All(3), {{0, av}, {1, bv}}),
        SegcoreError);
}

TEST(VectorAccessTest, IncrementalMergeReleasesReplacedLeaves) {
    auto initial = std::make_shared<std::vector<int64_t>>(64, -1);
    std::weak_ptr<void> weak = initial;
    auto view =
        ReadOnlyVectorView::BorrowImmutable<int64_t>(DataType::INT64,
                                                     initial->data(),
                                                     initial,
                                                     SelectionView::All(64),
                                                     SelectionView::All(64));
    initial.reset();
    for (size_t i = 0; i < 64; ++i) {
        VectorResultBuilder builder(DataType::INT64, 64);
        builder.Set<int64_t>(i, i);
        auto rows = OwnedSelection::FromIndices(64, {i});
        auto patch = std::move(builder).Finish(rows.view());
        view = view.Merge(patch, rows.view());
        for (size_t j = 0; j < 64; ++j)
            EXPECT_EQ(view.ValueAt<int64_t>(j),
                      j <= i ? static_cast<int64_t>(j) : -1);
    }
    EXPECT_TRUE(weak.expired());  // No retained chain of previous merge states.
}

TEST(VectorAccessTest, UnalignedBorrowedNumericLoadAndSlice) {
    auto bytes = std::make_shared<std::vector<char>>(1 + 3 * sizeof(int64_t));
    int64_t data[] = {-7, 0, 123};
    std::memcpy(bytes->data() + 1, data, sizeof(data));
    auto view = ReadOnlyVectorView::BorrowImmutable<int64_t>(
        DataType::INT64,
        reinterpret_cast<const int64_t*>(bytes->data() + 1),
        bytes,
        SelectionView::All(3),
        SelectionView::All(3));
    EXPECT_EQ(view.FlatData<int64_t>(), nullptr);
    const auto slice = view.Slice(1, 2);
    EXPECT_EQ(slice.ValueAt<int64_t>(0), 0);
    EXPECT_EQ(slice.ValueAt<int64_t>(1), 123);
}

TEST(VectorAccessTest, PrimitiveValidityAndBitmapPublication) {
    const auto oversized =
        std::numeric_limits<size_t>::max() / sizeof(void*) + 1;
    EXPECT_THROW(ReadOnlyVectorView::BorrowIndirect<bool>(
                     DataType::BOOL,
                     nullptr,
                     std::make_shared<int>(0),
                     SelectionView::None(oversized),
                     SelectionView::None(oversized)),
                 SegcoreError);
    BitmapResultBuilder builder(4);
    builder.Set<bool>(0, true);
    builder.Set<bool>(2, false);
    EXPECT_THROW(builder.Set<bool>(4, true), SegcoreError);
    const auto bitmap = std::move(builder).Finish(SelectionView::All(4));
    EXPECT_EQ(bitmap.encoding(), ValueEncoding::Bitmap);
    EXPECT_EQ(bitmap.TryValueAt<bool>(0), std::optional<bool>(true));
    EXPECT_FALSE(bitmap.TryValueAt<bool>(1).has_value());
    EXPECT_EQ(bitmap.TryValueAt<bool>(2), std::optional<bool>(false));
    EXPECT_EQ(bitmap.FlatData<bool>(), nullptr);
    EXPECT_THROW(std::move(builder).Finish(SelectionView::All(4)),
                 SegcoreError);
    const auto mapped = bitmap.Dictionary(
        std::make_shared<RowMapping>(std::vector<vector_size_t>{2, 1, 0, 0}));
    EXPECT_EQ(mapped.TryValueAt<bool>(0), std::optional<bool>(false));
    EXPECT_FALSE(mapped.TryValueAt<bool>(1).has_value());
    EXPECT_EQ(mapped.TryValueAt<bool>(3), std::optional<bool>(true));
    auto bytes = std::make_shared<std::vector<char>>(9);
    int64_t value = -42;
    std::memcpy(bytes->data() + 1, &value, sizeof(value));
    const auto unaligned = ReadOnlyVectorView::BorrowImmutable<int64_t>(
        DataType::INT64,
        reinterpret_cast<const int64_t*>(bytes->data() + 1),
        bytes,
        SelectionView::All(1),
        SelectionView::All(1));
    EXPECT_EQ(unaligned.TryValueAt<int64_t>(0), std::optional<int64_t>(-42));
}

TEST(VectorAccessTest, RecursiveArrayViewsKeepLayoutAndOwnersThroughMappings) {
    proto::schema::TypeSchema type;
    type.mutable_array_element()->mutable_array_element()->set_leaf_type(
        proto::schema::Int64);
    proto::schema::ScalarField empty, one;
    empty.mutable_array_data()->set_element_type(proto::schema::Int64);
    one.mutable_array_data()->set_element_type(proto::schema::Int64);
    one.mutable_array_data()->add_data()->mutable_long_data()->add_data(7);
    auto owner = std::make_shared<std::vector<ArrayValue>>();
    owner->emplace_back(empty, type);
    owner->emplace_back(one, type);
    owner->emplace_back();
    auto valid = OwnedSelection::FromIndices(3, {0, 1});
    auto source =
        ReadOnlyVectorView::BorrowImmutable<ArrayValue>(DataType::ARRAY,
                                                        owner->data(),
                                                        owner,
                                                        SelectionView::All(3),
                                                        valid.view());
    auto first = OwnedSelection::FromIndices(3, {1});
    auto remaining = OwnedSelection::FromIndices(3, {0, 2});
    auto combined =
        ReadOnlyVectorView::Nulls(DataType::ARRAY, SelectionView::None(3))
            .Merge(source.Restrict(first.view()), first.view())
            .Merge(source.Restrict(remaining.view()), remaining.view());
    auto mapped = combined
                      .Dictionary(std::make_shared<RowMapping>(
                          std::vector<vector_size_t>{1, 0, 1, 2}))
                      .Slice(1, 3);
    owner.reset();
    EXPECT_EQ(mapped.ArrayLengthAt(0), 0);
    EXPECT_EQ(mapped.ArrayLengthAt(1), 1);
    EXPECT_FALSE(mapped.IsValid(2));
    EXPECT_EQ(mapped.ValueAt<ArrayValue>(1)
                  .View()
                  .array_at(0)
                  .get_data_unchecked<int64_t>(0),
              7);
    EXPECT_THROW(mapped.ArrayAt(1), SegcoreError);
    EXPECT_THROW(mapped.ArrayLengthAt(2), SegcoreError);
    // Until legacy writable ARRAYs carry recursive schemas, reject conversion
    // explicitly instead of interpreting ArrayValue storage as flat Array.
    EXPECT_THROW(MakeWritableCopy(mapped), SegcoreError);
}

TEST(VectorAccessTest, VectorValiditySlotsHavePayloadFreeNullPlaceholders) {
    for (auto type : {DataType::VECTOR_FLOAT,
                      DataType::VECTOR_BINARY,
                      DataType::VECTOR_FLOAT16,
                      DataType::VECTOR_BFLOAT16,
                      DataType::VECTOR_INT8,
                      DataType::VECTOR_SPARSE_U32_F32}) {
        auto none = ReadOnlyVectorView::Nulls(type, SelectionView::None(3));
        EXPECT_THROW(none.IsValid(0), SegcoreError);
        auto all = ReadOnlyVectorView::Nulls(type, SelectionView::All(3));
        auto copied = ReadOnlyVectorView::CopyOf(all.AsVector());
        EXPECT_EQ(copied.logical_type(), type);
        EXPECT_EQ(copied.CountNulls(SelectionView::All(3)), 3);
        EXPECT_FALSE(copied.Slice(1, 2).IsValid(0));
    }
}

TEST(VectorAccessTest, BulkValidityPreservesCoverageMappingsAndOwnership) {
    constexpr size_t n = 131;
    auto owner = std::make_shared<std::vector<int64_t>>(n, 7);
    auto valid = OwnedSelection::All(n);
    for (size_t i = 0; i < n; ++i)
        if (i % 5 == 0)
            valid.SetSelected(i, false);
    auto borrowed =
        ReadOnlyVectorView::BorrowImmutable<int64_t>(DataType::INT64,
                                                     owner->data(),
                                                     owner,
                                                     SelectionView::All(n),
                                                     valid.view());
    auto numeric =
        ReadOnlyVectorView::CopyOf(MakeWritableCopy(borrowed).values);
    BitmapResultBuilder bitmap(n);
    for (size_t i = 0; i < n; ++i)
        if (i % 5 != 0)
            bitmap.Set<bool>(i, i % 2 == 0);
    auto boolean = std::move(bitmap).Finish(SelectionView::All(n));
    auto coverage = OwnedSelection::All(n);
    for (size_t i = 0; i < n; ++i)
        if (i % 7 == 0)
            coverage.SetSelected(i, false);
    for (const auto& base : {borrowed, numeric, boolean}) {
        auto restricted = base.Restrict(coverage.view());
        auto mask = restricted.CopyValidity();
        for (size_t i = 0; i < n; ++i)
            EXPECT_EQ(mask.view().IsSelected(i), i % 5 != 0 && i % 7 != 0);
        // Non-word-aligned, nested slices must rebase once, not read row 0.
        auto slice = restricted.Slice(3, 126).Slice(58, 68);
        auto sliced = slice.CopyValidity();
        for (size_t i = 0; i < 68; ++i)
            EXPECT_EQ(sliced.view().IsSelected(i),
                      (i + 61) % 5 != 0 && (i + 61) % 7 != 0);
        // Dictionary -> slice: fallback must retain outer coordinates.
        auto mapped =
            restricted
                .Dictionary(std::make_shared<RowMapping>(
                    std::vector<vector_size_t>{65, 64, 64, 63, 129, 130}))
                .Slice(1, 5);
        auto mapped_valid = mapped.CopyValidity();
        const bool expected[] = {true, true, false, true, false};
        for (size_t i = 0; i < 5; ++i)
            EXPECT_EQ(mapped_valid.view().IsSelected(i), expected[i]);
        mask.SetSelected(1, false);
        EXPECT_TRUE(restricted.IsValid(1));  // detached, immutable publication
    }
    auto detached = borrowed.CopyValidity();
    borrowed =
        ReadOnlyVectorView::Nulls(DataType::INT64, SelectionView::None(n));
    owner.reset();
    EXPECT_TRUE(detached.view().IsSelected(129));
    EXPECT_FALSE(detached.view().IsSelected(130));
}

TEST(VectorAccessTest,
     BulkValidityHandlesConstantFragmentsAndIncrementalMerge) {
    auto source = MakeColumn<int64_t>(DataType::INT64, {1, 2, 3, 4, 5, 6});
    source->nullAt(1);
    source->nullAt(4);
    auto view = ReadOnlyVectorView::CopyOf(source);
    auto selected = OwnedSelection::FromIndices(6, {0, 1, 3, 4, 5});
    auto sparse = view.Restrict(selected.view());
    auto fragments = ReadOnlyVectorView::Segments(
        DataType::INT64,
        selected.view(),
        {{0, sparse.Slice(0, 2)}, {3, sparse.Slice(3, 3)}});
    auto first = OwnedSelection::FromIndices(6, {0, 4});
    auto rest = OwnedSelection::FromIndices(6, {1, 3, 5});
    auto merged =
        ReadOnlyVectorView::Nulls(DataType::INT64, SelectionView::None(6))
            .Merge(view.Restrict(first.view()), first.view())
            .Merge(view.Restrict(rest.view()), rest.view());
    for (const auto& value : {fragments, merged}) {
        auto valid = value.CopyValidity();
        for (size_t i = 0; i < 6; ++i)
            EXPECT_EQ(valid.view().IsSelected(i), i == 0 || i == 3 || i == 5);
        auto sliced = value.Slice(1, 4).CopyValidity();
        for (size_t i = 0; i < 4; ++i)
            EXPECT_EQ(sliced.view().IsSelected(i), i == 2);
    }
    auto constant = ReadOnlyVectorView::CopyOf(
        std::make_shared<ConstantVector<int64_t>>(DataType::INT64, 6, 42));
    auto known = constant.Restrict(selected.view()).CopyValidity();
    EXPECT_EQ(known.view().selected_count(), 5);
    EXPECT_FALSE(known.view().IsSelected(2));
    for (auto type :
         {DataType::INT64, DataType::NONE, DataType::VECTOR_FLOAT}) {
        auto nulls =
            ReadOnlyVectorView::Nulls(type, selected.view()).CopyValidity();
        auto empty = ReadOnlyVectorView::Nulls(type, SelectionView::None(0))
                         .CopyValidity();
        EXPECT_TRUE(nulls.view().IsNone());
        EXPECT_TRUE(empty.view().IsNone());
    }
}

TEST(VectorAccessTest, PrimitiveSpansRetainOwnersAndRespectSlicesAndGaps) {
    constexpr size_t n = 193;
    auto owner = std::make_shared<std::vector<int64_t>>(n);
    for (size_t i = 0; i < n; ++i) (*owner)[i] = i * 7;
    auto valid = OwnedSelection::All(n);
    valid.SetSelected(64, false);
    auto source =
        ReadOnlyVectorView::BorrowImmutable<int64_t>(DataType::INT64,
                                                     owner->data(),
                                                     owner,
                                                     SelectionView::All(n),
                                                     valid.view());
    auto selected = OwnedSelection::All(n);
    for (size_t i = 71; i < 79; ++i) selected.SetSelected(i, false);
    auto parts = ReadOnlyVectorView::Segments(
        DataType::INT64,
        selected.view(),
        {{0, source.Slice(0, 71)}, {79, source.Slice(79, n - 79)}});
    auto sliced = parts.Slice(3, 187).Slice(5, 177);
    auto decoded = sliced.DecodePrimitive();
    ASSERT_TRUE(decoded);
    ASSERT_EQ(decoded->spans.size(), 2);
    EXPECT_EQ(decoded->spans[0].begin, 0);
    EXPECT_EQ(decoded->spans[0].count, 63);
    EXPECT_EQ(decoded->spans[1].begin, 71);
    EXPECT_FALSE(decoded->spans[0].valid.IsSelected(56));
    auto dictionary = source.Dictionary(
        std::make_shared<RowMapping>(std::vector<vector_size_t>{100, 7, 100}));
    EXPECT_FALSE(dictionary.DecodePrimitive());
    owner.reset();
    source = ReadOnlyVectorView::Nulls(DataType::INT64, SelectionView::None(n));
    parts = source;
    sliced = source;
    dictionary = source;
    EXPECT_EQ(static_cast<const int64_t*>(decoded->spans[0].data)[0], 56);
    EXPECT_EQ(static_cast<const int64_t*>(decoded->spans[1].data)[0], 553);
    EXPECT_FALSE(decoded->spans[0].valid.IsSelected(56));
}

TEST(VectorAccessTest, BatchedWritableCopiesKeepGapsNullsAndOwnedPayloads) {
    constexpr size_t n = 139;
    auto owner =
        std::make_shared<std::vector<std::string>>(n, std::string(256, 'x'));
    std::vector<std::string_view> strings;
    for (const auto& value : *owner) strings.push_back(value);
    auto valid = OwnedSelection::All(n);
    valid.SetSelected(64, false);
    auto source = ReadOnlyVectorView::BorrowImmutable<std::string_view>(
        DataType::VARCHAR,
        strings.data(),
        owner,
        SelectionView::All(n),
        valid.view());
    auto rows = OwnedSelection::All(n);
    rows.SetSelected(65, false);
    auto parts = ReadOnlyVectorView::Segments(
        DataType::VARCHAR,
        rows.view(),
        {{0, source.Slice(0, 65)}, {66, source.Slice(66, n - 66)}});
    auto slice = parts.Slice(3, 130);
    auto copy = MakeWritableCopy(slice);
    EXPECT_TRUE(copy.defined_rows.view().Equals(slice.defined_rows()));
    auto round_trip =
        ReadOnlyVectorView::CopyOf(copy.values, copy.defined_rows.view());
    for (size_t row = 0; row < 130; ++row) {
        if (!slice.defined_rows().IsSelected(row)) {
            EXPECT_FALSE(copy.values->ValidAt(row));
            EXPECT_THROW(round_trip.IsValid(row), SegcoreError);
        } else {
            EXPECT_EQ(round_trip.IsValid(row), row + 3 != 64);
            if (round_trip.IsValid(row))
                EXPECT_EQ(round_trip.StringViewAt(row), (*owner)[row + 3]);
        }
    }
    copy.values->SetValueAt<std::string>(0, "mutated copy");
    EXPECT_EQ(slice.StringViewAt(0), std::string(256, 'x'));
    EXPECT_EQ(round_trip.StringViewAt(0), std::string(256, 'x'));
    // NULL spans with no address must not be touched by the decoded copier.
    auto nulls =
        ReadOnlyVectorView::BorrowImmutable<int64_t>(DataType::INT64,
                                                     nullptr,
                                                     owner,
                                                     SelectionView::All(n),
                                                     SelectionView::None(n));
    auto null_copy = MakeWritableCopy(nulls);
    EXPECT_TRUE(null_copy.values->Validity().none());
    auto constant = ReadOnlyVectorView::CopyOf(
        std::make_shared<ConstantVector<std::string>>(
            DataType::VARCHAR, n, "prefix"));
    auto legacy = constant
                      .Dictionary(std::make_shared<RowMapping>(
                          std::vector<vector_size_t>{100, 4, 100}))
                      .CopyForLegacyFunction();
    ASSERT_TRUE(std::dynamic_pointer_cast<ConstantVector<std::string>>(legacy));
    EXPECT_EQ(legacy->size(), 3);
    EXPECT_THROW(constant.Restrict(rows.view()).CopyForLegacyFunction(),
                 SegcoreError);
    auto null_arg =
        ReadOnlyVectorView::Nulls(DataType::VARCHAR, SelectionView::All(n))
            .CopyForLegacyFunction();
    EXPECT_TRUE(std::dynamic_pointer_cast<ConstantVector<std::string>>(null_arg)
                    ->IsNull());
}

TEST(VectorAccessTest, ConstantBroadcastAndPrimitiveFallback) {
    auto one = ReadOnlyVectorView::CopyOf(
        std::make_shared<ConstantVector<int64_t>>(DataType::INT64, 1, 42));
    auto all = one.Broadcast(SelectionView::All(131));
    EXPECT_EQ(all.encoding(), ValueEncoding::Constant);
    auto decoded = all.DecodePrimitive();
    ASSERT_TRUE(decoded);
    ASSERT_EQ(decoded->spans.size(), 1);
    EXPECT_TRUE(decoded->spans[0].constant);
    EXPECT_EQ(*static_cast<const int64_t*>(decoded->spans[0].data), 42);
    EXPECT_EQ(all.ValueAt<int64_t>(130), 42);
    auto nulls =
        ReadOnlyVectorView::Nulls(DataType::INT64, SelectionView::All(1))
            .Broadcast(SelectionView::All(131));
    auto empty = nulls.DecodePrimitive();
    ASSERT_TRUE(empty);
    EXPECT_TRUE(empty->spans[0].valid.IsNone());
    EXPECT_EQ(empty->spans[0].data, nullptr);
    auto bytes = std::make_shared<std::vector<char>>(17);
    int64_t value = 23;
    std::memcpy(bytes->data() + 1, &value, sizeof(value));
    auto unaligned = ReadOnlyVectorView::BorrowImmutable<int64_t>(
        DataType::INT64,
        reinterpret_cast<const int64_t*>(bytes->data() + 1),
        bytes,
        SelectionView::All(1),
        SelectionView::All(1));
    EXPECT_FALSE(unaligned.DecodePrimitive());
    EXPECT_EQ(unaligned.ValueAt<int64_t>(0), 23);
}

TEST(VectorAccessTest, BooleanMasksPreserveEncodingNullCoverageAndOwnership) {
    constexpr size_t n = 277;
    TargetBitmap truth(n, false), valid(n, false);
    for (size_t row = 0; row < n; ++row) {
        truth.set(
            row,
            row % 3 == 0);  // Deliberately includes TRUE NULL payload bits.
        valid.set(row, row % 7 != 0);
    }
    auto base = ReadOnlyVectorView::CopyOf(
        std::make_shared<ColumnVector>(std::move(truth), std::move(valid)));
    auto check = [](const ReadOnlyVectorView& view) {
        auto rows = OwnedSelection::CopyOf(view.defined_rows());
        for (size_t row = 0; row < view.size(); row += 11)
            rows.SetSelected(row, false);
        auto masks = view.CopyBoolMasks(rows.view());
        auto published = ReadOnlyVectorView::FromBooleanMasks(
            masks.truth, masks.valid, rows.view());
        for (size_t row = 0; row < view.size(); ++row) {
            const bool known = rows.view().IsSelected(row) && view.IsValid(row);
            EXPECT_EQ(masks.valid.view().IsSelected(row), known);
            EXPECT_EQ(masks.truth.view().IsSelected(row),
                      known && view.ValueAt<bool>(row));
            EXPECT_EQ(published.defined_rows().IsSelected(row),
                      rows.view().IsSelected(row));
            if (rows.view().IsSelected(row)) {
                EXPECT_EQ(published.IsValid(row), known);
                if (known)
                    EXPECT_EQ(published.ValueAt<bool>(row),
                              view.ValueAt<bool>(row));
            }
        }
        masks.truth.ResetAll(view.size());
        masks.valid.ResetNone(view.size());
        return published;
    };
    check(base);
    auto sliced = base.Slice(3, 270).Slice(5, 259);
    check(sliced);
    check(base.Dictionary(std::make_shared<RowMapping>(
        std::vector<vector_size_t>{260, 7, 260, 129, 0, 1})));
    auto defined = OwnedSelection::All(n);
    for (size_t row = 65; row < 71; ++row) defined.SetSelected(row, false);
    auto segmented = ReadOnlyVectorView::Segments(
        DataType::BOOL,
        defined.view(),
        {{0, base.Slice(0, 65)}, {71, base.Slice(71, n - 71)}});
    check(segmented);
    auto patch_rows = OwnedSelection::FromIndices(n, {65, 66, 67, 68, 69, 70});
    check(segmented.Merge(base, patch_rows.view()));
    for (bool value : {false, true}) {
        auto constant =
            ReadOnlyVectorView::CopyOf(std::make_shared<ConstantVector<bool>>(
                                           DataType::BOOL, 1, value))
                .Broadcast(SelectionView::All(n));
        check(constant.Slice(3, 259));
    }
    check(ReadOnlyVectorView::Nulls(DataType::BOOL, SelectionView::All(n)));
    auto flat = std::make_shared<ColumnVector>(DataType::BOOL, n);
    for (size_t row = 0; row < n; ++row)
        flat->SetValueAt<bool>(row, row % 5 == 0);
    flat->nullAt(64);
    check(ReadOnlyVectorView::CopyOf(flat));
    auto published = check(sliced);
    base = ReadOnlyVectorView::Nulls(DataType::BOOL, SelectionView::None(n));
    sliced = base;
    EXPECT_TRUE(published.IsValid(1));
    EXPECT_TRUE(published.ValueAt<bool>(1));  // Original position 9.
    EXPECT_THROW(segmented.CopyBoolMasks(SelectionView::All(n)), SegcoreError);
    EXPECT_THROW(ReadOnlyVectorView::FromBooleanMasks(OwnedSelection::All(n),
                                                      OwnedSelection::None(n),
                                                      SelectionView::All(n)),
                 SegcoreError);
    EXPECT_THROW(ReadOnlyVectorView::FromBooleanMasks(OwnedSelection::None(n),
                                                      OwnedSelection::All(n),
                                                      SelectionView::None(n)),
                 SegcoreError);
}

TEST(VectorAccessTest, DensePayloadCapabilityKeepsCoverageAndOwnerBoundaries) {
    auto storage = std::make_shared<std::vector<int64_t>>(131, 42);
    auto valid = OwnedSelection::All(131);
    valid.SetSelected(64, false);
    auto full = ReadOnlyVectorView::BorrowImmutable<int64_t>(
        DataType::INT64,
        storage->data(),
        storage,
        SelectionView::All(131),
        valid.view(),
        PrimitivePayloadAccess::AllDefinedRows);
    auto slice = full.Slice(3, 127);
    auto decoded = slice.DecodePrimitive();
    ASSERT_TRUE(decoded);
    EXPECT_TRUE(decoded->spans[0].CanReadAll(0, 127));
    EXPECT_FALSE(decoded->spans[0].valid.IsSelected(61));
    auto rows = OwnedSelection::FromIndices(131, {1, 64, 129});
    auto sparse = full.Restrict(rows.view()).DecodePrimitive();
    ASSERT_TRUE(sparse);
    EXPECT_FALSE(sparse->spans[0].CanReadAll(0, 131));
    EXPECT_TRUE(sparse->spans[0].CanEvaluateAll(0, 131));
    auto partial_valid = OwnedSelection::Intersect(rows.view(), valid.view());
    auto partial = ReadOnlyVectorView::BorrowImmutable<int64_t>(
        DataType::INT64,
        storage->data(),
        storage,
        rows.view(),
        partial_valid.view(),
        PrimitivePayloadAccess::AllDefinedRows);
    EXPECT_FALSE(partial.DecodePrimitive()->spans[0].CanEvaluateAll(0, 131));
    auto again = full.Restrict(rows.view()).Slice(1, 129).DecodePrimitive();
    ASSERT_TRUE(again);
    EXPECT_TRUE(again->spans[0].CanEvaluateAll(0, 129));
    EXPECT_TRUE(sparse->spans[0].CanReadAll(64, 1));
    auto mapped = full.Dictionary(
        std::make_shared<RowMapping>(std::vector<vector_size_t>{64, 1, 64}));
    EXPECT_FALSE(mapped.DecodePrimitive());
    auto missing =
        ReadOnlyVectorView::BorrowImmutable<int64_t>(DataType::INT64,
                                                     nullptr,
                                                     storage,
                                                     SelectionView::All(131),
                                                     SelectionView::None(131));
    auto empty = missing.DecodePrimitive();
    ASSERT_TRUE(empty);
    EXPECT_FALSE(empty->spans[0].CanReadAll(0, 131));
    std::weak_ptr<void> owner = storage;
    storage.reset();
    full = missing;
    slice = missing;
    EXPECT_FALSE(owner.expired());  // decoded spans retain their source owner.
}

TEST(VectorAccessTest, BorrowedValidityAndDirectTrueCopyKeepNullAndCoverage) {
    constexpr size_t n = 259;
    TargetBitmap truth(n, true), valid(n, true);
    for (size_t i = 0; i < n; ++i) {
        truth.set(i, i % 3 != 0);
        valid.set(i, i % 7 != 0);
    }
    // Deliberately retain TRUE payloads under NULL: the consumer must mask them.
    auto base = ReadOnlyVectorView::CopyOf(
        std::make_shared<ColumnVector>(std::move(truth), std::move(valid)));
    auto check = [&](const ReadOnlyVectorView& view) {
        auto borrowed = view.TryBorrowValidity();
        auto owned = view.CopyValidity();
        if (borrowed)
            EXPECT_TRUE(borrowed->Equals(owned.view()));
        auto rows = OwnedSelection::CopyOf(view.defined_rows());
        for (size_t i = 0; i < view.size(); i += 5) rows.SetSelected(i, false);
        TargetBitmap output(view.size() + 13, true);
        view.CopyTrueTo(rows.view(), output, 7);
        for (size_t i = 0; i < output.size(); ++i) {
            bool expected = true;
            if (i >= 7 && i < 7 + view.size()) {
                auto row = i - 7;
                expected = rows.view().IsSelected(row) && view.IsValid(row) &&
                           view.ValueAt<bool>(row);
            }
            EXPECT_EQ(output[i], expected) << i;
        }
    };
    ASSERT_TRUE(base.TryBorrowValidity());
    check(base);
    check(base.Slice(3, 129));
    auto sparse = OwnedSelection::FromIndices(n, {1, 7, 64, 128, 258});
    check(base.Restrict(sparse.view()));
    check(base.Dictionary(std::make_shared<RowMapping>(
        std::vector<vector_size_t>{7, 1, 7, 128, 3})));
    check(ReadOnlyVectorView::Nulls(DataType::BOOL, SelectionView::All(n)));
    EXPECT_THROW(
        base.Restrict(sparse.view()).CopyTrueTo(SelectionView::All(n), truth),
        SegcoreError);
}

TEST(VectorAccessTest, StringSpansRetainOwnersAndPreserveRepresentation) {
    auto owner = std::make_shared<std::vector<std::string>>(
        std::initializer_list<std::string>{"", "a", "bc", "d"});
    auto valid = OwnedSelection::All(4);
    valid.SetSelected(2, false);
    auto source =
        ReadOnlyVectorView::BorrowImmutable<std::string>(DataType::VARCHAR,
                                                         owner->data(),
                                                         owner,
                                                         SelectionView::All(4),
                                                         valid.view());
    auto decoded = source.Slice(1, 3).DecodeStrings();
    ASSERT_TRUE(decoded);
    ASSERT_EQ(decoded->spans.size(), 1);
    EXPECT_FALSE(decoded->spans[0].string_views);
    EXPECT_FALSE(decoded->spans[0].valid.IsSelected(1));
    auto dictionary = source.Dictionary(
        std::make_shared<RowMapping>(std::vector<vector_size_t>{3, 2, 3, 0}));
    EXPECT_FALSE(dictionary.DecodeStrings());
    EXPECT_EQ(dictionary.TryStringViewAt(0), "d");
    EXPECT_FALSE(dictionary.TryStringViewAt(1));
    EXPECT_EQ(dictionary.TryStringViewAt(2), "d");
    EXPECT_EQ(dictionary.TryStringViewAt(3), "");
    owner.reset();
    source =
        ReadOnlyVectorView::Nulls(DataType::VARCHAR, SelectionView::All(4));
    dictionary = source;
    EXPECT_EQ(static_cast<const std::string*>(decoded->spans[0].data)[0], "a");
    auto nulls = source.DecodeStrings();
    ASSERT_TRUE(nulls);
    EXPECT_EQ(nulls->spans[0].data, nullptr);
    EXPECT_TRUE(nulls->spans[0].valid.IsNone());
}

TEST(VectorAccessTest, OptionalBitmapRunsPreserveNeighborsAndNulls) {
    constexpr size_t n = 273;
    for (size_t begin : {size_t{0}, size_t{1}, size_t{7}, size_t{63}}) {
        BitmapResultBuilder result(n);
        result.SetRange(0, n, true);
        const size_t count = 193;
        result.EvaluateOptionalRange(
            begin, count, [](size_t i) -> std::optional<bool> {
                if (i % 3 == 0)
                    return std::nullopt;
                return i % 3 == 1;
            });
        auto out = std::move(result).Finish(SelectionView::All(n));
        for (size_t i = 0; i < n; ++i) {
            const bool inside = i >= begin && i < begin + count;
            const bool valid = !inside || (i - begin) % 3 != 0;
            EXPECT_EQ(out.IsValid(i), valid) << begin << ":" << i;
            if (valid)
                EXPECT_EQ(out.ValueAt<bool>(i),
                          !inside || (i - begin) % 3 == 1);
        }
    }
}

TEST(VectorAccessTest, ObjectSpansRetainOwnerAndDictionaryNulls) {
    auto owner = std::make_shared<std::vector<Json>>();
    for (int i = 0; i < 4; ++i)
        owner->emplace_back(
            simdjson::padded_string("{\"x\":" + std::to_string(i) + "}"));
    auto valid = OwnedSelection::FromIndices(4, {0, 1, 3});
    auto view = ReadOnlyVectorView::BorrowImmutable<Json>(DataType::JSON,
                                                          owner->data(),
                                                          owner,
                                                          SelectionView::All(4),
                                                          valid.view());
    auto decoded = view.Slice(1, 3).DecodeObjects<Json>();
    ASSERT_TRUE(decoded);
    ASSERT_EQ(decoded->spans.size(), 1);
    EXPECT_FALSE(decoded->spans[0].valid.IsSelected(1));
    auto mapped = view.Dictionary(
        std::make_shared<RowMapping>(std::vector<vector_size_t>{3, 2, 3, 0}));
    EXPECT_FALSE(mapped.DecodeObjects<Json>());
    EXPECT_EQ(mapped.TryObjectAt<Json>(1), nullptr);
    EXPECT_EQ(mapped.TryObjectAt<Json>(0), mapped.TryObjectAt<Json>(2));
    owner.reset();
    view = ReadOnlyVectorView::Nulls(DataType::JSON, SelectionView::All(4));
    mapped = view;
    EXPECT_EQ(static_cast<const Json*>(decoded->spans[0].data)[0]
                  .at<int64_t>("/x")
                  .value(),
              1);
    auto nulls = view.DecodeObjects<Json>();
    ASSERT_TRUE(nulls);
    EXPECT_TRUE(nulls->spans[0].valid.IsNone());
}

TEST(VectorAccessTest,
     BorrowedArrayDescriptorsPreserveElementNullsAndMappings) {
    struct Owner {
        Array array;
        std::vector<ArrayView> views;
    };
    proto::schema::ScalarField values;
    values.mutable_long_data()->add_data(7);
    values.mutable_long_data()->add_data(99);
    values.add_valid_data(true);
    values.add_valid_data(false);
    auto owner = std::make_shared<Owner>();
    owner->array = Array(values, true);
    const auto& a = owner->array;
    owner->views.emplace_back(const_cast<char*>(a.data()),
                              a.length(),
                              a.byte_size(),
                              a.get_element_type(),
                              a.get_offsets_data(),
                              a.get_element_valid_data(),
                              a.is_element_nullable(),
                              a.has_invalid_element());
    owner->views.emplace_back();
    auto valid = OwnedSelection::FromIndices(2, {0});
    auto view =
        ReadOnlyVectorView::BorrowImmutable<ArrayView>(DataType::ARRAY,
                                                       owner->views.data(),
                                                       owner,
                                                       SelectionView::All(2),
                                                       valid.view());
    auto owned = ReadOnlyVectorView::CopyOf(MakeWritableCopy(view).values);
    auto mixed = ReadOnlyVectorView::Segments(
        DataType::ARRAY, SelectionView::All(4), {{0, view}, {2, owned}});
    auto mapped = mixed.Dictionary(
        std::make_shared<RowMapping>(std::vector<vector_size_t>{2, 0, 1, 0}));
    owner.reset();
    for (size_t i : {0, 1, 3}) {
        ASSERT_TRUE(mapped.TryArrayViewAt(i));
        EXPECT_EQ(mapped.ArrayViewAt(i).get_data_unchecked<int64_t>(0), 7);
        EXPECT_FALSE(mapped.ArrayViewAt(i).is_element_valid(1));
        EXPECT_EQ(mapped.TryArrayLengthAt(i), 2);
        auto copy = mapped.ValueAt<Array>(i);
        EXPECT_FALSE(copy.is_element_valid(1));
    }
    EXPECT_FALSE(mapped.TryArrayViewAt(2));
    EXPECT_FALSE(mapped.TryArrayLengthAt(2));
}

}  // namespace milvus
