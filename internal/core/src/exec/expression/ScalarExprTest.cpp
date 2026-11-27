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
#include <bit>

#include <limits>
#include <optional>
#include <random>
#include <sys/mman.h>
#include <unistd.h>

#include "common/ValueOp.h"
#include "exec/expression/InputRefExpr.h"
#include "exec/expression/ScalarCallExpr.h"
#include "exec/expression/ScalarBatchKernels.h"
#include "exec/expression/ScalarArithmetic.h"
#include "exec/expression/ScalarJsonPredicates.h"
#include "exec/expression/ScalarCollectionPredicates.h"
#include "exec/expression/ScalarObjectKernels.h"
#include "exec/expression/Element.h"
#include "exec/expression/ScalarPredicates.h"
#include "exec/expression/ScalarExprSet.h"
#include "exec/expression/function/impl/StringFunctions.h"

namespace milvus::exec::scalar {
namespace {

class OrderedBool final : public Expr {
 public:
    OrderedBool(int id, std::vector<int>& calls, bool fail = false)
        : Expr(DataType::BOOL), id_(id), calls_(calls), fail_(fail) {
    }

 protected:
    ReadOnlyVectorView
    EvalImpl(SelectionView rows, EvalCtx&) const override {
        calls_.push_back(id_);
        if (fail_)
            ThrowInfo(UnexpectedError, "ordered predicate failure");
        return ReadOnlyVectorView::CopyOf(
            std::make_shared<ConstantVector<bool>>(
                DataType::BOOL, rows.row_count(), true),
            rows);
    }

 private:
    int id_;
    std::vector<int>& calls_;
    bool fail_;
};

BatchInput
Input(size_t size, std::vector<DataType> types = {}) {
    std::vector<std::string> names;
    for (size_t i = 0; i < types.size(); ++i) {
        names.push_back("c" + std::to_string(i));
    }
    return BatchInput(
        std::make_shared<RowType>(std::move(names), std::move(types)), size);
}

ExprPtr
Ref(column_index_t slot, DataType type) {
    return std::make_shared<InputRefExpr>(slot, type);
}

template <typename T>
ExprPtr
Literal(DataType type, T value, bool is_null = false) {
    return MakeLiteral(std::make_shared<ConstantVector<T>>(
        type, 1, value, is_null ? std::optional<size_t>{1} : std::nullopt));
}

template <typename T>
ReadOnlyVectorView
Values(DataType type, std::vector<T> values) {
    auto column = std::make_shared<ColumnVector>(type, values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        column->template SetValueAt<T>(i, values[i]);
    }
    return ReadOnlyVectorView::CopyOf(column);
}

ReadOnlyVectorView
Bools(const std::vector<int>& values) {
    TargetBitmap data(values.size(), false), valid(values.size(), false);
    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i] >= 0) {
            data.set(i, values[i] != 0);
            valid.set(i);
        }
    }
    return ReadOnlyVectorView::CopyOf(
        std::make_shared<ColumnVector>(std::move(data), std::move(valid)));
}

int
Tri(const ReadOnlyVectorView& values, size_t row) {
    return values.IsValid(row) ? static_cast<int>(values.ValueAt<bool>(row))
                               : -1;
}

std::vector<size_t>
Indices(SelectionView rows) {
    std::vector<size_t> result;
    rows.ForEachSelected([&](size_t row) { result.push_back(row); });
    return result;
}

void
ExpectBools(const ReadOnlyVectorView& values, std::vector<int> expected) {
    ASSERT_EQ(values.size(), expected.size());
    for (size_t row = 0; row < expected.size(); ++row) {
        EXPECT_EQ(Tri(values, row), expected[row]) << "row " << row;
    }
}

}  // namespace

TEST(ScalarExprTest,
     FullThreeValuedTruthTableMatchesIndependentAndLegacyResults) {
    std::vector<int> left, right;
    for (int a : {-1, 0, 1}) {
        for (int b : {-1, 0, 1}) {
            left.push_back(a);
            right.push_back(b);
        }
    }
    auto input = Input(9, {DataType::BOOL, DataType::BOOL});
    const auto a = Bools(left), b = Bools(right);
    input.BindValues(0, a);
    input.BindValues(1, b);
    EvalCtx ctx(input);
    for (auto op : {LogicalOp::And, LogicalOp::Or}) {
        auto result =
            MakeLogical(op, {Ref(0, DataType::BOOL), Ref(1, DataType::BOOL)})
                ->Eval(SelectionView::All(9), ctx);
        // Independent SQL truth-table oracle.
        ExpectBools(result,
                    op == LogicalOp::And
                        ? std::vector<int>{-1, 0, -1, 0, 0, 0, -1, 0, 1}
                        : std::vector<int>{-1, -1, 1, -1, 0, 1, 1, 1, 1});
        auto legacy = MakeWritableCopy(a);
        auto rhs = MakeWritableCopy(b);
        if (op == LogicalOp::And) {
            common::ThreeValuedLogicOp::And(legacy.values, rhs.values);
        } else {
            common::ThreeValuedLogicOp::Or(legacy.values, rhs.values);
        }
        auto expected = ReadOnlyVectorView::CopyOf(legacy.values);
        for (size_t row = 0; row < 9; ++row) {
            EXPECT_EQ(Tri(result, row), Tri(expected, row));
        }
    }
    auto result =
        MakeNot(Ref(0, DataType::BOOL))->Eval(SelectionView::All(9), ctx);
    for (size_t row = 0; row < 9; ++row) {
        EXPECT_EQ(Tri(result, row), left[row] < 0 ? -1 : !left[row]);
        EXPECT_EQ(Tri(input.GetColumn(0, SelectionView::All(9)), row),
                  left[row]);
    }
}

TEST(ScalarExprTest, ThreeOperandsShortCircuitWithoutReadingUndefinedResults) {
    auto input = Input(4, {DataType::BOOL, DataType::BOOL, DataType::BOOL});
    input.BindValues(0, Bools({0, 1, -1, -1}));
    std::vector<std::vector<size_t>> requests;
    input.BindLoader(1, [&](SelectionView rows) {
        requests.push_back(Indices(rows));
        return Bools({1, 0, 1, -1}).Restrict(rows);
    });
    input.BindLoader(2, [&](SelectionView rows) {
        requests.push_back(Indices(rows));
        return Bools({1, 1, 0, 1}).Restrict(rows);
    });
    EvalCtx ctx(input);
    const auto result = MakeLogical(LogicalOp::And,
                                    {Ref(0, DataType::BOOL),
                                     Ref(1, DataType::BOOL),
                                     Ref(2, DataType::BOOL)})
                            ->Eval(SelectionView::All(4), ctx);
    ExpectBools(result, {0, 0, 0, -1});
    ASSERT_EQ(requests.size(), 2);
    EXPECT_EQ(requests[0], (std::vector<size_t>{1, 2, 3}));
    EXPECT_EQ(requests[1], (std::vector<size_t>{2, 3}));
}

TEST(ScalarExprTest, LogicalTriplesPreserveSparseCoverageAndShortCircuit) {
    constexpr size_t n = 139;
    std::vector<std::vector<int>> values(3, std::vector<int>(n));
    auto rows = OwnedSelection::All(n);
    for (size_t row = 0; row < n; ++row) {
        values[0][row] = int(row % 3) - 1;
        values[1][row] = int(row / 3 % 3) - 1;
        values[2][row] = int(row / 9 % 3) - 1;
        if (row % 11 == 0)
            rows.SetSelected(row, false);
    }
    for (auto op : {LogicalOp::And, LogicalOp::Or}) {
        auto input = Input(n, {DataType::BOOL, DataType::BOOL, DataType::BOOL});
        std::vector<std::vector<size_t>> requests(3);
        for (size_t slot = 0; slot < 3; ++slot)
            input.BindLoader(slot, [&, slot](SelectionView selected) {
                requests[slot] = Indices(selected);
                return Bools(values[slot]).Restrict(selected);
            });
        EvalCtx ctx(input);
        auto result = MakeLogical(op,
                                  {Ref(0, DataType::BOOL),
                                   Ref(1, DataType::BOOL),
                                   Ref(2, DataType::BOOL)})
                          ->Eval(rows.view(), ctx);
        EXPECT_TRUE(result.defined_rows().Equals(rows.view()));
        std::vector<std::vector<size_t>> expected_requests(3);
        const int decisive = op == LogicalOp::And ? 0 : 1;
        for (size_t row = 0; row < n; ++row) {
            if (!rows.view().IsSelected(row)) {
                EXPECT_THROW(result.IsValid(row), SegcoreError);
                continue;
            }
            bool stopped = false, unknown = false;
            for (size_t slot = 0; slot < 3; ++slot) {
                if (!stopped)
                    expected_requests[slot].push_back(row);
                stopped |= values[slot][row] == decisive;
                unknown |= values[slot][row] == -1;
            }
            const int expected = stopped   ? decisive
                                 : unknown ? -1
                                           : 1 - decisive;
            EXPECT_EQ(Tri(result, row), expected) << "row " << row;
        }
        EXPECT_EQ(requests, expected_requests);
    }
}

TEST(ScalarExprTest, WholeBatchShortCircuitAndEmptySelectionNeverLoadOrCall) {
    for (auto op : {LogicalOp::And, LogicalOp::Or}) {
        auto input = Input(4, {DataType::BOOL});
        int loads = 0, calls = 0;
        input.BindLoader(0, [&](SelectionView) -> ReadOnlyVectorView {
            ++loads;
            ThrowInfo(FileReadFailed, "must not load skipped input");
        });
        auto call =
            MakeCall(DataType::BOOL,
                     {Ref(0, DataType::BOOL)},
                     [&](const auto&, auto, auto&) -> ReadOnlyVectorView {
                         ++calls;
                         ThrowInfo(FileReadFailed, "must not call");
                     });
        EvalCtx ctx(input);
        auto decisive = Literal(DataType::BOOL, op == LogicalOp::Or);
        auto root = MakeLogical(op, {decisive, call});
        ExpectBools(root->Eval(SelectionView::All(4), ctx),
                    std::vector<int>(4, op == LogicalOp::Or));
        auto empty = call->Eval(SelectionView::None(4), ctx);
        EXPECT_EQ(empty.size(), 4);
        EXPECT_TRUE(empty.defined_rows().IsNone());
        EXPECT_EQ(loads, 0);
        EXPECT_EQ(calls, 0);
    }
}

TEST(ScalarExprTest, NotAndDifferentRootSelectionsDoNotMutateSharedInput) {
    auto input = Input(4, {DataType::BOOL});
    const auto values = Bools({1, 0, -1, 1});
    std::vector<std::vector<size_t>> requests;
    input.BindLoader(0, [&](SelectionView rows) {
        requests.push_back(Indices(rows));
        return values.Restrict(rows);
    });
    EvalCtx ctx(input);
    auto ref = Ref(0, DataType::BOOL);
    ExprSet set({MakeNot(ref), ref});
    const auto first = OwnedSelection::FromIndices(4, {1, 3});
    const auto second = OwnedSelection::FromIndices(4, {0, 1, 2});
    auto result = set.Eval({first.view(), second.view()}, ctx);
    ASSERT_EQ(requests.size(), 2);
    EXPECT_EQ(requests[0], (std::vector<size_t>{1, 3}));
    EXPECT_EQ(requests[1], (std::vector<size_t>{0, 2}));
    EXPECT_EQ(Tri(result[0], 1), 1);
    EXPECT_EQ(Tri(result[0], 3), 0);
    EXPECT_EQ(Tri(result[1], 1), 0);
    EXPECT_EQ(Tri(result[1], 2), -1);
    EXPECT_THROW(result[0].IsValid(0), SegcoreError);
    EXPECT_THROW(result[1].IsValid(3), SegcoreError);
    set.Eval(SelectionView::All(4), ctx);
    EXPECT_EQ(requests.size(), 2);
    EXPECT_EQ(Tri(result[0], 3), 0);  // retained result survived later loads
}

TEST(ScalarExprTest, SameFieldComparisonsReuseValuesAndKeepNulls) {
    auto column = std::make_shared<ColumnVector>(DataType::INT64, 4);
    for (size_t i = 0; i < 4; ++i) {
        column->SetValueAt<int64_t>(i, i * 5);
    }
    column->nullAt(2);
    const auto values = ReadOnlyVectorView::CopyOf(column);
    auto input = Input(4, {DataType::INT64});
    int loads = 0;
    input.BindLoader(0, [&](SelectionView rows) {
        ++loads;
        return values.Restrict(rows);
    });
    EvalCtx ctx(input);
    auto a = Ref(0, DataType::INT64);
    auto root = MakeLogical(
        LogicalOp::And,
        {MakeComparison(
             CompareOp::Greater, a, Literal<int64_t>(DataType::INT64, 4)),
         MakeComparison(
             CompareOp::Less, a, Literal<int64_t>(DataType::INT64, 10))});
    ExpectBools(root->Eval(SelectionView::All(4), ctx), {0, 1, -1, 0});
    EXPECT_EQ(loads, 1);
}

TEST(ScalarExprTest, ConstantsZeroColumnsZeroRowsAndRepeatedBatches) {
    auto root = Literal<int64_t>(DataType::INT64, 42);
    auto null = Literal(DataType::VARCHAR, std::string("ignored"), true);
    for (size_t size : {size_t{4}, size_t{0}, size_t{7}}) {
        auto input = Input(size);
        EvalCtx ctx(input);
        auto output = ExprSet({root, null}).Eval(SelectionView::All(size), ctx);
        EXPECT_EQ(output[0].size(), size);
        for (size_t row = 0; row < size; ++row) {
            EXPECT_EQ(output[0].ValueAt<int64_t>(row), 42);
            EXPECT_FALSE(output[1].IsValid(row));
        }
        EXPECT_TRUE(ExprSet({}).Eval(SelectionView::All(size), ctx).empty());
    }
}

TEST(ScalarExprTest, RangeEndpointsAndNullsAcrossComparableTypes) {
    auto check = []<typename T>(DataType type) {
        auto column = std::make_shared<ColumnVector>(type, 7);
        const std::vector<T> values{9, 10, 11, 99, 100, 101, 0};
        for (size_t i = 0; i < values.size(); ++i)
            column->template SetValueAt<T>(i, values[i]);
        column->nullAt(6);
        auto input = Input(7, {type});
        input.BindValues(0, ReadOnlyVectorView::CopyOf(column));
        EvalCtx ctx(input);
        auto lower = std::make_shared<ConstantVector<T>>(type, 1, T{10});
        auto upper = std::make_shared<ConstantVector<T>>(type, 1, T{100});
        for (bool lower_inclusive : {false, true}) {
            for (bool upper_inclusive : {false, true}) {
                auto range = MakeRange(Ref(0, type),
                                       lower,
                                       upper,
                                       lower_inclusive,
                                       upper_inclusive);
                ExpectBools(range->Eval(SelectionView::All(7), ctx),
                            {0, lower_inclusive, 1, 1, upper_inclusive, 0, -1});
                ExpectBools(
                    MakeNot(range)->Eval(SelectionView::All(7), ctx),
                    {1, !lower_inclusive, 0, 0, !upper_inclusive, 1, -1});
            }
        }
        ExpectBools(MakeRange(Ref(0, type), upper, lower, true, true)
                        ->Eval(SelectionView::All(7), ctx),
                    {0, 0, 0, 0, 0, 0, -1});
        ExpectBools(MakeRange(Ref(0, type), lower, lower, true, true)
                        ->Eval(SelectionView::All(7), ctx),
                    {0, 1, 0, 0, 0, 0, -1});
        ExpectBools(MakeRange(Ref(0, type), lower, lower, false, true)
                        ->Eval(SelectionView::All(7), ctx),
                    {0, 0, 0, 0, 0, 0, -1});
    };
    check.template operator()<int8_t>(DataType::INT8);
    check.template operator()<int16_t>(DataType::INT16);
    check.template operator()<int32_t>(DataType::INT32);
    check.template operator()<int64_t>(DataType::INT64);
    check.template operator()<int64_t>(DataType::TIMESTAMPTZ);
    check.template operator()<float>(DataType::FLOAT);
    check.template operator()<double>(DataType::DOUBLE);
}

TEST(ScalarExprTest, RangeFloatingSpecialValuesAndIntegerExtremes) {
    auto check = []<typename T>(DataType type) {
        auto input = Input(6, {type});
        input.BindValues(0,
                         Values<T>(type,
                                   {std::numeric_limits<T>::quiet_NaN(),
                                    -std::numeric_limits<T>::infinity(),
                                    -T{0},
                                    T{0},
                                    T{1},
                                    std::numeric_limits<T>::infinity()}));
        EvalCtx ctx(input);
        auto lower = std::make_shared<ConstantVector<T>>(type, 1, T{0});
        auto upper = std::make_shared<ConstantVector<T>>(type, 1, T{1});
        auto range = MakeRange(Ref(0, type), lower, upper, true, true);
        ExpectBools(range->Eval(SelectionView::All(6), ctx),
                    {0, 0, 1, 1, 1, 0});
        ExpectBools(MakeNot(range)->Eval(SelectionView::All(6), ctx),
                    {1, 1, 0, 0, 0, 1});
    };
    check.template operator()<float>(DataType::FLOAT);
    check.template operator()<double>(DataType::DOUBLE);

    auto input = Input(3, {DataType::INT64});
    input.BindValues(0,
                     Values<int64_t>(DataType::INT64,
                                     {std::numeric_limits<int64_t>::min(),
                                      0,
                                      std::numeric_limits<int64_t>::max()}));
    EvalCtx ctx(input);
    auto lower = std::make_shared<ConstantVector<int64_t>>(
        DataType::INT64, 1, std::numeric_limits<int64_t>::min());
    auto upper = std::make_shared<ConstantVector<int64_t>>(
        DataType::INT64, 1, std::numeric_limits<int64_t>::max());
    ExpectBools(MakeRange(Ref(0, DataType::INT64), lower, upper, true, true)
                    ->Eval(SelectionView::All(3), ctx),
                {1, 1, 1});
}

TEST(ScalarExprTest, RangeOwnsStringBoundsAndSupportsBool) {
    for (auto type : {DataType::STRING, DataType::VARCHAR, DataType::TEXT}) {
        const std::string prefix(64, 'x');
        auto lower = std::make_shared<ColumnVector>(type, 1);
        auto upper = std::make_shared<ColumnVector>(type, 1);
        lower->SetValueAt<std::string>(0, prefix + "a");
        upper->SetValueAt<std::string>(0, prefix + "c");
        auto range = MakeRange(Ref(0, type), lower, upper, false, true);
        lower->SetValueAt<std::string>(0, prefix + "z");
        upper.reset();
        auto input = Input(4, {type});
        input.BindValues(
            0,
            Values<std::string>(
                type,
                {prefix + "a", prefix + "b", prefix + "c", prefix + "d"}));
        EvalCtx ctx(input);
        ExpectBools(range->Eval(SelectionView::All(4), ctx), {0, 1, 1, 0});
    }
    auto input = Input(3, {DataType::BOOL});
    input.BindValues(0, Bools({0, 1, -1}));
    EvalCtx ctx(input);
    ExpectBools(
        MakeRange(
            Ref(0, DataType::BOOL),
            std::make_shared<ConstantVector<bool>>(DataType::BOOL, 1, false),
            std::make_shared<ConstantVector<bool>>(DataType::BOOL, 1, true),
            false,
            true)
            ->Eval(SelectionView::All(3), ctx),
        {0, 1, -1});
}

TEST(ScalarExprTest, RangeEvaluatesInputOnceAndOnlyOnSelectedRows) {
    auto input = Input(5, {DataType::INT64});
    std::vector<std::vector<size_t>> requests;
    input.BindLoader(0, [&](SelectionView rows) {
        requests.push_back(Indices(rows));
        return Values<int64_t>(DataType::INT64, {0, 10, 50, 100, 101})
            .Restrict(rows);
    });
    int evaluations = 0;
    auto child = MakeCall(DataType::INT64,
                          {Ref(0, DataType::INT64)},
                          [&](const auto& args, SelectionView, EvalCtx&) {
                              ++evaluations;
                              return args[0];
                          });
    auto range = MakeRange(
        child,
        std::make_shared<ConstantVector<int64_t>>(DataType::INT64, 1, 10),
        std::make_shared<ConstantVector<int64_t>>(DataType::INT64, 1, 100),
        false,
        false);
    EvalCtx ctx(input);
    range->Eval(SelectionView::None(5), ctx);
    MakeLogical(LogicalOp::And, {Literal<bool>(DataType::BOOL, false), range})
        ->Eval(SelectionView::All(5), ctx);
    EXPECT_EQ(evaluations, 0);
    EXPECT_TRUE(requests.empty());
    const auto selected = OwnedSelection::FromIndices(5, {1, 2, 3});
    auto output = range->Eval(selected.view(), ctx);
    EXPECT_EQ(evaluations, 1);
    EXPECT_EQ(requests, (std::vector<std::vector<size_t>>{{1, 2, 3}}));
    EXPECT_EQ(Tri(output, 1), 0);
    EXPECT_EQ(Tri(output, 2), 1);
    EXPECT_EQ(Tri(output, 3), 0);
    EXPECT_THROW(output.IsValid(0), SegcoreError);
    EXPECT_THROW(output.IsValid(4), SegcoreError);
}

TEST(ScalarExprTest, IsNullUsesValidityOnlyAndLaterReadsOnlyNonNullPayloads) {
    auto input = Input(4, {DataType::INT64});
    int payloads = 0, validities = 0;
    input.BindLoader(
        0,
        [&](SelectionView rows) {
            ++payloads;
            EXPECT_EQ(Indices(rows), (std::vector<size_t>{0, 2}));
            return Values<int64_t>(DataType::INT64, {10, 0, 20, 0})
                .Restrict(rows);
        },
        [&](SelectionView rows) {
            ++validities;
            return ValidityResult{OwnedSelection::CopyOf(rows),
                                  OwnedSelection::FromIndices(4, {0, 2})};
        });
    EvalCtx ctx(input);
    const auto ref = Ref(0, DataType::INT64);
    ExpectBools(MakeIsNull(ref)->Eval(SelectionView::All(4), ctx),
                {0, 1, 0, 1});
    ExpectBools(MakeIsNull(ref, true)->Eval(SelectionView::All(4), ctx),
                {1, 0, 1, 0});
    EXPECT_EQ(payloads, 0);
    EXPECT_EQ(validities, 1);
    auto values = ref->Eval(SelectionView::All(4), ctx);
    EXPECT_EQ(payloads, 1);
    EXPECT_EQ(values.ValueAt<int64_t>(2), 20);
    EXPECT_FALSE(values.IsValid(1));
    ExpectBools(MakeIsNull(Literal<int64_t>(DataType::INT64, 0, true))
                    ->Eval(SelectionView::All(4), ctx),
                {1, 1, 1, 1});
}

TEST(ScalarExprTest,
     NativeCallComputesOnlySelectedRowsAndFailureDoesNotPublish) {
    auto input = Input(4, {DataType::INT64});
    input.BindValues(0, Values<int64_t>(DataType::INT64, {0, 2, 0, 5}));
    EvalCtx ctx(input);
    size_t visits = 0;
    auto call = MakeCall(
        DataType::INT64,
        {Ref(0, DataType::INT64)},
        [&](const auto& args, SelectionView rows, EvalCtx&) {
            auto output = std::make_shared<ColumnVector>(DataType::INT64, 4);
            rows.ForEachSelected([&](size_t row) {
                ++visits;
                const auto divisor = args[0].template ValueAt<int64_t>(row);
                if (divisor == 0) {
                    ThrowInfo(ExprInvalid, "injected division by zero");
                }
                output->SetValueAt<int64_t>(row, 10 / divisor);
            });
            return ReadOnlyVectorView::CopyOf(output, rows);
        });
    const auto selected = OwnedSelection::FromIndices(4, {1, 3});
    auto result = call->Eval(selected.view(), ctx);
    EXPECT_EQ(visits, 2);
    EXPECT_EQ(result.ValueAt<int64_t>(1), 5);
    EXPECT_EQ(result.ValueAt<int64_t>(3), 2);
    EXPECT_THROW(result.IsValid(0), SegcoreError);
    try {
        result = call->Eval(SelectionView::All(4), ctx);
        FAIL();
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), ExprInvalid);
    }
    EXPECT_EQ(result.ValueAt<int64_t>(3), 2);
    EXPECT_FALSE(result.defined_rows().IsSelected(0));
}

TEST(ScalarExprTest,
     LegacyStartsWithGatherScatterHandlesSparseNullAndDictionary) {
    auto strings = Values<std::string>(DataType::VARCHAR,
                                       {"skip", "apple", "banana", "apricot"});
    strings = strings.Dictionary(
        std::make_shared<RowMapping>(std::vector<vector_size_t>{2, 1, 2, 3}));
    const auto null_row = OwnedSelection::FromIndices(4, {3});
    strings = strings.Merge(
        ReadOnlyVectorView::Nulls(DataType::VARCHAR, null_row.view()),
        null_row.view());
    auto input = Input(4, {DataType::VARCHAR});
    input.BindValues(0, strings);
    EvalCtx ctx(input);
    auto call = MakeLegacyCall(DataType::BOOL,
                               {Ref(0, DataType::VARCHAR),
                                Literal(DataType::VARCHAR, std::string("ap"))},
                               expression::function::StartsWithVarchar);
    const auto selected = OwnedSelection::FromIndices(4, {1, 3});
    const auto result = call->Eval(selected.view(), ctx);
    EXPECT_EQ(Tri(result, 1), 1);
    EXPECT_EQ(Tri(result, 3), -1);
    EXPECT_THROW(result.IsValid(0), SegcoreError);
    EXPECT_EQ(strings.StringViewAt(1), "apple");
    ExpectBools(call->Eval(SelectionView::All(4), ctx), {0, 1, 0, -1});
}

TEST(ScalarExprTest, LegacyKernelSeesOnlySelectedRowsAndCannotMutateInput) {
    auto input = Input(4, {DataType::INT64});
    const auto original = Values<int64_t>(DataType::INT64, {0, 2, 0, 5});
    input.BindValues(0, original);
    EvalCtx ctx(input);
    auto call = MakeLegacyCall(
        DataType::INT64,
        {Ref(0, DataType::INT64)},
        +[](const RowVector& args, VectorPtr& result) {
            EXPECT_EQ(args.size(), 2);
            auto column =
                std::dynamic_pointer_cast<ColumnVector>(args.child(0));
            for (size_t i = 0; i < column->size(); ++i) {
                const auto value = column->ValueAt<int64_t>(i);
                AssertInfo(value != 0, "legacy kernel read an unselected row");
                column->SetValueAt<int64_t>(i, 10 / value);
            }
            result = column;
        });
    const auto rows = OwnedSelection::FromIndices(4, {1, 3});
    auto result = call->Eval(rows.view(), ctx);
    EXPECT_EQ(result.ValueAt<int64_t>(1), 5);
    EXPECT_EQ(result.ValueAt<int64_t>(3), 2);
    EXPECT_EQ(original.ValueAt<int64_t>(1), 2);
    EXPECT_EQ(input.GetColumn(0, SelectionView::All(4)).ValueAt<int64_t>(3), 5);
}

TEST(ScalarExprTest, LegacyConstantsStayDetachedAcrossGatherAndOutputAliases) {
    static VectorPtr retained;
    const std::string original(256, 'x');
    auto constant = ReadOnlyVectorView::CopyOf(
        std::make_shared<ConstantVector<std::string>>(
            DataType::VARCHAR, 5, original));
    auto input = Input(5, {DataType::VARCHAR});
    input.BindValues(0, constant);
    EvalCtx ctx(input);
    auto call = MakeLegacyCall(
        DataType::VARCHAR,
        {Ref(0, DataType::VARCHAR)},
        +[](const RowVector& args, VectorPtr& output) {
            auto value = std::dynamic_pointer_cast<ConstantVector<std::string>>(
                args.child(0));
            EXPECT_TRUE(value);
            if (!value)
                return;
            *static_cast<std::string*>(value->RawValueAt(
                0, sizeof(std::string))) = "changed by function";
            retained = value;
            output = value;
        });
    for (const auto& selected :
         {OwnedSelection::All(5), OwnedSelection::FromIndices(5, {1, 3})}) {
        auto result = call->Eval(selected.view(), ctx);
        ASSERT_TRUE(retained);
        EXPECT_EQ(retained->size(), selected.view().selected_count());
        auto writable = std::dynamic_pointer_cast<SimpleVector>(retained);
        *static_cast<std::string*>(writable->RawValueAt(
            0, sizeof(std::string))) = "changed after publication";
        selected.view().ForEachSelected([&](size_t row) {
            EXPECT_EQ(result.StringViewAt(row), "changed by function");
            EXPECT_EQ(constant.StringViewAt(row), original);
        });
        retained.reset();
    }
}

TEST(ScalarExprTest, LegacyZeroArgumentFunctionPreservesCompactRowCount) {
    auto input = Input(5);
    EvalCtx ctx(input);
    auto call = MakeLegacyCall(
        DataType::INT64, {}, +[](const RowVector& args, VectorPtr& result) {
            EXPECT_EQ(args.size(), 2);
            EXPECT_TRUE(args.childrens().empty());
            result = std::make_shared<ConstantVector<int64_t>>(
                DataType::INT64, args.size(), 8);
        });
    auto rows = OwnedSelection::FromIndices(5, {1, 4});
    auto result = call->Eval(rows.view(), ctx);
    EXPECT_EQ(result.ValueAt<int64_t>(4), 8);
    EXPECT_FALSE(result.defined_rows().IsSelected(0));
    const auto empty = call->Eval(SelectionView::None(5), ctx);
    EXPECT_TRUE(empty.defined_rows().IsNone());
}

TEST(ScalarExprTest, LoadFailurePreservesCodeAndCanRetryMissingRows) {
    auto input = Input(4, {DataType::INT64});
    int attempts = 0;
    input.BindLoader(0, [&](SelectionView rows) {
        if (++attempts == 1) {
            ThrowInfo(FileReadFailed, "injected loader error");
        }
        return Values<int64_t>(DataType::INT64, {1, 2, 3, 4}).Restrict(rows);
    });
    EvalCtx ctx(input);
    auto root = MakeComparison(CompareOp::Greater,
                               Ref(0, DataType::INT64),
                               Literal<int64_t>(DataType::INT64, 2));
    try {
        root->Eval(SelectionView::All(4), ctx);
        FAIL();
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), FileReadFailed);
    }
    ExpectBools(root->Eval(SelectionView::All(4), ctx), {0, 0, 1, 1});
    EXPECT_EQ(attempts, 2);
}

TEST(ScalarExprTest, CancellationBudgetAndAllocationFailuresPropagate) {
    auto input = Input(2, {DataType::BOOL});
    int loads = 0;
    input.BindLoader(0, [&](SelectionView rows) {
        ++loads;
        return Bools({1, 1}).Restrict(rows);
    });
    auto ref = Ref(0, DataType::BOOL);
    for (auto code : {FollyCancel, InsufficientResource}) {
        EvalCtx ctx(input, [code] { ThrowInfo(code, "injected checkpoint"); });
        try {
            MakeNot(ref)->Eval(SelectionView::All(2), ctx);
            FAIL();
        } catch (const SegcoreError& error) {
            EXPECT_EQ(error.get_error_code(), code);
        }
    }
    EXPECT_EQ(loads, 0);
    EvalCtx ctx(input);
    auto oom = MakeCall(
        DataType::BOOL, {}, [](const auto&, auto, auto&) -> ReadOnlyVectorView {
            throw std::bad_alloc();
        });
    EXPECT_THROW(oom->Eval(SelectionView::All(2), ctx), std::bad_alloc);
    ExpectBools(ref->Eval(SelectionView::All(2), ctx), {1, 1});
    EXPECT_EQ(loads, 1);
}

TEST(ScalarExprTest, BadRootSelectionAndInputBindingRejectedBeforeLoading) {
    auto input = Input(4, {DataType::INT64});
    int loads = 0;
    input.BindLoader(0, [&](SelectionView rows) {
        ++loads;
        return Values<int64_t>(DataType::INT64, {1, 2, 3, 4}).Restrict(rows);
    });
    EvalCtx ctx(input);
    auto ref = Ref(0, DataType::INT64);
    ExprSet set({ref, ref});
    EXPECT_THROW(set.Eval({SelectionView::All(4), SelectionView::All(3)}, ctx),
                 SegcoreError);
    EXPECT_THROW(set.Eval(std::vector<SelectionView>{}, ctx), SegcoreError);
    EXPECT_THROW(ref->Eval(SelectionView::All(3), ctx), SegcoreError);
    EXPECT_THROW(Ref(0, DataType::BOOL)->Eval(SelectionView::All(4), ctx),
                 SegcoreError);
    EXPECT_THROW(Ref(3, DataType::INT64)->Eval(SelectionView::All(4), ctx),
                 SegcoreError);
    EXPECT_EQ(loads, 0);
}

TEST(ScalarExprTest, MalformedNativeAndLegacyResultsAreNotPublished) {
    auto input = Input(3);
    EvalCtx ctx(input);
    for (auto bad :
         {MakeCall(DataType::BOOL,
                   {},
                   [](const auto&, auto, auto&) {
                       return Bools({1, 0});
                   }),
          MakeCall(DataType::BOOL,
                   {},
                   [](const auto&, auto, auto&) {
                       return Values<int64_t>(DataType::INT64, {1, 2, 3});
                   }),
          MakeCall(DataType::BOOL,
                   {},
                   [](const auto&, auto, auto&) {
                       return Bools({1, 0, 1}).Restrict(SelectionView::None(3));
                   }),
          MakeLegacyCall(
              DataType::BOOL, {}, +[](const RowVector&, VectorPtr&) {}),
          MakeLegacyCall(
              DataType::BOOL,
              {},
              +[](const RowVector&, VectorPtr& out) {
                  out = std::make_shared<ConstantVector<bool>>(
                      DataType::BOOL, 2, true);
              }),
          MakeLegacyCall(
              DataType::BOOL, {}, +[](const RowVector&, VectorPtr& out) {
                  out = std::make_shared<ConstantVector<int64_t>>(
                      DataType::INT64, 3, 1);
              })}) {
        EXPECT_THROW(bad->Eval(SelectionView::All(3), ctx), SegcoreError);
    }
}

TEST(ScalarExprTest, InvalidConstructionIsRejected) {
    auto integer = Literal<int64_t>(DataType::INT64, 1);
    auto text = Literal(DataType::VARCHAR, std::string("1"));
    EXPECT_THROW(MakeNot(integer), SegcoreError);
    EXPECT_THROW(MakeLogical(LogicalOp::And, {}), SegcoreError);
    EXPECT_THROW(MakeLogical(LogicalOp::Or, {integer}), SegcoreError);
    EXPECT_THROW(MakeComparison(CompareOp::Equal, integer, text), SegcoreError);
    EXPECT_THROW(MakeComparison(static_cast<CompareOp>(99), integer, integer),
                 SegcoreError);
    EXPECT_THROW(MakeComparison(CompareOp::Equal, nullptr, integer),
                 SegcoreError);
    EXPECT_THROW(
        MakeLiteral(std::make_shared<ColumnVector>(DataType::INT64, 2)),
        SegcoreError);
    EXPECT_THROW(ExprSet({nullptr}), SegcoreError);
    EXPECT_THROW(MakeCall(DataType::BOOL, {}, {}), SegcoreError);
    EXPECT_THROW(MakeLegacyCall(DataType::BOOL, {}, nullptr), SegcoreError);
}

TEST(ScalarExprTest, AllComparisonOperatorsTypesAndNaNKeepNativeSemantics) {
    const std::vector<CompareOp> ops{CompareOp::Equal,
                                     CompareOp::NotEqual,
                                     CompareOp::Less,
                                     CompareOp::LessEqual,
                                     CompareOp::Greater,
                                     CompareOp::GreaterEqual};
    const std::vector<std::vector<int>> expected{
        {0, 1, 0}, {1, 0, 1}, {1, 0, 0}, {1, 1, 0}, {0, 0, 1}, {0, 1, 1}};
    auto check = [&]<typename T>(DataType type) {
        auto input = Input(3, {type});
        input.BindValues(0, Values<T>(type, {T(1), T(2), T(3)}));
        EvalCtx ctx(input);
        for (size_t i = 0; i < ops.size(); ++i) {
            ExpectBools(
                MakeComparison(ops[i], Ref(0, type), Literal<T>(type, T(2)))
                    ->Eval(SelectionView::All(3), ctx),
                expected[i]);
        }
    };
    check.template operator()<int8_t>(DataType::INT8);
    check.template operator()<int16_t>(DataType::INT16);
    check.template operator()<int32_t>(DataType::INT32);
    check.template operator()<int64_t>(DataType::INT64);
    check.template operator()<int64_t>(DataType::TIMESTAMPTZ);
    check.template operator()<float>(DataType::FLOAT);
    check.template operator()<double>(DataType::DOUBLE);
    auto input = Input(1, {DataType::DOUBLE});
    input.BindValues(
        0,
        Values<double>(DataType::DOUBLE,
                       {std::numeric_limits<double>::quiet_NaN()}));
    EvalCtx ctx(input);
    for (auto op : ops) {
        ExpectBools(
            MakeComparison(
                op, Ref(0, DataType::DOUBLE), Literal(DataType::DOUBLE, 0.0))
                ->Eval(SelectionView::All(1), ctx),
            {op == CompareOp::NotEqual ? 1 : 0});
    }
}

TEST(ScalarExprTest, StringComparisonSupportsMappedDuplicatesAndSlices) {
    for (auto type : {DataType::STRING, DataType::VARCHAR, DataType::TEXT}) {
        auto input = Input(4, {type});
        const auto values = Values<std::string>(type, {"a", "b", "c"});
        input.BindValues(0,
                         values.Dictionary(std::make_shared<RowMapping>(
                             std::vector<vector_size_t>{2, 0, 2, 1})));
        EvalCtx ctx(input);
        auto root = MakeComparison(
            CompareOp::Greater, Ref(0, type), Literal(type, std::string("b")));
        ExpectBools(root->Eval(SelectionView::All(4), ctx), {1, 0, 1, 0});
    }
}

TEST(ScalarExprTest,
     FilterProjectionAndDownstreamInputShareWithoutCursorState) {
    int loads = 0;
    auto input = Input(4, {DataType::INT64});
    auto values = std::make_shared<ColumnVector>(DataType::INT64, 4);
    for (size_t i = 0; i < 4; ++i) {
        values->SetValueAt<int64_t>(i, std::vector<int64_t>{17, 20, 0, 30}[i]);
    }
    values->nullAt(2);
    input.BindLoader(0, [&](SelectionView rows) {
        ++loads;
        return ReadOnlyVectorView::CopyOf(values, rows);
    });
    EvalCtx ctx(input);
    const auto ref = Ref(0, DataType::INT64);
    const auto filter = MakeComparison(
        CompareOp::GreaterEqual, ref, Literal<int64_t>(DataType::INT64, 18));
    auto mask = filter->Eval(SelectionView::All(4), ctx);
    auto selected = OwnedSelection::None(4);
    for (size_t i = 0; i < 4; ++i) {
        if (mask.IsValid(i) && mask.ValueAt<bool>(i)) {
            selected.SetSelected(i);
        }
    }
    auto add_one = MakeCall(
        DataType::INT64,
        {ref},
        [](const auto& args, SelectionView rows, auto&) {
            auto out = std::make_shared<ColumnVector>(DataType::INT64,
                                                      rows.row_count());
            rows.ForEachSelected([&](size_t row) {
                if (args[0].IsValid(row)) {
                    out->template SetValueAt<int64_t>(
                        row, args[0].template ValueAt<int64_t>(row) + 1);
                } else {
                    out->nullAt(row);
                }
            });
            return ReadOnlyVectorView::CopyOf(out, rows);
        });
    auto results = ExprSet({add_one, filter}).Eval(selected.view(), ctx);
    EXPECT_EQ(loads, 1);
    RowVector projected(4, {results[0].AsVector(), results[1].AsVector()});
    auto compact = SelectRows(
        projected,
        std::make_shared<RowMapping>(std::vector<vector_size_t>{1, 3}));
    BatchInput downstream(*compact);
    EvalCtx next(downstream);
    auto output = Ref(0, DataType::INT64)->Eval(SelectionView::All(2), next);
    EXPECT_EQ(output.ValueAt<int64_t>(0), 21);
    EXPECT_EQ(output.ValueAt<int64_t>(1), 31);

    // Same expression, fresh batch: no retained cursor/selection/loaded state.
    auto another = Input(2, {DataType::INT64});
    another.BindValues(0, Values<int64_t>(DataType::INT64, {10, 50}));
    EvalCtx next_batch(another);
    ExpectBools(filter->Eval(SelectionView::All(2), next_batch), {0, 1});
}

TEST(ScalarExprTest,
     RowShortCircuitSkipsThrowingFunctionButOtherRootStillEvaluates) {
    auto input = Input(4, {DataType::BOOL, DataType::INT64});
    input.BindValues(0, Bools({0, 1, 0, 1}));
    input.BindValues(1, Values<int64_t>(DataType::INT64, {0, 2, 0, 5}));
    EvalCtx ctx(input);
    std::vector<std::vector<size_t>> calls;
    auto guarded = MakeCall(
        DataType::BOOL,
        {Ref(1, DataType::INT64)},
        [&](const auto& args, SelectionView rows, auto&) {
            calls.push_back(Indices(rows));
            rows.ForEachSelected([&](size_t row) {
                if (args[0].template ValueAt<int64_t>(row) == 0) {
                    ThrowInfo(ExprInvalid, "unselected computation ran");
                }
            });
            return Bools({1, 1, 1, 1}).Restrict(rows);
        });
    auto root = MakeLogical(LogicalOp::And, {Ref(0, DataType::BOOL), guarded});
    ExpectBools(root->Eval(SelectionView::All(4), ctx), {0, 1, 0, 1});
    ASSERT_EQ(calls.size(), 1);
    EXPECT_EQ(calls[0], (std::vector<size_t>{1, 3}));
    // A sibling root does NOT inherit the first root's short-circuit selection.
    ExprSet set({root, guarded});
    EXPECT_THROW(set.Eval(SelectionView::All(4), ctx), SegcoreError);
    EXPECT_EQ(calls.back(), (std::vector<size_t>{0, 1, 2, 3}));
}

TEST(ScalarExprTest,
     FailedSecondRootKeepsOldOutputsAndSuccessfulInputCoverage) {
    auto input = Input(3, {DataType::INT64});
    int loads = 0, calls = 0;
    input.BindLoader(0, [&](SelectionView rows) {
        ++loads;
        return Values<int64_t>(DataType::INT64, {1, 2, 3}).Restrict(rows);
    });
    EvalCtx ctx(input);
    auto ref = Ref(0, DataType::INT64);
    auto second =
        MakeCall(DataType::INT64, {ref}, [&](const auto& args, auto, auto&) {
            if (++calls == 1) {
                ThrowInfo(InsufficientResource, "injected second root failure");
            }
            return args[0];
        });
    ExprSet set({ref, second});
    std::vector<ReadOnlyVectorView> output{
        Values<int64_t>(DataType::INT64, {99})};
    try {
        output = set.Eval(SelectionView::All(3), ctx);
        FAIL();
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), InsufficientResource);
    }
    ASSERT_EQ(output.size(), 1);
    EXPECT_EQ(output[0].ValueAt<int64_t>(0), 99);
    output = set.Eval(SelectionView::All(3), ctx);
    EXPECT_EQ(loads, 1);
    EXPECT_EQ(calls, 2);
    EXPECT_EQ(output[1].ValueAt<int64_t>(2), 3);
}

TEST(ScalarExprTest,
     RandomSparseThreeValuedLogicMatchesLegacyAcrossBitmapWords) {
    constexpr size_t count = 193;
    std::mt19937 rng(20261016);
    std::vector<int> a(count), b(count);
    auto rows = OwnedSelection::None(count);
    for (size_t row = 0; row < count; ++row) {
        a[row] = static_cast<int>(rng() % 3) - 1;
        b[row] = static_cast<int>(rng() % 3) - 1;
        if (rng() % 3 == 0) {
            rows.SetSelected(row);
        }
    }
    auto left = Bools(a), right = Bools(b);
    auto input = Input(count, {DataType::BOOL, DataType::BOOL});
    input.BindValues(0, left);
    input.BindValues(1, right);
    EvalCtx ctx(input);
    for (auto op : {LogicalOp::And, LogicalOp::Or}) {
        auto root =
            MakeLogical(op, {Ref(0, DataType::BOOL), Ref(1, DataType::BOOL)});
        auto result = root->Eval(rows.view(), ctx);
        auto lhs = MakeWritableCopy(left), rhs = MakeWritableCopy(right);
        if (op == LogicalOp::And) {
            common::ThreeValuedLogicOp::And(lhs.values, rhs.values);
        } else {
            common::ThreeValuedLogicOp::Or(lhs.values, rhs.values);
        }
        auto expected = ReadOnlyVectorView::CopyOf(lhs.values);
        for (size_t row = 0; row < count; ++row) {
            EXPECT_EQ(result.defined_rows().IsSelected(row),
                      rows.view().IsSelected(row));
            if (rows.view().IsSelected(row)) {
                EXPECT_EQ(Tri(result, row), Tri(expected, row));
            }
        }
    }
}

TEST(ScalarExprTest,
     LoaderFailureAfterPartialSuccessKeepsPreviouslyReturnedView) {
    auto input = Input(4, {DataType::INT64});
    int calls = 0;
    input.BindLoader(0, [&](SelectionView rows) {
        if (++calls == 2) {
            ThrowInfo(FileReadFailed, "injected incremental read failure");
        }
        return Values<int64_t>(DataType::INT64, {10, 20, 30, 40})
            .Restrict(rows);
    });
    EvalCtx ctx(input);
    auto ref = Ref(0, DataType::INT64);
    const auto selected = OwnedSelection::FromIndices(4, {1, 3});
    auto old = ref->Eval(selected.view(), ctx);
    EXPECT_THROW(ref->Eval(SelectionView::All(4), ctx), SegcoreError);
    EXPECT_EQ(old.ValueAt<int64_t>(1), 20);
    EXPECT_EQ(old.ValueAt<int64_t>(3), 40);
    EXPECT_FALSE(old.defined_rows().IsSelected(0));
    const auto next = ref->Eval(SelectionView::All(4), ctx);
    EXPECT_EQ(next.ValueAt<int64_t>(0), 10);
    EXPECT_EQ(calls, 3);
    EXPECT_FALSE(old.defined_rows().IsSelected(0));
}

TEST(ScalarExprTest, LegacyFailureKeepsOriginalErrorCode) {
    auto input = Input(3);
    EvalCtx ctx(input);
    auto call = MakeLegacyCall(
        DataType::BOOL, {}, +[](const RowVector&, VectorPtr& output) {
            output =
                std::make_shared<ConstantVector<bool>>(DataType::BOOL, 3, true);
            ThrowInfo(FileReadFailed,
                      "injected legacy error after output assignment");
        });
    try {
        call->Eval(SelectionView::All(3), ctx);
        FAIL();
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), FileReadFailed);
    }
}

TEST(ScalarExprTest, ReorderPreservesBarriersNotAndStableEqualTiers) {
    std::vector<int> calls;
    auto expensive = std::make_shared<OrderedBool>(1, calls);
    auto cheap = std::make_shared<OrderedBool>(2, calls);
    auto equal = std::make_shared<OrderedBool>(3, calls);
    auto barrier = std::make_shared<OrderedBool>(4, calls, true);
    auto classify = [&](const Expr* expr) -> ReorderProperty {
        if (expr == expensive.get())
            return {ReorderTier::String, true};
        if (expr == cheap.get() || expr == equal.get())
            return {ReorderTier::Numeric, true};
        return {};
    };
    auto input = Input(1);
    EvalCtx ctx(input);
    auto nested =
        MakeLogical(LogicalOp::And,
                    {expensive, MakeLogical(LogicalOp::And, {cheap, equal})});
    AdjustLogicalExpression(nested, classify)->Eval(SelectionView::All(1), ctx);
    EXPECT_EQ(calls, (std::vector<int>{2, 3, 1}));
    calls.clear();
    MakeNot(AdjustLogicalExpression(MakeNot(nested), classify))
        ->Eval(SelectionView::All(1), ctx);
    EXPECT_EQ(calls, (std::vector<int>{1, 2, 3}));
    calls.clear();
    auto guarded = MakeLogical(LogicalOp::And, {expensive, barrier, cheap});
    EXPECT_THROW(AdjustLogicalExpression(guarded, classify)
                     ->Eval(SelectionView::All(1), ctx),
                 SegcoreError);
    EXPECT_EQ(calls, (std::vector<int>{1, 4}));
}
TEST(ScalarExprTest, BulkComparisonsAndRangesAllNumericTypes) {
    auto check = []<typename T>(
                     DataType type, bool dense = false, bool sparse = true) {
        constexpr size_t n = 259;
        auto a = std::make_shared<std::vector<T>>(n);
        auto b = std::make_shared<std::vector<T>>(n);
        auto valid = OwnedSelection::All(n);
        auto rows = OwnedSelection::All(n);
        for (size_t i = 0; i < n; ++i) {
            (*a)[i] = T(int(i % 19) - 9);
            (*b)[i] = T(int(i % 13) - 6);
            if (i % 31 == 0)
                valid.SetSelected(i, false);
            if (sparse && i % 47 < 3)
                rows.SetSelected(i, false);
        }
        (*a)[64] = std::numeric_limits<T>::max();
        (*a)[65] = std::numeric_limits<T>::lowest();
        if constexpr (std::is_floating_point_v<T>) {
            (*a)[129] = std::numeric_limits<T>::quiet_NaN();
            (*a)[130] = std::numeric_limits<T>::infinity();
            (*b)[131] = -std::numeric_limits<T>::infinity();
        }
        auto left = ReadOnlyVectorView::BorrowImmutable<T>(
            type,
            a->data(),
            a,
            SelectionView::All(n),
            valid.view(),
            dense ? PrimitivePayloadAccess::AllDefinedRows
                  : PrimitivePayloadAccess::ValidRowsOnly);
        auto right = ReadOnlyVectorView::BorrowImmutable<T>(
            type, b->data(), b, SelectionView::All(n), SelectionView::All(n));
        // Deliberately different fragment boundaries and an unaligned slice.
        left = ReadOnlyVectorView::Segments(
            type,
            SelectionView::All(n),
            {{0, left.Slice(0, 67)}, {67, left.Slice(67, n - 67)}});
        right = ReadOnlyVectorView::Segments(
            type,
            SelectionView::All(n),
            {{0, right.Slice(0, 139)}, {139, right.Slice(139, n - 139)}});
        auto input = Input(n, {type, type});
        input.BindValues(0, left);
        input.BindValues(1, right);
        EvalCtx ctx(input);
        for (auto op : {CompareOp::Equal,
                        CompareOp::NotEqual,
                        CompareOp::Less,
                        CompareOp::LessEqual,
                        CompareOp::Greater,
                        CompareOp::GreaterEqual}) {
            auto compare = [op](T x, T y) {
                switch (op) {
                    case CompareOp::Equal:
                        return x == y;
                    case CompareOp::NotEqual:
                        return x != y;
                    case CompareOp::Less:
                        return x < y;
                    case CompareOp::LessEqual:
                        return x <= y;
                    case CompareOp::Greater:
                        return x > y;
                    default:
                        return x >= y;
                }
            };
            // Assert bulk eligibility directly, not only equality of results.
            BitmapResultBuilder builder(n);
            ASSERT_TRUE((ComparePrimitive<T, T>(
                left, right, rows.view(), op, builder, compare)));
            auto direct = std::move(builder).Finish(rows.view());
            for (int shape = 0; shape < 3; ++shape) {
                auto lhs = shape == 2 ? Literal<T>(type, T(2)) : Ref(0, type);
                auto rhs = shape == 1 ? Literal<T>(type, T(2)) : Ref(1, type);
                auto output =
                    MakeComparison(op, lhs, rhs)->Eval(rows.view(), ctx);
                for (size_t i = 0; i < n; ++i) {
                    ASSERT_EQ(output.defined_rows().IsSelected(i),
                              rows.view().IsSelected(i));
                    if (!rows.view().IsSelected(i))
                        continue;
                    const bool known = shape == 2 || valid.view().IsSelected(i);
                    ASSERT_EQ(output.IsValid(i), known);
                    if (known)
                        EXPECT_EQ(output.template ValueAt<bool>(i),
                                  compare(shape == 2 ? T(2) : (*a)[i],
                                          shape == 1 ? T(2) : (*b)[i]));
                    if (shape == 0)
                        EXPECT_EQ(Tri(output, i), Tri(direct, i));
                }
            }
        }
        auto lo = std::make_shared<ConstantVector<T>>(type, 1, T(-3));
        auto hi = std::make_shared<ConstantVector<T>>(type, 1, T(4));
        for (bool inc_lo : {false, true})
            for (bool inc_hi : {false, true}) {
                auto out = MakeRange(Ref(0, type), lo, hi, inc_lo, inc_hi)
                               ->Eval(rows.view(), ctx);
                rows.view().ForEachSelected([&](size_t i) {
                    EXPECT_EQ(out.IsValid(i), valid.view().IsSelected(i));
                    if (out.IsValid(i))
                        EXPECT_EQ(
                            out.template ValueAt<bool>(i),
                            (inc_lo ? (*a)[i] >= T(-3) : (*a)[i] > T(-3)) &&
                                (inc_hi ? (*a)[i] <= T(4) : (*a)[i] < T(4)));
                });
            }
    };
    check.template operator()<int8_t>(DataType::INT8);
    check.template operator()<int16_t>(DataType::INT16);
    check.template operator()<int32_t>(DataType::INT32);
    check.template operator()<int64_t>(DataType::INT64);
    check.template operator()<int64_t>(DataType::TIMESTAMPTZ);
    check.template operator()<float>(DataType::FLOAT);
    check.template operator()<double>(DataType::DOUBLE);
    for (bool sparse : {false, true}) {
        check.template operator()<int8_t>(DataType::INT8, true, sparse);
        check.template operator()<int16_t>(DataType::INT16, true, sparse);
        check.template operator()<int32_t>(DataType::INT32, true, sparse);
        check.template operator()<int64_t>(DataType::INT64, true, sparse);
        check.template operator()<int64_t>(DataType::TIMESTAMPTZ, true, sparse);
    }
}

TEST(ScalarExprTest,
     PrimitiveFallbackPreservesRepeatedRowsAndArithmeticErrors) {
    auto source = std::make_shared<ColumnVector>(DataType::INT64, 4);
    source->SetValueAt<int64_t>(0, std::numeric_limits<int64_t>::min());
    source->SetValueAt<int64_t>(1, 42);
    source->nullAt(2);
    source->SetValueAt<int64_t>(3, 9);
    auto flat = ReadOnlyVectorView::CopyOf(source);
    auto mapped = flat.Dictionary(
        std::make_shared<RowMapping>(std::vector<vector_size_t>{3, 2, 0, 3}));
    auto constant = [](int64_t value) {
        return std::make_shared<ConstantVector<int64_t>>(
            DataType::INT64, 1, value);
    };
    for (const auto& view : {flat, mapped}) {
        auto input = Input(4, {DataType::INT64});
        input.BindValues(0, view);
        EvalCtx ctx(input);
        auto rows = OwnedSelection::All(4);
        rows.SetSelected(view.encoding() == ValueEncoding::Dictionary ? 2 : 0,
                         false);
        auto expr = MakeArithmeticCompare(Ref(0, DataType::INT64),
                                          proto::plan::Div,
                                          CompareOp::Less,
                                          constant(-1),
                                          constant(-5));
        auto out = expr->Eval(rows.view(), ctx);
        rows.view().ForEachSelected([&](size_t row) {
            EXPECT_EQ(out.IsValid(row), view.IsValid(row));
            if (out.IsValid(row))
                EXPECT_EQ(out.ValueAt<bool>(row),
                          view.ValueAt<int64_t>(row) / -1 < -5);
        });
        EXPECT_THROW(expr->Eval(SelectionView::All(4), ctx), SegcoreError);
        auto terms = std::make_shared<ColumnVector>(DataType::INT64, 2);
        terms->SetValueAt<int64_t>(0, 9);
        terms->SetValueAt<int64_t>(1, 42);
        auto term =
            MakeTerm(Ref(0, DataType::INT64), terms)->Eval(rows.view(), ctx);
        rows.view().ForEachSelected([&](size_t row) {
            EXPECT_EQ(term.IsValid(row), view.IsValid(row));
            if (term.IsValid(row))
                EXPECT_TRUE(term.ValueAt<bool>(row));
        });
    }
}

TEST(ScalarExprTest,
     BulkLogicalMasksMatchThreeValuedOracleAcrossWordBoundaries) {
    constexpr size_t n = 273;
    std::vector<int> a(n + 5), b(n + 5), c(n + 5);
    for (size_t row = 0; row < n + 5; ++row) {
        a[row] = int(row % 3) - 1;
        b[row] = int((row / 3) % 3) - 1;
        c[row] = int((row / 9) % 3) - 1;
    }
    auto input = Input(n, {DataType::BOOL, DataType::BOOL, DataType::BOOL});
    input.BindValues(0, Bools(a).Slice(5, n));
    input.BindValues(1, Bools(b).Slice(5, n));
    input.BindValues(2, Bools(c).Slice(5, n));
    EvalCtx ctx(input);
    for (bool sparse : {false, true}) {
        auto rows = OwnedSelection::All(n);
        if (sparse)
            for (size_t row = 0; row < n; row += 5)
                rows.SetSelected(row, false);
        for (auto op : {LogicalOp::And, LogicalOp::Or}) {
            auto expr = MakeLogical(op,
                                    {Ref(0, DataType::BOOL),
                                     Ref(1, DataType::BOOL),
                                     Ref(2, DataType::BOOL)});
            auto out = expr->Eval(rows.view(), ctx);
            auto negated = MakeNot(expr)->Eval(rows.view(), ctx);
            for (size_t row = 0; row < n; ++row) {
                EXPECT_EQ(out.defined_rows().IsSelected(row),
                          rows.view().IsSelected(row));
                if (!rows.view().IsSelected(row))
                    continue;
                const std::vector<int> values{
                    a[row + 5], b[row + 5], c[row + 5]};
                const bool unknown =
                    std::find(values.begin(), values.end(), -1) != values.end();
                const int stop = op == LogicalOp::And ? 0 : 1;
                const bool decided =
                    std::find(values.begin(), values.end(), stop) !=
                    values.end();
                const int expected = decided ? stop : (unknown ? -1 : 1 - stop);
                EXPECT_EQ(Tri(out, row), expected);
                EXPECT_EQ(Tri(negated, row), expected < 0 ? -1 : 1 - expected);
            }
        }
    }
}

TEST(ScalarExprTest, BulkBooleanComparisonAndRangeKeepNulls) {
    constexpr size_t n = 259;
    std::vector<int> a(n), b(n);
    for (size_t row = 0; row < n; ++row) {
        a[row] = int(row % 3) - 1;
        b[row] = int(row / 3 % 3) - 1;
    }
    auto input = Input(n, {DataType::BOOL, DataType::BOOL});
    input.BindValues(0, Bools(a));
    input.BindValues(1, Bools(b));
    EvalCtx ctx(input);
    auto rows = OwnedSelection::All(n);
    rows.SetSelected(64, false);
    rows.SetSelected(129, false);
    for (auto op : {CompareOp::Equal,
                    CompareOp::NotEqual,
                    CompareOp::Less,
                    CompareOp::LessEqual,
                    CompareOp::Greater,
                    CompareOp::GreaterEqual}) {
        auto out =
            MakeComparison(op, Ref(0, DataType::BOOL), Ref(1, DataType::BOOL))
                ->Eval(rows.view(), ctx);
        rows.view().ForEachSelected([&](size_t row) {
            const bool valid = a[row] >= 0 && b[row] >= 0;
            EXPECT_EQ(out.IsValid(row), valid);
            if (!valid)
                return;
            bool expected;
            switch (op) {
                case CompareOp::Equal:
                    expected = a[row] == b[row];
                    break;
                case CompareOp::NotEqual:
                    expected = a[row] != b[row];
                    break;
                case CompareOp::Less:
                    expected = a[row] < b[row];
                    break;
                case CompareOp::LessEqual:
                    expected = a[row] <= b[row];
                    break;
                case CompareOp::Greater:
                    expected = a[row] > b[row];
                    break;
                default:
                    expected = a[row] >= b[row];
                    break;
            }
            EXPECT_EQ(out.ValueAt<bool>(row), expected);
        });
    }
    for (bool lo : {false, true})
        for (bool hi : {false, true})
            for (bool li : {false, true})
                for (bool ui : {false, true}) {
                    auto out = MakeRange(Ref(0, DataType::BOOL),
                                         std::make_shared<ConstantVector<bool>>(
                                             DataType::BOOL, 1, lo),
                                         std::make_shared<ConstantVector<bool>>(
                                             DataType::BOOL, 1, hi),
                                         li,
                                         ui)
                                   ->Eval(rows.view(), ctx);
                    rows.view().ForEachSelected([&](size_t row) {
                        EXPECT_EQ(out.IsValid(row), a[row] >= 0);
                        if (a[row] >= 0)
                            EXPECT_EQ(
                                out.ValueAt<bool>(row),
                                (li ? bool(a[row]) >= lo : bool(a[row]) > lo) &&
                                    (ui ? bool(a[row]) <= hi
                                        : bool(a[row]) < hi));
                    });
                }
}

TEST(ScalarExprTest, IntegerTermKernelsPreserveSelectionNullsAndEncodings) {
    auto check = []<typename T>(DataType type) {
        constexpr size_t n = 259;
        auto data = std::make_shared<std::vector<T>>(n);
        auto valid = OwnedSelection::All(n);
        for (size_t i = 0; i < n; ++i) {
            (*data)[i] = T(int(i % 101) - 50);
            if (i % 31 == 0)
                valid.SetSelected(i, false);
        }
        (*data)[64] = std::numeric_limits<T>::lowest();
        (*data)[65] = std::numeric_limits<T>::max();
        for (auto access : {PrimitivePayloadAccess::ValidRowsOnly,
                            PrimitivePayloadAccess::AllDefinedRows}) {
            const auto flat =
                ReadOnlyVectorView::BorrowImmutable<T>(type,
                                                       data->data(),
                                                       data,
                                                       SelectionView::All(n),
                                                       valid.view(),
                                                       access);
            const auto fragmented = ReadOnlyVectorView::Segments(
                type,
                SelectionView::All(n),
                {{0, flat.Slice(0, 67)}, {67, flat.Slice(67, n - 67)}});
            std::vector<vector_size_t> mapping(n);
            for (size_t i = 0; i < n; ++i)
                mapping[i] = (i * 7) % 131;  // reorder, repeat, and NULL
            const auto dictionary = flat.Dictionary(
                std::make_shared<RowMapping>(std::move(mapping)));
            const auto constant = ReadOnlyVectorView::CopyOf(
                std::make_shared<ConstantVector<T>>(type, n, T(5)));
            const auto nulls = ReadOnlyVectorView::CopyOf(
                std::make_shared<ConstantVector<T>>(type, n, T(5), n));
            for (size_t term_count : {0, 4, 1024}) {
                std::vector<T> terms;
                for (size_t i = 0; i < term_count; ++i)
                    terms.push_back(i % 4 == 0 ? T(5) : T(60 - int(i)));
                if (term_count) {
                    terms[1] = std::numeric_limits<T>::max();
                    terms[2] = std::numeric_limits<T>::lowest();
                }
                auto constants =
                    std::make_shared<ColumnVector>(type, terms.size());
                for (size_t i = 0; i < terms.size(); ++i)
                    constants->template SetValueAt<T>(i, terms[i]);
                for (const auto& view :
                     {flat, fragmented, dictionary, constant, nulls}) {
                    for (bool sparse : {false, true}) {
                        auto rows = OwnedSelection::All(n);
                        if (sparse)
                            for (size_t i = 0; i < n; ++i)
                                if (i % 47 < 3)
                                    rows.SetSelected(i, false);
                        auto input = Input(n, {type});
                        input.BindValues(0, view.Restrict(rows.view()));
                        EvalCtx ctx(input);
                        auto expr = MakeTerm(Ref(0, type), constants);
                        auto result = expr->Eval(rows.view(), ctx);
                        auto inverse = MakeNot(expr)->Eval(rows.view(), ctx);
                        for (size_t i = 0; i < n; ++i) {
                            ASSERT_EQ(result.defined_rows().IsSelected(i),
                                      rows.view().IsSelected(i));
                            if (!rows.view().IsSelected(i))
                                continue;
                            ASSERT_EQ(result.IsValid(i), view.IsValid(i));
                            ASSERT_EQ(inverse.IsValid(i), view.IsValid(i));
                            if (!view.IsValid(i))
                                continue;
                            const T value = view.template ValueAt<T>(i);
                            const bool expected =
                                std::find(terms.begin(), terms.end(), value) !=
                                terms.end();
                            ASSERT_EQ(result.template ValueAt<bool>(i),
                                      expected)
                                << "row " << i;
                            ASSERT_EQ(inverse.template ValueAt<bool>(i),
                                      !expected)
                                << "row " << i;
                        }
                    }
                }
            }
        }
    };
    check.template operator()<int8_t>(DataType::INT8);
    check.template operator()<int16_t>(DataType::INT16);
    check.template operator()<int32_t>(DataType::INT32);
    check.template operator()<int64_t>(DataType::INT64);
    check.template operator()<int64_t>(DataType::TIMESTAMPTZ);
}

TEST(ScalarExprTest, IntegerTermNeverReadsProtectedNullOrUnselectedSlots) {
    const auto page = size_t(sysconf(_SC_PAGESIZE));
    void* mapping = mmap(nullptr,
                         page * 2,
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS,
                         -1,
                         0);
    ASSERT_NE(mapping, MAP_FAILED);
    auto owner = std::shared_ptr<void>(
        mapping, [page](void* p) { munmap(p, page * 2); });
    constexpr size_t n = 67;
    auto* data =
        reinterpret_cast<int64_t*>(static_cast<char*>(mapping) + page) - n;
    for (size_t i = 0; i < n; ++i) data[i] = i;
    ASSERT_EQ(mprotect(static_cast<char*>(mapping) + page, page, PROT_NONE), 0);
    auto valid = OwnedSelection::All(n + 1);
    valid.SetSelected(n, false);
    auto constants = std::make_shared<ColumnVector>(DataType::INT64, 4);
    for (size_t i = 0; i < 4; ++i) constants->SetValueAt<int64_t>(i, 66 - i);
    for (bool null_slot : {false, true}) {
        auto rows = null_slot ? OwnedSelection::All(n + 1) : valid;
        auto view = ReadOnlyVectorView::BorrowImmutable<int64_t>(
            DataType::INT64,
            data,
            owner,
            rows.view(),
            valid.view(),
            null_slot ? PrimitivePayloadAccess::ValidRowsOnly
                      : PrimitivePayloadAccess::AllDefinedRows);
        auto input = Input(n + 1, {DataType::INT64});
        input.BindValues(0, view);
        EvalCtx ctx(input);
        auto result = MakeTerm(Ref(0, DataType::INT64), constants)
                          ->Eval(rows.view(), ctx);
        for (size_t i = 0; i < n; ++i)
            EXPECT_EQ(result.ValueAt<bool>(i), i >= 63);
        if (null_slot)
            EXPECT_FALSE(result.IsValid(n));
        else
            EXPECT_FALSE(result.defined_rows().IsSelected(n));
    }
}

TEST(ScalarExprTest, DenseComparisonKeepsTwoOperandNullsAndNot) {
    constexpr size_t n = 131;
    auto a = std::make_shared<std::vector<int64_t>>(n);
    auto b = std::make_shared<std::vector<int64_t>>(n);
    auto av = OwnedSelection::All(n), bv = OwnedSelection::All(n);
    for (size_t i = 0; i < n; ++i) {
        (*a)[i] = int64_t(i % 13) - 6;
        (*b)[i] = int64_t(i % 17) - 8;
        if (i % 7 == 0)
            av.SetSelected(i, false);
        if (i % 11 == 0)
            bv.SetSelected(i, false);
    }
    for (auto access : {PrimitivePayloadAccess::ValidRowsOnly,
                        PrimitivePayloadAccess::AllDefinedRows}) {
        auto left =
            ReadOnlyVectorView::BorrowImmutable<int64_t>(DataType::INT64,
                                                         a->data(),
                                                         a,
                                                         SelectionView::All(n),
                                                         av.view(),
                                                         access);
        auto right =
            ReadOnlyVectorView::BorrowImmutable<int64_t>(DataType::INT64,
                                                         b->data(),
                                                         b,
                                                         SelectionView::All(n),
                                                         bv.view(),
                                                         access);
        auto decoded = left.DecodePrimitive();
        ASSERT_TRUE(decoded);
        EXPECT_EQ(
            ChooseSelectionKernel(av.view(),
                                  decoded->spans[0].CanEvaluateAll(0, n),
                                  SelectionKernelCost::NumericComparison(),
                                  SelectionKernelPreference::Dense),
            access == PrimitivePayloadAccess::AllDefinedRows
                ? SelectionKernelMode::Dense
                : SelectionKernelMode::Sparse);
        EXPECT_EQ(
            ChooseSelectionKernel(av.view(),
                                  false,
                                  SelectionKernelCost::NumericComparison(),
                                  SelectionKernelPreference::Dense),
            SelectionKernelMode::Sparse);
        auto input = Input(n, {DataType::INT64, DataType::INT64});
        input.BindValues(0, left);
        input.BindValues(1, right);
        EvalCtx ctx(input);
        auto expr = MakeComparison(
            CompareOp::Less, Ref(0, DataType::INT64), Ref(1, DataType::INT64));
        auto result = expr->Eval(SelectionView::All(n), ctx);
        auto inverse = MakeNot(expr)->Eval(SelectionView::All(n), ctx);
        for (size_t i = 0; i < n; ++i) {
            const bool valid =
                av.view().IsSelected(i) && bv.view().IsSelected(i);
            EXPECT_EQ(result.IsValid(i), valid);
            EXPECT_EQ(inverse.IsValid(i), valid);
            if (valid) {
                EXPECT_EQ(result.ValueAt<bool>(i), (*a)[i] < (*b)[i]);
                EXPECT_EQ(inverse.ValueAt<bool>(i), !((*a)[i] < (*b)[i]));
            }
        }
    }
}

TEST(ScalarExprTest,
     DenseInputDoesNotAuthorizeArithmeticOnNullOrUnselectedRows) {
    auto values = std::make_shared<std::vector<int64_t>>(
        std::initializer_list<int64_t>{INT64_MIN, 12, INT64_MIN, 9});
    auto valid = OwnedSelection::FromIndices(4, {1, 2, 3});
    auto view = ReadOnlyVectorView::BorrowImmutable<int64_t>(
        DataType::INT64,
        values->data(),
        values,
        SelectionView::All(4),
        valid.view(),
        PrimitivePayloadAccess::AllDefinedRows);
    auto input = Input(4, {DataType::INT64});
    input.BindValues(0, view);
    EvalCtx ctx(input);
    auto constant = [](int64_t v) {
        return std::make_shared<ConstantVector<int64_t>>(DataType::INT64, 1, v);
    };
    auto expr = MakeArithmeticCompare(Ref(0, DataType::INT64),
                                      proto::plan::Div,
                                      CompareOp::Less,
                                      constant(-1),
                                      constant(-5));
    auto rows = OwnedSelection::FromIndices(4, {0, 1, 3});
    auto result = expr->Eval(rows.view(), ctx);
    EXPECT_FALSE(result.IsValid(0));
    EXPECT_TRUE(result.ValueAt<bool>(1));
    EXPECT_TRUE(result.ValueAt<bool>(3));
    EXPECT_THROW(result.IsValid(2), SegcoreError);
    EXPECT_THROW(expr->Eval(SelectionView::All(4), ctx), SegcoreError);
}

TEST(ScalarExprTest, ConservativeComparisonNeverReadsProtectedNullSlot) {
    const auto page = size_t(sysconf(_SC_PAGESIZE));
    void* mapping = mmap(nullptr,
                         page * 2,
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS,
                         -1,
                         0);
    ASSERT_NE(mapping, MAP_FAILED);
    auto owner = std::shared_ptr<void>(
        mapping, [page](void* p) { munmap(p, page * 2); });
    auto* data =
        reinterpret_cast<int64_t*>(static_cast<char*>(mapping) + page) - 1;
    *data = 5;
    ASSERT_EQ(mprotect(static_cast<char*>(mapping) + page, page, PROT_NONE), 0);
    auto valid = OwnedSelection::FromIndices(2, {0});
    auto values = ReadOnlyVectorView::BorrowImmutable<int64_t>(
        DataType::INT64, data, owner, SelectionView::All(2), valid.view());
    auto input = Input(2, {DataType::INT64});
    input.BindValues(0, values);
    EvalCtx ctx(input);
    auto expr = MakeComparison(CompareOp::Less,
                               Ref(0, DataType::INT64),
                               Literal<int64_t>(DataType::INT64, 10));
    auto result = expr->Eval(SelectionView::All(2), ctx);
    EXPECT_TRUE(result.ValueAt<bool>(0));
    EXPECT_FALSE(result.IsValid(1));
    auto range =
        MakeRange(
            Ref(0, DataType::INT64),
            std::make_shared<ConstantVector<int64_t>>(DataType::INT64, 1, 0),
            std::make_shared<ConstantVector<int64_t>>(DataType::INT64, 1, 10),
            true,
            false)
            ->Eval(SelectionView::All(2), ctx);
    EXPECT_TRUE(range.ValueAt<bool>(0));
    EXPECT_FALSE(range.IsValid(1));
}

TEST(ScalarExprTest, StringBatchesKeepMixedStorageNullsSparseRowsAndMappings) {
    constexpr size_t n = 137;
    const std::vector<std::string> samples{"",
                                           "a",
                                           "b",
                                           "bb",
                                           "\xE5\x8C\x97\xE4\xBA\xAC",
                                           std::string("b\0x", 3),
                                           std::string(512, 'b')};
    auto owner = std::make_shared<std::vector<std::string_view>>(n);
    auto valid = OwnedSelection::All(n);
    auto owned = std::make_shared<ColumnVector>(DataType::VARCHAR, n);
    for (size_t i = 0; i < n; ++i) {
        (*owner)[i] = samples[i % samples.size()];
        owned->SetValueAt<std::string>(i, samples[i % samples.size()]);
        if (i % 13 == 0) {
            valid.SetSelected(i, false);
            owned->nullAt(i);
        }
    }
    const auto borrowed = ReadOnlyVectorView::BorrowImmutable<std::string_view>(
        DataType::VARCHAR,
        owner->data(),
        owner,
        SelectionView::All(n),
        valid.view());
    const auto dense = ReadOnlyVectorView::BorrowImmutable<std::string_view>(
        DataType::VARCHAR,
        owner->data(),
        owner,
        SelectionView::All(n),
        valid.view(),
        PrimitivePayloadAccess::AllDefinedRows);
    const auto flat = ReadOnlyVectorView::CopyOf(owned);
    const auto fragmented = ReadOnlyVectorView::Segments(
        DataType::VARCHAR,
        SelectionView::All(n),
        {{0, borrowed.Slice(0, 65)}, {65, flat.Slice(65, n - 65)}});
    std::vector<vector_size_t> mapping(n);
    for (size_t i = 0; i < n; ++i) mapping[i] = (i * 3) % 67;
    const auto dictionary =
        fragmented.Dictionary(std::make_shared<RowMapping>(mapping));
    const auto rhs = ReadOnlyVectorView::Segments(
        DataType::VARCHAR,
        SelectionView::All(n),
        {{0, flat.Slice(0, 31)}, {31, borrowed.Slice(31, n - 31)}});
    for (const auto& view : {borrowed, dense, flat, fragmented, dictionary}) {
        for (bool sparse : {false, true}) {
            auto rows = OwnedSelection::All(n);
            if (sparse)
                for (size_t i = 1; i < n; i += 3) rows.SetSelected(i, false);
            auto input = Input(n, {DataType::VARCHAR, DataType::VARCHAR});
            input.BindValues(0, view);
            input.BindValues(1, rhs);
            EvalCtx ctx(input);
            const auto ref = Ref(0, DataType::VARCHAR);
            const auto literal = Literal<std::string>(DataType::VARCHAR, "b");
            for (auto op : {CompareOp::Equal,
                            CompareOp::NotEqual,
                            CompareOp::Less,
                            CompareOp::LessEqual,
                            CompareOp::Greater,
                            CompareOp::GreaterEqual,
                            CompareOp::Prefix}) {
                for (int operands = 0; operands < 3; ++operands) {
                    auto expr = MakeComparison(op,
                                               operands == 1 ? literal : ref,
                                               operands == 0 ? literal
                                               : operands == 1
                                                   ? ref
                                                   : Ref(1, DataType::VARCHAR));
                    const auto out = expr->Eval(rows.view(), ctx);
                    for (size_t i = 0; i < n; ++i) {
                        EXPECT_EQ(out.defined_rows().IsSelected(i),
                                  rows.view().IsSelected(i));
                        if (!rows.view().IsSelected(i))
                            continue;
                        const bool known = view.IsValid(i) &&
                                           (operands != 2 || rhs.IsValid(i));
                        ASSERT_EQ(out.IsValid(i), known);
                        if (!known)
                            continue;
                        const std::string a(operands == 1
                                                ? std::string_view("b")
                                                : view.StringViewAt(i));
                        const std::string b(
                            operands == 0   ? std::string_view("b")
                            : operands == 1 ? view.StringViewAt(i)
                                            : rhs.StringViewAt(i));
                        const int cmp = a.compare(b);
                        const bool expected =
                            op == CompareOp::Prefix      ? a.starts_with(b)
                            : op == CompareOp::Equal     ? cmp == 0
                            : op == CompareOp::NotEqual  ? cmp != 0
                            : op == CompareOp::Less      ? cmp < 0
                            : op == CompareOp::LessEqual ? cmp <= 0
                            : op == CompareOp::Greater   ? cmp > 0
                                                         : cmp >= 0;
                        EXPECT_EQ(out.ValueAt<bool>(i), expected);
                    }
                }
            }
            auto constants =
                std::make_shared<ColumnVector>(DataType::VARCHAR, 3);
            constants->SetValueAt<std::string>(0, samples[2]);
            constants->SetValueAt<std::string>(1, samples[5]);
            constants->SetValueAt<std::string>(2, samples[6]);
            auto terms = MakeTerm(ref, constants);
            auto range =
                MakeRange(ref,
                          std::make_shared<ConstantVector<std::string>>(
                              DataType::VARCHAR, 1, "a"),
                          std::make_shared<ConstantVector<std::string>>(
                              DataType::VARCHAR, 1, "c"),
                          true,
                          false);
            for (auto expr : {terms,
                              range,
                              MakeStringMatch(ref, StringOp::Prefix, "b"),
                              MakeStringMatch(ref, StringOp::Like, "b%")}) {
                const auto out = expr->Eval(rows.view(), ctx);
                rows.view().ForEachSelected([&](size_t i) {
                    ASSERT_EQ(out.IsValid(i), view.IsValid(i));
                    if (!view.IsValid(i))
                        return;
                    const auto value = view.StringViewAt(i);
                    const bool expected =
                        expr == terms
                            ? (value == samples[2] || value == samples[5] ||
                               value == samples[6])
                        : expr == range ? (value >= "a" && value < "c")
                                        : value.starts_with("b");
                    EXPECT_EQ(out.ValueAt<bool>(i), expected);
                });
            }
        }
    }
}

TEST(ScalarExprTest, StringMatchersCancelInsideBatchAndCanRetry) {
    constexpr size_t n = 2049;
    const auto values = Values<std::string>(
        DataType::VARCHAR, std::vector<std::string>(n, "value_1"));
    const auto ref = Ref(0, DataType::VARCHAR);
    auto expressions =
        std::vector<ExprPtr>{MakeStringMatch(ref, StringOp::Prefix, "value"),
                             MakeStringMatch(ref, StringOp::Suffix, "1"),
                             MakeStringMatch(ref, StringOp::Contains, "ue_"),
                             MakeStringMatch(ref, StringOp::Like, "value%"),
                             MakeStringMatch(ref, StringOp::Regex, "value.*")};
    const auto fused = FuseAdjacentStringMatches(
        LogicalOp::And, {expressions[0], expressions[1]});
    ASSERT_EQ(fused.size(), 1);
    expressions.push_back(fused.front());
    for (const auto& expr : expressions) {
        auto input = Input(n, {DataType::VARCHAR});
        bool loaded = false;
        input.BindLoader(0, [&](SelectionView rows) {
            loaded = true;
            return values.Restrict(rows);
        });
        size_t checks_after_load = 0;
        EvalCtx cancelled(input, [&] {
            if (loaded && ++checks_after_load == 3)
                ThrowInfo(FollyCancel, "cancel string kernel inside batch");
        });
        try {
            expr->Eval(SelectionView::All(n), cancelled);
            FAIL() << "cancellation was not checked inside the string kernel";
        } catch (const SegcoreError& error) {
            EXPECT_EQ(error.get_error_code(), FollyCancel);
        }
        EXPECT_EQ(checks_after_load, 3);
        // A partly written private bitmap must not become the retry result.
        EvalCtx retry(input);
        const auto result = expr->Eval(SelectionView::All(n), retry);
        for (size_t row = 0; row < n; ++row) {
            ASSERT_TRUE(result.IsValid(row));
            EXPECT_TRUE(result.ValueAt<bool>(row));
        }
    }
}

TEST(ScalarExprTest, StringBatchesNeverReadProtectedNullOrUnselectedSlots) {
    const auto page = size_t(sysconf(_SC_PAGESIZE));
    void* mapping = mmap(nullptr,
                         page * 2,
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS,
                         -1,
                         0);
    ASSERT_NE(mapping, MAP_FAILED);
    auto owner = std::shared_ptr<void>(
        mapping, [page](void* p) { munmap(p, page * 2); });
    constexpr size_t n = 67;
    auto* data = reinterpret_cast<std::string_view*>(
                     static_cast<char*>(mapping) + page) -
                 n;
    for (size_t i = 0; i < n; ++i)
        new (data + i) std::string_view(i % 2 ? "abc" : "def");
    ASSERT_EQ(mprotect(static_cast<char*>(mapping) + page, page, PROT_NONE), 0);
    auto valid = OwnedSelection::All(n + 1);
    valid.SetSelected(n, false);
    for (bool null_slot : {false, true}) {
        auto rows = null_slot ? OwnedSelection::All(n + 1) : valid;
        auto view = ReadOnlyVectorView::BorrowImmutable<std::string_view>(
            DataType::VARCHAR,
            data,
            owner,
            rows.view(),
            valid.view(),
            null_slot ? PrimitivePayloadAccess::ValidRowsOnly
                      : PrimitivePayloadAccess::AllDefinedRows);
        auto input = Input(n + 1, {DataType::VARCHAR});
        input.BindValues(0, view);
        EvalCtx ctx(input);
        auto ref = Ref(0, DataType::VARCHAR);
        auto constants = std::make_shared<ConstantVector<std::string>>(
            DataType::VARCHAR, 1, "abc");
        for (auto expr :
             {MakeComparison(CompareOp::Equal, ref, MakeLiteral(constants)),
              MakeTerm(ref, constants),
              MakeStringMatch(ref, StringOp::Prefix, "a")}) {
            auto out = expr->Eval(rows.view(), ctx);
            for (size_t i = 0; i < n; ++i)
                EXPECT_EQ(out.ValueAt<bool>(i), i % 2 != 0);
            if (null_slot)
                EXPECT_FALSE(out.IsValid(n));
            else
                EXPECT_FALSE(out.defined_rows().IsSelected(n));
        }
    }
}

TEST(ScalarExprTest, ArithmeticBatchDispatchMatchesScalarOracle) {
    constexpr size_t n = 271;
    auto column = std::make_shared<ColumnVector>(DataType::INT64, n);
    for (size_t i = 0; i < n; ++i) {
        column->SetValueAt<int64_t>(i, int64_t(i) - 135);
        if (i % 11 == 0)
            column->nullAt(i);
    }
    const auto view = ReadOnlyVectorView::CopyOf(column);
    auto input = Input(n, {DataType::INT64});
    input.BindValues(0, view);
    EvalCtx ctx(input);
    auto constant = [](int64_t v) {
        return std::make_shared<ConstantVector<int64_t>>(DataType::INT64, 1, v);
    };
    for (auto op : {proto::plan::Add,
                    proto::plan::Sub,
                    proto::plan::Mul,
                    proto::plan::Div,
                    proto::plan::Mod,
                    proto::plan::BitAnd,
                    proto::plan::BitOr,
                    proto::plan::BitXor,
                    proto::plan::Shl,
                    proto::plan::Shr}) {
        for (auto cmp : {CompareOp::Equal,
                         CompareOp::NotEqual,
                         CompareOp::Less,
                         CompareOp::LessEqual,
                         CompareOp::Greater,
                         CompareOp::GreaterEqual}) {
            auto expr = MakeArithmeticCompare(
                Ref(0, DataType::INT64), op, cmp, constant(2), constant(10));
            for (bool sparse : {false, true}) {
                auto rows = OwnedSelection::All(n);
                if (sparse)
                    for (size_t i = 0; i < n; i += 3)
                        rows.SetSelected(i, false);
                auto result = expr->Eval(rows.view(), ctx);
                for (size_t i = 0; i < n; ++i) {
                    if (!rows.view().IsSelected(i)) {
                        EXPECT_FALSE(result.defined_rows().IsSelected(i));
                        continue;
                    }
                    EXPECT_EQ(result.IsValid(i), view.IsValid(i));
                    if (!view.IsValid(i))
                        continue;
                    auto v = int64_t(i) - 135;
                    int64_t actual = 0;
                    switch (op) {
                        case proto::plan::Add:
                            actual = v + 2;
                            break;
                        case proto::plan::Sub:
                            actual = v - 2;
                            break;
                        case proto::plan::Mul:
                            actual = v * 2;
                            break;
                        case proto::plan::Div:
                            actual = v / 2;
                            break;
                        case proto::plan::Mod:
                            actual = v % 2;
                            break;
                        case proto::plan::BitAnd:
                            actual = v & 2;
                            break;
                        case proto::plan::BitOr:
                            actual = v | 2;
                            break;
                        case proto::plan::BitXor:
                            actual = v ^ 2;
                            break;
                        case proto::plan::Shl:
                            actual = std::bit_cast<int64_t>(uint64_t(v) << 2);
                            break;
                        case proto::plan::Shr:
                            actual = v >> 2;
                            break;
                        default:
                            FAIL();
                    }
                    bool expected = false;
                    switch (cmp) {
                        case CompareOp::Equal:
                            expected = actual == 10;
                            break;
                        case CompareOp::NotEqual:
                            expected = actual != 10;
                            break;
                        case CompareOp::Less:
                            expected = actual < 10;
                            break;
                        case CompareOp::LessEqual:
                            expected = actual <= 10;
                            break;
                        case CompareOp::Greater:
                            expected = actual > 10;
                            break;
                        case CompareOp::GreaterEqual:
                            expected = actual >= 10;
                            break;
                    }
                    EXPECT_EQ(result.ValueAt<bool>(i), expected)
                        << int(op) << ":" << i;
                }
            }
        }
    }
}

TEST(ScalarExprTest, FloatingTermsHandleNaNInfinitySignedZeroAndDuplicates) {
    auto check = []<typename T>(DataType type) {
        constexpr size_t n = 273;
        const std::vector<T> pattern{T(0),
                                     -T(0),
                                     T(1),
                                     std::numeric_limits<T>::quiet_NaN(),
                                     std::numeric_limits<T>::infinity(),
                                     -std::numeric_limits<T>::infinity()};
        auto column = std::make_shared<ColumnVector>(type, n);
        for (size_t i = 0; i < n; ++i) {
            column->template SetValueAt<T>(i, pattern[i % pattern.size()]);
            if (i % 17 == 0)
                column->nullAt(i);
        }
        auto values = ReadOnlyVectorView::CopyOf(column);
        auto constants = std::make_shared<ColumnVector>(type, 5);
        std::vector<T> terms{std::numeric_limits<T>::quiet_NaN(),
                             T(0),
                             -T(0),
                             T(1),
                             std::numeric_limits<T>::infinity()};
        for (size_t i = 0; i < terms.size(); ++i)
            constants->template SetValueAt<T>(i, terms[i]);
        std::vector<vector_size_t> indices(n);
        for (size_t i = 0; i < n; ++i) indices[i] = n - 1 - i;
        for (bool dictionary : {false, true}) {
            auto view =
                dictionary
                    ? values.Dictionary(std::make_shared<RowMapping>(indices))
                    : values;
            auto input = Input(n, {type});
            input.BindValues(0, view);
            EvalCtx ctx(input);
            auto out = MakeTerm(Ref(0, type), constants)
                           ->Eval(SelectionView::All(n), ctx);
            for (size_t i = 0; i < n; ++i) {
                EXPECT_EQ(out.IsValid(i), view.IsValid(i));
                if (!view.IsValid(i))
                    continue;
                auto value = view.template ValueAt<T>(i);
                const bool expected =
                    std::any_of(terms.begin(), terms.end(), [&](T term) {
                        return term == value;
                    });
                EXPECT_EQ(out.template ValueAt<bool>(i), expected) << i;
            }
        }
    };
    check.template operator()<float>(DataType::FLOAT);
    check.template operator()<double>(DataType::DOUBLE);
}

TEST(ScalarExprTest, ArithmeticWrapMatchesAcrossSimdHeadsTailsAndDictionary) {
    constexpr size_t n = 273;
    auto column = std::make_shared<ColumnVector>(DataType::INT64, n);
    for (size_t i = 0; i < n; ++i)
        column->SetValueAt<int64_t>(i, i % 2 ? INT64_MIN : INT64_MAX);
    auto flat = ReadOnlyVectorView::CopyOf(column);
    std::vector<vector_size_t> indices(n);
    for (size_t i = 0; i < n; ++i) indices[i] = i;
    for (auto view :
         {flat, flat.Dictionary(std::make_shared<RowMapping>(indices))}) {
        auto input = Input(n, {DataType::INT64});
        input.BindValues(0, view);
        EvalCtx ctx(input);
        auto constant = [](int64_t v) {
            return std::make_shared<ConstantVector<int64_t>>(
                DataType::INT64, 1, v);
        };
        for (auto op : {proto::plan::Add, proto::plan::Sub, proto::plan::Mul}) {
            auto expr = MakeArithmeticCompare(Ref(0, DataType::INT64),
                                              op,
                                              CompareOp::Less,
                                              constant(2),
                                              constant(0));
            auto rows = OwnedSelection::All(n);
            for (size_t i = 0; i < n; i += 71) rows.SetSelected(i, false);
            auto output = expr->Eval(rows.view(), ctx);
            rows.view().ForEachSelected([&](size_t row) {
                const uint64_t v = uint64_t(view.ValueAt<int64_t>(row));
                const auto wrapped =
                    std::bit_cast<int64_t>(op == proto::plan::Add   ? v + 2
                                           : op == proto::plan::Sub ? v - 2
                                                                    : v * 2);
                EXPECT_EQ(output.ValueAt<bool>(row), wrapped < 0)
                    << row << ":" << int(op);
            });
        }
    }
}

TEST(ScalarExprTest, FullLegacyCallDetachesEscapedArgumentsAndResults) {
    static ColumnVectorPtr escaped;
    auto input = Input(3, {DataType::INT64});
    const auto source = Values<int64_t>(DataType::INT64, {2, 4, 6});
    input.BindValues(0, source);
    EvalCtx ctx(input);
    auto call = MakeLegacyCall(
        DataType::INT64,
        {Ref(0, DataType::INT64)},
        +[](const RowVector& args, VectorPtr& output) {
            escaped = std::dynamic_pointer_cast<ColumnVector>(args.child(0));
            escaped->SetValueAt<int64_t>(0, 7);
            escaped->nullAt(1);
            output = escaped;
        });
    const auto result = call->Eval(SelectionView::All(3), ctx);
    EXPECT_EQ(source.ValueAt<int64_t>(0), 2);
    EXPECT_TRUE(source.IsValid(1));
    EXPECT_EQ(result.ValueAt<int64_t>(0), 7);
    EXPECT_FALSE(result.IsValid(1));
    escaped->SetValueAt<int64_t>(0, 99);
    escaped->SetValueAt<int64_t>(1, 99);
    EXPECT_EQ(result.ValueAt<int64_t>(0), 7);
    EXPECT_FALSE(result.IsValid(1));
    escaped.reset();
}

TEST(ScalarExprTest, BoolMasksDoNotReadProtectedNullPayload) {
    const auto page = size_t(sysconf(_SC_PAGESIZE));
    void* mapping = mmap(nullptr,
                         page * 2,
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS,
                         -1,
                         0);
    ASSERT_NE(mapping, MAP_FAILED);
    auto owner = std::shared_ptr<void>(
        mapping, [page](void* p) { munmap(p, page * 2); });
    constexpr size_t n = 67;
    auto* data =
        reinterpret_cast<bool*>(static_cast<char*>(mapping) + page) - n;
    for (size_t i = 0; i < n; ++i) data[i] = i % 2;
    ASSERT_EQ(mprotect(static_cast<char*>(mapping) + page, page, PROT_NONE), 0);
    auto valid = OwnedSelection::All(n + 1);
    valid.SetSelected(n, false);
    auto view = ReadOnlyVectorView::BorrowImmutable<bool>(
        DataType::BOOL, data, owner, SelectionView::All(n + 1), valid.view());
    auto input = Input(n + 1, {DataType::BOOL});
    input.BindValues(0, view);
    EvalCtx ctx(input);
    auto terms =
        std::make_shared<ConstantVector<bool>>(DataType::BOOL, 1, true);
    const auto result = MakeTerm(Ref(0, DataType::BOOL), terms)
                            ->Eval(SelectionView::All(n + 1), ctx);
    for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(result.ValueAt<bool>(i), i % 2 != 0);
    EXPECT_FALSE(result.IsValid(n));
}

TEST(ScalarExprTest,
     DenseFloatingKernelsPreserveNaNNullsAndCheckedConversions) {
    auto check = []<typename T>(DataType type) {
        constexpr size_t n = 273;
        auto storage = std::make_shared<std::vector<T>>(n);
        auto valid = OwnedSelection::All(n);
        for (size_t i = 0; i < n; ++i) {
            (*storage)[i] = i % 7 == 0   ? std::numeric_limits<T>::quiet_NaN()
                            : i % 7 == 1 ? std::numeric_limits<T>::infinity()
                                         : T(int(i % 19) - 9);
            if (i % 11 == 0)
                valid.SetSelected(i, false);
        }
        auto plain =
            ReadOnlyVectorView::BorrowImmutable<T>(type,
                                                   storage->data(),
                                                   storage,
                                                   SelectionView::All(n),
                                                   valid.view());
        auto dense = ReadOnlyVectorView::BorrowImmutable<T>(
            type,
            storage->data(),
            storage,
            SelectionView::All(n),
            valid.view(),
            PrimitivePayloadAccess::AllDefinedRows);
        auto literal = [&](T value) {
            return std::make_shared<ConstantVector<T>>(type, 1, value);
        };
        auto ref = Ref(0, type);
        std::vector<ExprPtr> expressions{MakeTerm(ref, literal(T(1)))};
        for (auto cmp : {CompareOp::Equal,
                         CompareOp::NotEqual,
                         CompareOp::Less,
                         CompareOp::LessEqual,
                         CompareOp::Greater,
                         CompareOp::GreaterEqual}) {
            expressions.push_back(
                MakeComparison(cmp, ref, MakeLiteral(literal(T(1)))));
            for (auto op :
                 {proto::plan::Add, proto::plan::Sub, proto::plan::Mul})
                expressions.push_back(MakeArithmeticCompare(
                    ref, op, cmp, literal(T(2)), literal(T(1))));
        }
        for (bool sparse : {false, true}) {
            auto rows = OwnedSelection::All(n);
            if (sparse)
                for (size_t i = 0; i < n; i += 3) rows.SetSelected(i, false);
            auto a = Input(n, {type}), b = Input(n, {type});
            a.BindValues(0, plain);
            b.BindValues(0, dense);
            EvalCtx ca(a), cb(b);
            for (const auto& expr : expressions) {
                auto expected = expr->Eval(rows.view(), ca),
                     actual = expr->Eval(rows.view(), cb);
                rows.view().ForEachSelected([&](size_t row) {
                    EXPECT_EQ(actual.IsValid(row), expected.IsValid(row));
                    if (expected.IsValid(row))
                        EXPECT_EQ(actual.template ValueAt<bool>(row),
                                  expected.template ValueAt<bool>(row));
                });
            }
        }
        auto guarded =
            std::make_shared<std::vector<T>>(std::initializer_list<T>{
                std::numeric_limits<T>::quiet_NaN(), T(1.25)});
        auto only_second = OwnedSelection::FromIndices(2, {1});
        auto view = ReadOnlyVectorView::BorrowImmutable<T>(
            type,
            guarded->data(),
            guarded,
            SelectionView::All(2),
            only_second.view(),
            PrimitivePayloadAccess::AllDefinedRows);
        auto input = Input(2, {type});
        input.BindValues(0, view);
        EvalCtx ctx(input);
        auto expr = MakeArithmeticCompare(ref,
                                          proto::plan::Mod,
                                          CompareOp::Less,
                                          literal(T(2.5)),
                                          literal(T(3)),
                                          true);
        auto result = expr->Eval(SelectionView::All(2), ctx);
        EXPECT_FALSE(result.IsValid(0));
        EXPECT_TRUE(result.template ValueAt<bool>(1));
    };
    check.template operator()<float>(DataType::FLOAT);
    check.template operator()<double>(DataType::DOUBLE);
}

TEST(ScalarExprTest, DenseMixedComparisonsPreservePromotionsAndNulls) {
    constexpr size_t n = 273;
    auto a = std::make_shared<std::vector<int64_t>>(n);
    auto b = std::make_shared<std::vector<double>>(n);
    auto av = OwnedSelection::All(n), bv = OwnedSelection::All(n);
    for (size_t i = 0; i < n; ++i) {
        (*a)[i] = i % 3 == 0 ? int64_t{9007199254740993LL} : int64_t(i) - 100;
        (*b)[i] = i % 3 == 0   ? 9007199254740992.0
                  : i % 3 == 1 ? std::numeric_limits<double>::quiet_NaN()
                               : double(i) - 99.5;
        if (i % 11 == 0)
            av.SetSelected(i, false);
        if (i % 13 == 0)
            bv.SetSelected(i, false);
    }
    auto input = Input(n, {DataType::INT64, DataType::DOUBLE});
    input.BindValues(0,
                     ReadOnlyVectorView::BorrowImmutable<int64_t>(
                         DataType::INT64,
                         a->data(),
                         a,
                         SelectionView::All(n),
                         av.view(),
                         PrimitivePayloadAccess::AllDefinedRows));
    input.BindValues(1,
                     ReadOnlyVectorView::BorrowImmutable<double>(
                         DataType::DOUBLE,
                         b->data(),
                         b,
                         SelectionView::All(n),
                         bv.view(),
                         PrimitivePayloadAccess::AllDefinedRows));
    EvalCtx ctx(input);
    for (auto op : {CompareOp::Equal,
                    CompareOp::NotEqual,
                    CompareOp::Less,
                    CompareOp::LessEqual,
                    CompareOp::Greater,
                    CompareOp::GreaterEqual}) {
        auto expr = MakeComparison(
            op, Ref(0, DataType::INT64), Ref(1, DataType::DOUBLE));
        for (bool sparse : {false, true}) {
            auto rows = OwnedSelection::All(n);
            if (sparse)
                for (size_t i = 0; i < n; i += 5) rows.SetSelected(i, false);
            auto result = expr->Eval(rows.view(), ctx);
            rows.view().ForEachSelected([&](size_t i) {
                const bool valid =
                    av.view().IsSelected(i) && bv.view().IsSelected(i);
                ASSERT_EQ(result.IsValid(i), valid);
                if (!valid)
                    return;
                const auto x = (*a)[i];
                const auto y = (*b)[i];
                const bool expected = op == CompareOp::Equal       ? x == y
                                      : op == CompareOp::NotEqual  ? x != y
                                      : op == CompareOp::Less      ? x < y
                                      : op == CompareOp::LessEqual ? x <= y
                                      : op == CompareOp::Greater   ? x > y
                                                                   : x >= y;
                EXPECT_EQ(result.ValueAt<bool>(i), expected);
            });
        }
    }
}

TEST(ScalarExprTest, AdaptiveSelectionUsesCostAndNeverOverridesEligibility) {
    constexpr size_t n = 8192;
    auto scattered = OwnedSelection::None(n),
         clustered = OwnedSelection::None(n);
    for (size_t i = 0; i < n; ++i) {
        if (i % 10 == 0)
            scattered.SetSelected(i);
        if (i < 819)
            clustered.SetSelected(i);
    }
    const auto cheap = SelectionKernelCost::NumericComparison();
    const SelectionKernelCost expensive{16, 20, 32, 2};
    EXPECT_EQ(ChooseSelectionKernel(scattered.view(), true, cheap),
              SelectionKernelMode::Dense);
    EXPECT_EQ(ChooseSelectionKernel(scattered.view(), false, cheap),
              SelectionKernelMode::Sparse);
    EXPECT_EQ(ChooseSelectionKernel(scattered.view(), true, expensive),
              SelectionKernelMode::Sparse);
    EXPECT_EQ(ChooseSelectionKernel(clustered.view(), true, cheap),
              SelectionKernelMode::Ranges);
    EXPECT_EQ(
        ChooseSelectionKernel(
            scattered.view(), false, cheap, SelectionKernelPreference::Dense),
        SelectionKernelMode::Sparse);
    EXPECT_EQ(ChooseSelectionKernel(SelectionView::All(n), true, cheap),
              SelectionKernelMode::Ranges);
    EXPECT_EQ(ChooseSelectionKernel(SelectionView::None(n), true, cheap),
              SelectionKernelMode::Ranges);
}

TEST(ScalarExprTest, AdaptiveAndFixedComparisonModesPreserveCoverageAndNulls) {
    constexpr size_t n = 515;
    auto data = std::make_shared<std::vector<int64_t>>(n);
    auto valid = OwnedSelection::All(n);
    for (size_t i = 0; i < n; ++i) {
        (*data)[i] = int64_t(i % 21) - 10;
        if (i % 17 == 0)
            valid.SetSelected(i, false);
    }
    for (int pattern = 0; pattern < 4; ++pattern) {
        auto rows = OwnedSelection::None(n);
        for (size_t i = 0; i < n; ++i)
            if (pattern == 0 || (pattern == 1 && i % 10 == 0) ||
                (pattern == 2 && i >= 127 && i < 287) ||
                (pattern == 3 && i == 512))
                rows.SetSelected(i);
        for (bool complete : {false, true}) {
            auto defined = complete ? OwnedSelection::All(n)
                                    : OwnedSelection::CopyOf(rows.view());
            auto input_valid =
                OwnedSelection::Intersect(valid.view(), defined.view());
            auto source = ReadOnlyVectorView::BorrowImmutable<int64_t>(
                DataType::INT64,
                data->data(),
                data,
                defined.view(),
                input_valid.view(),
                PrimitivePayloadAccess::AllDefinedRows);
            // Restriction and slicing may narrow output, never grant payload coverage.
            auto left = source.Restrict(rows.view()).Slice(1, n - 2);
            auto requested = rows.view().Slice(1, n - 2);
            auto right = ReadOnlyVectorView::CopyOf(
                             std::make_shared<ConstantVector<int64_t>>(
                                 DataType::INT64, 1, 0))
                             .Broadcast(requested);
            for (auto mode : {SelectionKernelPreference::Adaptive,
                              SelectionKernelPreference::Ranges,
                              SelectionKernelPreference::Sparse,
                              SelectionKernelPreference::Dense}) {
                for (auto op : {CompareOp::Less,
                                CompareOp::LessEqual,
                                CompareOp::Equal,
                                CompareOp::NotEqual,
                                CompareOp::Greater,
                                CompareOp::GreaterEqual}) {
                    auto compare = [op](int64_t a, int64_t b) {
                        switch (op) {
                            case CompareOp::Less:
                                return a < b;
                            case CompareOp::LessEqual:
                                return a <= b;
                            case CompareOp::Equal:
                                return a == b;
                            case CompareOp::NotEqual:
                                return a != b;
                            case CompareOp::Greater:
                                return a > b;
                            default:
                                return a >= b;
                        }
                    };
                    BitmapResultBuilder out(n - 2);
                    ASSERT_TRUE((ComparePrimitive<int64_t, int64_t>(
                        left, right, requested, op, out, compare, mode)));
                    auto result = std::move(out).Finish(requested);
                    EXPECT_TRUE(result.defined_rows().Equals(requested));
                    for (size_t j = 0; j < n - 2; ++j) {
                        if (!requested.IsSelected(j))
                            continue;
                        EXPECT_EQ(result.IsValid(j),
                                  valid.view().IsSelected(j + 1));
                        if (result.IsValid(j))
                            EXPECT_EQ(result.ValueAt<bool>(j),
                                      compare((*data)[j + 1], int64_t{0}));
                    }
                }
            }
        }
    }
}

TEST(ScalarExprTest,
     SparseBitmapWriterOnlyVisitsSelectedRowsAndPacksUnalignedBits) {
    BitmapResultBuilder out(200);
    out.SetRange(0, 200, true);
    auto rows = OwnedSelection::FromIndices(133, {0, 1, 63, 64, 65, 127, 132});
    size_t calls = 0;
    out.EvaluateSelected(3, rows.view(), [&](size_t row) {
        EXPECT_TRUE(rows.view().IsSelected(row));
        ++calls;
        return row % 2 == 0;
    });
    auto result = std::move(out).Finish(SelectionView::All(200));
    EXPECT_EQ(calls, 7);
    for (size_t row = 0; row < 200; ++row) {
        if (row < 3 || row >= 136) {
            EXPECT_TRUE(result.IsValid(row));
            EXPECT_TRUE(result.ValueAt<bool>(row));
        } else {
            EXPECT_EQ(result.IsValid(row), rows.view().IsSelected(row - 3));
            if (result.IsValid(row))
                EXPECT_EQ(result.ValueAt<bool>(row), (row - 3) % 2 == 0);
        }
    }
}

TEST(ScalarExprTest, AdaptiveComparisonCannotReadProtectedUndefinedPositions) {
    const auto page = size_t(sysconf(_SC_PAGESIZE));
    void* mapping = mmap(nullptr,
                         page * 2,
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS,
                         -1,
                         0);
    ASSERT_NE(mapping, MAP_FAILED);
    auto owner = std::shared_ptr<void>(
        mapping, [page](void* p) { munmap(p, page * 2); });
    auto* data = static_cast<int64_t*>(mapping);
    const auto n = page / sizeof(int64_t) * 2;
    const auto safe = n / 2;
    auto rows = OwnedSelection::None(n);
    for (size_t i = 0; i < safe; ++i) {
        data[i] = i;
        rows.SetSelected(i);
    }
    ASSERT_EQ(mprotect(static_cast<char*>(mapping) + page, page, PROT_NONE), 0);
    // Even producer-approved NULL slots do not authorize undefined positions.
    auto left = ReadOnlyVectorView::BorrowImmutable<int64_t>(
        DataType::INT64,
        data,
        owner,
        rows.view(),
        rows.view(),
        PrimitivePayloadAccess::AllDefinedRows);
    auto right =
        ReadOnlyVectorView::CopyOf(
            std::make_shared<ConstantVector<int64_t>>(DataType::INT64, 1, 10))
            .Broadcast(rows.view());
    for (auto preference : {SelectionKernelPreference::Adaptive,
                            SelectionKernelPreference::Dense}) {
        BitmapResultBuilder out(n);
        ASSERT_TRUE((ComparePrimitive<int64_t, int64_t>(
            left,
            right,
            rows.view(),
            CompareOp::Less,
            out,
            [](auto a, auto b) { return a < b; },
            preference)));
        auto result = std::move(out).Finish(rows.view());
        for (size_t i = 0; i < safe; ++i)
            EXPECT_EQ(result.ValueAt<bool>(i), i < 10);
    }
}

TEST(ScalarExprTest, AdaptiveMembershipModesAgreeForSimdAndHashSets) {
    constexpr size_t n = 1027;
    auto values = std::make_shared<std::vector<int64_t>>(n);
    auto valid = OwnedSelection::All(n);
    for (size_t i = 0; i < n; ++i) {
        (*values)[i] = int64_t(i % 1500) - 10;
        if (i % 17 == 0)
            valid.SetSelected(i, false);
    }
    for (bool clustered : {false, true}) {
        auto rows = OwnedSelection::None(n);
        for (size_t i = 0; i < n; ++i)
            if (clustered ? i % 256 < 31 : i % 10 == 0)
                rows.SetSelected(i);
        for (bool full : {false, true}) {
            auto defined = full ? SelectionView::All(n) : rows.view();
            auto known = OwnedSelection::Intersect(valid.view(), defined);
            auto input = ReadOnlyVectorView::BorrowImmutable<int64_t>(
                             DataType::INT64,
                             values->data(),
                             values,
                             defined,
                             known.view(),
                             PrimitivePayloadAccess::AllDefinedRows)
                             .Restrict(rows.view());
            auto check = [&](const auto& terms, bool simd) {
                for (auto mode : {SelectionKernelPreference::Adaptive,
                                  SelectionKernelPreference::Ranges,
                                  SelectionKernelPreference::Sparse,
                                  SelectionKernelPreference::Dense}) {
                    BitmapResultBuilder out(n);
                    ASSERT_TRUE((MatchPrimitive<int64_t>(
                        input,
                        rows.view(),
                        out,
                        terms,
                        [&](int64_t x) {
                            return terms.In(MultiElement::ValueType(
                                std::in_place_type<int64_t>, x));
                        },
                        SelectionKernelCost::NumericMembership(terms.Size(),
                                                               simd),
                        mode)));
                    auto result = std::move(out).Finish(rows.view());
                    EXPECT_TRUE(result.defined_rows().Equals(rows.view()));
                    rows.view().ForEachSelected([&](size_t i) {
                        ASSERT_EQ(result.IsValid(i),
                                  valid.view().IsSelected(i));
                        if (result.IsValid(i))
                            EXPECT_EQ(result.ValueAt<bool>(i),
                                      terms.In(MultiElement::ValueType(
                                          std::in_place_type<int64_t>,
                                          (*values)[i])));
                    });
                }
            };
            check(SimdBatchElement<int64_t>(std::vector<int64_t>{1, 5, 10, 90}),
                  true);
            std::vector<int64_t> many;
            for (int64_t i = 0; i < 1024; ++i) many.push_back(i * 2);
            check(SetElement<int64_t>(many), false);
        }
    }
}

TEST(ScalarExprTest, SparseOptionalWriterKeepsNullsAndUnalignedNeighbors) {
    BitmapResultBuilder out(150);
    out.SetRange(0, 150, true);
    auto rows = OwnedSelection::FromIndices(131, {0, 1, 63, 64, 65, 127, 130});
    size_t calls = 0;
    out.EvaluateOptionalSelected(
        3, rows.view(), [&](size_t i) -> std::optional<bool> {
            EXPECT_TRUE(rows.view().IsSelected(i));
            ++calls;
            if (i % 2)
                return std::nullopt;
            return i % 4 == 0;
        });
    auto result = std::move(out).Finish(SelectionView::All(150));
    EXPECT_EQ(calls, 7);
    for (size_t i = 0; i < 150; ++i) {
        if (i < 3 || i >= 134) {
            EXPECT_TRUE(result.IsValid(i));
            EXPECT_TRUE(result.ValueAt<bool>(i));
        } else {
            const auto j = i - 3;
            EXPECT_EQ(result.IsValid(i),
                      rows.view().IsSelected(j) && j % 2 == 0);
            if (result.IsValid(i))
                EXPECT_EQ(result.ValueAt<bool>(i), j % 4 == 0);
        }
    }
}

TEST(ScalarExprTest, AdaptiveJsonPredicatesKeepExistsAndUnknownSemantics) {
    constexpr size_t n = 1027;
    auto column = std::make_shared<ColumnVector>(DataType::JSON, n);
    for (size_t i = 0; i < n; ++i) {
        const std::string text = i % 3 == 0   ? "{\"x\":7,\"a\":[1,7]}"
                                 : i % 3 == 1 ? "{\"x\":\"wrong\",\"a\":null}"
                                              : "{}";
        column->SetValueAt<Json>(i, Json(simdjson::padded_string(text)));
        if (i % 11 == 0)
            column->nullAt(i);
    }
    auto values = ReadOnlyVectorView::CopyOf(column);
    auto input = Input(n, {DataType::JSON});
    input.BindValues(0, values);
    EvalCtx ctx(input);
    auto ref = Ref(0, DataType::JSON);
    proto::plan::GenericValue seven;
    seven.set_int64_val(7);
    auto exists = MakeJsonExists(ref, {"x"});
    auto compare = MakeJsonComparison(ref, {"x"}, proto::plan::Equal, seven);
    auto contains = MakeJsonContains(ref, {"a"}, {seven}, false, true);
    auto length = MakeCollectionLength(ref, {"a"});
    for (bool clustered : {false, true}) {
        auto rows = OwnedSelection::None(n);
        for (size_t i = 0; i < n; ++i)
            if (clustered ? i % 256 < 31 : i % 10 == 0)
                rows.SetSelected(i);
        auto e = exists->Eval(rows.view(), ctx);
        auto c = compare->Eval(rows.view(), ctx);
        auto a = contains->Eval(rows.view(), ctx);
        auto l = length->Eval(rows.view(), ctx);
        rows.view().ForEachSelected([&](size_t i) {
            EXPECT_TRUE(e.IsValid(i));
            EXPECT_EQ(e.ValueAt<bool>(i), i % 11 != 0 && i % 3 != 2);
            const bool typed = i % 11 != 0 && i % 3 == 0;
            EXPECT_EQ(c.IsValid(i), typed);
            if (typed) {
                EXPECT_TRUE(c.ValueAt<bool>(i));
                EXPECT_TRUE(a.ValueAt<bool>(i));
            }
            EXPECT_EQ(l.IsValid(i), typed);
            if (typed)
                EXPECT_EQ(l.ValueAt<int64_t>(i), 2);
        });
        EXPECT_TRUE(e.defined_rows().Equals(rows.view()));
        EXPECT_TRUE(l.defined_rows().Equals(rows.view()));
    }
}

TEST(ScalarExprTest,
     ArrayMembershipSmallAndLargeMasksHandleDuplicatesAndNulls) {
    auto make_array = [](size_t count, bool duplicates = false) {
        ScalarFieldProto field;
        auto* data = field.mutable_long_data();
        for (size_t i = 0; i < count; ++i) data->add_data(duplicates ? 0 : i);
        return Array(field);
    };
    const auto ref = Ref(0, DataType::ARRAY);
    for (size_t count : {0, 1, 2, 3, 4, 5, 63, 64, 65, 129}) {
        auto column = std::make_shared<ColumnVector>(DataType::ARRAY, 5);
        column->SetValueAt<Array>(0, make_array(count));
        column->SetValueAt<Array>(1, make_array(count ? count - 1 : 0));
        column->SetValueAt<Array>(2, make_array(100, true));
        column->SetValueAt<Array>(3, make_array(0));
        column->SetValueAt<Array>(4, make_array(count));
        column->nullAt(4);
        auto values = ReadOnlyVectorView::CopyOf(column);
        std::vector<proto::plan::GenericValue> terms;
        for (size_t i = 0; i < count; ++i) {
            proto::plan::GenericValue term;
            term.set_int64_val(i);
            terms.push_back(term);
            terms.push_back(term);
        }
        for (bool all : {false, true}) {
            auto expr = MakeArrayContains(ref, DataType::INT64, terms, all);
            auto input = Input(5, {DataType::ARRAY});
            input.BindValues(0, values);
            EvalCtx ctx(input);
            const std::vector<int> expected =
                all ? std::vector<int>{1,
                                       count == 0,
                                       count <= 1,
                                       count == 0,
                                       -1}
                    : std::vector<int>{count > 0, count > 1, count > 0, 0, -1};
            ExpectBools(expr->Eval(SelectionView::All(5), ctx), expected);
            const auto selected = OwnedSelection::FromIndices(5, {1, 3, 4});
            auto sparse = expr->Eval(selected.view(), ctx);
            selected.view().ForEachSelected([&](size_t row) {
                EXPECT_EQ(Tri(sparse, row), expected[row]);
            });
            auto mapped = values.Dictionary(std::make_shared<RowMapping>(
                std::vector<vector_size_t>{4, 2, 0, 2, 3}));
            auto mapped_input = Input(5, {DataType::ARRAY});
            mapped_input.BindValues(0, mapped);
            EvalCtx mapped_ctx(mapped_input);
            ExpectBools(
                expr->Eval(SelectionView::All(5), mapped_ctx),
                {-1, expected[2], expected[0], expected[2], expected[3]});
        }
    }
}

TEST(ScalarExprTest, SmallArrayMembershipPreservesFloatingEqualityAndNulls) {
    auto check = [&]<typename T>(DataType type) {
        const T nan = std::numeric_limits<T>::quiet_NaN();
        const T inf = std::numeric_limits<T>::infinity();
        const std::vector<std::vector<T>> arrays{
            {T{0}, T{1}, T{2}, nan, inf, T{3}, T{4}, T{5}},
            {T{-0.0}, T{0}, nan},
            {},
            {T{7}, T{8}}};
        auto column =
            std::make_shared<ColumnVector>(DataType::ARRAY, arrays.size());
        for (size_t row = 0; row < arrays.size(); ++row) {
            ScalarFieldProto field;
            for (size_t i = 0; i < arrays[row].size(); ++i) {
                if constexpr (std::is_same_v<T, float>)
                    field.mutable_float_data()->add_data(arrays[row][i]);
                else
                    field.mutable_double_data()->add_data(arrays[row][i]);
                field.add_valid_data(!(row == 0 && i == 1));
            }
            column->template SetValueAt<Array>(row, Array(field, true));
        }
        column->nullAt(3);
        auto input = Input(arrays.size(), {DataType::ARRAY});
        input.BindValues(0, ReadOnlyVectorView::CopyOf(column));
        EvalCtx ctx(input);
        const std::vector<T> constants{
            T{-0.0}, T{0}, T{1}, T{2}, nan, inf, T{3}, T{4}};
        for (size_t count : {0, 1, 2, 3, 4, 5, 6, 8}) {
            std::vector<proto::plan::GenericValue> terms(count);
            for (size_t i = 0; i < count; ++i)
                terms[i].set_float_val(constants[i]);
            for (bool all : {false, true}) {
                auto expr = MakeArrayContains(
                    Ref(0, DataType::ARRAY), type, terms, all);
                for (const auto& rows :
                     {OwnedSelection::All(arrays.size()),
                      OwnedSelection::FromIndices(arrays.size(), {1, 2, 3})}) {
                    auto result = expr->Eval(rows.view(), ctx);
                    rows.view().ForEachSelected([&](size_t row) {
                        ASSERT_EQ(result.IsValid(row), row != 3);
                        if (row == 3)
                            return;
                        bool expected = all;
                        for (size_t target = 0; target < count; ++target) {
                            bool found = false;
                            for (size_t i = 0; i < arrays[row].size(); ++i)
                                if (!(row == 0 && i == 1) &&
                                    arrays[row][i] == constants[target])
                                    found = true;
                            if (all)
                                expected &= found;
                            else
                                expected |= found;
                        }
                        EXPECT_EQ(result.template ValueAt<bool>(row), expected);
                    });
                }
            }
        }
    };
    check.template operator()<float>(DataType::FLOAT);
    check.template operator()<double>(DataType::DOUBLE);
}

TEST(ScalarExprTest, ArrayCancellationPollsWithinLargeRowsAndAcrossShortRows) {
    const auto ref = Ref(0, DataType::ARRAY);
    proto::plan::GenericValue absent;
    absent.set_int64_val(-1);
    for (const size_t rows : {size_t{1}, size_t{2049}}) {
        ScalarFieldProto field;
        const size_t length = rows == 1 ? 4096 : 2;
        for (size_t i = 0; i < length; ++i)
            field.mutable_long_data()->add_data(i);
        const auto values = Values<Array>(
            DataType::ARRAY, std::vector<Array>(rows, Array(field)));
        std::vector<ExprPtr> expressions{
            MakeArrayContains(ref, DataType::INT64, {absent}, false),
            MakeArrayContains(ref, DataType::INT64, {absent}, true)};
        if (rows > 1) {
            expressions.push_back(MakeArrayElement(ref, 1, DataType::INT64));
            expressions.push_back(MakeCollectionLength(ref));
        }
        for (const auto& expr : expressions) {
            auto input = Input(rows, {DataType::ARRAY});
            input.BindValues(0, values);
            size_t checks = 0;
            EvalCtx cancelled(input, [&] {
                if (++checks == 4)
                    ThrowInfo(FollyCancel, "cancel ARRAY work inside Eval");
            });
            try {
                expr->Eval(SelectionView::All(rows), cancelled);
                FAIL() << "expected cancellation during ARRAY work";
            } catch (const SegcoreError& error) {
                EXPECT_EQ(error.get_error_code(), FollyCancel);
            }
            EvalCtx retry(input);
            auto result = expr->Eval(SelectionView::All(rows), retry);
            for (size_t row = 0; row < rows; ++row) {
                ASSERT_TRUE(result.IsValid(row));
                if (expr->type() == DataType::BOOL)
                    EXPECT_FALSE(result.ValueAt<bool>(row));
                else
                    EXPECT_EQ(result.ValueAt<int64_t>(row),
                              expr == expressions[2] ? 1 : length);
            }
            checks = 0;
            EXPECT_NO_THROW(expr->Eval(SelectionView::None(rows), cancelled));
            EXPECT_EQ(checks, 0);
        }
    }
}

TEST(ScalarExprTest, AdaptiveObjectCallbacksNeverExpandOrReorderWork) {
    constexpr size_t n = 1024;
    auto data = std::make_shared<std::vector<Json>>();
    for (size_t i = 0; i < n; ++i)
        data->emplace_back(simdjson::padded_string(std::to_string(i)));
    auto source =
        ReadOnlyVectorView::BorrowImmutable<Json>(DataType::JSON,
                                                  data->data(),
                                                  data,
                                                  SelectionView::All(n),
                                                  SelectionView::All(n));
    for (bool clustered : {false, true}) {
        auto rows = OwnedSelection::None(n);
        for (size_t i = 0; i < n; ++i)
            if (clustered ? i < 100 : i % 10 == 0)
                rows.SetSelected(i);
        auto input = source.Restrict(rows.view());
        std::vector<size_t> expected, calls;
        rows.view().ForEachSelected([&](size_t i) { expected.push_back(i); });
        auto check = [&](const Json& json) -> std::optional<bool> {
            const auto i = size_t(&json - data->data());
            EXPECT_TRUE(rows.view().IsSelected(i));
            calls.push_back(i);
            return i % 3 ? std::optional<bool>(i % 2) : std::nullopt;
        };
        BitmapResultBuilder out(n);
        ObjectInput<Json>(input).MatchOptional(rows.view(), out, check);
        EXPECT_EQ(calls, expected);
        auto result = std::move(out).Finish(rows.view());
        calls.clear();
        VectorResultBuilder projection(DataType::INT64, n);
        ObjectInput<Json>(input).Project<int64_t>(
            rows.view(),
            projection,
            [&](const Json& json) -> std::optional<int64_t> {
                const auto i = size_t(&json - data->data());
                EXPECT_TRUE(rows.view().IsSelected(i));
                calls.push_back(i);
                return i;
            });
        EXPECT_EQ(calls, expected);
        auto projected = std::move(projection).Finish(rows.view());
        for (auto i : expected) EXPECT_EQ(projected.ValueAt<int64_t>(i), i);
    }
}

TEST(ScalarExprTest, AdaptiveLegacyGatherKeepsOneCallAndAscendingRows) {
    constexpr size_t n = 1024;
    std::vector<int64_t> data(n);
    for (size_t i = 0; i < n; ++i) data[i] = i;
    auto input = Input(n, {DataType::INT64});
    input.BindValues(0, Values<int64_t>(DataType::INT64, data));
    EvalCtx ctx(input);
    static size_t calls = 0;
    auto expr = MakeLegacyCall(
        DataType::INT64,
        {Ref(0, DataType::INT64)},
        +[](const RowVector& args, VectorPtr& output) {
            ++calls;
            auto values = ReadOnlyVectorView::CopyOf(args.child(0));
            int64_t previous = -1;
            for (size_t i = 0; i < values.size(); ++i) {
                EXPECT_GT(values.ValueAt<int64_t>(i), previous);
                previous = values.ValueAt<int64_t>(i);
            }
            output = args.child(0);
        });
    for (bool clustered : {false, true}) {
        auto rows = OwnedSelection::None(n);
        for (size_t i = 0; i < n; ++i)
            if (clustered ? i < 100 : i % 10 == 0)
                rows.SetSelected(i);
        calls = 0;
        auto result = expr->Eval(rows.view(), ctx);
        EXPECT_EQ(calls, 1);
        EXPECT_TRUE(result.defined_rows().Equals(rows.view()));
        rows.view().ForEachSelected(
            [&](size_t i) { EXPECT_EQ(result.ValueAt<int64_t>(i), i); });
    }
}

}  // namespace milvus::exec::scalar
