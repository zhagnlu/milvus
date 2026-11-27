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
#include "exec/expression/InputRefExpr.h"
#include "exec/expression/PredicateRefExpr.h"
#include "exec/expression/ScalarCallExpr.h"
#include "exec/predicate/PredicateBatch.h"
#include "segcore/reader/ReaderTestUtils.h"
#include "segcore/reader/ScalarIndexReader.h"

namespace milvus::exec::predicate {
namespace {
using namespace segcore::reader;
using namespace scalar;

BatchInput
Input(size_t n) {
    return BatchInput(
        std::make_shared<RowType>(std::vector<std::string>{"raw"},
                                  std::vector<DataType>{DataType::BOOL}),
        n);
}
ReadOnlyVectorView
Bools(const std::vector<int>& values) {
    TargetBitmap truth(values.size(), false), valid(values.size(), false);
    for (size_t i = 0; i < values.size(); ++i) {
        truth.set(i, values[i] > 0);
        valid.set(i, values[i] >= 0);
    }
    return ReadOnlyVectorView::CopyOf(
        std::make_shared<ColumnVector>(std::move(truth), std::move(valid)));
}
void
Expect(const ReadOnlyVectorView& values, const std::vector<int>& expected) {
    ASSERT_EQ(values.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(values.IsValid(i) ? int(values.ValueAt<bool>(i)) : -1,
                  expected[i])
            << "row " << i;
    }
}
std::vector<size_t>
Indices(SelectionView rows) {
    std::vector<size_t> result;
    rows.ForEachSelected([&](size_t i) { result.push_back(i); });
    return result;
}
IndexResult
Full() {
    auto yes = OwnedSelection::FromIndices(5, {1, 4});
    auto valid = OwnedSelection::FromIndices(5, {0, 1, 3, 4});
    return IndexResult::Full(yes.view(), valid.view());
}
IndexResult
Coarse() {
    auto candidates = OwnedSelection::FromIndices(5, {0, 3});
    auto no = OwnedSelection::FromIndices(5, {1});
    auto nulls = OwnedSelection::FromIndices(5, {2});
    return IndexResult::Candidates(candidates.view(), no.view(), nulls.view());
}
class TestReader : public IndexReader {
 public:
    explicit TestReader(std::function<IndexResult()> query, int64_t n = 5)
        : query(std::move(query)), n(n) {
    }
    int64_t
    row_count() const override {
        return n;
    }
    IndexResult
    Query(const ReadContext&) const override {
        ++calls;
        return query();
    }
    std::function<IndexResult()> query;
    int64_t n;
    mutable int calls = 0;
};
std::shared_ptr<IndexNode>
Node(std::shared_ptr<TestReader> reader) {
    return std::make_shared<IndexNode>(reader, std::make_shared<ReadContext>());
}
ExprPtr
Raw() {
    return std::make_shared<InputRefExpr>(0, DataType::BOOL);
}
ExprPtr
Pred(column_index_t slot = 0) {
    return std::make_shared<PredicateRefExpr>(slot);
}
}  // namespace

TEST(PredicateTest, ResultsOwnMasksAndRejectInvalidProofs) {
    auto yes = OwnedSelection::FromIndices(3, {1});
    auto valid = OwnedSelection::All(3);
    auto full = IndexResult::Full(yes.view(), valid.view());
    auto exact = IndexResult::TrueOnly(yes.view());
    yes.ResetNone(3);
    valid.ResetNone(3);
    EXPECT_TRUE(full.selected().IsSelected(1));
    EXPECT_TRUE(full.valid().IsAll());
    EXPECT_TRUE(exact.selected().IsSelected(1));
    EXPECT_THROW(IndexResult::Full(full.selected(), valid.view()),
                 SegcoreError);
    EXPECT_THROW(IndexResult::Full(full.selected(), SelectionView::All(4)),
                 SegcoreError);
    EXPECT_THROW(IndexResult::Candidates(
                     full.selected(), full.selected(), SelectionView::None(3)),
                 SegcoreError);
    EXPECT_THROW(IndexResult::Candidates(
                     SelectionView::None(3), full.selected(), full.selected()),
                 SegcoreError);
    EXPECT_THROW(IndexResult::Candidates(SelectionView::None(3),
                                         SelectionView::None(4),
                                         SelectionView::None(3)),
                 SegcoreError);
}

TEST(PredicateTest, FullResultMapsRangeOffsetsDuplicatesAndSparseRows) {
    auto result = Full();
    Expect(MapPredicate(
               result, ReadBatchBinding::Range(1, 3), SelectionView::All(3)),
           {1, -1, 0});
    const auto offsets = ReadBatchBinding::Offsets({4, 2, 0, 4});
    Expect(MapPredicate(result, offsets, SelectionView::All(4)), {1, -1, 0, 1});
    auto rows = OwnedSelection::FromIndices(4, {1, 3});
    auto sparse = MapPredicate(result, offsets, rows.view());
    EXPECT_FALSE(sparse.IsValid(1));
    EXPECT_TRUE(sparse.ValueAt<bool>(3));
    EXPECT_THROW(sparse.IsValid(0), SegcoreError);
    auto selected = MapFilterTrue(result, offsets, rows.view());
    EXPECT_EQ(Indices(selected.view()), (std::vector<size_t>{3}));
}

TEST(PredicateTest, SharedRangeViewsRetainMasksAndPreserveCoverage) {
    constexpr size_t n = 137;
    TargetBitmap truth(n, false), valid(n, false);
    for (size_t i = 0; i < n; ++i) {
        valid.set(i, i % 7 != 0);
        truth.set(i, i % 7 != 0 && i % 3 == 0);
    }
    auto result = std::make_shared<const IndexResult>(
        IndexResult::Full(OwnedSelection::OwnBitmap(std::move(truth)),
                          OwnedSelection::OwnBitmap(std::move(valid))));
    std::weak_ptr<const IndexResult> lifetime = result;
    auto rows = OwnedSelection::All(129);
    rows.SetSelected(5, false);
    rows.SetSelected(63, false);
    auto view =
        MapPredicate(result, ReadBatchBinding::Range(3, 129), rows.view());
    result.reset();
    EXPECT_FALSE(lifetime.expired());
    auto masks = view.CopyBoolMasks(rows.view());
    auto writable = MakeWritableCopy(view);
    auto restored = ReadOnlyVectorView::CopyOf(writable.values, rows.view());
    auto published = ReadOnlyVectorView::CopyOf(view.AsVector());
    TargetBitmap output(145, true);
    view.CopyTrueTo(rows.view(), output, 9);
    for (size_t i = 0; i < output.size(); ++i) {
        if (i < 9 || i >= 138) {
            EXPECT_TRUE(output[i]);
            continue;
        }
        const auto row = i - 9;
        const bool defined = rows.view().IsSelected(row);
        const bool valid_row = defined && (row + 3) % 7 != 0;
        const bool match = valid_row && (row + 3) % 3 == 0;
        EXPECT_EQ(output[i], match);
        EXPECT_EQ(masks.valid.view().IsSelected(row), valid_row);
        EXPECT_EQ(masks.truth.view().IsSelected(row), match);
        if (!defined) {
            EXPECT_THROW(view.TryValueAt<bool>(row), SegcoreError);
            continue;
        }
        EXPECT_EQ(view.IsValid(row), valid_row);
        EXPECT_EQ(restored.IsValid(row), valid_row);
        EXPECT_EQ(published.IsValid(row), valid_row);
        EXPECT_EQ(view.TryValueAt<bool>(row).has_value(), valid_row);
        if (valid_row) {
            EXPECT_EQ(view.ValueAt<bool>(row), match);
            EXPECT_EQ(*view.TryValueAt<bool>(row), match);
            EXPECT_EQ(restored.ValueAt<bool>(row), match);
            EXPECT_EQ(published.ValueAt<bool>(row), match);
        }
    }
    EXPECT_FALSE(view.DecodePrimitive());
    auto dictionary = view.Dictionary(
        std::make_shared<RowMapping>(std::vector<vector_size_t>{4, 0, 4, 1}));
    Expect(dictionary, {-1, 1, -1, 0});
    auto slice = view.Slice(60, 65);
    auto sliced_masks = slice.CopyBoolMasks(slice.defined_rows());
    for (size_t i = 0; i < 65; ++i) {
        EXPECT_EQ(sliced_masks.truth.view().IsSelected(i),
                  masks.truth.view().IsSelected(60 + i));
        EXPECT_EQ(sliced_masks.valid.view().IsSelected(i),
                  masks.valid.view().IsSelected(60 + i));
    }
    auto patch_rows = OwnedSelection::FromIndices(129, {5, 63});
    auto patch = ReadOnlyVectorView::Nulls(DataType::BOOL, patch_rows.view());
    auto merged = view.Merge(patch, patch_rows.view());
    EXPECT_TRUE(merged.defined_rows().IsAll());
    EXPECT_FALSE(merged.IsValid(5));
    EXPECT_TRUE(merged.ValueAt<bool>(0));
}

TEST(PredicateTest, BorrowedMasksReleaseOwnersAndRejectInvalidTruth) {
    std::weak_ptr<const IndexResult> lifetime;
    std::optional<ReadOnlyVectorView> view;
    {
        auto source = std::make_shared<const IndexResult>(Full());
        lifetime = source;
        EXPECT_THROW(ReadOnlyVectorView::BorrowBooleanMasks(
                         SelectionView::All(5), SelectionView::None(5), source),
                     SegcoreError);
        EXPECT_THROW(ReadOnlyVectorView::BorrowBooleanMasks(
                         SelectionView::None(4), SelectionView::All(5), source),
                     SegcoreError);
        EXPECT_THROW(ReadOnlyVectorView::BorrowBooleanMasks(
                         SelectionView::None(5), SelectionView::All(5), {}),
                     SegcoreError);
        view = MapPredicate(
            source, ReadBatchBinding::Range(1, 3), SelectionView::All(3));
    }
    EXPECT_FALSE(lifetime.expired());
    Expect(*view, {1, -1, 0});
    view.reset();
    EXPECT_TRUE(lifetime.expired());
}

TEST(PredicateTest, OwnedIndexMasksKeepValidationAndUniformDomains) {
    for (const size_t n : {size_t{0}, size_t{1}, size_t{65}}) {
        for (bool all_valid : {false, true}) {
            auto result = std::make_shared<const IndexResult>(IndexResult::Full(
                OwnedSelection::None(n),
                all_valid ? OwnedSelection::All(n) : OwnedSelection::None(n)));
            auto view = MapPredicate(
                result, ReadBatchBinding::Range(0, n), SelectionView::All(n));
            auto masks = view.CopyBoolMasks(SelectionView::All(n));
            EXPECT_TRUE(masks.truth.view().IsNone());
            EXPECT_EQ(masks.valid.view().selected_count(), all_valid ? n : 0);
        }
    }
    EXPECT_THROW(
        IndexResult::Full(OwnedSelection::All(65), OwnedSelection::None(65)),
        SegcoreError);
    EXPECT_THROW(
        IndexResult::Full(OwnedSelection::None(65), OwnedSelection::All(64)),
        SegcoreError);
    auto exact = std::make_shared<const IndexResult>(
        IndexResult::TrueOnly(SelectionView::All(2)));
    EXPECT_THROW(
        MapPredicate(
            exact, ReadBatchBinding::Range(0, 2), SelectionView::All(2)),
        SegcoreError);
}

TEST(PredicateTest, TrueOnlyAndCandidatesNeverImplicitlyBecomeFullBoolean) {
    auto full = Full();
    auto exact = IndexResult::TrueOnly(full.selected());
    auto binding = ReadBatchBinding::Range(0, 5);
    EXPECT_THROW(MapPredicate(exact, binding, SelectionView::All(5)),
                 SegcoreError);
    auto yes = MapFilterTrue(exact, binding, SelectionView::All(5));
    EXPECT_EQ(Indices(yes.view()), (std::vector<size_t>{1, 4}));
    EXPECT_THROW(MapPredicate(Coarse(), binding, SelectionView::All(5)),
                 SegcoreError);
    EXPECT_THROW(MapFilterTrue(Coarse(), binding, SelectionView::All(5)),
                 SegcoreError);
}

TEST(PredicateTest, IndexQueryOnceAcrossSkippedBackwardAndDuplicateBatches) {
    auto reader = std::make_shared<TestReader>(Full);
    auto node = Node(reader);
    node->Eval(ReadBatchBinding::Range(3, 2), SelectionView::None(2));
    EXPECT_EQ(reader->calls, 0);
    Expect(node->Eval(ReadBatchBinding::Range(3, 2), SelectionView::All(2)),
           {0, 1});
    Expect(node->Eval(ReadBatchBinding::Range(0, 3), SelectionView::All(3)),
           {0, 1, -1});
    Expect(
        node->Eval(ReadBatchBinding::Offsets({4, 1, 4}), SelectionView::All(3)),
        {1, 1, 1});
    EXPECT_EQ(reader->calls, 1);
}

TEST(PredicateTest,
     ZeroRowSourceNeverQueriesAndTrueOnlyHasExplicitFilterRoute) {
    auto reader = std::make_shared<TestReader>(Full, 0);
    auto node = Node(reader);
    auto empty =
        node->Eval(ReadBatchBinding::Range(0, 0), SelectionView::All(0));
    EXPECT_EQ(empty.size(), 0);
    EXPECT_EQ(reader->calls, 0);
    reader = std::make_shared<TestReader>([] {
        auto full = Full();
        return IndexResult::TrueOnly(full.selected());
    });
    node = Node(reader);
    const auto binding = ReadBatchBinding::Offsets({4, 2, 0, 4});
    auto selected = node->FilterTrue(binding, SelectionView::All(4));
    EXPECT_EQ(Indices(selected.view()), (std::vector<size_t>{0, 3}));
    EXPECT_THROW(node->Eval(binding, SelectionView::All(4)), SegcoreError);
    EXPECT_EQ(reader->calls, 1);
}

TEST(PredicateTest, BoundsAndDisabledIndexFailBeforeQuery) {
    auto reader = std::make_shared<TestReader>(Full);
    auto node = Node(reader);
    EXPECT_THROW(
        node->Eval(ReadBatchBinding::Range(4, 2), SelectionView::All(2)),
        SegcoreError);
    EXPECT_THROW(
        node->Eval(ReadBatchBinding::Offsets({1, 5}), SelectionView::None(2)),
        SegcoreError);
    EXPECT_THROW(
        node->Eval(ReadBatchBinding::Range(0, 2), SelectionView::All(3)),
        SegcoreError);
    EXPECT_THROW((IndexNode(reader, std::make_shared<ReadContext>(), false)),
                 SegcoreError);
    EXPECT_EQ(reader->calls, 0);
}

TEST(PredicateTest, QueryFailurePreservesCodeAndDoesNotPublish) {
    auto reader = std::make_shared<TestReader>([]() -> IndexResult {
        ThrowInfo(FileReadFailed, "injected index read failure");
    });
    auto node = Node(reader);
    try {
        node->Eval(ReadBatchBinding::Range(0, 5), SelectionView::All(5));
        FAIL() << "expected failure";
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), FileReadFailed);
    }
    EXPECT_EQ(reader->calls, 1);  // no automatic retry
    reader->query = Full;
    Expect(node->Eval(ReadBatchBinding::Range(0, 5), SelectionView::All(5)),
           {0, 1, -1, 0, 1});
    EXPECT_EQ(reader->calls, 2);
}

TEST(PredicateTest, AllocationFailureAndMalformedResultDoNotPublish) {
    auto reader = std::make_shared<TestReader>(
        []() -> IndexResult { throw std::bad_alloc(); });
    auto node = Node(reader);
    EXPECT_THROW(
        node->Eval(ReadBatchBinding::Range(0, 5), SelectionView::All(5)),
        std::bad_alloc);
    reader->query = [] {
        return IndexResult::Full(SelectionView::None(4), SelectionView::All(4));
    };
    EXPECT_THROW(
        node->Eval(ReadBatchBinding::Range(0, 5), SelectionView::All(5)),
        SegcoreError);
    reader->query = Full;
    node->Eval(ReadBatchBinding::Range(0, 5), SelectionView::All(5));
    EXPECT_EQ(reader->calls, 3);
}

TEST(PredicateTest, CancellationBeforePublicationAndOnCachedRead) {
    auto reader = std::make_shared<TestReader>(Full);
    auto context = std::make_shared<ReadContext>();
    int checks = 0, fail_at = 2;
    context->checkpoint = [&] {
        if (++checks == fail_at)
            ThrowInfo(FollyCancel, "injected cancellation");
    };
    IndexNode node(reader, context);
    EXPECT_THROW(
        node.Eval(ReadBatchBinding::Range(0, 5), SelectionView::All(5)),
        SegcoreError);
    EXPECT_EQ(reader->calls, 1);
    fail_at = -1;
    node.Eval(ReadBatchBinding::Range(0, 5), SelectionView::All(5));
    EXPECT_EQ(reader->calls, 2);
    fail_at = checks + 1;
    EXPECT_THROW(
        node.Eval(ReadBatchBinding::Range(0, 5), SelectionView::All(5)),
        SegcoreError);
    EXPECT_EQ(reader->calls, 2);
}

TEST(PredicateTest, RecursiveIndexQueryFailsAndGuardRecovers) {
    auto reader = std::make_shared<TestReader>(Full);
    auto node = Node(reader);
    reader->query = [&] {
        node->Read(ReadBatchBinding::Range(0, 5), SelectionView::All(5));
        return Full();
    };
    EXPECT_THROW(
        node->Eval(ReadBatchBinding::Range(0, 5), SelectionView::All(5)),
        SegcoreError);
    reader->query = Full;
    node->Eval(ReadBatchBinding::Range(0, 5), SelectionView::All(5));
    EXPECT_EQ(reader->calls, 2);
}

TEST(PredicateTest, PredicateSlotsAreSeparateAndNotOrPreserveNulls) {
    auto raw = Input(5);
    raw.BindValues(0, Bools({1, 0, 0, 0, 1}));
    auto reader = std::make_shared<TestReader>(Full);
    auto provider = std::make_shared<PredicateBatch>(
        raw,
        ReadBatchBinding::Range(0, 5),
        std::vector<PredicateSource>{BindIndex(Node(reader))});
    EvalCtx ctx(raw, {}, provider);
    Expect(MakeNot(Pred())->Eval(SelectionView::All(5), ctx), {1, 0, -1, 1, 0});
    Expect(MakeLogical(LogicalOp::Or, {Raw(), Pred()})
               ->Eval(SelectionView::All(5), ctx),
           {1, 1, -1, 0, 1});
    Expect(Raw()->Eval(SelectionView::All(5), ctx), {1, 0, 0, 0, 1});
    EXPECT_EQ(reader->calls, 1);
}

TEST(PredicateTest, LogicalShortCircuitAndEmptyRowsDoNotQueryIndex) {
    auto raw = Input(5);
    raw.BindValues(0, Bools({0, 0, 0, 0, 0}));
    auto reader = std::make_shared<TestReader>(Full);
    auto provider = std::make_shared<PredicateBatch>(
        raw,
        ReadBatchBinding::Range(0, 5),
        std::vector<PredicateSource>{BindIndex(Node(reader))});
    EvalCtx ctx(raw, {}, provider);
    Expect(MakeLogical(LogicalOp::And, {Raw(), Pred()})
               ->Eval(SelectionView::All(5), ctx),
           {0, 0, 0, 0, 0});
    Pred()->Eval(SelectionView::None(5), ctx);
    EXPECT_EQ(reader->calls, 0);
}

TEST(PredicateTest, RefineUsesProofsButEvaluatesUnknownNonCandidates) {
    auto raw = Input(5);
    std::vector<std::vector<size_t>> loads;
    raw.BindLoader(0, [&](SelectionView rows) {
        loads.push_back(Indices(rows));
        return Bools({1, 0, -1, 0, -1}).Restrict(rows);
    });
    auto reader = std::make_shared<TestReader>(Coarse);
    auto refine = std::make_shared<RefineNode>(Node(reader), Raw());
    auto provider = std::make_shared<PredicateBatch>(
        raw,
        ReadBatchBinding::Range(0, 5),
        std::vector<PredicateSource>{BindRefine(refine)});
    EvalCtx ctx(raw, {}, provider);
    Expect(Pred()->Eval(SelectionView::All(5), ctx), {1, 0, -1, 0, -1});
    ASSERT_EQ(loads.size(), 1);
    EXPECT_EQ(loads[0],
              (std::vector<size_t>{
                  0, 3, 4}));  // row 4 outside candidate remains NULL
    Expect(MakeNot(Pred())->Eval(SelectionView::All(5), ctx),
           {0, 1, -1, 1, -1});
    EXPECT_EQ(loads.size(), 1);
    Expect(Raw()->Eval(SelectionView::All(5), ctx), {1, 0, -1, 0, -1});
    ASSERT_EQ(loads.size(), 2);
    EXPECT_EQ(loads[1],
              (std::vector<size_t>{1, 2}));  // raw input shared with projection
    EXPECT_EQ(reader->calls, 1);
}

TEST(PredicateTest, RefineMapsReorderedAndDuplicatePositions) {
    auto raw = Input(6);
    std::vector<size_t> loaded;
    raw.BindLoader(0, [&](SelectionView rows) {
        loaded = Indices(rows);
        return Bools({-1, 1, -1, 1, 0, 0}).Restrict(rows);
    });
    EvalCtx ctx(raw);
    RefineNode refine(Node(std::make_shared<TestReader>(Coarse)), Raw());
    Expect(refine.Eval(ReadBatchBinding::Offsets({4, 0, 2, 0, 1, 3}),
                       SelectionView::All(6),
                       ctx),
           {-1, 1, -1, 1, 0, 0});
    EXPECT_EQ(loaded, (std::vector<size_t>{0, 1, 3, 5}));
}

TEST(PredicateTest, FullyProvenRowsDoNotEvaluateExactProgram) {
    auto raw = Input(2);
    int calls = 0;
    auto exact =
        MakeCall(DataType::BOOL,
                 {},
                 [&](const auto&, SelectionView, auto&) -> ReadOnlyVectorView {
                     ++calls;
                     ThrowInfo(FileReadFailed, "must not evaluate proven rows");
                 });
    EvalCtx ctx(raw);
    auto reader = std::make_shared<TestReader>(Coarse);
    RefineNode refine(Node(reader), exact);
    Expect(refine.Eval(
               ReadBatchBinding::Offsets({2, 1}), SelectionView::All(2), ctx),
           {-1, 0});
    refine.Eval(ReadBatchBinding::Offsets({2, 1}), SelectionView::None(2), ctx);
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(reader->calls, 1);
}

TEST(PredicateTest, FalseNegativeCandidateIsContractFailure) {
    auto raw = Input(5);
    raw.BindValues(0, Bools({1, 0, -1, 0, 1}));  // row 4 TRUE omitted by coarse
    EvalCtx ctx(raw);
    RefineNode refine(Node(std::make_shared<TestReader>(Coarse)), Raw());
    EXPECT_THROW(
        refine.Eval(ReadBatchBinding::Range(0, 5), SelectionView::All(5), ctx),
        SegcoreError);
}

TEST(PredicateTest, RefineFailureLeavesPredicateCoverageUnpublished) {
    auto raw = Input(5);
    bool fail = true;
    int exact_calls = 0;
    auto exact = MakeCall(
        DataType::BOOL, {}, [&](const auto&, SelectionView rows, auto&) {
            ++exact_calls;
            if (fail)
                ThrowInfo(FileReadFailed, "injected refine read failure");
            return Bools({1, 0, -1, 0, -1}).Restrict(rows);
        });
    auto reader = std::make_shared<TestReader>(Coarse);
    auto provider = std::make_shared<PredicateBatch>(
        raw,
        ReadBatchBinding::Range(0, 5),
        std::vector<PredicateSource>{
            BindRefine(std::make_shared<RefineNode>(Node(reader), exact))});
    EvalCtx ctx(raw, {}, provider);
    EXPECT_THROW(Pred()->Eval(SelectionView::All(5), ctx), SegcoreError);
    fail = false;
    Expect(Pred()->Eval(SelectionView::All(5), ctx), {1, 0, -1, 0, -1});
    EXPECT_EQ(exact_calls, 2);
    EXPECT_EQ(reader->calls,
              1);  // complete coarse result is reusable despite exact failure
}

TEST(PredicateTest, RefineDoesNotAcceptFullOrTrueOnlyResults) {
    auto raw = Input(5);
    EvalCtx ctx(raw);
    auto reader = std::make_shared<TestReader>(Full);
    RefineNode refine(Node(reader), Raw());
    EXPECT_THROW(
        refine.Eval(ReadBatchBinding::Range(0, 5), SelectionView::All(5), ctx),
        SegcoreError);
}

TEST(PredicateTest, PredicateProviderRejectsForeignRawBatchAndCycles) {
    auto raw = Input(5), foreign = Input(5);
    auto provider = std::make_shared<PredicateBatch>(
        raw,
        ReadBatchBinding::Range(0, 5),
        std::vector<PredicateSource>{
            [](const auto&, SelectionView rows, EvalCtx& ctx) {
                return ctx.GetPredicate(1, rows);
            },
            [](const auto&, SelectionView rows, EvalCtx& ctx) {
                return ctx.GetPredicate(0, rows);
            }});
    EvalCtx ctx(raw, {}, provider), other(foreign, {}, provider);
    EXPECT_THROW(Pred()->Eval(SelectionView::All(5), other), SegcoreError);
    EXPECT_THROW(Pred()->Eval(SelectionView::All(5), ctx), SegcoreError);
    EXPECT_THROW(Pred()->Eval(SelectionView::All(5), ctx),
                 SegcoreError);  // guards unwound
    EvalCtx unbound(raw);
    EXPECT_THROW(Pred()->Eval(SelectionView::All(5), unbound), SegcoreError);
}

TEST(PredicateTest, PredicateCoverageRequestsOnlyMissingRowsAndKeepsSnapshots) {
    auto raw = Input(5);
    std::vector<std::vector<size_t>> loads;
    auto provider = std::make_shared<PredicateBatch>(
        raw,
        ReadBatchBinding::Range(0, 5),
        std::vector<PredicateSource>{
            [&](const auto&, SelectionView rows, auto&) {
                loads.push_back(Indices(rows));
                return Bools({0, 1, -1, 0, 1}).Restrict(rows);
            }});
    EvalCtx ctx(raw, {}, provider);
    auto first = OwnedSelection::FromIndices(5, {1, 2});
    auto saved = Pred()->Eval(first.view(), ctx);
    Expect(Pred()->Eval(SelectionView::All(5), ctx), {0, 1, -1, 0, 1});
    EXPECT_EQ(loads, (std::vector<std::vector<size_t>>{{1, 2}, {0, 3, 4}}));
    EXPECT_TRUE(saved.ValueAt<bool>(1));
    EXPECT_FALSE(saved.IsValid(2));
    EXPECT_THROW(saved.IsValid(0), SegcoreError);
}

TEST(PredicateTest, LightweightCacheRejectsMalformedPatchWithoutPublishing) {
    auto raw = Input(5);
    int mode = 0;
    auto provider = std::make_shared<PredicateBatch>(
        raw,
        ReadBatchBinding::Range(0, 5),
        std::vector<PredicateSource>{[&](const auto&,
                                         SelectionView rows,
                                         auto&) {
            if (mode == 1)
                return Bools({0, 1, 0, 0, 1});  // changes cached NULL at 2
            if (mode == 2)
                return Bools({0, 1, -1, 0, 1}).Restrict(SelectionView::None(5));
            if (mode == 3)
                return ReadOnlyVectorView::Nulls(DataType::INT64, rows);
            if (mode == 4)
                ThrowInfo(FileReadFailed, "injected predicate failure");
            return Bools({0, 1, -1, 0, 1}).Restrict(rows);
        }});
    EvalCtx ctx(raw, {}, provider);
    auto first = OwnedSelection::FromIndices(5, {1, 2});
    auto saved = Pred()->Eval(first.view(), ctx);
    for (mode = 1; mode <= 4; ++mode) {
        EXPECT_THROW(Pred()->Eval(SelectionView::All(5), ctx), SegcoreError);
        auto hit = Pred()->Eval(first.view(), ctx);
        EXPECT_TRUE(hit.ValueAt<bool>(1));
        EXPECT_FALSE(hit.IsValid(2));
        EXPECT_FALSE(saved.IsValid(2));
    }
    mode = 0;
    Expect(Pred()->Eval(SelectionView::All(5), ctx), {0, 1, -1, 0, 1});
    EXPECT_THROW(saved.IsValid(0), SegcoreError);
}

namespace {
bool
Oracle(Int64IndexQuery q, int64_t value, bool valid) {
    switch (q.op) {
        case ScalarIndexOp::Equal:
            return value == q.lower;
        case ScalarIndexOp::NotEqual:
            return value != q.lower;
        case ScalarIndexOp::Less:
            return value < q.lower;
        case ScalarIndexOp::LessEqual:
            return value <= q.lower;
        case ScalarIndexOp::Greater:
            return value > q.lower;
        case ScalarIndexOp::GreaterEqual:
            return value >= q.lower;
        case ScalarIndexOp::Between:
            return (q.lower_inclusive ? value >= q.lower : value > q.lower) &&
                   (q.upper_inclusive ? value <= q.upper : value < q.upper);
        case ScalarIndexOp::IsNull:
            return !valid;
        case ScalarIndexOp::IsNotNull:
            return valid;
    }
    return false;
}
}  // namespace

TEST(PredicateTest, RealSortedIndexAllOperationsMatchNullableRawOracle) {
    const int64_t values[] = {std::numeric_limits<int64_t>::min(),
                              -3,
                              0,
                              4,
                              4,
                              9,
                              std::numeric_limits<int64_t>::max()};
    const bool valid[] = {true, false, true, true, false, true, true};
    auto index =
        std::make_shared<milvus::segcore::reader::TestScalarIndex<int64_t>>();
    index->Build(7, values, valid);
    for (auto op : {ScalarIndexOp::Equal,
                    ScalarIndexOp::NotEqual,
                    ScalarIndexOp::Less,
                    ScalarIndexOp::LessEqual,
                    ScalarIndexOp::Greater,
                    ScalarIndexOp::GreaterEqual,
                    ScalarIndexOp::Between,
                    ScalarIndexOp::IsNull,
                    ScalarIndexOp::IsNotNull}) {
        for (int64_t lower : values) {
            for (bool inclusive : {false, true}) {
                Int64IndexQuery query{op, lower, 9, inclusive, inclusive};
                ScalarIndexReader reader(index, 7, query);
                auto result = reader.Query(ReadContext{});
                auto mapped = MapPredicate(result,
                                           ReadBatchBinding::Range(0, 7),
                                           SelectionView::All(7));
                std::vector<int> expected;
                for (size_t i = 0; i < 7; ++i) {
                    bool nullable = op != ScalarIndexOp::IsNull &&
                                    op != ScalarIndexOp::IsNotNull;
                    expected.push_back(
                        nullable && !valid[i]
                            ? -1
                            : int(Oracle(query, values[i], valid[i])));
                }
                Expect(mapped, expected);
            }
        }
    }
}

TEST(PredicateTest, RealSortedIndexAllNullRowsAndEmptyIntervals) {
    const int64_t values[] = {1, 2, 3};
    const bool valid[] = {false, false, false};
    auto index =
        std::make_shared<milvus::segcore::reader::TestScalarIndex<int64_t>>();
    index->Build(3, values, valid);
    for (auto op : {ScalarIndexOp::Equal,
                    ScalarIndexOp::NotEqual,
                    ScalarIndexOp::Between,
                    ScalarIndexOp::IsNull,
                    ScalarIndexOp::IsNotNull}) {
        ScalarIndexReader reader(index, 3, {op, 3, 1});
        auto result = reader.Query(ReadContext{});
        Expect(
            MapPredicate(
                result, ReadBatchBinding::Range(0, 3), SelectionView::All(3)),
            std::vector<int>(3,
                             op == ScalarIndexOp::IsNull      ? 1
                             : op == ScalarIndexOp::IsNotNull ? 0
                                                              : -1));
    }
}

TEST(PredicateTest, ReloadedSortedIndexPreservesNullRowsAndSourceLifetime) {
    const int64_t values[] = {4, 0, -1, 4, 0};
    const bool valid[] = {true, false, true, true, false};
    auto built =
        std::make_shared<milvus::segcore::reader::TestScalarIndex<int64_t>>();
    built->Build(5, values, valid);
    auto loaded = built;
    auto reader = std::make_shared<ScalarIndexReader>(
        loaded, 5, Int64IndexQuery{ScalarIndexOp::NotEqual, 4});
    IndexNode node(reader, std::make_shared<ReadContext>());
    loaded.reset();  // the bound reader retains the actual index generation
    built.reset();
    reader.reset();
    Expect(node.Eval(ReadBatchBinding::Offsets({4, 2, 3, 0, 1, 2}),
                     SelectionView::All(6)),
           {-1, 1, 0, 0, -1, 1});
}

TEST(PredicateTest, MappingCrossesBitmapWordsWithoutChangingSelectionDomain) {
    auto yes = OwnedSelection::None(131), valid = OwnedSelection::None(131);
    for (size_t i = 0; i < 131; ++i) {
        if (i % 3)
            valid.SetSelected(i);
        if (i % 3 && i % 2)
            yes.SetSelected(i);
    }
    auto result = IndexResult::Full(yes.view(), valid.view());
    std::vector<int64_t> offsets;
    for (size_t i = 0; i < 135; ++i) offsets.push_back((i * 67) % 131);
    auto rows = OwnedSelection::None(offsets.size());
    for (size_t i = 0; i < offsets.size(); ++i)
        if (i % 4)
            rows.SetSelected(i);
    auto mapped =
        MapPredicate(result, ReadBatchBinding::Offsets(offsets), rows.view());
    for (size_t i = 0; i < offsets.size(); ++i) {
        if (i % 4 == 0) {
            EXPECT_FALSE(mapped.defined_rows().IsSelected(i));
            continue;
        }
        EXPECT_EQ(mapped.IsValid(i), offsets[i] % 3 != 0);
        if (mapped.IsValid(i))
            EXPECT_EQ(mapped.ValueAt<bool>(i), offsets[i] % 2 != 0);
    }
}

TEST(PredicateTest, ScalarIndexRejectsMismatchedDomainAndNestedGeneration) {
    const int64_t values[] = {1, 2, 3};
    auto index =
        std::make_shared<milvus::segcore::reader::TestScalarIndex<int64_t>>();
    index->Build(3, values);
    EXPECT_THROW((ScalarIndexReader(index, 4, {ScalarIndexOp::Equal, 1})),
                 SegcoreError);
    EXPECT_THROW(
        (ScalarIndexReader(index, 3, {static_cast<ScalarIndexOp>(999), 1})),
        SegcoreError);
    auto nested =
        std::make_shared<milvus::segcore::reader::TestScalarIndex<int64_t>>(
            storage::FileManagerContext(), true);
    EXPECT_THROW((ScalarIndexReader(nested, 0, {ScalarIndexOp::Equal, 1})),
                 SegcoreError);
}

namespace {
class FaultySortedIndex
    : public milvus::segcore::reader::TestScalarIndex<int64_t> {
 public:
    enum class Fault {
        None,
        ReadError,
        ShortTruth,
        ShortValidity,
        TrueOnNull,
        ChangedCount
    };
    Fault fault = Fault::None;
    int64_t
    Count() const override {
        return milvus::segcore::reader::TestScalarIndex<int64_t>::Count() +
               (fault == Fault::ChangedCount);
    }
    TargetBitmap
    In(size_t n, const int64_t* values) const override {
        if (fault == Fault::ReadError)
            ThrowInfo(FileReadFailed, "injected backend read failure");
        if (fault == Fault::ShortTruth)
            return TargetBitmap(1, false);
        if (fault == Fault::TrueOnNull)
            return TargetBitmap(3, true);
        return milvus::segcore::reader::TestScalarIndex<int64_t>::In(n, values);
    }
    TargetBitmap
    IsNotNull() const override {
        if (fault == Fault::ShortValidity)
            return TargetBitmap(1, true);
        return milvus::segcore::reader::TestScalarIndex<int64_t>::IsNotNull();
    }
};
}  // namespace

TEST(PredicateTest, BackendFailureAndInvalidBitmapsNeverPublishIndexResults) {
    const int64_t values[] = {1, 2, 1};
    const bool valid[] = {true, false, true};
    auto backend = std::make_shared<FaultySortedIndex>();
    backend->Build(3, values, valid);
    auto reader = std::make_shared<ScalarIndexReader>(
        backend, 3, Int64IndexQuery{ScalarIndexOp::Equal, 1});
    for (auto fault : {FaultySortedIndex::Fault::ReadError,
                       FaultySortedIndex::Fault::ShortTruth,
                       FaultySortedIndex::Fault::ShortValidity,
                       FaultySortedIndex::Fault::TrueOnNull,
                       FaultySortedIndex::Fault::ChangedCount}) {
        IndexNode node(reader, std::make_shared<ReadContext>());
        backend->fault = fault;
        try {
            node.Eval(ReadBatchBinding::Range(0, 3), SelectionView::All(3));
            FAIL() << "expected backend failure";
        } catch (const SegcoreError& error) {
            EXPECT_EQ(error.get_error_code(),
                      fault == FaultySortedIndex::Fault::ReadError
                          ? FileReadFailed
                          : UnexpectedError);
        }
        backend->fault = FaultySortedIndex::Fault::None;
        Expect(node.Eval(ReadBatchBinding::Range(0, 3), SelectionView::All(3)),
               {1, -1, 1});
    }
}

TEST(PredicateTest, RefineRecursionAndCancellationUnwindWithoutPublishing) {
    auto raw = Input(5);
    auto reader = std::make_shared<TestReader>(Coarse);
    std::shared_ptr<RefineNode> refine;
    bool recurse = true, cancel = false;
    auto exact = MakeCall(
        DataType::BOOL, {}, [&](const auto&, SelectionView rows, EvalCtx& ctx) {
            if (recurse)
                return refine->Eval(ReadBatchBinding::Range(0, 5), rows, ctx);
            cancel = true;  // fail the checkpoint after exact evaluation
            return Bools({1, 0, -1, 0, -1}).Restrict(rows);
        });
    refine = std::make_shared<RefineNode>(Node(reader), exact);
    auto provider = std::make_shared<PredicateBatch>(
        raw,
        ReadBatchBinding::Range(0, 5),
        std::vector<PredicateSource>{BindRefine(refine)});
    EvalCtx ctx(
        raw,
        [&] {
            if (cancel)
                ThrowInfo(FollyCancel, "cancel refine");
        },
        provider);
    EXPECT_THROW(Pred()->Eval(SelectionView::All(5), ctx), SegcoreError);
    recurse = false;
    EXPECT_THROW(Pred()->Eval(SelectionView::All(5), ctx), SegcoreError);
    EvalCtx recovered(raw, {}, provider);
    Expect(Pred()->Eval(SelectionView::All(5), recovered), {1, 0, -1, 0, -1});
    EXPECT_EQ(reader->calls, 1);
}
TEST(PredicateTest, BulkRangeMappingAndProofValidationAcrossBitmapWords) {
    constexpr size_t n = 277;
    auto truth = OwnedSelection::None(n);
    auto valid = OwnedSelection::All(n);
    for (size_t row = 0; row < n; ++row) {
        if (row % 7 == 0)
            valid.SetSelected(row, false);
        else if (row % 3 == 0)
            truth.SetSelected(row);
    }
    const auto full = IndexResult::Full(truth.view(), valid.view());
    auto rows = OwnedSelection::All(257);
    for (size_t row = 0; row < 257; row += 9) rows.SetSelected(row, false);
    const auto binding = ReadBatchBinding::Range(13, 257);
    const auto result = MapPredicate(full, binding, rows.view());
    const auto selected = MapFilterTrue(full, binding, rows.view());
    for (size_t row = 0; row < 257; ++row) {
        EXPECT_EQ(result.defined_rows().IsSelected(row),
                  rows.view().IsSelected(row));
        EXPECT_EQ(
            selected.view().IsSelected(row),
            rows.view().IsSelected(row) && truth.view().IsSelected(row + 13));
        if (rows.view().IsSelected(row)) {
            EXPECT_EQ(result.IsValid(row), valid.view().IsSelected(row + 13));
            if (result.IsValid(row))
                EXPECT_EQ(result.ValueAt<bool>(row),
                          truth.view().IsSelected(row + 13));
        }
    }
    auto invalid = OwnedSelection::CopyOf(truth.view());
    invalid.SetSelected(
        133);  // NULL proof at an unaligned position beyond two words.
    EXPECT_THROW(IndexResult::Full(invalid.view(), valid.view()), SegcoreError);
    auto overlap = OwnedSelection::FromIndices(n, {129});
    EXPECT_THROW(IndexResult::Candidates(
                     overlap.view(), overlap.view(), SelectionView::None(n)),
                 SegcoreError);
    EXPECT_THROW(IndexResult::Candidates(
                     overlap.view(), SelectionView::None(n), overlap.view()),
                 SegcoreError);
    EXPECT_THROW(IndexResult::Candidates(
                     SelectionView::None(n), overlap.view(), overlap.view()),
                 SegcoreError);
}

}  // namespace milvus::exec::predicate
