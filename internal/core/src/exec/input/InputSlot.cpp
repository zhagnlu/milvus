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

#include "exec/input/InputSlot.h"

namespace milvus::exec {
namespace {
void
RequireInputCoverage(SelectionView available, SelectionView requested) {
    AssertInfo(requested.IsSubsetOf(available),
               "loader did not cover requested rows");
}

struct InputLoadingGuard {
    explicit InputLoadingGuard(bool& flag) : flag_(flag) {
        AssertInfo(!flag_, "recursive loading of the same input slot");
        flag_ = true;
    }
    ~InputLoadingGuard() {
        flag_ = false;
    }
    bool& flag_;
};
}  // namespace

InputSlot::InputSlot(DataType type, size_t size)
    : empty_(ReadOnlyVectorView::Nulls(type, SelectionView::None(size))),
      state_{empty_, {OwnedSelection::None(size), OwnedSelection::None(size)}} {
}

void
InputSlot::Reset(size_t size) {
    AssertInfo(!loading_, "cannot reset an input during loading");
    if (empty_.size() != size)
        empty_ = ReadOnlyVectorView::Nulls(empty_.logical_type(),
                                           SelectionView::None(size));
    state_.values = empty_;
    state_.validity.defined.ResetNone(size);
    state_.validity.valid.ResetNone(size);
    bound_values_ = bound_validity_ = false;
}

void
InputSlot::ValidateValues(const ReadOnlyVectorView& values,
                          SelectionView requested) const {
    AssertInfo(values.size() == state_.values.size() &&
                   values.logical_type() == state_.values.logical_type(),
               "loader returned wrong value shape/type");
    RequireInputCoverage(values.defined_rows(), requested);
}

void
InputSlot::ValidateValidity(const ValidityResult& validity,
                            SelectionView requested) const {
    AssertInfo(validity.defined.view().row_count() == state_.values.size() &&
                   validity.valid.view().row_count() == state_.values.size(),
               "loader returned wrong validity shape");
    RequireInputCoverage(validity.defined.view(), requested);
    RequireInputCoverage(validity.defined.view(), validity.valid.view());
}

ValidityResult
InputSlot::ExtractValidity(const ReadOnlyVectorView& values) {
    const auto borrowed = values.TryBorrowValidity();
    return {
        OwnedSelection::CopyOf(values.defined_rows()),
        borrowed ? OwnedSelection::CopyOf(*borrowed) : values.CopyValidity()};
}

ValidityResult
InputSlot::MergeValidity(const ValidityResult& old,
                         const ValidityResult& patch) {
    const auto overlap =
        OwnedSelection::Intersect(old.defined.view(), patch.defined.view());
    if (!overlap.view().IsNone()) {
        AssertInfo(
            old.valid.view().EqualsOn(patch.valid.view(), overlap.view()),
            "loader changed validity within an immutable input batch");
    }
    return {OwnedSelection::Union(old.defined.view(), patch.defined.view()),
            OwnedSelection::Union(old.valid.view(), patch.valid.view())};
}

void
InputSlot::Publish(State state) noexcept {
    // No allocation after publication begins, including the caller's result.
    static_assert(std::is_nothrow_move_assignable_v<State>);
    static_assert(std::is_nothrow_move_constructible_v<State>);
    static_assert(std::is_nothrow_move_constructible_v<ValidityResult>);
    state_ = std::move(state);
}

void
InputSlot::BindValues(ReadOnlyVectorView values) {
    AssertInfo(!bound_values_ && !loading_, "input values may be bound once");
    ValidateValues(values, SelectionView::None(state_.values.size()));
    auto validity = MergeValidity(state_.validity, ExtractValidity(values));
    Publish({std::move(values), std::move(validity)});
    bound_values_ = true;
}

void
InputSlot::BindValidity(ValidityResult validity) {
    AssertInfo(!bound_validity_ && !loading_,
               "input validity may be bound once");
    ValidateValidity(validity, SelectionView::None(state_.values.size()));
    auto merged = MergeValidity(state_.validity, validity);
    Publish({state_.values, std::move(merged)});
    bound_validity_ = true;
}

void
InputSlot::BindLoader(InputLoaders loaders) {
    AssertInfo(
        !bound_loader_ && !loading_ && (loaders.values || loaders.validity),
        "input loader may be bound once and must have a capability");
    loaders_ = std::move(loaders);
    bound_loader_ = true;
}

InputSlot::State
InputSlot::LoadValues(SelectionView rows) {
    // No published values/validity to reconcile on the first load. Reader
    // results carry masks checked at the access boundary; ordinary loaders
    // still receive the full shape/coverage validation here.
    if (state_.values.defined_rows().IsNone() &&
        state_.validity.defined.view().IsNone()) {
        AssertInfo(loaders_.values, "input has no loader for missing values");
        auto result = loaders_.values(rows);
        // Keep the slot boundary check even for a validated Reader result:
        // a caller could forward a result from a different typed source.
        ValidateValues(result.values_, rows);
        if (result.validity_)
            return {std::move(result.values_), std::move(*result.validity_)};
        auto validity = ExtractValidity(result.values_);
        ValidateValidity(validity, rows);
        return {std::move(result.values_), std::move(validity)};
    }
    auto candidate = state_;
    auto missing =
        OwnedSelection::Subtract(rows, candidate.values.defined_rows());
    auto known_nulls = OwnedSelection::Subtract(
        candidate.validity.defined.view(), candidate.validity.valid.view());
    known_nulls.Intersect(missing.view());
    if (!known_nulls.view().IsNone()) {
        candidate.values = candidate.values.Merge(
            ReadOnlyVectorView::Nulls(candidate.values.logical_type(),
                                      known_nulls.view()),
            known_nulls.view());
        missing.Subtract(known_nulls.view());
    }
    if (!missing.view().IsNone()) {
        AssertInfo(loaders_.values, "input has no loader for missing values");
        auto result = loaders_.values(missing.view());
        auto& loaded = result.values_;
        ValidateValues(loaded, missing.view());
        auto validity = result.validity_ ? std::move(*result.validity_)
                                         : ExtractValidity(loaded);
        ValidateValidity(validity, missing.view());
        candidate.validity = MergeValidity(candidate.validity, validity);
        // Extra physical coverage may be retained, but never overwrite values
        // already published to another consumer of this immutable batch.
        auto added = OwnedSelection::Subtract(loaded.defined_rows(),
                                              candidate.values.defined_rows());
        candidate.values = candidate.values.Merge(loaded, added.view());
    }
    return candidate;
}

ReadOnlyVectorView
InputSlot::GetColumn(SelectionView rows) {
    if (rows.IsNone())
        return state_.values.Restrict(rows);
    if (rows.IsSubsetOf(state_.values.defined_rows()))
        return state_.values;
    InputLoadingGuard guard(loading_);
    auto candidate = LoadValues(rows);
    auto result = candidate.values;
    Publish(std::move(candidate));
    return result;
}

ValidityResult
InputSlot::GetValidity(SelectionView rows) {
    if (rows.IsNone()) {
        return {OwnedSelection::None(rows.row_count()),
                OwnedSelection::None(rows.row_count())};
    }
    if (rows.IsSubsetOf(state_.validity.defined.view()))
        return state_.validity;
    InputLoadingGuard guard(loading_);
    auto candidate = state_;
    auto missing =
        OwnedSelection::Subtract(rows, state_.validity.defined.view());
    if (loaders_.validity) {
        auto loaded = loaders_.validity(missing.view());
        ValidateValidity(loaded, missing.view());
        candidate.validity = MergeValidity(candidate.validity, loaded);
    } else {
        // Missing validity only: do not fetch payload for rows whose validity
        // is already known. Prepare the returned copy BEFORE publishing values.
        candidate = LoadValues(missing.view());
    }
    auto result = candidate.validity;
    Publish(std::move(candidate));
    return result;
}

}  // namespace milvus::exec
