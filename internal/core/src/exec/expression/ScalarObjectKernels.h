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

#include "exec/expression/ScalarBatchKernels.h"

namespace milvus::exec::scalar {

// JSON/ARRAY objects are borrowed, never copied into a temporary value per row.
// Parsing still happens on valid requested objects; no dense NULL evaluation.
template <typename T>
class ObjectInput {
 public:
    explicit ObjectInput(const ReadOnlyVectorView& input)
        : input_(input), decoded_(input.template DecodeObjects<T>()) {
    }

    bool
    decoded() const {
        return decoded_.has_value();
    }

    template <typename Test>
    void
    Match(SelectionView rows,
          BitmapResultBuilder& output,
          const Test& test) const {
        if (!decoded_) {
            rows.ForEachSelected([&](size_t row) {
                if (const auto* value = input_.template TryObjectAt<T>(row))
                    output.Set<bool>(row, test(*value));
            });
            return;
        }
        for (const auto& span : decoded_->spans) {
            auto active = OwnedSelection::Intersect(
                rows.Slice(span.begin, span.count), span.valid);
            const auto* data = static_cast<const T*>(span.data);
            auto probe = [&](size_t i) {
                return test(data[span.constant ? 0 : i]);
            };
            EvaluateSelection(
                span.begin,
                active.view(),
                false,
                SelectionKernelCost::ObjectCallback(),
                output,
                [&](size_t offset, size_t count) {
                    output.EvaluateRange(
                        span.begin + offset, count, [&](size_t i) {
                            return probe(offset + i);
                        });
                },
                probe);
        }
    }

    template <typename Test>
    void
    MatchOptional(SelectionView rows,
                  BitmapResultBuilder& output,
                  const Test& test,
                  SelectionKernelPreference preference =
                      SelectionKernelPreference::Adaptive) const {
        if (!decoded_) {
            rows.ForEachSelected([&](size_t row) {
                if (const auto* value = input_.template TryObjectAt<T>(row)) {
                    if (auto result = test(*value))
                        output.Set<bool>(row, *result);
                }
            });
            return;
        }
        for (const auto& span : decoded_->spans) {
            auto active = OwnedSelection::Intersect(
                rows.Slice(span.begin, span.count), span.valid);
            const auto* data = static_cast<const T*>(span.data);
            EvaluateOptionalSelection(
                span.begin,
                active.view(),
                SelectionKernelCost::ObjectCallback(),
                output,
                [&](size_t i) { return test(data[span.constant ? 0 : i]); },
                preference);
        }
    }

    template <typename Result, typename Test>
    void
    Project(SelectionView rows,
            VectorResultBuilder& output,
            const Test& test) const {
        if (!decoded_) {
            rows.ForEachSelected([&](size_t row) {
                if (const auto* value = input_.template TryObjectAt<T>(row)) {
                    if (auto result = test(*value))
                        output.Set<Result>(row, std::move(*result));
                }
            });
            return;
        }
        for (const auto& span : decoded_->spans) {
            auto active = OwnedSelection::Intersect(
                rows.Slice(span.begin, span.count), span.valid);
            const auto* data = static_cast<const T*>(span.data);
            auto probe = [&](size_t i) {
                return test(data[span.constant ? 0 : i]);
            };
            if (ChooseSelectionKernel(active.view(),
                                      false,
                                      SelectionKernelCost::ObjectCallback()) ==
                SelectionKernelMode::Sparse)
                output.EvaluateSelected<Result>(
                    span.begin, active.view(), probe);
            else
                active.view().ForEachSelectedRange(
                    [&](size_t offset, size_t count) {
                        output.EvaluateRange<Result>(
                            span.begin + offset, count, [&](size_t i) {
                                return probe(offset + i);
                            });
                    });
        }
    }

 private:
    const ReadOnlyVectorView& input_;
    std::optional<ReadOnlyVectorView::DecodedView> decoded_;
};

// ARRAY can be stored as owned Array or pinned ArrayView descriptors. Resolve
// representation at the span boundary; dictionary fallback borrows one descriptor.
class ArrayInput {
 public:
    explicit ArrayInput(const ReadOnlyVectorView& input) : input_(input) {
    }
    template <typename Test>
    void
    Match(SelectionView rows,
          BitmapResultBuilder& output,
          const Test& test) const {
        const ObjectInput<ArrayView> views(input_);
        if (views.decoded())
            return views.Match(rows, output, test);
        const ObjectInput<Array> owned(input_);
        if (owned.decoded())
            return owned.Match(rows, output, test);
        rows.ForEachSelected([&](size_t row) {
            if (auto value = input_.TryArrayViewAt(row))
                output.Set<bool>(row, test(*value));
        });
    }

    template <typename Result, typename Test>
    void
    Project(SelectionView rows,
            VectorResultBuilder& output,
            const Test& test) const {
        const ObjectInput<ArrayView> views(input_);
        if (views.decoded())
            return views.template Project<Result>(rows, output, test);
        const ObjectInput<Array> owned(input_);
        if (owned.decoded())
            return owned.template Project<Result>(rows, output, test);
        rows.ForEachSelected([&](size_t row) {
            if (auto value = input_.TryArrayViewAt(row)) {
                if (auto result = test(*value))
                    output.Set<Result>(row, std::move(*result));
            }
        });
    }

 private:
    const ReadOnlyVectorView& input_;
};

}  // namespace milvus::exec::scalar
