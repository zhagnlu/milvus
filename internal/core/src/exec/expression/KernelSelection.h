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

#include "common/Selection.h"

namespace milvus::exec::scalar {

enum class SelectionKernelMode { Ranges, Sparse, Dense };
// Explicit preferences are useful for paired benchmarks; even Dense cannot
// override the caller's proven input/operator eligibility.
enum class SelectionKernelPreference { Adaptive, Ranges, Sparse, Dense };

// Relative costs supplied by the concrete kernel, outside row loops. Safety
// eligibility is a separate input, never inferred from these estimates.
struct SelectionKernelCost {
    uint32_t batch_row;
    uint32_t sparse_row;
    uint32_t range_start;
    uint32_t mask_word;
    // Sparse execution still walks bitmap words and packs output, even when
    // very few payloads are selected. Omitting this term favors sparse too much.
    uint32_t sparse_word = 64;

    static constexpr SelectionKernelCost
    NumericComparison() {
        return {1, 12, 160, 2};
    }
    // Compile-time/bound-plan estimates: no per-row cost discovery. Hash IN
    // has little batch advantage; small SIMD lists can amortize range startup.
    static SelectionKernelCost
    NumericMembership(size_t terms, bool simd) {
        const auto work = static_cast<uint32_t>(std::min<size_t>(terms, 64));
        return simd ? SelectionKernelCost{std::max<uint32_t>(1, work / 4),
                                          12 + work / 2,
                                          160,
                                          2,
                                          32}
                    : SelectionKernelCost{24, 28, 48, 2, 8};
    }
    static constexpr SelectionKernelCost
    StringComparison() {
        return {24, 28, 64, 2, 8};
    }
    static constexpr SelectionKernelCost
    StringMembership() {
        return {64, 100, 64, 2, 8};
    }
    static constexpr SelectionKernelCost
    SelectedCallback() {
        return {64, 68, 32, 2, 32};
    }
    static constexpr SelectionKernelCost
    ObjectCallback() {
        return {256, 260, 32, 2, 32};
    }
    static constexpr SelectionKernelCost
    Arithmetic(bool bulk) {
        return bulk ? SelectionKernelCost{2, 16, 160, 2, 64}
                    : SelectionKernelCost{12, 16, 128, 2, 32};
    }
    static constexpr SelectionKernelCost
    ScalarComparison() {
        return {4, 8, 16, 2};
    }
};

inline SelectionKernelMode
ChooseSelectionKernel(SelectionView active,
                      bool can_evaluate_all,
                      SelectionKernelCost cost,
                      SelectionKernelPreference preference =
                          SelectionKernelPreference::Adaptive) {
    using Mode = SelectionKernelMode;
    using Preference = SelectionKernelPreference;
    if (preference == Preference::Sparse)
        return Mode::Sparse;
    if (preference == Preference::Ranges)
        return Mode::Ranges;
    if (preference == Preference::Dense)
        return can_evaluate_all ? Mode::Dense : Mode::Sparse;
    // OwnedSelection already canonicalizes uniform results. Do not rescan an
    // arbitrary bitmap to recognize All/None a second time.
    if (!active.HasBitmap())
        return Mode::
            Ranges;  // All is already one batch, with no mask to apply.
    const size_t n = active.row_count();
    if (n <= 64)
        return Mode::Sparse;
    const size_t words = (n + 63) / 64;
    const size_t samples = std::min<size_t>(8, words);
    uint64_t selected = 0, runs = 0, inspected = 0;
    for (size_t i = 0; i < samples; ++i) {
        const size_t word_index = i * (words - 1) / (samples - 1);
        const size_t begin = word_index * 64;
        const size_t count = std::min<size_t>(64, n - begin);
        const auto bits = active.ReadWord(begin, count);
        selected += __builtin_popcountll(bits);
        runs += __builtin_popcountll(bits & ~(bits << 1));
        inspected += count;
    }
    // Samples estimate cost only. Missing a cluster can choose a slower
    // kernel, but never changes which rows are valid or allowed to be read.
    const uint64_t sparse =
        selected * cost.sparse_row + samples * cost.sparse_word;
    const uint64_t ranges =
        selected * cost.batch_row + runs * cost.range_start + samples;
    const uint64_t dense =
        inspected * cost.batch_row + samples * cost.mask_word;
    if (can_evaluate_all && dense <= std::min(sparse, ranges))
        return Mode::Dense;
    return ranges <= sparse ? Mode::Ranges : Mode::Sparse;
}

// Strict traversal for projections, callbacks and gather. Both alternatives
// retain ascending selected-row order; no callback sees an unselected row.
// Supply a scalar callback directly, not a one-row invocation of a batch kernel.
template <typename Range, typename Value>
void
VisitSelected(SelectionView active,
              SelectionKernelCost cost,
              const Range& range,
              const Value& value) {
    if (ChooseSelectionKernel(active, false, cost) ==
        SelectionKernelMode::Sparse)
        active.ForEachSelected(value);
    else
        active.ForEachSelectedRange(range);
}

}  // namespace milvus::exec::scalar
