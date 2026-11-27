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

#include <functional>
#include <unordered_map>

#include "exec/input/InputLayout.h"
#include "exec/expression/ScalarExprSet.h"
#include "segcore/reader/ScalarIndexReader.h"
#include "segcore/reader/TextIndexReader.h"
#include "segcore/reader/GeometryIndexReader.h"

namespace milvus {
class Schema;
}

namespace milvus::exec::plan {

// Per-branch requirements, not the union stored in InputLayout. An indexed
// comparison must not force value loading for a separate validity-only root.
struct InputUse {
    column_index_t slot;
    bool values;
};

class CompiledExprPlan;
struct MatchPlan {
    std::shared_ptr<const CompiledExprPlan> child;
    std::shared_ptr<const Schema> schema;
    int64_t anchor;
    proto::plan::MatchType operation;
    int64_t threshold;
};
struct PredicatePlan {
    InputDependency field;
    // Integral ordinary fields use query; typed_query carries floating, string
    // and JSON path alternatives. json_stats is an independent optional source.
    segcore::reader::Int64IndexQuery query;
    // Eager raw program, or null when deferred via prepare_exact or unsupported.
    scalar::ExprPtr exact;
    std::vector<InputUse> raw_inputs;
    std::optional<segcore::reader::JsonStatsQuery> json_stats;
    std::optional<segcore::reader::TypedIndexQuery> typed_query;
    std::optional<segcore::reader::TextIndexQuery> text_query;
    std::optional<segcore::reader::GeometryIndexQuery> geometry_query;
    std::optional<segcore::reader::NgramQuery> ngram_query;
    std::optional<segcore::reader::JsonStatsPath> selected_stats;
    std::optional<MatchPlan> match;
    // Public compilation has already validated constants and recorded inputs.
    // Only materialize the raw lookup structure after choosing Raw or Refine.
    // The factory captures immutable data; it never reads query resources and
    // is invoked once per preparation, not once per batch or Eval.
    std::function<scalar::ExprPtr()> prepare_exact;

    bool
    HasRaw() const {
        return exact || prepare_exact;
    }

    scalar::ExprPtr
    PrepareRaw() const {
        return prepare_exact ? prepare_exact() : exact;
    }
};

struct ExpressionProperty {
    scalar::ReorderProperty reorder;
    std::optional<int64_t> namespace_candidate;
    std::optional<column_index_t> predicate_slot;
};
using ExpressionProperties =
    std::unordered_map<const scalar::Expr*, ExpressionProperty>;

// Immutable calculation/layout only: no query resource, loader, cursor or
// batch coverage. A shared plan can be prepared independently for each query.
class CompiledExprPlan {
 public:
    CompiledExprPlan(
        InputLayout layout,
        std::vector<scalar::ExprPtr> roots,
        RowTypePtr upstream_schema,
        std::optional<std::vector<InputUse>> direct_inputs = std::nullopt,
        std::vector<PredicatePlan> predicates = {},
        ExpressionProperties properties = {},
        bool whole_segment_safe = false)
        : layout_(std::move(layout)),
          expressions_(std::move(roots)),
          upstream_schema_(std::move(upstream_schema)),
          predicates_(std::move(predicates)),
          properties_(std::move(properties)),
          whole_segment_safe_(whole_segment_safe) {
        if (direct_inputs) {
            direct_inputs_ = std::move(*direct_inputs);
        } else {
            for (size_t i = 0; i < layout_.slots().size(); ++i)
                direct_inputs_.push_back({static_cast<column_index_t>(i),
                                          layout_.slots()[i].needs_values});
        }
    }

    const InputLayout&
    inputs() const {
        return layout_;
    }

    const scalar::ExprSet&
    expressions() const {
        return expressions_;
    }

    const RowTypePtr&
    upstream_schema() const {
        return upstream_schema_;
    }

    const std::vector<InputUse>&
    direct_inputs() const {
        return direct_inputs_;
    }
    const std::vector<PredicatePlan>&
    predicates() const {
        return predicates_;
    }

    const ExpressionProperties&
    properties() const {
        return properties_;
    }
    bool
    whole_segment_safe() const {
        return whole_segment_safe_;
    }

 private:
    InputLayout layout_;
    scalar::ExprSet expressions_;
    RowTypePtr upstream_schema_;
    std::vector<InputUse> direct_inputs_;
    std::vector<PredicatePlan> predicates_;
    ExpressionProperties properties_;
    bool whole_segment_safe_;
};
using CompiledExprPlanPtr = std::shared_ptr<const CompiledExprPlan>;

}  // namespace milvus::exec::plan
