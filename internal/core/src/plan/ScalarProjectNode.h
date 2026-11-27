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

#include <map>
#include <unordered_set>
#include "plan/PlanNode.h"
#include "common/ResultColumn.h"

namespace milvus::plan {

// Internal data-batch projection. The bitmap-to-positions source and external
// query protocol are separate integrations; this node never guesses input
// shape from its first column's physical encoding.
class ScalarProjectNode final : public PlanNode {
 public:
    ScalarProjectNode(
        const PlanNodeId& id,
        std::vector<expr::TypedExprPtr> expressions,
        std::vector<std::string> names,
        std::map<int64_t, column_index_t> upstream_columns,
        PlanNodePtr source,
        std::map<ResultColumnId, column_index_t> upstream_results = {})
        : PlanNode(id),
          expressions_(std::move(expressions)),
          upstream_columns_(std::move(upstream_columns)),
          upstream_results_(std::move(upstream_results)),
          sources_{std::move(source)} {
        AssertInfo(sources_[0] && sources_[0]->output_type() &&
                       sources_[0]->output_type() != RowType::None,
                   "scalar projection requires a typed data source");
        AssertInfo(names.size() == expressions_.size() && !names.empty(),
                   "scalar projection requires named expressions");
        std::vector<DataType> types;
        for (const auto& expression : expressions_) {
            AssertInfo(expression, "null projection expression");
            types.push_back(expression->type());
        }
        output_type_ =
            std::make_shared<RowType>(std::move(names), std::move(types));
    }

    const std::vector<PlanNodePtr>&
    sources() const override {
        return sources_;
    }
    RowTypePtr
    output_type() const override {
        return output_type_;
    }
    OutputKind
    output_kind() const override {
        return OutputKind::ScalarRows;
    }
    std::string_view
    name() const override {
        return "ScalarProjectNode";
    }
    std::string
    ToString() const override {
        return "ScalarProjectNode";
    }
    const auto&
    expressions() const {
        return expressions_;
    }
    const auto&
    upstream_columns() const {
        return upstream_columns_;
    }
    const auto&
    upstream_results() const {
        return upstream_results_;
    }

 private:
    const std::vector<expr::TypedExprPtr> expressions_;
    const std::map<int64_t, column_index_t> upstream_columns_;
    const std::map<ResultColumnId, column_index_t> upstream_results_;
    const std::vector<PlanNodePtr> sources_;
    RowTypePtr output_type_;
};

// Internal row-level scan projection. Its sole input is the exclusion bitmap
// AFTER MVCC; it enumerates PK-ordered positions and reads only one batch of
// scalar dependencies at a time. Element-level input is deliberately separate.
class ScalarProjectScanNode final : public PlanNode {
 public:
    ScalarProjectScanNode(const PlanNodeId& id,
                          std::vector<expr::TypedExprPtr> expressions,
                          std::vector<std::string> names,
                          std::shared_ptr<MvccNode> source)
        : PlanNode(id),
          expressions_(std::move(expressions)),
          sources_{std::move(source)} {
        AssertInfo(sources_[0] && !names.empty() &&
                       names.size() == expressions_.size(),
                   "scalar scan requires MVCC input and named expressions");
        std::unordered_set<std::string> unique_names;
        for (const auto& name : names)
            AssertInfo(!name.empty() && unique_names.insert(name).second,
                       "scalar scan output names must be nonempty and unique");
        std::vector<DataType> types;
        for (const auto& expression : expressions_) {
            AssertInfo(expression, "null scan projection expression");
            types.push_back(expression->type());
        }
        output_type_ =
            std::make_shared<RowType>(std::move(names), std::move(types));
    }
    const std::vector<PlanNodePtr>&
    sources() const override {
        return sources_;
    }
    RowTypePtr
    output_type() const override {
        return output_type_;
    }
    OutputKind
    output_kind() const override {
        return OutputKind::ScalarRows;
    }
    std::string_view
    name() const override {
        return "ScalarProjectScanNode";
    }
    std::string
    ToString() const override {
        return "ScalarProjectScanNode";
    }
    const auto&
    expressions() const {
        return expressions_;
    }

 private:
    const std::vector<expr::TypedExprPtr> expressions_;
    const std::vector<PlanNodePtr> sources_;
    RowTypePtr output_type_;
};

}  // namespace milvus::plan
