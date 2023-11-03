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

#include <memory>
#include <string>
#include <vector>

#include "common/Types.h"
#include "common/Vector.h"
#include "expr/ITypeExpr.h"
#include "exceptions/EasyAssert.h"
#include "segcore/SegmentInterface.h"

namespace milvus {
namespace plan {

typedef std::string PlanNodeId;
/** 
 * @brief Base class for all logic plan node
 * 
 */
class PlanNode {
 public:
    explicit PlanNode(const PlanNodeId& id) : id_(id) {
    }

    virtual ~PlanNode() = default;

    const PlanNodeId&
    id() const {
        return id_;
    }

    virtual DataType
    output_type() const = 0;

    virtual std::vector<std::shared_ptr<PlanNode>>
    sources() const = 0;

    virtual bool
    RequireSplits() const {
        return false;
    }

    virtual std::string_view
    name() const = 0;

 private:
    PlanNodeId id_;
};

using PlanNodePtr = std::shared_ptr<PlanNode>;

class SegmentNode : public PlanNode {
 public:
    SegmentNode(
        const PlanNodeId& id,
        const std::shared_ptr<milvus::segcore::SegmentInternalInterface>&
            segment)
        : PlanNode(id), segment_(segment) {
    }

    DataType
    output_type() const override {
        return DataType::ROW;
    }

    std::vector<std::shared_ptr<PlanNode>>
    sources() const override {
        return {};
    }

    std::string_view
    name() const override {
        return "SegmentNode";
    }

 private:
    std::shared_ptr<milvus::segcore::SegmentInternalInterface> segment_;
};

class ValuesNode : public PlanNode {
 public:
    ValuesNode(const PlanNodeId& id,
               const std::vector<RowVectorPtr>& values,
               bool parallelizeable = false)
        : PlanNode(id),
          values_{std::move(values)},
          output_type_(values[0]->type()) {
        AssertInfo(!values.empty(), "ValueNode must has value");
    }

    ValuesNode(const PlanNodeId& id,
               std::vector<RowVectorPtr>&& values,
               bool parallelizeable = false)
        : PlanNode(id),
          values_{std::move(values)},
          output_type_(values[0]->type()) {
        AssertInfo(!values.empty(), "ValueNode must has value");
    }

    DataType
    output_type() const override {
        return output_type_;
    }

    const std::vector<RowVectorPtr>&
    values() const {
        return values_;
    }

    std::vector<PlanNodePtr>
    sources() const override {
        return {};
    }

    bool
    parallelizable() {
        return parallelizable_;
    }

    std::string_view
    name() const override {
        return "Values";
    }

 private:
    DataType output_type_;
    const std::vector<RowVectorPtr> values_;
    bool parallelizable_;
};

class FilterNode : public PlanNode {
 public:
    FilterNode(const PlanNodeId& id,
               expr::TypedExprPtr filter,
               std::vector<PlanNodePtr> sources)
        : PlanNode(id),
          sources_{std::move(sources)},
          filter_(std::move(filter)) {
        AssertInfo(
            filter_->type() == DataType::BOOL,
            fmt::format("Filter expression must be of type BOOLEAN, Got {}",
                        filter_->type()));
    }

    DataType
    output_type() const override {
        return sources_[0]->output_type();
    }

    std::vector<PlanNodePtr>
    sources() const override {
        return sources_;
    }

    const expr::TypedExprPtr&
    filter() const {
        return filter_;
    }

    std::string_view
    name() const override {
        return "Filter";
    }

 private:
    const std::vector<PlanNodePtr> sources_;
    const expr::TypedExprPtr filter_;
};

class FilterBitsNode : public PlanNode {
 public:
    FilterBitsNode(
        const PlanNodeId& id,
        expr::TypedExprPtr filter,
        std::vector<PlanNodePtr> sources = std::vector<PlanNodePtr>{})
        : PlanNode(id),
          sources_{std::move(sources)},
          filter_(std::move(filter)) {
        AssertInfo(
            filter_->type() == DataType::BOOL,
            fmt::format("Filter expression must be of type BOOLEAN, Got {}",
                        filter_->type()));
    }

    DataType
    output_type() const override {
        return DataType::BOOL;
    }

    std::vector<PlanNodePtr>
    sources() const override {
        return sources_;
    }

    const expr::TypedExprPtr&
    filter() const {
        return filter_;
    }

    std::string_view
    name() const override {
        return "FilterBits";
    }

 private:
    const std::vector<PlanNodePtr> sources_;
    const expr::TypedExprPtr filter_;
};

enum class ExecutionStrategy {
    // Process splits as they come in any available driver.
    kUngrouped,
    // Process splits from each split group only in one driver.
    // It is used when split groups represent separate partitions of the data on
    // the grouping keys or join keys. In that case it is sufficient to keep only
    // the keys from a single split group in a hash table used by group-by or
    // join.
    kGrouped,
};
struct PlanFragment {
    std::shared_ptr<const PlanNode> plan_node_;
    ExecutionStrategy execution_strategy_{ExecutionStrategy::kUngrouped};
    int32_t num_splitgroups_{0};

    PlanFragment() = default;

    inline bool
    IsGroupedExecution() const {
        return execution_strategy_ == ExecutionStrategy::kGrouped;
    }

    explicit PlanFragment(std::shared_ptr<const PlanNode> top_node,
                          ExecutionStrategy strategy,
                          int32_t num_splitgroups)
        : plan_node_(std::move(top_node)),
          execution_strategy_(strategy),
          num_splitgroups_(num_splitgroups) {
    }

    explicit PlanFragment(std::shared_ptr<const PlanNode> top_node)
        : plan_node_(std::move(top_node)) {
    }
};

}  // namespace plan
}  // namespace milvus