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

#include "exec/operator/Operator.h"
#include "plan/ScalarProjectNode.h"

namespace milvus::exec {

struct ScalarProjectState;

class PhyScalarProjectNode final : public Operator {
 public:
    PhyScalarProjectNode(
        int32_t operator_id,
        DriverContext* ctx,
        const std::shared_ptr<const milvus::plan::ScalarProjectNode>& node);
    ~PhyScalarProjectNode() override;
    bool
    IsFilter() const override {
        return false;
    }
    bool
    NeedInput() const override {
        return !no_more_input_ && !input_;
    }
    void
    AddInput(RowVectorPtr& input) override;
    RowVectorPtr
    GetOutput() override;
    bool
    IsFinished() override {
        return no_more_input_ && !input_;
    }
    BlockingReason
    IsBlocked(ContinueFuture*) override {
        return BlockingReason::kNotBlocked;
    }
    std::string
    ToString() const override {
        return "Scalar Project Operator";
    }

 private:
    std::unique_ptr<ScalarProjectState> state_;
};

struct ScalarProjectScanState;

class PhyScalarProjectScanNode final : public Operator {
 public:
    PhyScalarProjectScanNode(
        int32_t operator_id,
        DriverContext* ctx,
        const std::shared_ptr<const milvus::plan::ScalarProjectScanNode>& node);
    ~PhyScalarProjectScanNode() override;
    bool
    IsFilter() const override {
        return false;
    }
    bool
    NeedInput() const override {
        return !no_more_input_ && !input_ && !finished_;
    }
    void
    AddInput(RowVectorPtr& input) override;
    RowVectorPtr
    GetOutput() override;
    bool
    IsFinished() override {
        return finished_ || (no_more_input_ && !input_);
    }
    BlockingReason
    IsBlocked(ContinueFuture*) override {
        return BlockingReason::kNotBlocked;
    }
    std::string
    ToString() const override {
        return "Scalar Project Scan Operator";
    }

 private:
    std::unique_ptr<ScalarProjectScanState> state_;
    bool finished_ = false;
};

}  // namespace milvus::exec
