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

#include <fmt/core.h>

#include "common/EasyAssert.h"
#include "common/Types.h"
#include "common/Vector.h"
#include "exec/expression/Expr.h"
#include "segcore/SegmentInterface.h"

namespace milvus {
namespace exec {

enum class LogicalOpType { Invalid = 0, And = 1, Or = 2, Xor = 3, Minus = 4 };

template <LogicalOpType op>
struct LogicalElementFunc {
    void
    operator()(bool* left, bool* right, int n) {
        for (size_t i = 0; i < n; ++i) {
            if constexpr (op == LogicalOpType::And) {
                left[i] &= right[i];
            } else if constexpr (op == LogicalOpType::Or) {
                left[i] |= right[i];
            } else if constexpr (op == LogicalOpType::Xor) {
                left[i] ^= right[i];
            } else if constexpr (op == LogicalOpType::Minus) {
                left[i] -= right[i];
            } else {
                PanicInfo(
                    OpTypeInvalid,
                    fmt::format("unsupported logical operator: {}", int(op)));
            }
        }
    }
};

class PhyLogicalBinaryExpr : public Expr {
 public:
    PhyLogicalBinaryExpr(
        const std::vector<std::shared_ptr<Expr>>& input,
        const std::shared_ptr<const milvus::expr::LogicalBinaryExpr>& expr,
        const std::string& name)
        : Expr(DataType::BOOL, std::move(input), name), expr_(expr) {
    }

    void
    Eval(EvalCtx& context, VectorPtr& result) override;

 private:
    std::shared_ptr<const milvus::expr::LogicalBinaryExpr> expr_;
};

}  //namespace exec
}  // namespace milvus
