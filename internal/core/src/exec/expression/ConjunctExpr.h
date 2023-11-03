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

#include "common/Types.h"
#include "common/Vector.h"
#include "exec/expression/Expr.h"
#include "segcore/SegmentInterface.h"
#include "exceptions/EasyAssert.h"

namespace milvus {
namespace exec {

template <bool is_and>
struct ConjunctElementFunc {
    void
    operator()(FlatVectorPtr& input_result, FlatVectorPtr& result) {
        bool* input_data = static_cast<bool*>(input_result->GetRawData());
        bool* res_data = static_cast<bool*>(result->GetRawData());
        for (int i = 0; i < result->size(); ++i) {
            if constexpr (is_and) {
                res_data[i] &= input_data[i];
            } else {
                res_data[i] |= input_data[i];
            }
        }
    }
};

class PhyConjunctFilterExpr : public Expr {
 public:
    PhyConjunctFilterExpr(std::vector<ExprPtr>&& inputs, bool is_and)
        : Expr(DataType::BOOL, std::move(inputs), is_and ? "and" : "or"),
          is_and_(is_and) {
        std::vector<DataType> input_types;
        input_types.reserve(inputs_.size());

        std::transform(inputs_.begin(),
                       inputs_.end(),
                       std::back_inserter(input_types),
                       [](const ExprPtr& expr) { return expr->type(); });

        ResolveType(input_types);
    }

    void
    Eval(EvalCtx& context, VectorPtr& result) override;

 private:
    int64_t
    UpdateResult(FlatVectorPtr& input_result,
                 EvalCtx& ctx,
                 FlatVectorPtr& result);

    static DataType
    ResolveType(const std::vector<DataType>& inputs);

    // true if conjunction (and), false if disjunction (or).
    bool is_and_;
    std::vector<int32_t> input_order_;
};
}  //namespace exec
}  // namespace milvus
