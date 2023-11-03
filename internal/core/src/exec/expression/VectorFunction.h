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
#include <memory>
#include <string>
#include <vector>

#include "common/Vector.h"
#include "exec/QueryContext.h"

namespace milvus {
namespace exec {

class FunctionSignature {
 public:
};

using FunctionSignaturePtr = std::shared_ptr<FunctionSignature>;

struct VectorFunctionEntry {
    std::vector<FunctionSignaturePtr> signature_;
};
class VectorFunction {
 public:
    virtual ~VectorFunction() = default;

    virtual void
    Apply(std::vector<VectorPtr>& args,
          DataType output_type,
          EvalCtx& context,
          VectorPtr& result) const = 0;
};

std::shared_ptr<VectorFunction>
GetVectorFunction(const std::string& name,
                  const std::vector<DataType>& input_types,
                  const QueryConfig& config);

template <typename T>
struct Equal {
    constexpr bool
    operator()(const T& l, const T& r) const {
        return l < r;
    }
};
}  // namespace exec
}  // namespace milvus