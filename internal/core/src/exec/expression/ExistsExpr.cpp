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

#include "ExistsExpr.h"
#include "common/Json.h"

namespace milvus {
namespace exec {

void
PhyExistsFilterExpr::Eval(EvalCtx& context, VectorPtr& result) {
    switch (expr_->column_.data_type_) {
        case DataType::JSON: {
            if (is_index_mode_) {
                PanicInfo("exists expr for json index mode not supportted");
            }
            result = EvalJsonExistsForDataSegment();
        }
        default:
            PanicInfo(fmt::format("unsupported data type: {}",
                                  expr_->column_.data_type_));
    }
}

VectorPtr
PhyExistsFilterExpr::EvalJsonExistsForDataSegment() {
    auto real_batch_size = GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }
    auto res_vec =
        std::make_shared<FlatVector>(DataType::BOOL, real_batch_size);
    bool* res = (bool*)res_vec->GetRawData();

    auto pointer = milvus::Json::pointer(expr_->column_.nested_path_);
    auto execute_sub_batch = [](const milvus::Json* data,
                                const int size,
                                bool* res,
                                const std::string& pointer) {
        for (int i = 0; i < size; ++i) {
            res[i] = data[i].exist(pointer);
        }
    };

    int processed_size =
        ProcessDataChunks<Json>(execute_sub_batch, res, pointer);
    AssertInfo(processed_size == real_batch_size,
               fmt::format("internal error: expr processed rows {} not equal "
                           "expect batch size {}",
                           processed_size,
                           real_batch_size));
    return res_vec;
}

}  //namespace exec
}  // namespace milvus
