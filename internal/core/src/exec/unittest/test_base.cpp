// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include <gtest/gtest.h>
#include <cstdint>
#include <memory>
#include <vector>

#include "common/Schema.h"
#include "exec/QueryContext.h"
#include "exec/Task.h"
#include "segcore/SegmentSealed.h"
#include "exec/operator/Operator.h"
#include "exec/operator/FilterBits.h"
#include "exec/expression/Expr.h"

namespace milvus {
namespace exec {

TEST(xx, q) {
}

int
main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

}  // namespace exec
}  // namespace milvus