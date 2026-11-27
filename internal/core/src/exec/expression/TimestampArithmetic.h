// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <cstdint>
#include "pb/plan.pb.h"

namespace milvus::exec::scalar {
// Shared with the legacy executor: UTC calendar normalization, pre-epoch
// fractions and overflow checks must not diverge between execution paths.
int64_t
ApplyTimestampInterval(int64_t timestamp_us,
                       proto::plan::ArithOpType operation,
                       const proto::plan::Interval& interval);
}  // namespace milvus::exec::scalar
