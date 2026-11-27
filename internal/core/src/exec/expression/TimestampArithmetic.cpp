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

#include "exec/expression/TimestampArithmetic.h"
#include <climits>
#include <ctime>
#include "common/EasyAssert.h"
#include "expr/ITypeExpr.h"

namespace milvus::exec::scalar {
namespace {
int
SafeAddTimestampField(int base, int64_t delta) {
    const int64_t min_delta = static_cast<int64_t>(INT_MIN) - base;
    const int64_t max_delta = static_cast<int64_t>(INT_MAX) - base;
    AssertInfo(delta >= min_delta && delta <= max_delta,
               "timestamp interval arithmetic overflow: {} + {} = {}",
               base,
               delta,
               delta < 0 ? "less than INT_MIN" : "greater than INT_MAX");
    return static_cast<int>(static_cast<int64_t>(base) + delta);
}

int64_t
ApplyIntervalSign(int64_t value, int op_sign) {
    if (op_sign > 0) {
        return value;
    }
    AssertInfo(value != INT64_MIN,
               "timestamp interval arithmetic overflow: cannot negate {}",
               value);
    return -value;
}

}  // namespace

int64_t
ApplyTimestampInterval(int64_t current_ts_us,
                       proto::plan::ArithOpType arith_op,
                       const proto::plan::Interval& interval) {
    int op_sign;
    switch (arith_op) {
        case proto::plan::ArithOpType::Add:
            op_sign = 1;
            break;
        case proto::plan::ArithOpType::Sub:
            op_sign = -1;
            break;
        case proto::plan::ArithOpType::Unknown:
            return current_ts_us;
        default:
            ThrowInfo(UnexpectedError,
                      "unsupported arithmetic op for "
                      "timestamptz_arith_compare_expr: {}",
                      arith_op);
    }

    // Floor division to correctly handle pre-epoch (negative) timestamps.
    // Keep the remainder separate so INT64_MIN does not overflow while
    // reconstructing the sub-second component.
    int64_t epoch_sec = current_ts_us / 1000000;
    int64_t sub_sec_us = current_ts_us % 1000000;
    if (sub_sec_us < 0) {
        --epoch_sec;
        sub_sec_us += 1000000;
    }

    std::time_t epoch_time = epoch_sec;
    struct std::tm tm_buf;
    if (::gmtime_r(&epoch_time, &tm_buf) == nullptr) {
        ThrowInfo(UnexpectedError,
                  "gmtime_r failed for timestamp {} us",
                  current_ts_us);
    }

    // Apply the operation sign without overflowing when an INT64_MIN interval
    // is subtracted.
    tm_buf.tm_year = SafeAddTimestampField(
        tm_buf.tm_year, ApplyIntervalSign(interval.years(), op_sign));
    tm_buf.tm_mon = SafeAddTimestampField(
        tm_buf.tm_mon, ApplyIntervalSign(interval.months(), op_sign));
    tm_buf.tm_mday = SafeAddTimestampField(
        tm_buf.tm_mday, ApplyIntervalSign(interval.days(), op_sign));
    tm_buf.tm_hour = SafeAddTimestampField(
        tm_buf.tm_hour, ApplyIntervalSign(interval.hours(), op_sign));
    tm_buf.tm_min = SafeAddTimestampField(
        tm_buf.tm_min, ApplyIntervalSign(interval.minutes(), op_sign));
    tm_buf.tm_sec = SafeAddTimestampField(
        tm_buf.tm_sec, ApplyIntervalSign(interval.seconds(), op_sign));

    // timegm normalizes the fields. -1 is a valid epoch second, so it cannot
    // be used as an error sentinel here.
    const std::time_t new_epoch_time = ::timegm(&tm_buf);
    const auto new_epoch_sec = static_cast<int64_t>(new_epoch_time);
    constexpr int64_t kMaxSec = (INT64_MAX - 999999) / 1000000;
    constexpr int64_t kMinSec = (INT64_MIN + 999999) / 1000000;
    AssertInfo(new_epoch_sec >= kMinSec && new_epoch_sec <= kMaxSec,
               "timestamp after interval arithmetic out of representable "
               "range: {} seconds from epoch",
               new_epoch_sec);
    return new_epoch_sec * 1000000 + sub_sec_us;
}

}  // namespace milvus::exec::scalar
