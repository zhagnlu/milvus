// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#pragma once

#include <cstddef>
#include <cstdint>

#include <emmintrin.h>
#include <iostream>

#include "common.h"
namespace milvus {
namespace simd {

BitsetBlockType
GetBitsetBlockSSE2(const bool* src);

template <typename T>
bool
FindTermSSE2(const T* src, size_t vec_size, T va) {
    CHECK_SUPPORTED_TYPE(T, "unsupported type for FindTermSSE2");
    return false;
}

template <>
bool
FindTermSSE2(const bool* src, size_t vec_size, bool val);

template <>
bool
FindTermSSE2(const int8_t* src, size_t vec_size, int8_t val);

template <>
bool
FindTermSSE2(const int16_t* src, size_t vec_size, int16_t val);

template <>
bool
FindTermSSE2(const int32_t* src, size_t vec_size, int32_t val);

template <>
bool
FindTermSSE2(const int64_t* src, size_t vec_size, int64_t val);

template <>
bool
FindTermSSE2(const float* src, size_t vec_size, float val);

template <>
bool
FindTermSSE2(const double* src, size_t vec_size, double val);

}  // namespace simd
}  // namespace milvus
