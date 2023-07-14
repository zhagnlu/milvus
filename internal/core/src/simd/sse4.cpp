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

#if defined(__x86_64__)

#include "sse4.h"
#include "sse2.h"

#include <emmintrin.h>
#include <smmintrin.h>
#include <iostream>

extern "C" {
extern int
sse2_strcmp(const char* s1, const char* s2);
}
namespace milvus {
namespace simd {

template <>
bool
FindTermSSE4(const int64_t* src, size_t vec_size, int64_t val) {
    size_t num_chunk = vec_size / 2;
    size_t remaining_size = vec_size % 2;

    __m128i xmm_target = _mm_set1_epi64x(val);
    for (size_t i = 0; i < num_chunk; ++i) {
        __m128i xmm_data =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i * 2));
        __m128i xmm_match = _mm_cmpeq_epi64(xmm_data, xmm_target);
        int mask = _mm_movemask_epi8(xmm_match);
        if (mask != 0) {
            return true;
        }
    }
    if (remaining_size == 1) {
        if (src[2 * num_chunk] == val) {
            return true;
        }
    }
    return false;
}

template <>
bool
FindTermSSE4(const std::string* src, size_t vec_size, std::string val) {
    for (size_t i = 0; i < vec_size; ++i) {
        if (StrCmpSSE4(src[i].c_str(), val.c_str())) {
            return true;
        }
    }
    return false;
}

template <>
bool
FindTermSSE4(const std::string_view* src,
             size_t vec_size,
             std::string_view val) {
    for (size_t i = 0; i < vec_size; ++i) {
        if (!StrCmpSSE4(src[i].data(), val.data())) {
            return true;
        }
    }
    return false;
}

int
StrCmpSSE4(const char* s1, const char* s2) {
    __m128i* ptr1 = reinterpret_cast<__m128i*>(const_cast<char*>(s1));
    __m128i* ptr2 = reinterpret_cast<__m128i*>(const_cast<char*>(s2));

    for (;; ptr1++, ptr2++) {
        const __m128i a = _mm_loadu_si128(ptr1);
        const __m128i b = _mm_loadu_si128(ptr2);

        const uint8_t mode = _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH |
                             _SIDD_NEGATIVE_POLARITY | _SIDD_LEAST_SIGNIFICANT;

        if (_mm_cmpistrc(a, b, mode)) {
            const auto idx = _mm_cmpistri(a, b, mode);
            const uint8_t b1 = (reinterpret_cast<char*>(ptr1))[idx];
            const uint8_t b2 = (reinterpret_cast<char*>(ptr2))[idx];

            if (b1 < b2) {
                return -1;
            } else if (b1 > b2) {
                return +1;
            } else {
                return 0;
            }
        } else if (_mm_cmpistrz(a, b, mode)) {
            break;
        }
    }
    return 0;
}

template <>
void
EqualValSSE4(const int16_t* src, size_t size, int16_t val, bool* res) {
    int num_chunk = size / 8;
    __m128i xmm_val = _mm_set1_epi16(val);  // Load val into xmm_val

    int index = 0;
    for (size_t i = 0; i < num_chunk; i += 8) {
        __m128i xmm_src =
            _mm_loadu_si128((__m128i*)&src[i]);  // Load 128 bits from src array

        __m128i xmm_cmp = _mm_cmpeq_epi16(
            xmm_src, xmm_val);  // Perform packed comparison between src and val

        // Store the comparison results in the res array
        uint16_t mask = _mm_movemask_epi8(xmm_cmp);
        res[index++] = (mask & 0x0003);
        res[index++] = (mask & 0x000C);
        res[index++] = (mask & 0x0030);
        res[index++] = (mask & 0x00C0);
        res[index++] = (mask & 0x0300);
        res[index++] = (mask & 0x0C00);
        res[index++] = (mask & 0x3000);
        res[index++] = (mask & 0xC000);
    }

    for (size_t i = 8 * num_chunk; i < size; ++i) {
        res[i] = src[i] == val;
    }
}

template <>
void
EqualValSSE4(const int32_t* src, size_t size, int32_t val, bool* res) {
    int num_chunk = size / 4;
    __m128i xmm_val = _mm_set1_epi32(val);  // Load val into xmm_val

    int index = 0;
    for (size_t i = 0; i < num_chunk; i += 4) {
        __m128i xmm_src =
            _mm_loadu_si128((__m128i*)&src[i]);  // Load 128 bits from src array

        __m128i xmm_cmp = _mm_cmpeq_epi32(
            xmm_src, xmm_val);  // Perform packed comparison between src and val

        // Store the comparison results in the res array
        uint16_t mask = _mm_movemask_epi8(xmm_cmp);
        res[index++] = (mask & 0x000F);
        res[index++] = (mask & 0x00F0);
        res[index++] = (mask & 0x0F00);
        res[index++] = (mask & 0xF000);
    }

    for (size_t i = 4 * num_chunk; i < size; ++i) {
        res[i] = src[i] == val;
    }
}

template <>
void
EqualValSSE4(const int64_t* src, size_t size, int64_t val, bool* res) {
    // __m128i target = _mm_set1_epi64x(val);
    int num_chunk = size / 2;
    __m128i xmm_val = _mm_set1_epi64x(val);  // Load val into xmm_val

    int index = 0;
    for (size_t i = 0; i < num_chunk; i += 2) {
        __m128i xmm_src =
            _mm_loadu_si128((__m128i*)&src[i]);  // Load 128 bits from src array

        __m128i xmm_cmp = _mm_cmpeq_epi64(
            xmm_src, xmm_val);  // Perform packed comparison between src and val

        // Store the comparison results in the res array
        uint16_t mask = _mm_movemask_epi8(xmm_cmp);
        res[index++] = (mask & 0x0001) != 0;
        res[index++] = (mask & 0x0100) != 0;
    }

    for (size_t i = 2 * num_chunk; i < size; ++i) {
        res[i] = src[i] == val;
    }
}

}  // namespace simd
}  // namespace milvus

#endif
