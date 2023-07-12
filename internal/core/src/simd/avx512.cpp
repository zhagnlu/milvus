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

#include "avx512.h"
#include <cassert>

#if defined(__x86_64__)
#include <immintrin.h>

namespace milvus {
namespace simd {

template <>
bool
FindTermAVX512(const bool* src, size_t vec_size, bool val) {
    __m512i zmm_target = _mm512_set1_epi8(val);
    __m512i zmm_data;
    size_t num_chunks = vec_size / 64;

    for (size_t i = 0; i < 64 * num_chunks; i += 64) {
        zmm_data =
            _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + i));
        __mmask64 mask = _mm512_cmpeq_epi8_mask(zmm_data, zmm_target);
        if (mask != 0) {
            return true;
        }
    }

    for (size_t i = 64 * num_chunks; i < vec_size; ++i) {
        if (src[i] == val) {
            return true;
        }
    }
    return false;
}

template <>
bool
FindTermAVX512(const int8_t* src, size_t vec_size, int8_t val) {
    __m512i zmm_target = _mm512_set1_epi8(val);
    __m512i zmm_data;
    size_t num_chunks = vec_size / 64;

    for (size_t i = 0; i < 64 * num_chunks; i += 64) {
        zmm_data =
            _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + i));
        __mmask64 mask = _mm512_cmpeq_epi8_mask(zmm_data, zmm_target);
        if (mask != 0) {
            return true;
        }
    }

    for (size_t i = 64 * num_chunks; i < vec_size; ++i) {
        if (src[i] == val) {
            return true;
        }
    }
    return false;
}

template <>
bool
FindTermAVX512(const int16_t* src, size_t vec_size, int16_t val) {
    __m512i zmm_target = _mm512_set1_epi16(val);
    __m512i zmm_data;
    size_t num_chunks = vec_size / 32;

    for (size_t i = 0; i < 32 * num_chunks; i += 32) {
        zmm_data =
            _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + i));
        __mmask32 mask = _mm512_cmpeq_epi16_mask(zmm_data, zmm_target);
        if (mask != 0) {
            return true;
        }
    }

    for (size_t i = 32 * num_chunks; i < vec_size; ++i) {
        if (src[i] == val) {
            return true;
        }
    }
    return false;
}

template <>
bool
FindTermAVX512(const int32_t* src, size_t vec_size, int32_t val) {
    __m512i zmm_target = _mm512_set1_epi32(val);
    __m512i zmm_data;
    size_t num_chunks = vec_size / 16;

    for (size_t i = 0; i < 16 * num_chunks; i += 16) {
        zmm_data =
            _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + i));
        __mmask16 mask = _mm512_cmpeq_epi32_mask(zmm_data, zmm_target);
        if (mask != 0) {
            return true;
        }
    }

    for (size_t i = 16 * num_chunks; i < vec_size; ++i) {
        if (src[i] == val) {
            return true;
        }
    }
    return false;
}

template <>
bool
FindTermAVX512(const int64_t* src, size_t vec_size, int64_t val) {
    __m512i zmm_target = _mm512_set1_epi64(val);
    __m512i zmm_data;
    size_t num_chunks = vec_size / 8;

    for (size_t i = 0; i < 8 * num_chunks; i += 8) {
        zmm_data =
            _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + i));
        __mmask8 mask = _mm512_cmpeq_epi64_mask(zmm_data, zmm_target);
        if (mask != 0) {
            return true;
        }
    }

    for (size_t i = 8 * num_chunks; i < vec_size; ++i) {
        if (src[i] == val) {
            return true;
        }
    }
    return false;
}

template <>
bool
FindTermAVX512(const float* src, size_t vec_size, float val) {
    __m512 zmm_target = _mm512_set1_ps(val);
    __m512 zmm_data;
    size_t num_chunks = vec_size / 16;

    for (size_t i = 0; i < 16 * num_chunks; i += 16) {
        zmm_data = _mm512_loadu_ps(src + i);
        __mmask16 mask = _mm512_cmp_ps_mask(zmm_data, zmm_target, _CMP_EQ_OQ);
        if (mask != 0) {
            return true;
        }
    }

    for (size_t i = 16 * num_chunks; i < vec_size; ++i) {
        if (src[i] == val) {
            return true;
        }
    }
    return false;
}

template <>
bool
FindTermAVX512(const double* src, size_t vec_size, double val) {
    __m512d zmm_target = _mm512_set1_pd(val);
    __m512d zmm_data;
    size_t num_chunks = vec_size / 8;

    for (size_t i = 0; i < 8 * num_chunks; i += 8) {
        zmm_data = _mm512_loadu_pd(src + i);
        __mmask8 mask = _mm512_cmp_pd_mask(zmm_data, zmm_target, _CMP_EQ_OQ);
        if (mask != 0) {
            return true;
        }
    }

    for (size_t i = 8 * num_chunks; i < vec_size; ++i) {
        if (src[i] == val) {
            return true;
        }
    }
    return false;
}

template <>
void
EqualValAVX512(const int8_t* src, size_t size, int8_t val, bool* res) {
    __m512i target = _mm512_set1_epi8(val);

    int num_chunk = size / 64;

    for (size_t i = 0; i < 64 * num_chunk; i += 64) {
        __m512i data =
            _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + i));

        __mmask64 cmp_res_mask = _mm512_cmpeq_epi8_mask(data, target);
        __m512i cmp_res = _mm512_maskz_set1_epi8(cmp_res_mask, 0xFF);
        _mm512_storeu_si512(res + i, cmp_res);
    }

    for (size_t i = 64 * num_chunk; i < size; ++i) {
        res[i] = src[i] == val;
    }
}

template <>
void
EqualValAVX512(const int16_t* src, size_t size, int16_t val, bool* res) {
    __m512i target = _mm512_set1_epi16(val);

    int num_chunk = size / 32;

    for (size_t i = 0; i < 32 * num_chunk; i += 32) {
        __m512i data =
            _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + i));

        __mmask32 cmp_res_mask = _mm512_cmpeq_epi16_mask(data, target);
        __m256i cmp_res = _mm256_maskz_set1_epi8(cmp_res_mask, 0xFF);
        _mm256_storeu_si256((__m256i*)(res + i), cmp_res);
    }

    for (size_t i = 64 * num_chunk; i < size; ++i) {
        res[i] = src[i] == val;
    }
}

template <>
void
EqualValAVX512(const int32_t* src, size_t size, int32_t val, bool* res) {
    __m512i target = _mm512_set1_epi32(val);

    int middle = size / 16 * 16;

    for (size_t i = 0; i < middle; i += 16) {
        __m512i data =
            _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + i));

        __mmask16 cmp_res_mask = _mm512_cmpeq_epi32_mask(data, target);
        __m128i cmp_res = _mm_maskz_set1_epi8(cmp_res_mask, 0xFF);
        _mm_storeu_si128((__m128i*)(res + i), cmp_res);
    }

    for (size_t i = middle; i < size; ++i) {
        res[i] = src[i] == val;
    }
}

template <>
void
EqualValAVX512(const int64_t* src, size_t size, int64_t val, bool* res) {
    __m512i target = _mm512_set1_epi64(val);
    int num_chunk = size / 8;
    int index = 0;
    for (size_t i = 0; i < 8 * num_chunk; i += 8) {
        __m512i data =
            _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + i));
        __mmask8 mask = _mm512_cmpeq_epi64_mask(data, target);
        __m128i cmp_res = _mm_maskz_set1_epi8(mask, 0xFF);
        _mm_storeu_si64((__m128i*)(res + i), cmp_res);
    }

    for (size_t i = 8 * num_chunk; i < size; ++i) {
        res[i] = src[i] == val;
    }
}

template <>
void
EqualValAVX512(const float* src, size_t size, float val, bool* res) {
    __m512 target = _mm512_set1_ps(val);

    int middle = size / 16 * 16;

    for (size_t i = 0; i < middle; i += 16) {
        __m512 data = _mm512_loadu_ps(src + i);

        __mmask16 cmp_res_mask = _mm512_cmpeq_ps_mask(data, target);
        __m128i cmp_res = _mm_maskz_set1_epi8(cmp_res_mask, 0xFF);
        _mm_storeu_si128((__m128i*)(res + i), cmp_res);
    }

    for (size_t i = middle; i < size; ++i) {
        res[i] = src[i] == val;
    }
}

template <>
void
EqualValAVX512(const double* src, size_t size, double val, bool* res) {
    __m512d target = _mm512_set1_pd(val);

    int middle = size / 8 * 8;

    for (size_t i = 0; i < middle; i += 8) {
        __m512d data = _mm512_loadu_pd(src + i);

        __mmask8 cmp_res_mask = _mm512_cmpeq_pd_mask(data, target);
        __m128i cmp_res = _mm_maskz_set1_epi8(cmp_res_mask, 0xFF);
        _mm_storeu_si64((res + i), cmp_res);
    }

    for (size_t i = middle; i < size; ++i) {
        res[i] = src[i] == val;
    }
}

}  // namespace simd
}  // namespace milvus
#endif
