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

#include <boost/format.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <unordered_set>
#include <boost/container/vector.hpp>

#if defined(__x86_64__)
#include "simd/hook.h"
#include "simd/sse2.h"
#include "simd/sse4.h"
#include "simd/avx2.h"
#include "simd/avx512.h"
#include "simd/ref.h"

using namespace std;
using namespace milvus::simd;

template <typename Type>
using FixedVector = boost::container::vector<Type>;

#define PRINT_SKPI_TEST                                                    \
    std::cout                                                              \
        << "skip "                                                         \
        << ::testing::UnitTest::GetInstance()->current_test_info()->name() \
        << std::endl;

TEST(GetBitSetBlock, base_test_sse) {
    FixedVector<bool> src;
    for (int i = 0; i < 64; ++i) {
        src.push_back(false);
    }

    auto res = GetBitsetBlockSSE2(src.data());
    std::cout << res << std::endl;
    ASSERT_EQ(res, 0);

    src.clear();
    for (int i = 0; i < 64; ++i) {
        src.push_back(true);
    }
    res = GetBitsetBlockSSE2(src.data());
    std::cout << std::hex << res << std::endl;
    ASSERT_EQ(res, 0xffffffffffffffff);

    src.clear();
    for (int i = 0; i < 64; ++i) {
        src.push_back(i % 2 == 0 ? true : false);
    }
    res = GetBitsetBlockSSE2(src.data());
    std::cout << std::hex << res << std::endl;
    ASSERT_EQ(res, 0x5555555555555555);

    src.clear();
    for (int i = 0; i < 64; ++i) {
        src.push_back(i % 4 == 0 ? true : false);
    }
    res = GetBitsetBlockSSE2(src.data());
    std::cout << std::hex << res << std::endl;
    ASSERT_EQ(res, 0x1111111111111111);

    src.clear();
    for (int i = 0; i < 64; ++i) {
        src.push_back(i % 8 == 0 ? true : false);
    }
    res = GetBitsetBlockSSE2(src.data());
    std::cout << std::hex << res << std::endl;
    ASSERT_EQ(res, 0x0101010101010101);

    src.clear();
    for (int i = 0; i < 64; ++i) {
        src.push_back(i % 16 == 0 ? true : false);
    }
    res = GetBitsetBlockSSE2(src.data());
    std::cout << std::hex << res << std::endl;
    ASSERT_EQ(res, 0x0001000100010001);

    src.clear();
    for (int i = 0; i < 64; ++i) {
        src.push_back(i % 32 == 0 ? true : false);
    }
    res = GetBitsetBlockSSE2(src.data());
    std::cout << std::hex << res << std::endl;
    ASSERT_EQ(res, 0x0000000100000001);

    src.clear();
    for (int i = 0; i < 64; ++i) {
        src.push_back(i % 5 == 0 ? true : false);
    }
    res = GetBitsetBlockSSE2(src.data());
    std::cout << std::hex << res << std::endl;
    ASSERT_EQ(res, 0x1084210842108421);
}

TEST(GetBitsetBlockPerf, bitset) {
    FixedVector<bool> srcs;
    for (size_t i = 0; i < 100000000; ++i) {
        srcs.push_back(i % 2 == 0);
    }
    std::cout << "start test" << std::endl;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 10000000; ++i)
        auto result = GetBitsetBlockSSE2(srcs.data() + i);
    std::cout << "cost: "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - start)
                     .count()
              << "us" << std::endl;
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < 10000000; ++i)
        auto result = GetBitsetBlockAVX2(srcs.data() + i);
    std::cout << "cost: "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - start)
                     .count()
              << "us" << std::endl;
}

TEST(GetBitSetBlock, base_test_avx2) {
    FixedVector<bool> src;
    for (int i = 0; i < 64; ++i) {
        src.push_back(false);
    }

    auto res = GetBitsetBlockAVX2(src.data());
    std::cout << res << std::endl;
    ASSERT_EQ(res, 0);

    src.clear();
    for (int i = 0; i < 64; ++i) {
        src.push_back(true);
    }
    res = GetBitsetBlockAVX2(src.data());
    std::cout << std::hex << res << std::endl;
    ASSERT_EQ(res, 0xffffffffffffffff);

    src.clear();
    for (int i = 0; i < 64; ++i) {
        src.push_back(i % 2 == 0 ? true : false);
    }
    res = GetBitsetBlockAVX2(src.data());
    std::cout << std::hex << res << std::endl;
    ASSERT_EQ(res, 0x5555555555555555);

    src.clear();
    for (int i = 0; i < 64; ++i) {
        src.push_back(i % 4 == 0 ? true : false);
    }
    res = GetBitsetBlockAVX2(src.data());
    std::cout << std::hex << res << std::endl;
    ASSERT_EQ(res, 0x1111111111111111);

    src.clear();
    for (int i = 0; i < 64; ++i) {
        src.push_back(i % 8 == 0 ? true : false);
    }
    res = GetBitsetBlockAVX2(src.data());
    std::cout << std::hex << res << std::endl;
    ASSERT_EQ(res, 0x0101010101010101);

    src.clear();
    for (int i = 0; i < 64; ++i) {
        src.push_back(i % 16 == 0 ? true : false);
    }
    res = GetBitsetBlockAVX2(src.data());
    std::cout << std::hex << res << std::endl;
    ASSERT_EQ(res, 0x0001000100010001);

    src.clear();
    for (int i = 0; i < 64; ++i) {
        src.push_back(i % 32 == 0 ? true : false);
    }
    res = GetBitsetBlockAVX2(src.data());
    std::cout << std::hex << res << std::endl;
    ASSERT_EQ(res, 0x0000000100000001);

    src.clear();
    for (int i = 0; i < 64; ++i) {
        src.push_back(i % 5 == 0 ? true : false);
    }
    res = GetBitsetBlockAVX2(src.data());
    std::cout << std::hex << res << std::endl;
    ASSERT_EQ(res, 0x1084210842108421);
}

TEST(FindTermSSE2, bool_type) {
    FixedVector<bool> vecs;
    vecs.push_back(false);

    auto res = FindTermSSE2(vecs.data(), vecs.size(), true);
    ASSERT_EQ(res, false);
    res = FindTermSSE2(vecs.data(), vecs.size(), false);
    ASSERT_EQ(res, true);

    for (int i = 0; i < 16; i++) {
        vecs.push_back(false);
    }

    res = FindTermSSE2(vecs.data(), vecs.size(), true);
    ASSERT_EQ(res, false);
    res = FindTermSSE2(vecs.data(), vecs.size(), false);
    ASSERT_EQ(res, true);

    vecs.push_back(true);
    for (int i = 0; i < 16; i++) {
        vecs.push_back(false);
    }
    res = FindTermSSE2(vecs.data(), vecs.size(), true);
    ASSERT_EQ(res, true);
}

TEST(FindTermSSE2, int8_type) {
    std::vector<int8_t> vecs;
    for (int i = 0; i < 100; i++) {
        vecs.push_back(i);
    }

    auto res = FindTermSSE2(vecs.data(), vecs.size(), (int8_t)0);
    ASSERT_EQ(res, true);
    res = FindTermSSE2(vecs.data(), vecs.size(), (int8_t)10);
    ASSERT_EQ(res, true);
    res = FindTermSSE2(vecs.data(), vecs.size(), (int8_t)99);
    ASSERT_EQ(res, true);
    res = FindTermSSE2(vecs.data(), vecs.size(), (int8_t)100);
    ASSERT_EQ(res, false);
    res = FindTermSSE2(vecs.data(), vecs.size(), (int8_t)127);
    ASSERT_EQ(res, false);
    vecs.push_back(127);
    res = FindTermSSE2(vecs.data(), vecs.size(), (int8_t)127);
    ASSERT_EQ(res, true);
}

TEST(FindTermSSE2, int16_type) {
    std::vector<int16_t> vecs;
    for (int i = 0; i < 1000; i++) {
        vecs.push_back(i);
    }

    auto res = FindTermSSE2(vecs.data(), vecs.size(), (int16_t)0);
    ASSERT_EQ(res, true);
    res = FindTermSSE2(vecs.data(), vecs.size(), (int16_t)10);
    ASSERT_EQ(res, true);
    res = FindTermSSE2(vecs.data(), vecs.size(), (int16_t)999);
    ASSERT_EQ(res, true);
    res = FindTermSSE2(vecs.data(), vecs.size(), (int16_t)1000);
    ASSERT_EQ(res, false);
    res = FindTermSSE2(vecs.data(), vecs.size(), (int16_t)1270);
    ASSERT_EQ(res, false);
    vecs.push_back(1000);
    res = FindTermSSE2(vecs.data(), vecs.size(), (int16_t)1000);
    ASSERT_EQ(res, true);
}

TEST(FindTermSSE2, int32_type) {
    std::vector<int32_t> vecs;
    for (int i = 0; i < 1000; i++) {
        vecs.push_back(i);
    }

    auto res = FindTermSSE2(vecs.data(), vecs.size(), 0);
    ASSERT_EQ(res, true);
    res = FindTermSSE2(vecs.data(), vecs.size(), 10);
    ASSERT_EQ(res, true);
    res = FindTermSSE2(vecs.data(), vecs.size(), 999);
    ASSERT_EQ(res, true);
    res = FindTermSSE2(vecs.data(), vecs.size(), 1000);
    ASSERT_EQ(res, false);

    vecs.push_back(1000);
    res = FindTermSSE2(vecs.data(), vecs.size(), 1000);
    ASSERT_EQ(res, true);
    res = FindTermSSE2(vecs.data(), vecs.size(), 1001);
    ASSERT_EQ(res, false);

    vecs.push_back(1001);
    res = FindTermSSE2(vecs.data(), vecs.size(), 1001);
    ASSERT_EQ(res, true);
    res = FindTermSSE2(vecs.data(), vecs.size(), 1002);
    ASSERT_EQ(res, false);

    vecs.push_back(1002);
    res = FindTermSSE2(vecs.data(), vecs.size(), 1002);
    ASSERT_EQ(res, true);
    res = FindTermSSE2(vecs.data(), vecs.size(), 1003);
    ASSERT_EQ(res, false);

    res = FindTermSSE2(vecs.data(), vecs.size(), 1270);
    ASSERT_EQ(res, false);
}

TEST(FindTermSSE2, int64_type) {
    std::vector<int64_t> vecs;
    for (int i = 0; i < 1000; i++) {
        vecs.push_back(i);
    }

    auto res = FindTermSSE2(vecs.data(), vecs.size(), (int64_t)0);
    ASSERT_EQ(res, true);
    res = FindTermSSE2(vecs.data(), vecs.size(), (int64_t)10);
    ASSERT_EQ(res, true);
    res = FindTermSSE2(vecs.data(), vecs.size(), (int64_t)999);
    ASSERT_EQ(res, true);
    res = FindTermSSE2(vecs.data(), vecs.size(), (int64_t)1000);
    ASSERT_EQ(res, false);
    res = FindTermSSE2(vecs.data(), vecs.size(), (int64_t)1270);
    ASSERT_EQ(res, false);
    vecs.push_back(1005);
    res = FindTermSSE2(vecs.data(), vecs.size(), (int64_t)1005);
    ASSERT_EQ(res, true);
}

TEST(FindTermSSE2, float_type) {
    std::vector<float> vecs;
    for (int i = 0; i < 10000; i++) {
        vecs.push_back(i + 0.01);
    }

    auto res = FindTermSSE2(vecs.data(), vecs.size(), (float)0.01);
    ASSERT_EQ(res, true);
    res = FindTermSSE2(vecs.data(), vecs.size(), (float)10.01);
    ASSERT_EQ(res, true);
    res = FindTermSSE2(vecs.data(), vecs.size(), (float)10000.01);
    ASSERT_EQ(res, false);
    res = FindTermSSE2(vecs.data(), vecs.size(), (float)12700.02);
    ASSERT_EQ(res, false);
    vecs.push_back(1.001);
    res = FindTermSSE2(vecs.data(), vecs.size(), (float)1.001);
    ASSERT_EQ(res, true);
}

TEST(FindTermSSE2, double_type) {
    std::vector<double> vecs;
    for (int i = 0; i < 10000; i++) {
        vecs.push_back(i + 0.01);
    }

    auto res = FindTermSSE2(vecs.data(), vecs.size(), 0.01);
    ASSERT_EQ(res, true);
    res = FindTermSSE2(vecs.data(), vecs.size(), 10.01);
    ASSERT_EQ(res, true);
    res = FindTermSSE2(vecs.data(), vecs.size(), 10000.01);
    ASSERT_EQ(res, false);
    res = FindTermSSE2(vecs.data(), vecs.size(), 12700.01);
    ASSERT_EQ(res, false);
    vecs.push_back(1.001);
    res = FindTermSSE2(vecs.data(), vecs.size(), 1.001);
    ASSERT_EQ(res, true);
}

TEST(FindTermSSE4, int64_type) {
    if (!cpu_support_sse4_2()) {
        PRINT_SKPI_TEST
        return;
    }
    std::vector<int64_t> srcs;
    for (size_t i = 0; i < 1000; i++) {
        srcs.push_back(i);
    }

    auto res = FindTermSSE4(srcs.data(), srcs.size(), (int64_t)0);
    ASSERT_EQ(res, true);
    res = FindTermSSE4(srcs.data(), srcs.size(), (int64_t)1);
    ASSERT_EQ(res, true);
    res = FindTermSSE4(srcs.data(), srcs.size(), (int64_t)999);
    ASSERT_EQ(res, true);
    res = FindTermSSE4(srcs.data(), srcs.size(), (int64_t)1000);
    ASSERT_EQ(res, false);
    res = FindTermSSE4(srcs.data(), srcs.size(), (int64_t)2000);
    ASSERT_EQ(res, false);
    srcs.push_back(1000);
    res = FindTermSSE4(srcs.data(), srcs.size(), (int64_t)1000);
    ASSERT_EQ(res, true);
}

TEST(FindTermAVX2, bool_type) {
    if (!cpu_support_avx2()) {
        PRINT_SKPI_TEST
        return;
    }
    std::vector<int64_t> srcs;
    for (size_t i = 0; i < 1000; i++) {
        srcs.push_back(i);
    }
    FixedVector<bool> vecs;
    vecs.push_back(false);

    auto res = FindTermAVX2(vecs.data(), vecs.size(), true);
    ASSERT_EQ(res, false);
    res = FindTermAVX2(vecs.data(), vecs.size(), false);
    ASSERT_EQ(res, true);

    for (int i = 0; i < 16; i++) {
        vecs.push_back(false);
    }

    res = FindTermAVX2(vecs.data(), vecs.size(), true);
    ASSERT_EQ(res, false);
    res = FindTermAVX2(vecs.data(), vecs.size(), false);
    ASSERT_EQ(res, true);

    vecs.push_back(true);
    for (int i = 0; i < 16; i++) {
        vecs.push_back(false);
    }
    res = FindTermAVX2(vecs.data(), vecs.size(), true);
    ASSERT_EQ(res, true);
}

TEST(FindTermAVX2, int8_type) {
    if (!cpu_support_avx2()) {
        PRINT_SKPI_TEST
        return;
    }
    std::vector<int8_t> vecs;
    for (int i = 0; i < 100; i++) {
        vecs.push_back(i);
    }

    auto res = FindTermAVX2(vecs.data(), vecs.size(), (int8_t)0);
    ASSERT_EQ(res, true);
    res = FindTermAVX2(vecs.data(), vecs.size(), (int8_t)10);
    ASSERT_EQ(res, true);
    res = FindTermAVX2(vecs.data(), vecs.size(), (int8_t)99);
    ASSERT_EQ(res, true);
    res = FindTermAVX2(vecs.data(), vecs.size(), (int8_t)100);
    ASSERT_EQ(res, false);
    res = FindTermAVX2(vecs.data(), vecs.size(), (int8_t)127);
    ASSERT_EQ(res, false);
    vecs.push_back(127);
    res = FindTermAVX2(vecs.data(), vecs.size(), (int8_t)127);
    ASSERT_EQ(res, true);
}

TEST(FindTermAVX2, int16_type) {
    if (!cpu_support_avx2()) {
        PRINT_SKPI_TEST
        return;
    }
    std::vector<int16_t> vecs;
    for (int i = 0; i < 1000; i++) {
        vecs.push_back(i);
    }

    auto res = FindTermAVX2(vecs.data(), vecs.size(), (int16_t)0);
    ASSERT_EQ(res, true);
    res = FindTermAVX2(vecs.data(), vecs.size(), (int16_t)10);
    ASSERT_EQ(res, true);
    res = FindTermAVX2(vecs.data(), vecs.size(), (int16_t)999);
    ASSERT_EQ(res, true);
    res = FindTermAVX2(vecs.data(), vecs.size(), (int16_t)1000);
    ASSERT_EQ(res, false);
    res = FindTermAVX2(vecs.data(), vecs.size(), (int16_t)1270);
    ASSERT_EQ(res, false);
    vecs.push_back(1270);
    res = FindTermAVX2(vecs.data(), vecs.size(), (int16_t)1270);
    ASSERT_EQ(res, true);
}

TEST(FindTermAVX2, int32_type) {
    if (!cpu_support_avx2()) {
        PRINT_SKPI_TEST
        return;
    }
    std::vector<int32_t> vecs;
    for (int i = 0; i < 1000; i++) {
        vecs.push_back(i);
    }

    auto res = FindTermAVX2(vecs.data(), vecs.size(), 0);
    ASSERT_EQ(res, true);
    res = FindTermAVX2(vecs.data(), vecs.size(), 10);
    ASSERT_EQ(res, true);
    res = FindTermAVX2(vecs.data(), vecs.size(), 999);
    ASSERT_EQ(res, true);
    res = FindTermAVX2(vecs.data(), vecs.size(), 1000);
    ASSERT_EQ(res, false);
    res = FindTermAVX2(vecs.data(), vecs.size(), 1270);
    ASSERT_EQ(res, false);
    vecs.push_back(1270);
    res = FindTermAVX2(vecs.data(), vecs.size(), 1270);
    ASSERT_EQ(res, true);
}

TEST(FindTermAVX2, int64_type) {
    if (!cpu_support_avx2()) {
        PRINT_SKPI_TEST
        return;
    }
    std::vector<int64_t> vecs;
    for (int i = 0; i < 1000; i++) {
        vecs.push_back(i);
    }

    auto res = FindTermAVX2(vecs.data(), vecs.size(), (int64_t)0);
    ASSERT_EQ(res, true);
    res = FindTermAVX2(vecs.data(), vecs.size(), (int64_t)10);
    ASSERT_EQ(res, true);
    res = FindTermAVX2(vecs.data(), vecs.size(), (int64_t)999);
    ASSERT_EQ(res, true);
    res = FindTermAVX2(vecs.data(), vecs.size(), (int64_t)1000);
    ASSERT_EQ(res, false);
    res = FindTermAVX2(vecs.data(), vecs.size(), (int64_t)1270);
    ASSERT_EQ(res, false);
    vecs.push_back(1270);
    res = FindTermAVX2(vecs.data(), vecs.size(), (int64_t)1270);
    ASSERT_EQ(res, true);
}

TEST(FindTermAVX2, float_type) {
    if (!cpu_support_avx2()) {
        PRINT_SKPI_TEST
        return;
    }
    std::vector<float> vecs;
    for (int i = 0; i < 10000; i++) {
        vecs.push_back(i + 0.01);
    }

    auto res = FindTermAVX2(vecs.data(), vecs.size(), (float)0.01);
    ASSERT_EQ(res, true);
    res = FindTermAVX2(vecs.data(), vecs.size(), (float)10.01);
    ASSERT_EQ(res, true);
    res = FindTermAVX2(vecs.data(), vecs.size(), (float)10000.01);
    ASSERT_EQ(res, false);
    res = FindTermAVX2(vecs.data(), vecs.size(), (float)12700.02);
    ASSERT_EQ(res, false);
    vecs.push_back(12700.02);
    res = FindTermAVX2(vecs.data(), vecs.size(), (float)12700.02);
    ASSERT_EQ(res, true);
}

TEST(FindTermAVX2, double_type) {
    if (!cpu_support_avx2()) {
        PRINT_SKPI_TEST
        return;
    }
    std::vector<double> vecs;
    for (int i = 0; i < 10000; i++) {
        vecs.push_back(i + 0.01);
    }

    auto res = FindTermAVX2(vecs.data(), vecs.size(), 0.01);
    ASSERT_EQ(res, true);
    res = FindTermAVX2(vecs.data(), vecs.size(), 10.01);
    ASSERT_EQ(res, true);
    res = FindTermAVX2(vecs.data(), vecs.size(), 10000.01);
    ASSERT_EQ(res, false);
    res = FindTermAVX2(vecs.data(), vecs.size(), 12700.01);
    ASSERT_EQ(res, false);
    vecs.push_back(12700.01);
    res = FindTermAVX2(vecs.data(), vecs.size(), 12700.01);
    ASSERT_EQ(res, true);
}

TEST(FindTermAVX512, bool_type) {
    if (!cpu_support_avx512()) {
        PRINT_SKPI_TEST
        return;
    }
    std::vector<int64_t> srcs;
    for (size_t i = 0; i < 1000; i++) {
        srcs.push_back(i);
    }
    FixedVector<bool> vecs;
    vecs.push_back(false);

    auto res = FindTermAVX512(vecs.data(), vecs.size(), true);
    ASSERT_EQ(res, false);
    res = FindTermAVX512(vecs.data(), vecs.size(), false);
    ASSERT_EQ(res, true);

    for (int i = 0; i < 16; i++) {
        vecs.push_back(false);
    }

    res = FindTermAVX512(vecs.data(), vecs.size(), true);
    ASSERT_EQ(res, false);
    res = FindTermAVX512(vecs.data(), vecs.size(), false);
    ASSERT_EQ(res, true);

    vecs.push_back(true);
    for (int i = 0; i < 16; i++) {
        vecs.push_back(false);
    }
    res = FindTermAVX512(vecs.data(), vecs.size(), true);
    ASSERT_EQ(res, true);
}

TEST(FindTermAVX512, int8_type) {
    if (!cpu_support_avx512()) {
        PRINT_SKPI_TEST
        return;
    }
    std::vector<int8_t> vecs;
    for (int i = 0; i < 100; i++) {
        vecs.push_back(i);
    }

    auto res = FindTermAVX512(vecs.data(), vecs.size(), (int8_t)0);
    ASSERT_EQ(res, true);
    res = FindTermAVX512(vecs.data(), vecs.size(), (int8_t)10);
    ASSERT_EQ(res, true);
    res = FindTermAVX512(vecs.data(), vecs.size(), (int8_t)99);
    ASSERT_EQ(res, true);
    res = FindTermAVX512(vecs.data(), vecs.size(), (int8_t)100);
    ASSERT_EQ(res, false);
    res = FindTermAVX512(vecs.data(), vecs.size(), (int8_t)127);
    ASSERT_EQ(res, false);
    vecs.push_back(127);
    res = FindTermAVX512(vecs.data(), vecs.size(), (int8_t)127);
    ASSERT_EQ(res, true);
}

TEST(FindTermAVX512, int16_type) {
    if (!cpu_support_avx512()) {
        PRINT_SKPI_TEST
        return;
    }
    std::vector<int16_t> vecs;
    for (int i = 0; i < 1000; i++) {
        vecs.push_back(i);
    }

    auto res = FindTermAVX512(vecs.data(), vecs.size(), (int16_t)0);
    ASSERT_EQ(res, true);
    res = FindTermAVX512(vecs.data(), vecs.size(), (int16_t)10);
    ASSERT_EQ(res, true);
    res = FindTermAVX512(vecs.data(), vecs.size(), (int16_t)999);
    ASSERT_EQ(res, true);
    res = FindTermAVX512(vecs.data(), vecs.size(), (int16_t)1000);
    ASSERT_EQ(res, false);
    res = FindTermAVX512(vecs.data(), vecs.size(), (int16_t)1270);
    ASSERT_EQ(res, false);
    vecs.push_back(1270);
    res = FindTermAVX512(vecs.data(), vecs.size(), (int16_t)1270);
    ASSERT_EQ(res, true);
}

TEST(FindTermAVX512, int32_type) {
    if (!cpu_support_avx512()) {
        PRINT_SKPI_TEST
        return;
    }
    std::vector<int32_t> vecs;
    for (int i = 0; i < 1000; i++) {
        vecs.push_back(i);
    }

    auto res = FindTermAVX512(vecs.data(), vecs.size(), 0);
    ASSERT_EQ(res, true);
    res = FindTermAVX512(vecs.data(), vecs.size(), 10);
    ASSERT_EQ(res, true);
    res = FindTermAVX512(vecs.data(), vecs.size(), 999);
    ASSERT_EQ(res, true);
    res = FindTermAVX512(vecs.data(), vecs.size(), 1000);
    ASSERT_EQ(res, false);
    res = FindTermAVX512(vecs.data(), vecs.size(), 1270);
    ASSERT_EQ(res, false);
    vecs.push_back(1270);
    res = FindTermAVX512(vecs.data(), vecs.size(), 1270);
    ASSERT_EQ(res, true);
}

TEST(FindTermAVX512, int64_type) {
    if (!cpu_support_avx512()) {
        PRINT_SKPI_TEST
        return;
    }
    std::vector<int64_t> vecs;
    for (int i = 0; i < 1000; i++) {
        vecs.push_back(i);
    }

    auto res = FindTermAVX512(vecs.data(), vecs.size(), (int64_t)0);
    ASSERT_EQ(res, true);
    res = FindTermAVX512(vecs.data(), vecs.size(), (int64_t)10);
    ASSERT_EQ(res, true);
    res = FindTermAVX512(vecs.data(), vecs.size(), (int64_t)999);
    ASSERT_EQ(res, true);
    res = FindTermAVX512(vecs.data(), vecs.size(), (int64_t)1000);
    ASSERT_EQ(res, false);
    res = FindTermAVX512(vecs.data(), vecs.size(), (int64_t)1270);
    ASSERT_EQ(res, false);
    vecs.push_back(1270);
    res = FindTermAVX512(vecs.data(), vecs.size(), (int64_t)1270);
    ASSERT_EQ(res, true);
}

TEST(FindTermAVX512, float_type) {
    if (!cpu_support_avx512()) {
        PRINT_SKPI_TEST
        return;
    }
    std::vector<float> vecs;
    for (int i = 0; i < 10000; i++) {
        vecs.push_back(i + 0.01);
    }

    auto res = FindTermAVX512(vecs.data(), vecs.size(), (float)0.01);
    ASSERT_EQ(res, true);
    res = FindTermAVX512(vecs.data(), vecs.size(), (float)10.01);
    ASSERT_EQ(res, true);
    res = FindTermAVX512(vecs.data(), vecs.size(), (float)10000.01);
    ASSERT_EQ(res, false);
    res = FindTermAVX512(vecs.data(), vecs.size(), (float)12700.02);
    ASSERT_EQ(res, false);
    vecs.push_back(12700.02);
    res = FindTermAVX512(vecs.data(), vecs.size(), (float)12700.02);
    ASSERT_EQ(res, true);
}

TEST(StrCmpSS4, string_type) {
    if (!cpu_support_sse4_2()) {
        PRINT_SKPI_TEST
        return;
    }

    std::vector<string> s1;
    for (int i = 0; i < 1000; ++i) {
        s1.push_back("test" + std::to_string(i));
    }

    for (int i = 0; i < 1000; ++i) {
        auto res = StrCmpSSE4(s1[i].c_str(), "test0");
    }

    string s2;
    string s3;
    for (int i = 0; i < 1000; ++i) {
        s2.push_back('x');
    }
    for (int i = 0; i < 1000; ++i) {
        s3.push_back('x');
    }

    auto res = StrCmpSSE4(s2.c_str(), s3.c_str());
    std::cout << res << std::endl;
}

TEST(FindTermAVX512, double_type) {
    if (!cpu_support_avx512()) {
        PRINT_SKPI_TEST
        return;
    }
    std::vector<double> vecs;
    for (int i = 0; i < 10000; i++) {
        vecs.push_back(i + 0.01);
    }

    auto res = FindTermAVX512(vecs.data(), vecs.size(), 0.01);
    ASSERT_EQ(res, true);
    res = FindTermAVX512(vecs.data(), vecs.size(), 10.01);
    ASSERT_EQ(res, true);
    res = FindTermAVX512(vecs.data(), vecs.size(), 10000.01);
    ASSERT_EQ(res, false);
    res = FindTermAVX512(vecs.data(), vecs.size(), 12700.01);
    ASSERT_EQ(res, false);
    vecs.push_back(12700.01);
    res = FindTermAVX512(vecs.data(), vecs.size(), 12700.01);
    ASSERT_EQ(res, true);
}

// Function to allocate aligned memory
void*
alignedMalloc(size_t size, size_t alignment) {
    void* ptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        ptr = nullptr;
    }
    return ptr;
}

TEST(EqualVal, perf_int8) {
    if (!cpu_support_avx512()) {
        PRINT_SKPI_TEST
        return;
    }
    std::vector<int8_t> srcs(1000000);
    for (int i = 0; i < 1000000; ++i) {
        srcs[i] = i % 128;
    }
    FixedVector<bool> res(1000000);
    auto start = std::chrono::steady_clock::now();
    EqualValRef(srcs.data(), 1000000, (int8_t)10, res.data());
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - start)
                     .count()
              << std::endl;
    start = std::chrono::steady_clock::now();
    EqualValAVX2(srcs.data(), 1000000, (int8_t)10, res.data());
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - start)
                     .count()
              << std::endl;
    start = std::chrono::steady_clock::now();
    EqualValAVX512(srcs.data(), 1000000, (int8_t)10, res.data());
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - start)
                     .count()
              << std::endl;
}

TEST(EqualVal, perf_int16) {
    if (!cpu_support_avx512()) {
        PRINT_SKPI_TEST
        return;
    }
    std::vector<int16_t> srcs(1000000);
    for (int i = 0; i < 1000000; ++i) {
        srcs[i] = i;
    }
    FixedVector<bool> res(1000000);
    auto start = std::chrono::steady_clock::now();
    EqualValRef(srcs.data(), 1000000, (int16_t)10, res.data());
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - start)
                     .count()
              << std::endl;
    start = std::chrono::steady_clock::now();
    EqualValAVX512(srcs.data(), 1000000, (int16_t)10, res.data());
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - start)
                     .count()
              << std::endl;
}

TEST(EqualVal, pref_int32) {
    if (!cpu_support_avx512()) {
        PRINT_SKPI_TEST
        return;
    }
    std::vector<int32_t> srcs(1000000);
    for (int i = 0; i < 1000000; ++i) {
        srcs[i] = i;
    }
    FixedVector<bool> res(1000000);
    auto start = std::chrono::steady_clock::now();
    EqualValRef(srcs.data(), 1000000, (int32_t)10, res.data());
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - start)
                     .count()
              << std::endl;
    start = std::chrono::steady_clock::now();
    EqualValAVX512(srcs.data(), 1000000, (int32_t)10, res.data());
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - start)
                     .count()
              << std::endl;
}

TEST(EqualVal, perf_int64) {
    if (!cpu_support_avx512()) {
        PRINT_SKPI_TEST
        return;
    }
    std::vector<int64_t> srcs(1000000);
    for (int i = 0; i < 1000000; ++i) {
        srcs[i] = i;
    }
    FixedVector<bool> res(1000000);
    auto start = std::chrono::steady_clock::now();
    EqualValRef(srcs.data(), 1000000, (int64_t)10, res.data());
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - start)
                     .count()
              << std::endl;
    start = std::chrono::steady_clock::now();
    EqualValAVX512(srcs.data(), 1000000, (int64_t)10, res.data());
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - start)
                     .count()
              << std::endl;
}

TEST(EqualVal, perf_float) {
    if (!cpu_support_avx512()) {
        PRINT_SKPI_TEST
        return;
    }
    std::vector<float> srcs(1000000);
    for (int i = 0; i < 1000000; ++i) {
        srcs[i] = float(i);
    }
    FixedVector<bool> res(1000000);
    auto start = std::chrono::steady_clock::now();
    EqualValRef(srcs.data(), 1000000, (float)100.1, res.data());
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - start)
                     .count()
              << std::endl;
    start = std::chrono::steady_clock::now();
    EqualValAVX512(srcs.data(), 1000000, (float)100.1, res.data());
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - start)
                     .count()
              << std::endl;
}

TEST(EqualVal, perf_double) {
    if (!cpu_support_avx512()) {
        PRINT_SKPI_TEST
        return;
    }
    std::vector<double> srcs(1000000);
    for (int i = 0; i < 1000000; ++i) {
        srcs[i] = double(i);
    }
    FixedVector<bool> res(1000000);
    auto start = std::chrono::steady_clock::now();
    EqualValRef(srcs.data(), 1000000, (double)100.1, res.data());
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - start)
                     .count()
              << std::endl;
    start = std::chrono::steady_clock::now();
    EqualValAVX512(srcs.data(), 1000000, (double)100.1, res.data());
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - start)
                     .count()
              << std::endl;
}

TEST(EqualVal, avx512_int8) {
    if (!cpu_support_avx512()) {
        PRINT_SKPI_TEST
        return;
    }
    std::vector<int8_t> vecs;
    for (int i = 0; i < 100; ++i) {
        vecs.push_back(i);
    }
    FixedVector<bool> res(1000);
    FixedVector<bool> expect(1000);

    int8_t target = 10;
    EqualValAVX512(vecs.data(), 100, target, res.data());
    for (int i = 0; i < 100; i++) {
        expect[i] = vecs[i] == target;
    }
    for (int i = 0; i < 100; i++) {
        ASSERT_EQ(res[i], expect[i]) << i;
    }

    target = 99;
    EqualValAVX512(vecs.data(), 100, target, res.data());
    for (int i = 0; i < 100; i++) {
        expect[i] = vecs[i] == target;
    }
    for (int i = 0; i < 100; i++) {
        ASSERT_EQ(res[i], expect[i]) << i;
    }
}

TEST(EqualVal, avx512_int16) {
    if (!cpu_support_avx512()) {
        PRINT_SKPI_TEST
        return;
    }
    std::vector<int16_t> vecs;
    for (int i = 0; i < 100; ++i) {
        vecs.push_back(i);
    }
    FixedVector<bool> res(1000);
    FixedVector<bool> expect(1000);

    int16_t target = 10;
    EqualValAVX512(vecs.data(), 100, target, res.data());
    for (int i = 0; i < 100; i++) {
        expect[i] = vecs[i] == target;
    }
    for (int i = 0; i < 100; i++) {
        ASSERT_EQ(res[i], expect[i]) << i;
    }

    target = 99;
    EqualValAVX512(vecs.data(), 100, target, res.data());
    for (int i = 0; i < 100; i++) {
        expect[i] = vecs[i] == target;
    }
    for (int i = 0; i < 100; i++) {
        ASSERT_EQ(res[i], expect[i]) << i;
    }
}

TEST(EqualVal, avx512_int32) {
    if (!cpu_support_avx512()) {
        PRINT_SKPI_TEST
        return;
    }
    std::vector<int32_t> vecs;
    for (int i = 0; i < 1000; ++i) {
        vecs.push_back(i);
    }
    FixedVector<bool> res(1000);
    FixedVector<bool> expect(1000);

    int32_t target = 10;
    EqualValAVX512(vecs.data(), 1000, target, res.data());
    for (int i = 0; i < 1000; i++) {
        expect[i] = vecs[i] == target;
    }
    for (int i = 0; i < 1000; i++) {
        ASSERT_EQ(res[i], expect[i]) << i;
    }

    target = 99;
    EqualValAVX512(vecs.data(), 1000, target, res.data());
    for (int i = 0; i < 1000; i++) {
        expect[i] = vecs[i] == target;
    }
    for (int i = 0; i < 1000; i++) {
        ASSERT_EQ(res[i], expect[i]) << i;
    }

    target = 999;
    EqualValAVX512(vecs.data(), 1000, target, res.data());
    for (int i = 0; i < 1000; i++) {
        expect[i] = vecs[i] == target;
    }
    for (int i = 0; i < 1000; i++) {
        ASSERT_EQ(res[i], expect[i]) << i;
    }
}
TEST(EqualVal, avx512_int64) {
    if (!cpu_support_avx512()) {
        PRINT_SKPI_TEST
        return;
    }
    std::vector<int64_t> vecs;
    for (int i = 0; i < 1000; ++i) {
        vecs.push_back(i);
    }
    FixedVector<bool> res(1000);
    FixedVector<bool> expect(1000);

    int64_t target = 10;
    EqualValAVX512(vecs.data(), 1000, target, res.data());
    for (int i = 0; i < 1000; i++) {
        expect[i] = vecs[i] == target;
    }
    for (int i = 0; i < 1000; i++) {
        ASSERT_EQ(res[i], expect[i]) << i;
    }

    target = 99;
    EqualValAVX512(vecs.data(), 1000, target, res.data());
    for (int i = 0; i < 1000; i++) {
        expect[i] = vecs[i] == target;
    }
    for (int i = 0; i < 1000; i++) {
        ASSERT_EQ(res[i], expect[i]) << i;
    }

    target = 999;
    EqualValAVX512(vecs.data(), 1000, target, res.data());
    for (int i = 0; i < 1000; i++) {
        expect[i] = vecs[i] == target;
    }
    for (int i = 0; i < 1000; i++) {
        ASSERT_EQ(res[i], expect[i]) << i;
    }
}

TEST(EqualVal, avx512_float) {
    if (!cpu_support_avx512()) {
        PRINT_SKPI_TEST
        return;
    }
    std::vector<float> vecs;
    for (int i = 0; i < 1000; ++i) {
        vecs.push_back(float(i + 0.01));
    }
    FixedVector<bool> res(1000);
    FixedVector<bool> expect(1000);

    float target = 10.01;
    EqualValAVX512(vecs.data(), 1000, target, res.data());
    for (int i = 0; i < 1000; i++) {
        expect[i] = vecs[i] == target;
    }
    for (int i = 0; i < 1000; i++) {
        ASSERT_EQ(res[i], expect[i]) << i;
    }

    target = 99.01;
    EqualValAVX512(vecs.data(), 1000, target, res.data());
    for (int i = 0; i < 1000; i++) {
        expect[i] = vecs[i] == target;
    }
    for (int i = 0; i < 1000; i++) {
        ASSERT_EQ(res[i], expect[i]) << i;
    }

    target = 999.01;
    EqualValAVX512(vecs.data(), 1000, target, res.data());
    for (int i = 0; i < 1000; i++) {
        expect[i] = vecs[i] == target;
    }
    for (int i = 0; i < 1000; i++) {
        ASSERT_EQ(res[i], expect[i]) << i;
    }
}

TEST(EqualVal, avx512_double) {
    if (!cpu_support_avx512()) {
        PRINT_SKPI_TEST
        return;
    }
    std::vector<double> vecs;
    for (int i = 0; i < 1000; ++i) {
        vecs.push_back(i + 0.001);
    }
    FixedVector<bool> res(1000);
    FixedVector<bool> expect(1000);

    double target = 10.001;
    EqualValAVX512(vecs.data(), 1000, target, res.data());
    for (int i = 0; i < 1000; i++) {
        expect[i] = vecs[i] == target;
    }
    for (int i = 0; i < 1000; i++) {
        ASSERT_EQ(res[i], expect[i]) << i;
    }

    target = 99.001;
    EqualValAVX512(vecs.data(), 1000, target, res.data());
    for (int i = 0; i < 1000; i++) {
        expect[i] = vecs[i] == target;
    }
    for (int i = 0; i < 1000; i++) {
        ASSERT_EQ(res[i], expect[i]) << i;
    }

    target = 999.001;
    EqualValAVX512(vecs.data(), 1000, target, res.data());
    for (int i = 0; i < 1000; i++) {
        expect[i] = vecs[i] == target;
    }
    for (int i = 0; i < 1000; i++) {
        ASSERT_EQ(res[i], expect[i]) << i;
    }
}

#endif

int
main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}