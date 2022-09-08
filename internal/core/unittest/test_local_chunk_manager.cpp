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

#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "storage/LocalChunkManager.h"

using namespace std;
using namespace milvus;
using namespace milvus::storage;

class LocalChunkManagerTest : public testing::Test {
 public:
    LocalChunkManagerTest() {
    }
    ~LocalChunkManagerTest() {
    }

    virtual void
    SetUp() {
        config::ChunkMangerConfig::SetLocalBucketName("/tmp/local-test-dir");
    }
};

TEST_F(LocalChunkManagerTest, DirPositive) {
    auto& lcm = LocalChunkManager::GetInstance();
    string pathPrex = "/tmp/local-test-dir";
    lcm.SetPathPrefix(pathPrex);
    lcm.RemoveDir(pathPrex);
    lcm.CreateDir(pathPrex);

    bool exist = lcm.DirExist(pathPrex);
    EXPECT_EQ(exist, true);

    lcm.RemoveDir(pathPrex);
    exist = lcm.DirExist(pathPrex);
    EXPECT_EQ(exist, false);
}

TEST_F(LocalChunkManagerTest, FilePositive) {
    auto& lcm = LocalChunkManager::GetInstance();
    string pathPrex = "/tmp/local-test-file";
    lcm.SetPathPrefix(pathPrex);
    lcm.RemoveDir(pathPrex);
    lcm.CreateDir(pathPrex);
    bool exist = lcm.DirExist(pathPrex);
    EXPECT_EQ(exist, true);

    string file = "/tmp/test-file";
    exist = lcm.Exist(file);
    EXPECT_EQ(exist, false);
    lcm.CreateFile(file);
    exist = lcm.Exist(file);
    EXPECT_EQ(exist, true);

    lcm.Remove(file);
    exist = lcm.Exist(file);
    EXPECT_EQ(exist, false);
}

TEST_F(LocalChunkManagerTest, WritePositive) {
    auto& lcm = LocalChunkManager::GetInstance();
    string pathPrex = "/tmp/local-test-write";
    lcm.SetPathPrefix(pathPrex);
    lcm.RemoveDir(pathPrex);
    lcm.CreateDir(pathPrex);
    bool exist = lcm.DirExist(pathPrex);
    EXPECT_EQ(exist, true);

    string file = "/tmp/test-write";
    exist = lcm.Exist(file);
    EXPECT_EQ(exist, false);

    uint8_t data[5] = {0x17, 0x32, 0x45, 0x34, 0x23};
    lcm.Write(file, data, sizeof(data));

    exist = lcm.Exist(file);
    EXPECT_EQ(exist, true);
    auto size = lcm.Size(file);
    EXPECT_EQ(size, 5);

    int datasize = 10000;
    uint8_t* bigdata = new uint8_t[datasize];
    srand((unsigned)time(NULL));
    for (int i = 0; i < datasize; ++i) {
        bigdata[i] = rand() % 256;
    }
    lcm.Write(file, bigdata, datasize);
    size = lcm.Size(file);
    EXPECT_EQ(size, datasize);
    delete[] bigdata;
}

TEST_F(LocalChunkManagerTest, ReadPositive) {
    auto& lcm = LocalChunkManager::GetInstance();
    string pathPrefix = "/tmp/test-read";
    lcm.SetPathPrefix(pathPrefix);
    lcm.RemoveDir(pathPrefix);
    lcm.CreateDir(pathPrefix);

    uint8_t data[5] = {0x17, 0x32, 0x45, 0x34, 0x23};
    string path = "/tmp/test-read";
    lcm.Write(path, data, sizeof(data));
    bool exist = lcm.Exist(path);
    EXPECT_EQ(exist, true);
    auto size = lcm.Size(path);
    EXPECT_EQ(size, 5);

    uint8_t readdata[20] = {0};
    size = lcm.Read(path, readdata, 20);
    EXPECT_EQ(size, 5);
    EXPECT_EQ(readdata[0], 0x17);
    EXPECT_EQ(readdata[1], 0x32);
    EXPECT_EQ(readdata[2], 0x45);
    EXPECT_EQ(readdata[3], 0x34);
    EXPECT_EQ(readdata[4], 0x23);

    size = lcm.Read(path, readdata, 3);
    EXPECT_EQ(size, 3);
    EXPECT_EQ(readdata[0], 0x17);
    EXPECT_EQ(readdata[1], 0x32);
    EXPECT_EQ(readdata[2], 0x45);

    uint8_t dataWithNULL[] = {0x17, 0x32, 0x00, 0x34, 0x23};
    lcm.Write(path, dataWithNULL, sizeof(dataWithNULL));
    exist = lcm.Exist(path);
    EXPECT_EQ(exist, true);
    size = lcm.Size(path);
    EXPECT_EQ(size, 5);
    size = lcm.Read(path, readdata, 20);
    EXPECT_EQ(size, 5);
    EXPECT_EQ(readdata[0], 0x17);
    EXPECT_EQ(readdata[1], 0x32);
    EXPECT_EQ(readdata[2], 0x00);
    EXPECT_EQ(readdata[3], 0x34);
    EXPECT_EQ(readdata[4], 0x23);
}

TEST_F(LocalChunkManagerTest, WriteOffset) {
    auto& lcm = LocalChunkManager::GetInstance();
    string pathPrex = "/tmp/test-writeoffset";
    lcm.SetPathPrefix(pathPrex);
    lcm.RemoveDir(pathPrex);
    lcm.CreateDir(pathPrex);

    string file = "/tmp/test-write";
    auto exist = lcm.Exist(file);
    EXPECT_EQ(exist, false);
    lcm.CreateFile(file);
    exist = lcm.Exist(file);
    EXPECT_EQ(exist, true);

    int offset = 0;
    uint8_t data[5] = {0x17, 0x32, 0x00, 0x34, 0x23};
    lcm.Write(file, offset, data, sizeof(data));

    exist = lcm.Exist(file);
    EXPECT_EQ(exist, true);
    auto size = lcm.Size(file);
    EXPECT_EQ(size, 5);

    offset = 5;
    lcm.Write(file, offset, data, sizeof(data));
    size = lcm.Size(file);
    EXPECT_EQ(size, 10);

    uint8_t readdata[20] = {0};
    size = lcm.Read(file, readdata, 20);
    EXPECT_EQ(size, 10);
    EXPECT_EQ(readdata[0], 0x17);
    EXPECT_EQ(readdata[1], 0x32);
    EXPECT_EQ(readdata[2], 0x00);
    EXPECT_EQ(readdata[3], 0x34);
    EXPECT_EQ(readdata[4], 0x23);
    EXPECT_EQ(readdata[5], 0x17);
    EXPECT_EQ(readdata[6], 0x32);
    EXPECT_EQ(readdata[7], 0x00);
    EXPECT_EQ(readdata[8], 0x34);
    EXPECT_EQ(readdata[9], 0x23);
}

TEST_F(LocalChunkManagerTest, ReadOffset) {
    auto& lcm = LocalChunkManager::GetInstance();
    string pathPrex = "/tmp/test-readoffset";
    lcm.SetPathPrefix(pathPrex);
    lcm.RemoveDir(pathPrex);
    lcm.CreateDir(pathPrex);

    string file = "/tmp/test-read";
    auto exist = lcm.Exist(file);
    EXPECT_EQ(exist, false);

    uint8_t data[] = {0x17, 0x32, 0x00, 0x34, 0x23, 0x23, 0x87, 0x98};
    lcm.Write(file, data, sizeof(data));

    exist = lcm.Exist(file);
    EXPECT_EQ(exist, true);

    uint8_t readdata[20];
    auto size = lcm.Read(file, 0, readdata, 3);
    EXPECT_EQ(size, 3);
    EXPECT_EQ(readdata[0], 0x17);
    EXPECT_EQ(readdata[1], 0x32);
    EXPECT_EQ(readdata[2], 0x00);
    size = lcm.Read(file, 3, readdata, 4);
    EXPECT_EQ(size, 4);
    EXPECT_EQ(readdata[0], 0x34);
    EXPECT_EQ(readdata[1], 0x23);
    EXPECT_EQ(readdata[2], 0x23);
    EXPECT_EQ(readdata[3], 0x87);
    size = lcm.Read(file, 7, readdata, 4);
    EXPECT_EQ(size, 1);
    EXPECT_EQ(readdata[0], 0x98);
}
