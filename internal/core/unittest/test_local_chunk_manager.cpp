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
        chunkManager_ = make_shared<LocalChunkManager>("");
    }

    virtual void
    TearDown() {
        chunkManager_.reset();
    }

    LocalChunkManagerSPtr chunkManager_;
};

TEST_F(LocalChunkManagerTest, DirPositive) {
    string pathPrex = "/tmp/local-test-dir";
    chunkManager_->SetPathPrefix(pathPrex);
    chunkManager_->RemoveDir(pathPrex);
    chunkManager_->CreateDir(pathPrex);

    bool exist = chunkManager_->DirExist(pathPrex);
    EXPECT_EQ(exist, true);

    chunkManager_->RemoveDir(pathPrex);
    exist = chunkManager_->DirExist(pathPrex);
    EXPECT_EQ(exist, false);
}

TEST_F(LocalChunkManagerTest, FilePositive) {
    string pathPrex = "/tmp/local-test-file";
    chunkManager_->SetPathPrefix(pathPrex);
    chunkManager_->RemoveDir(pathPrex);
    chunkManager_->CreateDir(pathPrex);
    bool exist = chunkManager_->DirExist(pathPrex);
    EXPECT_EQ(exist, true);

    string file = "test-file";
    exist = chunkManager_->Exist(file);
    EXPECT_EQ(exist, false);
    LocalChunkManager::CreateFile(pathPrex + "/" + file);
    exist = chunkManager_->Exist(file);
    EXPECT_EQ(exist, true);

    chunkManager_->Remove(file);
    exist = chunkManager_->Exist(file);
    EXPECT_EQ(exist, false);
}

TEST_F(LocalChunkManagerTest, WritePositive) {
    string pathPrex = "/tmp/local-test-write";
    chunkManager_->SetPathPrefix(pathPrex);
    chunkManager_->RemoveDir(pathPrex);
    chunkManager_->CreateDir(pathPrex);
    bool exist = chunkManager_->DirExist(pathPrex);
    EXPECT_EQ(exist, true);

    string file = "test-write";
    exist = chunkManager_->Exist(file);
    EXPECT_EQ(exist, false);

    uint8_t data[5] = {0x17, 0x32, 0x45, 0x34, 0x23};
    chunkManager_->Write(file, data, sizeof(data));

    exist = chunkManager_->Exist(file);
    EXPECT_EQ(exist, true);
    auto size = chunkManager_->Size(file);
    EXPECT_EQ(size, 5);

    int datasize = 10000;
    uint8_t* bigdata = new uint8_t[datasize];
    srand((unsigned)time(NULL));
    for (int i = 0; i < datasize; ++i) {
        bigdata[i] = rand() % 256;
    }
    chunkManager_->Write(file, bigdata, datasize);
    size = chunkManager_->Size(file);
    EXPECT_EQ(size, datasize);
    delete bigdata;
}

TEST_F(LocalChunkManagerTest, ReadPositive) {
    string pathPrefix = "/tmp/test-read";
    chunkManager_->SetPathPrefix(pathPrefix);
    chunkManager_->RemoveDir(pathPrefix);
    chunkManager_->CreateDir(pathPrefix);

    uint8_t data[5] = {0x17, 0x32, 0x45, 0x34, 0x23};
    string path = "test-read";
    chunkManager_->Write(path, data, sizeof(data));
    bool exist = chunkManager_->Exist(path);
    EXPECT_EQ(exist, true);
    auto size = chunkManager_->Size(path);
    EXPECT_EQ(size, 5);

    uint8_t readdata[20] = {0};
    size = chunkManager_->Read(path, readdata, 20);
    EXPECT_EQ(size, 5);
    EXPECT_EQ(readdata[0], 0x17);
    EXPECT_EQ(readdata[1], 0x32);
    EXPECT_EQ(readdata[2], 0x45);
    EXPECT_EQ(readdata[3], 0x34);
    EXPECT_EQ(readdata[4], 0x23);

    size = chunkManager_->Read(path, readdata, 3);
    EXPECT_EQ(size, 3);
    EXPECT_EQ(readdata[0], 0x17);
    EXPECT_EQ(readdata[1], 0x32);
    EXPECT_EQ(readdata[2], 0x45);

    uint8_t dataWithNULL[] = {0x17, 0x32, 0x00, 0x34, 0x23};
    chunkManager_->Write(path, dataWithNULL, sizeof(dataWithNULL));
    exist = chunkManager_->Exist(path);
    EXPECT_EQ(exist, true);
    size = chunkManager_->Size(path);
    EXPECT_EQ(size, 5);
    size = chunkManager_->Read(path, readdata, 20);
    EXPECT_EQ(size, 5);
    EXPECT_EQ(readdata[0], 0x17);
    EXPECT_EQ(readdata[1], 0x32);
    EXPECT_EQ(readdata[2], 0x00);
    EXPECT_EQ(readdata[3], 0x34);
    EXPECT_EQ(readdata[4], 0x23);
}
