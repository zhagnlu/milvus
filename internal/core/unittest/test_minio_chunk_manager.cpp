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

#include <boost/filesystem.hpp>
#include <boost/system/error_code.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "storage/MinioChunkManager.h"

using namespace std;
using namespace milvus;
using namespace milvus::storage;
using namespace boost::filesystem;

class MinioChunkManagerTest : public testing::Test {
 public:
    MinioChunkManagerTest() {
    }
    ~MinioChunkManagerTest() {
    }

    bool
    FindFile(const path& dir, const string& file_name, path& path_found) {
        const recursive_directory_iterator end;
        boost::system::error_code err;
        auto iter = recursive_directory_iterator(dir, err);
        while (iter != end) {
            try {
                if ((*iter).path().filename() == file_name) {
                    path_found = (*iter).path();
                    return true;
                }
                iter++;
            } catch (filesystem_error& e) {
            } catch (std::exception& e) {
                // ignore error
            }
        }
        return false;
    }

    string
    GetConfig() {
        char testPath[100];
        auto pwd = string(getcwd(testPath, sizeof(testPath)));
        path filepath;
        auto currentPath = path(pwd);
        while (!FindFile(currentPath, "milvus.yaml", filepath)) {
            currentPath = currentPath.append("../");
        }
        return filepath.string();
    }

    virtual void
    SetUp() {
        auto configPath = GetConfig();
        cout << configPath << endl;
        YAML::Node config;
        config = YAML::LoadFile(configPath);
        auto minioConfig = config["minio"];
        auto address = minioConfig["address"].as<string>();
        auto port = minioConfig["port"].as<string>();
        auto endpoint = address + ":" + port;
        auto accessKey = minioConfig["accessKeyID"].as<string>();
        auto accessValue = minioConfig["secretAccessKey"].as<string>();
        auto useSSL = minioConfig["useSSL"].as<bool>();
        auto bucketName = minioConfig["bucketName"].as<string>();
        chunkManager_ = std::make_shared<MinioChunkManager>(endpoint, accessKey, accessValue, bucketName, useSSL);
    }

    virtual void
    TearDown() {
        chunkManager_.reset();
    }

    MinioChunkManagerSPtr chunkManager_;
};

TEST_F(MinioChunkManagerTest, BucketPositive) {
    string testBucketName = "test-bucket";
    chunkManager_->SetBucketName(testBucketName);
    chunkManager_->DeleteBucket(testBucketName);
    bool exist = chunkManager_->BucketExists(testBucketName);
    EXPECT_EQ(exist, false);
    chunkManager_->CreateBucket(testBucketName);
    exist = chunkManager_->BucketExists(testBucketName);
    EXPECT_EQ(exist, true);
}

TEST_F(MinioChunkManagerTest, BucketNegtive) {
    string testBucketName = "test-bucket-ng";
    chunkManager_->SetBucketName(testBucketName);
    chunkManager_->DeleteBucket(testBucketName);

    // create already exist bucket
    chunkManager_->CreateBucket(testBucketName);
    try {
        chunkManager_->CreateBucket(testBucketName);
    } catch (S3ErrorException& e) {
        EXPECT_TRUE(std::string(e.what()).find("BucketAlreadyOwnedByYou") != string::npos);
    }
}

TEST_F(MinioChunkManagerTest, ObjectExist) {
    string testBucketName = "test-objexist";
    string objPath = "1/3";
    chunkManager_->SetBucketName(testBucketName);
    if (!chunkManager_->BucketExists(testBucketName)) {
        chunkManager_->CreateBucket(testBucketName);
    }

    bool exist = chunkManager_->Exist(objPath);
    EXPECT_EQ(exist, false);
}

TEST_F(MinioChunkManagerTest, WritePositive) {
    string testBucketName = "test-write";
    chunkManager_->SetBucketName(testBucketName);
    EXPECT_EQ(chunkManager_->GetBucketName(), testBucketName);

    if (!chunkManager_->BucketExists(testBucketName)) {
        chunkManager_->CreateBucket(testBucketName);
    }
    uint8_t data[5] = {0x17, 0x32, 0x45, 0x34, 0x23};
    string path = "1/3/5";
    chunkManager_->Write(path, data, sizeof(data));

    bool exist = chunkManager_->Exist(path);
    EXPECT_EQ(exist, true);

    auto size = chunkManager_->Size(path);
    EXPECT_EQ(size, 5);

    int datasize = 10000;
    uint8_t* bigdata = new uint8_t[datasize];
    srand((unsigned)time(NULL));
    for (int i = 0; i < datasize; ++i) {
        bigdata[i] = rand() % 256;
    }
    chunkManager_->Write(path, bigdata, datasize);
    size = chunkManager_->Size(path);
    EXPECT_EQ(size, datasize);
    delete bigdata;
}

TEST_F(MinioChunkManagerTest, ReadPositive) {
    string testBucketName = "test-read";
    chunkManager_->SetBucketName(testBucketName);
    EXPECT_EQ(chunkManager_->GetBucketName(), testBucketName);

    if (!chunkManager_->BucketExists(testBucketName)) {
        chunkManager_->CreateBucket(testBucketName);
    }
    uint8_t data[5] = {0x17, 0x32, 0x45, 0x34, 0x23};
    string path = "1/4/6";
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

TEST_F(MinioChunkManagerTest, RemovePositive) {
    string testBucketName = "test-remove";
    chunkManager_->SetBucketName(testBucketName);
    EXPECT_EQ(chunkManager_->GetBucketName(), testBucketName);

    if (!chunkManager_->BucketExists(testBucketName)) {
        chunkManager_->CreateBucket(testBucketName);
    }
    uint8_t data[5] = {0x17, 0x32, 0x45, 0x34, 0x23};
    string path = "1/7/8";
    chunkManager_->Write(path, data, sizeof(data));

    bool exist = chunkManager_->Exist(path);
    EXPECT_EQ(exist, true);

    chunkManager_->Remove(path);

    exist = chunkManager_->Exist(path);
    EXPECT_EQ(exist, false);
}

TEST_F(MinioChunkManagerTest, ListWithPrefixPositive) {
    string testBucketName = "test-listprefix";
    chunkManager_->SetBucketName(testBucketName);
    EXPECT_EQ(chunkManager_->GetBucketName(), testBucketName);

    if (!chunkManager_->BucketExists(testBucketName)) {
        chunkManager_->CreateBucket(testBucketName);
    }

    string path1 = "1/7/8";
    string path2 = "1/7/4";
    string path3 = "1/4/8";
    uint8_t data[5] = {0x17, 0x32, 0x45, 0x34, 0x23};
    chunkManager_->Write(path1, data, sizeof(data));
    chunkManager_->Write(path2, data, sizeof(data));
    chunkManager_->Write(path3, data, sizeof(data));

    vector<string> objs = chunkManager_->ListWithPrefix("1/7");
    EXPECT_EQ(objs.size(), 2);
    std::sort(objs.begin(), objs.end());
    EXPECT_EQ(objs[0], "1/7/4");
    EXPECT_EQ(objs[1], "1/7/8");

    objs = chunkManager_->ListWithPrefix("//1/7");
    EXPECT_EQ(objs.size(), 2);

    objs = chunkManager_->ListWithPrefix("1");
    EXPECT_EQ(objs.size(), 3);
    std::sort(objs.begin(), objs.end());
    EXPECT_EQ(objs[0], "1/4/8");
    EXPECT_EQ(objs[1], "1/7/4");
}
