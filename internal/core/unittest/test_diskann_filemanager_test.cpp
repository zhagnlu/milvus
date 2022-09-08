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
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "storage/Event.h"
#include "storage/MinioChunkManager.h"
#include "storage/LocalChunkManager.h"
#include "storage/DiskANNFileManagerImpl.h"
#include "config/ConfigChunkManager.h"
#include "config/ConfigKnowhere.h"

using namespace std;
using namespace milvus;
using namespace milvus::storage;
using namespace boost::filesystem;
using namespace knowhere;

class DiskAnnFileManagerTest : public testing::Test {
 public:
    DiskAnnFileManagerTest() {
    }
    ~DiskAnnFileManagerTest() {
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

    void
    InitRemoteChunkManager() {
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
        config::ChunkMangerConfig::SetAddress(endpoint);
        config::ChunkMangerConfig::SetAccessKey(accessKey);
        config::ChunkMangerConfig::SetAccessValue(accessValue);
        config::ChunkMangerConfig::SetBucketName(bucketName);
        config::ChunkMangerConfig::SetUseSSL(useSSL);
    }

    void
    InitLocalChunkManager() {
        config::ChunkMangerConfig::SetLocalBucketName("/tmp/milvus");
    }

    virtual void
    SetUp() {
        InitLocalChunkManager();
        InitRemoteChunkManager();
    }
};

void
GenerateLocalIndexFile(const std::string path, uint64_t indexSize) {
    uint32_t degree = 4;
    std::fstream file;
    file.open(path, ios::binary | ios::out);
    file.write((char*)(&indexSize), sizeof(indexSize));
    file.write((char*)(&degree), sizeof(degree));
    uint8_t start = 0x01;
    for (uint8_t i = start; i < indexSize + start; ++i) {
        file.write((char*)(&i), 1);
    }
    file.close();
}

TEST_F(DiskAnnFileManagerTest, AddFilePositive) {
    auto& lcm = LocalChunkManager::GetInstance();
    auto& rcm = MinioChunkManager::GetInstance();
    auto buildId = 1;
    string localPath = "/tmp/test_add_file";
    lcm.RemoveDir(localPath);
    lcm.CreateDir(localPath);
    lcm.SetPathPrefix(localPath);
    auto buildDir = localPath + "/" + std::to_string(buildId);
    lcm.RemoveDir(buildDir);
    lcm.CreateDir(buildDir);

    string testBucketName = "test-diskann";
    rcm.SetBucketName(testBucketName);
    EXPECT_EQ(rcm.GetBucketName(), testBucketName);

    if (!rcm.BucketExists(testBucketName)) {
        rcm.CreateBucket(testBucketName);
    }

    // local path patten ...../buildId/index_file
    std::string indexFilePath = buildDir + "/index";
    uint64_t indexSize = 104;
    GenerateLocalIndexFile(indexFilePath, indexSize);

    // collection_id: 1, partition_id: 2, segment_id: 3
    // field_id: 100, index_build_id: 1000, index_version: 1
    FieldDataMeta filed_data_meta = {1, 2, 3, 100};
    IndexMeta index_meta = {3, 100, 1000, 1, "index_test_key"};

    int sliceSize = 5;
    config::KnowhereSetIndexSliceSize(sliceSize);
    auto diskAnnFileManager = std::make_shared<DiskANNFileManagerImpl>(filed_data_meta, index_meta);

    diskAnnFileManager->AddFile("/tmp/test_add_file/" + std::to_string(buildId) + "/index");

    // check result
    auto remotePrefix = diskAnnFileManager->GetRemoteIndexObjectPrefix();
    auto remoteIndexFiles = rcm.ListWithPrefix(remotePrefix);
    auto num_slice = indexSize / (sliceSize << 20);
    EXPECT_EQ(remoteIndexFiles.size(), indexSize % (sliceSize << 20) == 0 ? num_slice : num_slice + 1);

    diskAnnFileManager->CacheIndexToDisk(remoteIndexFiles);
    auto fileSize1 = rcm.Size(remoteIndexFiles[0]);
    auto buf = std::unique_ptr<uint8_t[]>(new uint8_t[fileSize1]);
    rcm.Read(remoteIndexFiles[0], buf.get(), fileSize1);

    auto index = DeserializeFileData(buf.get(), fileSize1);
    auto payload = index->GetPayload();
    auto rows = payload->rows;
    auto rawData = payload->raw_data;
    EXPECT_EQ(rows, sliceSize);
    EXPECT_EQ(rawData[0], 1);
    EXPECT_EQ(rawData[4], 5);
}
