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

#include <chrono>
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <unistd.h>

#include "common/Slice.h"
#include "storage/Event.h"
#include "storage/LocalChunkManager.h"
#include "storage/MinioChunkManager.h"
#include "storage/DiskFileManagerImpl.h"
#include "storage/ThreadPool.h"
#include "config/ConfigChunkManager.h"
#include "config/ConfigKnowhere.h"
#include "test_utils/indexbuilder_test_utils.h"
#include "storage/PayloadReader.h"

using namespace std;
using namespace milvus;
using namespace milvus::storage;
using namespace boost::filesystem;
using namespace knowhere;

static size_t
getCurrentRSS() {
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE* fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
        return (size_t)0L; /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return (size_t)0L; /* Can't read? */
    }
    fclose(fp);
    return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);
}

class DiskAnnFileManagerTest : public testing::Test {
 public:
    DiskAnnFileManagerTest() {
    }
    ~DiskAnnFileManagerTest() {
    }

    virtual void
    SetUp() {
        ChunkMangerConfig::SetLocalRootPath("/tmp/diskann");
        storage_config_ = get_default_storage_config();
    }

 protected:
    StorageConfig storage_config_;
};

TEST_F(DiskAnnFileManagerTest, AddFilePositive) {
    auto& lcm = LocalChunkManager::GetInstance();
    auto rcm = std::make_unique<MinioChunkManager>(storage_config_);
    string testBucketName = "test-diskann";
    storage_config_.bucket_name = testBucketName;
    if (!rcm->BucketExists(testBucketName)) {
        rcm->CreateBucket(testBucketName);
    }

    std::string indexFilePath = "/tmp/diskann/index_files/1000/index";
    auto exist = lcm.Exist(indexFilePath);
    EXPECT_EQ(exist, false);
    uint64_t index_size = 1024;
    lcm.CreateFile(indexFilePath);
    std::vector<uint8_t> data(index_size);
    lcm.Write(indexFilePath, data.data(), index_size);

    // collection_id: 1, partition_id: 2, segment_id: 3
    // field_id: 100, index_build_id: 1000, index_version: 1
    FieldDataMeta filed_data_meta = {1, 2, 3, 100};
    IndexMeta index_meta = {3, 100, 1000, 1, "index"};

    int64_t slice_size = milvus::index_file_slice_size << 20;
    auto diskAnnFileManager = std::make_shared<DiskFileManagerImpl>(filed_data_meta, index_meta, storage_config_);
    auto ok = diskAnnFileManager->AddFile(indexFilePath);
    EXPECT_EQ(ok, true);

    auto remote_files_to_size = diskAnnFileManager->GetRemotePathsToFileSize();
    auto num_slice = index_size / slice_size;
    EXPECT_EQ(remote_files_to_size.size(), index_size % slice_size == 0 ? num_slice : num_slice + 1);

    std::vector<std::string> remote_files;
    for (auto& file2size : remote_files_to_size) {
        std::cout << file2size.first << std::endl;
        remote_files.emplace_back(file2size.first);
    }
    diskAnnFileManager->CacheIndexToDisk(remote_files);
    auto local_files = diskAnnFileManager->GetLocalFilePaths();
    for (auto& file : local_files) {
        auto file_size = lcm.Size(file);
        auto buf = std::unique_ptr<uint8_t[]>(new uint8_t[file_size]);
        lcm.Read(file, buf.get(), file_size);

        auto index = FieldData(buf.get(), file_size);
        auto payload = index.get_payload();
        auto rows = payload->rows;
        auto rawData = payload->raw_data;

        EXPECT_EQ(rows, index_size);
        EXPECT_EQ(rawData[0], data[0]);
        EXPECT_EQ(rawData[4], data[4]);
    }
}
void
testtt(StorageConfig& storage_config_) {
    auto& lcm = LocalChunkManager::GetInstance();
    auto rcm = std::make_unique<MinioChunkManager>(storage_config_);
    string testBucketName = "test-diskann";
    storage_config_.bucket_name = testBucketName;
    if (!rcm->BucketExists(testBucketName)) {
        rcm->CreateBucket(testBucketName);
    }

    std::string indexFilePath = "/tmp/diskann/index_files/1000/index";
    auto exist = lcm.Exist(indexFilePath);
    EXPECT_EQ(exist, false);
    uint64_t index_size = 300 << 20;

    lcm.CreateFile(indexFilePath);
    std::vector<uint8_t> data(index_size);
    for (int i = 0; i < index_size; ++i) {
        data[i] = random() % 256;
    }
    lcm.Write(indexFilePath, data.data(), index_size);

    // collection_id: 1, partition_id: 2, segment_id: 3
    // field_id: 100, index_build_id: 1000, index_version: 1
    FieldDataMeta filed_data_meta = {1, 2, 3, 100};
    IndexMeta index_meta = {3, 100, 1000, 1, "index"};

    int64_t slice_size = milvus::index_file_slice_size << 20;
    auto diskAnnFileManager = std::make_shared<DiskFileManagerImpl>(filed_data_meta, index_meta, storage_config_);
    auto ok = diskAnnFileManager->AddFile(indexFilePath);
    EXPECT_EQ(ok, true);

    auto remote_files_to_size = diskAnnFileManager->GetRemotePathsToFileSize();
    auto num_slice = index_size / slice_size;
    EXPECT_EQ(remote_files_to_size.size(), index_size % slice_size == 0 ? num_slice : num_slice + 1);

    std::vector<std::string> remote_files;
    for (auto& file2size : remote_files_to_size) {
        std::cout << file2size.first << std::endl;
        remote_files.emplace_back(file2size.first);
    }
    for (int i = 0; i < 1000; ++i) {
        sleep(5);
        std::cout << " cached done" << std::endl;
        diskAnnFileManager->CacheIndexToDisk(remote_files);
    }
    auto local_files = diskAnnFileManager->GetLocalFilePaths();
    for (auto& file : local_files) {
        auto file_size = lcm.Size(file);
        auto buf = std::unique_ptr<uint8_t[]>(new uint8_t[file_size]);
        lcm.Read(file, buf.get(), file_size);

        auto index = FieldData(buf.get(), file_size);
        auto payload = index.get_payload();
        auto rows = payload->rows;
        auto rawData = payload->raw_data;

        EXPECT_EQ(rows, index_size);
        EXPECT_EQ(rawData[0], data[0]);
        EXPECT_EQ(rawData[4], data[4]);
    }
}
TEST_F(DiskAnnFileManagerTest, AddFilePositiveParallel) {
    testtt(storage_config_);
}

int
test_worker(string s) {
    std::cout << s << std::endl;
    sleep(4);
    std::cout << s << std::endl;
    return 1;
}

TEST_F(DiskAnnFileManagerTest, TestThreadPool) {
    auto thread_pool = new milvus::ThreadPool(50);
    std::vector<std::future<int>> futures;
    auto start = chrono::system_clock::now();
    for (int i = 0; i < 100; i++) {
        futures.push_back(thread_pool->Submit(test_worker, "test_id" + std::to_string(i)));
    }
    for (auto& future : futures) {
        EXPECT_EQ(future.get(), 1);
    }
    auto end = chrono::system_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    auto second = double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;
    EXPECT_LT(second, 4 * 100);
}

int
test_exception(string s) {
    if (s == "test_id60") {
        throw std::runtime_error("run time error");
    }
    return 1;
}

TEST_F(DiskAnnFileManagerTest, TestThreadPoolException) {
    try {
        auto thread_pool = new milvus::ThreadPool(50);
        std::vector<std::future<int>> futures;
        for (int i = 0; i < 100; i++) {
            futures.push_back(thread_pool->Submit(test_exception, "test_id" + std::to_string(i)));
        }
        for (auto& future : futures) {
            future.get();
        }
    } catch (std::exception& e) {
        EXPECT_EQ(std::string(e.what()), "run time error");
    }
}

void
test_write(LocalChunkManager* lcm, std::string path, void* data, int size) {
    lcm->Write(path, data, size);
}

TEST_F(DiskAnnFileManagerTest, TESTWRITE) {
    auto& lcm = LocalChunkManager::GetInstance();
    lcm.CreateFile("/tmp/test.txt");
    int size = 40 << 20;
    std::cout << size << std::endl;
    std::vector<uint8_t> data(size);
    auto start = chrono::system_clock::now();
    int batch = size / 10;
    std::cout << batch << std::endl;
    for (int i = 0; i < size; i += batch) {
        lcm.Write("/tmp/test.txt", data.data() + i, batch);
        std::cout << i << std::endl;
    }
    auto end = chrono::system_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
    std::cout << "duration" << duration << "ms";

    start = chrono::system_clock::now();
    auto thread_pool = new milvus::ThreadPool(50);
    std::vector<std::future<void>> futures;
    for (int i = 0; i < size; i + batch) {
        futures.push_back(thread_pool->Submit(test_write, &lcm, "/tmp/test.txt", data.data() + i, batch));
    }
    for (auto& future : futures) {
        future.get();
    }
    end = chrono::system_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
    std::cout << "duration" << duration << "ms";
}

TEST_F(DiskAnnFileManagerTest, TESTPARQUET) {
    while (true) {
        auto mem_pool = arrow::default_memory_pool();

        std::unique_ptr<parquet::arrow::FileReader> reader;
        auto arrow_memory_mapped_file = arrow::io::MemoryMappedFile::Open("/tmp/a.parquet", arrow::io::FileMode::READ);
        std::shared_ptr<arrow::io::RandomAccessFile> input = arrow_memory_mapped_file.ValueOrDie();
        auto st = parquet::arrow::OpenFile(input, mem_pool, &reader);
        AssertInfo(st.ok(), "failed to get arrow file reader");
        std::shared_ptr<arrow::Table> table;
        std::cout << "read before" << getCurrentRSS() << std::endl;
        st = reader->ReadTable(&table);
        std::cout << "read after" << getCurrentRSS() << std::endl;
        AssertInfo(st.ok(), "failed to get reader data to arrow table");
        auto column = table->column(0);
        AssertInfo(column != nullptr, "returned arrow column is null");
        AssertInfo(column->chunks().size() == 1, "arrow chunk size in arrow column should be 1");
        auto array = column->chunk(0);
        AssertInfo(array != nullptr, "empty arrow array of PayloadReader");
        // sleep(1);
        std::cout << "read sleep" << getCurrentRSS() << std::endl;
    }
}

TEST_F(DiskAnnFileManagerTest, TESTPARQUET111) {
    while (true) {
        auto mem_pool = arrow::default_memory_pool();

        std::unique_ptr<parquet::arrow::FileReader> reader;
        auto arrow_memory_mapped_file = arrow::io::MemoryMappedFile::Open("/tmp/a.parquet", arrow::io::FileMode::READ);
        std::shared_ptr<arrow::io::RandomAccessFile> input = arrow_memory_mapped_file.ValueOrDie();
        auto st = parquet::arrow::OpenFile(input, mem_pool, &reader);
        AssertInfo(st.ok(), "failed to get arrow file reader");
        std::shared_ptr<arrow::Table> table;
        std::cout << "read before " << getCurrentRSS() << std::endl;
        std::shared_ptr<arrow::ChunkedArray> array;
        st = reader->ReadColumn(0, &array);
        AssertInfo(array != nullptr, "empty arrow array of PayloadReader");
        auto array1 = array->chunk(0);
        // sleep(4);
        std::cout << "read after " << getCurrentRSS() << std::endl;
    }
}

class A {
 public:
    A() {
    }
    ~A() {
        sleep(10);
    }
};

void
TT() {
    auto a = std::make_unique<A>();
}

TEST_F(DiskAnnFileManagerTest, TESTAA) {
    TT();
    std::cout << "done" << std::endl;
}
