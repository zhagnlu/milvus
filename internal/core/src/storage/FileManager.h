// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <vector>

#include "ChunkManager.h"
#include "MinioChunkManager.h"

using namespace milvus::storage;
namespace knowhere {

/**
 * @brief This FileManager is used to manage file, including its replication, backup, ect.
 * It will act as a cloud-like client, and Knowhere need to call load/add to better support
 * distribution of the whole service.
 *
 * (TODO) we need support finer granularity file operator (read/write),
 * so Knowhere doesn't need to offer any help for service in the future .
 */
class FileManager {
    /**
     * @brief Load a file to the local disk, so we can use stl lib to operate it.
     *
     * @param filename
     * @return false if any error, or return true.
     */
    virtual bool
    LoadFile(const std::string& filename) noexcept = 0;

    /**
     * @brief Add file to FileManager to manipulate it.
     *
     * @param filename
     * @return false if any error, or return true.
     */
    virtual bool
    AddFile(const std::string& filename) noexcept = 0;

    /**
     * @brief Check if a file exists.
     *
     * @param filename
     * @return std::nullopt if any error, or return if the file exists.
     */
    virtual std::optional<bool>
    IsExisted(const std::string& filename) noexcept = 0;

    /**
     * @brief Delete a file from FileManager.
     *
     * @param filename
     * @return false if any error, or return true.
     */
    virtual bool
    RemoveFile(const std::string& filename) noexcept = 0;
};

class DiskANNFileManagerImpl : public FileManager {
  public:
    explicit DiskANNFileManagerImpl(int64_t collectionId,
                                    int64_t partiitionId,
                                    int64_t segmentId,
                                    LocalChunkManagerSPtr localChunkManager,
                                    RemoteChunkManagerSPtr remoteChunkManager);

    virtual ~DiskANNFileManagerImpl();

    virtual bool
    LoadFile(const std::string& filename) noexcept;

    virtual bool
    AddFile(const std::string& filename) noexcept;

    virtual std::optional<bool>
    IsExisted(const std::string& filename) noexcept;

    virtual bool
    RemoveFile(const std::string& filename) noexcept;

 public:
    virtual std::string GetName() const { return "DiskANNFileManagerImpl"; }
    std::string GetRemoteObjectName(const std::string& localfile);

 private:
    int64_t collection_id_;
    int64_t partition_id_;
    int64_t segment_id_;
    mutable std::shared_mutex mutex_;
    // this map record (remote_object, local_file) pair
    std::map<std::string, std::string> file_map_;
    LocalChunkManagerSPtr local_chunk_manager_;
    RemoteChunkManagerSPtr remote_chunk_manager_;
};

}  // namespace knowwhere
