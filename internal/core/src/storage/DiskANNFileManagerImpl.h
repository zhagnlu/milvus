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
#include <shared_mutex>
#include <string>
#include <vector>

#include "FileManager.h"
#include "MinioChunkManager.h"
#include "LocalChunkManager.h"
#include "storage/IndexData.h"

using milvus::storage::LocalChunkManagerSPtr;
using milvus::storage::RemoteChunkManagerSPtr;

namespace knowhere {

class DiskANNFileManagerImpl : public FileManager {
 public:
    explicit DiskANNFileManagerImpl(int64_t collectionId,
                                    int64_t partiitionId,
                                    int64_t segmentId,
                                    LocalChunkManagerSPtr localChunkManager,
                                    RemoteChunkManagerSPtr remoteChunkManager);

    virtual ~DiskANNFileManagerImpl();

    void
    Init();

    virtual bool
    LoadFile(const std::string& filename) noexcept;

    virtual bool
    AddFile(const std::string& filename) noexcept;

    virtual std::optional<bool>
    IsExisted(const std::string& filename) noexcept;

    virtual bool
    RemoveFile(const std::string& filename) noexcept;

 public:
    virtual std::string
    GetName() const {
        return "DiskANNFileManagerImpl";
    }

    LocalChunkManagerSPtr
    GetLocalChunkManager() {
        return local_chunk_manager_;
    }

    RemoteChunkManagerSPtr
    GetRemoteChunkManager() {
        return remote_chunk_manager_;
    }

    std::string
    GetRemoteObjectPrefix(const milvus::storage::IndexMeta& indexMeta);

    void
    SetIndexSliceSize(int64_t size) {
        index_file_slice_size_ = size;
    }

    void
    SetIndexMeta(int64_t buildId, const milvus::storage::IndexMeta& indexMeta) {
        std::unique_lock lock(index_meta_mutex_);
        index_meta_map_[buildId] = indexMeta;
    }

    milvus::storage::IndexMeta
    GetIndexMeta(int64_t buildId, bool& found) {
        std::shared_lock lock(index_meta_mutex_);
        if (index_meta_map_.find(buildId) != index_meta_map_.end()) {
            found = true;
            return index_meta_map_[buildId];
        }
        found = false;
        return {};
    }

 private:
    int64_t
    GetIndexBuildId(const std::string& localfile);

    std::string
    GetFileName(const std::string& localfile);

 private:
    // collection meta
    int64_t collection_id_;
    int64_t partition_id_;
    int64_t segment_id_;
    int64_t index_file_slice_size_;
    int64_t index_version_;

    // index meta
    mutable std::shared_mutex index_meta_mutex_;
    // record (index_build_id, index_meta) map
    std::map<int64_t, milvus::storage::IndexMeta> index_meta_map_;

    // this map record (remote_object, local_file) pair
    std::map<std::string, std::string> file_map_;
    LocalChunkManagerSPtr local_chunk_manager_;
    RemoteChunkManagerSPtr remote_chunk_manager_;
};

}  // namespace knowhere
