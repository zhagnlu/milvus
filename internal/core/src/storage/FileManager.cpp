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

#include <mutex>
#include "FileManager.h"

#include "Exception.h"
//#include "log/Log.h"

#define FILEMANAGER_TRY    try{
#define FILEMANAGER_CATCH                                       \
    } catch (LocalChunkManagerException& e) {                   \
        std::cout << "LocalChunkManagerException:"     \
                            << e.what();                        \
        return false;                                           \
    } catch (MinioException& e) {                               \
        std::cout << "MinioException:" << e.what();    \
        return false;                                           \
    } catch (std::exception& e) {                               \
        std::cout  << "Exception:" << e.what();         \
        return false;
#define FILEMANAGER_END    }

using ReadLock = std::shared_lock<std::shared_mutex>;
using WriteLock = std::lock_guard<std::shared_mutex>;

using namespace milvus::storage;
namespace knowhere {

DiskANNFileManagerImpl::DiskANNFileManagerImpl(int64_t collectioinId, int64_t partitionId, int64_t segmentId,
                                               LocalChunkManagerSPtr localChunkManager, RemoteChunkManagerSPtr remoteChunkManager)
                                               : collection_id_(collectioinId), partition_id_(partitionId), segment_id_(segmentId),
                                               local_chunk_manager_(localChunkManager), remote_chunk_manager_(remoteChunkManager) {
}

DiskANNFileManagerImpl::~DiskANNFileManagerImpl() {
    local_chunk_manager_.reset();
    remote_chunk_manager_.reset();
    file_map_.clear();
}

bool DiskANNFileManagerImpl::LoadFile(const std::string& file) noexcept {
    return false;
}

bool DiskANNFileManagerImpl::AddFile(const std::string& file) noexcept {
    std::unique_ptr<uint8_t> buf;
    FILEMANAGER_TRY
    if (!local_chunk_manager_->Exist(file)) {
        return false;
    }
    auto fileSize = local_chunk_manager_->Size(file);
    auto buf = std::unique_ptr<uint8_t[]>(new uint8_t(fileSize));
    local_chunk_manager_->Read(file, buf.get(), fileSize);

    // // decode knowhere's index data to buf
    // auto originData = DeserializeFileData(buf.get(), fileSize);

    // // encode buf to remote file format
    // auto fileData = originData->serialize();

    // // put file to remote 
    // auto filepath = GetRemoteObjectName(file);
    // remote_chunk_manager_->Write(filepath, fileData.data(), fileData.size());
    FILEMANAGER_CATCH
    FILEMANAGER_END


    return false;
}

std::string DiskANNFileManagerImpl::GetRemoteObjectName(const std::string& localfile) {
    return std::to_string(collection_id_) + "/" + std::to_string(segment_id_);
}

bool DiskANNFileManagerImpl::RemoveFile(const std::string& file) noexcept {
    // remove local file 
    bool localExist = false;
    FILEMANAGER_TRY
    localExist = local_chunk_manager_->Exist(file);
    FILEMANAGER_CATCH
    FILEMANAGER_END
    if (!localExist) {
        FILEMANAGER_TRY
        local_chunk_manager_->Remove(file);
        FILEMANAGER_CATCH
        FILEMANAGER_END
    }

    // remove according remote file
    std::string remoteFile = "";
    bool remoteExist = false;
    FILEMANAGER_TRY
    remoteExist = remote_chunk_manager_->Exist(remoteFile);
    FILEMANAGER_CATCH
    FILEMANAGER_END
    if (!remoteExist) {
        FILEMANAGER_TRY
        remote_chunk_manager_->Remove(file);
        FILEMANAGER_CATCH
        FILEMANAGER_END
    }
    return true;
}

std::optional<bool>
DiskANNFileManagerImpl::IsExisted(const std::string& file) noexcept {
    bool isExist = false;
    try {
        isExist = local_chunk_manager_->Exist(file);
    } catch (LocalChunkManagerException& e) {
        //LOG_SEGCORE_DEBUG_ << "LocalChunkManagerException:"
        //                   << e.what();
        return std::nullopt;
    } catch (std::exception& e) {
        //LOG_SEGCORE_DEBUG_ << "Exception:" << e.what();
        return std::nullopt;
    }
    return isExist;
}

} // namespace knowhere
