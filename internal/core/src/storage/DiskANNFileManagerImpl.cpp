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

#include <algorithm>
#include <boost/filesystem.hpp>
#include <mutex>

#include "common/Consts.h"
#include "DiskANNFileManagerImpl.h"
#include "Exception.h"
#include "log/Log.h"
#include "storage/DataCodec.h"
#include "storage/FieldData.h"
#include "storage/IndexData.h"

#define FILEMANAGER_TRY try {
#define FILEMANAGER_CATCH                                                                  \
    }                                                                                      \
    catch (milvus::storage::LocalChunkManagerException & e) {                              \
        LOG_SEGCORE_INFO_C << "LocalChunkManagerException:" << e.what();                   \
        return false;                                                                      \
    }                                                                                      \
    catch (milvus::storage::MinioException & e) {                                          \
        LOG_SEGCORE_INFO_C << "milvus::storage::MinioException:" << e.what();              \
        return false;                                                                      \
    }                                                                                      \
    catch (milvus::storage::DiskANNFileManagerException & e) {                             \
        LOG_SEGCORE_INFO_C << "milvus::storage::DiskANNFileManagerException:" << e.what(); \
        return false;                                                                      \
    }                                                                                      \
    catch (milvus::storage::ArrowException & e) {                                          \
        LOG_SEGCORE_INFO_C << "milvus::storage::ArrowException:" << e.what();              \
        return false;                                                                      \
    }                                                                                      \
    catch (std::exception & e) {                                                           \
        LOG_SEGCORE_INFO_C << "Exception:" << e.what();                                    \
        return false;
#define FILEMANAGER_END }

using ReadLock = std::shared_lock<std::shared_mutex>;
using WriteLock = std::lock_guard<std::shared_mutex>;

namespace knowhere {

DiskANNFileManagerImpl::DiskANNFileManagerImpl(int64_t collectioinId,
                                               int64_t partitionId,
                                               int64_t segmentId,
                                               LocalChunkManagerSPtr localChunkManager,
                                               RemoteChunkManagerSPtr remoteChunkManager)
    : FileManager(FileManagerType::DiskANNFileManager),
      collection_id_(collectioinId),
      partition_id_(partitionId),
      segment_id_(segmentId),
      local_chunk_manager_(localChunkManager),
      remote_chunk_manager_(remoteChunkManager) {
    Init();
}

void
DiskANNFileManagerImpl::Init() {
    // Load config param
    // index_file_slice_size_ = ReadConfig();
    index_version_ = 1;
}

DiskANNFileManagerImpl::~DiskANNFileManagerImpl() {
    local_chunk_manager_.reset();
    remote_chunk_manager_.reset();
    file_map_.clear();
}

bool
DiskANNFileManagerImpl::LoadFile(const std::string& file) noexcept {
    return false;
}

bool
DiskANNFileManagerImpl::AddFile(const std::string& file) noexcept {
    std::unique_ptr<uint8_t> buf;
    FILEMANAGER_TRY
    if (!local_chunk_manager_->Exist(file)) {
        LOG_SEGCORE_INFO_C << "local file: " << file << " does not exist ";
        return false;
    }

    // locate index meta from local file path
    auto indexBuildId = GetIndexBuildId(file);
    auto fileName = GetFileName(file);
    bool found = false;
    auto indexMeta = GetIndexMeta(indexBuildId, found);
    if (!found) {
        std::stringstream err_msg;
        err_msg << "index meta not found for build_id:" << indexBuildId;
        throw milvus::storage::DiskANNFileManagerException(err_msg.str());
    }
    milvus::storage::FieldDataMeta fieldMeta = {collection_id_, partition_id_, indexMeta.segment_id,
                                                indexMeta.field_id};

    auto fileSize = local_chunk_manager_->Size(file);
    auto buf = std::unique_ptr<uint8_t[]>(new uint8_t[fileSize]);
    auto ret_size = local_chunk_manager_->Read(file, buf.get(), fileSize);

    // Decode knowhere's index data to buf
    auto localData = milvus::storage::DeserializeLocalIndexFileData(buf.get(), fileSize);
    auto localPayload = localData->GetPayload();
    assert(localPayload->data_type == milvus::DataType::INT8);
    int64_t localPayloadSize = localPayload->rows;

    // Split local data to multi part with specified size
    int slice_num = 0;
    auto remotePrefix = GetRemoteObjectPrefix(indexMeta);
    auto rawData = reinterpret_cast<const int8_t*>(localPayload->raw_data);
    for (int offset = 0; offset < localPayloadSize; slice_num++) {
        auto batch_size = std::min(index_file_slice_size_, localPayloadSize - offset);
        auto builder = std::make_shared<arrow::Int8Builder>();
        auto status = builder->AppendValues(rawData + offset, batch_size);
        if (!status.ok()) {
            std::stringstream err_msg;
            err_msg << "data append failed";
            throw milvus::storage::ArrowException(err_msg.str());
        }
        std::shared_ptr<arrow::Array> array;
        status = builder->Finish(&array);
        if (!status.ok()) {
            std::stringstream err_msg;
            err_msg << "data build failed";
            throw milvus::storage::ArrowException(err_msg.str());
        }
        auto fieldData = std::make_shared<milvus::storage::FieldData>(array, milvus::storage::DataType::INT8);
        auto indexData = std::make_shared<milvus::storage::IndexData>(fieldData);
        indexData->set_index_meta(indexMeta);
        indexData->SetFieldDataMeta(fieldMeta);
        auto subFile = indexData->serialize_to_remote_file();

        // Put file to remote
        char objectKey[200];
        sprintf(objectKey, "%s/%s_%d", remotePrefix.c_str(), fileName.c_str(), slice_num);
        remote_chunk_manager_->Write(objectKey, subFile.data(), subFile.size());
        offset += batch_size;
    }
    FILEMANAGER_CATCH
    FILEMANAGER_END

    return true;
}  // namespace knowhere

int64_t
DiskANNFileManagerImpl::GetIndexBuildId(const std::string& localfile) {
    // Parse localfile path to extract index build id to locate remote path
    // localfile pattern /tmp/...../indexbuildid/indexfile
    boost::filesystem::path localPath(localfile);
    auto parentPath = localPath.parent_path().filename();
    auto indexBuildId = std::stoll(parentPath.string());
    return indexBuildId;
}

std::string
DiskANNFileManagerImpl::GetFileName(const std::string& localfile) {
    boost::filesystem::path localPath(localfile);
    return localPath.filename().string();
}

std::string
DiskANNFileManagerImpl::GetRemoteObjectPrefix(const milvus::storage::IndexMeta& indexMeta) {
    char objPrefix[100];
    sprintf(objPrefix, "%s/%ld/%ld/%ld/%ld", INDEX_ROOT_PATH, indexMeta.build_id, index_version_, partition_id_,
            indexMeta.segment_id);
    return objPrefix;
}

bool
DiskANNFileManagerImpl::RemoveFile(const std::string& file) noexcept {
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
    } catch (milvus::storage::LocalChunkManagerException& e) {
        // LOG_SEGCORE_DEBUG_ << "LocalChunkManagerException:"
        //                   << e.what();
        return std::nullopt;
    } catch (std::exception& e) {
        // LOG_SEGCORE_DEBUG_ << "Exception:" << e.what();
        return std::nullopt;
    }
    return isExist;
}

}  // namespace knowhere
