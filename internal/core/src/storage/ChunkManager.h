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
#include <vector>

namespace milvus::storage {

/**
 * @brief This ChunkManager is abstract interface for milvus that
 * used to manager operation and interaction with storage
 */
class ChunkManager {
  public:
   /**
    * @brief Whether file exists or not 
    * @param filepath 
    * @return true 
    * @return false 
    */
    virtual bool
    Exist(const std::string& filepath) = 0;

   /**
    * @brief Get file size
    * @param filepath 
    * @return uint64_t 
    */
    virtual uint64_t
    Size(const std::string& filepath) = 0;

   /**
    * @brief Read file to buffer
    * @param filepath 
    * @param buf 
    * @param len 
    * @return uint64_t 
    */
    virtual uint64_t
    Read(const std::string& filepath, void* buf, uint64_t len) = 0;

   /**
    * @brief Write buffer to file with offset
    * @param filepath 
    * @param buf 
    * @param len 
    */
    virtual void
    Write(const std::string& filepath, void* buf, uint64_t len) = 0;

   /**
    * @brief Read file to buffer with offset
    * @param filepath 
    * @param buf 
    * @param len 
    * @return uint64_t 
    */
    virtual uint64_t
    Read(const std::string& filepath, uint64_t offset, void* buf, uint64_t len) = 0;

   /**
    * @brief Write buffer to file with offset
    * @param filepath 
    * @param buf 
    * @param len 
    */
    virtual void
    Write(const std::string& filepath, uint64_t offset, void* buf, uint64_t len) = 0;

   /**
    * @brief List files with same prefix
    * @param filepath 
    * @return std::vector<std::string> 
    */
    virtual std::vector<std::string>
    ListWithPrefix(const std::string& filepath) = 0;

   /**
    * @brief Remove specified file
    * @param filepath 
    */
    virtual void
    Remove(const std::string& filepath) = 0;

   /**
    * @brief Get the Name object
    * Used for forming diagnosis messages
    * @return std::string 
    */
    virtual std::string
    GetName() const = 0;
};

/**
 * @brief LocalChunkManager is responsible for read and write local file
 * that inherited from ChunkManager
 */
class LocalChunkManager : public ChunkManager {
 public:
    explicit LocalChunkManager(const std::string& path) : path_prefix_(path) {}

    virtual ~LocalChunkManager() {};

    virtual bool
    Exist(const std::string& filepath);

   /**
    * @brief Get file's size
    * if file not exist, throw exception
    * @param filepath 
    * @return uint64_t 
    */
    virtual uint64_t
    Size(const std::string& filepath);

    virtual uint64_t
    Read(const std::string& filepath, void* buf, uint64_t len);

    virtual void
    Write(const std::string& filepath, void* buf, uint64_t len);

    virtual uint64_t
    Read(const std::string& filepath, uint64_t offset, void* buf, uint64_t len);

    virtual void
    Write(const std::string& filepath, uint64_t offset, void* buf, uint64_t len);

    virtual std::vector<std::string>
    ListWithPrefix(const std::string& filepath);

   /**
    * @brief Remove file no matter whether file exists
    *  or not
    * @param filepath 
    */
    virtual void
    Remove(const std::string& filepath);

    virtual std::string GetName() const { return "LocalChunkManager"; }

 public:
    static bool
    CreateFile(const std::string& filepath);

    static bool
    DirExist(const std::string& dir);
   /**
    * @brief Delete directory totally
    * different from Remove, this interface drop local dir
    * instead of file, but for remote system, may has no
    * concept of directory, so just used in local chunk manager 
    * @param dir 
    */
    static void
    RemoveDir(const std::string& dir);

   /**
    * @brief Create a Dir object
    * if dir already exists, throw exception
    * @param dir 
    */
    static void
    CreateDir(const std::string& dir);

   

 private:
    std::string path_prefix_;
};

/**
 * @brief RemoteChunkManager is responsible for read and write Remote file
 * that inherited from ChunkManager. 
 */

class RemoteChunkManager : public ChunkManager {
 public:
    // some general interface for convert between remote data 
    // and local disk data
    //virtual InsertData
    //DeserializeRemoteInsertData(void* buf, uint64_t size);
    //virtual IndexData
    //DeserializeRemoteIndexData(void* buf, uint64_t size);
    //virtual void
    //SerializeInsertDataToLocalDisk(void* buf, 
    //                               uint64_t size, 
    //                               const std::string& toFile);
    //virtual void
    //SerializeIndexDataToLocalDisk(void* buf,
    //                              uint64_t size,
    //                              const std::string& toFile);
    virtual ~RemoteChunkManager() {}
    virtual std::string GetName() const { return "RemoteChunkManager"; }
};

using LocalChunkManagerSPtr = std::shared_ptr<milvus::storage::LocalChunkManager>;
using RemoteChunkManagerSPtr = std::shared_ptr<milvus::storage::RemoteChunkManager>;

}  // namespace storage
