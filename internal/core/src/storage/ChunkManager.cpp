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

#include "ChunkManager.h"
#include "Exception.h"

#include <sstream>
#include <boost/filesystem.hpp>
#include <boost/system/error_code.hpp>

using namespace boost;

#define THROWLOCALERROR(FUNCTION)                                           \
    do {                                                                    \
        std::stringstream err_msg;                                          \
        err_msg << "Error:" << #FUNCTION << ":"                             \
                << err.message();                                           \
        throw LocalChunkManagerException(err_msg.str());                    \
    } while (0)

namespace milvus::storage {

bool LocalChunkManager::Exist(const std::string& filepath) {
    filesystem::path prefix(path_prefix_);
    auto absPath = prefix.append(filepath);
    system::error_code err;
    bool isExist = filesystem::exists(absPath, err);
    if (err && err.value() != system::errc::no_such_file_or_directory) {
        THROWLOCALERROR(Exist);
    }
    return isExist;
}

uint64_t LocalChunkManager::Size(const std::string& filepath) {
    filesystem::path prefix(path_prefix_);
    auto absPath = prefix.append(filepath);
 
    if (!Exist(filepath)) {
        throw InvalidPathException("invalid local path:" + absPath.generic_string());
    }
    system::error_code err;
    int64_t size = filesystem::file_size(absPath, err);
    if (err) {
        THROWLOCALERROR(FileSize);
    }
    return size;
}

void LocalChunkManager::Remove(const std::string& filepath) {
    filesystem::path prefix(path_prefix_);
    auto absPath = prefix.append(filepath);
    system::error_code err;
    filesystem::remove(absPath, err);
    if (err) {
        THROWLOCALERROR(Remove);
    }
}

uint64_t LocalChunkManager::Read(const std::string& filepath, void* buf, uint64_t size) {
    return Read(filepath, 0, buf, size);
}

uint64_t LocalChunkManager::Read(const std::string& filepath, uint64_t offset, void* buf, uint64_t size) {
    filesystem::path prefix(path_prefix_);
    auto absPathStr = prefix.append(filepath).generic_string();
    std::ifstream infile;
    infile.open(absPathStr, std::ios_base::binary);
    if (infile.fail()) {
        std::stringstream err_msg;
        err_msg << "Error: open local file '" << absPathStr << "failed, " << strerror(errno);
        throw OpenFileException(err_msg.str());;
    }

    infile.seekg(offset, std::ios::beg);
    if (!infile.read(reinterpret_cast<char*>(buf), size)) {
        if (!infile.eof()) {
            std::stringstream err_msg;
            err_msg << "Error: read local file '" << absPathStr << "failed, " << strerror(errno);
            throw ReadFileException(err_msg.str());
        }
    }
    return infile.gcount();
}

void LocalChunkManager::Write(const std::string& filepath, void* buf, uint64_t size) {
    filesystem::path prefix(path_prefix_);
    auto absPathStr = prefix.append(filepath).generic_string();
    // if filepath not exists, will create this file automatically
    std::ofstream outfile;
    outfile.open(absPathStr, std::ios_base::binary);
       if (outfile.fail()) {
        std::stringstream err_msg;
        err_msg << "Error: open local file '" << absPathStr << "failed, " << strerror(errno);
        throw OpenFileException(err_msg.str());;
    }
    if (!outfile.write(reinterpret_cast<char*>(buf), size)) {
        std::stringstream err_msg;
        err_msg << "Error: write local file '" << absPathStr << "failed, " << strerror(errno);
        throw WriteFileException(err_msg.str());
    }
}

void LocalChunkManager::Write(const std::string& filepath, uint64_t offset, void* buf, uint64_t size) {
    filesystem::path prefix(path_prefix_);
    auto absPathStr = prefix.append(filepath).generic_string();
    std::fstream outfile;
    outfile.open(absPathStr, std::ios_base::in | std::ios_base::out | std::ios_base::binary);
    if (outfile.fail()) {
        std::stringstream err_msg;
        err_msg << "Error: open local file '" << absPathStr << "failed, " << strerror(errno);
        throw OpenFileException(err_msg.str());;
    }

    outfile.seekp(offset, std::ios::beg);
    if (!outfile.write(reinterpret_cast<char*>(buf), size)) {
        std::stringstream err_msg;
        err_msg << "Error: write local file '" << absPathStr << "failed, " << strerror(errno);
        throw WriteFileException(err_msg.str());
    }
}

std::vector<std::string>
LocalChunkManager::ListWithPrefix(const std::string& filepath) {
	throw NotImplementedException(GetName() + "::ListWithPrefix" + " not implement now");
}

bool LocalChunkManager::CreateFile(const std::string& filepath) {
    std::ofstream outfile(filepath);
    if (outfile.fail()) {
        std::stringstream err_msg;
        err_msg << "Error: create new local file '" << filepath << "failed, " << strerror(errno);
        throw CreateFileException(err_msg.str());
    }
    return true;
}

bool LocalChunkManager::DirExist(const std::string& dir) {
    filesystem::path dirPath(dir);
    system::error_code err;
    bool isExist = filesystem::exists(dirPath, err);
    if (err) {
        THROWLOCALERROR(DirExist);
    }
    return isExist;
}

void LocalChunkManager::CreateDir(const std::string& dir) {
    bool isExist = DirExist(dir);
    if (isExist) {
        throw PathAlreadyExistException("dir:" + dir + "alreay exists");
    }
    filesystem::path dirPath(dir);
    system::error_code err;
    filesystem::create_directory(dirPath, err);
    if (err) {
        THROWLOCALERROR(CreateDir);
    }
}

void LocalChunkManager::RemoveDir(const std::string& dir) {
    filesystem::path dirPath(dir);
    system::error_code err;
    filesystem::remove_all(dirPath);
    if (err) {
        THROWLOCALERROR(RemoveDir);
    }
}


} // namespace storage
