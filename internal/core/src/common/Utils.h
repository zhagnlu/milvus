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

#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include "exceptions/EasyAssert.h"
#include "config/ConfigChunkManager.h"
#include "common/Consts.h"
#include <google/protobuf/text_format.h>
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"

namespace milvus {

inline DatasetPtr
GenDataset(const int64_t nb, const int64_t dim, const void* xb) {
    return knowhere::GenDataset(nb, dim, xb);
}

inline const float*
GetDatasetDistance(const DatasetPtr& dataset) {
    return knowhere::GetDatasetDistance(dataset);
}

inline const int64_t*
GetDatasetIDs(const DatasetPtr& dataset) {
    return knowhere::GetDatasetIDs(dataset);
}

inline int64_t
GetDatasetRows(const DatasetPtr& dataset) {
    return knowhere::GetDatasetRows(dataset);
}

inline const void*
GetDatasetTensor(const DatasetPtr& dataset) {
    return knowhere::GetDatasetTensor(dataset);
}

inline int64_t
GetDatasetDim(const DatasetPtr& dataset) {
    return knowhere::GetDatasetDim(dataset);
}

inline bool
PrefixMatch(const std::string& str, const std::string& prefix) {
    auto ret = strncmp(str.c_str(), prefix.c_str(), prefix.length());
    if (ret != 0) {
        return false;
    }

    return true;
}

inline bool
PostfixMatch(const std::string& str, const std::string& postfix) {
    if (postfix.length() > str.length()) {
        return false;
    }

    int offset = str.length() - postfix.length();
    auto ret = strncmp(str.c_str() + offset, postfix.c_str(), postfix.length());
    if (ret != 0) {
        return false;
    }
    //
    //    int i = postfix.length() - 1;
    //    int j = str.length() - 1;
    //    for (; i >= 0; i--, j--) {
    //        if (postfix[i] != str[j]) {
    //            return false;
    //        }
    //    }
    return true;
}

inline int64_t
upper_align(int64_t value, int64_t align) {
    Assert(align > 0);
    auto groups = (value + align - 1) / align;
    return groups * align;
}

inline int64_t
upper_div(int64_t value, int64_t align) {
    Assert(align > 0);
    auto groups = (value + align - 1) / align;
    return groups;
}

inline std::string
read_string_from_file(const std::string& file_path) {
    const std::ifstream file(file_path, std::ios_base::binary);
    if (file.fail()) {
        throw std::runtime_error("failed to open file");
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

}  // namespace milvus
