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

#include <string>

namespace milvus::config {
class ChunkMangerConfig {
 public:
    static void
    SetAddress(const std::string& address);

    static std::string
    GetAddress();

    static void
    SetAccessKey(const std::string& access_key);

    static std::string
    GetAccessKey();

    static void
    SetAccessValue(const std::string& access_value);

    static std::string
    GetAccessValue();

    static void
    SetUseSSL(bool use_ssl);

    static bool
    GetUseSSL();

    static void
    SetBucketName(const std::string& bucket_name);

    static std::string
    GetBucketName();

    static void
    SetLocalBucketName(const std::string& path_prefix);

    static std::string
    GetLocalBucketName();

    static bool
    GetUseIAM();

    static void
    SetUseIAM(bool);

    static std::string
    GetDefaultSTSEndpoint();
};

}  // namespace milvus::config