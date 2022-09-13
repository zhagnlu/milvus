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

#include "config/ConfigChunkManager.h"

namespace milvus::config {

std::string MINIO_ADDRESS;
std::string MINIO_ACCESS_KEY;
std::string MINIO_ACCESS_VALUE;
std::string MINIO_BUCKET_NAME;
std::string DEFAULT_STS_ROLE_ENDPOINT = "https://sts.amazonaws.com";
bool MINIO_USE_SSL = false;
std::string LOCAL_BUCKET_NAME = "/tmp";
bool MINIO_USE_IAM = false;

void
ChunkMangerConfig::SetAddress(const std::string& address) {
    MINIO_ADDRESS = address;
}

std::string
ChunkMangerConfig::GetAddress() {
    return MINIO_ADDRESS;
}

void
ChunkMangerConfig::SetAccessKey(const std::string& access_key) {
    MINIO_ACCESS_KEY = access_key;
}

std::string
ChunkMangerConfig::GetAccessKey() {
    return MINIO_ACCESS_KEY;
}

void
ChunkMangerConfig::SetAccessValue(const std::string& access_value) {
    MINIO_ACCESS_VALUE = access_value;
}

std::string
ChunkMangerConfig::GetAccessValue() {
    return MINIO_ACCESS_VALUE;
}

void
ChunkMangerConfig::SetBucketName(const std::string& bucket_name) {
    MINIO_BUCKET_NAME = bucket_name;
}

std::string
ChunkMangerConfig::GetBucketName() {
    return MINIO_BUCKET_NAME;
}

void
ChunkMangerConfig::SetUseSSL(bool use_ssl) {
    MINIO_USE_SSL = use_ssl;
}

bool
ChunkMangerConfig::GetUseSSL() {
    return MINIO_USE_SSL;
}

void
ChunkMangerConfig::SetUseIAM(bool use_iam) {
    MINIO_USE_IAM = use_iam;
}

bool
ChunkMangerConfig::GetUseIAM() {
    return MINIO_USE_IAM;
}

std::string
ChunkMangerConfig::GetDefaultSTSEndpoint() {
    return DEFAULT_STS_ROLE_ENDPOINT;
}

void
ChunkMangerConfig::SetLocalBucketName(const std::string& path_prefix) {
    LOCAL_BUCKET_NAME = path_prefix;
}

std::string
ChunkMangerConfig::GetLocalBucketName() {
    return LOCAL_BUCKET_NAME;
}

}  // namespace milvus::config
