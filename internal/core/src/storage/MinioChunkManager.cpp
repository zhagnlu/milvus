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

#include <fstream>
#include <aws/core/auth/AWSCredentials.h>
#include <aws/core/internal/AWSHttpResourceClient.h>
#include <aws/s3/model/CreateBucketRequest.h>
#include <aws/s3/model/DeleteBucketRequest.h>
#include <aws/s3/model/DeleteObjectRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/s3/model/ListObjectsRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/sts/model/AssumeRoleWithWebIdentityRequest.h>
#include <aws/sts/STSClient.h>

#include "MinioChunkManager.h"

#include "common/Utils.h"
#include "log/Log.h"

#define THROWS3ERROR(FUNCTION)                                                                         \
    do {                                                                                               \
        auto& err = outcome.GetError();                                                                \
        std::stringstream err_msg;                                                                     \
        err_msg << "Error:" << #FUNCTION << ":" << err.GetExceptionName() << "  " << err.GetMessage(); \
        throw S3ErrorException(err_msg.str());                                                         \
    } while (0)

#define S3NoSuchBucket "NoSuchBucket"
namespace milvus::storage {

/**
 * @brief convert std::string to Aws::String
 * because Aws has String type internally
 * but has a copy of string content unfortunately
 * TODO: remove this convert
 * @param str
 * @return Aws::String
 */
inline Aws::String
ConvertToAwsString(const std::string& str) {
    return Aws::String(str.c_str(), str.size());
}

/**
 * @brief convert Aws::string to std::string
 * @param aws_str
 * @return std::string
 */
inline std::string
ConvertFromAwsString(const Aws::String& aws_str) {
    return std::string(aws_str.c_str(), aws_str.size());
}

bool
HasPrefix(const std::string str, const std::string prefix) {
    return str.length() >= prefix.length() && str.substr(0, prefix.length()) == prefix;
}

Aws::Auth::AWSCredentials
MinioChunkManager::GetIAMCred() {
    Aws::Auth::AWSCredentials cred;

    std::string endpoint;
    const char* awsRegion = std::getenv("AWS_REGION");
    if (awsRegion == NULL) {
        endpoint = config::ChunkMangerConfig::GetDefaultSTSEndpoint();
    } else if (HasPrefix(awsRegion, "cn-")) {
        endpoint = std::string("https://sts.") + awsRegion + ".amazonaws.com.cn";
    } else {
        endpoint = std::string("https://sts.") + awsRegion + ".amazonaws.com";
    }

    std::string webIdenToken;
    const char* tokenFilePath = std::getenv("AWS_WEB_IDENTITY_TOKEN_FILE");
    if (tokenFilePath == NULL) {
        std::stringstream err_msg;
        err_msg << "Error: GetIAMCred: AWS_WEB_IDENTITY_TOKEN_FILE not found in env";
        throw ConfigException(err_msg.str());
    }
    webIdenToken = read_string_from_file(tokenFilePath);

    const char* roleARN = std::getenv("AWS_ROLE_ARN");
    if (roleARN == NULL) {
        std::stringstream err_msg;
        err_msg << "Error: GetIAMCred: AWS_ROLE_ARN not found in env";
        throw ConfigException(err_msg.str());
    }

    const char* sessionName = std::getenv("AWS_ROLE_SESSION_NAME");

    Aws::Client::ClientConfiguration clientConfig;
    clientConfig.endpointOverride = ConvertToAwsString(endpoint);
    // Aws::STS::STSClient sts(clientConfig);
    // Aws::STS::Model::AssumeRoleWithWebIdentityRequest request;
    // request.SetRoleArn(roleARN);
    // request.SetWebIdentityToken(webIdenToken.c_str());
    // if (sessionName != NULL) {
    // request.SetRoleSessionName(sessionName);
    //}
    // auto outcome = sts.AssumeRoleWithWebIdentity(request);
    // auto result = outcome.GetResult();
    // cred = result.GetCredentials();
    //
    auto client = Aws::Internal::STSCredentialsClient(clientConfig);
    Aws::Internal::STSCredentialsClient::STSAssumeRoleWithWebIdentityRequest request;
    request.roleArn = ConvertToAwsString(roleARN);
    request.webIdentityToken = ConvertToAwsString(webIdenToken);
    if (sessionName != NULL) {
        request.roleSessionName = ConvertToAwsString(sessionName);
    } else {
        request.roleSessionName = ConvertToAwsString("");
    }
    LOG_SEGCORE_INFO_C << "AssumeRoleWithWebIdentityRequest: {roleARN: " << request.roleArn
                       << ", webIdenToken: " << request.webIdentityToken << ", sessionName:" << request.roleSessionName
                       << "};";
    cred = client.GetAssumeRoleWithWebIdentityCredentials(request).creds;
    LOG_SEGCORE_INFO_C << "AWSCredentials result: { access_id:" << cred.GetAWSAccessKeyId()
                       << ", access_key:" << cred.GetAWSSecretKey() << ", token:" << cred.GetSessionToken() << "}";

    return cred;
}  // namespace milvus::storage

MinioChunkManager::MinioChunkManager(const std::string& endpoint,
                                     const std::string& access_key,
                                     const std::string& access_value,
                                     const std::string& bucket_name,
                                     bool secure,
                                     bool use_iam)
    : default_bucket_name_(bucket_name) {
    Aws::InitAPI(sdk_options_);
    Aws::Client::ClientConfiguration config;
    config.endpointOverride = ConvertToAwsString(endpoint);

    if (secure) {
        config.scheme = Aws::Http::Scheme::HTTPS;
        config.verifySSL = true;
    } else {
        config.scheme = Aws::Http::Scheme::HTTP;
        config.verifySSL = false;
    }

    Aws::Auth::AWSCredentials cred;
    if (use_iam) {
        cred = GetIAMCred();
    } else {
        cred = Aws::Auth::AWSCredentials(ConvertToAwsString(access_key), ConvertToAwsString(access_value));
    }
    client_ = std::make_shared<Aws::S3::S3Client>(cred, config,
                                                  Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never, false);

    LOG_SEGCORE_INFO_C << "init MinioChunkManager with parameter[endpoint: '" << endpoint << "', access_key:'"
                       << access_key << "', access_value:'" << access_value << "', default_bucket_name:'" << bucket_name
                       << "', use_secure:'" << std::boolalpha << secure << "']";
}

MinioChunkManager::~MinioChunkManager() {
    Aws::ShutdownAPI(sdk_options_);
    client_.reset();
}

uint64_t
MinioChunkManager::Size(const std::string& filepath) {
    return GetObjectSize(default_bucket_name_, filepath);
}

bool
MinioChunkManager::Exist(const std::string& filepath) {
    return ObjectExists(default_bucket_name_, filepath);
}

void
MinioChunkManager::Remove(const std::string& filepath) {
    DeleteObject(default_bucket_name_, filepath);
}

std::vector<std::string>
MinioChunkManager::ListWithPrefix(const std::string& filepath) {
    return ListObjects(default_bucket_name_.c_str(), filepath.c_str());
}

uint64_t
MinioChunkManager::Read(const std::string& filepath, void* buf, uint64_t size) {
    if (!ObjectExists(default_bucket_name_, filepath)) {
        std::stringstream err_msg;
        err_msg << "object('" << default_bucket_name_ << "', " << filepath << "') not exists";
        throw ObjectNotExistException(err_msg.str());
    }
    return GetObjectBuffer(default_bucket_name_, filepath, buf, size);
}

void
MinioChunkManager::Write(const std::string& filepath, void* buf, uint64_t size) {
    PutObjectBuffer(default_bucket_name_, filepath, buf, size);
}

bool
MinioChunkManager::BucketExists(const std::string& bucket_name) {
    auto outcome = client_->ListBuckets();

    if (!outcome.IsSuccess()) {
        THROWS3ERROR(BucketExists);
    }
    for (auto&& b : outcome.GetResult().GetBuckets()) {
        if (ConvertFromAwsString(b.GetName()) == bucket_name) {
            return true;
        }
    }
    return false;
}

std::vector<std::string>
MinioChunkManager::ListBuckets() {
    std::vector<std::string> buckets;
    auto outcome = client_->ListBuckets();

    if (!outcome.IsSuccess()) {
        THROWS3ERROR(CreateBucket);
    }
    for (auto&& b : outcome.GetResult().GetBuckets()) {
        buckets.emplace_back(b.GetName().c_str());
    }
    return buckets;
}

bool
MinioChunkManager::CreateBucket(const std::string& bucket_name) {
    Aws::S3::Model::CreateBucketRequest request;
    request.SetBucket(bucket_name.c_str());

    auto outcome = client_->CreateBucket(request);

    if (!outcome.IsSuccess()) {
        THROWS3ERROR(CreateBucket);
    }
    return true;
}

bool
MinioChunkManager::DeleteBucket(const std::string& bucket_name) {
    Aws::S3::Model::DeleteBucketRequest request;
    request.SetBucket(bucket_name.c_str());

    auto outcome = client_->DeleteBucket(request);

    if (!outcome.IsSuccess()) {
        auto err = outcome.GetError();
        if (err.GetExceptionName() != S3NoSuchBucket) {
            THROWS3ERROR(DeleteBucket);
        }
        return false;
    }
    return true;
}

bool
MinioChunkManager::ObjectExists(const std::string& bucket_name, const std::string& object_name) {
    Aws::S3::Model::HeadObjectRequest request;
    request.SetBucket(bucket_name.c_str());
    request.SetKey(object_name.c_str());

    auto outcome = client_->HeadObject(request);

    if (!outcome.IsSuccess()) {
        auto& err = outcome.GetError();
        if (!err.GetExceptionName().empty()) {
            std::stringstream err_msg;
            err_msg << "Error: ObjectExists: " << err.GetMessage();
            throw S3ErrorException(err_msg.str());
        }
        return false;
    }
    return true;
}

int64_t
MinioChunkManager::GetObjectSize(const std::string& bucket_name, const std::string& object_name) {
    Aws::S3::Model::HeadObjectRequest request;
    request.SetBucket(bucket_name.c_str());
    request.SetKey(object_name.c_str());

    auto outcome = client_->HeadObject(request);
    if (!outcome.IsSuccess()) {
        THROWS3ERROR(GetObjectSize);
    }
    return outcome.GetResult().GetContentLength();
}

bool
MinioChunkManager::DeleteObject(const std::string& bucket_name, const std::string& object_name) {
    Aws::S3::Model::DeleteObjectRequest request;
    request.SetBucket(bucket_name.c_str());
    request.SetKey(object_name.c_str());

    auto outcome = client_->DeleteObject(request);

    if (!outcome.IsSuccess()) {
        // auto err = outcome.GetError();
        // std::stringstream err_msg;
        // err_msg << "Error: DeleteObject:" << err.GetMessage();
        // throw S3ErrorException(err_msg.str());
        THROWS3ERROR(DeleteObject);
    }
    return true;
}

bool
MinioChunkManager::PutObjectBuffer(const std::string& bucket_name,
                                   const std::string& object_name,
                                   void* buf,
                                   uint64_t size) {
    Aws::S3::Model::PutObjectRequest request;
    request.SetBucket(bucket_name.c_str());
    request.SetKey(object_name.c_str());

    const std::shared_ptr<Aws::IOStream> input_data = Aws::MakeShared<Aws::StringStream>("");

    input_data->write(reinterpret_cast<char*>(buf), size);
    request.SetBody(input_data);

    auto outcome = client_->PutObject(request);

    if (!outcome.IsSuccess()) {
        THROWS3ERROR(PutObjectBuffer);
    }
    return true;
}

uint64_t
MinioChunkManager::GetObjectBuffer(const std::string& bucket_name,
                                   const std::string& object_name,
                                   void* buf,
                                   uint64_t size) {
    Aws::S3::Model::GetObjectRequest request;
    request.SetBucket(bucket_name.c_str());
    request.SetKey(object_name.c_str());

    auto outcome = client_->GetObject(request);

    if (!outcome.IsSuccess()) {
        THROWS3ERROR(GetObjectBuffer);
    }
    std::stringstream ss;
    ss << outcome.GetResultWithOwnership().GetBody().rdbuf();
    uint64_t realSize = size;
    if (ss.str().size() <= size) {
        memcpy(buf, ss.str().data(), ss.str().size());
        realSize = ss.str().size();
    } else {
        memcpy(buf, ss.str().data(), size);
    }
    return realSize;
}

std::vector<std::string>
MinioChunkManager::ListObjects(const char* bucket_name, const char* prefix) {
    std::vector<std::string> objects_vec;
    Aws::S3::Model::ListObjectsRequest request;
    request.WithBucket(bucket_name);
    if (prefix != NULL) {
        request.SetPrefix(prefix);
    }

    auto outcome = client_->ListObjects(request);

    if (!outcome.IsSuccess()) {
        THROWS3ERROR(ListObjects);
    }
    auto objects = outcome.GetResult().GetContents();
    for (auto& obj : objects) {
        objects_vec.emplace_back(obj.GetKey().c_str());
    }
    return objects_vec;
}

}  // namespace milvus::storage
