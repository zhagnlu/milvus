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

#include <iostream>

namespace milvus {

class ExecDriverException : public std::exception {
 public:
    explicit ExecDriverException(const std::string& msg)
        : std::exception(), exception_message_(msg) {
    }
    const char*
    what() const noexcept {
        return exception_message_.c_str();
    }
    virtual ~ExecDriverException() {
    }

 private:
    std::string exception_message_;
};
class ExecOperatorException : public std::exception {
 public:
    explicit ExecOperatorException(const std::string& msg)
        : std::exception(), exception_message_(msg) {
    }
    const char*
    what() const noexcept {
        return exception_message_.c_str();
    }
    virtual ~ExecOperatorException() {
    }

 private:
    std::string exception_message_;
};

class NotImplementedException : public std::exception {
 public:
    explicit NotImplementedException(const std::string& msg)
        : std::exception(), exception_message_(msg) {
    }
    const char*
    what() const noexcept {
        return exception_message_.c_str();
    }
    virtual ~NotImplementedException() {
    }

 private:
    std::string exception_message_;
};

class NotSupportedDataTypeException : public std::exception {
 public:
    explicit NotSupportedDataTypeException(const std::string& msg)
        : std::exception(), exception_message_(msg) {
    }
    const char*
    what() const noexcept {
        return exception_message_.c_str();
    }
    virtual ~NotSupportedDataTypeException() {
    }

 private:
    std::string exception_message_;
};

class UnistdException : public std::runtime_error {
 public:
    explicit UnistdException(const std::string& msg) : std::runtime_error(msg) {
    }

    virtual ~UnistdException() {
    }
};

}  // namespace milvus
