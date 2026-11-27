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

#include "common/ResultColumn.h"
#include "expr/ITypeExpr.h"

namespace milvus::expr {
// These two unbound nodes are consumed only by the projection binder.
// They deliberately cannot be compiled or used as storage filter expressions.
class ProjectionNameExpr final : public ITypeExpr {
 public:
    explicit ProjectionNameExpr(std::string name)
        : ITypeExpr(DataType::NONE), name_(std::move(name)) {
    }
    const std::string&
    name() const {
        return name_;
    }
    std::string
    ToString() const override {
        return "ProjectionNameExpr";
    }

 private:
    const std::string name_;
};

class ProjectionArithmeticExpr final : public ITypeExpr {
 public:
    ProjectionArithmeticExpr(proto::plan::ArithOpType op,
                             TypedExprPtr left,
                             TypedExprPtr right)
        : ITypeExpr(DataType::NONE, {std::move(left), std::move(right)}),
          op_(op) {
    }
    proto::plan::ArithOpType
    op() const {
        return op_;
    }
    std::string
    ToString() const override {
        return "ProjectionArithmeticExpr";
    }

 private:
    const proto::plan::ArithOpType op_;
};

// Explicit upstream result reference; no storage lookup or name resolution at
// execution time. Type and nullability must match the producer descriptor.
class ResultColumnExpr final : public ITypeExpr {
 public:
    ResultColumnExpr(ResultColumnId id, DataType type, bool nullable = false)
        : ITypeExpr(type), id_(id), nullable_(nullable) {
    }
    ResultColumnId
    id() const {
        return id_;
    }
    bool
    nullable() const {
        return nullable_;
    }
    std::string
    ToString() const override {
        return fmt::format("ResultColumnExpr[{}]", id_.value);
    }

 private:
    const ResultColumnId id_;
    const bool nullable_;
};
}  // namespace milvus::expr
