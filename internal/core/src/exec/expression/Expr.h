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

#include <memory>
#include <string>

#include "exec/expression/EvalCtx.h"
#include "exec/expression/VectorFunction.h"
#include "exec/QueryContext.h"
#include "expr/ITypeExpr.h"

namespace milvus {
namespace exec {

class Expr {
 public:
    Expr(DataType type,
         const std::vector<std::shared_ptr<Expr>>&& inputs,
         const std::string& name)
        : type_(type),
          inputs_(std::move(inputs)),
          name_(name),
          vector_func_(nullptr) {
    }

    Expr(DataType type,
         const std::vector<std::shared_ptr<Expr>>&& inputs,
         std::shared_ptr<VectorFunction> vec_func,
         const std::string& name)
        : type_(type),
          inputs_(std::move(inputs)),
          name_(name),
          vector_func_(vec_func) {
    }
    virtual ~Expr() = default;

    const DataType&
    type() const {
        return type_;
    }

    virtual void
    Eval(EvalCtx& context, VectorPtr& result) {
    }

 protected:
    DataType type_;
    const std::vector<std::shared_ptr<Expr>> inputs_;
    std::string name_;
    std::shared_ptr<VectorFunction> vector_func_;
};

using ExprPtr = std::shared_ptr<Expr>;

class SegmentExpr : public Expr {
 public:
    SegmentExpr(const std::vector<std::shared_ptr<Expr>>&& input,
                const std::string& name,
                const segcore::SegmentInternalInterface* segment,
                Timestamp query_timestamp,
                int64_t batch_size)
        : Expr(DataType::BOOL, std::move(input), name),
          segment_(segment),
          query_timestamp_(query_timestamp),
          batch_size_(batch_size) {
        num_rows_ = segment_->get_active_count(query_timestamp_);
        size_per_chunk_ = segment_->size_per_chunk();
        AssertInfo(batch_size_ > 0, "expr batch size should greater than zero");
    }

 protected:
    const segcore::SegmentInternalInterface* segment_;
    Timestamp query_timestamp_;
    int64_t batch_size_;

    // State indicate position that expr computing at
    // because expr maybe called for every batch.
    bool is_index_mode_{false};
    bool is_data_mode_{false};

    int32_t num_rows_{0};
    int32_t current_num_rows_{0};
    int32_t num_data_chunk_{0};
    int32_t num_index_chunk_{0};
    int32_t current_data_chunk_{0};
    int32_t current_data_chunk_pos_{0};
    int32_t current_index_chunk_{0};
    int32_t size_per_chunk_{0};
};

std::vector<ExprPtr>
CompileExpressions(const std::vector<expr::TypedExprPtr>& logical_exprs,
                   ExecContext* context,
                   const std::unordered_set<std::string>& flatten_cadidates =
                       std::unordered_set<std::string>(),
                   bool enable_constant_folding = false);

std::vector<ExprPtr>
CompileInputs(const expr::TypedExprPtr& expr,
              QueryContext* config,
              const std::unordered_set<std::string>& flatten_cadidates);

ExprPtr
CompileExpression(const expr::TypedExprPtr& expr,
                  QueryContext* context,
                  const std::unordered_set<std::string>& flatten_cadidates,
                  bool enable_constant_folding);

class ExprSet {
 public:
    explicit ExprSet(const std::vector<expr::TypedExprPtr>& logical_exprs,
                     ExecContext* exec_ctx) {
        exprs_ = CompileExpressions(logical_exprs, exec_ctx);
    }

    virtual ~ExprSet() = default;

    void
    Eval(EvalCtx& ctx, std::vector<VectorPtr>& results) {
        Eval(0, exprs_.size(), true, ctx, results);
    }

    virtual void
    Eval(int32_t begin,
         int32_t end,
         bool initialize,
         EvalCtx& ctx,
         std::vector<VectorPtr>& result);

    void
    Clear() {
        exprs_.clear();
    }

    ExecContext*
    get_exec_context() const {
        return exec_ctx_;
    }

    size_t
    size() const {
        return exprs_.size();
    }

    const std::vector<std::shared_ptr<Expr>>&
    exprs() const {
        return exprs_;
    }

    const std::shared_ptr<Expr>&
    expr(int32_t index) const {
        return exprs_[index];
    }

 private:
    std::vector<std::shared_ptr<Expr>> exprs_;
    ExecContext* exec_ctx_;
};

template <typename T>
T
GetValueFromProto(const milvus::proto::plan::GenericValue& value_proto) {
    if constexpr (std::is_same_v<T, bool>) {
        Assert(value_proto.val_case() ==
               milvus::proto::plan::GenericValue::kBoolVal);
        return static_cast<T>(value_proto.bool_val());
    } else if constexpr (std::is_integral_v<T>) {
        Assert(value_proto.val_case() ==
               milvus::proto::plan::GenericValue::kInt64Val);
        return static_cast<T>(value_proto.int64_val());
    } else if constexpr (std::is_floating_point_v<T>) {
        Assert(value_proto.val_case() ==
               milvus::proto::plan::GenericValue::kFloatVal);
        return static_cast<T>(value_proto.float_val());
    } else if constexpr (std::is_same_v<T, std::string>) {
        Assert(value_proto.val_case() ==
               milvus::proto::plan::GenericValue::kStringVal);
        return static_cast<T>(value_proto.string_val());
    } else {
        PanicInfo("unsupported generic value type");
    }
};

}  //namespace exec
}  // namespace milvus