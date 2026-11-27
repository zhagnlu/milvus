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

#include "exec/expression/ScalarCallExpr.h"
#include "exec/expression/KernelSelection.h"

namespace milvus::exec::scalar {
namespace {

class CallExpr final : public Expr {
 public:
    CallExpr(DataType type,
             std::vector<ExprPtr> inputs,
             ScalarFunction function)
        : Expr(type),
          inputs_(std::move(inputs)),
          function_(std::move(function)) {
        AssertInfo(function_, "scalar call requires a function");
        for (const auto& input : inputs_) {
            AssertInfo(input, "scalar call contains a null argument");
        }
    }

 protected:
    ReadOnlyVectorView
    EvalImpl(SelectionView rows, EvalCtx& ctx) const override {
        std::vector<ReadOnlyVectorView> args;
        args.reserve(inputs_.size());
        for (const auto& input : inputs_) {
            args.push_back(input->Eval(rows, ctx));
        }
        return function_(args, rows, ctx);
    }

 private:
    std::vector<ExprPtr> inputs_;
    ScalarFunction function_;
};

}  // namespace

ExprPtr
MakeCall(DataType result_type,
         std::vector<ExprPtr> inputs,
         ScalarFunction function) {
    return std::make_shared<CallExpr>(
        result_type, std::move(inputs), std::move(function));
}

ExprPtr
MakeLegacyCall(DataType result_type,
               std::vector<ExprPtr> inputs,
               expression::FilterFunctionPtr function) {
    AssertInfo(function, "legacy scalar call requires a function");
    return MakeCall(
        result_type,
        std::move(inputs),
        [function, result_type](const std::vector<ReadOnlyVectorView>& args,
                                SelectionView rows,
                                EvalCtx&) {
            if (rows.IsAll()) {
                // Full batches need neither an identity dictionary nor scatter.
                // Legacy functions still receive detached writable arguments.
                std::vector<VectorPtr> gathered;
                gathered.reserve(args.size());
                for (const auto& arg : args)
                    gathered.push_back(arg.CopyForLegacyFunction());
                RowVector batch(rows.row_count(), std::move(gathered));
                VectorPtr output;
                function(batch, output);
                AssertInfo(
                    output && output->size() == rows.row_count(),
                    "legacy scalar function returned wrong result length");
                const auto values = ReadOnlyVectorView::CopyOf(output);
                AssertInfo(values.logical_type() == result_type,
                           "legacy scalar function returned wrong result type");
                return values;
            }
            std::vector<vector_size_t> positions;
            positions.reserve(rows.selected_count());
            // Optimize only position gathering. The opaque legacy function
            // still receives exactly the selected rows in one invocation.
            VisitSelected(
                rows,
                SelectionKernelCost{1, 2, 16, 0, 4},
                [&](size_t begin, size_t count) {
                    for (size_t i = 0; i < count; ++i)
                        positions.push_back(begin + i);
                },
                [&](size_t row) { positions.push_back(row); });
            auto mapping = std::make_shared<RowMapping>(std::move(positions));
            const auto count = mapping->indices().size();
            std::vector<VectorPtr> gathered;
            gathered.reserve(args.size());
            for (const auto& arg : args) {
                // Never hand published/dictionary vectors to legacy kernels.
                gathered.push_back(
                    arg.Dictionary(mapping).CopyForLegacyFunction());
            }
            RowVector compact(count, std::move(gathered));
            VectorPtr output;
            function(compact, output);
            AssertInfo(output && output->size() == count,
                       "legacy scalar function returned wrong result length");
            const auto values = ReadOnlyVectorView::CopyOf(output);
            AssertInfo(values.logical_type() == result_type,
                       "legacy scalar function returned wrong result type");
            auto scattered = MakeWritableCopy(ReadOnlyVectorView::Nulls(
                result_type, SelectionView::None(rows.row_count())));
            CopySelected(
                values, SelectionView::All(count), *mapping, scattered);
            return ReadOnlyVectorView::CopyOf(scattered.values,
                                              scattered.defined_rows.view());
        });
}

}  // namespace milvus::exec::scalar
