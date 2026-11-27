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

#include "exec/plan/PreparedExecution.h"

namespace milvus::exec {
// Keep reset private to the execution lifecycle, rather than allowing callers
// to clear a BatchInput while its loaders still capture an earlier binding.
class BatchInputResetAccess {
 public:
    static void
    Reset(BatchInput& input, size_t size) {
        input.Reset(size);
    }
};
}  // namespace milvus::exec

namespace milvus::exec::plan {
namespace {
struct EvaluationGuard {
    bool& evaluating;
    EvaluationGuard(bool& state, bool ready) : evaluating(state) {
        AssertInfo(ready && !evaluating, "execution workspace is unavailable");
        evaluating = true;
    }
    ~EvaluationGuard() {
        evaluating = false;
    }
};

class DeferredInput {
 public:
    DeferredInput(PreparedInput source, ReadBatchBinding binding)
        : source_(std::move(source)), binding_(std::move(binding)) {
    }

    void
    Reset(const ReadBatchBinding& binding) {
        AssertInfo(!initializing_,
                   "cannot reset a source during initialization");
        loaders_.reset();
        binding_ = binding;
    }

    InputValueResult
    Values(SelectionView rows) {
        return source_.read_values ? source_.read_values(binding_, rows)
                                   : Get().values(rows);
    }

    ValidityResult
    Validity(SelectionView rows) {
        return source_.read_validity ? source_.read_validity(binding_, rows)
                                     : Get().validity(rows);
    }

    const InputLoaders&
    Get() {
        AssertInfo(!initializing_, "recursive scalar loader initialization");
        if (!loaders_) {
            struct Guard {
                bool& flag;
                explicit Guard(bool& value) : flag(value) {
                    flag = true;
                }
                ~Guard() {
                    flag = false;
                }
            } guard(initializing_);
            auto loaded = source_.make_loaders(binding_);
            AssertInfo(!source_.supports_values || loaded.values,
                       "prepared source omitted its value loader");
            AssertInfo(!source_.supports_validity || loaded.validity,
                       "prepared source omitted its validity loader");
            // Do not publish a partially initialized factory result on failure.
            loaders_ = std::move(loaded);
        }
        return *loaders_;
    }

 private:
    PreparedInput source_;
    ReadBatchBinding binding_;
    std::optional<InputLoaders> loaders_;
    bool initializing_ = false;
};

}  // namespace

std::vector<std::optional<InputDependency>>
RequiredInputs(const CompiledExprPlan& plan,
               const std::vector<InputUse>& uses) {
    std::vector<std::optional<InputDependency>> required(
        plan.inputs().slots().size());
    for (const auto& use : uses) {
        AssertInfo(use.slot < required.size(),
                   "raw dependency slot out of bounds");
        if (!required[use.slot]) {
            auto dependency = plan.inputs().slots()[use.slot];
            dependency.needs_values = false;
            dependency.needs_validity = true;
            required[use.slot] = std::move(dependency);
        }
        required[use.slot]->needs_values |= use.values;
    }
    return required;
}

PreparedExecution::PreparedExecution(
    CompiledExprPlanPtr plan,
    std::vector<std::optional<PreparedInput>> inputs,
    std::optional<PreparedPredicates> predicates)
    : plan_(std::move(plan)), inputs_(std::move(inputs)) {
    AssertInfo(plan_ && inputs_.size() == plan_->inputs().slots().size(),
               "prepared scalar input layout mismatch");
    if (predicates) {
        predicates_ = std::move(*predicates);
    } else {
        AssertInfo(plan_->predicates().empty(),
                   "physical predicates were not prepared");
        predicates_.raw_inputs = plan_->direct_inputs();
    }
    AssertInfo(predicates_.supported,
               "cannot publish unsupported predicate sources");
    AssertInfo(predicates_.sources.size() >= plan_->predicates().size() &&
                   predicates_.routes.size() == predicates_.sources.size(),
               "prepared predicate layout mismatch");
    requirements_ = RequiredInputs(*plan_, predicates_.raw_inputs);
    for (size_t i = 0; i < inputs_.size(); ++i) {
        if (!requirements_[i]) {
            AssertInfo(!inputs_[i],
                       "unselected raw dependency must not bind storage");
            continue;
        }
        const auto& dependency = *requirements_[i];
        AssertInfo(!dependency.result_column || dependency.upstream_column,
                   "result dependency must not bind storage");
        if (dependency.upstream_column) {
            AssertInfo(!inputs_[i], "upstream input must not bind storage");
            const auto& schema = plan_->upstream_schema();
            AssertInfo(
                schema &&
                    *dependency.upstream_column < schema->column_count() &&
                    schema->column_type(*dependency.upstream_column) ==
                        dependency.type,
                "upstream dependency does not match compiled schema");
            continue;
        }
        AssertInfo(inputs_[i], "missing prepared scalar source");
        const auto& source = *inputs_[i];
        AssertInfo(source.type == dependency.type && source.owner &&
                       source.make_loaders,
                   "invalid prepared scalar source type/owner/factory");
        if ((dependency.needs_values && !source.supports_values) ||
            (dependency.needs_validity && !source.supports_validity &&
             !source.supports_values)) {
            ThrowInfo(Unsupported,
                      "scalar source lacks required value/validity capability");
        }
    }
}

std::unique_ptr<BatchInput>
PreparedExecution::MakeBatch(const ReadBatchBinding& binding,
                             const RowVector* upstream,
                             std::optional<SelectionView> upstream_rows) const {
    AssertInfo(plan_->predicates().empty(),
               "physical predicate plan requires MakeExecutionBatch");
    return MakeRawBatch(binding, upstream, upstream_rows);
}

std::unique_ptr<BatchInput>
PreparedExecution::MakeRawBatch(
    const ReadBatchBinding& binding,
    const RowVector* upstream,
    std::optional<SelectionView> upstream_rows,
    std::vector<std::function<void(const ReadBatchBinding&)>>* resets) const {
    AssertInfo(!upstream_rows ||
                   (upstream && upstream_rows->row_count() == binding.size()),
               "upstream coverage requires a matching upstream batch");
    const bool needs_upstream = std::any_of(
        requirements_.begin(), requirements_.end(), [](const auto& dependency) {
            return dependency && dependency->upstream_column.has_value();
        });
    AssertInfo(!resets || !needs_upstream,
               "reusable workspace cannot bind upstream inputs");
    // Validate upstream before constructing any deferred source.
    if (needs_upstream) {
        AssertInfo(upstream && upstream->size() == binding.size(),
                   "missing or misaligned scalar upstream batch");
        const auto& expected = plan_->upstream_schema();
        const auto& actual = upstream->row_type();
        AssertInfo(actual->column_count() == expected->column_count(),
                   "scalar upstream schema changed");
        for (size_t i = 0; i < expected->column_count(); ++i) {
            AssertInfo(actual->column_name(i) == expected->column_name(i) &&
                           actual->column_type(i) == expected->column_type(i),
                       "scalar upstream schema changed");
        }
    }
    auto input =
        std::make_unique<BatchInput>(plan_->inputs().schema(), binding.size());
    for (size_t i = 0; i < inputs_.size(); ++i) {
        if (!requirements_[i])
            continue;
        const auto& dependency = *requirements_[i];
        if (dependency.upstream_column) {
            input->BindColumn(
                static_cast<column_index_t>(i),
                *upstream,
                *dependency.upstream_column,
                upstream_rows.value_or(SelectionView::All(binding.size())));
        } else {
            if (inputs_[i]->make_geometry_lookup)
                input->BindGeometryLookup(
                    i, inputs_[i]->make_geometry_lookup(binding));
            auto state = std::make_shared<DeferredInput>(*inputs_[i], binding);
            if (resets) {
                resets->push_back([state,
                                   raw = input.get(),
                                   slot = i,
                                   geometry = inputs_[i]->make_geometry_lookup](
                                      const ReadBatchBinding& next) {
                    state->Reset(next);
                    if (geometry)
                        raw->BindGeometryLookup(slot, geometry(next));
                });
            }
            BatchInput::ValueLoader values;
            BatchInput::ValidityLoader validity;
            if (inputs_[i]->supports_values) {
                values = [state](SelectionView rows) {
                    return state->Values(rows);
                };
            }
            if (inputs_[i]->supports_validity) {
                validity = [state](SelectionView rows) {
                    return state->Validity(rows);
                };
            }
            input->BindLoader(static_cast<column_index_t>(i),
                              std::move(values),
                              std::move(validity));
        }
    }
    return input;
}

std::unique_ptr<ExecutionBatch>
PreparedExecution::MakeExecutionBatch(
    const ReadBatchBinding& binding,
    const RowVector* upstream,
    std::optional<SelectionView> upstream_rows) const {
    return std::make_unique<ExecutionBatch>(
        plan_,
        MakeRawBatch(binding, upstream, upstream_rows),
        binding,
        predicates_.sources,
        predicates_.expressions);
}

std::unique_ptr<ExecutionBatch>
PreparedExecution::MakeReusableBatch(const ReadBatchBinding& binding) const {
    std::vector<std::function<void(const ReadBatchBinding&)>> resets;
    auto input = MakeRawBatch(binding, nullptr, std::nullopt, &resets);
    auto batch = std::make_unique<ExecutionBatch>(plan_,
                                                  std::move(input),
                                                  binding,
                                                  predicates_.sources,
                                                  predicates_.expressions);
    batch->resets_ = std::move(resets);
    return batch;
}

void
ExecutionBatch::Reset(const ReadBatchBinding& binding) {
    AssertInfo(resets_ && !evaluating_ && ready_,
               "execution workspace cannot be reset");
    ready_ = false;
    BatchInputResetAccess::Reset(*input_, binding.size());
    for (auto& reset : *resets_) reset(binding);
    predicates_->Reset(binding);
    ready_ = true;
}

ExecutionBatch::ExecutionBatch(
    CompiledExprPlanPtr plan,
    std::unique_ptr<BatchInput> input,
    ReadBatchBinding binding,
    std::vector<predicate::PredicateSource> sources,
    std::shared_ptr<const scalar::ExprSet> expressions)
    : plan_(std::move(plan)),
      input_(std::move(input)),
      expressions_(std::move(expressions)) {
    AssertInfo(plan_ && input_ && sources.size() >= plan_->predicates().size(),
               "execution batch does not match its plan");
    predicates_ = std::make_shared<predicate::PredicateBatch>(
        *input_, std::move(binding), std::move(sources));
}

std::vector<ReadOnlyVectorView>
ExecutionBatch::Eval(SelectionView rows, std::function<void()> checkpoint) {
    EvaluationGuard guard(evaluating_, ready_);
    scalar::EvalCtx ctx(*input_, std::move(checkpoint), predicates_);
    return (expressions_ ? *expressions_ : plan_->expressions())
        .Eval(rows, ctx);
}

std::vector<ReadOnlyVectorView>
ExecutionBatch::Eval(const std::vector<SelectionView>& rows,
                     std::function<void()> checkpoint) {
    EvaluationGuard guard(evaluating_, ready_);
    scalar::EvalCtx ctx(*input_, std::move(checkpoint), predicates_);
    return (expressions_ ? *expressions_ : plan_->expressions())
        .Eval(rows, ctx);
}

}  // namespace milvus::exec::plan
