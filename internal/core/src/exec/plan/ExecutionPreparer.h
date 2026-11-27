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

#include "exec/plan/PreparedExecution.h"

namespace milvus::exec::plan {

// Called once per stored dependency for metadata/resource binding only.
// This hook MUST NOT fetch field payloads or construct a prefetching cursor.
using BindInput = std::function<PreparedInput(const InputDependency&)>;

struct PreparedIndex {
    segcore::reader::IndexResultKind kind;
    std::shared_ptr<const segcore::reader::IndexReader> reader;
    std::shared_ptr<const segcore::reader::ReadContext> context;
    PredicateBackend backend = PredicateBackend::ScalarIndex;
    predicate::PredicateSource selected_source;
    bool selected_needs_raw = false;
    bool force_batch = false;
    PredicateRoute selected_route = PredicateRoute::ExactIndex;
};

// nullopt means no eligible index (metadata only). Exceptions are failures,
// never capability misses. Binder must match field/query and the same fixed
// source snapshot as the raw binder; do not execute the query during binding.
using BindIndex =
    std::function<std::optional<PreparedIndex>(const PredicatePlan&)>;
struct PhysicalPrepareOptions {
    bool use_index = true;
    BindIndex bind_index;
    bool reorder = true;
    std::optional<int64_t> namespace_field;
};

// Select routes before binding raw dependencies. Capability misses are values,
// not exceptions. No query or field payload is read during this phase.
PreparedPredicates
AdjustExecution(const CompiledExprPlan& plan,
                const PhysicalPrepareOptions& physical = {});

// Compatibility convenience gate for INT64. A missing or unmigrated backend
// returns nullopt; invalid Count/context still fails. PrepareIntegerIndex below
// also serves BOOL and the narrower signed scalar types (not nested indexes).
std::optional<PreparedIndex>
PrepareInt64Index(std::shared_ptr<const index::IIndexReaderBase> index,
                  int64_t row_count,
                  segcore::reader::Int64IndexQuery query,
                  std::shared_ptr<const segcore::reader::ReadContext> context);

template <typename T>
std::optional<PreparedIndex>
PrepareIntegerIndex(
    std::shared_ptr<const index::IIndexReaderBase> index,
    int64_t row_count,
    segcore::reader::Int64IndexQuery query,
    std::shared_ptr<const segcore::reader::ReadContext> context) {
    if (!index || !index->Caps().predicate ||
        index->CoordDomain() != index::Domain::Row ||
        !index->Caps().value_lookup)
        return std::nullopt;
    AssertInfo(context, "integer index binding requires a read context");
    return PreparedIndex{
        segcore::reader::IndexResultKind::FullThreeValued,
        std::make_shared<segcore::reader::IntegerIndexReader<T>>(
            std::move(index), row_count, std::move(query)),
        std::move(context)};
}

PreparedExecution
Prepare(CompiledExprPlanPtr plan,
        const BindInput& bind = {},
        const PhysicalPrepareOptions& physical = {});

}  // namespace milvus::exec::plan
