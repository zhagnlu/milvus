// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <functional>
#include "segcore/OffsetBatchIterator.h"
#include "exec/input/InputLayout.h"
#include "segcore/reader/ColumnReader.h"
#include "exec/plan/ExecutionPreparer.h"

namespace milvus {
class Schema;
class IStructElementOffsets;
}  // namespace milvus
namespace milvus::segcore {
class SegmentInternalInterface;
class SegmentReadSnapshot;
class InsertRecordGrowing;

}  // namespace milvus::segcore

namespace milvus::exec::access {

// Traversal and columns use the same fixed query prefix/snapshot. The caller
// keeps the exclusion bitmap alive and immutable until the cursor is released.
segcore::OffsetBatchIterator
MakeSegmentOffsetIterator(
    const segcore::SegmentInternalInterface& segment,
    std::shared_ptr<const segcore::SegmentReadSnapshot> snapshot,
    BitsetTypeView excluded,
    const segcore::reader::ReadContext& context);

// nullptr means an unsupported source capability, before reading payloads.
// Binding errors propagate. Returned Readers own their resources independently.
using BindColumnReader =
    std::function<std::shared_ptr<const segcore::reader::ColumnReader>(
        const InputDependency&)>;

// Element IDs are mapped to document rows only at this boundary. Both the
// mapping and parent readers must belong to the same snapshot/fixed prefix.
BindColumnReader
MakeElementReaderBinding(BindColumnReader parents,
                         std::shared_ptr<const Schema> schema,
                         std::shared_ptr<const IStructElementOffsets> offsets,
                         int64_t document_count,
                         int64_t element_count);

BindColumnReader
MakeSealedReaderBinding(
    std::shared_ptr<const segcore::SegmentReadSnapshot> snapshot,
    int64_t active_count,
    OpContext* op_context = nullptr);

plan::BindIndex
MakeElementIndexBinding(
    std::shared_ptr<const segcore::SegmentReadSnapshot> snapshot,
    std::shared_ptr<const Schema> schema,
    std::shared_ptr<const IStructElementOffsets> offsets,
    std::shared_ptr<const segcore::reader::ReadContext> context);

plan::BindIndex
MakeSealedIndexBinding(
    std::shared_ptr<const segcore::SegmentReadSnapshot> snapshot,
    int64_t active_count,
    std::shared_ptr<const segcore::reader::ReadContext> context,
    bool use_json_stats = true);

plan::BindIndex
MakeSegmentIndexBinding(
    const segcore::SegmentInternalInterface& segment,
    std::shared_ptr<const segcore::SegmentReadSnapshot> snapshot,
    int64_t active_count,
    std::shared_ptr<const segcore::reader::ReadContext> context,
    bool use_json_stats = true);

// Captures the published schema; borrows record/indexes until binding completes.
// The supplied prefix must already be acknowledged by that record.
BindColumnReader
MakeGrowingReaderBinding(const segcore::InsertRecordGrowing& record,
                         std::shared_ptr<const Schema> schema,
                         int64_t active_count);

// The binder borrows segment for synchronous preparation ONLY. Never retain it
// as a lazy loader. Growing captures the published schema and uses the caller's
// fixed query prefix, not a fresh active_count/ACK for each field or batch.
BindColumnReader
MakeSegmentReaderBinding(
    const segcore::SegmentInternalInterface& segment,
    std::shared_ptr<const segcore::SegmentReadSnapshot> snapshot,
    int64_t active_count,
    OpContext* op_context = nullptr);

}  // namespace milvus::exec::access
