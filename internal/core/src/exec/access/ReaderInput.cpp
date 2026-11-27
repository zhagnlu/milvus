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

#include "exec/access/ReaderInput.h"
#include "common/GeometryCache.h"
#include "exec/access/ScanNode.h"
#include "exec/access/PositionNode.h"

namespace milvus::exec::access {
// Only this bridge constructs an InputValueResult with supplied validity.
// Node::Read validates shape, coverage and value/mask agreement before returning.
class ReaderInputAccess {
 public:
    template <typename Node>
    static InputValueResult
    Read(const Node& node, SelectionView rows) {
        auto result =
            node.Read(rows, segcore::reader::ReadProjection::ValuesAndValidity);
        return InputValueResult(
            std::move(*result.values),
            {std::move(result.defined), std::move(result.valid)});
    }
};

namespace {
template <typename Node>
InputLoaders
Loaders(std::shared_ptr<const Node> node) {
    using segcore::reader::ReadProjection;
    return {[node](SelectionView rows) {
                return ReaderInputAccess::Read(*node, rows);
            },
            [node](SelectionView rows) {
                auto result = node->Read(rows, ReadProjection::ValidityOnly);
                return ValidityResult{std::move(result.defined),
                                      std::move(result.valid)};
            }};
}
}  // namespace

plan::PreparedInput
PrepareReaderInput(std::shared_ptr<const segcore::reader::ColumnReader> reader,
                   std::shared_ptr<const segcore::reader::ReadContext> context,
                   bool needs_values) {
    AssertInfo(reader && context, "reader input requires source and context");
    plan::PreparedInput input;
    input.type = reader->type();
    input.supports_values = needs_values;
    input.supports_validity = true;
    input.owner = reader;
    if (auto cache = reader->geometry_cache()) {
        input.make_geometry_lookup =
            [cache = std::move(cache)](const ReadBatchBinding& binding) {
                return [cache, binding](size_t row) {
                    return cache->GetByOffset(binding.offset(row));
                };
            };
    }
    input.read_values = [reader, context](const ReadBatchBinding& binding,
                                          SelectionView rows) {
        if (binding.is_range())
            return ReaderInputAccess::Read(ScanNode(reader, binding, context),
                                           rows);
        return ReaderInputAccess::Read(PositionNode(reader, binding, context),
                                       rows);
    };
    input.read_validity = [reader, context](const ReadBatchBinding& binding,
                                            SelectionView rows) {
        using segcore::reader::ReadProjection;
        auto result = binding.is_range()
                          ? ScanNode(reader, binding, context)
                                .Read(rows, ReadProjection::ValidityOnly)
                          : PositionNode(reader, binding, context)
                                .Read(rows, ReadProjection::ValidityOnly);
        return ValidityResult{std::move(result.defined),
                              std::move(result.valid)};
    };
    input.make_loaders =
        [reader = std::move(reader),
         context = std::move(context)](const ReadBatchBinding& binding) {
            if (binding.is_range())
                return Loaders(
                    std::make_shared<const ScanNode>(reader, binding, context));
            return Loaders(
                std::make_shared<const PositionNode>(reader, binding, context));
        };
    return input;
}
}  // namespace milvus::exec::access
