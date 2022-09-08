// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include <map>

#include "exceptions/EasyAssert.h"
#include "indexbuilder/VecIndexCreator.h"
#include "index/Utils.h"
#include "index/IndexFactory.h"
#include "storage/DiskANNFileManagerImpl.h"

namespace milvus::indexbuilder {

VecIndexCreator::VecIndexCreator(DataType data_type,
                                 const char* serialized_type_params,
                                 const char* serialized_index_params)
    : data_type_(data_type) {
    milvus::Index::ParseFromString(type_params_, std::string(serialized_type_params));
    milvus::Index::ParseFromString(index_params_, std::string(serialized_index_params));

    for (auto i = 0; i < type_params_.params_size(); ++i) {
        const auto& param = type_params_.params(i);
        config_[param.key()] = param.value();
    }

    for (auto i = 0; i < index_params_.params_size(); ++i) {
        const auto& param = index_params_.params(i);
        config_[param.key()] = param.value();
    }

    Index::CreateIndexInfo index_info;
    index_info.field_type = data_type_;
    index_info.index_mode = Index::GetIndexModeFromConfig(config_);
    index_info.index_type = Index::GetIndexTypeFromConfig(config_);
    index_info.metric_type = Index::GetMetricTypeFromConfig(config_);

    std::shared_ptr<storage::FileManagerImpl> file_manager = nullptr;
    if (Index::is_in_disk_list(index_info.index_type)) {
        // For now, only support diskann index
        file_manager = std::make_shared<storage::DiskANNFileManagerImpl>(Index::GetFieldDataMetaFromConfig(config_),
                                                                         Index::GetIndexMetaFromConfig(config_));
    }

    index_ = Index::IndexFactory::GetInstance().CreateIndex(index_info, file_manager);
    AssertInfo(index_ != nullptr, "[VecIndexCreator]Index is null after create index");
}

int64_t
VecIndexCreator::dim() {
    return Index::GetDimFromConfig(config_);
}

void
VecIndexCreator::Build(const milvus::DatasetPtr& dataset) {
    index_->BuildWithDataset(dataset, config_);
}

milvus::BinarySet
VecIndexCreator::Serialize() {
    return index_->Serialize(config_);
}

void
VecIndexCreator::Load(const milvus::BinarySet& binary_set) {
    index_->Load(binary_set, config_);
}

std::unique_ptr<SearchResult>
VecIndexCreator::Query(const milvus::DatasetPtr& dataset, const SearchInfo& search_info, const BitsetView& bitset) {
    auto vector_index = dynamic_cast<Index::VectorIndex*>(index_.get());
    return vector_index->Query(dataset, search_info, bitset);
}

void
VecIndexCreator::CleanLocalData() {
    auto vector_index = dynamic_cast<Index::VectorIndex*>(index_.get());
    vector_index->CleanLocalData();
}

}  // namespace milvus::indexbuilder
