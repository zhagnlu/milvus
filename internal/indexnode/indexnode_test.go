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

//go:build linux
// +build linux

package indexnode

import (
	"container/list"
	"context"
	"path"
	"strconv"
	"testing"
	"time"

	"go.uber.org/zap"

	"github.com/golang/protobuf/proto"
	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus/internal/common"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/etcdpb"
	"github.com/milvus-io/milvus/internal/proto/indexpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	"github.com/milvus-io/milvus/internal/proto/schemapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/dependency"
	"github.com/milvus-io/milvus/internal/util/etcd"
	"github.com/milvus-io/milvus/internal/util/metricsinfo"
	"github.com/milvus-io/milvus/internal/util/sessionutil"
)

func TestIndexNode(t *testing.T) {
	ctx := context.Background()

	indexID := UniqueID(999)
	indexBuildID1 := UniqueID(54321)
	indexBuildID2 := UniqueID(12345)
	floatVectorFieldID := UniqueID(101)
	binaryVectorFieldID := UniqueID(102)
	tsFieldID := UniqueID(1)
	collectionID := UniqueID(201)
	floatVectorFieldName := "float_vector"
	binaryVectorFieldName := "binary_vector"
	metaPath1 := "FloatVector"
	metaPath2 := "BinaryVector"
	metaPath3 := "FloatVectorDeleted"
	floatVectorBinlogPath := "float_vector_binlog"
	binaryVectorBinlogPath := "binary_vector_binlog"

	factory := dependency.NewDefaultFactory(true)
	in, err := NewIndexNode(ctx, factory)
	assert.Nil(t, err)
	Params.Init()

	etcdCli, err := etcd.GetEtcdClient(&Params.EtcdCfg)
	assert.NoError(t, err)
	in.SetEtcdClient(etcdCli)
	defer etcdCli.Close()

	err = in.Init()
	assert.Nil(t, err)

	err = in.Start()
	assert.Nil(t, err)

	err = in.Register()
	assert.Nil(t, err)

	in.chunkManager = storage.NewLocalChunkManager(storage.RootPath("/tmp/lib/milvus"))
	t.Run("CreateIndex FloatVector", func(t *testing.T) {
		var insertCodec storage.InsertCodec

		insertCodec.Schema = &etcdpb.CollectionMeta{
			ID: collectionID,
			Schema: &schemapb.CollectionSchema{
				Fields: []*schemapb.FieldSchema{
					{
						FieldID:      floatVectorFieldID,
						Name:         floatVectorFieldName,
						IsPrimaryKey: false,
						DataType:     schemapb.DataType_FloatVector,
					},
				},
			},
		}
		data := make(map[UniqueID]storage.FieldData)
		tsData := make([]int64, nb)
		for i := 0; i < nb; i++ {
			tsData[i] = int64(i + 100)
		}
		data[tsFieldID] = &storage.Int64FieldData{
			NumRows: []int64{nb},
			Data:    tsData,
		}
		data[floatVectorFieldID] = &storage.FloatVectorFieldData{
			NumRows: []int64{nb},
			Data:    generateFloatVectors(),
			Dim:     dim,
		}
		insertData := storage.InsertData{
			Data: data,
			Infos: []storage.BlobInfo{
				{
					Length: 10,
				},
			},
		}
		binLogs, _, err := insertCodec.Serialize(999, 888, &insertData)
		assert.Nil(t, err)
		kvs := make(map[string][]byte, len(binLogs))
		paths := make([]string, 0, len(binLogs))
		for i, blob := range binLogs {
			key := path.Join(floatVectorBinlogPath, strconv.Itoa(i))
			paths = append(paths, key)
			kvs[key] = blob.Value[:]
		}
		err = in.chunkManager.MultiWrite(kvs)
		assert.Nil(t, err)

		indexMeta := &indexpb.IndexMeta{
			IndexBuildID: indexBuildID1,
			State:        commonpb.IndexState_InProgress,
			IndexVersion: 1,
		}

		value, err := proto.Marshal(indexMeta)
		assert.Nil(t, err)
		err = in.etcdKV.Save(metaPath1, string(value))
		assert.Nil(t, err)
		req := &indexpb.CreateIndexRequest{
			IndexBuildID: indexBuildID1,
			IndexName:    "FloatVector",
			IndexID:      indexID,
			Version:      1,
			MetaPath:     metaPath1,
			DataPaths:    paths,
			TypeParams: []*commonpb.KeyValuePair{
				{
					Key:   "dim",
					Value: "8",
				},
			},
			IndexParams: []*commonpb.KeyValuePair{
				{
					Key:   "index_type",
					Value: "IVF_SQ8",
				},
				{
					Key:   "params",
					Value: "{\"nlist\": 128}",
				},
				{
					Key:   "metric_type",
					Value: "L2",
				},
			},
		}

		status, err2 := in.CreateIndex(ctx, req)
		assert.Nil(t, err2)
		assert.Equal(t, commonpb.ErrorCode_Success, status.ErrorCode)

		strValue, err3 := in.etcdKV.Load(metaPath1)
		assert.Nil(t, err3)
		indexMetaTmp := indexpb.IndexMeta{}
		err = proto.Unmarshal([]byte(strValue), &indexMetaTmp)
		assert.Nil(t, err)
		for indexMetaTmp.State != commonpb.IndexState_Finished {
			time.Sleep(100 * time.Millisecond)
			strValue, err := in.etcdKV.Load(metaPath1)
			assert.Nil(t, err)
			err = proto.Unmarshal([]byte(strValue), &indexMetaTmp)
			assert.Nil(t, err)
		}
		defer in.chunkManager.MultiRemove(indexMetaTmp.IndexFilePaths)
		defer func() {
			for k := range kvs {
				err = in.chunkManager.Remove(k)
				assert.Nil(t, err)
			}
		}()

		defer in.etcdKV.RemoveWithPrefix(metaPath1)
	})
	t.Run("CreateIndex BinaryVector", func(t *testing.T) {
		var insertCodec storage.InsertCodec

		insertCodec.Schema = &etcdpb.CollectionMeta{
			ID: collectionID,
			Schema: &schemapb.CollectionSchema{
				Fields: []*schemapb.FieldSchema{
					{
						FieldID:      binaryVectorFieldID,
						Name:         binaryVectorFieldName,
						IsPrimaryKey: false,
						DataType:     schemapb.DataType_BinaryVector,
					},
				},
			},
		}
		data := make(map[UniqueID]storage.FieldData)
		tsData := make([]int64, nb)
		for i := 0; i < nb; i++ {
			tsData[i] = int64(i + 100)
		}
		data[tsFieldID] = &storage.Int64FieldData{
			NumRows: []int64{nb},
			Data:    tsData,
		}
		data[binaryVectorFieldID] = &storage.BinaryVectorFieldData{
			NumRows: []int64{nb},
			Data:    generateBinaryVectors(),
			Dim:     dim,
		}
		insertData := storage.InsertData{
			Data: data,
			Infos: []storage.BlobInfo{
				{
					Length: 10,
				},
			},
		}
		binLogs, _, err := insertCodec.Serialize(999, 888, &insertData)
		assert.Nil(t, err)
		kvs := make(map[string][]byte, len(binLogs))
		paths := make([]string, 0, len(binLogs))
		for i, blob := range binLogs {
			key := path.Join(binaryVectorBinlogPath, strconv.Itoa(i))
			paths = append(paths, key)
			kvs[key] = blob.Value[:]
		}
		err = in.chunkManager.MultiWrite(kvs)
		assert.Nil(t, err)

		indexMeta := &indexpb.IndexMeta{
			IndexBuildID: indexBuildID2,
			State:        commonpb.IndexState_InProgress,
			IndexVersion: 1,
		}

		value, err := proto.Marshal(indexMeta)
		assert.Nil(t, err)
		err = in.etcdKV.Save(metaPath2, string(value))
		assert.Nil(t, err)
		req := &indexpb.CreateIndexRequest{
			IndexBuildID: indexBuildID2,
			IndexName:    "BinaryVector",
			IndexID:      indexID,
			Version:      1,
			MetaPath:     metaPath2,
			DataPaths:    paths,
			TypeParams: []*commonpb.KeyValuePair{
				{
					Key:   "dim",
					Value: "8",
				},
			},
			IndexParams: []*commonpb.KeyValuePair{
				{
					Key:   "index_type",
					Value: "BIN_FLAT",
				},
				{
					Key:   "metric_type",
					Value: "JACCARD",
				},
			},
		}

		status, err2 := in.CreateIndex(ctx, req)
		assert.Nil(t, err2)
		assert.Equal(t, commonpb.ErrorCode_Success, status.ErrorCode)

		strValue, err3 := in.etcdKV.Load(metaPath2)
		assert.Nil(t, err3)
		indexMetaTmp := indexpb.IndexMeta{}
		err = proto.Unmarshal([]byte(strValue), &indexMetaTmp)
		assert.Nil(t, err)
		for indexMetaTmp.State != commonpb.IndexState_Finished {
			time.Sleep(100 * time.Millisecond)
			strValue, err = in.etcdKV.Load(metaPath2)
			assert.Nil(t, err)
			err = proto.Unmarshal([]byte(strValue), &indexMetaTmp)
			assert.Nil(t, err)
		}
		defer in.chunkManager.MultiRemove(indexMetaTmp.IndexFilePaths)
		defer func() {
			for k := range kvs {
				err = in.chunkManager.Remove(k)
				assert.Nil(t, err)
			}
		}()

		defer in.etcdKV.RemoveWithPrefix(metaPath2)
	})

	t.Run("Create DeletedIndex", func(t *testing.T) {
		var insertCodec storage.InsertCodec

		insertCodec.Schema = &etcdpb.CollectionMeta{
			ID: collectionID,
			Schema: &schemapb.CollectionSchema{
				Fields: []*schemapb.FieldSchema{
					{
						FieldID:      floatVectorFieldID,
						Name:         floatVectorFieldName,
						IsPrimaryKey: false,
						DataType:     schemapb.DataType_FloatVector,
					},
				},
			},
		}
		data := make(map[UniqueID]storage.FieldData)
		tsData := make([]int64, nb)
		for i := 0; i < nb; i++ {
			tsData[i] = int64(i + 100)
		}
		data[tsFieldID] = &storage.Int64FieldData{
			NumRows: []int64{nb},
			Data:    tsData,
		}
		data[floatVectorFieldID] = &storage.FloatVectorFieldData{
			NumRows: []int64{nb},
			Data:    generateFloatVectors(),
			Dim:     dim,
		}
		insertData := storage.InsertData{
			Data: data,
			Infos: []storage.BlobInfo{
				{
					Length: 10,
				},
			},
		}
		binLogs, _, err := insertCodec.Serialize(999, 888, &insertData)
		assert.Nil(t, err)
		kvs := make(map[string][]byte, len(binLogs))
		paths := make([]string, 0, len(binLogs))
		for i, blob := range binLogs {
			key := path.Join(floatVectorBinlogPath, strconv.Itoa(i))
			paths = append(paths, key)
			kvs[key] = blob.Value[:]
		}
		err = in.chunkManager.MultiWrite(kvs)
		assert.Nil(t, err)

		indexMeta := &indexpb.IndexMeta{
			IndexBuildID: indexBuildID1,
			State:        commonpb.IndexState_InProgress,
			IndexVersion: 1,
			MarkDeleted:  true,
		}

		value, err := proto.Marshal(indexMeta)
		assert.Nil(t, err)
		err = in.etcdKV.Save(metaPath3, string(value))
		assert.Nil(t, err)
		req := &indexpb.CreateIndexRequest{
			IndexBuildID: indexBuildID1,
			IndexName:    "FloatVector",
			IndexID:      indexID,
			Version:      1,
			MetaPath:     metaPath3,
			DataPaths:    paths,
			TypeParams: []*commonpb.KeyValuePair{
				{
					Key:   "dim",
					Value: "8",
				},
			},
			IndexParams: []*commonpb.KeyValuePair{
				{
					Key:   "index_type",
					Value: "IVF_SQ8",
				},
				{
					Key:   "params",
					Value: "{\"nlist\": 128}",
				},
				{
					Key:   "metric_type",
					Value: "L2",
				},
			},
		}

		status, err2 := in.CreateIndex(ctx, req)
		assert.Nil(t, err2)
		assert.Equal(t, commonpb.ErrorCode_Success, status.ErrorCode)
		time.Sleep(100 * time.Millisecond)
		strValue, err3 := in.etcdKV.Load(metaPath3)
		assert.Nil(t, err3)
		indexMetaTmp := indexpb.IndexMeta{}
		err = proto.Unmarshal([]byte(strValue), &indexMetaTmp)
		assert.Nil(t, err)
		assert.Equal(t, true, indexMetaTmp.MarkDeleted)
		assert.Equal(t, int64(1), indexMetaTmp.IndexVersion)
		//for indexMetaTmp.State != commonpb.IndexState_Finished {
		//	time.Sleep(100 * time.Millisecond)
		//	strValue, err := in.etcdKV.Load(metaPath3)
		//	assert.Nil(t, err)
		//	err = proto.Unmarshal([]byte(strValue), &indexMetaTmp)
		//	assert.Nil(t, err)
		//}
		defer in.chunkManager.MultiRemove(indexMetaTmp.IndexFilePaths)
		defer func() {
			for k := range kvs {
				err = in.chunkManager.Remove(k)
				assert.Nil(t, err)
			}
		}()

		defer in.etcdKV.RemoveWithPrefix(metaPath3)
	})

	t.Run("GetComponentStates", func(t *testing.T) {
		resp, err := in.GetComponentStates(ctx)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
		assert.Equal(t, internalpb.StateCode_Healthy, resp.State.StateCode)
	})

	t.Run("GetTimeTickChannel", func(t *testing.T) {
		resp, err := in.GetTimeTickChannel(ctx)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	})

	t.Run("GetStatisticsChannel", func(t *testing.T) {
		resp, err := in.GetStatisticsChannel(ctx)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	})

	t.Run("GetMetrics_system_info", func(t *testing.T) {
		req, err := metricsinfo.ConstructRequestByMetricType(metricsinfo.SystemInfoMetrics)
		assert.Nil(t, err)
		resp, err := in.GetMetrics(ctx, req)
		assert.Nil(t, err)
		log.Info("GetMetrics_system_info",
			zap.String("resp", resp.Response),
			zap.String("name", resp.ComponentName))
	})
	err = in.etcdKV.RemoveWithPrefix("session/IndexNode")
	assert.Nil(t, err)

	err = in.Stop()
	assert.Nil(t, err)
}

func TestCreateIndexFailed(t *testing.T) {
	ctx := context.Background()

	indexID := UniqueID(1001)
	indexBuildID1 := UniqueID(54322)
	indexBuildID2 := UniqueID(54323)
	floatVectorFieldID := UniqueID(102)
	tsFieldID := UniqueID(1)
	collectionID := UniqueID(202)
	floatVectorFieldName := "float_vector"
	metaPath1 := "FloatVector1"
	metaPath2 := "FloatVector2"
	floatVectorBinlogPath := "float_vector_binlog"

	factory := dependency.NewDefaultFactory(true)
	in, err := NewIndexNode(ctx, factory)
	assert.Nil(t, err)
	Params.Init()

	etcdCli, err := etcd.GetEtcdClient(&Params.EtcdCfg)
	assert.NoError(t, err)
	in.SetEtcdClient(etcdCli)
	defer etcdCli.Close()

	err = in.Init()
	assert.Nil(t, err)

	err = in.Start()
	assert.Nil(t, err)

	err = in.Register()
	assert.Nil(t, err)

	in.chunkManager = storage.NewLocalChunkManager(storage.RootPath("/tmp/lib/milvus"))
	t.Run("CreateIndex error", func(t *testing.T) {
		var insertCodec storage.InsertCodec

		insertCodec.Schema = &etcdpb.CollectionMeta{
			ID: collectionID,
			Schema: &schemapb.CollectionSchema{
				Fields: []*schemapb.FieldSchema{
					{
						FieldID:      floatVectorFieldID,
						Name:         floatVectorFieldName,
						IsPrimaryKey: false,
						DataType:     schemapb.DataType_FloatVector,
					},
				},
			},
		}
		data := make(map[UniqueID]storage.FieldData)
		tsData := make([]int64, nb)
		for i := 0; i < nb; i++ {
			tsData[i] = int64(i + 100)
		}
		data[tsFieldID] = &storage.Int64FieldData{
			NumRows: []int64{nb},
			Data:    tsData,
		}
		data[floatVectorFieldID] = &storage.FloatVectorFieldData{
			NumRows: []int64{nb},
			Data:    generateFloatVectors(),
			Dim:     dim,
		}
		insertData := storage.InsertData{
			Data: data,
			Infos: []storage.BlobInfo{
				{
					Length: 10,
				},
			},
		}
		binLogs, _, err := insertCodec.Serialize(999, 888, &insertData)
		assert.Nil(t, err)
		kvs := make(map[string][]byte, len(binLogs))
		paths := make([]string, 0, len(binLogs))
		for i, blob := range binLogs {
			key := path.Join(floatVectorBinlogPath, strconv.Itoa(i))
			paths = append(paths, key)
			kvs[key] = blob.Value[:]
		}
		err = in.chunkManager.MultiWrite(kvs)
		assert.Nil(t, err)

		indexMeta := &indexpb.IndexMeta{
			IndexBuildID: indexBuildID1,
			State:        commonpb.IndexState_InProgress,
			IndexVersion: 1,
		}

		value, err := proto.Marshal(indexMeta)
		assert.Nil(t, err)
		err = in.etcdKV.Save(metaPath1, string(value))
		assert.Nil(t, err)
		req := &indexpb.CreateIndexRequest{
			IndexBuildID: indexBuildID1,
			IndexName:    "FloatVector",
			IndexID:      indexID,
			Version:      1,
			MetaPath:     metaPath1,
			DataPaths:    paths,
			TypeParams: []*commonpb.KeyValuePair{
				{
					Key:   "dim",
					Value: "8",
				},
				{
					Key:   "dim",
					Value: "8",
				},
			},
			IndexParams: []*commonpb.KeyValuePair{
				{
					Key:   "index_type",
					Value: "IVF_SQ8",
				},
				{
					Key:   "params",
					Value: "{\"nlist\": 128}",
				},
				{
					Key:   "metric_type",
					Value: "L2",
				},
			},
		}

		status, err2 := in.CreateIndex(ctx, req)
		assert.Nil(t, err2)
		assert.Equal(t, commonpb.ErrorCode_Success, status.ErrorCode)

		strValue, err3 := in.etcdKV.Load(metaPath1)
		assert.Nil(t, err3)
		indexMetaTmp := indexpb.IndexMeta{}
		err = proto.Unmarshal([]byte(strValue), &indexMetaTmp)
		assert.Nil(t, err)
		for indexMetaTmp.State != commonpb.IndexState_Failed {
			time.Sleep(100 * time.Millisecond)
			strValue, err = in.etcdKV.Load(metaPath1)
			assert.Nil(t, err)
			err = proto.Unmarshal([]byte(strValue), &indexMetaTmp)
			assert.Nil(t, err)
		}
		defer in.chunkManager.MultiRemove(indexMetaTmp.IndexFilePaths)
		defer func() {
			for k := range kvs {
				err = in.chunkManager.Remove(k)
				assert.Nil(t, err)
			}
		}()
	})

	t.Run("Invalid Param", func(t *testing.T) {
		var insertCodec storage.InsertCodec

		insertCodec.Schema = &etcdpb.CollectionMeta{
			ID: collectionID,
			Schema: &schemapb.CollectionSchema{
				Fields: []*schemapb.FieldSchema{
					{
						FieldID:      floatVectorFieldID,
						Name:         floatVectorFieldName,
						IsPrimaryKey: false,
						DataType:     schemapb.DataType_FloatVector,
					},
				},
			},
		}
		data := make(map[UniqueID]storage.FieldData)
		tsData := make([]int64, nb)
		for i := 0; i < nb; i++ {
			tsData[i] = int64(i + 100)
		}
		data[tsFieldID] = &storage.Int64FieldData{
			NumRows: []int64{nb},
			Data:    tsData,
		}
		data[floatVectorFieldID] = &storage.FloatVectorFieldData{
			NumRows: []int64{nb},
			Data:    generateFloatVectors(),
			Dim:     dim,
		}
		insertData := storage.InsertData{
			Data: data,
			Infos: []storage.BlobInfo{
				{
					Length: 10,
				},
			},
		}
		binLogs, _, err := insertCodec.Serialize(999, 888, &insertData)
		assert.Nil(t, err)
		kvs := make(map[string][]byte, len(binLogs))
		paths := make([]string, 0, len(binLogs))
		for i, blob := range binLogs {
			key := path.Join(floatVectorBinlogPath, strconv.Itoa(i))
			paths = append(paths, key)
			kvs[key] = blob.Value[:]
		}
		err = in.chunkManager.MultiWrite(kvs)
		assert.Nil(t, err)

		indexMeta2 := &indexpb.IndexMeta{
			IndexBuildID: indexBuildID2,
			State:        commonpb.IndexState_InProgress,
			IndexVersion: 1,
		}

		value2, err := proto.Marshal(indexMeta2)
		assert.Nil(t, err)
		err = in.etcdKV.Save(metaPath2, string(value2))
		assert.Nil(t, err)

		req2 := &indexpb.CreateIndexRequest{
			IndexBuildID: indexBuildID2,
			IndexName:    "FloatVector",
			IndexID:      indexID,
			Version:      1,
			MetaPath:     metaPath2,
			DataPaths:    paths,
			TypeParams: []*commonpb.KeyValuePair{
				{
					Key:   "dim",
					Value: "8",
				},
				{
					Key:   "params",
					Value: "value",
				},
			},
			IndexParams: []*commonpb.KeyValuePair{
				{
					Key:   "index_type",
					Value: "IVF_SQ8",
				},
				{
					Key:   "params",
					Value: "{\"nlist\": 128}",
				},
				{
					Key:   "metric_type",
					Value: "L2",
				},
			},
		}

		status, err2 := in.CreateIndex(ctx, req2)
		assert.Nil(t, err2)
		assert.Equal(t, commonpb.ErrorCode_Success, status.ErrorCode)

		strValue, err3 := in.etcdKV.Load(metaPath2)
		assert.Nil(t, err3)
		indexMetaTmp := indexpb.IndexMeta{}
		err = proto.Unmarshal([]byte(strValue), &indexMetaTmp)
		assert.Nil(t, err)
		for indexMetaTmp.State != commonpb.IndexState_Failed {
			time.Sleep(100 * time.Millisecond)
			strValue, err = in.etcdKV.Load(metaPath2)
			assert.Nil(t, err)
			err = proto.Unmarshal([]byte(strValue), &indexMetaTmp)
			assert.Nil(t, err)
		}
		defer in.chunkManager.MultiRemove(indexMetaTmp.IndexFilePaths)
		defer func() {
			for k := range kvs {
				err = in.chunkManager.Remove(k)
				assert.Nil(t, err)
			}
		}()
	})

	t.Run("CreateIndex server not healthy", func(t *testing.T) {
		in.UpdateStateCode(internalpb.StateCode_Initializing)
		status, err := in.CreateIndex(ctx, &indexpb.CreateIndexRequest{})
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, status.ErrorCode)
	})

	err = in.etcdKV.RemoveWithPrefix("session/IndexNode")
	assert.Nil(t, err)

	err = in.Stop()
	assert.Nil(t, err)
}

func TestIndexNode_Error(t *testing.T) {
	ctx := context.Background()

	factory := dependency.NewDefaultFactory(true)
	in, err := NewIndexNode(ctx, factory)
	assert.Nil(t, err)
	Params.Init()

	etcdCli, err := etcd.GetEtcdClient(&Params.EtcdCfg)
	assert.NoError(t, err)
	in.SetEtcdClient(etcdCli)
	defer etcdCli.Close()

	err = in.Init()
	assert.Nil(t, err)

	err = in.Start()
	assert.Nil(t, err)

	err = in.Register()
	assert.Nil(t, err)

	in.UpdateStateCode(internalpb.StateCode_Initializing)

	in.chunkManager = storage.NewLocalChunkManager(storage.RootPath("/tmp/lib/milvus"))
	t.Run("CreateIndex", func(t *testing.T) {
		status, err := in.CreateIndex(ctx, &indexpb.CreateIndexRequest{})
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, status.ErrorCode)
	})

	t.Run("GetMetrics", func(t *testing.T) {
		resp, err := in.GetMetrics(ctx, &milvuspb.GetMetricsRequest{})
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, resp.Status.ErrorCode)
	})

	in.UpdateStateCode(internalpb.StateCode_Healthy)

	t.Run("Request Illegal", func(t *testing.T) {
		resp, err := in.GetMetrics(ctx, &milvuspb.GetMetricsRequest{})
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, resp.Status.ErrorCode)
	})

	t.Run("MetricsTypeIllegal", func(t *testing.T) {
		req, err := metricsinfo.ConstructRequestByMetricType("GetIndexNodeMetrics")
		assert.Nil(t, err)
		resp, err := in.GetMetrics(ctx, req)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, resp.Status.ErrorCode)
	})

	err = in.etcdKV.RemoveWithPrefix("session/IndexNode")
	assert.Nil(t, err)

	err = in.Stop()
	assert.Nil(t, err)
}

func TestIndexNode_InitError(t *testing.T) {
	ctx := context.Background()
	in := &IndexNode{
		sched: &TaskScheduler{
			IndexBuildQueue: &IndexBuildTaskQueue{
				BaseTaskQueue: BaseTaskQueue{
					unissuedTasks: list.New(),
					activeTasks:   make(map[UniqueID]task),
					maxTaskNum:    0,
					utBufChan:     make(chan int, 1024),
				},
			},
		},
	}
	in.UpdateStateCode(internalpb.StateCode_Healthy)

	t.Run("CreateIndex", func(t *testing.T) {
		status, err := in.CreateIndex(ctx, &indexpb.CreateIndexRequest{})
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, status.ErrorCode)
	})
}

func TestIndexNode_GetComponentStates(t *testing.T) {
	n := &IndexNode{}
	n.stateCode.Store(internalpb.StateCode_Healthy)
	resp, err := n.GetComponentStates(context.Background())
	assert.NoError(t, err)
	assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	assert.Equal(t, common.NotRegisteredID, resp.State.NodeID)
	n.session = &sessionutil.Session{}
	n.session.UpdateRegistered(true)
	resp, err = n.GetComponentStates(context.Background())
	assert.NoError(t, err)
	assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
}
