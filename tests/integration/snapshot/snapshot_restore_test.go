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

package snapshot

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
	"go.uber.org/zap"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/pkg/v2/common"
	"github.com/milvus-io/milvus/pkg/v2/log"
	"github.com/milvus-io/milvus/pkg/v2/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v2/util/merr"
	"github.com/milvus-io/milvus/pkg/v2/util/metric"
	"github.com/milvus-io/milvus/tests/integration"
)

type SnapshotRestoreSuite struct {
	integration.MiniClusterSuite
}

func TestSnapshotRestore(t *testing.T) {
	suite.Run(t, new(SnapshotRestoreSuite))
}

// TestSnapshotRestoreWithDynamicFields verifies the full snapshot restore chain
// for a collection with dynamic fields (enable_dynamic_field=true).
//
// This is a regression test for https://github.com/milvus-io/milvus/issues/48579
// where json_key_index buildID remapping caused the LOON manifest to reference
// stale buildIDs, resulting in QueryNode 404 errors and LoadCollection hanging.
func (s *SnapshotRestoreSuite) TestSnapshotRestoreWithDynamicFields() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	c := s.Cluster

	const (
		dim    = 128
		rowNum = 3000
	)

	collectionName := "snap_restore_" + funcutil.GenRandomStr()
	snapshotName := "snap_" + funcutil.GenRandomStr()
	restoredCollName := "restored_" + collectionName

	// Step 1: Create collection WITH dynamic fields enabled
	schema := &schemapb.CollectionSchema{
		Name:               collectionName,
		EnableDynamicField: true,
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "id", IsPrimaryKey: true, DataType: schemapb.DataType_Int64},
			{FieldID: 101, Name: "vector", DataType: schemapb.DataType_FloatVector,
				TypeParams: []*commonpb.KeyValuePair{{Key: common.DimKey, Value: fmt.Sprintf("%d", dim)}}},
		},
	}
	marshaledSchema, err := proto.Marshal(schema)
	s.NoError(err)

	createStatus, err := c.MilvusClient.CreateCollection(ctx, &milvuspb.CreateCollectionRequest{
		CollectionName: collectionName,
		Schema:         marshaledSchema,
		ShardsNum:      2,
	})
	s.NoError(err)
	s.NoError(merr.Error(createStatus))
	log.Info("collection created", zap.String("name", collectionName))

	// Step 2: Create index on vector field
	createIdxStatus, err := c.MilvusClient.CreateIndex(ctx, &milvuspb.CreateIndexRequest{
		CollectionName: collectionName,
		FieldName:      "vector",
		IndexName:      "_default",
		ExtraParams:    integration.ConstructIndexParam(dim, integration.IndexFaissIvfFlat, metric.L2),
	})
	s.NoError(err)
	s.NoError(merr.Error(createIdxStatus))
	s.WaitForIndexBuilt(ctx, collectionName, "vector")

	// Step 3: Insert data
	fVecColumn := integration.NewFloatVectorFieldData("vector", rowNum, dim)
	hashKeys := integration.GenerateHashKeys(rowNum)
	insertResult, err := c.MilvusClient.Insert(ctx, &milvuspb.InsertRequest{
		CollectionName: collectionName,
		FieldsData:     []*schemapb.FieldData{fVecColumn},
		HashKeys:       hashKeys,
		NumRows:        uint32(rowNum),
	})
	s.NoError(err)
	s.NoError(merr.Error(insertResult.GetStatus()))
	log.Info("data inserted", zap.Int("rows", rowNum))

	// Step 4: Flush
	flushResp, err := c.MilvusClient.Flush(ctx, &milvuspb.FlushRequest{
		CollectionNames: []string{collectionName},
	})
	s.NoError(err)
	segmentIDs := flushResp.GetCollSegIDs()[collectionName]
	s.NotEmpty(segmentIDs.GetData())
	flushTs := flushResp.GetCollFlushTs()[collectionName]
	s.WaitForFlush(ctx, segmentIDs.GetData(), flushTs, "", collectionName)
	log.Info("flush completed")

	// Step 5: Create snapshot
	createSnapStatus, err := c.MilvusClient.CreateSnapshot(ctx, &milvuspb.CreateSnapshotRequest{
		CollectionName: collectionName,
		Name:           snapshotName,
		Description:    "test snapshot for restore",
	})
	s.NoError(err)
	s.NoError(merr.Error(createSnapStatus))
	log.Info("snapshot created", zap.String("name", snapshotName))

	// Step 6: Restore snapshot to new collection
	restoreResp, err := c.MilvusClient.RestoreSnapshot(ctx, &milvuspb.RestoreSnapshotRequest{
		Name:           snapshotName,
		CollectionName: restoredCollName,
	})
	s.NoError(err)
	s.NoError(merr.Error(restoreResp.GetStatus()))
	jobID := restoreResp.GetJobId()
	log.Info("restore initiated", zap.Int64("jobID", jobID))

	// Step 7: Wait for restore to complete
	s.Eventually(func() bool {
		stateResp, err := c.MilvusClient.GetRestoreSnapshotState(ctx, &milvuspb.GetRestoreSnapshotStateRequest{
			JobId: jobID,
		})
		if err != nil {
			return false
		}
		state := stateResp.GetInfo().GetState()
		log.Info("restore state", zap.Any("state", state), zap.Int32("progress", stateResp.GetInfo().GetProgress()))
		return state == milvuspb.RestoreSnapshotState_RestoreSnapshotCompleted
	}, 2*time.Minute, 1*time.Second, "restore should complete within 2 minutes")

	// Step 8: Load restored collection — this is where the bug manifested
	// (loadProgress stuck at 90% due to json_stats 404)
	loadStatus, err := c.MilvusClient.LoadCollection(ctx, &milvuspb.LoadCollectionRequest{
		CollectionName: restoredCollName,
	})
	s.NoError(err)
	s.NoError(merr.Error(loadStatus))
	s.WaitForLoad(ctx, restoredCollName)
	log.Info("restored collection loaded successfully")

	// Step 9: Query restored collection to verify data
	queryResult, err := c.MilvusClient.Query(ctx, &milvuspb.QueryRequest{
		CollectionName: restoredCollName,
		Expr:           "",
		OutputFields:   []string{"count(*)"},
	})
	s.NoError(err)
	s.NoError(merr.Error(queryResult.GetStatus()))
	s.NotEmpty(queryResult.GetFieldsData())
	log.Info("query on restored collection succeeded",
		zap.Any("result", queryResult.GetFieldsData()))

	// Step 10: Cleanup
	c.MilvusClient.DropCollection(ctx, &milvuspb.DropCollectionRequest{CollectionName: restoredCollName})
	c.MilvusClient.DropSnapshot(ctx, &milvuspb.DropSnapshotRequest{Name: snapshotName})
	c.MilvusClient.DropCollection(ctx, &milvuspb.DropCollectionRequest{CollectionName: collectionName})
}
