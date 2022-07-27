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

package indexcoord

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"testing"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"

	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/common"
	grpcindexnode "github.com/milvus-io/milvus/internal/distributed/indexnode"
	"github.com/milvus-io/milvus/internal/indexnode"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/indexpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	"github.com/milvus-io/milvus/internal/proto/schemapb"
	"github.com/milvus-io/milvus/internal/util/dependency"
	"github.com/milvus-io/milvus/internal/util/etcd"
	"github.com/milvus-io/milvus/internal/util/metricsinfo"
	"github.com/milvus-io/milvus/internal/util/sessionutil"
)

func TestIndexCoord(t *testing.T) {
	ctx := context.Background()
	inm0 := &indexnode.Mock{}
	Params.Init()
	etcdCli, err := etcd.GetEtcdClient(&Params.EtcdCfg)
	assert.NoError(t, err)
	inm0.SetEtcdClient(etcdCli)
	err = inm0.Init()
	assert.Nil(t, err)
	err = inm0.Register()
	assert.Nil(t, err)
	err = inm0.Start()
	assert.Nil(t, err)
	factory := dependency.NewDefaultFactory(true)
	ic, err := NewIndexCoord(ctx, factory)
	assert.Nil(t, err)
	ic.reqTimeoutInterval = time.Second * 10

	dcm := &DataCoordMock{
		Err:  false,
		Fail: false,
	}
	err = ic.SetDataCoord(dcm)
	assert.Nil(t, err)

	ic.SetEtcdClient(etcdCli)
	err = ic.Init()
	assert.Nil(t, err)

	ccm := &ChunkManagerMock{
		Err:  false,
		Fail: false,
	}
	ic.chunkManager = ccm

	err = ic.Register()
	assert.Nil(t, err)
	err = ic.Start()
	assert.Nil(t, err)

	err = inm0.Stop()
	assert.Nil(t, err)

	t.Run("create index without indexnodes", func(t *testing.T) {
		indexID := int64(rand.Int())
		req := &indexpb.BuildIndexRequest{
			IndexID:   indexID,
			DataPaths: []string{"NoIndexNode-1", "NoIndexNode-2"},
			NumRows:   10,
			TypeParams: []*commonpb.KeyValuePair{
				{
					Key:   "dim",
					Value: "128",
				},
			},
			FieldSchema: &schemapb.FieldSchema{
				DataType: schemapb.DataType_FloatVector,
			},
		}
		resp, err := ic.BuildIndex(ctx, req)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
		time.Sleep(time.Second)
		status, err := ic.DropIndex(ctx, &indexpb.DropIndexRequest{
			IndexID: indexID,
		})
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, status.ErrorCode)
	})

	in, err := grpcindexnode.NewServer(ctx, factory)
	assert.Nil(t, err)
	assert.NotNil(t, in)
	inm := &indexnode.Mock{
		Build:   true,
		Failure: false,
	}

	inm.SetEtcdClient(etcdCli)
	err = in.SetClient(inm)
	assert.Nil(t, err)

	err = in.Run()
	assert.Nil(t, err)

	state, err := ic.GetComponentStates(ctx)
	assert.Nil(t, err)
	assert.Equal(t, internalpb.StateCode_Healthy, state.State.StateCode)

	indexID := int64(rand.Int())

	var indexBuildID UniqueID

	t.Run("Create Index", func(t *testing.T) {
		req := &indexpb.BuildIndexRequest{
			IndexID:   indexID,
			DataPaths: []string{"DataPath-1", "DataPath-2"},
			NumRows:   0,
			TypeParams: []*commonpb.KeyValuePair{
				{
					Key:   "dim",
					Value: "128",
				},
			},
			FieldSchema: &schemapb.FieldSchema{
				DataType: schemapb.DataType_FloatVector,
			},
		}
		resp, err := ic.BuildIndex(ctx, req)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
		indexBuildID = resp.IndexBuildID

		resp2, err := ic.BuildIndex(ctx, req)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
		assert.Equal(t, indexBuildID, resp2.IndexBuildID)
		assert.Equal(t, "already have same index", resp2.Status.Reason)

		req2 := &indexpb.BuildIndexRequest{
			IndexID:   indexID,
			DataPaths: []string{"DataPath-3", "DataPath-4"},
			NumRows:   1000,
			TypeParams: []*commonpb.KeyValuePair{
				{
					Key:   "dim",
					Value: "128",
				},
			},
			FieldSchema: &schemapb.FieldSchema{
				DataType: schemapb.DataType_FloatVector,
			},
		}
		resp3, err := ic.BuildIndex(ctx, req2)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp3.Status.ErrorCode)
	})

	t.Run("Get Index State", func(t *testing.T) {
		req := &indexpb.GetIndexStatesRequest{
			IndexBuildIDs: []UniqueID{indexBuildID},
		}
		for {
			resp, err := ic.GetIndexStates(ctx, req)
			assert.Nil(t, err)
			assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
			if resp.States[0].State == commonpb.IndexState_Finished {
				break
			}
			time.Sleep(500 * time.Millisecond)
		}
	})

	t.Run("Get IndexFile Paths", func(t *testing.T) {
		req := &indexpb.GetIndexFilePathsRequest{
			IndexBuildIDs: []UniqueID{indexBuildID},
		}
		resp, err := ic.GetIndexFilePaths(ctx, req)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
		assert.Equal(t, 1, len(resp.FilePaths))
		assert.Equal(t, 2, len(resp.FilePaths[0].IndexFilePaths))
		assert.Equal(t, "IndexFilePath-1", resp.FilePaths[0].IndexFilePaths[0])
		assert.Equal(t, "IndexFilePath-2", resp.FilePaths[0].IndexFilePaths[1])
	})

	t.Run("Drop Index", func(t *testing.T) {
		req := &indexpb.DropIndexRequest{
			IndexID: indexID,
		}
		resp, err := ic.DropIndex(ctx, req)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.ErrorCode)
	})

	t.Run("GetMetrics, system info", func(t *testing.T) {
		req, err := metricsinfo.ConstructRequestByMetricType(metricsinfo.SystemInfoMetrics)
		assert.Nil(t, err)
		resp, err := ic.GetMetrics(ctx, req)
		assert.Nil(t, err)
		log.Info("GetMetrics, system info",
			zap.String("name", resp.ComponentName),
			zap.String("resp", resp.Response))
	})

	t.Run("GetTimeTickChannel", func(t *testing.T) {
		resp, err := ic.GetTimeTickChannel(ctx)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	})

	t.Run("GetStatisticsChannel", func(t *testing.T) {
		resp, err := ic.GetStatisticsChannel(ctx)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
	})

	t.Run("GetMetrics when indexcoord is not healthy", func(t *testing.T) {
		ic.UpdateStateCode(internalpb.StateCode_Abnormal)
		req, err := metricsinfo.ConstructRequestByMetricType(metricsinfo.SystemInfoMetrics)
		assert.Nil(t, err)
		resp, err := ic.GetMetrics(ctx, req)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, resp.Status.ErrorCode)
		ic.UpdateStateCode(internalpb.StateCode_Healthy)
	})

	t.Run("GetMetrics when request is illegal", func(t *testing.T) {
		req, err := metricsinfo.ConstructRequestByMetricType("GetIndexNodeMetrics")
		assert.Nil(t, err)
		resp, err := ic.GetMetrics(ctx, req)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, resp.Status.ErrorCode)
	})

	t.Run("GetMetrics request without metricType", func(t *testing.T) {
		req := &milvuspb.GetMetricsRequest{
			Request: "GetIndexCoordMetrics",
		}
		resp, err := ic.GetMetrics(ctx, req)
		assert.Nil(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, resp.Status.ErrorCode)
	})

	err = in.Stop()
	assert.Nil(t, err)
	err = ic.Stop()
	assert.Nil(t, err)
}

func TestIndexCoord_watchNodeLoop(t *testing.T) {
	ech := make(chan *sessionutil.SessionEvent)
	in := &IndexCoord{
		loopWg:    sync.WaitGroup{},
		loopCtx:   context.Background(),
		eventChan: ech,
		session: &sessionutil.Session{
			TriggerKill: true,
			ServerID:    0,
		},
	}
	in.loopWg.Add(1)

	flag := false
	closed := false
	sigDone := make(chan struct{}, 1)
	sigQuit := make(chan struct{}, 1)
	sc := make(chan os.Signal, 1)
	signal.Notify(sc, syscall.SIGINT)
	defer signal.Reset(syscall.SIGINT)

	go func() {
		in.watchNodeLoop()
		flag = true
		sigDone <- struct{}{}
	}()
	go func() {
		<-sc
		closed = true
		sigQuit <- struct{}{}
	}()

	close(ech)
	<-sigDone
	<-sigQuit
	assert.True(t, flag)
	assert.True(t, closed)
}

func TestIndexCoord_watchMetaLoop(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	ic := &IndexCoord{
		loopCtx: ctx,
		loopWg:  sync.WaitGroup{},
	}

	watchChan := make(chan clientv3.WatchResponse, 1024)

	client := &mockETCDKV{
		watchWithRevision: func(s string, i int64) clientv3.WatchChan {
			return watchChan
		},
	}
	mt := &metaTable{
		client:            client,
		indexBuildID2Meta: map[UniqueID]*Meta{},
		etcdRevision:      0,
		lock:              sync.RWMutex{},
	}
	ic.metaTable = mt

	t.Run("watch chan panic", func(t *testing.T) {
		ic.loopWg.Add(1)
		watchChan <- clientv3.WatchResponse{Canceled: true}

		assert.Panics(t, func() {
			ic.watchMetaLoop()
		})
		ic.loopWg.Wait()
	})

	t.Run("watch chan new meta table panic", func(t *testing.T) {
		client = &mockETCDKV{
			watchWithRevision: func(s string, i int64) clientv3.WatchChan {
				return watchChan
			},
			loadWithRevisionAndVersions: func(s string) ([]string, []string, []int64, int64, error) {
				return []string{}, []string{}, []int64{}, 0, fmt.Errorf("error occurred")
			},
		}
		mt = &metaTable{
			client:            client,
			indexBuildID2Meta: map[UniqueID]*Meta{},
			etcdRevision:      0,
			lock:              sync.RWMutex{},
		}
		ic.metaTable = mt
		ic.loopWg.Add(1)
		watchChan <- clientv3.WatchResponse{CompactRevision: 10}
		assert.Panics(t, func() {
			ic.watchMetaLoop()
		})
		ic.loopWg.Wait()
	})

	t.Run("watch chan new meta success", func(t *testing.T) {
		ic.loopWg = sync.WaitGroup{}
		client = &mockETCDKV{
			watchWithRevision: func(s string, i int64) clientv3.WatchChan {
				return watchChan
			},
			loadWithRevisionAndVersions: func(s string) ([]string, []string, []int64, int64, error) {
				return []string{}, []string{}, []int64{}, 0, nil
			},
		}
		mt = &metaTable{
			client:            client,
			indexBuildID2Meta: map[UniqueID]*Meta{},
			etcdRevision:      0,
			lock:              sync.RWMutex{},
		}
		ic.metaTable = mt
		ic.loopWg.Add(1)
		watchChan <- clientv3.WatchResponse{CompactRevision: 10}
		go ic.watchMetaLoop()
		cancel()
		ic.loopWg.Wait()
	})
}

func TestIndexCoord_GetComponentStates(t *testing.T) {
	n := &IndexCoord{}
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

func TestIndexCoord_NotHealthy(t *testing.T) {
	ic := &IndexCoord{}
	ic.stateCode.Store(internalpb.StateCode_Abnormal)
	req := &indexpb.BuildIndexRequest{}
	resp, err := ic.BuildIndex(context.Background(), req)
	assert.Error(t, err)
	assert.Equal(t, commonpb.ErrorCode_UnexpectedError, resp.Status.ErrorCode)

	req2 := &indexpb.DropIndexRequest{}
	status, err := ic.DropIndex(context.Background(), req2)
	assert.Nil(t, err)
	assert.Equal(t, commonpb.ErrorCode_UnexpectedError, status.ErrorCode)

	req3 := &indexpb.GetIndexStatesRequest{}
	resp2, err := ic.GetIndexStates(context.Background(), req3)
	assert.Nil(t, err)
	assert.Equal(t, commonpb.ErrorCode_UnexpectedError, resp2.Status.ErrorCode)

	req4 := &indexpb.GetIndexFilePathsRequest{
		IndexBuildIDs: []UniqueID{1, 2},
	}
	resp4, err := ic.GetIndexFilePaths(context.Background(), req4)
	assert.Nil(t, err)
	assert.Equal(t, commonpb.ErrorCode_UnexpectedError, resp4.Status.ErrorCode)

	req5 := &indexpb.RemoveIndexRequest{}
	resp5, err := ic.RemoveIndex(context.Background(), req5)
	assert.Nil(t, err)
	assert.Equal(t, commonpb.ErrorCode_UnexpectedError, resp5.GetErrorCode())
}

func TestIndexCoord_GetIndexFilePaths(t *testing.T) {
	ic := &IndexCoord{
		metaTable: &metaTable{
			indexBuildID2Meta: map[UniqueID]*Meta{
				1: {
					indexMeta: &indexpb.IndexMeta{
						IndexBuildID:   1,
						State:          commonpb.IndexState_Finished,
						IndexFilePaths: []string{"indexFiles-1", "indexFiles-2"},
					},
				},
				2: {
					indexMeta: &indexpb.IndexMeta{
						IndexBuildID: 2,
						State:        commonpb.IndexState_Failed,
					},
				},
			},
		},
	}

	ic.stateCode.Store(internalpb.StateCode_Healthy)

	t.Run("GetIndexFilePaths success", func(t *testing.T) {
		resp, err := ic.GetIndexFilePaths(context.Background(), &indexpb.GetIndexFilePathsRequest{IndexBuildIDs: []UniqueID{1}})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
		assert.Equal(t, 1, len(resp.FilePaths))
		assert.ElementsMatch(t, resp.FilePaths[0].IndexFilePaths, []string{"indexFiles-1", "indexFiles-2"})
	})

	t.Run("GetIndexFilePaths failed", func(t *testing.T) {
		resp, err := ic.GetIndexFilePaths(context.Background(), &indexpb.GetIndexFilePathsRequest{IndexBuildIDs: []UniqueID{2}})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.Status.ErrorCode)
		assert.Equal(t, 0, len(resp.FilePaths[0].IndexFilePaths))
	})

	t.Run("set DataCoord with nil", func(t *testing.T) {
		err := ic.SetDataCoord(nil)
		assert.Error(t, err)
	})
}

func Test_tryAcquireSegmentReferLock(t *testing.T) {
	ic := &IndexCoord{
		session: &sessionutil.Session{
			ServerID: 1,
		},
	}
	dcm := &DataCoordMock{
		Err:  false,
		Fail: false,
	}
	cmm := &ChunkManagerMock{
		Err:  false,
		Fail: false,
	}

	ic.dataCoordClient = dcm
	ic.chunkManager = cmm

	t.Run("success", func(t *testing.T) {
		err := ic.tryAcquireSegmentReferLock(context.Background(), 1, 1, []UniqueID{1})
		assert.Nil(t, err)
	})

	t.Run("error", func(t *testing.T) {
		dcmE := &DataCoordMock{
			Err:  true,
			Fail: false,
		}
		ic.dataCoordClient = dcmE
		err := ic.tryAcquireSegmentReferLock(context.Background(), 1, 1, []UniqueID{1})
		assert.Error(t, err)
	})

	t.Run("Fail", func(t *testing.T) {
		dcmF := &DataCoordMock{
			Err:  false,
			Fail: true,
		}
		ic.dataCoordClient = dcmF
		err := ic.tryAcquireSegmentReferLock(context.Background(), 1, 1, []UniqueID{1})
		assert.Error(t, err)
	})
}

func Test_tryReleaseSegmentReferLock(t *testing.T) {
	ic := &IndexCoord{
		session: &sessionutil.Session{
			ServerID: 1,
		},
	}
	dcm := &DataCoordMock{
		Err:  false,
		Fail: false,
	}

	ic.dataCoordClient = dcm

	t.Run("success", func(t *testing.T) {
		err := ic.tryReleaseSegmentReferLock(context.Background(), 1, 1)
		assert.NoError(t, err)
	})
}

func TestIndexCoord_RemoveIndex(t *testing.T) {
	ic := &IndexCoord{
		metaTable: &metaTable{},
		indexBuilder: &indexBuilder{
			notify: make(chan struct{}, 10),
		},
	}
	ic.stateCode.Store(internalpb.StateCode_Healthy)
	status, err := ic.RemoveIndex(context.Background(), &indexpb.RemoveIndexRequest{BuildIDs: []UniqueID{0}})
	assert.Nil(t, err)
	assert.Equal(t, commonpb.ErrorCode_Success, status.GetErrorCode())
}
