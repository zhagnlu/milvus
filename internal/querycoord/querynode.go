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

package querycoord

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"

	nodeclient "github.com/milvus-io/milvus/internal/distributed/querynode/client"
	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/internal/util/metricsinfo"
)

// Node provides many interfaces to access querynode via grpc
type Node interface {
	start() error
	stop()
	getNodeInfo() (Node, error)

	releaseCollection(ctx context.Context, in *querypb.ReleaseCollectionRequest) error
	releasePartitions(ctx context.Context, in *querypb.ReleasePartitionsRequest) error

	watchDmChannels(ctx context.Context, in *querypb.WatchDmChannelsRequest) error
	watchDeltaChannels(ctx context.Context, in *querypb.WatchDeltaChannelsRequest) error
	syncReplicaSegments(ctx context.Context, in *querypb.SyncReplicaSegmentsRequest) error
	//removeDmChannel(collectionID UniqueID, channels []string) error

	hasWatchedDeltaChannel(collectionID UniqueID) bool

	setState(state nodeState)
	getState() nodeState
	isOnline() bool
	isOffline() bool

	getSegmentInfo(ctx context.Context, in *querypb.GetSegmentInfoRequest) (*querypb.GetSegmentInfoResponse, error)
	loadSegments(ctx context.Context, in *querypb.LoadSegmentsRequest) error
	releaseSegments(ctx context.Context, in *querypb.ReleaseSegmentsRequest) error
	getComponentInfo(ctx context.Context) *internalpb.ComponentInfo

	getMetrics(ctx context.Context, in *milvuspb.GetMetricsRequest) (*milvuspb.GetMetricsResponse, error)
}

type queryNode struct {
	ctx      context.Context
	cancel   context.CancelFunc
	id       int64
	address  string
	client   types.QueryNode
	kvClient *etcdkv.EtcdKV

	sync.RWMutex
	watchedDeltaChannels map[UniqueID][]*datapb.VchannelInfo
	state                nodeState
	stateLock            sync.RWMutex

	totalMem     uint64
	memUsage     uint64
	memUsageRate float64
	cpuUsage     float64
}

func newQueryNode(ctx context.Context, address string, id UniqueID, kv *etcdkv.EtcdKV) (Node, error) {
	watchedDeltaChannels := make(map[UniqueID][]*datapb.VchannelInfo)
	childCtx, cancel := context.WithCancel(ctx)
	client, err := nodeclient.NewClient(childCtx, address)
	if err != nil {
		cancel()
		return nil, err
	}
	node := &queryNode{
		ctx:                  childCtx,
		cancel:               cancel,
		id:                   id,
		address:              address,
		client:               client,
		kvClient:             kv,
		watchedDeltaChannels: watchedDeltaChannels,
		state:                disConnect,
	}

	return node, nil
}

func (qn *queryNode) start() error {
	if err := qn.client.Init(); err != nil {
		log.Error("start: init queryNode client failed", zap.Int64("nodeID", qn.id), zap.String("error", err.Error()))
		return err
	}
	if err := qn.client.Start(); err != nil {
		log.Error("start: start queryNode client failed", zap.Int64("nodeID", qn.id), zap.String("error", err.Error()))
		return err
	}

	qn.stateLock.Lock()
	if qn.state < online {
		qn.state = online
	}
	qn.stateLock.Unlock()
	log.Info("start: queryNode client start success", zap.Int64("nodeID", qn.id), zap.String("address", qn.address))
	return nil
}

func (qn *queryNode) stop() {
	//qn.stateLock.Lock()
	//defer qn.stateLock.Unlock()
	//qn.state = offline
	if qn.client != nil {
		qn.client.Stop()
	}
	qn.cancel()
}

func (qn *queryNode) hasWatchedDeltaChannel(collectionID UniqueID) bool {
	qn.RLock()
	defer qn.RUnlock()

	_, ok := qn.watchedDeltaChannels[collectionID]
	return ok
}

func (qn *queryNode) setDeltaChannelInfo(collectionID int64, infos []*datapb.VchannelInfo) {
	qn.Lock()
	defer qn.Unlock()

	qn.watchedDeltaChannels[collectionID] = infos
}

func (qn *queryNode) setState(state nodeState) {
	qn.stateLock.Lock()
	defer qn.stateLock.Unlock()

	qn.state = state
}

func (qn *queryNode) getState() nodeState {
	qn.stateLock.RLock()
	defer qn.stateLock.RUnlock()

	return qn.state
}

func (qn *queryNode) isOnline() bool {
	qn.stateLock.RLock()
	defer qn.stateLock.RUnlock()

	return qn.state == online
}

func (qn *queryNode) isOffline() bool {
	qn.stateLock.RLock()
	defer qn.stateLock.RUnlock()

	return qn.state == offline
}

//***********************grpc req*************************//
func (qn *queryNode) watchDmChannels(ctx context.Context, in *querypb.WatchDmChannelsRequest) error {
	if !qn.isOnline() {
		return errors.New("WatchDmChannels: queryNode is offline")
	}

	status, err := qn.client.WatchDmChannels(qn.ctx, in)
	if err != nil {
		return err
	}
	if status.ErrorCode != commonpb.ErrorCode_Success {
		return errors.New(status.Reason)
	}

	return nil
}

func (qn *queryNode) watchDeltaChannels(ctx context.Context, in *querypb.WatchDeltaChannelsRequest) error {
	if !qn.isOnline() {
		return errors.New("WatchDeltaChannels: queryNode is offline")
	}

	status, err := qn.client.WatchDeltaChannels(qn.ctx, in)
	if err != nil {
		return err
	}
	if status.ErrorCode != commonpb.ErrorCode_Success {
		return errors.New(status.Reason)
	}
	qn.setDeltaChannelInfo(in.CollectionID, in.Infos)
	return err
}

func (qn *queryNode) releaseCollection(ctx context.Context, in *querypb.ReleaseCollectionRequest) error {
	if !qn.isOnline() {
		log.Warn("ReleaseCollection: the QueryNode has been offline, the release request is no longer needed", zap.Int64("nodeID", qn.id))
		return nil
	}

	status, err := qn.client.ReleaseCollection(qn.ctx, in)
	if err != nil {
		return err
	}
	if status.ErrorCode != commonpb.ErrorCode_Success {
		return errors.New(status.Reason)
	}

	qn.Lock()
	delete(qn.watchedDeltaChannels, in.CollectionID)
	qn.Unlock()

	return nil
}

func (qn *queryNode) releasePartitions(ctx context.Context, in *querypb.ReleasePartitionsRequest) error {
	if !qn.isOnline() {
		return nil
	}

	status, err := qn.client.ReleasePartitions(qn.ctx, in)
	if err != nil {
		return err
	}
	if status.ErrorCode != commonpb.ErrorCode_Success {
		return errors.New(status.Reason)
	}

	return nil
}

func (qn *queryNode) getSegmentInfo(ctx context.Context, in *querypb.GetSegmentInfoRequest) (*querypb.GetSegmentInfoResponse, error) {
	if !qn.isOnline() {
		return nil, fmt.Errorf("getSegmentInfo: queryNode %d is offline", qn.id)
	}

	res, err := qn.client.GetSegmentInfo(qn.ctx, in)
	if err != nil {
		return nil, err
	}
	if res.Status.ErrorCode != commonpb.ErrorCode_Success {
		return nil, errors.New(res.Status.Reason)
	}

	return res, nil
}

func (qn *queryNode) getComponentInfo(ctx context.Context) *internalpb.ComponentInfo {
	if !qn.isOnline() {
		return &internalpb.ComponentInfo{
			NodeID:    qn.id,
			StateCode: internalpb.StateCode_Abnormal,
		}
	}

	res, err := qn.client.GetComponentStates(qn.ctx)
	if err != nil || res.Status.ErrorCode != commonpb.ErrorCode_Success {
		return &internalpb.ComponentInfo{
			NodeID:    qn.id,
			StateCode: internalpb.StateCode_Abnormal,
		}
	}

	return res.State
}

func (qn *queryNode) getMetrics(ctx context.Context, in *milvuspb.GetMetricsRequest) (*milvuspb.GetMetricsResponse, error) {
	if !qn.isOnline() {
		return nil, fmt.Errorf("getMetrics: queryNode %d is offline", qn.id)
	}

	return qn.client.GetMetrics(qn.ctx, in)
}

func (qn *queryNode) loadSegments(ctx context.Context, in *querypb.LoadSegmentsRequest) error {
	if !qn.isOnline() {
		return errors.New("LoadSegments: queryNode is offline")
	}

	status, err := qn.client.LoadSegments(qn.ctx, in)
	if err != nil {
		return err
	}
	if status.ErrorCode != commonpb.ErrorCode_Success {
		return errors.New(status.Reason)
	}

	return nil
}

func (qn *queryNode) releaseSegments(ctx context.Context, in *querypb.ReleaseSegmentsRequest) error {
	if !qn.isOnline() {
		return errors.New("ReleaseSegments: queryNode is offline")
	}

	status, err := qn.client.ReleaseSegments(qn.ctx, in)
	if err != nil {
		return err
	}
	if status.ErrorCode != commonpb.ErrorCode_Success {
		return errors.New(status.Reason)
	}

	return nil
}

func (qn *queryNode) getNodeInfo() (Node, error) {
	qn.Lock()
	defer qn.Unlock()

	if !qn.isOnline() {
		return nil, errors.New("getNodeInfo: queryNode is offline")
	}

	req, err := metricsinfo.ConstructRequestByMetricType(metricsinfo.SystemInfoMetrics)
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(qn.ctx, time.Second)
	defer cancel()
	resp, err := qn.client.GetMetrics(ctx, req)
	if err != nil {
		return nil, err
	}
	if resp.Status.ErrorCode != commonpb.ErrorCode_Success {
		return nil, errors.New(resp.Status.Reason)
	}
	infos := metricsinfo.QueryNodeInfos{}
	err = metricsinfo.UnmarshalComponentInfos(resp.Response, &infos)
	if err != nil {
		return nil, err
	}
	qn.cpuUsage = infos.HardwareInfos.CPUCoreUsage
	qn.totalMem = infos.HardwareInfos.Memory
	qn.memUsage = infos.HardwareInfos.MemoryUsage
	qn.memUsageRate = float64(qn.memUsage) / float64(qn.totalMem)
	return &queryNode{
		id:      qn.id,
		address: qn.address,
		state:   qn.getState(),

		totalMem:     qn.totalMem,
		memUsage:     qn.memUsage,
		memUsageRate: qn.memUsageRate,
		cpuUsage:     qn.cpuUsage,
	}, nil
}

func (qn *queryNode) syncReplicaSegments(ctx context.Context, in *querypb.SyncReplicaSegmentsRequest) error {
	if !qn.isOnline() {
		return errors.New("SyncSegments: queryNode is offline")
	}

	status, err := qn.client.SyncReplicaSegments(ctx, in)
	if err != nil {
		return err
	}
	if status.ErrorCode != commonpb.ErrorCode_Success {
		return errors.New(status.Reason)
	}

	return nil
}
