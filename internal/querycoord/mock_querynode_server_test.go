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
	"net"
	"strconv"
	"sync"
	"time"

	"go.uber.org/zap"
	"google.golang.org/grpc"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/util/etcd"
	"github.com/milvus-io/milvus/internal/util/funcutil"
	"github.com/milvus-io/milvus/internal/util/metricsinfo"
	"github.com/milvus-io/milvus/internal/util/retry"
	"github.com/milvus-io/milvus/internal/util/sessionutil"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

const (
	defaultTotalmemPerNode = 6000000
)

var (
	GlobalSegmentInfos  = make(map[UniqueID]*querypb.SegmentInfo)
	globalSegInfosMutex sync.RWMutex
)

type rpcHandler func() (*commonpb.Status, error)

type queryNodeServerMock struct {
	querypb.QueryNodeServer
	ctx         context.Context
	cancel      context.CancelFunc
	session     *sessionutil.Session
	grpcErrChan chan error
	grpcServer  *grpc.Server

	queryNodeIP   string
	queryNodePort int64
	queryNodeID   int64

	rwmutex             sync.RWMutex // guard for all modification
	watchDmChannels     rpcHandler
	watchDeltaChannels  rpcHandler
	loadSegment         rpcHandler
	releaseCollection   rpcHandler
	releasePartition    rpcHandler
	releaseSegments     rpcHandler
	syncReplicaSegments rpcHandler
	getSegmentInfos     func() (*querypb.GetSegmentInfoResponse, error)
	getMetrics          func() (*milvuspb.GetMetricsResponse, error)

	segmentInfos map[UniqueID]*querypb.SegmentInfo

	totalMem uint64
}

func newQueryNodeServerMock(ctx context.Context) *queryNodeServerMock {
	ctx1, cancel := context.WithCancel(ctx)
	return &queryNodeServerMock{
		ctx:         ctx1,
		cancel:      cancel,
		grpcErrChan: make(chan error),

		rwmutex:             sync.RWMutex{},
		watchDmChannels:     returnSuccessResult,
		watchDeltaChannels:  returnSuccessResult,
		loadSegment:         returnSuccessResult,
		releaseCollection:   returnSuccessResult,
		releasePartition:    returnSuccessResult,
		releaseSegments:     returnSuccessResult,
		syncReplicaSegments: returnSuccessResult,
		getSegmentInfos:     returnSuccessGetSegmentInfoResult,
		getMetrics:          returnSuccessGetMetricsResult,

		segmentInfos: GlobalSegmentInfos,

		totalMem: defaultTotalmemPerNode,
	}
}

func (qs *queryNodeServerMock) setRPCInterface(interfacePointer *rpcHandler, newhandler rpcHandler) {
	qs.rwmutex.Lock()
	defer qs.rwmutex.Unlock()
	*interfacePointer = newhandler
}

func (qs *queryNodeServerMock) Register() error {
	log.Debug("query node session info", zap.String("metaPath", Params.EtcdCfg.MetaRootPath))
	etcdCli, err := etcd.GetEtcdClient(&Params.EtcdCfg)
	if err != nil {
		return err
	}
	qs.session = sessionutil.NewSession(qs.ctx, Params.EtcdCfg.MetaRootPath, etcdCli)
	qs.session.Init(typeutil.QueryNodeRole, qs.queryNodeIP+":"+strconv.FormatInt(qs.queryNodePort, 10), false, false)
	qs.queryNodeID = qs.session.ServerID
	log.Debug("query nodeID", zap.Int64("nodeID", qs.queryNodeID))
	log.Debug("query node address", zap.String("address", qs.session.Address))
	qs.session.Register()

	return nil
}

func (qs *queryNodeServerMock) init() error {
	qs.queryNodeIP = funcutil.GetLocalIP()
	grpcPort := Params.QueryCoordCfg.Port

	go func() {
		var lis net.Listener
		var err error
		err = retry.Do(qs.ctx, func() error {
			addr := ":" + strconv.Itoa(grpcPort)
			lis, err = net.Listen("tcp", addr)
			if err == nil {
				qs.queryNodePort = int64(lis.Addr().(*net.TCPAddr).Port)
			} else {
				// set port=0 to get next available port
				grpcPort = 0
			}
			return err
		}, retry.Attempts(2))
		if err != nil {
			qs.grpcErrChan <- err
		}

		qs.grpcServer = grpc.NewServer()
		querypb.RegisterQueryNodeServer(qs.grpcServer, qs)
		go funcutil.CheckGrpcReady(qs.ctx, qs.grpcErrChan)
		if err = qs.grpcServer.Serve(lis); err != nil {
			qs.grpcErrChan <- err
		}
	}()

	err := <-qs.grpcErrChan
	if err != nil {
		return err
	}

	if err := qs.Register(); err != nil {
		return err
	}

	return nil
}

func (qs *queryNodeServerMock) start() error {
	return nil
}

func (qs *queryNodeServerMock) stop() error {
	qs.cancel()
	if qs.session != nil {
		qs.session.Revoke(time.Second)
	}
	if qs.grpcServer != nil {
		qs.grpcServer.GracefulStop()
	}

	return nil
}

func (qs *queryNodeServerMock) run() error {
	if err := qs.init(); err != nil {
		return err
	}

	if err := qs.start(); err != nil {
		return err
	}

	return nil
}

func (qs *queryNodeServerMock) GetComponentStates(ctx context.Context, req *internalpb.GetComponentStatesRequest) (*internalpb.ComponentStates, error) {
	return &internalpb.ComponentStates{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		},
	}, nil
}

func (qs *queryNodeServerMock) WatchDmChannels(ctx context.Context, req *querypb.WatchDmChannelsRequest) (*commonpb.Status, error) {
	return qs.watchDmChannels()
}

func (qs *queryNodeServerMock) WatchDeltaChannels(ctx context.Context, req *querypb.WatchDeltaChannelsRequest) (*commonpb.Status, error) {
	return qs.watchDeltaChannels()
}

func (qs *queryNodeServerMock) LoadSegments(ctx context.Context, req *querypb.LoadSegmentsRequest) (*commonpb.Status, error) {
	sizePerRecord, err := typeutil.EstimateSizePerRecord(req.Schema)
	if err != nil {
		return returnFailedResult()
	}
	for _, info := range req.Infos {
		segmentInfo := &querypb.SegmentInfo{
			SegmentID:    info.SegmentID,
			PartitionID:  info.PartitionID,
			CollectionID: info.CollectionID,
			NodeID:       qs.queryNodeID,
			SegmentState: commonpb.SegmentState_Sealed,
			MemSize:      info.NumOfRows * int64(sizePerRecord),
			NumRows:      info.NumOfRows,
			NodeIds:      []UniqueID{qs.queryNodeID},
		}
		globalSegInfosMutex.Lock()
		qs.segmentInfos[info.SegmentID] = segmentInfo
		globalSegInfosMutex.Unlock()
	}

	return qs.loadSegment()
}

func (qs *queryNodeServerMock) ReleaseCollection(ctx context.Context, req *querypb.ReleaseCollectionRequest) (*commonpb.Status, error) {
	qs.rwmutex.RLock()
	defer qs.rwmutex.RUnlock()
	return qs.releaseCollection()
}

func (qs *queryNodeServerMock) ReleasePartitions(ctx context.Context, req *querypb.ReleasePartitionsRequest) (*commonpb.Status, error) {
	return qs.releasePartition()
}

func (qs *queryNodeServerMock) ReleaseSegments(ctx context.Context, req *querypb.ReleaseSegmentsRequest) (*commonpb.Status, error) {
	return qs.releaseSegments()
}

func (qs *queryNodeServerMock) GetSegmentInfo(ctx context.Context, req *querypb.GetSegmentInfoRequest) (*querypb.GetSegmentInfoResponse, error) {
	segmentInfos := make([]*querypb.SegmentInfo, 0)
	globalSegInfosMutex.RLock()
	for _, info := range qs.segmentInfos {
		if info.CollectionID == req.CollectionID && info.NodeID == qs.queryNodeID {
			segmentInfos = append(segmentInfos, info)
		}
	}
	globalSegInfosMutex.RUnlock()

	res, err := qs.getSegmentInfos()
	if err == nil {
		res.Infos = segmentInfos
	}
	return res, err
}

func (qs *queryNodeServerMock) SyncReplicaSegments(ctx context.Context, req *querypb.SyncReplicaSegmentsRequest) (*commonpb.Status, error) {
	return qs.syncReplicaSegments()
}

func (qs *queryNodeServerMock) GetMetrics(ctx context.Context, req *milvuspb.GetMetricsRequest) (*milvuspb.GetMetricsResponse, error) {
	response, err := qs.getMetrics()
	if err != nil {
		return nil, err
	}

	// check whether the memory usage has been set
	if len(response.Response) > 0 {
		return response, nil
	}

	if response.Status.ErrorCode != commonpb.ErrorCode_Success {
		return nil, errors.New("query node do task failed")
	}

	totalMemUsage := uint64(0)
	globalSegInfosMutex.RLock()
	for _, info := range qs.segmentInfos {
		if nodeIncluded(qs.queryNodeID, info.NodeIds) {
			totalMemUsage += uint64(info.MemSize)
		}
	}
	globalSegInfosMutex.RUnlock()
	nodeInfos := metricsinfo.QueryNodeInfos{
		BaseComponentInfos: metricsinfo.BaseComponentInfos{
			Name: metricsinfo.ConstructComponentName(typeutil.QueryNodeRole, qs.queryNodeID),
			HardwareInfos: metricsinfo.HardwareMetrics{
				IP:          qs.queryNodeIP,
				Memory:      qs.totalMem,
				MemoryUsage: totalMemUsage,
			},
			Type: typeutil.QueryNodeRole,
			ID:   qs.queryNodeID,
		},
	}
	resp, err := metricsinfo.MarshalComponentInfos(nodeInfos)
	if err != nil {
		response.Status.ErrorCode = commonpb.ErrorCode_UnexpectedError
		response.Status.Reason = err.Error()
		return response, err
	}
	response.Response = resp
	return response, nil
}

func startQueryNodeServer(ctx context.Context) (*queryNodeServerMock, error) {
	node := newQueryNodeServerMock(ctx)
	err := node.run()
	if err != nil {
		return nil, err
	}

	return node, nil
}

func returnSuccessResult() (*commonpb.Status, error) {
	return &commonpb.Status{
		ErrorCode: commonpb.ErrorCode_Success,
	}, nil
}

func returnFailedResult() (*commonpb.Status, error) {
	return &commonpb.Status{
		ErrorCode: commonpb.ErrorCode_UnexpectedError,
	}, errors.New("query node do task failed")
}

func returnSuccessGetSegmentInfoResult() (*querypb.GetSegmentInfoResponse, error) {
	return &querypb.GetSegmentInfoResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		},
	}, nil
}

func returnFailedGetSegmentInfoResult() (*querypb.GetSegmentInfoResponse, error) {
	return &querypb.GetSegmentInfoResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
		},
	}, errors.New("query node do task failed")
}

func returnSuccessGetMetricsResult() (*milvuspb.GetMetricsResponse, error) {
	return &milvuspb.GetMetricsResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		},
	}, nil
}

func returnFailedGetMetricsResult() (*milvuspb.GetMetricsResponse, error) {
	return &milvuspb.GetMetricsResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
		},
	}, errors.New("query node do task failed")
}
