package proxy

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/milvus-io/milvus/internal/common"

	"github.com/golang/protobuf/proto"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/proto/schemapb"
	"github.com/milvus-io/milvus/internal/types"

	"github.com/milvus-io/milvus/internal/util/funcutil"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

func TestQueryTask_all(t *testing.T) {
	Params.Init()

	var (
		err error
		ctx = context.TODO()

		rc = NewRootCoordMock()
		qc = NewQueryCoordMock(withValidShardLeaders())
		qn = &QueryNodeMock{}

		shardsNum      = int32(2)
		collectionName = t.Name() + funcutil.GenRandomStr()

		expr   = fmt.Sprintf("%s > 0", testInt64Field)
		hitNum = 10

		errPolicy = func(context.Context, *shardClientMgr, func(context.Context, int64, types.QueryNode, []string) error, map[string][]nodeInfo) error {
			return fmt.Errorf("fake error")
		}
	)

	mockCreator := func(ctx context.Context, address string) (types.QueryNode, error) {
		return qn, nil
	}

	mgr := newShardClientMgr(withShardClientCreator(mockCreator))

	rc.Start()
	defer rc.Stop()
	qc.Start()
	defer qc.Stop()

	err = InitMetaCache(rc, qc, mgr)
	assert.NoError(t, err)

	fieldName2Types := map[string]schemapb.DataType{
		testBoolField:     schemapb.DataType_Bool,
		testInt32Field:    schemapb.DataType_Int32,
		testInt64Field:    schemapb.DataType_Int64,
		testFloatField:    schemapb.DataType_Float,
		testDoubleField:   schemapb.DataType_Double,
		testFloatVecField: schemapb.DataType_FloatVector,
	}
	if enableMultipleVectorFields {
		fieldName2Types[testBinaryVecField] = schemapb.DataType_BinaryVector
	}

	schema := constructCollectionSchemaByDataType(collectionName, fieldName2Types, testInt64Field, false)
	marshaledSchema, err := proto.Marshal(schema)
	assert.NoError(t, err)

	createColT := &createCollectionTask{
		Condition: NewTaskCondition(ctx),
		CreateCollectionRequest: &milvuspb.CreateCollectionRequest{
			CollectionName: collectionName,
			Schema:         marshaledSchema,
			ShardsNum:      shardsNum,
		},
		ctx:       ctx,
		rootCoord: rc,
	}

	require.NoError(t, createColT.OnEnqueue())
	require.NoError(t, createColT.PreExecute(ctx))
	require.NoError(t, createColT.Execute(ctx))
	require.NoError(t, createColT.PostExecute(ctx))

	collectionID, err := globalMetaCache.GetCollectionID(ctx, collectionName)
	assert.NoError(t, err)

	status, err := qc.LoadCollection(ctx, &querypb.LoadCollectionRequest{
		Base: &commonpb.MsgBase{
			MsgType:  commonpb.MsgType_LoadCollection,
			SourceID: Params.ProxyCfg.GetNodeID(),
		},
		CollectionID: collectionID,
	})
	require.NoError(t, err)
	require.Equal(t, commonpb.ErrorCode_Success, status.ErrorCode)

	// test begins
	task := &queryTask{
		Condition: NewTaskCondition(ctx),
		RetrieveRequest: &internalpb.RetrieveRequest{
			Base: &commonpb.MsgBase{
				MsgType:  commonpb.MsgType_Retrieve,
				SourceID: Params.ProxyCfg.GetNodeID(),
			},
			CollectionID:   collectionID,
			OutputFieldsId: make([]int64, len(fieldName2Types)),
		},
		ctx: ctx,
		result: &milvuspb.QueryResults{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_Success,
			},
		},
		request: &milvuspb.QueryRequest{
			Base: &commonpb.MsgBase{
				MsgType:  commonpb.MsgType_Retrieve,
				SourceID: Params.ProxyCfg.GetNodeID(),
			},
			CollectionName: collectionName,
			Expr:           expr,
		},
		qc:       qc,
		shardMgr: mgr,
	}

	assert.NoError(t, task.OnEnqueue())

	// test query task with timeout
	ctx1, cancel1 := context.WithTimeout(ctx, 10*time.Second)
	defer cancel1()
	// before preExecute
	assert.Equal(t, typeutil.ZeroTimestamp, task.TimeoutTimestamp)
	task.ctx = ctx1
	assert.NoError(t, task.PreExecute(ctx))
	// after preExecute
	assert.Greater(t, task.TimeoutTimestamp, typeutil.ZeroTimestamp)

	task.ctx = ctx
	task.queryShardPolicy = errPolicy
	assert.Error(t, task.Execute(ctx))

	task.queryShardPolicy = mergeRoundRobinPolicy
	result1 := &internalpb.RetrieveResults{
		Base: &commonpb.MsgBase{MsgType: commonpb.MsgType_RetrieveResult},
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		},
		Ids: &schemapb.IDs{
			IdField: &schemapb.IDs_IntId{
				IntId: &schemapb.LongArray{Data: generateInt64Array(hitNum)},
			},
		},
	}

	outputFieldIDs := make([]UniqueID, 0, len(fieldName2Types))
	for i := 0; i < len(fieldName2Types); i++ {
		outputFieldIDs = append(outputFieldIDs, int64(common.StartOfUserFieldID+i))
	}
	task.RetrieveRequest.OutputFieldsId = outputFieldIDs
	for fieldName, dataType := range fieldName2Types {
		result1.FieldsData = append(result1.FieldsData, generateFieldData(dataType, fieldName, hitNum))
	}

	task.ctx = ctx
	qn.queryError = fmt.Errorf("mock error")
	assert.Error(t, task.Execute(ctx))

	qn.queryError = nil
	qn.withQueryResult = &internalpb.RetrieveResults{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_NotShardLeader,
		},
	}
	assert.Equal(t, task.Execute(ctx), errInvalidShardLeaders)

	qn.withQueryResult = &internalpb.RetrieveResults{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
		},
	}
	assert.Error(t, task.Execute(ctx))

	qn.withQueryResult = result1

	assert.NoError(t, task.Execute(ctx))

	assert.NoError(t, task.PostExecute(ctx))
}
