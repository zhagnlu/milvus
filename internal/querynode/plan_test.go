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

package querynode

import (
	"context"
	"math"
	"testing"

	"github.com/golang/protobuf/proto"
	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus/internal/common"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/planpb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
)

func TestPlan_Plan(t *testing.T) {
	collectionID := UniqueID(0)
	schema := genTestCollectionSchema()

	collection := newCollection(collectionID, schema)

	dslString := "{\"bool\": { \n\"vector\": {\n \"floatVectorField\": {\n \"metric_type\": \"L2\", \n \"params\": {\n \"nprobe\": 10 \n},\n \"query\": \"$0\",\n \"topk\": 10 \n,\"round_decimal\": 6\n } \n } \n } \n }"

	plan, err := createSearchPlan(collection, dslString)
	assert.NoError(t, err)
	assert.NotEqual(t, plan, nil)
	topk := plan.getTopK()
	assert.Equal(t, int(topk), 10)
	metricType := plan.getMetricType()
	assert.Equal(t, metricType, "L2")
	plan.delete()
	deleteCollection(collection)
}

func TestPlan_createSearchPlanByExpr(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	historical, err := genSimpleReplicaWithSealSegment(ctx)
	assert.NoError(t, err)

	col, err := historical.getCollectionByID(defaultCollectionID)
	assert.NoError(t, err)

	planNode := &planpb.PlanNode{
		OutputFieldIds: []FieldID{rowIDFieldID},
	}
	expr, err := proto.Marshal(planNode)
	assert.NoError(t, err)

	_, err = createSearchPlanByExpr(col, expr)
	assert.Error(t, err)
}

func TestPlan_NilCollection(t *testing.T) {
	collection := &Collection{
		id: defaultCollectionID,
	}

	_, err := createSearchPlan(collection, "")
	assert.Error(t, err)

	_, err = createSearchPlanByExpr(collection, nil)
	assert.Error(t, err)
}

func TestPlan_PlaceholderGroup(t *testing.T) {
	collectionID := UniqueID(0)
	schema := genTestCollectionSchema()
	collection := newCollection(collectionID, schema)

	dslString := "{\"bool\": { \n\"vector\": {\n \"floatVectorField\": {\n \"metric_type\": \"L2\", \n \"params\": {\n \"nprobe\": 10 \n},\n \"query\": \"$0\",\n \"topk\": 10 \n,\"round_decimal\": 6\n } \n } \n } \n }"
	plan, err := createSearchPlan(collection, dslString)
	assert.NoError(t, err)
	assert.NotNil(t, plan)

	var searchRawData1 []byte
	var searchRawData2 []byte
	var vec = generateFloatVectors(1, defaultDim)
	for i, ele := range vec {
		buf := make([]byte, 4)
		common.Endian.PutUint32(buf, math.Float32bits(ele+float32(i*2)))
		searchRawData1 = append(searchRawData1, buf...)
	}
	for i, ele := range vec {
		buf := make([]byte, 4)
		common.Endian.PutUint32(buf, math.Float32bits(ele+float32(i*4)))
		searchRawData2 = append(searchRawData2, buf...)
	}
	placeholderValue := commonpb.PlaceholderValue{
		Tag:    "$0",
		Type:   commonpb.PlaceholderType_FloatVector,
		Values: [][]byte{searchRawData1, searchRawData2},
	}

	placeholderGroup := commonpb.PlaceholderGroup{
		Placeholders: []*commonpb.PlaceholderValue{&placeholderValue},
	}

	placeGroupByte, err := proto.Marshal(&placeholderGroup)
	assert.Nil(t, err)
	holder, err := parseSearchRequest(plan, placeGroupByte)
	assert.NoError(t, err)
	assert.NotNil(t, holder)
	numQueries := holder.getNumOfQuery()
	assert.Equal(t, int(numQueries), 2)

	holder.delete()
	deleteCollection(collection)
}

func TestPlan_newSearchRequest(t *testing.T) {
	iReq, _ := genSearchRequest(defaultNQ, IndexHNSW, genTestCollectionSchema())
	collection := newCollection(defaultCollectionID, genTestCollectionSchema())
	req := &querypb.SearchRequest{
		Req:             iReq,
		DmlChannels:     []string{defaultDMLChannel},
		SegmentIDs:      []UniqueID{defaultSegmentID},
		FromShardLeader: true,
		Scope:           querypb.DataScope_Historical,
	}
	searchReq, err := newSearchRequest(collection, req, req.Req.GetPlaceholderGroup())
	assert.NoError(t, err)

	assert.Equal(t, simpleFloatVecField.id, searchReq.searchFieldID)
	assert.EqualValues(t, defaultNQ, searchReq.getNumOfQuery())

	searchReq.delete()
	deleteCollection(collection)
}
