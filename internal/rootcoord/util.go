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

package rootcoord

import (
	"encoding/json"
	"fmt"

	"github.com/milvus-io/milvus/internal/common"

	"github.com/milvus-io/milvus/internal/metastore/model"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus/internal/mq/msgstream"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

// EqualKeyPairArray check whether 2 KeyValuePairs are equal
func EqualKeyPairArray(p1 []*commonpb.KeyValuePair, p2 []*commonpb.KeyValuePair) bool {
	if len(p1) != len(p2) {
		return false
	}
	m1 := make(map[string]string)
	for _, p := range p1 {
		m1[p.Key] = p.Value
	}
	for _, p := range p2 {
		val, ok := m1[p.Key]
		if !ok {
			return false
		}
		if val != p.Value {
			return false
		}
	}
	return true
}

// GetFieldSchemaByID return field schema by id
func GetFieldSchemaByID(coll *model.Collection, fieldID typeutil.UniqueID) (*model.Field, error) {
	for _, f := range coll.Fields {
		if f.FieldID == fieldID {
			return f, nil
		}
	}
	return nil, fmt.Errorf("field id = %d not found", fieldID)
}

// GetFieldSchemaByIndexID return field schema by it's index id
func GetFieldSchemaByIndexID(coll *model.Collection, idxID typeutil.UniqueID) (*model.Field, error) {
	var fieldID typeutil.UniqueID
	exist := false
	for _, t := range coll.FieldIDToIndexID {
		if t.Value == idxID {
			fieldID = t.Key
			exist = true
			break
		}
	}
	if !exist {
		return nil, fmt.Errorf("index id = %d is not attach to any field", idxID)
	}
	return GetFieldSchemaByID(coll, fieldID)
}

// EncodeDdOperation serialize DdOperation into string
func EncodeDdOperation(m proto.Message, ddType string) (string, error) {
	mByte, err := proto.Marshal(m)
	if err != nil {
		return "", err
	}
	ddOp := DdOperation{
		Body: mByte,
		Type: ddType,
	}
	ddOpByte, err := json.Marshal(ddOp)
	if err != nil {
		return "", err
	}
	return string(ddOpByte), nil
}

// DecodeDdOperation deserialize string to DdOperation
func DecodeDdOperation(str string, ddOp *DdOperation) error {
	return json.Unmarshal([]byte(str), ddOp)
}

// EncodeMsgPositions serialize []*MsgPosition into string
func EncodeMsgPositions(msgPositions []*msgstream.MsgPosition) (string, error) {
	if len(msgPositions) == 0 {
		return "", nil
	}
	resByte, err := json.Marshal(msgPositions)
	if err != nil {
		return "", err
	}
	return string(resByte), nil
}

// DecodeMsgPositions deserialize string to []*MsgPosition
func DecodeMsgPositions(str string, msgPositions *[]*msgstream.MsgPosition) error {
	if str == "" || str == "null" {
		return nil
	}
	return json.Unmarshal([]byte(str), msgPositions)
}

func Int64TupleSliceToMap(s []common.Int64Tuple) map[int]common.Int64Tuple {
	ret := make(map[int]common.Int64Tuple, len(s))
	for i, e := range s {
		ret[i] = e
	}
	return ret
}

func Int64TupleMapToSlice(s map[int]common.Int64Tuple) []common.Int64Tuple {
	ret := make([]common.Int64Tuple, 0, len(s))
	for _, e := range s {
		ret = append(ret, e)
	}
	return ret
}
