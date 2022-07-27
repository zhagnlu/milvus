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

package datanode

import (
	"github.com/milvus-io/milvus/internal/mq/msgstream"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/util/flowgraph"
)

type (
	// Msg is flowgraph.Msg
	Msg = flowgraph.Msg

	// MsgStreamMsg is flowgraph.MsgStreamMsg
	MsgStreamMsg = flowgraph.MsgStreamMsg
)

type flowGraphMsg struct {
	insertMessages []*msgstream.InsertMsg
	deleteMessages []*msgstream.DeleteMsg
	timeRange      TimeRange
	startPositions []*internalpb.MsgPosition
	endPositions   []*internalpb.MsgPosition
	//segmentsToFlush is the signal used by insertBufferNode to notify deleteNode to flush
	segmentsToFlush []UniqueID
	dropCollection  bool
	dropPartitions  []UniqueID
}

func (fgMsg *flowGraphMsg) TimeTick() Timestamp {
	return fgMsg.timeRange.timestampMax
}

// flush Msg is used in flowgraph insertBufferNode to flush the given segment
type flushMsg struct {
	msgID        UniqueID
	timestamp    Timestamp
	segmentID    UniqueID
	collectionID UniqueID
	flushed      bool
}

type resendTTMsg struct {
	msgID      UniqueID
	segmentIDs []UniqueID
}
