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
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestHistorical_statistic(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	t.Run("test statistics", func(t *testing.T) {
		his, err := genSimpleReplicaWithSealSegment(ctx)
		assert.NoError(t, err)

		_, _, _, err = statisticHistorical(his, defaultCollectionID, nil, []UniqueID{defaultSegmentID})
		assert.NoError(t, err)
	})

	t.Run("test no collection - statistics partitions", func(t *testing.T) {
		his, err := genSimpleReplicaWithSealSegment(ctx)
		assert.NoError(t, err)

		err = his.removeCollection(defaultCollectionID)
		assert.NoError(t, err)

		_, _, _, err = statisticHistorical(his, defaultCollectionID, nil, nil)
		assert.Error(t, err)
	})

	t.Run("test no collection - statistics all collection", func(t *testing.T) {
		his, err := genSimpleReplicaWithSealSegment(ctx)
		assert.NoError(t, err)

		err = his.removeCollection(defaultCollectionID)
		assert.NoError(t, err)

		_, _, _, err = statisticHistorical(his, defaultCollectionID, []UniqueID{defaultPartitionID}, nil)
		assert.Error(t, err)
	})

	t.Run("test load partition and partition has been released", func(t *testing.T) {
		his, err := genSimpleReplicaWithSealSegment(ctx)
		assert.NoError(t, err)

		col, err := his.getCollectionByID(defaultCollectionID)
		assert.NoError(t, err)
		col.setLoadType(loadTypePartition)

		err = his.removePartition(defaultPartitionID)
		assert.NoError(t, err)

		_, _, _, err = statisticHistorical(his, defaultCollectionID, nil, nil)
		assert.Error(t, err)
	})

	t.Run("test no partition in collection", func(t *testing.T) {
		his, err := genSimpleReplicaWithSealSegment(ctx)
		assert.NoError(t, err)

		err = his.removePartition(defaultPartitionID)
		assert.NoError(t, err)

		res, _, ids, err := statisticHistorical(his, defaultCollectionID, nil, nil)
		assert.Equal(t, 0, len(res))
		assert.Equal(t, 0, len(ids))
		assert.NoError(t, err)
	})
}

func TestStreaming_statistics(t *testing.T) {
	t.Run("test statistics", func(t *testing.T) {
		streaming, err := genSimpleReplicaWithGrowingSegment()
		assert.NoError(t, err)

		res, _, _, err := statisticStreaming(streaming,
			defaultCollectionID,
			[]UniqueID{defaultPartitionID},
			defaultDMLChannel)
		assert.NoError(t, err)
		assert.Len(t, res, 1)
	})

	t.Run("test run empty partition", func(t *testing.T) {
		streaming, err := genSimpleReplicaWithGrowingSegment()
		assert.NoError(t, err)

		res, _, _, err := statisticStreaming(streaming,
			defaultCollectionID,
			[]UniqueID{defaultPartitionID},
			defaultDMLChannel)
		assert.NoError(t, err)
		assert.Len(t, res, 1)
	})

	t.Run("test run empty partition and loadCollection", func(t *testing.T) {
		streaming, err := genSimpleReplicaWithGrowingSegment()
		assert.NoError(t, err)

		col, err := streaming.getCollectionByID(defaultCollectionID)
		assert.NoError(t, err)
		col.setLoadType(loadTypeCollection)

		err = streaming.removePartition(defaultPartitionID)
		assert.NoError(t, err)

		res, _, _, err := statisticStreaming(streaming,
			defaultCollectionID,
			[]UniqueID{defaultPartitionID},
			defaultDMLChannel)

		assert.Error(t, err)
		assert.Equal(t, 0, len(res))
	})

	t.Run("test run empty partition and loadPartition", func(t *testing.T) {
		streaming, err := genSimpleReplicaWithGrowingSegment()
		assert.NoError(t, err)

		col, err := streaming.getCollectionByID(defaultCollectionID)
		assert.NoError(t, err)
		col.setLoadType(loadTypePartition)

		err = streaming.removePartition(defaultPartitionID)
		assert.NoError(t, err)

		_, _, _, err = statisticStreaming(streaming,
			defaultCollectionID,
			[]UniqueID{defaultPartitionID},
			defaultDMLChannel)
		assert.Error(t, err)
	})

	t.Run("test no partitions in collection", func(t *testing.T) {
		streaming, err := genSimpleReplicaWithGrowingSegment()
		assert.NoError(t, err)

		err = streaming.removePartition(defaultPartitionID)
		assert.NoError(t, err)

		res, _, _, err := statisticStreaming(streaming,
			defaultCollectionID,
			[]UniqueID{},
			defaultDMLChannel)
		assert.NoError(t, err)
		assert.Equal(t, 0, len(res))
	})
}
