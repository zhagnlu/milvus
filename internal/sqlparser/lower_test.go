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

package sqlparser

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

func TestLowerAggregatePartialLayout(t *testing.T) {
	bound := bindSQL(t, "SELECT v, avg(pk), count(DISTINCT data->>'x'), count(DISTINCT data->>'y'), max(pk) FROM events GROUP BY v")
	plan, err := LowerAggregate(bound, bindingSchema(), "UTC")
	require.NoError(t, err)
	require.Len(t, plan.GroupIDs, 3)
	require.Len(t, plan.Project.Columns, 4)
	require.Len(t, plan.Aggregates, 3)
	require.Equal(t, planpb.AggregateOp_sum, plan.Aggregates[0].Op)
	require.Equal(t, planpb.AggregateOp_count, plan.Aggregates[1].Op)
	require.Equal(t, planpb.AggregateOp_max, plan.Aggregates[2].Op)
	require.Equal(t, plan.Aggregates[0].FieldId, plan.Aggregates[2].FieldId)
	require.Equal(t, []int{3, 4}, plan.Inputs[0].Columns)
	require.Equal(t, 1, plan.Inputs[1].DistinctGroup)
	require.Equal(t, 2, plan.Inputs[2].DistinctGroup)
	require.Equal(t, []int{5}, plan.Inputs[3].Columns)
	for _, column := range plan.Project.Columns {
		require.Greater(t, column.Id, int64(102))
	}
}

func TestLowerJSONBench(t *testing.T) {
	for name, sql := range jsonBenchQueries {
		t.Run(name, func(t *testing.T) {
			plan, err := LowerAggregate(bindSQL(t, sql), bindingSchema(), "UTC")
			require.NoError(t, err)
			require.NotEmpty(t, plan.Project.Columns)
			if name == "Q3" {
				require.Equal(t, "extract", plan.Project.Columns[1].Expression.Op)
				require.Equal(t, "hour", plan.Project.Columns[1].Expression.TimeUnit)
			}
		})
	}
}

func TestCountStarRequiresSQLProjection(t *testing.T) {
	plan, err := LowerAggregate(bindSQL(t, "SELECT count(*) FROM events WHERE pk > 1"), bindingSchema(), "UTC")
	require.NoError(t, err)
	require.NotNil(t, plan.Project.Predicate)
	require.Len(t, plan.Project.Columns, 1)
	require.Greater(t, plan.Aggregates[0].FieldId, int64(102))
	require.Equal(t, plan.Project.Columns[0].Id, plan.Aggregates[0].FieldId)
	require.Equal(t, int64(1), plan.Project.Columns[0].Expression.Literal.GetInt64Val())
}
