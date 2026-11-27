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

package sqlquery

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/sqlparser"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

func testField(t *testing.T, id int64, typ schemapb.DataType, values ...any) *schemapb.FieldData {
	t.Helper()
	field, err := typeutil.GenEmptyFieldData(&schemapb.FieldSchema{FieldID: id, DataType: typ, Nullable: true})
	require.NoError(t, err)
	for _, value := range values {
		require.NoError(t, appendValue(field, value))
	}
	return field
}

func TestAggregateAcrossShards(t *testing.T) {
	paramtable.Init()
	bound := testBind(t, "SELECT v, count(DISTINCT data->>'x') AS unique_x, avg(pk) AS mean, count(*) AS n FROM events GROUP BY v ORDER BY n DESC LIMIT 1")
	plan, err := sqlparser.LowerAggregate(bound, testSchema(), "UTC")
	require.NoError(t, err)
	// Physical keys are (v, x), states are (sum(pk), count(pk), count(*)).
	// Neither shard's local winner is the global winner v=1. The same x=a
	// occurs on both shards and must count once in COUNT DISTINCT.
	partial := func(groups []any, xs []any, sums, counts []any) *internalpb.RetrieveResults {
		return &internalpb.RetrieveResults{FieldsData: []*schemapb.FieldData{
			testField(t, plan.GroupIDs[0], schemapb.DataType_Int64, groups...),
			testField(t, plan.GroupIDs[1], schemapb.DataType_VarChar, xs...),
			testField(t, plan.Aggregates[0].FieldId, schemapb.DataType_Int64, sums...),
			testField(t, plan.Aggregates[1].FieldId, schemapb.DataType_Int64, counts...),
			testField(t, plan.Aggregates[2].FieldId, schemapb.DataType_Int64, counts...),
		}}
	}
	result, err := ReduceAggregate(t.Context(), plan, testSchema(), []*internalpb.RetrieveResults{
		partial([]any{int64(1), int64(2)}, []any{"a", "x"}, []any{int64(40), int64(90)}, []any{int64(4), int64(9)}),
		partial([]any{int64(1), int64(1), int64(3)}, []any{"a", "b", "z"}, []any{int64(100), int64(40), int64(160)}, []any{int64(5), int64(2), int64(8)}),
	}, 1<<20)
	require.NoError(t, err)
	require.Equal(t, []int64{1}, result.FieldsData[0].GetScalars().GetLongData().Data)
	require.Equal(t, []int64{2}, result.FieldsData[1].GetScalars().GetLongData().Data)
	require.InDelta(t, 180.0/11, result.FieldsData[2].GetScalars().GetDoubleData().Data[0], 1e-12)
	require.Equal(t, []int64{11}, result.FieldsData[3].GetScalars().GetLongData().Data)
}

func TestAggregateEmptyInput(t *testing.T) {
	paramtable.Init()
	bound := testBind(t, "SELECT count(*), count(DISTINCT v), avg(v), min(v) FROM events")
	plan, err := sqlparser.LowerAggregate(bound, testSchema(), "UTC")
	require.NoError(t, err)
	result, err := ReduceAggregate(t.Context(), plan, testSchema(), nil, 1<<20)
	require.NoError(t, err)
	require.Len(t, result.FieldsData, 4)
	for _, field := range result.FieldsData[:2] {
		require.Equal(t, []int64{0}, field.GetScalars().GetLongData().Data)
	}
	for _, field := range result.FieldsData[2:] {
		require.Equal(t, []bool{false}, typeutil.GetFieldDataValidData(field))
	}
}

func TestProjectFilterSortSliceAndNull(t *testing.T) {
	bound := testBind(t, "SELECT pk AS id, v + 1 AS next FROM events WHERE pk > 1 ORDER BY v DESC NULLS LAST OFFSET 1 LIMIT 2")
	input := &milvuspb.QueryResults{FieldsData: []*schemapb.FieldData{
		testField(t, 100, schemapb.DataType_Int64, int64(1), int64(2), int64(3), int64(4)),
		testField(t, 101, schemapb.DataType_Int64, int64(100), nil, int64(2), int64(4)),
	}}
	result, err := Project(t.Context(), bound, input, "UTC", 1<<20)
	require.NoError(t, err)
	require.Equal(t, []string{"id", "next"}, result.OutputFields)
	require.Equal(t, []int64{3, 2}, result.FieldsData[0].GetScalars().GetLongData().Data)
	require.Equal(t, int64(3), result.FieldsData[1].GetScalars().GetLongData().Data[0])
	require.Equal(t, []bool{true, false}, typeutil.GetFieldDataValidData(result.FieldsData[1]))
	_, err = Project(t.Context(), bound, input, "UTC", 1)
	require.Error(t, err)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err = Project(ctx, bound, input, "UTC", 1<<20)
	require.ErrorIs(t, err, context.Canceled)
}

func TestGlobalTimestampSpan(t *testing.T) {
	paramtable.Init()
	bound := testBind(t, "SELECT v, extract(epoch FROM (max(to_timestamp(pk)) - min(to_timestamp(pk)))) * 1000 AS span FROM events GROUP BY v ORDER BY span DESC LIMIT 1")
	plan, err := sqlparser.LowerAggregate(bound, testSchema(), "UTC")
	require.NoError(t, err)
	partial := func(maximum, minimum int64) *internalpb.RetrieveResults {
		return &internalpb.RetrieveResults{FieldsData: []*schemapb.FieldData{
			testField(t, plan.GroupIDs[0], schemapb.DataType_Int64, int64(1)),
			testField(t, plan.Aggregates[0].FieldId, schemapb.DataType_Timestamptz, maximum),
			testField(t, plan.Aggregates[1].FieldId, schemapb.DataType_Timestamptz, minimum),
		}}
	}
	result, err := ReduceAggregate(t.Context(), plan, testSchema(), []*internalpb.RetrieveResults{
		partial(11_000_000, 10_000_000), partial(102_000_000, 100_000_000),
	}, 1<<20)
	require.NoError(t, err)
	require.Equal(t, []float64{92000}, result.FieldsData[1].GetScalars().GetDoubleData().Data)
}

func TestAggregateTimestampGroupAndProjection(t *testing.T) {
	paramtable.Init()
	bound := testBind(t, "SELECT to_timestamp(pk) AS instant, count(*) AS n FROM events GROUP BY instant ORDER BY instant")
	plan, err := sqlparser.LowerAggregate(bound, testSchema(), "UTC")
	require.NoError(t, err)
	partial := &internalpb.RetrieveResults{FieldsData: []*schemapb.FieldData{
		testField(t, plan.GroupIDs[0], schemapb.DataType_Timestamptz, int64(2_000_000), int64(1_000_000)),
		testField(t, plan.Aggregates[0].FieldId, schemapb.DataType_Int64, int64(3), int64(2)),
	}}
	result, err := ReduceAggregate(t.Context(), plan, testSchema(), []*internalpb.RetrieveResults{partial, partial}, 1<<20)
	require.NoError(t, err)
	require.Equal(t, []int64{1_000_000, 2_000_000}, result.FieldsData[0].GetScalars().GetTimestamptzData().Data)
	require.Equal(t, []int64{4, 6}, result.FieldsData[1].GetScalars().GetLongData().Data)
}

func TestProjectNullableJSONAndCompactVector(t *testing.T) {
	schema := testSchema()
	schema.Fields = append(schema.Fields, &schemapb.FieldSchema{FieldID: 103, Name: "vec", DataType: schemapb.DataType_FloatVector, Nullable: true})
	components, err := sqlparser.ExtractSqlComponents("SELECT data->>'x' AS x, vec FROM events WHERE pk > 1 ORDER BY pk DESC")
	require.NoError(t, err)
	bound, err := sqlparser.Bind(components, schema)
	require.NoError(t, err)
	input := &milvuspb.QueryResults{FieldsData: []*schemapb.FieldData{
		testField(t, 100, schemapb.DataType_Int64, int64(1), int64(2), int64(3)),
		testField(t, 102, schemapb.DataType_JSON, []byte(`{"x":"a"}`), nil, []byte(`{"x":"c"}`)),
		{FieldId: 103, Type: schemapb.DataType_FloatVector, ValidData: []bool{false, true, true}, Field: &schemapb.FieldData_Vectors{Vectors: &schemapb.VectorField{Dim: 2, Data: &schemapb.VectorField_FloatVector{FloatVector: &schemapb.FloatArray{Data: []float32{2, 20, 3, 30}}}}}},
	}}
	result, err := Project(t.Context(), bound, input, "UTC", 1<<20)
	require.NoError(t, err)
	require.Equal(t, []bool{true, false}, typeutil.GetFieldDataValidData(result.FieldsData[0]))
	require.Equal(t, "c", result.FieldsData[0].GetScalars().GetStringData().Data[0])
	require.Equal(t, []float32{3, 30, 2, 20}, result.FieldsData[1].GetVectors().GetFloatVector().Data)
	// Search can compact scalar JSON as well as vectors.
	input.FieldsData[1].GetScalars().GetJsonData().Data = [][]byte{[]byte(`{"x":"a"}`), []byte(`{"x":"c"}`)}
	compact, err := Project(t.Context(), bound, input, "UTC", 1<<20)
	require.NoError(t, err)
	require.Equal(t, result.String(), compact.String())
}

func TestProjectFormattedSearchTimestamp(t *testing.T) {
	schema := testSchema()
	schema.Fields = append(schema.Fields, &schemapb.FieldSchema{FieldID: 103, Name: "created", DataType: schemapb.DataType_Timestamptz})
	components, err := sqlparser.ExtractSqlComponents("SELECT extract(hour FROM created) AS hour FROM events")
	require.NoError(t, err)
	bound, err := sqlparser.Bind(components, schema)
	require.NoError(t, err)
	field := testField(t, 103, schemapb.DataType_VarChar, "2024-01-01T08:00:00+08:00")
	field.Type = schemapb.DataType_Timestamptz
	result, err := Project(t.Context(), bound, &milvuspb.QueryResults{FieldsData: []*schemapb.FieldData{field}}, "UTC", 1<<20)
	require.NoError(t, err)
	require.Equal(t, []float64{0}, result.FieldsData[0].GetScalars().GetDoubleData().Data)
	require.NotNil(t, field.GetScalars().GetStringData(), "normalization must not mutate the Search response")
}

func TestSQLTopKOffsetAndOverflow(t *testing.T) {
	input := &milvuspb.QueryResults{FieldsData: []*schemapb.FieldData{
		testField(t, 100, schemapb.DataType_Int64, int64(0), int64(1), int64(2), int64(3), int64(4), int64(5), int64(6), int64(7)),
		testField(t, 101, schemapb.DataType_Int64, int64(2), nil, int64(1), int64(2), nil, int64(1), int64(2), int64(1)),
	}}
	for _, order := range []string{"v ASC NULLS FIRST", "v DESC NULLS LAST", "v ASC NULLS LAST, pk DESC", "v DESC NULLS FIRST, pk ASC"} {
		query := "SELECT pk FROM events ORDER BY " + order
		full, err := Project(t.Context(), testBind(t, query), input, "UTC", 1<<20)
		require.NoError(t, err)
		all := full.FieldsData[0].GetScalars().GetLongData().Data
		for _, tc := range []struct {
			clause     string
			start, end int
		}{
			{"LIMIT 3", 0, 3}, {"OFFSET 2 LIMIT 3", 2, 5}, {"OFFSET 7 LIMIT 3", 7, 8},
			{"OFFSET 1 LIMIT 9223372036854775807", 1, 8}, {"OFFSET 2", 2, 8},
			{"OFFSET 20 LIMIT 3", 0, 0}, {"LIMIT 0", 0, 0},
		} {
			got, err := Project(t.Context(), testBind(t, query+" "+tc.clause), input, "UTC", 1<<20)
			require.NoError(t, err, order+" "+tc.clause)
			if tc.start == tc.end {
				if len(got.FieldsData) > 0 {
					require.Empty(t, got.FieldsData[0].GetScalars().GetLongData().Data)
				}
			} else {
				require.Equal(t, all[tc.start:tc.end], got.FieldsData[0].GetScalars().GetLongData().Data, order+" "+tc.clause)
			}
		}
	}
}

func TestDistinctCollapseMultipleKeysAndNulls(t *testing.T) {
	paramtable.Init()
	bound := testBind(t, "SELECT v, count(DISTINCT data->>'x') AS nx, count(DISTINCT data->>'y') AS ny, count(*) AS n FROM events GROUP BY v ORDER BY v NULLS FIRST")
	plan, err := sqlparser.LowerAggregate(bound, testSchema(), "UTC")
	require.NoError(t, err)
	partial := func() *internalpb.RetrieveResults {
		return &internalpb.RetrieveResults{FieldsData: []*schemapb.FieldData{
			testField(t, plan.GroupIDs[0], schemapb.DataType_Int64, nil, int64(1), int64(1), int64(1)),
			testField(t, plan.GroupIDs[1], schemapb.DataType_VarChar, "", "a\x00b", "a", "a\x00b"),
			testField(t, plan.GroupIDs[2], schemapb.DataType_VarChar, nil, "c", "\x00bc", nil),
			testField(t, plan.Aggregates[0].FieldId, schemapb.DataType_Int64, int64(1), int64(2), int64(3), int64(4)),
		}}
	}
	for i := 0; i < 2; i++ {
		result, err := ReduceAggregate(t.Context(), plan, testSchema(), []*internalpb.RetrieveResults{partial(), partial()}, 1<<20)
		require.NoError(t, err)
		require.Equal(t, []int64{1, 2}, result.FieldsData[1].GetScalars().GetLongData().Data)
		require.Equal(t, []int64{0, 2}, result.FieldsData[2].GetScalars().GetLongData().Data)
		require.Equal(t, []int64{2, 18}, result.FieldsData[3].GetScalars().GetLongData().Data)
	}
}
