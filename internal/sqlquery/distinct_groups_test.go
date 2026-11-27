// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

package sqlquery

import (
	"context"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/sqlparser"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

func TestDistinctSingleExtensionAcrossMergeShapes(t *testing.T) {
	paramtable.Init()
	bound := testBind(t, "SELECT v, count(DISTINCT data->>'x') AS nx, count(DISTINCT data->>'x') AS again, count(*) AS n FROM events GROUP BY v ORDER BY v NULLS FIRST")
	plan, err := sqlparser.LowerAggregate(bound, testSchema(), "UTC")
	require.NoError(t, err)
	partial := func(groups, xs, counts []any) *internalpb.RetrieveResults {
		return &internalpb.RetrieveResults{FieldsData: []*schemapb.FieldData{
			testField(t, plan.GroupIDs[0], schemapb.DataType_Int64, groups...),
			testField(t, plan.GroupIDs[1], schemapb.DataType_VarChar, xs...),
			testField(t, plan.Aggregates[0].FieldId, schemapb.DataType_Int64, counts...),
		}}
	}
	groups := []any{nil, nil, int64(1), int64(1), int64(1), int64(1)}
	xs := []any{"", nil, strings.Repeat("wide", 80), "a\x00b", "a", "a\x00b"}
	counts := []any{int64(1), int64(2), int64(3), int64(4), int64(5), int64(6)}
	moreGroups := []any{nil, int64(1), int64(1), int64(1)}
	moreXs := []any{"", "a\x00b", "b", nil}
	moreCounts := []any{int64(7), int64(8), int64(9), int64(10)}
	first, second := partial(groups, xs, counts), partial(moreGroups, moreXs, moreCounts)
	combined := partial(append(groups, moreGroups...), append(xs, moreXs...), append(counts, moreCounts...))
	empty := partial(nil, nil, nil)
	shapes := [][]*internalpb.RetrieveResults{{first, second}, {combined}, {empty, combined}, {first, empty, second}}
	for i, inputs := range shapes {
		t.Run(fmt.Sprintf("shape_%d", i), func(t *testing.T) {
			before := make([]proto.Message, len(inputs))
			for i, input := range inputs {
				before[i] = proto.Clone(input)
			}
			result, err := ReduceAggregate(t.Context(), plan, testSchema(), inputs, 1<<20)
			require.NoError(t, err)
			require.Equal(t, []int64{1, 4}, result.FieldsData[1].GetScalars().GetLongData().Data)
			require.Equal(t, []int64{1, 4}, result.FieldsData[2].GetScalars().GetLongData().Data)
			require.Equal(t, []int64{10, 45}, result.FieldsData[3].GetScalars().GetLongData().Data)
			for i, input := range inputs {
				require.True(t, proto.Equal(before[i], input), "retained partial mutated")
			}
		})
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err = ReduceAggregate(ctx, plan, testSchema(), []*internalpb.RetrieveResults{first, second}, 1<<20)
	require.ErrorIs(t, err, context.Canceled)
}

func TestDistinctSingleExtensionExactIntegersAndAllNull(t *testing.T) {
	paramtable.Init()
	plan, err := sqlparser.LowerAggregate(testBind(t, "SELECT count(DISTINCT v) AS n FROM events"), testSchema(), "UTC")
	require.NoError(t, err)
	for _, tc := range []struct {
		values []any
		want   int64
	}{
		{[]any{int64(9007199254740992), int64(9007199254740993), nil, int64(9007199254740992)}, 2},
		{[]any{nil, nil}, 0},
	} {
		partial := &internalpb.RetrieveResults{FieldsData: []*schemapb.FieldData{
			testField(t, plan.GroupIDs[0], schemapb.DataType_Int64, tc.values...),
		}}
		result, err := ReduceAggregate(t.Context(), plan, testSchema(), []*internalpb.RetrieveResults{partial, partial}, 1<<20)
		require.NoError(t, err)
		require.Equal(t, []int64{tc.want}, result.FieldsData[0].GetScalars().GetLongData().Data)
	}
}

func TestDistinctFloatExtensionKeepsSetEquality(t *testing.T) {
	paramtable.Init()
	schema := testSchema()
	schema.Fields = append(schema.Fields, &schemapb.FieldSchema{FieldID: 103, Name: "f", DataType: schemapb.DataType_Double})
	components, err := sqlparser.ExtractSqlComponents("SELECT count(DISTINCT f) AS n, count(*) AS total FROM events")
	require.NoError(t, err)
	bound, err := sqlparser.Bind(components, schema)
	require.NoError(t, err)
	plan, err := sqlparser.LowerAggregate(bound, schema, "UTC")
	require.NoError(t, err)
	partial := &internalpb.RetrieveResults{FieldsData: []*schemapb.FieldData{
		testField(t, plan.GroupIDs[0], schemapb.DataType_Double, 0.0, math.Copysign(0, -1), math.NaN(), math.NaN(), math.Inf(1), math.Inf(1)),
		testField(t, plan.Aggregates[0].FieldId, schemapb.DataType_Int64, int64(1), int64(1), int64(1), int64(1), int64(1), int64(1)),
	}}
	result, err := ReduceAggregate(t.Context(), plan, schema, []*internalpb.RetrieveResults{partial, partial}, 1<<20)
	require.NoError(t, err)
	// Go set equality merges signed zeros and infinities; every NaN remains
	// distinct. Counting physical groups directly would overcount signed zero.
	require.Equal(t, []int64{6}, result.FieldsData[0].GetScalars().GetLongData().Data)
	require.Equal(t, []int64{12}, result.FieldsData[1].GetScalars().GetLongData().Data)
}

func TestDistinctSingleExtensionKeepsResourceAccounting(t *testing.T) {
	paramtable.Init()
	plan, err := sqlparser.LowerAggregate(testBind(t, "SELECT count(DISTINCT data->>'x') AS n FROM events"), testSchema(), "UTC")
	require.NoError(t, err)
	values := make([]any, 1000)
	for i := range values {
		values[i] = fmt.Sprintf("value-%04d", i)
	}
	partial := &internalpb.RetrieveResults{FieldsData: []*schemapb.FieldData{
		testField(t, plan.GroupIDs[0], schemapb.DataType_VarChar, values...),
	}}
	// Input fits, but the conservative distinct-state estimate exceeds the
	// budget. Eliding a set must not silently remove the existing guard.
	_, err = ReduceAggregate(t.Context(), plan, testSchema(), []*internalpb.RetrieveResults{partial, partial}, int64(2*proto.Size(partial)+100))
	require.ErrorIs(t, err, merr.ErrParameterInvalid)
	require.Contains(t, err.Error(), "SQL aggregation exceeds")
}

func TestDistinctMixedStringAndNumericDimensions(t *testing.T) {
	paramtable.Init()
	plan, err := sqlparser.LowerAggregate(testBind(t, "SELECT count(DISTINCT data->>'x') AS nx, count(DISTINCT v) AS nv, count(*) AS n FROM events"), testSchema(), "UTC")
	require.NoError(t, err)
	partial := func(xs, vs []any) *internalpb.RetrieveResults {
		counts := make([]any, len(xs))
		for i := range counts {
			counts[i] = int64(1)
		}
		return &internalpb.RetrieveResults{FieldsData: []*schemapb.FieldData{
			testField(t, plan.GroupIDs[0], schemapb.DataType_VarChar, xs...),
			testField(t, plan.GroupIDs[1], schemapb.DataType_Int64, vs...),
			testField(t, plan.Aggregates[0].FieldId, schemapb.DataType_Int64, counts...),
		}}
	}
	result, err := ReduceAggregate(t.Context(), plan, testSchema(), []*internalpb.RetrieveResults{
		partial([]any{"same", "same", "", nil}, []any{int64(1), int64(2), nil, int64(3)}),
		partial([]any{"same", "other", nil}, []any{int64(1), int64(2), nil}),
	}, 1<<20)
	require.NoError(t, err)
	for i, want := range []int64{3, 3, 7} {
		require.Equal(t, []int64{want}, result.FieldsData[i].GetScalars().GetLongData().Data)
	}
}

func TestDistinctStringSQLGroupKeyRetainsIdentity(t *testing.T) {
	paramtable.Init()
	plan, err := sqlparser.LowerAggregate(testBind(t, "SELECT data->>'x' AS x, count(DISTINCT data->>'x') AS nx, count(*) AS n FROM events GROUP BY x ORDER BY x NULLS FIRST"), testSchema(), "UTC")
	require.NoError(t, err)
	partial := &internalpb.RetrieveResults{FieldsData: []*schemapb.FieldData{
		testField(t, plan.GroupIDs[0], schemapb.DataType_VarChar, "", "a\x00b", nil),
		testField(t, plan.Aggregates[0].FieldId, schemapb.DataType_Int64, int64(1), int64(1), int64(1)),
	}}
	result, err := ReduceAggregate(t.Context(), plan, testSchema(), []*internalpb.RetrieveResults{partial, partial}, 1<<20)
	require.NoError(t, err)
	require.Equal(t, []string{"", "", "a\x00b"}, result.FieldsData[0].GetScalars().GetStringData().Data)
	require.Equal(t, []bool{false, true, true}, result.FieldsData[0].GetValidData())
	require.Equal(t, []int64{0, 1, 1}, result.FieldsData[1].GetScalars().GetLongData().Data)
	require.Equal(t, []int64{2, 2, 2}, result.FieldsData[2].GetScalars().GetLongData().Data)
}
