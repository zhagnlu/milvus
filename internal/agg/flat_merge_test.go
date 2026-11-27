package agg

import (
	"context"
	"math"
	"testing"

	"github.com/bytedance/mockey"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

func TestGroupedMergeCollisionOwnershipAndLimit(t *testing.T) {
	// Force every key, including NULL, into the same hash bucket. Groups must
	// still compare all keys and keep independent accumulator storage.
	patch := mockey.Mock((*StringFieldAccessor).Hash).To(func(*StringFieldAccessor, int) uint64 { return 1 }).Build()
	defer patch.UnPatch()
	keys, err := genEmptyFieldDataByType(schemapb.DataType_VarChar)
	require.NoError(t, err)
	keys.GetScalars().GetStringData().Data = []string{"a", "b", "", "ignored"}
	typeutil.SetFieldDataValidData(keys, []bool{true, true, true, false})
	values := genEmptyLongFieldData(schemapb.DataType_Int64, []int64{1, 10, 100, 1000})
	originalKeys, originalValues := proto.Clone(keys), proto.Clone(values)
	for _, limit := range []int64{-1, 2} {
		r := makeGroupAggReducer()
		r.groupLimit = limit
		partial, err := r.DecodePartial([]*schemapb.FieldData{keys, values}, 4)
		require.NoError(t, err)
		for attempt := 0; attempt < 2; attempt++ {
			merged, err := r.Merge(t.Context(), []*PartialAggregationResult{partial, partial})
			require.NoError(t, err)
			actual := make(map[any]int64)
			outKeys, outValues := merged.fieldDatas[0], merged.fieldDatas[1]
			for i, value := range outValues.GetScalars().GetLongData().Data {
				var key any
				if typeutil.GetFieldDataValidData(outKeys)[i] {
					key = outKeys.GetScalars().GetStringData().Data[i]
				}
				actual[key] = value
			}
			want := map[any]int64{"a": 2, "b": 20}
			if limit == -1 {
				want[""], want[nil] = 200, 2000
			}
			require.Equal(t, want, actual)
			require.EqualValues(t, 8, merged.allRetrieveCount)
			require.True(t, proto.Equal(originalKeys, keys))
			require.True(t, proto.Equal(originalValues, values))
		}
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		_, err = r.Merge(ctx, []*PartialAggregationResult{partial, partial})
		require.ErrorIs(t, err, context.Canceled)
	}
}

func TestGroupedMergeScalarKeys(t *testing.T) {
	for _, tc := range []struct {
		typ    schemapb.DataType
		values []any
	}{
		{schemapb.DataType_Bool, []any{true, false}},
		{schemapb.DataType_Int32, []any{int32(-1), int32(1)}},
		{schemapb.DataType_Int64, []any{int64(9007199254740992), int64(9007199254740993)}},
		{schemapb.DataType_Timestamptz, []any{int64(-500000), int64(500001)}},
		{schemapb.DataType_Float, []any{float32(1.25), float32(-2.5)}},
		{schemapb.DataType_Double, []any{float64(1.25), float64(-2.5)}},
	} {
		t.Run(tc.typ.String(), func(t *testing.T) {
			schema := &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{
				{FieldID: 1, Name: "key", DataType: tc.typ, Nullable: true},
				{FieldID: 2, Name: "count", DataType: schemapb.DataType_Int64},
			}}
			r := NewGroupAggReducer([]int64{1}, []*planpb.Aggregate{{Op: planpb.AggregateOp_count, FieldId: 2}}, -1, schema)
			keys, err := typeutil.GenEmptyFieldData(&schemapb.FieldSchema{DataType: tc.typ, Nullable: true})
			require.NoError(t, err)
			for _, value := range tc.values {
				require.NoError(t, AssembleSingleValue(NewFieldValue(value), keys))
			}
			require.NoError(t, AssembleSingleValue(NewNullFieldValue(), keys))
			partial, err := r.DecodePartial([]*schemapb.FieldData{keys, genEmptyLongFieldData(schemapb.DataType_Int64, []int64{1, 2, 3})}, 6)
			require.NoError(t, err)
			merged, err := r.Merge(t.Context(), []*PartialAggregationResult{partial, partial})
			require.NoError(t, err)
			require.Equal(t, []int64{2, 4, 6}, merged.fieldDatas[1].GetScalars().GetLongData().Data)
			require.Equal(t, []bool{true, true, false}, typeutil.GetFieldDataValidData(merged.fieldDatas[0]))
		})
	}
	// Retain the existing IEEE equality contract: NaNs never equal one another,
	// even when the hash bits are identical and they share a collision chain.
	schema := &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{{FieldID: 1, Name: "key", DataType: schemapb.DataType_Double}, {FieldID: 2, Name: "count", DataType: schemapb.DataType_Int64}}}
	r := NewGroupAggReducer([]int64{1}, []*planpb.Aggregate{{Op: planpb.AggregateOp_count, FieldId: 2}}, -1, schema)
	keys, err := genEmptyFieldDataByType(schemapb.DataType_Double)
	require.NoError(t, err)
	keys.GetScalars().GetDoubleData().Data = []float64{math.NaN(), 1}
	partial, err := r.DecodePartial([]*schemapb.FieldData{keys, genEmptyLongFieldData(schemapb.DataType_Int64, []int64{1, 1})}, 2)
	require.NoError(t, err)
	merged, err := r.Merge(t.Context(), []*PartialAggregationResult{partial, partial})
	require.NoError(t, err)
	require.ElementsMatch(t, []int64{1, 2, 1}, merged.fieldDatas[1].GetScalars().GetLongData().Data)
}

func TestGroupedMergeCompositeCollisionAfterLimit(t *testing.T) {
	strings := mockey.Mock((*StringFieldAccessor).Hash).To(func(*StringFieldAccessor, int) uint64 { return 1 }).Build()
	defer strings.UnPatch()
	integers := mockey.Mock((*Int64FieldAccessor).Hash).To(func(*Int64FieldAccessor, int) uint64 { return 1 }).Build()
	defer integers.UnPatch()
	schema := &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{
		{FieldID: 1, Name: "text", DataType: schemapb.DataType_VarChar},
		{FieldID: 2, Name: "integer", DataType: schemapb.DataType_Int64},
		{FieldID: 3, Name: "sum", DataType: schemapb.DataType_Int64, Nullable: true},
	}}
	r := NewGroupAggReducer([]int64{1, 2}, []*planpb.Aggregate{{Op: planpb.AggregateOp_sum, FieldId: 3}}, 2, schema)
	makePartial := func(text []string, integer, sum []int64, valid []bool) *PartialAggregationResult {
		keys, err := genEmptyFieldDataByType(schemapb.DataType_VarChar)
		require.NoError(t, err)
		keys.GetScalars().GetStringData().Data = text
		values := genEmptyLongFieldData(schemapb.DataType_Int64, sum)
		typeutil.SetFieldDataValidData(values, valid)
		partial, err := r.DecodePartial([]*schemapb.FieldData{keys, genEmptyLongFieldData(schemapb.DataType_Int64, integer), values}, int64(len(text)))
		require.NoError(t, err)
		return partial
	}
	// The first two groups share their first key and collide in every hasher.
	// A new group after the limit must not prevent later matches from updating
	// retained groups, including initialization of a NULL accumulator.
	first := makePartial([]string{"a\x00b", "a\x00b", "ignored"}, []int64{math.MinInt64, math.MaxInt64, 0}, []int64{0, 10, 999}, []bool{false, true, true})
	second := makePartial([]string{"a\x00b", "ignored-again", "a\x00b"}, []int64{math.MaxInt64, 0, math.MinInt64}, []int64{2, 88, 7}, []bool{true, true, true})
	merged, err := r.Merge(t.Context(), []*PartialAggregationResult{first, second})
	require.NoError(t, err)
	require.Equal(t, []string{"a\x00b", "a\x00b"}, merged.fieldDatas[0].GetScalars().GetStringData().Data)
	require.Equal(t, []int64{math.MinInt64, math.MaxInt64}, merged.fieldDatas[1].GetScalars().GetLongData().Data)
	require.Equal(t, []int64{7, 12}, merged.fieldDatas[2].GetScalars().GetLongData().Data)
	require.Equal(t, []bool{true, true}, typeutil.GetFieldDataValidData(merged.fieldDatas[2]))
}

func TestGroupedMergeProbePreservesUpdateError(t *testing.T) {
	want := merr.Wrap(merr.ErrServiceUnavailable, "injected aggregate update failure")
	patch := mockey.Mock((*SumAggregate).Update).To(func(*SumAggregate, *FieldValue, *FieldValue) error { return want }).Build()
	defer patch.UnPatch()
	r := makeGroupAggReducer()
	keys, err := genEmptyFieldDataByType(schemapb.DataType_VarChar)
	require.NoError(t, err)
	keys.GetScalars().GetStringData().Data = []string{"retained"}
	partial, err := r.DecodePartial([]*schemapb.FieldData{keys, genEmptyLongFieldData(schemapb.DataType_Int64, []int64{1})}, 1)
	require.NoError(t, err)
	_, err = r.Merge(t.Context(), []*PartialAggregationResult{partial, partial})
	require.Same(t, want, err)
	require.ErrorIs(t, err, merr.ErrServiceUnavailable)
}
