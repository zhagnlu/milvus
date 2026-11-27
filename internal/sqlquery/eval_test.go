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
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/sqlparser"
)

func testSchema() *schemapb.CollectionSchema {
	return &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{
		{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
		{FieldID: 101, Name: "v", DataType: schemapb.DataType_Int64, Nullable: true},
		{FieldID: 102, Name: "data", DataType: schemapb.DataType_JSON, Nullable: true},
	}}
}

func testBind(t *testing.T, sql string) *sqlparser.BoundQuery {
	t.Helper()
	comp, err := sqlparser.ExtractSqlComponents(sql)
	require.NoError(t, err)
	bound, err := sqlparser.Bind(comp, testSchema())
	require.NoError(t, err)
	return bound
}

func TestEvaluatePrecisionNullAndErrors(t *testing.T) {
	for _, test := range []struct {
		expression string
		want       any
		fail       bool
	}{
		{"9223372036854775807 - 1", int64(9223372036854775806), false},
		{"(data->>'n')::bigint", int64(9007199254740993), false},
		{"v + 1", nil, false},
		{"9223372036854775807 + 1", nil, true},
		{"1 / 0", nil, true},
		{"'32768'::smallint", nil, true},
		{"''::bigint", nil, true},
	} {
		t.Run(test.expression, func(t *testing.T) {
			bound := testBind(t, "SELECT "+test.expression+" FROM events")
			value, err := (evaluation{columns: map[int64]any{102: []byte(`{"n":9007199254740993}`)}, zone: time.UTC}).eval(bound.Outputs[0].Expr)
			if test.fail {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.Equal(t, test.want, value)
		})
	}
}

func TestJSONNullAndTextExtraction(t *testing.T) {
	for _, test := range []struct {
		sql  string
		want any
	}{
		{"SELECT data->'x' FROM events", []byte("null")},
		{"SELECT data->>'x' FROM events", nil},
		{"SELECT data->'missing' FROM events", nil},
		{"SELECT (data->'s')::text FROM events", `"hello"`},
		{"SELECT data->'array'->>'0' FROM events", nil},
		{"SELECT data->'object'->>'0' FROM events", "value"},
	} {
		bound := testBind(t, test.sql)
		value, err := (evaluation{columns: map[int64]any{102: []byte(`{"x":null,"s":"hello","array":["a"],"object":{"0":"value"}}`)}, zone: time.UTC}).eval(bound.Outputs[0].Expr)
		require.NoError(t, err, test.sql)
		require.Equal(t, test.want, value, test.sql)
	}
}

func TestEvaluateSQLThreeValuedPredicates(t *testing.T) {
	for _, test := range []struct {
		filter string
		want   any
	}{
		{`data->>'n' = '1'`, true},
		{`data->>'nil' IS NULL`, true},
		{`NOT (v = 1 OR v = 2)`, nil},
		{`pk NOT IN (1, NULL)`, nil},
		{`pk IN (2, NULL)`, true},
		{`pk = 3 AND v = 1`, false},
		{`pk = 2 OR v = 1`, true},
		{`data->>'text' LIKE '_界'`, true},
	} {
		t.Run(test.filter, func(t *testing.T) {
			bound := testBind(t, "SELECT pk FROM events WHERE "+test.filter)
			value, err := (evaluation{columns: map[int64]any{100: int64(2), 102: []byte(`{"n":1,"nil":null,"text":"世界"}`)}, zone: time.UTC}).eval(bound.Predicate)
			require.NoError(t, err)
			require.Equal(t, test.want, value)
		})
	}
}

func TestEvaluateTimezoneAndNegativeEpoch(t *testing.T) {
	bound := testBind(t, "SELECT extract(hour FROM to_timestamp(-0.5)), extract(second FROM to_timestamp(-0.5)) FROM events")
	zone, err := time.LoadLocation("Asia/Shanghai")
	require.NoError(t, err)
	eval := evaluation{zone: zone}
	hour, err := eval.eval(bound.Outputs[0].Expr)
	require.NoError(t, err)
	require.Equal(t, float64(7), hour)
	second, err := eval.eval(bound.Outputs[1].Expr)
	require.NoError(t, err)
	require.Equal(t, 59.5, second)
}
