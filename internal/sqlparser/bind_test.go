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

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
)

func bindingSchema() *schemapb.CollectionSchema {
	return &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{
		{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
		{FieldID: 101, Name: "data", DataType: schemapb.DataType_JSON, Nullable: true},
		{FieldID: 102, Name: "v", DataType: schemapb.DataType_Int64},
	}}
}

func bindSQL(t *testing.T, sql string) *BoundQuery {
	t.Helper()
	comp, err := ExtractSqlComponents(sql)
	require.NoError(t, err)
	bound, err := Bind(comp, bindingSchema())
	require.NoError(t, err)
	return bound
}

func TestBindJSONBench(t *testing.T) {
	for name, sql := range jsonBenchQueries {
		t.Run(name, func(t *testing.T) { bindSQL(t, sql) })
	}
}

func TestBindComputedGroupingBeforeAggregation(t *testing.T) {
	bound := bindSQL(t, jsonBenchQueries["Q3"])
	require.Len(t, bound.GroupKeys, 2)
	require.Equal(t, "extract", bound.GroupKeys[1].Op)
	require.Equal(t, "hour", bound.GroupKeys[1].Value)
	require.Equal(t, "to_timestamp", bound.GroupKeys[1].Args[0].Op)
	require.Equal(t, "slot", bound.Outputs[1].Expr.Op)
	require.Equal(t, 1, bound.Outputs[1].Expr.Slot)
}

func TestBindDistinctAndAggregateIdentity(t *testing.T) {
	bound := bindSQL(t, `SELECT data->>'city' AS city, data->>'country' AS country,
count(DISTINCT data->>'user') AS users, min(v) AS lo, max(v) AS hi
FROM events GROUP BY city, country ORDER BY hi DESC`)
	require.Len(t, bound.GroupKeys, 2)
	require.Len(t, bound.Aggregates, 3)
	require.True(t, bound.Aggregates[0].Distinct)
	require.Equal(t, schemapb.DataType_VarChar, bound.Aggregates[0].Input.Type)
	require.NotEqual(t, bound.Aggregates[1].Output.Slot, bound.Aggregates[2].Output.Slot)
	require.Equal(t, bound.Aggregates[2].Output.Slot, bound.OrderBy[0].Expr.Slot)
	require.Equal(t, []int64{101, 102}, bound.SourceFields)
}

func TestBindFinalProjectionUsesReducedStates(t *testing.T) {
	bound := bindSQL(t, jsonBenchQueries["Q5"])
	require.Len(t, bound.Aggregates, 2)
	require.Equal(t, "max", bound.Aggregates[0].Name)
	require.Equal(t, "min", bound.Aggregates[1].Name)
	require.Equal(t, "*", bound.Outputs[1].Expr.Op)
	require.False(t, containsSourceColumn(bound.Outputs[1].Expr))
}

func TestBindRejectInvalidSemantics(t *testing.T) {
	for _, sql := range []string{
		"SELECT missing FROM events",
		"SELECT v, count(*) FROM events",
		"SELECT count(*) FROM events ORDER BY v",
		"SELECT sum(data->>'x') FROM events",
		"SELECT sum(count(v)) FROM events",
		"SELECT row_number() OVER () FROM events",
		"SELECT count(*) FROM events GROUP BY sum(v)",
		"SELECT unknown_function(v) FROM events",
		"SELECT to_timestamp(VARIADIC v) FROM events",
		"SELECT count(VARIADIC v) FROM events",
		"SELECT v FROM events WHERE other.v = 1",
		"SELECT v FROM events WHERE missing = 1",
		"SELECT events.v FROM events AS e",
		"SELECT custom.sum(v) FROM events",
		"SELECT count(*) FROM events GROUP BY data",
		"SELECT v FROM events ORDER BY data",
		"SELECT v FROM events ORDER BY 2",
		"SELECT v FROM events GROUP BY 0",
		"SELECT v::numeric FROM events",
		"SELECT v::varchar(1) FROM events",
		"SELECT v::int[] FROM events",
		"SELECT 9223372036854775808 FROM events",
	} {
		t.Run(sql, func(t *testing.T) {
			comp, err := ExtractSqlComponents(sql)
			if err == nil {
				_, err = Bind(comp, bindingSchema())
			}
			require.Error(t, err)
		})
	}
}

func TestExtractRejectsDiscardedClauses(t *testing.T) {
	for _, sql := range []string{
		"SELECT v FROM events HAVING count(*) > 1",
		"WITH a AS (SELECT v FROM events) SELECT v FROM a",
		"SELECT v FROM events UNION SELECT v FROM events",
		"SELECT DISTINCT v FROM events",
		"SELECT v INTO another FROM events",
		"SELECT v FROM events ORDER BY v USING >",
		"SELECT v FROM events FOR UPDATE",
		"SELECT v FROM events LIMIT -1",
		"SELECT v FROM events OFFSET -1",
		"SELECT count() FROM events",
		"SELECT l2_distance(v, '[1]') OVER () FROM events",
		"SELECT custom.l2_distance(v, '[1]') FROM events",
		"SELECT l2_distance(DISTINCT v, '[1]') FROM events",
		"SELECT sum(DISTINCT v) FROM events",
		"SELECT v FROM events AS e(renamed)",
		"SELECT v FROM events WHERE (data->>'n')::bigint > 1",
		"SELECT data->>'x'->>'y' FROM events",
	} {
		t.Run(sql, func(t *testing.T) { _, err := ExtractSqlComponents(sql); require.Error(t, err) })
	}
}

func TestExtractPaginationAndNulls(t *testing.T) {
	comp, err := ExtractSqlComponents("SELECT v FROM events WHERE v IS NULL ORDER BY v DESC NULLS LAST LIMIT 0 OFFSET 2")
	require.NoError(t, err)
	require.Zero(t, comp.Limit)
	require.EqualValues(t, 2, comp.Offset)
	require.Equal(t, "v is null", comp.Where.MilvusExpr)
	require.False(t, comp.OrderBy[0].NullsFirst)
}

func TestJSONPathEscaping(t *testing.T) {
	col := &ColumnRef{FieldName: "data", NestedPath: []string{"a\"b", "c\\d"}}
	require.Equal(t, `data["a\"b"]["c\\d"]`, col.MilvusExpr())
}

func TestBindQualifiedGroupIdentity(t *testing.T) {
	bound := bindSQL(t, "SELECT e.v, min(e.pk), min(pk) FROM events AS e GROUP BY v ORDER BY e.v")
	require.Len(t, bound.GroupKeys, 1)
	require.Len(t, bound.Aggregates, 1)
	require.Equal(t, "slot", bound.Outputs[0].Expr.Op)
	require.Equal(t, bound.Outputs[0].Expr.Slot, bound.OrderBy[0].Expr.Slot)
	require.Equal(t, bound.Outputs[1].Expr.Slot, bound.Outputs[2].Expr.Slot)
}

func TestBindPositions(t *testing.T) {
	bound := bindSQL(t, "SELECT v, count(*) FROM events GROUP BY 1 ORDER BY 2 DESC")
	require.Equal(t, int64(102), bound.GroupKeys[0].FieldID)
	require.Equal(t, bound.Aggregates[0].Output.Slot, bound.OrderBy[0].Expr.Slot)
	require.True(t, bound.OrderBy[0].Descending)
}

func TestBindPreservesNumericTypes(t *testing.T) {
	bound := bindSQL(t, "SELECT 9223372036854775807, v::smallint AS s, v::integer AS i, v::real AS r FROM events")
	require.Equal(t, int64(9223372036854775807), bound.Outputs[0].Expr.Value)
	require.Equal(t, schemapb.DataType_Int16, bound.Outputs[1].Expr.Type)
	require.Equal(t, schemapb.DataType_Int32, bound.Outputs[2].Expr.Type)
	require.Equal(t, schemapb.DataType_Float, bound.Outputs[3].Expr.Type)
}

func TestWherePreservesPrecedence(t *testing.T) {
	for sql, expected := range map[string]string{
		"SELECT v FROM events WHERE NOT (v = 1 OR pk = 2)": `not ((v == 1) or (pk == 2))`,
		"SELECT v FROM events WHERE (v + 1) * 2 > 4":       `((v + 1) * 2) > 4`,
		"SELECT v FROM events WHERE v > -(pk + 2)":         `v > (-(pk + 2))`,
	} {
		t.Run(sql, func(t *testing.T) {
			comp, err := ExtractSqlComponents(sql)
			require.NoError(t, err)
			require.Equal(t, expected, comp.Where.MilvusExpr)
		})
	}
}
