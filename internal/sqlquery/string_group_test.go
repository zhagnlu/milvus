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
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/sqlparser"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

// Adding a constant second key forces generic tuple encoding. Both paths must
// retain identical group identities and exact sets for every input merge shape.
func TestStringGroupMatchesCompositeGrouping(t *testing.T) {
	paramtable.Init()
	keys := []any{nil, "", "a\x00b", "世界", strings.Repeat("long", 80), "a\x00b", nil, ""}
	xs := []any{"a", "", "x", nil, "long", "x", "b", ""}
	vs := []any{int64(1), int64(2), int64(3), nil, int64(4), int64(5), int64(1), nil}
	for _, multi := range []bool{false, true} {
		var first proto.Message
		for _, composite := range []bool{false, true} {
			sql := "SELECT data->>'g' AS g, count(DISTINCT data->>'x') AS nx, count(DISTINCT v) AS nv, count(*) AS n FROM events GROUP BY g"
			if composite {
				sql += ", pk"
			}
			sql += " ORDER BY g NULLS FIRST"
			plan, err := sqlparser.LowerAggregate(testBind(t, sql), testSchema(), "UTC")
			require.NoError(t, err)
			fields := []*schemapb.FieldData{testField(t, plan.GroupIDs[0], schemapb.DataType_VarChar, keys...)}
			if composite {
				constants := make([]any, len(keys))
				for i := range constants {
					constants[i] = int64(7)
				}
				fields = append(fields, testField(t, plan.GroupIDs[1], schemapb.DataType_Int64, constants...))
			}
			offset := len(fields)
			fields = append(fields, testField(t, plan.GroupIDs[offset], schemapb.DataType_VarChar, xs...), testField(t, plan.GroupIDs[offset+1], schemapb.DataType_Int64, vs...), testField(t, plan.Aggregates[0].FieldId, schemapb.DataType_Int64, int64(1), int64(2), int64(3), int64(4), int64(5), int64(6), int64(7), int64(8)))
			input := &internalpb.RetrieveResults{FieldsData: fields}
			before := proto.Clone(input)
			inputs := []*internalpb.RetrieveResults{input}
			if multi {
				inputs = append(inputs, input)
			}
			got, err := ReduceAggregate(t.Context(), plan, testSchema(), inputs, 1<<20)
			require.NoError(t, err)
			require.True(t, proto.Equal(before, input), "retained input mutated")
			if first == nil {
				first = got
			} else {
				require.True(t, proto.Equal(first, got), "string grouping differs from generic composite grouping: %v / %v", first, got)
			}
			ctx, cancel := context.WithCancel(t.Context())
			cancel()
			_, err = ReduceAggregate(ctx, plan, testSchema(), inputs, 1<<20)
			require.ErrorIs(t, err, context.Canceled)
		}
	}
}

func TestStringGroupRetainsResourceLimit(t *testing.T) {
	paramtable.Init()
	plan, err := sqlparser.LowerAggregate(testBind(t, "SELECT data->>'g' AS g, count(DISTINCT v) AS n FROM events GROUP BY g"), testSchema(), "UTC")
	require.NoError(t, err)
	keys, values := make([]any, 300), make([]any, 300)
	for i := range keys {
		keys[i] = ""
		values[i] = int64(i)
	}
	input := &internalpb.RetrieveResults{FieldsData: []*schemapb.FieldData{testField(t, plan.GroupIDs[0], schemapb.DataType_VarChar, keys...), testField(t, plan.GroupIDs[1], schemapb.DataType_Int64, values...)}}
	_, err = ReduceAggregate(t.Context(), plan, testSchema(), []*internalpb.RetrieveResults{input}, 16000)
	require.ErrorContains(t, err, "SQL aggregation exceeds the output size limit")
}
