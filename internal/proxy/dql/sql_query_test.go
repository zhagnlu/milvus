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

package dql

import (
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

func sqlTask(statement string) *QueryTask {
	return &QueryTask{
		RetrieveRequest: &internalpb.RetrieveRequest{}, queryParams: &queryParams{},
		request: &milvuspb.QueryRequest{CollectionName: "events", QueryParams: []*commonpb.KeyValuePair{{Key: SQLQueryKey, Value: statement}}},
		schema: mustNewSchemaInfo(&schemapb.CollectionSchema{Name: "events", Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 101, Name: "data", DataType: schemapb.DataType_JSON},
		}}),
	}
}

func TestSQLPreparationRetainsProjectionAndGlobalLimit(t *testing.T) {
	task := sqlTask("SELECT data->>'kind' AS kind, count(*) AS n FROM events WHERE pk > 2 GROUP BY kind ORDER BY n DESC LIMIT 1")
	prepared, err := task.prepareSQL(&planparserv2.ParserVisitorArgs{})
	require.NoError(t, err)
	require.True(t, prepared)
	require.Equal(t, int64(typeutil.Unlimited), task.Limit)
	require.Empty(t, task.GetOrderByFields())
	require.NotNil(t, task.SqlProject.Predicate)
	require.True(t, proto.Equal(task.SqlProject, task.plan.GetQuery().SqlProject))
	require.Equal(t, task.GroupByFieldIds, task.plan.GetQuery().GroupByFieldIds)
	require.Greater(t, task.Aggregates[0].FieldId, int64(101))
	require.Empty(t, task.OutputFieldsId)
}

func TestSQLPreparationPreservesComputedDependencies(t *testing.T) {
	task := sqlTask("SELECT (data->>'x')::bigint + pk AS value FROM events LIMIT 3 OFFSET 2")
	prepared, err := task.prepareSQL(&planparserv2.ParserVisitorArgs{})
	require.NoError(t, err)
	require.False(t, prepared)
	require.ElementsMatch(t, []string{"data", "pk"}, task.request.OutputFields)
	require.Equal(t, int64(5), task.Limit)
	task = sqlTask("SELECT pk FROM events WHERE data->>'kind' = 'commit' LIMIT 3")
	_, err = task.prepareSQL(&planparserv2.ParserVisitorArgs{})
	require.NoError(t, err)
	require.Equal(t, int64(typeutil.Unlimited), task.Limit, "filtering must precede pagination")
}

func TestSQLPreparationRejectsCollectionMismatchAndDuplicateSQL(t *testing.T) {
	task := sqlTask("SELECT pk FROM secret LIMIT 1")
	_, err := task.prepareSQL(&planparserv2.ParserVisitorArgs{})
	require.ErrorContains(t, err, "authorized Query collection")
	task = sqlTask("")
	task.request.QueryParams = append(task.request.QueryParams, &commonpb.KeyValuePair{Key: SQLQueryKey, Value: "SELECT pk FROM events"})
	_, err = task.prepareSQL(&planparserv2.ParserVisitorArgs{})
	require.ErrorContains(t, err, "duplicate SQL")
}
