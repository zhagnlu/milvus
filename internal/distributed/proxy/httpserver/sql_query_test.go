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

package httpserver

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"

	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/mocks"
	"github.com/milvus-io/milvus/internal/proxy"
	"github.com/milvus-io/milvus/internal/proxy/dql"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

func TestSQLHTTPUsesAuthorizedQueryRequest(t *testing.T) {
	paramtable.Init()
	key := paramtable.Get().QuotaConfig.QuotaAndLimitsEnabled.Key
	require.NoError(t, paramtable.Get().Save(key, "false"))
	t.Cleanup(func() { _ = paramtable.Get().Reset(key) })
	mp := mocks.NewMockProxy(t)
	mp.EXPECT().DescribeCollection(mock.Anything, mock.Anything).Return(&milvuspb.DescribeCollectionResponse{
		Status: merr.Success(), Schema: &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true}}},
	}, nil).Once()
	mp.EXPECT().Query(mock.Anything, mock.Anything).RunAndReturn(func(_ context.Context, req *milvuspb.QueryRequest) (*milvuspb.QueryResults, error) {
		require.Equal(t, "events", req.CollectionName)
		require.Equal(t, "analytics", req.DbName)
		require.Equal(t, dql.SQLQueryKey, req.QueryParams[0].Key)
		require.Equal(t, "SELECT count(*) AS n FROM events WHERE pk > $1", req.QueryParams[0].Value)
		require.Equal(t, int64(7), req.ExprTemplateValues["sql_param_1"].GetInt64Val())
		return &milvuspb.QueryResults{Status: merr.Success(), OutputFields: []string{"n"}, FieldsData: []*schemapb.FieldData{{FieldName: "n", Type: schemapb.DataType_Int64, Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{Data: &schemapb.ScalarField_LongData{LongData: &schemapb.LongArray{Data: []int64{9}}}}}}}}, nil
	}).Once()
	engine := initHTTPServerV2(mp, false)
	req := httptest.NewRequest(http.MethodPost, "/v2/vectordb/entities/sql_query", strings.NewReader(`{"dbName":"analytics","sql":"SELECT count(*) AS n FROM events WHERE pk > $1","params":{"$1":7}}`))
	req.Header.Set(HTTPHeaderAllowInt64, "true")
	w := httptest.NewRecorder()
	engine.ServeHTTP(w, req)
	require.Equal(t, http.StatusOK, w.Code)
	require.Equal(t, int64(0), gjson.Get(w.Body.String(), "code").Int(), w.Body.String())
	require.Equal(t, int64(9), gjson.Get(w.Body.String(), "data.0.n").Int(), w.Body.String())
}

func TestSQLHTTPRejectsInvalidStatementsBeforeRPC(t *testing.T) {
	engine := initHTTPServerV2(mocks.NewMockProxy(t), false)
	for _, body := range []string{`{}`, `{"sql":"DELETE FROM events"}`, `{"sql":"SELECT * FROM events; SELECT * FROM secret"}`} {
		w := httptest.NewRecorder()
		engine.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v2/vectordb/entities/sql_query", strings.NewReader(body)))
		require.NotEqual(t, int64(0), gjson.Get(w.Body.String(), "code").Int(), w.Body.String())
	}
}

func TestSQLHTTPRequiresAuthentication(t *testing.T) {
	engine := initHTTPServerV2(mocks.NewMockProxy(t), true)
	w := httptest.NewRecorder()
	engine.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v2/vectordb/entities/sql_query", strings.NewReader(`{"sql":"SELECT pk FROM events LIMIT 1"}`)))
	require.Equal(t, http.StatusUnauthorized, w.Code)
}

func TestSQLTraceRedactsParametersWithoutMutation(t *testing.T) {
	request := &SQLQueryReqV2{
		SQL:          "SELECT pk FROM events WHERE pk = $1",
		Params:       map[string]json.RawMessage{"$1": json.RawMessage(`42`)},
		SearchParams: map[string]string{"$2": "[0.1, 0.2]"},
	}
	logged := getTraceLogRequestFieldWithoutSensitiveInfo(request).Interface.(*SQLQueryReqV2)
	require.Equal(t, request.SQL, logged.SQL)
	var marker string
	require.NoError(t, json.Unmarshal(logged.Params["$1"], &marker))
	require.Equal(t, proxy.RedactedValue, marker)
	require.Equal(t, proxy.RedactedValue, logged.SearchParams["$2"])
	require.Equal(t, json.RawMessage(`42`), request.Params["$1"])
	require.Equal(t, "[0.1, 0.2]", request.SearchParams["$2"])
}
