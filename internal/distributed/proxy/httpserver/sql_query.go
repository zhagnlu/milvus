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
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus/internal/proxy"
	"github.com/milvus-io/milvus/internal/proxy/dql"
	"github.com/milvus-io/milvus/internal/sqlparser"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func (h *HandlersV2) sqlQuery(ctx context.Context, c *gin.Context, anyReq any, dbName string) (interface{}, error) {
	request := anyReq.(*SQLQueryReqV2)
	if request.Timezone == "" {
		request.Timezone = "UTC"
	}
	fail := func(err error) (interface{}, error) {
		HTTPAbortReturn(c, http.StatusOK, gin.H{HTTPReturnCode: merr.Code(err), HTTPReturnMessage: err.Error()})
		return nil, err
	}
	components, err := sqlparser.ExtractSqlComponents(request.SQL)
	if err != nil {
		return fail(err)
	}
	schema, err := h.GetCollectionSchema(ctx, c, dbName, components.Collection)
	if err != nil {
		return nil, err
	}
	consistency, useDefault, err := convertConsistencyLevel(request.ConsistencyLevel)
	if err != nil {
		return fail(err)
	}
	values, err := generateExpressionTemplate(request.Params)
	if err != nil {
		return fail(err)
	}
	values, err = sqlparser.BuildSQLParamTemplateValues(values)
	if err != nil {
		return fail(err)
	}
	if components.VectorSearch != nil {
		return h.sqlSearch(ctx, c, request, dbName, components, schema, values, consistency, useDefault)
	}
	req := &milvuspb.QueryRequest{
		DbName: dbName, CollectionName: components.Collection, PartitionNames: request.PartitionNames,
		ConsistencyLevel: consistency, UseDefaultConsistency: useDefault, ExprTemplateValues: values,
		QueryParams: []*commonpb.KeyValuePair{{Key: dql.SQLQueryKey, Value: request.SQL}},
	}
	if request.Timezone != "" {
		req.QueryParams = append(req.QueryParams, &commonpb.KeyValuePair{Key: "timezone", Value: request.Timezone})
	}
	c.Set(ContextRequest, req)
	response, err := h.wrapperProxyWithLimit(ctx, c, req, h.checkAuth, false, "/milvus.proto.milvus.MilvusService/Query", true, h.proxy, func(reqCtx context.Context, req any) (interface{}, error) {
		return h.proxy.Query(reqCtx, req.(*milvuspb.QueryRequest))
	})
	if err != nil {
		return response, err
	}
	result := response.(*milvuspb.QueryResults)
	allowInt64, _ := strconv.ParseBool(c.Request.Header.Get(HTTPHeaderAllowInt64))
	rows, err := buildQueryResp(0, result.OutputFields, result.FieldsData, nil, nil, allowInt64, schema)
	if err != nil {
		return fail(err)
	}
	HTTPReturnStream(c, http.StatusOK, gin.H{HTTPReturnCode: merr.Code(nil), HTTPReturnData: rows, HTTPReturnCost: proxy.GetCostValue(result.Status)})
	return response, nil
}
