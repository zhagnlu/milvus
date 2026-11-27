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
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/proxy"
	"github.com/milvus-io/milvus/internal/proxy/dql"
	"github.com/milvus-io/milvus/internal/sqlparser"
	"github.com/milvus-io/milvus/internal/sqlquery"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

func (h *HandlersV2) sqlSearch(ctx context.Context, c *gin.Context, request *SQLQueryReqV2, dbName string, components *sqlparser.SqlComponents, schema *schemapb.CollectionSchema, values map[string]*schemapb.TemplateValue, consistency commonpb.ConsistencyLevel, useDefault bool) (interface{}, error) {
	fail := func(err error) (interface{}, error) {
		HTTPAbortReturn(c, http.StatusOK, gin.H{HTTPReturnCode: merr.Code(err), HTTPReturnMessage: err.Error()})
		return nil, err
	}
	plan, err := sqlparser.BindSearchWithParams(components, schema, values)
	if err != nil {
		return fail(err)
	}
	distance := plan.Distance
	rawVector := ""
	if distance.Literal != nil {
		rawVector = distance.Literal.Raw
	} else if distance.ParamRef != "" {
		rawVector = request.SearchParams[distance.ParamRef]
		if rawVector == "" {
			raw := request.Params[distance.ParamRef]
			if len(raw) == 0 {
				raw = request.Params[strings.TrimPrefix(distance.ParamRef, "$")]
			}
			rawVector = string(raw)
			var text string
			if json.Unmarshal(raw, &text) == nil {
				rawVector = text
			}
		}
	}
	if rawVector == "" {
		return fail(merr.WrapErrParameterInvalidMsg("missing SQL query vector"))
	}
	placeholder, err := sqlparser.BuildTypedPlaceholderGroupFromRaw(rawVector, plan.VectorField)
	if err != nil {
		return fail(err)
	}
	params := make(map[string]any)
	for key, value := range request.SearchParams {
		if strings.HasPrefix(key, "$") {
			continue
		}
		switch key {
		case "anns_field", "metric_type", "topk", "offset", "round_decimal":
			return fail(merr.WrapErrParameterInvalidMsg("SQL controls search parameter %q", key))
		}
		var decoded any
		if json.Unmarshal([]byte(value), &decoded) == nil {
			params[key] = decoded
		} else {
			params[key] = value
		}
	}
	encodedParams, err := json.Marshal(params)
	if err != nil {
		return fail(merr.WrapErrParameterInvalidErr(err, "invalid search parameters"))
	}
	req := &milvuspb.SearchRequest{
		DbName: dbName, CollectionName: components.Collection, PartitionNames: request.PartitionNames,
		DslType: commonpb.DslType_BoolExprV1, Nq: 1, ExprTemplateValues: values,
		ConsistencyLevel: consistency, UseDefaultConsistency: useDefault,
		SearchInput: &milvuspb.SearchRequest_PlaceholderGroup{PlaceholderGroup: placeholder},
		SearchParams: []*commonpb.KeyValuePair{
			{Key: proxy.AnnsFieldKey, Value: plan.VectorField.Name},
			{Key: proxy.TopKKey, Value: strconv.FormatInt(plan.TopK, 10)},
			{Key: "metric_type", Value: distance.Operator.MetricType()},
			{Key: proxy.RoundDecimalKey, Value: "-1"},
			{Key: proxy.ParamsKey, Value: string(encodedParams)},
		},
	}
	if components.Where != nil {
		req.Dsl = components.Where.MilvusExpr
	}
	if request.Timezone != "" {
		req.SearchParams = append(req.SearchParams, &commonpb.KeyValuePair{Key: "timezone", Value: request.Timezone})
	}
	var primary *schemapb.FieldSchema
	for _, field := range schema.Fields {
		if field.IsPrimaryKey {
			primary = field
		}
		for _, id := range plan.Bound.SourceFields {
			if id == field.FieldID {
				req.OutputFields = append(req.OutputFields, field.Name)
				break
			}
		}
	}
	c.Set(ContextRequest, req)
	response, err := h.wrapperProxyWithLimit(ctx, c, req, h.checkAuth, false, "/milvus.proto.milvus.MilvusService/Search", true, h.proxy, func(reqCtx context.Context, req any) (interface{}, error) {
		return h.proxy.Search(reqCtx, req.(*milvuspb.SearchRequest))
	})
	if err != nil {
		return response, err
	}
	result := response.(*milvuspb.SearchResults)
	data := result.GetResults()
	fields := append([]*schemapb.FieldData(nil), data.GetFieldsData()...)
	if primary != nil {
		found := false
		for _, field := range fields {
			if field.FieldId == primary.FieldID {
				found = true
				break
			}
		}
		if !found {
			pk := &schemapb.FieldData{FieldId: primary.FieldID, FieldName: primary.Name, Type: primary.DataType}
			if primary.DataType == schemapb.DataType_Int64 {
				pk.Field = &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{Data: &schemapb.ScalarField_LongData{LongData: &schemapb.LongArray{Data: data.GetIds().GetIntId().GetData()}}}}
			} else {
				pk.Field = &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{Data: &schemapb.ScalarField_StringData{StringData: &schemapb.StringArray{Data: data.GetIds().GetStrId().GetData()}}}}
			}
			fields = append(fields, pk)
		}
	}
	scores := make([]float64, len(data.GetScores()))
	for i, score := range data.GetScores() {
		scores[i] = sqlparser.SQLDistance(distance.Operator, score)
	}
	fields = append(fields, &schemapb.FieldData{FieldId: plan.DistanceID, Type: schemapb.DataType_Double, Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{Data: &schemapb.ScalarField_DoubleData{DoubleData: &schemapb.DoubleArray{Data: scores}}}}})
	projected, err := sqlquery.Project(ctx, plan.Bound, &milvuspb.QueryResults{FieldsData: fields}, request.Timezone, paramtable.Get().QuotaConfig.MaxOutputSize.GetAsInt64())
	if err != nil {
		return fail(err)
	}
	allowInt64, _ := strconv.ParseBool(c.Request.Header.Get(HTTPHeaderAllowInt64))
	if err := dql.FormatTimestamptzFields(projected.FieldsData, request.Timezone); err != nil {
		return fail(err)
	}
	rows, err := buildQueryResp(0, projected.OutputFields, projected.FieldsData, nil, nil, allowInt64, schema)
	if err != nil {
		return fail(err)
	}
	HTTPReturnStream(c, http.StatusOK, gin.H{HTTPReturnCode: merr.Code(nil), HTTPReturnData: rows, HTTPReturnCost: proxy.GetCostValue(result.Status)})
	return response, nil
}
