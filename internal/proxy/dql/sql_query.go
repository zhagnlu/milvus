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
	"math"

	"github.com/milvus-io/milvus/internal/agg"
	"github.com/milvus-io/milvus/internal/parser/planparserv2"
	"github.com/milvus-io/milvus/internal/sqlparser"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// SQLQueryKey transports SQL over the existing Query request. This retains the
// collection-level authorization and shared DQL lifecycle without a private
// public-proto fork or a second copy of QueryTask.
const SQLQueryKey = "sql"

func (t *QueryTask) prepareSQL(visitorArgs *planparserv2.ParserVisitorArgs) (bool, error) {
	var statement string
	found := false
	for _, parameter := range t.request.GetQueryParams() {
		if parameter.Key == SQLQueryKey {
			if found {
				return false, merr.WrapErrParameterInvalidMsg("duplicate SQL query parameter")
			}
			statement = parameter.Value
			found = true
		}
	}
	if !found {
		return false, nil
	}
	if statement == "" {
		return false, merr.WrapErrParameterInvalidMsg("SQL statement must not be empty")
	}
	if t.sqlBound != nil {
		return t.sqlAggregate != nil, nil
	}
	if t.queryParams.isIterator || t.reQuery || t.internalTask {
		return false, merr.WrapErrParameterInvalidMsg("SQL is not supported for internal queries or query iterators")
	}
	components, err := sqlparser.ExtractSqlComponents(statement)
	if err != nil {
		return false, err
	}
	if components.Collection != t.request.CollectionName {
		return false, merr.WrapErrParameterInvalidMsg("SQL collection must match the authorized Query collection")
	}
	if components.VectorSearch != nil {
		return false, merr.WrapErrParameterInvalidMsg("vector SQL must use the SQL search endpoint")
	}
	t.request.ExprTemplateValues, err = sqlparser.BuildSQLParamTemplateValues(t.request.ExprTemplateValues)
	if err != nil {
		return false, err
	}
	t.sqlBound, err = sqlparser.BindWithParams(components, t.schema.CollectionSchema, t.request.ExprTemplateValues)
	if err != nil {
		return false, err
	}
	t.request.Expr = ""
	t.queryParams.limit = typeutil.Unlimited
	t.queryParams.offset = 0
	t.queryParams.orderByFields = nil
	t.queryParams.groupByFields = nil
	t.Limit = typeutil.Unlimited
	if len(t.sqlBound.GroupKeys) == 0 && len(t.sqlBound.Aggregates) == 0 {
		// Row projection does not change cardinality. With no ORDER BY, a
		// native limit can bound materialization before final SQL pagination.
		if t.sqlBound.Predicate == nil && len(components.OrderBy) == 0 && components.Limit > 0 {
			if components.Offset > math.MaxInt64-components.Limit {
				return false, merr.WrapErrParameterInvalidMsg("SQL pagination overflow")
			}
			t.Limit = components.Offset + components.Limit
			t.queryParams.limit = t.Limit
		}
		t.request.OutputFields = nil
		for _, id := range t.sqlBound.SourceFields {
			field, err := t.schema.SchemaHelper.GetFieldFromID(id)
			if err != nil {
				return false, err
			}
			t.request.OutputFields = append(t.request.OutputFields, field.Name)
		}
		if len(t.request.OutputFields) == 0 {
			pk, err := t.schema.GetPkField()
			if err != nil {
				return false, err
			}
			t.request.OutputFields = []string{pk.Name}
		}
		return false, nil
	}
	t.sqlAggregate, err = sqlparser.LowerAggregate(t.sqlBound, t.schema.CollectionSchema, t.resolvedTimezoneStr)
	if err != nil {
		return false, err
	}
	t.plan, err = planparserv2.CreateRetrievePlanArgs(t.schema.SchemaHelper, t.request.Expr, t.request.ExprTemplateValues, visitorArgs)
	if err != nil {
		return false, WrapPlanCreationError(err, "failed to create SQL predicate plan")
	}
	t.GroupByFieldIds = t.sqlAggregate.GroupIDs
	t.Aggregates = t.sqlAggregate.Aggregates
	t.SqlProject = t.sqlAggregate.Project
	if predicate := sqlparser.LowerJSONPredicate(t.sqlBound.Predicate); predicate != nil {
		t.plan.GetQuery().Predicates = predicate
		t.SqlProject.Predicate = nil
	}
	t.OutputFieldsId = nil
	t.plan.OutputFieldIds = nil
	t.plan.GetQuery().GroupByFieldIds = t.GroupByFieldIds
	t.plan.GetQuery().Aggregates = t.Aggregates
	t.plan.GetQuery().SqlProject = t.SqlProject
	for _, aggregate := range t.Aggregates {
		op, err := agg.FromPB(aggregate)
		if err != nil {
			return false, err
		}
		t.userAggregates = append(t.userAggregates, op)
	}
	for _, output := range t.sqlBound.Outputs {
		t.userOutputFields = append(t.userOutputFields, output.Name)
	}
	return true, nil
}
