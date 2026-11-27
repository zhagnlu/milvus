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
	"math"
	"strconv"

	pg_query "github.com/pganalyze/pg_query_go/v6"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

type SearchPlan struct {
	Bound       *BoundQuery
	VectorField *schemapb.FieldSchema
	Distance    *VectorDistExpr
	DistanceID  int64
	TopK        int64
}

func BindSearch(comp *SqlComponents, schema *schemapb.CollectionSchema) (*SearchPlan, error) {
	return BindSearchWithParams(comp, schema, nil)
}

func BindSearchWithParams(comp *SqlComponents, schema *schemapb.CollectionSchema, params map[string]*schemapb.TemplateValue) (*SearchPlan, error) {
	if comp == nil || schema == nil || comp.VectorSearch == nil {
		return nil, merr.WrapErrParameterInvalidMsg("missing vector SQL definition")
	}
	if len(comp.GroupBy) != 0 || len(comp.OrderBy) != 1 || comp.OrderBy[0].Desc || comp.Limit < 0 {
		return nil, merr.WrapErrParameterInvalidMsg("vector SQL requires one ascending distance ORDER BY and an explicit LIMIT, without GROUP BY")
	}
	if comp.Offset > math.MaxInt64-comp.Limit {
		return nil, merr.WrapErrParameterInvalidMsg("SQL pagination overflow")
	}
	var selected *SelectItem
	distancePosition := 0
	for i := range comp.SelectItems {
		item := &comp.SelectItems[i]
		if item.Type == SelectVectorDist {
			if selected != nil {
				return nil, merr.WrapErrParameterInvalidMsg("only one vector distance is supported")
			}
			selected, distancePosition = item, i
		}
	}
	if selected == nil {
		return nil, merr.WrapErrParameterInvalidMsg("missing SELECT distance")
	}
	normalizer := &binder{query: &BoundQuery{Components: comp}}
	order := comp.OrderBy[0]
	match := selected.Alias != "" && order.Ref == selected.Alias
	if constant := order.RawExpr.GetAConst(); constant != nil && constant.GetIval() != nil {
		match = int(constant.GetIval().Ival) == distancePosition+1
	}
	match = match || normalizer.nodeKey(order.RawExpr) == normalizer.nodeKey(selected.RawExpr)
	if !match {
		return nil, merr.WrapErrParameterInvalidMsg("vector SQL must order by the selected distance")
	}
	if selected.Distance.Column.Qualifier != "" && !normalizer.validQualifier(selected.Distance.Column.Qualifier) {
		return nil, merr.WrapErrParameterInvalidMsg("unknown vector field qualifier")
	}
	if len(selected.Distance.Column.NestedPath) > 0 {
		return nil, merr.WrapErrParameterInvalidMsg("vector SQL requires a vector field")
	}
	nextID, err := nextQueryFieldID(schema)
	if err != nil {
		return nil, err
	}
	plan := &SearchPlan{Distance: selected.Distance, TopK: comp.Offset + comp.Limit, DistanceID: nextID}
	for _, field := range schema.Fields {
		if field.Name == comp.VectorSearch.FieldName {
			plan.VectorField = field
		}
	}
	if plan.VectorField == nil {
		return nil, merr.WrapErrParameterInvalidMsg("unknown vector field %q", comp.VectorSearch.FieldName)
	}
	if plan.TopK == 0 {
		plan.TopK = 1
	}
	name := "__sql_distance_" + strconv.FormatInt(plan.DistanceID, 10)
	for _, field := range schema.Fields {
		if field.Name == name {
			return nil, merr.WrapErrParameterInvalidMsg("SQL temporary name conflicts with schema")
		}
	}
	column := &pg_query.Node{Node: &pg_query.Node_ColumnRef{ColumnRef: &pg_query.ColumnRef{Fields: []*pg_query.Node{{Node: &pg_query.Node_String_{String_: &pg_query.String{Sval: name}}}}}}}
	modified := *comp
	modified.SelectItems = append([]SelectItem(nil), comp.SelectItems...)
	modified.SelectItems[distancePosition] = SelectItem{Type: SelectColumn, Column: &ColumnRef{FieldName: name}, Alias: selected.Alias, RawExpr: column}
	if selected.Alias == "" {
		modified.SelectItems[distancePosition].Alias = "distance"
	}
	modified.OrderBy = []OrderItem{{RawExpr: column}}
	modifiedSchema := proto.Clone(schema).(*schemapb.CollectionSchema)
	modifiedSchema.Fields = append(modifiedSchema.Fields, &schemapb.FieldSchema{FieldID: plan.DistanceID, Name: name, DataType: schemapb.DataType_Double, IsDynamic: true})
	plan.Bound, err = BindWithParams(&modified, modifiedSchema, params)
	if err != nil {
		return nil, err
	}
	if len(plan.Bound.Aggregates) > 0 {
		return nil, merr.WrapErrParameterInvalidMsg("aggregation with vector search is not supported")
	}
	if !nativeSearchPredicate(plan.Bound.Predicate) {
		return nil, merr.WrapErrParameterInvalidMsg("vector SQL filters require scalar fields and literals; JSON extraction, casts and arithmetic require query execution")
	}
	// Search applies the scalar predicate before selecting nearest neighbors.
	plan.Bound.Predicate = nil
	return plan, nil
}

func nativeSearchPredicate(expr *BoundExpr) bool {
	if expr == nil {
		return true
	}
	switch expr.Op {
	case "column", "literal", "and", "or", "not", "is_null", "is_not_null", "in", "like", "=", "<>", "<", "<=", ">", ">=":
	default:
		return false
	}
	for _, arg := range expr.Args {
		if !nativeSearchPredicate(arg) {
			return false
		}
	}
	return true
}

// SQLDistance converts Milvus metric scores to pgvector distance conventions.
func SQLDistance(op VectorDistOp, score float32) float64 {
	switch op {
	case VectorDistL2:
		return math.Sqrt(math.Max(0, float64(score)))
	case VectorDistCosine:
		return 1 - float64(score)
	case VectorDistIP:
		return -float64(score)
	default:
		return float64(score)
	}
}
