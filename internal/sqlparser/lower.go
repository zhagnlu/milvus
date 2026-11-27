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

	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// AggregateInput maps a logical aggregate to columns in the native partial
// result. AVG has sum and count columns. DISTINCT uses a grouping-key column;
// its cardinality is determined only after all partial groups have been merged.
type AggregateInput struct {
	Columns       []int
	DistinctGroup int // -1 unless this aggregate is COUNT DISTINCT
}

// AggregatePlan keeps physical partial states separate from logical outputs.
// The native grouping includes distinct inputs, so the final stage may collapse
// multiple physical groups into one SQL group. No intermediate limit is safe.
type AggregatePlan struct {
	Bound      *BoundQuery
	Project    *planpb.SQLProject
	GroupIDs   []int64
	Aggregates []*planpb.Aggregate
	Inputs     []AggregateInput
}

func LowerAggregate(bound *BoundQuery, schema *schemapb.CollectionSchema, timezone string) (*AggregatePlan, error) {
	if bound == nil || schema == nil {
		return nil, merr.WrapErrParameterInvalidMsg("bound SQL and schema are required")
	}
	if len(bound.GroupKeys) == 0 && len(bound.Aggregates) == 0 {
		return nil, merr.WrapErrParameterInvalidMsg("query has no grouping or aggregation")
	}
	result := &AggregatePlan{Bound: bound, Project: &planpb.SQLProject{Version: 1, Timezone: timezone}}
	if bound.Predicate != nil {
		var err error
		result.Project.Predicate, err = LowerRowExpression(bound.Predicate)
		if err != nil {
			return nil, err
		}
	}
	nextID, err := nextQueryFieldID(schema)
	if err != nil {
		return nil, err
	}
	columns := make(map[string]int64)
	project := func(expr *BoundExpr) (int64, error) {
		wire, err := LowerRowExpression(expr)
		if err != nil {
			return 0, err
		}
		key, err := proto.MarshalOptions{Deterministic: true}.Marshal(wire)
		if err != nil {
			return 0, merr.WrapErrServiceInternalErr(err, "serialize SQL row expression")
		}
		if id, exists := columns[string(key)]; exists {
			return id, nil
		}
		if nextID == math.MaxInt64 {
			return 0, merr.WrapErrParameterInvalidMsg("query layout ID range exhausted")
		}
		id := nextID
		nextID++
		columns[string(key)] = id
		result.Project.Columns = append(result.Project.Columns, &planpb.SQLProjectionColumn{Id: id, Expression: wire})
		return id, nil
	}
	for _, group := range bound.GroupKeys {
		id, err := project(group)
		if err != nil {
			return nil, err
		}
		result.GroupIDs = append(result.GroupIDs, id)
	}
	// Extend the native key with DISTINCT arguments. This creates mergeable
	// exact equivalence classes rather than non-mergeable local cardinalities.
	for _, aggregate := range bound.Aggregates {
		input := AggregateInput{DistinctGroup: -1}
		if aggregate.Distinct {
			id, err := project(aggregate.Input)
			if err != nil {
				return nil, err
			}
			for i, groupID := range result.GroupIDs {
				if id == groupID {
					input.DistinctGroup = i
					break
				}
			}
			if input.DistinctGroup < 0 {
				input.DistinctGroup = len(result.GroupIDs)
				result.GroupIDs = append(result.GroupIDs, id)
			}
		}
		result.Inputs = append(result.Inputs, input)
	}
	for i, aggregate := range bound.Aggregates {
		if aggregate.Distinct {
			continue
		}
		input := aggregate.Input
		if input == nil {
			// COUNT(*) is COUNT(1). Requiring a projected field also makes
			// older workers reject this plan instead of ignoring its predicate.
			input = &BoundExpr{Op: "literal", Type: schemapb.DataType_Int64, Value: int64(1)}
		}
		fieldID, err := project(input)
		if err != nil {
			return nil, err
		}
		ops := map[string][]planpb.AggregateOp{
			"count": {planpb.AggregateOp_count},
			"sum":   {planpb.AggregateOp_sum},
			"avg":   {planpb.AggregateOp_sum, planpb.AggregateOp_count},
			"min":   {planpb.AggregateOp_min},
			"max":   {planpb.AggregateOp_max},
		}[aggregate.Name]
		if len(ops) == 0 {
			return nil, merr.WrapErrParameterInvalidMsg("unsupported aggregate %q", aggregate.Name)
		}
		for _, op := range ops {
			result.Inputs[i].Columns = append(result.Inputs[i].Columns, len(result.GroupIDs)+len(result.Aggregates))
			result.Aggregates = append(result.Aggregates, &planpb.Aggregate{Op: op, FieldId: fieldID})
		}
	}
	return result, nil
}

func nextQueryFieldID(schema *schemapb.CollectionSchema) (int64, error) {
	var maximum int64 = 99
	for _, field := range schema.Fields {
		maximum = max(maximum, field.FieldID)
	}
	for _, parent := range schema.StructArrayFields {
		maximum = max(maximum, parent.FieldID)
		for _, field := range parent.Fields {
			maximum = max(maximum, field.FieldID)
		}
	}
	if maximum == math.MaxInt64 {
		return 0, merr.WrapErrParameterInvalidMsg("query layout ID range exhausted")
	}
	return maximum + 1, nil
}

func LowerRowExpression(expr *BoundExpr) (*planpb.SQLScalarExpr, error) {
	if expr == nil || expr.Op == "slot" {
		return nil, merr.WrapErrParameterInvalidMsg("row expression cannot reference an aggregate output")
	}
	result := &planpb.SQLScalarExpr{
		Op: expr.Op, DataType: expr.Type, SourceFieldId: expr.FieldID, JsonPath: expr.Path,
	}
	if expr.Op == "extract" {
		result.TimeUnit, _ = expr.Value.(string)
	} else if expr.Op == "literal" && expr.Value != nil {
		result.Literal = &planpb.GenericValue{}
		switch value := expr.Value.(type) {
		case int64:
			result.Literal.Val = &planpb.GenericValue_Int64Val{Int64Val: value}
		case float64:
			result.Literal.Val = &planpb.GenericValue_FloatVal{FloatVal: value}
		case string:
			result.Literal.Val = &planpb.GenericValue_StringVal{StringVal: value}
		case bool:
			result.Literal.Val = &planpb.GenericValue_BoolVal{BoolVal: value}
		default:
			return nil, merr.WrapErrServiceInternalMsg("unexpected SQL literal %T", value)
		}
	}
	for _, arg := range expr.Args {
		child, err := LowerRowExpression(arg)
		if err != nil {
			return nil, err
		}
		result.Args = append(result.Args, child)
	}
	return result, nil
}
