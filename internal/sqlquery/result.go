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
	"math"
	"time"

	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/agg"
	"github.com/milvus-io/milvus/internal/sqlparser"
	"github.com/milvus-io/milvus/internal/util/queryutil"
	"github.com/milvus-io/milvus/internal/util/reduce/orderby"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

type groupState struct {
	keys           []any
	states         []*agg.FieldValue
	distinct       []map[any]struct{}
	stringDistinct []map[string]struct{}
	distinctCount  int64
}

// A full merge of multiple non-empty partials deduplicates all physical keys.
// With one extra non-floating key, each surviving non-NULL key therefore adds
// exactly one distinct value to its SQL group. Single-input forwarding and
// multiple distinct dimensions keep the set path. Floating keys keep it too:
// their bitwise hashes need not identify Go-equal values such as signed zero.
func canCountDistinctGroups(plan *sqlparser.AggregatePlan, fields []*schemapb.FieldData, partials []*agg.PartialAggregationResult) bool {
	baseKeys := len(plan.Bound.GroupKeys)
	if len(plan.GroupIDs) != baseKeys+1 {
		return false
	}
	for _, input := range plan.Inputs {
		if input.DistinctGroup >= 0 && input.DistinctGroup != baseKeys {
			return false
		}
	}
	switch fields[baseKeys].GetType() {
	case schemapb.DataType_Bool, schemapb.DataType_Int8, schemapb.DataType_Int16,
		schemapb.DataType_Int32, schemapb.DataType_Int64, schemapb.DataType_Timestamptz,
		schemapb.DataType_VarChar, schemapb.DataType_String:
	default:
		return false
	}
	values := 0
	for _, partial := range partials {
		if !partial.IsEmpty() {
			values++
		}
	}
	return values > 1
}

// ReduceAggregate preserves native partial states through the last merge, then
// collapses DISTINCT key extensions, finalizes AVG, evaluates output expressions
// and applies ORDER BY/OFFSET/LIMIT. Neither QN nor Delegator truncates groups.
func ReduceAggregate(ctx context.Context, plan *sqlparser.AggregatePlan, schema *schemapb.CollectionSchema, results []*internalpb.RetrieveResults, maxBytes int64) (*milvuspb.QueryResults, error) {
	schema, err := queryutil.WithSQLProjectionSchema(schema, plan.Project)
	if err != nil {
		return nil, err
	}
	zone, err := location(plan.Project.Timezone)
	if err != nil {
		return nil, err
	}
	var inputSize int64
	for _, result := range results {
		inputSize += int64(proto.Size(result))
	}
	if maxBytes > 0 && inputSize > maxBytes {
		return nil, merr.WrapErrParameterInvalidMsg("SQL partial results exceed the output size limit")
	}
	reducer := agg.NewGroupAggReducer(plan.GroupIDs, plan.Aggregates, -1, schema)
	partials, err := reducer.DecodeInternalResults(results)
	if err != nil {
		return nil, err
	}
	merged, err := reducer.Merge(ctx, partials)
	if err != nil {
		return nil, err
	}
	final, err := reducer.Finalize(merged)
	if err != nil {
		return nil, err
	}
	fields := final.GetFieldDatas()
	accessors := make([]agg.FieldAccessor, len(fields))
	rows := 0
	for i, field := range fields {
		accessors[i], err = agg.NewFieldAccessor(field.Type)
		if err != nil {
			return nil, err
		}
		accessors[i].SetVals(field)
		rows = accessors[i].RowCount()
	}
	needsDistinct := false
	for _, input := range plan.Inputs {
		needsDistinct = needsDistinct || input.DistinctGroup >= 0
	}
	if !needsDistinct {
		// Native reduction has already produced one row per SQL group. Avoid
		// regrouping the same keys and rebuilding per-group accumulator maps.
		return finishMergedAggregate(ctx, plan, accessors, rows, zone, maxBytes)
	}
	baseKeys := len(plan.Bound.GroupKeys)
	countDistinctGroups := canCountDistinctGroups(plan, fields, partials)
	stringDistinctReaders := make([]*agg.StringFieldAccessor, len(plan.Inputs))
	unboxedKeys := make([]bool, len(fields))
	hasStringDistinct := false
	for i, input := range plan.Inputs {
		if input.DistinctGroup >= 0 {
			if reader, ok := accessors[input.DistinctGroup].(*agg.StringFieldAccessor); ok {
				stringDistinctReaders[i] = reader
				hasStringDistinct = true
				// Extension keys are used only for DISTINCT, while SQL group
				// keys still need their boxed values for final group identity.
				if input.DistinctGroup >= baseKeys {
					unboxedKeys[input.DistinctGroup] = true
				}
			}
		}
	}
	ops := make([]agg.AggregateBase, len(plan.Aggregates))
	for i, definition := range plan.Aggregates {
		ops[i], err = agg.FromPB(definition)
		if err != nil {
			return nil, err
		}
	}
	newGroup := func(keys []any) *groupState {
		group := &groupState{keys: keys, states: make([]*agg.FieldValue, len(ops))}
		for i, definition := range plan.Aggregates {
			group.states[i] = agg.NewNullFieldValue()
			if definition.Op == planpb.AggregateOp_count {
				group.states[i] = agg.NewFieldValue(int64(0))
			}
		}
		if !countDistinctGroups {
			group.distinct = make([]map[any]struct{}, len(plan.Inputs))
			if hasStringDistinct {
				group.stringDistinct = make([]map[string]struct{}, len(plan.Inputs))
			}
			for i, input := range plan.Inputs {
				if input.DistinctGroup >= 0 {
					if stringDistinctReaders[i] != nil {
						group.stringDistinct[i] = make(map[string]struct{})
					} else {
						group.distinct[i] = make(map[any]struct{})
					}
				}
			}
		}
		return group
	}
	groups := make(map[string]*groupState)
	// A single string key already has an exact map representation. Avoid
	// boxing and encoding it on every physical DISTINCT group. NULL has its
	// own group so it remains distinct from an empty string or any byte value.
	var stringGroupReader *agg.StringFieldAccessor
	var nullStringGroup *groupState
	if baseKeys == 1 {
		stringGroupReader, _ = accessors[0].(*agg.StringFieldAccessor)
		if stringGroupReader != nil {
			unboxedKeys[0] = true
		}
	}
	ordered := make([]*groupState, 0)
	if baseKeys == 0 {
		group := newGroup(nil)
		groups[""] = group
		ordered = append(ordered, group)
	}
	var memory int64
	values := make([]any, len(fields))
	keyBuffer := make([]byte, 0, 128)
	for row := 0; row < rows; row++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		for i, accessor := range accessors {
			values[i] = nil
			if !unboxedKeys[i] && !accessor.IsNullAt(row) {
				values[i] = accessor.ValAt(row)
			}
		}
		key := ""
		var group *groupState
		nullStringKey := false
		if stringGroupReader != nil {
			nullStringKey = stringGroupReader.IsNullAt(row)
			if nullStringKey {
				group = nullStringGroup
			} else {
				key = stringGroupReader.StringAt(row)
				group = groups[key]
			}
		} else if baseKeys > 0 {
			encoded, err := appendSQLGroupKey(keyBuffer[:0], values[:baseKeys])
			if err != nil {
				return nil, merr.WrapErrParameterInvalidErr(err, "invalid SQL group key")
			}
			keyBuffer = encoded
			key = string(encoded)
			group = groups[key]
		} else {
			group = groups[key]
		}
		if group == nil {
			if stringGroupReader != nil && !nullStringKey {
				values[0] = key
			}
			group = newGroup(append([]any(nil), values[:baseKeys]...))
			keyBytes := len(key)
			if stringGroupReader != nil {
				// Preserve the generic path's conservative accounting, even
				// though only one encoded buffer is now needed per query.
				encoded, err := appendSQLGroupKey(keyBuffer[:0], group.keys)
				if err != nil {
					return nil, merr.Wrap(err, "invalid SQL string group key")
				}
				keyBuffer = encoded
				keyBytes = len(encoded)
			}
			if stringGroupReader != nil && nullStringKey {
				nullStringGroup = group
			} else {
				groups[key] = group
			}
			ordered = append(ordered, group)
			memory += int64(keyBytes + 128 + 32*len(ops))
		}
		for i, op := range ops {
			value := values[len(plan.GroupIDs)+i]
			if value != nil {
				if err := op.Update(group.states[i], agg.NewFieldValue(value)); err != nil {
					return nil, err
				}
			}
		}
		if countDistinctGroups {
			if !accessors[baseKeys].IsNullAt(row) {
				group.distinctCount++
				// Retain the existing conservative resource accounting even
				// though these sets no longer need to be allocated.
				for _, input := range plan.Inputs {
					if input.DistinctGroup >= 0 {
						if reader, ok := accessors[baseKeys].(*agg.StringFieldAccessor); ok {
							memory += int64(len(reader.StringAt(row))) + 64
						} else {
							memory += valueSize(values[baseKeys]) + 64
						}
					}
				}
			}
		} else {
			for i, input := range plan.Inputs {
				if input.DistinctGroup < 0 {
					continue
				}
				if reader := stringDistinctReaders[i]; reader != nil {
					if !reader.IsNullAt(row) {
						text := reader.StringAt(row)
						if _, exists := group.stringDistinct[i][text]; !exists {
							group.stringDistinct[i][text] = struct{}{}
							memory += int64(len(text)) + 64
						}
					}
					continue
				}
				value := values[input.DistinctGroup]
				if value == nil {
					continue
				}
				if _, exists := group.distinct[i][value]; !exists {
					group.distinct[i][value] = struct{}{}
					memory += valueSize(value) + 64
				}
			}
		}
		if maxBytes > 0 && memory > maxBytes {
			return nil, merr.WrapErrParameterInvalidMsg("SQL aggregation exceeds the output size limit")
		}
	}
	evaluations := make([]evaluation, 0, len(ordered))
	for _, group := range ordered {
		slots := append(append([]any(nil), group.keys...), make([]any, len(plan.Bound.Aggregates))...)
		for i, aggregate := range plan.Bound.Aggregates {
			input := plan.Inputs[i]
			var value any
			if input.DistinctGroup >= 0 {
				if countDistinctGroups {
					value = group.distinctCount
				} else if stringDistinctReaders[i] != nil {
					value = int64(len(group.stringDistinct[i]))
				} else {
					value = int64(len(group.distinct[i]))
				}
			} else if aggregate.Name == "avg" {
				sum := group.states[input.Columns[0]-len(plan.GroupIDs)].Value()
				count := group.states[input.Columns[1]-len(plan.GroupIDs)].Value().(int64)
				if count > 0 && sum != nil {
					numerator, err := number(sum)
					if err != nil {
						return nil, err
					}
					value = numerator / float64(count)
				}
			} else {
				value = group.states[input.Columns[0]-len(plan.GroupIDs)].Value()
			}
			slots[aggregate.Output.Slot] = value
		}
		evaluations = append(evaluations, evaluation{slots: slots, zone: zone})
	}
	return finish(ctx, plan.Bound, evaluations, nil, maxBytes)
}

// finishMergedAggregate finalizes rows that need no DISTINCT key collapse.
// This is reached only after the distributed reducer has merged every shard.
func finishMergedAggregate(ctx context.Context, plan *sqlparser.AggregatePlan, accessors []agg.FieldAccessor, rowCount int, zone *time.Location, maxBytes int64) (*milvuspb.QueryResults, error) {
	valueAt := func(column, row int) any {
		if accessors[column].IsNullAt(row) {
			return nil
		}
		return accessors[column].ValAt(row)
	}
	evaluations := make([]evaluation, 0, rowCount)
	baseKeys := len(plan.Bound.GroupKeys)
	for row := 0; row < rowCount; row++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		slots := make([]any, baseKeys+len(plan.Bound.Aggregates))
		for key := 0; key < baseKeys; key++ {
			slots[key] = valueAt(key, row)
		}
		for i, aggregate := range plan.Bound.Aggregates {
			input := plan.Inputs[i]
			value := valueAt(input.Columns[0], row)
			if aggregate.Name == "avg" {
				countValue := valueAt(input.Columns[1], row)
				count := int64(0)
				if countValue != nil {
					count = countValue.(int64)
				}
				if count > 0 && value != nil {
					numerator, err := number(value)
					if err != nil {
						return nil, err
					}
					value = numerator / float64(count)
				} else {
					value = nil
				}
			} else if aggregate.Name == "count" && value == nil {
				value = int64(0)
			}
			slots[aggregate.Output.Slot] = value
		}
		evaluations = append(evaluations, evaluation{slots: slots, zone: zone})
	}
	return finish(ctx, plan.Bound, evaluations, nil, maxBytes)
}

func Project(ctx context.Context, bound *sqlparser.BoundQuery, input *milvuspb.QueryResults, timezone string, maxBytes int64) (*milvuspb.QueryResults, error) {
	zone, err := location(timezone)
	if err != nil {
		return nil, err
	}
	sources := make(map[int64]*schemapb.FieldData)
	getters := make(map[int64]func(int) any)
	rows := -1
	fields := make([]*schemapb.FieldData, len(input.FieldsData))
	vectorIndices := make([][]int, len(fields))
	for i, original := range input.FieldsData {
		field, err := expandScalar(original)
		if err != nil {
			return nil, err
		}
		if field.Type == schemapb.DataType_Timestamptz && field.GetScalars().GetStringData() != nil {
			// Search formats timestamps before returning to the REST adapter.
			// Recover instants before evaluating SQL temporal expressions.
			values := make([]int64, len(field.GetScalars().GetStringData().Data))
			for row, value := range field.GetScalars().GetStringData().Data {
				if valid := typeutil.GetFieldDataValidData(field); len(valid) > 0 && !valid[row] {
					continue
				}
				instant, err := time.Parse(time.RFC3339Nano, value)
				if err != nil {
					return nil, merr.WrapErrServiceInternalErr(err, "invalid timestamp in SQL source result")
				}
				values[row] = instant.UnixMicro()
			}
			field = proto.Clone(field).(*schemapb.FieldData)
			field.GetScalars().Data = &schemapb.ScalarField_TimestamptzData{TimestamptzData: &schemapb.TimestamptzArray{Data: values}}
		}
		fields[i] = field
		sources[field.FieldId] = field
		getters[field.FieldId] = typeutil.GetDataIterator(field)
		if field.Type == schemapb.DataType_JSON {
			getters[field.FieldId] = func(row int) any {
				if valid := typeutil.GetFieldDataValidData(field); len(valid) > 0 && !valid[row] {
					return nil
				}
				return field.GetScalars().GetJsonData().Data[row]
			}
		}
		if typeutil.IsCompactNullableVectorFieldData(field) {
			vectorIndices[i], _ = typeutil.BuildNullableVectorDataIndices(typeutil.GetFieldDataValidData(field))
		}
		count, err := funcutil.GetNumRowOfFieldData(field)
		if err != nil {
			return nil, err
		}
		if rows >= 0 && rows != int(count) {
			return nil, merr.WrapErrServiceInternalMsg("SQL source columns have inconsistent row counts")
		}
		rows = int(count)
	}
	if rows < 0 {
		rows = 0
	}
	if maxBytes > 0 && int64(proto.Size(input)) > maxBytes {
		return nil, merr.WrapErrParameterInvalidMsg("SQL source results exceed the output size limit")
	}
	evaluations := make([]evaluation, 0, rows)
	selected := make([]int64, 0, rows)
	for row := 0; row < rows; row++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		current := evaluation{columns: make(map[int64]any, len(getters)), zone: zone}
		for id, get := range getters {
			current.columns[id] = get(row)
		}
		if bound.Predicate != nil {
			match, err := current.eval(bound.Predicate)
			if err != nil {
				return nil, err
			}
			if match != true {
				continue
			}
		}
		selected = append(selected, int64(row))
		evaluations = append(evaluations, current)
	}
	if len(selected) != rows {
		subset := typeutil.PrepareResultFieldData(fields, int64(len(selected)))
		indices := make([]int64, len(fields))
		for _, row := range selected {
			for i, mapping := range vectorIndices {
				indices[i] = row
				if mapping != nil {
					indices[i] = int64(mapping[row])
				}
			}
			typeutil.AppendFieldData(subset, fields, row, indices...)
		}
		for _, field := range subset {
			if field != nil {
				sources[field.FieldId] = field
			}
		}
	}
	return finish(ctx, bound, evaluations, sources, maxBytes)
}

// Query returns full scalar arrays, while Search may return compact nullable
// arrays. Normalize only compact scalars so sorting and slicing use row indices.
func expandScalar(field *schemapb.FieldData) (*schemapb.FieldData, error) {
	if field == nil {
		return nil, merr.WrapErrServiceInternalMsg("missing SQL source column")
	}
	valid := typeutil.GetFieldDataValidData(field)
	if field.GetScalars() == nil || len(valid) == 0 {
		return field, nil
	}
	scalar := field.GetScalars().ProtoReflect()
	kind := scalar.WhichOneof(scalar.Descriptor().Oneofs().ByName("data"))
	if kind == nil {
		return nil, merr.WrapErrServiceInternalMsg("missing SQL scalar data")
	}
	array := scalar.Get(kind).Message()
	data := array.Descriptor().Fields().ByName("data")
	if data == nil || !data.IsList() {
		return nil, merr.WrapErrServiceInternalMsg("invalid SQL scalar array")
	}
	values := array.Get(data).List()
	if values.Len() == len(valid) {
		return field, nil
	}
	count := 0
	for _, present := range valid {
		if present {
			count++
		}
	}
	if count != values.Len() {
		return nil, merr.WrapErrServiceInternalMsg("SQL scalar validity does not match its values")
	}
	result := proto.Clone(field).(*schemapb.FieldData)
	output := result.GetScalars().ProtoReflect().Get(kind).Message().Mutable(data).List()
	output.Truncate(0)
	next := 0
	for _, present := range valid {
		if present {
			output.Append(values.Get(next))
			next++
		} else {
			output.Append(output.NewElement())
		}
	}
	return result, nil
}

func location(timezone string) (*time.Location, error) {
	if timezone == "" {
		timezone = "UTC"
	}
	zone, err := time.LoadLocation(timezone)
	if err != nil {
		return nil, merr.WrapErrParameterInvalidErr(err, "invalid SQL timezone")
	}
	return zone, nil
}

func valueSize(value any) int64 {
	switch value := value.(type) {
	case string:
		return int64(len(value))
	case []byte:
		return int64(len(value))
	default:
		return 16
	}
}

func finish(ctx context.Context, bound *sqlparser.BoundQuery, rows []evaluation, sources map[int64]*schemapb.FieldData, maxBytes int64) (*milvuspb.QueryResults, error) {
	expressions := make([]*sqlparser.BoundExpr, 0, len(bound.Outputs)+len(bound.OrderBy))
	names := make([]string, 0, len(bound.Outputs))
	for _, output := range bound.Outputs {
		expressions = append(expressions, output.Expr)
		names = append(names, output.Name)
	}
	if bound.Components.Limit == 0 {
		return &milvuspb.QueryResults{Status: merr.Success(), OutputFields: names}, nil
	}
	for _, order := range bound.OrderBy {
		expressions = append(expressions, order.Expr)
	}
	fields := make([]*schemapb.FieldData, len(expressions))
	var total int64
	for i, expr := range expressions {
		name := ""
		if i < len(names) {
			name = names[i]
		}
		if expr.Op == "column" && sources[expr.FieldID] != nil {
			fields[i] = proto.Clone(sources[expr.FieldID]).(*schemapb.FieldData)
			fields[i].FieldName = name
			fields[i].FieldId = int64(i + 1)
			fields[i].IsDynamic = false
			total += int64(proto.Size(fields[i]))
			if maxBytes > 0 && total > maxBytes {
				return nil, merr.WrapErrParameterInvalidMsg("SQL result exceeds the output size limit")
			}
			continue
		}
		typ := expr.Type
		if typ == schemapb.DataType_None {
			typ = schemapb.DataType_VarChar
		}
		var err error
		fields[i], err = typeutil.GenEmptyFieldData(&schemapb.FieldSchema{FieldID: int64(i + 1), Name: name, DataType: typ, Nullable: true})
		if err != nil {
			return nil, err
		}
		for _, row := range rows {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			value, err := row.eval(expr)
			if err != nil {
				return nil, err
			}
			total += valueSize(value) + 1
			if maxBytes > 0 && total > maxBytes {
				return nil, merr.WrapErrParameterInvalidMsg("SQL result exceeds the output size limit")
			}
			if err := appendValue(fields[i], value); err != nil {
				return nil, err
			}
		}
	}
	result := &internalpb.RetrieveResults{FieldsData: fields}
	if len(bound.OrderBy) > 0 {
		orders := make([]*orderby.OrderByField, len(bound.OrderBy))
		positions := make([]int, len(orders))
		for i, order := range bound.OrderBy {
			positions[i] = len(bound.Outputs) + i
			orders[i] = orderby.NewOrderByField(int64(positions[i]+1), "", order.Expr.Type, orderby.WithAscending(!order.Descending), orderby.WithNullsFirst(order.NullsFirst))
		}
		// Apply Top-K only after all partial aggregates and computed ordering
		// expressions have been finalized globally. Saturate an overflowing
		// OFFSET+LIMIT to the existing unlimited sort.
		topK := int64(-1)
		if limit, offset := bound.Components.Limit, bound.Components.Offset; limit > 0 && offset <= math.MaxInt64-limit {
			topK = offset + limit
		}
		out, err := queryutil.NewOrderByLimitOperatorWithPositions(orders, positions, topK).Run(ctx, nil, result)
		if err != nil {
			return nil, err
		}
		result = out[0].(*internalpb.RetrieveResults)
	}
	out, err := queryutil.NewSliceOperator(bound.Components.Limit, bound.Components.Offset).Run(ctx, nil, result)
	if err != nil {
		return nil, err
	}
	result = out[0].(*internalpb.RetrieveResults)
	if len(result.FieldsData) != 0 {
		result.FieldsData = result.FieldsData[:len(bound.Outputs)]
	}
	return &milvuspb.QueryResults{Status: merr.Success(), FieldsData: result.FieldsData, OutputFields: names}, nil
}

func appendValue(field *schemapb.FieldData, value any) error {
	if field.Type == schemapb.DataType_JSON {
		var raw []byte
		if value != nil {
			raw = value.([]byte)
		}
		field.GetScalars().GetJsonData().Data = append(field.GetScalars().GetJsonData().Data, raw)
		typeutil.SetFieldDataValidData(field, append(typeutil.GetFieldDataValidData(field), value != nil))
		return nil
	}
	if value == nil {
		return agg.AssembleSingleValue(agg.NewNullFieldValue(), field)
	}
	value = normalize(value)
	switch field.Type {
	case schemapb.DataType_Int8, schemapb.DataType_Int16, schemapb.DataType_Int32:
		value = int32(value.(int64))
	case schemapb.DataType_Float:
		value = float32(value.(float64))
	}
	return agg.AssembleSingleValue(agg.NewFieldValue(value), field)
}
