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
	"fmt"
	"math"
	"strconv"
	"strings"

	pg_query "github.com/pganalyze/pg_query_go/v6"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// BoundExpr is independent of the PostgreSQL AST. FieldID identifies a source
// field; Slot identifies a value in this query's result layout. They are never
// interchangeable. Value holds a typed literal (int64, float64, string or bool).
type BoundExpr struct {
	Op       string
	Type     schemapb.DataType
	Nullable bool
	FieldID  int64
	Slot     int
	Path     []string
	Value    any
	Args     []*BoundExpr
}

type BoundAggregate struct {
	Name     string
	Distinct bool
	Input    *BoundExpr // nil only for COUNT(*)
	Output   *BoundExpr
}

type BoundOutput struct {
	Name string
	Expr *BoundExpr
}

type BoundOrder struct {
	Expr       *BoundExpr
	Descending bool
	NullsFirst bool
}

// BoundQuery separates row expressions from final expressions. GroupKeys and
// aggregate inputs execute before partial aggregation; Outputs and OrderBy
// execute after global aggregation. This is essential for temporal grouping.
type BoundQuery struct {
	Components   *SqlComponents
	GroupKeys    []*BoundExpr
	Aggregates   []BoundAggregate
	Outputs      []BoundOutput
	OrderBy      []BoundOrder
	SourceFields []int64
	Predicate    *BoundExpr
}

type binder struct {
	query      *BoundQuery
	fields     map[string]*schemapb.FieldSchema
	aliases    map[string]*pg_query.Node
	groups     map[string]*BoundExpr
	aggregates map[string]*BoundExpr
	sources    map[int64]bool
	nextSlot   int
	params     map[string]*schemapb.TemplateValue
}

// Bind resolves identifiers and types without fetching data or constructing a
// task. Unsupported expressions fail here, before any query is dispatched.
func Bind(comp *SqlComponents, schema *schemapb.CollectionSchema) (*BoundQuery, error) {
	return BindWithParams(comp, schema, nil)
}

func BindWithParams(comp *SqlComponents, schema *schemapb.CollectionSchema, params map[string]*schemapb.TemplateValue) (*BoundQuery, error) {
	if comp == nil || schema == nil {
		return nil, merr.WrapErrParameterInvalidMsg("SQL components and schema are required")
	}
	b := &binder{
		query: &BoundQuery{Components: comp}, fields: make(map[string]*schemapb.FieldSchema),
		aliases: make(map[string]*pg_query.Node), groups: make(map[string]*BoundExpr),
		aggregates: make(map[string]*BoundExpr), sources: make(map[int64]bool), params: params,
	}
	for _, field := range schema.Fields {
		b.fields[field.Name] = field
	}
	if comp.Where != nil {
		predicate, err := b.predicate(comp.Where.RawExpr)
		if err != nil {
			return nil, merr.Wrap(err, "WHERE")
		}
		b.query.Predicate = predicate
	}
	for _, item := range comp.SelectItems {
		if item.Alias != "" {
			if _, exists := b.aliases[item.Alias]; exists {
				return nil, merr.WrapErrParameterInvalidMsg("duplicate SELECT alias %q", item.Alias)
			}
			b.aliases[item.Alias] = item.RawExpr
		}
	}
	for _, group := range comp.GroupBy {
		node, err := b.resolveReference(group.RawExpr, false)
		if err != nil {
			return nil, err
		}
		expr, err := b.expression(node, false)
		if err != nil {
			return nil, merr.Wrap(err, "GROUP BY")
		}
		if !comparableType(expr.Type) {
			return nil, merr.WrapErrParameterInvalidMsg("GROUP BY requires a scalar comparable value")
		}
		key := b.nodeKey(node)
		if _, exists := b.groups[key]; !exists {
			b.groups[key] = &BoundExpr{Op: "slot", Slot: b.nextSlot, Type: expr.Type, Nullable: expr.Nullable}
			b.nextSlot++
			b.query.GroupKeys = append(b.query.GroupKeys, expr)
		}
	}
	for _, item := range comp.SelectItems {
		if item.Type == SelectStar {
			if len(schema.StructArrayFields) != 0 {
				return nil, merr.WrapErrParameterInvalidMsg("SQL SELECT * does not support struct-array fields; select scalar fields explicitly")
			}
			if len(comp.GroupBy) > 0 {
				return nil, merr.WrapErrParameterInvalidMsg("SELECT * is not supported with GROUP BY")
			}
			for _, field := range schema.Fields {
				if field.IsDynamic {
					continue
				}
				expr := b.column(field)
				b.query.Outputs = append(b.query.Outputs, BoundOutput{Name: field.Name, Expr: expr})
			}
			continue
		}
		expr, err := b.expression(item.RawExpr, true)
		if err != nil {
			return nil, merr.Wrap(err, "SELECT")
		}
		name := item.Alias
		if name == "" {
			if item.Column != nil {
				name = item.Column.MilvusExpr()
			} else {
				name = fmt.Sprintf("expr_%d", len(b.query.Outputs))
			}
		}
		b.query.Outputs = append(b.query.Outputs, BoundOutput{Name: name, Expr: expr})
	}
	outputNames := make(map[string]bool, len(b.query.Outputs))
	for _, output := range b.query.Outputs {
		if outputNames[output.Name] {
			return nil, merr.WrapErrParameterInvalidMsg("duplicate SQL output name %q; use distinct aliases", output.Name)
		}
		outputNames[output.Name] = true
	}
	for _, order := range comp.OrderBy {
		node, err := b.resolveReference(order.RawExpr, true)
		if err != nil {
			return nil, err
		}
		expr, err := b.expression(node, true)
		if err != nil {
			return nil, merr.Wrap(err, "ORDER BY")
		}
		if !comparableType(expr.Type) {
			return nil, merr.WrapErrParameterInvalidMsg("ORDER BY requires a scalar comparable value")
		}
		b.query.OrderBy = append(b.query.OrderBy, BoundOrder{Expr: expr, Descending: order.Desc, NullsFirst: order.NullsFirst})
	}
	if len(b.query.GroupKeys) > 0 || len(b.query.Aggregates) > 0 {
		for _, output := range b.query.Outputs {
			if containsSourceColumn(output.Expr) {
				return nil, merr.WrapErrParameterInvalidMsg("output %q must appear in GROUP BY or an aggregate", output.Name)
			}
		}
		for _, order := range b.query.OrderBy {
			if containsSourceColumn(order.Expr) {
				return nil, merr.WrapErrParameterInvalidMsg("ORDER BY expression must appear in GROUP BY or an aggregate")
			}
		}
	}
	return b.query, nil
}

func (b *binder) nodeKey(node *pg_query.Node) string {
	// Locations differ between SELECT and GROUP BY. PG AST locations are not
	// part of an expression's identity; normalize them recursively on a clone.
	if node == nil {
		return ""
	}
	copy := proto.Clone(node)
	b.normalizeIdentity(copy.ProtoReflect())
	data, _ := proto.MarshalOptions{Deterministic: true}.Marshal(copy)
	return string(data)
}

func (b *binder) normalizeIdentity(message protoreflect.Message) {
	// A valid table qualifier does not change the identity of a source column.
	if column, ok := message.Interface().(*pg_query.ColumnRef); ok && len(column.Fields) == 2 {
		if qualifier := column.Fields[0].GetString_(); qualifier != nil && b.validQualifier(qualifier.Sval) {
			column.Fields = column.Fields[1:]
		}
	}
	message.Range(func(field protoreflect.FieldDescriptor, value protoreflect.Value) bool {
		if strings.HasSuffix(string(field.Name()), "location") {
			message.Clear(field)
			return true
		}
		if field.Kind() != protoreflect.MessageKind {
			return true
		}
		if field.IsList() {
			list := value.List()
			for i := 0; i < list.Len(); i++ {
				b.normalizeIdentity(list.Get(i).Message())
			}
		} else if !field.IsMap() {
			b.normalizeIdentity(value.Message())
		}
		return true
	})
}

func (b *binder) resolveReference(node *pg_query.Node, order bool) (*pg_query.Node, error) {
	if constant := node.GetAConst(); constant != nil && constant.GetIval() != nil {
		position := int(constant.GetIval().Ival)
		if position < 1 || position > len(b.query.Components.SelectItems) {
			return nil, merr.WrapErrParameterInvalidMsg("GROUP BY / ORDER BY position %d is out of range", position)
		}
		item := b.query.Components.SelectItems[position-1]
		if item.Type == SelectStar {
			return nil, merr.WrapErrParameterInvalidMsg("positional references to SELECT * are not supported")
		}
		return item.RawExpr, nil
	}
	cr := node.GetColumnRef()
	if cr == nil || len(cr.Fields) != 1 {
		return node, nil
	}
	name := cr.Fields[0].GetString_().GetSval()
	// PostgreSQL resolves GROUP BY ambiguity to input columns and ORDER BY
	// ambiguity to output aliases.
	if !order && b.fields[name] != nil {
		return node, nil
	}
	if alias := b.aliases[name]; alias != nil {
		return alias, nil
	}
	return node, nil
}

func (b *binder) validQualifier(qualifier string) bool {
	if b.query.Components.TableAlias != "" {
		return qualifier == b.query.Components.TableAlias
	}
	return qualifier == b.query.Components.Collection
}

func (b *binder) column(field *schemapb.FieldSchema) *BoundExpr {
	if !b.sources[field.FieldID] {
		b.sources[field.FieldID] = true
		b.query.SourceFields = append(b.query.SourceFields, field.FieldID)
	}
	return &BoundExpr{Op: "column", Type: field.DataType, Nullable: field.Nullable, FieldID: field.FieldID}
}

func (b *binder) expression(node *pg_query.Node, final bool) (*BoundExpr, error) {
	if node == nil {
		return nil, merr.WrapErrParameterInvalidMsg("missing expression")
	}
	if parameter := node.GetParamRef(); parameter != nil {
		value := b.params[SQLParamTemplateName(int(parameter.Number))]
		if value == nil {
			return nil, merr.WrapErrParameterInvalidMsg("missing SQL parameter $%d", parameter.Number)
		}
		literal := &BoundExpr{Op: "literal"}
		switch value := value.Val.(type) {
		case *schemapb.TemplateValue_Int64Val:
			literal.Type, literal.Value = schemapb.DataType_Int64, value.Int64Val
		case *schemapb.TemplateValue_FloatVal:
			if math.IsNaN(value.FloatVal) || math.IsInf(value.FloatVal, 0) {
				return nil, merr.WrapErrParameterInvalidMsg("SQL parameters must be finite")
			}
			literal.Type, literal.Value = schemapb.DataType_Double, value.FloatVal
		case *schemapb.TemplateValue_StringVal:
			literal.Type, literal.Value = schemapb.DataType_VarChar, value.StringVal
		case *schemapb.TemplateValue_BoolVal:
			literal.Type, literal.Value = schemapb.DataType_Bool, value.BoolVal
		default:
			return nil, merr.WrapErrParameterInvalidMsg("SQL parameter must be scalar")
		}
		return literal, nil
	}
	if final {
		if group := b.groups[b.nodeKey(node)]; group != nil {
			return group, nil
		}
	}
	if cast := node.GetTypeCast(); cast != nil {
		if cast.TypeName == nil || len(cast.TypeName.Typmods) != 0 || len(cast.TypeName.ArrayBounds) != 0 {
			return nil, merr.WrapErrParameterInvalidMsg("cast modifiers and array casts are not supported")
		}
		input, err := b.expression(cast.Arg, final)
		if err != nil {
			return nil, err
		}
		typ, err := castDataType(extractTypeName(cast.TypeName))
		if err != nil {
			return nil, err
		}
		if names := cast.TypeName.Names; len(names) > 2 || (len(names) == 2 && names[0].GetString_().GetSval() != "pg_catalog") {
			return nil, merr.WrapErrParameterInvalidMsg("only built-in SQL cast types are supported")
		}
		if !supportedCast(input.Type, typ) {
			return nil, merr.WrapErrParameterInvalidMsg("unsupported SQL cast from %s to %s", input.Type, typ)
		}
		return &BoundExpr{Op: "cast", Type: typ, Nullable: input.Nullable, Args: []*BoundExpr{input}}, nil
	}
	if col, err := extractColumnRef(node); err != nil {
		return nil, err
	} else if col != nil {
		if col.Qualifier != "" && !b.validQualifier(col.Qualifier) {
			return nil, merr.WrapErrParameterInvalidMsg("unknown table qualifier %q", col.Qualifier)
		}
		field := b.fields[col.FieldName]
		if field == nil {
			return nil, merr.WrapErrParameterInvalidMsg("unknown SQL field %q", col.FieldName)
		}
		expr := b.column(field)
		if len(col.NestedPath) != 0 {
			if field.DataType != schemapb.DataType_JSON {
				return nil, merr.WrapErrParameterInvalidMsg("JSON path on non-JSON field %q", field.Name)
			}
			typ := schemapb.DataType_JSON
			if col.IsText {
				typ = schemapb.DataType_VarChar
			}
			expr = &BoundExpr{Op: "json_path", Type: typ, Nullable: true, Path: col.NestedPath, Args: []*BoundExpr{expr}}
		}
		return expr, nil
	}
	if ac := node.GetAConst(); ac != nil {
		expr := &BoundExpr{Op: "literal"}
		switch {
		case ac.Isnull:
			expr.Nullable = true
			expr.Type = schemapb.DataType_None
		case ac.GetIval() != nil:
			expr.Type = schemapb.DataType_Int64
			expr.Value = int64(ac.GetIval().Ival)
		case ac.GetFval() != nil:
			// PostgreSQL represents integer literals outside int32 as Float nodes.
			// Preserve exact int64 values instead of rounding through float64.
			if integer, err := strconv.ParseInt(ac.GetFval().Fval, 10, 64); err == nil {
				expr.Type = schemapb.DataType_Int64
				expr.Value = integer
				break
			}
			value, err := strconv.ParseFloat(ac.GetFval().Fval, 64)
			if err != nil {
				return nil, merr.WrapErrParameterInvalidErr(err, "invalid numeric literal")
			}
			if math.IsNaN(value) || math.IsInf(value, 0) || !strings.ContainsAny(ac.GetFval().Fval, ".eE") {
				return nil, merr.WrapErrParameterInvalidMsg("numeric literal is outside the supported range")
			}
			expr.Type = schemapb.DataType_Double
			expr.Value = value
		case ac.GetSval() != nil:
			expr.Type = schemapb.DataType_VarChar
			expr.Value = ac.GetSval().Sval
		case ac.GetBoolval() != nil:
			expr.Type = schemapb.DataType_Bool
			expr.Value = ac.GetBoolval().Boolval
		default:
			return nil, merr.WrapErrParameterInvalidMsg("unsupported SQL literal")
		}
		return expr, nil
	}
	if fc := node.GetFuncCall(); fc != nil {
		if len(fc.Funcname) > 2 || (len(fc.Funcname) == 2 && fc.Funcname[0].GetString_().GetSval() != "pg_catalog") {
			return nil, merr.WrapErrParameterInvalidMsg("only built-in SQL functions are supported")
		}
		name := getFuncName(fc)
		if supportedAggFuncs[name] {
			if !final {
				return nil, merr.WrapErrParameterInvalidMsg("aggregate %s cannot appear in a row expression", name)
			}
			return b.aggregate(node, fc)
		}
		if fc.Over != nil || fc.AggFilter != nil || fc.AggDistinct || fc.AggStar || fc.AggWithinGroup || len(fc.AggOrder) != 0 || fc.FuncVariadic {
			return nil, merr.WrapErrParameterInvalidMsg("window and filtered functions are not supported")
		}
		args := make([]*BoundExpr, len(fc.Args))
		for i, arg := range fc.Args {
			var err error
			args[i], err = b.expression(arg, final)
			if err != nil {
				return nil, err
			}
		}
		switch name {
		case "to_timestamp":
			if len(args) != 1 || !numericType(args[0].Type) {
				return nil, merr.WrapErrParameterInvalidMsg("to_timestamp requires one numeric argument")
			}
			return &BoundExpr{Op: name, Type: schemapb.DataType_Timestamptz, Nullable: args[0].Nullable, Args: args}, nil
		case "extract", "date_part":
			if len(args) != 2 || args[0].Op != "literal" || args[0].Type != schemapb.DataType_VarChar {
				return nil, merr.WrapErrParameterInvalidMsg("extract requires a literal time unit and a timestamp")
			}
			unit := strings.ToLower(args[0].Value.(string))
			switch unit {
			case "hour", "minute", "second", "day", "month", "year", "dow", "doy", "epoch":
			default:
				return nil, merr.WrapErrParameterInvalidMsg("unsupported extract unit %q", unit)
			}
			if args[1].Type != schemapb.DataType_Timestamptz && !(unit == "epoch" && args[1].Op == "timestamp_diff") {
				return nil, merr.WrapErrParameterInvalidMsg("extract requires a timestamp or epoch interval")
			}
			return &BoundExpr{Op: "extract", Type: schemapb.DataType_Double, Nullable: args[1].Nullable, Value: unit, Args: args[1:]}, nil
		default:
			return nil, merr.WrapErrParameterInvalidMsg("unsupported SQL function %q", name)
		}
	}
	if ae := node.GetAExpr(); ae != nil {
		op := getAExprOpName(ae)
		if op != "+" && op != "-" && op != "*" && op != "/" {
			return nil, merr.WrapErrParameterInvalidMsg("unsupported SQL expression operator %q", op)
		}
		var left *BoundExpr
		var err error
		if ae.Lexpr != nil {
			left, err = b.expression(ae.Lexpr, final)
			if err != nil {
				return nil, err
			}
		}
		right, err := b.expression(ae.Rexpr, final)
		if err != nil {
			return nil, err
		}
		if ae.Lexpr == nil {
			if !numericType(right.Type) || (op != "+" && op != "-") {
				return nil, merr.WrapErrParameterInvalidMsg("invalid unary arithmetic")
			}
			return &BoundExpr{Op: "unary" + op, Type: right.Type, Nullable: right.Nullable, Args: []*BoundExpr{right}}, nil
		}
		if op == "-" && left.Type == schemapb.DataType_Timestamptz && right.Type == schemapb.DataType_Timestamptz {
			return &BoundExpr{Op: "timestamp_diff", Type: schemapb.DataType_Double, Nullable: left.Nullable || right.Nullable, Args: []*BoundExpr{left, right}}, nil
		}
		if !numericType(left.Type) || !numericType(right.Type) {
			return nil, merr.WrapErrParameterInvalidMsg("arithmetic requires numeric operands; use an explicit cast")
		}
		typ := schemapb.DataType_Int64
		if floatingType(left.Type) || floatingType(right.Type) {
			typ = schemapb.DataType_Double
		}
		return &BoundExpr{Op: op, Type: typ, Nullable: left.Nullable || right.Nullable, Args: []*BoundExpr{left, right}}, nil
	}
	return nil, merr.WrapErrParameterInvalidMsg("unsupported SQL expression %T", node.Node)
}

func (b *binder) aggregate(node *pg_query.Node, fc *pg_query.FuncCall) (*BoundExpr, error) {
	key := b.nodeKey(node)
	if output := b.aggregates[key]; output != nil {
		return output, nil
	}
	definition, err := extractAggFunc(fc)
	if err != nil {
		return nil, err
	}
	var input *BoundExpr
	if !fc.AggStar {
		input, err = b.expression(fc.Args[0], false)
		if err != nil {
			return nil, err
		}
	}
	typ := schemapb.DataType_Int64
	if definition.FuncName != "count" {
		typ = input.Type
		if definition.FuncName == "sum" || definition.FuncName == "avg" {
			if !numericType(typ) {
				return nil, merr.WrapErrParameterInvalidMsg("%s requires a numeric input", definition.FuncName)
			}
			if definition.FuncName == "avg" || floatingType(typ) {
				typ = schemapb.DataType_Double
			} else {
				typ = schemapb.DataType_Int64
			}
		}
	}
	if input != nil && input.Type == schemapb.DataType_None && definition.FuncName == "count" {
		input.Type = schemapb.DataType_Int64
	}
	if input != nil && input.Type == schemapb.DataType_Bool && definition.FuncName != "count" {
		return nil, merr.WrapErrParameterInvalidMsg("%s does not support boolean inputs", definition.FuncName)
	}
	if input != nil && !comparableType(input.Type) {
		return nil, merr.WrapErrParameterInvalidMsg("aggregate inputs require a scalar value; use ->> or an explicit cast for JSON")
	}
	output := &BoundExpr{Op: "slot", Slot: b.nextSlot, Type: typ, Nullable: definition.FuncName != "count"}
	b.nextSlot++
	b.aggregates[key] = output
	b.query.Aggregates = append(b.query.Aggregates, BoundAggregate{Name: definition.FuncName, Distinct: definition.Distinct, Input: input, Output: output})
	return output, nil
}

func containsSourceColumn(expr *BoundExpr) bool {
	if expr.Op == "column" {
		return true
	}
	for _, arg := range expr.Args {
		if containsSourceColumn(arg) {
			return true
		}
	}
	return false
}

func numericType(typ schemapb.DataType) bool {
	return typ == schemapb.DataType_Int8 || typ == schemapb.DataType_Int16 || typ == schemapb.DataType_Int32 || typ == schemapb.DataType_Int64 || floatingType(typ)
}

func comparableType(typ schemapb.DataType) bool {
	return numericType(typ) || typ == schemapb.DataType_Bool || typ == schemapb.DataType_VarChar || typ == schemapb.DataType_String || typ == schemapb.DataType_Timestamptz
}

func floatingType(typ schemapb.DataType) bool {
	return typ == schemapb.DataType_Float || typ == schemapb.DataType_Double
}

func supportedCast(from, to schemapb.DataType) bool {
	if from == schemapb.DataType_None || from == to {
		return true
	}
	// Timestamp values use microseconds internally. Exposing that representation
	// as SQL numeric/text casts would silently implement different SQL semantics.
	if from == schemapb.DataType_Timestamptz || to == schemapb.DataType_Timestamptz {
		return false
	}
	if from == schemapb.DataType_JSON {
		return to == schemapb.DataType_VarChar
	}
	if !comparableType(from) {
		return false
	}
	if to == schemapb.DataType_Bool {
		return !floatingType(from)
	}
	if floatingType(to) && from == schemapb.DataType_Bool {
		return false
	}
	return true
}

func castDataType(name string) (schemapb.DataType, error) {
	switch strings.ToLower(name) {
	case "text", "varchar":
		return schemapb.DataType_VarChar, nil
	case "bigint", "int8":
		return schemapb.DataType_Int64, nil
	case "integer", "int4":
		return schemapb.DataType_Int32, nil
	case "smallint", "int2":
		return schemapb.DataType_Int16, nil
	case "float8", "double precision":
		return schemapb.DataType_Double, nil
	case "float4", "real":
		return schemapb.DataType_Float, nil
	case "bool", "boolean":
		return schemapb.DataType_Bool, nil
	case "timestamptz":
		return schemapb.DataType_Timestamptz, nil
	default:
		return schemapb.DataType_None, merr.WrapErrParameterInvalidMsg("unsupported SQL cast %q", name)
	}
}
