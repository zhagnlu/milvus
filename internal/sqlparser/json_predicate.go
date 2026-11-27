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
	"encoding/json"
	"strings"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

// LowerJSONPredicate returns an equivalent native WHERE filter, or nil when
// SQL text extraction cannot safely be expressed by native JSON comparisons.
// It is all-or-nothing: unsupported clauses keep the original SQL evaluator.
// A literal that is itself valid JSON might match a non-string through ->>
// (e.g. 123 or true), so a native string comparison would lose valid rows.
func LowerJSONPredicate(expr *BoundExpr) *planpb.Expr {
	if expr == nil {
		return nil
	}
	if expr.Op == "and" || expr.Op == "or" {
		var result *planpb.Expr
		for _, arg := range expr.Args {
			child := LowerJSONPredicate(arg)
			if child == nil {
				return nil
			}
			if result == nil {
				result = child
				continue
			}
			op := planpb.BinaryExpr_LogicalAnd
			if expr.Op == "or" {
				op = planpb.BinaryExpr_LogicalOr
			}
			result = &planpb.Expr{Expr: &planpb.Expr_BinaryExpr{BinaryExpr: &planpb.BinaryExpr{Op: op, Left: result, Right: child}}}
		}
		return result
	}
	if (expr.Op != "=" && expr.Op != "in") || len(expr.Args) < 2 {
		return nil
	}
	path, literals := expr.Args[0], expr.Args[1:]
	if expr.Op == "=" && path.Op == "literal" && len(literals) == 1 {
		path, literals = literals[0], []*BoundExpr{path}
	}
	if path.Op != "json_path" || path.Type != schemapb.DataType_VarChar ||
		len(path.Args) != 1 || path.Args[0].Op != "column" ||
		path.Args[0].Type != schemapb.DataType_JSON || len(path.Path) == 0 {
		return nil
	}
	for _, component := range path.Path {
		// Native JSON Pointer could interpret a numeric key as an array index,
		// whereas SQL -> '0' only addresses an object key.
		if component == "" || strings.Trim(component, "0123456789") == "" {
			return nil
		}
	}
	values := make([]*planpb.GenericValue, 0, len(literals))
	for _, literal := range literals {
		text, ok := literal.Value.(string)
		if literal.Op != "literal" || !ok || json.Valid([]byte(text)) {
			return nil
		}
		values = append(values, &planpb.GenericValue{Val: &planpb.GenericValue_StringVal{StringVal: text}})
	}
	column := &planpb.ColumnInfo{FieldId: path.Args[0].FieldID, DataType: schemapb.DataType_JSON,
		NestedPath: append([]string(nil), path.Path...), Nullable: path.Args[0].Nullable}
	if expr.Op == "=" {
		return &planpb.Expr{Expr: &planpb.Expr_UnaryRangeExpr{UnaryRangeExpr: &planpb.UnaryRangeExpr{
			ColumnInfo: column, Op: planpb.OpType_Equal, Value: values[0],
		}}}
	}
	return &planpb.Expr{Expr: &planpb.Expr_TermExpr{TermExpr: &planpb.TermExpr{ColumnInfo: column, Values: values}}}
}
