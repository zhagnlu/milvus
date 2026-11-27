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
	pg_query "github.com/pganalyze/pg_query_go/v6"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func boolean(op string, args ...*BoundExpr) *BoundExpr {
	return &BoundExpr{Op: op, Type: schemapb.DataType_Bool, Nullable: true, Args: args}
}

func (b *binder) predicate(node *pg_query.Node) (*BoundExpr, error) {
	if be := node.GetBoolExpr(); be != nil {
		var op string
		switch be.Boolop {
		case pg_query.BoolExprType_AND_EXPR:
			op = "and"
		case pg_query.BoolExprType_OR_EXPR:
			op = "or"
		case pg_query.BoolExprType_NOT_EXPR:
			op = "not"
		default:
			return nil, merr.WrapErrParameterInvalidMsg("unsupported SQL boolean operator")
		}
		args := make([]*BoundExpr, len(be.Args))
		for i, arg := range be.Args {
			var err error
			args[i], err = b.predicate(arg)
			if err != nil {
				return nil, err
			}
		}
		return boolean(op, args...), nil
	}
	if nt := node.GetNullTest(); nt != nil {
		arg, err := b.expression(nt.Arg, false)
		if err != nil {
			return nil, err
		}
		op := "is_null"
		if nt.Nulltesttype == pg_query.NullTestType_IS_NOT_NULL {
			op = "is_not_null"
		}
		return boolean(op, arg), nil
	}
	if ae := node.GetAExpr(); ae != nil {
		left, err := b.expression(ae.Lexpr, false)
		if err != nil {
			return nil, err
		}
		switch ae.Kind {
		case pg_query.A_Expr_Kind_AEXPR_OP, pg_query.A_Expr_Kind_AEXPR_LIKE:
			right, err := b.expression(ae.Rexpr, false)
			if err != nil {
				return nil, err
			}
			op := getAExprOpName(ae)
			if op == "~~" || op == "!~~" {
				if left.Type != schemapb.DataType_VarChar || right.Type != schemapb.DataType_VarChar {
					return nil, merr.WrapErrParameterInvalidMsg("LIKE requires text operands")
				}
				result := boolean("like", left, right)
				if op == "!~~" {
					result = boolean("not", result)
				}
				return result, nil
			}
			return comparison(op, left, right)
		case pg_query.A_Expr_Kind_AEXPR_IN:
			args := []*BoundExpr{left}
			for _, item := range ae.Rexpr.GetList().GetItems() {
				right, err := b.expression(item, false)
				if err != nil {
					return nil, err
				}
				cmp, err := comparison("=", left, right)
				if err != nil {
					return nil, err
				}
				args = append(args, cmp.Args[1])
			}
			result := boolean("in", args...)
			if getAExprOpName(ae) == "<>" {
				result = boolean("not", result)
			}
			return result, nil
		case pg_query.A_Expr_Kind_AEXPR_BETWEEN, pg_query.A_Expr_Kind_AEXPR_NOT_BETWEEN:
			items := ae.Rexpr.GetList().GetItems()
			if len(items) != 2 {
				return nil, merr.WrapErrParameterInvalidMsg("BETWEEN requires two bounds")
			}
			lo, err := b.expression(items[0], false)
			if err != nil {
				return nil, err
			}
			hi, err := b.expression(items[1], false)
			if err != nil {
				return nil, err
			}
			lower, err := comparison(">=", left, lo)
			if err != nil {
				return nil, err
			}
			upper, err := comparison("<=", left, hi)
			if err != nil {
				return nil, err
			}
			result := boolean("and", lower, upper)
			if ae.Kind == pg_query.A_Expr_Kind_AEXPR_NOT_BETWEEN {
				result = boolean("not", result)
			}
			return result, nil
		}
	}
	value, err := b.expression(node, false)
	if err != nil {
		return nil, err
	}
	if value.Type != schemapb.DataType_Bool {
		return nil, merr.WrapErrParameterInvalidMsg("WHERE requires a boolean expression")
	}
	return value, nil
}

func comparison(op string, left, right *BoundExpr) (*BoundExpr, error) {
	switch op {
	case "=", "<>", "<", "<=", ">", ">=":
	default:
		return nil, merr.WrapErrParameterInvalidMsg("unsupported SQL comparison %q", op)
	}
	// Untyped string literals can be coerced to the other operand's type.
	// A text expression (including ->>) is not an untyped literal.
	if numericType(left.Type) && right.Op == "literal" && right.Type == schemapb.DataType_VarChar {
		right = &BoundExpr{Op: "cast", Type: left.Type, Args: []*BoundExpr{right}}
	}
	if numericType(right.Type) && left.Op == "literal" && left.Type == schemapb.DataType_VarChar {
		left = &BoundExpr{Op: "cast", Type: right.Type, Args: []*BoundExpr{left}}
	}
	if left.Type != schemapb.DataType_None && right.Type != schemapb.DataType_None &&
		(!(numericType(left.Type) && numericType(right.Type)) && (left.Type != right.Type || !comparableType(left.Type))) {
		return nil, merr.WrapErrParameterInvalidMsg("incompatible SQL comparison types %s and %s", left.Type, right.Type)
	}
	return boolean(op, left, right), nil
}
