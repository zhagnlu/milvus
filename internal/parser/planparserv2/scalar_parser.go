package planparserv2

import (
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/rewriter"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// ParseScalarExpr is an internal projection entry, not a query API. ParseExpr
// remains BOOL-only. Value arithmetic stays in the tree so constant and column
// division have identical selected-row error semantics. Aliases and nested
// value projection are intentionally not accepted by this first entry.
func ParseScalarExpr(schema *typeutil.SchemaHelper, text string, templates map[string]*schemapb.TemplateValue) (*planpb.Expr, schemapb.DataType, error) {
	if isEmptyExpression(text) {
		return nil, schemapb.DataType_None, merr.WrapErrQueryPlanMsg("empty scalar expression")
	}
	// Existing predicates retain their own folding and coercion rules. The
	// scalar context is only needed for value expressions; its deferred
	// arithmetic would otherwise break predicates such as "a > 1 + 2".
	ret := handleExprInternal(schema, text, &ParserVisitorArgs{scalarProjection: true})
	if parsed := getExpr(ret); parsed == nil || parsed.dataType == schemapb.DataType_Bool {
		predicate := handleExprInternal(schema, text, &ParserVisitorArgs{})
		if typed := getExpr(predicate); typed != nil && typed.dataType == schemapb.DataType_Bool {
			ret = predicate
		}
	}
	if err := getError(ret); err != nil {
		return nil, schemapb.DataType_None, merr.Wrap(err, "cannot parse scalar expression")
	}
	parsed := getExpr(ret)
	if parsed == nil {
		return nil, schemapb.DataType_None, merr.WrapErrServiceInternalMsg("scalar parser returned no expression")
	}
	values, err := UnmarshalExpressionValues(templates)
	if err != nil {
		return nil, schemapb.DataType_None, err
	}
	dtype, err := bindScalarExpr(schema, parsed.expr, parsed.dataType, values)
	if err != nil {
		return nil, schemapb.DataType_None, err
	}
	// BOOL literals are values here; the predicate rewriter would turn them
	// into AlwaysTrue/NOT AlwaysTrue filter nodes.
	if parsed.expr.GetValueExpr() != nil {
		return parsed.expr, dtype, nil
	}
	return rewriter.RewriteExpr(parsed.expr), dtype, nil
}

func bindScalarExpr(schema *typeutil.SchemaHelper, expr *planpb.Expr, inferred schemapb.DataType, values map[string]*planpb.GenericValue) (schemapb.DataType, error) {
	none := schemapb.DataType_None
	switch node := expr.GetExpr().(type) {
	case *planpb.Expr_ValueExpr:
		if expr.GetIsTemplate() {
			value, ok := values[node.ValueExpr.GetTemplateVariableName()]
			if !ok {
				return none, merr.WrapErrQueryPlanMsg("missing scalar template value: %s", node.ValueExpr.GetTemplateVariableName())
			}
			node.ValueExpr.Value = value
			node.ValueExpr.TemplateVariableName = ""
			expr.IsTemplate = false
		}
		value := node.ValueExpr.GetValue()
		switch {
		case IsBool(value):
			return schemapb.DataType_Bool, nil
		case IsInteger(value):
			return schemapb.DataType_Int64, nil
		case IsFloating(value):
			return schemapb.DataType_Double, nil
		case IsString(value):
			return schemapb.DataType_VarChar, nil
		default:
			return none, merr.WrapErrQueryPlanMsg("unsupported scalar literal type")
		}
	case *planpb.Expr_ColumnExpr:
		info := node.ColumnExpr.GetInfo()
		dtype := info.GetDataType()
		if len(info.GetNestedPath()) != 0 || info.GetIsElementLevel() ||
			!(typeutil.IsArithmetic(dtype) || typeutil.IsBoolType(dtype) || typeutil.IsStringType(dtype)) {
			return none, merr.WrapErrQueryPlanMsg("unsupported scalar projection column type or path")
		}
		return dtype, nil
	case *planpb.Expr_BinaryArithExpr:
		arith := node.BinaryArithExpr
		switch arith.GetOp() {
		case planpb.ArithOpType_Add, planpb.ArithOpType_Sub, planpb.ArithOpType_Mul, planpb.ArithOpType_Div:
		default:
			return none, merr.WrapErrQueryPlanMsg("unsupported scalar arithmetic operation: %s", arith.GetOp())
		}
		left, err := bindScalarExpr(schema, arith.GetLeft(), none, values)
		if err != nil {
			return none, err
		}
		right, err := bindScalarExpr(schema, arith.GetRight(), none, values)
		if err != nil {
			return none, err
		}
		if !typeutil.IsArithmetic(left) || !typeutil.IsArithmetic(right) {
			return none, merr.WrapErrQueryPlanMsg("scalar arithmetic requires numeric operands")
		}
		expr.IsTemplate = false
		return getTargetType(left, right)
	default:
		if inferred != schemapb.DataType_Bool {
			return none, merr.WrapErrQueryPlanMsg("unsupported scalar expression type")
		}
		if err := fillExpressionValueWithBudgetAndSchema(expr, values, nil, schema); err != nil {
			return none, err
		}
		return schemapb.DataType_Bool, nil
	}
}
