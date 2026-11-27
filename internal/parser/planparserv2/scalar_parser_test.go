package planparserv2

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

func TestParseScalarExpr(t *testing.T) {
	schema, err := typeutil.CreateSchemaHelper(newTestSchema(true))
	require.NoError(t, err)
	for _, tc := range []struct {
		text  string
		dtype schemapb.DataType
	}{
		{"Int8Field", schemapb.DataType_Int8},
		{"Int8Field + 1", schemapb.DataType_Int64},
		{"(Int64Field + Int32Field) * 2 - 3", schemapb.DataType_Int64},
		{"Int64Field / 2", schemapb.DataType_Int64},
		{"FloatField + Int64Field", schemapb.DataType_Double},
		{"FloatField + FloatField", schemapb.DataType_Double},
		{"1.0000000001", schemapb.DataType_Double},
		{"VarCharField", schemapb.DataType_VarChar},
		{"\"hello\"", schemapb.DataType_VarChar},
		{"true", schemapb.DataType_Bool},
		{"false", schemapb.DataType_Bool},
		{"BoolField", schemapb.DataType_Bool},
		{"Int64Field > 10", schemapb.DataType_Bool},
		{"Int64Field > 1 + 2", schemapb.DataType_Bool},
		{"1 + 1 > 1", schemapb.DataType_Bool},
		{"1 / 0", schemapb.DataType_Int64},
		{"-9223372036854775808 / -1", schemapb.DataType_Int64},
	} {
		t.Run(tc.text, func(t *testing.T) {
			expr, dtype, err := ParseScalarExpr(schema, tc.text, nil)
			require.NoError(t, err)
			require.NotNil(t, expr)
			require.Equal(t, tc.dtype, dtype)
		})
	}
	// Value arithmetic must not be folded before a row selection is known.
	expr, _, err := ParseScalarExpr(schema, "1 / 0", nil)
	require.NoError(t, err)
	require.Equal(t, planpb.ArithOpType_Div, expr.GetBinaryArithExpr().GetOp())
	expr, _, err = ParseScalarExpr(schema, "false", nil)
	require.NoError(t, err)
	require.NotNil(t, expr.GetValueExpr())
	require.False(t, expr.GetValueExpr().GetValue().GetBoolVal())
	_, err = ParseExpr(schema, "Int64Field + 1", nil)
	require.Error(t, err)
}

func TestParseScalarExprTemplates(t *testing.T) {
	schema, err := typeutil.CreateSchemaHelper(newTestSchema(true))
	require.NoError(t, err)
	values := map[string]*schemapb.TemplateValue{
		"factor": {Val: &schemapb.TemplateValue_FloatVal{FloatVal: 1.0000000001}},
		"offset": {Val: &schemapb.TemplateValue_Int64Val{Int64Val: 3}},
	}
	expr, dtype, err := ParseScalarExpr(schema, "(Int32Field + {offset}) * {factor}", values)
	require.NoError(t, err)
	require.Equal(t, schemapb.DataType_Double, dtype)
	require.False(t, expr.GetIsTemplate())
	require.Equal(t, 1.0000000001, expr.GetBinaryArithExpr().GetRight().GetValueExpr().GetValue().GetFloatVal())
	_, dtype, err = ParseScalarExpr(schema, "{offset} + {factor}", values)
	require.NoError(t, err)
	require.Equal(t, schemapb.DataType_Double, dtype)
	_, dtype, err = ParseScalarExpr(schema, "Int64Field > {offset}", values)
	require.NoError(t, err)
	require.Equal(t, schemapb.DataType_Bool, dtype)
	_, _, err = ParseScalarExpr(schema, "Int64Field + {missing}", values)
	require.Error(t, err)
	require.ErrorIs(t, err, merr.ErrQueryPlan)
	values["factor"] = &schemapb.TemplateValue{Val: &schemapb.TemplateValue_StringVal{StringVal: "bad"}}
	_, _, err = ParseScalarExpr(schema, "Int64Field + {factor}", values)
	require.Error(t, err)
}

func TestParseScalarExprRejectsUnsupported(t *testing.T) {
	schema, err := typeutil.CreateSchemaHelper(newTestSchema(true))
	require.NoError(t, err)
	for _, text := range []string{"", " ", "ArrayField", "JSONField[\"x\"]", "Int64Field % 2", "Int64Field + true", "Int64Field AS score", "FloatVectorField"} {
		t.Run(text, func(t *testing.T) {
			_, _, err := ParseScalarExpr(schema, text, nil)
			require.Error(t, err)
		})
	}
}
