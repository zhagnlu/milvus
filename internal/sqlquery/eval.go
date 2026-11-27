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
	"bytes"
	"encoding/json"
	"math"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/sqlparser"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

type evaluation struct {
	columns map[int64]any
	slots   []any
	zone    *time.Location
}

func normalize(value any) any {
	switch value := value.(type) {
	case int8:
		return int64(value)
	case int16:
		return int64(value)
	case int32:
		return int64(value)
	case float32:
		return float64(value)
	default:
		return value
	}
}

func number(value any) (float64, error) {
	switch value := normalize(value).(type) {
	case int64:
		return float64(value), nil
	case float64:
		return value, nil
	default:
		return 0, merr.WrapErrParameterInvalidMsg("SQL arithmetic requires numeric operands")
	}
}

func finite(value float64) (any, error) {
	if math.IsInf(value, 0) || math.IsNaN(value) {
		return nil, merr.WrapErrParameterInvalidMsg("SQL floating point overflow")
	}
	return value, nil
}

func integer(value any) (int64, error) {
	switch value := normalize(value).(type) {
	case int64:
		return value, nil
	case string:
		parsed, err := strconv.ParseInt(strings.Trim(value, " \t\r\n\f\v"), 10, 64)
		if err != nil {
			return 0, merr.WrapErrParameterInvalidErr(err, "invalid SQL integer cast")
		}
		return parsed, nil
	case bool:
		if value {
			return 1, nil
		}
		return 0, nil
	case float64:
		value = math.RoundToEven(value)
		if math.IsNaN(value) || value < -9223372036854775808.0 || value >= 9223372036854775808.0 {
			return 0, merr.WrapErrParameterInvalidMsg("SQL integer cast overflow")
		}
		return int64(value), nil
	default:
		return 0, merr.WrapErrParameterInvalidMsg("unsupported SQL integer cast from %T", value)
	}
}

func cast(value any, typ schemapb.DataType) (any, error) {
	if value == nil {
		return nil, nil
	}
	value = normalize(value)
	switch typ {
	case schemapb.DataType_Int8, schemapb.DataType_Int16, schemapb.DataType_Int32, schemapb.DataType_Int64, schemapb.DataType_Timestamptz:
		result, err := integer(value)
		if err != nil {
			return nil, err
		}
		if (typ == schemapb.DataType_Int8 && (result < math.MinInt8 || result > math.MaxInt8)) ||
			(typ == schemapb.DataType_Int16 && (result < math.MinInt16 || result > math.MaxInt16)) ||
			(typ == schemapb.DataType_Int32 && (result < math.MinInt32 || result > math.MaxInt32)) {
			return nil, merr.WrapErrParameterInvalidMsg("SQL integer cast overflow")
		}
		return result, nil
	case schemapb.DataType_Float, schemapb.DataType_Double:
		var result float64
		var err error
		if text, ok := value.(string); ok {
			if strings.ContainsAny(text, "xXpP_") {
				return nil, merr.WrapErrParameterInvalidMsg("invalid SQL floating point cast")
			}
			result, err = strconv.ParseFloat(strings.Trim(text, " \t\r\n\f\v"), 64)
		} else {
			result, err = number(value)
		}
		if err != nil {
			return nil, merr.WrapErrParameterInvalidErr(err, "invalid SQL floating point cast")
		}
		if typ == schemapb.DataType_Float {
			result = float64(float32(result))
		}
		return finite(result)
	case schemapb.DataType_VarChar, schemapb.DataType_String:
		switch value := value.(type) {
		case string:
			return value, nil
		case int64:
			return strconv.FormatInt(value, 10), nil
		case float64:
			return strconv.FormatFloat(value, 'g', -1, 64), nil
		case bool:
			return strconv.FormatBool(value), nil
		}
	case schemapb.DataType_Bool:
		switch value := value.(type) {
		case bool:
			return value, nil
		case int64:
			return value != 0, nil
		case string:
			result, err := strconv.ParseBool(strings.ToLower(strings.Trim(value, " \t\r\n\f\v")))
			if err != nil {
				return nil, merr.WrapErrParameterInvalidErr(err, "invalid SQL boolean cast")
			}
			return result, nil
		}
	}
	return nil, merr.WrapErrParameterInvalidMsg("unsupported SQL cast from %T to %s", value, typ)
}

func (e evaluation) eval(expr *sqlparser.BoundExpr) (any, error) {
	switch expr.Op {
	case "and", "or", "not", "is_null", "is_not_null", "in", "like", "=", "<>", "<", "<=", ">", ">=":
		return e.predicate(expr)
	}
	switch expr.Op {
	case "column":
		return normalize(e.columns[expr.FieldID]), nil
	case "slot":
		if expr.Slot < 0 || expr.Slot >= len(e.slots) {
			return nil, merr.WrapErrServiceInternalMsg("invalid SQL result slot %d", expr.Slot)
		}
		return normalize(e.slots[expr.Slot]), nil
	case "literal":
		return expr.Value, nil
	case "json_path":
		value, err := e.eval(expr.Args[0])
		if err != nil || value == nil {
			return value, err
		}
		raw, ok := value.([]byte)
		if !ok {
			return nil, merr.WrapErrServiceInternalMsg("JSON source has type %T", value)
		}
		if !json.Valid(raw) {
			return nil, merr.WrapErrServiceInternalMsg("invalid stored JSON")
		}
		for _, path := range expr.Path {
			raw = bytes.TrimSpace(raw)
			if len(raw) == 0 || raw[0] != '{' {
				return nil, nil
			}
			var object map[string]json.RawMessage
			if err := json.Unmarshal(raw, &object); err != nil {
				return nil, merr.WrapErrServiceInternalErr(err, "invalid stored JSON")
			}
			raw = object[path]
		}
		raw = bytes.TrimSpace(raw)
		if len(raw) == 0 {
			return nil, nil
		}
		if expr.Type == schemapb.DataType_JSON {
			return raw, nil
		}
		if bytes.Equal(raw, []byte("null")) {
			return nil, nil
		}
		if raw[0] == '"' {
			var text string
			if err := json.Unmarshal(raw, &text); err != nil {
				return nil, merr.WrapErrServiceInternalErr(err, "invalid stored JSON string")
			}
			return text, nil
		}
		return string(raw), nil
	}
	args := make([]any, len(expr.Args))
	for i, arg := range expr.Args {
		value, err := e.eval(arg)
		if err != nil {
			return nil, err
		}
		if value == nil {
			return nil, nil
		}
		args[i] = value
	}
	switch expr.Op {
	case "cast":
		value := args[0]
		if raw, ok := value.([]byte); ok {
			value = string(raw)
		}
		return cast(value, expr.Type)
	case "to_timestamp":
		seconds, err := number(args[0])
		if err != nil {
			return nil, err
		}
		return integer(math.Round(seconds * 1000000))
	case "timestamp_diff":
		// Subtract seconds and remainders separately to preserve small spans
		// between large timestamps without overflowing int64.
		a, b := args[0].(int64), args[1].(int64)
		return float64(a/1000000-b/1000000) + float64(a%1000000-b%1000000)/1000000, nil
	case "extract":
		if expr.Value == "epoch" && expr.Args[0].Op == "timestamp_diff" {
			return args[0], nil
		}
		micros := args[0].(int64)
		timestamp := time.UnixMicro(micros).In(e.zone)
		switch expr.Value {
		case "hour":
			return float64(timestamp.Hour()), nil
		case "minute":
			return float64(timestamp.Minute()), nil
		case "second":
			return float64(timestamp.Second()) + float64(timestamp.Nanosecond())/1e9, nil
		case "day":
			return float64(timestamp.Day()), nil
		case "month":
			return float64(timestamp.Month()), nil
		case "year":
			return float64(timestamp.Year()), nil
		case "dow":
			return float64(timestamp.Weekday()), nil
		case "doy":
			return float64(timestamp.YearDay()), nil
		case "epoch":
			return float64(micros) / 1000000, nil
		}
	case "unary+":
		return args[0], nil
	case "unary-":
		if integer, ok := args[0].(int64); ok {
			if integer == math.MinInt64 {
				return nil, merr.WrapErrParameterInvalidMsg("SQL integer arithmetic overflow")
			}
			return -integer, nil
		}
		value, err := number(args[0])
		if err != nil {
			return nil, err
		}
		return finite(-value)
	case "+", "-", "*", "/":
		if expr.Type == schemapb.DataType_Int64 {
			a, b := args[0].(int64), args[1].(int64)
			var result int64
			overflow := false
			switch expr.Op {
			case "+":
				result = a + b
				overflow = (b > 0 && result < a) || (b < 0 && result > a)
			case "-":
				result = a - b
				overflow = (b > 0 && result > a) || (b < 0 && result < a)
			case "*":
				result = a * b
				overflow = (a == math.MinInt64 && b == -1) || (b != 0 && result/b != a)
			case "/":
				if b == 0 {
					return nil, merr.WrapErrParameterInvalidMsg("SQL division by zero")
				}
				overflow = a == math.MinInt64 && b == -1
				if !overflow {
					result = a / b
				}
			}
			if overflow {
				return nil, merr.WrapErrParameterInvalidMsg("SQL integer arithmetic overflow")
			}
			return result, nil
		}
		a, err := number(args[0])
		if err != nil {
			return nil, err
		}
		b, err := number(args[1])
		if err != nil {
			return nil, err
		}
		switch expr.Op {
		case "+":
			return finite(a + b)
		case "-":
			return finite(a - b)
		case "*":
			return finite(a * b)
		case "/":
			if b == 0 {
				return nil, merr.WrapErrParameterInvalidMsg("SQL division by zero")
			}
			return finite(a / b)
		}
	}
	return nil, merr.WrapErrParameterInvalidMsg("unsupported SQL expression %q", expr.Op)
}

func compare(a, b any) (int, error) {
	a, b = normalize(a), normalize(b)
	if x, ok := a.(int64); ok {
		if y, ok := b.(int64); ok {
			if x < y {
				return -1, nil
			}
			if x > y {
				return 1, nil
			}
			return 0, nil
		}
	}
	if x, ok := a.(string); ok {
		y, ok := b.(string)
		if !ok {
			return 0, merr.WrapErrServiceInternalMsg("inconsistent SQL text comparison")
		}
		return strings.Compare(x, y), nil
	}
	if x, ok := a.(bool); ok {
		y, ok := b.(bool)
		if !ok {
			return 0, merr.WrapErrServiceInternalMsg("inconsistent SQL boolean comparison")
		}
		if x == y {
			return 0, nil
		}
		if !x {
			return -1, nil
		}
		return 1, nil
	}
	x, err := number(a)
	if err != nil {
		return 0, err
	}
	y, err := number(b)
	if err != nil {
		return 0, err
	}
	if math.IsNaN(x) || math.IsNaN(y) {
		return 0, merr.WrapErrParameterInvalidMsg("NaN SQL comparisons are not supported")
	}
	if x < y {
		return -1, nil
	}
	if x > y {
		return 1, nil
	}
	return 0, nil
}

func (e evaluation) predicate(expr *sqlparser.BoundExpr) (any, error) {
	if expr.Op == "and" || expr.Op == "or" {
		unknown := false
		for _, arg := range expr.Args {
			value, err := e.eval(arg)
			if err != nil {
				return nil, err
			}
			if value == nil {
				unknown = true
				continue
			}
			if expr.Op == "and" && value == false {
				return false, nil
			}
			if expr.Op == "or" && value == true {
				return true, nil
			}
		}
		if unknown {
			return nil, nil
		}
		return expr.Op == "and", nil
	}
	left, err := e.eval(expr.Args[0])
	if err != nil {
		return nil, err
	}
	switch expr.Op {
	case "is_null":
		return left == nil, nil
	case "is_not_null":
		return left != nil, nil
	}
	if left == nil {
		return nil, nil
	}
	if expr.Op == "not" {
		return !left.(bool), nil
	}
	if expr.Op == "in" {
		unknown := false
		for _, arg := range expr.Args[1:] {
			right, err := e.eval(arg)
			if err != nil {
				return nil, err
			}
			if right == nil {
				unknown = true
				continue
			}
			cmp, err := compare(left, right)
			if err != nil {
				return nil, err
			}
			if cmp == 0 {
				return true, nil
			}
		}
		if unknown {
			return nil, nil
		}
		return false, nil
	}
	right, err := e.eval(expr.Args[1])
	if err != nil {
		return nil, err
	}
	if right == nil {
		return nil, nil
	}
	if expr.Op == "like" {
		var pattern strings.Builder
		pattern.WriteString("(?s)^")
		escaped := false
		for _, ch := range right.(string) {
			if escaped {
				pattern.WriteString(regexp.QuoteMeta(string(ch)))
				escaped = false
				continue
			}
			switch ch {
			case '\\':
				escaped = true
			case '%':
				pattern.WriteString(".*")
			case '_':
				pattern.WriteString(".")
			default:
				pattern.WriteString(regexp.QuoteMeta(string(ch)))
			}
		}
		if escaped {
			return nil, merr.WrapErrParameterInvalidMsg("LIKE pattern ends with escape")
		}
		pattern.WriteString("$")
		compiled, err := regexp.Compile(pattern.String())
		if err != nil {
			return nil, merr.WrapErrParameterInvalidErr(err, "invalid LIKE pattern")
		}
		return compiled.MatchString(left.(string)), nil
	}
	cmp, err := compare(left, right)
	if err != nil {
		return nil, err
	}
	switch expr.Op {
	case "=":
		return cmp == 0, nil
	case "<>":
		return cmp != 0, nil
	case "<":
		return cmp < 0, nil
	case "<=":
		return cmp <= 0, nil
	case ">":
		return cmp > 0, nil
	case ">=":
		return cmp >= 0, nil
	}
	return nil, merr.WrapErrServiceInternalMsg("unknown SQL predicate %q", expr.Op)
}
