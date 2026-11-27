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
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math"
	"sort"
	"strconv"
	"strings"

	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// BuildTypedPlaceholderGroupFromRaw builds the same PlaceholderGroup shape that
// normal Search receives from SDKs, but starts from a SQL parameter/literal
// string and resolves the concrete placeholder type from the vector field.
func BuildTypedPlaceholderGroupFromRaw(raw string, field *schemapb.FieldSchema) ([]byte, error) {
	if field == nil {
		return nil, merr.WrapErrParameterInvalidMsg("vector field schema is required")
	}

	switch field.GetDataType() {
	case schemapb.DataType_FloatVector, schemapb.DataType_Float16Vector, schemapb.DataType_BFloat16Vector:
		vector, err := parseVectorLiteral(raw)
		if err != nil {
			return nil, err
		}
		return BuildPlaceholderGroup([][]float32{vector})
	case schemapb.DataType_BinaryVector:
		value, err := parseBinaryVectorLiteral(raw, vectorDim(field))
		if err != nil {
			return nil, err
		}
		return buildPlaceholderGroup(commonpb.PlaceholderType_BinaryVector, [][]byte{value})
	case schemapb.DataType_SparseFloatVector:
		value, err := parseSparseVectorLiteral(raw)
		if err != nil {
			return nil, err
		}
		return buildPlaceholderGroup(commonpb.PlaceholderType_SparseFloatVector, [][]byte{value})
	default:
		return nil, merr.WrapErrParameterInvalidMsg("field %s is not a supported vector field type: %s", field.GetName(), field.GetDataType())
	}
}

func buildPlaceholderGroup(placeholderType commonpb.PlaceholderType, values [][]byte) ([]byte, error) {
	pg := &commonpb.PlaceholderGroup{
		Placeholders: []*commonpb.PlaceholderValue{
			{
				Tag:    "$0",
				Type:   placeholderType,
				Values: values,
			},
		},
	}
	return proto.Marshal(pg)
}

// BuildPlaceholderGroup encodes float vectors using the Search wire format.
func BuildPlaceholderGroup(vectors [][]float32) ([]byte, error) {
	values := make([][]byte, len(vectors))
	for i, vector := range vectors {
		values[i] = make([]byte, len(vector)*4)
		for j, value := range vector {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, merr.WrapErrParameterInvalidMsg("vector values must be finite")
			}
			binary.LittleEndian.PutUint32(values[i][j*4:], math.Float32bits(value))
		}
	}
	return buildPlaceholderGroup(commonpb.PlaceholderType_FloatVector, values)
}

func parseBinaryVectorLiteral(raw string, dim int) ([]byte, error) {
	literal := strings.TrimSpace(raw)
	if literal == "" {
		return nil, merr.WrapErrParameterInvalidMsg("binary vector literal cannot be empty")
	}

	if strings.HasPrefix(literal, "base64:") {
		value, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(literal, "base64:"))
		if err != nil {
			return nil, merr.WrapErrParameterInvalidErr(err, "invalid base64 binary vector literal")
		}
		if dim > 0 && len(value)*8 != dim {
			return nil, merr.WrapErrParameterInvalidMsg("binary vector dimension mismatch: got %d bits, expected %d", len(value)*8, dim)
		}
		return value, nil
	}

	literal = strings.TrimPrefix(literal, "0b")
	literal = strings.NewReplacer(" ", "", "_", "").Replace(literal)
	if len(literal)%8 != 0 {
		return nil, merr.WrapErrParameterInvalidMsg("binary vector bit length must be a multiple of 8, got %d", len(literal))
	}
	if dim > 0 && len(literal) != dim {
		return nil, merr.WrapErrParameterInvalidMsg("binary vector dimension mismatch: got %d bits, expected %d", len(literal), dim)
	}

	value := make([]byte, len(literal)/8)
	for i, ch := range literal {
		if ch != '0' && ch != '1' {
			return nil, merr.WrapErrParameterInvalidMsg("invalid binary vector bit %q at position %d", ch, i)
		}
		if ch == '1' {
			value[i/8] |= 1 << uint(7-i%8)
		}
	}
	return value, nil
}

func parseSparseVectorLiteral(raw string) ([]byte, error) {
	literal := strings.TrimSpace(raw)
	if literal == "" {
		return nil, merr.WrapErrParameterInvalidMsg("sparse vector literal cannot be empty")
	}

	if strings.HasPrefix(literal, "{") {
		if slash := strings.LastIndex(literal, "/"); slash >= 0 {
			literal = strings.TrimSpace(literal[:slash])
		}
	}

	entries, err := parseSparseEntries(literal)
	if err != nil {
		return nil, err
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].pos < entries[j].pos
	})

	value := make([]byte, len(entries)*8)
	for i, entry := range entries {
		binary.LittleEndian.PutUint32(value[i*8:], entry.pos)
		binary.LittleEndian.PutUint32(value[i*8+4:], math.Float32bits(entry.value))
	}
	return value, nil
}

type sparseEntry struct {
	pos   uint32
	value float32
}

func parseSparseEntries(literal string) ([]sparseEntry, error) {
	var rawMap map[string]float32
	if err := json.Unmarshal([]byte(literal), &rawMap); err == nil {
		return sparseEntriesFromMap(rawMap)
	}

	body := strings.TrimSpace(literal)
	if !strings.HasPrefix(body, "{") || !strings.HasSuffix(body, "}") {
		return nil, merr.WrapErrParameterInvalidMsg("invalid sparse vector literal %q", literal)
	}
	body = strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(body, "{"), "}"))
	if body == "" {
		return nil, merr.WrapErrParameterInvalidMsg("sparse vector literal cannot be empty")
	}

	entries := make([]sparseEntry, 0)
	for _, part := range strings.Split(body, ",") {
		kv := strings.SplitN(part, ":", 2)
		if len(kv) != 2 {
			return nil, merr.WrapErrParameterInvalidMsg("invalid sparse vector entry %q", part)
		}
		pos64, err := strconv.ParseUint(strings.TrimSpace(kv[0]), 10, 32)
		if err != nil {
			return nil, merr.WrapErrParameterInvalidErr(err, "invalid sparse vector position")
		}
		value64, err := strconv.ParseFloat(strings.TrimSpace(kv[1]), 32)
		if err != nil {
			return nil, merr.WrapErrParameterInvalidErr(err, "invalid sparse vector value")
		}
		entries = append(entries, sparseEntry{pos: uint32(pos64), value: float32(value64)})
	}
	return entries, nil
}

func sparseEntriesFromMap(rawMap map[string]float32) ([]sparseEntry, error) {
	if len(rawMap) == 0 {
		return nil, merr.WrapErrParameterInvalidMsg("sparse vector literal cannot be empty")
	}
	entries := make([]sparseEntry, 0, len(rawMap))
	for key, value := range rawMap {
		pos64, err := strconv.ParseUint(key, 10, 32)
		if err != nil {
			return nil, merr.WrapErrParameterInvalidErr(err, "invalid sparse vector position")
		}
		entries = append(entries, sparseEntry{pos: uint32(pos64), value: value})
	}
	return entries, nil
}

func vectorDim(field *schemapb.FieldSchema) int {
	for _, param := range field.GetTypeParams() {
		if param.GetKey() != "dim" {
			continue
		}
		dim, err := strconv.Atoi(param.GetValue())
		if err == nil {
			return dim
		}
	}
	return 0
}
