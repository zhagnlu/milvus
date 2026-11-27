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

package queryutil

import (
	"strconv"

	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// WithSQLProjectionSchema extends only the reducer's query-local schema. The
// collection schema used for segment reads must remain unchanged. Physical and
// projected IDs cannot collide, even when a query projects the same field more
// than once through different expressions.
func WithSQLProjectionSchema(schema *schemapb.CollectionSchema, project *planpb.SQLProject) (*schemapb.CollectionSchema, error) {
	if project == nil {
		return schema, nil
	}
	if project.Version != 1 {
		return nil, merr.WrapErrParameterInvalidMsg("unsupported SQL projection version %d", project.Version)
	}
	if schema == nil {
		return nil, merr.WrapErrServiceInternalMsg("missing collection schema for SQL projection")
	}
	result := proto.Clone(schema).(*schemapb.CollectionSchema)
	ids := make(map[int64]bool, len(schema.Fields)+len(project.Columns))
	for _, field := range schema.Fields {
		ids[field.FieldID] = true
	}
	for _, parent := range schema.StructArrayFields {
		ids[parent.FieldID] = true
		for _, field := range parent.Fields {
			ids[field.FieldID] = true
		}
	}
	for _, column := range project.Columns {
		if column.GetId() <= 0 || ids[column.GetId()] || column.GetExpression() == nil {
			return nil, merr.WrapErrParameterInvalidMsg("invalid SQL projection ID %d", column.GetId())
		}
		typ := column.Expression.DataType
		switch typ {
		case schemapb.DataType_Bool, schemapb.DataType_Int8, schemapb.DataType_Int16,
			schemapb.DataType_Int32, schemapb.DataType_Int64, schemapb.DataType_Float,
			schemapb.DataType_Double, schemapb.DataType_String, schemapb.DataType_VarChar,
			schemapb.DataType_Timestamptz:
		default:
			return nil, merr.WrapErrParameterInvalidMsg("unsupported SQL projection type %s", typ)
		}
		ids[column.Id] = true
		result.Fields = append(result.Fields, &schemapb.FieldSchema{
			FieldID: column.Id, Name: "__sql_slot_" + strconv.FormatInt(column.Id, 10),
			DataType: typ, Nullable: true,
		})
	}
	return result, nil
}
