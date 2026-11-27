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

	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Validate clauses before extraction: accepting and discarding a clause changes
// the query's meaning, even when the PostgreSQL parser accepts its syntax.
func validateSelectStatement(stmt *pg_query.SelectStmt) error {
	checks := []struct {
		present bool
		name    string
	}{
		{stmt.WithClause != nil, "WITH"},
		{stmt.Op != pg_query.SetOperation_SETOP_NONE, "set operations"},
		{stmt.HavingClause != nil, "HAVING"},
		{len(stmt.WindowClause) != 0, "WINDOW"},
		{len(stmt.DistinctClause) != 0, "SELECT DISTINCT"},
		{stmt.IntoClause != nil, "SELECT INTO"},
		{len(stmt.LockingClause) != 0, "row locking"},
		{len(stmt.ValuesLists) != 0, "VALUES"},
		{stmt.GroupDistinct, "GROUP BY DISTINCT"},
		{stmt.LimitOption == pg_query.LimitOption_LIMIT_OPTION_WITH_TIES, "FETCH WITH TIES"},
	}
	for _, check := range checks {
		if check.present {
			return merr.WrapErrParameterInvalidMsg("%s is not supported in SQL queries", check.name)
		}
	}
	return nil
}
