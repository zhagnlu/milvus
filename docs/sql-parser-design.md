# Milvus SQL Parser Design

## Purpose

This change adds a SQL query entrypoint to Milvus so users can run scalar queries, JSON-field analytics, aggregation, ordering, and vector search with PostgreSQL-style `SELECT` syntax.

The SQL parser PR should contain the runtime feature and its correctness tests. JSONBench benchmark coverage is intentionally kept in a separate PR so reviewers can evaluate the core SQL execution path without benchmark-specific data-loading code.

## Scope

Included in this PR:

- Parse PostgreSQL-style `SELECT` SQL with `github.com/pganalyze/pg_query_go/v6`.
- Convert parsed SQL into a Milvus-specific intermediate representation.
- Build Milvus query plans for scalar, JSON, aggregate, sort, projection, and limit queries.
- Route vector-distance SQL to the existing search pipeline.
- Add REST v2 endpoint support: `POST /v2/vectordb/entities/sql_query`.
- Add unit, integration, and SQL query E2E coverage.

Out of scope for this PR:

- JSONBench 1M benchmark runner and benchmark report generation.
- JOIN, subquery, HAVING, window functions, CTE, UNION, and DML.
- Stateful PostgreSQL session behavior such as `SET ivfflat.probes = ...`.

## Architecture

```text
Client
  |
  | POST /v2/vectordb/entities/sql_query
  v
HTTP v2 handler
  |
  | milvuspb.SqlQueryRequest
  v
Proxy.SqlQuery
  |
  | ExtractSqlComponents(sql)
  v
SqlComponents IR
  |
  +-- QueryTypeQuery --------------------+
  |                                      |
  | BuildQueryPlan                       |
  | sqlQueryTask                         |
  | QueryNode.Query                      |
  | Reducer.Reduce                       |
  | ApplyProjection                      |
  |                                      |
  +-- QueryTypeSearch / QueryTypeHybrid -+
                                         |
                                         | sqlComponentsToSearchRequest
                                         | Proxy.search
                                         | searchResultsToSqlResults
                                         v
                                  SqlQueryResults
```

The design keeps parsing, planning, and execution routing separate:

- `internal/sqlparser` owns SQL parsing and SQL-to-plan translation.
- `internal/proxy` owns task lifecycle, shard execution, reducer integration, and search routing.
- `internal/distributed/proxy/httpserver` owns REST request/response adaptation.
- `internal/core` and `pkg/proto/plan.proto` provide execution-plan support for aggregate, sort, and projection nodes.

## SQL IR

`SqlComponents` is the boundary between parser details and Milvus planning:

```go
type SqlComponents struct {
    Collection   string
    SelectItems  []SelectItem
    Where        *WhereClause
    GroupBy      []GroupByItem
    OrderBy      []OrderItem
    Limit        int64
    VectorSearch *VectorSearchDef
    QueryType    QueryType
}
```

The parser normalizes SQL into the following select item types:

| Type | Meaning | Example |
| --- | --- | --- |
| `SelectColumn` | Field or JSON field reference | `id`, `data->>'city'` |
| `SelectAgg` | Aggregate function | `count(*)`, `sum(score)` |
| `SelectComputed` | Scalar or arithmetic expression | `extract(hour FROM ...)`, `max(ts) - min(ts)` |
| `SelectVectorDist` | Vector distance expression | `embedding <-> $1`, `l2_distance(embedding, $1)` |
| `SelectStar` | Wildcard projection | `SELECT *` |

JSON field access is represented as a `ColumnRef`:

```go
type ColumnRef struct {
    FieldName  string
    NestedPath []string
    IsText     bool
    CastType   string
}
```

For example, `data->'commit'->>'collection'` becomes:

```text
FieldName:  data
NestedPath: [commit, collection]
IsText:     true
MilvusExpr: data["commit"]["collection"]
```

## Supported SQL

### Projection

```sql
SELECT id, name FROM users LIMIT 10;
SELECT * FROM users LIMIT 5;
SELECT data->>'category' AS category FROM events;
```

### Filtering

```sql
WHERE age > 30 AND score <= 100
WHERE name IN ('a', 'b', 'c')
WHERE age BETWEEN 25 AND 35
WHERE name LIKE 'user_%'
WHERE data->>'kind' = 'commit'
```

The WHERE converter maps PostgreSQL operators to Milvus filter syntax:

| PostgreSQL | Milvus expression |
| --- | --- |
| `=` | `==` |
| `<>` | `!=` |
| `AND` / `OR` / `NOT` | `and` / `or` / `not` |
| `IN (...)` | `in [...]` |
| `LIKE` | `like` |
| `BETWEEN a AND b` | `>= a and <= b` |
| `data->>'key'` | `data["key"]` |

### Aggregation

```sql
SELECT count(*) FROM users;
SELECT sum(score), avg(score), min(age), max(age) FROM users;
SELECT count(DISTINCT data->>'did') FROM events;
SELECT data->>'category' AS category, count(*) AS cnt
FROM events
GROUP BY category
ORDER BY cnt DESC;
```

### Computed expressions

```sql
SELECT extract(hour FROM to_timestamp((data->>'time_us')::bigint / 1000000.0)) AS hour_of_day
FROM events;

SELECT (data->>'did')::text AS user_id,
       max(to_timestamp((data->>'time_us')::bigint / 1000000.0)) -
       min(to_timestamp((data->>'time_us')::bigint / 1000000.0)) AS activity_span
FROM events
GROUP BY user_id;
```

Computed expressions are planned in two phases:

1. Extract any nested aggregate inputs required by segcore.
2. Evaluate the final expression in proxy after shard results are reduced.

### Vector search

The parser supports both pgvector-style operators and compatibility functions:

```sql
SELECT id, embedding <-> $1 AS distance
FROM articles
ORDER BY distance
LIMIT 10;

SELECT id, l2_distance(embedding, $1) AS distance
FROM articles
WHERE category = 'tech'
ORDER BY distance
LIMIT 10;
```

| SQL syntax | Milvus metric |
| --- | --- |
| `<->`, `l2_distance(field, vector)` | `L2` |
| `<=>`, `cosine_distance(field, vector)` | `COSINE` |
| `<#>`, `ip_distance(field, vector)` | `IP` |
| `<~>`, `hamming_distance(field, vector)` | `HAMMING` |
| `<%>`, `jaccard_distance(field, vector)` | `JACCARD` |

Vector parameters are passed through `SqlQueryRequest.SearchParams`:

```json
{
  "sql": "SELECT id, embedding <-> $1 AS distance FROM articles ORDER BY distance LIMIT 10",
  "searchParams": {
    "$1": "[0.1, 0.2, 0.3]",
    "nprobe": "16"
  }
}
```

## Query Planning

For scalar and aggregate SQL, `BuildQueryPlan` creates a `QueryPlanNode`:

```text
Scan
  -> Filter
  -> Aggregate
  -> Sort
  -> Project
  -> Limit
```

The plan extensions added in `pkg/proto/plan.proto` include:

- `AggregateOp.count_distinct`
- JSON-aware `Aggregate.nested_path`
- `AggregateNode`
- `SortNode`
- `ProjectNode`
- Additional fields on `QueryPlanNode` for aggregate, sort, and project metadata

The generated protobuf file `pkg/proto/planpb/plan.pb.go` is updated accordingly.

## Execution

### Scalar and aggregate path

`Proxy.SqlQuery` creates a `sqlQueryTask` for `QueryTypeQuery`.

`PreExecute`:

- Validates the target collection.
- Loads schema metadata.
- Builds the query plan.
- Builds aggregation field mappings for reducer compatibility.
- Serializes the plan into `SerializedExprPlan`.
- Resolves consistency level, partitions, and timestamps.

`Execute`:

- Uses the existing load-balancing policy.
- Sends retrieve requests to QueryNode shards.
- Collects `RetrieveResults`.

`PostExecute`:

- Uses reducer to merge shard results.
- Applies computed projection.
- Applies post-projection sort when ordering depends on computed output.
- Returns `SqlQueryResults` with column names and field data.

### Vector path

`QueryTypeSearch` and `QueryTypeHybrid` are converted to `SearchRequest`.

- SQL `WHERE` is converted to the search filter DSL.
- SQL `LIMIT` becomes `topk`.
- SQL distance expression selects metric type.
- `$1`-style vector parameters are read from `SearchParams`.
- The existing search pipeline executes the query.
- Search results are converted back to SQL tabular results.

## Code Layout

```text
internal/sqlparser/
  parser.go                 PG parser wrapper
  extract.go                AST -> SqlComponents extraction
  json_path.go              JSON path and type-cast extraction
  where_converter.go        WHERE AST -> Milvus expression
  plan_builder.go           SqlComponents -> PlanNode
  project_executor.go       Proxy-side computed projection
  *_test.go                 parser, extractor, planner, projection tests

internal/proxy/
  impl.go                   Proxy.SqlQuery entrypoint and routing
  task_sql_query.go         sqlQueryTask and search request conversion
  task_sql_query_test.go    proxy task tests

internal/distributed/proxy/httpserver/
  constant.go               sql_query REST action
  request_v2.go             SqlQueryReqV2 request model
  handler_v2.go             REST handler bridge

internal/core/
  exec/operator/*           aggregate/project execution support
  query/PlanProto.cpp       plan deserialization support
  unittest/test_sort_project.cpp

tests/e2e/sql_query_test.go
tests/integration/httpserver/sql_query_test.go
```

## PR Boundary

This SQL parser PR should include:

- SQL parser implementation.
- Proto and segcore plan support.
- Proxy and REST SQL query path.
- Unit tests.
- SQL query E2E and HTTP integration tests.

The JSONBench PR should be stacked on top of this PR and include only the benchmark E2E runner:

```text
tests/e2e/jsonbench_1m_test.go
```

