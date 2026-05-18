# JSONBench SQL Query E2E Design

## Purpose

This PR adds an end-to-end benchmark runner for executing JSONBench-style analytical SQL queries against Milvus through the new SQL query endpoint.

The JSONBench change is intentionally split from the SQL parser implementation:

- The SQL parser PR introduces feature behavior and correctness coverage.
- The JSONBench PR adds benchmark data loading and performance measurement only.

This keeps the main parser review focused on product behavior while allowing benchmark setup, query selection, and result reporting to evolve independently.

## Dependency

The JSONBench E2E runner depends on the SQL parser PR because it calls:

```text
POST /v2/vectordb/entities/sql_query
```

It should be submitted as a stacked PR after the SQL parser PR.

Expected stacked history:

```text
test: add JSONBench SQL query E2E
feat: add SQL query parser
origin/master
```

## Scope

Included in this PR:

- `tests/e2e/jsonbench_1m_test.go`
- Data loader for a gzipped Bluesky JSON lines file.
- Five JSONBench-inspired SQL queries.
- Cold and hot latency measurement for each query.
- Summary output in the Go test log.

Out of scope:

- SQL parser implementation changes.
- Query execution engine changes.
- Automatic dataset download.
- CI execution by default.
- Formal benchmark result publication.

## Test Entry

The benchmark is implemented as a Go E2E test:

```text
tests/e2e/jsonbench_1m_test.go
```

Run command:

```bash
cd tests/e2e
go test -v -run TestJSONBench1M \
  -milvus-addr=localhost:19530 \
  -bluesky-data=$HOME/data/bluesky/file_0001.json.gz \
  -timeout 30m
```

The test is skipped unless `-bluesky-data` is provided.

## Dataset

Input format:

- Gzipped JSON lines file.
- One JSON object per line.
- The file path is provided by `-bluesky-data`.

The loader creates a Milvus collection named `bluesky`:

| Field | Type | Purpose |
| --- | --- | --- |
| `pk` | `Int64`, primary key, auto ID | Row identity |
| `data` | `JSON` | Raw Bluesky event payload |
| `dummy_vec` | `FloatVector(2)` | Required vector field for collection load/index flow |

Insert behavior:

- Reads the gzip stream line by line.
- Parses each line into a JSON object.
- Inserts rows in batches of 5000.
- Logs progress every 50000 rows.

The loader first checks:

```sql
SELECT count(*) AS cnt FROM bluesky
```

If the collection already contains queryable data, loading is skipped so repeated benchmark runs can reuse the existing collection.

## Collection Lifecycle

When data is not already loaded:

1. Drop existing `bluesky` collection, ignoring errors.
2. Create the collection.
3. Insert JSON data in batches.
4. Wait briefly for data to seal.
5. Create a `FLAT` index on `dummy_vec`.
6. Load the collection.
7. Poll load state until `LoadStateLoaded`.

The benchmark uses REST v2 APIs only. It does not start an embedded Milvus cluster.

## Query Set

The benchmark runs five analytical SQL queries over the `data` JSON field.

### Q1: Top event types

```sql
SELECT data->'commit'->>'collection' AS event, count(*) AS count
FROM bluesky
GROUP BY event
ORDER BY count DESC
```

Coverage:

- JSON path projection.
- Group by JSON expression alias.
- Count aggregation.
- Sort by aggregate alias.

### Q2: Top event types with unique users

```sql
SELECT data->'commit'->>'collection' AS event, count(*) AS count,
       count(DISTINCT data->>'did') AS users
FROM bluesky
WHERE data->>'kind' = 'commit' AND data->'commit'->>'operation' = 'create'
GROUP BY event
ORDER BY count DESC
```

Coverage:

- JSON path filters.
- `count(DISTINCT ...)`.
- Multiple aggregate outputs.

### Q3: Hour-of-day activity

```sql
SELECT data->'commit'->>'collection' AS event,
       extract(hour FROM to_timestamp((data->>'time_us')::bigint / 1000000.0)) AS hour_of_day,
       count(*) AS count
FROM bluesky
WHERE data->>'kind' = 'commit' AND data->'commit'->>'operation' = 'create'
  AND data->'commit'->>'collection' IN ('app.bsky.feed.post', 'app.bsky.feed.repost',
                                         'app.bsky.feed.like')
GROUP BY event, hour_of_day
ORDER BY hour_of_day, event
```

Coverage:

- Computed projection.
- Type cast.
- `to_timestamp`.
- `extract(hour FROM ...)`.
- Multi-key group by and order by.

### Q4: Earliest posters

```sql
SELECT (data->>'did')::text AS user_id,
       min(to_timestamp((data->>'time_us')::bigint / 1000000.0)) AS first_post_ts
FROM bluesky
WHERE data->>'kind' = 'commit' AND data->'commit'->>'operation' = 'create'
  AND data->'commit'->>'collection' = 'app.bsky.feed.post'
GROUP BY user_id
ORDER BY first_post_ts ASC LIMIT 3
```

Coverage:

- Casted JSON projection.
- `min` over a computed timestamp input.
- Ascending order and limit.

### Q5: Longest user activity span

```sql
SELECT (data->>'did')::text AS user_id,
       extract(epoch FROM (
           max(to_timestamp((data->>'time_us')::bigint / 1000000.0)) -
           min(to_timestamp((data->>'time_us')::bigint / 1000000.0))
       )) * 1000 AS activity_span
FROM bluesky
WHERE data->>'kind' = 'commit' AND data->'commit'->>'operation' = 'create'
  AND data->'commit'->>'collection' = 'app.bsky.feed.post'
GROUP BY user_id
ORDER BY activity_span DESC LIMIT 3
```

Coverage:

- Hidden nested aggregates.
- Arithmetic over aggregate results.
- Computed order by.
- Limit.

## Measurement Model

For each query, the benchmark performs:

1. One cold run.
2. One hot run.
3. One additional hot run.

The test records:

- Query name.
- Cold latency.
- Hot run 1 latency.
- Hot run 2 latency.
- Number of returned rows.
- First row sample as JSON.
- Error message, if the query fails.

The benchmark does not currently compute percentiles or repeat each query enough times for statistical reporting. It is intended as an E2E performance smoke test and a manually-run comparison harness.

## Output

The test logs per-query details and a final summary table:

```text
=== JSONBench 1M Results Summary ===
Q     Cold        Hot1        Hot2        Rows    Status
----  ----------  ----------  ----------  ------  ------
Q1    ...         ...         ...         ...     OK
Q2    ...         ...         ...         ...     OK
```

Failures are recorded in the summary instead of stopping the benchmark immediately after the first failed query. This makes it easier to see which SQL features still fail on a given build.

## Operational Notes

- The benchmark expects a running external Milvus instance.
- The test can take a long time because it loads and indexes up to 1M JSON rows.
- It should not be part of normal unit-test or CI paths.
- Reusing an already-loaded `bluesky` collection is recommended for repeated query tuning.
- The benchmark file reuses helper functions in the `tests/e2e` package, including `baseURL` and `postJSON`.

## PR Boundary

This JSONBench PR should contain only:

```text
tests/e2e/jsonbench_1m_test.go
docs/jsonbench-e2e-design.md
```

It should not include SQL parser, proxy, proto, or segcore changes. Those belong to the SQL parser PR.

