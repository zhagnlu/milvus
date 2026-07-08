# JSONBench SQL Query E2E Design

## Purpose

This PR adds an end-to-end benchmark runner for executing JSONBench-style analytical SQL queries against Milvus through the new SQL query endpoint.

The JSONBench change is intentionally split from the SQL parser implementation:

- The SQL parser PR introduces feature behavior and correctness coverage.
- The JSONBench PR adds benchmark data loading and performance measurement only.

This keeps the main parser review focused on product behavior while allowing benchmark setup, query selection, and result reporting to evolve independently.

The benchmark PR should be easy to review after the parser PR lands: it contains no parser, proxy, proto, or segcore changes. Any parser behavior needed by the benchmark must be introduced by the lower SQL parser PR first.

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

Review rule for this stacked PR:

- If a query fails because parser support is missing, fix the parser PR and rebase this PR.
- If loading, timing, logging, or dataset handling is wrong, fix this PR.
- Do not use this PR to hide runtime behavior changes needed by SQL correctness tests.

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
- Parser correctness assertions beyond checking whether each benchmark query can run.
- Cluster provisioning, image building, or environment bootstrapping.

## Environment Assumptions

The runner assumes:

- A Milvus instance is already running and reachable through REST v2.
- The SQL parser PR is present in that Milvus build.
- The caller has a local gzipped Bluesky JSON lines file.
- The target database can create, drop, insert into, index, and load a `bluesky` collection.
- The machine running the test has enough time, memory, and network bandwidth to insert up to 1M JSON rows.

The test intentionally does not download datasets or start Milvus. Keeping those steps outside the PR avoids adding external network dependencies to normal test paths.

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

The test is skipped unless `-bluesky-data` is provided. This skip behavior is the CI safety mechanism: normal E2E runs can include the file without requiring a large local dataset.

Useful run variants:

```bash
# Reuse a previously loaded collection when possible.
go test -v -run TestJSONBench1M \
  -milvus-addr=localhost:19530 \
  -bluesky-data=$HOME/data/bluesky/file_0001.json.gz \
  -timeout 30m

# Force a clean run by dropping the bluesky collection manually first.
curl -X POST http://localhost:19530/v2/vectordb/collections/drop \
  -H 'Content-Type: application/json' \
  -d '{"collectionName":"bluesky"}'
```

## Dataset

Input format:

- Gzipped JSON lines file.
- One JSON object per line.
- The file path is provided by `-bluesky-data`.
- The expected source is the Bluesky data shape used by JSONBench-style queries: top-level fields such as `did`, `kind`, and `time_us`, plus nested `commit.collection` and `commit.operation`.

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

If the collection already contains queryable data, loading is skipped so repeated benchmark runs can reuse the existing collection. This makes tuning iterations faster, but it also means a benchmark run may depend on previously loaded data. For a clean comparison, drop the collection before running.

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

Lifecycle failure handling:

- Drop errors are ignored because the collection may not exist.
- Insert and load errors fail the test setup.
- Query failures are recorded per query so later queries can still run and produce a summary.
- Reusing an existing collection skips load work but does not validate that the collection came from the same dataset file.

## Query Set

The benchmark runs five analytical SQL queries over the `data` JSON field. They are chosen to exercise the SQL features added in the parser PR without adding new behavior in this benchmark PR.

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

Interpretation guidance:

- Cold latency includes the first observed execution after loading or reusing data.
- Hot latencies are the next two executions in the same process.
- The test does not reset caches between queries.
- Results are useful for catching large regressions and comparing local builds under the same environment.
- Results should not be presented as formal benchmark numbers without a separate repeated-run methodology.

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

The summary is deliberately compact so it can be pasted into PR comments during manual validation. Per-query logs include the first row sample when available, which helps verify that a query returned plausible tabular output without dumping the full result set.

## Operational Notes

- The benchmark expects a running external Milvus instance.
- The test can take a long time because it loads and indexes up to 1M JSON rows.
- It should not be part of normal unit-test or CI paths.
- Reusing an already-loaded `bluesky` collection is recommended for repeated query tuning.
- The benchmark file reuses helper functions in the `tests/e2e` package, including `baseURL` and `postJSON`.
- The benchmark depends on server-side SQL support but should not introduce new production code paths.
- If a query exposes a parser or execution bug, move the fix into the SQL parser PR and keep this PR as the harness only.

## CI Policy

This test should remain opt-in:

- It requires a large local dataset path.
- It mutates a real Milvus instance by creating and dropping `bluesky`.
- It can take much longer than normal E2E tests.
- It measures latency, which is noisy in shared CI environments.

The `-bluesky-data` flag is therefore required. Without it, `TestJSONBench1M` skips immediately.

## PR Boundary

This JSONBench PR should contain only:

```text
tests/e2e/jsonbench_1m_test.go
docs/jsonbench-e2e-design.md
```

It should not include SQL parser, proxy, proto, or segcore changes. Those belong to the SQL parser PR.

Reviewer checklist:

- The PR diff contains only `tests/e2e/jsonbench_1m_test.go` and this design document.
- The test skips when `-bluesky-data` is absent.
- The runner can reuse an existing `bluesky` collection for repeated manual runs.
- Query failures are summarized instead of hiding later query results.
- No dataset download URL, credential, or machine-specific path is hardcoded.

