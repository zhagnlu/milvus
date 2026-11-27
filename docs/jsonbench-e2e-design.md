# JSONBench SQL Query E2E Design

Latest records (2026-11-11):

- [Standalone shutdown fix](#standalone-shutdown-verification-2026-11-11):
  original exit `-6` reproduced; repaired binary exits `0` after SQL E2E.

- [Actual CPU and memory usage](#recorded-cpu-and-memory-usage-2026-11-10):
  all eight engines, per-query CPU utilization and comparable post-query RSS snapshots.
- [Completion audit and local comparison summary](#continuation-completion-audit-2026-11-10):
  all three previously pending local engines completed; historical results preserved.
- [Milvus versus ClickHouse CPU retest](#clickhouse-cpu-retest-results-2026-11-10):
  two completed paired rounds, independently checked responses and server CPU counters.
- [Milvus versus Elasticsearch with exact semantics](#recorded-same-host-elasticsearch-comparison-2026-11-10):
  two completed paired rounds with exact Q2 distinct counts and microsecond precision.
- [Milvus versus StarRocks](#recorded-same-host-starrocks-comparison-2026-11-10):
  two completed paired rounds with query caching disabled and full result checks.
- [Milvus versus Doris](#recorded-same-host-doris-comparison-2026-11-10):
  two completed paired rounds with SQL result and query caches disabled.

- [Milvus versus DuckDB on the same host](#recorded-same-host-comparison-2026-11-06):
  complete HTTP timings, independent result oracle and reader/CPU diagnostics.
- [Milvus versus ClickHouse on the same host](#recorded-same-host-clickhouse-comparison-2026-11-06):
  complete result timings with typed JSON paths, exact DISTINCT and aligned precision.
- [Milvus versus PostgreSQL/MongoDB on the same host](#recorded-same-host-postgresqlmongodb-comparison-2026-11-06):
  completed resumed measurements, exact results and separate execution plans.
- [All-system reference comparison](#latest-all-system-reference-comparison-2026-11-06):
  current Milvus numbers alongside ClickHouse, PostgreSQL, MongoDB, StarRocks,
  Doris and Elasticsearch, with hardware/query/index/precision qualifications.

The [November 9 attempt](#clickhouse-cpu-retest-preparation-2026-11-09) was blocked
by socket permissions. Approved execution outside the sandbox resolved that
block for the November 10 ClickHouse retest and Elasticsearch comparison.
Local StarRocks, Doris and Elasticsearch comparisons are complete. Doris's
primary comparison explicitly disables SQL result caching; the exploratory
cached run is retained separately.

## Local continuation (2026-11-24)

The [final record](benchmark-results/20261124-jsonbench-final.txt) contains a fresh
eight-system, same-host comparison after the retained P16 Go regrouping change.
All 800 timed and 160 warmup responses passed the independent million-row oracle;
complete responses and CPU counters are archived. P16's Q2 improvement is
qualified separately from Q1 latency variation. SQL semantics, schema/index
policies, CPU accounting and memory qualifications accompany the measurements.

The [P16 report](benchmark-results/20261124-jsonbench-p16-and-peers.json) preserves
all controls and allocation/phase evidence. The
[P17 report](benchmark-results/20261124-jsonbench-p17-rejected.json) records the
rejected native literal-batching experiment, including an excluded contaminated
cohort and clean repeats. Earlier sections below remain dated historical records.
No QPS or full-suite/cloud/fault-injection result is claimed.

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

This scope and the environment assumptions below describe the Go E2E entry
point. The separate, opt-in
[same-host comparison](#recorded-same-host-comparison-2026-11-06) also manages
isolated local database processes and retains its own preparation checkpoints.

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
go test -tags dynamic,test -gcflags='all=-N -l' -count=1 -v -run TestJSONBench1M \
  -milvus-addr=localhost:19530 \
  -bluesky-data=$HOME/data/bluesky/file_0001.json.gz \
  -timeout 30m
```

The test requires `-milvus-addr` and either `-bluesky-data` or explicit `-bluesky-reuse`; otherwise it skips. This skip behavior is the CI safety mechanism: normal E2E runs can include the file without requiring a large local dataset.

Useful run variants:

```bash
# Reuse a previously loaded collection when possible.
go test -tags dynamic,test -gcflags='all=-N -l' -count=1 -v -run TestJSONBench1M \
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

Reuse requires `-bluesky-reuse` and a verified count of exactly 1,000,000 rows. The caller must identify the dataset; a reused collection is not fingerprinted. Without the flag, creation fails if `bluesky` already exists. The harness never deletes an existing collection.

## Collection Lifecycle

When data is not already loaded:

1. Create a new collection; do not replace an existing collection.
2. Insert exactly the first 1,000,000 valid JSON rows in batches. Preserve raw JSON integer precision and fail on malformed input or a short dataset.
3. Create a `FLAT` index on `dummy_vec`.
4. Load the collection.
5. Poll load state until `LoadStateLoaded`. Loading does not prove that all rows are sealed.

The benchmark uses REST v2 APIs only. It does not start an embedded Milvus cluster.

Lifecycle failure handling:

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

Failures are recorded in the summary and mark the Go test as failed, without
stopping after the first failed query. Errors from all three executions are
checked. This makes it easier to see which SQL features still fail on a build
without reporting a misleading passing test.

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
- It mutates a real Milvus instance by creating and inserting into `bluesky`,
  or explicitly reuses an existing collection; the harness does not drop it.
- It can take much longer than normal E2E tests.
- It measures latency, which is noisy in shared CI environments.

The test requires an explicit target address and a dataset path, or `-bluesky-reuse` for an already prepared dataset.

## PR Boundary

This JSONBench PR should contain only:

```text
tests/e2e/jsonbench_1m_test.go
docs/jsonbench-e2e-design.md
```

It should not include SQL parser, proxy, proto, or segcore changes. Those belong to the SQL parser PR.

Reviewer checklist:

- The PR diff contains only `tests/e2e/jsonbench_1m_test.go` and this design document.
- The test skips without an explicit target and data/reuse option.
- The runner can reuse an existing `bluesky` collection for repeated manual runs.
- Query failures are summarized instead of hiding later query results.
- No dataset download URL, credential, or machine-specific path is hardcoded.

## Recorded measurements (2026-11-05)

**Runtime correction:** the original audited timings below used a Go executable
built at 12:15, before the JSON predicate pushdown added at 16:59. Stats were
actually read, but those timings do not describe the current source. See the
[rebuilt-runtime measurements and diagnosis](#rebuilt-runtime-and-bottleneck-diagnosis).
The historical on/off measurements remain useful for that binary only.

This is a local performance smoke test and an external published-results
reference, not a same-machine system ranking. The numeric record is saved in
[20261105-jsonbench-1m.json](benchmark-results/20261105-jsonbench-1m.json):
it contains the measured samples, reader counts, binary hashes, external raw
timings, source URLs and the upstream commit. It does not depend on the local
ignored `_artifacts` directory to reproduce the tables below.

### Local setup and verification

- Source baseline: `95607764c375e9401f4f29159e7ab6607baa6078`, plus the
  uncommitted SQL implementation in this workspace. This is a development build.
- Hardware: 16 vCPU, AMD EPYC 7H12, approximately 61 GiB RAM.
- Deployment: standalone, local storage, RocksMQ and an isolated embedded etcd.
  C++ used the Release build in `cmake_build/src/libmilvus_core.so`; the installed
  copy under `internal/core/output/lib` was stale and was not used for these
  measurements. The Go service was an optimized build.
- Dataset: the first 1,000,000 events of `file_0001.json.gz`; compressed size
  135,176,827 bytes, SHA256
  `2e781e750b8826041763e48e87dbf6497162ca51501e67a098f64e2e011f0aa2`.
- All rows were flushed and sorted; JSON stats construction completed before
  release/load. The final layout was two sealed segments containing 965,000 and
  35,000 rows. Segment IDs and row counts stayed unchanged across the measured
  configurations. An exploratory run affected by background merging is excluded.
- `common.groupBy.maxGroups=1000000` was necessary: Q2's exact DISTINCT expands
  to 271,874 intermediate groups and fails with the default limit of 100,000.
  Both configurations used the same raised limit. Stats construction remained
  enabled; only `common.usingJSONShreddingForQuery` changed between them.
- Background compaction was disabled during measurement. No concurrent writes
  or other client queries were issued. Strong `COUNT(*)` confirmed 1,000,000 rows
  before each configuration.

The repeat-run audit used a separate temporary driver, rather than changing
`TestJSONBench1M`'s three-execution contract. It wrapped the real
`JsonKeyStats::ShreddingReader::Get` and `CreateShreddingReader` functions with
`LD_PRELOAD`, forwarding their arguments and results. Counters recorded actual
non-null STRING and INT64 returns; reader-creation records included each
segment's row count and the availability of each requested typed path. Production
source files and the core library were not changed for this audit.

Every enabled sample had successful stats reads. Required readers covered all
1,000,000 segment rows, and Q3-Q5 had successful INT64 reads. Every disabled
sample had zero reader calls and no reader creation. The complete query results
in both configurations matched the previously independently computed oracle.
All 27 SQL functional E2E subtests also passed in this session (5.772 seconds).

### Stats on/off performance

Each query had two warmups followed by ten timed executions, sequentially with
one client. The table reports the median in milliseconds. The timer covered
HTTP request execution through reading the complete response body, excluding
client JSON decoding, counter snapshots and correctness checks. The temporary
reader counters and creation records were active during measurement, and their
overhead has not been independently quantified. The OS page cache was not cleared.

| Query | Stats enabled median (ms) | Stats disabled median (ms) | Disabled / enabled | Enabled min-max (ms) |
| --- | ---: | ---: | ---: | ---: |
| Q1 | 711.333 | 1151.189 | 1.62x | 695.969-728.486 |
| Q2 | 3037.772 | 4913.417 | 1.62x | 3013.799-3090.735 |
| Q3 | 2973.255 | 5239.467 | 1.76x | 2942.404-3003.683 |
| Q4 | 2063.215 | 3679.871 | 1.78x | 2043.516-2122.832 |
| Q5 | 2036.289 | 3670.155 | 1.80x | 2002.026-2090.913 |

Actual reader returns per enabled query were stable across the measured samples:

| Query | Successful STRING values | Successful INT64 values | Missing stats values |
| --- | ---: | ---: | ---: |
| Q1 | 994,672 | 0 | 5,328 |
| Q2 | 3,903,894 | 0 | 0 |
| Q3 | 3,537,611 | 588,328 | 0 |
| Q4 | 3,036,095 | 86,812 | 0 |
| Q5 | 3,036,095 | 86,812 | 0 |

These are value reads across multiple JSON paths, not unique row counts.
Missing/NULL paths can still fall back to raw JSON. This verifies that the
queries use stats, not that every raw JSON read has been eliminated. It also
does not establish P95/P99, concurrency throughput, remote-storage performance
or nullable/mixed-type coverage. The earlier growing/sealed smoke run had a
different segment layout and must not be used to infer the stats speedup.
The embedded-etcd client still panics with `context canceled` on shutdown after
the timed queries; that separate issue remains unresolved.

Detailed local-only logs, reader traces and the audit sources are archived at
`tests/e2e/_artifacts/jsonbench-jsonstats-20261105/`. They are ignored by Git;
the numeric snapshot above preserves the measurements and provenance.

### External published reference: one million rows

JSONBench now lives in
[ClickBench's `json/` directory][upstream-methodology]. This reference was
retrieved on 2026-11-05 and pinned to commit
`ff15af5d8aae5ce5ce79b8e479379582f9050fdc`. The selected public runs were dated
2026-11-02 or 2026-11-03, with all 1,000,000 documents loaded. Their recorded
machine is `m6i.8xlarge`; the upstream default is 32 vCPU, 128 GB RAM and a
2,000 GB gp3 volume, running Ubuntu 24.04. None of these external systems was
rerun on the local Milvus machine.

The upstream driver runs each query once cold and twice hot. For this table,
the external number is `median(result[q][1], result[q][2]) * 1000`, which for
two samples is their arithmetic mean. The cold sample is excluded. The Milvus
rows identify their sample schedules: three measured runs without the reader
audit for the rebuilt executable, and ten audited runs for the historical
executable. This is an explicitly derived comparison metric, not a reproduction
of the dashboard's ranking.

| System / version | Run date | Q1 (ms) | Q2 (ms) | Q3 (ms) | Q4 (ms) | Q5 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Milvus rebuilt Go, stats enabled, median of 3 | 2026-11-05 | 690.094 | 1880.528 | 991.966 | 346.718 | 313.086 |
| Milvus earlier Go, audited, median of 10 | 2026-11-05 | 711.333 | 3037.772 | 2973.255 | 2063.215 | 2036.289 |
| [ClickHouse 26.10.3.7][result-clickhouse] | 2026-11-03 | 4.0 | 20.5 | 7.5 | 15.5 | 15.5 |
| [DuckDB 2.0.0, JSON][result-duckdb] | 2026-11-02 | 103.0 | 363.0 | 376.0 | 237.0 | 247.5 |
| [PostgreSQL 18.6][result-postgresql] | 2026-11-02 | 109.5 | 2701.5 | 429.0 | 191.0 | 208.0 |
| [MongoDB 9.0.4][result-mongodb] | 2026-11-02 | 1056.5 | 1518.5 | 1363.5 | 213.5 | 232.0 |
| [StarRocks 4.1.7][result-starrocks] | 2026-11-02 | 59.0 | 79.0 | 76.5 | 51.0 | 55.5 |

Additional published workloads with stronger comparability qualifications:

| System / version | Q1 (ms) | Q2 (ms) | Q3 (ms) | Q4 (ms) | Q5 (ms) | Qualification |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| [Apache Doris 4.1.5][result-doris] | 55.0 | 60.0 | 60.0 | 40.0 | 45.0 | Queries use generated columns derived from VARIANT data. |
| [Elasticsearch 9.5.6][result-elasticsearch] | 91.0 | 78.0 | 84.0 | 65.0 | 82.0 | Approximate Q2 distinct count; different Q3 ordering; server-side timing. |

### Cross-hardware latency ratios

The latest rebuilt Milvus results give the following ratios against the public
warm timings. Each cell is **local Milvus latency / published system latency**:
above 1 means the local measurement took longer, below 1 means it took less time.
These are **raw cross-environment latency ratios, not engine slowdown factors**.
They must not be presented as a hardware-normalized ranking.
The exact calculation and source references are preserved in
[20261105-jsonbench-comparison.json](benchmark-results/20261105-jsonbench-comparison.json).

| Published system | Q1 | Q2 | Q3 | Q4 | Q5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ClickHouse | 172.52x | 91.73x | 132.26x | 22.37x | 20.20x |
| DuckDB | 6.70x | 5.18x | 2.64x | 1.46x | 1.26x |
| PostgreSQL | 6.30x | 0.70x | 2.31x | 1.82x | 1.51x |
| MongoDB | 0.65x | 1.24x | 0.73x | 1.62x | 1.35x |
| StarRocks | 11.70x | 23.80x | 12.97x | 6.80x | 5.64x |
| Apache Doris, generated-column workload | 12.55x | 31.34x | 16.53x | 8.67x | 6.96x |
| Elasticsearch, non-equivalent Q2/Q3 | 7.58x | 24.11x | 11.81x | 5.33x | 3.82x |

The last two rows retain the schema/semantic qualifications above. For example,
the ClickHouse Q1 cell means 690.094 ms / 4 ms = 172.52 for these two recorded
environments. It does not establish that the Milvus engine is 172.52 times
slower on identical hardware.

### What the external comparison does and does not show

The published ClickHouse, DuckDB and StarRocks latencies are lower than the
rebuilt Milvus run for all five queries. Rebuilt Milvus Q2 is numerically below
the published PostgreSQL value, while Milvus Q1 is lower than the published
MongoDB value; PostgreSQL's other four values and MongoDB's other four values
are lower. These observations describe the recorded numbers, not normalized
engine speed ratios or a same-host ranking.

The query and schema audit found the following material differences:

- Hardware and sampling differ: local Milvus has 16 vCPU and about 61 GiB RAM,
  versus the upstream 32 vCPU / 128 GB default; three rebuilt-runtime samples
  (or ten historical samples) versus two hot samples. Do not multiply or divide
  results by the CPU count to normalize them.
- Timing boundaries differ. Milvus includes HTTP response transfer.
  [ClickHouse's runner][query-clickhouse] uses client timing and suppresses
  output with `FORMAT Null`; [PostgreSQL][query-postgresql] uses psql timing;
  [MongoDB][query-mongodb] times the aggregation through `toArray()`;
  [Elasticsearch][query-elasticsearch] uses the server's `took` field.
- Physical layouts differ. [ClickHouse][schema-clickhouse] declares typed JSON
  paths and a workload-specific sort order. [PostgreSQL][schema-postgresql] and
  [MongoDB][schema-mongodb] define compound indexes on the queried paths.
  [StarRocks][schema-starrocks] retains JSON with a generated sort key.
  [Doris][schema-doris] exposes generated scalar/time columns, which its queries
  read directly. These are not equivalent unindexed raw-JSON scans.
- Elasticsearch Q2 uses `COUNT_DISTINCT(did, 40000)`, which remains
  [approximate][elastic-distinct], unlike Milvus's exact DISTINCT. Its
  [Q3][queries-elasticsearch] sorts by count and event rather than hour and
  event. Its full Q1-Q5 vector is therefore not treated as semantically identical.
- Timestamp precision and evaluation order differ. MongoDB converts timestamps
  to millisecond Date values. Several Q5 variants return integral millisecond
  spans while Milvus retains fractional milliseconds. DuckDB Q4/Q5 aggregate
  extracted timestamp text before casting, whereas Milvus casts before MIN/MAX.
  Missing-path/NULL representations can also differ. External result sets were
  not independently checked against the local oracle.

An apples-to-apples engine comparison remains a follow-up: run the selected
systems on the same hardware with the same dataset fingerprint, aligned query
results/precision, explicit schema/index choices, matching timing boundaries,
and the same warmup/sample schedule. This document records public reference
numbers without claiming that that follow-up has happened.

### External CPU configuration audit

The selected public results identify a 32-vCPU machine, but do not record
per-query CPU seconds, active cores or launched query workers. The shared
[benchmark driver][benchmark-driver] runs queries sequentially, with three
executions per query by default (one cold and two hot). This is a single-query
latency test; a single client does not imply a single execution thread.

The install/start/load/query scripts at the pinned upstream revision show:

| System | Configuration evidence | Actual cores used by Q1-Q5 |
| --- | --- | --- |
| ClickHouse | No fixed `max_threads` override in the audited scripts; automatic/default server settings. | Not recorded. |
| DuckDB | No `SET threads` override; the documented default is the CPU core count. | Not recorded; a thread budget is not utilization. |
| PostgreSQL | The installer explicitly retains default settings. PostgreSQL 18 defaults to at most two parallel workers per Gather; the leader may participate. | Not recorded; a plan may launch fewer workers or use no parallel plan. |
| MongoDB | One mongod; no explicit query CPU/thread cap in the audited scripts. | Not recorded. |
| StarRocks | One FE and one BE on the host; no explicit query DOP override found. | Not recorded. |
| Apache Doris | Load script sets `parallel_pipeline_task_num=32` and `enable_parallel_scan=false`. | Not recorded; 32 is a per-fragment task setting, not measured busy cores. |
| Elasticsearch | One node; index template sets one primary shard and zero replicas; no explicit processor cap found. | Not recorded; one shard does not establish one CPU core. |

Sources: the pinned upstream per-system install/start/load/query scripts,
[Doris load settings][load-doris], [Elasticsearch index template][index-elasticsearch],
[PostgreSQL installer][install-postgresql],
[PostgreSQL parallel-worker defaults][postgresql-parallel-settings],
[DuckDB thread defaults][duckdb-thread-settings], and
[Doris task-parallelism semantics][doris-task-parallelism]. Defaults describe
expected configuration, not a captured runtime-settings or query-profile dump.

The local Q2 profile's approximately 1.08 average process CPU cores is a
different kind of evidence from the public host's 32 available vCPU. There is
no basis to say the comparison is "Milvus used one core and every other system
used 32", or to divide the latency ratios by 32. A controlled retest must
capture database process-tree CPU seconds / query wall seconds and query plans
or profiles, as well as the configured resource budget. Keep background work
and profiling overhead separate from clean latency samples.

### How to make a fair hardware comparison

CPU count is not a conversion factor. Doubling vCPU does not generally halve
latency: single-core performance, memory bandwidth, cache sizes, disk behavior,
query parallelism and physical layout all affect the result differently. The
local Q2 CPU profile averaged only about 1.08 process CPU cores on a 16-vCPU
machine. More available cores alone would not establish a proportional gain.
There is no measured correction factor for the AMD host versus the upstream
`m6i.8xlarge`, so none is applied here.

The current evidence supports these different levels of comparison:

| Existing evidence | Supported use | Limitation |
| --- | --- | --- |
| Clean old/rebuilt Milvus on this host | Measure the rebuilt runtime's effect with the same data, stats setting and segment layout. | Three timed samples; not an external-system comparison. |
| Historical Milvus stats on/off audit | Verify actual reader use and compare those configurations of that same historical executable. | Not a measurement of the rebuilt executable's on/off gain. |
| Local Milvus versus published other systems | Identify reference latency ranges and candidates for a controlled retest. | Different hardware and measurement policies; no engine ranking or normalized speed ratio. |
| Profile and native phase measurements | Locate work inside this Milvus implementation, such as projection CPU or fallback row counts. | Diagnostic instrumentation; not a substitute for external-system benchmarking. |

For a defensible cross-system comparison, rerun the selected systems on the
current Milvus host, sequentially, with only the tested database active. This
avoids estimating a cross-machine correction. Alternatively, rerun Milvus and
the selected systems on the upstream machine configuration; changing only the
Milvus machine is insufficient if the other setup details still differ.
At the end of the 2026-11-05 investigation, neither retest had been executed.
The [2026-11-06 DuckDB comparison](#recorded-same-host-comparison-2026-11-06)
below subsequently tested both engines on the local AMD host.

The controlled run should use:

1. The same CPU affinity/resource limit, memory budget, filesystem/storage and
   client placement. Record actual CPU time, memory and I/O rather than assuming
   every engine uses all available cores equally.
2. The same dataset fingerprint and 1,000,000-row count, with complete loading,
   persistence, stats/index builds and no concurrent ingestion or background
   compaction during measurement. Verify the exact executable/library hashes.
3. Aligned query results: exact DISTINCT, UTC, agreed timestamp precision,
   missing/NULL behavior and ordering. Exclude approximate or differently sorted
   variants from an exact-semantics comparison.
4. An explicit schema/index policy. Record native JSON optimizations and indexes;
   report generated scalar-column workloads separately unless the same option
   is deliberately allowed for every system. Include preparation time and
   storage size when evaluating those choices.
5. The same timing boundary, preferably client wall time through consuming the
   complete response, plus two warmups and ten timed executions per query.
   Report medians and ranges; run cold-cache measurements as a separate series
   with a common reset procedure. Rotate system order across repeated rounds.
6. A separate reader/plan validation pass and an uninstrumented timing pass with
   identical binaries, data layout and configuration. This keeps proof of stats
   use while avoiding unequal audit overhead across engines.

Until that run exists, the public table must remain labeled as reference data.
The current implementation bottlenecks below are supported by local measurements
and do not rely on attributing the cross-hardware ratios to engine efficiency.

### Rebuilt runtime and bottleneck diagnosis

The follow-up found an execution-artifact mismatch: C++ had been rebuilt and
selected correctly, but `bin/milvus` was still the 12:15 Go build. Only two
runtime Go files in the SQL path were newer than that binary:
`internal/sqlparser/json_predicate.go` and `internal/proxy/dql/sql_query.go`,
both updated at 16:59. The old binary did not contain `LowerJSONPredicate`;
the rebuilt binary does. As a result, the old execution interpreted JSON WHERE
clauses inside the scalar projector instead of using the new native predicate
path. Rebuilding Go restored this existing source behavior; no new production
optimization was introduced by this diagnosis.

The same dataset, core library, stats-enabled configuration and two sealed
segments were used for clean old/new measurements. Neither executable used
the per-value reader audit wrapper. Each query had one warmup and three timed
runs; the table is their median, not the previous ten-sample statistic.
Full samples, binary hashes, native phase events and CPU-profile summaries are
in [20261105-jsonbench-diagnosis.json](benchmark-results/20261105-jsonbench-diagnosis.json).

| Query | Old Go, no audit (ms) | Rebuilt Go, no audit (ms) | Old / rebuilt |
| --- | ---: | ---: | ---: |
| Q1 | 690.941 | 690.094 | 1.00x |
| Q2 | 3000.525 | 1880.528 | 1.60x |
| Q3 | 2897.052 | 991.966 | 2.92x |
| Q4 | 2017.805 | 346.718 | 5.82x |
| Q5 | 1998.798 | 313.086 | 6.38x |

Removing the original audit wrapper changed the old-runtime medians by only
about 1%-3%, rather than explaining the large gap. Native phase records from
the rebuilt runtime confirm that filtering happens before scalar projection:
Q2, Q3 and Q4/Q5 project 954,611, 588,328 and 86,812 rows respectively.
All five query results matched the million-row oracle. All 27 functional SQL
subtests passed against the rebuilt executable (6.568 seconds).

Remaining measured costs:

| Query | Diagnostic HTTP wall (ms) | Largest segment projection wall (ms) | Projection CPU, summed (ms) | Native grouping CPU, summed (ms) | Raw JSON rows fetched by projection |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q1 | 721.308 | 581.865 | 609.411 | 99.692 | 995,720 |
| Q2 | 2018.533 | 749.749 | 776.507 | 262.651 | 0 |
| Q3 | 991.634 | 873.104 | 904.794 | 60.955 | 0 |
| Q4 | 336.173 | 109.598 | 113.751 | 13.403 | 0 |
| Q5 | 319.068 | 111.516 | 115.513 | 14.405 | 0 |

These are separate coarse-instrumentation samples after warmup, not additional
clean latency medians. CPU sums include both segment workers and must not be
subtracted directly from HTTP wall time.

1. **The scalar projection remains row-at-a-time.**
   `EvaluateSqlProjection` loops over each output column and selected row,
   recursively interprets the protobuf scalar expression and writes a
   materialized value. Stats reads avoid reparsing JSON on a hit, but do not
   remove per-row dispatch, variant handling, string construction/copies or
   intermediate column writes. The large Q3 projection cost is measured;
   assigning exact percentages to individual expression operations would
   require a more detailed native profile.
2. **Missing typed cells amplify raw fallback.** Q1 has only 5,328 missing
   stats values, but the 1,024-row `JsonSource` cache caused 973 raw bulk reads
   covering 995,720 rows. Those bulk reads consumed 95.816 ms of CPU in the
   diagnostic sample. The amplification is real, but it is only part of Q1's
   609.411 ms projection CPU cost. A fix must preserve mixed-type/path semantics;
   an absent typed cell must not simply be treated as SQL NULL.
3. **DISTINCT carries a large intermediate group set through Go reducers.**
   Q2 expands to 271,874 `(event, did)` keys rather than only its 13 final groups.
   Its 15-second CPU profile recorded 16.21 CPU seconds: approximately 15% of
   sampled CPU was under Proxy `sqlquery.ReduceAggregate`, approximately 11%
   under QueryNode `GroupAggReducer.Merge`, and approximately 7% under background
   Go GC. These are process CPU samples, not wall-time percentages; nested
   cumulative functions must not be added repeatedly. The Proxy reducer also
   encodes logical group keys with `json.Marshal` per intermediate row.
4. **The test layout exposes limited parallelism.** One segment holds 96.5%
   of the rows. Projection loops execute serially within that segment. The Q2
   profile averaged about 1.08 process CPU cores over its 15-second interval,
   despite the machine having 16 vCPU. This is observed utilization for this
   workload, not a claim about all Milvus query parallelism.
5. **Even simple SQL COUNT materializes unnecessary rows.** Rebuilt
   `SELECT count(*)` took 145.607 ms versus 1.830 ms for the ordinary REST
   `outputFields: ["count(*)"]` path. Its diagnostic projection produced
   1,000,000 literal values and consumed 139.822 ms CPU, while native global
   grouping consumed only 0.014 ms. The SQL lowering currently uses COUNT(1)
   with a projected literal for compatibility, missing the ordinary count path.

The remaining optimization priorities are batch/typed expression evaluation
and constant/count fast paths, sparse raw fallback over only the missing
offsets, more efficient exact-DISTINCT state merging, and bounded parallel work
inside large segments. They have been diagnosed but not implemented here.
The embedded-etcd shutdown panic also remains unchanged.

### Recorded same-host comparison (2026-11-06)

The controlled local retest is complete for **Milvus versus DuckDB 1.5.6**.
The [opt-in comparison runner](../tests/e2e/jsonbench_compare.py) starts the
engines sequentially, preserves its dataset/runtime/checkpoints on disk and
refuses an incomplete Milvus ingestion. The
[numeric record](benchmark-results/20261106-jsonbench-same-host.json) contains
every timed sample, both dialects' SQL, executable hashes, schemas, hardware,
CPU observations and ordering of the two rounds. No production execution code
was optimized in this investigation.

Measurement conditions:

- Same 16-vCPU AMD EPYC 7H12 host, presented as eight cores with two threads
  each, about 61 GiB RAM; both database processes used CPU affinity `0-15`.
  DuckDB used `threads=16`. It had a 48 GiB database memory limit; there was no
  common enforced process memory cap. This is a shared host resource comparison,
  not an identical memory-limit experiment.
- Identical fingerprinted source gzip and first 1,000,000 nonempty JSON records.
  The source SHA-256 remains
  `2e781e750b8826041763e48e87dbf6497162ca51501e67a098f64e2e011f0aa2`;
  the extracted NDJSON SHA-256 is
  `7beb29f6c036fe784754ff34d68d1f216c6cc89de12155da06f725bdf5c8536e`.
  DuckDB's stored JSON was streamed back in insertion order and fingerprinted
  against that NDJSON before each engine run.
- Milvus retained the previous Go/core hashes, used local storage/RocksMQ and
  embedded etcd, enabled JSON stats and set `common.groupBy.maxGroups=1000000`.
  Sorting/stats preparation finished before timing; compaction was disabled
  during timing. Five sealed segments contained **225,000 / 224,011 / 220,989 /
  225,000 / 105,000** rows. IDs and row counts stayed fixed across both rounds.
- Both engines used native JSON data, without generated scalar columns or JSON
  expression indexes. Milvus additionally required the same dummy vector/FLAT
  index as the original loader. Its built-in shredding layout was enabled;
  DuckDB's recorded plans scan its JSON column.
- Round one ran Milvus then DuckDB; round two reversed the order. Each query
  received two warmups and ten timed executions per round, giving 20 measured
  samples per engine/query. The table reports the median of those 20 samples.
  These are warm single-client runs; OS caches were not reset.
- A shared Python HTTP client consumed the complete response before stopping
  each timer. Milvus used its REST service; DuckDB used a small Python HTTP
  adapter with persistent connections and TCP_NODELAY. Database startup, load,
  preparation, response JSON decoding and correctness checks were excluded.
  The adapter/framework difference remains part of this comparison's scope.

DuckDB received equivalent direct `json_extract_string(data, '$.…')`
expressions. Its arrow operator precedence differs from PostgreSQL's; sending
the original unparenthesized predicates directly failed. UTC, exact DISTINCT,
grouping, ordering and timestamp/span calculations were otherwise aligned.
Every one of the **200 timed queries and 40 warmups** matched independent Python
counters, sets and min/max calculations over the source. Integer counts and
distinct cardinalities matched exactly; timestamp and span tolerances were
1 microsecond and 0.0011 milliseconds respectively.

| Query | Milvus median (ms) | DuckDB median (ms) | Milvus / DuckDB latency |
| --- | ---: | ---: | ---: |
| Q1 | 189.385 | 119.501 | 1.585x |
| Q2 | 1115.076 | 516.568 | 2.159x |
| Q3 | 252.661 | 632.925 | 0.399x |
| Q4 | 233.563 | 393.015 | 0.594x |
| Q5 | 208.195 | 374.760 | 0.556x |

Ratios above one mean higher Milvus latency. In this local configuration the
remaining largest gap is Q2 exact DISTINCT. Q3-Q5 have lower Milvus latency.
These results do not establish a general ranking or reproduce the published
DuckDB result: that record used **DuckDB 2.0.0**, different hardware and native
benchmark timing. The original public Q4 also constructs timestamps after MIN,
and Q5 uses integer `date_diff('milliseconds', …)`; that Q5 precision differs
from the fractional-millisecond acceptance query used here.

The new Milvus numbers must also not be presented as a source-code speedup
over November 5. The binary/core hashes are unchanged, but the earlier layout
was 965,000 + 35,000 rows. The present layout exposes more segment workers.
The observed timing/utilization difference supports investigating layout and
parallelism; it does not isolate their causal contribution.

Process CPU seconds divided by query wall seconds, aggregated over the timed
samples, gives the following observed average busy-CPU equivalents. Background
threads are included; these are not launched-worker counts:

| Query | Milvus CPU / wall | DuckDB CPU / wall |
| --- | ---: | ---: |
| Q1 | 4.351 | 7.274 |
| Q2 | 2.326 | 7.856 |
| Q3 | 4.395 | 7.820 |
| Q4 | 1.908 | 7.748 |
| Q5 | 1.989 | 7.697 |

For Q2, average process CPU per query was 2.593 seconds for Milvus and
3.983 seconds for DuckDB. The longer Milvus wall time together with lower CPU
utilization makes its serial reduction/materialization path a concrete next
investigation target. CPU-count scaling still cannot predict a speedup.

Separate validation and diagnostics, excluded from the clean samples:

- An [actual reader audit](benchmark-results/20261106-jsonbench-reader-audit.json)
  used an `LD_PRELOAD` wrapper on the same library and segment IDs. Required
  readers covered all 1,000,000 rows. Q1 returned 994,672 string cells and 5,328
  missing cells; Q2 returned 1,909,222 string cells; Q3 returned 588,328 strings
  and 588,328 INT64 values; Q4/Q5 each returned 86,812 of each. All five results
  again matched the oracle. No audit wrapper was loaded for the clean timings.
  This retest did not measure the current layout with stats disabled.
- A [15-second Q2 CPU profile and DuckDB plans](benchmark-results/20261106-jsonbench-diagnosis.json)
  were captured in a separate pass. Fourteen profiled Q2 requests passed the
  oracle. Of 27.45 sampled CPU seconds, QueryNode's `GroupAggReducer.Merge`
  accounted for 17.78% cumulatively, Proxy `sqlquery.ReduceAggregate` 17.63%,
  background Go marking 7.98%, and `json.Marshal` 5.06%. Nested cumulative
  percentages overlap and must not be added. Native unwinding is incomplete;
  these are sampled process CPU shares, not wall-time phase percentages.
  Source inspection confirms native DISTINCT extends the grouping key,
  QueryNode builds/merges row objects and reassembles protobuf columns, and
  Proxy serializes logical keys and rebuilds exact sets. The input has 271,874
  distinct `(event, did)` classes but only 13 final event groups.
- DuckDB's Q2 plan scans 1,000,000 JSON rows, filters to 954,611, and uses a
  `HASH_GROUP_BY` with `count(DISTINCT)` yielding 13 rows. Its profiled total
  wall time was 0.530 seconds; parallel operator times must not be treated as
  additive wall times. The adapter's failed-query response was also exercised:
  a missing-table error produced a nonzero code and was rejected by the client.
- Equivalent Q4/Q5 SQL variants delayed timestamp construction until after
  numeric MIN/MAX, or computed the integer-microsecond difference directly.
  One additional ten-sample series gave Milvus 230.180 / 190.404 ms and DuckDB
  399.520 / 391.236 ms. All results passed the same oracle. This is a separate
  exploratory series, not evidence of a robust cross-round optimization gain;
  neither the acceptance queries nor production code were changed.

Reproduction with the same checked-out SQL binary/core builds:

```sh
python3 -m pip install --target .benchmarks/jsonbench-run/python \
  duckdb==1.5.6 pymilvus==3.0.2 requests==2.34.2
# Download the source gzip separately and verify its SHA-256 above.
PYTHONPATH=.benchmarks/jsonbench-run/python python3 tests/e2e/jsonbench_compare.py \
  --run-dir .benchmarks/jsonbench-run --data /absolute/path/file_0001.json.gz
```

The completed run's data, logs, profile, wrapper source/library and diagnostic
scripts remain under `.benchmarks/jsonbench-20261106/`, which is ignored by Git.
The reviewable numeric snapshots are saved under `docs/benchmark-results/`.
Both engines were stopped after testing. Milvus still emitted the existing
`context canceled` shutdown panic after requested termination (exit -6); that
shutdown problem was not fixed. Nullable/mixed-type stats comparisons and cloud
deployment also remain separate unfinished work. No Go/C++ production source
was changed by this comparison, and the prior functional suite was not rerun.

### Recorded same-host ClickHouse comparison (2026-11-06)

The [paired numeric record](benchmark-results/20261106-jsonbench-clickhouse-same-host.json)
contains two reversed-order rounds of Milvus and ClickHouse 26.10.3.7 on the
same host and source dataset as the DuckDB run. Each query received two warmups
and ten timed executions per engine/round. Every result passed the same
independent oracle. The medians below use all 20 timed samples per engine/query;
the Milvus samples belong to this pair, rather than the earlier DuckDB pair.

| Query | Milvus median (ms) | ClickHouse median (ms) | Milvus / ClickHouse latency |
| --- | ---: | ---: | ---: |
| Q1 | 189.543 | 4.712 | 40.226x |
| Q2 | 1099.743 | 31.379 | 35.047x |
| Q3 | 256.101 | 13.056 | 19.616x |
| Q4 | 228.739 | 16.271 | 14.058x |
| Q5 | 205.062 | 16.343 | 12.548x |

ClickHouse retained the upstream typed JSON paths, LowCardinality categories
and workload-specific sort order. Its `commit.collection` path was made
nullable to preserve the missing-path NULL group, and Q5 used an integer
microsecond difference divided by 1000.0 to retain fractional milliseconds.
It returned complete JSON responses instead of the upstream `FORMAT Null`.
Timing includes consuming those responses over HTTP and excludes client JSON
decoding and correctness checks. ClickHouse used `max_threads=16` and a 48 GiB
memory limit; this is not a common enforced memory cap across both engines.

The Milvus binary/core and five-segment layout matched the DuckDB comparison.
ClickHouse's active parts stayed fixed during each timing series. The captured
PID was its watchdog, so the recorded zero CPU counters do not measure server
work and must not be used to compare CPU utilization. The runner now reads the
process tree, but validating that correction requires another timed run.
Separate plans and reader statistics are collected outside clean latency
measurements. These results describe the recorded schema policies, rather than
a raw-JSON scan ranking. Both ClickHouse runs exited normally; Milvus retained its existing
exit -6 problem after requested termination.

### Standalone shutdown verification (2026-11-11)

The shutdown follow-up fixes `kvfactory.CloseEtcdClient`, which previously
panicked on the `context.Canceled` returned by a successful embedded etcd
client close. The source trace is `CreateEtcdClient` → `GetEmbedEtcdClient` →
etcd `v3client.New` → `clientv3.NewCtxClient`. In etcd v3.5.33, `Client.Close`
cancels the client context, closes watchers and leases, then returns the context
error when there is no gRPC connection. This expected cancellation now permits
the cached client and metadata root path to be reset. Other close errors retain
their existing panic behavior.

The [shutdown verification record](benchmark-results/20261111-milvus-shutdown-verification.json)
preserves binary, core and source hashes, test results, and the verification
script. Both runs used fresh isolated embedded-etcd/local-storage/RocksMQ data
directories and the same C++ core library search path:

| Binary | Healthy before SIGTERM | Exit | Cancellation panic | Graceful-stop marker |
| --- | --- | ---: | --- | --- |
| Original benchmark `bin/milvus` | Yes | -6 | Yes, at `CloseEtcdClient` | No |
| Current-worktree build, separate output | Yes | 0 | No | Yes |

The repaired instance also passed all **27 SQL E2E subtests**, including vector
search, invalid SQL, exact DISTINCT, and JSONBench queries before flush and after
flush/reload. The focused Go regression passed for the etcd client factory,
SQL parser, SQL query executor, query utilities and shallow-copy utilities,
using `-tags dynamic,test -gcflags='all=-N -l' -count=1`. Client-close tests cover
uninitialized, embedded, remote and previously closed clients, wrapped
cancellation, deadline expiry and an unexpected internal error. The ten tested
listener ports were vacant after both runs; neither run required a forced kill.

The original benchmark binary and historical result/source files remain
unchanged. These are shutdown and functional checks; performance results were
not rerun with the repaired binary. Real standalone shutdown was tested with
embedded etcd; remote clients were covered by unit tests. Full Go/C++ regression,
nullable/mixed-type reader-stat comparisons and cloud deployment remain pending.

### Recorded CPU and memory usage (2026-11-10)

The [resource summary generator](../tests/e2e/jsonbench_resources.py) recomputes
actual usage from the recorded samples without starting databases. The
[JSON evidence](benchmark-results/20261110-jsonbench-resource-usage.json) retains
source hashes, per-query and combined metrics, and the individual PID RSS values
behind each memory snapshot. The
[CSV export](benchmark-results/20261110-jsonbench-resource-usage.csv) has all
Q1-Q5 and combined rows for each engine. An independent calculation checked
**1,100 CPU/wall samples, 110 memory snapshots and all 48 CSV rows**; source
hashes and observed CPU affinities were checked as well.

All engines had CPU affinity 0-15 on the same host, with 16 available logical
CPUs and approximately 61.4 GiB RAM. CPU below is the sum of timed process CPU
deltas divided by the sum of complete HTTP wall seconds. One busy logical CPU
equals 100% in a per-core convention; host utilization divides by all 16 logical
CPUs. This is utilization over the timed-query windows, not over loading,
startup, warmups, pauses or the entire elapsed benchmark session. It includes
background database work. CPU reads immediately surround the HTTP wall timer,
so their measurement windows differ slightly.

| Engine | Q1 busy CPUs | Q2 busy CPUs | Q3 busy CPUs | Q4 busy CPUs | Q5 busy CPUs | All-query busy CPUs | Host CPU utilization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Milvus | 4.39 | 2.35 | 4.39 | 1.89 | 1.99 | 2.72 | 17.0% |
| ClickHouse | 2.84 | 7.92 | 4.57 | 1.04 | 1.08 | 4.11 | 25.7% |
| StarRocks | 2.42 | 3.75 | 2.66 | 4.07 | 4.10 | 3.23 | 20.2% |
| Doris | 4.77 | 5.76 | 4.73 | 4.91 | 5.14 | 5.15 | 32.2% |
| Elasticsearch | 8.09 | 6.85 | 3.14 | 5.69 | 4.64 | 5.70 | 35.7% |
| DuckDB | 7.27 | 7.86 | 7.82 | 7.75 | 7.70 | 7.76 | 48.5% |
| PostgreSQL | 4.09 | 1.00 | 2.85 | 3.32 | 3.16 | 1.61 | 10.1% |
| MongoDB | 1.01 | 1.01 | 1.01 | 1.00 | 0.99 | 1.01 | 6.3% |

Memory uses each query's last timed-sample snapshot in each round. The table
sums RSS across unique live database PIDs at that snapshot, including the
ClickHouse watchdog and server, StarRocks/Doris FE and BE, PostgreSQL live
children, and the Elasticsearch JVM/controller tree. A root already present in
its process tree is counted only once. Denser memory snapshots available for
some peers are excluded to keep the sampling schedule consistent.

| Engine | Median sampled RSS sum (GiB) | Sampled minimum (GiB) | Sampled maximum (GiB) | Snapshots |
| --- | ---: | ---: | ---: | ---: |
| Milvus | 2.590 | 2.469 | 2.879 | 40 |
| ClickHouse | 0.960 | 0.783 | 0.992 | 10 |
| StarRocks | 1.193 | 0.964 | 1.283 | 10 |
| Doris | 2.984 | 2.730 | 3.046 | 10 |
| Elasticsearch | 32.402 | 32.165 | 32.628 | 10 |
| DuckDB | 1.153 | 1.014 | 1.378 | 10 |
| PostgreSQL | 0.630 | 0.176 | 0.797 | 10 |
| MongoDB | 0.957 | 0.680 | 1.202 | 10 |

These are **sampled process RSS sums, not continuous peak or unique physical
memory**. Shared resident pages are repeated across processes, OS page cache is
excluded, and PostgreSQL parallel workers that have exited before a post-query
snapshot are absent. No PSS or cgroup memory history was recorded, so those
quantities cannot be reconstructed. Elasticsearch's approximately 32.4 GiB RSS
includes its preconfigured 31 GiB JVM heap and does not mean each query allocated
that much additional memory. No common process memory cap was enforced.

Milvus pools the four primary November 10 pairs: 80 timed samples per query and
40 post-query memory snapshots overall. Each peer contributes 20 timed samples
per query and ten memory snapshots overall. DuckDB/PostgreSQL/MongoDB evidence
comes from November 6; the other engines are from November 10. Cached Doris and
the invalid historical watchdog-only ClickHouse CPU records are excluded.
DuckDB's CPU includes its in-process Python HTTP adapter; the other database
CPU measurements exclude external clients/adapters. PostgreSQL CPU includes
live descendants and reaped-worker accounting even though memory snapshots
contain only live processes.

Recompute the derived artifacts from the preserved reports:

```sh
python3 tests/e2e/jsonbench_resources.py
```

### Continuation completion audit (2026-11-10)

The four primary paired runs below completed on the original fingerprinted
1,000,000-row input with the same Milvus binary/core and fixed five-segment
layout. The [final independent audit](benchmark-results/20261110-jsonbench-continuation-verification.json)
rechecked **800 timed samples, 80 persisted full responses and 400 raw peer CPU
counter pairs**. The runners also oracle-checked 160 warmup executions at runtime;
warmups are not stored as timing samples. Six offline CPU collector regressions
passed. A separate exploratory Doris run contributes another 200 checked samples
and 20 persisted responses, and is labelled as using SQL result caching.

| Local peer | Q1 median (ms) | Q2 median (ms) | Q3 median (ms) | Q4 median (ms) | Q5 median (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ClickHouse, new CPU retest | 5.057 | 31.129 | 13.188 | 16.533 | 16.709 |
| StarRocks, query cache disabled | 50.611 | 57.984 | 60.543 | 25.723 | 25.277 |
| Doris, SQL result/query caches disabled | 37.525 | 58.488 | 33.095 | 27.765 | 29.803 |
| Elasticsearch, exact aligned queries | 25.694 | 151.591 | 74.024 | 46.090 | 43.569 |

Each section retains its own paired Milvus medians and ratios. These are warm,
single-client, complete-response comparisons with different recorded storage
policies and transport adapters. They do not establish a raw-JSON scan ranking
or a common-memory-limit comparison. The earlier DuckDB, PostgreSQL and MongoDB
local records remain available in their November 6 sections.

All three historical reports and exported snapshots retain their previous
hashes. Executed runners, helpers and verification scripts are preserved as
[source evidence](benchmark-results/20261110-jsonbench-sources/), with hashes in
the final audit. These are exact source snapshots from the original benchmark
directory layout. The audit confirmed that all twelve test listener ports were
vacant, all active-process markers were stopped, and the temporary
`vm.max_map_count` setting was restored from 2000000 to its original **65530**.

The benchmark continuation does not fix Milvus's requested-shutdown exit `-6`,
rerun the full Go/C++ functional regression, complete nullable/mixed-type reader
statistics comparisons, or deploy cloud instances. Those remain follow-up work.
The historical ClickHouse watchdog-only CPU counters remain invalid; the new
paired retest is the server CPU evidence.

### Recorded same-host Doris comparison (2026-11-10)

Apache Doris 4.1.5 completed two reverse-order paired rounds with full output
checking. Both `enable_sql_cache` and `enable_query_cache` were explicitly false
in the actual measured MySQL session; its effective variables and separate
post-timing EXPLAIN plans are retained. The same input, Milvus binary/core and
five-segment layout were checked before timing. Each query had two warmups and
ten timed complete HTTP responses per engine per round. All 200 timed queries
and 40 warmups passed the independent oracle.

| Query | Paired Milvus median (ms) | Doris median (ms) | Milvus / Doris |
| --- | ---: | ---: | ---: |
| Q1 | 201.372 | 37.525 | 5.37x |
| Q2 | 1129.624 | 58.488 | 19.31x |
| Q3 | 262.080 | 33.095 | 7.92x |
| Q4 | 232.831 | 27.765 | 8.39x |
| Q5 | 206.357 | 29.803 | 6.92x |

This is the upstream **generated-column workload** over typed VARIANT input,
with one FE, one BE and one replica. It is a different storage policy from the
Milvus JSON reader. The generated time column is DATETIME(6), and Q5 uses
`MICROSECONDS_DIFF` divided by 1000.0 to retain fractional milliseconds. Q2 uses
exact COUNT(DISTINCT), Q3 orders by hour/event, and timestamps are checked against
the UTC microsecond oracle. The function follows the official
[microsecond-difference contract](https://doris.apache.org/docs/4.x/sql-manual/sql-functions/scalar-functions/date-time-functions/microseconds-diff/).

Ten bounded Stream Load batches were checked for exact row counts and zero
filtered/unselected rows; the stored table count was 1,000,000. The persistent
local MySQL-to-HTTP adapter uses TCP_NODELAY, and its protocol/serialization work
is included in HTTP wall time. CPU sums the FE and BE trees, excludes the adapter,
includes background work, and is read outside the HTTP timer. Page caches remain
warm, and recorded default engine parallelism is retained. A common enforced
process memory cap is not established.

The [primary raw report](benchmark-results/20261110-jsonbench-doris-same-host.json)
and [independent verification](benchmark-results/20261110-jsonbench-doris-verification.json)
retain all samples/responses, ingestion records, effective DDL/session settings,
separate plans and 100 raw peer CPU counter pairs. The official archive passed
its upstream SHA512 check, and download provenance is included.

The first completed exploratory pair reported `enable_sql_cache=true` in this
build. Its peer medians were 7.498, 5.273, 5.241, 5.217 and 4.516 ms; those values
are not used in the primary table. The
[cached raw report](benchmark-results/20261110-jsonbench-doris-cached-same-host.json)
and [cached verification](benchmark-results/20261110-jsonbench-doris-cached-verification.json)
are preserved with that qualification. The cache-disabled pair is a separate
fresh measurement, rather than a correction applied to old timing samples.

### Recorded same-host StarRocks comparison (2026-11-10)

StarRocks 4.1.7 completed two reverse-order paired rounds with the unchanged
Milvus baseline binary/core, five sealed segments and fingerprinted input. Both
engines consumed complete HTTP responses; StarRocks uses a persistent local
MySQL-to-HTTP adapter with TCP_NODELAY. Each query had two warmups and ten timed
samples per engine per round. All 200 timed queries and 40 warmups passed the
independent oracle, and all persisted results were rechecked separately.

| Query | Paired Milvus median (ms) | StarRocks median (ms) | Milvus / StarRocks |
| --- | ---: | ---: | ---: |
| Q1 | 192.232 | 50.611 | 3.80x |
| Q2 | 1099.664 | 57.984 | 18.97x |
| Q3 | 257.340 | 60.543 | 4.25x |
| Q4 | 231.879 | 25.723 | 9.01x |
| Q5 | 205.215 | 25.277 | 8.12x |

The table follows the upstream JSON-column schema with a generated VARBINARY
sort key over kind, operation, collection and did, and one replica on one BE.
This storage policy differs from Milvus's JSON reader layout. Q2 retains exact
COUNT(DISTINCT); Q4 uses microsecond-scale `to_datetime` in a UTC session; Q5
subtracts numeric microseconds and returns fractional milliseconds instead of
the upstream integer-millisecond date difference. Effective DDL and measured
session variables are captured in the report. Query cache is explicitly disabled
in that actual session. This run does not independently prove which paths were
stored in StarRocks's automatic JSON flattening format.

Stream Load used bounded batches, strict mode and zero allowed filtering. Each
batch response was checked for exact loaded-row count and no filtered or
unselected rows. The stored table count was 1,000,000 before timing. FE and BE
were the only database processes for that engine; their CPU trees are summed
without the adapter. CPU counters include background work and have 10 ms
precision. Warm page caches, default engine parallelism and the lack of a common
enforced process memory cap remain qualifications.

The [raw paired report](benchmark-results/20261110-jsonbench-starrocks-same-host.json)
contains effective schema/session settings, separate post-timing EXPLAIN plans,
per-batch ingestion evidence and complete responses. The
[independent verification](benchmark-results/20261110-jsonbench-starrocks-verification.json)
rechecked all 200 samples, 20 persisted responses and 100 peer CPU counter pairs.
The official download URL and computed archive SHA256 are recorded; no
independent upstream checksum was available for this StarRocks archive.
Milvus still exited `-6` on requested termination; the StarRocks FE/BE shutdown
codes are retained separately in the evidence.

### Recorded same-host Elasticsearch comparison (2026-11-10)

Elasticsearch 9.5.6 completed two reverse-order paired rounds against the same
Milvus binary/core, five sealed segments and fingerprinted 1,000,000-row input.
Each query had two warmups and ten timed complete HTTP responses per engine per
round. All 200 timed queries and 40 warmups passed the independent oracle.

| Query | Paired Milvus median (ms) | Elasticsearch median (ms) | Milvus / Elasticsearch |
| --- | ---: | ---: | ---: |
| Q1 | 192.305 | 25.694 | 7.48x |
| Q2 | 1111.733 | 151.591 | 7.33x |
| Q3 | 256.809 | 74.024 | 3.47x |
| Q4 | 236.461 | 46.090 | 5.13x |
| Q5 | 205.602 | 43.569 | 4.72x |

This is an aligned local workload, with explicit changes from the published
reference: Q2 groups by event/did, then counts the resulting did groups and sums
their row counts. It does not use approximate `COUNT_DISTINCT`. Q3 sorts by
hour/event. Raw `time_us` is indexed as a long without the Filebeat division by
1000; Q4 converts the minimum microsecond value to `date_nanos` in ES|QL, and Q5
subtracts integer microseconds before dividing by 1000.0. The conversion follows
the official [TO_DATE_NANOS contract](https://www.elastic.co/docs/reference/query-languages/esql/functions-operators/type-conversion-functions/to_date_nanos).
All five queries return the oracle's complete columns and ordering. Warning
headers and partial ES|QL results are rejected.

The isolated single-node index has one primary shard, zero replicas, upstream
keyword/sort-key policy, retained full JSON `_source`, and only the queried paths
indexed. The workload uses a 31 GiB JVM heap and 16 configured processors; mmap
is disabled, and shard request caching is disabled. Ingestion checked every
bulk item, loaded exactly 1,000,000 documents and merged the index to one segment
before timing. Warm caches and different index/schema policies remain material
qualifications; these numbers are not raw-JSON scan rankings.

The [raw paired report](benchmark-results/20261110-jsonbench-elasticsearch-same-host.json)
retains all samples, full responses, effective mapping/settings and CPU counter
snapshots. The [independent verification](benchmark-results/20261110-jsonbench-elasticsearch-verification.json)
rechecked 200 timing samples, 20 persisted responses, 100 CPU counter pairs,
input/binary identity and reversed order. The official archive passed SHA512
verification; its download provenance is included in the verification record.
CPU includes background work and is read outside the HTTP wall timer.

Both Elasticsearch processes exited 143 following the requested SIGTERM; their
server logs confirm the node stopped and closed. Milvus retained its existing
shutdown exit `-6`. No functional-regression or shutdown-fix claim is made by
this benchmark run.

### ClickHouse CPU retest results (2026-11-10)

Approved execution outside the sandbox completed two reverse-order rounds on
the same 1,000,000-row input, unchanged Milvus binary/core and five sealed
segments. Each query had two warmups and ten timed complete HTTP responses per
engine per round. All 200 timed executions and 40 warmups passed the independent
result oracle. The ClickHouse schema, exact DISTINCT and fractional-millisecond
Q5 expression are the same as the November 6 comparison. No database or adapter
was left running by this retest.

| Query | Milvus median (ms) | ClickHouse median (ms) | Milvus / ClickHouse | Milvus CPU / wall | ClickHouse CPU / wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q1 | 185.319 | 5.057 | 36.65x | 4.41 | 2.84 |
| Q2 | 1090.668 | 31.129 | 35.04x | 2.34 | 7.92 |
| Q3 | 250.673 | 13.188 | 19.01x | 4.39 | 4.57 |
| Q4 | 231.213 | 16.533 | 13.98x | 1.89 | 1.04 |
| Q5 | 203.047 | 16.709 | 12.15x | 1.98 | 1.08 |

CPU / wall is the sum of per-sample process CPU seconds divided by the sum of
HTTP wall seconds across both rounds. It estimates average busy cores, includes
background work, and is not configured parallelism or a percentage of host CPU.
ClickHouse counters cover its watchdog and live server child; Milvus counters
cover its standalone process. CPU reads occur immediately outside the HTTP wall
timer, so the windows differ slightly. OS counters have 10 ms precision; short
individual queries can report zero ticks. Warm caches, workload-specific typed
JSON paths/sorting and the absence of a common enforced process memory cap
remain qualifications of this comparison.

The original bug is reproduced in the raw evidence: the watchdog alone recorded
zero Q2 CPU in both rounds, whereas the complete tree recorded **2.41 seconds**
and **2.38 seconds** across each round's ten Q2 samples. Every timed ClickHouse
sample retained before/after counters and the same two PIDs. All 100 counter
pairs were independently checked for stable PIDs, nondecreasing counters and
exact agreement with the stored CPU deltas. The six collector regression tests
also passed, including busy child/reaped child accounting, watchdog-only
rejection, changing process trees and counter resets. The collector's field
indices were checked against the Linux [/proc/PID/stat contract](https://man7.org/linux/man-pages/man5/proc_pid_stat.5.html).

Evidence:

- [Complete paired samples and responses](benchmark-results/20261110-jsonbench-clickhouse-cpu-same-host.json).
- [Independent verification and CPU summaries](benchmark-results/20261110-jsonbench-clickhouse-cpu-verification.json):
  200 timing samples, 20 persisted responses, all raw ClickHouse counter pairs,
  input/binary/runner fingerprints, reversed order and unchanged baseline layout.
- Executed runners are archived under `.benchmarks/jsonbench-20261106/` with
  the suffix `-cpu-20261110-executed.py`; the verification record retains hashes.

All historical report and exported snapshot hashes remain unchanged. Their
watchdog-only CPU values remain invalid; the table here uses fresh paired
measurements. ClickHouse exited normally in both rounds. Milvus still exited
`-6` after requested shutdown; this run does not fix or explain that issue.
Full Go/C++ regression and the other pending database comparisons are not
claimed complete by this benchmark retest.

### ClickHouse CPU retest preparation (2026-11-09)

The continuation reached Milvus port preflight, where creating a socket failed
with `PermissionError: [Errno 1] Operation not permitted`. The execution policy
also rejected the previously approved command's request to run outside the
sandbox. Neither database started in that attempt, and the November 6 numeric
records were then the latest measurements. StarRocks/Doris/Elasticsearch also
remained pending at that checkpoint. The completed November 10 measurements
above supersede this blocked attempt.

The [continuation verification record](benchmark-results/20261109-jsonbench-continuation-verification.json)
confirms that all three historical reports and exported snapshots retain their
previously verified hashes. All 700 stored timing samples' per-query statistics
and 70 persisted query responses were checked again against the independent
oracle. This adds no new execution samples.

The ClickHouse runner now records root and process-tree CPU counters with live
PIDs before and after each timed sample. It rejects changing process trees and
decreasing counters. Each round must capture positive Q2 CPU, including child
server work when a watchdog is present. These checks run outside the HTTP wall
timer. CPU counters retain OS clock-tick precision and include background work;
individual short queries may still record zero ticks.

Six offline regression tests passed, including a real idle-parent/busy-child
process tree, CPU accounting after child reaping, watchdog-only counters, one
invalid round after a valid round, changing/reset counters, and preservation of
existing reports. This verified the collector and rejection paths; live
ClickHouse server CPU was still unverified by that attempt and was later checked
in the November 10 retest. Before starting databases the runner
checks the prepared dataset hash, Milvus binary/core hashes and CPU affinity
against the historical baseline. Those preflight checks passed on this attempt.

Use a new output prefix to keep historical reports, timing samples and logs:

```bash
cd /root/milvus
PYTHONPATH=/root/milvus/.benchmarks/jsonbench-20261106/python \
  python3 tests/e2e/jsonbench_clickhouse.py \
  --run-dir /root/milvus/.benchmarks/jsonbench-20261106 \
  --clickhouse-binary /root/milvus/.benchmarks/jsonbench-20261106/clickhouse/clickhouse \
  --output-prefix clickhouse-cpu-20261109
```

This command requires an execution environment that permits local listeners.
The intended run remains two reversed-order rounds with two warmups and ten
timed executions per query/engine. New results belong in a separate dated
snapshot after full-response oracle and CPU validation. Existing comparison
reports are refused rather than overwritten; an interrupted attempt that has
saved a round needs a fresh prefix.

### Recorded same-host PostgreSQL/MongoDB comparison (2026-11-06)

The interrupted native-client run completed at 10:28 UTC on November 6. Its
[numeric record](benchmark-results/20261106-jsonbench-native-same-host.json)
contains every measured sample, complete final responses, database and runner
hashes, schemas/indexes, settings and both rounds. The first attempt had failed
before database startup because socket creation was denied by the execution
sandbox; resuming with the required execution permission produced the results
below. No failed attempt was counted as a timing sample.

The run reused the fingerprinted million-row source, Milvus binary/core and
fixed five-segment layout from the DuckDB comparison. All three database
processes inherited CPU affinity `0-15` on the same host. Round one ran Milvus,
PostgreSQL, MongoDB; round two reversed that order. Each query received two
warmups and ten timed executions per engine/round. All **300 timed queries and
60 warmups** passed the independent oracle, with exact counts/DISTINCT and the
same 1 microsecond timestamp / 0.0011 millisecond span tolerances. The medians
below use 20 timed samples per engine/query and this run's own Milvus samples.

| Query | Milvus median (ms) | PostgreSQL 18.6 median (ms) | MongoDB 9.0.4 median (ms) | Milvus / PostgreSQL | Milvus / MongoDB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q1 | 191.854 | 173.832 | 1733.158 | 1.104x | 0.111x |
| Q2 | 1111.346 | 3494.828 | 2263.655 | 0.318x | 0.491x |
| Q3 | 254.782 | 705.632 | 2892.235 | 0.361x | 0.088x |
| Q4 | 229.380 | 237.202 | 130.098 | 0.967x | 1.763x |
| Q5 | 207.785 | 277.976 | 142.455 | 0.747x | 1.459x |

Ratios above one mean higher Milvus latency. These are warm single-client
measurements under the recorded schema policies, with no shared enforced
process memory cap or cache reset. PostgreSQL and MongoDB use persistent native
client connections behind Python HTTP adapters; timing includes consuming the
complete HTTP response and adapter serialization, while client response JSON
decoding and oracle checks occur after the timer stops. Database CPU counters
exclude the adapter process and include database background work; they are not
launched-worker counts.

Configuration and precision qualifications:

- PostgreSQL used the pinned upstream JSONB/LZ4 schema and compound expression
  index. The input was copied without the upstream loader's escape removal or
  skipped records, in one transaction, followed by `VACUUM ANALYZE` and a
  verified count of 1,000,000. Sessions used UTC and the acceptance SQL above,
  including fractional timestamp arithmetic. Settings remained
  `shared_buffers=128MB`, `work_mem=4MB`, `max_parallel_workers=8` and
  `max_parallel_workers_per_gather=2`. Although `jit=on`, the local source build
  has no JIT provider: the diagnostic `pg_jit_available()` returned false.
- MongoDB used BSON/zstd, the upstream compound index on kind, operation,
  collection, did and time_us, and the covered-scan configuration. Q2 used exact
  `$addToSet`; Q4/Q5 aggregated integer microseconds before timestamp/span
  conversion, preserving the acceptance precision. The pipelines, index and
  version are stored in the record. A deterministic source-row `_id` supports
  resuming a verified contiguous ingestion prefix.

The separate [execution-plan record](benchmark-results/20261106-jsonbench-native-diagnostics.json)
contains ClickHouse reader statistics/pipelines, PostgreSQL
`EXPLAIN (ANALYZE, BUFFERS)` and MongoDB `executionStats`, collected after the
clean measurements. In that diagnostic execution, PostgreSQL launched two
parallel workers for Q1 and Q3-Q5; Q2 used an index scan without a parallel
Gather. Q3-Q5 sorts spilled to disk at the recorded memory settings. MongoDB
Q1/Q2/Q4/Q5 examined zero source documents through its compound index; Q3
fetched 588,328 documents. These plans describe observed execution choices;
they do not isolate the causal contribution of any one setting to latency.

A [persisted-result verification record](benchmark-results/20261106-jsonbench-result-verification.json)
checks all three local comparison records: 700 timed samples, their stored
responses, medians/ratios, dataset and binary fingerprints, fixed Milvus layout
and reversed round order. Native runner fingerprints and the archived DuckDB
runner are also checked. Warmups were verified during execution but are not
stored as timing samples. The historical ClickHouse CPU counters remain
invalid as explained above; this verification checks latency and result
integrity, not CPU accounting.

All benchmark database and adapter processes were stopped. PostgreSQL and
MongoDB exited normally; Milvus still exited -6 after requested termination.
At the November 6 checkpoint the shutdown problem, nullable/mixed-type stats
comparisons, cloud deployment and local StarRocks/Doris/Elasticsearch comparisons
were unfinished. The three local peer comparisons were completed on November 10
as recorded above; the other follow-ups remain open. No production Go/C++ code
changed in this benchmark continuation, and the prior functional suite was not
rerun.

### Latest all-system reference comparison (2026-11-06)

This historical table uses the Milvus samples from the DuckDB comparison above,
replacing the November 5 Milvus baseline **for this table only**. The historical
tables retain their original dates and measurements. Results, schema/query/
loader/timing source and content hashes are saved in
[20261106-jsonbench-external-comparison.json](benchmark-results/20261106-jsonbench-external-comparison.json).
All seven external result vectors and 105 upstream files were fetched again
from pinned revision `ff15af5d8aae5ce5ce79b8e479379582f9050fdc`.

The first two rows are this host's actual measurements: 16 logical CPUs, about
61 GiB RAM, two warmups and ten timed executions per round in two reversed-order
rounds, median of 20. The remaining rows are **published reference data** from
`m6i.8xlarge` (upstream documented default: 32 vCPU, 128 GB RAM, 2,000 GB gp3),
median of the two warm executions. Each published record reports 1,000,000
loaded documents, but contains no input fingerprint or verified result sets.
No CPU-count or hardware normalization is applied.

| System | Evidence | Q1 (ms) | Q2 (ms) | Q3 (ms) | Q4 (ms) | Q5 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Milvus, five sealed segments, stats enabled | Local, oracle checked | 189.385 | 1115.076 | 252.661 | 233.563 | 208.195 |
| DuckDB 1.5.6 | Same host, oracle checked | 119.501 | 516.568 | 632.925 | 393.015 | 374.760 |
| [ClickHouse 26.10.3.7][result-clickhouse] | Published, 2026-11-03 | 4.000 | 20.500 | 7.500 | 15.500 | 15.500 |
| [DuckDB 2.0.0][result-duckdb] | Published, 2026-11-02 | 103.000 | 363.000 | 376.000 | 237.000 | 247.500 |
| [PostgreSQL 18.6][result-postgresql] | Published, 2026-11-02 | 109.500 | 2701.500 | 429.000 | 191.000 | 208.000 |
| [MongoDB 9.0.4][result-mongodb] | Published, 2026-11-02 | 1056.500 | 1518.500 | 1363.500 | 213.500 | 232.000 |
| [StarRocks 4.1.7][result-starrocks] | Published, 2026-11-02 | 59.000 | 79.000 | 76.500 | 51.000 | 55.500 |
| [Apache Doris 4.1.5][result-doris] | Published, generated-column workload | 55.000 | 60.000 | 60.000 | 40.000 | 45.000 |
| [Elasticsearch 9.5.6][result-elasticsearch] | Published, approximate/precision/order differences | 91.000 | 78.000 | 84.000 | 65.000 | 82.000 |

For numerical comparison, the following cells are current local Milvus latency
divided by each **published** latency. A value above one means a larger recorded
Milvus number. These are raw ratios across different environments and query
policies; they do not establish same-host engine slowdown factors.

| Published reference | Q1 | Q2 | Q3 | Q4 | Q5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ClickHouse | 47.35x | 54.39x | 33.69x | 15.07x | 13.43x |
| DuckDB 2.0.0 | 1.84x | 3.07x | 0.67x | 0.99x | 0.84x |
| PostgreSQL | 1.73x | 0.41x | 0.59x | 1.22x | 1.00x |
| MongoDB | 0.18x | 0.73x | 0.19x | 1.09x | 0.90x |
| StarRocks | 3.21x | 14.11x | 3.30x | 4.58x | 3.75x |
| Doris, generated-column workload | 3.44x | 18.58x | 4.21x | 5.84x | 4.63x |
| Elasticsearch, non-equivalent query vector | 2.08x | 14.30x | 3.01x | 3.59x | 2.54x |

Interpretation and the source audit:

- **ClickHouse:** the published values remain far below the local Milvus
  values. Its [schema][schema-clickhouse] declares typed JSON paths,
  LowCardinality categories and a workload-specific sort order; the
  [driver][query-clickhouse] uses `FORMAT Null`. Q2 uses exact `uniqExact`.
  A comparable retest must return the full results, align missing-path values
  and retain fractional-millisecond Q5 output. Typed paths/sort order should be
  recorded as an explicit schema policy rather than treated as an unindexed
  raw-JSON scan.
- **PostgreSQL:** current Milvus's Q2/Q3 values are numerically lower than the
  published values; Q1/Q4 are higher, and Q5 is close. The
  [schema][schema-postgresql] uses LZ4-compressed JSONB and a compound expression
  index, while [timing][query-postgresql] comes from psql. Q3 divides to integer
  seconds, and the loader can remove escaped U+0000 and skip invalid input.
  Full output validation and an explicit index policy remain necessary.
- **MongoDB:** current Milvus's Q1-Q3 values are numerically lower; Q4 is
  higher and Q5 lower. Its [compound index][schema-mongodb] and covered-scan
  configuration differ from the Milvus layout. The aggregation pipelines use
  exact `$addToSet` for Q2, but convert microseconds to millisecond Date values
  before Q4/Q5 aggregation. Those two precision-reduced queries are not exact
  comparisons against the microsecond oracle.
- **StarRocks:** all five published values are lower. The
  [schema][schema-starrocks] keeps JSON/FlatJSON and adds a generated sort key;
  it does not project every queried path into a generated scalar column.
  The runner measures mysql client process wall time. Q4 explicitly retains
  six timestamp digits; Q5 returns integer milliseconds. A retest should
  preserve the same semantic output and report its FlatJSON/sort-key policy.
- **Doris:** all five published values are lower, but queries read generated
  scalar columns from the [VARIANT schema][schema-doris]. In addition, its
  generated `time` is declared `DATETIME` with no scale. Doris documents the
  [default as DATETIME(0)](https://doris.apache.org/docs/dev/sql-manual/basic-element/sql-data-types/date-time/DATETIME/),
  so this DDL does not retain the oracle's microseconds. Q4/Q5 need a precision
  aligned retest, with DATETIME(6) or numeric microseconds. The loader also allows
  filtering up to 10% of a file; the reported million-row count is not a
  fingerprint check. Generated-column results remain a separate workload.
- **Elasticsearch:** all five published values are lower, but its
  [Q2][queries-elasticsearch] uses approximate `COUNT_DISTINCT(did,40000)` and
  Q3 sorts by count/event rather than hour/event. Its keyword/index-sort policy
  uses one primary shard and no replicas. `time_us` is mapped as `date` and the
  Filebeat processor divides microseconds by 1000 before ingestion, so Q4/Q5
  also lose the oracle's microsecond precision. Its driver reports server
  `took`. Only an aligned-query, complete-response retest can provide an exact
  same-host comparison.

The table above retains the published-reference audit checkpoint. Local
ClickHouse, PostgreSQL and MongoDB results are now recorded in the separate
local comparisons above; their published rows remain historical reference
data. StarRocks, Doris and Elasticsearch now also have completed local,
aligned-query comparisons in the November 10 sections above. Their source
snapshots and explicit precision/algorithm qualifications keep the local
measurements distinct from these published reference vectors.

[upstream-methodology]: https://github.com/ClickHouse/ClickBench/blob/ff15af5d8aae5ce5ce79b8e479379582f9050fdc/json/README.md
[result-clickhouse]: https://github.com/ClickHouse/ClickBench/blob/ff15af5d8aae5ce5ce79b8e479379582f9050fdc/json/clickhouse/results/20261103/m6i.8xlarge_bluesky_1m.json
[result-duckdb]: https://github.com/ClickHouse/ClickBench/blob/ff15af5d8aae5ce5ce79b8e479379582f9050fdc/json/duckdb/results/20261102/m6i.8xlarge_bluesky_1m.json
[result-postgresql]: https://github.com/ClickHouse/ClickBench/blob/ff15af5d8aae5ce5ce79b8e479379582f9050fdc/json/postgresql/results/20261102/m6i.8xlarge_bluesky_1m.json
[result-mongodb]: https://github.com/ClickHouse/ClickBench/blob/ff15af5d8aae5ce5ce79b8e479379582f9050fdc/json/mongodb/results/20261102/m6i.8xlarge_bluesky_1m.json
[result-starrocks]: https://github.com/ClickHouse/ClickBench/blob/ff15af5d8aae5ce5ce79b8e479379582f9050fdc/json/starrocks/results/20261102/m6i.8xlarge_bluesky_1m.json
[result-doris]: https://github.com/ClickHouse/ClickBench/blob/ff15af5d8aae5ce5ce79b8e479379582f9050fdc/json/doris/results/20261102/m6i.8xlarge_bluesky_1m.json
[result-elasticsearch]: https://github.com/ClickHouse/ClickBench/blob/ff15af5d8aae5ce5ce79b8e479379582f9050fdc/json/elasticsearch/results/20261102/m6i.8xlarge_bluesky_1m.json
[query-clickhouse]: https://github.com/ClickHouse/ClickBench/blob/ff15af5d8aae5ce5ce79b8e479379582f9050fdc/json/clickhouse/query
[query-postgresql]: https://github.com/ClickHouse/ClickBench/blob/ff15af5d8aae5ce5ce79b8e479379582f9050fdc/json/postgresql/query
[query-mongodb]: https://github.com/ClickHouse/ClickBench/blob/ff15af5d8aae5ce5ce79b8e479379582f9050fdc/json/mongodb/query
[query-elasticsearch]: https://github.com/ClickHouse/ClickBench/blob/ff15af5d8aae5ce5ce79b8e479379582f9050fdc/json/elasticsearch/query
[schema-clickhouse]: https://github.com/ClickHouse/ClickBench/blob/ff15af5d8aae5ce5ce79b8e479379582f9050fdc/json/clickhouse/create.sql
[schema-postgresql]: https://github.com/ClickHouse/ClickBench/blob/ff15af5d8aae5ce5ce79b8e479379582f9050fdc/json/postgresql/ddl.sql
[schema-mongodb]: https://github.com/ClickHouse/ClickBench/blob/ff15af5d8aae5ce5ce79b8e479379582f9050fdc/json/mongodb/ddl.js
[schema-starrocks]: https://github.com/ClickHouse/ClickBench/blob/ff15af5d8aae5ce5ce79b8e479379582f9050fdc/json/starrocks/create.sql
[schema-doris]: https://github.com/ClickHouse/ClickBench/blob/ff15af5d8aae5ce5ce79b8e479379582f9050fdc/json/doris/create.sql
[queries-elasticsearch]: https://github.com/ClickHouse/ClickBench/blob/ff15af5d8aae5ce5ce79b8e479379582f9050fdc/json/elasticsearch/queries.txt
[elastic-distinct]: https://www.elastic.co/docs/reference/query-languages/esql/functions-operators/aggregation-functions/count_distinct
[benchmark-driver]: https://github.com/ClickHouse/ClickBench/blob/ff15af5d8aae5ce5ce79b8e479379582f9050fdc/json/lib/benchmark-common.sh
[load-doris]: https://github.com/ClickHouse/ClickBench/blob/ff15af5d8aae5ce5ce79b8e479379582f9050fdc/json/doris/load
[index-elasticsearch]: https://github.com/ClickHouse/ClickBench/blob/ff15af5d8aae5ce5ce79b8e479379582f9050fdc/json/elasticsearch/index_template.json
[install-postgresql]: https://github.com/ClickHouse/ClickBench/blob/ff15af5d8aae5ce5ce79b8e479379582f9050fdc/json/postgresql/install
[postgresql-parallel-settings]: https://www.postgresql.org/docs/18/runtime-config-resource.html
[duckdb-thread-settings]: https://duckdb.org/docs/2.0/configuration/overview
[doris-task-parallelism]: https://doris.apache.org/docs/4.x/query-acceleration/tuning/tuning-execution/parallelism-tuning/
