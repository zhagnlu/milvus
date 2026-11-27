# Scalar execution performance comparison

Routine performance comparisons must use the deployment batch size (the current
Python benchmark default is 8,192). Do not set Raw batch size to the Segment row count
except for an explicit bottleneck investigation. Such measurements are diagnostic
controls and must be reported separately from deployment-representative results.
Planner-selected WholeSegment execution for index/stats is a separate, real path.
The service default is defined in `pkg/util/paramtable/component_param.go` and
`internal/core/src/common/Consts.h`; override `--batch` if the deployment differs.
For direct native benchmark invocation, pass `MILVUS_SCALAR_PERF_BATCH=8192`
explicitly when overriding the deployment; the native harness now also defaults to 8,192.

This module measures the real `PhyFilterBitsNode` with
`expression.enable_expr_execution_v2` explicitly set to false and true. It
uses the same sealed segment and expression in both modes:

```text
a < 10 && b > 10 && a != 8
```

Both fields are nullable INT64. Data is deterministic: `a = row * 17 % 100`,
`b = ((row / 100) * 37 + row * 13) % 100`; every 97th a and 89th b is NULL.
The independent oracle evaluates the predicate and folds FALSE/NULL into
exclusion bits. Every measured invocation checks the entire output bitmap
outside the timer, not just the hit count.

Three cases use identical values:

| Case | Indexes | Expected new path |
| --- | --- | --- |
| Raw | None | Batch, three Raw predicates |
| Mixed | STL_SORT on b | Batch, one ExactIndex and two Raw predicates |
| AllIndex | STL_SORT on a and b | WholeSegment, three ExactIndex predicates |

The test asserts the actual new route, so legacy fallback cannot masquerade as
a successful new-mode benchmark. Reordering is enabled for both modes; the
previous global setting is restored on test exit. Cross-query predicate-result
caching is explicitly disabled. Physical columns/indexes are warm after the
warmup rounds; this is not a cold-storage benchmark.

## Build and run

Use an existing C++ dependency setup, an optimized build and no sanitizer:

```bash
cmake --build cmake_build --target scalar_execution_perf -j4
source scripts/setenv.sh
python3 scripts/benchmark/scalar_execution.py \
  --output /tmp/scalar-perf-run1 \
  --rows 262144 --batch 8192 --rounds 21 --warmup 3
```

The output directory must be new. On Linux, optionally add `--cpu N` to pin the
process to an allowed CPU. The runner sets `OMP_NUM_THREADS=1` and
`OPENBLAS_NUM_THREADS=1`, captures build settings, CPU affinity, git state and
linked libraries, and runs three small correctness cases plus three performance
cases. Ensure the loaded `libmilvus_core.so` belongs to the build under test;
`scripts/setenv.sh` may prefer installed libraries, so install the current build
first if needed. Do not rebuild/install libraries while the benchmark runs.

The `scalar_execution_perf` target is excluded from the default build. The same
source is also discovered by `all_tests`: normal UT runs only the small
`CorrectnessAndRouting` tests (3,073 rows), while timings require explicit opt-in:

```bash
MILVUS_SCALAR_PERF_ROWS=262144 \
MILVUS_SCALAR_PERF_BATCH=8192 \
MILVUS_SCALAR_PERF_ROUNDS=21 \
MILVUS_SCALAR_PERF_WARMUP=3 \
  cmake_build/unittest/scalar_execution_perf \
  --gtest_also_run_disabled_tests \
  '--gtest_filter=Paths/ScalarExecutionPerformanceTest.*' \
  --gtest_output=xml:/tmp/scalar-perf.xml
```

Settings: rows 1–10,000,000; rounds 3–1,000; warmup 1–100; batch 1–rows.
Use a batch smaller than rows to exercise multiple batches. For controlled diagnosis, `--expression unary` selects `a < 10` and
`--nullable 0` makes both fields non-nullable. `--case raw|mixed|index` restricts
the runner to one layout (`all` is the default). The corresponding UT environment
variables are `MILVUS_SCALAR_PERF_EXPRESSION=compound|unary` and
`MILVUS_SCALAR_PERF_NULLABLE=0|1`. Smoke correctness tests retain the original
compound/nullable setup; measured cases validate the selected variant against
its independent oracle. In the unary variant, the Mixed layout's b index is
unused and the expected route is Raw. To add further expressions, extend the
predicate builder, oracle and expected route checks together.

## Timing contract and output

Each invocation creates a fresh QueryConfig, QueryContext, snapshot, Task,
DriverContext and operator. It executes the expression once and closes/destroys
the operator. No completed execution or query-local index result is reused
between invocations. The logical expression tree and loaded segment are reused
by both modes. There is no external asynchronous prefetch; lazy loading and any
internal prefetch wait are included in execution.

- `prepare`: context/snapshot/task construction and operator compilation/Prepare.
- `eval`: `GetOutput()`, including first-use index queries, per-batch evaluation
  and final exclusion-bitmap construction.
- `total`: preparation, evaluation and query/operator cleanup. The final result
  stays alive for validation after timing, so its release is excluded.

Data generation, column loading, index construction, warmup and result checking
are outside the reported samples. Modes alternate first/second position each
round. `report.json` contains all samples and median/P95/min/max for each phase;
`results.xml` contains the UT properties and `run.log` preserves failures.
`summary.md` compares total latency. `new_over_old > 1` means the new mode is
slower; `old_over_new_speedup > 1` means the new mode is faster. Individual phase
medians need not sum to the total median.

No speed threshold fails CI: timing depends on CPU, scheduling and data shape.
Repeat on the same machine/build, inspect P95 and samples, and test other sizes
and selectivities before generalizing. This measures single-thread FilterBits,
not RPC, ANN, MVCC, deletes, concurrent service throughput or cold I/O. Equal
filter bitmaps do not prove full BOOL-projection three-valued equivalence; that
is covered by the functional expression tests.

## Measured baseline (2026-10-29)

Intel Core i7-8700 @ 3.20 GHz, Linux, GCC Release (`-O3 -DNDEBUG`),
ASAN off, process pinned to CPU 2. There are 262,144 rows, batch size 1,024,
5 warmup invocations per mode and 31 measured invocations per mode. Both
fields are nullable; 20,547 rows match. Numbers below are total latency
medians, including query preparation and cleanup.

| Layout | Old median (ms) | New median (ms) | New / old |
| --- | ---: | ---: | ---: |
| raw | 2.393 | 55.895 | 23.36 |
| b_index | 1.725 | 50.768 | 29.43 |
| all_index | 0.117 | 24.787 | 211.67 |

All six correctness/routing and timing cases passed. This baseline exposes a
large regression in the new engine; it is not evidence of a speedup. Preparation
takes only tens of microseconds in both modes: the gap is in evaluation. Two
earlier runs at the same size also showed the regression. At 1,048,576 rows
(21 rounds), the observed ratios were 22.32 / 27.27 / 197.11 for Raw / Mixed /
AllIndex. Those are measurements of this data shape and warm-cache setup, not
general service slowdown factors.

Code inspection identifies row-by-row predicate mapping (`MapPredicate`),
logical result processing (`LogicalExpr::EvalImpl`) and final FilterBits
conversion as follow-up profiling candidates. Their individual contribution
is not measured: Linux perf sampling was unavailable on this host. No
production optimization is included in this benchmark change.

## Follow-up CPU investigation (2026-10-29)

The same binary and core-library hashes were verified against the baseline.
Although Linux perf was unavailable, the installed gperftools `libprofiler`
provided user-space sampling at 997 Hz. Each case used 101 measured rounds and
5 warmup rounds per mode. All three sampled benchmark cases passed. Sampling
runs are for attribution, not replacement latency measurements.

Stacks were filtered to new `FilterBitsExecution::GetOutput`; setup, old-mode
execution and post-timing validation are excluded. There were 2,699 new-mode
samples for AllIndex, 6,594 for Raw and 5,893 for Mixed. AllIndex's disjoint
stack groups were approximately:

| Work | CPU samples |
| --- | ---: |
| Logical combination and other batch work | 41.5% |
| Final output conversion and batch setup | 27.0% |
| Extracting input/predicate validity | 15.9% |
| Mapping index results to views | 10.1% |
| IndexResult normalization/validation | 5.0% |
| Other predicate work | 0.5% |

`IsValid`, `IsSelected`, `BoolValueAt` and `defined_rows` alone accounted for
54.3% of AllIndex's new-execution **self** samples. This overlaps the phase
breakdown and must not be added to it. WholeSegment removes batch splitting,
but currently still maps, combines and converts bitmaps row by row. In
particular, callers check `IsValid` then `ValueAt<bool>` checks validity again.
InputSlot also extracts a new validity mask from an already-produced BOOL view.

Raw additionally spends about 8.9% on LiteralExpr broadcasting (a full-size
zero RowMapping plus Dictionary validation), 8.8% on Reader result validation,
and 15.5% on extracting input/predicate validity. Its numeric comparison uses
row-by-row view access rather than the legacy `inplace_compare_val` batch
kernel. Mixed shows the same mechanisms. These are observed hotspots; the
benefit of replacing them has not yet been measured.

Prioritize encoding-aware bitmap logic/output, direct validity-mask reuse and
word-wise coverage checks, then Flat + Constant numeric kernels and allocation
reuse. Keep full NULL/coverage/short-circuit guarantees; deleting validation is
not the proposed fix. Filter-only null rejection is a separate opportunity and
its individual contribution was not measured.

Example user-space profiling command (requires installed libprofiler):

```bash
# Put taskset BEFORE env: do not start a profiling timer in a process that execs.
source scripts/setenv.sh
taskset -c 2 env \
  LD_PRELOAD="${LD_PRELOAD:+${LD_PRELOAD}:}/lib/x86_64-linux-gnu/libprofiler.so" \
  CPUPROFILE=/tmp/scalar-all-index.prof CPUPROFILE_FREQUENCY=997 \
  MILVUS_SCALAR_PERF_ROUNDS=101 MILVUS_SCALAR_PERF_WARMUP=5 \
  cmake_build/unittest/scalar_execution_perf \
  --gtest_also_run_disabled_tests \
  '--gtest_filter=Paths/ScalarExecutionPerformanceTest.DISABLED_CompareModes/AllIndex'
```

Use the available profiler's symbolization tools against the exact loaded
libraries. The sample percentages are approximate CPU attribution, not exact
wall-clock phase timings. No production execution code was changed for this
investigation.

## Raw-only control experiments (2026-10-29)

The benchmark now supports a unary expression and non-nullable fields to
separate basic per-row overhead from conjunction and NULL handling:

```bash
python3 scripts/benchmark/scalar_execution.py \
  --output /tmp/scalar-raw-unary \
  --case raw --expression unary --nullable 0 \
  --rows 262144 --batch 1024 --rounds 31 --warmup 5 --cpu 2
```

On the same host/core-library build, total median latency was:

| Expression | Nullable | Old ms | New ms | New / old |
| --- | --- | ---: | ---: | ---: |
| a < 10 AND b > 10 AND a != 8 | yes | 2.404 | 55.769 | 23.20 |
| a < 10 | yes | 0.667 | 36.244 | 54.35 |
| a < 10 | no | 0.555 | 30.567 | 55.10 |
| a < 10 AND b > 10 AND a != 8 | no | 1.774 | 51.109 | 28.81 |

All results and routes passed validation. Both expression shapes additionally
passed all three index layouts with non-nullable fields at 3,073 rows. The
regression persists without AND or NULLs, so the basic Raw path is the first
optimization target.

A further 997 Hz CPU profile of non-nullable `a < 10` collected 3,499 samples
inside new GetOutput: comparison/other input work 34.5%, final output/batch
setup 26.9%, Reader result validation 19.2%, validity extraction 15.3%, and
constant broadcast 3.5%. The original compound Raw profile was refined to
separate field validity extraction (6.4%) from predicate-result validity
extraction (9.0%); Reader and read-view construction accounted for 4.8%. These
are CPU shares in warm-cache execution, not physical I/O latency.

ReaderInput currently passes only `values` from a values+validity ReadResult;
InputSlot then extracts validity again from that view. Raw predicates have a
second InputSlot for their BOOL result and repeat this extraction. Together
with the lost batch numeric kernel, per-row view checks and final bit-by-bit
conversion, these account for most of the observed cost. Production execution
code remains unchanged by the control experiments.

## Bulk validity optimization (2026-10-29)

The first production optimization adds `ReadOnlyVectorView::CopyValidity()`
and uses it in InputSlot extraction and Reader result validation. Flat and
bitmap vectors, constants, and their contiguous slices use bitmap operations;
dictionary mappings and fragmented views retain the per-row fallback. The
returned mask is detached and clipped to defined rows. Payload ownership and
NULL/undefined semantics are unchanged. Coverage and overlapping-validity
checks now use bitmap set operations instead of repeated row access.

Before/after measurements below use Release, CPU 2, 262,144 rows, batch size
1,024, warm storage, result cache disabled, 5 warmups and 31 measured rounds.
Values are total medians, including preparation. Each run validates the route
and output against an independent oracle.

| Expression / layout | Nullable | New before ms | New after ms | Latency reduction | Old after ms | After new / old |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Compound / Raw | yes | 57.454 | 46.619 | 18.9% | 2.796 | 16.67 |
| a < 10 / Raw | yes | 35.525 | 24.178 | 31.9% | 0.673 | 35.90 |
| a < 10 / Raw | no | 30.984 | 22.183 | 28.4% | 0.585 | 37.95 |
| Compound / b index | yes | 52.153 | 40.514 | 22.3% | 1.969 | 20.58 |
| Compound / all index | yes | 24.678 | 22.472 | 8.9% | 0.120 | 187.94 |

Compound means `a < 10 AND b > 10 AND a != 8`. These are local observations,
not cross-machine guarantees. The old compound Raw control varied from
2.431 ms before to 2.796 ms after; run-to-run noise remains. The new path is
still substantially slower than the old path. SIMD numeric comparison,
logical combination and final filter bitmap conversion are not optimized by
this change.

A separate post-change 997 Hz profile collected 5,285 samples inside new Raw
GetOutput. Input/predicate validity extraction accounted for 1.34% and Reader
validation for 1.32%, versus approximately 15.4% and 8.8% in the earlier
compound Raw profile. Remaining groups were comparison/other input work
36.8%, logical combination/other batch work 27.1%, final output/batch setup
22.4%, and literal broadcast 9.8%. These are sampled CPU shares, not elapsed
phase timings; wall-clock measurements above ran without the profiler.

Validation: 331 focused C++ tests passed, including added multiword, unaligned
slice, repeated dictionary mapping, fragmented/merged view, immutable mask,
NULL, partial coverage and bidirectional validity mismatch cases. Existing
loader failure/retry and overlapping-validity rejection tests also passed.
The standalone service with the new mode enabled passed 285 E2E cases; case
names and statuses match the previously captured old-mode baseline. Runtime
configuration and loaded core-library paths were checked; the service was
stopped after validation.

Local reports, raw timing samples and profile evidence are in
`/mnt/dev_debug/scalar-validity-opt-20261029/`.

## Primitive batch kernels (2026-10-30)

`DecodePrimitive()` retains the source owner and exposes aligned numeric
fragments, constants and their validity masks. Comparison kernels intersect
selection and validity, enumerate contiguous valid runs, and write bitmap
ranges directly. They never load NULL/undefined payloads. Sliced/segmented
inputs retain batch coordinates; arbitrary mappings, routed merges and
unaligned/indirect storage use the original access path.

All six comparisons support column/constant, constant/column and column/column
operands. Same-type integers, TIMESTAMPTZ, FLOAT and DOUBLE use the existing
bulk bitset kernels; four range endpoint combinations use the same machinery.
Mixed-type comparisons preserve C++ promotions through decoded scalar loops.
Numeric IN, Roaring/Bloom membership and arithmetic comparisons share decoded
input access, while retaining their existing probe/arithmetic semantics. Owned
literals broadcast as Constant rather than allocating an N-entry dictionary.
JSON/ARRAY parsing, logical combination and final FilterBits conversion are
unchanged by this optimization.

The runner additionally accepts `--expression eq|ne|le|gt|ge|range|in|arith|compare`.
`unary` remains `a < 10`; `compound` remains the default. `arith` and `compare`
require `--case raw`. For example:

```bash
source scripts/setenv.sh
python3 scripts/benchmark/scalar_execution.py \
  --output /tmp/scalar-range --case raw --expression range --nullable 1 \
  --rows 262144 --batch 1024 --rounds 31 --warmup 5 --cpu 2
```

Same Release setup, warm storage, result cache disabled, 262,144 rows, batch
1,024, CPU 2, 31 measured rounds and 5 warmups. Total median milliseconds:

| Raw expression | Nullable | Old | New | New / old |
| --- | --- | ---: | ---: | ---: |
| a < 10 && b > 10 && a != 8 | yes | 2.424 | 26.856 | 11.08 |
| a < 10 | yes | 0.620 | 10.516 | 16.95 |
| a == 10 | yes | 0.656 | 10.421 | 15.88 |
| a != 10 | yes | 0.638 | 10.441 | 16.38 |
| a <= 10 | yes | 0.653 | 10.623 | 16.28 |
| a > 10 | yes | 0.628 | 10.402 | 16.56 |
| a >= 10 | yes | 0.648 | 10.435 | 16.10 |
| 10 <= a < 20 | yes | 0.628 | 10.452 | 16.63 |
| a IN [1, 5, 10, 90] | yes | 0.708 | 12.726 | 17.97 |
| a + 2 < 12 | yes | 0.644 | 11.840 | 18.37 |
| a < b | yes | 0.995 | 13.102 | 13.17 |
| a < 10 | no | 0.553 | 9.868 | 17.85 |

Relative to the previous validity-only optimization, observed new-mode Raw
compound latency decreased from 46.619 to 26.856 ms (42.4%); nullable unary
from 24.178 to 10.516 ms (56.5%); non-nullable unary from 22.183 to 9.868 ms
(55.5%). These are local before/after runs on different dates; the old compound
control also changed from 2.796 to 2.424 ms, so percentages include host noise.
The new path remains substantially slower than the old path. Compound mixed
index/Raw measured 23.717 ms new versus 1.733 ms old. All-index measured
21.110 ms versus 0.108 ms; this change does not optimize index bitmap mapping.

All benchmark variants passed independent output and actual-route checks.
348 focused C++ tests passed, including numeric types, NaN/infinities/extremes,
fragment boundaries, NULLs, sparse selections, owner lifetime, unaligned
fallback, repeated mappings and selected/unselected arithmetic overflow.

A separate 997 Hz Raw compound profile collected 3,009 new-mode samples.
Logical combination/other batch work accounted for 40.6%, final output/batch
setup 26.6%, and comparison/other input work 23.7%. The last group includes
Reader/input setup: the named `ComparePrimitive` frame itself accounted for
about 0.47% inclusive samples, and literal broadcast for 0.53%. Small sample
counts are approximate attribution, not precise kernel timings. Remaining
per-row BOOL access is concentrated in logical combination and output.

Reports, raw timing samples and profile evidence are retained in
`/mnt/dev_debug/scalar-batch-kernels-20261030/`.

Standalone new-mode E2E passed 285 cases; case names/statuses match the saved
old-mode baseline. Effective service configuration and loaded core library were
verified, and the service was stopped after validation.

## BOOL bitmap execution (2026-10-30)

The next stage keeps BOOL results in bitmap form through AND/OR/NOT,
IS NULL/IS NOT NULL, BOOL comparisons/ranges, index Range mapping, candidate
refinement and filter consumers. `CopyBoolMasks()` returns detached TRUE and
validity masks; `FromBooleanMasks()` checks truth/validity/coverage and takes
ownership of private buffers. Bitmap/constant slices use bulk operations;
other representations retain a validity-aware gather fallback.

AND only skips definite FALSE, OR only skips definite TRUE, and later operands
can override earlier UNKNOWN. The order of expression calls and cancellation
checkpoints is retained. Index proof conflicts, missing coverage and false
negative candidate indexes still fail; NULL is folded into exclusion only at
the filter consumers. Range index mapping copies bitmap slices; offsets keep
their position mapping, including duplicates.

Same local Release protocol: 262,144 rows, batch 1,024, CPU 2, 5 warmups and
31 measured rounds, warm storage, query result cache disabled. Total median
latencies in milliseconds (the prior stage is the immediately preceding
primitive-kernel measurement):

| Expression / layout | Nullable | Prior new | Current new | Current old | Reduction | New / old |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| a < 10 && b > 10 && a != 8 / raw | yes | 26.856 | 9.417 | 2.346 | 64.9% | 4.01 |
| a < 10 && b > 10 && a != 8 / b_index | yes | 23.717 | 5.803 | 1.732 | 75.5% | 3.35 |
| a < 10 && b > 10 && a != 8 / all_index | yes | 21.110 | 0.360 | 0.101 | 98.3% | 3.57 |
| a < 10 / raw | yes | 10.516 | 3.628 | 0.615 | 65.5% | 5.90 |
| a == 10 / raw | yes | 10.421 | 3.564 | 0.611 | 65.8% | 5.83 |
| a != 10 / raw | yes | 10.441 | 3.692 | 0.655 | 64.6% | 5.63 |
| a <= 10 / raw | yes | 10.623 | 3.512 | 0.601 | 66.9% | 5.85 |
| a > 10 / raw | yes | 10.402 | 3.626 | 0.628 | 65.1% | 5.78 |
| a >= 10 / raw | yes | 10.435 | 3.484 | 0.626 | 66.6% | 5.57 |
| 10 <= a < 20 / raw | yes | 10.452 | 3.586 | 0.617 | 65.7% | 5.81 |
| a IN [1, 5, 10, 90] / raw | yes | 12.726 | 5.637 | 0.675 | 55.7% | 8.35 |
| a + 2 < 12 / raw | yes | 11.840 | 4.840 | 0.597 | 59.1% | 8.11 |
| a < b / raw | yes | 13.102 | 5.785 | 0.926 | 55.8% | 6.25 |
| a < 10 / raw | no | 9.868 | 2.924 | 0.513 | 70.4% | 5.70 |

All 14 measured layout/expression variants passed independent output and
actual-path checks. These are local observations, not cross-machine guarantees;
the old controls also fluctuate. The new path remains slower than the old path.

353 focused C++ tests passed, including complete three-valued truth tables,
word boundaries, sparse selections, unaligned and overlapping copies, immutable
mask publication, all BOOL comparison/range combinations, index proof conflicts,
cancellation and failed/refined-result publication.

A separate 997 Hz profile over 501 Raw compound rounds collected 5,208 samples
inside the new GetOutput. Logical combination/other batch work decreased to
6.6% and final output/batch setup to 8.4% of current samples (previously 40.6%
and 26.6%). Comparison/other input work now accounts for 61.3%; this grouping
includes field loading and must not be interpreted as numeric comparison
kernel time. Reader `ReadRange` is 33.9% inclusive and its `ResultBuilder::AddBatch`
19.6% inclusive (nested, not additive). AddBatch still visits selected rows and
writes both per-result and per-fragment validity. Reader validation, repeated
input metadata construction and allocations remain optimization targets. Sample
shares are approximate attribution; the latency table ran without profiling.

Artifacts: `/mnt/dev_debug/scalar-bool-kernels-20261030/`.

Standalone new-mode E2E passed 285 cases; case names/statuses match the saved
old-mode baseline. Effective service configuration and loaded core library were
checked, and the service was stopped after validation.

## Reader validity construction and metadata reuse (2026-10-30)

Sealed Scan now converts validity once per physical span. Expanded bool flags
are packed in 64-row blocks; packed flags support unaligned slices without
assuming padding. The result is intersected with selection and copied into the
batch mask in bulk, while the same fragment mask is used by the borrowed view.
Validity-only and already-owned Take results share this conversion. Lazy,
scattered Take retains its position-by-position access.

The Reader bridge carries the values and validated defined/valid masks through
the deferred loader into InputSlot. A private constructor restricts this path
to the bridge after ScanNode/PositionNode validation. Ordinary value loaders
still return a view and derive validity inside InputSlot. Value/mask agreement,
coverage, immutable-batch consistency and transactional publication remain
checked; extra validity coverage is retained independently of value coverage.
Growing and reverse-lookup Readers also benefit from metadata propagation;
their physical validity-building loops were not changed in this stage.

Same Release benchmark protocol: 262,144 rows, batch 1,024, CPU 2, five warmups
and 31 measured rounds, warm storage, query result cache disabled. The prior
column is the immediately preceding BOOL bitmap stage; old is rerun alongside
current new. Total median latencies in milliseconds:

| Expression / layout | Nullable | Prior new | Current new | Current old | Reduction | New / old |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| a < 10 && b > 10 && a != 8 / raw | yes | 9.417 | 7.027 | 2.512 | 25.4% | 2.80 |
| a < 10 && b > 10 && a != 8 / b_index | yes | 5.803 | 4.219 | 1.784 | 27.3% | 2.37 |
| a < 10 && b > 10 && a != 8 / all_index | yes | 0.360 | 0.384 | 0.104 | -6.5% | 3.69 |
| a < 10 / raw | yes | 3.628 | 2.070 | 0.652 | 42.9% | 3.17 |
| a == 10 / raw | yes | 3.564 | 2.074 | 0.648 | 41.8% | 3.20 |
| a != 10 / raw | yes | 3.692 | 2.096 | 0.642 | 43.2% | 3.26 |
| a <= 10 / raw | yes | 3.512 | 2.205 | 0.642 | 37.2% | 3.43 |
| a > 10 / raw | yes | 3.626 | 2.068 | 0.619 | 43.0% | 3.34 |
| a >= 10 / raw | yes | 3.484 | 2.064 | 0.612 | 40.8% | 3.37 |
| 10 <= a < 20 / raw | yes | 3.586 | 1.987 | 0.635 | 44.6% | 3.13 |
| a IN [1, 5, 10, 90] / raw | yes | 5.637 | 4.208 | 0.710 | 25.4% | 5.93 |
| a + 2 < 12 / raw | yes | 4.840 | 3.152 | 0.638 | 34.9% | 4.94 |
| a < b / raw | yes | 5.785 | 2.694 | 0.898 | 53.4% | 3.00 |
| a < 10 / raw | no | 2.924 | 1.710 | 0.548 | 41.5% | 3.12 |

All 14 variants passed independent output and actual-path checks. All-index does
not exercise the optimized Reader and showed no gain; its slight change is not
attributed to this optimization. Old-mode controls fluctuate between runs.
The nullable unary case improves by 42.9%, but remains about 3.17 times the old
mode; this does not close the framework's performance gap.

356 focused C++ tests passed. New tests compare physical validity conversion
with a scalar oracle for All/None/sparse selections, all-valid/all-null/mixed
flags, offsets 0–15, word boundaries and short tails. Reader tests cross three
unaligned Cells and exercise both projections. Range and offsets bridge tests
verify extra validity reuse without another read, reject value/mask mismatch
and later validity conflicts, and verify successful retry after failed loads.
Existing tests cover cancellation, truncated Cells, lazy Take failures, NULL
semantics and duplicate offsets. These checks are focused, not a full repo run.

A separate 997 Hz profile over 1,000 nullable unary Raw rounds collected 2,371
samples inside new GetOutput. Reader ReadRange decreased from 61.8% to 34.0%
inclusive; OwnedSelection::SetSelected, previously 29.7% self, had no samples.
The remaining 0.7% in InputSlot::ExtractValidity includes non-Reader predicate
results; those still use view-derived validity. Allocator self samples are
12.4%, and final output/batch setup is 23.0% of current samples. These are CPU
sample shares, not additive wall-time phases, and indicate remaining work
rather than proving a single next bottleneck. Latencies above exclude profiling.

Artifacts: `/mnt/dev_debug/scalar-reader-validity-20261030/`.

Standalone new-mode E2E passed all 285 cases; case results match the saved
old-mode baseline (timings excluded). Effective new-mode configuration and
loaded core library were verified, and the service was stopped afterwards.

## Two primitive evaluation modes (2026-11-03)

The framework now separates physical payload readability from SQL validity.
Borrowed inputs default to `PrimitivePayloadAccess::ValidRowsOnly`. A producer
may explicitly grant `AllDefinedRows` for direct integer storage. Raw Scan
sets `integer_null_payload_readable` only for fixed-width integer targets whose
buffer length covers all physical slots; the flag resets on every output.
Validity-only/skipped requests and other backends do not implicitly grant it.
The Reader carries this capability into immutable views. Decoded spans retain
both their current view's defined coverage and the source capability.

A separate kernel policy selects `DenseThenMask` or `ValidRowsOnly`. Currently
same-type integer/TIMESTAMPTZ comparisons and ranges are eligible for dense
execution when the entire overlap is selected, defined and readable. They
compute a whole span and then preserve the intersection of operand validity,
clearing NULL truth bits. Partial selections retain valid-run execution; no
extra field reads are requested. Arithmetic/casts/parsers, floating/mixed-type
comparisons and BOOL retain their previous evaluation policy. All-NULL views
with no payload and mapped/unaligned inputs remain safe fallbacks. There is no
new user configuration switch and the default batch size is unchanged.

361 focused C++ tests passed, including both modes across integer widths,
six comparisons, four range boundaries, differing fragment boundaries,
NULL/NOT semantics, capability propagation and sparse coverage. An inaccessible
memory page backs a NULL slot to prove conservative comparison/range kernels
do not read it. Arithmetic tests place INT64_MIN in NULL and unselected slots,
then verify that division by -1 only fails when a valid offending row is
selected. Physical Scan tests verify capability creation and reset when
switching to validity-only.

Default-batch protocol: Release, 262,144 rows, batch 1,024, CPU 2, five warmups
and 31 measured rounds, warm storage and result cache disabled. Prior new is
the historical Reader-validity-stage measurement; current old is measured
alongside current new. Total medians in milliseconds:

| Expression / layout | Nullable | Prior new | Current new | Current old | Reduction | New / old |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| a < 10 && b > 10 && a != 8 / raw | yes | 7.027 | 7.272 | 2.384 | -3.5% | 3.05 |
| a < 10 && b > 10 && a != 8 / b_index | yes | 4.219 | 4.283 | 1.708 | -1.5% | 2.51 |
| a < 10 && b > 10 && a != 8 / all_index | yes | 0.384 | 0.385 | 0.102 | -0.2% | 3.75 |
| a < 10 / raw | yes | 2.070 | 1.892 | 0.625 | 8.6% | 3.03 |
| a == 10 / raw | yes | 2.074 | 1.938 | 0.617 | 6.6% | 3.14 |
| a != 10 / raw | yes | 2.096 | 1.931 | 0.639 | 7.9% | 3.02 |
| a <= 10 / raw | yes | 2.205 | 1.953 | 0.614 | 11.4% | 3.18 |
| a > 10 / raw | yes | 2.068 | 1.922 | 0.615 | 7.0% | 3.13 |
| a >= 10 / raw | yes | 2.064 | 1.985 | 0.636 | 3.8% | 3.12 |
| 10 <= a < 20 / raw | yes | 1.987 | 1.780 | 0.618 | 10.4% | 2.88 |
| a IN [1, 5, 10, 90] / raw | yes | 4.208 | 4.266 | 0.695 | -1.4% | 6.14 |
| a + 2 < 12 / raw | yes | 3.152 | 3.207 | 0.600 | -1.8% | 5.35 |
| a < b / raw | yes | 2.694 | 2.388 | 0.893 | 11.3% | 2.68 |
| a < 10 / raw | no | 1.710 | 1.692 | 0.568 | 1.0% | 2.98 |

All 14 variants passed independent output and actual new/old route checks.
Unary, ranges and column comparison benefit. Compound did not improve in this
measurement (Raw 7.03 -> 7.27 ms); IN/arithmetic were not enabled for dense
NULL evaluation and also showed no gain. Historical old controls fluctuate,
so these are observations, not a universal speedup claim.

Two additional unary diagnostics use one batch of 262,144 rows, CPU 2,
10 warmups and 51 measured rounds. The matching historical baseline comes
from `/mnt/dev_debug/scalar-gap-analysis-20261030/`:

| Unary layout | Prior new eval | Current new eval | Current old eval | Current new total | Current old total |
| --- | ---: | ---: | ---: | ---: | ---: |
| nullable | 0.341 | 0.121 | 0.115 | 0.133 | 0.123 |
| non-nullable | 0.066 | 0.068 | 0.054 | 0.080 | 0.062 |

For the nullable whole-batch case, pure execution is now 0.121 ms versus old
0.115 ms; previous new was 0.341 ms. This supports the dense-kernel change,
but does not recommend whole-segment Raw evaluation for every workload.
The default-batch nullable unary still takes about 3.03 times current old.
Batch construction, input/predicate caches, allocations and Reader lifecycle
remain separate work.

A separate 997 Hz profile over 1,000 nullable unary Raw rounds collected 2,061
samples in new GetOutput. The baseline scalar op_compare_val symbol had no
samples (previously 193 inclusive samples); the explicit AVX2 comparison path
was sampled. CompareConstant decreased from 278/2,371 to 69/2,061 inclusive
samples. These are CPU samples, not additive wall-time phases; the benchmark
table was measured without profiling. No full repository test run is claimed.

Artifacts: `/mnt/dev_debug/scalar-dense-mode-20261103/`.

Standalone new-mode E2E passed 285 cases; values/statuses match the saved
old-mode baseline, excluding timings. Effective service configuration and
loaded core library were checked, and the service was stopped after testing.

## Reusable batch workspace and metadata cost (2026-11-04)

This iteration reduces framework work without increasing the default batch size:

- `FilterBitsExecution` creates one reusable `ExecutionBatch` per execution.
  Reset clears field/predicate coverage and rebinds explicit positions, retaining
  slots, source callbacks and container capacity. Published immutable views keep
  their original owners; reset does not overwrite payload held by a consumer.
- `PredicateBatch` stores optional BOOL views directly, removing its second
  `BatchInput`/`InputSlot` layer. Field inputs retain their own value/validity cache.
- First field reads adopt validated Reader results without merging empty state.
  Borrowed validity and allocation-free mask comparisons remove temporary masks;
  direct TRUE-bit output avoids another BOOL validity/result conversion. Type,
  shape, requested coverage and NULL consistency remain checked.
- Prepared Reader callbacks use stack access nodes; a sealed Reader retains its
  physical Scan cursor for forward requests under the same OpContext. Backward
  requests, context changes and read failures recreate it. Every request still
  supplies its own position; ResultOwned pins preserve returned views.
- Comparison type dispatch is bound when the expression is built; the common
  single primitive span uses inline storage. Existing dense/valid-run policies,
  mixed-type comparison rules and exception boundaries remain intact.

Release on the same host, pinned CPU 2, 262,144 rows, batch 1,024, nullable INT64,
warm physical storage and cross-query result caching disabled. Fresh before and
after measurements use 10 warmups per mode, 101 rounds for unary and 51 for the
compound expression. Modes alternate execution order and validate the complete
output and actual routes outside timing. Total medians in milliseconds:

| Expression / layout | Before new | After new | After old | New reduction | After new / old |
| --- | ---: | ---: | ---: | ---: | ---: |
| a < 10 / raw | 1.936 | 1.103 | 0.637 | 43.0% | 1.73 |
| a < 10 && b > 10 && a != 8 / raw | 7.277 | 5.105 | 2.367 | 29.9% | 2.16 |
| a < 10 && b > 10 && a != 8 / b_index | 4.257 | 2.603 | 1.704 | 38.8% | 1.53 |
| a < 10 && b > 10 && a != 8 / all_index | 0.376 | 0.238 | 0.098 | 36.6% | 2.43 |

Unary evaluation alone decreased from 1.920 to 1.088 ms; prepare stayed around
0.013 ms. Total P95 decreased from 2.109 to 1.182 ms.
The old-mode unary control moved from 0.611 to 0.637 ms, so the table reports
observations rather than attributing every microsecond to an individual change.
New unary is still about 1.73 times old; this is not performance parity.

A separate GDB call-count check used 4,096 rows in four 1,024-row batches, one
warmup and three measured invocations. Counts were identical in all four new
invocations; debugger timings are excluded from latency results:

| Calls per unary execution | Before new | After new | After old |
| --- | ---: | ---: | ---: |
| Execution batch/workspace construction | 4 | 1 | n/a |
| InputSlot construction | 8 | 1 | n/a |
| InputSlot GetColumn / LoadValues | 8 / 8 | 4 / 4 | n/a |
| Physical Scan creation | 4 | 1 | 1 |
| GetChunk | 4 | 4 | 4 |
| InputSlot ExtractValidity | 4 | 0 | n/a |

The prior counts come from `/mnt/dev_debug/scalar-batch-gap-20261103/`.
This confirms container/cursor reuse, not elimination of field reads or pins.

Additional after-change Raw cases use 31 rounds and five warmups, otherwise the
same default-batch settings. All validate full output and physical routes:

| Expression | Nullable | New total ms | Old total ms |
| --- | --- | ---: | ---: |
| a == 10 | yes | 1.059 | 0.615 |
| a != 10 | yes | 1.113 | 0.616 |
| a <= 10 | yes | 1.042 | 0.624 |
| a > 10 | yes | 1.118 | 0.614 |
| a >= 10 | yes | 1.042 | 0.617 |
| 10 <= a < 20 | yes | 0.992 | 0.640 |
| a IN [1, 5, 10, 90] | yes | 3.361 | 0.692 |
| a + 2 < 12 | yes | 2.584 | 0.635 |
| a < b | yes | 1.556 | 0.927 |
| a < 10 | no | 0.956 | 0.514 |

Two one-batch unary diagnostics (262,144 rows per batch) also passed; nullable
new/old total was 0.176/0.263 ms, non-nullable 0.068/0.058 ms. These diagnostics
do not justify changing the default batch size. IN and arithmetic retain a
substantial gap; their scalar computation was not replaced in this iteration.

A separate 997 Hz profile of 1,000 unary rounds collected 1,170 samples inside
new GetOutput. Reader ReadRange accounted for 573 inclusive samples (49.0%),
workspace reset for 104 (8.9%), CompareConstant for 46 (3.9%), and direct output
copy for 19 (1.6%). Allocator self samples were 141 (12.1%) and overlap these
phases. These are approximate CPU samples, not additive wall-time costs.
Reader result/view/validity construction, per-batch output allocations and
resource release remain; reusing a workspace does not reuse immutable payload
buffers while returned views may still hold them.

376 focused C++ tests passed. Added/updated cases cover bitmap tails, unaligned
slices and overlapping aliases; sparse/NULL/dictionary fallback; wrong typed or
incomplete Reader results; incremental predicate validity disagreements;
Range/Offsets/empty/tail batch rebinding; old views after reset; query index cache
reuse; recursive reset rejection and failed-reset poisoning; backward physical
reads, context switches, cursor failure recovery and retained pins. Real
FilterBits read failures (FileReadFailed/DataFormatBroken), cancellation and
allocation failure are injected at cursor Next, after a prior batch succeeded,
and checked for preserved errors, no output publication and no legacy fallback.
No full repository test run is claimed.

Artifacts: `/mnt/dev_debug/scalar-batch-opt-20261104/` (before/after JSON samples,
UT XML, call counts, profile and runtime evidence).

Standalone new-mode E2E passed 285 cases; all case values/statuses match both
the saved old-mode baseline and the previous new-mode run, excluding timings.
Effective new-mode service configuration and the loaded core library were
verified. The local service was stopped after validation.

## Expression, backend and batch matrix (2026-11-04)

`scalar_execution_matrix.py` runs the same operator harness across two axes:

- Expressions: integer comparisons/range/IN/NULL/arithmetic/column comparison;
  string equality/prefix/LIKE; JSON numeric comparison/range/IN, contains/all/any,
  exists, arithmetic, and a sparse shared-BSON numeric path.
- Execution: Raw at the configured batch size (default 8,192); ordinary scalar
  STL_SORT and BITMAP indexes; real built/loaded JSON stats; a mixed-index
  conjunction. Exact index/stats cases assert WholeSegment in both engines,
  except JSON arithmetic: old uses Raw/Batch and new uses Stats/WholeSegment.
  That case is explicitly a path-change comparison, not framework overhead.

The optional `--diagnostic-full-batch` adds one-full-size Raw controls only for
bottleneck investigations. They are excluded from `summary.md` and written to
`diagnostic-summary.md`, with `diagnostic: true` in `matrix.json`. These are not
production performance results or a new planner setting. It separates batch
administration from work that scales with rows. Reports include each engine's
actual backend and execution granularity, the resulting number of batches,
Prepare/Eval/cleanup/total distributions, complete output checks and hit counts.
Preparation includes each query's compilation and resource binding. Cleanup is
measured before the final result is released, as in the existing timing contract.
No result is reused across timed queries.

```bash
source scripts/setenv.sh
python3 scripts/benchmark/scalar_execution_matrix.py \
  --output /tmp/scalar-matrix --rows 262144 \
  --rounds 31 --warmup 5 --cpu 2

# Repeat selected controls in reversed case order on alternate passes.
python3 scripts/benchmark/scalar_execution_matrix.py \
  --output /tmp/scalar-matrix-repeat --repeats 3 \
  --only eq in json_contains json_unary --cpu 2

# Individual cells remain runnable through the existing runner.
python3 scripts/benchmark/scalar_execution.py \
  --output /tmp/scalar-json-stats --expression json_contains --case stats --cpu 2
python3 scripts/benchmark/scalar_execution.py \
  --output /tmp/scalar-bitmap-in --expression in64 --case index --index-type bitmap --cpu 2
```

The default matrix has 59 cells (25 Raw cases, 22 numeric index cases,
9 JSON stats cases, 3 conjunction layouts). `--batch` should match the deployment
configuration; `--diagnostic-full-batch` adds 25 Raw diagnostic controls. The runner executes them sequentially
and preserves failed-case logs; any failure yields a nonzero exit status. Each
cell first checks 3,073 rows, including the short final batch, then measures the
requested row count while checking every result against an independent oracle.
Timing is not a CI pass/fail threshold. `--only` filters expressions; `--repeats`
repeats whole matrix passes, separately from measured query rounds within a cell.
`--binary` can select a preserved executable; use `LD_LIBRARY_PATH` to select
its matching preserved core library. Each cell records the actual loaded hashes.

`scalar_execution_summary.py MATRIX_JSON --repeats N` also generates
`aggregate.json` and `phase-summary.md` (the matrix runner invokes it automatically).
It groups only matching clocks, data and routes, reports medians of per-run
medians and ratio ranges, and separates Prepare/Eval/cleanup deltas in microseconds.
Incomplete or failed groups are marked and excluded from performance conclusions.
It reports backend/granularity/effective-batch counts for both modes and labels
path changes separately. Diagnostic full-batch controls get a separate report.

The 4-term integer IN uses the original domain [0,99]. The 64/1,024-term variants
both use [0,8191], with terms [0,63] / [0,1023]; these have different selectivity,
so their latency difference does not isolate list size alone. JSON documents
include SQL NULL (every 97th row), missing paths (every 23rd), JSON null (every
29th remaining row), and wrong types (every 31st remaining row). Normal documents
contain `n` and a two-element `arr`; `rare` occurs every 101st row. A stats cell
builds/uploads/loads a real stats generation outside timing, then both modes
query the same resources. Raw JSON retains the same documents. The oracle treats
JSON exists according to Milvus nonempty semantics, including JSON null as false.

The matrix does not yet cover JSON path cast indexes, text/GIS backends, ANN
candidate Offsets, cold storage, concurrent queries or service RPC latency.
Backend-specific computations can differ even when both plans say JsonStats;
a new/old latency delta alone is not a measurement of pure framework overhead.

`--clock thread_cpu` is an additional CPU-cost control, available on both runners.
It uses `CLOCK_THREAD_CPUTIME_ID` for the same Prepare/Eval/cleanup boundaries;
`--clock wall` remains the default. CPU measurements exclude time when the query
thread is descheduled, and exclude worker-thread CPU and I/O wait. They do not
remove frequency or shared-cache interference and must not be presented as
request latency. Compare old/new within one clock mode; do not subtract a CPU
measurement from a wall measurement. Each JSON report records `clock` explicitly.

### Matrix results and attribution

All 70 matrix cells passed full output and physical-route checks at 3,073 and
262,144 rows; each measured mode ran 31 rounds after five warmups. The production
core library hash is unchanged from the preceding workspace-optimization run.
This iteration changes only benchmark code, runners and documentation.

Wall results below use CPU 2, Release, nullable data and warm physical caches.
A separate task later started a benchmark pinned to CPU 2 during the Stats part
of the run. All original Stats and conjunction wall timings are therefore
excluded from conclusions (their correctness/route checks still passed). Their
raw artifacts are retained with the interference note. Seventeen additional
controls use thread CPU time on CPU 4, including all nine Stats predicates;
these also passed. The shared host still had other work, so thread CPU results
exclude scheduler wait but retain frequency/cache effects. They are CPU costs,
not isolated-machine latency; compare engines within each row/clock only.

Raw wall-time total medians, milliseconds. The full-batch control has one batch
of 262,144 rows; it remains a Raw/Batch plan in both engines:

| Expression | Old, batch 1,024 | New, batch 1,024 | New / old | Old, full batch | New, full batch | New / old |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| a < 10 | 0.615 | 1.048 | 1.71 | 0.132 | 0.104 | 0.79 |
| a == 10 | 0.629 | 1.093 | 1.74 | 0.131 | 0.102 | 0.78 |
| a != 10 | 0.617 | 1.057 | 1.71 | 0.139 | 0.110 | 0.79 |
| 10 <= a < 20 | 0.632 | 0.963 | 1.52 | 0.143 | 0.115 | 0.80 |
| a IN [1, 5, 10, 90] | 0.714 | 3.312 | 4.64 | 0.185 | 2.546 | 13.75 |
| a IN [0..63] | 2.642 | 3.863 | 1.46 | 2.187 | 3.053 | 1.40 |
| a IN [0..1023] | 2.637 | 4.572 | 1.73 | 2.123 | 3.657 | 1.72 |
| a + 2 < 12 | 0.624 | 2.558 | 4.10 | 0.141 | 1.732 | 12.24 |
| a < b | 0.984 | 1.528 | 1.55 | 0.197 | 0.157 | 0.80 |
| a IS NULL | 0.506 | 0.719 | 1.42 | 0.066 | 0.060 | 0.90 |
| s == 'value_10' | 2.102 | 18.657 | 8.88 | 1.595 | 17.774 | 11.15 |
| s LIKE 'value_1%' | 2.144 | 12.617 | 5.88 | 1.693 | 11.729 | 6.93 |
| s LIKE '%1%' | 5.081 | 15.561 | 3.06 | 4.518 | 14.634 | 3.24 |
| j['n'] < 10 | 23.971 | 34.867 | 1.45 | 28.204 | 38.323 | 1.36 |
| 10 <= j['n'] < 20 | 25.204 | 35.546 | 1.41 | 28.693 | 38.769 | 1.35 |
| j['n'] IN [1,5,10,90] | 25.797 | 35.060 | 1.36 | 30.596 | 39.259 | 1.28 |
| JSON_CONTAINS(j['arr'], 10) | 30.596 | 42.075 | 1.38 | 35.746 | 46.523 | 1.30 |
| JSON_CONTAINS_ALL(j['arr'], [10,11]) | 29.891 | 47.939 | 1.60 | 34.112 | 51.429 | 1.51 |
| JSON_CONTAINS_ANY(j['arr'], [1,5,10,90]) | 29.370 | 56.520 | 1.92 | 34.400 | 61.013 | 1.77 |
| EXISTS j['n'] | 21.373 | 31.672 | 1.48 | 26.201 | 36.100 | 1.38 |
| j['n'] + 2 < 12 | 24.294 | 36.570 | 1.51 | 28.828 | 40.823 | 1.42 |
| j['rare'] < 10 | 22.054 | 31.656 | 1.44 | 26.874 | 35.872 | 1.33 |

Ordinary index wall-time medians, milliseconds; both modes actually execute one
WholeSegment query despite the configured batch size of 1,024:

| Expression | STL_SORT old | STL_SORT new | New / old | BITMAP old | BITMAP new | New / old |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| a < 10 | 0.036 | 0.058 | 1.60 | 0.024 | 0.046 | 1.92 |
| a == 10 | 0.020 | 0.042 | 2.13 | 0.017 | 0.039 | 2.34 |
| 10 <= a < 20 | 0.037 | 0.059 | 1.61 | 0.025 | 0.047 | 1.91 |
| a IN [1, 5, 10, 90] | 0.029 | 0.054 | 1.85 | 0.021 | 0.045 | 2.20 |
| a IN [0..63] | 0.021 | 0.065 | 3.10 | 0.035 | 0.080 | 2.30 |
| a IN [0..1023] | 0.074 | 0.449 | 6.10 | 0.283 | 0.671 | 2.37 |
| a IS NULL | 0.017 | 0.028 | 1.66 | 0.017 | 0.028 | 1.64 |

JSON Stats **thread CPU** total medians, milliseconds (different clock and CPU
from the wall tables above). Except arithmetic, both modes assert JsonStats and
WholeSegment. Arithmetic compares old Raw/Batch with new Stats/WholeSegment and
is explicitly excluded from same-backend framework-overhead claims:

| Expression | Old CPU ms | New CPU ms | New / old | Path comparison |
| --- | ---: | ---: | ---: | --- |
| j['n'] < 10 | 3.748 | 4.807 | 1.28 | same_backend |
| 10 <= j['n'] < 20 | 4.066 | 6.219 | 1.53 | same_backend |
| j['n'] IN [1,5,10,90] | 5.590 | 6.915 | 1.24 | same_backend |
| JSON_CONTAINS(j['arr'], 10) | 37.796 | 45.226 | 1.20 | same_backend |
| JSON_CONTAINS_ALL(j['arr'], [10,11]) | 49.239 | 49.934 | 1.01 | same_backend |
| JSON_CONTAINS_ANY(j['arr'], [1,5,10,90]) | 36.460 | 55.487 | 1.52 | same_backend |
| EXISTS j['n'] | 1.032 | 1.103 | 1.07 | same_backend |
| j['n'] + 2 < 12 | 51.607 | 6.107 | 0.12 | path_change |
| j['rare'] < 10 | 0.737 | 0.889 | 1.21 | same_backend |

The measurements identify separate issues:

1. **Per-batch framework cost remains for simple Raw predicates.** Equality
   takes 1.093/0.629 ms (new/old) at batch 1,024 but 0.102/0.131 ms in one full
   batch. Range and NULL show the same trend. This bounds a batch-related gap;
   it is not a precise decomposition because batching also changes locality
   and kernel invocation size. JSON Raw becomes slower in one full batch in
   both engines, so increasing every production batch is not the remedy.
2. **Raw IN/arithmetic have additional kernel gaps.** Four-term IN still takes
   2.546/0.185 ms in one full batch. In a separate 997 Hz profile, 1,571/1,636
   new GetOutput self samples (96.0%) are in the primitive IN lambda, which
   includes set lookup, valid-row traversal and per-row bitmap output. New
   [ScalarPredicates.cpp](../src/exec/expression/ScalarPredicates.cpp) always
   constructs `std::unordered_set`; old
   [TermExpr.cpp](../src/exec/expression/TermExpr.cpp) chooses small-list SIMD
   `SimdBatchElement` or the existing large-set implementation and caches its
   batch filter. These paths currently do not use equivalent computation.
3. **Indexed large IN pays for an unused Raw alternative.** For 1,024 terms on
   STL_SORT, new/old Prepare is 0.355/0.007 ms, Eval 0.082/0.066 ms, and total
   0.449/0.074 ms. Most of this gap is preparation. The compiler recursively
   builds the exact Raw expression before physical selection, creates per-term
   literal/vector views, packs constants and constructs the Raw hash set.
   Prepare later chooses ExactIndex, leaving that work unused. Source trace:
   [ExprCompiler.cpp](../src/exec/plan/ExprCompiler.cpp) → `MakeTerm` →
   [ExecutionPreparer.cpp](../src/exec/plan/ExecutionPreparer.cpp). A filtered
   profile contains 438 new compilation samples inside the timed Run;
   `ReadOnlyVectorView::CopyOf` and `MakeTerm` are substantial inclusive stacks.
   Their inclusive counts overlap and must not be added. This is a concrete
   compile/prepare-layer issue, separate from batch management.
4. **WholeSegment still has bitmap adaptation overhead.** Small indexed
   comparisons add roughly 18 microseconds of Eval at 262,144 rows in the wall
   run, plus preparation. Thread-CPU size controls for BITMAP equality show:

| Rows | Old Eval CPU us | New Eval CPU us | Difference us |
| ---: | ---: | ---: | ---: |
| 4,096 | 10.79 | 14.08 | 3.29 |
| 262,144 | 19.45 | 52.80 | 33.35 |
| 1,048,576 | 58.68 | 192.12 | 133.44 |

   Prepare does not grow in the same proportion. The fourfold Eval-gap increase
   between 262,144 and 1,048,576 rows is consistent with repeated whole-bitmap
   work. Code trace shows normalization/validation in `IndexResult::Full`,
   truth/validity materialization in `MapPredicate` / `FromBooleanMasks`, and
   final copying in `CopyTrueTo`. The size control supports investigating this
   bridge; it does not measure the individual cost of each copy/check.
5. **The profiled Stats gaps lie mostly inside the backend implementation.**
   `JsonStatsReader::Query` contains 786/799 new GetOutput samples for JSON
   unary (98.4%) and 4,508/4,514 for contains (99.9%). The surrounding workspace,
   mapping and final output are small in these two profiles. Stats contains
   allocates a `vector<bool>` per BSON array and performs generic value/term
   dispatch; allocator self samples are 14.2% of new execution. Raw JSON
   contains similarly spends 93.0% inside its generic predicate function,
   including view access, parsing and comparison; Reader ReadRange is 4.5%.
   These are inclusive/self sample shares, not additive timings or a claim
   about every Stats predicate. See
   [ScalarIndexReader.cpp](../src/segcore/reader/ScalarIndexReader.cpp) and
   [ScalarJsonPredicates.cpp](../src/exec/expression/ScalarJsonPredicates.cpp).

Follow-up priorities are therefore distinct: defer constructing unused Raw
executors/constants after physical selection; reuse legacy IN kernel selection
and batch output; reduce full-index bitmap adaptation copies/checks; optimize
Stats/JSON inner kernels separately. This benchmark iteration implements none
of those production optimizations and makes no service-throughput claim.

Artifacts: `/mnt/dev_debug/scalar-matrix-20261104/`. `full/matrix.json` retains all
70 wall runs, `contention.json` marks excluded groups, `cpu/runs.json` contains
17 thread-CPU controls, and per-case reports preserve samples, Prepare/Eval/
cleanup, route/shape checks, clocks, library hashes and environment. Five 997 Hz
profiles run on CPU 4 provide attribution only; their timings are excluded.
The matching pre-clock-extension benchmark executable is retained as
`profile-binary` for reproducible symbolization. No new service E2E or full
repository test run is claimed for this test-only change.

## Integer IN batch kernels (2026-11-05)

The migrated integer IN now reuses the legacy runtime-dispatched
`SimdBatchElement` for small sets and `SetElement` for larger sets, using the
same lane-count threshold. Constants are sorted and deduplicated locally:
direct C++ callers do not necessarily pass through the Go rewriter. This
covers INT8/16/32/64 and TIMESTAMPTZ. FLOAT/DOUBLE, BOOL and string IN retain
their existing paths; in particular this change does not apply the legacy
binary-search SIMD tails to floating-point NaNs.

`MatchPrimitive` decodes each span once. For fully selected, defined integer
payloads whose NULL slots are explicitly readable, it filters the whole span
then applies validity to both masks. Otherwise it filters only contiguous
selected valid ranges. Constant spans evaluate membership once per range;
arbitrary dictionary mappings retain the encoding-aware fallback. The bitmap
builder clears and writes private result ranges and publishes validity in bulk.
Large hash sets still perform per-element probes inside the batch kernel.

Validation: 378 focused C++ tests passed, including two added tests covering
all five integer logical types, empty/unsorted/duplicate/large term sets,
extreme values, segmented input with unaligned bitmap boundaries, sparse
coverage, repeated dictionary rows, constants, all-NULL input and NOT.
Protected pages verify that a NULL without readable-payload permission and an
undefined/unselected slot are not read. Reader, planner, candidate filter,
FilterBits and JSON stats regression tests were included. No service E2E or
sanitizer run was performed in this iteration.

The full CMake build started rebuilding unrelated Rust storage dependencies.
After compiling the affected production C++ objects, validation reused the
existing dependency libraries and the exact CMake compile/link commands for
the core and focused tests. The new core was loaded from `cmake_build/src`;
installed service libraries were not replaced. The saved pre-change core and
benchmark executable were loaded separately for before measurements.

The following are warm-column wall-time medians, 262,144 rows, CPU 4, 31
measured rounds after 5 warmups. Every invocation checks the full result
bitmap against an independent oracle and asserts the physical route; old/new
execution modes alternate within each process. All 22 before/after runs
passed. “Legacy” is the legacy mode in the after-run; “before” and “after”
refer to the new execution mode. No compilation ran concurrently with this
matrix. Per-run library hashes, P95s, samples and phase splits are retained.

Configured batch size (1,024), used for routine conclusions:

| Expression / path | Nullable | Batch size | Legacy ms | New before ms | New after ms | After / legacy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| in / raw | True | 1,024 | 0.685 | 3.400 | 1.047 | 1.53x |
| in / raw | False | 1,024 | 0.588 | 3.643 | 0.974 | 1.66x |
| in64 / raw | True | 1,024 | 2.594 | 3.858 | 1.800 | 0.69x |
| in1024 / raw | True | 1,024 | 2.494 | 4.671 | 2.127 | 0.85x |
| in / all_index | True | 1,024 | 0.030 | 0.057 | 0.058 | 1.94x |
| in1024 / all_index | True | 1,024 | 0.077 | 0.468 | 0.429 | 5.57x |
| unary / raw | True | 1,024 | 0.609 | 1.122 | 1.096 | 1.80x |

Historical diagnostic controls (Raw batch=Segment rows); excluded from routine
performance conclusions:

| Expression / path | Nullable | Batch size | Legacy ms | New before ms | New after ms | After / legacy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| in / raw | True | 262,144 | 0.188 | 2.491 | 0.161 | 0.86x |
| in / raw | False | 262,144 | 0.112 | 2.871 | 0.127 | 1.13x |
| in64 / raw | True | 262,144 | 2.094 | 2.949 | 0.945 | 0.45x |
| in1024 / raw | True | 262,144 | 2.059 | 3.590 | 1.181 | 0.57x |

At the configured batch size of 1,024, four-item nullable Raw IN improved from
3.400 ms to 1.047 ms (69.2% lower), but remains 1.53x legacy (0.685 ms). This
is the primary conclusion for this optimization.

In the historical four-item nullable Raw IN full-batch diagnostic, execution fell
from 2.491 ms to 0.161 ms (93.5% lower); the legacy control was 0.188 ms.
After-change Eval medians were 0.144 ms new versus 0.180 ms legacy; Prepare
was 0.015 ms versus 0.007 ms. The original 13.75x regression is removed for
this diagnostic workload; that is not evidence of production parity. This is not a claim that all IN predicates or new-mode paths
are faster: four-item non-nullable full-batch IN remains about 1.13x legacy,
and nullable batch-1,024 IN remains 1.53x. Exact-index preparation overhead
also remains (1,024 terms: 0.429 ms total versus 0.077 ms legacy).

Artifacts: `/mnt/dev_debug/scalar-in-opt-20261105/`, including `matrix.json`,
per-case `report.json`, `focused-tests.xml`, protected-page tests in
`ScalarExprTest.cpp`, and retained build commands and pre-change binaries.

## Deployment-default performance matrix (2026-11-05)

The service default is **8,192**, not the earlier UT default of 1,024:
`QueryNodeCfg.ExprEvalBatchSize` in `pkg/util/paramtable/component_param.go`
is passed through `internal/util/initcore/query_node.go` /
`SetDefaultExprEvalBatchSize` to `QueryConfig`; `common/Consts.h` also defaults
to 8,192. Both Python runners now default to 8,192. A deployment override
must be supplied with `--batch`. The earlier 1,024 measurements remain valid
for that configuration but are not the primary comparison for the service
default. The initial partial 1,024 run in this artifact directory was
interrupted after checking the service default and excluded from conclusions.

No production C++ code was changed in this iteration. The tested library
includes the preceding integer IN optimization. Frozen copies of the
benchmark executable and core library were used throughout, with stable
hashes verified for all reports. Other dependencies were reused from the
existing native setup; remote Dev CLI authentication was unavailable.

Method: 262,144 rows, configured batch 8,192, nullable fields, warm physical
columns/indexes, cross-query result cache disabled, Release, Intel i7-8700,
single query thread pinned to CPU 4. This remains a shared development host.
Three independent matrix passes (alternating case order), each with 31
measured rounds after five warmups per engine. Old/new modes alternate order
within each round. All **59 combinations × 3 passes = 177 runs** passed
complete output-bitmap and actual-route checks at 3,073 and 262,144 rows.
Nine separate 997 Hz profiles and six index-size controls also passed.
No timing from profiled runs is used in the tables. No service E2E, ANN,
RPC, cold-I/O, or concurrent-query throughput is measured.

Raw executes 32 batches in both engines. Ordinary Index and JSON Stats
predicates use their genuine WholeSegment route (one execution); the configured
batch was never inflated to the Segment size. JSON arithmetic is explicitly
different: legacy Raw/32 batches versus new JSON Stats/one execution. Its
speedup must not be described as reduced framework overhead.

Tables show medians of the three per-run medians, in milliseconds; ratios
divide these displayed aggregate times (before rounding). Full per-run ratio
ranges, P95s, samples, phase deltas and routes are in `phase-summary.md` and
`aggregate.json`. Phase deltas are medians of paired run-median differences,
so they need not exactly equal differences between independently aggregated
columns or add up to total.

### Raw (32 batches)

| Expression | Backend | Old ms | New ms | New / old | ΔPrepare µs | ΔEval µs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| a < 10 | Raw | 0.205 | 0.233 | 1.14 | 4.33 | 18.82 |
| a == 10 | Raw | 0.208 | 0.225 | 1.08 | 4.07 | 12.94 |
| a != 10 | Raw | 0.214 | 0.241 | 1.13 | 4.31 | 18.75 |
| a <= 10 | Raw | 0.214 | 0.236 | 1.10 | 4.26 | 17.19 |
| a > 10 | Raw | 0.212 | 0.245 | 1.16 | 4.50 | 19.31 |
| a >= 10 | Raw | 0.210 | 0.235 | 1.12 | 4.17 | 20.45 |
| 10 <= a < 20 | Raw | 0.224 | 0.237 | 1.06 | 4.39 | 5.68 |
| a IN [1, 5, 10, 90] | Raw | 0.267 | 0.276 | 1.03 | 7.73 | 3.52 |
| a IN [0..63] | Raw | 2.148 | 1.049 | 0.49 | 26.65 | -1125.32 |
| a IN [0..1023] | Raw | 2.132 | 1.323 | 0.62 | 330.35 | -1145.75 |
| a + 2 < 12 | Raw | 0.211 | 2.059 | 9.77 | 4.32 | 1845.75 |
| a < b | Raw | 0.325 | 0.341 | 1.05 | 3.37 | 12.21 |
| a IS NULL | Raw | 0.131 | 0.141 | 1.07 | 3.58 | 4.05 |
| s == 'value_10' | Raw | 1.493 | 18.097 | 12.12 | 6.25 | 16530.69 |
| s LIKE 'value_1%' | Raw | 1.672 | 11.854 | 7.09 | 4.21 | 10207.92 |
| s LIKE '%1%' | Raw | 4.463 | 14.490 | 3.25 | 4.43 | 10022.33 |
| j['n'] < 10 | Raw | 23.250 | 33.581 | 1.44 | 14.17 | 10395.94 |
| 10 <= j['n'] < 20 | Raw | 23.812 | 34.449 | 1.45 | 14.50 | 11081.56 |
| j['n'] IN [1,5,10,90] | Raw | 25.778 | 34.578 | 1.34 | 14.04 | 8913.23 |
| JSON_CONTAINS(j['arr'], 10) | Raw | 29.998 | 41.062 | 1.37 | 13.28 | 11404.05 |
| JSON_CONTAINS_ALL(j['arr'], [10,11]) | Raw | 29.046 | 46.184 | 1.59 | 14.72 | 17284.16 |
| JSON_CONTAINS_ANY(j['arr'], [1,5,10,90]) | Raw | 28.530 | 55.428 | 1.94 | 15.83 | 26885.03 |
| EXISTS j['n'] | Raw | 20.821 | 30.347 | 1.46 | 12.91 | 9513.49 |
| j['n'] + 2 < 12 | Raw | 23.464 | 35.568 | 1.52 | 12.33 | 11766.91 |
| j['rare'] < 10 | Raw | 21.114 | 30.449 | 1.44 | 11.96 | 9317.90 |
| a < 10 && b > 10 && a != 8 | Raw | 0.993 | 2.429 | 2.45 | 10.77 | 1424.92 |

### Ordinary indexes (one execution)

| Expression | Backend | Old ms | New ms | New / old | ΔPrepare µs | ΔEval µs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| a < 10 | STL_SORT | 0.037 | 0.061 | 1.65 | 3.49 | 20.63 |
| a < 10 | BITMAP | 0.025 | 0.050 | 1.98 | 3.40 | 20.52 |
| a == 10 | STL_SORT | 0.020 | 0.045 | 2.25 | 3.19 | 21.05 |
| a == 10 | BITMAP | 0.017 | 0.042 | 2.44 | 3.17 | 21.02 |
| a != 10 | STL_SORT | 0.019 | 0.044 | 2.26 | 3.40 | 20.70 |
| a != 10 | BITMAP | 0.017 | 0.042 | 2.41 | 3.32 | 20.87 |
| a <= 10 | STL_SORT | 0.039 | 0.063 | 1.63 | 3.22 | 20.47 |
| a <= 10 | BITMAP | 0.026 | 0.050 | 1.95 | 3.13 | 20.59 |
| a > 10 | STL_SORT | 0.044 | 0.068 | 1.56 | 3.37 | 20.76 |
| a > 10 | BITMAP | 0.084 | 0.109 | 1.29 | 3.41 | 21.17 |
| a >= 10 | STL_SORT | 0.042 | 0.066 | 1.56 | 3.63 | 20.36 |
| a >= 10 | BITMAP | 0.084 | 0.109 | 1.29 | 3.52 | 20.93 |
| 10 <= a < 20 | STL_SORT | 0.036 | 0.061 | 1.69 | 3.43 | 20.81 |
| 10 <= a < 20 | BITMAP | 0.026 | 0.050 | 1.97 | 3.66 | 20.89 |
| a IN [1, 5, 10, 90] | STL_SORT | 0.030 | 0.058 | 1.94 | 6.64 | 21.05 |
| a IN [1, 5, 10, 90] | BITMAP | 0.020 | 0.048 | 2.38 | 6.50 | 21.03 |
| a IN [0..63] | STL_SORT | 0.021 | 0.067 | 3.22 | 24.53 | 20.68 |
| a IN [0..63] | BITMAP | 0.035 | 0.081 | 2.32 | 24.87 | 20.83 |
| a IN [0..1023] | STL_SORT | 0.075 | 0.413 | 5.53 | 322.37 | 15.76 |
| a IN [0..1023] | BITMAP | 0.280 | 0.622 | 2.22 | 325.78 | 17.84 |
| a IS NULL | STL_SORT | 0.017 | 0.029 | 1.68 | 2.67 | 8.96 |
| a IS NULL | BITMAP | 0.017 | 0.029 | 1.69 | 2.53 | 8.95 |
| a < 10 && b > 10 && a != 8 | STL_SORT | 0.101 | 0.251 | 2.48 | 8.89 | 138.82 |

### JSON Stats

| Expression | Backend | Old ms | New ms | New / old | ΔPrepare µs | ΔEval µs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| j['n'] < 10 | JSON Stats | 1.850 | 2.436 | 1.32 | 6.40 | 591.08 |
| 10 <= j['n'] < 20 | JSON Stats | 2.174 | 2.994 | 1.38 | 6.27 | 816.96 |
| j['n'] IN [1,5,10,90] | JSON Stats | 2.907 | 4.040 | 1.39 | 7.21 | 1076.86 |
| JSON_CONTAINS(j['arr'], 10) | JSON Stats | 19.112 | 19.764 | 1.03 | 10.94 | 530.93 |
| JSON_CONTAINS_ALL(j['arr'], [10,11]) | JSON Stats | 21.512 | 20.798 | 0.97 | 12.85 | -597.88 |
| JSON_CONTAINS_ANY(j['arr'], [1,5,10,90]) | JSON Stats | 17.846 | 23.398 | 1.31 | 10.48 | 5540.46 |
| EXISTS j['n'] | JSON Stats | 0.312 | 0.343 | 1.10 | 5.65 | 21.57 |
| j['n'] + 2 < 12 [path change] | JSON Stats | 23.829 | 3.366 | 0.14 | 10.10 | -20484.67 |
| j['rare'] < 10 | JSON Stats | 0.152 | 0.181 | 1.19 | 5.19 | 24.79 |

### Mixed Raw/index conjunction

| Expression | Backend | Old ms | New ms | New / old | ΔPrepare µs | ΔEval µs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| a < 10 && b > 10 && a != 8 | STL_SORT | 0.698 | 1.010 | 1.45 | 11.09 | 300.30 |

### Attribution and remaining work

1. **Simple Raw scalar execution is close at the service batch size.** Six
   integer comparisons, range, four-term IN, NULL and column comparison have
   modest residual overhead (roughly 3–16% in these aggregate measurements).
   The 64/1,024-term IN kernels are faster than legacy, although large-IN
   preparation still costs more. The earlier batch-1,024 gap must not be
   generalized to the service default.
2. **String view abstraction is a demonstrated framework hot spot.** New
   string equality takes 18.097 ms versus 1.493 ms. Of 2,993 new GetOutput
   CPU samples, 2,135 (71.3%) are disjoint self samples in `IsValid`,
   `StringViewAt`, `SelectionView::IsSelected`, `ValueAddress` and
   `defined_rows`. `ReadSelectedValue` checks validity, then `StringViewAt`
   checks it again and resolves the representation/address; even the constant
   operand is accessed per row. This is not string payload copying. The
   next target is span-level string decoding, constant binding and bulk result
   construction while retaining sparse/NULL semantics. See
   [ScalarExpr.cpp](../src/exec/expression/ScalarExpr.cpp) and
   [VectorAccess.cpp](../src/common/VectorAccess.cpp).
3. **Compound sparse execution has expensive validation and small-range work.**
   The Raw conjunction takes 2.429 ms versus 0.993 ms. New execution has
   323/1,301 samples (24.8%, inclusive) under `ReadResult::Validate`, and
   47.0% under `ComparePrimitive`. A segmented view can fail
   `TryBorrowValidity` and make validation reconstruct validity through
   per-row access. Selected valid runs also invoke comparison/write helpers
   repeatedly after short circuiting. These measurements support optimizing
   validity propagation and short-range kernels; they do not isolate the
   benefit of each proposed change. See
   [ReadResult.h](../src/segcore/reader/ReadResult.h) and
   [ScalarBatchKernels.h](../src/exec/expression/ScalarBatchKernels.h).
4. **Indexed large IN has unnecessary compile/prepare work.** STL_SORT with
   1,024 terms costs 0.413 ms versus 0.075 ms; approximately 322 µs of the
   paired gap is Prepare, versus 16 µs Eval. The compiler constructs the Raw
   alternative, per-term literal/vector views and packed constants before
   choosing ExactIndex. The filtered profile contains 326 new compile
   samples, including 143 under `ReadOnlyVectorView::CopyOf` and 49 under
   `MakeTerm` (inclusive, overlapping counts; do not add). Defer construction
   of unused Raw evaluators after physical selection. See
   [ExprCompiler.cpp](../src/exec/plan/ExprCompiler.cpp:1197).
5. **Small index queries retain result-adaptation overhead.** Equality adds
   about 21 µs Eval plus about 3 µs Prepare at 262,144 rows, for both ordinary
   index backends. Additional BITMAP equality controls, each repeated three
   times with configured batch 8,192 and actual WholeSegment execution, show:

| Segment rows | Old Eval µs | New Eval µs | Paired Eval delta µs |
| ---: | ---: | ---: | ---: |
| 8,192 | 5.844 | 7.733 | 1.839 |
| 262,144 | 10.255 | 30.877 | 21.019 |
| 1,048,576 | 25.258 | 103.535 | 78.813 |

   This scaling is consistent with additional full-bitmap normalization,
   validation, view construction and final copying in `IndexResult::Full`,
   `MapPredicate`, `FromBooleanMasks` and `CopyTrueTo`. It does not measure
   each operation independently; it rules out treating all of the delta as
   constant batch-initialization cost.
6. **Arithmetic still needs kernel migration work.** `a + 2 < 12` takes
   2.059 ms versus 0.211 ms. About 94% of new GetOutput self samples are in
   arithmetic evaluation, scalar `Arithmetic` and scalar `Compare`; legacy
   uses the batch arithmetic-comparison kernel. This is separate from
   workspace construction and should preserve the existing integer overflow
   and exception rules when optimized.
7. **JSON costs include both generic row access and backend computation.**
   Raw contains is 1.37x and contains_any is 1.94x legacy. Its profile includes
   view access, parsing and element matching, so the entire delta must not be
   labeled framework overhead. In the three profiled JSON Stats cases,
   99.1–99.9% of new GetOutput samples are inside `JsonStatsReader::Query`.
   Stats contains is approximately at parity, contains_all is slightly faster,
   but contains_any remains 1.31x. BSON traversal, generic term dispatch and
   per-array `vector<bool>` allocation are backend hot spots. The outer
   whole-segment workspace/mapping is small for these measured Stats cases.

Priorities from this run: eliminate per-row string view resolution; avoid
validity reconstruction and expensive tiny-range calls in sparse compounds;
lazily construct unused Raw index alternatives; reduce whole-index bitmap
adaptation; migrate arithmetic and JSON kernels separately. These are identified
follow-ups, not production fixes implemented in this benchmark iteration.

Artifacts: `/mnt/dev_debug/scalar-matrix-production-20261105/`. Primary results
are in `b8192-wall/matrix.json`, `aggregate.json`, `phase-summary.md`, and
per-case reports/XML. `size-controls.json` contains the six extra controls;
`profile-summary.json`, per-case `*-detail.json` and `index-prepare.json` retain
the sampling evidence. The interrupted `wall/` directory is excluded.


## String span kernels (2026-11-11, cloud validation)

String comparisons decode physical spans once, including mixed owned strings and
borrowed string views. Nullable comparisons use valid runs by default; sealed
StringViews explicitly guarantee readable NULL slots and permit dense comparison
followed by truth/validity masking. Pattern matching and IN retain valid-only
execution, and mapped/indirect fallback resolves each selected position once.
The IN set owns its constants and accepts string_view lookup without copying the
candidate. It reuses the legacy large-set `SetElement<std::string>` implementation
and bulk membership writer. Prefix/suffix/contains dispatch outside the row loop
and check cancellation between chunks of at most 256 valid rows; LIKE/regex keep
per-row checks. No generic string NULL-read guarantee is assumed.

The benchmark now includes `string_range`, `string_in` (four short constants),
and `string_in_long` (the same domain with a 256-byte prefix). Both the native
harness and Python runner default to batch 8192. Six string cases can also be
selected through Dev CLI's gtest-filter interface, without environment forwarding:

```bash
milvus-dev-cli cpp-ut local /path/to/milvus -b master --allow-dirty \
  -t internal/core/src/exec/operator/ScalarExecutionPerformanceTest.cpp \
  -k 'Paths/ScalarExecutionPerformanceTest.StringBatchMatrix/Raw'
```

`StringBatchMatrix` skips normal wildcard UT runs and requires its name explicitly
in the filter. It runs three passes over equality, range, prefix, LIKE, short IN,
and long IN; each compares alternating old/new executions of 262144 rows with
batch 8192, five warmups and 31 measured rounds. Complete result bitmaps and
actual Raw/32-batch routes are checked. JSON records prefixed `SCALAR_STRING_PERF`
are emitted into the cloud test log, including Prepare/Eval/cleanup/total samples.
Cloud timings must be reported separately from the November 5 pinned local
measurements; comparisons within a record share a process and input.

Verification covers both operand orders, unequal fragment boundaries, mixed
storage, sparse selections, duplicate mappings, empty and long strings, embedded
NUL, and protected NULL/unselected slots. Existing fusion tests exercise lazy
invalid-regex construction and three-valued logic. An additional test cancels
inside each matcher (including a fused pair), checks the propagated cancellation
code, and retries to verify that no partial result was published.

Dev CLI cloud C++ UT passed **138 tests**; the task reached `succeeded`.
Task: `cpp-ut-local-lu-zhang-head-cab9d65-6b357367-u59677`, attempt created
`2026-11-11T08:04:23.359404Z`, completed `2026-11-11T08:27:33.976046Z`.
The task ID was reused across source revisions; these results refer only to this
attempt and its 138-test log. This was a Release cloud build, not local execution,
PR Jenkins, or a service E2E run. No explicit CPU affinity was configured.

The table uses the median of three per-pass median total times. Each pass has
31 timed executions per mode, with Prepare/Eval/cleanup included. Ratio ranges
are the three paired per-pass ratios, not confidence intervals.

| Expression (Raw, nullable) | Old ms | New ms | New / old | Paired ratio range |
| --- | ---: | ---: | ---: | ---: |
| String equality | 1.164 | 1.288 | 1.11x | 0.95–1.11x |
| String range | 2.771 | 3.259 | 1.18x | 1.17–1.18x |
| Prefix | 1.356 | 1.379 | 1.02x | 1.01–1.02x |
| LIKE | 3.660 | 4.263 | 1.16x | 1.16–1.16x |
| IN, four short strings | 4.245 | 2.104 | 0.50x | 0.50–0.50x |
| IN, four strings with 256-byte prefix | 9.556 | 9.070 | 0.95x | 0.95–0.96x |

The measured prefix path is close to legacy; short IN is faster and long IN is
slightly faster. Equality varies from 0.95x to 1.11x across passes, so the 1.11x
aggregate is not a stable per-run penalty. Range and LIKE retain about 18% and
16% total overhead respectively. Their Eval deltas are about 472 µs and 574 µs,
while Prepare deltas are only about 12 µs and 14 µs. The remaining gap is mainly
inside evaluation for these cases; this run does not isolate validity-run setup,
matcher checks and bitmap writing with a new profile.

**IN algorithm caveat:** legacy uses linear comparison for up to four string
constants and a hash set for larger lists; the new path measured here uses the
legacy hash-set implementation for both. These are valid end-to-end comparisons,
but their differences cannot be attributed solely to framework overhead. The
selection is not a claim that hashing wins for every list size or text domain.

Do not subtract these cloud timings from the November 5 local baseline or the
previous cloud attempt: workers and build environments can differ. The old/new
measurements within each final record share the executable, data and process.
No service throughput or Index/Stats speedup is claimed by this string Raw run.

Artifacts: `/mnt/dev_debug/scalar-string-opt-20261111/`. `cloud-final.json`
contains the final source's test log, `cloud-reports.json` its 18 records, and
`cloud-summary.json` the aggregate. `final-snapshot-hashes.json` identifies the
uploaded source; prior attempt logs are retained separately.


## Shared batch kernels (2026-11-12)

The same span access contract now covers arithmetic, FLOAT/DOUBLE IN,
Roaring/Bloom membership, mixed numeric comparison, JSON predicates and ARRAY
operations. BOOL flat storage is packed by typed runs before bitmap algebra.
Arithmetic dispatches the operation and comparison outside the value loop;
Add/Sub/Mul reuse existing bulk kernels, while division, modulo, conversions and
parsing retain selected-valid-only evaluation and their original error checks.
Same-type FLOAT/DOUBLE comparisons, ranges, IN and Add/Sub/Mul may evaluate a
whole selected span then mask NULLs only when its producer grants readable
physical slots. Fixed-width Sealed Scan verifies the buffer length and explicitly
grants this capability; validity-only reads clear it. Mixed numeric comparisons also use readable dense spans while retaining native
C++ promotions (including INT64/DOUBLE precision and NaN semantics). Geometry
does not inherit string dense capability. Conversions, division, modulo and parsers stay valid-only.
Integer scalar heads/tails explicitly wrap to match the SIMD kernel. Floating
IN rejects NaN probes in the scalar fallback as well as SIMD.

Sealed ordinary ARRAY Scan and already-owned Take results retain pinned ArrayView
descriptors instead of copying each array payload into Array. Explicit writable
copies still detach; lazy Take and recursive/vector-array representations retain
their existing materialization boundaries. Dictionary and mixed owned/borrowed
fragments preserve element NULLs.

JSON/ARRAY borrow typed object spans and write optional BOOL results by bitmap
words. Homogeneous JSON contains prepares typed constants and extracts one probe per
array element; heterogeneous and nested-array coercion remains unchanged.
ANY contains allocates no ALL scratch; ALL scratch is local to an Eval
(or a synchronous stats query), then reused across documents. ARRAY membership
prepares a constant hash table and borrows string elements. GEOS parsing and
MATCH's independent element batches remain semantic work; their shared input
access/result reduction is optimized. Full-selected legacy calls omit identity
mapping and scatter but still detach writable arguments and publish snapshots.

Run the expanded matrix through Dev CLI (cloud execution):

```bash
milvus-dev-cli cpp-ut local /path/to/milvus -b master --allow-dirty \
  -t internal/core/src/exec/operator/ScalarExecutionPerformanceTest.cpp \
  -k 'Paths/ScalarExecutionPerformanceTest.KernelBatchMatrix/*'
```

This opt-in matrix uses **batch 8192, 262144 rows, five warmups, 31 alternating
old/new measurements, and three passes**. Raw cases include Add/Sub/Mul/Div/Mod,
FLOAT/DOUBLE arithmetic and IN, BOOL IN, mixed comparison, Roaring, integer/string
Bloom, ARRAY element/length/equality/ANY/ALL, legacy function, JSON unary/IN/
contains/arith, compound and string controls. It also measures JSON stats and
Mixed/AllIndex controls; WholeSegment is accepted only after route assertions.
The normal `TypedKernelCorrectness/Raw` test checks the new typed fixtures against
an independent exclusion-bitmap oracle. ARRAY performance fixtures have nullable
rows and non-null elements; separate migration tests cover element NULL semantics
where the old implementation is not a valid oracle.

`SCALAR_KERNEL_PERF` JSON records retain Prepare/Eval/cleanup/total samples,
actual routes and effective batch counts. `StringBatchMatrix` retains its existing
`SCALAR_STRING_PERF` prefix. The Python single-case and matrix runners accept the
same expanded expression names. These are deterministic workload comparisons,
not production throughput predictions or pass/fail latency thresholds.

Validation: **280 focused C++ tests passed** through Dev CLI cloud execution;
36 workloads produced **108 records** with matching results and asserted routes.
Task `cpp-ut-local-lu-zhang-head-cab9d65-7b8db42a-u59677`, created
`2026-11-12T13:38:36.939678Z`, completed `2026-11-12T14:00:05.545159Z`.
Source hash: `src-v2-1b167956499ab476c511`. This is not PR Jenkins CI or service E2E.
The focused tests include NaN/Inf/signed zero, integer overflow across SIMD
heads/tails, sparse/dictionary inputs, protected NULL payloads, cancellation,
JSON coercion, array element validity, borrowed-view pin lifetime and explicit
detachment. Python benchmark entry points pass syntax checks.

The table uses the median of three per-pass total-time medians (Prepare + Eval +
cleanup), in milliseconds. Each pass alternates old/new execution in the same
process; ratios below 1 favor the new mode. Raw uses 32 batches of 8192 rows;
Index/Stats may use WholeSegment only when the physical plan asserts it.
Do not compare absolute timings with previous local/cloud workers.

| Expression | Path | Old ms | New ms | New / old |
| --- | --- | ---: | ---: | ---: |
| `in1024` | AllIndex | 0.073 | 0.428 | 5.89× |
| `compound` | Raw | 1.060 | 2.476 | 2.33× |
| `in` | AllIndex | 0.025 | 0.052 | 2.09× |
| `array_all` | Raw | 4.100 | 8.475 | 2.07× |
| `legacy_call` | Raw | 9.598 | 17.784 | 1.85× |
| `array_any` | Raw | 4.098 | 7.539 | 1.84× |
| `unary` | AllIndex | 0.034 | 0.059 | 1.72× |
| `array_element` | Raw | 2.780 | 4.554 | 1.64× |
| `json_in` | JSON Stats | 2.878 | 4.283 | 1.49× |
| `array_length` | Raw | 1.947 | 2.799 | 1.44× |
| `compound` | Mixed | 0.711 | 1.020 | 1.43× |
| `json_unary` | JSON Stats | 1.762 | 2.505 | 1.42× |
| `mixed_compare` | Raw | 0.465 | 0.569 | 1.22× |
| `json_contains_any` | JSON Stats | 18.867 | 22.722 | 1.20× |
| `float_in` | Raw | 0.218 | 0.261 | 1.20× |
| `float_arith` | Raw | 0.188 | 0.224 | 1.19× |
| `double_in` | Raw | 0.251 | 0.295 | 1.17× |
| `arith_sub` | Raw | 0.224 | 0.260 | 1.16× |
| `arith_mod` | Raw | 0.969 | 1.119 | 1.15× |
| `arith_mul` | Raw | 0.237 | 0.273 | 1.15× |
| `double_arith` | Raw | 0.225 | 0.259 | 1.15× |
| `arith` | Raw | 0.224 | 0.258 | 1.15× |
| `arith_div` | Raw | 0.969 | 1.089 | 1.12× |
| `json_unary` | Raw | 21.516 | 22.795 | 1.06× |
| `array_eq` | Raw | 3.754 | 3.950 | 1.05× |
| `json_arith` | Raw | 21.392 | 22.397 | 1.05× |
| `json_contains_any` | Raw | 25.897 | 27.107 | 1.05× |
| `json_contains_all` | Raw | 26.381 | 26.268 | 1.00× |
| `roaring` | Raw | 3.092 | 3.053 | 0.99× |
| `string_eq` | Raw | 1.246 | 1.214 | 0.97× |
| `json_in` | Raw | 23.588 | 22.514 | 0.95× |
| `bloom_string` | Raw | 3.375 | 3.181 | 0.94× |
| `bloom_int` | Raw | 3.034 | 2.694 | 0.89× |
| `json_contains_all` | JSON Stats | 23.922 | 20.932 | 0.88× |
| `string_in` | Raw | 4.164 | 2.045 | 0.49× |
| `bool_in` | Raw | 1.057 | 0.399 | 0.38× |

The expanded matrix exposed an additional shared ARRAY Reader cost: converting
every `ArrayView` to an owned `Array` before evaluation. In the intermediate
cloud run, the five ARRAY cases ranged from 7.48× to 13.79×. Keeping pinned
ArrayView descriptors reduced the final paired ratios to 1.05×–2.07×. This
benefits subscript, length, equality and both contains variants through the same
input representation. It does not remove the materialization required by
explicit writable copies, lazy Take, recursive arrays or vector arrays.

Remaining gaps are retained in the table, not treated as fixed:

- AllIndex `in1024`: about 337.5 us of the new mode's 427.8 us total is Prepare;
  this is a separate planning cost, not per-row Raw view dispatch.
- Raw compound: 2.33×; the aggregate test does not isolate its remaining Eval
  cost. Further profiling is needed before attributing it to a single layer.
- ARRAY contains: 1.84×–2.07× and subscript 1.64× remain after removing payload
  copies. This run does not claim their remaining inner-loop/intermediate-output
  costs are eliminated or separately profiled.
- Legacy call: 1.85×. Identity mapping/scatter is removed for full selections;
  writable arguments and published results still require ownership isolation.
- JSON stats unary/IN remain 1.42×–1.49×. Stats contains ALL is 0.88×; this does
  not imply that every stats predicate became faster.

Artifacts: `/mnt/dev_debug/scalar-kernels-20261112/`. `log7.json` and
`status7.json` are the final run; `final-reports.json` preserves all samples and
plans, `final-summary.json` aggregates them, and `submitted7-hashes.json`
identifies the uploaded source. `phase1-*` and `log4.json` retain the successful
intermediate run that exposed ARRAY copying. Earlier failed/ cancelled attempts
are kept separately and are not used as final validation.


## IN preparation after physical selection (2026-11-13)

Ordinary scalar IN now separates semantic constant compilation from the Raw
lookup structure. Compile validates types/non-finite values, drops out-of-range
narrow integer terms, and publishes one typed constant view. It no longer builds
and snapshots a separate vector for every literal. Index eligibility uses the
same allocation-free representability check. A planned IN retains a Raw factory;
AdjustExecution invokes it only for Raw or candidate Refine, once per prepared
query, and reuses the resulting program across batches. ExactIndex never builds
the Raw SIMD/hash structure. Explicit raw-only compilation already knows its
route and constructs its program directly. JSON/ARRAY-specific IN kernels retain
their existing preparation and empty/NULL/coercion rules.

Validated with Dev CLI **cloud C++ UT**, baseline master, task
`cpp-ut-local-lu-zhang-head-cab9d65-4107c716-u59677` (final attempt created
`2026-11-13T04:20:47.418670Z`, completed `2026-11-13T04:24:57.024248Z`):
**197 passed**, two intentional skips for the IN benchmark's Mixed/Stats
parameters. This was not a service E2E or Jenkins run. The new physical-plan
checks verify zero Raw factory calls for ExactIndex, one for Raw/Refine, no
rebuild across sparse Eval and repeated/reordered offsets, independent preparation
of a shared plan, narrow overflow/all-overflow/empty lists, NULL and NOT, typed
constants, malformed constants rejected before binding, and index failures
propagated without a Raw fallback. Existing compiler, migration, scalar,
FilterBits and complex-operator checks are included.

`Paths/ScalarExecutionPerformanceTest.InPrepareMatrix/*` is an explicit opt-in
benchmark: 4/64/1024 terms, Raw and STL_SORT AllIndex, three passes each (18 JSON
records), 262,144 nullable rows, Raw batch **8,192**, five warmups and 31
alternating old/new measured executions per pass. Every execution validates the
exclusion bitmap; physical-route checks require 32 Raw batches or one genuinely
all-index execution. Storage is warm and result caching is disabled. Totals
include Prepare, Eval and cleanup. Each table entry is the median of the three
per-pass medians, so independently summarized phases need not sum exactly.

| Terms / path | Old total ms | New total ms | New / old | Old Prepare µs | New Prepare µs | Old Eval µs | New Eval µs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 / raw | 0.2455 | 0.2848 | 1.16× | 8.10 | 15.54 | 235.56 | 266.53 |
| 64 / raw | 0.8529 | 0.8951 | 1.05× | 8.29 | 20.52 | 842.52 | 871.71 |
| 1024 / raw | 1.8017 | 0.9454 | 0.52× | 8.52 | 91.26 | 1792.66 | 850.73 |
| 4 / all_index | 0.0255 | 0.0507 | 1.99× | 4.20 | 6.87 | 20.48 | 42.79 |
| 64 / all_index | 0.0159 | 0.0423 | 2.66× | 4.16 | 8.59 | 10.72 | 32.59 |
| 1024 / all_index | 0.0669 | 0.1181 | 1.77× | 5.17 | 36.62 | 60.38 | 80.50 |

For 1024-term AllIndex, the previous run recorded 337.50 µs new Prepare and
427.85 µs new total; this run records 36.62 µs and 118.14 µs respectively
(about 89% and 72% lower). Those are historical cross-run comparisons, not a
same-process before/after experiment. The stronger current paired comparison
still shows **1.77×** legacy total (66.89 µs). The remaining gap includes roughly
31.45 µs extra Prepare and 20.12 µs extra Eval; this run does not isolate the
individual functions responsible. Small indexed IN still has about 22 µs extra
Eval, and Raw 4/64-term IN remains 1.16×/1.05× legacy. Do not describe this change
as eliminating all framework overhead.


## Adaptive selection kernels (2026-11-13)

Numeric comparisons and ranges now choose among contiguous-range batch kernels,
a packed sparse loop and dense computation followed by selection/validity masks.
The decision samples at most eight bitmap words, using kernel-specific relative
row, range-start and bitmap costs. All/None encodings bypass sampling. The model
is a bounded heuristic, not an exact row/run count or a universal density cutoff.
Mixed numeric comparisons have a separate cost profile from same-type SIMD.

The eligibility check is independent of cost: dense execution requires a total
kernel and already readable input payloads. `CanEvaluateAll` retains full physical
coverage through a logical Restrict, while `CanReadAll` keeps its strict logical
coverage contract. Only numeric comparisons/ranges use the new permission;
strings, IN, checked arithmetic, JSON parsing and external callbacks retain their
existing policies. The change does not issue additional Reader requests or
publish results outside selection. A partially defined borrowed input remains
ineligible even if the producer permits reads of its defined NULL slots.

`ScalarSelectionPerformanceTest.ModeMatrix` explicitly measures all four
preferences (Ranges, Sparse, Dense and Adaptive) at 1/5/10/25/50/90/100 percent,
with scattered and clustered selections and full versus partial payload
coverage: 28 records, 8,192 positions each, five warmup rounds, 31 measured rounds
and 16 repetitions. Preference order rotates. Timing includes decoding,
selection handling, kernel execution and result publication; correctness checks
are outside the timer. Forced Dense falls back to Sparse when ineligible. The
reported chosen mode uses the same canonicalized active mask as the kernel.

`Paths/ScalarExecutionPerformanceTest.AdaptiveBatchMatrix/*` pairs old/new full
operator executions for Raw unary, equality, range, compound and mixed-type
comparison, plus compound with a b index. There are three passes per case,
262,144 nullable rows, batch **8,192**, five warmups and 31 alternating measured
executions per pass. Every run checks the exclusion bitmap and selected physical
route. Storage is warm and result caching is off. AllIndex/Stats parameters are
intentionally skipped because this change concerns Raw kernels. Phase tables
use the median of three per-pass medians; independently aggregated phases need
not add exactly to the total.

Validated by Dev CLI **cloud C++ UT**, baseline master: **187 passed**, two
intentional AllIndex/Stats benchmark skips. Task
`cpp-ut-local-lu-zhang-head-cab9d65-0b71d4d2-u59677`, final attempt created
`2026-11-13T07:12:36.280089Z`; test log reports Remote test completed; task metadata still running at capture.
Checks include six comparison operators across fixed/adaptive modes, NULL and
partial coverage, unaligned slices/writes, dictionary fallback, mixed numeric
semantics and protected undefined memory. Even forced Dense must fall back when
that memory is not readable. Existing exception/short-circuit checks also pass.
This is not service E2E or Jenkins validation.

| Expression / path | Old total ms | New total ms | New / old | Old Prepare µs | New Prepare µs | Old Eval µs | New Eval µs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unary / raw | 0.2933 | 0.3316 | 1.13× | 11.25 | 19.31 | 281.05 | 308.91 |
| eq / raw | 0.2918 | 0.3320 | 1.14× | 11.41 | 18.89 | 278.67 | 307.99 |
| range / raw | 0.3123 | 0.3406 | 1.09× | 11.97 | 20.58 | 297.90 | 315.96 |
| compound / raw | 1.5696 | 2.1173 | 1.35× | 28.23 | 62.07 | 1534.99 | 2045.55 |
| mixed_compare / raw | 0.7993 | 0.6988 | 0.87× | 13.37 | 20.77 | 781.93 | 671.37 |
| compound / b_index | 0.9867 | 0.7189 | 0.73× | 19.48 | 38.60 | 964.93 | 672.82 |

Selected fixed-mode comparisons below are per 8,192-position invocation in µs,
including decode/selection/publication. A dash means Dense is not eligible;
forced Dense would actually use Sparse. These are whole microbench timings,
not isolated decision-cycle measurements.

| Pattern / selected | Full payload | Adaptive choice | Ranges µs | Sparse µs | Dense µs | Adaptive µs |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| scattered / 1% | True | dense | 3.444 | 3.209 | 1.762 | 1.909 |
| scattered / 1% | False | sparse | 3.454 | 3.224 | — | 3.369 |
| scattered / 10% | True | dense | 20.757 | 4.054 | 1.610 | 1.746 |
| scattered / 10% | False | sparse | 21.017 | 4.071 | — | 4.178 |
| scattered / 90% | True | dense | 29.915 | 14.053 | 3.065 | 1.747 |
| scattered / 90% | False | sparse | 29.992 | 14.065 | — | 14.105 |
| scattered / 100% | True | ranges | 1.440 | 15.762 | 2.769 | 1.441 |
| scattered / 100% | False | ranges | 1.435 | 15.752 | 2.723 | 1.443 |
| clustered / 1% | True | ranges | 1.464 | 2.923 | 1.601 | 1.597 |
| clustered / 1% | False | ranges | 1.481 | 2.943 | — | 1.616 |
| clustered / 50% | True | dense | 2.020 | 10.139 | 1.619 | 1.905 |
| clustered / 50% | False | ranges | 1.884 | 9.380 | — | 2.018 |

Raw compound previously measured 2.476 ms new versus 1.060 ms old (2.33×).
The final run measures 2.117 ms versus 1.570 ms (1.35×).
The change relative to 2.476 ms is a historical cross-run comparison;
use the current paired ratio for the remaining gap. Reader, selection/cache and
logical-combination overhead have not been individually re-profiled here, so
this result does not assign the residual cost to one function.

Initial calibration runs exposed missing sparse bitmap scan/packing costs,
underestimated tiny-range startup and unnecessary dense masking for uniform All.
The final model includes those terms and uses a single range for uniform All.
The final matrix has a worst adaptive/best-fixed ratio of about 1.18×;
clustered selections still show small mischoices. Median adaptive-minus-chosen
fixed time is about 0.13 µs across these records, but whole-invocation timing
also includes dispatch effects and noise, so this is not an isolated scheduler
cost. In particular, some dense records are faster in adaptive mode than in
forced Dense; their difference cannot be credited to negative decision cost.
Sampling can still miss a distribution, and cost constants are platform/kernel
estimates; the mode matrix deliberately retains those limitations instead of
asserting an absolute timing threshold. Do not extrapolate numeric comparison
costs to expensive JSON/string/callback kernels.

Artifacts: `/mnt/dev_debug/scalar-adaptive-20261113/`. Final `log3.json`,
`status3.json`, `selection-reports3.json`, `kernel-reports3.json`,
`final-summary.json` and `submitted3-hashes.json` retain logs, individual samples,
physical routes and the exact submitted source manifest. Earlier logs preserve
calibration runs; only the third attempt validates the final cost parameters.


## Remaining Raw kernels: adaptive selection (2026-11-13)

Numeric SIMD/hash IN, string comparisons/ranges/hash IN, string matchers,
ordinary arithmetic, JSON/ARRAY predicates, ARRAY/collection projections and
timestamp calendar projections now use the shared selector. IN binds costs when
its chosen Raw kernel is prepared, using deduplicated term count and SIMD/hash
representation; it does not rebuild terms per batch. Strings, arithmetic and
strict callbacks have separate cost classes. Packed sparse callbacks avoid
calling a batch kernel for each single selected row. Optional predicates pack
truth and validity together; typed projections bind the output type/address once.

Dense permission remains independent of cost. Only audited total numeric/string
kernels may expand computation over already-readable physical payloads. JSON,
ARRAY, LIKE/regex, GEOS parsing, checked conversions, division/modulus and
calendar arithmetic stay selected-only. String literal matcher checkpoints
remain outside contiguous inner loops (at most 256 selected rows per check);
LIKE/regex keep per-row checks. JSON EXISTS establishes requested validity after
sparse output, retaining FALSE for SQL NULL. Legacy calls optimize only selected
position gathering, keeping exactly one function invocation, argument isolation,
ascending candidate order and scatter. BOOL masks, index/stats and MATCH parent
scheduling retain their specialized implementations; migrated MATCH children
use their corresponding scalar kernels.

`ScalarSelectionPerformanceTest.RemainingModeMatrix` measures 4-term SIMD IN,
1024-term hash IN, string IN, JSON unary, integer addition and division at
10/50/90/99 percent, scattered and clustered, with full/partial payload coverage
(96 records). Each uses 8,192
positions, five warmups, 31 rotated fixed/adaptive rounds and four repeats.
Numbers are full decode/selection/kernel/publication latency, not isolated
scheduler time. Forced Dense uses Sparse when ineligible (always for JSON and
division).
Every output is checked outside the timer.

`Paths/ScalarExecutionPerformanceTest.RemainingAdaptiveBatchMatrix/Raw` covers
32 expression shapes with three passes each: 262,144 nullable rows, batch
**8,192**, five warmups and 31 alternating old/new measurements per pass.
Prepare, Eval and total are recorded separately; each result and physical route
is checked. Storage is warm and result caching is disabled. Index/stats are not
part of this Raw-kernel change. This is Dev CLI cloud C++ UT, not service E2E or
Jenkins. No absolute timing threshold is a correctness assertion.

Final validation: **171 tests passed** in Dev CLI cloud C++ UT, baseline
master. Task `cpp-ut-local-lu-zhang-head-cab9d65-a8531784-u59677`, attempt created
`2026-11-13T10:33:16.644143Z`. The test log reports completion; task metadata at capture
is `running`. Source is identified by `submitted3-hashes.json`.
Tests cover SIMD/hash IN under all preferences, NULL/partial coverage/protected
memory, sparse optional outputs and unaligned boundaries, JSON EXISTS and missing
paths, callback order, typed projection, cancellation/retry, one legacy invocation,
and calendar overflow in unselected gaps. Existing scalar migration, geometry,
complex-operator and FilterBits checks are included. This is not a service E2E.

Each entry below is the median of three per-pass medians; phases need not sum
exactly after independent aggregation. Ratios compare old/new in the same run,
not the current run with a previous machine/measurement.

| Raw expression | Old total ms | New total ms | New / old | Old Prepare µs | New Prepare µs | Old Eval µs | New Eval µs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| array_all | 5.5420 | 12.9650 | 2.34× | 15.55 | 29.13 | 5517.55 | 12928.95 |
| array_any | 5.6014 | 12.9273 | 2.31× | 15.71 | 28.29 | 5578.22 | 12891.52 |
| legacy_call | 10.3061 | 19.8448 | 1.93× | 23.18 | 29.82 | 10285.23 | 19812.65 |
| array_element | 3.5331 | 5.4345 | 1.54× | 14.22 | 27.08 | 3518.63 | 5412.13 |
| array_length | 2.4092 | 3.6714 | 1.52× | 16.61 | 21.61 | 2391.31 | 3644.62 |
| arith_div | 2.4459 | 3.0392 | 1.24× | 16.43 | 23.42 | 2426.00 | 3008.28 |
| arith_mod | 2.3701 | 2.8970 | 1.22× | 13.05 | 25.69 | 2349.53 | 2865.13 |
| string_like | 4.2445 | 5.1538 | 1.21× | 13.51 | 24.47 | 4228.93 | 5122.84 |
| float_arith | 0.2440 | 0.2817 | 1.15× | 10.96 | 19.48 | 230.76 | 260.29 |
| float_in | 0.2604 | 0.2960 | 1.14× | 11.06 | 20.34 | 248.05 | 272.53 |
| arith | 0.2683 | 0.3046 | 1.14× | 11.01 | 18.64 | 255.06 | 283.20 |
| double_arith | 0.2685 | 0.3035 | 1.13× | 11.19 | 19.68 | 256.19 | 281.00 |
| timestamp_arith | 29.6872 | 33.1256 | 1.12× | 25.26 | 35.53 | 29657.76 | 33085.93 |
| double_in | 0.2983 | 0.3324 | 1.11× | 11.15 | 20.47 | 285.42 | 307.60 |
| json_arith | 25.5857 | 28.3830 | 1.11× | 21.15 | 36.27 | 25563.15 | 28349.90 |
| in | 0.2838 | 0.3091 | 1.09× | 10.78 | 19.96 | 270.48 | 284.79 |
| json_unary | 25.8274 | 28.0936 | 1.09× | 20.95 | 36.37 | 25808.36 | 28050.75 |
| json_contains_any | 31.4015 | 33.9440 | 1.08× | 19.79 | 40.28 | 31374.15 | 33904.64 |
| string_prefix | 1.6498 | 1.7779 | 1.08× | 14.86 | 26.64 | 1634.91 | 1747.28 |
| array_eq | 4.9317 | 5.2235 | 1.06× | 16.37 | 23.64 | 4906.64 | 5199.15 |
| in64 | 0.8628 | 0.9125 | 1.06× | 11.38 | 25.35 | 851.35 | 883.63 |
| json_exists | 23.0920 | 24.2933 | 1.05× | 16.24 | 28.17 | 23074.93 | 24241.68 |
| json_contains_all | 32.0051 | 32.8804 | 1.03× | 20.70 | 36.11 | 31983.08 | 32840.56 |
| bloom_string | 4.1345 | 4.2049 | 1.02× | 19.33 | 25.99 | 4115.69 | 4177.32 |
| roaring | 4.0146 | 4.0428 | 1.01× | 17.67 | 22.12 | 3995.63 | 4015.89 |
| string_eq | 1.4507 | 1.4548 | 1.00× | 12.24 | 27.41 | 1430.32 | 1424.10 |
| bloom_int | 3.6431 | 3.6517 | 1.00× | 16.87 | 20.93 | 3619.06 | 3626.45 |
| string_in_long | 9.7506 | 9.7509 | 1.00× | 22.57 | 52.45 | 9711.16 | 9677.76 |
| json_in | 28.6353 | 28.4817 | 0.99× | 20.91 | 40.24 | 28605.92 | 28413.75 |
| string_range | 3.0431 | 2.7750 | 0.91× | 14.04 | 26.66 | 3027.87 | 2740.05 |
| string_in | 4.5939 | 3.4396 | 0.75× | 15.56 | 27.44 | 4572.33 | 3410.19 |
| in1024 | 2.1765 | 1.1092 | 0.51× | 12.48 | 110.41 | 2156.08 | 995.34 |

Selected microbench records (µs per 8,192-position invocation). Dense is omitted
where forbidden; its forced preference falls back to Sparse. Fixed/adaptive
timings include dispatch and publication, so subtracting them does not isolate
the decision's CPU cost.

| Kernel / pattern / selected | Full payload | Choice | Ranges µs | Sparse µs | Dense µs | Adaptive µs |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| in4 / scattered / 10% | True | dense | 27.432 | 7.145 | 2.988 | 3.116 |
| in1024 / scattered / 10% | True | sparse | 16.779 | 6.829 | 25.578 | 6.958 |
| arith_div / scattered / 10% | True | sparse | 15.693 | 2.454 | — | 2.533 |
| arith_add / scattered / 10% | True | dense | 18.761 | 2.718 | 1.414 | 1.530 |
| string_in / scattered / 10% | True | sparse | 22.206 | 13.760 | 60.352 | 13.787 |
| json_unary / scattered / 10% | True | sparse | 86.733 | 79.438 | — | 79.530 |
| in4 / scattered / 10% | False | sparse | 24.446 | 6.327 | — | 6.420 |
| in1024 / scattered / 10% | False | sparse | 19.073 | 7.793 | — | 7.919 |
| arith_div / scattered / 10% | False | sparse | 18.547 | 2.937 | — | 3.084 |
| arith_add / scattered / 10% | False | sparse | 19.989 | 3.198 | — | 3.080 |
| string_in / scattered / 10% | False | sparse | 25.614 | 15.184 | — | 14.999 |
| json_unary / scattered / 10% | False | sparse | 89.224 | 81.689 | — | 81.921 |
| arith_div / scattered / 90% | True | sparse | 40.230 | 18.549 | — | 18.424 |
| string_in / scattered / 90% | True | dense | 82.780 | 97.994 | 62.221 | 59.455 |
| in4 / scattered / 90% | False | sparse | 49.490 | 31.011 | — | 31.027 |
| in1024 / scattered / 90% | False | sparse | 39.493 | 39.682 | — | 39.775 |
| arith_div / scattered / 90% | False | sparse | 21.744 | 10.150 | — | 10.229 |
| arith_add / scattered / 90% | False | sparse | 30.391 | 11.695 | — | 11.796 |
| string_in / scattered / 90% | False | ranges | 89.919 | 113.864 | — | 91.311 |
| json_unary / scattered / 90% | False | ranges | 710.383 | 702.020 | — | 710.674 |
| arith_div / scattered / 99% | True | ranges | 10.064 | 10.505 | — | 10.195 |
| in4 / clustered / 50% | True | dense | 2.642 | 20.241 | 2.784 | 2.865 |
| in1024 / clustered / 50% | True | ranges | 14.999 | 22.445 | 27.078 | 15.158 |
| arith_div / clustered / 50% | True | ranges | 3.922 | 6.180 | — | 4.197 |
| arith_add / clustered / 50% | True | dense | 1.647 | 6.940 | 1.470 | 1.586 |
| string_in / clustered / 50% | True | ranges | 35.566 | 56.743 | 66.182 | 36.067 |
| json_unary / clustered / 50% | True | ranges | 382.068 | 387.336 | — | 381.577 |

The worst observed adaptive/best-eligible-fixed ratio is 1.12×
(`arith_add`, clustered, 10%, full payload
True). Bounded sampling and cost classes are approximations;
they do not promise the fastest mode for every distribution, string length or
platform. Whole-operator gaps remain visible in the table. This run does not
attribute residual reader/cache/planning costs to individual functions.

Artifacts: `/mnt/dev_debug/scalar-adaptive-rest-20261113/`, final `log3.json`,
`status3.json`, `operator-reports3.json`, `operator-summary3.json`,
`selection-reports3.json` and `submitted3-hashes.json`. For attempt 3,
`log3-raw.json` preserves the aggregated provider output; `log3.json` excludes
the identified previous-attempt Pod `k9w2d`, retaining the current attempt. Earlier attempts are
kept separately and do not validate later cleanup or calibration changes.

## Migrated expression regression audit (2026-11-14)

Production execution code is unchanged from the 2026-11-13 final validated
snapshot. This change expands the test module, not the execution implementation.
The audit covers **50 Raw expression shapes / 85 expression-backend combinations**:
50 Raw, 2 mixed (STL_SORT/BITMAP), 24 ordinary-index and 9 JSON-stats combinations.
Each has three passes of five warmups and 31 alternating old/new measurements:
**255 operator records**, 262,144 nullable rows, Raw **batch 8,192**. Route and full
exclusion-bitmap checks run outside the timers. Index/stats WholeSegment routes
are checked explicitly; this is not Raw with a segment-sized batch.

Wall-clock Dev CLI cloud C++ UT, warm storage and result cache disabled. The main
run passed 10 tests. A second targeted run passed the two diagnostic tests,
including the real legacy function's copy stages. No service E2E or production
throughput claim is made. The two cloud runs are not compared by absolute time;
all ratios and diagnostic controls compare alternatives within their own run.

Values below are medians of three per-pass medians. Phase medians need not sum
exactly. The screening rule is total new/old >= 1.20 OR total increase >= 1 ms;
this is a reporting threshold, not a correctness gate or a production SLA.
It flags 45 of 85 combinations (18 Raw, 23 ordinary-index, 4 stats).
The matrix includes one backend change: JSON arithmetic uses Raw in the old
executor and stats in the new executor. Its speedup is excluded from claims about
same-backend framework overhead. Raw IN also moves lazy constant-set construction
from old Eval to new Prepare, so a Prepare-only ratio is not an end-to-end loss.

### Results requiring attention

Ordered by slowdown ratio; inspect the absolute increase before prioritizing.
`legacy_call` is the existing synthetic less-than callback; `legacy_starts_with`
uses the actual registered `StartsWithVarchar` implementation in both modes.

| Expression | Path / backend | Old ms | New ms | Ratio | Increase ms | ΔPrepare µs | ΔEval µs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy_starts_with | raw | 8.5259 | 36.9793 | 4.34× | 28.4535 | 17.55 | 28432.19 |
| eq | all_index/BITMAP | 0.0127 | 0.0362 | 2.84× | 0.0235 | 2.40 | 20.88 |
| ne | all_index/BITMAP | 0.0132 | 0.0368 | 2.79× | 0.0236 | 2.46 | 20.94 |
| in64 | all_index/STL_SORT | 0.0156 | 0.0416 | 2.66× | 0.0260 | 4.54 | 21.03 |
| in | all_index/BITMAP | 0.0148 | 0.0387 | 2.62× | 0.0240 | 2.55 | 21.03 |
| eq | all_index/STL_SORT | 0.0149 | 0.0387 | 2.60× | 0.0238 | 2.50 | 20.81 |
| compound | all_index/STL_SORT | 0.1063 | 0.2734 | 2.57× | 0.1671 | 11.34 | 154.95 |
| ne | all_index/STL_SORT | 0.0155 | 0.0397 | 2.56× | 0.0242 | 2.44 | 21.40 |
| unary | all_index/BITMAP | 0.0188 | 0.0434 | 2.31× | 0.0246 | 2.72 | 21.36 |
| range | all_index/BITMAP | 0.0196 | 0.0444 | 2.26× | 0.0248 | 3.03 | 21.28 |
| le | all_index/BITMAP | 0.0197 | 0.0445 | 2.26× | 0.0248 | 3.04 | 21.58 |
| compound | all_index/BITMAP | 0.1586 | 0.3310 | 2.09× | 0.1724 | 11.58 | 158.46 |
| array_all | raw | 4.1479 | 8.5312 | 2.06× | 4.3833 | 9.65 | 4371.48 |
| in | all_index/STL_SORT | 0.0241 | 0.0481 | 2.00× | 0.0240 | 2.79 | 20.30 |
| legacy_call | raw | 9.3501 | 17.8135 | 1.91× | 8.4633 | 13.45 | 8457.76 |
| in64 | all_index/BITMAP | 0.0287 | 0.0543 | 1.89× | 0.0256 | 4.58 | 20.84 |
| is_null | all_index/STL_SORT | 0.0134 | 0.0248 | 1.86× | 0.0115 | 1.96 | 9.10 |
| is_null | all_index/BITMAP | 0.0134 | 0.0248 | 1.85× | 0.0114 | 2.06 | 9.05 |
| array_any | raw | 4.1286 | 7.5496 | 1.83× | 3.4211 | 10.51 | 3403.30 |
| range | all_index/STL_SORT | 0.0327 | 0.0584 | 1.79× | 0.0257 | 3.16 | 21.79 |
| in1024 | all_index/STL_SORT | 0.0664 | 0.1173 | 1.77× | 0.0510 | 31.57 | 19.01 |
| unary | all_index/STL_SORT | 0.0339 | 0.0588 | 1.73× | 0.0249 | 2.88 | 21.50 |
| le | all_index/STL_SORT | 0.0360 | 0.0613 | 1.70× | 0.0253 | 2.94 | 21.85 |
| ge | all_index/STL_SORT | 0.0375 | 0.0620 | 1.65× | 0.0244 | 2.88 | 21.37 |
| gt | all_index/STL_SORT | 0.0400 | 0.0651 | 1.63× | 0.0251 | 2.79 | 21.67 |
| compound | raw | 1.0591 | 1.6978 | 1.60× | 0.6386 | 14.70 | 621.92 |
| array_length | raw | 1.9675 | 3.0475 | 1.55× | 1.0800 | 5.77 | 1072.31 |
| array_element | raw | 2.8124 | 4.3047 | 1.53× | 1.4923 | 8.67 | 1477.79 |
| json_range | json_stats | 2.1625 | 3.2375 | 1.50× | 1.0750 | 9.37 | 1067.69 |
| json_in | json_stats | 2.8760 | 4.1802 | 1.45× | 1.3042 | 10.42 | 1292.21 |
| json_unary | json_stats | 1.7600 | 2.5202 | 1.43× | 0.7602 | 8.81 | 748.72 |
| gt | all_index/BITMAP | 0.1094 | 0.1393 | 1.27× | 0.0299 | 4.26 | 24.58 |
| ge | all_index/BITMAP | 0.1103 | 0.1396 | 1.27× | 0.0293 | 4.33 | 24.45 |
| is_null | raw | 0.1274 | 0.1596 | 1.25× | 0.0321 | 2.67 | 28.74 |
| compare | raw | 0.3584 | 0.4394 | 1.23× | 0.0810 | 4.50 | 75.65 |
| gt | raw | 0.2293 | 0.2795 | 1.22× | 0.0503 | 5.86 | 43.22 |
| ge | raw | 0.2299 | 0.2801 | 1.22× | 0.0502 | 6.02 | 42.93 |
| unary | raw | 0.2287 | 0.2785 | 1.22× | 0.0498 | 6.07 | 43.21 |
| ne | raw | 0.2298 | 0.2785 | 1.21× | 0.0487 | 5.81 | 42.24 |
| eq | raw | 0.2300 | 0.2786 | 1.21× | 0.0486 | 5.81 | 42.49 |
| mixed_compare | raw | 0.4553 | 0.5492 | 1.21× | 0.0939 | 4.69 | 87.59 |
| float_in | raw | 0.2109 | 0.2544 | 1.21× | 0.0434 | 7.87 | 34.31 |
| le | raw | 0.2317 | 0.2793 | 1.21× | 0.0476 | 6.17 | 41.33 |
| json_contains_any | json_stats | 19.0119 | 21.8331 | 1.15× | 2.8212 | 26.65 | 2784.98 |
| json_arith | raw | 21.6639 | 23.0223 | 1.06× | 1.3583 | 26.79 | 1325.92 |

The largest absolute increase is the real `starts_with` call: its three pass
ratios are 4.31–4.34×. Next are the synthetic legacy call, ARRAY ALL/ANY and stats
contains-any. Scalar index EQ/NE ratios are high because old execution is only
about 13–16 µs; most single index predicates add about 20–22 µs in Eval. Indexed
compound conditions add roughly 155–158 µs in Eval. IN(1024) additionally adds
about 31 µs in Prepare. All of these index paths already execute once.

Raw integer comparisons add roughly 48–50 µs total (about 41–43 µs in Eval),
consistent with shared fixed/batch overhead; no CPU sample assigns this total to
individual framework functions. Raw division/modulus are about 1.17× in this run;
LIKE about 1.10×. Their smaller current differences must not be replaced by old
measurements from a different revision or machine.

### Source investigation and measured controls

1. **Legacy function adapter.** `ScalarCallExpr.cpp::MakeLegacyCall` detaches each
   argument with `MakeWritableCopy`, then snapshots the returned vector with
   `ReadOnlyVectorView::CopyOf`. `VectorAccess.cpp::CopyFlat` still uses per-row
   `IsValid` / `ValueAt`; broadcast constants become a full Flat writable column.
   In contrast, `ValueExpr.cpp` keeps a ConstantVector and `CallExpr.cpp` directly
   passes evaluated arguments to the same function. The new real-function test
   confirms this boundary is important, rather than generalizing the synthetic
   callback's result to every function.
2. **ARRAY membership.** Legacy ANY uses `ankerl::unordered_dense::set`; legacy
   ALL uses a dense value-to-bit map, uint64 state for <=64 terms, and a length
   precheck. `ScalarCollectionPredicates.cpp` uses a node-based unordered_map and
   clears/tests a byte-vector for ALL, with per-row type checks and row/element
   checkpoints. The code differences are confirmed; only checkpoint binding is
   separately measured here. The individual hash/validation shares are unmeasured.
   This fixture has non-NULL elements: optimizations must retain new element-NULL
   semantics rather than copy old unsupported behavior.
3. **ARRAY projection predicates.** New array-element and length filters produce
   a typed intermediate vector and then compare it; old `UnaryExpr.h` and
   `BinaryArithOpEvalRangeExpr.cpp::ExecArrayLength` write BOOL directly. This is
   a concrete fusion opportunity, not a measured isolated cost for the projection.
4. **Ordinary indexes.** `IntegerIndexReader::Query` returns matches and validity;
   `IndexResult::Full` checks and copies masks; `MapPredicate` intersects them into
   a batch BOOL; FilterBits copies TRUE and flips it. `IndexNode::FilterTrue` /
   `MapFilterTrue` exist but production FilterBits calls full `ExecutionBatch::Eval`.
   These extra passes explain what to profile/optimize; they are not evidence that
   one specific pass accounts for the entire 21 µs. A TRUE-only optimization must
   be confined to a compatible filter consumer, preserving projection/NOT/3VL.
5. **JSON stats.** WholeSegment scheduling is active. `QueryTypedStats` still
   computes optional results row by row, with `MatchStats` inspecting operation,
   protobuf bounds and numeric rules; collection stats use generic BSON term
   loops. Thus the difference includes adapter/kernel work, not just batch setup.
   Preserve cross-type and large-number comparison semantics when moving dispatch
   and constant handling outside the loop.

The checkpoint control constructs the same new FilterBitsExecution twice: the
normal checkCancellation callback versus an unbound callback. The callback uses
the same raw QueryContext capture as production and the query outlives execution.
This fixture has no OpContext, as in the main benchmark; it measures callback /
null-context checking overhead, not the cost of a live cancellation token.
Counts are collected in a separate untimed run. Removing cancellation is only a
diagnostic control and is not a production optimization.

| Expression | Checkpoint calls / query | Checked Eval ms | Unbound Eval ms | Difference ms |
| --- | ---: | ---: | ---: | ---: |
| array_any | 726696 | 7.7994 | 6.4561 | 1.3433 |
| array_all | 737073 | 8.7458 | 7.3154 | 1.4304 |
| array_element | 259731 | 4.2601 | 4.0941 | 0.1660 |
| array_length | 259731 | 2.9929 | 2.2582 | 0.7347 |
| string_like | 259699 | 4.0033 | 3.6394 | 0.3639 |
| legacy_call | 226 | 18.3813 | 18.3726 | 0.0087 |
| legacy_starts_with | 258 | 39.1631 | 39.1761 | -0.0130 |

Legacy adapter stage control: 8,192 rows with the same short strings and NULL
pattern, three passes of 31 rotated rounds/four repeats. Inputs, function and
output are validated outside timers. This uses real MakeWritableCopy, CopyOf and
StartsWithVarchar, with no Reader or planner. The stage values describe the new
adapter's own work; they are **not** each stage's net increase over the old path,
and must not be subtracted from a full-operator timing from another cloud run.

| Stage | Median µs / 8,192 rows |
| --- | ---: |
| input_copy | 629.197 |
| constant_expand | 585.124 |
| function | 70.027 |
| result_copy | 31.283 |
| adapter_total | 1311.555 |

The input copy and broadcast-constant expansion are the dominant stages in
this control (629 and 585 µs), versus 70 µs for the actual function and 31 µs
for result copying; the integrated adapter is 1,312 µs. The adapter still pays
per-row generic vector access during copies even though scalar predicate kernels
were converted to span-based access. This identifies a remaining shared hot path.
Checkpoint removal does not measurably improve the legacy function control.

The useful next steps are shared bulk copying/constant preservation at the
legacy adapter boundary, ARRAY membership kernel/state and bounded cancellation
checks, then filter-consumer mask ownership and stats dispatch. Argument isolation,
result immutability, element NULL and cancellation responsiveness are requirements;
blindly deleting copies/checks is not a valid optimization.

### Coverage limits and reproducibility

This is not every migrated feature: Text/Phrase/Fuzzy, GIS/cache/RTree refinement,
MATCH/nested indexes, ARRAY/JSON-path index backends, Growing, candidate offsets /
IterativeFilter / Boost, OR/NOT and additional string/arithmetic/type distributions
still lack this same paired matrix. Existing correctness migration tests do not
establish their performance. Raw bool IN, large IN, string IN/range and mixed
Raw/index conjunctions show no large regression in this run; see the full CSV.

Opt in with `Paths/ScalarExecutionPerformanceTest.RegressionAuditMatrix/*`.
Diagnostics use `Paths/ScalarExecutionPerformanceTest.CheckpointDiagnostic/Raw`
and `ScalarRegressionDiagnostics.LegacyAdapterStages`. The matrix fixes both
ordinary index implementations and rejects a conflicting index-type environment
override. Its Raw shapes share `PerformanceShapes()` with the measurement whitelist.

Artifacts: `/mnt/dev_debug/scalar-regression-audit-20261114/`:
`operator-summary1.csv` / `operator-summary1.json` contain all 85 combinations,
including per-pass ratio ranges and per-phase medians. `operator-reports1.json`
retains all samples. `checkpoint-summary2.json`, `adapter-summary2.json`, both
logs/statuses and `submitted1-hashes.json` / `submitted2-hashes.json` distinguish
the main matrix from the final targeted diagnostics. Production source hashes
match the previously validated implementation; only this test module and report
changed in this audit.

Main cloud task: `cpp-ut-local-lu-zhang-head-cab9d65-5aa0e576-u59677`,
created `2026-11-14T12:26:18.297574Z` (10 tests passed).
Targeted diagnostic task: `cpp-ut-local-lu-zhang-head-cab9d65-9992e3b3-u59677`,
created `2026-11-14T12:40:46.943243Z` (2 tests passed).

## Index mask ownership and logical accumulation (2026-11-14)

Follow-up to the regression audit above. The BITMAP slowdown is largely shared
framework work after the backend returns: the former full-result path allocated
a temporary difference bitmap to validate TRUE/validity, copied and normalized
both masks into IndexResult, copied them again when mapping a Range, and applied
validity again while emitting FilterBits. WholeSegment was already active in
both executors; this was not a batching or index-selection regression. The old
FilterBits WholeSegment path converts its mutable result in place, so retaining
an immutable cached result has a real output-copy/ownership cost to consider.

Implemented:

- Transfer locally owned scalar/ARRAY/stats result masks into IndexResult; borrowed-mask inputs still detach.
- Validate TRUE subset validity with allocation-free word reduction. Share immutable masks for Range BOOL views, retaining the IndexResult owner. Offsets still gather in request order, including duplicates and NULLs.
- Copy canonical TRUE directly into FilterBits output. BOOL mask consumers clip each borrowed mask once rather than rebuilding and re-clipping validity.
- Accumulate SQL AND/OR with TRUE and FALSE sets, consuming detached child validity as its FALSE set. Reuse accumulated FALSE storage for final validity. Child order and decisive-row short circuit are unchanged; NULL belongs to neither set.

### Measurement and validation

Cloud C++ UT through milvus-dev-cli, baseline master, dirty curated source.
The final run passed **219 tests** and produced **255 paired operator records**
(85 combinations, three passes), plus three isolated bitmap-stage diagnostic records.
Each operator pass uses five warmups and 31 alternating old/new measurements;
262,144 nullable rows, Raw batch **8,192**, warm storage, result cache disabled.
Index/stats WholeSegment paths are checked as one genuine physical execution.
Route and full exclusion-bitmap equality checks remain outside timing. Raw and
stats were rerun because BOOL view handling and AND/OR are shared framework code.

Test coverage includes unaligned ranges, sparse coverage, NULL truth tables,
27 three-input combinations across bitmap words with exact child-request checks,
NOT/OR, repeated and backward offsets, owner release and views surviving sources,
copy/publication/dictionary/slice/merge consumers, invalid masks, failed/canceled
queries and cache publication. No service E2E, sanitizer, cold-storage or concurrent
production-throughput claim is made.

All times below are medians of three per-pass medians, in **microseconds**.
Old and final-new columns are paired within the final run. Prior ratios are from
the preceding audit, each with its own old control; they are not a same-process
before/after experiment. Phase medians need not add to total medians.

### Index results

| Backend | Expression | Old total µs | New total µs | Prior new/old | Final new/old | ΔPrepare µs | ΔEval µs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BITMAP | eq | 12.276 | 23.905 | 2.84× | 1.95× | 2.479 | 8.822 |
| BITMAP | ne | 12.634 | 24.245 | 2.79× | 1.92× | 2.438 | 8.943 |
| BITMAP | unary | 17.662 | 29.739 | 2.31× | 1.68× | 2.689 | 9.061 |
| BITMAP | range | 18.950 | 31.433 | 2.26× | 1.66× | 2.964 | 9.154 |
| BITMAP | in | 14.406 | 26.113 | 2.62× | 1.81× | 2.573 | 8.910 |
| BITMAP | in64 | 28.480 | 42.002 | 1.89× | 1.47× | 4.534 | 8.724 |
| BITMAP | in1024 | 273.140 | 309.330 | 1.19× | 1.13× | 30.501 | 5.799 |
| BITMAP | is_null | 13.156 | 18.659 | 1.85× | 1.42× | 1.970 | 3.288 |
| BITMAP | compound | 173.298 | 252.180 | 2.09× | 1.46× | 12.496 | 64.772 |
| STL_SORT | eq | 14.435 | 26.447 | 2.60× | 1.83× | 2.521 | 9.129 |
| STL_SORT | ne | 14.903 | 26.529 | 2.56× | 1.78× | 2.433 | 8.715 |
| STL_SORT | unary | 33.340 | 45.977 | 1.73× | 1.38× | 2.833 | 9.304 |
| STL_SORT | range | 32.100 | 44.659 | 1.79× | 1.39× | 3.112 | 9.125 |
| STL_SORT | in | 24.905 | 36.578 | 2.00× | 1.47× | 2.579 | 8.767 |
| STL_SORT | in64 | 15.270 | 28.704 | 2.66× | 1.88× | 4.410 | 8.613 |
| STL_SORT | in1024 | 66.873 | 105.086 | 1.77× | 1.57× | 30.816 | 6.496 |
| STL_SORT | is_null | 13.062 | 18.999 | 1.86× | 1.45× | 1.935 | 3.554 |
| STL_SORT | compound | 105.037 | 181.166 | 2.57× | 1.72× | 11.525 | 62.165 |

`compound` is `a < 10 && b > 10 && a != 8`. These are real index queries,
not Raw evaluation with an artificial segment-sized batch. Both BITMAP and
STL_SORT benefit because their results use the same ownership/mapping layer.

### Isolated result-processing control

The same immutable 262,144-row nullable result is consumed by both alternatives.
No backend query or Prepare time is included. Three passes, five warmups, 31
rotating rounds, 16 repetitions per sample; output/proof equality is checked after
each timed sample. These stages must not be added to end-to-end measurements.

| Stage | Copy/materialize µs | Direct/share µs |
| --- | ---: | ---: |
| TRUE subset validity validation | 3.894 | 1.212 |
| Range BOOL mapping + CopyTrueTo | 9.734 | 2.045 |

### Shared-path controls

| Path | Expression | Old total µs | New total µs | Prior new/old | Final new/old |
| --- | --- | ---: | ---: | ---: | ---: |
| raw/raw | compound | 1028.998 | 1673.653 | 1.60× | 1.63× |
| raw/raw | eq | 230.817 | 275.688 | 1.21× | 1.19× |
| raw/raw | in | 246.640 | 285.028 | 1.16× | 1.16× |
| raw/raw | string_eq | 1248.984 | 1192.301 | 0.89× | 0.95× |
| raw/raw | bool_in | 1059.225 | 401.778 | 0.35× | 0.38× |
| b_index/BITMAP | compound | 790.311 | 642.345 | 0.87× | 0.81× |
| b_index/STL_SORT | compound | 708.631 | 555.609 | 0.85× | 0.78× |
| json_stats/JSON_STATS | json_unary | 1751.882 | 2489.339 | 1.43× | 1.42× |

The broader controls do not establish blanket non-regression. Cross-run shifts
also occurred outside the index path:

| Raw expression | Prior old/new µs | Final old/new µs | Prior/final ratio |
| --- | ---: | ---: | ---: |
| json_contains_all | 28568.2 / 26026.4 | 26356.4 / 27669.1 | 0.91× / 1.05× |
| string_prefix | 1573.1 / 1356.3 | 1423.3 / 1414.8 | 0.86× / 0.99× |
| string_range | 3110.2 / 2476.1 | 2847.7 / 2503.7 | 0.80× / 0.88× |

Both old and new timings moved. The before/after runs do not isolate whether
these shifts come from this change or environment/cache variation; they remain
follow-up controls, rather than evidence that every Raw path is unaffected.

### Remaining overhead and artifacts

Single-index equality still has query-local Prepare/cache/view/output costs;
we have reduced the avoidable mask work, not established parity with the old
executor. Large IN still has a separate typed-constant/Prepare gap (especially
visible with STL_SORT); this change does not resolve it. Logical accumulation
still materializes child TRUE/validity masks. The precise share of the residual
cost among those stages requires further measurement.

Artifacts: `/mnt/dev_debug/scalar-index-overhead-20261114/`, including submitted
source hashes, full logs, per-sample JSON, operator summaries, diagnostics and
source/ownership audit. The final hashes include the report update.

- First ownership-only run: `cpp-ut-local-lu-zhang-head-cab9d65-01e3ec6c-u59677` (157 tests; 78 operator records).
- Final run: `cpp-ut-local-lu-zhang-head-cab9d65-e4dd4d5a-u59677`.


## Legacy function adaptation and ARRAY work (2026-11-14)

### Scope and findings

The next audit prioritized absolute excess time as well as new/old ratios.
The preceding run had `legacy_starts_with` at 37.037 ms versus 8.052 ms
(4.60×), `legacy_call` at 17.945 ms versus 9.618 ms (1.87×), and ARRAY
ANY/ALL at 1.90–2.02×. These are operator measurements, including Prepare,
Eval and cleanup; the legacy-call case is a synthetic registered numeric callback,
whereas starts-with invokes the real `StartsWithVarchar` implementation.

The legacy adapter expanded constant arguments into one mutable value per row,
resolved ordinary inputs through per-row view accessors, and snapshotted returned
BOOL data bit by bit. The old function executor could pass a ConstantVector.
The new path now preserves a separately owned ConstantVector, copies primitive
and string spans by valid selected ranges, and snapshots BOOL masks with bulk
bitmap operations. Sparse arguments still gather exactly the requested rows;
opaque callbacks still execute once. Mutable/retained arguments and returned
aliases remain detached from input views and published outputs.

ARRAY membership now uses a dense hash map and a 64-bit found mask for at most
64 distinct targets, falling back to a reusable word vector for larger ALL sets.
Duplicate targets and repeated array values do not change membership semantics.
Cheap ARRAY length/numeric-element work and membership share a bounded cancellation
budget: poll initially and within every 256 row/element work items, including
inside one large array. JSON parsing, string element copying and array equality
retain their previous polling boundaries. This reduces callback overhead without
removing cancellation checks; the bound is work items, not wall-clock time.

### Paired operator results

Cloud C++ UT via milvus-dev-cli, master baseline, curated dirty source, warm
storage, result cache disabled, 262,144 nullable rows. Raw batch size is **8,192**
(32 batches), never artificially enlarged to segment size. Index/stats use their
actual WholeSegment plan. Each of three passes has five warmups and 31 alternating
old/new samples; route and full exclusion-bitmap equality checks are outside timing.

Times are medians of the three pass medians, in **milliseconds**. Old/new columns
are paired in the final run. Prior ratios come from the preceding audit with its
own old control; cross-run changes are not a controlled same-process before/after.

| Raw expression | Old ms | New ms | Prior new/old | Final new/old |
| --- | ---: | ---: | ---: | ---: |
| legacy_starts_with | 8.348 | 4.959 | 4.60× | 0.59× |
| legacy_call | 9.247 | 7.298 | 1.87× | 0.79× |
| array_any | 4.054 | 6.452 | 1.90× | 1.59× |
| array_all | 4.238 | 6.731 | 2.02× | 1.59× |
| array_element | 2.776 | 4.166 | 1.53× | 1.50× |
| array_length | 1.942 | 2.713 | 1.44× | 1.40× |

The two legacy-function regressions are eliminated in this matrix. ARRAY improved,
but remains slower than the old executor; the change does not establish ARRAY parity.

### Isolated adapter and cancellation diagnostics

The adapter diagnostic uses prebuilt warm 8,192-row inputs and the real starts-with
function, excludes Reader/Prepare, and checks results outside timing. Three passes,
five warmups, 31 rotating rounds, four repetitions per sample. These component
measurements must not be added to operator totals.

| Adapter stage | Median µs |
| --- | ---: |
| input_copy | 51.322 |
| constant_expand | 43.267 |
| constant_preserve | 0.177 |
| function | 85.236 |
| result_copy | 0.367 |
| adapter_total | 135.738 |

`constant_expand` is a diagnostic control using the generic writable-copy API;
`constant_preserve` is the actual legacy-adapter path. The snapshot of returned
BOOL output is retained. The diagnostic includes allocation/replacement costs,
not just kernel arithmetic.

The cancellation diagnostic compares the new executor with its callback bound
versus unbound, both without an operation context. It is a callback-cost control,
not a production cancellation-latency estimate. Final callback counts are measured
across all 262,144 rows; prior counts used the same fixture.

| ARRAY shape | Prior callback count | Final callback count | Final checked − unbound Eval µs |
| --- | ---: | ---: | ---: |
| array_any | 726,696 | 3,106 | -10.599 |
| array_all | 737,073 | 3,138 | 12.544 |
| array_element | 259,731 | 1,314 | 8.921 |
| array_length | 259,731 | 1,314 | 3.336 |

Small negative/positive residual timing differences are measurement noise, not
negative callback cost. ARRAY's remaining gap is not explained by the now-small
bound-versus-unbound callback difference; Reader/view handling, projection and
kernel execution need separate attribution before choosing the next change.

### Remaining regressions and controls

The full matrix contains **85 combinations / 255 paired reports**, covering Raw,
STL_SORT, BITMAP, mixed index/Raw and JSON stats. Representative remaining gaps:

| Expression | Path | Old ms | New ms | New/old | Excess ms |
| --- | --- | ---: | ---: | ---: | ---: |
| json_exists | raw/raw | 20.111 | 23.210 | 1.15× | 3.099 |
| json_contains_any | json_stats/JSON_STATS | 18.998 | 21.693 | 1.14× | 2.695 |
| array_all | raw/raw | 4.238 | 6.731 | 1.59× | 2.493 |
| array_any | raw/raw | 4.054 | 6.452 | 1.59× | 2.398 |
| json_arith | raw/raw | 21.140 | 22.976 | 1.09× | 1.835 |
| array_element | raw/raw | 2.776 | 4.166 | 1.50× | 1.390 |
| json_in | json_stats/JSON_STATS | 2.845 | 4.137 | 1.45× | 1.292 |
| json_range | json_stats/JSON_STATS | 2.143 | 3.230 | 1.51× | 1.087 |
| array_length | raw/raw | 1.942 | 2.713 | 1.40× | 0.771 |
| json_unary | json_stats/JSON_STATS | 1.774 | 2.492 | 1.40× | 0.718 |
| compound | raw/raw | 1.093 | 1.703 | 1.56× | 0.610 |
| compound | all_index/BITMAP | 0.173 | 0.255 | 1.47× | 0.081 |
| eq | all_index/BITMAP | 0.013 | 0.024 | 1.84× | 0.011 |
| in1024 | all_index/STL_SORT | 0.066 | 0.104 | 1.58× | 0.038 |

Single-index ratios remain large because the old baseline is only tens of
microseconds; absolute gaps must be considered alongside the ratios. JSON stats
numeric predicates still have Eval overhead, while large indexed IN retains a
Prepare gap. Neither is claimed fixed by this change.

Raw JSON EXISTS/arithmetic and string equality also shifted relative to the prior
run. These paths were not directly changed, but timing alone cannot rule out
shared-code layout/cache effects; do not claim blanket non-regression:

| Raw control | Prior new/old | Final old ms | Final new ms | Final new/old |
| --- | ---: | ---: | ---: | ---: |
| json_exists | 1.03× | 20.111 | 23.210 | 1.15× |
| json_arith | 1.05× | 21.140 | 22.976 | 1.09× |
| string_eq | 0.95× | 1.155 | 1.223 | 1.06× |
| json_contains_any | 1.09× | 26.783 | 26.782 | 1.00× |
| json_contains_all | 1.05× | 26.938 | 25.876 | 0.96× |
| string_prefix | 0.99× | 1.567 | 1.354 | 0.86× |
| in | 1.16× | 0.262 | 0.304 | 1.16× |
| in1024 | 0.53× | 1.803 | 0.955 | 0.53× |

### Correctness validation and artifacts

Final cloud run passed **272 tests**. Coverage includes borrowed/owned strings,
unaligned slices, segmented gaps, NULL spans with no payload address, dictionary
repeats, constant arguments under full/sparse selection, callback mutation and
retained output aliases. ARRAY coverage includes empty/duplicate terms, 63/64/65/129
unique targets, parent/element NULLs, typed sealed/growing arrays, sparse/dictionary
inputs, and cancellation inside a long row and across short rows followed by retry.
Existing loader/function failure propagation, publication and candidate-filter
checks are included. No sanitizer, service E2E, cold-storage or concurrent
production-throughput result is claimed.

- Adapter-only run: `cpp-ut-local-lu-zhang-head-cab9d65-0e250b74-u59677`, 143 tests passed.
- Final run: `cpp-ut-local-lu-zhang-head-cab9d65-492d6b0b-u59677`, succeeded, 272 tests passed.
- Artifacts: `/mnt/dev_debug/scalar-adapter-array-opt-20261114/`, including full
  logs, submitted source hashes, all samples, operator CSV/JSON summaries,
  adapter/cancellation diagnostics and source/ownership audit.

The design document's section 3.2 describes constant-preserving legacy arguments,
batched writable copies and the ARRAY cancellation budget. Production code and
tests match the final submitted snapshot; the performance report is added afterward.


## Small ARRAY membership and stage attribution (2026-11-16)

### Scope and diagnosis

Added an opt-in `ArrayStageDiagnostic` to the real Sealed ARRAY fixture. It
explicitly preloads the one field slot, then measures cached expression Eval,
workspace reset and bitmap output separately for each normal 8,192-row batch.
All 262,144 output bits are checked against the independent expected result.
Preparation is outside these diagnostic timers. Instrumented stages are not
operator latency and must not be added to the paired operator measurements.

The initial cloud diagnostic identified cached ARRAY ANY/ALL evaluation as the
largest stage (about 6.8 ms), versus 1.5–1.6 ms loading and about 0.01 ms output.
The change therefore targets membership computation, leaving Reader and view
protocols unchanged:

- At most four distinct numeric/BOOL targets use direct equality comparisons;
  larger sets and strings retain the existing hash lookup. Dispatch is outside
  the row loop. Both paths use the same converted constants and unique target IDs.
- ALL rejects a valid array when its logical element count is smaller than the
  distinct target count. Parent NULL still yields NULL. Payload-type validation
  and the row checkpoint occur before this early return.
- Element validity, duplicate handling, FLOAT contains versus IN-field precision,
  the 64-target mask boundary, large-set scratch reuse and cancellation remain
  governed by the existing contracts.

### Paired operator results

Final Dev CLI cloud C++ UT, master baseline, dirty curated source: **102 tests
passed**, **85 combinations / 255 paired reports**. Warm storage, result cache
disabled, 262,144 nullable rows, Raw batch **8,192**, five warmups and 31
alternating old/new samples per pass, three passes. Full output equality and
actual route checks are outside timing. The ARRAY timing fixture uses INT64
arrays with zero or two elements, two distinct ALL targets and four ANY targets.

Times are medians of three per-pass medians, in **milliseconds**. Old/new columns
are paired in this final run. Prior ratios refer to the 2026-11-14 run with its
own old control; they are cross-run references, not controlled before/after gains.

| Raw expression | Old ms | New ms | Prior new/old | Final new/old |
| --- | ---: | ---: | ---: | ---: |
| array_any | 5.121 | 7.305 | 1.59× | 1.43× |
| array_all | 6.032 | 7.291 | 1.59× | 1.21× |
| array_element | 3.698 | 5.162 | 1.50× | 1.40× |
| array_length | 2.586 | 3.389 | 1.40× | 1.31× |
| array_eq | 4.811 | 5.268 | 1.12× | 1.09× |

ANY ratios across the three passes were 1.425–1.439; ALL ratios were
1.209–1.227. The remaining absolute excess is about 2.18 ms for ANY and
1.26 ms for ALL. Neither path has reached parity with the old executor.
Unchanged element/length/equality controls also shifted across runs, so the
ratio changes cannot be attributed entirely to the lookup change.

### Stage diagnostics

Each entry below is a median of three per-pass medians in **milliseconds**.
Before and after are separate cloud runs. These diagnose where time is spent;
they do not establish an isolated same-process speedup.

| Expression | Before load | Final load | Before cached Eval | Final cached Eval |
| --- | ---: | ---: | ---: | ---: |
| array_any | 1.620 | 1.763 | 6.753 | 5.340 |
| array_all | 1.538 | 1.763 | 6.814 | 5.239 |
| array_element | 1.569 | 1.800 | 3.451 | 3.184 |
| array_length | 1.989 | 1.773 | 1.955 | 1.499 |
| array_eq | 1.514 | 1.739 | 3.370 | 3.323 |

Final bitmap output remains about 0.011–0.014 ms. ARRAY membership computation
is still the largest stage; element projection and length computation also
retain costs. The cancellation diagnostic recorded 3,106 ANY and 3,138 ALL
callbacks, unchanged for this fixture. Checked-versus-unbound differences were
small and changed sign; they are noise-sensitive controls, not cancellation
latency measurements. Cancellation is still checked within a long array.

### Other paths and remaining work

These selected controls were rerun in the same final matrix. This change does
not claim to fix their residual costs or establish blanket non-regression.

| Expression | Path | Old ms | New ms | New/old |
| --- | --- | ---: | ---: | ---: |
| compound | raw/raw | 1.435 | 1.908 | 1.33× |
| eq | all_index/BITMAP | 0.015 | 0.027 | 1.81× |
| compound | all_index/STL_SORT | 0.121 | 0.212 | 1.76× |
| json_unary | json_stats/JSON_STATS | 1.801 | 2.683 | 1.49× |
| json_in | json_stats/JSON_STATS | 3.086 | 4.579 | 1.48× |
| json_range | json_stats/JSON_STATS | 2.460 | 3.591 | 1.46× |
| json_exists | raw/raw | 23.085 | 24.060 | 1.04× |
| legacy_starts_with | raw/raw | 8.858 | 5.637 | 0.64× |

Next work remains ARRAY computation/projection, JSON stats Eval and general
Prepare/bitmap costs. Non-INT64, long-array and large-target membership
performance has not been established by this fixture. Correctness coverage
for those shapes does not prove performance parity.

### Correctness and artifacts

The final run includes numeric/string/BOOL ARRAY operations on Growing and
Sealed, sparse and dictionary inputs, duplicate targets/elements, parent and
element NULL, empty sets, 0/1/2/3/4/5/63/64/65/129 target boundaries, and FLOAT/
DOUBLE NaN, infinities and signed zero. Existing tests exercise cancellation
inside a large row and across short rows, then successful retry. The source
audit traces both compiler factory sites, typed constant construction, valid-row
selection, target IDs, the length proof and exception/publication behavior.
No new error code or error translation was introduced.

This is cloud C++ UT and native operator measurement, not PR Jenkins CI or a
service E2E run. No service E2E, ASAN, cold-storage or concurrent production
throughput result is claimed. The production feature switch stays disabled by
default.

Artifacts: `/mnt/dev_debug/scalar-array-stages-20261116/`, including source
hashes, before snapshots, turn-only patch, full logs, 255 operator reports,
15 stage reports per diagnostic run, 21 cancellation reports and source audit.

- Baseline stage task: `cpp-ut-local-lu-zhang-head-cab9d65-bf8c001a-u59677`, succeeded.
- Final task: `cpp-ut-local-lu-zhang-head-cab9d65-aadd6b10-u59677`, succeeded,
  created **2026-11-16T08:47:24.953516Z**. The ID was reused on resubmission;
  `log3.json`/`status3.json` identify the successful run. `log2.json` preserves
  an earlier canceled attempt with two test-only missing-template-keyword
  compilation errors, both corrected before the final submission.

Production code and tests match `submitted3-hashes.json`; only the performance
report and the design document's validation paragraph were updated afterward.

## ARRAY cost attribution and selection controls (2026-11-16)

This investigation changes only the opt-in `ArrayCostDiagnostic` and this
record; it does not change production execution or remove correctness checks.
The fixture is the same 262,144-row Sealed INT64 ARRAY input, batch 8,192,
with empty or two-element arrays. ANY uses four targets; ALL uses two.
Compare both non-nullable input and parent NULL every 97 rows. Every group has
three passes, five warmup rounds and 31 measured rounds per pass. Values below
are medians of the three pass medians, in milliseconds; no timing threshold is
asserted.

The final Dev CLI cloud C++ run is
`cpp-ut-local-lu-zhang-head-cab9d65-7383699b-u59677`, created at
`2026-11-16T13:00:51.053153Z`. It runs the Raw correctness/routing test and the
new diagnostic: both tests passed, with 12 diagnostic records and all truth/
validity checks passing. This is cloud C++ validation, not Jenkins CI or service E2E.
The earlier attempt with the same ID at `12:53:59.937265Z` was cancelled after
a diagnostic-only compile error (taking a borrowed view of a temporary owned
selection); the final source stores the owner before borrowing its view.

### Real operator totals and input shape

| Expression | Parent NULL | Old operator | New operator | New / old | Valid ranges | Adaptive Ranges / Sparse spans |
|---|---|---:|---:|---:|---:|---:|
| array_any | False | 4.982 | 6.010 | 1.206 | 32 | 32 / 0 |
| array_any | True | 5.172 | 7.255 | 1.403 | 2734 | 32 / 0 |
| array_all | False | 5.486 | 5.735 | 1.045 | 32 | 32 / 0 |
| array_all | True | 5.903 | 7.084 | 1.200 | 2734 | 32 / 0 |

### Instrumented stages

The legacy scan invokes the real public `ProcessDataChunks<ArrayView>` with a
test copy of the fixture's typed small-set ANY/ALL callback. Callback time is
measured within the scan, and subtracted from that same invocation to obtain
`scan other`. This residual includes physical Scan, descriptor construction,
pinning and cursor work. It is not a direct instrumentation of production old
Eval. The callback is specialized for this fixture; ALL also omits per-batch
matcher construction. Compare `legacy scan total` with the real old operator
before interpreting this approximation.

New stages time the real reusable execution batch: Reset, GetColumn, Eval with
the field already cached, and output bitmap copy/flip. Cached Eval includes
framework work and kernel execution. These instrumented phases must not be
subtracted from the separately measured operator totals as exact accounting.

| Expression | NULL | Legacy scan total | Legacy callback | Legacy scan other | New load | New cached Eval | New reset | New output |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| array_any | False | 4.398 | 2.752 | 1.664 | 1.727 | 4.258 | 0.052 | 0.014 |
| array_any | True | 4.764 | 2.984 | 1.779 | 1.904 | 5.499 | 0.056 | 0.013 |
| array_all | False | 5.226 | 3.591 | 1.651 | 1.708 | 4.057 | 0.049 | 0.015 |
| array_all | True | 5.468 | 3.718 | 1.746 | 1.857 | 5.289 | 0.050 | 0.015 |

### Same-input computation controls

All controls consume the exact new Reader view, retain the input owner, and
check the complete result outside timers. Both truth and validity are consumed
and validated. The fixture's INT64 payload type and absence of element NULLs
are verified before invoking unchecked controls. Their costs include decoding;
Object controls also include the real bitmap builder and result publication.

- `direct hash`: old-style row loop and hash lookup over decoded spans.
- `object hash`: the same hash probe through production ObjectInput traversal.
- `tiny`: ObjectInput with small-target direct equality.
- `checked`: tiny plus payload type compatibility, element validity and the
  same bounded row/element cancellation budget as the production kernel.
- `typed`: unchecked tiny with direct INT64 payload access, bypassing the
  generic accessor's bounds/type dispatch; this is a prevalidated lower-cost
  control, not a safe replacement for arbitrary inputs.
- `checked ranges` / `checked sparse`: same checked test kernel and decoded
  traversal, with only the selection preference changed. Both preserve NULL
  and cancellation checks. Dense is not enabled.

| Expression | NULL | Direct hash | Object hash | Tiny | Checked | Typed | Checked ranges | Checked sparse |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| array_any | False | 3.637 | 2.480 | 3.378 | 3.647 | 1.450 | 3.989 | 3.910 |
| array_any | True | 4.151 | 3.851 | 3.463 | 4.700 | 1.707 | 4.889 | 3.919 |
| array_all | False | 4.008 | 2.842 | 1.715 | 3.195 | 1.002 | 3.126 | 3.069 |
| array_all | True | 3.829 | 3.959 | 2.919 | 4.204 | 1.204 | 4.117 | 3.048 |

Controls fix ANY/ALL at template instantiation and are compiled separately from
the production Contains implementation. Differences include inlining, code
layout and cache effects, not only the cost of individual source branches.
Forced-mode results are diagnostic headroom, not an achieved production
speedup. No CPU sampling or assembly inspection was performed. The legacy
scan approximation and these controls do not justify assigning every
microsecond of the real operator gap to a specific instruction.

### Interpretation and next optimization target

- Most of the roughly 1.7--1.9 ms input loading is shared work. The legacy scan
  residual is 1.65--1.78 ms; new loading is about 0.06--0.13 ms higher. Both
  paths construct ArrayView descriptors in `ArrayChunk::Views`. It is incorrect
  to count the entire new load phase as additional framework overhead.
- Parent NULLs turn 32 all-valid ranges into 2,734 ranges. The real operator
  gap grows from 1.027 to 2.084 ms for ANY, and from 0.249 to 1.181 ms for ALL.
  Adaptive selection chooses Ranges for all 32 spans in every case.
- With the same checked test kernel, forcing Sparse instead of Ranges saves
  0.970 ms for nullable ANY and 1.068 ms for nullable ALL (about 20% and 26% of
  the forced-Ranges control). Without NULLs the differences are only 0.079 and
  0.057 ms. This identifies traversal of fragmented valid ranges as a concrete
  optimization target. The shared ObjectCallback cost estimate is not well
  calibrated for these short ARRAY controls. Production-mode A/B validation is
  still needed before claiming the same savings in the production executor.
- The unchecked tiny controls fall from 3.463 to 1.707 ms (ANY), and 2.919 to
  1.204 ms (ALL), when using a prevalidated direct INT64 load. This establishes
  headroom in generic payload access/bounds/type dispatch and its generated
  loop. Those accesses also exist in the old path; this is an optimization
  opportunity, not an exclusive explanation of the old/new gap. Any real
  specialization must obtain type/validity guarantees from the input source
  and retain a checked fallback.
- The adaptive checked controls add roughly 1.24/1.28 ms over the unchecked
  tiny controls in the nullable fixture. This combines payload validation,
  element-null checks, cancellation budgets and their code-generation effects;
  it does not isolate each check's cost. The real cached Eval is still about
  0.80/1.09 ms above these checked controls. That residual includes the actual
  generic kernel and Eval machinery, and remains unattributed at instruction
  level. Do not add these control differences as an exact operator breakdown.

The first exploration's legacy callback timings changed noticeably after
adding more control instantiations, despite unchanged production code. The
final legacy scan totals are about 5--12% below real old operator totals. This
reinforces why the callback split is approximate and why the real operator
and paired forced-mode controls are the stronger evidence here.

Priority is to calibrate ARRAY traversal using a production-kernel A/B, then
specialize typed payload access under explicit input guarantees. The evidence
does not justify removing correctness checks or globally retuning JSON and
all object kernels from these two short-array cases.

Artifacts (source manifests, complete samples, summaries, status and logs):
`/mnt/dev_debug/scalar-array-attribution-20261116/`. The first successful
exploration is `cpp-ut-local-lu-zhang-head-cab9d65-e9a40229-u59677`; its data
remain separate from the final controls. Do not use cross-run absolute deltas
as an optimization result. Long arrays, other element types, large target sets,
cold storage, production concurrency and service E2E are outside this run.
