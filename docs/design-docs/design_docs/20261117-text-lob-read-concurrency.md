# TEXT LOB reads using the upstream asynchronous API

Updated: 2026-11-23. Target: Milvus master. Rollout is disabled by default.

## Dependency and scope

Use milvus-storage [PR #810](https://github.com/milvus-io/milvus-storage/pull/810),
commit `07f135ea23b6baa921227a9c591746d59649cabe`. It provides concurrent
`LobColumnReader` calls and `ReadBatchAsync`, which returns owned strings and
fans out native Vortex reads across every file in a batch. The caller chooses
the Folly executor and controls admission. Milvus pins this upstream commit;
there is no companion storage patch or replacement lifecycle implementation.
The dependency advance also includes the preceding upstream generic-reader
cancellation API (#803), Take routing (#793), Talon lifetime and filesystem fixes.
Milvus call sites remain source-compatible through default arguments; test
reader overrides adopt the new signatures. Passing request tokens into all
non-LOB storage APIs is outside this change.

The previous prototype implemented concurrency inside storage's synchronous
API. This design supersedes that prototype. Its benchmark and sanitizer results
are not evidence for the current implementation.

The change affects TEXT reads in growing/sealed segments, grep, retrieve/search
output, and index feeds that use `TextColumnCache`. It does not alter storage
paths, file formats, manifests, writes or replication state. Settings and
scheduling are local to each process.

## Milvus integration

The public `TextColumnCache::ReadBatch` and `ReadBatchInto` interfaces remain
synchronous. They accept an optional cancellation token and return a complete
batch or throw. `ReadBatchInto` moves owned strings into its pre-sized output.

1. Decode inline references and preserve null/empty entries. Validate every
   non-inline reference before admitting file reads.
2. Group references by file UUID. Sort/deduplicate file-local row offsets and
   retain every original output position.
3. Acquire a process-wide batch permit. Inline-only batches bypass the permit.
4. Submit at most K single-file `ReadBatchAsync` calls. Splitting at file
   boundaries bounds upstream fan-out, including deferred cold opens.
5. Consume any ready file, verify its result count, map strings back to original
   positions, and refill the window. A slow first file does not block
   replenishment of completed slots. Duplicated references reproduce their
   values at every requested position.
6. Release the permit only after all admitted futures finish and their results
   have been consumed or discarded.

A dedicated four-thread Folly executor runs deferred file opens and result
continuations. Native scans execute on the storage Rust Tokio runtime. Query
and load workers may synchronously wait on the cache, so their own executors
must not also run these continuations. The dedicated workers never wait for
sibling futures. The caller polls ready slots at a 1 ms interval; this adds up
to roughly one polling interval to completion discovery.

K counts admitted file operations, including completed but unconsumed results;
it is not a count of GET requests. B active batches imply at most B × K admitted
file operations through this cache. Neither limit bounds decoded bytes or RSS:
one file can contain a large selection, output strings survive file completion,
and waiting requests also retain their inputs and plans.

## Cache and lifecycle

The global cache retains up to 64 LOB paths. This is not a bound on underlying
Vortex file handles. Each caller holds a shared reader throughout its operation.
Cache lookup/publication uses a short lock. Eviction and clear detach entries
under the lock and destroy them outside it, so slow closes cannot block cache
access. The small outer LRU lives in Milvus; no storage LRU API extension is
needed. The existing default-filesystem/path key remains unchanged.

Upstream owns the file-reader cache and close semantics:

- Concurrent sync/async reads are supported. Cold misses may open a file more
  than once before publishing a shared winner; this is not a single-open cache.
- `ClearCache` swaps the cache snapshot. In-flight tasks keep their old snapshot
  and cannot populate the new one.
- `Close` rejects new admissions. Already admitted reads, including deferred
  futures, may complete after close or reader destruction.

Milvus eviction does not explicitly close a reader still used by another call.
Its outer clear may overlap a cache miss that subsequently publishes a reader;
this is resource eviction, not a data-version invalidation barrier.

## Cancellation and failure handling

Cancellation is threaded from `OpContext` through growing/sealed bulk output,
the Arrow take output paths for retrieve/search, and sealed TEXT index reads.
The growing background index feed has no request context and uses an empty
token. Waiting for the global budget or the fallback per-path lock is
cancellable. Inline-only reads check cancellation on entry.

After admission, cancellation stops submitting files and drains all admitted
futures before releasing the batch permit. There is no native cancellation
handle in the upstream API: this does not interrupt an S3 request or guarantee
prompt cancellation of a hung backend operation. Cancellation may therefore
wait until the backend's own timeout/retry policy finishes.

Arrow errors pass through the existing `ToSegcoreError` translation without
stringifying before classification. Exceptional futures propagate their
exception. Once an error has been observed, unwind drains siblings without
checking cancellation again; later cancellation does not replace that error.
Failures never return partially populated output as success.

The review traces are:

| Origin | Path through this integration | Boundary / limitation |
| --- | --- | --- |
| Invalid encoded size/flag/negative offset | Validate before any file submission | `DataFormatBroken`; stored data corruption, not user input |
| File open / native read error | Upstream Result → `CheckedTexts` → existing `ToSegcoreError` | Structured categories retained; siblings drain |
| Timeout / throttle marker | Existing filesystem → Vortex bridge marker → Arrow extended status → Segcore | Existing mapping retained; actual S3 fault injection is still needed |
| Missing file / corrupt file | Upstream open/scan status → existing translation | ENOENT and invalid-data mapping retained; backend variants are not universally verified |
| Wrong row count | Upstream async shape validation or Milvus count assertion | Whole batch fails; no silent empty-text substitution |
| Exceptional future / allocation exception | Future get or caller allocation throws → drain guard | No new catch-all conversion; an allocation failure before admission needs no drain |
| Cancellation | Token check / budget wait → drain if admitted → `FutureCancellation` | Existing Milvus future boundary handles cancellation |

This is not an end-to-end retry-classification change. Existing upstream
callback catch-all paths can turn C++ exceptions into plain Arrow IOError;
this adapter cannot recover a category already lost there. In particular, it
does not claim that every callback allocation failure reaches Milvus as OOM.
No cgo error projection, retry policy or metric-label contract is modified.

## Configuration and fallback

All settings are startup-only and passed through `SegcoreSetTextLobReadConcurrency`
before the global cache is first used.

| Setting under `queryNode.segcore` | Default | Behavior |
| --- | --- | --- |
| `textLobConcurrentReads` | `false` | `false`: serialize per path and use synchronous file reads. `true`: shared-reader asynchronous reads |
| `textLobMaxConcurrentBatches` | `4` | Process-wide batch budget; clamp to 1–64, invalid text uses 4 |
| `textLobMaxConcurrentFilesPerBatch` | `1` | Async file window; clamp to 1–64, invalid text uses 1; used only when enabled |

When enabled, K=1 still uses the upstream async API but admits only one file per
batch. Increase K to allow cross-file overlap. The fallback also shares the
batch budget and deduplicates rows, while retaining synchronous storage calls
and the per-path mutex. Enablement remains opt-in pending backend throughput,
latency and memory acceptance. No production speedup is asserted here.

## Verification

Focused tests cover shared-reader overlap, batch admission independent of
executor thread count, cancellation while queued, reader lifetime through clear
and eviction, duplicate/order/null preservation, file-window replenishment,
exception/Arrow-error draining, and malformed-reference rejection before I/O.
Real local-filesystem Vortex tests cover cache reuse, two requests sharing one
open file, and K=1/K=2 cross-file reads. Grep tests exercise integration with
TEXT/VARCHAR and growing/sealed segments. Go tests cover config defaults/clamps.

Validation uses Milvus Dev CLI cloud jobs, not local test execution or PR
Jenkins. Final task IDs and results are recorded with the submission. Actual
S3/Talon fault injection, throughput/RSS acceptance, and full Rust/FFI race
detection remain follow-up work.

### Completed validation (2026-11-23)

- Dev CLI cloud C++ task
  `cpp-ut-local-lu-zhang-lob-fix-be12b07-53eb7aeb9bcd1751-u59677`:
  succeeded. `all_tests`: 137 passed, including all 21 TEXT cache/real Vortex
  cases, async-load pipeline, manifest translator, metadata-open cancellation
  and staged warmup coverage. `index_tests`: 90 grep cases passed and 70
  inapplicable parameter combinations skipped by the existing fixture.
- Dev CLI cloud Go task
  `go-ut-local-lu-zhang-lob-fix-be12b07-34fea171df1ff4be-u59677`: succeeded.
  `TestTextLobConcurrencySettings` passed, including its 8 clamp subtests.
  `internal/util/initcore` compiled and linked against the rebuilt native
  library; the filter selected no test cases in that package.
- The dependency was verified in the build log as the official `07f135ea`
  revision. A stale cached prototype patch caused the incremental build to
  fall back to a clean build; the successful tests used the clean dependency.
- No new ASAN/TSAN, S3/Talon fault-injection, or performance/RSS result is
  claimed by this validation.

Reproduce the targeted C++ run with:

```bash
milvus-dev-cli cpp-ut local -b master --allow-dirty -t all_tests -t index_tests \
  -k 'TextColumnCache.*:ReadAPIs/TextColumnCacheReadTest.*:*GrepScanTest.*:AsyncLoadPipelineTest.*:*ManifestGroupTranslator*:*TestLazyManifest.LazyManifestMatchCancellationDuringMetadataOpen*:SealedSegmentCowState.StagedStorageV2*:test_chunk_segment.EmptyViewScopeRetainsPrimaryKeyMetadata'
```
