#!/usr/bin/env python3
# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build a standalone delete preparation/lookup benchmark from current kernels.

Does not run Milvus or measure cgo, storage pins, skiplist insertion, snapshots,
or tSafe. Extracts the current reorder and sorted-lookup code to avoid benchmarking
a separately maintained approximation of the optimization.

Example from the repository root:
  python3 internal/core/benchmark/run_delete_batch_benchmark.py \
      --output /tmp/delete-threshold.csv --cpu 2

Both paths are forced, independently of the production threshold. "old" in the
CSV denotes the current binary-search fallback, including timestamp/PK sorting
for streaming and stable timestamp sorting when needed for replay. It is not
the latest master's unconditional merge lookup or the pre-PR tuple-copying
implementation. "new" includes index sorting, permutation, timestamp copying
and the ordered-input check. IDs parsing
is modeled by a PkType copy before timing; common existence filtering is omitted.
Every timed invocation checks its callback count and checksum against an
independent oracle. This validates the benchmark's matches, not MVCC or tSafe.

Each case reports median microseconds over alternating old/new paired rounds.
speedup is the median paired old/new ratio (>1 favors new); min/max are paired
round extrema, not confidence intervals. Independent batches rotate between
rounds, with no explicit cache flush. The fixed-width VARCHAR column uses a
contiguous payload and uint32 offsets; production StringChunk bounds checks and
storage pin wrappers are not measured. stream_same assigns one timestamp to the
entire batch; stream_mixed assigns random timestamps. dense shuffles a contiguous
key range, ordered leaves it sorted, uniform samples the whole segment, repeated
uses 16 keys, uniform_unique samples without replacement (batch <= rows),
repeated_half samples count/2 keys and repeats them, and miss
samples absent keys within the segment's key range.

By default, chunks evenly divide the column. --chunk-rows supplies explicit
positive row counts (one --rows value, summing to it), including a short tail;
it overrides --chunks. It does not read real segment metadata by itself.
"""

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows", default="32768,2000000,44000000")
    parser.add_argument("--batches", default="128,512,1024,2048,4096,16384,65536")
    parser.add_argument("--chunks", default="1,8")
    parser.add_argument("--chunk-rows", help="explicit comma-separated chunk row counts")
    parser.add_argument("--types", default="int64,varchar16")
    parser.add_argument("--shapes", default="uniform,dense,ordered,repeated,miss")
    parser.add_argument("--modes", default="load,stream_mixed,stream_same")
    parser.add_argument("--rounds", default=7, type=int)
    parser.add_argument("--seed", default=53901, type=int)
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    for name in ("rows", "batches", "chunks"):
        try:
            values = [int(value) for value in getattr(args, name).split(",")]
        except ValueError:
            parser.error(f"--{name} must contain comma-separated integers")
        if not values or min(values) < 1:
            parser.error(f"--{name} must contain positive integers")
        if name == "rows" and max(values) * 16 >= 2**32:
            parser.error("VARCHAR column exceeds the uint32 offset range")
    for name, choices in (
        ("types", {"int64", "varchar16"}),
        ("shapes", {"uniform", "uniform_unique", "dense", "ordered", "repeated", "repeated_half", "miss"}),
        ("modes", {"load", "stream_mixed", "stream_same"}),
    ):
        if not set(getattr(args, name).split(",")) <= choices:
            parser.error(f"invalid --{name}; allowed values: {sorted(choices)}")
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.chunk_rows is not None:
        try:
            chunk_rows = [int(value) for value in args.chunk_rows.split(",")]
            rows = [int(value) for value in args.rows.split(",")]
        except ValueError:
            parser.error("--chunk-rows must contain positive integers")
        if not chunk_rows or min(chunk_rows) < 1:
            parser.error("--chunk-rows must contain positive integers")
        if len(rows) != 1 or sum(chunk_rows) != rows[0]:
            parser.error("--chunk-rows must sum to the single --rows value")
        args.chunks = str(len(chunk_rows))
    root = Path(__file__).resolve().parents[3]
    source_path = root / "internal/core/src/segcore/ChunkedSegmentSealedImpl.cpp"
    source = source_path.read_text()
    reorder = source[source.index("template <typename Compare>\nvoid\nReorderDeleteBatch"):
                     source.index("// Formats field identifiers")]
    start = source.index("    size_t first_pk = 0;", source.index("ChunkedSegmentSealedImpl::search_batch_pks"))
    kernel = source[start:source.index("\n    switch (", start)]
    delete_record_path = root / "internal/core/src/segcore/DeletedRecord.h"
    delete_record = delete_record_path.read_text()
    sort_start = delete_record.index("    static const Timestamp*\n    SortByTimestamp")
    sort_end = delete_record.index("\n    // Matches for each row", sort_start)
    sort_by_timestamp = delete_record[sort_start:sort_end]
    generated = reorder + sort_by_timestamp + """
template <typename Chunks, typename GetTimestamp, typename ReadTimestamp,
          typename Callback>
void SortedLookup(const std::vector<PkType>& pks, Chunks chunks,
                  GetTimestamp get_timestamp, ReadTimestamp read_ts,
                  Callback callback) {
    const auto timestamp_hit = [](Timestamp a, Timestamp b) { return a < b; };
""" + kernel + "\n    chunks(search_sorted_chunk);\n}\n"
    cpu = args.cpu if args.cpu is not None else min(os.sched_getaffinity(0))
    if cpu not in os.sched_getaffinity(0):
        parser.error(f"CPU {cpu} is not in this process's allowed affinity set")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="delete-threshold-") as tmp:
        temp = Path(tmp)
        (temp / "delete_kernels.h").write_text(generated)
        executable = temp / "delete_batch_benchmark"
        command = ["c++", "-O3", "-DNDEBUG", "-std=c++20", "-Wall", "-Wextra",
                   "-I" + str(root / "internal/core/src"), "-I" + str(temp),
                   str(Path(__file__).with_name("DeleteBatchBenchmark.cpp")),
                   "-o", str(executable)]
        subprocess.run(command, check=True)
        args.output.with_suffix(".metadata.txt").write_text(
            "command=" + " ".join(command) + "\n"
            + "arguments=" + repr(vars(args)) + "\n"
            + "source_sha256=" + hashlib.sha256(source.encode()).hexdigest() + "\n"
            + "delete_record_sha256=" + hashlib.sha256(delete_record.encode()).hexdigest() + "\n"
            + "kernel_sha256=" + hashlib.sha256(generated.encode()).hexdigest() + "\n"
            + "benchmark_sha256=" + hashlib.sha256(Path(__file__).with_name("DeleteBatchBenchmark.cpp").read_bytes()).hexdigest() + "\n"
            + "cpu=" + str(cpu) + "\n"
            + subprocess.check_output(["c++", "--version"], text=True)
            + subprocess.check_output(["lscpu"], text=True))
        with args.output.open("w") as output:
            run_command = ["taskset", "-c", str(cpu), str(executable), args.rows,
                           args.batches, args.chunks, args.types, args.shapes,
                           args.modes, str(args.rounds), str(args.seed)]
            if args.chunk_rows is not None:
                run_command.append(args.chunk_rows)
            subprocess.run(run_command, stdout=output, check=True)


if __name__ == "__main__":
    main()
