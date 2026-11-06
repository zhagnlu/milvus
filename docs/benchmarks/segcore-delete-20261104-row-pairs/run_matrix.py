#!/usr/bin/env python3
"""Measure the requested row/pair matrix with modeled row-group boundaries.

Run from the repository root. This is preparation+lookup, not Milvus E2E.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading


ROWS = [500000, 1000000, 2000000, 4000000, 5000000]
BATCHES = [100, 1000, 10000, 50000, 100000]
WIDTHS = {"int64": 24, "varchar16": 36}
MODES = ["stream_same", "stream_mixed", "load"]
SEEDS = [110401, 110417]
ROUNDS = 21
RG_BYTES = 1024 * 1024
CELL_BYTES = 4 * RG_BYTES


def layouts(rows, width):
    # Constant-width approximation of ParquetFileWriter::Flush: when the
    # remaining budget is smaller than one row, it still writes one row.
    full_group_rows = (RG_BYTES + width - 1) // width
    row_groups = [full_group_rows] * (rows // full_group_rows)
    if rows % full_group_rows:
        row_groups.append(rows % full_group_rows)
    average_size = sum(n * width for n in row_groups) // len(row_groups)
    per_cell = max(1, CELL_BYTES // average_size)
    yield "v2_stats_on", 1, row_groups, row_groups
    chunks = [sum(row_groups[i:i + per_cell])
              for i in range(0, len(row_groups), per_cell)]
    yield "v2_stats_off", per_cell, row_groups, chunks


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observe_environment():
    cpus = {}
    for line in Path("/proc/stat").read_text().splitlines():
        fields = line.split()
        if fields[0].startswith("cpu") and fields[0] != "cpu":
            values = list(map(int, fields[1:9]))
            cpus[fields[0]] = [sum(values), values[3] + values[4]]
    return dict(time=datetime.now(timezone.utc).isoformat(),
                loadavg=Path("/proc/loadavg").read_text().strip(),
                cpu_pressure=Path("/proc/pressure/cpu").read_text().strip(),
                memory_pressure=Path("/proc/pressure/memory").read_text().strip(),
                cpu_total_idle_ticks=cpus)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    runner = root / "internal/core/benchmark/run_delete_batch_benchmark.py"
    source_paths = [
        "internal/core/src/segcore/ChunkedSegmentSealedImpl.cpp",
        "internal/core/src/segcore/DeletedRecord.h",
        "internal/core/src/common/GallopingSearch.h",
        "internal/core/benchmark/DeleteBatchBenchmark.cpp",
        "internal/core/benchmark/run_delete_batch_benchmark.py",
        "internal/core/src/segcore/storagev2translator/GroupCTMeta.h",
        "internal/core/src/segcore/storagev2translator/GroupChunkTranslator.cpp",
    ]
    before = {p: sha(root / p) for p in source_paths}
    jobs = []
    for seed in ([args.seed] if args.seed else SEEDS):
        for kind, width in WIDTHS.items():
            for rows in ROWS:
                for layout, per_cell, row_groups, chunks in layouts(rows, width):
                    output = out / str(seed) / f"{kind}-{rows}-{layout}.csv"
                    command = [sys.executable, str(runner), "--output", str(output),
                               "--rows", str(rows), "--batches", ",".join(map(str, BATCHES)),
                               "--chunk-rows", ",".join(map(str, chunks)),
                               "--types", kind, "--shapes", "uniform_unique",
                               "--modes", ",".join(MODES), "--rounds", str(ROUNDS),
                               "--seed", str(seed), "--cpu", str(args.cpu)]
                    jobs.append(dict(seed=seed, type=kind, rows=rows, layout=layout,
                                     group_row_bytes=width, row_group_rows=row_groups,
                                     row_groups_per_cell=per_cell, chunk_rows=chunks,
                                     chunks=len(chunks), output=str(output.relative_to(out)),
                                     command=command, status="pending"))
    manifest = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        purpose="5 row counts x 5 pair counts; two PK types; two modeled V2 layouts; three modes",
        assumptions=["one logical column-group file; no extra partition/clustering fields",
                     "constant estimated row width: 24 B int64, 36 B VARCHAR16",
                     "row groups use 1 MiB target with at least one whole row",
                     "actual file/segment metadata not read; explicit full groups plus tail",
                     "old/current binary fallback vs forced new; not original master E2E",
                     "100/1000 pairs do not enable new lookup in production (threshold 1024)",
                     "random unique existing PKs; mixed-mode timestamps can reject some row versions"],
        batches=BATCHES, modes=MODES, rounds=ROUNDS, cpu=args.cpu,
        source_sha256_before=before, expected_records=len(jobs) * len(BATCHES) * len(MODES),
        status="planned", runs=jobs)
    manifest_path = out / (f"plan-{args.seed}.json" if args.seed else "plan.json")

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    matrix_path = out / (f"matrix-{args.seed}.csv" if args.seed else "matrix.csv")
    environment_log = out / (f"environment-{args.seed}.jsonl" if args.seed else "environment.jsonl")
    stop_monitor = threading.Event()

    def monitor_environment():
        with environment_log.open("w") as stream:
            while True:
                stream.write(json.dumps(observe_environment()) + "\n")
                stream.flush()
                if stop_monitor.wait(5):
                    break

    monitor = threading.Thread(target=monitor_environment, daemon=True)
    monitor.start()
    manifest["status"] = "running"
    with matrix_path.open("w", newline="") as matrix:
        writer = None
        for i, job in enumerate(jobs):
            print(f"START {i + 1}/{len(jobs)} seed={job['seed']} {job['type']} "
                  f"N={job['rows']} C={job['chunks']} {job['layout']}", flush=True)
            job["status"] = "running"
            job["started_at"] = datetime.now(timezone.utc).isoformat()
            save()
            result = subprocess.run(job["command"], cwd=root)
            job["exit_code"] = result.returncode
            if result.returncode:
                job["status"] = manifest["status"] = "failed"
                save()
                raise SystemExit(result.returncode)
            records = list(csv.DictReader((out / job["output"]).open()))
            assert len(records) == len(BATCHES) * len(MODES)
            for row in records:
                row.update(seed=job["seed"], layout=job["layout"],
                           group_row_bytes=job["group_row_bytes"],
                           row_groups_per_cell=job["row_groups_per_cell"],
                           chunk_rows=json.dumps(job["chunk_rows"], separators=(",", ":")))
                if writer is None:
                    writer = csv.DictWriter(matrix, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
            matrix.flush()
            job["status"] = "completed_exit_0"
            job["completed_at"] = datetime.now(timezone.utc).isoformat()
            job["records"] = len(records)
            save()
            print(f"DONE {i + 1}/{len(jobs)}", flush=True)
    after = {p: sha(root / p) for p in source_paths}
    manifest["source_sha256_after"] = after
    assert before == after, "source changed while measuring"
    manifest["status"] = "measurements_completed"
    manifest["matrix_sha256"] = sha(matrix_path)
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    stop_monitor.set()
    monitor.join()
    save()


if __name__ == "__main__":
    main()
