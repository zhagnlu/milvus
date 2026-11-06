#!/usr/bin/env python3
"""Measure B=1/10/100/500 at C=16/32; standard-library component benchmark."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ROWS = [1000000, 5000000, 44000000]
CHUNKS = [16, 32]
BATCHES = [1, 10, 100, 500]
TYPES = ["int64", "varchar16"]
SHAPES = ["uniform_unique", "repeated", "miss"]
MODES = ["stream_same", "stream_mixed", "load"]
SEEDS = [110501, 110517]
ROUNDS = 21
CPU = 2


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    return {str(p.relative_to(ROOT)): sha(p) for p in [
        ROOT / "internal/core/src/segcore/ChunkedSegmentSealedImpl.cpp",
        ROOT / "internal/core/src/common/GallopingSearch.h",
        ROOT / "internal/core/benchmark/DeleteBatchBenchmark.cpp",
        ROOT / "internal/core/benchmark/run_delete_batch_benchmark.py"]}


def observe():
    cpus = {}
    for line in Path("/proc/stat").read_text().splitlines():
        fields = line.split()
        if fields[0].startswith("cpu") and fields[0] != "cpu":
            values = list(map(int, fields[1:9]))
            cpus[fields[0]] = [sum(values), values[3] + values[4]]
    return dict(time=datetime.now(timezone.utc).isoformat(),
                cpu_pressure=Path("/proc/pressure/cpu").read_text().strip(),
                cpu_total_idle_ticks=cpus)


def main():
    runner = ROOT / "internal/core/benchmark/run_delete_batch_benchmark.py"
    expected_per_seed = len(ROWS) * len(CHUNKS) * len(BATCHES) * len(TYPES) * len(SHAPES) * len(MODES)
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(),
                rows=ROWS, chunks=CHUNKS, batches=BATCHES, types=TYPES,
                shapes=SHAPES, modes=MODES, seeds=SEEDS, rounds=ROUNDS, cpu=CPU,
                expected_per_seed=expected_per_seed, source_before=sources(),
                status="running", runs=[])
    for seed in SEEDS:
        output = HERE / f"seed-{seed}.csv"
        command = [sys.executable, str(runner), "--output", str(output),
                   "--rows", ",".join(map(str, ROWS)),
                   "--chunks", ",".join(map(str, CHUNKS)),
                   "--batches", ",".join(map(str, BATCHES)),
                   "--types", ",".join(TYPES), "--shapes", ",".join(SHAPES),
                   "--modes", ",".join(MODES), "--rounds", str(ROUNDS),
                   "--seed", str(seed), "--cpu", str(CPU)]
        plan["runs"].append(dict(seed=seed, output=output.name, command=command, status="pending"))

    def save():
        (HERE / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    stop = threading.Event()

    def monitor():
        with (HERE / "environment.jsonl").open("w") as stream:
            while True:
                stream.write(json.dumps(observe()) + "\n")
                stream.flush()
                if stop.wait(5):
                    break

    observer = threading.Thread(target=monitor, daemon=True)
    observer.start()
    save()
    try:
        for job in plan["runs"]:
            job["status"] = "running"
            job["started_at"] = datetime.now(timezone.utc).isoformat()
            save()
            print(f"START seed={job['seed']}, expected={expected_per_seed}", flush=True)
            result = subprocess.run(job["command"], cwd=ROOT)
            job["exit_code"] = result.returncode
            if result.returncode:
                job["status"] = plan["status"] = "failed"
                save()
                raise SystemExit(result.returncode)
            records = list(csv.DictReader((HERE / job["output"]).open()))
            assert len(records) == expected_per_seed
            job["records"] = len(records)
            job["status"] = "completed"
            job["completed_at"] = datetime.now(timezone.utc).isoformat()
            job["csv_sha256"] = sha(HERE / job["output"])
            save()
            print(f"DONE seed={job['seed']}, records={len(records)}", flush=True)
        plan["source_after"] = sources()
        plan["status"] = "completed"
        plan["completed_at"] = datetime.now(timezone.utc).isoformat()
        save()
    finally:
        stop.set()
        observer.join()


if __name__ == "__main__":
    main()
