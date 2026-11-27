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

"""Opt-in closed-loop JSONBench throughput sweep over prepared local databases.

Each database is run alone. Increase clients until CPU saturation or a measured
throughput plateau, then confirm the best error-free concurrency three times.
Every successful response, including warmup/drain responses, is oracle checked.
"""

import argparse
import concurrent.futures
import datetime as dt
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import tarfile
import threading
import time
from types import SimpleNamespace

import requests

import jsonbench_compare as bench
import jsonbench_clickhouse as clickhouse
import jsonbench_native_clients as native

ROOT = Path(__file__).resolve().parents[2]
ENGINES = ("milvus", "clickhouse", "starrocks", "doris", "elasticsearch", "duckdb", "postgresql", "mongodb")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def window_statistics(records, start, end):
    completed = [row for row in records if start <= row["end_s"] < end]
    succeeded = [row for row in completed if row["ok"]]
    # Include drain completions for requests ADMITTED in the window, so a slow
    # tail is not silently omitted from the latency distribution.
    cohort = [row for row in records if start <= row["start_s"] < end]
    latencies = [(row["end_s"] - row["start_s"]) * 1000 for row in cohort if row["ok"]]
    return {"window_seconds": end - start, "successful_completions": len(succeeded),
            "error_completions": len(completed) - len(succeeded),
            "qps": len(succeeded) / (end - start), "latency_admission_cohort": len(cohort),
            "p50_ms": percentile(latencies, .5), "p95_ms": percentile(latencies, .95),
            "p99_ms": percentile(latencies, .99), "all_phase_errors": sum(not row["ok"] for row in records),
            "all_phase_successes": sum(row["ok"] for row in records)}


def memory_of(state):
    processes = {}
    for component in state.get("processes", [state]):
        for item in component["process_tree"]:
            value, unit = item["memory"]["VmRSS"].split()
            if unit != "kB":
                raise RuntimeError("Unexpected RSS unit")
            processes[item["pid"]] = int(value)
    return {"rss_sum_kib": sum(processes.values()), "pid_rss_kib": processes}


def database_state(roots):
    states = []
    for child in roots:
        if child.poll() is not None:
            raise RuntimeError(f"Database root exited during load: {child.pid}, {child.returncode}")
        states.append(native.native_cpu(child.pid))
    return {"cpu_seconds": sum(value["cpu_seconds"] for value in states), "processes": states}


class Client:
    def __init__(self, engine, queries, oracle):
        self.engine, self.queries, self.oracle = engine, queries, oracle
        self.session = requests.Session()
        self.session.trust_env = False
        if engine == "milvus":
            self.session.headers["Accept-Type-Allow-Int64"] = "true"

    def close(self):
        self.session.close()

    def execute(self, name):
        statement = self.queries[name]
        if self.engine == "clickhouse":
            response = self.session.post("http://127.0.0.1:18123", params={
                "max_threads": 16, "max_memory_usage": 48 * 1024**3, "use_query_cache": 0,
                "output_format_json_quote_64bit_integers": 0, "date_time_output_format": "iso", "session_timezone": "UTC"},
                data=(statement.rstrip("; \n") + " FORMAT JSON").encode(), timeout=(5, 100))
        elif self.engine == "elasticsearch":
            response = self.session.post("http://127.0.0.1:19200/_query", json={"query": statement, "columnar": False}, timeout=(5, 100))
        else:
            port = 19530 if self.engine == "milvus" else 19531
            response = self.session.post(f"http://127.0.0.1:{port}" + bench.HTTP_PATH,
                json={"sql": statement, "query_name": name, "timezone": "UTC", "consistencyLevel": "Strong"}, timeout=(5, 100))
        raw = response.content
        ended = time.perf_counter()
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code}: {raw[:2000]!r}")
        data = json.loads(raw)
        if self.engine == "elasticsearch":
            if data.get("is_partial") or response.headers.get("Warning"):
                raise RuntimeError(f"Partial ES|QL result/warning: {data}, {response.headers.get('Warning')}")
            columns = [item["name"] for item in data["columns"]]
            rows = [dict(zip(columns, row)) for row in data["values"]]
        else:
            if self.engine != "clickhouse" and data.get("code") != 0:
                raise RuntimeError(f"Database error: {data}")
            rows = data["data"]
        bench.check(name, rows, self.oracle)
        return ended


def stage(engine, roots, queries, oracle, names, concurrency, seconds, warmup, adapter, phase):
    ready = threading.Barrier(concurrency + 1)
    stop = threading.Event()

    def worker(index):
        client = Client(engine, queries, oracle)
        records = []
        try:
            # A checked query also establishes the HTTP and native connections.
            client.execute(names[index % len(names)])
            ready.wait(timeout=150)
            number = index
            while not stop.is_set():
                name = names[number % len(names)]
                number += 1
                started = time.perf_counter()
                try:
                    ended = client.execute(name)
                    row = {"query": name, "start_s": started, "end_s": ended, "ok": True}
                except Exception as error:
                    row = {"query": name, "start_s": started, "end_s": time.perf_counter(),
                           "ok": False, "error_type": type(error).__name__, "error": str(error)[:2500]}
                    stop.set()
                records.append(row)
            return records
        except BaseException:
            stop.set()
            ready.abort()
            raise
        finally:
            client.close()

    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        try:
            futures = [pool.submit(worker, index) for index in range(concurrency)]
            ready.wait(timeout=150)
            warmup_start = time.perf_counter()
            stop.wait(warmup)
            before = database_state(roots)
            adapter_before = native.native_cpu(adapter.pid) if adapter is not None and adapter not in roots else None
            driver_before = time.process_time()
            started = time.perf_counter()
            requested_end = started + seconds
            snapshots = [{"time_s": started, "database": before, "memory": memory_of(before)}]
            while time.perf_counter() < requested_end and not stop.is_set():
                stop.wait(min(1, max(0, requested_end - time.perf_counter())))
                if time.perf_counter() < requested_end:
                    snapshot = database_state(roots)
                    snapshots.append({"time_s": time.perf_counter(), "database": snapshot, "memory": memory_of(snapshot)})
            ended = time.perf_counter()
            stop.set()
            after = database_state(roots)
            driver_cpu = time.process_time() - driver_before
            adapter_after = native.native_cpu(adapter.pid) if adapter_before is not None else None
            snapshots.append({"time_s": ended, "database": after, "memory": memory_of(after)})
        finally:
            stop.set()
        records = [row for future in futures for row in future.result(timeout=120)]
    records.sort(key=lambda row: (row["start_s"], row["end_s"]))
    cpu = after["cpu_seconds"] - before["cpu_seconds"]
    if cpu < 0:
        raise RuntimeError("Database process-tree CPU counter moved backwards")
    result = {"phase": phase, "concurrency": concurrency, "warmup_seconds": started - warmup_start,
              "measurement_start_s": started, "measurement_end_s": ended, "cpu_before": before, "cpu_after": after,
              "database_cpu_seconds": cpu, "average_busy_logical_cpus": cpu / (ended - started),
              "average_host_cpu_percent": cpu / (ended - started) / 16 * 100,
              "load_generator_cpu_seconds": driver_cpu, "load_generator_busy_cpus": driver_cpu / (ended - started),
              "monitor_samples": snapshots, "rss_sampled_max_gib": max(row["memory"]["rss_sum_kib"] for row in snapshots) / 1024**2,
              "rss_sampled_median_gib": statistics.median(row["memory"]["rss_sum_kib"] for row in snapshots) / 1024**2,
              "requests": records, **window_statistics(records, started, ended)}
    if adapter_before is not None:
        result["adapter_busy_cpus"] = (adapter_after["cpu_seconds"] - adapter_before["cpu_seconds"]) / (ended - started)
    result["valid"] = result["all_phase_errors"] == 0 and result["successful_completions"] > 0 and ended - started >= seconds * .98
    bench.progress(f"{engine} {phase} C={concurrency}: QPS={result['qps']:.3f}, CPU={result['average_host_cpu_percent']:.1f}%, p95={result['p95_ms']}, errors={result['all_phase_errors']}")
    return result


class Runtime:
    def __init__(self, args, engine, label):
        self.args, self.engine, self.label = args, engine, label
        self.roots, self.adapter = [], None
        self.stop_method = None
        self.metadata = {}

    def start_adapter(self, queries):
        args = self.args
        bench.vacant(19531)
        workload = args.output_dir / f"{self.engine}-workload.json"
        bench.save(workload, {"queries": queries, "dataset_ndjson_sha256": args.baseline["dataset"]["ndjson_sha256"]})
        with (args.output_dir / f"{self.engine}-adapter.log").open("w") as stream:
            child = subprocess.Popen([sys.executable, str(Path(__file__).with_name("jsonbench_qps_worker.py")),
                                      "--engine", self.engine, "--run-dir", str(args.run_dir), "--workload", str(workload)],
                                      cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        child.benchmark_marker = args.output_dir / f"{self.engine}-adapter-active.json"
        bench.save(child.benchmark_marker, {"pid": child.pid, "status": "running"})
        self.adapter = child
        with requests.Session() as session:
            session.trust_env = False
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise RuntimeError(f"{self.engine} threaded adapter exited; see adapter log")
                try:
                    response = session.get("http://127.0.0.1:19531/healthz", timeout=2)
                    if response.ok and response.json().get("threaded") is True:
                        return
                except requests.RequestException:
                    pass
                time.sleep(1)
        raise TimeoutError("Threaded adapter readiness timeout")

    def start(self):
        args, engine = self.args, self.engine
        prepared = SimpleNamespace(run_dir=args.run_dir, clickhouse_binary=args.run_dir / "clickhouse/clickhouse",
            postgres_bin=Path("/tmp/jsonbench-postgresql-18.6/bin"), postgres_state=Path("/tmp/jsonbench-postgresql-state-20261106"),
            mongodb_binary=args.run_dir / "mongodb-linux-x86_64-ubuntu2204-9.0.4/bin/mongod")
        queries = args.baseline["queries"]
        with requests.Session() as session:
            session.trust_env = False
            if engine == "milvus":
                child = bench.start(prepared, engine, self.label)
                self.roots, self.stop_method = [child], lambda: bench.stop(child)
                session.headers.update({"Content-Type": "application/json", "Accept-Type-Allow-Int64": "true"})
                layout = bench.load_milvus(prepared, session, args.baseline["dataset"])
                if layout != args.baseline["rounds"][0]["milvus_segments"]:
                    raise RuntimeError("Milvus segment layout differs from baseline")
                self.metadata["milvus_segments"] = layout
            elif engine == "clickhouse":
                child = clickhouse.start(prepared, self.label)
                self.roots, self.stop_method = [child], lambda: bench.stop(child)
                report = json.loads((ROOT / "docs/benchmark-results/20261110-jsonbench-clickhouse-cpu-same-host.json").read_text())
                queries = report["clickhouse_queries"]
                schema = (args.run_dir / "external-sources/json__clickhouse__create.sql").read_text()
                self.metadata["parts"] = clickhouse.prepare(prepared, session, args.baseline["dataset"], schema)
                self.metadata["cache_policy"] = "use_query_cache=0 on every request"
            elif engine in ("starrocks", "doris"):
                columnar = module("qps_columnar", args.run_dir / "run-columnar.py")
                children, home, backends = columnar.start(engine, self.label)
                self.roots, self.stop_method = children, lambda: columnar.stop(children)
                schema, queries = columnar.setup_queries(engine)
                self.metadata["schema"] = columnar.prepare(engine, schema, args.baseline["dataset"])
                self.metadata["backends"] = backends
                self.metadata["cache_policy"] = "query cache disabled and checked per HTTP/native connection; Doris SQL cache also disabled"
                self.start_adapter(queries)
            elif engine == "elasticsearch":
                elastic = module("qps_elastic", args.run_dir / "run-elasticsearch.py")
                child = elastic.start(self.label)
                self.roots, self.stop_method = [child], lambda: elastic.stop(child)
                queries = elastic.QUERIES
                self.metadata["effective_index"] = elastic.prepare(session, args.baseline["dataset"])
                self.metadata["cache_policy"] = "index.requests.cache.enable=false"
            elif engine == "duckdb":
                report = json.loads((args.run_dir / "comparison.json").read_text())
                queries = report["duckdb_queries"]
                self.start_adapter(queries)
                self.roots = [self.adapter]
                self.metadata["database_threads"] = 16
                self.metadata["database_memory_limit"] = "48GiB"
            else:
                if engine == "postgresql":
                    restore_postgres(args)
                child = native.launch(prepared, engine, self.label)
                self.roots, self.stop_method = [child], lambda: bench.stop(child)
                self.metadata["settings"] = native.prepare(prepared, engine, args.baseline["dataset"])
                self.start_adapter(queries)
        return queries

    def stop(self):
        codes = {}
        if self.adapter is not None:
            codes["adapter"] = bench.stop(self.adapter)
        if self.stop_method is not None:
            codes["database"] = self.stop_method()
        return codes


def restore_postgres(args):
    path = Path("/tmp/jsonbench-postgresql-18.6")
    if not (path / "bin/postgres").exists():
        archive = args.run_dir / "postgres-install.tar.gz"
        expected = json.loads((args.run_dir / "postgres-install.json").read_text())["sha256"]
        if bench.digest(archive) != expected:
            raise RuntimeError("PostgreSQL install archive fingerprint differs")
        with tarfile.open(archive) as source:
            if any(not (item.name == path.name or item.name.startswith(path.name + "/")) for item in source.getmembers()):
                raise RuntimeError("Unexpected PostgreSQL install archive path")
            source.extractall(path.parent, filter="data")
    expected = json.loads((args.run_dir / "native-comparison.json").read_text())["postgres_binary_sha256"]
    if bench.digest(path / "bin/postgres") != expected:
        raise RuntimeError("PostgreSQL binary differs from the previous native run")


def run_engine(args, engine):
    path = args.output_dir / f"{engine}.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("status") == "complete":
            bench.progress(f"{engine}: reusing completed QPS checkpoint")
            return previous
        raise FileExistsError(f"Preserving incomplete {path}; use a new output directory for retry")
    report = {"engine": engine, "status": "running", "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "queries_tested": args.queries, "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "dataset_ndjson_sha256": args.baseline["dataset"]["ndjson_sha256"], "stages": [], "confirmations": []}
    bench.save(path, report)
    runtime = Runtime(args, engine, f"qps-{args.output_dir.name}-{engine}")
    try:
        queries = runtime.start()
        report["queries"] = queries
        report["metadata"] = runtime.metadata
        client = Client(engine, queries, args.baseline["dataset"]["oracle"])
        try:
            begin = time.perf_counter()
            for name in args.queries:
                client.execute(name)
            baseline_latency = (time.perf_counter() - begin) / len(args.queries)
        finally:
            client.close()
        best = None
        plateau = 0
        for concurrency in (1, 2, 4, 8, 16, 32, 64):
            if concurrency > args.max_concurrency:
                break
            warmup = max(args.warmup, min(20, concurrency * baseline_latency / 2))
            result = stage(engine, runtime.roots, queries, args.baseline["dataset"]["oracle"], args.queries,
                           concurrency, args.seconds, warmup, runtime.adapter, "sweep")
            report["stages"].append(result)
            bench.save(path, report)
            if not result["valid"]:
                report["sweep_stop_reason"] = "errors_or_invalid_measurement"
                break
            if best is not None:
                gain = result["qps"] / best["qps"] - 1
                plateau = plateau + 1 if gain < .05 else 0
                if plateau >= 1 and result["average_host_cpu_percent"] >= args.cpu_target:
                    report["sweep_stop_reason"] = "CPU_target_and_throughput_plateau"
                elif plateau >= 2:
                    report["sweep_stop_reason"] = "throughput_plateau"
            if best is None or result["qps"] > best["qps"]:
                best = result
            if "sweep_stop_reason" in report:
                break
        if best is None:
            raise RuntimeError("No error-free throughput stage")
        concurrency = best["concurrency"]
        for _ in range(args.repetitions):
            warmup = max(args.warmup, min(20, concurrency * baseline_latency / 2))
            result = stage(engine, runtime.roots, queries, args.baseline["dataset"]["oracle"], args.queries,
                           concurrency, args.confirm_seconds, warmup, runtime.adapter, "confirmation")
            report["confirmations"].append(result)
            bench.save(path, report)
        if not all(row["valid"] for row in report["confirmations"]):
            raise RuntimeError("Selected concurrency failed confirmation")
        report["summary"] = {"selected_concurrency": concurrency,
            "confirmed_median_qps": statistics.median(row["qps"] for row in report["confirmations"]),
            "confirmed_qps_range": [min(row["qps"] for row in report["confirmations"]), max(row["qps"] for row in report["confirmations"])],
            "confirmed_median_host_cpu_percent": statistics.median(row["average_host_cpu_percent"] for row in report["confirmations"]),
            "confirmed_median_p95_ms": statistics.median(row["p95_ms"] for row in report["confirmations"]),
            "confirmed_sampled_max_rss_gib": max(row["rss_sampled_max_gib"] for row in report["confirmations"]),
            "cpu_target_reached_in_confirmation": all(row["average_host_cpu_percent"] >= args.cpu_target for row in report["confirmations"])}
        if engine == "milvus" and bench.segments() != runtime.metadata["milvus_segments"]:
            raise RuntimeError("Milvus layout changed during throughput timing")
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        bench.progress(f"{engine} FAILED: {error}")
    finally:
        report["shutdown_exit_codes"] = runtime.stop()
        report["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        bench.save(path, report)
    return report


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.baseline = json.loads((args.run_dir / "comparison.json").read_text())
    if sorted(os.sched_getaffinity(0)) != args.baseline["cpu_affinity"]:
        raise RuntimeError("CPU affinity differs from the historical baseline")
    if bench.digest(args.run_dir / "first-million.ndjson") != args.baseline["dataset"]["ndjson_sha256"]:
        raise RuntimeError("Dataset fingerprint differs")
    if (bench.digest(ROOT / "bin/milvus") != args.baseline["milvus_binary_sha256"]
            or bench.digest(ROOT / "cmake_build/src/libmilvus_core.so") != args.baseline["milvus_core_sha256"]):
        raise RuntimeError("Milvus binaries differ")
    original_sysctl = Path("/proc/sys/vm/max_map_count").read_text().strip()
    manifest = {"status": "running", "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_ndjson_sha256": args.baseline["dataset"]["ndjson_sha256"], "milvus_binary_sha256": args.baseline["milvus_binary_sha256"],
        "milvus_core_sha256": args.baseline["milvus_core_sha256"], "available_logical_cpus": 16,
        "queries": args.queries, "settings": {key: getattr(args, key) for key in ("seconds", "confirm_seconds", "warmup", "repetitions", "max_concurrency", "cpu_target")},
        "source_sha256": {path.name: bench.digest(path) for path in (Path(__file__), Path(__file__).with_name("jsonbench_qps_worker.py"))},
        "engines": {}, "methodology": [
            "Closed-loop concurrent HTTP clients, complete responses; every successful warmup/timed/drain response checked against independent source oracle.",
            "QPS counts successful completions inside the window, including requests started before it; latency cohort includes all requests admitted in the window, including drain completions.",
            "Databases run sequentially; 16 logical CPU affinity, warm caches, unchanged prepared layouts, no identical cgroup memory cap.",
            "Query/result caches disabled for ClickHouse/StarRocks/Doris; Elasticsearch shard request cache disabled. Data/page caches remain warm.",
            "Database process-tree CPU includes live/reaped workers and background work; driver and external threaded-adapter CPU measured separately.",
            "RSS sampled about once per second; live PID RSS sum repeats shared pages and is not PSS or a guaranteed instantaneous peak.",
            "Each threaded-adapter connection owns an independent native database connection/cursor; DuckDB CPU includes its embedded adapter.",
            "CPU target is an observed threshold, not assumed saturation; a QPS plateau below it is reported explicitly."]}
    status_path = args.output_dir / "status.json"
    bench.save(status_path, manifest)
    try:
        if "doris" in args.engines:
            subprocess.run(["sysctl", "-w", "vm.max_map_count=2000000"], check=True)
        for engine in args.engines:
            manifest["active_engine"] = engine
            bench.save(status_path, manifest)
            report = run_engine(args, engine)
            manifest["engines"][engine] = {"status": report["status"], "summary": report.get("summary"), "error": report.get("error")}
            bench.save(status_path, manifest)
        manifest["status"] = "complete" if all(value["status"] == "complete" for value in manifest["engines"].values()) else "incomplete"
    finally:
        if "doris" in args.engines:
            restored = subprocess.run(["sysctl", "-w", "vm.max_map_count=" + original_sysctl], capture_output=True, text=True)
            manifest["sysctl_restored"] = restored.returncode == 0
            manifest["sysctl_restored_value"] = Path("/proc/sys/vm/max_map_count").read_text().strip()
        manifest["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        bench.save(status_path, manifest)
        print(json.dumps({"status": manifest["status"], "engines": manifest["engines"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / ".benchmarks/jsonbench-20261106")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=list(ENGINES))
    parser.add_argument("--queries", nargs="+", choices=[f"Q{i}" for i in range(1, 6)], default=["Q2"])
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--confirm-seconds", type=float, default=30)
    parser.add_argument("--warmup", type=float, default=5)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--cpu-target", type=float, default=90)
    args = parser.parse_args()
    if args.seconds <= 0 or args.confirm_seconds <= 0 or args.warmup < 0 or args.repetitions < 1 or args.max_concurrency < 1:
        parser.error("Invalid duration/concurrency/repetition counts")
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main(args)
