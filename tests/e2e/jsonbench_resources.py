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

"""Summarize persisted JSONBench CPU samples and comparable post-query RSS.

No databases are started. Memory is sampled RSS, not a continuous peak or PSS.
Use --results-dir to select the directory containing the recorded reports.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics


SOURCES = [
    ("20261106-jsonbench-same-host.json", ("duckdb",)),
    ("20261106-jsonbench-native-same-host.json", ("postgresql", "mongodb")),
    ("20261110-jsonbench-clickhouse-cpu-same-host.json", ("milvus", "clickhouse")),
    ("20261110-jsonbench-starrocks-same-host.json", ("milvus", "starrocks")),
    ("20261110-jsonbench-doris-same-host.json", ("milvus", "doris")),
    ("20261110-jsonbench-elasticsearch-same-host.json", ("milvus", "elasticsearch")),
]
ORDER = ("milvus", "clickhouse", "starrocks", "doris", "elasticsearch", "duckdb", "postgresql", "mongodb")
QUERIES = tuple(f"Q{i}" for i in range(1, 6))


def rss_kib(memory):
    value, unit = memory["VmRSS"].split()
    if unit != "kB" or int(value) < 0:
        raise ValueError(f"Unexpected /proc RSS value: {memory['VmRSS']}")
    return int(value)


def processes(state):
    """Return each live PID once, without double-counting a tree's root."""
    if "processes" in state:
        found = {}
        for component in state["processes"]:
            for key, value in processes(component).items():
                if key in found and found[key] != value:
                    raise ValueError(f"Inconsistent duplicate PID {key}")
                found[key] = value
        return found
    if "process_tree" in state:
        if not state["process_tree"]:
            raise ValueError("Missing live process-tree memory")
        found = {}
        for process in state["process_tree"]:
            key = str(process["pid"])
            value = rss_kib(process["memory"])
            if key in found and found[key] != value:
                raise ValueError(f"Inconsistent duplicate PID {key}")
            found[key] = value
        return found
    return {"single_process": rss_kib(state["memory"])}


def post_query_memory(result):
    # Every report retains the state after the LAST timed sample of each query.
    # Ignore denser before/after memory snapshots available only for some peers.
    if "process_after" in result:
        state = result["process_after"]
    else:
        state = result["cpu_counters"][-1]["after"]
    values = processes(state)
    return {"rss_sum_kib": sum(values.values()), "pid_rss_kib": values,
            "live_process_count": len(values)}


def summarize(cpu, wall, memory):
    if not cpu or len(cpu) != len(wall) or not memory:
        raise ValueError("Incomplete CPU/wall/memory evidence")
    if not all(math.isfinite(value) and value >= 0 for value in cpu):
        raise ValueError("Invalid CPU sample")
    if not all(math.isfinite(value) and value > 0 for value in wall):
        raise ValueError("Invalid HTTP wall sample")
    seconds = sum(wall) / 1000
    rss = [item["rss_sum_kib"] / 1024**2 for item in memory]
    busy = sum(cpu) / seconds
    return {"timed_sample_count": len(cpu), "total_cpu_seconds": sum(cpu),
            "total_http_wall_seconds": seconds, "mean_cpu_seconds_per_query": statistics.mean(cpu),
            "average_busy_logical_cpus": busy, "average_host_cpu_percent": busy / 16 * 100,
            "post_query_rss_sample_count": len(rss), "post_query_rss_median_gib": statistics.median(rss),
            "post_query_rss_min_gib": min(rss), "post_query_rss_max_gib": max(rss)}


def main(args):
    raw = {}
    source_hashes = {}
    dataset_hash = None
    for filename, selected_engines in SOURCES:
        path = args.results_dir / filename
        data = path.read_bytes()
        report = json.loads(data)
        source_hashes[filename] = hashlib.sha256(data).hexdigest()
        if report["cpu_affinity"] != list(range(16)) or report["dataset"]["rows"] != 1_000_000:
            raise ValueError(f"Unexpected affinity/dataset: {filename}")
        if (len(report["rounds"]) != 2 or report["warmups_per_query_per_round"] != 2
                or report["samples_per_query_per_round"] != 10):
            raise ValueError(f"Unexpected sampling schedule: {filename}")
        current_hash = report["dataset"]["ndjson_sha256"]
        if dataset_hash is not None and current_hash != dataset_hash:
            raise ValueError("Input fingerprints differ")
        dataset_hash = current_hash
        for round_number, rnd in enumerate(report["rounds"], 1):
            for engine in selected_engines:
                if engine in ("starrocks", "doris") and report.get("query_caches_disabled") is not True:
                    raise ValueError("Cached exploratory report cannot enter the primary summary")
                target = raw.setdefault(engine, {query: {"cpu": [], "wall": [], "memory": []} for query in QUERIES})
                for query in QUERIES:
                    result = rnd["engines"][engine][query]
                    cpu, wall = result["process_cpu_seconds"], result["samples_ms"]
                    if len(cpu) != 10 or len(wall) != 10 or result["oracle_passed_every_sample"] is not True:
                        raise ValueError(f"Incomplete verified samples: {filename}, {engine}, {query}")
                    if not math.isclose(result["cpu_seconds_over_wall_seconds"], sum(cpu) / (sum(wall) / 1000), rel_tol=1e-12):
                        raise ValueError("Stored CPU ratio disagrees with raw samples")
                    memory = post_query_memory(result)
                    memory.update({"source_report": filename, "round": round_number, "query": query})
                    target[query]["cpu"].extend(cpu)
                    target[query]["wall"].extend(wall)
                    target[query]["memory"].append(memory)
    engines = {}
    for engine in ORDER:
        values = raw[engine]
        per_query = {query: summarize(**values[query]) for query in QUERIES}
        all_cpu = [value for query in QUERIES for value in values[query]["cpu"]]
        all_wall = [value for query in QUERIES for value in values[query]["wall"]]
        all_memory = [value for query in QUERIES for value in values[query]["memory"]]
        engines[engine] = {"queries": per_query, "all_queries": summarize(all_cpu, all_wall, all_memory),
                           "post_query_memory_evidence": all_memory}
    output = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "dataset_rows": 1_000_000,
              "dataset_ndjson_sha256": dataset_hash, "available_logical_cpus": 16,
              "available_host_memory_gib": 64398556 / 1024**2,
              "source_reports_sha256": source_hashes,
              "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "methodology": {
                  "cpu": "sum timed process CPU deltas / sum complete HTTP wall seconds; background work included; CPU reads are just outside the wall timer",
                  "cpu_percent": "average busy logical CPUs / 16 * 100; one busy logical CPU equals 100% in per-core/top convention",
                  "memory": "sum VmRSS of unique live database PIDs at the last timed sample of each query in each round, then summarize those snapshots",
                  "rss_limitations": "RSS sums repeat shared pages across processes, exclude OS page cache and workers already exited by the snapshot; not PSS, query allocations, or continuous peak memory",
                  "sampling": "10 post-query memory snapshots per peer; Milvus pools the four November 10 paired runs (40 snapshots and 400 timed samples)",
                  "measurement_scope": "query measurement only; excludes loading, startup, warmups and pauses; no identical process memory cap was enforced",
                  "dates": "DuckDB/PostgreSQL/MongoDB: November 6; Milvus/ClickHouse/StarRocks/Doris/Elasticsearch: November 10",
                  "adapter_cpu": "DuckDB runs inside its Python HTTP adapter process; other database CPU excludes external adapters/clients",
                  "excluded": "Historical watchdog-only ClickHouse CPU and exploratory cached Doris report are excluded",
              }, "engines": engines}
    totals = {"timed_cpu_wall_samples": sum(value["all_queries"]["timed_sample_count"] for value in engines.values()),
              "comparable_post_query_rss_snapshots": sum(value["all_queries"]["post_query_rss_sample_count"] for value in engines.values())}
    if totals != {"timed_cpu_wall_samples": 1100, "comparable_post_query_rss_snapshots": 110}:
        raise ValueError(f"Unexpected combined evidence counts: {totals}")
    output["evidence_counts"] = totals
    prefix = args.results_dir / args.output_prefix
    prefix.with_suffix(".json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    columns = ["engine", "query", "timed_sample_count", "average_busy_logical_cpus", "average_host_cpu_percent",
               "mean_cpu_seconds_per_query", "post_query_rss_sample_count", "post_query_rss_median_gib",
               "post_query_rss_min_gib", "post_query_rss_max_gib"]
    with prefix.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for engine, values in engines.items():
            for query, value in list(values["queries"].items()) + [("ALL", values["all_queries"])]:
                writer.writerow({"engine": engine, "query": query, **value})
    print("CPU: average busy logical CPUs during timed queries")
    print("| Engine | Q1 | Q2 | Q3 | Q4 | Q5 | All queries | Host CPU % |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for engine, values in engines.items():
        cpu = " | ".join(f"{values['queries'][query]['average_busy_logical_cpus']:.2f}" for query in QUERIES)
        overall = values["all_queries"]
        print(f"| {engine} | {cpu} | {overall['average_busy_logical_cpus']:.2f} | {overall['average_host_cpu_percent']:.1f}% |")
    print("\nMemory: post-query process-tree RSS sum; GiB, not a continuous peak")
    print("| Engine | Median RSS | Sampled min | Sampled max | Snapshots |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for engine, values in engines.items():
        memory = values["all_queries"]
        print(f"| {engine} | {memory['post_query_rss_median_gib']:.3f} | {memory['post_query_rss_min_gib']:.3f} | {memory['post_query_rss_max_gib']:.3f} | {memory['post_query_rss_sample_count']} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[2] / "docs/benchmark-results")
    parser.add_argument("--output-prefix", default="20261110-jsonbench-resource-usage")
    options = parser.parse_args()
    if Path(options.output_prefix).name != options.output_prefix:
        parser.error("output-prefix must not contain directories")
    main(options)
