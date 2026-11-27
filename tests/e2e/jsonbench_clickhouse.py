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

"""Opt-in paired Milvus/ClickHouse HTTP benchmark using a prepared JSONBench run.

Requires requests/pymilvus and an independently downloaded/verified ClickHouse
binary. All state is kept below --run-dir; neither database is downloaded here.
"""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import requests

import jsonbench_compare as bench
from jsonbench_native_clients import native_cpu


def cpu_state(pid):
    state = native_cpu(pid)
    return {"cpu_seconds": state["cpu_seconds"],
            "root_cpu_seconds": bench.process_state(pid)["cpu_seconds"],
            "pids": sorted(item["pid"] for item in state["process_tree"])}


def validate_cpu(rounds):
    for rnd in rounds:
        counters = rnd["engines"]["clickhouse"]["Q2"]["cpu_counters"]
        total = sum(item["after"]["cpu_seconds"] - item["before"]["cpu_seconds"] for item in counters)
        root = sum(item["after"]["root_cpu_seconds"] - item["before"]["root_cpu_seconds"] for item in counters)
        if total <= 0:
            raise RuntimeError("ClickHouse CPU source captured no server work")
        if any(len(item["before"]["pids"]) > 1 for item in counters) and total - root <= 0:
            raise RuntimeError("ClickHouse child server CPU was not captured")


def sql(session, statement, data=None):
    params = {"query": statement, "max_threads": 16,
              "max_memory_usage": 48 * 1024**3,
              "output_format_json_quote_64bit_integers": 0,
              "date_time_output_format": "iso", "session_timezone": "UTC"}
    response = session.post("http://127.0.0.1:18123", params=params, data=data, timeout=600)
    if not response.ok:
        raise RuntimeError(f"ClickHouse HTTP {response.status_code}: {response.text[:2000]}")
    response.raise_for_status()
    return response


def start(args, label):
    bench.vacant(18123)
    bench.vacant(19000)
    directory = args.run_dir / "clickhouse"
    directory.mkdir(exist_ok=True)
    for name in ("data", "tmp", "user_files", "format_schemas"):
        (directory / name).mkdir(exist_ok=True)
    (directory / "config.xml").write_text(f"""<clickhouse>
  <logger><level>warning</level><console>true</console></logger>
  <path>{directory}/data/</path><tmp_path>{directory}/tmp/</tmp_path>
  <user_files_path>{directory}/user_files/</user_files_path>
  <format_schema_path>{directory}/format_schemas/</format_schema_path>
  <listen_host>127.0.0.1</listen_host><http_port>18123</http_port><tcp_port>19000</tcp_port>
  <timezone>UTC</timezone><users_config>users.xml</users_config>
  <max_server_memory_usage>51539607552</max_server_memory_usage>
  <max_thread_pool_size>256</max_thread_pool_size>
  <background_pool_size>8</background_pool_size>
  <background_schedule_pool_size>16</background_schedule_pool_size>
  <background_buffer_flush_schedule_pool_size>4</background_buffer_flush_schedule_pool_size>
  <background_message_broker_schedule_pool_size>4</background_message_broker_schedule_pool_size>
  <background_distributed_schedule_pool_size>4</background_distributed_schedule_pool_size>
  <merge_tree><number_of_free_entries_in_pool_to_execute_mutation>4</number_of_free_entries_in_pool_to_execute_mutation>
    <number_of_free_entries_in_pool_to_lower_max_size_of_merge>4</number_of_free_entries_in_pool_to_lower_max_size_of_merge>
    <number_of_free_entries_in_pool_to_execute_optimize_entire_partition>4</number_of_free_entries_in_pool_to_execute_optimize_entire_partition>
  </merge_tree>
</clickhouse>
""")
    (directory / "users.xml").write_text("""<clickhouse>
  <users><default><password></password><networks><ip>127.0.0.1</ip></networks>
    <profile>default</profile><quota>default</quota></default></users>
  <profiles><default><max_threads>16</max_threads><max_memory_usage>51539607552</max_memory_usage></default></profiles>
  <quotas><default></default></quotas>
</clickhouse>
""")
    with (directory / f"{label}.log").open("w") as output:
        child = subprocess.Popen([str(args.clickhouse_binary), "server", "--config-file", str(directory / "config.xml")],
                                 stdout=output, stderr=subprocess.STDOUT)
    child.benchmark_marker = args.run_dir / "active-process.json"
    bench.save(child.benchmark_marker, {"engine": "clickhouse", "pid": child.pid, "status": "running"})
    deadline = time.monotonic() + 90
    try:
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError(f"ClickHouse exited; see {directory / (label + '.log')}")
            try:
                response = requests.get("http://127.0.0.1:18123/ping", timeout=1)
                if response.ok:
                    bench.progress(f"ClickHouse ready, pid={child.pid}")
                    return child
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError("ClickHouse startup timeout")
    except BaseException:
        bench.stop(child)
        raise


def prepare(args, session, dataset, original_schema):
    marker = args.run_dir / "clickhouse/loaded.json"
    if marker.exists():
        if json.loads(marker.read_text())["ndjson_sha256"] != dataset["ndjson_sha256"]:
            raise RuntimeError("ClickHouse dataset checkpoint differs")
    else:
        exists = sql(session, "EXISTS TABLE bluesky").text.strip()
        if exists != "0":
            raise RuntimeError("unrecognized/partial ClickHouse table; choose a new run directory")
        # Nullable preserves the acceptance query's missing-path NULL group.
        schema = original_schema.replace("commit.collection LowCardinality(String)",
                                         "commit.collection LowCardinality(Nullable(String))")
        schema = schema.replace("SETTINGS object_serialization_version", "SETTINGS allow_nullable_key = 1, object_serialization_version")
        sql(session, schema)
        begin = time.monotonic()
        with (args.run_dir / "first-million.ndjson").open("rb") as source:
            sql(session, "INSERT INTO bluesky SETTINGS min_insert_block_size_rows=1000000, min_insert_block_size_bytes=0 FORMAT JSONAsObject", source)
        sql(session, "OPTIMIZE TABLE bluesky FINAL")
        bench.save(marker, {"ndjson_sha256": dataset["ndjson_sha256"], "rows": bench.ROWS,
                            "insert_and_optimize_seconds": time.monotonic() - begin, "schema": schema})
    count = int(sql(session, "SELECT count() FROM bluesky").text)
    if count != bench.ROWS:
        raise RuntimeError(f"ClickHouse has {count} rows")
    sql(session, "SYSTEM STOP MERGES bluesky")
    return sql(session, "SELECT name, rows, bytes_on_disk FROM system.parts WHERE active AND database='default' AND table='bluesky' ORDER BY name FORMAT JSON").json()["data"]


def measure(args, child, session, queries, oracle, label):
    output = {}
    for name, statement in queries.items():
        samples, cpu, counters = [], [], []
        payload = (statement.rstrip("; \n") + " FORMAT JSON").encode()
        url = requests.Request("POST", "http://127.0.0.1:18123", params={
            "max_threads": 16, "max_memory_usage": 48 * 1024**3,
            "output_format_json_quote_64bit_integers": 0,
            "date_time_output_format": "iso", "session_timezone": "UTC"}).prepare().url
        for index in range(args.warmups + args.samples):
            before = cpu_state(child.pid)
            begin = time.perf_counter_ns()
            response = session.post(url, data=payload, timeout=300)
            raw = response.content
            elapsed = (time.perf_counter_ns() - begin) / 1_000_000
            after = cpu_state(child.pid)
            if before["pids"] != after["pids"]:
                raise RuntimeError("ClickHouse process tree changed during a sample")
            if after["cpu_seconds"] < before["cpu_seconds"]:
                raise RuntimeError("ClickHouse CPU counter moved backwards")
            if not response.ok:
                raise RuntimeError(f"ClickHouse HTTP {response.status_code}: {raw[:2000]!r}")
            rows = json.loads(raw)["data"]
            bench.check(name, rows, oracle)
            if index >= args.warmups:
                samples.append(elapsed)
                cpu.append(after["cpu_seconds"] - before["cpu_seconds"])
                counters.append({"before": before, "after": after})
        output[name] = {"median_ms": statistics.median(samples), "min_ms": min(samples), "max_ms": max(samples),
                        "samples_ms": samples, "process_cpu_seconds": cpu, "cpu_counters": counters,
                        "process_after": native_cpu(child.pid),
                        "cpu_seconds_over_wall_seconds": sum(cpu) / (sum(samples) / 1000),
                        "oracle_passed_every_sample": True, "response": rows}
        bench.save(args.run_dir / f"{label}-clickhouse-timings.json", output)
        bench.progress(f"{label} ClickHouse {name}: {output[name]['median_ms']:.3f} ms")
    return output


def main(args):
    if args.samples < 1 or args.warmups < 0:
        raise ValueError("samples must be positive and warmups nonnegative")
    if not args.output_prefix or Path(args.output_prefix).name != args.output_prefix:
        raise ValueError("output-prefix must be a filename prefix without directories")
    report_path = args.run_dir / f"{args.output_prefix}-comparison.json"
    if report_path.exists():
        raise FileExistsError(f"Preserving {report_path}; choose a new --output-prefix")
    existing = json.loads((args.run_dir / "comparison.json").read_text())
    if bench.digest(args.run_dir / "first-million.ndjson") != existing["dataset"]["ndjson_sha256"]:
        raise RuntimeError("Prepared dataset fingerprint differs from baseline")
    for path, expected in ((bench.ROOT / "bin/milvus", existing["milvus_binary_sha256"]),
                           (bench.ROOT / "cmake_build/src/libmilvus_core.so", existing["milvus_core_sha256"])):
        if bench.digest(path) != expected:
            raise RuntimeError(f"Baseline binary fingerprint differs: {path}")
    if sorted(os.sched_getaffinity(0)) != existing["cpu_affinity"]:
        raise RuntimeError("CPU affinity differs from baseline")
    source = args.run_dir / "external-sources"
    original_schema = (source / "json__clickhouse__create.sql").read_text()
    original_queries = (source / "json__clickhouse__queries.sql").read_text().splitlines()
    if len(original_queries) != 5:
        raise RuntimeError("expected five upstream ClickHouse queries")
    queries = dict(zip(existing["queries"], original_queries))
    queries["Q5"] = queries["Q5"].replace(
        "date_diff( 'milliseconds', min(fromUnixTimestamp64Micro(data.time_us)), max(fromUnixTimestamp64Micro(data.time_us)))",
        "(max(data.time_us) - min(data.time_us)) / 1000.0")
    report = {"recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(), "comparison_class": "paired same-host complete HTTP response",
              "dataset": existing["dataset"], "milvus_queries": existing["queries"], "clickhouse_queries": queries,
              "clickhouse_binary_sha256": bench.digest(args.clickhouse_binary), "runner_sha256": bench.digest(__file__),
              "clickhouse_version": subprocess.check_output([str(args.clickhouse_binary), "--version"], text=True).strip(),
              "milvus_binary_sha256": bench.digest(bench.ROOT / "bin/milvus"),
              "core_sha256": bench.digest(bench.ROOT / "cmake_build/src/libmilvus_core.so"),
              "warmups_per_query_per_round": args.warmups, "samples_per_query_per_round": args.samples, "rounds": [],
              "status": "running", "output_prefix": args.output_prefix,
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"),
              "native_cpu_runner_sha256": bench.digest(Path(__file__).with_name("jsonbench_native_clients.py")),
              "shared_runner_sha256": bench.digest(bench.ROOT / "tests/e2e/jsonbench_compare.py"),
              "clickhouse_cpu_measurement": "Process tree including watchdog/server and reaped children; not watchdog-only CPU",
              "qualifications": ["ClickHouse uses explicit typed JSON paths and workload-specific sorting, as in its upstream schema.",
                                 "commit.collection is Nullable to preserve missing-path SQL NULL; Q5 retains fractional milliseconds.",
                                 "Full result JSON is consumed; FORMAT Null from the upstream runner is not used.",
                                 "Warm caches; no common enforced process memory cap; CPU seconds include background threads."]}
    session = requests.Session()
    session.trust_env = False
    for number, order in enumerate([["milvus", "clickhouse"], ["clickhouse", "milvus"]], 1):
        label = f"{args.output_prefix}-round-{number}"
        rnd = {"order": order, "engines": {}, "shutdown_exit_codes": {}}
        for engine in order:
            child = bench.start(args, "milvus", label) if engine == "milvus" else start(args, label)
            try:
                if engine == "milvus":
                    session.headers.update({"Content-Type": "application/json", "Accept-Type-Allow-Int64": "true"})
                    layout = bench.load_milvus(args, session, existing["dataset"])
                    if layout != existing["rounds"][0]["milvus_segments"]:
                        raise RuntimeError("Milvus segment layout differs from the established baseline")
                    rnd["engines"][engine] = bench.measure(args, child, engine, session, existing["queries"], existing["dataset"]["oracle"], label)
                    if bench.segments() != layout:
                        raise RuntimeError("Milvus layout changed during timing")
                    rnd["milvus_segments"] = layout
                else:
                    session.headers.pop("Content-Type", None)
                    parts = prepare(args, session, existing["dataset"], original_schema)
                    rnd["engines"][engine] = measure(args, child, session, queries, existing["dataset"]["oracle"], label)
                    after = sql(session, "SELECT name, rows, bytes_on_disk FROM system.parts WHERE active AND database='default' AND table='bluesky' ORDER BY name FORMAT JSON").json()["data"]
                    if parts != after:
                        raise RuntimeError("ClickHouse parts changed during timing")
                    rnd["clickhouse_parts"] = parts
            finally:
                session.close()
                rnd["shutdown_exit_codes"][engine] = bench.stop(child)
        report["rounds"].append(rnd)
        bench.save(report_path, report)
    report["summary"] = {}
    for name in queries:
        medians = {engine: statistics.median([ms for rnd in report["rounds"] for ms in rnd["engines"][engine][name]["samples_ms"]])
                   for engine in ("milvus", "clickhouse")}
        report["summary"][name] = {"median_ms": medians, "milvus_over_clickhouse": medians["milvus"] / medians["clickhouse"]}
    report["schema"] = json.loads((args.run_dir / "clickhouse/loaded.json").read_text())["schema"]
    validate_cpu(report["rounds"])
    report["status"] = "complete"
    bench.save(report_path, report)
    bench.progress(json.dumps(report["summary"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--clickhouse-binary", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--output-prefix", default="clickhouse",
                        help="Prefix for new reports, timing files and logs; existing reports are preserved")
    options = parser.parse_args()
    options.run_dir = options.run_dir.resolve()
    options.clickhouse_binary = options.clickhouse_binary.resolve()
    main(options)
