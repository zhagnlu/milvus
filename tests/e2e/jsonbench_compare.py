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

"""Opt-in, sequential same-host JSONBench comparison; no automatic downloads.

Install duckdb, requests and pymilvus in an isolated environment, then run:
  python3 tests/e2e/jsonbench_compare.py --run-dir .benchmarks/jsonbench-run \
      --data /path/to/file_0001.json.gz

This starts an isolated embedded Milvus, inserts/flushes/reloads one million
rows, and stops it before starting DuckDB. Two rounds reverse engine order.
It preserves all data and checkpoints; incomplete Milvus ingestion is refused.
DuckDB executes equivalent JSONPath SQL through a localhost HTTP adapter, so both
latencies include full HTTP response consumption. It uses native JSON columns,
without generated scalar columns or expression indexes. This is a warm,
single-client comparison, not an upstream ClickBench reproduction.
"""

import argparse
import collections
import datetime as dt
import gzip
import hashlib
import http.server
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA = "2e781e750b8826041763e48e87dbf6497162ca51501e67a098f64e2e011f0aa2"
ROWS = 1_000_000
HTTP_PATH = "/v2/vectordb/entities/sql_query"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def progress(message):
    print(f"{dt.datetime.now(dt.timezone.utc).isoformat()} {message}", flush=True)


def prepare(args):
    """Fingerprint and build an independent Python oracle from the exact input."""
    if digest(args.data) != SOURCE_SHA:
        raise RuntimeError("source gzip fingerprint differs from the recorded dataset")
    normalized = args.run_dir / "first-million.ndjson"
    marker = args.run_dir / "dataset.json"
    if marker.exists():
        info = json.loads(marker.read_text())
        if (info["source_sha256"] != SOURCE_SHA or info["rows"] != ROWS
                or digest(normalized) != info["ndjson_sha256"]):
            raise RuntimeError("prepared dataset fingerprint/count mismatch")
        return info
    counts, creates, hours = collections.Counter(), collections.Counter(), collections.Counter()
    users = collections.defaultdict(set)
    mins, maxs = {}, {}
    total = 0
    with gzip.open(args.data, "rb") as source, normalized.open("wb") as target:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            target.write(line.rstrip(b"\r\n") + b"\n")
            commit = item.get("commit") or {}
            event, did = commit.get("collection"), item.get("did")
            counts[event] += 1
            if item.get("kind") == "commit" and commit.get("operation") == "create":
                creates[event] += 1
                if did is not None:
                    users[event].add(did)
                ts = int(item["time_us"])
                if event in ("app.bsky.feed.post", "app.bsky.feed.repost", "app.bsky.feed.like"):
                    hours[event, ts // 3_600_000_000 % 24] += 1
                if event == "app.bsky.feed.post":
                    mins[did] = min(mins.get(did, ts), ts)
                    maxs[did] = max(maxs.get(did, ts), ts)
            total += 1
            if total == ROWS:
                break
    if total != ROWS:
        raise RuntimeError(f"expected {ROWS} input rows, got {total}")
    epoch = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    oracle = {
        "Q1": [{"event": e, "count": n} for e, n in counts.most_common()],
        "Q2": [{"event": e, "count": n, "users": len(users[e])} for e, n in creates.most_common()],
        "Q3": [{"event": e, "hour_of_day": h, "count": hours[e, h]}
               for e, h in sorted(hours, key=lambda key: (key[1], key[0]))],
        "Q4": [{"user_id": did, "first_post_ts": (epoch + dt.timedelta(microseconds=ts)).isoformat()}
               for did, ts in sorted(mins.items(), key=lambda pair: pair[1])[:3]],
        "Q5": [{"user_id": did, "activity_span": (maxs[did] - mins[did]) / 1000}
               for did in sorted(mins, key=lambda did: maxs[did] - mins[did], reverse=True)[:3]],
    }
    info = {"source_sha256": SOURCE_SHA, "ndjson_sha256": digest(normalized),
            "rows": total, "q2_intermediate_groups": sum(map(len, users.values())),
            "oracle": oracle}
    save(marker, info)
    progress(f"prepared exact dataset: {total} rows")
    return info


def check(name, actual, oracle):
    expected = oracle[name]
    if len(actual) != len(expected):
        raise RuntimeError(f"{name}: result length {len(actual)} != {len(expected)}")
    for got, want in zip(actual, expected):
        if got.keys() != want.keys():
            raise RuntimeError(f"{name}: output columns differ: {got}")
        for key, value in want.items():
            observed = got[key]
            if key == "first_post_ts":
                delta = dt.datetime.fromisoformat(observed.replace("Z", "+00:00")) - dt.datetime.fromisoformat(value)
                matches = abs(delta.total_seconds()) <= 0.000001
            elif key == "activity_span":
                matches = math.isclose(observed, value, rel_tol=0, abs_tol=0.0011)
            else:
                matches = observed == value
            if not matches:
                raise RuntimeError(f"{name} {key}: {observed!r} != {value!r}")


def request(session, port, path, payload):
    response = session.post(f"http://127.0.0.1:{port}/v2/vectordb{path}", json=payload, timeout=300)
    response.raise_for_status()
    result = response.json()
    if result.get("code") != 0:
        raise RuntimeError(f"{path}: {result}")
    return result.get("data")


def process_state(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    cpu = (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    memory = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:", "Threads:", "Cpus_allowed_list:")):
            key, value = line.split(":", 1)
            memory[key] = value.strip()
    io = {}
    try:
        io = {key: int(value) for key, value in (line.split(":") for line in Path(f"/proc/{pid}/io").read_text().splitlines())}
    except PermissionError:
        pass
    return {"cpu_seconds": cpu, "memory": memory, "io": io}


def vacant(port):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


def start(args, engine, label, preparing=False):
    import requests

    port = 19530 if engine == "milvus" else 19531
    vacant(port)
    env = os.environ.copy()
    if engine == "milvus":
        vacant(2379)
        vacant(2380)
        vacant(9091)
        conf = args.run_dir / "configs"
        conf.mkdir(exist_ok=True)
        for name in ("milvus.yaml", "glog.conf", "hook.yaml"):
            shutil.copyfile(ROOT / "configs" / name, conf / name)
        (conf / "user.yaml").write_text("""common:
  groupBy:
    maxGroups: 1000000
  enabledJSONShredding: true
  usingJSONShreddingForQuery: true
dataCoord:
  enableCompaction: PREPARING
  compaction:
    enableAutoCompaction: false
""".replace("PREPARING", "true" if preparing else "false"))
        (conf / "embedEtcd.yaml").write_text("""listen-client-urls: http://127.0.0.1:2379
advertise-client-urls: http://127.0.0.1:2379
listen-peer-urls: http://127.0.0.1:2380
initial-advertise-peer-urls: http://127.0.0.1:2380
initial-cluster: default=http://127.0.0.1:2380
quota-backend-bytes: 4294967296
auto-compaction-mode: revision
auto-compaction-retention: '1000'
""")
        lib = ROOT / "internal/core/output/lib"
        env.update({
            "LD_LIBRARY_PATH": f"{ROOT}/cmake_build/src:{lib}:{lib}64:{ROOT}/cmake_build/lib",
            "LD_PRELOAD": str(lib / "libjemalloc.so"), "DEPLOY_MODE": "STANDALONE",
            "MILVUSCONF": str(conf), "ETCD_USE_EMBED": "true", "ETCD_AUTH_ENABLED": "false",
            "ETCD_CONFIG_PATH": str(conf / "embedEtcd.yaml"),
            "ETCD_DATA_DIR": str(args.run_dir / "etcd"), "ETCD_ROOTPATH": "jsonbench-same-host",
            "COMMON_STORAGETYPE": "local", "LOCALSTORAGE_PATH": str(args.run_dir / "storage"),
            "MQ_TYPE": "rocksmq", "ROCKSMQ_PATH": str(args.run_dir / "rocksmq"),
            "LOG_LEVEL": "warn", "LOG_FILE_ROOTPATH": str(args.run_dir / "logs"),
            "QUERYNODE_MMAP_MMAPDIRPATH": str(args.run_dir / "mmap"), "LOCAL_STORAGE_SIZE": "100",
        })
        command = [str(ROOT / "bin/milvus"), "run", "standalone"]
    else:
        command = [sys.executable, str(Path(__file__).resolve()), "--run-dir", str(args.run_dir),
                   "--duckdb-worker", "--threads", str(args.threads)]
    log = (args.run_dir / f"{label}-{engine}.log").open("w")
    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    log.close()
    child.benchmark_marker = args.run_dir / "active-process.json"
    save(child.benchmark_marker, {"engine": engine, "pid": child.pid, "status": "running"})
    deadline = time.monotonic() + 180
    try:
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError(f"{engine} exited: see {label}-{engine}.log")
            try:
                response = requests.get(f"http://127.0.0.1:{9091 if engine == 'milvus' else port}/healthz", timeout=1)
                if response.ok:
                    progress(f"{engine} ready, pid={child.pid}")
                    return child
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError(f"{engine} startup timeout")
    except BaseException:
        stop(child)
        raise


def stop(child):
    if child.poll() is None:
        child.send_signal(signal.SIGTERM)
        try:
            child.wait(timeout=45)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
    progress(f"stopped pid={child.pid}, exit={child.returncode}")
    save(child.benchmark_marker, {"pid": child.pid, "status": "stopped", "returncode": child.returncode})
    return child.returncode


def segments(persistent=False):
    import grpc
    from pymilvus.grpc_gen import milvus_pb2, milvus_pb2_grpc

    with grpc.insecure_channel("127.0.0.1:19530") as channel:
        stub = milvus_pb2_grpc.MilvusServiceStub(channel)
        method = "GetPersistentSegmentInfo" if persistent else "GetQuerySegmentInfo"
        payload = getattr(milvus_pb2, method + "Request")(collectionName="bluesky")
        result = getattr(stub, method)(payload, timeout=30)
        if result.status.code or result.status.error_code:
            raise RuntimeError(str(result.status))
        return sorted([{"id": info.segmentID, "rows": info.num_rows, "state": int(info.state)}
                       for info in result.infos if info.num_rows > 0], key=lambda item: item["id"])


def stats_artifacts(args, layout):
    artifacts = {}
    for path in (args.run_dir / "storage/json_stats").rglob("meta.json"):
        segment = int(path.parent.parts[-2])
        metadata = json.loads(path.read_text())
        if any(item["id"] == segment and item["rows"] == metadata["num_rows"] for item in layout):
            artifacts[str(segment)] = {"metadata_file": str(path), "sha256": digest(path), **metadata}
    return artifacts


def load_milvus(args, session, dataset):
    def api(path, payload=None):
        return request(session, 19530, path, {"collectionName": "bluesky", **(payload or {})})

    marker = args.run_dir / "milvus-loaded.json"
    if marker.exists():
        if json.loads(marker.read_text())["ndjson_sha256"] != dataset["ndjson_sha256"]:
            raise RuntimeError("Milvus dataset marker mismatch")
    else:
        if "bluesky" in request(session, 19530, "/collections/list", {}):
            raise RuntimeError("incomplete/unknown bluesky collection: use a new run directory")
        api("/collections/create", {"schema": {"autoId": True, "fields": [
            {"fieldName": "pk", "dataType": "Int64", "isPrimary": True, "autoID": True},
            {"fieldName": "data", "dataType": "JSON"},
            {"fieldName": "dummy_vec", "dataType": "FloatVector", "elementTypeParams": {"dim": 2}},
        ]}})
        begin = time.monotonic()
        batch, total = [], 0
        with (args.run_dir / "first-million.ndjson").open() as source:
            for line in source:
                batch.append({"data": json.loads(line), "dummy_vec": [0.0, 0.0]})
                if len(batch) == 5000:
                    api("/entities/insert", {"data": batch})
                    total += len(batch)
                    batch.clear()
                    if total % 50000 == 0:
                        progress(f"inserted {total}/{ROWS}, elapsed={time.monotonic() - begin:.1f}s")
        if batch or total != ROWS:
            raise RuntimeError("unexpected ingestion row count")
        api("/indexes/create", {"indexParams": [{"fieldName": "dummy_vec", "indexName": "dummy_idx",
                                                 "metricType": "L2", "indexType": "FLAT"}]})
        api("/collections/flush")
        save(marker, {"ndjson_sha256": dataset["ndjson_sha256"], "rows": total,
                      "insert_and_flush_seconds": time.monotonic() - begin})
    api("/collections/load")
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        state = api("/collections/get_load_state")
        if state.get("loadState") == "LoadStateLoaded":
            layout = segments()
            if sum(item["rows"] for item in layout) == ROWS and all(item["state"] == 3 for item in layout):
                count = api("/entities/query", {"filter": "", "outputFields": ["count(*)"], "consistencyLevel": "Strong"})
                if count != [{"count(*)": ROWS}]:
                    raise RuntimeError(f"stored row count mismatch: {count}")
                return layout
        time.sleep(2)
    raise TimeoutError("million-row sealed layout not ready")


def duckdb_worker(args):
    import duckdb

    con = duckdb.connect(str(args.run_dir / "duckdb.db"))
    con.execute(f"SET threads={args.threads}")
    con.execute("SET memory_limit='48GiB'")
    con.execute("SET TimeZone='UTC'")
    if not con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='bluesky'").fetchone()[0]:
        begin = time.monotonic()
        con.execute("CREATE TABLE bluesky AS SELECT json AS data FROM read_ndjson_objects(?)",
                    [str(args.run_dir / "first-million.ndjson")])
        con.execute("CHECKPOINT")
        progress(f"DuckDB loaded in {time.monotonic() - begin:.3f}s")
    if con.execute("SELECT count(*) FROM bluesky").fetchone()[0] != ROWS:
        raise RuntimeError("DuckDB row count differs")
    # Verify the stored input, rather than trusting an equal row count on reuse.
    stored_hash = hashlib.sha256()
    scan = con.execute("SELECT data FROM bluesky ORDER BY rowid")
    while batch := scan.fetchmany(10000):
        for (value,) in batch:
            stored_hash.update(value.encode() + b"\n")
    dataset = json.loads((args.run_dir / "dataset.json").read_text())
    if stored_hash.hexdigest() != dataset["ndjson_sha256"]:
        raise RuntimeError("stored DuckDB JSON fingerprint differs from the input")
    save(args.run_dir / "duckdb-settings.json", {
        "version": duckdb.__version__, "threads": con.execute("SELECT current_setting('threads')").fetchone()[0],
        "memory_limit": con.execute("SELECT current_setting('memory_limit')").fetchone()[0],
        "timezone": con.execute("SELECT current_setting('TimeZone')").fetchone()[0],
        "stored_ndjson_sha256": stored_hash.hexdigest(),
    })

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def log_message(self, *unused):
            pass

        def respond(self, value):
            encoded = json.dumps(value, default=lambda obj: obj.isoformat(), allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            # The readiness client must not retain the only handler while idle.
            self.close_connection = True
            self.respond({"code": 0})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path != HTTP_PATH or payload.get("timezone") != "UTC":
                raise ValueError("unexpected benchmark request")
            try:
                result = con.execute(payload["sql"])
            except duckdb.Error as error:
                self.respond({"code": 1, "message": str(error)})
                return
            names = [column[0] for column in result.description]
            self.respond({"code": 0, "data": [dict(zip(names, row)) for row in result.fetchall()]})

    http.server.HTTPServer(("127.0.0.1", 19531), Handler).serve_forever()


def measure(args, child, engine, session, queries, oracle, label, checkpoint=None):
    port = 19530 if engine == "milvus" else 19531
    result = {}
    for name, sql in queries.items():
        payload = json.dumps({"sql": sql, "timezone": "UTC", "consistencyLevel": "Strong"}).encode()
        samples, cpu_samples = [], []
        for index in range(args.warmups + args.samples):
            before = process_state(child.pid)
            begin = time.perf_counter_ns()
            response = session.post(f"http://127.0.0.1:{port}{HTTP_PATH}", data=payload, timeout=300)
            raw = response.content
            elapsed = (time.perf_counter_ns() - begin) / 1_000_000
            after = process_state(child.pid)
            response.raise_for_status()
            data = json.loads(raw)
            if data.get("code") != 0:
                raise RuntimeError(f"{engine} {name}: {data}")
            check(name, data["data"], oracle)
            if index >= args.warmups:
                samples.append(elapsed)
                cpu_samples.append(after["cpu_seconds"] - before["cpu_seconds"])
        result[name] = {"median_ms": statistics.median(samples), "min_ms": min(samples), "max_ms": max(samples),
                        "samples_ms": samples, "process_cpu_seconds": cpu_samples,
                        "cpu_seconds_over_wall_seconds": sum(cpu_samples) / (sum(samples) / 1000),
                        "process_after": after, "response": data["data"], "oracle_passed_every_sample": True}
        save(args.run_dir / f"{label}-{engine}-timings.json", result)
        if checkpoint is not None:
            checkpoint(result)
        progress(f"{label} {engine} {name}: median {result[name]['median_ms']:.3f} ms")
    return result


def main(args):
    import requests

    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.duckdb_worker:
        duckdb_worker(args)
        return
    if args.samples < 1 or args.warmups < 0 or args.rounds < 1:
        raise ValueError("invalid measurement counts")
    dataset = prepare(args)
    queries = {"Q" + key: sql for key, sql in re.findall(
        r"var benchQ([1-5]) = `([^`]+)`", (ROOT / "tests/e2e/jsonbench_1m_test.go").read_text())}
    if len(queries) != 5:
        raise RuntimeError("expected five SQL acceptance queries")
    # DuckDB's arrow precedence differs from PostgreSQL's. Use its native
    # direct JSONPath accessor, preserving the bound paths and text values.
    def jsonpath(match):
        path = ".".join(re.findall(r"->>?'([^']+)'", match.group()))
        return "json_extract_string(data, '$." + path + "')"

    duckdb_queries = {name: re.sub(r"data(?:->>?'[^']+')+", jsonpath, sql)
                      for name, sql in queries.items()}
    report = {"recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "comparison_class": "same-host warm native-JSON HTTP single-client latency",
              "dataset": dataset, "queries": queries, "duckdb_queries": duckdb_queries,
              "warmups_per_query_per_round": args.warmups, "samples_per_query_per_round": args.samples,
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "host_cpu": subprocess.check_output(["lscpu"], text=True),
              "host_meminfo": Path("/proc/meminfo").read_text(),
              "milvus_binary_sha256": digest(ROOT / "bin/milvus"),
              "milvus_core_sha256": digest(ROOT / "cmake_build/src/libmilvus_core.so"),
              "runner_sha256": digest(__file__), "rounds": [],
              "limitations": ["Warm only; OS caches are not reset.",
                              "DuckDB uses a minimal Python HTTP adapter; Milvus uses its REST service.",
                              "Same CPU affinity, but no common enforced process memory cap.",
                              "Process CPU includes background threads; it is not query-only CPU.",
                              "Reader-hit instrumentation is separate from clean timing."]}
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"Content-Type": "application/json", "Accept-Type-Allow-Int64": "true"})
    if not (args.run_dir / "milvus-prepared.json").exists():
        child = start(args, "milvus", "prepare", preparing=True)
        try:
            load_milvus(args, session, dataset)
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                layout = segments(persistent=True)
                artifacts = stats_artifacts(args, layout)
                if (sum(item["rows"] for item in layout) == ROWS
                        and len(artifacts) == len(layout)):
                    save(args.run_dir / "milvus-prepared.json", {"layout": layout, "stats": artifacts})
                    progress("all persisted segments have JSON stats artifacts")
                    break
                progress(f"waiting for JSON stats: {len(artifacts)}/{len(layout)} segments")
                time.sleep(3)
            else:
                raise TimeoutError("JSON stats build timeout")
        finally:
            stop(child)
    for round_index in range(args.rounds):
        order = ["milvus", "duckdb"] if round_index % 2 == 0 else ["duckdb", "milvus"]
        round_result = {"order": order, "engines": {}, "shutdown_exit_codes": {}}
        label = f"round-{round_index + 1}"
        for engine in order:
            child = start(args, engine, label)
            try:
                if engine == "milvus":
                    layout = load_milvus(args, session, dataset)
                    save(args.run_dir / f"{label}-segments-before.json", layout)
                    # Releasing/reloading ensures complete persisted stats are selected.
                    request(session, 19530, "/collections/release", {"collectionName": "bluesky"})
                    layout = load_milvus(args, session, dataset)
                    progress(f"sealed layout: {layout}")
                engine_queries = queries if engine == "milvus" else duckdb_queries
                round_result["engines"][engine] = measure(args, child, engine, session, engine_queries, dataset["oracle"], label)
                if engine == "milvus":
                    after_layout = segments()
                    save(args.run_dir / f"{label}-segments-after.json", after_layout)
                    if after_layout != layout:
                        raise RuntimeError("segment layout changed during measurement")
                    round_result["milvus_segments"] = layout
                else:
                    round_result["duckdb_settings"] = json.loads((args.run_dir / "duckdb-settings.json").read_text())
            finally:
                round_result["shutdown_exit_codes"][engine] = stop(child)
        report["rounds"].append(round_result)
        save(args.run_dir / "comparison.json", report)
    summary = {}
    for name in queries:
        medians = {}
        for engine in ("milvus", "duckdb"):
            samples = [sample for rnd in report["rounds"] for sample in rnd["engines"][engine][name]["samples_ms"]]
            medians[engine] = statistics.median(samples)
        summary[name] = {"median_ms": medians, "milvus_over_duckdb": medians["milvus"] / medians["duckdb"]}
    report["summary"] = summary
    save(args.run_dir / "comparison.json", report)
    progress(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--threads", type=int, default=len(os.sched_getaffinity(0)))
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--duckdb-worker", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    options.run_dir = options.run_dir.resolve()
    if not options.duckdb_worker and options.data is None:
        parser.error("--data is required")
    main(options)
