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

"""Opt-in local PostgreSQL/MongoDB comparison through a common HTTP adapter.

Requires a prepared jsonbench_compare.py run and separately verified database
binaries. PostgreSQL runs as UID/GID 65534 in a fresh --postgres-state directory;
the benchmark driver must be allowed to change that directory's ownership.
"""

import argparse
import datetime as dt
import decimal
import http.server
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import tarfile
import time

import requests

import jsonbench_compare as bench


def native_cpu(pid):
    """Include live PostgreSQL children and CPU of reaped parallel workers."""
    states = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            values = path.read_text().rsplit(")", 1)[1].split()
            states[int(path.parent.name)] = (int(values[1]), sum(int(values[i]) for i in (11, 12, 13, 14)))
        except (OSError, ValueError, IndexError):
            continue
    descendants = {pid}
    while True:
        expanded = descendants | {child for child, (parent, _) in states.items() if parent in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    state = SINGLE_PROCESS_STATE(pid)
    state["cpu_seconds"] = sum(states[child][1] for child in descendants if child in states) / os.sysconf("SC_CLK_TCK")
    state["live_processes"] = len(descendants)
    state["memory_scope"] = "Root PID only; process_tree records live database workers. Shared RSS is not summed."
    tree = []
    for child in sorted(descendants):
        try:
            item = SINGLE_PROCESS_STATE(child)
            tree.append({"pid": child, "memory": item["memory"], "io": item["io"]})
        except OSError:
            continue
    state["process_tree"] = tree
    return state


SINGLE_PROCESS_STATE = bench.process_state


def mongo_pipelines():
    match = {"kind": "commit", "commit.operation": "create"}
    posts = {**match, "commit.collection": "app.bsky.feed.post"}
    return {
        "Q1": [{"$group": {"_id": "$commit.collection", "count": {"$sum": 1}}},
               {"$project": {"_id": 0, "event": "$_id", "count": 1}}, {"$sort": {"count": -1}}],
        "Q2": [{"$match": match}, {"$group": {"_id": "$commit.collection", "count": {"$sum": 1}, "users": {"$addToSet": "$did"}}},
               {"$project": {"_id": 0, "event": "$_id", "count": 1, "users": {"$size": "$users"}}}, {"$sort": {"count": -1}}],
        "Q3": [{"$match": {**match, "commit.collection": {"$in": ["app.bsky.feed.post", "app.bsky.feed.repost", "app.bsky.feed.like"]}}},
               {"$project": {"event": "$commit.collection", "hour": {"$hour": {"date": {"$toDate": {"$divide": ["$time_us", 1000]}}, "timezone": "UTC"}}}},
               {"$group": {"_id": {"event": "$event", "hour": "$hour"}, "count": {"$sum": 1}}},
               {"$project": {"_id": 0, "event": "$_id.event", "hour_of_day": "$_id.hour", "count": 1}},
               {"$sort": {"hour_of_day": 1, "event": 1}}],
        "Q4": [{"$match": posts}, {"$group": {"_id": "$did", "first_post_us": {"$min": "$time_us"}}},
               {"$sort": {"first_post_us": 1}}, {"$limit": 3},
               {"$project": {"_id": 0, "user_id": "$_id", "first_post_us": 1}}],
        "Q5": [{"$match": posts}, {"$group": {"_id": "$did", "min_us": {"$min": "$time_us"}, "max_us": {"$max": "$time_us"}}},
               {"$project": {"_id": 0, "user_id": "$_id", "activity_span": {"$divide": [{"$subtract": ["$max_us", "$min_us"]}, 1000]}}},
               {"$sort": {"activity_span": -1}}, {"$limit": 3}],
    }


def pg_connect():
    import psycopg2
    connection = psycopg2.connect(host="127.0.0.1", port=15432, user="nobody", dbname="postgres", connect_timeout=5)
    connection.autocommit = True
    with connection.cursor() as cursor:
        cursor.execute("SET TIME ZONE 'UTC'")
    return connection


def mongo_connect():
    from pymongo import MongoClient
    return MongoClient("mongodb://127.0.0.1:27018", serverSelectionTimeoutMS=1000, connectTimeoutMS=1000)


def worker(args):
    baseline = json.loads((args.run_dir / "comparison.json").read_text())
    if args.worker == "postgresql":
        connection = pg_connect()

        def execute(statement):
            with connection.cursor() as cursor:
                cursor.execute(statement)
                names = [column.name for column in cursor.description]
                return [dict(zip(names, row)) for row in cursor.fetchall()]
    else:
        connection = mongo_connect()
        pipelines = mongo_pipelines()
        names = {sql: name for name, sql in baseline["queries"].items()}

        def execute(statement):
            name = names[statement]
            rows = list(connection.bluesky.bluesky.aggregate(pipelines[name], allowDiskUse=True))
            if name == "Q4":
                epoch = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
                for row in rows:
                    row["first_post_ts"] = (epoch + dt.timedelta(microseconds=row.pop("first_post_us"))).isoformat()
            return rows

    def encode(value):
        if isinstance(value, decimal.Decimal):
            return float(value)
        return value.isoformat()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def log_message(self, *unused):
            pass

        def respond(self, value):
            encoded = json.dumps(value, default=encode, allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            self.close_connection = True
            self.respond({"code": 0})

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path != bench.HTTP_PATH or request.get("timezone") != "UTC":
                raise ValueError("unexpected benchmark request")
            try:
                self.respond({"code": 0, "data": execute(request["sql"])})
            except Exception as error:
                self.respond({"code": 1, "message": str(error)})

    http.server.HTTPServer(("127.0.0.1", 19531), Handler).serve_forever()


def launch(args, engine, label):
    if engine == "postgresql":
        bench.vacant(15432)
        base = args.postgres_state
        base.mkdir(parents=True, exist_ok=True)
        data, sockets = base / "data", base / "sockets"
        archive = args.run_dir / "postgres-state.tar.gz"
        marker_path = args.run_dir / "postgresql-loaded.json"
        if marker_path.exists():
            marker = json.loads(marker_path.read_text())
            archive = args.run_dir / marker.get("state_archive_file", archive.name)
        if not (data / "PG_VERSION").exists() and archive.exists():
            marker = json.loads(marker_path.read_text())
            if bench.digest(archive) != marker["state_archive_sha256"]:
                raise RuntimeError("PostgreSQL persisted state archive differs")
            with tarfile.open(archive) as stored:
                if any(not (item.name == base.name or item.name.startswith(base.name + "/")) for item in stored.getmembers()):
                    raise RuntimeError("PostgreSQL archive has unexpected paths")
                stored.extractall(base.parent, filter="data")
            for directory, dirs, files in os.walk(base):
                for path in [Path(directory), *(Path(directory) / name for name in dirs + files)]:
                    os.chown(path, 65534, 65534, follow_symlinks=False)
        if not (data / "PG_VERSION").exists() and marker_path.exists():
            # /tmp can disappear before the first stopped-state archive exists.
            marker_path.replace(args.run_dir / "postgresql-loaded-lost-tmp.json")
            bench.progress("PostgreSQL temporary state lost; reloading verified source")
        for path in (data, sockets):
            path.mkdir(exist_ok=True, mode=0o700)
            os.chown(path, 65534, 65534)
        prefix = ["setpriv", "--reuid=65534", "--regid=65534", "--clear-groups"]
        if not (data / "PG_VERSION").exists():
            subprocess.run(prefix + [str(args.postgres_bin / "initdb"), "-D", str(data), "-U", "nobody", "-A", "trust", "--no-locale", "--encoding=UTF8"],
                           cwd="/tmp", check=True)
        command = prefix + [str(args.postgres_bin / "postgres"), "-D", str(data), "-h", "127.0.0.1", "-p", "15432", "-k", str(sockets)]
    else:
        bench.vacant(27018)
        data = args.run_dir / "mongodb/data"
        data.mkdir(parents=True, exist_ok=True)
        command = [str(args.mongodb_binary), "--dbpath", str(data), "--bind_ip", "127.0.0.1", "--port", "27018",
                   "--setParameter", "internalQueryMaxAddToSetBytes=1073741824",
                   "--setParameter", "internalQueryPlannerGenerateCoveredWholeIndexScans=true",
                   "--wiredTigerCollectionBlockCompressor", "zstd"]
    with (args.run_dir / f"{label}-{engine}.log").open("w") as output:
        child = subprocess.Popen(command, cwd="/tmp", stdout=output, stderr=subprocess.STDOUT)
    child.benchmark_marker = args.run_dir / "active-native.json"
    bench.save(child.benchmark_marker, {"engine": engine, "pid": child.pid, "status": "running"})
    deadline = time.monotonic() + 90
    try:
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError(f"{engine} startup failed; see {label}-{engine}.log")
            try:
                if engine == "postgresql":
                    pg_connect().close()
                else:
                    with mongo_connect() as client:
                        client.admin.command("ping")
                bench.progress(f"{engine} ready, pid={child.pid}")
                return child
            except Exception:
                time.sleep(1)
        raise TimeoutError(f"{engine} readiness timeout")
    except BaseException:
        bench.stop(child)
        raise


def prepare(args, engine, dataset):
    marker = args.run_dir / f"{engine}-loaded.json"
    loading = args.run_dir / f"{engine}-loading.json"
    if marker.exists() and json.loads(marker.read_text())["ndjson_sha256"] != dataset["ndjson_sha256"]:
        raise RuntimeError(f"{engine} input checkpoint differs")
    if loading.exists() and json.loads(loading.read_text())["ndjson_sha256"] != dataset["ndjson_sha256"]:
        raise RuntimeError(f"{engine} ingestion checkpoint differs")
    if engine == "postgresql":
        connection = pg_connect()
        try:
            with connection.cursor() as cursor:
                if not marker.exists():
                    cursor.execute("SELECT to_regclass('bluesky')")
                    exists = cursor.fetchone()[0] is not None
                    if exists and not loading.exists():
                        raise RuntimeError("partial/unknown PostgreSQL table")
                    schema = (args.run_dir / "external-sources/json__postgresql__ddl.sql").read_text()
                    begin = time.monotonic()
                    if not exists:
                        bench.save(loading, {"ndjson_sha256": dataset["ndjson_sha256"], "schema": schema})
                        bench.progress("PostgreSQL loading 1000000 rows in one transaction")
                        connection.autocommit = False
                        try:
                            cursor.execute(schema)
                            with (args.run_dir / "first-million.ndjson").open() as source:
                                cursor.copy_expert("COPY bluesky FROM STDIN WITH (FORMAT csv, QUOTE E'\\x01', DELIMITER E'\\x02', ESCAPE E'\\x01')", source)
                            connection.commit()
                        except BaseException:
                            connection.rollback()
                            raise
                        finally:
                            connection.autocommit = True
                    cursor.execute("SELECT count(*) FROM bluesky")
                    if cursor.fetchone()[0] != bench.ROWS:
                        raise RuntimeError("PostgreSQL ingestion cardinality differs")
                    cursor.execute("VACUUM ANALYZE bluesky")
                    bench.save(marker, {"ndjson_sha256": dataset["ndjson_sha256"], "schema": schema,
                                        "insert_vacuum_seconds": time.monotonic() - begin})
                cursor.execute("SELECT count(*) FROM bluesky")
                if cursor.fetchone()[0] != bench.ROWS:
                    raise RuntimeError("PostgreSQL row count differs")
                settings = {}
                for key in ("server_version", "shared_buffers", "work_mem", "max_parallel_workers", "max_parallel_workers_per_gather", "jit", "TimeZone"):
                    cursor.execute("SHOW " + key)
                    settings[key] = cursor.fetchone()[0]
                return settings
        finally:
            connection.close()
    with mongo_connect() as client:
        collection = client.bluesky.bluesky
        if not marker.exists():
            exists = "bluesky" in client.bluesky.list_collection_names()
            if exists and not loading.exists():
                raise RuntimeError("partial/unknown MongoDB collection")
            if not exists:
                bench.save(loading, {"ndjson_sha256": dataset["ndjson_sha256"], "id_policy": "source row number starting at zero"})
                client.bluesky.create_collection("bluesky", storageEngine={"wiredTiger": {"configString": "block_compressor=zstd"}})
            collection.create_index([(field, 1) for field in ("kind", "commit.operation", "commit.collection", "did", "time_us")])
            begin = time.monotonic()
            total = collection.count_documents({})
            if total:
                first = collection.find_one({}, {"_id": 1}, sort=[("_id", 1)])["_id"]
                last = collection.find_one({}, {"_id": 1}, sort=[("_id", -1)])["_id"]
                if first != 0 or last != total - 1 or total > bench.ROWS:
                    raise RuntimeError("MongoDB partial input is not a contiguous source prefix")
            bench.progress(f"MongoDB ingestion resumes at {total}/{bench.ROWS}")
            batch = []
            with (args.run_dir / "first-million.ndjson").open() as source:
                for index, line in enumerate(source):
                    if index < total:
                        continue
                    document = json.loads(line)
                    if "_id" in document:
                        raise RuntimeError("source already has a MongoDB _id")
                    document["_id"] = index
                    batch.append(document)
                    if len(batch) == 5000:
                        collection.insert_many(batch)
                        total += len(batch)
                        batch.clear()
                        if total % 100000 == 0:
                            bench.progress(f"MongoDB inserted {total}/{bench.ROWS}")
                            bench.save(loading, {"ndjson_sha256": dataset["ndjson_sha256"], "inserted_rows": total,
                                                 "id_policy": "source row number starting at zero"})
            if batch or total != bench.ROWS:
                raise RuntimeError("MongoDB input cardinality differs")
            client.admin.command("fsync")
            bench.save(marker, {"ndjson_sha256": dataset["ndjson_sha256"], "rows": total, "insert_fsync_seconds": time.monotonic() - begin})
        if collection.count_documents({}) != bench.ROWS:
            raise RuntimeError("MongoDB stored cardinality differs")
        return {"version": client.server_info()["version"], "indexes": collection.index_information(), "pipelines": mongo_pipelines()}


def start_adapter(args, engine, label, session):
    bench.vacant(19531)
    with (args.run_dir / f"{label}-{engine}-adapter.log").open("w") as output:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--run-dir", str(args.run_dir), "--worker", engine],
                                 stdout=output, stderr=subprocess.STDOUT)
    child.benchmark_marker = args.run_dir / "active-adapter.json"
    bench.save(child.benchmark_marker, {"engine": engine, "pid": child.pid, "status": "running"})
    try:
        for _ in range(30):
            if child.poll() is not None:
                raise RuntimeError(f"{engine} adapter exited")
            try:
                if session.get("http://127.0.0.1:19531/healthz", timeout=1).ok:
                    return child
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError("adapter readiness timeout")
    except BaseException:
        bench.stop(child)
        raise


def main(args):
    if args.worker:
        worker(args)
        return
    if args.samples < 1 or args.warmups < 0:
        raise ValueError("invalid measurement counts")
    existing = json.loads((args.run_dir / "comparison.json").read_text())
    if bench.digest(args.run_dir / "first-million.ndjson") != existing["dataset"]["ndjson_sha256"]:
        raise RuntimeError("native comparison source fingerprint differs")
    if (bench.digest(bench.ROOT / "bin/milvus") != existing["milvus_binary_sha256"]
            or bench.digest(bench.ROOT / "cmake_build/src/libmilvus_core.so") != existing["milvus_core_sha256"]):
        raise RuntimeError("Milvus binaries differ from the paired baseline")
    report = {"recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "comparison_class": "same-host complete HTTP response via native-client adapters", "dataset": existing["dataset"],
              "queries": existing["queries"], "rounds": [], "runner_sha256": bench.digest(__file__),
              "milvus_binary_sha256": bench.digest(bench.ROOT / "bin/milvus"),
              "core_sha256": bench.digest(bench.ROOT / "cmake_build/src/libmilvus_core.so"),
              "postgres_binary_sha256": bench.digest(args.postgres_bin / "postgres"),
              "mongodb_binary_sha256": bench.digest(args.mongodb_binary),
              "shared_runner_sha256": bench.digest(bench.__file__),
              "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "host_cpu": Path("/proc/cpuinfo").read_text(),
              "host_meminfo": Path("/proc/meminfo").read_text(),
              "resume_policy": "Reuse completed engines in each round; rerun an interrupted engine with fresh warmups.",
              "warmups_per_query_per_round": args.warmups, "samples_per_query_per_round": args.samples,
              "qualifications": ["PostgreSQL uses upstream JSONB/LZ4 + compound expression index and default memory/parallel settings.",
                                 "MongoDB uses upstream BSON/zstd + compound index; Q4/Q5 aggregate integer microseconds to retain precision.",
                                 "Native-client HTTP adapters add transport/serialization overhead; their CPU is excluded from database CPU measurements.",
                                 "PostgreSQL CPU includes process descendants and their reaped workers; CPU is not comparable to launched-worker counts.",
                                 "Warm caches; same inherited CPU affinity; no common enforced process memory cap."]}
    checkpoint_path = args.run_dir / "native-comparison.json"
    if checkpoint_path.exists():
        previous = json.loads(checkpoint_path.read_text())
        for key in ("dataset", "queries", "runner_sha256", "shared_runner_sha256", "milvus_binary_sha256",
                    "core_sha256", "postgres_binary_sha256", "mongodb_binary_sha256",
                    "warmups_per_query_per_round", "samples_per_query_per_round"):
            if previous.get(key) != report[key]:
                raise RuntimeError(f"native resume checkpoint differs: {key}")
        report = previous
        bench.progress("Resuming native comparison from durable engine checkpoints")
    bench.save(checkpoint_path, report)
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"Content-Type": "application/json", "Accept-Type-Allow-Int64": "true"})
    for number, order in enumerate([["milvus", "postgresql", "mongodb"], ["mongodb", "postgresql", "milvus"]], 1):
        label = f"native-round-{number}"
        if len(report["rounds"]) < number:
            report["rounds"].append({"order": order, "engines": {}, "settings": {}, "shutdown_exit_codes": {},
                                     "completed_engines": [], "attempts": {}})
        rnd = report["rounds"][number - 1]
        for engine in order:
            if engine in rnd["completed_engines"]:
                bench.progress(f"{label} {engine}: already complete; skipping")
                continue
            attempt = rnd["attempts"].get(engine, 0) + 1
            rnd["attempts"][engine] = attempt
            attempt_label = f"{label}-attempt-{attempt}"
            report["progress"] = {"round": number, "engine": engine, "stage": "starting", "attempt": attempt}
            bench.save(checkpoint_path, report)
            bench.progress(f"{label} {engine}: starting attempt {attempt}")
            child = bench.start(args, engine, attempt_label) if engine == "milvus" else launch(args, engine, attempt_label)
            adapter = None
            try:
                if engine == "milvus":
                    layout = bench.load_milvus(args, session, existing["dataset"])
                    if layout != existing["rounds"][0]["milvus_segments"]:
                        raise RuntimeError("Milvus layout changed")
                    rnd["milvus_segments"] = layout
                else:
                    rnd["settings"][engine] = prepare(args, engine, existing["dataset"])
                    adapter = start_adapter(args, engine, attempt_label, session)
                if engine == "postgresql":
                    bench.process_state = native_cpu
                report["progress"]["stage"] = "measuring"
                bench.save(checkpoint_path, report)

                def checkpoint(results):
                    rnd["engines"][engine] = results
                    report["progress"]["completed_queries"] = list(results)
                    bench.save(checkpoint_path, report)

                rnd["engines"][engine] = bench.measure(args, child, engine, session, existing["queries"],
                                                      existing["dataset"]["oracle"], attempt_label, checkpoint=checkpoint)
                if engine == "milvus" and bench.segments() != layout:
                    raise RuntimeError("Milvus layout changed during timing")
            finally:
                bench.process_state = SINGLE_PROCESS_STATE
                session.close()
                if adapter:
                    bench.stop(adapter)
                rnd["shutdown_exit_codes"][engine] = bench.stop(child)
                if engine == "postgresql" and (args.run_dir / "postgresql-loaded.json").exists():
                    report["progress"]["stage"] = "archiving_postgres_state"
                    bench.save(checkpoint_path, report)
                    archive = args.run_dir / f"postgres-state-round-{number}-attempt-{attempt}.tar.gz"
                    temporary = archive.with_suffix(".tmp")
                    with tarfile.open(temporary, "w:gz", compresslevel=1) as stored:
                        stored.add(args.postgres_state, arcname=args.postgres_state.name)
                    with temporary.open("rb") as stream:
                        os.fsync(stream.fileno())
                    temporary.replace(archive)
                    marker = json.loads((args.run_dir / "postgresql-loaded.json").read_text())
                    marker["state_archive_sha256"] = bench.digest(archive)
                    marker["state_archive_file"] = archive.name
                    bench.save(args.run_dir / "postgresql-loaded.json", marker)
            rnd["completed_engines"].append(engine)
            report["progress"]["stage"] = "engine_complete"
            bench.save(checkpoint_path, report)
            bench.progress(f"{label} {engine}: durable checkpoint complete")
    report["summary"] = {}
    for name in existing["queries"]:
        medians = {engine: statistics.median([ms for rnd in report["rounds"] for ms in rnd["engines"][engine][name]["samples_ms"]])
                   for engine in ("milvus", "postgresql", "mongodb")}
        report["summary"][name] = {"median_ms": medians,
                                  "milvus_over_peer": {engine: medians["milvus"] / medians[engine] for engine in ("postgresql", "mongodb")}}
    report["progress"] = {"stage": "complete"}
    bench.save(checkpoint_path, report)
    bench.progress(json.dumps(report["summary"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--postgres-bin", type=Path, default=Path("/tmp/jsonbench-postgresql-18.6/bin"))
    parser.add_argument("--postgres-state", type=Path, default=Path("/tmp/jsonbench-postgresql-state-20261106"))
    parser.add_argument("--mongodb-binary", type=Path)
    parser.add_argument("--worker", choices=["postgresql", "mongodb"])
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    options = parser.parse_args()
    options.run_dir = options.run_dir.resolve()
    main(options)
