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

"""Threaded native-client HTTP adapters for the opt-in QPS benchmark."""

import argparse
import datetime as dt
import decimal
import hashlib
import http.server
import json
from pathlib import Path
import socket

import jsonbench_compare as bench
import jsonbench_native_clients as native


def main(args):
    workload = json.loads(args.workload.read_text())
    queries = workload["queries"]
    anchor = mongo = None
    if args.engine == "duckdb":
        import duckdb
        anchor = duckdb.connect(str(args.run_dir / "duckdb.db"), read_only=True)
        anchor.execute("SET threads=16")
        anchor.execute("SET memory_limit='48GiB'")
        anchor.execute("SET TimeZone='UTC'")
        stored = hashlib.sha256()
        cursor = anchor.execute("SELECT data FROM bluesky ORDER BY rowid")
        count = 0
        while batch := cursor.fetchmany(10000):
            for (value,) in batch:
                stored.update(value.encode() + b"\n")
                count += 1
        if count != bench.ROWS or stored.hexdigest() != workload["dataset_ndjson_sha256"]:
            raise RuntimeError("DuckDB stored input differs from the prepared baseline")
    elif args.engine == "mongodb":
        mongo = native.mongo_connect()
        pipelines = native.mongo_pipelines()

    def encode(value):
        if isinstance(value, decimal.Decimal):
            return float(value)
        if isinstance(value, dt.datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=dt.timezone.utc)
            return value.isoformat()
        raise TypeError(type(value).__name__)

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.database = None
            if args.engine == "duckdb":
                # A DuckDB cursor is an independent connection to the same instance.
                self.database = anchor.cursor()
                self.database.execute("SET TimeZone='UTC'")
            elif args.engine == "postgresql":
                self.database = native.pg_connect()
                with self.database.cursor() as cursor:
                    cursor.execute("SET statement_timeout='90s'")
            elif args.engine in ("starrocks", "doris"):
                import pymysql
                self.database = pymysql.connect(host="127.0.0.1", port=9030, user="root", database="jsonbench",
                                                autocommit=True, read_timeout=100, connect_timeout=5,
                                                cursorclass=pymysql.cursors.DictCursor)
                with self.database.cursor() as cursor:
                    cursor.execute("SET time_zone='+00:00'")
                    cursor.execute("SET enable_query_cache=false")
                    if args.engine == "doris":
                        cursor.execute("SET enable_sql_cache=false")
                    cursor.execute("SHOW VARIABLES")
                    settings = {row["Variable_name"]: row["Value"] for row in cursor.fetchall()}
                    if str(settings["enable_query_cache"]).lower() != "false":
                        raise RuntimeError("Query cache is enabled in a measured adapter session")
                    if args.engine == "doris" and str(settings["enable_sql_cache"]).lower() != "false":
                        raise RuntimeError("SQL result cache is enabled in a measured adapter session")

        def finish(self):
            try:
                super().finish()
            finally:
                if self.database is not None:
                    self.database.close()

        def log_message(self, *unused):
            pass

        def respond(self, result):
            body = json.dumps(result, default=encode, allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.close_connection = True
            self.respond({"code": 0, "engine": args.engine, "threaded": True})

        def do_POST(self):
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                name = payload["query_name"]
                if self.path != bench.HTTP_PATH or payload["sql"] != queries[name] or payload["timezone"] != "UTC":
                    raise ValueError("Unexpected benchmark request")
                if args.engine == "duckdb":
                    cursor = self.database.execute(payload["sql"])
                    columns = [column[0] for column in cursor.description]
                    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
                elif args.engine == "postgresql":
                    with self.database.cursor() as cursor:
                        cursor.execute(payload["sql"])
                        columns = [column.name for column in cursor.description]
                        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
                elif args.engine in ("starrocks", "doris"):
                    with self.database.cursor() as cursor:
                        cursor.execute(payload["sql"])
                        rows = cursor.fetchall()
                else:
                    rows = list(mongo.bluesky.bluesky.aggregate(pipelines[name], allowDiskUse=True, maxTimeMS=90000))
                    if name == "Q4":
                        epoch = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
                        for row in rows:
                            row["first_post_ts"] = (epoch + dt.timedelta(microseconds=row.pop("first_post_us"))).isoformat()
                self.respond({"code": 0, "data": rows})
            except Exception as error:
                self.respond({"code": 1, "error_type": type(error).__name__, "message": str(error)})

    class Server(http.server.ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 128

    Server(("127.0.0.1", 19531), Handler).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("duckdb", "postgresql", "mongodb", "starrocks", "doris"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    main(parser.parse_args())
