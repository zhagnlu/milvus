#!/usr/bin/env python3
# Licensed to the LF AI & Data foundation under one or more contributor
# license agreements. See the NOTICE file distributed with this work for
# additional information regarding copyright ownership. The ASF licenses
# this file to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run the opt-in scalar UT benchmark and retain machine-readable samples."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import xml.etree.ElementTree as ET


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=repo / "cmake_build")
    parser.add_argument("--binary", type=Path, help="Defaults to build-dir/unittest/scalar_execution_perf")
    parser.add_argument("--output", type=Path, required=True, help="New directory for XML, log and JSON")
    parser.add_argument("--rows", type=int, default=262144)
    parser.add_argument("--rounds", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--expression", choices=('float_in', 'double_in', 'bool_in', 'mixed_compare', 'float_arith', 'double_arith', 'roaring', 'bloom_int', 'bloom_string', 'array_element', 'array_length', 'array_any', 'array_all', 'array_eq', 'legacy_call', "compound", "unary", "eq", "ne", "le", "gt", "ge", "range", "in", "arith", "arith_sub", "arith_mul", "arith_div", "arith_mod", "compare", "in64", "in1024", "is_null", "string_eq", "string_in", "string_in_long", "string_range", "string_prefix", "string_like", "json_unary", "json_range", "json_in", "json_contains", "json_contains_all", "json_contains_any", "json_exists", "json_arith", "json_rare"), default="compound")
    parser.add_argument("--nullable", type=int, choices=(0, 1), default=1)
    parser.add_argument("--case", choices=("all", "raw", "mixed", "index", "stats"), default="all")
    parser.add_argument("--index-type", choices=("sorted", "bitmap"), default="sorted")
    parser.add_argument("--clock", choices=("wall", "thread_cpu"), default="wall")
    parser.add_argument("--cpu", type=int, help="Linux CPU affinity for the benchmark process")
    args = parser.parse_args()
    if not (1 <= args.rows <= 10000000 and 3 <= args.rounds <= 1000
            and 1 <= args.warmup <= 100 and 1 <= args.batch <= args.rows):
        parser.error("Invalid rows/rounds/warmup/batch; see the benchmark README")
    if args.expression in ('float_in', 'double_in', 'bool_in', 'mixed_compare', 'float_arith', 'double_arith', 'roaring', 'bloom_int', 'bloom_string', 'array_element', 'array_length', 'array_any', 'array_all', 'array_eq', 'legacy_call', "arith", "arith_sub", "arith_mul", "arith_div", "arith_mod", "compare") and args.case != "raw":
        parser.error("Typed/arithmetic benchmarks require --case raw")
    if args.expression.startswith("json_") and args.case not in ("raw", "stats"):
        parser.error("JSON benchmarks require --case raw or stats")
    if args.expression.startswith("string_") and args.case != "raw":
        parser.error("String benchmarks currently require --case raw")
    if args.case == "stats" and not args.expression.startswith("json_"):
        parser.error("Stats benchmarks require a JSON expression")
    binary = (args.binary or args.build_dir / "unittest/scalar_execution_perf").resolve()
    if not binary.is_file():
        parser.error(f"Build the scalar_execution_perf target first: {binary}")
    cache = {}
    for line in (args.build_dir / "CMakeCache.txt").read_text().splitlines():
        if line and not line.startswith(("#", "//")) and "=" in line:
            key, value = line.split("=", 1)
            cache[key.split(":", 1)[0]] = value
    if cache.get("CMAKE_BUILD_TYPE") not in ("Release", "RelWithDebInfo"):
        parser.error("Use a Release or RelWithDebInfo build for performance measurements")
    if cache.get("USE_ASAN") == "ON":
        parser.error("Use a build without ASAN for performance measurements")
    if args.cpu is not None:
        if args.cpu not in os.sched_getaffinity(0):
            parser.error("--cpu must belong to this process's allowed affinity")
        os.sched_setaffinity(0, {args.cpu})
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for name in ("rows", "rounds", "warmup", "batch", "expression", "nullable", "index_type", "clock"):
        env[f"MILVUS_SCALAR_PERF_{name.upper()}"] = str(getattr(args, name))
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    cases = {"all": (("Raw", "Mixed", "AllIndex"), {"raw", "b_index", "all_index"}),
             "raw": (("Raw",), {"raw"}), "mixed": (("Mixed",), {"b_index"}),
             "index": (("AllIndex",), {"all_index"}), "stats": (("Stats",), {"json_stats"})}
    suffixes, expected_layouts = cases[args.case]
    filters = ":".join(f"Paths/ScalarExecutionPerformanceTest.*/{s}" for s in suffixes)
    command = [str(binary), "--gtest_also_run_disabled_tests",
               f"--gtest_filter={filters}",
               f"--gtest_output=xml:{output / 'results.xml'}"]
    started = time.time()
    with (output / "run.log").open("w") as log:
        completed = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise SystemExit(f"UT failed (exit {completed.returncode}); inspect {output / 'run.log'}")
    tree = ET.parse(output / "results.xml").getroot()
    if (int(tree.get("failures", 0)) or int(tree.get("errors", 0))
            or int(tree.get("tests", 0)) != 2 * len(expected_layouts) or int(tree.get("skipped", 0))):
        raise SystemExit("UT reported failures; performance results are not accepted")
    results = [json.loads(prop.get("value")) for prop in tree.iter("property")
               if prop.get("name") == "scalar_performance"]
    if len(results) != len(expected_layouts) or {r["layout"] for r in results} != expected_layouts:
        raise SystemExit("Missing benchmark cases; check binary freshness and run.log")
    for result in results:
        expected_index = "JSON_STATS" if args.case == "stats" else "BITMAP" if args.index_type == "bitmap" else "STL_SORT"
        if (result.get("clock") != args.clock
                or result.get("expression_kind") != args.expression
                or result.get("rows") != args.rows
                or result.get("batch_size") != args.batch
                or result.get("nullable") != bool(args.nullable)
                or result.get("index_type") != expected_index
                or "physical_plans" not in result
                or "cleanup" not in result["new"]):
            raise SystemExit("Benchmark metadata differs from requested settings; check binary freshness")
    libraries = subprocess.check_output(["ldd", str(binary)], text=True)
    core = next((Path(line.split("=>", 1)[1].split(" (", 1)[0].strip())
                 for line in libraries.splitlines() if "libmilvus_core.so =>" in line), None)
    metadata = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "platform": platform.platform(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cpu_model": next((line.split(":", 1)[1].strip() for line in
                           Path("/proc/cpuinfo").read_text().splitlines()
                           if line.startswith("model name")), "unknown"),
        "build": {k: cache.get(k) for k in ("CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER",
                  "CMAKE_CXX_FLAGS", "CMAKE_CXX_FLAGS_RELEASE", "CMAKE_CXX_FLAGS_RELWITHDEBINFO", "USE_ASAN")},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True),
        "binary": str(binary), "binary_sha256": sha256(binary), "command": command,
        "environment": {k: v for k, v in env.items() if k.startswith("MILVUS_SCALAR_PERF_")
                        or k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LD_LIBRARY_PATH")},
        "linked_libraries": libraries,
        "core_library": str(core) if core else None,
        "core_library_sha256": sha256(core) if core else None,
        "elapsed_seconds": time.time() - started,
        "scope": "single-thread FilterBits on warm sealed columns; no ANN, MVCC, RPC or cold I/O",
    }
    report = {"metadata": metadata, "results": results}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["| Layout | Old median (us) | New median (us) | New / old | Old P95 (us) | New P95 (us) |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        old, new = r["old"]["total"], r["new"]["total"]
        lines.append(f'| {r["layout"]} | {old["median_us"]:.2f} | {new["median_us"]:.2f} | '
                     f'{r["new_over_old"]:.3f} | {old["p95_us"]:.2f} | {new["p95_us"]:.2f} |')
    table = "\n".join(lines) + "\n"
    (output / "summary.md").write_text(table)
    print(f"Clock: {args.clock}")
    print(table)
    print(f"Full samples, preparation/evaluation splits and environment: {output / 'report.json'}")


if __name__ == "__main__":
    main()
