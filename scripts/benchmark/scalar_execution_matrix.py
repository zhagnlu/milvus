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
"""Run expression/backend/batch controls sequentially; preserve failed cases."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def cases(rows, batch_size=8192, diagnostic_full_batch=False):
    batch_size = min(batch_size, rows)
    numeric = ('float_in', 'double_in', 'bool_in', 'mixed_compare', 'float_arith', 'double_arith', 'roaring', 'bloom_int', 'bloom_string', 'array_element', 'array_length', 'array_any', 'array_all', 'array_eq', 'legacy_call', "unary", "eq", "ne", "le", "gt", "ge", "range", "in", "in64", "in1024",
               "arith", "arith_sub", "arith_mul", "arith_div", "arith_mod", "compare", "is_null")
    strings = ("string_eq", "string_prefix", "string_like")
    documents = ("json_unary", "json_range", "json_in", "json_contains",
                 "json_contains_all", "json_contains_any", "json_exists",
                 "json_arith", "json_rare")
    for expression in numeric + strings + documents:
        batches = (batch_size, rows) if diagnostic_full_batch else (batch_size,)
        for batch in dict.fromkeys(batches):
            yield expression, "raw", "sorted", batch
    for expression in ("unary", "eq", "ne", "le", "gt", "ge", "range", "in", "in64", "in1024", "is_null"):
        for index in ("sorted", "bitmap"):
            yield expression, "index", index, batch_size
    for expression in documents:
        yield expression, "stats", "sorted", batch_size
    for layout in ("raw", "mixed", "index"):
        yield "compound", layout, "sorted", batch_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--binary", type=Path,
                        help="Optional saved benchmark executable for reproducible runs")
    parser.add_argument("--rows", type=int, default=262144)
    parser.add_argument("--batch", type=int, default=8192,
                        help="Batch size matching the deployment configuration (default: 8192)")
    parser.add_argument("--diagnostic-full-batch", action="store_true",
                        help="Add Raw batch=rows controls for bottleneck investigation only; report separately")
    parser.add_argument("--rounds", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--clock", choices=("wall", "thread_cpu"), default="wall")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--only", nargs="+", help="Expression names to include")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.rows < 1 or args.batch < 1:
        parser.error("--rows and --batch must be positive")
    selected = [c for c in cases(args.rows, args.batch, args.diagnostic_full_batch)
                if not args.only or c[0] in args.only]
    if not selected:
        parser.error("No cases selected")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runner = Path(__file__).with_name("scalar_execution.py")
    records = []
    for repeat in range(args.repeats):
        # Reverse case order on alternating repeats; each case also alternates
        # old/new modes internally. Never benchmark concurrently with another case.
        for expression, layout, index, batch in selected[::1 if repeat % 2 == 0 else -1]:
            name = f"r{repeat}-{expression}-{layout}-{index}-b{batch}"
            dest = output / name
            command = [sys.executable, str(runner), "--output", str(dest),
                       "--expression", expression, "--case", layout,
                       "--index-type", index, "--batch", str(batch),
                       "--rows", str(args.rows), "--rounds", str(args.rounds),
                       "--warmup", str(args.warmup), "--nullable", "1", "--clock", args.clock]
            if args.cpu is not None:
                command += ["--cpu", str(args.cpu)]
            if args.build_dir:
                command += ["--build-dir", str(args.build_dir)]
            if args.binary:
                command += ["--binary", str(args.binary)]
            with (output / (name + ".log")).open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            record = {"name": name, "expression": expression, "layout": layout,
                      "index_type": index, "batch": batch, "repeat": repeat,
                      "diagnostic": layout == "raw" and batch > min(args.batch, args.rows),
                      "exit_code": result.returncode, "command": command}
            if result.returncode == 0:
                record["report"] = str(dest / "report.json")
                record["result"] = json.loads((dest / "report.json").read_text())["results"][0]
            records.append(record)
            (output / "matrix.json").write_text(json.dumps(records, indent=2) + "\n")
            print(name, "PASS" if not result.returncode else "FAILED", flush=True)
    header = [f"Clock: {args.clock}", "", "| Expression | Layout / index | Batch | Old total ms | New total ms | New / old | Old eval ms | New eval ms |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    summaries = {False: list(header), True: [
        "Diagnostic Raw batch=rows controls only; do not use as production performance results.",
        "", *header]}
    for record in records:
        lines = summaries[record["diagnostic"]]
        if "result" not in record:
            lines.append(f'| {record["expression"]} | {record["layout"]} / {record["index_type"]} | {record["batch"]} | FAILED | | | | |')
            continue
        r = record["result"]
        old, new = r["old"], r["new"]
        lines.append(f'| {r["expression"]} | {record["layout"]} / {record["index_type"]} | {record["batch"]} | '
                     f'{old["total"]["median_us"]/1000:.3f} | {new["total"]["median_us"]/1000:.3f} | '
                     f'{r["new_over_old"]:.2f} | {old["eval"]["median_us"]/1000:.3f} | {new["eval"]["median_us"]/1000:.3f} |')
    (output / "summary.md").write_text("\n".join(summaries[False]) + "\n")
    if any(r["diagnostic"] for r in records):
        (output / "diagnostic-summary.md").write_text("\n".join(summaries[True]) + "\n")
    failed = sum(r["exit_code"] != 0 for r in records)
    summary = subprocess.run([
        sys.executable, str(Path(__file__).with_name("scalar_execution_summary.py")),
        str(output / "matrix.json"), "--repeats", str(args.repeats)])
    print(f"{len(records)-failed}/{len(records)} cases passed; results: {output}")
    return 1 if failed or summary.returncode else 0


if __name__ == "__main__":
    raise SystemExit(main())
