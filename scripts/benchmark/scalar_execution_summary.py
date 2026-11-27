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
"""Summarize repeated scalar benchmarks without mixing paths or clock types."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median


def summarize(records, expected_repeats):
    groups = defaultdict(list)
    for record in records:
        key = (record["expression"], record["layout"], record["index_type"],
               record["batch"], record.get("diagnostic", False))
        groups[key].append(record)
    output = []
    for key, runs in groups.items():
        expression, layout, index_type, batch, diagnostic = key
        row = dict(expression=expression, layout=layout, index_type=index_type,
                   batch=batch, diagnostic=diagnostic, runs=len(runs))
        complete = (len(runs) == expected_repeats
                    and {r["repeat"] for r in runs} == set(range(expected_repeats))
                    and all(r["exit_code"] == 0 and "result" in r for r in runs))
        row["complete"] = complete
        if not complete:
            output.append(row)
            continue
        results = [r["result"] for r in runs]
        # Reject averaging incompatible data even if expression names match.
        for field in ("clock", "rows", "nullable", "batch_size", "physical_plans",
                      "comparison_kind", "matches", "result_cache", "storage_cache"):
            if any(r[field] != results[0][field] for r in results):
                raise ValueError(f"{key}: {field} differs between repeats")
            row[field] = results[0][field]
        row["label"] = results[0]["expression"]
        for mode in ("old", "new"):
            row[mode] = {}
            for phase in ("prepare", "eval", "cleanup", "total"):
                values = [r[mode][phase]["median_us"] for r in results]
                row[mode][phase] = dict(median_us=median(values),
                                        min_median_us=min(values), max_median_us=max(values),
                                        median_p95_us=median(r[mode][phase]["p95_us"] for r in results))
        ratios = [r["new_over_old"] for r in results]
        row["ratio"] = dict(median=median(ratios), minimum=min(ratios), maximum=max(ratios),
                            of_medians=row["new"]["total"]["median_us"] / row["old"]["total"]["median_us"])
        row["delta_us"] = {
            phase: median(r["new"][phase]["median_us"] - r["old"][phase]["median_us"]
                          for r in results)
            for phase in ("prepare", "eval", "cleanup", "total")}
        output.append(row)
    return output


def route(plan):
    backend = "+".join(plan["backends"]) or "direct scalar (no predicate slot)"
    return backend + f' / {plan["granularity"]} / {plan["effective_batches"]} batches'


def layout_name(row):
    if row["layout"] == "raw":
        return "Raw"
    if row["layout"] == "stats":
        return "JSON Stats"
    index = "STL_SORT" if row["index_type"] == "sorted" else "BITMAP"
    return f'{row["layout"]} / {index}'


def write_report(rows, path, diagnostic):
    selected = [r for r in rows if r["diagnostic"] == diagnostic]
    lines = ["Repeated scalar execution comparison", "",
             "Times are medians of per-run medians; the ratio divides these times, with the per-run ratio range in parentheses.",
             "Phase deltas are new minus old, in microseconds; phase medians need not sum to total.",
             "Execution differences include backend kernels and are not pure framework overhead.", ""]
    if diagnostic:
        lines += ["DIAGNOSTIC batch=rows controls; excluded from routine performance conclusions.", ""]
    for clock in sorted({r["clock"] for r in selected if r["complete"]}):
        lines += [f"Clock: {clock}", "",
                  "| Expression | Layout / index | Runs | Old ms | New ms | Ratio (range) | ΔPrepare µs | ΔEval µs | ΔCleanup µs |",
                  "|---|---|---:|---:|---:|---|---:|---:|---:|"]
        for row in selected:
            if not row["complete"] or row["clock"] != clock:
                continue
            ratio = row["ratio"]
            lines.append(f'| {row["label"]} | {layout_name(row)} | {row["runs"]} | '
                         f'{row["old"]["total"]["median_us"]/1000:.3f} | {row["new"]["total"]["median_us"]/1000:.3f} | '
                         f'{ratio["of_medians"]:.2f} ({ratio["minimum"]:.2f}–{ratio["maximum"]:.2f}) | '
                         f'{row["delta_us"]["prepare"]:.2f} | {row["delta_us"]["eval"]:.2f} | {row["delta_us"]["cleanup"]:.2f} |')
        lines += ["", "| Expression / layout / index | Old route | New route | Comparison |",
                  "|---|---|---|---|"]
        for row in selected:
            if row["complete"] and row["clock"] == clock:
                plans = row["physical_plans"]
                lines.append(f'| {row["expression"]} / {layout_name(row)} | '
                             f'{route(plans["old"])} | {route(plans["new"])} | {row["comparison_kind"]} |')
        lines.append("")
    for row in selected:
        if not row["complete"]:
            lines.append(f'INCOMPLETE: {row["expression"]} / {row["layout"]} / {row["index_type"]}; no performance conclusion.')
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--repeats", type=int, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    rows = summarize(json.loads(args.matrix.read_text()), args.repeats)
    dest = args.matrix.parent
    (dest / "aggregate.json").write_text(json.dumps(rows, indent=2) + "\n")
    write_report(rows, dest / "phase-summary.md", False)
    if any(r["diagnostic"] for r in rows):
        write_report(rows, dest / "diagnostic-phase-summary.md", True)
    return 0 if rows and all(r["complete"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
