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

"""Sweep segment rows and unique delete pairs using estimated chunk layouts.

Each PK chunk shares boundaries with a modeled system column group: 24 B/row
for int64 (row_id + timestamp + PK), 36 B/row for varchar16 (including offsets).
Chunks = ceil(rows * modeled row bytes / target bytes). Chunks are evenly split
by the underlying component benchmark; these are NOT real loaded file metadata.
This measures preparation + lookup, NOT full apply. Both paths are forced.
Batch sizes larger than segment rows are skipped to keep all PKs unique.
"""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


def integers(value):
    values = [int(v) for v in value.split(",")]
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("expected positive integers")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=integers,
                        default=integers("100000,500000,1000000,2000000,8000000,44000000"))
    parser.add_argument("--batches", type=integers,
                        default=integers("128,1024,8192,65536,262144,440000"))
    parser.add_argument("--cell-mib", type=integers, default=[4])
    parser.add_argument("--types", default="int64,varchar16")
    parser.add_argument("--modes", default="stream_same,load")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=880301)
    parser.add_argument("--cpu", type=int, default=2)
    args = parser.parse_args()
    widths = {"int64": 24, "varchar16": 36}
    kinds = args.types.split(",")
    if not set(kinds) <= widths.keys():
        parser.error("types must be int64 or varchar16")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(vars(args), output_dir=str(args.output_dir), runs=[])
    runner = Path(__file__).with_name("run_delete_batch_benchmark.py")
    with (args.output_dir / "matrix.csv").open("w") as combined:
        writer = None
        for mib in args.cell_mib:
            for kind in kinds:
                for rows in args.rows:
                    batches = [b for b in args.batches if b <= rows]
                    if not batches:
                        continue
                    target = mib * 1024 * 1024
                    chunks = (rows * widths[kind] + target - 1) // target
                    output = args.output_dir / f"{kind}-{rows}-{mib}MiB.csv"
                    command = [sys.executable, str(runner), "--output", str(output),
                               "--rows", str(rows), "--batches", ",".join(map(str, batches)),
                               "--chunks", str(chunks), "--types", kind,
                               "--shapes", "uniform_unique", "--modes", args.modes,
                               "--rounds", str(args.rounds), "--seed", str(args.seed),
                               "--cpu", str(args.cpu)]
                    print(f"START {kind}: rows={rows}, chunks={chunks}, target={mib}MiB, batches={batches}", flush=True)
                    subprocess.run(command, check=True)
                    manifest["runs"].append({"command": command, "chunks": chunks,
                                             "skipped_batches": [b for b in args.batches if b > rows]})
                    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
                    with output.open() as source:
                        for row in csv.DictReader(source):
                            row.update(cell_mib=mib, group_row_bytes=widths[kind],
                                       batch_fraction=int(row["batch"]) / rows)
                            if writer is None:
                                writer = csv.DictWriter(combined, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                    combined.flush()
                    print(f"DONE {output.name}", flush=True)


if __name__ == "__main__":
    main()
