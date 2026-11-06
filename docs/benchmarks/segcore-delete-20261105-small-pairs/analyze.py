#!/usr/bin/env python3
"""Validate the small-pair measurements and regenerate tables without rerunning."""

import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import statistics

from run import BATCHES, CHUNKS, CPU, HERE, MODES, ROUNDS, ROWS, SEEDS, SHAPES, TYPES


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):
    return (row["type"], int(row["rows"]), int(row["chunks"]), row["shape"], row["mode"], int(row["batch"]))


def main():
    plan = json.loads((HERE / "plan.json").read_text())
    assert plan["status"] == "completed"
    assert plan["source_before"] == plan["source_after"]
    expected = set(itertools.product(TYPES, ROWS, CHUNKS, SHAPES, MODES, BATCHES))
    snapshot = HERE.parent / "segcore-delete-20261104-row-pairs/benchmark-snapshot"
    kernel_sha = digest(snapshot / "delete_kernels.h")
    assert digest(snapshot / "common/GallopingSearch.h") == plan["source_after"]["internal/core/src/common/GallopingSearch.h"]
    assert digest(snapshot / "DeleteBatchBenchmark.cpp") == plan["source_after"]["internal/core/benchmark/DeleteBatchBenchmark.cpp"]
    records, runs = [], {}
    for job in plan["runs"]:
        assert job["exit_code"] == 0 and job["records"] == len(expected)
        path = HERE / job["output"]
        assert digest(path) == job["csv_sha256"]
        rows = list(csv.DictReader(path.open()))
        index = {key(r): r for r in rows}
        assert len(rows) == len(index) == len(expected)
        assert set(index) == expected
        metadata = path.with_suffix(".metadata.txt").read_text()
        assert "source_sha256=" + plan["source_after"]["internal/core/src/segcore/ChunkedSegmentSealedImpl.cpp"] in metadata
        assert "kernel_sha256=" + kernel_sha in metadata
        assert "benchmark_sha256=" + plan["source_after"]["internal/core/benchmark/DeleteBatchBenchmark.cpp"] in metadata
        for r in rows:
            assert int(r["samples"]) == ROUNDS * min(8, 4096 // int(r["batch"]))
            assert 0 < float(r["round_min_speedup"]) <= float(r["speedup"]) <= float(r["round_max_speedup"])
            records.append(dict(r, seed=job["seed"]))
        runs[job["seed"]] = index
    assert len(records) == 864
    with (HERE / "matrix.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    cells = []
    for k in sorted(expected):
        a, b = [runs[seed][k] for seed in SEEDS]
        cells.append(dict(type=k[0], rows=k[1], chunks=k[2], shape=k[3], mode=k[4], batch=k[5],
                          old_us=float(a["old_total_us"]), new_us=float(a["new_total_us"]),
                          confirm_old_us=float(b["old_total_us"]), confirm_new_us=float(b["new_total_us"]),
                          min_seed_speedup=min(float(r["speedup"]) for r in (a, b)),
                          max_seed_speedup=max(float(r["speedup"]) for r in (a, b)),
                          minimum_round_speedup=min(float(r["round_min_speedup"]) for r in (a, b)),
                          both_cost_medians_lower=all(float(r["new_total_us"]) < float(r["old_total_us"]) for r in (a, b))))
    with (HERE / "summary.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)

    def stats(selected):
        return dict(records=len(selected),
                    min_paired_median_speedup=min(float(r["speedup"]) for r in selected),
                    min_round_speedup=min(float(r["round_min_speedup"]) for r in selected),
                    median_cost_increases=sum(float(r["new_total_us"]) > float(r["old_total_us"]) for r in selected))

    env = [json.loads(line) for line in (HERE / "environment.jsonl").read_text().splitlines()]
    busy = []
    for a, b in zip(env, env[1:]):
        active = total = 0
        for cpu, (ticks, idle) in b["cpu_total_idle_ticks"].items():
            if cpu == f"cpu{CPU}":
                continue
            old_ticks, old_idle = a["cpu_total_idle_ticks"][cpu]
            total += ticks - old_ticks
            active += ticks - old_ticks - (idle - old_idle)
        if total:
            busy.append(100 * active / total)
    validation = dict(validated_at=datetime.now(timezone.utc).isoformat(),
                      records=len(records), configurations=len(cells),
                      timed_old_new_pairs=sum(int(r["samples"]) for r in records),
                      source_unchanged_during_run=True, matrix_complete=True,
                      per_call_oracle_passed=True, kernel_matches_20261104=True,
                      by_batch={str(batch): stats([r for r in records if int(r["batch"]) == batch]) for batch in BATCHES},
                      requested_batches_configurations=sum(c["batch"] in (10, 100, 500) for c in cells),
                      requested_batches_both_cost_medians_lower=all(c["both_cost_medians_lower"] for c in cells if c["batch"] in (10, 100, 500)),
                      environment=dict(samples=len(env), max_cpu_some_avg10=max(float(re.search(r"some avg10=([0-9.]+)", e["cpu_pressure"]).group(1)) for e in env),
                                       median_other_cpu_busy_pct=statistics.median(busy), max_other_cpu_busy_pct=max(busy)),
                      files={p.name: digest(p) for p in sorted(HERE.iterdir()) if p.is_file() and p.suffix in (".csv", ".py", ".txt", ".jsonl")})
    (HERE / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")

    lines = ["# Delete 小批量补测：1024 是否是必要下限", "",
             "测试日期：2026-11-05。显式指定 C=16/32，测 B=1/10/100/500；100 万、500 万、4400 万行；int64/VARCHAR16；随机不重复、少量候选键重复、未命中；同 TS 流式、混合 TS 流式、加载回放。", "",
             "**10/100/500 pairs 的 324 项配置，在两个 seed 中 old/new 耗时中位数均下降。这些结果不支持把 B≥1024 当作 C 已足够大时的必要启用条件。** 最弱场景只有约 3% 收益，不能保证每次调用都有提升。", "",
             "## 测量口径", "",
             "- 两个 seed 为 110501、110517，每项 21 轮，每轮 8 组 old/new 调用；共 432 项不同配置、864 条记录、145,152 组计时配对。", 
             "- 使用与 2026-11-04 完全相同的 benchmark、提取的排序/查找内核及 GallopingSearch；两个 seed 的 metadata 核对内核哈希一致。所有计时调用通过命中数量/哈希 oracle。", 
             "- 本次显式均匀切分成 16/32 个 chunk，用于隔离批量影响；不是这些行数必然产生的真实存储布局。物理 PK 唯一、有序；VARCHAR 固定 16 字节。", 
             "- old 是当前 PR 内的二分回退，new 强制执行排序和有序查找，均包含整理成本。生产代码仍要求 B≥1024，所以本次所有 new 都是强制算法对照；生产启用条件未修改。", 
             "- 同一 i7-8700、固定 CPU 2、GCC 11.4、O3；不含真实存储 pin/I/O、删除结构更新、完整 MVCC、并发或 RSS。亲和性不代表独占 CPU。", "",
             f"环境记录 {len(env)} 次，CPU 等待压力 some avg10 最高 {validation['environment']['max_cpu_some_avg10']:.2f}%；其他 CPU 的平均忙碌比例中位数 {statistics.median(busy):.2f}%，最高 {max(busy):.2f}%。见 [environment.jsonl](environment.jsonl)。", "",
             "## 全部已测场景的最低收益", "",
             "配对中位加速比为各轮 old/new 比值的中位数，不必等于两侧耗时中位数相除。", "",
             "| pairs | 记录数（含两个 seed） | 最低配对中位加速比 | 最低单轮比值 | 新耗时中位数增加的记录 |",
             "|---:|---:|---:|---:|---:|"]
    for batch, s in validation["by_batch"].items():
        lines.append(f"| {batch} | {s['records']} | {s['min_paired_median_speedup']:.4f}× | {s['min_round_speedup']:.4f}× | {s['median_cost_increases']} |")
    lines += ["", "B=1 的例外：500 万行、int64、C=16、未命中、加载回放，首测 old/new 耗时中位数为 0.593→0.597 μs，但配对中位比值为 1.0674×。两个统计量方向不一致，应视为接近持平、有波动，不能写成所有 B=1 情况都稳定提升。所有单轮不利极值均保留。", "",
              "## 100 万行、随机不重复输入的具体耗时", "",
              "单元格为旧→新耗时（μs）；取 seed=110501 的耗时中位数。复测 cost 和范围在 [summary.csv](summary.csv)。"]
    mode_labels = {"stream_same": "流式，同 TS", "stream_mixed": "流式，混合 TS", "load": "加载回放"}
    for chunks, kind in itertools.product(CHUNKS, TYPES):
        lines += ["", f"### {kind}，C={chunks}", "", "| 路径 | 1 pair | 10 pairs | 100 pairs | 500 pairs |", "|---|---:|---:|---:|---:|"]
        for mode in MODES:
            rows = [runs[SEEDS[0]][kind, 1000000, chunks, "uniform_unique", mode, batch] for batch in BATCHES]
            lines.append("| " + mode_labels[mode] + " | " + " | ".join(f"{float(r['old_total_us']):.3f}→{float(r['new_total_us']):.3f}" for r in rows) + " |")
    lines += ["", "## 对启用规则的影响", "",
              "1024 是之前保留的既有下限，不是本次测得的收益分界。C 已达到候选门槛后，10/100/500 不应仅因批量小而排除；移除该下限有性能证据支持，但不意味着取消 chunk 条件，也不意味着任意大批量回放没有退化。", "",
              "完整规则仍需在排序前统一选择，并覆盖真实 chunk 布局、门槛两侧和删除语义。若采用流式 C≥16、回放 C≥32 的分路径候选，本次小批量数据支持取消 B≥1024；该生产规则尚未实现和验证。若回放也统一用 C≥16，历史 150 万 pairs 的退化仍需另行处理。", "",
              "## 原始数据与复现", "",
              "[plan.json](plan.json) 保存命令、退出码及源码哈希；[matrix.csv](matrix.csv) 汇总 864 条原始记录；[summary.csv](summary.csv) 将两个 seed 对齐为 432 项配置；[validation.json](validation.json) 验证完整性与哈希。原始文件为 [seed-110501.csv](seed-110501.csv)、[seed-110517.csv](seed-110517.csv)，相邻 metadata 保存编译环境。", "",
              "本轮实际提取内核与[此前源码快照](../segcore-delete-20261104-row-pairs/benchmark-snapshot/README.md)相同，已逐份核对内核、benchmark 和 GallopingSearch 哈希。", "",
              "```bash", "python3 docs/benchmarks/segcore-delete-20261105-small-pairs/run.py", "python3 docs/benchmarks/segcore-delete-20261105-small-pairs/analyze.py", "```", "",
              "重跑会覆盖该目录同名结果，保留历史时先备份。分析脚本只校验和生成报告，不运行 benchmark。", ""]
    (HERE / "README.md").write_text("\n".join(lines))
    print(json.dumps({k: validation[k] for k in ("records", "configurations", "timed_old_new_pairs", "by_batch", "requested_batches_both_cost_medians_lower", "environment")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
