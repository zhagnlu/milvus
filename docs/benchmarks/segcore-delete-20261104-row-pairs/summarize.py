#!/usr/bin/env python3
"""Validate this archive and render its report from measured CSVs."""

import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import statistics

from run_matrix import BATCHES, MODES, ROUNDS, ROWS, SEEDS, WIDTHS, layouts


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LAYOUTS = ["v2_stats_on", "v2_stats_off"]
MODE_LABELS = {"stream_same": "流式，同 TS", "stream_mixed": "流式，混合 TS", "load": "加载回放"}
TYPE_LABELS = {"int64": "int64", "varchar16": "VARCHAR16"}
LAYOUT_LABELS = {"v2_stats_on": "默认 V2：stats skip 开启", "v2_stats_off": "V2：stats skip 关闭，4 MiB 合并目标"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):
    return (row["type"], int(row["rows"]), row["layout"], row["mode"], int(row["batch"]), int(row["seed"]))


def ms(row, side):
    return float(row[side + "_total_us"]) / 1000


def main():
    plan = json.loads((HERE / "plan.json").read_text())
    assert plan["status"] == "measurements_completed"
    records = list(csv.DictReader((HERE / "matrix.csv").open()))
    expected = set(itertools.product(WIDTHS, ROWS, LAYOUTS, MODES, BATCHES, SEEDS))
    index = {key(r): r for r in records}
    assert len(records) == len(index) == len(expected) == 600
    assert set(index) == expected
    assert plan["source_sha256_before"] == plan["source_sha256_after"]
    assert all(digest(ROOT / p) == v for p, v in plan["source_sha256_after"].items())
    assert digest(HERE / "matrix.csv") == plan["matrix_sha256"]
    kernel_sha = digest(HERE / "benchmark-snapshot/delete_kernels.h")
    assert digest(HERE / "benchmark-snapshot/common/GallopingSearch.h") == plan["source_sha256_after"]["internal/core/src/common/GallopingSearch.h"]
    for run in plan["runs"]:
        assert run["exit_code"] == 0 and run["records"] == 15
        assert run["status"] == "completed_exit_0"
        raw = list(csv.DictReader((HERE / run["output"]).open()))
        assert len(raw) == 15
        metadata = (HERE / run["output"]).with_suffix(".metadata.txt").read_text()
        assert ("source_sha256=" + plan["source_sha256_after"][
            "internal/core/src/segcore/ChunkedSegmentSealedImpl.cpp"]) in metadata
        assert ("benchmark_sha256=" + plan["source_sha256_after"][
            "internal/core/benchmark/DeleteBatchBenchmark.cpp"]) in metadata
        assert ("kernel_sha256=" + kernel_sha) in metadata
        for r in raw:
            k = (r["type"], int(r["rows"]), run["layout"], r["mode"], int(r["batch"]), run["seed"])
            assert all(index[k][name] == value for name, value in r.items())
            assert int(r["chunks"]) == run["chunks"]
            assert sum(json.loads(index[k]["chunk_rows"])) == int(r["rows"])
            assert int(r["samples"]) == ROUNDS * max(1, min(8, 4096 // int(r["batch"])))
            assert float(r["old_total_us"]) > 0 and float(r["new_total_us"]) > 0
            assert 0 < float(r["round_min_speedup"]) <= float(r["speedup"]) <= float(r["round_max_speedup"])

    cells = []
    for kind, rows, layout, mode, batch in itertools.product(WIDTHS, ROWS, LAYOUTS, MODES, BATCHES):
        pair = [index[kind, rows, layout, mode, batch, seed] for seed in SEEDS]
        speedups = [float(r["speedup"]) for r in pair]
        deltas = [100 * (ms(r, "new") / ms(r, "old") - 1) for r in pair]
        if max(deltas) < -5:
            verdict = "提升"
        elif min(deltas) > 5:
            verdict = "退化"
        else:
            verdict = "接近或跨 seed 不稳定"
        first = pair[0]
        cells.append(dict(type=kind, rows=rows, layout=layout, chunks=int(first["chunks"]), mode=mode,
                          batch=batch, verdict=verdict, old_ms=ms(first, "old"), new_ms=ms(first, "new"),
                          cost_delta_pct=deltas[0], confirm_old_ms=ms(pair[1], "old"),
                          confirm_new_ms=ms(pair[1], "new"), confirm_cost_delta_pct=deltas[1],
                          min_seed_speedup=min(speedups), max_seed_speedup=max(speedups),
                          minimum_round_speedup=min(float(r["round_min_speedup"]) for r in pair),
                          current_lookup_enabled=batch >= 1024,
                          candidate_gate_enabled=(int(first["chunks"]) >= 64 and 1024 <= batch <= 10000)))
    with (HERE / "summary.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)

    def stats(selected):
        return dict(configurations=len(selected),
                    verdicts={v: sum(c["verdict"] == v for c in selected)
                              for v in ["提升", "退化", "接近或跨 seed 不稳定"]},
                    minimum_paired_median_speedup=min(c["min_seed_speedup"] for c in selected),
                    minimum_round_speedup=min(c["minimum_round_speedup"] for c in selected))

    environment = [json.loads(line) for line in (HERE / "environment.jsonl").read_text().splitlines()]
    assert len(environment) > 1
    pressure = [float(re.search(r"some avg10=([0-9.]+)", e["cpu_pressure"]).group(1)) for e in environment]
    background_busy = []
    for before, after in zip(environment, environment[1:]):
        active = total = 0
        for cpu, (ticks, idle) in after["cpu_total_idle_ticks"].items():
            if cpu == f"cpu{plan['cpu']}":
                continue
            old_ticks, old_idle = before["cpu_total_idle_ticks"][cpu]
            total += ticks - old_ticks
            active += ticks - old_ticks - (idle - old_idle)
        if total:
            background_busy.append(100 * active / total)
    environment_summary = dict(samples=len(environment),
                               max_cpu_some_avg10=max(pressure),
                               median_non_benchmark_cpu_busy_percent=statistics.median(background_busy),
                               max_non_benchmark_cpu_busy_percent=max(background_busy),
                               note="Other CPU usage includes this runner's compilation/tool activity; affinity is not exclusive CPU reservation.")
    harness_checks = {}
    for name, chunks in [("smoke", [4]), ("default-layout-smoke", [1, 4])]:
        smoke = list(csv.DictReader((HERE / (name + ".csv")).open()))
        smoke_expected = set(itertools.product(WIDTHS, chunks,
                                              ["uniform_unique", "repeated", "miss"],
                                              MODES, [100, 1000]))
        smoke_keys = {(r["type"], int(r["chunks"]), r["shape"], r["mode"], int(r["batch"]))
                      for r in smoke}
        assert len(smoke) == len(smoke_keys) == len(smoke_expected)
        assert smoke_keys == smoke_expected
        for r in smoke:
            assert int(r["rows"]) == 1000
            assert int(r["samples"]) == 3 * max(1, min(8, 4096 // int(r["batch"])))
        metadata = (HERE / (name + ".metadata.txt")).read_text()
        assert "kernel_sha256=" + kernel_sha in metadata
        assert "benchmark_sha256=" + plan["source_sha256_after"][
            "internal/core/benchmark/DeleteBatchBenchmark.cpp"] in metadata
        harness_checks[name] = dict(configurations=len(smoke), complete=True,
                                   note="Each CSV row is emitted only after all invocation oracle checks pass")

    validation = dict(validated_at=datetime.now(timezone.utc).isoformat(), records=len(records),
                      distinct_configurations=len(cells), rounds=ROUNDS, seeds=SEEDS,
                      timed_old_new_pairs=sum(int(r["samples"]) for r in records),
                      timed_paired_rounds=len(records) * ROUNDS,
                      source_unchanged=True, matrix_complete=True, raw_aggregate_match=True,
                      all_oracles_passed=True,
                      environment=environment_summary, harness_checks=harness_checks,
                      note="Oracle checks match count/hash per call; not end-to-end MVCC or memory verification",
                      groups={layout + ":" + mode: stats([c for c in cells if c["layout"] == layout and c["mode"] == mode])
                              for layout, mode in itertools.product(LAYOUTS, MODES)},
                      active_groups={layout + ":" + mode: stats([c for c in cells if c["layout"] == layout and c["mode"] == mode and c["current_lookup_enabled"]])
                                     for layout, mode in itertools.product(LAYOUTS, MODES)},
                      candidate_gate=stats([c for c in cells if c["candidate_gate_enabled"]]),
                      files={str(p.relative_to(HERE)): digest(p) for p in sorted(HERE.rglob("*"))
                             if p.is_file() and "busy-host" not in p.relative_to(HERE).parts
                             and (p.suffix in (".csv", ".py", ".cpp", ".h", ".jsonl") or p.name.endswith(".metadata.txt"))})
    (HERE / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")

    lines = ["# Delete：50 万～500 万行 × 100～10 万 pairs 实测", "",
             "日期：2026-11-04。覆盖用户指定的五档行数、五档 pairs；int64 与固定 16 字节 VARCHAR；默认 V2 和关闭 stats skip 后的合并布局；同 TS 流式、混合 TS 流式及加载回放。", "",
             "**chunk 数是有明确假设的存储布局估算，耗时是本机实际运行的排序＋查找组件测量。没有读取线上段 metadata，也没有测完整 Milvus Delete 接口。**", "",
             "首轮曾受到其他 C++/Rust 编译干扰，原始 600 条记录完整保留在 [busy-host](busy-host/README.md)，不纳入下面的正式结果。用户停止其他编译后，重新执行本报告全部两次 seed；正式结果来自这次低负载重测。", "",
             "## 1. chunk 如何估算", "",
             "假设 PK 列组为 row_id、timestamp、PK，无其他分区键/聚簇键；int64 按 24 B/行、VARCHAR16 按 36 B/行估算。假设一个逻辑文件，忽略额外缓冲/有效位开销。", "",
             "按当前 Parquet writer 的约 1 MiB row group 目标划分，剩余容量不足一行时仍放入一行，因此固定行宽模型的完整组约为 int64 43,691 行、VARCHAR16 29,128 行。末尾保留真实余数，不再把所有 chunk 均分。", "",
             "默认 V2 的 stats skip 开启：每组一个 cell/PK chunk。关闭时用 `R=max(1,floor(4MiB / floor(总估算字节/row group 数)))`，本矩阵所有配置均得 R=4，再按每 4 组合并。多文件会改变短尾及分组，V3 的投影大小也会改变结果；下表不能直接视为任意线上段的真实布局。", "",
             "| 行数 | int64，默认 V2 | int64，合并布局 | VARCHAR16，默认 V2 | VARCHAR16，合并布局 |",
             "|---:|---:|---:|---:|---:|"]
    for n in ROWS:
        counts = [len(chunks) for width in WIDTHS.values() for _, _, _, chunks in layouts(n, width)]
        lines.append(f"| {n // 10000} 万 | " + " | ".join(map(str, counts)) + " |")
    lines += ["", "切分规则及代码证据见 [storage → cell → PK chunk 分析](../segcore-delete-20261028/storage-cell-chunk.md)。所有具体行边界在 [plan.json](plan.json) 和 [matrix.csv](matrix.csv) 中归档。", "",
              "## 2. 测量口径与当前启用条件", "",
              "- 限定有序 PK 原始列查找路径，段内 PK 全局有序且物理 PK 唯一；删除 PK 随机、不重复且存在。字符串是固定 16 字节十六进制编码。混合 TS 与回放会按插入/删除时间戳过滤版本；B 是输入事件数，不保证等于最终删除行数。",
              "- 每项配置两次 seed（110401、110417），每次 21 轮，old/new 交替执行顺序；两次运行顺序执行，固定 CPU 2。共 300 项不同配置、600 条记录、37,800 组计时 old/new 调用配对。",
              "- CPU/编译器见每份 metadata；延续本机 i7-8700、GCC 11.4、`-O3 -DNDEBUG -std=c++20` 的组件口径。",
              "- old 为当前 PR 框架内的二分回退，不是原始 master 完整实现；new 强制 PK 排序和有序查找。两侧计入 TS 复制、排序、重排、定位及回调计算，不含公共 IDs 解析、存在性过滤、真实 pin/I/O、跳表/bitmap/快照更新、cgo、Go 层或 tSafe。",
              "- 字符串列模拟连续 payload＋uint32 offsets；未计真实 StringChunk 的边界检查和存储包装开销。StreamPush/LoadPush 的删除结构维护、整批时间戳统计等后续处理不在计时范围内。",
              "- 本次 benchmark 新增显式 chunk 行数参数，边界访问改为预计算的 offsets；生产优化源码未修改。不同日期报告的绝对耗时不应直接拼接为 A/B。",
              "- **100/1000 pairs 是强制新算法对照；当前 PR 的门槛仍是 B≥1024，这两档实际走二分回退。1 万/5 万/10 万在有序 PK 列路径会启用新算法；尚无 chunk 门槛或批量上限。**",
              "- 上次建议的 `C≥64 && 1024≤B≤10000` 尚未落地。按本次布局，仅默认 V2 的 int64 400/500 万行、VARCHAR16 200/400/500 万行，在 1 万 pairs 这档满足该候选。",
              "- 每次调用都核对匹配数量/哈希 oracle；这不替代 MVCC、并发和完整进程 RSS 验证。", "",
              f"每 5 秒记录一次环境，共 {environment_summary['samples']} 次；CPU 等待压力 `some avg10` 最高 {environment_summary['max_cpu_some_avg10']:.2f}%，除固定 benchmark CPU 外的其他 CPU 平均忙碌比例，中位数 {environment_summary['median_non_benchmark_cpu_busy_percent']:.2f}%、最高 {environment_summary['max_non_benchmark_cpu_busy_percent']:.2f}%。后者也包含本轮编译 benchmark 和工具调用的活动。见 [environment.jsonl](environment.jsonl)。", "",
              "固定 CPU 亲和性不代表独占核心。所有单轮极值保留在 CSV，没有剔除不利结果。两次 seed 的中位数与范围用于观察复测稳定性，不能当作置信区间或线上收益概率。", "",
              "## 3. 结果概览", "",
              "下表只统计 **当前 B≥1024** 的配置。判定规则：两次 seed 的耗时中位数均降低超过 5% 记“提升”，均增加超过 5% 记“退化”，其余保留为接近/不稳定；不是线上请求概率。", "",
              "| 布局 | 路径 | 配置数 | 提升 | 退化 | 接近/不稳定 | 最低配对中位加速比 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for layout, mode in itertools.product(LAYOUTS, MODES):
        s = validation["active_groups"][layout + ":" + mode]
        v = s["verdicts"]
        lines.append(f"| {LAYOUT_LABELS[layout]} | {MODE_LABELS[mode]} | {s['configurations']} | {v['提升']} | {v['退化']} | {v['接近或跨 seed 不稳定']} | {s['minimum_paired_median_speedup']:.2f}× |")

    lines += ["", "## 4. 全部耗时矩阵", "",
              "各单元格为 **旧耗时 → 新耗时（ms）；两次 seed 的配对中位加速比范围**。cost 固定取首测 seed=110401，避免挑选更好结果；复测 cost 与百分比见第 5 节。加速比大于 1 表示更快，小于 1 表示更慢；配对中位比值不必等于两侧耗时中位数相除。`†` 表示当前代码未启用的新算法对照。", ""]
    for layout, kind, mode in itertools.product(LAYOUTS, WIDTHS, MODES):
        lines += [f"### {LAYOUT_LABELS[layout]} / {TYPE_LABELS[kind]} / {MODE_LABELS[mode]}", "",
                  "| 行数 | chunks | 100 pairs† | 1000 pairs† | 1 万 pairs | 5 万 pairs | 10 万 pairs |",
                  "|---:|---:|---|---|---|---|---|"]
        for n in ROWS:
            group = [c for c in cells if c["layout"] == layout and c["type"] == kind and c["mode"] == mode and c["rows"] == n]
            by_batch = {c["batch"]: c for c in group}
            values = [f"{by_batch[b]['old_ms']:.3f} → {by_batch[b]['new_ms']:.3f}；{by_batch[b]['min_seed_speedup']:.2f}–{by_batch[b]['max_seed_speedup']:.2f}×" for b in BATCHES]
            lines.append(f"| {n // 10000} 万 | {group[0]['chunks']} | " + " | ".join(values) + " |")
        lines.append("")

    lines += ["## 5. 逐项结论与复测", "",
              "完整 300 项结果在 [summary.csv](summary.csv)，包括首测/复测的 old、new cost、耗时变化百分比、两次 seed 加速比范围、最低单轮比值、当前启用与候选门槛标记。", "",
              "下面保留所有未达到“两次 seed 均降低超过 5%”的配置，不因其中部分处于未启用批量而删除。百分比为 `100*(新耗时/旧耗时-1)`；正数表示耗时增加。", "",
              "| 布局 | PK | 行数 | C | pairs | 路径 | 首测 old→new/ms | 变化 | 复测 old→new/ms | 变化 | 结论 |",
              "|---|---|---:|---:|---:|---|---|---:|---|---:|---|"]
    for c in cells:
        if c["verdict"] != "提升":
            lines.append(f"| {c['layout']} | {c['type']} | {c['rows']} | {c['chunks']} | {c['batch']} | {MODE_LABELS[c['mode']]} | {c['old_ms']:.3f}→{c['new_ms']:.3f} | {c['cost_delta_pct']:+.1f}% | {c['confirm_old_ms']:.3f}→{c['confirm_new_ms']:.3f} | {c['confirm_cost_delta_pct']:+.1f}% | {c['verdict']} |")
    lines += ["", "## 6. 复现及证据边界", "",
              "```bash", "python3 docs/benchmarks/segcore-delete-20261104-row-pairs/run_matrix.py --cpu 2",
              "python3 docs/benchmarks/segcore-delete-20261104-row-pairs/summarize.py", "```", "",
              "重跑会覆盖本目录同名结果，需保留历史时先传 `--output-dir` 指向新目录。",
              "",
              "[plan.json](plan.json) 保存全部命令、输入边界及源码哈希；[matrix.csv](matrix.csv) 汇总全部原始记录；两个 seed 的子目录保存原始 CSV/编译环境；[validation.json](validation.json) 核对配置完整性、采样数、原始与汇总一致性及源码不变。[源码快照](benchmark-snapshot/README.md) 保存实际 benchmark 和提取内核，内核哈希与所有 metadata 核对一致。", "",
              "显式不等长边界 `[1,332,666,1]` 的 [36 项 smoke](smoke.csv)，以及兼容原有均分布局的 [72 项 smoke](default-layout-smoke.csv)，均覆盖两种 PK、两档批量、三个删除路径和存在/重复/未命中输入；每次调用的数量与哈希校验通过。它们用于核对 benchmark 新增参数及边界处理，不混入正式性能统计。", "",
              "本轮只覆盖这套 PK 分布、两种类型和模型布局。文件分片、额外列组字段、变长/长字符串、重复物理版本、真实缓存缺页、并发及其他机器可能改变收益。不得把模型 chunk 数称为线上实测，也不得把组件加速倍数当成完整 apply 或 tSafe 的收益。", ""]
    (HERE / "README.md").write_text("\n".join(lines))
    print(json.dumps({k: validation[k] for k in ["records", "distinct_configurations", "timed_old_new_pairs", "active_groups", "candidate_gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
