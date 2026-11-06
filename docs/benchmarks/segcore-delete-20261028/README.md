# Segcore 删除查找测试归档

主要测试结果和当前结论见 [评审摘要（2026-10-30）](../segcore-delete-test-conclusions-20261030.zh.md)。

后续新增实测见 [100 万行、小批量 delete 对照（2026-10-30）](../segcore-delete-20261030-basic-1m/README.md)，数据保存在独立目录，不混入下方历史矩阵。

最新补测见 [50 万～500 万行 × 100～10 万 pairs（2026-11-04）](../segcore-delete-20261104-row-pairs/README.md)，包括按 V2 row group 规则估算的 chunk 数、两种布局、三个删除路径及两个 seed 的实际耗时。

小批量下限补测见 [C=16/32、B=1/10/100/500（2026-11-05）](../segcore-delete-20261105-small-pairs/README.md)，用于验证 1024 是否是必要下限。

分析、表格解释和候选启用策略见 [主报告](../segcore-delete-20261028.md)。整理日期 2026-10-28；保留历史目录日期与原始内容，不把归档日期当作所有测试的执行日期。

实际存储布局的逐层核查见 [storage、cell 和 PK chunk 的切分规则](storage-cell-chunk.md)，包含 V1/V2/V3 计算公式与测试估算的适用范围。

| 目录／文件 | 内容 |
|---|---|
| [lookup-only](lookup-only/delete-opt-lookup-bench.log) | 最早 40 条仅查找的测试；不含排序。源码与驱动一并保留 |
| [initial/delete-threshold-all.csv](initial/delete-threshold-all.csv) | 初期 1600 条汇总，来自 sweep 1260、boundary 320、large-sparse 20；不要与分文件重复计数 |
| initial 其他 CSV/metadata | 初期 smoke，以及 PK-only/system-group 布局纠偏结果；同名 metadata 记录运行参数 |
| [layouts/delete-layout-results.csv](layouts/delete-layout-results.csv) | 扩展布局 2786 条汇总；所有分组原始 CSV、metadata 和历史分析均保留 |
| [delete-size-grid-4MiB](delete-size-grid-4MiB/matrix.csv) | 第一轮段行数／批量矩阵，136 条 |
| [delete-size-grid-1MiB](delete-size-grid-1MiB/matrix.csv) | 1 MiB 布局对照，32 条 |
| [delete-size-grid-bulk](delete-size-grid-bulk/matrix.csv) | 150 万 pairs 扩展，12 条 |
| [sizes/delete-size-gate-bulk.csv](sizes/delete-size-gate-bulk.csv) | 16/32 chunks 的大批量对照，12 条 |
| [sizes/delete-size-results.csv](sizes/delete-size-results.csv) | 上述四部分合计 192 条；还保存验证信息、图表和独立 smoke 结果 |
| [delete-size-midpoints-20261027/matrix.csv](delete-size-midpoints-20261027/matrix.csv) | 1024～65536 加密矩阵原始 192 条 |
| [delete-size-midpoints-20261027/results.csv](delete-size-midpoints-20261027/results.csv) | 替换 12 条 15 轮复测后的 192 条；保留 original_speedup、retested、rounds |
| [recommended-gate-selected.csv](recommended-gate-selected.csv) | 流式 C≥16/回放 C≥32 且 B≥1024 的筛选记录，合计 639 条，含重复配置 |
| [recommended-gate-summary.json](recommended-gate-summary.json) | 上述候选规则的计数、最低加速比和批量范围 |
| [archive-manifest.json](archive-manifest.json) | 167 个从原位置复制的文件、原路径、字节数、SHA256；不包括本次新生成的 README/报告/筛选文件/工具 |
| benchmark-snapshot | 当前组件 benchmark 与两个 runner 的副本，供版本溯源；运行入口请用仓库 internal/core/benchmark 中的文件 |
| historical-tools | 原分析/画图脚本，仍含原 /tmp 路径；修改路径后才能直接复用 |
| [render_report_tables.py](render_report_tables.py) | 从归档 CSV 重新生成主报告的矩阵和反例表格，无第三方依赖、不运行 benchmark |

## 字段含义

- `old`：当前 PR 框架内强制二分，不是原始 master；`new`：强制整理后使用有序查找。
- `rows`：段行数；`chunks`：合成 PK 列的 chunk 数；`batch`：输入删除 pair 数。
- `old/new_prepare_us`：时间戳复制、排序与重排；`old/new_lookup_us`：查找；`old/new_total_us`：两阶段总计。各列分别取中位数。
- `speedup`：配对轮次 old/new 总耗时比值的中位数；大于 1 更快。min/max 是轮次极值，不是置信区间。
- `stream_same`：批内同一删除时间戳；`stream_mixed`：批内混合时间戳；`load`：回放，二分侧不排序。
- `uniform`：全段有放回随机抽样；`uniform_unique`：无放回随机抽样；`dense`：连续区间打乱；`ordered`：有序连续输入；`repeated`：少量候选 PK 重复；`repeated_half`：约半批候选 PK 重复；`miss`：段内范围中不存在的 PK。
- `cell_mib/group_row_bytes/batch_fraction`：布局目标大小、建模行宽和 B/N，不是实际加载文件的统计。

所有 .metadata.txt / manifest.json 中的 /tmp 路径仅记录原运行位置，对应数据已复制到本目录。用户提供的原型 Milvus A/B 只有报告数字，没有原始日志，见主报告第 5 节，不混入本地 CSV。

## 校验与重建

```bash
python3 docs/benchmarks/segcore-delete-20261028/render_report_tables.py
python3 docs/benchmarks/segcore-delete-20261028/verify_archive.py
```

此操作只重建文档表格、校验归档，不代表重新执行性能测试或并发正确性测试。
