# 从 storage 到 cell，再到删除路径的 PK chunk

核查日期：2026-10-28。依据当前工作区源码及本地 milvus-storage 检出 `0828aac90b4a0c27026a4e1210b58da9967539eb`，依赖工作区无修改。本次为源码链路核查，未读取运行中集群的真实段文件元数据，也未重新执行性能测试。旧读取结构文档记录的是更早基线，只作为分层说明参考。

## 1. 同名 chunk 必须分层

```text
写入数据，先决定列组（哪些字段存在一起）
  → 每个列组写一个或多个文件
  → Parquet 文件内部划分 row group
  → 加载端读取真实 row group 的行数和大小元数据
  → Translator 将一个或多个存储读取单元映射到一个缓存 cell
  → 加载并解码这些单元，构造一个 GroupChunk
  → GroupChunk 中每个字段得到一个子 Chunk，具有相同的行范围
  → PK 的 ProxyChunkColumn 暴露这组子 Chunk
  → 删除查找使用 pk_column->num_chunks()/chunk_row_nums()
```

Parquet 的 column chunk 是“一个 row group 中某个列的编码数据”；Arrow 的 array chunk 是读取结果的内存分段；segcore 的 `Chunk` 是转换后的字段数据块。三者不是一一对应的同一种对象。多个 Arrow arrays 会被 `create_group_chunk` 写入一个字段 Chunk，不会按 Arrow arrays 的数量再次增加 PK chunk 数。

本文后续的 cell 与 PK chunk 一一对应，限定为删除优化涉及的普通 Raw PK 列路径。当前 `ResolveVortexColumnGroupLocalFormats` 明确让含 PK 的列组走 Raw；其他字段的直接 Vortex 本地格式有独立的物理 cell 规划，不能把这里的公式套到所有 Column 实现。

## 2. 写入：先定列组，再切 row group

默认系统列组策略包含 row_id、timestamp、PK，还可以包含 partition key、clustering key；已有段可以沿用已有分组，local format 设置也会参与分组。因此“PK 列组恒为 24 B/行”不成立。24 B 只是三个 int64 字段、无其他相关字段时的建模行宽。

证据：[分组策略](../../../internal/storagecommon/split_policy.go)、[默认策略顺序](../../../internal/storagecommon/column_group_splitter.go)、[沿用已有分组](../../../internal/flushcommon/syncmgr/task.go)。

当前 Parquet writer 的 `DEFAULT_MAX_ROW_GROUP_SIZE = 1024 * 1024`，即 1 MiB。切分逻辑：

1. `Write` 用 `arrow::util::ReferencedBufferSize(record)` 估算传入列组 RecordBatch 的数据量并缓存。
2. `Flush` 计算该 batch 的平均每行大小 `batch_size / rows`，按当前 row group 剩余容量计算可放多少行，通过 `Slice` 切 batch。
3. 每次至少放一行，因此 row group 可以略超目标；不能切开单行。
4. 未满的尾部保留给下次 Flush，Close 时再写出最后一组；每个关闭的文件可能产生自己的短尾组。
5. `write_row_group` 写入行数、估算 memory_size 和行偏移，Close 将它们保存到文件 KV metadata。文件压缩后的字节数不是这个切分依据。

证据：[常量](../../../cmake_build/thirdparty/milvus-storage/milvus-storage-src/cpp/include/milvus-storage/common/config.h)、[ParquetFileWriter::Write/Flush/Close](../../../cmake_build/thirdparty/milvus-storage/milvus-storage-src/cpp/src/format/parquet/parquet_writer.cpp)。V2 Packed writer 先按列组拆 RecordBatch，再分别交给这些 writer；V3 Parquet 路径也使用该格式 writer。文件滚动、写入缓冲、对象存储 multipart 大小是另一层控制，不等于 row group 或 segcore cell 大小。

故 `1 MiB / 24 ≈ 43,690 行` 只能帮助估算；实际行数应读取文件 metadata，尤其是变长字段、短尾文件、历史文件和外部文件。

## 3. V1：一个字段文件对应一个 cell/chunk

`storagev1translator::ChunkTranslator` 对每个 `FileInfo` 建立一个 cell：

```text
C = PK 字段 file_infos.size()
rows[c] = file_infos[c].row_count
prefix[c+1] = prefix[c] + rows[c]
```

加载该文件得到的多个 Arrow batch 会合成一个 Chunk。文件内部即使有多个 Parquet row group，这条 Translator 路径也不按它们拆 segcore chunk。`virtual_chunk_config` 只是辅助行号定位，不改变实际 chunk 数。

证据：[ChunkTranslator 构造、num_cells、get_cells](../../../internal/core/src/segcore/storagev1translator/ChunkTranslator.cpp)。

## 4. V2：先读每个文件的 row group metadata，再按策略分组

`PlanGroupChunkPacking` 读取真实文件元数据；`GroupChunkTranslator` 得到每个文件的 row group 数，以及每组的 `row_num()`、`memory_size()`。

设文件 f 有 `G[f]` 个 row group，总数 `M = sum(G[f])`，第 i 组估算字节数 `S[i]`。目标 `T = cellTargetSizeBytes`，默认 4 MiB。

**默认开启 `common.parquetStatsSkipIndex.enabled`：**

```text
R = 1                         # 每个 cell 的 row group 数
C = sum(G[f])                 # 每个 row group 恰好一个 cell
```

该配置本身决定 1:1 布局；即使缺少可用的 footer 统计，也不会因此自动退回合并。此时 4 MiB 目标不参与分组，必须统计真实 row group 数。

**关闭该配置、允许合并时：**

```text
avg = floor(sum(S[i]) / M)
R = max(1, floor(T / avg))     # avg <= 0 或无输入时实现回退为 1
C = sum_f ceil(G[f] / R)       # 每个文件单独分组，不跨文件
```

每个文件按原顺序以 R 组为一批，最后一批可不足 R。注意这是“先取全体平均大小，再按固定组数切”，不是逐个 cell 累加字节凑到 4 MiB。若平均 1.05 MiB，R=3；不能固定认为 R=4。遇到大小不均的 row group，实际 cell 可以超过 4 MiB。

每个 cell 的行数是它所覆盖各 row group 行数之和；`chunk_memory_size_` 是它们估算字节数之和。源码：[GroupCTMeta::ComputeRowGroupsPerCell](../../../internal/core/src/segcore/storagev2translator/GroupCTMeta.h)、[GroupChunkTranslator 构造](../../../internal/core/src/segcore/storagev2translator/GroupChunkTranslator.cpp)、[加载入口和 packing plan](../../../internal/core/src/segcore/ChunkedSegmentSealedImpl.cpp)。

## 5. Manifest/V3 Raw：投影后的逻辑存储 chunk 序列，再合并 cell

`LoadColumnGroup` 根据本次需要的字段构造 ChunkReader，`ManifestGroupTranslator` 获取：

- `get_chunk_rows()`：各逻辑存储 chunk 的行数。
- `get_chunk_column_estimated_size()`：各字段、各逻辑存储 chunk 的估算大小，优先按本次投影字段求和。
- 不可用时按代码选择全组估算、采样行宽等 fallback；因此不能始终拿写入时完整列组行宽计算加载后的 cell。

Parquet 情况下，storage 的 `ColumnGroupReader` 遍历列组的所有文件，将 row group 展平。如果 manifest 文件条目只暴露物理文件的 `[start_index,end_index)`，则先对 row group 求交集；逻辑 chunk 行数只保留交集部分，估算大小也按行数比例缩放。空交集被跳过。因此 storage API 这里的 chunk 不一定是一个完整物理 row group。

Parquet 字段估算也不是重新按 C++ 类型宽度精确求和：当前 reader 会结合记录的 row group memory_size 和各列的 Parquet uncompressed-size 权重分配估算。它与最终 Raw Chunk 实际内存大小仍有区别。

设展平后的有效逻辑存储 chunk 总数为 M，本次投影估算为 `P[i]`：

```text
avg = floor(sum(P[i]) / M)
R = max(1, floor(T / avg))
C = ceil(M / R)
cell[c] = logical_chunks[c*R : min((c+1)*R, M)]
```

该层没有像 V2 那样按物理文件重置分组，**一个 cell 可以覆盖前一文件尾部和后一文件头部**。下层 reader 用各逻辑 chunk 记录的 file_index 分别读取。源码里 “single data source” 指一个统一 ChunkReader，并不表示只有一个物理文件。

当前 V2 的 stats-skip 1:1 开关没有传入这条 ManifestGroupTranslator 分组逻辑。即使物理格式不是 Parquet，也应使用 ChunkReader 提供的实际单元和元数据，不能假定都是 1 MiB Parquet row group。

证据：[ManifestGroupTranslator](../../../internal/core/src/segcore/storagev2translator/ManifestGroupTranslator.cpp)、[storage ColumnGroupReader 的文件范围求交与 chunk 元数据](../../../cmake_build/thirdparty/milvus-storage/milvus-storage-src/cpp/src/format/column_group_reader.cpp)、[Parquet 字段大小估算](../../../cmake_build/thirdparty/milvus-storage/milvus-storage-src/cpp/src/format/parquet/parquet_format_reader.cpp)、[MakeChunkReaderFactory](../../../internal/core/src/segcore/memory_planner.cpp)。

## 6. 从 cell 到最终 PK Chunk：边界不会按字节再次切分

Translator 根据 cell 已确定的 row group 范围读 Arrow 数据，将同一字段的 arrays 汇集，调用 `create_group_chunk`。每个字段创建一个 Chunk；字段数据在内存或 mmap 区域内按对齐要求排列，同一 GroupChunk 内字段行数一致。

这一步会跳过 row_id 等不保留字段，也可能做类型规范化；因此“写入端估算 4 MiB”“规划时估算 4 MiB”“最终分配 4 MiB”不是同一个承诺。它不会为了让 PK 自身达到 4 MiB 再把 PK 重新分组。

最终关系：

```text
translator.num_cells()
  = CacheSlot 的 cell 数
  = ChunkedColumnGroup.num_chunks()
  = PK ProxyChunkColumn.num_chunks()

PK chunk_row_nums(c)
  = num_rows_until_chunk[c+1] - num_rows_until_chunk[c]
  = cell 覆盖的逻辑存储单元行数之和
```

这统计的是列的全部逻辑 chunk，包括尚未驻留缓存的块，不是当前 pin/命中块数。对于尚未初始化的 lazy 列组，`num_chunks()` 会进入 `EnsureMaterialized()` 创建 translator/cache slot，可能读取 metadata；不能默认它在所有状态下都是零开销。删除策略应优先复用已建立的布局，避免为了判断门槛引入新的初始化或全量加载。

证据：[ChunkWriter::create_group_chunk](../../../internal/core/src/common/ChunkWriter.cpp)、[ChunkedColumnGroup / ProxyChunkColumn](../../../internal/core/src/mmap/ChunkedColumnGroup.h)。

## 7. 一个可手算的布局例子

假设两个文件各有 5 个有效 row group；每组估算恰好 1 MiB、每组 10,000 行，目标 4 MiB。这是解释分组算法的例子，不是 int64 行宽模型。

| 模式 | cell 切分 | PK chunk 数 | 各 chunk 行数 |
|---|---|---:|---|
| V2 stats skip 开启 | 每组一个 cell | 10 | 10,000 × 10 |
| V2 stats skip 关闭 | 文件 A：4+1；文件 B：4+1 | 4 | 40,000 / 10,000 / 40,000 / 10,000 |
| V3 Raw，投影估算仍是每组 1 MiB | 展平后 4+4+2 | 3 | 40,000 / 40,000 / 20,000 |

若 V3 只加载部分列、每组投影估算降为 0.5 MiB，则 R=8，10 组被划成 8+2，即 2 个 PK chunk。可见相同段行数、相同物理文件，不同加载路径/投影也可能有不同 chunk 数。

## 8. 对删除性能矩阵和后续门槛的影响

此前矩阵用 `ceil(N * 24或36 / 1或4MiB)` 产生 C，再均分 N 行。该模型用于隔离 N、B、C 对查找性能的影响；它没有复现文件短尾、按平均大小计算整数 R、V3 投影、manifest 范围裁剪及不均匀 chunk。

因此 44M int64 的 1008/252 chunks、44M VARCHAR16 的 1511/378 chunks 仍然是测试参数，不能称为真实段的必然布局。原始性能测量无需改写，但“对应生产场景”的解释必须保留此限制。

落实策略时应在相同发布状态下取 PK Column 的 C 和行边界。验证生产代表性时，至少采集：storage 版本/物理格式、PK 列组字段、实际投影、文件数与有效范围、每个存储单元的行数/估算大小、stats-skip 配置、R、最终 C 和各 chunk 行数。下一轮基准应能输入真实行数边界，而不只输入 C 后均分。

本次没有读取真实段 metadata，因此仍不能宣称此前 16/32 chunks 的启用规则已在实际存储布局上证明不退化。
