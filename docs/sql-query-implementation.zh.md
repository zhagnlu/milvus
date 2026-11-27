# Milvus SQL Query 实现说明

整理日期：2026 年 11 月 25 日。适用范围：当前工作区中的 SQL Query 实现，包含尚未提交的改动；Git HEAD 为 `95607764c375e9401f4f29159e7ab6607baa6078`。该提交号仅标识基线，不能单独还原本文描述的工作区版本。本文不代表已发布版本的功能承诺。

SQL Query 将 PostgreSQL 风格的单集合 SELECT 编译为 Milvus 已有的 Query 或 Search 请求。普通行查询在 Proxy 完成 SQL 表达式计算；聚合查询把分组前表达式下推到 segcore，沿 QueryNode 和 Delegator 合并中间状态，最后由 Proxy 计算最终结果、排序和分页；向量查询直接复用 Search。理解这三条路径，是理解整个实现的关键。

## 01 能力与边界

| 能力 | 当前实现 |
| --- | --- |
| 数据范围 | 一个 collection；通过请求的 dbName 选择数据库，支持 partitionNames |
| 投影 | 字段、JSON 路径、别名、SELECT *、受支持的标量计算 |
| 过滤 | 比较、AND / OR / NOT、IN、BETWEEN、LIKE、IS NULL 等受支持表达式 |
| 聚合 | COUNT、SUM、AVG、MIN、MAX、精确 COUNT DISTINCT |
| 排序分页 | ORDER BY、NULLS FIRST / LAST、LIMIT、OFFSET |
| 时间表达式 | to_timestamp、extract、时间戳差及受支持的算术表达式 |
| 向量检索 | pgvector 风格距离运算符及相应函数，复用已有 ANN Search |
| 明确拒绝 | JOIN、子查询、CTE、HAVING、窗口、集合运算、写语句、SELECT DISTINCT 等 |

支持 PostgreSQL 风格语法不等于兼容完整 PostgreSQL。类型转换只实现了一个子集；WHERE 中的显式 cast 仍会在提取阶段被拒绝。COUNT(DISTINCT x) 与 SELECT DISTINCT x 是不同功能，后者当前不支持。REST 用 JSON 对象表示一行，因此重复输出列名必须通过不同别名消除。

## 02 请求接口与使用示例

入口为 `POST /v2/vectordb/entities/sql_query`。集合名称来自 SQL 的 FROM，不需要再传 collectionName。

```json
{
  "dbName": "default",
  "sql": "SELECT pk, age FROM users WHERE age >= $1 ORDER BY age DESC LIMIT 10",
  "params": {"$1": 18},
  "consistencyLevel": "Strong",
  "timezone": "UTC"
}
```

| 参数 | 作用 |
| --- | --- |
| sql | 必填，只允许一条 SELECT |
| dbName | 数据库名称；不要使用 database.collection 形式的 FROM |
| params | 标量位置参数；$1 和 1 都归一化到 sql_param_1，同时提供冲突键会报错 |
| searchParams | 向量输入和检索调优参数；值为字符串，例如 nprobe 和 $1 |
| partitionNames | 限定目标分区 |
| consistencyLevel | 由已有 Query / Search 一致性逻辑处理 |
| timezone | 默认为 UTC，控制时间提取与时间戳格式化 |

成功响应沿用 REST v2 的 `code`、`data` 和 `cost` 结构。客户端应检查业务 code；不能只以 HTTP 200 判断成功。Int64 输出沿用现有 REST 序列化约定，受 `Accept-Type-Allow-Int64` 请求头控制。

下面的 SQL 展示分组与精确去重。示例假设 events 包含 JSON 字段 data。

```sql
SELECT data->>'city' AS city,
       count(*) AS events,
       count(DISTINCT data->>'user') AS users
FROM events
GROUP BY city
ORDER BY users DESC
LIMIT 10;
```

向量请求仍使用同一个 SQL HTTP 入口：

```json
{
  "sql": "SELECT pk, embedding <-> $1 AS distance FROM articles ORDER BY distance LIMIT 5",
  "searchParams": {"$1": "[0.1, 0.2, 0.3]", "nprobe": "10"},
  "consistencyLevel": "Strong"
}
```

向量维度和类型必须匹配集合 schema，调优参数必须适用于实际索引。

## 03 总体执行链路

```text
REST SQL 请求
    ↓
HTTP 解析 SQL，定位 collection，读取 schema，准备参数
    ↓
按是否包含向量距离表达式分流
    ├─ 普通行查询 → QueryRequest → QueryTask → QueryNode
    │                              ↓
    │                       Proxy Project → 排序 → 分页
    │
    ├─ 聚合查询 → QueryTask → segcore SQL Project → 原生聚合
    │                                           ↓
    │                    QueryNode 合并 → Delegator 合并
    │                                           ↓
    │                    Proxy 最终归并 → 最终表达式 → 排序 → 分页
    │
    └─ 向量查询 → SearchRequest → 已有 Search → 距离转换 → SQL 投影
    ↓
REST 行对象数组
```

HTTP 的 sqlQuery / sqlSearch 通过现有 wrapperProxyWithLimit 调用 Query / Search，复用鉴权、限流和请求控制。路由还接入 DQL admission、请求大小限制和超时中间件。普通 SQL 被放入 QueryRequest.QueryParams 的 `sql` 参数；DQL 层在 prepareSQL 中识别它，而不是新增一套 SQL RPC 或复制任务生命周期。

prepareSQL 再次检查 SQL 中的 collection 与已授权 QueryRequest.CollectionName 相同，并拒绝 SQL iterator、内部查询和从 Query 路径进入的向量 SQL。HTTP 的前置解析负责路由，DQL 的解析和绑定负责真正执行计划。

## 04 从 SQL 文本到类型化计划

### PostgreSQL AST 与 SqlComponents

`internal/sqlparser/parser.go` 使用 pg_query_go/v6 解析 PostgreSQL AST。ParseOne 强制单条语句，ExtractSqlComponents 强制 SELECT，再提取 FROM、SELECT、WHERE、GROUP BY、ORDER BY、LIMIT 和 OFFSET。

语法可以被 PostgreSQL parser 接受，并不表示能被 Milvus 执行。validateSelectStatement 显式拒绝不支持的子句，FROM 提取只接受一个普通集合，避免解析成功后静默丢弃子句。WhereClause 同时保留原始 AST 和转换后的 Milvus filter 表达式。

### Schema 绑定与三个身份空间

BindWithParams 将 SqlComponents 绑定到 collection schema，生成 BoundQuery。其核心是严格区分三种身份：

| 身份 | 表示什么 | 使用位置 |
| --- | --- | --- |
| 物理 FieldID | collection 中真实存在的字段 | 读取 segment 原始数据 |
| 查询投影 ID | 一次查询中的类型化中间列 | 原生 Project、聚合和 reducer schema |
| Slot | 逻辑分组键或聚合输出在最终布局中的位置 | Proxy 最终表达式求值 |

例如 `data->>'city'` 和 `data->>'country'` 共享一个物理 JSON 字段，但必须成为不同的中间值。`min(v)` 和 `max(v)` 也需要不同的输出 Slot。把它们都按原始 FieldID 标识，会导致结果串列或覆盖。

BoundExpr 保存 Op、Type、Nullable、FieldID、Slot、Path、Value 和 Args。BoundQuery 将 GroupKeys、Aggregates、Outputs、OrderBy、SourceFields 和 Predicate 分开。分组键与聚合输入属于逐行阶段，Outputs 和 OrderBy 可以引用分组后 Slot。

绑定器解析别名、合法表限定符及分组或排序的位置引用，检查类型、重复输出名和聚合合法性。用于表达式身份比较的 AST 副本会去掉位置标记并归一化合法表限定符，避免同一表达式在 SELECT 与 GROUP BY 中被误认成不同表达式。

### LowerAggregate 与线协议

LowerAggregate 将聚合查询编译为 SQLProject、GroupIDs、原生 Aggregates 和逻辑到物理结果列的 Inputs 映射。相同逐行表达式可复用投影列；新 ID 从 schema 最大字段 ID 之后分配，覆盖 struct-array 子字段的 ID 范围检查。

当前协议为：

```text
QueryPlanNode.sql_project = 8
RetrieveRequest.sql_project = 28
SQLProject.version = 1

SQLProject
  ├─ columns: [投影 ID → SQLScalarExpr]
  ├─ timezone
  └─ predicate
```

QueryPlanNode 的字段号 7 保留给 query_iterator_cursor。这里沿用原生 group_by_field_ids 与 aggregates，并没有迁入历史方案的 aggregate_node 字段。SQLProject 的版本及投影 ID 在相应解码和 schema 构建处检查；这不等于已完成所有混合版本集群验证。

## 05 普通行查询如何执行

没有 GROUP BY 和聚合的 SQL 复用 QueryTask 获取 SourceFields。Proxy 收到合并后的结果，再由 sqlquery.Project 执行 SQL Predicate，计算 SELECT / ORDER BY 表达式，最后排序和切片。

prepareSQL 清空普通 Query 的过滤字符串、排序和分组配置，并将下层 limit 设为 Unlimited，避免先分页再做 SQL 过滤。只有在没有 SQL Predicate、没有 ORDER BY 且 LIMIT 为正时，才把 OFFSET + LIMIT 下推，限制初始读取量；该分支检查分页相加溢出。

若 SQL 仅选择常量，没有源字段，准备阶段仍取主键以维持行数。LIMIT 0 有专门的空结果处理。SQL 排序列可以是未出现在 SELECT 中的表达式：finish 将其作为临时列求值，完成排序和分页后再移除。

这条路径会在 Proxy 物化所需源列。即使最终 LIMIT 很小，带过滤或排序的查询也可能读取大量候选数据，受到现有输出大小限制约束。不要把它理解为已具备通用的 SQL 索引下推或全局成本优化器。

## 06 聚合前表达式如何下推

聚合查询通过 SQLProject 在 segcore 的 ProjectNode 内执行逐行表达式，之后接现有 AggregationNode。PlanProto.cpp 根据查询投影 ID 找到对应类型，ProjectNode.cpp 在选出可见行的 offsets 后调用 EvaluateSqlProjection。

执行顺序为：可见性与已有原生过滤 → SQL Predicate → 分组键和聚合输入计算 → 原生聚合。SQL Predicate 为 FALSE 或 NULL 的行不会进入聚合。

例如按小时分组：

```sql
SELECT extract(hour FROM to_timestamp(
         (data->>'time_us')::bigint / 1000000.0
       )) AS hour_of_day,
       count(*) AS events
FROM events
GROUP BY hour_of_day
ORDER BY hour_of_day;
```

每行应先算出 hour_of_day，再按小时聚合。若先按原始 time_us 分组，再把组键格式化为小时，会保留本应合并的多个组。BoundQuery 的阶段划分和 SQLProject 的执行位置共同保证这一顺序。

原生执行器先将协议表达式准备为内部操作树，支持所实现的标量、JSON、算术、转换和时间操作，并在逐行循环及批量读取中检查取消。它使用类型化值，Int64 输入不会统一先转为 double。

## 07 分布式聚合如何保持正确

### 中间结果必须可再次合并

segcore 返回的列布局为物理分组键加聚合状态。QueryNode 与 Delegator 使用共享 reduction pipeline，经 DecodeInternalResults → Merge → EncodePartial 继续传递状态；Proxy 的 ReduceAggregate 完成最后一层合并，再进行 SQL 最终化。

WithSQLProjectionSchema 只克隆并扩展 reducer 的查询局部 schema，不修改 collection schema，也不让 segment 把虚拟列当作真实存储字段。PartialAggregationResult 与 FinalAggregationResult 分成不同类型，约束最终展示数据不再回流到 Merge。

### AVG 合并 sum 和 count

```text
分片 A：sum = 10，count = 1
分片 B：sum = 90，count = 9
全局 AVG = (10 + 90) / (1 + 9) = 10
```

LowerAggregate 把 AVG 展开为 SUM 和 COUNT，Proxy 最后相除。直接平均两个局部 AVG，会在分片样本量不同时产生错误。COUNT(*) 被编译为 COUNT(投影常量 1)，让计数与 SQL Predicate 在同一个原生路径中执行。

### COUNT DISTINCT 合并精确等价类

DISTINCT 参数被追加到物理 grouping key。例如逻辑上按 city 分组、去重 user，原生会按 `(city, user)` 聚合。跨 segment 和 shard 合并这些物理键后，Proxy 再折叠回 city，统计非 NULL 的不同 user，同时合并其他聚合状态。

```text
分片 A 的 user：u1、u2
分片 B 的 user：u2、u3
全局精确结果：3
局部基数相加：2 + 2 = 4，错误
```

多 DISTINCT 参数会扩展更多物理键，不能用局部基数求和替代。当前代码对已经完成多输入合并、仅增加一个非浮点 DISTINCT 键的特定形状，可直接数已唯一化的物理组；单输入、多个 DISTINCT 维度及浮点键等情况保留集合路径。

最终折叠只有一个字符串 SQL 组键时，当前实现直接使用不可变字符串作 map key，并单独保存 NULL 组。空字符串与 NULL 不混淆，资源计费仍沿用保守的编码长度。复合键使用带类型标签和字符串长度的 tuple 编码。

### 排序与分页必须在全局最终化后执行

SQL 准备阶段清除原生排序并保留 Unlimited，使下层不按局部聚合值截断组。Proxy 合并全部状态后，才计算最终 SELECT 和 ORDER BY 表达式，再执行 OFFSET / LIMIT。

例如 shard A 的计数为 X=6、Y=5，shard B 为 Z=6、Y=5。各取局部 top 1 会丢弃 Y，但全局第一名其实是 Y=10。同理，`max(timestamp) - min(timestamp)` 必须在全局 min/max 合并后计算，不能先计算局部跨度再取最大值。

## 08 JSON 读取与语义一致性

聚合路径的 JSON 表达式在原生 SQL evaluator 中执行。对于可使用 JSON stats 的文本路径，执行器尝试 STRING shredding reader；相应整数转换也可使用 INT64 reader。路径不适用、stats 不可用或 typed cell 缺失时，按需要读取原始 JSON。

typed cell 缺失不能直接解释成 SQL NULL：原始 JSON 可能含有其他类型，并在 ->> 文本提取后产生有效值。直接文本路径的列内核会收集缺失 offsets，以最多 1,024 行一批读取原始 JSON；通用逐行路径也使用有界 raw JSON 缓存。

SQL 的 `-> '0'` 表示对象键，不是数组下标。当前代码为数字字符串路径检查父节点类型，避免 JSON Pointer 的数组解释改变 SQL 含义。数值、字符串、JSON null 和路径不存在也必须按实际提取方式处理。

聚合查询还可以通过 LowerJSONPredicate 下推一个保守的原生过滤子集：文本 JSON 路径与合适字符串字面量的等值、IN，以及可整体转换的 AND / OR。字面量自身能解析为 JSON、数字形式路径等情况会保留 SQL evaluator。转换按整棵支持的谓词进行，避免不等价的局部替换。

原始读取失败与“typed 路径不可用”属于不同情况。SQL evaluator 的正常回退路径没有把任意读取异常吞成 NULL；存储故障的完整错误分类和重试行为仍需按端到端故障注入结果判断。

## 09 向量 SQL 如何复用 Search

BindSearchWithParams 校验一个被选出的距离表达式、唯一的升序距离 ORDER BY、显式 LIMIT，并拒绝 GROUP BY 和聚合。Search 的 TopK 取 OFFSET + LIMIT；返回结果经距离转换与 SQL 投影后再分页。

| SQL 距离 | Milvus metric | 返回到 SQL 的距离 |
| --- | --- | --- |
| <-> 或 l2_distance | L2 | sqrt(max(0, score)) |
| <=> 或 cosine_distance | COSINE | 1 - score |
| <#> 或 ip_distance | IP | -score |
| <~> 或 hamming_distance | HAMMING | 原始 score |
| <%> 或 jaccard_distance | JACCARD | 原始 score |

向量参数按 schema 转成 typed placeholder。WHERE 的受支持标量字段与字面量表达式由 Search 在选邻居前执行；JSON 提取、cast 和算术等不满足该过滤路径的表达式会被拒绝，避免在已经截断的 ANN 结果上过滤而改变 top-k 含义。

SQL 控制 anns_field、metric_type、topk、offset 和 round_decimal，searchParams 不允许覆盖这些键。检索精度仍取决于使用的 ANN 索引与配置，SQL 距离语法不改变其召回性质。

## 10 资源和兼容性约束

精确 DISTINCT 可能产生大量中间组，多 DISTINCT 维度会进一步扩大物理分组基数。`common.groupBy.maxGroups` 约束原生组数；Proxy 还检查输入 partial 总大小、中间去重状态和输出大小。当前实现没有 SQL 状态落盘，不能把这些检查视为全链路内存峰值保证。

无 ORDER BY 的查询没有稳定行序承诺。排序支持 NULLS FIRST / LAST；默认 ASC 为 NULLS LAST，DESC 为 NULLS FIRST。全局无输入聚合与空 GROUP BY 不相同：前者 COUNT 为 0，SUM / AVG / MIN / MAX 为 NULL；后者没有分组行。

SQLProject 使用显式版本，未知版本或非法虚拟 ID 在相应检查处失败。COUNT(*) 引用投影常量，使不识别虚拟列的旧节点不能仅忽略 SQLProject 后悄悄执行原生 count 快捷路径。升级兼容性仍应覆盖各版本组合的实际执行，不应仅凭 protobuf 可解码推断可用。

该功能执行只读查询，没有引入持久元数据修改或新的写入消息路径。现有普通 Query 的 Arrow 和延迟物化路径继续由原有执行分支处理；SQL 行查询在 Proxy 的物化成本需要独立评估。

## 11 验证证据与复现入口

代码中的测试分为 parser / binder / lowering、Go 求值与归并、HTTP / DQL、C++ SQL projection 和真实服务 E2E。核心反例包括跨分片重复 DISTINCT、同小时不同时间戳、局部与全局 top-k 不一致、NULL 与空输入、同字段多路径、Int64 精度及取消。

截至 2026 年 11 月 24 日的仓库验证记录报告：相关 Go 包、SQL E2E 的 27 个子测试、保留版本的原生 26 个用例，以及 growing / sealed、stats 开关、双分片下的混合 JSON 与 grouped DISTINCT 矩阵共 54 项检查通过。这里引用的是既有验证记录；本次整理没有重新运行 Milvus 功能测试。

该记录同时明确：全仓 Go / C++ 回归、云部署、legacy stats 以及 S3 / 数据损坏 / OOM 注入尚未完成。本文对错误回退和资源约束的说明不能延伸为这些故障下重试已验证的结论。性能优化也不能据此宣称全部查询无回退。

在已经构建好匹配依赖、Go 二进制及核心动态库的环境中，可按以下入口复现。Go 测试必须保留仓库要求的 tags 与 gcflags。

```bash
source scripts/setenv.sh

go test -tags dynamic,test -gcflags="all=-N -l" -count=1 \
  ./internal/sqlparser ./internal/sqlquery ./internal/agg \
  ./internal/util/queryutil ./internal/util/shallowcopy

go test -tags dynamic,test -gcflags="all=-N -l" -count=1 \
  -run TestSQL ./internal/proxy/dql ./internal/distributed/proxy/httpserver

cmake --build cmake_build --target sql_projection_tests -j 4
cmake_build/unittest/sql_projection_tests

# 先启动使用匹配 Go/core 的本地 Milvus 服务，再运行：
go test -tags dynamic,test -gcflags="all=-N -l" -count=1 -v \
  ./tests/e2e -run '^TestSqlQuery$' \
  -milvus-addr localhost:19530 -timeout 10m
```

## 12 源码与设计资料索引

以下是工作区相对路径。部分文件及改动尚未提交，不应将公开仓库基线链接当作当前实现的完整源码。

| 环节 | 文件与关键入口 |
| --- | --- |
| HTTP 路由与请求 | internal/distributed/proxy/httpserver/handler_v2.go；request_v2.go 的 SQLQueryReqV2 |
| HTTP 标量入口 | internal/distributed/proxy/httpserver/sql_query.go 的 sqlQuery |
| HTTP 向量适配 | internal/distributed/proxy/httpserver/sql_search.go 的 sqlSearch |
| SQL 解析与语法约束 | internal/sqlparser/parser.go、extract.go、validate.go、where_converter.go |
| 类型绑定 | internal/sqlparser/bind.go 的 BindWithParams；predicate.go |
| 聚合编译与过滤下推 | internal/sqlparser/lower.go 的 LowerAggregate；json_predicate.go |
| 向量绑定与距离转换 | internal/sqlparser/search.go 的 BindSearchWithParams、SQLDistance；placeholder.go |
| DQL 接入 | internal/proxy/dql/sql_query.go 的 prepareSQL；task_query.go 的 PostExecute |
| 原生执行计划 | internal/core/src/query/PlanProto.cpp；exec/operator/ProjectNode.cpp |
| 原生表达式执行 | internal/core/src/exec/operator/SqlProjection.cpp 的 EvaluateSqlProjection |
| QueryNode 与 Delegator 归并 | internal/querynodev2/segments/query_pipeline.go 的 RunQNQueryPipeline、RunDelegatorQueryPipeline |
| 共享归并和局部 schema | internal/util/queryutil/pipeline_builders.go、reduce_by_groups_op.go、sql_projection.go |
| 可合并聚合状态 | internal/agg/aggregate_reducer.go 的 DecodePartial、Merge、EncodePartial、Finalize |
| Proxy SQL 最终化 | internal/sqlquery/result.go 的 ReduceAggregate、Project、finish；eval.go、group_key.go |
| 协议定义 | pkg/proto/plan.proto；pkg/proto/internal.proto |
| 功能测试 | internal/sqlparser/*_test.go；internal/sqlquery/*_test.go；internal/proxy/dql/sql_query_test.go；internal/distributed/proxy/httpserver/sql_query_test.go |
| 原生与 E2E 测试 | internal/core/unittest/test_sql_projection.cpp；tests/e2e/sql_query_test.go |

当前设计与验证脉络见 `docs/design-docs/design_docs/20261104-sql-query.md`；`docs/sql-parser-design.md` 已标记为历史方案；最新既有验证边界见 `docs/benchmark-results/20261124-jsonbench-final.txt`。实现说明以当前 SQL 代码与现行设计交叉核对，历史 RPC 和历史 aggregate_node 方案不属于本文执行链路。
