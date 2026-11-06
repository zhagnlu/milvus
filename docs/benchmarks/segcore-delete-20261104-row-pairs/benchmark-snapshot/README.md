# 测量源码快照

这里保存实际使用的 benchmark、runner、从生产代码提取的排序/查找内核及 GallopingSearch.h。内核哈希与每份 metadata 一致。

可仅用标准库独立编译，参数顺序见 DeleteBatchBenchmark.cpp 和 plan.json 中的 runner 参数：

```bash
c++ -O3 -DNDEBUG -std=c++20 -Wall -Wextra -I docs/benchmarks/segcore-delete-20261104-row-pairs/benchmark-snapshot docs/benchmarks/segcore-delete-20261104-row-pairs/benchmark-snapshot/DeleteBatchBenchmark.cpp -o /tmp/delete-row-pairs-snapshot
```
