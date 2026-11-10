# 数据集划分审计报告

- 随机种子：`20260840`
- 实体分配目标：`70%/15%/15%`
- 主原则：全局实体先划分，任务样本后派生；混合 unseen 组合不进入主 benchmark。

## PHT 主划分

| 部分 | 行数 | 占保留样本 |
|---|---:|---:|
| train | 4,168,959 | 41.58% |
| val | 2,993,620 | 29.86% |
| test | 2,863,675 | 28.56% |
| excluded mixed | 3,045,128 | - |
| quarantined conflicts | 5,694 | - |

## 任务行数

```json
{
  "PH": {
    "excluded": 98661,
    "test_unseen_hla": 84100,
    "test_unseen_pair": 985,
    "test_unseen_peptide": 100123,
    "train": 466216,
    "val_unseen_hla": 146236,
    "val_unseen_pair": 978,
    "val_unseen_peptide": 99861
  },
  "PT": {
    "excluded": 134154,
    "test_unseen_pair": 11802,
    "test_unseen_peptide": 147620,
    "test_unseen_tcr": 154822,
    "train": 721467,
    "val_unseen_pair": 11810,
    "val_unseen_peptide": 164666,
    "val_unseen_tcr": 154855
  },
  "PHT": {
    "excluded": 3045128,
    "quarantined_label_conflict": 5694,
    "test_unseen_hla": 1061753,
    "test_unseen_peptide": 855143,
    "test_unseen_tcr": 892230,
    "test_unseen_triple": 54549,
    "train": 4168959,
    "val_unseen_hla": 996654,
    "val_unseen_peptide": 1044384,
    "val_unseen_tcr": 893719,
    "val_unseen_triple": 58863
  }
}
```

## PHT 四类分布

```json
{
  "excluded": {
    "000": 1500073,
    "010": 1414976,
    "100": 115304,
    "111": 14775
  },
  "quarantined_label_conflict": {
    "111": 5694
  },
  "test_unseen_hla": {
    "000": 579088,
    "010": 447512,
    "100": 25318,
    "111": 9835
  },
  "test_unseen_peptide": {
    "000": 411465,
    "010": 378797,
    "100": 57822,
    "111": 7059
  },
  "test_unseen_tcr": {
    "000": 467665,
    "010": 347875,
    "100": 72841,
    "111": 3849
  },
  "test_unseen_triple": {
    "000": 22738,
    "010": 30657,
    "100": 824,
    "111": 330
  },
  "train": {
    "000": 2184686,
    "010": 1626858,
    "100": 339367,
    "111": 18048
  },
  "val_unseen_hla": {
    "000": 582887,
    "010": 381537,
    "100": 31323,
    "111": 907
  },
  "val_unseen_peptide": {
    "000": 480509,
    "010": 470189,
    "100": 82430,
    "111": 11256
  },
  "val_unseen_tcr": {
    "000": 465414,
    "010": 351624,
    "100": 72882,
    "111": 3799
  },
  "val_unseen_triple": {
    "000": 27993,
    "010": 29397,
    "100": 1407,
    "111": 66
  }
}
```

## 重要数据质量警告

1. `PHT_111.csv` 的 HLA 已映射为 33/34 aa 序列；当前实际长度分布见 `audit/report.json` 的 `hla_row_formats`。
2. PHT 与 PH/PT 的二元标签查找结果见 `audit/report.json`；`missing` 不会被当成负样本。
3. 明确标签冲突行未被改标，而是隔离在 `PHT/excluded/cross_task_label_conflicts.csv`。
4. `excluded/mixed_entity_splits.csv` 保留了多实体混合 unseen 样本，避免归入单实体 benchmark 造成语义混淆。

## 8–15 aa 过滤与四类训练集平衡后的正式输出

本版本仅包含长度为 8–15 aa（含边界）的 peptide，并沿用随机种子 `20260840`、全局实体比例 `70%/15%/15%` 重新划分。正式使用的后平衡数据如下：

- PH train/val/test：291,052 / 168,680 / 110,576，三个部分的正负比均为 1:1。
- PT train/val/test：664,952 / 307,670 / 298,034，三个部分的正负比均为 1:1。
- PHT train：72,192 条，PHT_111/PHT_000/PHT_100/PHT_010 各 18,048 条，四类比例为 1:1:1:1。
- PHT val/test：32,056 / 42,146，聚合 PHT 正负比均为 1:1；场景化结果仍以 `PHT/benchmarks/` 为准。

PHT train 的四类等量意味着聚合二分类标签必然为 18,048 个正样本和 54,144 个负样本，即正:负为 1:3。这是已确认的训练集特例，不适用 0.8–1.2 的二分类比例约束；PH、PT、PHT val/test 及所有正式 benchmark 仍执行原二分类比例约束。

`PHT/classes/PHT_<code>/train.csv` 是 `PHT/train.csv` 的四个互斥分类视图，不是额外新增样本。完整机器可读结果见 `audit/balance_report.json` 和 `audit/final_validation.json`。
