# Stage16–18 工程验收记录

本记录对应“根据三个计划文档实现代码”，不代表35轮正式实验或性能提升已完成。使用说明见 [STAGE16_18_IMPLEMENTATION.md](../STAGE16_18_IMPLEMENTATION.md)。

## 需求到证据

| 需求 | 实现/证据 | 状态 |
|---|---|---|
| Stage16独立PT完整验证与三个同轨迹选择器 | `stage16_evaluation.py`、`stage16_selection.py`；gate/同分/无可行解/计数/NaN测试 | 已实现并测试 |
| Stage17 U/F/J/P/Z 参数范围 | `stage17_model.py`；全部条件权限、带AdamW历史状态的隔离、Z保持测试 | 已实现并测试 |
| Stage18回放、peptide×label权重、smoothing、KL | `stage18_config.py`、`stage18_loss.py`；窗口梯度等价、KL方向/T²/alpha0/极端值测试 | 已实现并测试 |
| 教师使用原始编码器，不能复用学生微调表示 | `stage18_teacher.py`、实际encoder张量校验；教师缓存不变、错误坐标拒绝测试 | 已实现并测试 |
| 缓存行ID/输入/角色/来源与字节完整性 | schema3身份、sidecar SHA、抽查；错序/错角色/错来源/有限值篡改拒绝测试 | 已实现并测试 |
| 评估不影响下一步训练 | 模式、requires_grad、CPU/CUDA/Python/NumPy RNG保护；下一步一致性测试 | 已实现并测试 |
| 缓存命中与未命中公平 | 学生权重加载后重置训练RNG；两条路径完整轨迹比较 | 已实现并测试 |
| 有效窗口及中途恢复 | plain CE/KD partial-window等价；Stage16及Stage18中途恢复逐参数一致 | 已实现并测试 |
| 完整微调encoder和三个head的推理导出 | 新类型encoder-inclusive state；训练后PT/PHT预测roundtrip完全一致 | 已实现并测试 |
| 模型锁定后评测与数据边界 | 10个CLI可启动；临时PH/PT/PHT benchmark端到端；debug不能锁定、锁文件不可覆盖 | 已实现并测试 |
| 多seed共用warmup生成 | 新common_warmup类型和ready_config；两轮warmup仅更新PHT、保留二元权重测试 | 已实现并测试 |
| 消融报告与配对统计 | [debug汇总](stage16_18_implementation_ablation_summary.md)；bootstrap身份对齐及相同预测零差值测试 | 已实现并测试 |
| 全量数据/来源审计 | [数据manifest](stage16/implementation_audit/data_manifest.json)，原始有效样本数及SHA均核对 | 已验证 |
| 真实PLM接线 | Stage16完整轮、Stage17恢复、Stage18 R3/COMB完整轮 | 已验证，均为debug |
| 当前缓存协议的真实恢复 | [Stage18最终记录](stage18/implementation_final_smoke_20260925/run_manifest.json)：round1、global_step3、finished | 已验证，debug |
| 35轮、全量teacher预计算、全部消融/多seed/真实test | 代码接口和默认预算已提供，本次没有执行科学实验 | 待正式实验 |

## 测试执行

环境：`/home/mjp/miniconda3/envs/unifyimmun/bin/python`，`OMP_NUM_THREADS=2`、`MKL_THREADING_LAYER=GNU`。

```text
python -m unittest discover -s tests -p 'test_plm_stage1*.py' -q
Ran 30 tests ... OK

python -m unittest discover -s tests -p 'test_stage8_pt_retention.py' -v
Ran 8 tests ... OK
```

所有规划中命名的实现文件已核对存在，文档链接与代码块已检查；新增Python代码已格式化，未发现TODO/NotImplemented占位。既有Stage8源文件及原始训练/测试CSV未修改。

真实debug运行只读train/validation，PT每split8行、PHT每类型2行；其结果不应出现在正式性能比较表中。最终真实test入口需要锁定的非debug checkpoint清单，尚未执行。开发早期的debug目录保留了对应版本证据；当前协议与旧debug缓存不兼容时会拒绝恢复，不会静默替换参考。
