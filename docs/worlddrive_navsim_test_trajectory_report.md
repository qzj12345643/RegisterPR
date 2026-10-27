# WorldDrive Navtest 全量轨迹评测报告

## 1. 结论摘要

完成 NAVSIM v1 navtest 全部 **12,146** 个场景，每场景评估 256 条候选，共 **3,109,376** 条轨迹。
Stage 1 PDMS 为 **87.7627**，FAR 为 **89.2488**，FAR 相对变化 **+1.4861** 分（0–100）。
Top-5 oracle 为 **93.8365**，256 候选 oracle 为 **99.3590**。Oracle 使用真实标签挑选，属于候选集合上界，不能部署。

## 2. 协议与完整性

- 模型：`worlddrive_stage2_train_89p2_pdms.ckpt`；同时读取其 Stage 1 选择与 FAR 最终选择。
- 每条候选 8 个未来位姿，间隔 0.5 s；官方 PDM 仿真 40 步 × 0.1 s。
- 图像与 metric cache 来自 DriveWorld-VLA 数据目录；模型、PDMSimulator、PDMScorer 来自本 WorldDrive 仓库。
- FP32 推理；VAE posterior 采样种子为 `int(token,16) % (2**31-1)`，可断点重现。
- 批量评估 256 候选，但 EP 按每个候选与参考轨迹组成的二元组独立归一化；不以全体候选最大进度作为分母。
- 每个场景的 Stage 1、FAR 选择均与官方单轨迹 `pdm_score` 复核，最大子指标绝对误差 **0**。
- PDMS = NC × DAC × (5 EP + 5 TTC + 2 Comfort) / 12；DDC 权重为 0。
- 下表统一使用 0–1 标度；零 regret、oracle 覆盖与 FAR 增减的容差为 1e-6。

## 3. 总体结果

| 指标 | Stage 1 | FAR | Top-5 oracle | 256 oracle |
|---|---:|---:|---:|---:|
| NC | 0.984069 | 0.983863 | 0.994278 | 1.000000 |
| DAC | 0.958340 | 0.969867 | 0.989462 | 0.999918 |
| EP | 0.816147 | 0.837581 | 0.885272 | 0.984814 |
| TTC | 0.950683 | 0.950519 | 0.982628 | 0.999918 |
| Comfort | 0.999918 | 0.999918 | 1.000000 | 0.999918 |
| DDC | 0.976494 | 0.974971 | 0.974436 | 0.909641 |
| PDMS | 0.877627 | 0.892488 | 0.938365 | 0.993590 |

## 4. 排序质量

| 指标 | Stage 1 | FAR |
|---|---:|---:|
| mean_regret | 0.115963 | 0.101102 |
| median_regret | 0.050683 | 0.042253 |
| p90_regret | 0.190646 | 0.167228 |
| zero_regret_rate | 0.306356 | 0.339206 |
| first_argmax_match | 0.026428 | 0.035814 |

Stage 1 全 256 候选逐场景 Spearman 均值：0.667880（有效 12145 场景）；FAR 仅 Top 5：0.395656（有效 9269 场景）。常数排名场景不计入相关性均值。

| Stage 1 top-k | 包含并列 PDM oracle 的比例 |
|---|---:|
| 5 | 41.02% |
| 10 | 50.28% |
| 16 | 59.13% |
| 32 | 73.86% |
| 64 | 85.89% |

零 regret 接受并列最优；首个 argmax 一致率仅按候选索引比较，两者不等价。

## 5. FAR 的实际收益

FAR 改选 9653 个场景；真实 PDMS 提升 4845 个、下降 1891 个、持平 5410 个。
FAR 在 Top 5 内达到并列最优的比例为 60.85%。
候选召回损失（256 oracle − Top-5 oracle）为 5.5225 分；Top-5 排序损失（Top-5 oracle − FAR）为 4.5877 分。

这些结果检验最终重排收益，不能单独证明未来 latent 的语义预测是收益来源；要判断其是否优于等容量 Transformer，需要控制参数量、训练数据与训练预算的重训练消融。

## 6. 安全与进度损失

| 事件 | Stage 1 场景数 | FAR 场景数 |
|---|---:|---:|
| NC_lt1 | 211 | 211 |
| NC_zero | 176 | 181 |
| DAC_lt1 | 506 | 366 |
| TTC_lt1 | 599 | 601 |
| any_failure | 1053 | 932 |
| safety_inferior_to_oracle | 1052 | 931 |

各事件可能重叠；NC<1 包括部分碰撞扣分，TTC 失败不是实际碰撞。
只将 FAR 所选轨迹的 EP 替换成同场景 256 oracle EP，其余指标保持不变，反事实平均 PDMS 为 0.936020，相对 FAR 变化 +0.043533。此项仅作误差分解，不是可实现收益。

## 7. Top-5 轨迹差异

Top-5 内 10 对轨迹的平均 ADE（xy 欧氏距离）的场景中位数为 0.7633 m，P90 为 1.4790 m；平均 FDE 的中位数为 1.6972 m。
此指标按相同未来时间点比较，既包含路线也包含速度差异；本评测没有对全量未来 latent 做语义或图像相似度检验。

## 8. 产物与复现

- [汇总 JSON](../outputs/navtest_worlddrive_stage2_full/aggregate_summary.json)
- [逐场景 CSV](../outputs/navtest_worlddrive_stage2_full/per_scene.csv)
- [完整性检查](../outputs/navtest_worlddrive_stage2_full/input_audit.json)
- 每场景候选与预测分数：`outputs/navtest_worlddrive_stage2_full/candidates/`
- 每条候选真实 PDM 子指标：`outputs/navtest_worlddrive_stage2_full/pdm/`
- 实现：`scripts/prepare_worlddrive_navtest.py`、`scripts/infer_worlddrive_navtest.py`、`scripts/score_worlddrive_navtest.py`、`scripts/report_worlddrive_navtest.py`。

![评测汇总](../outputs/navtest_worlddrive_stage2_full/evaluation_summary.png)
