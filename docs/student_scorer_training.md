# Student register scorer 训练

从 `distill-epoch=09-step=5920.ckpt` 加载冻结的基础 planner、motion encoder 与 student FAR。
同一 checkpoint 内冻结的 `teacher_registers` 来自 epoch05，使用其 `window_heads/global_heads`
初始化评分头。严格加载所需参数；不创建 DiT、teacher readout，也不进行未来生成或 token MSE。

## 网络与选候选

Student 仍使用一个 generating-anchor + planner-offset motion token；第一个 decoder
同时读取视觉和 motion，后续 fusion 交互 scene registers、ego 与 motion。
评分器读取融合后的前 37 个 token（36 scene + 1 ego）。新增返回选项保留原蒸馏的 36-token 接口。

评分头规格与 teacher 一致：512→1024→1，ReLU。各 12 个 scene registers 平均池化，
得到三个窗口的 NC、DAC；37 个 tokens 平均池化，经五个独立 heads 输出 TTC、Comfort 和三个 EP。
训练和验证从 planner Top-16 取固定前三条 + 剩余十三条随机三条，无放回；验证种子固定。
每条选中的轨迹调用一次 PDM scorer/simulator，全程及分段标签复用同一次结果。每 scene 六次，
不使用真实 PDM 做候选选择。推理给整个 Top-16 评分，取最高分，选轨迹路径不访问真实标签。

## 标签与聚合

时间窗口保留 `(0,.5]、(.5,2.5]、(2.5,4]`，但此新阶段将 **t=0 的安全状态计入第一段**，
使跨时间 min 与官方包含初始时刻的全程 NC/DAC 一致。这不是修改此前 teacher 的标签。

EP 改为全程官方 EP 的分段贡献：计算三个时间段内的非负进度增量 Δ_i，
当增量和为正时，标签 `EP_i = official_EP × Δ_i / sum(Δ)`；
总进度增量为零时，按窗口时长比例 `[.125,.5,.375]` 分配 official_EP。
停驶分配仅为标签约定，不能解释为实际移动距离。这样可以兼容官方低进度阈值、安全门控与饱和逻辑，
确保三段 EP 标签相加等于全程 EP，而不是同时强制两个不一致的监督目标。

使用同一可微聚合函数训练和推理：

```
NC = min(NC_1, NC_2, NC_3)
DAC = min(DAC_1, DAC_2, DAC_3)
EP = clamp(EP_1 + EP_2 + EP_3, 0, 1)
PDMS = NC * DAC * (5*EP + 5*TTC + 2*Comfort) / 12
```

实现检查分段安全标签与全程指标、PDMS 公式与真实 simulator 分数一致，不静默接受其他版本的评分权重。
最终 PDMS 不增加独立 head，梯度通过聚合回传至分项预测。min/clamp 的梯度特点由分段监督补充。

## 四类损失

```
L_segment = mean(BCEWithLogits(NC_windows), BCEWithLogits(DAC_windows), BCEWithLogits(EP_windows))
L_global = mean(BCEWithLogits(TTC), BCEWithLogits(Comfort))
L_aggregate = MSE(predicted_full_[NC,DAC,EP], simulator_full_[NC,DAC,EP])
L_pdm = MSE(predicted_PDMS, simulator_PDMS)
L = segment_loss_weight*L_segment + global_loss_weight*L_global
    + aggregate_loss_weight*L_aggregate + pdm_loss_weight*L_pdm
```

四个权重默认均为 1，在 YAML 中集中列出。NC=.5 和 EP 等软标签不做二值化。
聚合指标及 PDMS 的 target 直接来自 simulator；分项日志不再被重复相加到 loss。

## 优化策略

默认共 10 epochs；epoch00 只训练评分头（初始 LR 5e-5），FAR 以 eval/no_grad 前向、LR=0。
epoch01 起联合微调，FAR LR 为评分头的 0.1 倍，二者余弦衰减。基础 planner、适配器与 motion
encoder 始终冻结。FAR 参数从构造时就注册到 DDP，但预热期间无梯度和优化更新；
DDP 使用 `find_unused_parameters=True` 支持预热到联合微调的切换，不启用 static_graph。
日志额外记录 `far_learning_rate`、`far_finetuning`。

仍采用此前缓存划分：85,109 training / 18,179 validation，每 epoch 全量验证并保存。
默认每卡 batch size 36、4 GPU、BF16；配置解析已覆盖该设置，smoke 仅验证小 batch。
W&B key 与之前一样从外部 wandb.md 读取。

## 启动与续训

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash /mnt/cfs/qiuzijun/WorldDrive/scripts/training/run_student_scorer.sh
```

配置：`navsim/planning/script/config/training/student_scorer_4gpu.yaml`。
可通过 `STUDENT_CHECKPOINT` 替换权重初始化来源；`SCORER_INIT_CHECKPOINT` 可单独指定评分头来源，
默认 null 则使用 student checkpoint 内的 heads。
`TRAIN_METRIC_CACHE` 指定训练/验证用的 metric cache。

本阶段中断续训时使用 `SCORER_RESUME_CHECKPOINT`，恢复 optimizer/scheduler/epoch，
回调根据恢复的 epoch 切换 FAR 训练状态，不会重新开始预热。
不要将蒸馏阶段 checkpoint 当成评分阶段 full-state resume。

```bash
export SCORER_RESUME_CHECKPOINT='/absolute/path/to/student-scorer-epoch=02-step=1776.ckpt'
bash /mnt/cfs/qiuzijun/WorldDrive/scripts/training/run_student_scorer.sh
```

本阶段 checkpoint 不包含 DiT。训练后完整 navtest 入口：

```bash
export STUDENT_SCORER_CHECKPOINT='/absolute/path/to/student-scorer-epoch=09-step=5920.ckpt'
CUDA_VISIBLE_DEVICES=6,7 GPUS=2 bash /mnt/cfs/qiuzijun/WorldDrive/scripts/evaluation/run_student_scorer_pdm_score.sh
```

`tests/test_student_scorer.py` 覆盖聚合梯度、软标签、初始时刻、停驶/饱和 EP、权重复用、
37-token 接口和优化阶段切换。`scripts/check_student_scorer.py` 在真实缓存和 checkpoint 上
运行预热及解冻短训练、验证与 Top-16 无 GT 推理。

本次验证结果：15 项相关测试通过；真实容器的单卡及双卡 DDP smoke 均通过。
双卡检查覆盖 FAR 预热零更新、解冻后梯度与参数同步、每候选一次 simulator、
Top-16 无标签推理，以及保存/重新加载本阶段 checkpoint 后分数完全一致。
结果见 `outputs/register_readiness/student_scorer_smoke_2gpu.json`。
这些短测试验证训练链路，不代表规划性能已提升；完整训练和 navtest 需要另外启动。
