# 单 motion token 的纯蒸馏适应阶段

入口：`scripts/training/run_register_distill.sh`，配置：
`navsim/planning/script/config/training/register_distill_4gpu.yaml`。

冻结基础 planner、视觉适配器、轨迹编码器、DiT 和 teacher registers（包括评分头）；
仅 `student_far.*` 参数参与优化，冻结组件始终保持 eval 模式。
保留原有 navtrain 缓存和 train/val 划分。兼容读取原 trajectory target 缓存，
但此阶段 forward 不使用 target，不加载 metric cache，不构造或调用 PDM simulator。

每个 scene 用冻结 planner 的预测综合分数选出 Top-16：保留前三条，
再从其余 13 条无放回随机选三条，共六条，一次性送入 teacher。
这是与原始 PDM winner/loser 不同的采样方案。训练每次重新抽样；
验证按固定种子抽样并固定生成噪声。同一 scene 的六条候选共享初始噪声。

teacher 使用原先的五个检索 motion tokens；student 使用原始 FAR 推理的构造方法：
`trajencoder_wm.traj_vocab_encoder(norm_odo(generating_anchor)) +
trajencoder_wm.traj_offset_encoder(planner_offset)`。
保留生成 anchor 的索引，不能重新检索最近 anchor，也不是截取五个 teacher tokens 中的第一个。
student 的 motion 输入为 `[B*6,1,512]`，teacher 为 `[B*6,5,512]`。

student 的第一个 4 层 decoder 使用 `[visual tokens, motion token]` 作为
cross-attention memory，使输出的 36 个 local registers 从这一阶段就依赖候选动作。
随后将 local registers、ego token、motion token 拼接，经过原有的 4 层 fusion
self-attention，取前 36 个 registers 蒸馏。加入 motion memory 不新增参数，
epoch05 权重可严格加载；前向行为改变后通过本阶段继续适应。

teacher 在 36 scene registers、一个 ego token、五个 motion tokens 交互后取前 36 个；
student 同样输出 `[B*6,36,512]`。不执行 teacher scorer heads；唯一目标为
`loss = distill_loss_weight * mean((student.float() - teacher.float()) ** 2)`，默认权重 1。
日志中的 `register_distill_loss` 与 `loss` 是同一个 MSE 的原值/加权值，不会相加。

## 启动

默认双方均从已有 `epoch=05-step=3552` 联合 checkpoint 初始化，严格加载全部需要的参数。
改成单 motion token 不改变 FAR 参数形状，因此保留 student 权重。
初始 LR 为 `5e-5`（上一阶段 `2e-4` 的四分之一），默认 10 epochs、每卡 36 scenes、
每个 epoch 验证和保存一次。会新建 optimizer 和 scheduler，不恢复上一阶段优化器。
W&B 配置与联合阶段一致，从外部 `wandb.md` 读取 key。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash /mnt/cfs/qiuzijun/WorldDrive/scripts/training/run_register_distill.sh
```

指定 teacher 快照和可选的独立 student 初始快照（否则 student 使用同一文件）：

```bash
export TEACHER_REGISTER_CHECKPOINT='/absolute/path/to/joint.ckpt'
export STUDENT_INIT_CHECKPOINT='/absolute/path/to/joint.ckpt'
bash /mnt/cfs/qiuzijun/WorldDrive/scripts/training/run_register_distill.sh
```

如果中断后续训本阶段，设置 `DISTILL_RESUME_CHECKPOINT` 为本阶段生成的 checkpoint。
它会恢复本阶段 optimizer/scheduler/global step；不要用联合阶段 checkpoint 作为该参数。
`TEACHER_REGISTER_CHECKPOINT` / `STUDENT_INIT_CHECKPOINT` 是权重初始化，
与 full-state resume 不同；原始联合训练使用的 `RESUME_CHECKPOINT` 环境变量在此不生效。

```bash
export DISTILL_RESUME_CHECKPOINT='/absolute/path/to/distill-epoch=00-step=592.ckpt'
bash /mnt/cfs/qiuzijun/WorldDrive/scripts/training/run_register_distill.sh
```

## 验证

`tests/test_register_distill.py` 检查采样、原 anchor 单 token 编码、冻结模式、
不执行评分头且 teacher token 不变、student 权重兼容，以及第一个 decoder 的
输出随 motion 变化且不同候选相互独立。
`scripts/check_register_distill.py` 用真实 checkpoint 和两条缓存样本执行单 GPU
Lightning 训练及验证，断言只有 student 有梯度并更新、teacher/student motion 分别为 5/1，
以及没有 PDM 调用。结果写入 `outputs/register_readiness/distill_single_motion_smoke.json`。
