# WorldDrive 单场景推理

本地入口运行官方 Stage 1 规划模块，读取真实 NAVSIM 前视相机历史，输出未来 4 秒、间隔 0.5 秒的 8 个 `(x, y, heading)` 轨迹点。无需地图、PDM 缓存或完整世界模型。

## 环境

仓库内 `.venv` 使用 Python 3.10，基于本机 `/mnt/cfs/liuchupeng/miniconda/FastWAM` 的只读共享依赖创建，额外在 `.venv` 内安装 `diffusers==0.34.0` 和 `timm==1.0.28`。未修改基础环境。该环境依赖基础路径继续存在。

主要版本：PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128、transformers 4.44.0、numpy 1.26.4。最小推理依赖记录在 `requirements-inference.txt`；它不是完整 NAVSIM 训练环境。

原仓库 `requirements.txt` 指定 torch 2.0.1，而 `docs/Installation.md` 指定 torch 2.3.0、diffusers 0.34.0、transformers 4.56.2，二者并不一致。这个轻量入口直接使用原有 `Adapters`、`TrajEncoder`、`TrajWorldPlanner`，避免导入训练、PDM 和视频生成模块。

## 权重

来源：[WorldDrive](https://huggingface.co/tabguigui/WorldDrive/tree/main) 和 [CogVideoX-2b VAE](https://huggingface.co/zai-org/CogVideoX-2b/tree/main/vae)。

| 本地文件 | 字节数 |
| --- | ---: |
| `ckpts/worlddrive_stage1_train_87p7_pdms.ckpt` | 376024474 |
| `ckpts/trajectory_anchors_256.npy` | 24704 |
| `ckpts/CogVideoX-2b/vae/diffusion_pytorch_model.safetensors` | 862388596 |

上述文件已逐一核对官方 SHA256，记录见 `ckpts/download_manifest.json`。VAE 配置位于 `ckpts/CogVideoX-2b/vae/config.json`。

下载使用 `/mnt/cfs/qiuzijun/xray` 已运行的 HTTP 代理 `http://127.0.0.1:1080`。VAE 单连接遇到中断后，使用 aria2 的 8 连接断点续传完成。Stage 2 和 TA-DWM 生成权重未下载，简单轨迹推理不需要它们。

后续 FAR 测试已补充下载并验证 `ckpts/worlddrive_stage2_train_89p2_pdms.ckpt`（456784080 字节）。原 case 的 FAR 特征、VAE 诊断解码、相似度及报告位于 `outputs/single_scene/far_future/`；复跑脚本为 `scripts/probe_far_future.py`。FAR 的时间平均潜变量解码不等同于未来视频生成；TA-DWM 生成权重仍未下载。

## 运行

在仓库根目录执行，先确认所选 GPU 空闲：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run_simple_inference.sh
```

默认从 `/mnt/cfs/datasets/navsim-v1` 的 `trainval` 中选择四帧图像齐全的场景。默认场景日志为 `2021.05.12.19.36.12_veh-35_00005_00204.pkl`，当前帧索引 351，token `381ec4a38449526a`。可显式指定：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run_simple_inference.sh \
  --log /mnt/cfs/datasets/navsim-v1/navsim_logs/trainval/2021.05.12.19.36.12_veh-35_00005_00204.pkl \
  --frame-index 351 \
  --output outputs/single_scene
```

`--data-root`、`--split`、`--checkpoint`、`--anchors`、`--vae` 可覆盖路径。启动脚本默认禁用 Hugging Face 联网，推理只读本地权重。不要用原先的 `driveworld-vla` Python 启动此脚本，其 torch 2.0.1 环境不具备所需 diffusers 依赖。

## 处理与输出

图像处理遵循 `worlddrive_features.py`：RGB 前视图上下各裁 28 像素，缩放到 1024×512，四帧历史之前补四帧黑图，归一化至 `[-1,1]`，VAE posterior 采样并乘 scaling factor。默认随机种子为 0。只计算规划需要的高分辨率 latent。

规划器使用全部 221 个 checkpoint 项严格加载，并核对外部轨迹锚点与 checkpoint 一致。预测调用仓库原有奖励加权与轨迹选择函数。

输出目录：

- `result.json`：输入来源、状态、预测/真实轨迹、形状、时间、单样本 ADE/FDE。
- `trajectory.npy`：`[8,3]`，单位为米、米、弧度，位于当前自车坐标系。
- `planner_outputs.npz`：预测轨迹、256 个候选得分、偏移、真实轨迹。
- `trajectory.png`：真实前视图与轨迹对比。

本次是 trainval 单样本运行检查，不代表 navtest 基准性能，不产生 PDM 分数，也不生成未来视频。JSON 中的用时为 VAE 编码加规划耗时，不含模型加载，不是纯规划器延迟基准。

## 本机验证结果（2026-09-17）

使用物理 GPU 1（A800 80GB）成功运行，进程退出码 0。全部 221 个权重项严格匹配，锚点一致；真实图像编码输出 `[1,2,16,64,128]`，预测 `[8,3]` 和候选得分均为有限数值。

- 场景：`381ec4a38449526a`，选择候选 187。
- 单样本 ADE：0.33357 米；FDE：0.59382 米。
- VAE 编码加规划耗时：4.243 秒。
- 实际结果保存在 `outputs/single_scene/`，完整日志见 `logs/simple_inference.log`。

已检查生成的相机/轨迹对比图。当前共享文件系统首次加载 Python/CUDA 库较慢，启动等待时间可能明显长于上述推理耗时。
