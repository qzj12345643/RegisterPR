"""Stratified multi-scene diagnostic of Stage 1 top-five proposal diversity."""
import argparse
import csv
import inspect
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/mnt/cfs/datasets/navsim-v1')


def ground_truth(frames, idx):
    def pose(f):
        w, x, y, z = f['ego2global_rotation']
        return np.array([*f['ego2global_translation'][:2], np.arctan2(2*(w*z-x*y), 1-2*(y*y+z*z))])
    origin = pose(frames[idx])
    future = np.stack([pose(f) for f in frames[idx+1:idx+9]])
    c, s = np.cos(origin[2]), np.sin(origin[2])
    future[:, :2] = (future[:, :2]-origin[:2]) @ np.array([[c, -s], [s, c]])
    future[:, 2] = (future[:, 2]-origin[2]+np.pi) % (2*np.pi)-np.pi
    return future


def prepare(out):
    # Select before seeing model predictions; at most one scene per log.
    rng = np.random.default_rng(20260917)
    logs = sorted((DATA / 'navsim_logs/trainval').glob('*.pkl'))
    rng.shuffle(logs)
    buckets = {name: [] for name in ['straight', 'left', 'right', 'slow_stop']}
    scanned = 0
    for log in logs:
        scanned += 1
        with log.open('rb') as f:
            frames = pickle.load(f)
        indices = list(range(3, len(frames)-8, 12))
        rng.shuffle(indices)
        for idx in indices:
            gt = ground_truth(frames, idx)
            distance = float(np.linalg.norm(gt[-1, :2]))
            heading = float(np.rad2deg(gt[-1, 2]))
            category = 'slow_stop' if distance < 2 else 'left' if heading > 15 else 'right' if heading < -15 else 'straight'
            if len(buckets[category]) >= 3:
                continue
            paths = [DATA / 'sensor_blobs/trainval' / f['cams']['CAM_F0']['data_path'] for f in frames[idx-3:idx+1]]
            if not all(p.is_file() for p in paths):
                continue
            buckets[category].append({'log': str(log), 'frame_index': idx, 'token': frames[idx]['token'], 'category': category, 'gt_end_distance_m': distance, 'gt_heading_change_deg': heading})
            print(f'Selected {category}: {frames[idx]["token"]}', flush=True)
            break
        if all(len(v) == 3 for v in buckets.values()):
            break
    samples = [item for bucket in buckets.values() for item in bucket]
    if not samples:
        raise RuntimeError('No complete scenes found')
    manifest = {'seed': 20260917, 'logs_scanned': scanned, 'selection': 'Shuffled logs and frame windows; max one scene/log; quota 3/category, selected before model inference', 'category_definition': 'slow_stop: GT endpoint distance <2m; else left/right: GT heading change >15/<-15 deg; else straight', 'samples': samples}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print({k: len(v) for k, v in buckets.items()}, flush=True)


def pair_distances(trajectories):
    xy = trajectories[..., :2]
    i, j = np.triu_indices(len(xy), 1)
    distances = np.linalg.norm(xy[i]-xy[j], axis=-1)
    return distances.mean(axis=1), distances[:, -1]


def shape_diversity(trajectories):
    # Compare paths at equal traveled distances, separating shape from speed.
    paths = [np.vstack([np.zeros((1, 2)), t[:, :2]]) for t in trajectories]
    arcs = [np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))] for p in paths]
    common = min(s[-1] for s in arcs)
    grid = np.linspace(0, common, 50)
    samples = []
    for p, s in zip(paths, arcs):
        s, indices = np.unique(s, return_index=True)
        samples.append(np.stack([np.interp(grid, s, p[indices, k]) for k in range(2)], axis=-1))
    distances, _ = pair_distances(np.stack(samples))
    return float(distances.mean()), float(distances.max()), float(common)


def run(out):
    import sys
    sys.path.insert(0, str(ROOT))
    import cv2
    import torch
    from PIL import Image
    from diffusers import AutoencoderKLCogVideoX
    from navsim.agents.worlddrive.worlddrive_adapters import Adapters
    from navsim.agents.worlddrive.worlddrive_planner import TrajEncoder, TrajWorldPlanner
    torch.set_num_threads(8)
    torch.manual_seed(0)
    device = 'cuda:0'
    anchors = torch.from_numpy(np.load(ROOT / 'ckpts/trajectory_anchors_256.npy')).float()
    modules = torch.nn.ModuleDict({'trajencoder': TrajEncoder(traj_vocab=anchors), 'trajplanner': TrajWorldPlanner(with_wm_proj=True), 'adapters': Adapters(inchannel_size=32, hidden_size=64)})
    ckpt = torch.load(ROOT / 'ckpts/worlddrive_stage1_train_87p7_pdms.ckpt', map_location='cpu', weights_only=False)
    modules.load_state_dict({k.removeprefix('agent.'): v for k, v in ckpt['state_dict'].items()}, strict=True)
    torch.testing.assert_close(modules['trajencoder'].traj_vocab.detach(), anchors)
    del ckpt
    modules.to(device).eval()
    vae = AutoencoderKLCogVideoX.from_pretrained(str(ROOT / 'ckpts/CogVideoX-2b'), subfolder='vae', local_files_only=True).to(device).eval()
    provenance = {'planner_source': inspect.getfile(TrajWorldPlanner), 'adapter_source': inspect.getfile(Adapters), 'data_root': str(DATA), 'torch': torch.__version__, 'checkpoint_strict_load': True, 'vae_dtype': str(vae.dtype), 'seed_per_scene': 0}
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    manifest = json.loads((out / 'manifest.json').read_text())
    with torch.inference_mode():
        embedding = modules['trajencoder'](traj_train=True)[None]
        for sample in manifest['samples']:
            dest = out / sample['token']
            dest.mkdir(exist_ok=True)
            with open(sample['log'], 'rb') as f:
                frames = pickle.load(f)
            idx = sample['frame_index']
            images = [np.asarray(Image.open(DATA / 'sensor_blobs/trainval' / f['cams']['CAM_F0']['data_path']).convert('RGB')) for f in frames[idx-3:idx+1]]
            resized = np.stack([cv2.resize(im[28:-28], (1024, 512)) for im in images])
            video = torch.from_numpy(np.concatenate([np.zeros_like(resized), resized])).float()
            video = ((video / 255-.5)*2).permute(3, 0, 1, 2)[None].to(device)
            torch.manual_seed(0)
            latent = vae.encode(video).latent_dist.sample() * vae.config.scaling_factor
            visual = latent.permute(0, 2, 1, 3, 4)
            current = frames[idx]
            status = torch.tensor(np.concatenate([current['driving_command'], current['ego_dynamic_state']]), dtype=torch.float32, device=device)[None]
            pred, rewards, offsets, _ = modules['trajplanner'](modules['adapters'](visual), status, embedding, anchors.to(device), eval_mode=True)
            proposals = anchors.numpy() + offsets[0].cpu().numpy().reshape(256, 8, 3)
            scores = rewards[0].cpu().numpy()
            top5 = np.argsort(-scores, kind='stable')[:5]
            prediction = pred[0].cpu().numpy()
            np.testing.assert_allclose(proposals[top5[0]], prediction, rtol=1e-5, atol=1e-5)
            assert np.isfinite(proposals).all() and np.isfinite(scores).all()
            gt = ground_truth(frames, idx)
            np.savez(dest / 'proposals.npz', proposals=proposals, anchors=anchors.numpy(), scores=scores, top5=top5, prediction=prediction, ground_truth=gt)
            Image.fromarray(images[-1]).resize((768, 432)).save(dest / 'front.png')
            (dest / 'sample.json').write_text(json.dumps(sample, indent=2)+'\n')
            print(f'Inferred {sample["category"]} {sample["token"]}, top5={top5.tolist()}', flush=True)
            del video, latent, visual


def analyze(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    samples = json.loads((out / 'manifest.json').read_text())['samples']
    rows = []
    colors = ['#e63946', '#0077b6', '#8e44ad', '#f4a261', '#16a085']
    fig, axes = plt.subplots(4, 3, figsize=(18, 20), squeeze=False)
    fig_all, axes_all = plt.subplots(4, 3, figsize=(18, 20), squeeze=False)
    fig_cam, axes_cam = plt.subplots(4, 3, figsize=(18, 12), squeeze=False)
    for ax in [*axes.flat, *axes_all.flat, *axes_cam.flat]:
        ax.set_visible(False)
    for n, sample in enumerate(samples):
        saved = np.load(out / sample['token'] / 'proposals.npz')
        proposals, ids, gt = saved['proposals'], saved['top5'], saved['ground_truth']
        selected = proposals[ids]
        ade, fde = pair_distances(selected)
        all_ade, _ = pair_distances(proposals)
        anchor_ade, _ = pair_distances(saved['anchors'][ids])
        shape_mean, shape_max, common = shape_diversity(selected)
        error = np.linalg.norm(saved['prediction'][:, :2]-gt[:, :2], axis=-1)
        row = dict(sample, top5_ids=','.join(map(str, ids)), pair_mean_m=float(ade.mean()), pair_max_m=float(ade.max()), endpoint_mean_m=float(fde.mean()), endpoint_max_m=float(fde.max()), all256_pair_mean_m=float(all_ade.mean()), selected_anchor_pair_mean_m=float(anchor_ade.mean()), shape_mean_m=shape_mean, shape_max_m=shape_max, common_path_m=common, top1_ADE_m=float(error.mean()), top1_FDE_m=float(error[-1]))
        rows.append(row)
        for ax, show_all in [(axes.flat[n], False), (axes_all.flat[n], True)]:
            ax.set_visible(True)
            if show_all:
                for proposal in proposals:
                    points = np.vstack([np.zeros((1, 2)), proposal[:, :2]])
                    ax.plot(points[:, 0], points[:, 1], color='#9aa5b1', alpha=.35, lw=.6)
            for rank in range(4, -1, -1):
                points = np.vstack([np.zeros((1, 2)), selected[rank, :, :2]])
                ax.plot(points[:, 0], points[:, 1], color=colors[rank], lw=2 if rank else 2.8, marker='o' if rank == 0 else None, ms=3)
            points = np.vstack([np.zeros((1, 2)), gt[:, :2]])
            ax.plot(points[:, 0], points[:, 1], 'k--', lw=1.8)
            ax.set_aspect('equal', adjustable='datalim')
            ax.grid(alpha=.2)
            ax.set(xlabel='Forward x (m)', ylabel='Left y (m)', title=f'{n+1}. {sample["category"]} | {sample["token"]}\nTop5 pair mean/max: {ade.mean():.2f}/{ade.max():.2f} m')
        ax = axes_cam.flat[n]
        ax.set_visible(True)
        ax.imshow(plt.imread(out / sample['token'] / 'front.png'))
        ax.axis('off')
        ax.set_title(f'{n+1}. {sample["category"]} | {sample["token"]}', fontsize=10)
    handles = [Line2D([], [], color=c, lw=2, label=f'Top {i+1}'+(' / FINAL' if i == 0 else '')) for i, c in enumerate(colors)]
    handles.append(Line2D([], [], color='black', ls='--', label='GT'))
    for figure, title, name in [(fig, 'Top 5 and GT — equal metric axes in every panel', 'top5_equal_scale'), (fig_all, 'All 256 proposals, Top 5 and GT — equal metric axes', 'all256_equal_scale')]:
        figure.suptitle(title, fontsize=18, y=.995)
        figure.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .98), ncol=6)
        figure.tight_layout(rect=(0, 0, 1, .96))
        figure.savefig(out / f'{name}.png', dpi=150)
        figure.savefig(out / f'{name}.pdf')
        plt.close(figure)
    fig_cam.tight_layout()
    fig_cam.savefig(out / 'scene_inputs.png', dpi=130)
    plt.close(fig_cam)
    with (out / 'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {'n': len(rows), 'metrics': rows, 'counts_pair_max_below_m': {str(t): sum(r['pair_max_m'] < t for r in rows) for t in [.5, 1, 2, 3]}, 'counts_shape_max_below_m': {str(t): sum(r['shape_max_m'] < t for r in rows) for t in [.25, .5, 1]}, 'median_pair_mean_m': float(np.median([r['pair_mean_m'] for r in rows])), 'median_shape_mean_m': float(np.median([r['shape_mean_m'] for r in rows]))}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = [
        '# WorldDrive Stage 1 Top 5 场景检查', '',
        '12 个新场景按固定随机种子预先选择，来自不同日志；直行、左转、右转、低速/停车各 3 个。这是分类型的小样本诊断，不是整个 NAVSIM 数据集的发生率估计。', '',
        '使用 WorldDrive/navsim/agents/worlddrive 中的原始规划模块；实际导入路径见 provenance.json。数据来自 /mnt/cfs/datasets/navsim-v1。未加载 DriveWorld-VLA/navsim，也未运行完整 NAVSIM PDM 评估。', '',
        '## 指标', '',
        '- Top5 配对距离：对 5 条轨迹的 10 对组合，在相同的 8 个未来时刻计算 XY 欧氏距离并取时间平均，再报告 10 对的均值/最大值。',
        '- 终点距离：同样 10 对在 4 秒处的距离。',
        '- 路径形状距离：从原点起按弧长重采样，在五条路径共有的行驶长度内比较 50 个等距位置；不比较共有长度之外的部分，停车/极短路径时解释价值有限。',
        '- 距离小于 1/2/3 米等阈值是本次诊断口径，不是官方指标或故障判据。', '',
        '## 结果', '',
        f'Top5 最大配对距离 < 1/2/3 米的场景数：{summary["counts_pair_max_below_m"]["1"]}/{summary["counts_pair_max_below_m"]["2"]}/{summary["counts_pair_max_below_m"]["3"]}，分母均为 {len(rows)}。', '',
        '| 场景 | 类型 | 配对均值 m | 配对最大 m | 终点最大 m | 形状最大 m | 共有长度 m |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        lines.append(f'| {r["token"]} | {r["category"]} | {r["pair_mean_m"]:.3f} | {r["pair_max_m"]:.3f} | {r["endpoint_max_m"]:.3f} | {r["shape_max_m"]:.3f} | {r["common_path_m"]:.2f} |')
    lines += ['', '## 机制与限制', '',
        'Stage 1 使用 imitation 与预测 NC/DAC/EP/TTC/comfort 的加权得分排序，最终直接选 argmax；这里展示的 Top5 为同一得分的前五名，没有 NMS、距离约束或多样性重排。得分相近的候选可以沿相同路径、只在速度或终点上有所差别。', '',
        '聚集本身不能证明模型出错或模式坍塌；还需要结合路径合理性、驾驶指令、GT 和碰撞/道路评估。这里没有执行 Stage 2 FAR。', '',
        '图 top5_equal_scale.png 和 all256_equal_scale.png 每个子图横纵轴保持等米制比例；各场景视野范围可不同。scene_inputs.png 给出实际相机输入。每个 token 子目录保存全部 256 条 proposal、得分、Top5 索引和 GT。', '',
        '复跑：先 prepare，再在空闲 GPU 上 run，最后 analyze。脚本为 scripts/analyze_top5_scenes.py。',
    ]
    (out / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'metrics'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'run', 'analyze'])
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/top5_multiscene')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    {'prepare': prepare, 'run': run, 'analyze': analyze}[args.phase](args.output)
