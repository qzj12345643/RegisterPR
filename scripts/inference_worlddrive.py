"""Single-scene Stage 1 inference using the official planner components.

No maps, PDM simulation, training dependencies, or world-model generator needed.
Only load trusted NAVSIM pickle files and official checkpoints.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKLCogVideoX

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navsim.agents.worlddrive.worlddrive_adapters import Adapters
from navsim.agents.worlddrive.worlddrive_planner import TrajEncoder, TrajWorldPlanner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/mnt/cfs/datasets/navsim-v1'))
    parser.add_argument('--split', default='trainval')
    parser.add_argument('--log', type=Path)
    parser.add_argument('--frame-index', type=int, help='Current frame; default: first complete scene')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'ckpts/worlddrive_stage1_train_87p7_pdms.ckpt')
    parser.add_argument('--anchors', type=Path, default=ROOT / 'ckpts/trajectory_anchors_256.npy')
    parser.add_argument('--vae', type=Path, default=ROOT / 'ckpts/CogVideoX-2b')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/single_scene')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use a GPU host or explicitly pass --device cpu')
    args.output.mkdir(parents=True, exist_ok=True)
    logs = [args.log] if args.log else sorted((args.data_root / 'navsim_logs' / args.split).glob('*.pkl'))
    selected = None
    for log in logs:
        with log.open('rb') as stream:
            frames = pickle.load(stream)
        indices = [args.frame_index] if args.frame_index is not None else range(3, len(frames)-8, 12)
        for idx in indices:
            if idx < 3 or idx + 8 >= len(frames):
                continue
            paths = [args.data_root / 'sensor_blobs' / args.split / f['cams']['CAM_F0']['data_path'] for f in frames[idx-3:idx+1]]
            if all(p.is_file() for p in paths):
                selected = (log, frames, idx, paths)
                break
        if selected:
            break
    if selected is None:
        raise FileNotFoundError('No scene with four front images and eight future poses was found')
    log, frames, idx, image_paths = selected
    current = frames[idx]
    print(f'Scene: {log.name}, frame {idx}, token {current["token"]}', flush=True)
    images = [np.asarray(Image.open(p).convert('RGB')) for p in image_paths]
    # Match TrajWorldFeatureBuilder: crop 28 rows, resize, prepend four black frames.
    resized = np.stack([cv2.resize(im[28:-28], (1024, 512)) for im in images])
    video = torch.from_numpy(np.concatenate([np.zeros_like(resized), resized])).float()
    video = (video / 255.0 - 0.5) * 2
    video = video.permute(3, 0, 1, 2).unsqueeze(0).to(device)
    status = torch.tensor(np.concatenate([current['driving_command'], current['ego_dynamic_state']]), dtype=torch.float32, device=device)[None]
    anchors = torch.from_numpy(np.load(args.anchors)).float()
    modules = torch.nn.ModuleDict({
        'trajencoder': TrajEncoder(traj_vocab=anchors),
        'trajplanner': TrajWorldPlanner(with_wm_proj=True),
        'adapters': Adapters(inchannel_size=32, hidden_size=64),
    })
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    weights = {k.removeprefix('agent.'): v for k, v in checkpoint['state_dict'].items()}
    modules.load_state_dict(weights, strict=True)
    torch.testing.assert_close(modules['trajencoder'].traj_vocab.detach(), anchors)
    del checkpoint, weights
    modules.to(device).eval()
    print('All planner checkpoint keys matched (strict=True).', flush=True)
    vae = AutoencoderKLCogVideoX.from_pretrained(str(args.vae), subfolder='vae', local_files_only=True).to(device).eval()
    start = time.perf_counter()
    with torch.inference_mode():
        # Official feature caching uses float32 and samples the posterior.
        latent = vae.encode(video).latent_dist.sample() * vae.config.scaling_factor
        visual = latent.permute(0, 2, 1, 3, 4)
        print(f'Encoded real images: {tuple(visual.shape)}', flush=True)
        del vae, video
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        adapted = modules['adapters'](visual)
        embedding = modules['trajencoder'](traj_train=True)[None]
        trajectory, rewards, offsets, _ = modules['trajplanner'](adapted, status, embedding, anchors.to(device), eval_mode=True)
        prediction = trajectory[0].cpu().numpy()
        rewards = rewards[0].cpu().numpy()
    elapsed = time.perf_counter() - start
    if prediction.shape != (8, 3) or not np.isfinite(prediction).all() or not np.isfinite(rewards).all():
        raise RuntimeError('Invalid model output')
    # NAVSIM ground truth: future global poses expressed in the current ego frame.
    def pose(frame):
        w, x, y, z = frame['ego2global_rotation']
        yaw = np.arctan2(2 * (w*z - x*y), 1 - 2 * (y*y + z*z))
        return np.array([*frame['ego2global_translation'][:2], yaw])
    origin = pose(current)
    future = np.stack([pose(f) for f in frames[idx+1:idx+9]])
    c, s = np.cos(origin[2]), np.sin(origin[2])
    gt = np.empty((8, 3))
    gt[:, :2] = (future[:, :2] - origin[:2]) @ np.array([[c, -s], [s, c]])
    gt[:, 2] = (future[:, 2] - origin[2] + np.pi) % (2*np.pi) - np.pi
    distances = np.linalg.norm(prediction[:, :2] - gt[:, :2], axis=-1)
    report = {
        'token': current['token'], 'log': str(log), 'split': args.split,
        'frame_index': idx, 'checkpoint': str(args.checkpoint), 'vae': str(args.vae),
        'image_paths': list(map(str, image_paths)), 'seed': args.seed,
        'device': str(device), 'torch_version': torch.__version__,
        'strict_checkpoint_load': True, 'visual_shape': list(visual.shape),
        'status': status[0].cpu().tolist(), 'trajectory_xy_heading': prediction.tolist(),
        'ground_truth_xy_heading': gt.tolist(), 'horizon_seconds': 4.0,
        'interval_seconds': 0.5, 'selected_anchor': int(rewards.argmax()),
        'single_sample_ADE_m': float(distances.mean()), 'single_sample_FDE_m': float(distances[-1]),
        'encode_and_plan_seconds': elapsed,
        'note': 'Single trainval scene smoke test; not a NAVSIM benchmark or PDM score.',
    }
    np.save(args.output / 'trajectory.npy', prediction)
    np.savez(args.output / 'planner_outputs.npz', trajectory=prediction, rewards=rewards, offsets=offsets.cpu().numpy(), ground_truth=gt)
    (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(images[-1]); axes[0].axis('off'); axes[0].set_title('NAVSIM current front camera')
    axes[1].plot(np.r_[0, prediction[:, 0]], np.r_[0, prediction[:, 1]], 'o-', label='WorldDrive Stage 1')
    axes[1].plot(np.r_[0, gt[:, 0]], np.r_[0, gt[:, 1]], 'o--', label='Recorded future')
    axes[1].set(xlabel='Forward x (m)', ylabel='Left y (m)', title='4-second trajectory (ego frame)')
    axes[1].axis('equal'); axes[1].grid(True); axes[1].legend()
    fig.tight_layout(); fig.savefig(args.output / 'trajectory.png', dpi=150); plt.close(fig)
    print(json.dumps({k: report[k] for k in ['token', 'selected_anchor', 'single_sample_ADE_m', 'single_sample_FDE_m', 'encode_and_plan_seconds']}, indent=2), flush=True)
    print(f'Output: {args.output}', flush=True)


if __name__ == '__main__':
    main()
