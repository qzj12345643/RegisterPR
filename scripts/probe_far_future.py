"""Probe FAR future features for the exact previously saved Stage 1 top five.

VAE-decoded time-averaged features are diagnostic images, not future video frames.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKLCogVideoX

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navsim.agents.worlddrive.worlddrive_adapters import Adapters
from navsim.agents.worlddrive.worlddrive_planner import TrajEncoder, TrajWorldPlanner
from navsim.agents.worlddrive.worlddrive_refiner import TrajWorldRefiner


def similarity(values):
    x = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    norm = np.linalg.norm(x, axis=1)
    cosine = (x @ x.T) / np.maximum(norm[:, None]*norm[None, :], 1e-12)
    n = len(x)
    rmse = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            rmse[i, j] = np.sqrt(np.mean((x[i]-x[j])**2))
    rms = np.sqrt(np.mean(x*x, axis=1))
    return {'cosine': cosine.tolist(), 'rmse': rmse.tolist(), 'relative_rmse': (rmse / np.maximum((rms[:, None]+rms[None, :])/2, 1e-12)).tolist()}


def ssim(a, b):
    # Per-channel SSIM, 11x11 Gaussian window, sigma 1.5, RGB range [0,1].
    a, b = a.astype(np.float64), b.astype(np.float64)
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)[5:-5, 5:-5]
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a*a)-ma*ma, blur(b*b)-mb*mb, blur(a*b)-ma*mb
    return float((((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, default=ROOT / 'outputs/single_scene')
    args = parser.parse_args()
    out = args.case / 'far_future'
    out.mkdir(exist_ok=True)
    report = json.loads((args.case / 'result.json').read_text())
    saved = np.load(args.case / 'all_proposals.npz')
    ids = saved['top5_indices']
    original = saved['proposals'][ids].copy()
    anchors_np = np.load(ROOT / 'ckpts/trajectory_anchors_256.npy')
    ckpt_path = ROOT / 'ckpts/worlddrive_stage2_train_89p2_pdms.ckpt'
    expected = 'a9facb540cc8a76641629dc768e99464961d4964e377841e227db78f534ee158'
    digest = hashlib.sha256()
    with ckpt_path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise RuntimeError('Stage 2 checkpoint SHA256 does not match official file')
    torch.set_num_threads(8)
    torch.manual_seed(0)
    device = 'cuda:0'
    anchors = torch.from_numpy(anchors_np).float()
    modules = torch.nn.ModuleDict({'trajencoder': TrajEncoder(traj_vocab=anchors), 'trajplanner': TrajWorldPlanner(with_wm_proj=True), 'adapters': Adapters(inchannel_size=32, hidden_size=64), 'trajrefiner': TrajWorldRefiner()})
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    weights = {k.removeprefix('agent.'): v for k, v in checkpoint['state_dict'].items()}
    prefixes = sorted(set(k.split('.')[0] for k in weights))
    print('Checkpoint prefixes:', prefixes, flush=True)
    keys = set(modules.state_dict())
    missing = sorted(keys-set(weights))
    extra = sorted(set(weights)-keys)
    if missing:
        raise RuntimeError(f'Missing required FAR/planner weights: {missing}')
    modules.load_state_dict({k: weights[k] for k in keys}, strict=True)
    torch.testing.assert_close(modules['trajencoder'].traj_vocab.detach(), anchors)
    # Training forward_wm uses trajencoder for anchors + offsets. Some agent
    # evaluation paths instead refer to a separately frozen trajencoder_wm.
    encoder = modules['trajencoder']
    encoder_source = 'trajencoder (matches forward_wm training path)'
    if any(k.startswith('trajencoder_wm.') for k in weights):
        encoder = TrajEncoder(traj_vocab=anchors)
        encoder.load_state_dict({k.removeprefix('trajencoder_wm.'): v for k, v in weights.items() if k.startswith('trajencoder_wm.')}, strict=True)
        encoder_source = 'trajencoder_wm (checkpoint-provided)'
    stage1 = torch.load(ROOT / 'ckpts/worlddrive_stage1_train_87p7_pdms.ckpt', map_location='cpu', weights_only=False)['state_dict']
    differing = [k for k, v in stage1.items() if not torch.equal(v, weights[k.removeprefix('agent.')])]
    print('Strict load passed; Stage1-vs-Stage2 differing backbone keys:', len(differing), flush=True)
    del stage1, checkpoint, weights
    modules.to(device).eval()
    encoder.to(device).eval()
    vae = AutoencoderKLCogVideoX.from_pretrained(str(ROOT / 'ckpts/CogVideoX-2b'), subfolder='vae', local_files_only=True).to(device).eval()
    images = [np.asarray(Image.open(p).convert('RGB')) for p in report['image_paths']]
    resized = np.stack([cv2.resize(im[28:-28], (1024, 512)) for im in images])
    video = torch.from_numpy(np.concatenate([np.zeros_like(resized), resized])).float()
    video = ((video/255-.5)*2).permute(3, 0, 1, 2)[None].to(device)
    # New FAR modules consume extra RNG. Reset the posterior seed explicitly;
    # this controlled re-encoding is shared by all conditions below, while the
    # previously saved trajectory coordinates are used unchanged.
    torch.manual_seed(0)
    farthest = int(np.linalg.norm(saved['proposals'][..., :2]-original[0, :, :2], axis=-1).mean(axis=-1).argmax())
    gt = saved['ground_truth'].astype(np.float32)
    gt_id = int(np.linalg.norm(anchors_np[..., :2]-gt[None, :, :2], axis=-1).sum(axis=-1).argmin())
    trajectories = np.concatenate([original, gt[None], saved['proposals'][farthest:farthest+1], original[:1]], axis=0)
    anchor_ids = [*map(int, ids), gt_id, farthest, int(ids[0])]
    labels = [f'Top{i+1} #{idx}' for i, idx in enumerate(ids)] + ['GT trajectory', f'Far control #{farthest}', 'Top1 duplicate']
    captured = []
    handle = modules['trajrefiner'].refine_decoder.register_forward_hook(lambda module, inputs, result: captured.append(result.detach().cpu().numpy()[0]))
    feature_latents, decoded, scores = [], [], []
    with torch.inference_mode():
        latent = vae.encode(video).latent_dist.sample() * vae.config.scaling_factor
        visual = modules['adapters'](latent.permute(0, 2, 1, 3, 4))
        status = torch.tensor(report['status'], device=device)[None]
        _, stage2_scores, stage2_offsets, (visual_tokens, _) = modules['trajplanner'](visual, status, modules['trajencoder'](traj_train=True)[None], anchors.to(device), eval_mode=True)
        for i, (trajectory, anchor_id) in enumerate(zip(trajectories, anchor_ids)):
            traj = torch.from_numpy(trajectory).float().to(device)[None, None]
            anchor = encoder.traj_vocab[anchor_id][None, None]
            offset = traj-anchor
            embedding = encoder.traj_vocab_encoder(encoder.norm_odo(anchor).reshape(1, 1, -1)) + encoder.traj_offset_encoder(offset.reshape(1, 1, -1))
            _, future, _, reward = modules['trajrefiner'](traj, embedding, visual_tokens, None)
            # This 16x32x64 feature is supervised against temporally averaged
            # low-resolution future VAE latents, not a time-indexed latent video.
            latent_map = future[0, 0].T.reshape(16, 32, 64)
            diagnostic = vae.decode(latent_map[None, :, None] / vae.config.scaling_factor).sample[0, :, 0]
            rgb = (diagnostic.permute(1, 2, 0).clamp(-1, 1)/2+.5).cpu().numpy()
            feature_latents.append(latent_map.cpu().numpy())
            decoded.append(rgb)
            scores.append(float(reward.item()))
            Image.fromarray(np.round(rgb*255).astype(np.uint8)).save(out / f'condition_{i+1}.png')
            print(labels[i], 'FAR score', scores[-1], 'feature', tuple(latent_map.shape), flush=True)
    handle.remove()
    np.testing.assert_array_equal(trajectories[:5], original)
    np.testing.assert_allclose(feature_latents[0], feature_latents[-1], rtol=0, atol=1e-6)
    matrices = {'decoder_hidden_512': similarity(captured), 'projected_latent_16': similarity(feature_latents), 'decoded_rgb': similarity(decoded)}
    matrices['decoded_rgb']['ssim'] = [[ssim(a, b) for b in decoded] for a in decoded]
    pairs = np.triu_indices(5, 1)
    summaries = {}
    for space, metrics in matrices.items():
        summaries[space] = {}
        for name, matrix in metrics.items():
            values = np.asarray(matrix)[pairs]
            summaries[space][name] = {'min': float(values.min()), 'mean': float(values.mean()), 'max': float(values.max())}
    far_order = np.argsort(-np.array(scores[:5]))
    result = {'token': report['token'], 'labels': labels, 'anchor_ids': anchor_ids, 'far_scores': scores, 'far_ranking_of_original_top5': (far_order+1).tolist(), 'far_selected_anchor': int(ids[far_order[0]]), 'original_stage1_top5_unchanged': True, 'checkpoint': str(ckpt_path), 'checkpoint_sha256_verified': digest.hexdigest(), 'checkpoint_prefixes': prefixes, 'checkpoint_keys_outside_core_modules': extra, 'checkpoint_keys_not_used': [k for k in extra if not k.startswith('trajencoder_wm.')], 'backbone_keys_different_from_stage1': differing, 'trajectory_encoder_source': encoder_source, 'refiner_source': inspect.getfile(TrajWorldRefiner), 'latent_shape': [16, 32, 64], 'decoded_shape': list(decoded[0].shape), 'posterior_seed': 0, 'input_note': 'Re-encoded identical real history once with fixed seed; original trajectory coordinates unchanged; not exact original stochastic VAE cache.', 'interpretation': 'FAR time-averaged future latent decoded as a one-latent diagnostic image. Not a real future frame, not a generated video, not ground-truth similarity.', 'matrices': matrices, 'top5_pair_summaries': summaries}
    (out / 'results.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(out / 'features_and_scores.npz', trajectories=trajectories, anchor_ids=anchor_ids, decoder_hidden=np.stack(captured), future_latents=np.stack(feature_latents), decoded_rgb=np.stack(decoded), far_scores=scores)
    render(out, result, decoded, trajectories, images[-1])
    print(json.dumps({'far_order': result['far_ranking_of_original_top5'], 'top5_pair_summaries': summaries}, indent=2), flush=True)


def render(out, result, images, trajectories, current):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    axes.flat[0].imshow(current); axes.flat[0].set_title('Actual current front image')
    for i, im in enumerate(images):
        axes.flat[i+1].imshow(im)
        axes.flat[i+1].set_title(f'{result["labels"][i]} | FAR {result["far_scores"][i]:.4f}')
    for ax in axes.flat:
        ax.axis('off')
    fig.suptitle('FAR latent decode diagnostics — time-averaged features, NOT future video frames', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(out / 'decoded_comparison.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, space, metric, title in zip(axes, ['decoder_hidden_512', 'projected_latent_16', 'decoded_rgb'], ['cosine', 'relative_rmse', 'ssim'], ['Decoder hidden cosine', 'Projected latent relative RMSE', 'Diagnostic image SSIM']):
        matrix = np.array(result['matrices'][space][metric])[:7, :7]
        im = ax.imshow(matrix, cmap='viridis', vmin=0 if metric == 'relative_rmse' else None)
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center', fontsize=8, color='white' if matrix[i,j] < (matrix.min()+matrix.max())/2 else 'black')
        ax.set_xticks(range(7), result['labels'][:7], rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(7), result['labels'][:7], fontsize=8)
        ax.set_title(title); fig.colorbar(im, ax=ax, fraction=.046)
    fig.tight_layout(); fig.savefig(out / 'similarity_matrices.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    for i, ax in enumerate(axes, start=1):
        # Fixed shared display scale: 0..0.1 RGB absolute difference.
        difference = np.abs(images[i]-images[0]).mean(axis=-1)
        im = ax.imshow(difference, vmin=0, vmax=.1, cmap='inferno')
        ax.set_title(f'Top{i+1} vs Top1 | mean {difference.mean():.4f}'); ax.axis('off')
    fig.colorbar(im, ax=list(axes), fraction=.02, label='RGB absolute difference [0,1]')
    fig.savefig(out / 'top5_difference_maps.png', dpi=160, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    main()
