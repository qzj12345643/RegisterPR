"""Resumable multi-GPU shards of WorldDrive Stage1 + FAR inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor

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


def load_video(scene):
    images = [cv2.resize(np.asarray(Image.open(p).convert('RGB'))[28:-28], (1024, 512)) for p in scene['image_paths']]
    array = np.stack(images)
    return np.concatenate([np.zeros_like(array), array])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/navtest_worlddrive_stage2_full')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--save-visual', action='store_true')
    args = parser.parse_args()
    out = args.output
    candidates_dir = out/'candidates'; candidates_dir.mkdir(exist_ok=True)
    visual_dir = out/'visual_cache'
    if args.save_visual:
        visual_dir.mkdir(exist_ok=True)
    manifest = json.loads((out/'manifest.json').read_text())
    if args.limit:
        manifest = manifest[:args.limit]
    scenes = manifest[args.rank::args.world_size]
    pending = [s for s in scenes if not (candidates_dir/(s['token']+'.npz')).is_file()]
    torch.set_num_threads(4); cv2.setNumThreads(1)
    torch.manual_seed(0)
    device = 'cuda:0'
    anchors = torch.from_numpy(np.load(ROOT/'ckpts/trajectory_anchors_256.npy')).float()
    model = torch.nn.ModuleDict({'trajencoder': TrajEncoder(traj_vocab=anchors), 'trajplanner': TrajWorldPlanner(with_wm_proj=True), 'adapters': Adapters(inchannel_size=32, hidden_size=64), 'trajrefiner': TrajWorldRefiner(), 'trajencoder_wm': TrajEncoder(traj_vocab=anchors)})
    raw = torch.load(ROOT/'ckpts/worlddrive_stage2_train_89p2_pdms.ckpt', map_location='cpu', weights_only=False)['state_dict']
    weights = {k.removeprefix('agent.'): v for k,v in raw.items()}
    model.load_state_dict({k: weights[k] for k in model.state_dict()}, strict=True)
    torch.testing.assert_close(model['trajencoder'].traj_vocab.detach(), anchors)
    torch.testing.assert_close(model['trajencoder_wm'].traj_vocab.detach(), anchors)
    del raw, weights
    model.to(device).eval()
    vae = AutoencoderKLCogVideoX.from_pretrained(str(ROOT/'ckpts/CogVideoX-2b'), subfolder='vae', local_files_only=True).to(device).eval()
    anchors = anchors.to(device)
    captured = {}
    def hook(name):
        return lambda module, inputs, output: captured.__setitem__(name, output)
    handles = [model['trajplanner'].cls_branch.register_forward_hook(hook('imitation'))]
    handles += [head.register_forward_hook(hook(f'metric_{i}')) for i, head in enumerate(model['trajplanner'].sim_reward_heads)]
    progress = {'rank': args.rank, 'total': len(scenes), 'already_done': len(scenes)-len(pending), 'new_done': 0, 'status': 'running', 'batch_size': args.batch_size, 'save_visual': args.save_visual}
    metadata = {'rank': args.rank, 'torch': torch.__version__, 'dtype': 'float32', 'vae_posterior': 'sample; seed=int(token,16) % (2**31-1)', 'planner_source': str(ROOT/'navsim/agents/worlddrive/worlddrive_planner.py'), 'checkpoint': str(ROOT/'ckpts/worlddrive_stage2_train_89p2_pdms.ckpt'), 'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32, 'batch_size': args.batch_size, 'candidate_count':256, 'far_topk':5}
    (out/f'inference_metadata_rank{args.rank}.json').write_text(json.dumps(metadata, indent=2)+'\n')
    start = time.perf_counter()
    print(f'Rank {args.rank}: {len(pending)} pending / {len(scenes)} total', flush=True)
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=4) as io_pool:
        vocab_embed = model['trajencoder'](traj_train=True)[None]
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset:offset+args.batch_size]
            videos = np.stack(list(io_pool.map(load_video, batch)))
            video = torch.from_numpy(videos).to(device=device, dtype=torch.float32).permute(0, 4, 1, 2, 3)
            video = (video/255-.5)*2
            posterior = vae.encode(video).latent_dist
            for b, scene in enumerate(batch):
                seed = int(scene['token'],16) % (2**31-1)
                generator = torch.Generator(device=device).manual_seed(seed)
                mean, std = posterior.mean[b:b+1], posterior.std[b:b+1]
                latent = (mean+std*torch.randn(mean.shape, generator=generator, device=device, dtype=mean.dtype))*vae.config.scaling_factor
                raw_visual = latent.permute(0,2,1,3,4)
                visual = model['adapters'](raw_visual)
                status = torch.tensor(scene['status'], device=device, dtype=torch.float32)[None]
                prediction, scores, offsets, (visual_tokens, _) = model['trajplanner'](visual, status, vocab_embed, anchors, eval_mode=True)
                proposals = anchors+offsets[0].reshape(256,8,3)
                top_scores, ids = scores.topk(5, dim=1)
                selected_anchors = model['trajencoder_wm'].traj_vocab[ids]
                norm = model['trajencoder_wm'].norm_odo(selected_anchors)
                selected_offset = offsets[:,ids[0]].reshape(1,5,8,3)
                top_traj = selected_anchors+selected_offset
                embedding = model['trajencoder_wm'].traj_vocab_encoder(norm.reshape(1,5,-1))+model['trajencoder_wm'].traj_offset_encoder(selected_offset.reshape(1,5,-1))
                far_trajectory, _, _, far_weight = model['trajrefiner'](top_traj, embedding, visual_tokens, None)
                stage1_id = int(scores.argmax().item())
                far_rank = int(far_weight[0,:,0].argmax().item())
                stage2_id = int(ids[0,far_rank].item())
                torch.testing.assert_close(prediction[0], proposals[stage1_id])
                torch.testing.assert_close(far_trajectory[0], proposals[stage2_id])
                assert torch.isfinite(proposals).all() and torch.isfinite(scores).all() and torch.isfinite(far_weight).all()
                data = {'proposals':proposals.cpu().numpy(), 'stage1_scores':scores[0].cpu().numpy(), 'top5_ids':ids[0].cpu().numpy(), 'far_scores':far_weight[0,:,0].cpu().numpy(), 'stage1_id':stage1_id, 'stage2_id':stage2_id, 'imitation_probability':captured['imitation'][0,:,0].softmax(dim=0).cpu().numpy(), 'predicted_metrics':torch.stack([captured[f'metric_{i}'][0,:,0].sigmoid() for i in range(5)], dim=-1).cpu().numpy(), 'posterior_seed':seed}
                if args.save_visual:
                    with (visual_dir/(scene['token']+'.tmp')).open('wb') as f:
                        np.save(f, raw_visual[0].cpu().numpy())
                    os.replace(visual_dir/(scene['token']+'.tmp'), visual_dir/(scene['token']+'.npy'))
                target = candidates_dir/(scene['token']+'.npz')
                temporary = candidates_dir/(scene['token']+'.tmp')
                with temporary.open('wb') as f:
                    np.savez(f, **data)
                os.replace(temporary, target)
                progress['new_done'] += 1
            del video, posterior
            elapsed = time.perf_counter()-start
            progress.update(elapsed_s=elapsed, seconds_per_scene=elapsed/max(progress['new_done'],1), peak_gpu_gb=torch.cuda.max_memory_allocated()/1024**3, completed=progress['already_done']+progress['new_done'])
            temp = out/f'progress_gpu{args.rank}.tmp'
            temp.write_text(json.dumps(progress, indent=2)+'\n')
            os.replace(temp,out/f'progress_gpu{args.rank}.json')
            if progress['new_done'] % 40 == 0 or offset+args.batch_size>=len(pending):
                print(json.dumps(progress), flush=True)
    progress['status']='complete'
    (out/f'progress_gpu{args.rank}.json').write_text(json.dumps(progress, indent=2)+'\n')
    for handle in handles:
        handle.remove()


if __name__ == '__main__':
    main()
