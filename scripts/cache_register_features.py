"""Resumable, log-sharded WorldDrive cache; run via cache_register_features.sh."""
import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from navsim.common.dataclasses import AgentInput, Scene, SensorConfig
from navsim.agents.worlddrive.worlddrive_features import TrajWorldFeatureBuilder, TrajectoryTargetBuilder
from navsim.planning.training.dataset import dump_feature_target_to_pickle, load_feature_target_from_pickle
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


def atomic_cache(path, value):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    for attempt in range(3):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            dump_feature_target_to_pickle(tmp, value)
            # Exercise the same deserializer used by training before committing.
            restored = load_feature_target_from_pickle(tmp)
            for key, tensor in value.items():
                assert torch.equal(tensor, restored[key]), key
            os.replace(tmp, path)
            return
        except FileNotFoundError:
            # Network filesystems may report ENOENT after a rename was committed
            # but its reply was lost. Accept only an exact verified destination;
            # otherwise rewrite the same computed tensors, with bounded retries.
            if path.is_file():
                restored = load_feature_target_from_pickle(path)
                if restored.keys() == value.keys() and all(torch.equal(tensor, restored[key]) for key, tensor in value.items()):
                    return
            if attempt == 2:
                raise
            print(f'Retrying cache write: {path}', flush=True)
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/mnt/cfs/datasets/navsim-v1'))
    parser.add_argument('--cache-path', type=Path, default=Path('exp/worlddrive_train_cache'))
    parser.add_argument('--vae-path', default='ckpts/CogVideoX-2b')
    parser.add_argument('--filter', type=Path, default=Path('navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml'))
    parser.add_argument('--limit', type=int, default=0, help='Maximum samples per worker, 0 means full split')
    parser.add_argument('--status-dir', type=Path, default=Path('exp/register_feature_caching'))
    parser.add_argument('--min-free-gib', type=float, default=24, help='Wait for this much usable CUDA memory before each sample')
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    torch.set_num_threads(2)
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    spec = yaml.safe_load(args.filter.read_text())
    tokens = set(spec['tokens'])
    logs = sorted(spec['log_names'])[rank::world]
    sensors = SensorConfig.build_all_sensors(False)
    sensors.cam_f0 = True

    def wait_for_memory():
        while True:
            free, _ = torch.cuda.mem_get_info()
            reusable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            if free + reusable >= args.min_free_gib * 2**30:
                return
            print(f'rank={rank} waiting for GPU memory: {(free+reusable)/2**30:.1f} GiB available, need {args.min_free_gib}', flush=True)
            time.sleep(30)

    wait_for_memory()
    builder = TrajWorldFeatureBuilder(pretrained_vae_path=args.vae_path, cache_mode=True)
    target_builder = TrajectoryTargetBuilder(TrajectorySampling(num_poses=8, interval_length=0.5))
    args.status_dir.mkdir(parents=True, exist_ok=True)
    stats = dict(rank=rank, world_size=world, generated=0, skipped=0, failed=0, logs_done=0, total_logs=len(logs), complete=False)
    started = time.time()

    def status():
        stats['elapsed_seconds'] = round(time.time() - started, 1)
        tmp = args.status_dir / f'rank{rank}.json.tmp'
        tmp.write_text(json.dumps(stats, indent=2))
        os.replace(tmp, args.status_dir / f'rank{rank}.json')
        print(json.dumps(stats), flush=True)

    status()
    for log in logs:
        with (args.data_root / 'navsim_logs/trainval' / f'{log}.pkl').open('rb') as stream:
            frames = pickle.load(stream)
        for index, frame in enumerate(frames):
            token = frame['token']
            if token not in tokens:
                continue
            destination = args.cache_path / log / token
            feature_path = destination / 'trajworld_feature.gz'
            target_path = destination / 'trajectory_target.gz'
            if feature_path.is_file() and target_path.is_file():
                stats['skipped'] += 1
                continue
            try:
                assert index >= 3 and index + 8 < len(frames), (log, token, index)
                agent_input = AgentInput.from_scene_dict_list(frames[index-3:index+1], args.data_root / 'sensor_blobs/trainval', 4, sensors)
                # The target builder only needs ego poses and history length. Reuse
                # the official Scene trajectory transform without loading maps.
                scene = Scene(
                    scene_metadata=SimpleNamespace(num_history_frames=4),
                    map_api=None,
                    frames=[SimpleNamespace(ego_status=Scene._build_ego_status(f)) for f in frames[index-3:index+9]],
                )
                torch.manual_seed(int(hashlib.sha256(token.encode()).hexdigest()[:8], 16))
                wait_for_memory()
                with torch.inference_mode():
                    features = builder.compute_features(agent_input)
                    targets = target_builder.compute_targets(scene)
                assert features['visual_token'].shape == (2, 16, 64, 128)
                assert features['low_visual_token'].shape == (2, 16, 32, 64)
                assert targets['trajectory'].shape == (8, 3)
                assert all(torch.isfinite(v).all() for v in list(features.values()) + list(targets.values()))
                destination.mkdir(parents=True, exist_ok=True)
                atomic_cache(feature_path, features)
                atomic_cache(target_path, targets)
                stats['generated'] += 1
                stats['last_token'] = token
            except Exception as error:
                stats['failed'] += 1
                status()
                raise RuntimeError(f'Failed {log}/{token}: {error}') from error
            if stats['generated'] <= 3 or stats['generated'] % 20 == 0:
                status()
            if args.limit and stats['generated'] >= args.limit:
                stats['complete'] = True
                status()
                return
        stats['logs_done'] += 1
        status()
    stats['complete'] = True
    status()


if __name__ == '__main__':
    main()
