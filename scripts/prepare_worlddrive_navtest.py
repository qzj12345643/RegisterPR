"""Build a complete, explicit NAVSIM navtest manifest; fail on missing inputs."""
import json
import pickle
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/navtest_worlddrive_stage2_full'
DATA = Path('/mnt/cfs/qiuzijun/DriveWorld-VLA/dataset')
CACHE = Path('/mnt/cfs/qiuzijun/DriveWorld-VLA/exp/metric_cache')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / 'navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml').read_text())
    wanted = set(cfg['tokens'])
    assert len(wanted) == len(cfg['tokens']) == 12146
    scenes, missing = [], []
    for name in sorted(cfg['log_names']):
        log = DATA / 'navsim_logs/test' / (name+'.pkl')
        with log.open('rb') as f:
            frames = pickle.load(f)
        for idx in range(3, len(frames)-10):
            current = frames[idx]
            if current['token'] not in wanted or not current['roadblock_ids']:
                continue
            paths = [str(DATA / 'sensor_blobs/test' / f['cams']['CAM_F0']['data_path']) for f in frames[idx-3:idx+1]]
            metric = CACHE / current['log_name'] / 'unknown' / current['token'] / 'metric_cache.pkl'
            for path in [*paths, str(metric)]:
                if not Path(path).is_file():
                    missing.append({'token': current['token'], 'path': path})
            scenes.append({'token': current['token'], 'log': str(log), 'frame_index': idx, 'image_paths': paths, 'status': np.concatenate([current['driving_command'], current['ego_dynamic_state']]).astype(float).tolist(), 'metric_cache': str(metric)})
    scenes.sort(key=lambda s: s['token'])
    found = {s['token'] for s in scenes}
    audit = {'expected': len(wanted), 'found': len(scenes), 'unique': len(found), 'missing_tokens': sorted(wanted-found), 'missing_files': missing, 'data_root': str(DATA), 'metric_cache_root': str(CACHE)}
    (OUT / 'input_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    (OUT / 'manifest.json').write_text(json.dumps(scenes, indent=2)+'\n')
    print(json.dumps({**audit, 'missing_files': missing[:10]}, indent=2), flush=True)
    assert found == wanted and len(scenes) == len(wanted) and not missing, 'Incomplete navtest inputs; see input_audit.json'


if __name__ == '__main__':
    main()
