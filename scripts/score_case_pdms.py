"""Compute simulator-based NAVSIM-v1 PDM scores for saved case proposals."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('NUPLAN_MAPS_ROOT', '/mnt/cfs/datasets/navsim-v1/maps')
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / 'outputs/.numba'))

import argparse
from dataclasses import asdict
import inspect
import json
import lzma
import pickle
import numpy as np

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Scene, SensorConfig, Trajectory
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, default=ROOT / 'outputs/single_scene')
    args = parser.parse_args()
    out = args.case / 'pdms'
    out.mkdir(exist_ok=True)
    report = json.loads((args.case / 'result.json').read_text())
    saved = np.load(args.case / 'all_proposals.npz')
    with open(report['log'], 'rb') as f:
        frames = pickle.load(f)
    idx = report['frame_index']
    # Cache processor needs 5 seconds of observations (10 future frames),
    # while candidate trajectories and PDM score use a 4-second horizon.
    window = frames[idx-3:idx+11]
    assert len(window) == 14 and window[3]['token'] == report['token']
    scene = Scene.from_scene_dict_list(window, None, num_history_frames=4, num_future_frames=10, sensor_config=SensorConfig.build_no_sensors())
    np.testing.assert_allclose(scene.get_future_trajectory(8).poses, saved['ground_truth'], atol=1e-4)
    scenario = NavSimScenario(scene, os.environ['NUPLAN_MAPS_ROOT'], 'nuplan-maps-v1.0')
    assert scenario.token == report['token']
    print('Building metric cache for', scenario.token, scene.scene_metadata.map_name, flush=True)
    processor = MetricCacheProcessor(str(out / 'metric_cache'), force_feature_computation=False)
    metadata = processor.compute_metric_cache(scenario)
    try:
        with lzma.open(metadata.file_name, 'rb') as f:
            cache = pickle.load(f)
    except (EOFError, lzma.LZMAError, pickle.UnpicklingError):
        print('Rebuilding incomplete metric cache left by interrupted write', flush=True)
        processor = MetricCacheProcessor(str(out / 'metric_cache'), force_feature_computation=True)
        metadata = processor.compute_metric_cache(scenario)
        with lzma.open(metadata.file_name, 'rb') as f:
            cache = pickle.load(f)
    sampling = TrajectorySampling(num_poses=40, interval_length=.1)
    trajectory_sampling = TrajectorySampling(num_poses=8, interval_length=.5)
    config = PDMScorerConfig()
    candidates = [(str(int(i)), saved['proposals'][i]) for i in saved['top5_indices']]
    candidates.append(('GT', saved['ground_truth']))
    rows = []
    for label, poses in candidates:
        # Standard pdm_score evaluates reference + one candidate, independently.
        simulator = PDMSimulator(sampling)
        scorer = PDMScorer(sampling, config)
        result = pdm_score(cache, Trajectory(poses, trajectory_sampling), sampling, simulator, scorer)
        row = {'candidate': label, **{k: float(v) for k, v in asdict(result).items()}}
        assert all(np.isfinite(v) for k, v in row.items() if k != 'candidate')
        row['score_100'] = row['score']*100
        row['reference_progress_raw_m'] = float(scorer._progress_raw[0])
        row['candidate_progress_raw_m'] = float(scorer._progress_raw[1])
        rows.append(row)
        print(json.dumps(row), flush=True)
    scores = {r['candidate']: r['score'] for r in rows}
    difference = scores['187']-scores['118']
    summary = {'token': scenario.token, 'map_name': scene.scene_metadata.map_name, 'metric_cache': str(metadata.file_name), 'pdm_score_source': inspect.getfile(pdm_score), 'simulator_source': inspect.getfile(PDMSimulator), 'scorer_source': inspect.getfile(PDMScorer), 'python': sys.executable, 'scorer_config': asdict(config), 'simulation_horizon_s': 4, 'simulation_interval_s': .1, 'candidate_interval_s': .5, 'cache_observation_horizon_s': 5, 'candidates': rows, 'score_187_minus_118': difference, 'winner_187_vs_118': 'tie' if abs(difference) < 1e-10 else '187' if difference > 0 else '118', 'note': 'Actual NAVSIM-v1 PDM simulator/scorer output, not learned FAR reward. Each candidate scored separately with the same PDM-Closed reference.'}
    (out / 'results.json').write_text(json.dumps(summary, indent=2)+'\n')
    import csv
    with (out / 'scores.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print('187 vs 118 winner:', summary['winner_187_vs_118'], 'difference:', difference, flush=True)


if __name__ == '__main__':
    main()
