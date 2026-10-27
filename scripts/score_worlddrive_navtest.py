"""Stream candidate-oracle PDM scoring with standard pairwise normalization."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'outputs/.matplotlib'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT/'outputs/.numba'))

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import json
import lzma
import pickle
import time
import traceback
import numpy as np

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score, transform_trajectory, get_trajectory_as_array
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex as MI, WeightedMetricIndex as WI

METRICS = ['no_at_fault_collisions','drivable_area_compliance','ego_progress','time_to_collision_within_bound','comfort','driving_direction_compliance','score']


def score_scene(scene, root):
    start = time.perf_counter()
    root = Path(root)
    sample = np.load(root/'candidates'/(scene['token']+'.npz'))
    trajectories = sample['proposals']
    assert trajectories.shape == (256,8,3)
    with lzma.open(scene['metric_cache'],'rb') as f:
        cache = pickle.load(f)
    sampling = TrajectorySampling(num_poses=40, interval_length=.1)
    candidate_sampling = TrajectorySampling(num_poses=8, interval_length=.5)
    simulator, scorer = PDMSimulator(sampling), PDMScorer(sampling)
    initial = cache.ego_state
    reference = get_trajectory_as_array(cache.trajectory,sampling,initial.time_point)
    states = [reference]
    for trajectory in trajectories:
        transformed = transform_trajectory(Trajectory(trajectory,candidate_sampling),initial)
        states.append(get_trajectory_as_array(transformed,sampling,initial.time_point))
    simulated = simulator.simulate_proposals(np.stack(states),initial)
    scorer.score_proposals(simulated,cache.observation,cache.centerline,cache.route_lane_ids,cache.drivable_area_map)
    # Only progress normalization couples proposals. Restore the EXACT standard
    # pair context (reference, this candidate) separately for every candidate.
    multiply = scorer._multi_metrics.prod(axis=0)
    raw_progress = scorer._progress_raw*multiply
    denominator = np.maximum(raw_progress[0],raw_progress[1:])
    progress = np.where(multiply[1:] == 0,0.,1.)
    moving = denominator > scorer._config.progress_distance_threshold
    progress[moving] = raw_progress[1:][moving]/denominator[moving]
    weighted = scorer._weighted_metrics[:,1:].copy()
    weighted[WI.PROGRESS] = progress
    weights = scorer._config.weighted_metrics_array
    scores = multiply[1:]*(weighted*weights[:,None]).sum(axis=0)/weights.sum()
    metrics = np.column_stack([scorer._multi_metrics[MI.NO_COLLISION,1:],scorer._multi_metrics[MI.DRIVABLE_AREA,1:],progress,weighted[WI.TTC],weighted[WI.COMFORTABLE],weighted[WI.DRIVING_DIRECTION],scores])
    collision_time = scorer._collision_time_idcs[1:]*.1
    ttc_time = scorer._ttc_time_idcs[1:]*.1
    assert np.isfinite(metrics).all()
    # Every scene checks both deployed selections against independent official
    # single-trajectory evaluation, not only a small smoke-test subset.
    selected = sorted(set([int(sample['stage1_id']),int(sample['stage2_id'])]))
    max_error = 0.
    for idx in selected:
        result = pdm_score(cache,Trajectory(trajectories[idx],candidate_sampling),sampling,PDMSimulator(sampling),PDMScorer(sampling))
        standard = np.array([asdict(result)[key] for key in METRICS])
        max_error = max(max_error,float(np.max(np.abs(standard-metrics[idx]))))
        np.testing.assert_allclose(metrics[idx],standard,rtol=1e-8,atol=1e-8)
    target = root/'pdm'/(scene['token']+'.npz')
    temp = target.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez(f,metrics=metrics,metric_names=METRICS,raw_progress_m=scorer._progress_raw[1:],collision_time_s=collision_time,ttc_time_s=ttc_time,single_trajectory_max_error=max_error)
    os.replace(temp,target)
    return {'token':scene['token'],'seconds':time.perf_counter()-start,'stage1':float(scores[int(sample['stage1_id'])]),'stage2':float(scores[int(sample['stage2_id'])]),'oracle':float(scores.max()),'max_error':max_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/navtest_worlddrive_stage2_full')
    parser.add_argument('--workers',type=int,default=24)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    out=args.output
    (out/'pdm').mkdir(exist_ok=True);(out/'errors').mkdir(exist_ok=True)
    scenes=json.loads((out/'manifest.json').read_text())
    if args.limit:
        scenes=scenes[:args.limit]
    remaining={s['token']:s for s in scenes if not (out/'pdm'/(s['token']+'.npz')).is_file()}
    existing=len(scenes)-len(remaining)
    done=0;failures={};seconds=[];started=time.perf_counter()
    inflight={}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while remaining or inflight:
            for token,scene in list(remaining.items()):
                if len(inflight)>=2*args.workers:
                    break
                if (out/'candidates'/(token+'.npz')).is_file():
                    inflight[pool.submit(score_scene,scene,str(out))]=token
                    del remaining[token]
            if not inflight:
                if not args.watch:
                    break
                time.sleep(5)
                continue
            finished,_=wait(inflight,timeout=10,return_when=FIRST_COMPLETED)
            for future in finished:
                token=inflight.pop(future)
                try:
                    result=future.result();done+=1;seconds.append(result['seconds'])
                except Exception:
                    failures[token]=traceback.format_exc()
                    (out/'errors'/(token+'.txt')).write_text(failures[token])
                    print(f'FAILED {token}: {failures[token]}',flush=True)
            progress={'total':len(scenes),'completed':existing+done,'existing':existing,'new_done':done,'failed':len(failures),'inflight':len(inflight),'waiting':len(remaining),'elapsed_s':time.perf_counter()-started,'mean_worker_seconds':float(np.mean(seconds)) if seconds else None,'status':'running'}
            temp=out/'progress_pdm.tmp';temp.write_text(json.dumps(progress,indent=2)+'\n');os.replace(temp,out/'progress_pdm.json')
            if finished and (done%50==0 or not remaining and not inflight):
                print(json.dumps(progress),flush=True)
    progress={'total':len(scenes),'completed':existing+done,'failed':len(failures),'remaining':len(remaining),'status':'complete' if not failures and not remaining else 'incomplete','elapsed_s':time.perf_counter()-started}
    (out/'progress_pdm.json').write_text(json.dumps(progress,indent=2)+'\n')
    print(json.dumps(progress),flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} PDM failures; inspect errors/')


if __name__=='__main__':
    main()
