"""Validate downloaded TA-DWM loading and one real five-candidate readout step."""
import json
import time
import argparse
from pathlib import Path

import torch
import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.worlddrive.future_register_agent import FutureRegisterAgent
from navsim.planning.training.dataset import load_feature_target_from_pickle

torch.set_num_threads(2)
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--weights-only', action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
agent = FutureRegisterAgent(
    trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
    checkpoint_path=str(root / 'ckpts/worlddrive_stage1_train_87p7_pdms.ckpt'),
    worldmodel_checkpoint_path=str(root / 'ckpts/worldtraj_stage1_512_tadwm.pkl'),
    pretrained_vae_path=str(root / 'ckpts/CogVideoX-2b'),
    teacher_scheduler_path=str(root / 'ckpts/CogVideoX-2b'),
    vocab_path=str(root / 'ckpts/trajectory_anchors_256.npy'),
    metric_cache_path='/mnt/cfs/qiuzijun/DrivoR/exp/train_metric_cache_ego_area_v2',
    sim_reward_path=None,
)
agent.initialize(from_world_model=True)
teacher = torch.load(agent._worldmodel_checkpoint_path, map_location='cpu')['model_state_dict']
mapped = {}
for key, value in teacher.items():
    key = key.removeprefix('module.')
    if key.startswith('traj_encoder.'):
        mapped[key.replace('traj_encoder.', 'trajencoder_wm.', 1)] = value
    elif key.startswith('adapters.'):
        mapped[key.replace('adapters.', 'adapters_wm.', 1)] = value
    elif key.startswith('transformer.'):
        mapped['trajworldmodel.' + key] = value
state = agent.state_dict()
expected = [key for key in state if key.startswith(('trajencoder_wm.', 'adapters_wm.', 'trajworldmodel.'))]
assert expected
for key in expected:
    if key == 'trajencoder_wm.traj_vocab':
        # The published teacher keeps its frozen anchor vocabulary in a
        # separate .npy asset, not in model_state_dict.
        assert torch.equal(state[key], torch.as_tensor(np.load(agent.vocab_path), dtype=state[key].dtype))
        continue
    assert key in mapped, f'Missing teacher tensor: {key}'
    assert torch.equal(state[key], mapped[key]), f'Teacher tensor differs: {key}'
print(f'Validated {len(expected)-1} teacher tensors and external anchor vocabulary', flush=True)
(root / 'outputs/register_readiness/teacher_weights.json').write_text(json.dumps({'teacher_tensors': len(expected)-1, 'external_vocabulary_matched': True, 'all_matched': True}, indent=2))
del teacher, mapped, state
if args.weights_only:
    raise SystemExit(0)
while torch.cuda.mem_get_info()[0] < 24 * 2**30:
    print('Waiting for 24 GiB free GPU memory for teacher forward/backward check', flush=True)
    time.sleep(30)
sample = next((root / 'exp/worlddrive_train_cache').glob('*/*/trajworld_feature.gz'))
features = {key: value.unsqueeze(0).cuda() for key, value in load_feature_target_from_pickle(sample).items()}
targets = {key: value.unsqueeze(0).cuda() for key, value in load_feature_target_from_pickle(sample.parent / 'trajectory_target.gz').items()}
agent = agent.cuda().train()
with torch.autocast('cuda', dtype=torch.bfloat16):
    losses = agent(features, targets, [sample.parent.name])
    loss = losses['loss']
assert torch.isfinite(loss)
loss.backward()
assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in agent.future_registers.parameters())
assert all(p.grad is None for name, p in agent.named_parameters() if not name.startswith('future_registers.'))
result = dict(teacher_tensors=len(expected), token=sample.parent.name, candidates=agent.register_candidates,
              losses={key: float(value.detach()) for key, value in losses.items()}, backward_passed=True)
(root / 'outputs/register_readiness/teacher_smoke.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
