"""Real GPU smoke: scorer warmup, FAR unfreeze, all PDM losses and inference."""
import os
import json
from pathlib import Path
from unittest.mock import patch
import torch
import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from navsim.planning.training.dataset import load_feature_target_from_pickle
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.agents.worlddrive.training_metric_cache import score_student_candidate

root=Path(__file__).resolve().parents[1]
for k,v in dict(NAVSIM_EXP_ROOT=root/'exp',OPENSCENE_DATA_ROOT='/mnt/cfs/datasets/navsim-v1',
                NAVSIM_DEVKIT_ROOT=root,NUPLAN_MAPS_ROOT='/mnt/cfs/datasets/navsim-v1/maps').items():
    os.environ.setdefault(k,str(v))
torch.set_num_threads(2)
devices=int(os.environ.get('SCORER_SMOKE_DEVICES','1'))
pl.seed_everything(7,workers=True)
with initialize_config_dir(config_dir=str(root/'navsim/planning/script/config/training'),version_base=None):
    cfg=compose(config_name='student_scorer_4gpu',overrides=[f'hydra.searchpath=[file://{root}/navsim/planning/script/config/common]'])
def forbidden(*args,**kwargs):raise AssertionError('DiT must not be constructed')
with patch('navsim.agents.worlddrive.worlddrive_agent.TrajWorldModel',forbidden):
    agent=instantiate(cfg.agent)
agent.initialize()
assert not hasattr(agent,'trajworldmodel')
print('Loaded distill epoch09 and embedded teacher epoch05 heads',flush=True)
samples=[]
for p in Path(cfg.cache_path).glob('*/*/trajworld_feature.gz'):
    samples.append((load_feature_target_from_pickle(p),{},p.parent.name))
    if len(samples)==2:break
loader=DataLoader(samples,batch_size=2)
initial=agent.student_far.query.detach().clone()
initial_head=agent.student_scorer.window_heads[0][0].weight.detach().clone()
observed=[]
class Verify(pl.Callback):
    def on_after_backward(self,trainer,module):
        a=module.agent
        assert all(p.grad is None for n,p in a.named_parameters() if not n.startswith(('student_far.','student_scorer.')))
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.student_scorer.parameters())
        if trainer.current_epoch==0:assert all(p.grad is None for p in a.student_far.parameters())
        else:assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.student_far.parameters())
        observed.append(trainer.current_epoch)
    def on_train_epoch_end(self,trainer,module):
        if trainer.current_epoch==0:
            torch.testing.assert_close(initial,module.agent.student_far.query.detach().cpu(),rtol=0,atol=0)
            assert not torch.equal(initial_head,module.agent.student_scorer.window_heads[0][0].weight.detach().cpu())
trainer=pl.Trainer(accelerator='gpu',devices=devices,
                   strategy='ddp_find_unused_parameters_true' if devices>1 else 'auto',
                   precision='bf16-mixed',max_epochs=2,
                   limit_train_batches=1,limit_val_batches=1,num_sanity_val_steps=0,logger=False,
                   enable_checkpointing=False,enable_model_summary=False,gradient_clip_val=1.0,
                   callbacks=agent.get_training_callbacks()+[Verify()])
with patch('navsim.agents.worlddrive.student_scorer_agent.score_student_candidate',wraps=score_student_candidate) as score:
    trainer.fit(AgentLightningModule(agent),loader,loader)
    calls=score.call_count
assert observed==[0,1] and calls==48//devices
assert not torch.equal(initial,agent.student_far.query.detach().cpu())
# Inference must use all 16 candidates, without accessing GT or simulator.
agent.cuda().eval()
features={k:v[None].cuda() for k,v in samples[0][0].items()}
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16),patch.object(agent,'_ensure_pdm_helpers',side_effect=AssertionError('GT in inference')):
    pred=agent(features,eval=True)
assert pred['student_score'].shape==(1,16) and pred['pred_traj'].shape==(1,8,3)
if devices>1:
    # Ensure FAR updates are synchronized even though it was unused in epoch00.
    parameters=agent.student_far.query.detach().contiguous()
    copies=[torch.empty_like(parameters) for _ in range(devices)]
    torch.distributed.all_gather(copies,parameters)
    for other in copies:torch.testing.assert_close(parameters,other,rtol=0,atol=0)
smoke_checkpoint='/tmp/student_scorer_smoke_roundtrip.ckpt'
trainer.save_checkpoint(smoke_checkpoint)
trainer.strategy.barrier()
with patch('navsim.agents.worlddrive.worlddrive_agent.TrajWorldModel',forbidden):
    restored=instantiate(cfg.agent,student_checkpoint_path=smoke_checkpoint,scorer_checkpoint_path=None)
restored.initialize()
restored.cuda().eval()
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    restored_pred=restored(features,eval=True)
torch.testing.assert_close(pred['student_score'],restored_pred['student_score'],rtol=0,atol=0)
assert not hasattr(restored,'metric_cache_loader')
result=dict(passed=True,checkpoint=cfg.agent.student_checkpoint_path,pdm_calls=calls,
            dit_constructed=False,warmup_far_unchanged=True,finetune_far_updated=True,
            devices=devices,checkpoint_inference_roundtrip=True,
            inference_candidates=16,metrics={k:float(v) for k,v in trainer.callback_metrics.items()})
out=root/f'outputs/register_readiness/student_scorer_smoke_{devices}gpu.json'
if trainer.is_global_zero:
    out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
