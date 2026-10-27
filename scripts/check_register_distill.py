"""Real single-GPU Lightning smoke for the MSE-only adaptation stage."""
import json
import os
from pathlib import Path
from unittest.mock import patch
import torch
import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from navsim.planning.training.dataset import load_feature_target_from_pickle
from navsim.planning.training.agent_lightning_module import AgentLightningModule

root = Path(__file__).resolve().parents[1]
os.environ.setdefault('NAVSIM_EXP_ROOT', str(root / 'exp'))
os.environ.setdefault('OPENSCENE_DATA_ROOT', '/mnt/cfs/datasets/navsim-v1')
os.environ.setdefault('NAVSIM_DEVKIT_ROOT', str(root))
os.environ.setdefault('NUPLAN_MAPS_ROOT', '/mnt/cfs/datasets/navsim-v1/maps')
os.environ.setdefault('FEATURE_CACHE', str(root / 'exp/worlddrive_train_cache'))
torch.set_num_threads(2)
torch.set_float32_matmul_precision('high')
pl.seed_everything(7, workers=True)
with initialize_config_dir(config_dir=str(root / 'navsim/planning/script/config/training'), version_base=None):
    cfg = compose(config_name='register_distill_4gpu', overrides=[
        f'hydra.searchpath=[file://{root}/navsim/planning/script/config/common]'])

def forbidden(*args, **kwargs):
    raise AssertionError('PDM or scorer head was called in pure distillation')

with patch('navsim.agents.worlddrive.worlddrive_agent.PDMSimulator', forbidden), \
     patch('navsim.agents.worlddrive.worlddrive_agent.MetricCacheLoader', forbidden), \
     patch('navsim.agents.worlddrive.future_register_agent.score_training_candidate', forbidden), \
     patch('navsim.agents.worlddrive.future_register_agent.load_training_metric_cache', forbidden):
    agent = instantiate(cfg.agent)
    agent.initialize()
    assert not hasattr(agent, 'simulator') and not hasattr(agent, 'metric_cache_loader')
    trainable = [name for name, p in agent.named_parameters() if p.requires_grad]
    assert trainable and all(name.startswith('student_far.') for name in trainable)
    for head in [*agent.teacher_registers.window_heads, *agent.teacher_registers.global_heads]:
        head.forward = forbidden
    shapes = {}
    def observe(name):
        def hook(module, inputs, output):
            shapes[name] = {'motion': list(inputs[1].shape),
                            'registers': list((output['tokens'] if isinstance(output, dict) else output).shape)}
        return hook
    agent.teacher_registers.register_forward_hook(observe('teacher'))
    agent.student_far.register_forward_hook(observe('student'))
    decoder_checks = []
    current_student_inputs = {}
    def capture_student_inputs(module, inputs):
        current_student_inputs['visual'] = inputs[0].detach()
        current_student_inputs['motion'] = inputs[1].detach()
    def check_decoder_memory(module, inputs):
        visual = current_student_inputs['visual']
        motion = current_student_inputs['motion']
        memory = inputs[1]
        assert memory.shape[1] == visual.shape[1] + motion.shape[1]
        torch.testing.assert_close(memory[:, :visual.shape[1]], visual.to(memory))
        torch.testing.assert_close(memory[:, visual.shape[1]:], motion.to(memory))
        decoder_checks.append(True)
    agent.student_far.register_forward_pre_hook(capture_student_inputs)
    agent.student_far.decoder.register_forward_pre_hook(check_decoder_memory)
    samples = []
    for path in Path(os.environ['FEATURE_CACHE']).glob('*/*/trajworld_feature.gz'):
        samples.append((load_feature_target_from_pickle(path), {}, path.parent.name))
        if len(samples) == 2:
            break
    assert len(samples) == 2
    loader = DataLoader(samples, batch_size=2)
    before = agent.student_far.query.detach().clone()
    class Verify(pl.Callback):
        def on_after_backward(self, trainer, module):
            assert all(p.grad is None for name, p in module.agent.named_parameters() if not name.startswith('student_far.'))
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.agent.student_far.parameters())
            assert not module.agent.teacher_registers.training
            assert not module.agent.trajplanner.training
    trainer = pl.Trainer(accelerator='gpu', devices=1, precision='bf16-mixed',
                         max_epochs=1, limit_train_batches=1, limit_val_batches=1,
                         num_sanity_val_steps=0, logger=False, enable_checkpointing=False,
                         enable_model_summary=False, callbacks=[Verify()], gradient_clip_val=1.0)
    trainer.fit(AgentLightningModule(agent), loader, loader)
    assert not torch.equal(before, agent.student_far.query.detach().cpu())
    assert shapes['teacher'] == {'motion': [12, 5, 512], 'registers': [12, 36, 512]}
    assert shapes['student'] == {'motion': [12, 1, 512], 'registers': [12, 36, 512]}
    assert len(decoder_checks) >= 2  # train and validation both condition early
    metrics = {name: float(value) for name, value in trainer.callback_metrics.items()}
    assert not any('scorer' in name or 'cls_loss' in name or 'sim_loss' in name for name in metrics)
    result = {'passed': True, 'checkpoint': cfg.agent.teacher_register_checkpoint_path,
              'shapes': shapes, 'metrics': metrics, 'student_updated': True,
              'only_student_gradients': True, 'pdm_calls': 0, 'batch_size': 2,
              'decoder_reads_visual_and_motion': True,
              'peak_gpu_GiB': torch.cuda.max_memory_allocated() / 2**30}
    out = root / 'outputs/register_readiness/distill_single_motion_smoke.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
