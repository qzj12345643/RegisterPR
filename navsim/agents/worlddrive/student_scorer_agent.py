"""Post-distillation student scoring, with no DiT or teacher forward."""
import copy
import math
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from .worlddrive_agent import WorldDriveAgent
from .worlddrive_refiner import FutureRegisterFAR
from .future_register_agent import RegisterDistillAgent
from .student_register_scorer import StudentRegisterScorer
from .training_metric_cache import TrainingPDMScorer, load_training_metric_cache, score_student_candidate


class ScorerTrainingPhase(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.agent.set_training_epoch(trainer.current_epoch)
        pl_module.log('far_learning_rate', trainer.optimizers[0].param_groups[1]['lr'], sync_dist=True)
        pl_module.log('far_finetuning', float(pl_module.agent.finetune_far), sync_dist=True)

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.agent.set_training_epoch(trainer.current_epoch)


class StudentRegisterScorerAgent(WorldDriveAgent):
    def __init__(self, student_checkpoint_path, metric_cache_path=None,
                 scorer_checkpoint_path=None, register_candidates=16, score_candidates=6,
                 validation_seed=1234, scorer_warmup_epochs=1, training_epochs=10,
                 far_lr_scale=0.1, segment_loss_weight=1.0, global_loss_weight=1.0,
                 aggregate_loss_weight=1.0, pdm_loss_weight=1.0, **kwargs):
        if not student_checkpoint_path:
            raise ValueError('student_checkpoint_path is required.')
        if register_candidates != 16 or score_candidates != 6:
            raise ValueError('Training uses a Top-16 pool with Top-3 plus random-3.')
        if not 0 <= scorer_warmup_epochs < training_epochs or far_lr_scale <= 0:
            raise ValueError('Require 0 <= warmup < epochs and positive FAR learning-rate scale.')
        # Construct only the base planner; do not allocate a DiT to delete later.
        kwargs.update(training_mode='traj', with_wm_proj=True, visualize=False, build_pdm_helpers=False)
        super().__init__(metric_cache_path=metric_cache_path, **kwargs)
        self.trajencoder_wm = copy.deepcopy(self.trajencoder)
        self.requires_grad_(False)
        self.student_far = FutureRegisterFAR()
        self.student_scorer = StudentRegisterScorer(
            segment_loss_weight=segment_loss_weight, global_loss_weight=global_loss_weight,
            aggregate_loss_weight=aggregate_loss_weight, pdm_loss_weight=pdm_loss_weight)
        self.student_checkpoint_path = student_checkpoint_path
        self.scorer_checkpoint_path = scorer_checkpoint_path
        self.metric_cache_path = metric_cache_path
        self.register_candidates, self.score_candidates = register_candidates, score_candidates
        self.validation_seed = validation_seed
        self.scorer_warmup_epochs, self.training_epochs = scorer_warmup_epochs, training_epochs
        self.far_lr_scale = far_lr_scale
        self.set_training_epoch(0)
        self.train()

    _checkpoint_state = staticmethod(RegisterDistillAgent._checkpoint_state)
    _sample_candidate_indices = RegisterDistillAgent._sample_candidate_indices
    _student_motion = RegisterDistillAgent._student_motion

    def initialize(self, from_world_model=True):
        state = self._checkpoint_state(self.student_checkpoint_path)
        required = self.state_dict()
        mapped = {key: state[key] for key in required if not key.startswith('student_scorer.')}
        if self.scorer_checkpoint_path:
            head_state = self._checkpoint_state(self.scorer_checkpoint_path)
        else:
            head_state = state
        for key in required:
            if not key.startswith('student_scorer.'):
                continue
            suffix = key.removeprefix('student_scorer.')
            source = next((prefix + suffix for prefix in ('student_scorer.', 'teacher_registers.', 'future_registers.')
                           if prefix + suffix in head_state), None)
            if source is None:
                raise RuntimeError(f'Checkpoint missing trained scoring head: {key}')
            mapped[key] = head_state[source]
        self.load_state_dict(mapped, strict=True)
        self.traj_vocab = self.trajencoder.traj_vocab.detach().cpu()
        if not torch.equal(self.trajencoder.traj_vocab, self.trajencoder_wm.traj_vocab):
            raise ValueError('Planner and motion encoder anchor indices must agree.')
        self.train(self.training)

    def set_training_epoch(self, epoch):
        self.finetune_far = int(epoch) >= self.scorer_warmup_epochs
        # Keep FAR registered with DDP from construction. During warmup it runs
        # under no_grad and has lr=0: no gradients, optimizer state or updates.
        self.student_far.train(self.training and self.finetune_far)

    def train(self, mode=True):
        super().train(False)
        self.training = mode
        self.student_scorer.train(mode)
        self.student_far.train(mode and self.finetune_far)
        return self

    def get_training_callbacks(self):
        return [ScorerTrainingPhase()]

    def get_coslr_optimizers(self):
        optimizer = torch.optim.AdamW([
            {'params': self.student_scorer.parameters(), 'lr': self._lr},
            {'params': self.student_far.parameters(), 'lr': self._lr * self.far_lr_scale},
        ], weight_decay=1e-4, betas=(0.9, 0.95))
        def decay(epoch):
            e = min(epoch, self.training_epochs)
            return 0.02 + 0.98 * (1 + math.cos(math.pi * e / self.training_epochs)) / 2
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [
            decay, lambda epoch: 0.0 if epoch < self.scorer_warmup_epochs else decay(epoch)])
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def _ensure_pdm_helpers(self):
        if not hasattr(self, 'metric_cache_loader'):
            if not self.metric_cache_path:
                raise ValueError('Training needs metric_cache_path; inference does not.')
            sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)
            self.simulator = PDMSimulator(sampling)
            self.train_scorer = TrainingPDMScorer(sampling)
            self.metric_cache_loader = MetricCacheLoader(Path(self.metric_cache_path))

    def forward(self, features, targets=None, token_list=None, eval=False):
        with torch.no_grad():
            visual = self.adapters(features['visual_token'])
            b = visual.shape[0]
            embed = self.trajencoder(traj_train=True)[None].expand(b, -1, -1)
            scores, offsets, (visual_tokens, _) = self.trajplanner(
                visual, features['status_feature'], embed, self.traj_vocab,
                eval_mode=True, return_candidates_only=True)
            indices = (scores.topk(self.register_candidates, dim=1).indices if eval
                       else self._sample_candidate_indices(scores))
            k = indices.shape[1]
            batch = torch.arange(b, device=indices.device)[:, None]
            selected_offsets = offsets[batch, indices]
            trajectories = self.trajencoder.traj_vocab[indices] + selected_offsets.reshape(b, k, 8, 3)
            motion = self._student_motion(indices, selected_offsets)
            ids = torch.arange(b, device=visual.device).repeat_interleave(k)
            status = features['status_feature'][ids].to(motion)
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.finetune_far and not eval):
            registers = self.student_far(visual_tokens[ids], motion, status, return_scorer_tokens=True)
        prediction = self.student_scorer(registers)
        if eval:
            pdm = prediction['pdm_score'].reshape(b, k)
            best = pdm.argmax(1)
            return dict(pred_traj=trajectories[torch.arange(b, device=pdm.device), best],
                        candidate_trajs=trajectories, student_score=pdm,
                        student_metric_scores=prediction['full_scores'].reshape(b, k, 5))
        if token_list is None or len(token_list) != b:
            raise ValueError('Training/validation requires scene tokens for simulator targets.')
        self._ensure_pdm_helpers()
        labels = []
        # Each selected candidate is simulated exactly once. Never rank the
        # candidate pool using ground-truth PDM labels.
        for row, token in zip(trajectories.detach().float().cpu().numpy(), token_list):
            cache = load_training_metric_cache(self.metric_cache_loader.metric_cache_paths[token])
            for trajectory in row:
                labels.append(score_student_candidate(cache, Trajectory(trajectory, self._trajectory_sampling),
                                                      self.simulator, self.train_scorer))
        labels = {key: torch.as_tensor(np.stack([row[key] for row in labels]), device=registers.device,
                                     dtype=torch.float32) for key in labels[0]}
        return self.student_scorer.loss(prediction, labels)

    def get_sensor_config(self):
        sensors = SensorConfig.build_no_sensors()
        sensors.cam_f0 = [0, 1, 2, 3]
        return sensors

    def compute_trajectory(self, features, targets=None, tokens=None):
        if isinstance(features, AgentInput):
            raw = features
            features = {}
            for builder in self.get_feature_builders():
                features.update(builder.compute_features(raw))
        self.eval()
        device = next(self.parameters()).device
        features = {key: value.to(device=device, dtype=torch.float32)[None] for key, value in features.items()}
        with torch.no_grad():
            prediction = self.forward(features, eval=True)
        return Trajectory(prediction['pred_traj'][0].float().cpu().numpy(), self._trajectory_sampling)
