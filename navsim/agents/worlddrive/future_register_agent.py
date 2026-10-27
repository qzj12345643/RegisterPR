"""Stage one of planning-register learning: frozen planner/DiT, PDM supervision."""

import numpy as np
import torch

from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from .worlddrive_agent import WorldDriveAgent
from .future_registers import FutureRegisterReadout
from .worlddrive_refiner import FutureRegisterFAR
from .training_metric_cache import TrainingPDMScorer, load_training_metric_cache, score_training_candidate
from .blocks.lr_scheduler import WarmupCosLR


class FutureRegisterAgent(WorldDriveAgent):
    def __init__(self, metric_cache_path, register_candidates=16, teacher_candidates=6,
                 teacher_chunk_size=None,
                 teacher_steps=1, register_layers=2, validation_seed=1234,
                 nc_loss_weight=1.0, dac_loss_weight=1.0, ttc_loss_weight=1.0,
                 comfort_loss_weight=1.0, ep_loss_weight=1.0,
                 ep_global_loss_weight=0.25, **kwargs):
        if register_candidates < 1 or teacher_candidates < 1 or teacher_steps < 1:
            raise ValueError('Candidate counts and teacher steps must be positive.')
        if teacher_candidates > register_candidates:
            raise ValueError('teacher_candidates cannot exceed register_candidates.')
        build_pdm_helpers = kwargs.get('build_pdm_helpers', True)
        if build_pdm_helpers and not metric_cache_path:
            raise ValueError('metric_cache_path is required for register supervision.')
        kwargs['training_mode'] = 'wm'
        kwargs['with_wm_proj'] = True
        kwargs['visualize'] = False
        super().__init__(metric_cache_path=metric_cache_path, **kwargs)
        if build_pdm_helpers:
            self.train_scorer = TrainingPDMScorer(self.simulator.proposal_sampling)
        del self.trajrefiner
        self.requires_grad_(False)
        self.future_registers = FutureRegisterReadout(
            layers=register_layers,
            loss_weights={
                'nc': nc_loss_weight, 'dac': dac_loss_weight,
                'ttc': ttc_loss_weight, 'comfort': comfort_loss_weight,
                'ep': ep_loss_weight, 'ep_global': ep_global_loss_weight,
            })
        self.register_candidates = register_candidates
        self.teacher_candidates = teacher_candidates
        # Kept for config compatibility; the aligned six-candidate path is
        # intentionally evaluated in one world-model call.
        self.teacher_chunk_size = teacher_candidates
        self.teacher_steps = teacher_steps
        self.validation_seed = validation_seed
        if register_candidates > self.traj_vocab.shape[0]:
            raise ValueError('register_candidates exceeds trajectory vocabulary size.')

    def _select_teacher_candidates(self, candidates, token_list, cache):
        """Select teacher candidates and retain PDM results computed for ranking.

        The winner/loser path has to score all planner candidates to rank them.
        Return the complete score records for the selected candidates so callers
        can reuse them when constructing the supervision labels instead of
        simulating the selected trajectories a second time.
        """
        if self.teacher_candidates != 6:
            return candidates[:, :self.teacher_candidates], None
        rng = np.random.default_rng(None if self.training else self.validation_seed)
        selected = []
        selected_records = []
        for scene_idx, token in enumerate(token_list):
            records = []
            for trajectory in candidates[scene_idx]:
                record = score_training_candidate(
                    cache[token], Trajectory(trajectory.float().cpu().numpy(), self._trajectory_sampling),
                    self.simulator, self.train_scorer)
                records.append(record)
            scores = [record[0].score for record in records]
            order = np.argsort(np.asarray(scores))
            winners = order[:3]
            loser_pool = order[-10:]
            losers = rng.choice(loser_pool, size=3, replace=False)
            selected_indices = np.concatenate((winners, losers))
            selected.append(torch.as_tensor(selected_indices, device=candidates.device))
            selected_records.extend(records[index] for index in selected_indices)
        selected = torch.stack(selected)
        batch = torch.arange(candidates.shape[0], device=candidates.device)[:, None]
        return candidates[batch, selected], selected_records

    def get_coslr_optimizers(self):
        # The readout is the only trainable component in this stage.
        optimizer = torch.optim.AdamW(
            self.future_registers.parameters(), lr=self._lr, weight_decay=1e-4,
            betas=(0.9, 0.95))
        return {'optimizer': optimizer, 'lr_scheduler': WarmupCosLR(
            optimizer=optimizer, lr=self._lr, min_lr=1e-6,
            epochs=10, warmup_epochs=0)}

    def compute_trajectory(self, *args, **kwargs):
        raise NotImplementedError('This agent trains the teacher readout; student deployment is a later stage.')

    def forward(self, features, targets=None, token_list=None, eval=False):
        if eval:
            raise ValueError('Use forward for readout training/validation, not trajectory deployment.')
        if token_list is None or targets is None:
            raise ValueError('Scene tokens and planner targets are required for online PDM labels.')
        frozen = (self.trajplanner, self.trajencoder, self.adapters,
                  self.trajworldmodel, self.trajencoder_wm, self.adapters_wm)
        for module in frozen:
            module.eval()  # Lightning's train() must not update frozen BN/dropout.
        with torch.no_grad():
            visual = self.adapters(features['visual_token'])
            b = visual.shape[0]
            embed = self.trajencoder(traj_train=True)[None].expand(b, -1, -1)
            scores, offsets, *_ = self.trajplanner(
                visual, features['status_feature'], embed, self.traj_vocab, targets)
            indices = scores.topk(self.register_candidates, dim=1).indices
            batch = torch.arange(b, device=indices.device)[:, None]
            candidates = self.trajencoder.traj_vocab[indices] + offsets[batch, indices].reshape(b, -1, 8, 3)
            history = self.adapters_wm(features['low_visual_token'])
            if history.shape[1] != 2:
                raise ValueError('Expected two history latent positions.')
        k = self.teacher_candidates
        trajectories = candidates.reshape(b*k, 8, 3)
        scene_indices = torch.arange(b, device=history.device).repeat_interleave(k)
        tokens = [token for token in token_list for _ in range(k)]
        if len(tokens) != b*k:
            raise ValueError('Scene token batch does not match features.')
        cache = {}
        for token in token_list:
            if token not in cache:
                cache[token] = load_training_metric_cache(self.metric_cache_loader.metric_cache_paths[token])
        candidates, selected_records = self._select_teacher_candidates(candidates, token_list, cache)
        trajectories = candidates.reshape(b * k, 8, 3)
        rng = None
        if not self.training:
            rng = torch.Generator(device=history.device).manual_seed(self.validation_seed)
        # Same scene noise across candidates reduces noise-induced ranking variation.
        noise = torch.randn(b, 5, 16, *history.shape[-2:], device=history.device,
                            dtype=history.dtype, generator=rng)
        losses = {}
        for start, end in ((0, b * k),):
            traj = trajectories[start:end]
            ids = scene_indices[start:end]
            with torch.no_grad():
                # Teacher must use its own pretrained trajectory encoder, not planner's.
                motion = self.trajencoder_wm(traj, traj_train=False)
                teacher = self.trajworldmodel.step_eval(
                    history[ids], noise[ids].clone(), motion, no_grad=True,
                    guidance_scale=1, timesteps=self.teacher_steps)
                futures = teacher[:, 2:]
            window_labels, global_labels, scalar_ep_labels = [], [], []
            records = selected_records[start:end] if selected_records is not None else [None] * (end - start)
            for trajectory, token, record in zip(traj, tokens[start:end], records):
                if record is None:
                    record = score_training_candidate(
                        cache[token],
                        Trajectory(trajectory.float().cpu().numpy(), self._trajectory_sampling),
                        self.simulator, self.train_scorer)
                result, window_scores, ep_scores = record
                window_labels.append(window_scores)
                scalar_ep_labels.append(result.ego_progress)
                global_labels.append([result.time_to_collision_within_bound, result.comfort,
                                      *ep_scores[0]])
            labels = {'window_scores': torch.as_tensor(np.stack(window_labels), device=traj.device),
                      'global_scores': torch.tensor(global_labels, device=traj.device),
                      'global_ep': torch.tensor(scalar_ep_labels, device=traj.device)}
            prediction = self.future_registers(
                futures.detach().to(dtype=visual.dtype), motion,
                features['status_feature'][ids].to(device=motion.device, dtype=motion.dtype))
            chunk_losses = self.future_registers.loss(prediction, labels)
            weight = (end-start)/(b*k)
            for key, value in chunk_losses.items():
                losses[key] = losses.get(key, 0) + weight * value
        return losses


class TeacherRegisterEvalAgent(FutureRegisterAgent):
    """Inference-only teacher-register scorer for NAVSIM evaluation.

    The training agent uses PDM winner/loser sampling to construct labels.  That
    path must not be used at evaluation time because it would read the ground
    truth metric cache before selecting a trajectory.  This class instead scores
    the planner's complete candidate pool with the frozen DiT and teacher
    readout, then returns the candidate with the highest DrivoR-style aggregate
    score.  NAVSIM subsequently computes the official PDM score for the returned
    trajectory.
    """

    def __init__(self, teacher_register_checkpoint_path, **kwargs):
        # The parent needs a metric-cache path to construct its simulator/cache
        # helpers, although evaluation never reads the cache for selection.
        kwargs['teacher_candidates'] = kwargs.get('register_candidates', 16)
        super().__init__(**kwargs)
        if not teacher_register_checkpoint_path:
            raise ValueError('teacher_register_checkpoint_path is required.')
        self.teacher_register_checkpoint_path = teacher_register_checkpoint_path

    def initialize(self, from_world_model=False) -> None:
        # Joint checkpoints contain the complete frozen planner and teacher,
        # including the teacher's distinct motion encoder and visual adapter.
        # Load them together so evaluation exactly matches the training run.
        checkpoint = torch.load(self.teacher_register_checkpoint_path, map_location='cpu')
        state = checkpoint.get('state_dict', checkpoint)
        state = {
            key.removeprefix('agent.'): value
            for key, value in state.items()
        }
        required = self.state_dict().keys()
        self.load_state_dict({key: state[key] for key in required}, strict=True)
        self.traj_vocab = self.trajencoder.traj_vocab.detach().cpu()
        self.requires_grad_(False)
        self.eval()

    def get_sensor_config(self):
        sensors = SensorConfig.build_no_sensors()
        sensors.cam_f0 = [0, 1, 2, 3]
        return sensors

    @staticmethod
    def _aggregate_teacher_score(prediction):
        """Aggregate temporal heads into a DrivoR-style NAVSIM-v1 score."""
        window_scores = prediction['window_scores'].float()  # [N, 3, NC/DAC]
        global_scores = prediction['global_scores'].float()  # [N, TTC, C, EP_1, EP_2, EP_3]
        nc = window_scores[..., 0].amin(dim=1)
        dac = window_scores[..., 1].amin(dim=1)
        ttc = global_scores[..., 0]
        comfort = global_scores[..., 1]
        ep = global_scores[..., 2:].sum(dim=-1).clamp(max=1.0)
        eps = torch.finfo(global_scores.dtype).eps
        score = (
            torch.log(nc.clamp_min(eps))
            + torch.log(dac.clamp_min(eps))
            + torch.log((5.0 * ttc + 5.0 * ep + 2.0 * comfort).clamp_min(eps))
        )
        return score, {'nc': nc, 'dac': dac, 'ttc': ttc,
                       'comfort': comfort, 'ep': ep}

    def forward(self, features, targets=None, token_list=None, eval=False):
        if self.training or not eval:
            raise ValueError('TeacherRegisterEvalAgent is inference-only; call forward(..., eval=True).')

        # Dataset-backed evaluation supplies unbatched feature tensors.  This
        # mirrors WorldDriveAgent's evaluation branch and avoids target/cache
        # access when ranking candidates.
        with torch.no_grad():
            visual_latents = self.adapters(features['visual_token'][None])
            status = features['status_feature'][None]
            batch_size = visual_latents.shape[0]
            planner_embed = self.trajencoder(traj_train=True)[None].expand(batch_size, -1, -1)
            planner_out = self.trajplanner(
                visual_latents, status, planner_embed, self.traj_vocab,
                None, eval_mode=True,
            )
            _, planner_scores, offsets, _ = planner_out
            indices = planner_scores.topk(self.register_candidates, dim=1).indices
            batch = torch.arange(batch_size, device=indices.device)[:, None]
            candidates = self.trajencoder.traj_vocab[indices]
            candidates = candidates + offsets[batch, indices].reshape(
                batch_size, self.register_candidates, 8, 3
            )

            history = self.adapters_wm(features['low_visual_token'][None])
            if history.shape[1] != 2:
                raise ValueError(f'Expected two history latent positions, got {history.shape[1]}.')
            candidate_count = candidates.shape[1]
            trajectories = candidates.reshape(batch_size * candidate_count, 8, 3)
            scene_indices = torch.arange(batch_size, device=history.device).repeat_interleave(candidate_count)
            motion = self.trajencoder_wm(trajectories, traj_train=False)
            rng = torch.Generator(device=history.device).manual_seed(self.validation_seed)
            noise = torch.randn(
                batch_size, 5, 16, *history.shape[-2:], device=history.device,
                dtype=history.dtype, generator=rng,
            )
            teacher_output = self.trajworldmodel.step_eval(
                history[scene_indices], noise[scene_indices].clone(), motion,
                no_grad=True, guidance_scale=1, timesteps=self.teacher_steps,
            )
            prediction = self.future_registers(
                teacher_output[:, 2:].to(dtype=visual_latents.dtype), motion,
                status[scene_indices].to(device=motion.device, dtype=motion.dtype),
            )
            scores, metric_scores = self._aggregate_teacher_score(prediction)
            scores = scores.reshape(batch_size, candidate_count)
            selected = scores.argmax(dim=1)
            selected_batch = torch.arange(batch_size, device=scores.device)
            trajectory = candidates[selected_batch, selected]

        return {
            'pred_traj': trajectory,
            'teacher_score': scores,
            'teacher_metric_scores': {
                key: value.reshape(batch_size, candidate_count)
                for key, value in metric_scores.items()
            },
            'candidate_trajs': candidates,
        }

    def compute_trajectory(self, features, targets=None, tokens=None):
        # Keep compatibility with the repository's dataset-backed PDM evaluator.
        # AgentInput support is included for submission-style callers.
        if isinstance(features, AgentInput):
            built_features = {}
            for builder in self.get_feature_builders():
                built_features.update(builder.compute_features(features))
            features = built_features
            targets = None
            tokens = None
        self.eval()
        device = next(self.parameters()).device
        features = {key: value.to(device=device, dtype=torch.float32) for key, value in features.items()}
        with torch.no_grad():
            predictions = self.forward(features, targets, tokens, eval=True)
        return Trajectory(predictions['pred_traj'].float().cpu().squeeze(0).numpy(), self._trajectory_sampling)


class RegisterDistillAgent(FutureRegisterAgent):
    """Adapt a warm-started single-motion-token student using only register MSE.

    The frozen teacher retains its five retrieved motion tokens. Candidate
    sampling uses only the frozen planner's predicted ranks, never PDM labels.
    """
    def __init__(self, teacher_register_checkpoint_path,
                 student_init_checkpoint_path=None, distill_epochs=10,
                 distill_loss_weight=1.0, **kwargs):
        if not teacher_register_checkpoint_path:
            raise ValueError('teacher_register_checkpoint_path is required.')
        if distill_epochs < 1 or distill_loss_weight <= 0:
            raise ValueError('Distillation epochs and MSE weight must be positive.')
        kwargs['build_pdm_helpers'] = False
        kwargs['metric_cache_path'] = None
        super().__init__(**kwargs)
        if self.register_candidates != 16 or self.teacher_candidates != 6:
            raise ValueError('Pure distillation requires a Top-16 pool and 3 + 3 candidates.')
        self.teacher_register_checkpoint_path = teacher_register_checkpoint_path
        self.student_init_checkpoint_path = student_init_checkpoint_path or teacher_register_checkpoint_path
        self.distill_epochs = distill_epochs
        self.distill_loss_weight = distill_loss_weight
        self.teacher_registers = self.future_registers
        del self.future_registers
        self.student_far = FutureRegisterFAR()
        self.requires_grad_(False)
        self.student_far.requires_grad_(True)
        self.train()

    @staticmethod
    def _checkpoint_state(path):
        checkpoint = torch.load(path, map_location='cpu')
        return {key.removeprefix('agent.'): value
                for key, value in checkpoint.get('state_dict', checkpoint).items()}

    def initialize(self, from_world_model=True):
        # Load the full joint snapshot, including its distinct teacher adapters
        # and motion encoder. Never reinitialize these from the stage-one files.
        state = self._checkpoint_state(self.teacher_register_checkpoint_path)
        expected = self.state_dict()
        mapped = {}
        for key in expected:
            if key.startswith('student_far.'):
                continue
            source = key.replace('teacher_registers.', 'future_registers.', 1)
            if key not in state and source not in state:
                raise RuntimeError(f'Teacher checkpoint is missing {key}')
            mapped[key] = state[key] if key in state else state[source]
        student_state = (state if self.student_init_checkpoint_path == self.teacher_register_checkpoint_path
                         else self._checkpoint_state(self.student_init_checkpoint_path))
        for key in expected:
            if key.startswith('student_far.'):
                if key not in student_state:
                    raise RuntimeError(f'Student checkpoint is missing {key}; warm-start is required.')
                mapped[key] = student_state[key]
        self.load_state_dict(mapped, strict=True)
        self.traj_vocab = self.trajencoder.traj_vocab.detach().cpu()
        if not torch.equal(self.trajencoder.traj_vocab, self.trajencoder_wm.traj_vocab):
            raise ValueError('Planner and teacher vocabularies must share anchor indices.')
        self.train(self.training)

    def train(self, mode=True):
        # Lightning recursively switches modules to train mode. Keep the entire
        # target-generation path deterministic (frozen BN and dropout) anyway.
        super().train(False)
        self.training = mode
        self.student_far.train(mode)
        return self

    def get_coslr_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.student_far.parameters(), lr=self._lr, weight_decay=1e-4,
            betas=(0.9, 0.95))
        return {'optimizer': optimizer, 'lr_scheduler': WarmupCosLR(
            optimizer=optimizer, lr=self._lr, min_lr=1e-6,
            epochs=self.distill_epochs, warmup_epochs=0)}

    def _sample_candidate_indices(self, scores):
        pool = scores.topk(self.register_candidates, dim=1, sorted=True).indices
        generator = None if self.training else torch.Generator(device=scores.device).manual_seed(self.validation_seed)
        tail = torch.stack([
            torch.randperm(13, device=scores.device, generator=generator)[:3] + 3
            for _ in range(scores.shape[0])])
        return torch.cat((pool[:, :3], pool.gather(1, tail)), dim=1)

    def _student_motion(self, indices, selected_offsets):
        # Match original FAR inference exactly: the generating anchor and its
        # planner-predicted offset. This is NOT one of five retrieved anchors.
        anchors = self.trajencoder_wm.traj_vocab[indices]
        normalized = self.trajencoder_wm.norm_odo(anchors)
        n = indices.numel()
        return (self.trajencoder_wm.traj_vocab_encoder(normalized.reshape(n, -1))
                + self.trajencoder_wm.traj_offset_encoder(selected_offsets.reshape(n, -1)))[:, None]

    def forward(self, features, targets=None, token_list=None, eval=False):
        if eval:
            raise ValueError('Pure distillation produces register tokens, not deployment scores.')
        with torch.no_grad():
            visual = self.adapters(features['visual_token'])
            b = visual.shape[0]
            embed = self.trajencoder(traj_train=True)[None].expand(b, -1, -1)
            scores, offsets, (planner_visual, _) = self.trajplanner(
                visual, features['status_feature'], embed, self.traj_vocab,
                eval_mode=True, return_candidates_only=True)
            indices = self._sample_candidate_indices(scores)
            batch = torch.arange(b, device=indices.device)[:, None]
            selected_offsets = offsets[batch, indices]
            k = self.teacher_candidates
            trajectories = (self.trajencoder.traj_vocab[indices]
                            + selected_offsets.reshape(b, k, 8, 3)).reshape(b * k, 8, 3)
            student_motion = self._student_motion(indices, selected_offsets)
            teacher_motion = self.trajencoder_wm(trajectories, traj_train=False)
            if teacher_motion.shape[1] != 5:
                raise ValueError('Teacher must retain its pretrained five-token motion representation.')
            history = self.adapters_wm(features['low_visual_token'])
            if history.shape[1] != 2:
                raise ValueError('Expected two history latent positions.')
            ids = torch.arange(b, device=history.device).repeat_interleave(k)
            rng = None if self.training else torch.Generator(device=history.device).manual_seed(self.validation_seed)
            noise = torch.randn(b, 5, 16, *history.shape[-2:], device=history.device,
                                dtype=history.dtype, generator=rng)
            # One call for all six candidates; share noise within each scene.
            teacher_output = self.trajworldmodel.step_eval(
                history[ids], noise[ids].clone(), teacher_motion, no_grad=True,
                guidance_scale=1, timesteps=self.teacher_steps)
            status = features['status_feature'][ids].to(teacher_motion)
            teacher_tokens = self.teacher_registers(
                teacher_output[:, 2:].detach(), teacher_motion, status,
                return_tokens_only=True)['tokens']
        student_tokens = self.student_far(planner_visual[ids], student_motion, status)
        loss = torch.nn.functional.mse_loss(student_tokens.float(), teacher_tokens.float())
        return {'register_distill_loss': loss, 'loss': self.distill_loss_weight * loss}


class JointFutureRegisterAgent(FutureRegisterAgent):
    """Joint first stage: train teacher registers and student FAR together.

    Teacher registers receive PDM supervision.  Student FAR receives only MSE
    against detached teacher registers in the same step.
    """
    def __init__(self, distill_loss_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        if distill_loss_weight < 0:
            raise ValueError('distill_loss_weight must be non-negative.')
        self.distill_loss_weight = distill_loss_weight
        self.student_far = FutureRegisterFAR()
        self.requires_grad_(False)
        for module in (self.future_registers, self.student_far):
            module.train()
            for parameter in module.parameters():
                parameter.requires_grad = True

    def get_coslr_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.future_registers.parameters()) + list(self.student_far.parameters()),
            lr=self._lr, weight_decay=1e-4, betas=(0.9, 0.95))
        return {'optimizer': optimizer, 'lr_scheduler': WarmupCosLR(
            optimizer=optimizer, lr=self._lr, min_lr=1e-6,
            epochs=10, warmup_epochs=0)}

    def forward(self, features, targets=None, token_list=None, eval=False):
        if token_list is None or targets is None:
            raise ValueError('Scene tokens and planner targets are required.')
        frozen = (self.trajplanner, self.trajencoder, self.adapters,
                  self.trajworldmodel, self.trajencoder_wm, self.adapters_wm)
        for module in frozen:
            module.eval()
        with torch.no_grad():
            visual_latents = self.adapters(features['visual_token'])
            b = visual_latents.shape[0]
            planner_embed = self.trajencoder(traj_train=True)[None].expand(b, -1, -1)
            scores, offsets, _, _, _, planner_outputs, _ = self.trajplanner(
                visual_latents, features['status_feature'], planner_embed,
                self.traj_vocab, targets)
            indices = scores.topk(self.register_candidates, dim=1).indices
            batch = torch.arange(b, device=indices.device)[:, None]
            candidates = self.trajencoder.traj_vocab[indices] + offsets[batch, indices].reshape(b, -1, 8, 3)
            history = self.adapters_wm(features['low_visual_token'])
            planner_visual, _ = planner_outputs
            if history.shape[1] != 2:
                raise ValueError('Expected two history latent positions.')

        k = self.teacher_candidates
        tokens = [token for token in token_list for _ in range(k)]
        cache = {token: load_training_metric_cache(self.metric_cache_loader.metric_cache_paths[token])
                 for token in token_list}
        candidates, selected_records = self._select_teacher_candidates(candidates, token_list, cache)
        trajectories = candidates.reshape(b * k, 8, 3)
        scene_indices = torch.arange(b, device=history.device).repeat_interleave(k)
        rng = None if self.training else torch.Generator(device=history.device).manual_seed(self.validation_seed)
        noise = torch.randn(b, 5, 16, *history.shape[-2:], device=history.device,
                            dtype=history.dtype, generator=rng)
        teacher_losses, student_losses, teacher_component_losses = [], [], {}
        motion = self.trajencoder_wm(trajectories, traj_train=False)
        for start, end in ((0, b * k),):
            traj = trajectories[start:end]
            ids = scene_indices[start:end]
            with torch.no_grad():
                teacher_output = self.trajworldmodel.step_eval(
                    history[ids], noise[ids].clone(), motion[start:end], no_grad=True,
                    guidance_scale=1, timesteps=self.teacher_steps)
            window_labels, global_labels, scalar_ep_labels = [], [], []
            records = selected_records[start:end] if selected_records is not None else [None] * (end - start)
            for trajectory, token, record in zip(traj, tokens[start:end], records):
                if record is None:
                    record = score_training_candidate(
                        cache[token], Trajectory(trajectory.float().cpu().numpy(), self._trajectory_sampling),
                        self.simulator, self.train_scorer)
                result, window_scores, ep_scores = record
                window_labels.append(window_scores)
                global_labels.append([result.time_to_collision_within_bound, result.comfort,
                                      *ep_scores[0]])
                scalar_ep_labels.append(result.ego_progress)
            labels = {
                'window_scores': torch.as_tensor(np.stack(window_labels), device=traj.device),
                'global_scores': torch.tensor(global_labels, device=traj.device),
                'global_ep': torch.tensor(scalar_ep_labels, device=traj.device),
            }
            teacher_prediction = self.future_registers(
                teacher_output[:, 2:].detach().to(dtype=visual_latents.dtype), motion[start:end],
                features['status_feature'][ids].to(device=motion.device, dtype=motion.dtype))
            teacher_loss_dict = self.future_registers.loss(teacher_prediction, labels)
            teacher_loss = teacher_loss_dict['loss']
            student_prediction = self.student_far(
                planner_visual[ids], motion[start:end],
                features['status_feature'][ids].to(device=motion.device, dtype=motion.dtype))
            student_loss = torch.nn.functional.mse_loss(
                student_prediction, teacher_prediction['tokens'].detach())
            teacher_losses.append(teacher_loss)
            student_losses.append(student_loss)
            for key, value in teacher_loss_dict.items():
                if key != 'loss':
                    teacher_component_losses.setdefault(key, []).append(value)
        teacher_loss = torch.stack(teacher_losses).mean()
        distill_loss = torch.stack(student_losses).mean()
        result = {
            'teacher_register_loss': teacher_loss,
            'register_distill_loss': distill_loss,
            'loss': teacher_loss + self.distill_loss_weight * distill_loss,
        }
        result.update({f'teacher_{key}': torch.stack(values).mean()
                       for key, values in teacher_component_losses.items()})
        return result
