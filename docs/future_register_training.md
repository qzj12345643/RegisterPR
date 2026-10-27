# Planning future registers

This stage learns a readout of **frozen DiT outputs**, not a student predicting
registers from observations. Only `future_registers.*` is optimized. Both planner
and teacher encoders remain frozen and in eval mode. No future images are needed;
current/history VAE features, scene tokens, planner targets and metric caches are
required. Simulator future information is used only to generate training labels.

## Network

For each candidate, teacher outputs have shape `[5,16,32,64]`. Remove the first two
history positions, without averaging the remaining three. Each future latent has
its own 12 learned 512-dimensional queries, plus a learned time embedding. Spatial
position features accompany projected latent pixels. A shared two-layer decoder
reads the corresponding latent independently for each group, then a two-layer
Transformer jointly processes the 36 registers, one projected ego-status token, and
the corresponding candidate's motion tokens. The frozen teacher trajectory encoder supplies `[N,L,512]` motion
tokens (currently L=5 anchor-neighbor encodings per candidate, not five candidates).
Motion tokens remain at 512 dimensions and are concatenated directly with
`[N,36,512]` registers. Concatenation gives `[N,37+L,512]`; after self-attention,
the first 37 positions are used by the scorer, while only the first 36 scene-register
positions are retained as the distillation target. No motion projection MLP is used.
Transformer FFNs use GELU. Heads pool within each register time group for NC/DAC and across all 37 tokens
for TTC/comfort and three temporally segmented EP targets. Each metric uses an independent
`512 -> 1024 -> 1` ReLU MLP
head, following DrivoR's per-metric scorer design. Motion inputs are detached.

`FutureRegisterReadout.forward()` returns `tokens: [N,36,512]`, window logits/scores
`[N,3,2]` (NC,DAC), and global logits/scores `[N,5]` (TTC,comfort,EP-0.5s,EP-2.5s,EP-4.0s). The **conditioned
output tokens**, not the shared query parameters alone, are future student targets.

## Supervision

Windows are `(0,0.5]`, `(0.5,2.5]`, `(2.5,4.0]` seconds. At 0.1 s simulator resolution,
these use indices 1–5, 6–25 and 26–40, respectively. Time zero is intentionally
excluded. This is a task definition inspired by VAE downsampling, not a claim of
disjoint receptive fields. The teacher's nominal +4.5 s endpoint is beyond the
4 s label horizon; no labels after 4 s are fabricated.

PDMScorer records **local** at-fault collision scores and drivable-area compliance
at every simulated step, preserving the original evaluator and its scalar scores.
Each window takes the minimum. NC preserves non-agent collision score 0.5 (it is a
soft compliance score, not strictly a Bernoulli probability). Repeated contacts may
appear in multiple windows; an earlier event is not automatically propagated into
later windows. DAC comes from the simulated vehicle footprint. TTC and comfort use
full-horizon evaluator labels. EP uses evaluator-normalized continuous progress.
BCE-with-logits trains NC/DAC/TTC/comfort and each EP window. A small auxiliary MSE
keeps the sum of the EP windows consistent with the scalar PDM progress target; the five component
losses have equal weight, with NC/DAC each averaged over the three windows.

## Run

Use an environment containing both NAVSIM/nuPlan and CogVideoX-compatible diffusers
(and the project's training dependencies). `driveworld-vla` now has diffusers 0.30.3 installed alongside nuPlan;
the four-GPU runtime preflight passes. No training was launched.

```bash
export PLANNER_CHECKPOINT=/path/to/stage1.ckpt
export WORLD_MODEL_CHECKPOINT=/path/to/TA-DWM.ckpt
export FEATURE_CACHE=/path/to/features
export METRIC_CACHE=/path/to/metric_cache
export TRAJECTORY_VOCAB=/path/to/trajectory_anchors_256.npy
export NAVSIM_EXP_ROOT=/path/to/experiments
# Also set the standard NAVSIM_DEVKIT_ROOT, OPENSCENE_DATA_ROOT, NUPLAN_MAPS_ROOT.
bash scripts/training/run_future_registers.sh \
  agent.teacher_scheduler_path=/path/to/CogVideoX-2b \
  agent.register_candidates=5 agent.teacher_chunk_size=2 \
  agent.teacher_steps=1 trainer.params.precision=bf16-mixed
```

The scheduler directory must contain `scheduler/scheduler_config.json`; the new
agent defaults to `zai-org/CogVideoX-2b` if a local copy is not provided. Teacher
checkpoint is required separately; a FAR checkpoint alone does not replace it.
First smoke-test with `trainer.params.fast_dev_run=true` and fewer candidates.

The existing WorldDrive training entrypoint calls `initialize(from_world_model=True)`:
planner weights come from `state_dict`, teacher weights from `model_state_dict`.
For exact resumption including optimizer state, use Lightning's training resume
mechanism; passing a checkpoint here only initializes model weights.

Candidates use the original stage2 sampling pattern: the planner supplies a top-16
pool, PDM scores all 16, three lowest-score winners are kept, and three candidates
are sampled from the highest-score group of ten losers. Those six candidates are
processed by the teacher in one call. The teacher uses its **own** trajectory encoder.
Candidates from the same scene share diffusion noise; validation seeds it for
repeatability. Teacher calls are chunked. Readout computation graphs are retained
until Lightning backward, so chunk size does not bound total readout activation
memory. Lower candidate count or batch size if needed. Labels are evaluated online
per candidate; this can be CPU-heavy and is not yet a label-caching pipeline.

Cached training now uses only `train_logs`; validation logs are not mixed into
training. `compute_trajectory()` is intentionally unsupported for this readout
training agent; student deployment/distillation is a separate next stage.

## Student register distillation

The student stage is now wired as `RegisterDistillAgent` with
`register_distill_4gpu.yaml`. It reuses the frozen planner and TA-DWM teacher,
but creates a new `FutureRegisterFAR` with 36 queries and 512-dimensional
outputs. For each selected candidate, the teacher produces `[36,512]` fused
registers and the student predicts `[36,512]` from planner visual tokens and the
same 512-dimensional motion token. The only loss is token-level MSE against
detached teacher tokens; no scorer or refine loss is used in this stage.

Provide a completed teacher-register checkpoint from the preceding teacher
stage:

```bash
export TEACHER_REGISTER_CHECKPOINT=/path/to/registers-teacher.ckpt
bash scripts/training/run_register_distill.sh
```

The student checkpoint is initialized from the stage-1 planner/TA-DWM weights,
while only `student_far.*` is optimized. The teacher readout checkpoint is loaded
strictly into the frozen 512-dimensional teacher readout.

For the intended single-stage version, use `JointFutureRegisterAgent` instead.
It creates teacher registers and student FAR from scratch in one run. The teacher
receives the existing PDM supervision, the student receives MSE against detached
teacher tokens, and one optimizer updates both modules:

```bash
bash scripts/training/run_joint_future_registers.sh
```

The combined loss is `teacher_register_loss + distill_loss_weight *
register_distill_loss`; no scorer is copied into the student path.

## Checks

```bash
OMP_NUM_THREADS=2 /mnt/cfs/qiuzijun/miniconda3/envs/driveworld-vla/bin/python -m unittest discover -s tests -p 'test_future_registers.py' -v
```

Tests cover window boundaries, non-cumulative labels, fractional NC, horizon
validation, per-latent attention isolation, candidate isolation, output shapes,
nonzero gradients for all three register groups and the motion projection,
frozen motion inputs, motion-dependent scores, and invalid motion shapes.
Existing observation caches remain compatible. Older readout checkpoints lack
the new motion projection parameters and cannot strictly resume this architecture.

The scorer instrumentation is also checked against controlled collision events in
an environment with nuPlan:

```bash
MPLCONFIGDIR=/tmp/worlddrive-mpl /mnt/cfs/qiuzijun/miniconda3/envs/driveworld-vla/bin/python -m unittest discover -s tests -p 'test_pdm_temporal_labels.py' -v
```

This checks the actual scorer methods preserve scalar NC/DAC while recording
per-window events. It is not a full dataset rollout or a pretrained DiT smoke run.

## Platform: four GPUs, five candidates

The platform launcher follows DrivoR's environment isolation, explicit Python
selection and runtime logging:

- Launcher: `scripts/run_future_registers_4gpu.sh`
- Full training config: `navsim/planning/script/config/training/future_registers_4gpu.yaml`
- Fixed interpreter (edit the script to change):
  `/mnt/cfs/qiuzijun/miniconda3/envs/driveworld-vla/bin/python`

Defaults: one node, four GPUs, BF16, one scene per GPU, five candidates per scene,
teacher chunk size two (2+2+1), 10 epochs, checkpoints every epoch. Only the readout
is trained. Both the agent's Python default and its base YAML now use five candidates.

Set the actual training asset paths in the platform environment (or edit the YAML):

```bash
export WORLD_MODEL_CHECKPOINT=/path/to/TA-DWM.ckpt
export FEATURE_CACHE=/path/to/worlddrive_train_cache
export METRIC_CACHE=/path/to/navsim_metric_cache
# Optional: local scheduler root containing scheduler/scheduler_config.json
export TEACHER_SCHEDULER_PATH=/path/to/CogVideoX-2b
bash /mnt/cfs/qiuzijun/WorldDrive/scripts/run_future_registers_4gpu.sh
```

The YAML defaults to the existing stage-1 planner checkpoint and the local anchor
path. `PLANNER_CHECKPOINT` and `TRAJECTORY_VOCAB` can override them. Feature caches
must contain WorldDrive's two-resolution VAE features, not DrivoR features.

To supply an alternate YAML or Hydra overrides, put the YAML first, as in DrivoR:

```bash
bash scripts/run_future_registers_4gpu.sh \
  /mnt/cfs/qiuzijun/WorldDrive/navsim/planning/script/config/training/future_registers_4gpu.yaml \
  trainer.params.fast_dev_run=true
```

`WORLDDRIVE_PREFLIGHT_ONLY=1` runs only the environment/CUDA checks. Runtime logs
are written under `$NAVSIM_EXP_ROOT/runtime_logs`. GPU IDs default to 0,1,2,3 and
respect `CUDA_VISIBLE_DEVICES`. No WANDB config was copied: the WorldDrive training
entrypoint does not currently wire up DrivoR's custom WANDB integration.

See `docs/future_register_readiness.md` for the audited asset locations and compact
DrivoR metric-cache compatibility adapter.
