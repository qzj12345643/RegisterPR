# Future-register training asset audit

Environment: `/mnt/cfs/qiuzijun/miniconda3/envs/driveworld-vla/bin/python`, as selected
by `DriveWorld-VLA/scripts/driveworld_env.sh`. Installed `diffusers==0.30.3`; retained
Torch 2.0.1+cu118, torchvision 0.15.2+cu118 and transformers 4.37.2. CogVideoX VAE,
DPM scheduler and FutureRegisterAgent imports pass. Platform preflight passes with
four visible GPUs, BF16 support and a CUDA arithmetic check. No training launched.

## Weights

| Asset | Status |
| --- | --- |
| Stage-1 planner | Present: `ckpts/worlddrive_stage1_train_87p7_pdms.ckpt` |
| Stage-2 FAR | Present, but not required for training the new readout |
| Trajectory vocabulary | Present: `ckpts/trajectory_anchors_256.npy` |
| CogVideoX VAE | Present: `ckpts/CogVideoX-2b/vae/` |
| CogVideoX scheduler config | Downloaded: `ckpts/CogVideoX-2b/scheduler/scheduler_config.json` |
| TA-DWM teacher | Downloaded: `ckpts/worldtraj_stage1_512_tadwm.pkl`; SHA-256 verified against official LFS |

The official Hugging Face model inventory contains both
`worldtraj_stage1_512_tadwm.pkl` and `worldtraj_stage1_1024_tadwm.pkl`:
https://huggingface.co/tabguigui/WorldDrive/tree/main

The current teacher runs at 256x512 and the YAML now defaults to
`ckpts/worldtraj_stage1_512_tadwm.pkl`; override `WORLD_MODEL_CHECKPOINT` to select a
different compatible TA-DWM checkpoint. The existing planner and FAR checkpoints
are not substitutes for the teacher checkpoint. The teacher download and feature-cache generation were subsequently requested and started.

## Data

Full filename/metadata coverage audit against this checkout's navtrain selection:

- 1,192 requested logs, all present in `/mnt/cfs/datasets/navsim-v1/navsim_logs/trainval`.
- 103,288 requested tokens, all found in those logs.
- 152,495 distinct front-camera files required by current + three history frames,
  all present under `/mnt/cfs/datasets/navsim-v1/sensor_blobs/trainval`.
- Maps: `/mnt/cfs/datasets/navsim-v1/maps`, pointing to DriveWorld-VLA's maps.
- This verifies filename availability, not JPEG decoding of every file.

`DrivoR/exp/train_metric_cache_ego_area_v2` indexes all 103,288 required tokens.
SafeDrive's `exp/train_metric_cache_navtrain` indexes only 31,055 records.
DriveWorld-VLA's own raw dataset paths mainly contain test data; use the shared
trainval paths above, which the DrivoR training setup also uses.

DrivoR/SafeDrive training metric caches are **not** standard WorldDrive caches:
there is no reference trajectory; reference progress is saved instead, and map
polygons are serialized as coordinate arrays. `training_metric_cache.py` now
adapts these in memory and uses the masked reference progress for EP normalization.
The register agent uses this loader/scorer. Standard caches remain supported.
Three real DrivoR caches from the beginning/middle/end of its index passed rollout,
scalar scoring and temporal-label extraction in this checkout. This is a sample
compatibility check, not a full deserialization audit of all metric files.

WorldDrive's required feature format is `trajworld_feature.gz` containing
`visual_token`, `low_visual_token`, `status_feature`, paired with
`trajectory_target.gz`. No compatible cache has been found in the three projects.
DrivoR's `drivor_feature_wm_f8.gz` is a different format and cannot be renamed to
replace it. The original raw current/history frames and VAE weights are available;
the WorldDrive cache must still be generated before full training.

Also fixed the target builder's missing return when no static-anchor reward cache
is supplied. That cache is unnecessary here because labels come from the simulator.

## Reports

- `outputs/register_readiness/data_audit.json`: per-log required/missing image counts.
- `outputs/register_readiness/metric_reuse.json`: three real cache scoring checks.
- `outputs/register_readiness/tests.log`: unit checks in the selected environment.
- `exp/runtime_logs/`: platform environment checks.

The launch YAML now points to the shared raw data, DrivoR's complete metric cache,
local planner/VAE/anchors/scheduler and the expected teacher path. Full training must wait until feature caching is complete.

Audit limits: the broad recursive search through all experiment directories was
stopped after several minutes of cold filesystem traversal with no matching
`trajworld_feature.gz` found. Named cache roots and their formats were inspected;
this does not prove no such file exists anywhere in the projects. The raw-data
coverage audit above did complete for all selected logs/tokens. Eight tests passed
in the DriveWorld-VLA environment, including compact-reference normalization.


## Download and cache execution (2026-09-18)

Downloaded and SHA-256 verified (matching official LFS). The teacher is `ckpts/worldtraj_stage1_512_tadwm.pkl`, official size
3,535,352,359 bytes; official LFS SHA-256:
`41b351e24938463736bff1ec4c1a493168266a75dee5ef3feccd3e1c99e104c5`.

Four real samples passed VAE encoding, shape/finite checks and gzip round-trip.
Full navtrain caching is launched on CUDA devices 0,1,2,3, in the DriveWorld-VLA
Python environment. The implementation reuses the original FP32 VAE feature
builder and posterior sampling, seeded per token for reproducible resumption.
Only current + three history CAM_F0 images are loaded. Target generation reuses
Scene's ego-pose and relative-trajectory functions without loading maps.

- Launcher: `scripts/cache_register_features.sh`
- Output: `exp/worlddrive_train_cache/<log>/<token>/`
- Log: `exp/register_feature_caching/run.log`
- Launcher PID: `exp/register_feature_caching/launcher.pid`
- Per-worker progress: `exp/register_feature_caching/rank{0,1,2,3}.json`
- Progress summary: `python3 scripts/register_cache_status.py`
- Resume after interruption: `bash scripts/cache_register_features.sh`

Features: visual_token [2,16,64,128], low_visual_token [2,16,32,64],
status_feature [8]; target: trajectory [8,3]. Cache files are atomically committed
only after reloading and comparing their tensors. A failed sample stops its
worker (and torchrun), so incomplete runs are visible rather than silently skipped.
Existing complete feature/target pairs are skipped on resume. Workers wait when
less than 24 GiB CUDA memory (including their reusable allocator cache) is
available; initialization/wait messages are in run.log.
These are **history observation feature caches**, not precomputed candidate future
latents. Candidate futures and candidate-dependent PDM labels remain online during
register training. No full training run has been launched.

The teacher instantiation check exposed a diffusers 0.30.3 API mismatch in
3-D positional embeddings. `blocks/cogvideox.py` now imports the existing local
Torch implementation from `blocks/embeddings.py`, which supports device/output_type.

The metric-cache index loader also now accepts absolute paths under named cache
roots (DrivoR train_metric_cache_ego_area_v2) while preserving legacy /metric_cache/
relocation. Small regression checks live in tests/test_teacher_runtime_compat.py.

Final verification: official teacher SHA-256 matches; all 761 checkpoint teacher
tensors match the instantiated model, and the external anchor vocabulary matches.
See outputs/register_readiness/teacher_weights.json and teacher_download.json.
Two runtime compatibility regression tests pass. The four-GPU full cache job is
running (launcher PID in exp/register_feature_caching/launcher.pid); it is not yet
complete. The teacher check was CPU loading/parameter verification only; a full
five-candidate GPU forward/backward remains a separate smoke check, available via
scripts/check_register_teacher.sh.

GPU migration: caching now runs on GPU 4,5,6,7; the cache launcher's default
was updated accordingly (CUDA_VISIBLE_DEVICES can override it). See active_run.json
in exp/register_feature_caching for the current launcher PID and device list.
A network-filesystem rename returned ENOENT despite a committed destination.
Atomic cache writes now verify that destination against the computed tensors,
and retry up to three times if needed; two injected-failure regression tests pass.
