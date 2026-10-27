# Resume joint future-register training on the platform

Run in the foreground as the platform job command, with four allocated GPUs:

```bash
export RESUME_CHECKPOINT=/mnt/cfs/qiuzijun/WorldDrive/exp/joint_future_registers_top16_pdm6_4gpu/2026.09.19.22.21.18/checkpoints/joint-registers-epoch=00-step=592.ckpt
bash /mnt/cfs/qiuzijun/WorldDrive/scripts/training/run_joint_future_registers.sh
```

`RESUME_CHECKPOINT` resolves to the top-level `resume_checkpoint_path` in
`joint_future_registers_4gpu.yaml`. The training entrypoint validates that file
before initializing W&B or loading data, then passes it to `Trainer.fit(ckpt_path=...)`.
Lightning restores model parameters, optimizer, LR scheduler and loop progress.
The planner/DiT initialization checkpoints remain separate from this full-state
resume checkpoint.

`max_epochs: 10` is the total target, including completed epochs. Resuming the
epoch-0 checkpoint continues with epoch 1 and leaves nine epochs to complete.
The interrupted portion of epoch 1 was not saved and must run again.

Without `RESUME_CHECKPOINT`, the default is `null` (fresh training). A Hydra
override takes precedence over the environment variable. Keep the inner quotes
when the filename contains `=`:

```bash
bash scripts/training/run_joint_future_registers.sh \
  'resume_checkpoint_path="/absolute/path/joint-registers-epoch=00-step=592.ckpt"'
```

The persistent wrapper also forwards Hydra arguments and inherits the environment.
For platform jobs, the foreground launcher lets the platform manage the process
and capture its output.

Model resume and W&B resume are separate. By default this creates a new W&B run
and a new output directory, retaining the existing run and checkpoint. Do not set
`WANDB_RUN_ID` unless deliberately appending to an existing run; the previous run
contains steps beyond this checkpoint, so reusing it can create overlapping steps.
Checkpoints from the resumed run are saved in its new output directory.
