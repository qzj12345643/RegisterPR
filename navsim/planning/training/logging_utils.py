"""Training logger construction for TensorBoard and optional Weights & Biases."""

import logging
import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

logger = logging.getLogger(__name__)


def _distributed_process_rank() -> int:
    """Return the launcher rank before Lightning has initialized process groups."""
    for variable in ("LOCAL_RANK", "RANK", "SLURM_PROCID"):
        value = os.environ.get(variable)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            logger.warning("Ignoring invalid distributed rank %s=%r", variable, value)
    return 0


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _initialize_wandb(wandb_logger: WandbLogger, cfg: DictConfig) -> None:
    """Initialize W&B metadata without logging credentials or the full environment."""
    keys = (
        "experiment_name",
        "seed",
        "resume_checkpoint_path",
        "agent",
        "trainer",
        "dataloader",
        "cache_path",
    )
    hyperparameters = {
        key: OmegaConf.to_container(cfg[key], resolve=True)
        if OmegaConf.is_config(cfg[key])
        else cfg[key]
        for key in keys
        if key in cfg
    }
    wandb_logger.log_hyperparams(hyperparameters)
    run = wandb_logger.experiment
    if run.url:
        logger.info("W&B run: %s", run.url)
    else:
        logger.info("W&B mode=%s; local run directory: %s", cfg.wandb.mode, run.dir)


def build_training_loggers(cfg: DictConfig):
    """Build TensorBoard plus a single rank-zero W&B logger when enabled."""
    options = cfg.get("wandb")
    if not options or not _as_bool(options.get("enabled", False)):
        return True

    mode = str(options.get("mode", "online"))
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb.mode must be online, offline or disabled")
    if mode == "disabled":
        return True

    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard = TensorBoardLogger(save_dir=str(output_dir), name="lightning_logs")

    # Lightning may execute this function once per DDP subprocess.  Only rank
    # zero may initialize an online W&B run; the other ranks keep TensorBoard
    # so Lightning still has a logger for synchronized self.log calls.
    if _distributed_process_rank() != 0:
        logger.info("Skipping W&B logger on distributed rank %s", _distributed_process_rank())
        return [tensorboard]

    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        raise ValueError(
            "W&B is enabled in online mode but WANDB_API_KEY is missing. "
            "Set it in the launch environment or provide the configured credentials file."
        )

    wandb_logger = WandbLogger(
        project=options.get("project", "worlddrive"),
        entity=options.get("entity"),
        name=options.get("name") or cfg.experiment_name,
        save_dir=str(output_dir),
        mode=mode,
        id=os.environ.get("WANDB_RUN_ID"),
        resume=os.environ.get("WANDB_RESUME", "allow"),
        log_model=False,
    )
    _initialize_wandb(wandb_logger, cfg)
    return [tensorboard, wandb_logger]
