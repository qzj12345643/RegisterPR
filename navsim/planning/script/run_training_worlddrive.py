from typing import Tuple
from pathlib import Path
import logging
import torch
from tqdm import tqdm

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.logging_utils import build_training_loggers

torch.set_float32_matmul_precision('high')

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    # Validate before constructing loggers/models or scanning the feature cache.
    resume_checkpoint_path = cfg.get("resume_checkpoint_path")
    if resume_checkpoint_path is not None:
        checkpoint_path = Path(str(resume_checkpoint_path)).expanduser()
        if not str(resume_checkpoint_path).strip() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_checkpoint_path}")
        resume_checkpoint_path = str(checkpoint_path.resolve())
        logger.info("Resuming full training state from: %s", resume_checkpoint_path)

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")
    training_loggers = build_training_loggers(cfg)

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize(from_world_model=True)
    logger.info("Load from pretrained world model")

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    # for debug
    # train_data = Subset(train_data, range(2))
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)
    print(len(train_data))
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")

    if "checkpoint" in cfg.trainer.params:
        checkpoint_params = cfg.trainer.params["checkpoint"]
        trainer_params = OmegaConf.to_container(cfg.trainer.params, resolve=True)  # 转换为普通字典
        del trainer_params["checkpoint"]
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_params.get('dirpath', 'path/to/save'),  # 默认保存路径
            filename=checkpoint_params.get('filename', 'd-epoch={epoch}-step={step}'),  # 默认保存文件格式
            every_n_epochs=checkpoint_params.get('every_n_epochs', 10),  # 每10个epoch保存一次
            save_on_train_epoch_end=checkpoint_params.get('save_on_train_epoch_end', None),
            save_top_k=-1,  # 保存所有检查点
            verbose=True,
        )
        callbacks = [checkpoint_callback] + agent.get_training_callbacks()
    else:
        callbacks = agent.get_training_callbacks()
        trainer_params = cfg.trainer.params

    if "strategy" not in cfg.trainer.params:
        trainer = pl.Trainer(
            strategy=DDPStrategy(find_unused_parameters=True),
            **trainer_params,
            callbacks=callbacks,
            logger=training_loggers,
        )
    else:
        trainer = pl.Trainer(**trainer_params, callbacks=callbacks, logger=training_loggers)

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=resume_checkpoint_path,
    )


if __name__ == "__main__":
    main()
