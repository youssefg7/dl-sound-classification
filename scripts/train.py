#!/usr/bin/env python
"""
Trains any model / dataset described by the Hydra configs, logs everything to MLflow, and saves Lightning checkpoints.

Examples:
- Train EnvNet-v2 on ESC-50 fold 0
  ```bash
  python scripts/train.py dataset=esc50 dataset.fold=0 model=envnet_v2
  ```

- Resume from a checkpoint
  ```bash
  python scripts/train.py +ckpt_path=outputs/checkpoints/epoch=19-step=900.ckpt
  ```
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

# Add project root to path before local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict

import numpy as np
import lightning.pytorch as pl
import torch
import mlflow
import mlflow.pytorch
from hydra import compose, initialize, initialize_config_dir, main as hydra_main
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# local imports
from src.training.engine import build_from_cfg
from src.training.callbacks import build_callbacks   # optional helper

# --------------------------------------------------------------------------- #
# Utility: deterministic seeding                                              #
# --------------------------------------------------------------------------- #
def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)


# --------------------------------------------------------------------------- #
# Hydra entry-point                                                           #
# --------------------------------------------------------------------------- #
@hydra_main(version_base=None,
            config_path="../configs",   # <- relative to THIS file
            config_name="training")
def train(cfg: DictConfig) -> None:     # noqa: D401
    """Main training function (Hydra injects `cfg`)."""
    # --------------------------------------------------------------------- #
    # 0)  Reproducibility
    # --------------------------------------------------------------------- #
    fix_seed(cfg.seed)

    # --------------------------------------------------------------------- #
    # 1)  Instantiate DataModule & LightningModule
    # --------------------------------------------------------------------- #
    # The datamodule's __init__ doesn't accept all keys in the config (e.g.,
    # num_classes, which is metadata for the model), so we build a new config
    # for it that only contains the arguments it knows about.
    datamodule_cfg = {
        "_target_": cfg.dataset._target_,
        "root": cfg.dataset.root,
        "fold": cfg.dataset.fold,
        "val_split": cfg.dataset.val_split,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "augment": cfg.dataset.augment,
    }
    datamodule = instantiate(datamodule_cfg)
    lit_model  = build_from_cfg(cfg)             # generic classifier

    # --------------------------------------------------------------------- #
    # 2)  MLflow logger & autolog
    # --------------------------------------------------------------------- #
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    logger = pl.loggers.MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=tracking_uri,
    )
    mlflow.pytorch.autolog(log_models=False)     # Lightning-aware

    # --------------------------------------------------------------------- #
    # 3)  Trainer with callbacks
    # --------------------------------------------------------------------- #
    trainer = pl.Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=build_callbacks(cfg),        # ModelCheckpoint, etc.
    )

    # --------------------------------------------------------------------- #
    # 4)  Fit & (optionally) test
    # --------------------------------------------------------------------- #
    ckpt_path = cfg.get("ckpt_path", None)       # resume or None
    trainer.fit(lit_model, datamodule=datamodule, ckpt_path=ckpt_path)
    trainer.test(ckpt_path="best", datamodule=datamodule)

    print("✓ training run finished — metrics & checkpoints logged.")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    train()      # Hydra will parse CLI overrides automatically
