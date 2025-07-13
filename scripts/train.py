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

os.environ["HYDRA_FULL_ERROR"] = "1"

# Add project root to path before local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
import numpy as np
import torch
from hydra import main as hydra_main
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.training.callbacks import build_callbacks  # optional helper

# local imports
from src.training.engine import build_from_cfg


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
@hydra_main(
    version_base=None,
    config_path="../configs",  # <- relative to THIS file
    config_name="training",
)
def train(cfg: DictConfig) -> None:  # noqa: D401
    """Main training function (Hydra injects `cfg`)."""
    # --------------------------------------------------------------------- #
    # 0)  Reproducibility & Performance
    # --------------------------------------------------------------------- #
    fix_seed(cfg.seed)
    
    # Optimize for H100/A100 Tensor Cores
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    # --------------------------------------------------------------------- #
    # 1)  Instantiate DataModule & LightningModule
    # --------------------------------------------------------------------- #
    # Build complete datamodule configuration merging model-specific overrides
    datamodule_cfg = {
        "_target_": cfg.dataset._target_,
        "root": cfg.dataset.root,
        "fold": cfg.dataset.fold,
        "val_split": cfg.dataset.val_split,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "num_classes": cfg.dataset.get("num_classes", 50),
        # Dataset-level augmentation settings
        "enable_bc_mixing": cfg.dataset.get("enable_bc_mixing", False),
        "enable_mixup": cfg.dataset.get("enable_mixup", False),
        "mixup_alpha": cfg.dataset.get("mixup_alpha", 0.5),
    }
    
    # Extract dataset overrides from model config before creating the model
    model_cfg = dict(cfg.model)
    dataset_overrides = model_cfg.pop("dataset_overrides", None)
    
    # Apply model-specific dataset overrides if they exist
    if dataset_overrides:
        for key, value in dataset_overrides.items():
            datamodule_cfg[key] = value
        print(f"✓ Applied model-specific dataset overrides: {list(dataset_overrides.keys())}")
    else:
        # Fallback to default values
        datamodule_cfg.update({
            "preprocessing_mode": cfg.dataset.get("preprocessing_mode", "envnet_v2"),
            "preprocessing_config": cfg.dataset.get("preprocessing_config", {}),
            "augment": cfg.dataset.get("augment", {}),
            "is_spectrogram": cfg.dataset.get("is_spectrogram", False),
        })
        print("⚠ No model-specific dataset overrides found, using defaults")
    
    # Create datamodule with merged config
    datamodule = instantiate(datamodule_cfg)
    
    # Create model config without dataset_overrides for clean instantiation
    from omegaconf import OmegaConf
    clean_model_cfg = OmegaConf.create(model_cfg)
    clean_cfg = OmegaConf.create(dict(cfg, model=clean_model_cfg))
    lit_model = build_from_cfg(clean_cfg)  # generic classifier

    # --------------------------------------------------------------------- #
    # 2)  MLflow logger & autolog
    # --------------------------------------------------------------------- #
    # Use absolute path for MLflow to avoid permission issues with Hydra's changing directories
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_mlruns = f"file:{project_root}/mlruns"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", default_mlruns)
    logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=tracking_uri,
    )
    
    # Log all configuration parameters to MLflow
    def flatten_config(config_dict, parent_key='', sep='_'):
        """Flatten nested config dictionary for MLflow logging."""
        items = []
        for k, v in config_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_config(v, new_key, sep=sep).items())
            else:
                # Convert to string for MLflow compatibility
                if v is None:
                    v = "None"
                elif isinstance(v, (list, tuple)):
                    v = str(v)
                elif hasattr(v, '__name__'):  # For classes/functions
                    v = str(v)
                items.append((f"cfg_{new_key}", v))  # Prefix to avoid conflicts
        return dict(items)
    
    # Convert OmegaConf to regular dict and flatten
    from omegaconf import OmegaConf
    # Convert to container to resolve interpolations and get regular Python dict
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    flat_params = flatten_config(config_dict)
    
    # Log all parameters to MLflow manually (before trainer setup to avoid conflicts)
    if hasattr(logger, 'experiment') and hasattr(logger, 'run_id'):
        for param_name, param_value in flat_params.items():
            try:
                # Convert to string and truncate if too long (MLflow has limits)
                param_str = str(param_value)
                if len(param_str) > 250:  # MLflow parameter value limit
                    param_str = param_str[:247] + "..."
                logger.experiment.log_param(logger.run_id, param_name, param_str)
            except Exception as e:
                print(f"⚠ Failed to log parameter {param_name}: {e}")
    print(f"✓ Logged {len(flat_params)} configuration parameters to MLflow")

    # --------------------------------------------------------------------- #
    # 3)  Calculate dynamic logging frequency and setup Trainer
    # --------------------------------------------------------------------- #
    # Calculate steps per epoch for dynamic logging
    trainer_cfg = dict(cfg.trainer)
    
    # Calculate steps per epoch if not already set
    if "log_every_n_steps" not in trainer_cfg or trainer_cfg.get("log_every_n_steps") is None:
        try:
            # Setup datamodule to get train dataloader length
            datamodule.setup("fit")
            train_loader = datamodule.train_dataloader()
            steps_per_epoch = len(train_loader)
            
            # Set log_every_n_steps to log approximately once per epoch
            trainer_cfg["log_every_n_steps"] = max(1, steps_per_epoch)
            print(f"✓ Dynamic logging: {steps_per_epoch} steps per epoch, logging every {trainer_cfg['log_every_n_steps']} steps")
        except Exception as e:
            print(f"⚠ Could not calculate dynamic logging frequency: {e}, using default")
            trainer_cfg["log_every_n_steps"] = 50  # fallback
    
    trainer = pl.Trainer(
        **trainer_cfg,
        logger=logger,
        callbacks=build_callbacks(cfg),  # ModelCheckpoint, etc.
    )

    # --------------------------------------------------------------------- #
    # 4)  Fit & (optionally) test
    # --------------------------------------------------------------------- #
    ckpt_path = cfg.get("ckpt_path", None)  # resume or None
    trainer.fit(lit_model, datamodule=datamodule, ckpt_path=ckpt_path)
    trainer.test(ckpt_path="best", datamodule=datamodule)

    print("✓ training run finished — metrics & checkpoints logged.")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    train()  # Hydra will parse CLI overrides automatically
