"""
Factory helpers that return pre-configured Lightning callbacks based on values in the Hydra config tree.


YAML snippet
------------
checkpoint:
  monitor: val/acc
  mode: max
  save_top_k: 3
  dirpath: outputs/checkpoints
early_stop:
  monitor: val/acc
  mode: max
  patience: 8
swa:
  enabled: false           # toggle Stochastic Weight Averaging
  swa_lrs: 1e-4
"""

from __future__ import annotations

from typing import List

import lightning.pytorch as pl
from lightning.pytorch.callbacks import TQDMProgressBar


# --------------------------------------------------------------------------- #
# Core callback builders                                                      #
# --------------------------------------------------------------------------- #
def _build_checkpoint(cfg) -> pl.callbacks.ModelCheckpoint:
    """Top-k checkpoint on a metric (defaults come from YAML)."""
    import os
    from hydra.core.hydra_config import HydraConfig
    
    # Get checkpoint config
    checkpoint_cfg = dict(cfg.checkpoint)
    
    # Use Hydra's output directory for checkpoints
    try:
        # Get Hydra's runtime configuration
        hydra_cfg = HydraConfig.get()
        output_dir = hydra_cfg.runtime.output_dir
        dirpath = checkpoint_cfg.get("dirpath", "checkpoints")
        
        # Always use Hydra's output directory
        checkpoint_cfg["dirpath"] = os.path.join(output_dir, dirpath)
        
    except Exception:
        # Fallback if Hydra config is not available
        dirpath = checkpoint_cfg.get("dirpath", "checkpoints")
        if not os.path.isabs(dirpath):
            checkpoint_cfg["dirpath"] = os.path.join(os.getcwd(), dirpath)
    
    return pl.callbacks.ModelCheckpoint(**checkpoint_cfg)


def _build_early_stop(cfg) -> pl.callbacks.EarlyStopping | None:
    """Early stopping—only if `early_stop` section exists."""
    if "early_stop" not in cfg:
        return None
    return pl.callbacks.EarlyStopping(**cfg.early_stop)


def _build_lr_monitor() -> pl.callbacks.LearningRateMonitor:
    """Log LR each epoch (useful for schedulers)."""
    return pl.callbacks.LearningRateMonitor(logging_interval="epoch")


def _build_swa(cfg) -> pl.callbacks.StochasticWeightAveraging | None:
    """Optional Stochastic Weight Averaging."""
    if not cfg.get("swa", {}).get("enabled", False):
        return None
    swa_cfg = cfg.swa
    return pl.callbacks.StochasticWeightAveraging(
        swa_lrs=swa_cfg.get("swa_lrs", 1e-3),
        swa_epoch_start=swa_cfg.get("swa_epoch_start", 0.75),
    )


# --------------------------------------------------------------------------- #
# Public factory                                                              #
# --------------------------------------------------------------------------- #
def build_callbacks(cfg) -> List[pl.callbacks.Callback]:
    """
    Assemble a list of callbacks based solely on the config tree.

    Returns
    -------
    list[Callback]
        Ready to be passed to `pl.Trainer(callbacks=...)`.
    """
    cbs: list[pl.callbacks.Callback | None] = [
        _build_checkpoint(cfg),
        _build_early_stop(cfg),
        _build_lr_monitor(),
        _build_swa(cfg),
        TQDMProgressBar(leave=True),  # Keep epoch lines visible
    ]
    # filter out None entries (callbacks disabled by config)
    return [cb for cb in cbs if cb is not None]
