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


# --------------------------------------------------------------------------- #
# Core callback builders                                                      #
# --------------------------------------------------------------------------- #
def _build_checkpoint(cfg) -> pl.callbacks.ModelCheckpoint:
    """Top-k checkpoint on a metric (defaults come from YAML)."""
    return pl.callbacks.ModelCheckpoint(**cfg.checkpoint)


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
    ]
    # filter out None entries (callbacks disabled by config)
    return [cb for cb in cbs if cb is not None]
