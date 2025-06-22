"""
LightningModule wrapper that can train any classification backbone described in a Hydra config.

Example YAML stanza
-------------------
model:
  _target_: src.models.envnet_v2.EnvNetV2
  num_classes: ${dataset.num_classes}   # auto-filled by dataset cfg

loss:
  _target_: torch.nn.CrossEntropyLoss
  label_smoothing: 0.1
"""

from __future__ import annotations

from typing import Any, Dict

import lightning.pytorch as pl
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig


def _adapt_head_if_possible(model: nn.Module, num_classes: int) -> None:
    """
    If the backbone exposes a `replace_head()` or `classifier.out_features`
    attribute, resize it to match `num_classes`. Silently does nothing when
    the interface isn't found.
    """
    if hasattr(model, "replace_head") and callable(model.replace_head):
        model.replace_head(num_classes)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        in_feat = model.classifier.in_features
        model.classifier = nn.Linear(in_feat, num_classes)
    elif hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_feat = model.fc.in_features
        model.fc = nn.Linear(in_feat, num_classes)
    # else: leave unchanged - assume caller set correct `num_classes`


class LitClassifier(pl.LightningModule):
    """
    Generic train-loop wrapper.

    Args
    ----
    model_cfg : DictConfig
        Hydra node describing the *backbone* (e.g. EnvNetV2).
    optim_cfg : DictConfig
        Optimiser spec, e.g. torch.optim.SGD.
    sched_cfg : DictConfig
        LR scheduler spec; set to `null` in YAML to skip.
    loss_cfg  : DictConfig
        Loss function spec; CrossEntropyLoss by default.
    """

    def __init__(
        self,
        model_cfg: DictConfig,
        optim_cfg: DictConfig,
        sched_cfg: DictConfig | None = None,
        loss_cfg: DictConfig | None = None,
        metric_cfg: DictConfig | None = None,
    ) -> None:
        super().__init__()

        # 1) Instantiate backbone
        self.model: nn.Module = instantiate(model_cfg)

        # 2) Auto-adapt head if dataset defined `num_classes`
        if "num_classes" in model_cfg and isinstance(
            model_cfg["num_classes"], (int, float)
        ):
            _adapt_head_if_possible(self.model, int(model_cfg["num_classes"]))

        # 3) Loss & metrics
        self.criterion: nn.Module = (
            instantiate(loss_cfg) if loss_cfg is not None else nn.CrossEntropyLoss()
        )

        # optional metrics (e.g. torchmetrics.Accuracy)
        self.metric = instantiate(metric_cfg) if metric_cfg is not None else None

        # 4) Store optimiser/scheduler cfg for later
        self._optim_cfg = optim_cfg
        self._sched_cfg = sched_cfg

        # 5) Lightning will auto-save everything in `self.hparams`
        #    Convert nested DictConfig → Python dict so JSON-serialisable
        self.save_hyperparameters(
            {
                "optim": (
                    dict(self._optim_cfg)
                    if isinstance(self._optim_cfg, DictConfig)
                    else self._optim_cfg
                ),
                "sched": (
                    dict(self._sched_cfg)
                    if isinstance(self._sched_cfg, DictConfig)
                    else self._sched_cfg
                ),
                "loss": str(self.criterion),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch: Any, stage: str) -> torch.Tensor:
        """Shared logic for train/val/test batches."""
        # The DataLoader can either return a tuple (x, y) or a dict with more keys
        if isinstance(batch, dict):
            x, y = batch["inputs"], batch["labels"]
        else:
            x, y = batch  # type: ignore[misc]

        logits = self(x)
        loss = self.criterion(logits, y)

        # Lightning metric logging
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        if self.metric is not None:
            self.metric.update(logits, y)

        return loss

    # Lightning hooks ------------------------------------------------------- #
    def training_step(self, batch: Any, batch_idx: int):  # type: ignore[override]
        return self._step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int):  # type: ignore[override]
        self._step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int):  # type: ignore[override]
        self._step(batch, "test")

    def on_validation_epoch_end(self) -> None:  # noqa: D401
        if self.metric is not None:
            val_acc = self.metric.compute()
            self.log("val/acc", val_acc, prog_bar=True)
            self.metric.reset()

    def on_test_epoch_end(self) -> None:  # noqa: D401
        if self.metric is not None:
            test_acc = self.metric.compute()
            self.log("test/acc", test_acc, prog_bar=True)
            self.metric.reset()

    def configure_optimizers(self):
        optim = instantiate(self._optim_cfg, params=self.parameters())

        if self._sched_cfg is None:
            return optim

        sched = instantiate(self._sched_cfg, optimizer=optim)

        # Lightning expects a dict for schedulers with extra settings
        if isinstance(sched, dict):
            return {"optimizer": optim, "lr_scheduler": sched}
        return {"optimizer": optim, "lr_scheduler": {"scheduler": sched}}


def build_from_cfg(cfg: DictConfig) -> LitClassifier:
    """
    Helper so scripts can do:
        from src.training.engine import build_from_cfg
        model = build_from_cfg(cfg)
    """
    return LitClassifier(
        model_cfg=cfg.model,
        optim_cfg=cfg.optimizer,
        sched_cfg=cfg.scheduler,
        loss_cfg=cfg.get("loss"),
        metric_cfg=cfg.get("metric"),
    )
