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

from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchmetrics.classification import MulticlassF1Score, MulticlassAUROC, MulticlassConfusionMatrix, MulticlassAccuracy
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tempfile


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
    metric_cfg : DictConfig
        Metric function spec; Accuracy by default.
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

        # Helper method to check if using KLDivLoss
        self._is_kl_loss = self._is_kl_divergence_loss()

        # optional metrics (e.g. torchmetrics.Accuracy)
        self.metric = instantiate(metric_cfg) if metric_cfg is not None else None
        if self.metric is None:
            metric_cfg = DictConfig(
                {
                    "_target_": "torchmetrics.classification.MulticlassAccuracy",
                    "num_classes": model_cfg.get("num_classes", None),
                }
            )
            
        self.train_acc = instantiate(metric_cfg)
        self.val_acc = instantiate(metric_cfg)
        self.test_acc = instantiate(metric_cfg)
        self.test_f1 = MulticlassF1Score(num_classes=model_cfg.get("num_classes", None))
        self.test_auroc = MulticlassAUROC(num_classes=model_cfg.get("num_classes", None))
        self.test_confmat = MulticlassConfusionMatrix(num_classes=model_cfg.get("num_classes", None))
        self.test_class_acc = MulticlassAccuracy(num_classes=model_cfg.get("num_classes", None),
                                                 average=None)

        self.train_acc_history = []
        self.val_acc_history = []
        self.epoch_history = []

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

    def _is_kl_divergence_loss(self) -> bool:
        """Check if the criterion is KLDivLoss."""
        return (hasattr(self.criterion, '__class__') and 
                'KL' in self.criterion.__class__.__name__)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch: Any, stage: str) -> torch.Tensor:
        """Shared logic for train/val/test batches."""
        # The DataLoader can either return a tuple (x, y) or a dict with more keys
        if isinstance(batch, dict):
            x, y = batch["inputs"], batch["labels"]
        else:
            x, y = batch 

        # Handle multi-crop testing (x is list of crops)
        if isinstance(x, (list, tuple)) and stage == "test":
            # x is list of length 10, each (B, 1, 220500)
            logits_stacked = torch.stack([self(x_i) for x_i in x], dim=0)  # (10, B, C)
            logits = logits_stacked.mean(dim=0)                            # (B, C)
        else:
            logits = self(x)
        
        # Handle both hard labels (int) and soft labels (float tensor)
        if y.dtype == torch.float32 and y.dim() > 1:
            # Soft labels (e.g., from BC mixing) - use KL divergence or similar
            if self._is_kl_loss:
                # For KL divergence loss, convert logits to log probabilities
                log_probs = torch.nn.functional.log_softmax(logits, dim=1)
                # KLDivLoss expects target to be probabilities (not log probabilities)
                target_probs = y
                loss = self.criterion(log_probs, target_probs)
            else:
                # For CrossEntropyLoss with soft labels, use manual soft cross-entropy
                # CrossEntropyLoss expects logits and hard labels, but we have soft labels
                probs = torch.nn.functional.softmax(logits, dim=1)
                loss = -torch.sum(y * torch.log(probs + 1e-8), dim=1).mean()
            
            # For metrics, convert soft labels to hard labels (argmax)
            hard_labels = torch.argmax(y, dim=1)
        else:
            # Hard labels (standard case)
            loss = self.criterion(logits, y)
            hard_labels = y

        # Lightning metric logging
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        if stage == "train":
            self.train_acc.update(logits, hard_labels)
        elif stage == "val":
            self.val_acc.update(logits, hard_labels)
        elif stage == "test":
            self.test_acc.update(logits, hard_labels)
            self.test_f1.update(logits, hard_labels)
            self.test_auroc.update(logits, hard_labels)
            self.test_confmat.update(logits, hard_labels)
            self.test_class_acc.update(logits, hard_labels)
        return loss

    # Lightning hooks ------------------------------------------------------- #
    def training_step(self, batch: Any, batch_idx: int):
        return self._step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int):
        self._step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int):
        self._step(batch, "test")

    def on_train_epoch_end(self) -> None:
        """Compute and log training accuracy at the end of each epoch."""
        train_acc = self.train_acc.compute()
        self.log("train/acc", train_acc, prog_bar=True)
        self.train_acc.reset()
        self.train_acc_history.append(float(train_acc))
        self.epoch_history.append(self.current_epoch)

    def on_validation_epoch_end(self) -> None:
        val_acc = self.val_acc.compute()
        self.log("val/acc", val_acc, prog_bar=True)
        self.val_acc.reset()
        self.val_acc_history.append(float(val_acc))

    def on_test_epoch_end(self) -> None:
        test_acc = self.test_acc.compute()
        test_f1 = self.test_f1.compute()
        test_auroc = self.test_auroc.compute()
        test_confmat = self.test_confmat.compute()
        test_class_acc = self.test_class_acc.compute()
        self.log("test/acc", test_acc, prog_bar=True)
        self.log("test/f1", test_f1, prog_bar=True)
        self.log("test/auroc", test_auroc, prog_bar=True)
        confmat = test_confmat.cpu().numpy()
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(confmat, annot=False, fmt="d", ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        if hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "log_figure"):
            self.logger.experiment.log_figure(self.logger.run_id, fig, "confmat.png")
        plt.close(fig)

        # Plot per-class accuracy
        class_acc = test_class_acc.cpu().numpy()
        class_indices = list(range(len(class_acc)))
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=class_indices, y=class_acc, ax=ax)
        ax.set_xlabel("Class Index")
        ax.set_ylabel("Accuracy")
        ax.set_title("Per-Class Accuracy")
        ax.set_ylim(0, 1)
        if hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "log_figure"):
            self.logger.experiment.log_figure(self.logger.run_id, fig, "per_class_accuracy.png")
        plt.close(fig)
        
        
        # Save confusion matrix and per-class accuracy
        test_confmat_path = None
        test_class_acc_path = None

        if hasattr(self.logger, "log_dir") and self.logger.log_dir is not None:
            # For loggers with log_dir (e.g., TensorBoardLogger)
            os.makedirs(self.logger.log_dir, exist_ok=True)
            test_confmat_path = os.path.join(self.logger.log_dir, "test_confmat.pt")
            test_class_acc_path = os.path.join(self.logger.log_dir, "test_class_acc.pt")
            torch.save(test_confmat.cpu(), test_confmat_path)
            torch.save(test_class_acc.cpu(), test_class_acc_path)
        elif hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "log_artifact"):
            # For MLflowLogger: save to temp dir, then log as artifact
            with tempfile.TemporaryDirectory() as tmpdir:
                test_confmat_path = os.path.join(tmpdir, "test_confmat.pt")
                test_class_acc_path = os.path.join(tmpdir, "test_class_acc.pt")
                torch.save(test_confmat.cpu(), test_confmat_path)
                torch.save(test_class_acc.cpu(), test_class_acc_path)
                self.logger.experiment.log_artifact(self.logger.run_id, test_confmat_path)
                self.logger.experiment.log_artifact(self.logger.run_id, test_class_acc_path)
        else:
            print("Warning: Could not determine logger directory or artifact method. Not saving confusion matrix/class accuracy tensors.")

        self.test_acc.reset()
        self.test_f1.reset()
        self.test_auroc.reset()
        self.test_confmat.reset()
        self.test_class_acc.reset()

    def on_fit_end(self) -> None:
        # Plot train/val accuracy per epoch
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(self.train_acc_history)), self.train_acc_history, label="Train Acc")
        ax.plot(range(len(self.val_acc_history)), self.val_acc_history, label="Val Acc")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Train/Val Accuracy per Epoch")
        ax.set_ylim(0, 1)
        ax.legend()
        if hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "log_figure"):
            self.logger.experiment.log_figure(self.logger.run_id, fig, "train_val_accuracy.png")
        plt.close(fig)

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
