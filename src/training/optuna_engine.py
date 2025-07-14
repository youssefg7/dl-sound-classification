"""
Optuna-aware Lightning module for hyperparameter optimization.

This module extends the base LitClassifier with Optuna-specific features
for hyperparameter optimization, including trial state management and
enhanced logging.
"""

from __future__ import annotations

from typing import Optional

import lightning.pytorch as pl
import optuna
from omegaconf import DictConfig

from .engine import LitClassifier


class OptunaLitClassifier(LitClassifier):
    """
    Optuna-aware Lightning classifier.
    
    This class extends the base LitClassifier with features specifically
    designed for Optuna hyperparameter optimization, including trial
    state management and enhanced logging.
    
    Args:
        model_cfg: Model configuration
        optim_cfg: Optimizer configuration
        sched_cfg: Scheduler configuration
        loss_cfg: Loss function configuration
        metric_cfg: Metric configuration
        trial: Optuna trial object (optional)
    """
    
    def __init__(
        self,
        model_cfg: DictConfig,
        optim_cfg: DictConfig,
        sched_cfg: DictConfig | None = None,
        loss_cfg: DictConfig | None = None,
        metric_cfg: DictConfig | None = None,
        trial: Optional[optuna.Trial] = None,
    ) -> None:
        super().__init__(model_cfg, optim_cfg, sched_cfg, loss_cfg, metric_cfg)
        
        # Store trial reference for logging
        self.trial = trial
        
        # Enhanced logging state
        self.trial_metrics = {}
        self.epoch_metrics = {}
        
        # Store trial number for logging
        self.trial_number = trial.number if trial else None
    
    def on_train_epoch_end(self) -> None:
        """Enhanced epoch end with trial-specific logging."""
        super().on_train_epoch_end()
        
        if self.trial is not None:
            # Log trial-specific metrics
            train_acc = self.train_acc.compute()
            self.trial_metrics[f"trial_{self.trial_number}/train_acc"] = train_acc
            
            # Log to MLflow with trial context
            if self.logger:
                self.logger.log_metrics(
                    {
                        f"trial_{self.trial_number}/train_acc": train_acc,
                        f"trial_{self.trial_number}/epoch": self.current_epoch,
                    },
                    step=self.current_epoch
                )
    
    def on_validation_epoch_end(self) -> None:
        """Enhanced validation epoch end with trial-specific logging."""
        super().on_validation_epoch_end()
        
        if self.trial is not None:
            # Log trial-specific metrics
            val_acc = self.val_acc.compute()
            self.trial_metrics[f"trial_{self.trial_number}/val_acc"] = val_acc
            
            # Log to MLflow with trial context
            if self.logger:
                self.logger.log_metrics(
                    {
                        f"trial_{self.trial_number}/val_acc": val_acc,
                        f"trial_{self.trial_number}/epoch": self.current_epoch,
                    },
                    step=self.current_epoch
                )
    
    def on_train_end(self) -> None:
        """Enhanced training end with trial summary."""
        if self.trial is not None and self.logger:
            # Log final trial summary
            final_metrics = {
                f"trial_{self.trial_number}/final_train_acc": self.train_acc.compute(),
                f"trial_{self.trial_number}/final_val_acc": self.val_acc.compute(),
                f"trial_{self.trial_number}/total_epochs": self.current_epoch,
            }
            
            self.logger.log_metrics(final_metrics, step=self.current_epoch)
    
    def log_hyperparameters(self, params: dict) -> None:
        """
        Log hyperparameters with trial context.
        
        Args:
            params: Hyperparameters to log
        """
        if self.trial is not None and self.logger:
            # Add trial context to hyperparameters
            trial_params = {
                f"trial_{self.trial_number}/{k}": v 
                for k, v in params.items()
            }
            
            # Add trial metadata
            trial_params.update({
                f"trial_{self.trial_number}/trial_number": self.trial_number,
            })
            
            self.logger.log_hyperparams(trial_params)
    
    def get_trial_summary(self) -> dict:
        """
        Get summary of trial metrics.
        
        Returns:
            Dictionary with trial summary
        """
        return {
            "trial_number": self.trial_number,
            "metrics": self.trial_metrics.copy(),
        }


def build_optuna_model(cfg: DictConfig, trial: Optional[optuna.Trial] = None) -> OptunaLitClassifier:
    """
    Build Optuna-aware model from configuration.
    
    Args:
        cfg: Configuration object
        trial: Optuna trial object (optional)
        
    Returns:
        OptunaLitClassifier instance
    """
    return OptunaLitClassifier(
        model_cfg=cfg.model,
        optim_cfg=cfg.optimizer,
        sched_cfg=cfg.scheduler,
        loss_cfg=cfg.get("loss"),
        metric_cfg=cfg.get("metric"),
        trial=trial,
    ) 