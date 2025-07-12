"""
Optuna pruning callbacks for PyTorch Lightning.

This module provides callbacks that integrate Optuna's pruning functionality
with PyTorch Lightning training loops, enabling early stopping of unpromising trials.
"""

from __future__ import annotations

from typing import Any, Optional

import lightning.pytorch as pl
import optuna
from lightning.pytorch.callbacks import Callback


class OptunaPruningCallback(Callback):
    """
    PyTorch Lightning callback for Optuna pruning.
    
    This callback integrates Optuna's pruning functionality with PyTorch Lightning,
    allowing trials to be pruned early if they show poor performance.
    
    Args:
        trial: Optuna trial object
        monitor: Metric to monitor for pruning decisions
        mode: Whether to minimize or maximize the monitored metric
        patience: Number of epochs to wait before pruning after a plateau
        min_epochs: Minimum number of epochs before pruning is allowed
    """
    
    def __init__(
        self,
        trial: optuna.Trial,
        monitor: str = "val/acc",
        mode: str = "max",
        patience: int = 3,
        min_epochs: int = 10,
    ):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_epochs = min_epochs
        
        # Track monitoring state
        self.best_value = None
        self.epochs_without_improvement = 0
        self.pruned = False
        
        # Determine comparison function based on mode
        if mode == "min":
            self.is_better = lambda new, best: new < best
        else:
            self.is_better = lambda new, best: new > best
    
    def on_validation_epoch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
    ) -> None:
        """
        Called when the validation epoch ends.
        
        Reports the current metric value to Optuna and checks if the trial
        should be pruned.
        """
        # Get the current epoch
        current_epoch = trainer.current_epoch
        
        # Skip pruning if we're in the minimum epochs period
        if current_epoch < self.min_epochs:
            return
        
        # Get the monitored metric value
        metric_value = self._get_metric_value(trainer)
        if metric_value is None:
            return
        
        # Report intermediate value to Optuna
        self.trial.report(metric_value, current_epoch)
        
        # Check if trial should be pruned
        if self.trial.should_prune():
            self.pruned = True
            message = f"Trial was pruned at epoch {current_epoch}"
            raise optuna.TrialPruned(message)
        
        # Update best value and patience counter
        self._update_best_value(metric_value)
    
    def _get_metric_value(self, trainer: pl.Trainer) -> Optional[float]:
        """
        Extract the monitored metric value from the trainer.
        
        Args:
            trainer: PyTorch Lightning trainer
            
        Returns:
            Current metric value or None if not available
        """
        # Get logged metrics
        logged_metrics = trainer.logged_metrics
        
        if self.monitor in logged_metrics:
            return float(logged_metrics[self.monitor])
        
        # Try callback metrics
        if hasattr(trainer, 'callback_metrics'):
            callback_metrics = trainer.callback_metrics
            if self.monitor in callback_metrics:
                return float(callback_metrics[self.monitor])
        
        # Try progress bar metrics
        if hasattr(trainer, 'progress_bar_metrics'):
            progress_metrics = trainer.progress_bar_metrics
            if self.monitor in progress_metrics:
                return float(progress_metrics[self.monitor])
        
        return None
    
    def _update_best_value(self, current_value: float) -> None:
        """
        Update the best value and patience counter.
        
        Args:
            current_value: Current metric value
        """
        if self.best_value is None:
            self.best_value = current_value
            self.epochs_without_improvement = 0
        elif self.is_better(current_value, self.best_value):
            self.best_value = current_value
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called when training ends.
        
        Reports the final metric value to Optuna.
        """
        if not self.pruned:
            # Report final value
            final_value = self._get_metric_value(trainer)
            if final_value is not None:
                self.trial.report(final_value, trainer.current_epoch)


class HyperbandPruningCallback(OptunaPruningCallback):
    """
    Enhanced pruning callback with Hyperband-specific features.
    
    This callback extends the base pruning callback with features
    specifically designed for Hyperband pruning.
    
    Args:
        trial: Optuna trial object
        monitor: Metric to monitor for pruning decisions
        mode: Whether to minimize or maximize the monitored metric
        patience: Number of epochs to wait before pruning after a plateau
        min_epochs: Minimum number of epochs before pruning is allowed
        resource_attr: Name of the resource attribute (usually 'epoch')
        enable_intermediate_logging: Whether to log intermediate values
    """
    
    def __init__(
        self,
        trial: optuna.Trial,
        monitor: str = "val/acc",
        mode: str = "max",
        patience: int = 3,
        min_epochs: int = 10,
        resource_attr: str = "epoch",
        enable_intermediate_logging: bool = True,
    ):
        super().__init__(trial, monitor, mode, patience, min_epochs)
        self.resource_attr = resource_attr
        self.enable_intermediate_logging = enable_intermediate_logging
        
        # Track resource usage
        self.resource_values = []
        self.metric_values = []
    
    def on_validation_epoch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
    ) -> None:
        """
        Called when the validation epoch ends.
        
        Enhanced version with resource tracking and intermediate logging.
        """
        current_epoch = trainer.current_epoch
        
        # Skip pruning if we're in the minimum epochs period
        if current_epoch < self.min_epochs:
            return
        
        # Get the monitored metric value
        metric_value = self._get_metric_value(trainer)
        if metric_value is None:
            return
        
        # Track resource and metric values
        self.resource_values.append(current_epoch)
        self.metric_values.append(metric_value)
        
        # Report intermediate value to Optuna
        self.trial.report(metric_value, current_epoch)
        
        # Log intermediate values if enabled
        if self.enable_intermediate_logging:
            self._log_intermediate_value(trainer, metric_value, current_epoch)
        
        # Check if trial should be pruned
        if self.trial.should_prune():
            self.pruned = True
            message = (
                f"Trial was pruned at epoch {current_epoch} "
                f"(resource={current_epoch}, value={metric_value:.4f})"
            )
            raise optuna.TrialPruned(message)
        
        # Update best value and patience counter
        self._update_best_value(metric_value)
    
    def _log_intermediate_value(
        self, 
        trainer: pl.Trainer, 
        metric_value: float, 
        epoch: int
    ) -> None:
        """
        Log intermediate values for monitoring.
        
        Args:
            trainer: PyTorch Lightning trainer
            metric_value: Current metric value
            epoch: Current epoch
        """
        # Log to trainer's logger if available
        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {
                    f"optuna/intermediate_value": metric_value,
                    f"optuna/trial_number": self.trial.number,
                    f"optuna/resource": epoch,
                },
                step=epoch
            )
    
    def get_resource_efficiency(self) -> float:
        """
        Calculate resource efficiency score.
        
        Returns:
            Resource efficiency score (performance per unit resource)
        """
        if not self.metric_values or not self.resource_values:
            return 0.0
        
        # Calculate efficiency as final performance / total resource used
        final_performance = self.metric_values[-1]
        total_resource = self.resource_values[-1]
        
        return final_performance / max(total_resource, 1)
    
    def get_learning_curve(self) -> tuple[list[int], list[float]]:
        """
        Get the learning curve data.
        
        Returns:
            Tuple of (resource_values, metric_values)
        """
        return self.resource_values.copy(), self.metric_values.copy()


def create_pruning_callback(
    trial: optuna.Trial,
    monitor: str = "val/acc",
    mode: str = "max",
    callback_type: str = "hyperband",
    **kwargs
) -> OptunaPruningCallback:
    """
    Factory function to create pruning callbacks.
    
    Args:
        trial: Optuna trial object
        monitor: Metric to monitor for pruning decisions
        mode: Whether to minimize or maximize the monitored metric
        callback_type: Type of pruning callback ('basic' or 'hyperband')
        **kwargs: Additional arguments for the callback
        
    Returns:
        Pruning callback instance
    """
    if callback_type == "hyperband":
        return HyperbandPruningCallback(
            trial=trial,
            monitor=monitor,
            mode=mode,
            **kwargs
        )
    else:
        return OptunaPruningCallback(
            trial=trial,
            monitor=monitor,
            mode=mode,
            **kwargs
        ) 