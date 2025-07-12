"""
Main Optuna trainer for hyperparameter optimization.

This module provides the main interface for running Optuna optimization studies
with PyTorch Lightning, integrating study management, hyperparameter spaces,
and pruning callbacks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Callable

import lightning.pytorch as pl
import optuna
from optuna.trial import FrozenTrial
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from .study_manager import StudyManager
from .hyperparameter_space import HyperparameterSpace
from .pruning_callbacks import create_pruning_callback


class OptunaTrainer:
    """
    Main trainer for Optuna hyperparameter optimization.
    
    This class orchestrates the entire optimization process, including
    study management, hyperparameter space definition, trial execution,
    and result analysis.
    
    Args:
        config: Base configuration for training
        study_manager: Study management instance
        hyperparameter_space: Hyperparameter space definition
        objective_fn: Function to build and train the model
        n_trials: Number of trials to run
        timeout: Timeout for the optimization (in seconds)
        mlflow_experiment_name: Name for MLflow experiment
    """
    
    def __init__(
        self,
        config: DictConfig,
        study_manager: StudyManager,
        hyperparameter_space: HyperparameterSpace,
        objective_fn: Optional[Callable] = None,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        mlflow_experiment_name: str = "optuna_optimization",
    ):
        self.config = config
        self.study_manager = study_manager
        self.hyperparameter_space = hyperparameter_space
        self.objective_fn = objective_fn or self._default_objective
        self.n_trials = n_trials
        self.timeout = timeout
        self.mlflow_experiment_name = mlflow_experiment_name
        
        # Optimize tensor core performance on H100/A100 GPUs
        try:
            import torch
            torch.set_float32_matmul_precision('medium')
        except Exception:
            pass  # Ignore if not available
        
        # Track optimization state
        self.best_trial = None
        self.best_value = None
        self.completed_trials = 0
        self.pruned_trials = 0
        
    def optimize(self) -> optuna.Study:
        """
        Run the optimization process.
        
        Returns:
            Completed Optuna study object
        """
        print("🚀 Starting Optuna optimization...")
        print(f"   Study name: {self.study_manager.study_name}")
        print(f"   Number of trials: {self.n_trials}")
        print(f"   Timeout: {self.timeout}s" if self.timeout else "   No timeout")
        
        # Print hyperparameter space summary
        self.hyperparameter_space.print_space_summary()
        
        # Get or create the study
        study = self.study_manager.study
        
        # Run optimization
        study.optimize(
            self.objective_fn,
            n_trials=self.n_trials,
            timeout=self.timeout,
            callbacks=[self._trial_callback],
        )
        
        # Update final statistics
        self._update_statistics(study)
        
        # Print optimization summary
        self._print_optimization_summary(study)
        
        return study
    
    def _default_objective(self, trial: optuna.Trial) -> float:
        """
        Default objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Objective value to optimize
        """
        # Get suggested hyperparameters
        suggested_params = self.hyperparameter_space.suggest_parameters(trial)
        
        # Update configuration with suggested parameters
        trial_config = self.hyperparameter_space.update_config_with_params(
            self.config, suggested_params
        )
        
        # Build model and data module
        model, datamodule = self._build_model_and_data(trial_config)
        
        # Create pruning callback
        pruning_callback = create_pruning_callback(
            trial=trial,
            monitor=trial_config.get("optuna", {}).get("monitor", "val/acc"),
            mode=trial_config.get("optuna", {}).get("mode", "max"),
            callback_type="hyperband",
            min_epochs=trial_config.get("optuna", {}).get("min_epochs", 10),
        )
        
        # Create trainer with pruning callback
        trainer = self._create_trainer(trial_config, pruning_callback, trial)
        
        # Train the model
        try:
            trainer.fit(model, datamodule=datamodule)
        except optuna.TrialPruned:
            # Handle pruned trials
            self.pruned_trials += 1
            raise
        
        # Get final metric value
        metric_name = trial_config.get("optuna", {}).get("monitor", "val/acc")
        final_value = self._get_final_metric_value(trainer, metric_name)
        
        if final_value is None:
            raise optuna.TrialPruned("Could not retrieve final metric value")
        
        return final_value
    
    def _build_model_and_data(self, config: DictConfig) -> tuple[pl.LightningModule, pl.LightningDataModule]:
        """
        Build model and data module from configuration.
        
        Args:
            config: Configuration with hyperparameters
            
        Returns:
            Tuple of (model, datamodule)
        """
        # Import here to avoid circular imports
        from src.training.engine import build_from_cfg
        from hydra.utils import instantiate
        
        # Build datamodule
        datamodule_cfg = {
            "_target_": config.dataset._target_,
            "root": config.dataset.root,
            "fold": config.dataset.fold,
            "val_split": config.dataset.val_split,
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "augment": config.dataset.augment,
        }
        datamodule = instantiate(datamodule_cfg)
        
        # Build model
        model = build_from_cfg(config)
        
        return model, datamodule
    
    def _create_trainer(
        self, 
        config: DictConfig, 
        pruning_callback: Any, 
        trial: optuna.Trial
    ) -> pl.Trainer:
        """
        Create PyTorch Lightning trainer with Optuna integration.
        
        Args:
            config: Configuration for training
            pruning_callback: Optuna pruning callback
            trial: Optuna trial object
            
        Returns:
            Configured PyTorch Lightning trainer
        """
        # Create MLflow logger with trial information
        logger = MLFlowLogger(
            experiment_name=self.mlflow_experiment_name,
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
            tags={
                "optuna_study": self.study_manager.study_name,
                "optuna_trial": trial.number,
            }
        )
        
        # Build callbacks (include pruning callback, but exclude checkpoint and progress bar callbacks)
        from src.training.callbacks import build_callbacks
        from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
        
        all_callbacks = build_callbacks(config)
        
        # Filter out checkpoint and progress bar callbacks since we disable them for optimization
        callbacks = [
            cb for cb in all_callbacks 
            if not isinstance(cb, (ModelCheckpoint, TQDMProgressBar))
        ]
        callbacks.append(pruning_callback)
        
        # Create trainer configuration
        trainer_cfg = dict(config.trainer)
        trainer_cfg.update({
            "logger": logger,
            "callbacks": callbacks,
            "enable_checkpointing": False,  # Disable checkpointing for optimization
            "enable_progress_bar": False,   # Reduce verbosity
        })
        
        # Handle multiprocessing cleanup gracefully
        import warnings
        warnings.filterwarnings("ignore", message=".*can only test a child process.*")
        
        return pl.Trainer(**trainer_cfg)
    
    def _get_final_metric_value(
        self, 
        trainer: pl.Trainer, 
        metric_name: str
    ) -> Optional[float]:
        """
        Extract final metric value from trainer.
        
        Args:
            trainer: PyTorch Lightning trainer
            metric_name: Name of the metric to extract
            
        Returns:
            Final metric value or None if not available
        """
        # Try different sources for the metric
        sources = [
            trainer.logged_metrics,
            getattr(trainer, 'callback_metrics', {}),
            getattr(trainer, 'progress_bar_metrics', {}),
        ]
        
        # Debug: print available metrics
        print(f"🔍 Looking for metric: {metric_name}")
        for i, source in enumerate(sources):
            print(f"  Source {i}: {list(source.keys())}")
        
        for source in sources:
            if metric_name in source:
                value = float(source[metric_name])
                print(f"✓ Found {metric_name} = {value}")
                return value
        
        # If not found, try to get from the model's validation metrics
        if hasattr(trainer, 'lightning_module'):
            if hasattr(trainer.lightning_module, 'val_acc'):
                try:
                    val_acc_obj = trainer.lightning_module.val_acc
                    val_acc = val_acc_obj.compute()
                    print(f"✓ Computed val_acc directly: {val_acc}")
                    return float(val_acc)
                except Exception as e:
                    print(f"⚠ Error computing val_acc: {e}")
        
        print(f"⚠ Could not find metric {metric_name}")
        return None
    
    def _trial_callback(
        self, 
        study: optuna.Study, 
        trial: FrozenTrial
    ) -> None:
        """
        Callback called after each trial completion.
        
        Args:
            study: Optuna study object
            trial: Completed trial object
        """
        self.completed_trials += 1
        
        # Update best trial if this is the best so far
        if study.best_trial and study.best_trial.number == trial.number:
            self.best_trial = trial
            self.best_value = trial.value
            
            print(f"🎯 New best trial #{trial.number}:")
            print(f"   Value: {trial.value:.4f}")
            print(f"   Params: {trial.params}")
    
    def _update_statistics(self, study: optuna.Study) -> None:
        """
        Update optimization statistics.
        
        Args:
            study: Completed study object
        """
        self.best_trial = study.best_trial
        self.best_value = study.best_value
        
        # Count trial states
        from collections import Counter
        trial_states = Counter(trial.state.name for trial in study.trials)
        self.completed_trials = trial_states.get("COMPLETE", 0)
        self.pruned_trials = trial_states.get("PRUNED", 0)
    
    def _print_optimization_summary(self, study: optuna.Study) -> None:
        """
        Print optimization summary.
        
        Args:
            study: Completed study object
        """
        print("\n" + "="*60)
        print("📊 OPTIMIZATION SUMMARY")
        print("="*60)
        
        print(f"Study name: {study.study_name}")
        print(f"Total trials: {len(study.trials)}")
        print(f"Completed trials: {self.completed_trials}")
        print(f"Pruned trials: {self.pruned_trials}")
        print(f"Failed trials: {len(study.trials) - self.completed_trials - self.pruned_trials}")
        
        if self.best_trial:
            print(f"\n🏆 Best trial: #{self.best_trial.number}")
            print(f"Best value: {self.best_value:.4f}")
            print(f"Best parameters:")
            for key, value in self.best_trial.params.items():
                print(f"  {key}: {value}")
        
        print("\n" + "="*60)
    
    def get_best_config(self) -> Optional[DictConfig]:
        """
        Get the best configuration from optimization.
        
        Returns:
            Best configuration or None if no successful trials
        """
        if not self.best_trial:
            return None
        
        # Update base configuration with best parameters
        best_config = self.hyperparameter_space.update_config_with_params(
            self.config, self.best_trial.params
        )
        
        return best_config
    
    def save_best_config(self, output_path: str) -> None:
        """
        Save the best configuration to a file.
        
        Args:
            output_path: Path to save the configuration
        """
        best_config = self.get_best_config()
        if best_config is None:
            print("⚠ No best configuration available to save")
            return
        
        # Save configuration
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        OmegaConf.save(best_config, output_path_obj)
        print(f"✓ Best configuration saved to: {output_path}")
    
    @classmethod
    def from_config(
        cls, 
        config: DictConfig, 
        optuna_config: DictConfig,
        hyperparameter_spaces: list[DictConfig],
    ) -> "OptunaTrainer":
        """
        Create OptunaTrainer from configuration files.
        
        Args:
            config: Base training configuration
            optuna_config: Optuna-specific configuration
            hyperparameter_spaces: List of hyperparameter space configurations
            
        Returns:
            OptunaTrainer instance
        """
        # Create study manager
        study_manager = StudyManager.from_config(optuna_config)
        
        # Create hyperparameter space
        hyperparameter_space = HyperparameterSpace.from_multiple_configs(
            hyperparameter_spaces
        )
        
        # Create trainer
        return cls(
            config=config,
            study_manager=study_manager,
            hyperparameter_space=hyperparameter_space,
            n_trials=optuna_config.get("n_trials", 100),
            timeout=optuna_config.get("timeout", None),
            mlflow_experiment_name=optuna_config.get("mlflow_experiment_name", "optuna_optimization"),
        ) 