#!/usr/bin/env python
"""
Debug version of Optuna optimization script with progress bars and detailed logging.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.optimization import OptunaTrainer, HyperparameterSpace
from src.optimization.optuna_trainer import OptunaTrainer as BaseOptunaTrainer

import lightning.pytorch as pl
import optuna
from lightning.pytorch.loggers import MLFlowLogger


class DebugOptunaTrainer(BaseOptunaTrainer):
    """Debug version of OptunaTrainer with progress bars enabled."""
    
    def _create_trainer(
        self, 
        config: DictConfig, 
        pruning_callback, 
        trial: optuna.Trial
    ) -> pl.Trainer:
        """Create trainer with progress bars enabled for debugging."""
        # Create MLflow logger with trial information
        logger = MLFlowLogger(
            experiment_name=self.mlflow_experiment_name,
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
            tags={
                "optuna_study": self.study_manager.study_name,
                "optuna_trial": trial.number,
            }
        )
        
        # Build callbacks - keep progress bars for debugging
        from src.training.callbacks import build_callbacks
        from lightning.pytorch.callbacks import ModelCheckpoint
        
        all_callbacks = build_callbacks(config)
        
        # Only filter out checkpoint callbacks (keep progress bars)
        callbacks = [cb for cb in all_callbacks if not isinstance(cb, ModelCheckpoint)]
        callbacks.append(pruning_callback)
        
        # Create trainer configuration with progress bars enabled
        trainer_cfg = dict(config.trainer)
        trainer_cfg.update({
            "logger": logger,
            "callbacks": callbacks,
            "enable_checkpointing": False,  # Disable checkpointing for optimization
            "enable_progress_bar": True,    # Enable progress bars for debugging
            "log_every_n_steps": 10,       # Log more frequently
        })
        
        return pl.Trainer(**trainer_cfg)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="optimization",
)
def debug_optimize(cfg: DictConfig) -> None:
    """Debug optimization with progress bars."""
    print("🐛 Starting DEBUG Optuna Optimization")
    print("=" * 60)
    
    # Get Optuna configuration from main config
    if "optuna" not in cfg:
        raise KeyError("Optuna configuration not found in config.")
    
    optuna_cfg = cfg.optuna
    
    print(f"📊 Optimization Configuration:")
    print(f"   Study name: {optuna_cfg.study_name}")
    print(f"   Number of trials: {optuna_cfg.n_trials}")
    print(f"   Direction: {optuna_cfg.direction}")
    print(f"   Storage: {optuna_cfg.storage_path}")
    
    # Load hyperparameter space configurations (modular)
    hyperparameter_space = HyperparameterSpace.from_model_config(cfg)
    
    # Create DEBUG OptunaTrainer with progress bars
    from src.optimization.study_manager import StudyManager
    study_manager = StudyManager.from_config(optuna_cfg)
    
    trainer = DebugOptunaTrainer(
        config=cfg,
        study_manager=study_manager,
        hyperparameter_space=hyperparameter_space,
        n_trials=optuna_cfg.get("n_trials", 100),
        timeout=optuna_cfg.get("timeout", None),
        mlflow_experiment_name=optuna_cfg.get("mlflow_experiment_name", "debug_optuna_optimization"),
    )
    
    print("\n🐛 DEBUG MODE: Progress bars enabled, detailed logging")
    print("=" * 60)
    
    # Run optimization
    study = trainer.optimize()
    
    print("\n✅ Debug optimization completed!")


if __name__ == "__main__":
    debug_optimize() 