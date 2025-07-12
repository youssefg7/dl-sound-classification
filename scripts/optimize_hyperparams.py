#!/usr/bin/env python
"""
Optuna hyperparameter optimization script.

This script runs hyperparameter optimization using Optuna with TPE sampling
and Hyperband pruning for any model/dataset combination.

Examples:
- Run basic optimization:
  ```bash
  python scripts/optimize_hyperparams.py
  ```

- Run with custom settings:
  ```bash
  python scripts/optimize_hyperparams.py \
    optuna.n_trials=200 \
    optuna.study_name="custom_study" \
    dataset=esc50 \
    model=envnet_v2
  ```

- Resume existing study:
  ```bash
  python scripts/optimize_hyperparams.py \
    optuna.study_name="existing_study" \
    optuna.n_trials=50
  ```
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


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="optimization",
)
def optimize_hyperparams(cfg: DictConfig) -> None:
    """
    Main optimization function.
    
    Args:
        cfg: Hydra configuration object
    """
    print("🎯 Starting Optuna Hyperparameter Optimization")
    print("=" * 60)
    
    # Get Optuna configuration from main config
    if "optuna" not in cfg:
        raise KeyError("Optuna configuration not found in config. Make sure you're using the optimization config.")
    
    optuna_cfg = cfg.optuna
    
    # Ensure optuna_cfg is a DictConfig
    if not isinstance(optuna_cfg, DictConfig):
        raise ValueError(f"Expected DictConfig, got {type(optuna_cfg)}")
    
    print(f"📊 Optimization Configuration:")
    print(f"   Study name: {optuna_cfg.study_name}")
    print(f"   Number of trials: {optuna_cfg.n_trials}")
    print(f"   Direction: {optuna_cfg.direction}")
    print(f"   Storage: {optuna_cfg.storage_path}")
    
    # Load hyperparameter space configurations
    hp_space_dir = Path("configs/optimization/hyperparameter_spaces")
    hp_space_configs = []
    
    for hp_file in hp_space_dir.glob("*.yaml"):
        print(f"   Loading HP space: {hp_file.name}")
        hp_config = OmegaConf.load(hp_file)
        hp_space_configs.append(hp_config)
    
    if not hp_space_configs:
        raise ValueError(f"No hyperparameter space configs found in {hp_space_dir}")
    
    # Create OptunaTrainer
    trainer = OptunaTrainer.from_config(
        config=cfg,
        optuna_config=optuna_cfg,
        hyperparameter_spaces=hp_space_configs,
    )
    
    # Run optimization
    study = trainer.optimize()
    
    # Save best configuration
    output_dir = Path(optuna_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_config_path = output_dir / optuna_cfg.best_config_path
    trainer.save_best_config(str(best_config_path))
    
    # Print study summary
    print("\n" + "=" * 60)
    print("📈 STUDY SUMMARY")
    print("=" * 60)
    
    study_summary = trainer.study_manager.get_study_summary()
    for key, value in study_summary.items():
        print(f"   {key}: {value}")
    
    print(f"\n💾 Results saved to: {output_dir}")
    print(f"📊 MLflow experiment: {optuna_cfg.mlflow_experiment_name}")
    print(f"🗃️  Study database: {optuna_cfg.storage_path}")
    
    print("\n✅ Optimization completed successfully!")


if __name__ == "__main__":
    optimize_hyperparams() 