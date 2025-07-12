"""
Study management for Optuna optimization with SQLite persistence.

This module handles the creation, loading, and management of Optuna studies
with SQLite database storage for persistence across runs.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from omegaconf import DictConfig


class StudyManager:
    """
    Manages Optuna studies with SQLite persistence.
    
    Args:
        storage_path: Path to SQLite database file
        study_name: Name of the study
        direction: Optimization direction ('minimize' or 'maximize')
        sampler_config: Configuration for TPE sampler
        pruner_config: Configuration for Hyperband pruner
    """
    
    def __init__(
        self,
        storage_path: str = "optuna_studies.db",
        study_name: str = "default_study",
        direction: str = "maximize",
        sampler_config: Optional[Dict[str, Any]] = None,
        pruner_config: Optional[Dict[str, Any]] = None,
    ):
        self.storage_path = Path(storage_path)
        self.study_name = study_name
        self.direction = direction
        self.sampler_config = sampler_config or {}
        self.pruner_config = pruner_config or {}
        
        # Create storage directory if it doesn't exist
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create SQLite URL
        self.storage_url = f"sqlite:///{self.storage_path}"
        
        # Initialize components
        self._sampler = None
        self._pruner = None
        self._study = None
    
    @property
    def sampler(self) -> TPESampler:
        """Get or create TPE sampler."""
        if self._sampler is None:
            self._sampler = TPESampler(
                n_startup_trials=self.sampler_config.get("n_startup_trials", 10),
                n_ei_candidates=self.sampler_config.get("n_ei_candidates", 24),
                seed=self.sampler_config.get("seed", 42),
            )
        return self._sampler
    
    @property
    def pruner(self) -> HyperbandPruner:
        """Get or create Hyperband pruner."""
        if self._pruner is None:
            self._pruner = HyperbandPruner(
                min_resource=self.pruner_config.get("min_resource", 1),
                max_resource=self.pruner_config.get("max_resource", 100),
                reduction_factor=self.pruner_config.get("reduction_factor", 3),
            )
        return self._pruner
    
    @property
    def study(self) -> optuna.Study:
        """Get or create Optuna study."""
        if self._study is None:
            self._study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage_url,
                direction=self.direction,
                sampler=self.sampler,
                pruner=self.pruner,
                load_if_exists=True,
            )
        return self._study
    
    def create_study(
        self,
        study_name: Optional[str] = None,
        direction: Optional[str] = None,
        overwrite: bool = False,
    ) -> optuna.Study:
        """
        Create a new study or load existing one.
        
        Args:
            study_name: Name for the study (uses default if None)
            direction: Optimization direction (uses default if None)
            overwrite: Whether to delete existing study with same name
            
        Returns:
            Optuna study object
        """
        name = study_name or self.study_name
        dir_val = direction or self.direction
        
        if overwrite:
            try:
                optuna.delete_study(study_name=name, storage=self.storage_url)
                print(f"✓ Deleted existing study: {name}")
            except Exception as e:
                print(f"⚠ Could not delete study {name}: {e}")
        
        study = optuna.create_study(
            study_name=name,
            storage=self.storage_url,
            direction=dir_val,
            sampler=self.sampler,
            pruner=self.pruner,
            load_if_exists=True,
        )
        
        print(f"✓ Created/loaded study: {name}")
        print(f"  - Direction: {dir_val}")
        print(f"  - Storage: {self.storage_url}")
        print(f"  - Sampler: {type(self.sampler).__name__}")
        print(f"  - Pruner: {type(self.pruner).__name__}")
        
        return study
    
    def load_study(self, study_name: str) -> optuna.Study:
        """
        Load an existing study.
        
        Args:
            study_name: Name of the study to load
            
        Returns:
            Optuna study object
            
        Raises:
            ValueError: If study doesn't exist
        """
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=self.storage_url,
                sampler=self.sampler,
                pruner=self.pruner,
            )
            print(f"✓ Loaded study: {study_name}")
            return study
        except Exception as e:
            raise ValueError(f"Could not load study {study_name}: {e}")
    
    def get_study_summary(self, study_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary information about a study.
        
        Args:
            study_name: Name of the study (uses default if None)
            
        Returns:
            Dictionary with study summary information
        """
        study = self.load_study(study_name or self.study_name)
        
        summary = {
            "study_name": study.study_name,
            "direction": study.direction.name,
            "n_trials": len(study.trials),
            "best_value": study.best_value if study.best_trial else None,
            "best_params": study.best_params if study.best_trial else None,
            "best_trial_number": study.best_trial.number if study.best_trial else None,
        }
        
        # Add trial state counts
        from collections import Counter
        trial_states = Counter(trial.state.name for trial in study.trials)
        summary["trial_states"] = dict(trial_states)
        
        return summary
    
    def list_studies(self) -> list[str]:
        """
        List all studies in the storage.
        
        Returns:
            List of study names
        """
        if not self.storage_path.exists():
            return []
        
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            cursor.execute("SELECT study_name FROM studies")
            studies = [row[0] for row in cursor.fetchall()]
            conn.close()
            return studies
        except Exception as e:
            print(f"⚠ Could not list studies: {e}")
            return []
    
    def delete_study(self, study_name: str) -> bool:
        """
        Delete a study from storage.
        
        Args:
            study_name: Name of the study to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            optuna.delete_study(study_name=study_name, storage=self.storage_url)
            print(f"✓ Deleted study: {study_name}")
            return True
        except Exception as e:
            print(f"⚠ Could not delete study {study_name}: {e}")
            return False
    
    @classmethod
    def from_config(cls, cfg: DictConfig) -> "StudyManager":
        """
        Create StudyManager from Hydra configuration.
        
        Args:
            cfg: Hydra configuration object
            
        Returns:
            StudyManager instance
        """
        return cls(
            storage_path=cfg.get("storage_path", "optuna_studies.db"),
            study_name=cfg.get("study_name", "default_study"),
            direction=cfg.get("direction", "maximize"),
            sampler_config=cfg.get("sampler", {}),
            pruner_config=cfg.get("pruner", {}),
        ) 