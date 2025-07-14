"""
Hyperparameter space management for Optuna optimization.

This module provides utilities for defining and managing hyperparameter spaces
from configuration files, supporting various parameter types and distributions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from enum import Enum

import optuna
from omegaconf import DictConfig, OmegaConf


class ParameterType(str, Enum):
    """Supported parameter types for hyperparameter spaces."""
    FLOAT = "float"
    INT = "int"
    CATEGORICAL = "categorical"
    DISCRETE = "discrete"


class HyperparameterSpace:
    """
    Manages hyperparameter spaces for Optuna optimization.
    
    This class handles the definition and sampling of hyperparameter spaces
    from configuration files, supporting various parameter types and distributions.
    """
    
    def __init__(self, config: Optional[DictConfig] = None):
        """
        Initialize hyperparameter space.
        
        Args:
            config: Configuration containing hyperparameter space definitions
        """
        self.config = config or DictConfig({})
        self.space_definitions = {}
        self._parse_config()
    
    def _parse_config(self) -> None:
        """Parse configuration and extract hyperparameter space definitions."""
        if not self.config:
            return
        
        # Recursively parse the configuration
        self.space_definitions = self._parse_section(self.config)
    
    def _parse_section(self, section: DictConfig, prefix: str = "") -> Dict[str, Any]:
        """
        Recursively parse a configuration section.
        
        Args:
            section: Configuration section to parse
            prefix: Prefix for nested parameter names
            
        Returns:
            Dictionary of parameter definitions
        """
        definitions = {}
        
        for key, value in section.items():
            param_name = f"{prefix}.{str(key)}" if prefix else str(key)
            
            if isinstance(value, DictConfig):
                # Check if this is a parameter definition
                if "type" in value:
                    definitions[param_name] = self._parse_parameter(value)
                else:
                    # Recursively parse nested sections
                    nested_defs = self._parse_section(value, param_name)
                    definitions.update(nested_defs)
            
        return definitions
    
    def _parse_parameter(self, param_config: DictConfig) -> Dict[str, Any]:
        """
        Parse a single parameter definition.
        
        Args:
            param_config: Configuration for a single parameter
            
        Returns:
            Parameter definition dictionary
        """
        param_type = param_config.get("type", "float")
        
        if param_type == ParameterType.FLOAT:
            return {
                "type": "float",
                "low": param_config.get("low", 0.0),
                "high": param_config.get("high", 1.0),
                "log": param_config.get("log", False),
                "step": param_config.get("step", None),
            }
        
        elif param_type == ParameterType.INT:
            return {
                "type": "int",
                "low": param_config.get("low", 0),
                "high": param_config.get("high", 100),
                "log": param_config.get("log", False),
                "step": param_config.get("step", 1),
            }
        
        elif param_type == ParameterType.CATEGORICAL:
            return {
                "type": "categorical",
                "choices": param_config.get("choices", []),
            }
        
        elif param_type == ParameterType.DISCRETE:
            return {
                "type": "discrete",
                "choices": param_config.get("choices", []),
            }
        
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")
    
    def suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        params = {}
        
        for param_name, param_def in self.space_definitions.items():
            param_type = param_def["type"]
            
            if param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    low=param_def["low"],
                    high=param_def["high"],
                    log=param_def.get("log", False),
                    step=param_def.get("step", None),
                )
            
            elif param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    low=param_def["low"],
                    high=param_def["high"],
                    log=param_def.get("log", False),
                    step=param_def.get("step", 1),
                )
            
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    choices=param_def["choices"],
                )
            
            elif param_type == "discrete":
                params[param_name] = trial.suggest_discrete_uniform(
                    param_name,
                    low=min(param_def["choices"]),
                    high=max(param_def["choices"]),
                    q=param_def["choices"][1] - param_def["choices"][0],
                )
        
        return params
    
    def update_config_with_params(
        self, 
        base_config: DictConfig, 
        params: Dict[str, Any]
    ) -> DictConfig:
        """
        Update a base configuration with suggested parameters.
        
        Args:
            base_config: Base configuration to update
            params: Dictionary of suggested parameters
            
        Returns:
            Updated configuration
        """
        # Create a deep copy of the base config
        updated_config = OmegaConf.create(OmegaConf.to_yaml(base_config))
        
        # Update with suggested parameters using OmegaConf.update
        for param_name, value in params.items():
            # Use OmegaConf structured update
            OmegaConf.update(updated_config, param_name, value, merge=True)
        
        # Ensure we return a DictConfig
        if not isinstance(updated_config, DictConfig):
            raise ValueError(f"Expected DictConfig, got {type(updated_config)}")
        return updated_config
    
    def get_parameter_names(self) -> List[str]:
        """
        Get list of all parameter names in the space.
        
        Returns:
            List of parameter names
        """
        return list(self.space_definitions.keys())
    
    def get_parameter_info(self, param_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific parameter.
        
        Args:
            param_name: Name of the parameter
            
        Returns:
            Parameter definition or None if not found
        """
        return self.space_definitions.get(param_name)
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Validate that parameters are within defined bounds.
        
        Args:
            params: Dictionary of parameters to validate
            
        Returns:
            True if all parameters are valid, False otherwise
        """
        for param_name, value in params.items():
            if param_name not in self.space_definitions:
                print(f"⚠ Unknown parameter: {param_name}")
                return False
            
            param_def = self.space_definitions[param_name]
            param_type = param_def["type"]
            
            if param_type in ["float", "int"]:
                if not (param_def["low"] <= value <= param_def["high"]):
                    print(f"⚠ Parameter {param_name} out of bounds: {value}")
                    return False
            
            elif param_type == "categorical":
                if value not in param_def["choices"]:
                    print(f"⚠ Parameter {param_name} not in choices: {value}")
                    return False
        
        return True
    
    def print_space_summary(self) -> None:
        """Print a summary of the hyperparameter space."""
        print("=== Hyperparameter Space Summary ===")
        print(f"Total parameters: {len(self.space_definitions)}")
        print()
        
        for param_name, param_def in self.space_definitions.items():
            param_type = param_def["type"]
            print(f"• {param_name} ({param_type})")
            
            if param_type == "float":
                log_str = " (log)" if param_def.get("log", False) else ""
                step_str = f", step={param_def['step']}" if param_def.get("step") else ""
                print(f"  Range: [{param_def['low']}, {param_def['high']}]{log_str}{step_str}")
            
            elif param_type == "int":
                log_str = " (log)" if param_def.get("log", False) else ""
                step_str = f", step={param_def['step']}" if param_def.get("step") else ""
                print(f"  Range: [{param_def['low']}, {param_def['high']}]{log_str}{step_str}")
            
            elif param_type == "categorical":
                choices_str = ", ".join(map(str, param_def["choices"]))
                print(f"  Choices: {choices_str}")
            
            elif param_type == "discrete":
                choices_str = ", ".join(map(str, param_def["choices"]))
                print(f"  Values: {choices_str}")
            
            print()
    
    @classmethod
    def from_config_file(cls, config_path: str) -> "HyperparameterSpace":
        """
        Create HyperparameterSpace from a configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            HyperparameterSpace instance
        """
        config = OmegaConf.load(config_path)
        if not isinstance(config, DictConfig):
            raise ValueError(f"Expected DictConfig, got {type(config)}")
        return cls(config)
    
    @classmethod
    def from_multiple_configs(cls, configs: List[DictConfig]) -> "HyperparameterSpace":
        """
        Create HyperparameterSpace from multiple configuration objects.
        
        Args:
            configs: List of configuration objects to merge
            
        Returns:
            HyperparameterSpace instance
        """
        merged_config = OmegaConf.create({})
        for config in configs:
            merged_config = OmegaConf.merge(merged_config, config)
        
        if not isinstance(merged_config, DictConfig):
            raise ValueError(f"Expected DictConfig, got {type(merged_config)}")
        return cls(merged_config)

    @classmethod
    def from_model_config(cls, model_config: DictConfig, hp_spaces_dir: str = "configs/optimization/hyperparameter_spaces") -> "HyperparameterSpace":
        """
        Create HyperparameterSpace by automatically loading relevant configs based on model.
        
        This method implements a modular hierarchy:
        - Generic spaces: training.yaml, loss.yaml (always loaded)
        - Model-specific: models/{model_name}.yaml (loaded based on model._target_)
        
        Args:
            model_config: Full model configuration containing model._target_
            hp_spaces_dir: Directory containing hyperparameter space configs
            
        Returns:
            HyperparameterSpace instance with only relevant parameters
            
        Raises:
            ValueError: If no relevant hyperparameter spaces found
        """
        from pathlib import Path
        
        hp_space_dir = Path(hp_spaces_dir)
        configs = []
        
        # 1. Extract model name from model._target_ field
        model_name = None
        if hasattr(model_config, 'model') and hasattr(model_config.model, '_target_'):
            model_target = model_config.model._target_
            # Extract model name from target path (e.g., "src.models.envnet_v2.EnvNetV2" -> "envnet_v2")
            if '.' in model_target:
                # Split by dots and find the model name (usually the second-to-last part)
                parts = model_target.split('.')
                # Look for common patterns: src.models.{model_name}.{ClassName}
                if len(parts) >= 3 and parts[0] == 'src' and parts[1] == 'models':
                    model_name = parts[2]
                else:
                    # Fallback: use the last part before the class name
                    model_name = parts[-2] if len(parts) >= 2 else parts[-1]
            else:
                model_name = model_target
        
        print(f"🔍 Model detected: {model_name}")
        
        # 2. Load generic hyperparameter spaces (always loaded)
        generic_files = ['training.yaml', 'loss.yaml']
        for file_name in generic_files:
            file_path = hp_space_dir / file_name
            if file_path.exists():
                print(f"   ✓ Loading generic: {file_name}")
                config = OmegaConf.load(file_path)
                configs.append(config)
            else:
                print(f"   ⚠ Generic file not found: {file_name}")
        
        # 3. Load model-specific hyperparameter space
        if model_name:
            model_file = hp_space_dir / "models" / f"{model_name}.yaml"
            if model_file.exists():
                print(f"   ✓ Loading model-specific: models/{model_name}.yaml")
                config = OmegaConf.load(model_file)
                configs.append(config)
            else:
                print(f"   ⚠ Model-specific file not found: models/{model_name}.yaml")
                available_models = [f.stem for f in (hp_space_dir / 'models').glob('*.yaml')]
                if available_models:
                    print(f"     Available models: {available_models}")
                else:
                    print(f"     No model files found in {hp_space_dir / 'models'}")
        else:
            print("   ⚠ Could not determine model name - only generic spaces loaded")
        
        if not configs:
            raise ValueError(f"No relevant hyperparameter space configs found in {hp_space_dir}")
        
        print(f"   📊 Total configs loaded: {len(configs)}")
        return cls.from_multiple_configs(configs) 