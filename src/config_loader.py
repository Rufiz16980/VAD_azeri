import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

class Config:
    """A wrapper class around config dictionaries to allow attribute-style access."""
    def __init__(self, data: Dict[str, Any]):
        # Use __dict__ directly to avoid infinite recursion with __getattr__
        self.__dict__["_data"] = data

    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            val = self._data[name]
            if isinstance(val, dict):
                return Config(val)
            return val
        raise AttributeError(f"'Config' object has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        val = self._data[key]
        if isinstance(val, dict):
            return Config(val)
        return val

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._data:
            val = self._data[key]
            if isinstance(val, dict):
                return Config(val)
            return val
        return default

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"Config({self._data})"

    def to_dict(self) -> Dict[str, Any]:
        return self._data


def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges dict2 into dict1."""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(experiment_name: Optional[str] = None) -> Config:
    """Loads and merges configuration layers.
    
    Layers merged in order (later overrides earlier):
    1. configs/base.yaml
    2. configs/data_pipeline.yaml
    3. configs/models/<model_config>.yaml (if experiment_name is provided)
    4. configs/experiments/<experiment_name>.yaml (if experiment_name is provided)
    """
    repo_root = Path(__file__).resolve().parent.parent
    configs_dir = repo_root / "configs"

    # 1. Load base configuration
    base_path = configs_dir / "base.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"Base configuration file not found at {base_path}")
    with open(base_path, "r", encoding="utf-8") as f:
        merged_data = yaml.safe_load(f) or {}

    # 2. Load data pipeline configuration
    pipeline_path = configs_dir / "data_pipeline.yaml"
    if pipeline_path.exists():
        with open(pipeline_path, "r", encoding="utf-8") as f:
            pipeline_data = yaml.safe_load(f) or {}
        merged_data = deep_merge(merged_data, pipeline_data)

    if experiment_name:
        # 3. Load experiment configuration
        exp_path = configs_dir / "experiments" / f"{experiment_name}.yaml"
        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment configuration file not found at {exp_path}")
        with open(exp_path, "r", encoding="utf-8") as f:
            exp_data = yaml.safe_load(f) or {}

        # Get model configuration name from experiment config
        model_config = exp_data.get("model_config")
        if not model_config:
            raise KeyError(f"Experiment config '{experiment_name}' is missing required key 'model_config'")

        # 4. Load model configuration
        model_path = configs_dir / "models" / f"{model_config}.yaml"
        if not model_path.exists():
            raise FileNotFoundError(f"Model configuration file not found at {model_path}")
        with open(model_path, "r", encoding="utf-8") as f:
            model_data = yaml.safe_load(f) or {}

        # Merge model first, then experiment
        merged_data = deep_merge(merged_data, model_data)
        merged_data = deep_merge(merged_data, exp_data)

    return Config(merged_data)
