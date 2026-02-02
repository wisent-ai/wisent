"""Base class for WisentConfigManager with core utilities."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

from ..types import ModelConfig, NumpyEncoder, DEFAULT_CONFIG_DIR


class WisentConfigManagerBase:
    """Base class providing initialization and core utilities for config management."""

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the config manager with optional custom config directory."""
        self.config_dir = Path(os.path.expanduser(config_dir or DEFAULT_CONFIG_DIR))
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, ModelConfig] = {}

    def _sanitize_model_name(self, model_name: str) -> str:
        """Convert model name to a safe filename."""
        sanitized = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        sanitized = "".join(c for c in sanitized if c.isalnum() or c in "._-")
        return sanitized

    def _get_config_path(self, model_name: str) -> Path:
        """Get the full path to the config file for a model."""
        sanitized_name = self._sanitize_model_name(model_name)
        return self.config_dir / f"{sanitized_name}.json"

    def _load_model_config(self, model_name: str) -> ModelConfig:
        """Load or create a model config with caching."""
        if model_name in self._cache:
            return self._cache[model_name]

        config_path = self._get_config_path(model_name)

        if config_path.exists():
            try:
                with open(config_path) as f:
                    data = json.load(f)
                config = ModelConfig.from_dict(data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to load config for {model_name}: {e}")
                config = ModelConfig(model_name=model_name)
        else:
            config = ModelConfig(model_name=model_name)

        self._cache[model_name] = config
        return config

    def _save_model_config(self, config: ModelConfig) -> Path:
        """Save a model config to disk and update cache."""
        config.updated_at = datetime.now().isoformat()
        config_path = self._get_config_path(config.model_name)

        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2, cls=NumpyEncoder)

        self._cache[config.model_name] = config
        return config_path
