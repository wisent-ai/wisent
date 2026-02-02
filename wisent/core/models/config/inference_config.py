"""
Inference configuration for model generation.

This module provides a single, global inference configuration that can be
viewed and updated from the CLI. The config is stored as a JSON file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any


# Default config file location
CONFIG_DIR = Path.home() / ".wisent"
CONFIG_FILE = CONFIG_DIR / "inference_config.json"


@dataclass
class InferenceConfig:
    """Configuration for model inference/generation."""

    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 32768
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    enable_thinking: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_generate_kwargs(self, **overrides) -> dict[str, Any]:
        """Convert config to kwargs for model.generate()."""
        kwargs = {
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_new_tokens,
        }

        if self.do_sample:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
            kwargs["top_k"] = self.top_k

        if self.repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = self.repetition_penalty

        if self.no_repeat_ngram_size > 0:
            kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size

        kwargs.update(overrides)
        return kwargs

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferenceConfig":
        """Create config from dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def save(self, path: Path | str | None = None) -> None:
        """Save config to JSON file."""
        path = Path(path) if path else CONFIG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "InferenceConfig":
        """Load config from JSON file, or return defaults if not found."""
        path = Path(path) if path else CONFIG_FILE
        if path.exists():
            with open(path) as f:
                return cls.from_dict(json.load(f))
        return cls()


# Global config instance
_config: InferenceConfig | None = None


def get_config() -> InferenceConfig:
    """Get the global inference config (loads from file if needed)."""
    global _config
    if _config is None:
        _config = InferenceConfig.load()
    return _config


def set_config(config: InferenceConfig) -> None:
    """Set the global inference config."""
    global _config
    _config = config


def save_config() -> None:
    """Save the current global config to file."""
    get_config().save()


def update_config(**kwargs) -> InferenceConfig:
    """Update config values and save."""
    config = get_config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.save()
    return config


def reset_config() -> InferenceConfig:
    """Reset config to defaults and save."""
    global _config
    _config = InferenceConfig()
    _config.save()
    return _config


def get_generate_kwargs(**overrides) -> dict[str, Any]:
    """Get generation kwargs from global config with optional overrides."""
    return get_config().to_generate_kwargs(**overrides)
