"""Global instance and convenience functions for config manager."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .types import ClassificationConfig, SteeringConfig, WeightModificationConfig
from .manager import WisentConfigManager


# Global instance
_config_manager: Optional[WisentConfigManager] = None


def get_config_manager() -> WisentConfigManager:
    """Get the global config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = WisentConfigManager()
    return _config_manager


# Classification convenience functions
def save_classification_config(model_name: str, **kwargs) -> Path:
    """Save classification config using global manager."""
    return get_config_manager().save_classification_config(model_name, **kwargs)


def get_classification_config(model_name: str, task_name: Optional[str] = None) -> Optional[ClassificationConfig]:
    """Get classification config using global manager."""
    return get_config_manager().get_classification_config(model_name, task_name)


# Steering convenience functions
def save_steering_config(model_name: str, **kwargs) -> Path:
    """Save steering config using global manager."""
    return get_config_manager().save_steering_config(model_name, **kwargs)


def get_steering_config(model_name: str, task_name: Optional[str] = None) -> Optional[SteeringConfig]:
    """Get steering config using global manager."""
    return get_config_manager().get_steering_config(model_name, task_name)


# Weight modification convenience functions
def save_weight_modification_config(model_name: str, **kwargs) -> Path:
    """Save weight modification config using global manager."""
    return get_config_manager().save_weight_modification_config(model_name, **kwargs)


def get_weight_modification_config(model_name: str, task_name: Optional[str] = None) -> Optional[WeightModificationConfig]:
    """Get weight modification config using global manager."""
    return get_config_manager().get_weight_modification_config(model_name, task_name)


# Trait convenience functions
def save_trait_classification_config(model_name: str, trait_name: str, **kwargs) -> Path:
    """Save classification config for a trait using global manager."""
    return get_config_manager().save_trait_classification_config(model_name, trait_name, **kwargs)


def get_trait_classification_config(model_name: str, trait_name: str) -> Optional[ClassificationConfig]:
    """Get classification config for a trait using global manager."""
    return get_config_manager().get_trait_classification_config(model_name, trait_name)


def save_trait_steering_config(model_name: str, trait_name: str, **kwargs) -> Path:
    """Save steering config for a trait using global manager."""
    return get_config_manager().save_trait_steering_config(model_name, trait_name, **kwargs)


def get_trait_steering_config(model_name: str, trait_name: str) -> Optional[SteeringConfig]:
    """Get steering config for a trait using global manager."""
    return get_config_manager().get_trait_steering_config(model_name, trait_name)


def save_trait_weight_modification_config(model_name: str, trait_name: str, **kwargs) -> Path:
    """Save weight modification config for a trait using global manager."""
    return get_config_manager().save_trait_weight_modification_config(model_name, trait_name, **kwargs)


def get_trait_weight_modification_config(model_name: str, trait_name: str) -> Optional[WeightModificationConfig]:
    """Get weight modification config for a trait using global manager."""
    return get_config_manager().get_trait_weight_modification_config(model_name, trait_name)
