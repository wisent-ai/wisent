"""Backward-compatible ModelConfigManager class."""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, Any, List

from ..convenience import (
    get_config_manager, save_classification_config, save_steering_config,
)


class ModelConfigManager:
    """Backward-compatible wrapper class for the unified config manager."""

    def __init__(self, config_dir: Optional[str] = None):
        self._manager = get_config_manager()

    def _sanitize_model_name(self, model_name: str) -> str:
        return self._manager._sanitize_model_name(model_name)

    def _get_config_path(self, model_name: str) -> str:
        return str(self._manager._get_config_path(model_name))

    def save_model_config(
        self, model_name: str, classification_layer: int,
        steering_layer: Optional[int] = None, token_aggregation: str = "average",
        detection_threshold: float = 0.6, optimization_method: str = "manual",
        optimization_metrics: Optional[Dict[str, Any]] = None,
        task_specific_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        """Save model configuration using unified config manager."""
        if steering_layer is None:
            steering_layer = classification_layer

        save_classification_config(
            model_name=model_name, layer=classification_layer,
            token_aggregation=token_aggregation, detection_threshold=detection_threshold,
            optimization_method=optimization_method, set_as_default=True,
        )
        save_steering_config(
            model_name=model_name, layer=steering_layer,
            optimization_method=optimization_method, set_as_default=True,
        )
        return str(self._manager._get_config_path(model_name))

    def load_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load model configuration in legacy format."""
        config = self._manager.get_model_config(model_name)
        if not config.default_classification and not config.default_steering:
            return None

        result = {
            "model_name": model_name,
            "optimal_parameters": {},
            "task_specific_overrides": {},
            "optimization_metrics": {},
            "config_version": "2.0",
        }
        if config.default_classification:
            result["optimal_parameters"]["classification_layer"] = config.default_classification.layer
            result["optimal_parameters"]["token_aggregation"] = config.default_classification.token_aggregation
            result["optimal_parameters"]["detection_threshold"] = config.default_classification.detection_threshold
        if config.default_steering:
            result["optimal_parameters"]["steering_layer"] = config.default_steering.layer
        return result

    def has_model_config(self, model_name: str) -> bool:
        """Check if a model has a saved configuration."""
        return self._manager.has_config(model_name)

    def get_optimal_parameters(
        self, model_name: str, task_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get optimal parameters for a model."""
        config = self.load_model_config(model_name)
        if not config:
            return None
        return config.get("optimal_parameters", {})

    def list_model_configs(self) -> List[Dict[str, Any]]:
        """List all available model configurations."""
        configs = []
        for model_name in self._manager.list_models():
            config = self.load_model_config(model_name)
            if config:
                configs.append({
                    "model_name": model_name,
                    "classification_layer": config.get("optimal_parameters", {}).get("classification_layer"),
                    "steering_layer": config.get("optimal_parameters", {}).get("steering_layer"),
                })
        return configs

    def remove_model_config(self, model_name: str) -> bool:
        """Remove a model configuration."""
        return self._manager.delete_config(model_name)


def get_default_manager() -> ModelConfigManager:
    """Get a default ModelConfigManager instance (backward compatible)."""
    return ModelConfigManager()


def save_model_config(model_name: str, **kwargs) -> str:
    """Save model configuration using default manager (backward compatible)."""
    return ModelConfigManager().save_model_config(model_name, **kwargs)


def load_model_config(model_name: str) -> Optional[Dict[str, Any]]:
    """Load model configuration using default manager (backward compatible)."""
    return ModelConfigManager().load_model_config(model_name)


def get_optimal_parameters(model_name: str, task_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get optimal parameters using default manager (backward compatible)."""
    return ModelConfigManager().get_optimal_parameters(model_name, task_name)
