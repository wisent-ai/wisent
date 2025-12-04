"""
Steering method rotator for discovering and selecting steering methods.

Uses BaseRotator for common plugin discovery and resolution logic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Type, Union

from wisent.core.steering_methods.core.atoms import BaseSteeringError, BaseSteeringMethod
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.activations.core.atoms import LayerActivations
from wisent.core.utils.base_rotator import BaseRotator

__all__ = [
    "SteeringMethodRotator",
]

logger = logging.getLogger(__name__)


class SteeringMethodRotator(BaseRotator[BaseSteeringMethod]):
    """
    Discover/select a steering method and train it on a ContrastivePairSet.

    Extends BaseRotator with steering method-specific functionality:
    - Train method for generating steering vectors
    - Kwargs override support during training
    """

    def __init__(
        self,
        method: Union[str, BaseSteeringMethod, Type[BaseSteeringMethod], None] = None,
        methods_location: Union[str, Path] = "wisent.core.steering_methods.methods",
        autoload: bool = True,
        **default_method_kwargs: Any,
    ) -> None:
        """
        Initialize the steering method rotator.

        Args:
            method: Method name, instance, class, or None for auto-selection.
            methods_location: Module path or directory for method discovery.
            autoload: Whether to auto-discover methods on init.
            **default_method_kwargs: Default kwargs passed to method.
        """
        super().__init__(
            plugin=method,
            location=methods_location,
            autoload=autoload,
            **default_method_kwargs,
        )

    def _get_registry_class(self) -> Type[BaseSteeringMethod]:
        return BaseSteeringMethod

    def _get_error_class(self) -> Type[Exception]:
        return BaseSteeringError

    def _get_plugin_type_name(self) -> str:
        return "steering method"

    # Keep static method for backward compatibility
    @staticmethod
    def discover_methods(location: Union[str, Path]) -> None:
        """
        Import all steering method modules so subclasses self-register.

        Static method for backward compatibility.
        """
        rotator = SteeringMethodRotator.__new__(SteeringMethodRotator)
        rotator._scope_prefix = ""
        rotator.discover(location)

    @staticmethod
    def list_methods() -> List[Dict[str, Any]]:
        """List all registered steering methods."""
        return [
            {
                "name": name,
                "description": getattr(cls, "description", ""),
                "class": f"{cls.__module__}.{cls.__name__}",
            }
            for name, cls in sorted(
                BaseSteeringMethod.list_registered().items(),
                key=lambda kv: kv[0]
            )
        ]

    def use(self, method: Union[str, BaseSteeringMethod, Type[BaseSteeringMethod]], **kwargs: Any) -> None:
        """Switch to a different steering method."""
        merged_kwargs = {**self._default_kwargs, **kwargs}
        self._plugin = self._resolve(method, **merged_kwargs)

    def train(self, pair_set: ContrastivePairSet, **overrides: Any) -> LayerActivations:
        """
        Train the steering method on a contrastive pair set.

        Args:
            pair_set: ContrastivePairSet to train on.
            **overrides: Kwargs to override for this training call.

        Returns:
            LayerActivations containing the trained steering vectors.
        """
        old = dict(self._plugin.kwargs)
        try:
            self._plugin.kwargs = {**old, **overrides}
            return self._plugin.train(pair_set)
        finally:
            self._plugin.kwargs = old


if __name__ == "__main__":
    rot = SteeringMethodRotator()
    print("Available steering methods:")
    for m in rot.list_methods():
        print(f" - {m['name']}: {m['description']} ({m['class']})")
