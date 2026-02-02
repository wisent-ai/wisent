"""
Manager subpackage for WisentConfigManager.

Assembles the full WisentConfigManager class from mixins.
"""

from __future__ import annotations

from .base import WisentConfigManagerBase
from .classification import ClassificationMixin
from .steering import SteeringMixin
from .traits import TraitsMixin


class WisentConfigManager(
    ClassificationMixin,
    SteeringMixin,
    TraitsMixin,
    WisentConfigManagerBase
):
    """
    Unified configuration manager for all Wisent optimization parameters.

    Stores one JSON file per model at ~/.wisent/configs/{model_name}.json

    Methods are provided by mixins:
    - ClassificationMixin: save/get classification config
    - SteeringMixin: save/get steering and weight modification config
    - TraitsMixin: save/get trait-specific config plus general methods
    - WisentConfigManagerBase: core utilities and initialization
    """
    pass


__all__ = ["WisentConfigManager"]
