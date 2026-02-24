"""Constant metadata registry for empirical validation of named constants."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from wisent.core import constants as _constants_module

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConstantMeta:
    """Metadata for a single named constant in constants.py.

    Attributes:
        name: Constant name as it appears in constants.py (e.g. GROM_LEARNING_RATE).
        group: Classification group - "D" for ML hyperparameters, "E" for thresholds.
        current_value: The default value from constants.py.
        dtype: "int" or "float".
        low: Lower bound of the valid search range.
        high: Upper bound of the valid search range.
        log_scale: Whether to search in log space (useful for learning rates).
        method: Steering method or subsystem this constant belongs to.
        description: Human-readable description of what this constant controls.
    """

    name: str
    group: str
    current_value: float
    dtype: str
    low: float
    high: float
    log_scale: bool
    method: str
    description: str

    def validate(self) -> List[str]:
        """Return a list of validation errors, empty if valid."""
        errors = []
        if self.group not in ("D", "E"):
            errors.append(f"{self.name}: group must be 'D' or 'E', got '{self.group}'")
        if self.dtype not in ("int", "float"):
            errors.append(f"{self.name}: dtype must be 'int' or 'float', got '{self.dtype}'")
        if self.low >= self.high:
            errors.append(f"{self.name}: low ({self.low}) must be < high ({self.high})")
        if self.log_scale and self.low <= 0:
            errors.append(f"{self.name}: log_scale requires low > 0, got {self.low}")
        actual = getattr(_constants_module, self.name, None)
        if actual is None:
            errors.append(f"{self.name}: not found in constants module")
        return errors

    @property
    def range_width(self) -> float:
        """Width of the search range."""
        return self.high - self.low

    def sample_linspace(self, steps: int) -> List[float]:
        """Generate evenly spaced values across the search range."""
        if steps < 2:
            return [self.current_value]
        if self.log_scale:
            import math
            log_low = math.log(self.low)
            log_high = math.log(self.high)
            values = [math.exp(log_low + i * (log_high - log_low) / (steps - 1)) for i in range(steps)]
        else:
            values = [self.low + i * (self.high - self.low) / (steps - 1) for i in range(steps)]
        if self.dtype == "int":
            seen = set()
            int_values = []
            for v in values:
                iv = int(round(v))
                if iv not in seen:
                    seen.add(iv)
                    int_values.append(float(iv))
            return int_values
        return values

    def cast_value(self, value: float) -> float:
        """Cast a value to the correct dtype."""
        if self.dtype == "int":
            return float(int(round(value)))
        return float(value)

    def clamp(self, value: float) -> float:
        """Clamp a value to the valid range."""
        return max(self.low, min(self.high, value))


# Module-level registry cache
_REGISTRY_CACHE: Optional[Dict[str, ConstantMeta]] = None


def _build_registry() -> Dict[str, ConstantMeta]:
    """Build the registry from registry_data entries."""
    from .registry_data import REGISTRY_ENTRIES
    registry = {}
    for entry in REGISTRY_ENTRIES:
        meta = ConstantMeta(**entry)
        errors = meta.validate()
        if errors:
            for err in errors:
                logger.warning("Registry validation: %s", err)
        registry[meta.name] = meta
    return registry


def get_registry() -> Dict[str, ConstantMeta]:
    """Get the full constant registry, building it on first access."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = _build_registry()
    return _REGISTRY_CACHE


def get_constants_by_group(group: str) -> Dict[str, ConstantMeta]:
    """Filter registry to constants in the given group ('D' or 'E')."""
    return {k: v for k, v in get_registry().items() if v.group == group}


def get_constants_by_method(method: str) -> Dict[str, ConstantMeta]:
    """Filter registry to constants belonging to a specific method."""
    method_lower = method.lower()
    return {k: v for k, v in get_registry().items() if v.method.lower() == method_lower}


def get_constant_value(name: str) -> float:
    """Read current value of a constant from the constants module."""
    val = getattr(_constants_module, name, None)
    if val is None:
        raise KeyError(f"Constant {name} not found in constants module")
    return float(val)


def set_constant_value(name: str, value: float) -> float:
    """Set a constant value in the constants module, returning the old value."""
    old = get_constant_value(name)
    setattr(_constants_module, name, value)
    return old


def reset_constant(name: str, meta: ConstantMeta) -> None:
    """Reset a constant to its default value."""
    setattr(_constants_module, name, meta.current_value)


def reset_all_constants() -> None:
    """Reset all registered constants to their default values."""
    for name, meta in get_registry().items():
        setattr(_constants_module, name, meta.current_value)


def patch_constants(overrides: Dict[str, float]) -> Dict[str, float]:
    """Patch multiple constants, returning dict of old values."""
    old_values = {}
    registry = get_registry()
    for name, value in overrides.items():
        if name not in registry:
            logger.warning("Skipping unknown constant: %s", name)
            continue
        meta = registry[name]
        clamped = meta.clamp(value)
        cast = meta.cast_value(clamped)
        old_values[name] = get_constant_value(name)
        set_constant_value(name, cast)
    return old_values


def restore_constants(old_values: Dict[str, float]) -> None:
    """Restore constants from a dict of old values."""
    for name, value in old_values.items():
        setattr(_constants_module, name, value)


class ConstantPatcher:
    """Context manager for temporarily patching constants."""

    def __init__(self, overrides: Dict[str, float]):
        self.overrides = overrides
        self._old_values: Dict[str, float] = {}

    def __enter__(self) -> ConstantPatcher:
        self._old_values = patch_constants(self.overrides)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        restore_constants(self._old_values)


def summarize_registry() -> Dict[str, int]:
    """Return summary counts of registry contents."""
    registry = get_registry()
    group_d = sum(1 for v in registry.values() if v.group == "D")
    group_e = sum(1 for v in registry.values() if v.group == "E")
    methods = {}
    for v in registry.values():
        methods[v.method] = methods.get(v.method, 0) + 1
    log_scale_count = sum(1 for v in registry.values() if v.log_scale)
    int_count = sum(1 for v in registry.values() if v.dtype == "int")
    return {
        "total": len(registry),
        "group_D": group_d,
        "group_E": group_e,
        "log_scale": log_scale_count,
        "int_typed": int_count,
        "methods": methods,
    }
