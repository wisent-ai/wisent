"""Empirically optimal parameter values from validation studies.

Reads proven results from parameters_to_validate.json.
These are optimization outputs, not defaults.
"""
import json
from pathlib import Path
from typing import Any, Optional

_JSON_PATH = Path(__file__).parent / "parameters_to_validate.json"


def _load_optimal() -> dict:
    """Load the optimal section from parameters_to_validate.json."""
    with open(_JSON_PATH) as f:
        data = json.load(f)
    return data.get("_meta", {}).get("optimal", {})


def _load_validated() -> dict:
    """Load the validated section from parameters_to_validate.json."""
    with open(_JSON_PATH) as f:
        data = json.load(f)
    return data.get("_meta", {}).get("validated", {})


def get_optimal(param: str, method: Optional[str] = None) -> Any:
    """Return the empirically optimal value for a parameter.

    For global params: get_optimal("extraction_strategy")
    For method params: get_optimal("num_directions", method="grom")
    Raises ValueError if no optimal value exists.
    """
    optimal = _load_optimal()
    if method:
        method_section = optimal.get(method, {})
        value = method_section.get(param)
        if value is None:
            raise ValueError(
                f"No optimal {method}.{param} in parameters_to_validate.json"
            )
    else:
        value = optimal.get(param)
        if value is None:
            raise ValueError(
                f"No optimal {param} in parameters_to_validate.json"
            )
    return value


def get_optimal_extraction_strategy() -> str:
    """Convenience wrapper for the most commonly used optimal."""
    return get_optimal("extraction_strategy")


def get_validated_extraction_strategies() -> list[str]:
    """Return the list of validated extraction strategies.

    These are strategies confirmed to work by empirical studies,
    though not necessarily the optimal one.
    Raises ValueError if no validated strategies exist.
    """
    validated = _load_validated()
    values = validated.get("extraction_strategy")
    if not values:
        raise ValueError(
            "No validated extraction_strategy list in parameters_to_validate.json"
        )
    return values
