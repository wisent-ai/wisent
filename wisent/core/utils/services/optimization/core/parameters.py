"""Backend-agnostic parameter definitions for unified optimization."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from wisent.core.utils.config_tools.constants import QUANTIZATION_STEP_DEFAULT


@dataclass
class FloatParam:
    """Float parameter with a distribution prior.

    Supported distributions:
    - "normal": unbounded, centred on mu with spread sigma
    - "lognormal": positive-only, centred on exp(mu) with spread sigma
    - "uniform": bounded, requires low and high
    """

    distribution: str
    mu: float | None = None
    sigma: float | None = None
    low: float | None = None
    high: float | None = None
    log_scale: bool = False

    def __post_init__(self) -> None:
        if self.distribution in ("normal", "lognormal"):
            if self.mu is None or self.sigma is None:
                raise ValueError(
                    f"{self.distribution} requires mu and sigma"
                )
        elif self.distribution == "uniform":
            if self.low is None or self.high is None:
                raise ValueError("uniform requires low and high")


@dataclass
class IntParam:
    """Integer parameter.

    Supported distributions:
    - "randint": uniform over [low, high)
    - "qnormal": quantised normal (mu, sigma, q)
    - "qlognormal": quantised lognormal (mu, sigma, q)
    """

    distribution: str
    mu: float | None = None
    sigma: float | None = None
    q: int = QUANTIZATION_STEP_DEFAULT
    low: int | None = None
    high: int | None = None

    def __post_init__(self) -> None:
        if self.distribution == "randint":
            if self.low is None or self.high is None:
                raise ValueError("randint requires low and high")
        elif self.distribution in ("qnormal", "qlognormal"):
            if self.mu is None or self.sigma is None:
                raise ValueError(
                    f"{self.distribution} requires mu and sigma"
                )


@dataclass
class CategoricalParam:
    """Categorical parameter with explicit choice list."""

    choices: list = field(default_factory=list)


Param = FloatParam | IntParam | CategoricalParam


@dataclass
class OptimizationRun:
    """Result of a completed optimization run."""

    best_params: dict[str, Any]
    best_score: float
    all_trials: list[dict]
    n_trials: int
    backend: str
