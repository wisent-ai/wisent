"""Solver implementations for WICHER steering."""

from __future__ import annotations

from typing import Callable

from .broyden import wicher_broyden_step
from .newton import wicher_newton_step
from .halley import wicher_halley_step
from wisent.core.utils.config_tools.constants import (
    WICHER_SOLVER_BROYDEN,
    WICHER_SOLVER_NEWTON,
    WICHER_SOLVER_HALLEY,
)

__all__ = [
    "wicher_broyden_step",
    "wicher_newton_step",
    "wicher_halley_step",
    "get_solver_fn",
]


def get_solver_fn(solver: str) -> Callable:
    """Map solver name string to the corresponding step function."""
    _dispatch = {
        WICHER_SOLVER_BROYDEN: wicher_broyden_step,
        WICHER_SOLVER_NEWTON: wicher_newton_step,
        WICHER_SOLVER_HALLEY: wicher_halley_step,
    }
    fn = _dispatch.get(solver)
    if fn is None:
        raise ValueError(
            f"Unknown WICHER solver: {solver!r}. "
            f"Choose from: {WICHER_SOLVER_BROYDEN}, "
            f"{WICHER_SOLVER_NEWTON}, {WICHER_SOLVER_HALLEY}"
        )
    return fn
