"""Weight modification CLI command package.

This package provides the modify-weights command for permanently modifying
model weights using steering vectors via directional projection or additive methods.
"""

from .entry import execute_modify_weights

__all__ = ["execute_modify_weights"]
