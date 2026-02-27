"""Input/output utilities for contrastive pairs.

Note: This package contains serialization and response modules.
For serialization functions, import from the submodule directly to avoid
circular imports: from .serialization import save_contrastive_pair_set
"""

from .response import (
    Response,
    PositiveResponse,
    NegativeResponse,
)

# Note: serialization imports are not included here to avoid circular imports
# with pair.py. Import directly from wisent.core.contrastive_pairs.core.io.serialization

__all__ = [
    "Response",
    "PositiveResponse",
    "NegativeResponse",
]
