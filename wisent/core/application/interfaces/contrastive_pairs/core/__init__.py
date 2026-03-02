"""Bridge to primitives/contrastive_pairs/ for backward-compatible imports."""
import os as _os

_base = _os.path.dirname(__file__)
_primitives = _os.path.normpath(
    _os.path.join(_base, "..", "..", "..", "..", "primitives", "contrastive_pairs")
)
if _os.path.isdir(_primitives):
    __path__.append(_primitives)
