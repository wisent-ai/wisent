"""Bridge to reading/diagnostics/ for backward-compatible imports."""
import os as _os

_base = _os.path.dirname(__file__)
_reading_diag = _os.path.normpath(
    _os.path.join(_base, "..", "..", "..", "..", "reading", "diagnostics")
)
if _os.path.isdir(_reading_diag):
    __path__.append(_reading_diag)

# Re-export from the actual location
from wisent.core.reading.diagnostics import *  # noqa: F401,F403
