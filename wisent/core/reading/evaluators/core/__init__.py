import os as _os
_base = _os.path.dirname(__file__)

# Bridge to extended/core/ first for BaseEvaluator atoms
_ext_core = _os.path.join(_os.path.dirname(_base), "extended", "core")
if _os.path.isdir(_ext_core):
    __path__.append(_ext_core)

for _root, _dirs, _files in _os.walk(_base):
    _dirs[:] = sorted(d for d in _dirs if not d.startswith((".", "_")))
    if _root != _base:
        __path__.append(_root)

"""Auto-grouped modules."""
