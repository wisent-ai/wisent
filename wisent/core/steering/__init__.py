import os as _os
_base = _os.path.dirname(__file__)
for _root, _dirs, _files in _os.walk(_base):
    _dirs[:] = sorted(d for d in _dirs if not d.startswith((".", "_")))
    if _root != _base:
        __path__.append(_root)
_sc = _os.path.join(_os.path.dirname(_base), "control", "steering_core")
if _os.path.isdir(_sc):
    for _root, _dirs, _files in _os.walk(_sc):
        _dirs[:] = sorted(d for d in _dirs if not d.startswith((".", "_")))
        __path__.append(_root)
"""Auto-grouped modules."""
