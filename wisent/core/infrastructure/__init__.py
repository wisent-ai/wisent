import os as _os
_base = _os.path.dirname(__file__)
for _root, _dirs, _files in _os.walk(_base):
    _dirs[:] = sorted(d for d in _dirs if not d.startswith((".", "_")))
    if _root != _base:
        __path__.append(_root)
_config = _os.path.join(_os.path.dirname(_base), "utils", "config")
if _os.path.isdir(_config):
    __path__.append(_config)
"""Auto-grouped modules."""
