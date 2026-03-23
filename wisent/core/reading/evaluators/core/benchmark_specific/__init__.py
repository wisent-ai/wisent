import os as _os
import importlib as _importlib

_base = _os.path.dirname(__file__)
_pkg = __name__

# Extend __path__ for backwards-compatible flattened imports like
# wisent.core.reading.evaluators.benchmark_specific.math_parsing.scripts
for _root, _dirs, _files in _os.walk(_base):
    _dirs[:] = sorted(d for d in _dirs if not d.startswith((".", "_")))
    if _root != _base:
        __path__.append(_root)

# Import all evaluator modules using FULL dotted paths to avoid
# creating duplicate class objects from __path__ aliasing.
# This triggers BaseEvaluator.__init_subclass__ registration.
for _root, _dirs, _files in _os.walk(_base):
    _dirs[:] = sorted(d for d in _dirs if not d.startswith((".", "_")))
    for _f in sorted(_files):
        if not _f.endswith(".py") or _f.startswith("_"):
            continue
        _rel = _os.path.relpath(_os.path.join(_root, _f), _base)
        _mod = _rel.replace(_os.sep, ".").removesuffix(".py")
        try:
            _importlib.import_module(f"{_pkg}.{_mod}")
        except Exception:
            pass
