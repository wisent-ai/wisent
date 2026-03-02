import os as _os
_base = _os.path.dirname(__file__)
for _root, _dirs, _files in _os.walk(_base):
    _dirs[:] = sorted(d for d in _dirs if not d.startswith((".", "_")))
    if _root != _base:
        __path__.append(_root)

# Bridge to extractors/lm_eval/_registry/manifest/ for LMEvalBenchmarkExtractor
_wisent = _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.dirname(_base))
)))
_manifest = _os.path.join(_wisent, "extractors", "lm_eval", "_registry", "manifest")
if _os.path.isdir(_manifest):
    __path__.append(_manifest)
