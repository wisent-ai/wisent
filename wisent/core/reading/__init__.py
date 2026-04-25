import os as _os
import pkgutil as _pkgutil
# Merge with sibling installs: wisent-evaluators contributes
# wisent/core/reading/evaluators/. Without extend_path, the regular
# package above would shadow the namespace install.
__path__ = _pkgutil.extend_path(__path__, __name__)
_base = _os.path.dirname(__file__)
for _entry in sorted(_os.listdir(_base)):
    _path = _os.path.join(_base, _entry)
    if _os.path.isdir(_path) and not _entry.startswith(('.', '_')):
        __path__.append(_path)

"""Representation reading: geometry, zwiad, classifiers, diagnostics, evaluators."""
