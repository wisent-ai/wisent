import os as _os
_base = _os.path.dirname(__file__)
for _entry in sorted(_os.listdir(_base)):
    _path = _os.path.join(_base, _entry)
    if _os.path.isdir(_path) and not _entry.startswith(('.', '_')):
        __path__.append(_path)

"""Auto-grouped modules."""
