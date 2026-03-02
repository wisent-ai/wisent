"""Auto-grouped modules."""
import importlib as _importlib


def __getattr__(name):
    _mod = _importlib.import_module(".registry", __name__)
    return getattr(_mod, name)
