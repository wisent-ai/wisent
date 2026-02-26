import os as _os
_base = _os.path.dirname(__file__)
for _root, _dirs, _files in _os.walk(_base):
    _dirs[:] = sorted(d for d in _dirs if d.startswith(("grp_", "sub_", "mid_")))
    if _root != _base:
        __path__.append(_root)

"""Steering optimization parser arguments."""

from .optimize_steering_parser import setup_steering_optimizer_parser
from .optimize_classification_parser import setup_classification_optimizer_parser
from .optimize_sample_size_parser import setup_sample_size_optimizer_parser
from .tune_recommendation_parser import setup_tune_recommendation_parser

__all__ = [
    'setup_steering_optimizer_parser',
    'setup_classification_optimizer_parser',
    'setup_sample_size_optimizer_parser',
    'setup_tune_recommendation_parser',
]
