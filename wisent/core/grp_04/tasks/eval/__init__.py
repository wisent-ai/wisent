import os as _os
_base = _os.path.dirname(__file__)
for _root, _dirs, _files in _os.walk(_base):
    _dirs[:] = sorted(d for d in _dirs if d.startswith(("grp_", "sub_", "mid_")))
    if _root != _base:
        __path__.append(_root)

"""Evaluation task implementations."""

from .lm_eval_task import (
    LMEvalTask,
    MBPPTask,
    HumanEvalTask,
    MBPPPlusTask,
    GSM8KTask,
    ArithmeticBaseTask,
    Arithmetic1dcTask,
    Arithmetic2daTask,
    Arithmetic2dmTask,
)
from .hle_task import HLETask, HLEExactMatchTask, HLEMultipleChoiceTask
from .supergpqa_task import (
    SuperGPQATask,
    SuperGPQAPhysicsTask,
    SuperGPQAChemistryTask,
    SuperGPQABiologyTask,
)

__all__ = [
    'LMEvalTask',
    'MBPPTask',
    'HumanEvalTask',
    'MBPPPlusTask',
    'GSM8KTask',
    'ArithmeticBaseTask',
    'Arithmetic1dcTask',
    'Arithmetic2daTask',
    'Arithmetic2dmTask',
    'HLETask',
    'HLEExactMatchTask',
    'HLEMultipleChoiceTask',
    'SuperGPQATask',
    'SuperGPQAPhysicsTask',
    'SuperGPQAChemistryTask',
    'SuperGPQABiologyTask',
]
