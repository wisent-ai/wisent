"""Task-specific extractors for HuggingFace datasets."""

import os as _os
_base = _os.path.dirname(__file__)
for _root, _dirs, _files in _os.walk(_base):
    _dirs[:] = sorted(d for d in _dirs if not d.startswith((".", "_")))
    if _root != _base:
        __path__.append(_root)

from wisent.extractors.hf.hf_task_extractors.aime import AIMEExtractor
from wisent.extractors.hf.hf_task_extractors.apps import AppsExtractor
from wisent.extractors.hf.hf_task_extractors.codexglue import CodexglueExtractor
from wisent.extractors.hf.hf_task_extractors.conala import ConalaExtractor
from wisent.extractors.hf.hf_task_extractors.concode import ConcodeExtractor
from wisent.extractors.hf.hf_task_extractors.ds_1000 import Ds1000Extractor
from wisent.extractors.hf.hf_task_extractors.hle import HleExtractor
from wisent.extractors.hf.hf_task_extractors.hmmt import HMMTExtractor
from wisent.extractors.hf.hf_task_extractors.humaneval import (
    HumanEvalExtractor,
    HumanEval64Extractor,
    HumanEvalPlusExtractor,
    HumanEvalInstructExtractor,
    HumanEval64InstructExtractor,
)
from wisent.extractors.hf.hf_task_extractors.livecodebench import LivecodebenchExtractor
from wisent.extractors.hf.hf_task_extractors.livemathbench import LiveMathBenchExtractor
from wisent.extractors.hf.hf_task_extractors.math500 import MATH500Extractor
from wisent.extractors.hf.hf_task_extractors.mercury import MercuryExtractor
from wisent.extractors.hf.hf_task_extractors.multipl_e import MultiplEExtractor
from wisent.extractors.hf.hf_task_extractors.polymath import PolyMathExtractor
from wisent.extractors.hf.hf_task_extractors.recode import RecodeExtractor
from wisent.extractors.hf.hf_task_extractors.super_gpqa import SuperGpqaExtractor

__all__ = [
    "AIMEExtractor",
    "AppsExtractor",
    "CodexglueExtractor",
    "ConalaExtractor",
    "ConcodeExtractor",
    "Ds1000Extractor",
    "HleExtractor",
    "HMMTExtractor",
    "HumanEvalExtractor",
    "HumanEval64Extractor",
    "HumanEvalPlusExtractor",
    "HumanEvalInstructExtractor",
    "HumanEval64InstructExtractor",
    "LivecodebenchExtractor",
    "LiveMathBenchExtractor",
    "MATH500Extractor",
    "MercuryExtractor",
    "MultiplEExtractor",
    "PolyMathExtractor",
    "RecodeExtractor",
    "SuperGpqaExtractor",
]

