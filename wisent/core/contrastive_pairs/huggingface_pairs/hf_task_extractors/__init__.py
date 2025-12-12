"""Task-specific extractors for HuggingFace datasets."""

from wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors.aime import AIMEExtractor
from wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors.apps import AppsExtractor
from wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors.codexglue import CodexglueExtractor
from wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors.conala import ConalaExtractor
from wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors.concode import ConcodeExtractor
from wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors.ds_1000 import Ds1000Extractor
from wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors.hle import HleExtractor
from wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors.hmmt import HMMTExtractor
from wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors.humaneval import (
    HumanEvalExtractor,
    HumanEval64Extractor,
    HumanEvalPlusExtractor,
    HumanEvalInstructExtractor,
    HumanEval64InstructExtractor,
)
from wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors.livecodebench import LivecodebenchExtractor
from wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors.livemathbench import LiveMathBenchExtractor
from wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors.math500 import MATH500Extractor
from wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors.mercury import MercuryExtractor
from wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors.multipl_e import MultiplEExtractor
from wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors.polymath import PolyMathExtractor
from wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors.recode import RecodeExtractor
from wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors.super_gpqa import SuperGpqaExtractor

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

