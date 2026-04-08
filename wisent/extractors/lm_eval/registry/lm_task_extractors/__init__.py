"""LM-Eval benchmark extractors."""

import os as _os
import sys as _sys
_base = _os.path.dirname(__file__)
for _root, _dirs, _files in _os.walk(_base):
    _dirs[:] = sorted(d for d in _dirs if not d.startswith((".", "_")))
    if _root != _base:
        __path__.append(_root)

from wisent.extractors.lm_eval.lm_task_extractors.specialized.safety.content_safety.model_written_evals import ModelWrittenEvalsExtractor

# Create module alias for model_written_evals extractor
import types as _types
_model_written_evals_mod = _types.ModuleType("model_written_evals")
_model_written_evals_mod.ModelWrittenEvalsExtractor = ModelWrittenEvalsExtractor
_sys.modules["wisent.extractors.lm_eval.lm_task_extractors.model_written_evals"] = _model_written_evals_mod
# Also register under the registry path (used by LM_EVAL_EXTRACTOR_BASE_IMPORT)
_sys.modules["wisent.extractors.lm_eval.registry.lm_task_extractors.model_written_evals"] = _model_written_evals_mod

from wisent.extractors.lm_eval.lm_task_extractors.specialized.safety.ai_risk.advanced import AdvancedExtractor

# Create module alias for advanced extractor
_advanced_mod = _types.ModuleType("advanced")
_advanced_mod.AdvancedExtractor = AdvancedExtractor
_sys.modules["wisent.extractors.lm_eval.lm_task_extractors.advanced"] = _advanced_mod
# Also register under the registry path (used by LM_EVAL_EXTRACTOR_BASE_IMPORT)
# This prevents the applied/math/advanced package from shadowing the ai_risk AdvancedExtractor
_sys.modules["wisent.extractors.lm_eval.registry.lm_task_extractors.advanced"] = _advanced_mod

from wisent.extractors.lm_eval.lm_task_extractors.specialized.language.asian_african.african.afrimgsm import AfrimgsmExtractor

# Create module alias for afrimgsm extractor
_afrimgsm_mod = _types.ModuleType("afrimgsm")
_afrimgsm_mod.AfrimgsmExtractor = AfrimgsmExtractor
_sys.modules["wisent.extractors.lm_eval.lm_task_extractors.afrimgsm"] = _afrimgsm_mod
# Also register under the registry path
_sys.modules["wisent.extractors.lm_eval.registry.lm_task_extractors.afrimgsm"] = _afrimgsm_mod

from wisent.extractors.lm_eval.lm_task_extractors.evaluation.reasoning.applied.multi_step.agieval import AgievalExtractor, AgievalMathExtractor, AgievalLogiQAExtractor

# Create module alias for agieval extractor
_agieval_mod = _types.ModuleType("agieval")
_agieval_mod.AgievalExtractor = AgievalExtractor
_agieval_mod.AgievalMathExtractor = AgievalMathExtractor
_agieval_mod.AgievalLogiQAExtractor = AgievalLogiQAExtractor
_sys.modules["wisent.extractors.lm_eval.lm_task_extractors.agieval"] = _agieval_mod
# Also register under the registry path
_sys.modules["wisent.extractors.lm_eval.registry.lm_task_extractors.agieval"] = _agieval_mod

from wisent.extractors.lm_eval.lm_task_extractors.evaluation.knowledge.analysis.text_classification.twenty_newsgroups import TwentyNewsgroupsExtractor

# Create module alias for twenty_newsgroups extractor
_twenty_newsgroups_mod = _types.ModuleType("twenty_newsgroups")
_twenty_newsgroups_mod.TwentyNewsgroupsExtractor = TwentyNewsgroupsExtractor
_sys.modules["wisent.extractors.lm_eval.lm_task_extractors.twenty_newsgroups"] = _twenty_newsgroups_mod
# Also register under the registry path
_sys.modules["wisent.extractors.lm_eval.registry.lm_task_extractors.twenty_newsgroups"] = _twenty_newsgroups_mod

from wisent.extractors.lm_eval.lm_task_extractors.specialized.language.text_and_translation.translation_regional.regional_misc.gaokao import GaokaoExtractor

# Create module alias for gaokao extractor
_gaokao_mod = _types.ModuleType("gaokao")
_gaokao_mod.GaokaoExtractor = GaokaoExtractor
_sys.modules["wisent.extractors.lm_eval.lm_task_extractors.gaokao"] = _gaokao_mod
# Also register under the registry path
_sys.modules["wisent.extractors.lm_eval.registry.lm_task_extractors.gaokao"] = _gaokao_mod

# benchmarks: there's a `benchmarks` folder in the walk path that shadows the
# `benchmarks.py` file inside `evaluation/knowledge/benchmarks/reference_benchmarks/`.
# Force the file's BenchmarksExtractor to take precedence by aliasing the module.
from wisent.extractors.lm_eval.registry.lm_task_extractors.evaluation.knowledge.benchmarks.reference_benchmarks.benchmarks import BenchmarksExtractor
_benchmarks_mod = _types.ModuleType("benchmarks")
_benchmarks_mod.BenchmarksExtractor = BenchmarksExtractor
_sys.modules["wisent.extractors.lm_eval.lm_task_extractors.benchmarks"] = _benchmarks_mod
_sys.modules["wisent.extractors.lm_eval.registry.lm_task_extractors.benchmarks"] = _benchmarks_mod

# translation: same folder/file collision as benchmarks
from wisent.extractors.lm_eval.registry.lm_task_extractors.specialized.language.text_and_translation.text_processing.text_tasks.translation import TranslationExtractor
_translation_mod = _types.ModuleType("translation")
_translation_mod.TranslationExtractor = TranslationExtractor
_sys.modules["wisent.extractors.lm_eval.lm_task_extractors.translation"] = _translation_mod
_sys.modules["wisent.extractors.lm_eval.registry.lm_task_extractors.translation"] = _translation_mod

from wisent.extractors.lm_eval.lm_task_extractors.ai2_arc import AI2ARCExtractor
from wisent.extractors.lm_eval.lm_task_extractors.anli import ANLIExtractor
from wisent.extractors.lm_eval.lm_task_extractors.arc_challenge import ArcChallengeExtractor
from wisent.extractors.lm_eval.lm_task_extractors.arc_easy import ArcEasyExtractor
from wisent.extractors.lm_eval.lm_task_extractors.arithmetic import ArithmeticExtractor
from wisent.extractors.lm_eval.lm_task_extractors.asdiv import ASDivExtractor
from wisent.extractors.lm_eval.lm_task_extractors.bigbench import BigBenchExtractor
from wisent.extractors.lm_eval.lm_task_extractors.boolq import BoolQExtractor
from wisent.extractors.lm_eval.lm_task_extractors.cb import CBExtractor
from wisent.extractors.lm_eval.lm_task_extractors.coqa import CoQAExtractor
from wisent.extractors.lm_eval.lm_task_extractors.drop import DropExtractor
from wisent.extractors.lm_eval.lm_task_extractors.glue import GLUEExtractor
from wisent.extractors.lm_eval.lm_task_extractors.gpqa import GPQAExtractor
from wisent.extractors.lm_eval.lm_task_extractors.gsm8k import GSM8KExtractor
from wisent.extractors.lm_eval.lm_task_extractors.headqa import HeadQAExtractor
from wisent.extractors.lm_eval.lm_task_extractors.hellaswag import HellaSwagExtractor
from wisent.extractors.lm_eval.lm_task_extractors.hle import HLEExtractor
from wisent.extractors.lm_eval.lm_task_extractors.lambada import LambadaExtractor
from wisent.extractors.lm_eval.lm_task_extractors.logiqa import LogiQAExtractor
from wisent.extractors.lm_eval.lm_task_extractors.logiqa2 import LogiQA2Extractor
from wisent.extractors.lm_eval.lm_task_extractors.mathqa import MathQAExtractor
from wisent.extractors.lm_eval.lm_task_extractors.medqa import MedQAExtractor
from wisent.extractors.lm_eval.lm_task_extractors.mmlu import MMLUExtractor
from wisent.extractors.lm_eval.lm_task_extractors.mrpc import MRPCExtractor
from wisent.extractors.lm_eval.lm_task_extractors.multilingual import MultilingualExtractor
from wisent.extractors.lm_eval.lm_task_extractors.multirc import MultiRCExtractor
from wisent.extractors.lm_eval.lm_task_extractors.mutual import MutualExtractor
from wisent.extractors.lm_eval.lm_task_extractors.nq_open import NQOpenExtractor
from wisent.extractors.lm_eval.lm_task_extractors.openbookqa import OpenBookQAExtractor
from wisent.extractors.lm_eval.lm_task_extractors.pawsx import PawsXExtractor
from wisent.extractors.lm_eval.lm_task_extractors.piqa import PIQAExtractor
from wisent.extractors.lm_eval.lm_task_extractors.prost import ProstExtractor
from wisent.extractors.lm_eval.lm_task_extractors.pubmedqa import PubMedQAExtractor
from wisent.extractors.lm_eval.lm_task_extractors.qa4mre import QA4MREExtractor
from wisent.extractors.lm_eval.lm_task_extractors.qasper import QasperExtractor
from wisent.extractors.lm_eval.lm_task_extractors.qnli import QNLIExtractor
from wisent.extractors.lm_eval.lm_task_extractors.qqp import QQPExtractor
from wisent.extractors.lm_eval.lm_task_extractors.quac import QuACExtractor
from wisent.extractors.lm_eval.lm_task_extractors.race import RACEExtractor
from wisent.extractors.lm_eval.lm_task_extractors.rte import RTEExtractor
from wisent.extractors.lm_eval.lm_task_extractors.sciq import SciQExtractor
from wisent.extractors.lm_eval.lm_task_extractors.social_iqa import SocialIQAExtractor
from wisent.extractors.lm_eval.lm_task_extractors.squad2 import SQuAD2Extractor
from wisent.extractors.lm_eval.lm_task_extractors.sst2 import SST2Extractor
from wisent.extractors.lm_eval.lm_task_extractors.superglue import SuperGLUEExtractor
from wisent.extractors.lm_eval.lm_task_extractors.supergpqa import SuperGPQAExtractor
from wisent.extractors.lm_eval.lm_task_extractors.swag import SwagExtractor
from wisent.extractors.lm_eval.lm_task_extractors.triviaqa import TriviaQAExtractor
from wisent.extractors.lm_eval.lm_task_extractors.truthfulqa_mc1 import TruthfulQAMC1Extractor
from wisent.extractors.lm_eval.lm_task_extractors.truthfulqa_mc2 import TruthfulQAMC2Extractor
from wisent.extractors.lm_eval.lm_task_extractors.webqs import WebQSExtractor
from wisent.extractors.lm_eval.lm_task_extractors.wic import WiCExtractor
from wisent.extractors.lm_eval.lm_task_extractors.wikitext import WikitextExtractor
from wisent.extractors.lm_eval.lm_task_extractors.winogrande import WinograndeExtractor
from wisent.extractors.lm_eval.lm_task_extractors.wnli import WNLIExtractor
from wisent.extractors.lm_eval.lm_task_extractors.wsc import WSCExtractor
from wisent.extractors.lm_eval.lm_task_extractors.xnli import XNLIExtractor
from wisent.extractors.lm_eval.lm_task_extractors.xstorycloze import XStoryClozeExtractor
from wisent.extractors.lm_eval.lm_task_extractors.xwinograd import XWinogradExtractor

__all__ = [
    "AfrimgsmExtractor",
    "AgievalExtractor",
    "AgievalMathExtractor",
    "AgievalLogiQAExtractor",
    "ModelWrittenEvalsExtractor",
    "AI2ARCExtractor",
    "ANLIExtractor",
    "ArcChallengeExtractor",
    "ArcEasyExtractor",
    "ArithmeticExtractor",
    "ASDivExtractor",
    "BigBenchExtractor",
    "BoolQExtractor",
    "CBExtractor",
    "COPAExtractor",
    "CoQAExtractor",
    "DropExtractor",
    "GaokaoExtractor",
    "GLUEExtractor",
    "GPQAExtractor",
    "GSM8KExtractor",
    "HeadQAExtractor",
    "HellaSwagExtractor",
    "HLEExtractor",
    "LambadaExtractor",
    "LogiQAExtractor",
    "LogiQA2Extractor",
    "MathQAExtractor",
    "MedQAExtractor",
    "MMLUExtractor",
    "MRPCExtractor",
    "MultilingualExtractor",
    "MultiRCExtractor",
    "MutualExtractor",
    "NQOpenExtractor",
    "OpenBookQAExtractor",
    "PawsXExtractor",
    "PIQAExtractor",
    "ProstExtractor",
    "PubMedQAExtractor",
    "QA4MREExtractor",
    "QasperExtractor",
    "QNLIExtractor",
    "QQPExtractor",
    "QuACExtractor",
    "RACEExtractor",
    "RecordExtractor",
    "RTEExtractor",
    "SciQExtractor",
    "SocialIQAExtractor",
    "SQuAD2Extractor",
    "SST2Extractor",
    "SuperGLUEExtractor",
    "SuperGPQAExtractor",
    "SwagExtractor",
    "TriviaQAExtractor",
    "TruthfulQAMC1Extractor",
    "TruthfulQAMC2Extractor",
    "TwentyNewsgroupsExtractor",
    "WebQSExtractor",
    "WiCExtractor",
    "WikitextExtractor",
    "WinograndeExtractor",
    "WNLIExtractor",
    "WSCExtractor",
    "XNLIExtractor",
    "XStoryClozeExtractor",
    "XWinogradExtractor",
]
