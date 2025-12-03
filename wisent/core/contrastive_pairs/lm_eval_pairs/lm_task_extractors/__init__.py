"""LM-Eval benchmark extractors."""

from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.ai2_arc import AI2ARCExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.anli import ANLIExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.arc_challenge import ArcChallengeExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.arc_easy import ArcEasyExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.arithmetic import ArithmeticExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.asdiv import ASDivExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.bigbench import BigBenchExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.boolq import BoolQExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.cb import CBExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.coqa import CoQAExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.drop import DropExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.glue import GLUEExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.gpqa import GPQAExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.gsm8k import GSM8KExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.headqa import HeadQAExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.hellaswag import HellaSwagExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.hle import HLEExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.lambada import LambadaExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.logiqa import LogiQAExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.logiqa2 import LogiQA2Extractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.mathqa import MathQAExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.medqa import MedQAExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.mmlu import MMLUExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.mrpc import MRPCExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.multilingual import MultilingualExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.multirc import MultiRCExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.mutual import MutualExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.nq_open import NQOpenExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.openbookqa import OpenBookQAExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.pawsx import PawsXExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.piqa import PIQAExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.prost import ProstExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.pubmedqa import PubMedQAExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.qa4mre import QA4MREExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.qasper import QasperExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.qnli import QNLIExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.qqp import QQPExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.quac import QuACExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.race import RACEExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.rte import RTEExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.sciq import SciQExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.social_iqa import SocialIQAExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.squad2 import SQuAD2Extractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.sst2 import SST2Extractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.superglue import SuperGLUEExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.supergpqa import SuperGPQAExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.swag import SwagExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.triviaqa import TriviaQAExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.truthfulqa_mc1 import TruthfulQAMC1Extractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.truthfulqa_mc2 import TruthfulQAMC2Extractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.webqs import WebQSExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.wic import WiCExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.wikitext import WikitextExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.winogrande import WinograndeExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.wnli import WNLIExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.wsc import WSCExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.xnli import XNLIExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.xstorycloze import XStoryClozeExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.xwinograd import XWinogradExtractor

__all__ = [
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
