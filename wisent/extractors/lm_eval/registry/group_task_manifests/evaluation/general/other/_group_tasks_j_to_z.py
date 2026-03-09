"""Group task imports - Part 2 (J-Z benchmarks)."""
from __future__ import annotations

from .inverse import INVERSE_TASKS
from .japanese_leaderboard import JAPANESE_LEADERBOARD_TASKS
from .jsonschema_bench import JSONSCHEMA_BENCH_TASKS
from .kbl import KBL_TASKS
from .kmmlu import KMMLU_TASKS
from .kobest import KOBEST_TASKS
from .kormedmcqa import KORMEDMCQA_TASKS
from .lambada import LAMBADA_TASKS
from .leaderboard import LEADERBOARD_TASKS
from .libra import LIBRA_TASKS
from .lingoly import LINGOLY_TASKS
from .longbench import LONGBENCH_TASKS
from .m import M_TASKS
from .mastermind import MASTERMIND_TASKS
from .med import MED_TASKS
from .meddialog import MEDDIALOG_TASKS
from .medqa import MEDQA_TASKS
from .mela import MELA_TASKS
from .metabench import METABENCH_TASKS
from .mgsm import MGSM_TASKS
from .minerva_math import MINERVA_MATH_TASKS
from .mlqa import MLQA_TASKS
from .mmlu import MMLU_TASKS
from .mmlu_pro import MMLU_PRO_TASKS
from .mmlu_pro_plus import MMLU_PRO_PLUS_TASKS
from .mmlu_prox import MMLU_PROX_TASKS
from .mmlusr import MMLUSR_TASKS
from .mmmu import MMMU_TASKS
from .model_written_evals import MODEL_WRITTEN_EVALS_TASKS
from .multiblimp import MULTIBLIMP_TASKS
from .non import NON_TASKS
from .noreval import NOREVAL_TASKS
from .noridiom import NORIDIOM_TASKS
from .nortruthfulqa import NORTRUTHFULQA_TASKS
from .nrk import NRK_TASKS
from .okapi import OKAPI_TASKS
from .okapi_arc_multilingual import OKAPI_ARC_MULTILINGUAL_TASKS
from .okapi_hellaswag_multilingual import OKAPI_HELLASWAG_MULTILINGUAL_TASKS
from .okapi_mmlu_multilingual import OKAPI_MMLU_MULTILINGUAL_TASKS
from .okapi_truthfulqa_multilingual import OKAPI_TRUTHFULQA_MULTILINGUAL_TASKS
from .paloma import PALOMA_TASKS
from .pawsx import PAWSX_TASKS
from .persona import PERSONA_TASKS
from .pile import PILE_TASKS
from .polemo2 import POLEMO2_TASKS
from .portuguese_bench import PORTUGUESE_BENCH_TASKS
from .prompt import PROMPT_TASKS
from .qa4mre import QA4MRE_TASKS
from .qasper import QASPER_TASKS
from .ru import RU_TASKS
from .ruler import RULER_TASKS
from .score import SCORE_TASKS
from .scrolls import SCROLLS_TASKS
from .self_consistency import SELF_CONSISTENCY_TASKS
from .spanish_bench import SPANISH_BENCH_TASKS
from .storycloze import STORYCLOZE_TASKS
from .tinyBenchmarks import TINYBENCHMARKS_TASKS
from .tmlu import TMLU_TASKS
from .tmmluplus import TMMLUPLUS_TASKS
from .translation import TRANSLATION_TASKS
from .truthfulqa_multi import TRUTHFULQA_MULTI_TASKS
from .truthfulqa import TRUTHFULQA_TASKS
from .turkishmmlu import TURKISHMMLU_TASKS
from .unscramble import UNSCRAMBLE_TASKS
from .unitxt import UNITXT_TASKS
from .winogender import WINOGENDER_TASKS
from .wmdp import WMDP_TASKS
from .wmt14 import WMT14_TASKS
from .wmt16 import WMT16_TASKS
from .wsc273 import WSC273_TASKS
from .xcopa import XCOPA_TASKS
from .xnli import XNLI_TASKS
from .xnli_eu import XNLI_EU_TASKS
from .xquad import XQUAD_TASKS
from .xstorycloze import XSTORYCLOZE_TASKS
from .xwinograd import XWINOGRAD_TASKS
from .super_glue_t5_prompt import SUPER_GLUE_T5_PROMPT_TASKS


def get_j_to_z_mappings() -> dict[str, str]:
    """Get group task mappings for Part 2 benchmarks."""
    all_mappings = {}
    all_mappings.update(JAPANESE_LEADERBOARD_TASKS)
    all_mappings.update(JSONSCHEMA_BENCH_TASKS)
    all_mappings.update(KBL_TASKS)
    all_mappings.update(KMMLU_TASKS)
    all_mappings.update(KOBEST_TASKS)
    all_mappings.update(KORMEDMCQA_TASKS)
    all_mappings.update(LAMBADA_TASKS)
    all_mappings.update(LEADERBOARD_TASKS)
    all_mappings.update(LIBRA_TASKS)
    all_mappings.update(LINGOLY_TASKS)
    all_mappings.update(LONGBENCH_TASKS)
    all_mappings.update(M_TASKS)
    all_mappings.update(MASTERMIND_TASKS)
    all_mappings.update(MED_TASKS)
    all_mappings.update(MEDDIALOG_TASKS)
    all_mappings.update(MEDQA_TASKS)
    all_mappings.update(MELA_TASKS)
    all_mappings.update(METABENCH_TASKS)
    all_mappings.update(MGSM_TASKS)
    all_mappings.update(MINERVA_MATH_TASKS)
    all_mappings.update(MLQA_TASKS)
    all_mappings.update(MMLU_TASKS)
    all_mappings.update(MMLU_PRO_TASKS)
    all_mappings.update(MMLU_PRO_PLUS_TASKS)
    all_mappings.update(MMLU_PROX_TASKS)
    all_mappings.update(MMLUSR_TASKS)
    all_mappings.update(MMMU_TASKS)
    all_mappings.update(MODEL_WRITTEN_EVALS_TASKS)
    all_mappings.update(MULTIBLIMP_TASKS)
    all_mappings.update(NON_TASKS)
    all_mappings.update(NOREVAL_TASKS)
    all_mappings.update(NORIDIOM_TASKS)
    all_mappings.update(NORTRUTHFULQA_TASKS)
    all_mappings.update(NRK_TASKS)
    all_mappings.update(OKAPI_TASKS)
    all_mappings.update(OKAPI_ARC_MULTILINGUAL_TASKS)
    all_mappings.update(OKAPI_HELLASWAG_MULTILINGUAL_TASKS)
    all_mappings.update(OKAPI_MMLU_MULTILINGUAL_TASKS)
    all_mappings.update(OKAPI_TRUTHFULQA_MULTILINGUAL_TASKS)
    all_mappings.update(PALOMA_TASKS)
    all_mappings.update(PAWSX_TASKS)
    all_mappings.update(PERSONA_TASKS)
    all_mappings.update(PILE_TASKS)
    all_mappings.update(POLEMO2_TASKS)
    all_mappings.update(PORTUGUESE_BENCH_TASKS)
    all_mappings.update(PROMPT_TASKS)
    all_mappings.update(QA4MRE_TASKS)
    all_mappings.update(QASPER_TASKS)
    all_mappings.update(RU_TASKS)
    all_mappings.update(RULER_TASKS)
    all_mappings.update(SCORE_TASKS)
    all_mappings.update(SCROLLS_TASKS)
    all_mappings.update(SELF_CONSISTENCY_TASKS)
    all_mappings.update(SPANISH_BENCH_TASKS)
    all_mappings.update(STORYCLOZE_TASKS)
    all_mappings.update(TINYBENCHMARKS_TASKS)
    all_mappings.update(TMLU_TASKS)
    all_mappings.update(TMMLUPLUS_TASKS)
    all_mappings.update(TRANSLATION_TASKS)
    all_mappings.update(TRUTHFULQA_MULTI_TASKS)
    all_mappings.update(TRUTHFULQA_TASKS)
    all_mappings.update(TURKISHMMLU_TASKS)
    all_mappings.update(UNSCRAMBLE_TASKS)
    all_mappings.update(UNITXT_TASKS)
    all_mappings.update(WINOGENDER_TASKS)
    all_mappings.update(WMDP_TASKS)
    all_mappings.update(WMT14_TASKS)
    all_mappings.update(WMT16_TASKS)
    all_mappings.update(WSC273_TASKS)
    all_mappings.update(XCOPA_TASKS)
    all_mappings.update(XNLI_TASKS)
    all_mappings.update(XNLI_EU_TASKS)
    all_mappings.update(XQUAD_TASKS)
    all_mappings.update(XSTORYCLOZE_TASKS)
    all_mappings.update(XWINOGRAD_TASKS)
    all_mappings.update(SUPER_GLUE_T5_PROMPT_TASKS)
    return all_mappings
