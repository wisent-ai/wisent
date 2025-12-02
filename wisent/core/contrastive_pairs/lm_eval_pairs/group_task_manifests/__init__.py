"""Group task manifests for LM Eval benchmarks with multiple subtasks."""

from __future__ import annotations

# Import all group task manifests
from .aclue import ACLUE_TASKS
from .acp import ACP_TASKS
from .advanced_ai_risk import ADVANCED_AI_RISK_TASKS
from .acpbench import ACPBENCH_TASKS
from .aexams import AEXAMS_TASKS
from .afrimgsm import AFRIMGSM_TASKS
from .afrimmlu import AFRIMMLU_TASKS
from .afrixnli import AFRIXNLI_TASKS
from .afrobench import AFROBENCH_TASKS
from .afrobench_adr import AFROBENCH_ADR_TASKS
from .afrobench_afriqa import AFROBENCH_AFRIQA_TASKS
from .afrobench_afrisenti import AFROBENCH_AFRISENTI_TASKS
from .afrobench_belebele import AFROBENCH_BELEBELE_TASKS
from .afrobench_flores import AFROBENCH_FLORES_TASKS
from .afrobench_injongointent import AFROBENCH_INJONGOINTENT_TASKS
from .afrobench_mafand import AFROBENCH_MAFAND_TASKS
from .afrobench_masakhaner import AFROBENCH_MASAKHANER_TASKS
from .afrobench_masakhanews import AFROBENCH_MASAKHANEWS_TASKS
from .afrobench_masakhapos import AFROBENCH_MASAKHAPOS_TASKS
from .afrobench_naijarc import AFROBENCH_NAIJARC_TASKS
from .afrobench_nollysenti import AFROBENCH_NOLLYSENTI_TASKS
from .afrobench_ntrex import AFROBENCH_NTREX_TASKS
from .afrobench_openai_mmlu import AFROBENCH_OPENAI_MMLU_TASKS
from .afrobench_salt import AFROBENCH_SALT_TASKS
from .afrobench_sib import AFROBENCH_SIB_TASKS
from .afrobench_uhura_arc_easy import AFROBENCH_UHURA_ARC_EASY_TASKS
from .afrobench_xlsum import AFROBENCH_XLSUM_TASKS
from .agieval import AGIEVAL_TASKS
from .anli import ANLI_TASKS
from .arab_culture import ARAB_CULTURE_TASKS
from .arabic_leaderboard_acva import ARABIC_LEADERBOARD_ACVA_TASKS
from .arabic_leaderboard_acva_light import ARABIC_LEADERBOARD_ACVA_LIGHT_TASKS
from .arabic_leaderboard_complete import ARABIC_LEADERBOARD_COMPLETE_TASKS
from .arabic_leaderboard_light import ARABIC_LEADERBOARD_LIGHT_TASKS
from .arabicmmlu import ARABICMMLU_TASKS
from .aradice import ARADICE_TASKS
from .arc import ARC_TASKS
from .arithmetic import ARITHMETIC_TASKS
from .basque_bench import BASQUE_BENCH_TASKS
from .bbh import BBH_TASKS
from .bbq import BBQ_TASKS
from .belebele import BELEBELE_TASKS
from .bertaqa import BERTAQA_TASKS
from .bigbench import BIGBENCH_TASKS
from .blimp import BLIMP_TASKS
from .careqa import CAREQA_TASKS
from .catalan_bench import CATALAN_BENCH_TASKS
from .ceval_valid import CEVAL_VALID_TASKS
from .cmmlu import CMMLU_TASKS
from .code_x_glue import CODE_X_GLUE_TASKS
from .copal_id import COPAL_ID_TASKS
from .crows_pairs import CROWS_PAIRS_TASKS
from .csatqa import CSATQA_TASKS
from .darija import DARIJA_TASKS
from .darijammlu import DARIJAMMLU_TASKS
from .egymmlu import EGYMMLU_TASKS
from .eus import EUS_TASKS
from .evalita_mp import EVALITA_MP_TASKS
from .fld import FLD_TASKS
from .flores import FLORES_TASKS
from .freebase import FREEBASE_TASKS
from .french_bench import FRENCH_BENCH_TASKS
from .galician_bench import GALICIAN_BENCH_TASKS
from .glianorex import GLIANOREX_TASKS
from .global_mmlu import GLOBAL_TASKS
from .gpqa import GPQA_TASKS
from .gsm8k import GSM8K_TASKS
from .gsm8k_platinum import GSM8K_PLATINUM_TASKS
from .haerae import HAERAE_TASKS
from .headqa import HEADQA_TASKS
from .hellaswag import HELLASWAG_TASKS
from .hendrycks_ethics import HENDRYCKS_ETHICS_TASKS
from .hendrycks_math import HENDRYCKS_MATH_TASKS
from .hrm8k import HRM8K_TASKS
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

__all__ = [
    "ACLUE_TASKS",
    "ACP_TASKS",
    "ADVANCED_AI_RISK_TASKS",
    "ACPBENCH_TASKS",
    "AEXAMS_TASKS",
    "AFRIMGSM_TASKS",
    "AFRIMMLU_TASKS",
    "AFRIXNLI_TASKS",
    "AFROBENCH_TASKS",
    "AFROBENCH_ADR_TASKS",
    "AFROBENCH_AFRIQA_TASKS",
    "AFROBENCH_AFRISENTI_TASKS",
    "AFROBENCH_BELEBELE_TASKS",
    "AFROBENCH_FLORES_TASKS",
    "AFROBENCH_INJONGOINTENT_TASKS",
    "AFROBENCH_MAFAND_TASKS",
    "AFROBENCH_MASAKHANER_TASKS",
    "AFROBENCH_MASAKHANEWS_TASKS",
    "AFROBENCH_MASAKHAPOS_TASKS",
    "AFROBENCH_NAIJARC_TASKS",
    "AFROBENCH_NOLLYSENTI_TASKS",
    "AFROBENCH_NTREX_TASKS",
    "AFROBENCH_OPENAI_MMLU_TASKS",
    "AFROBENCH_SALT_TASKS",
    "AFROBENCH_SIB_TASKS",
    "AFROBENCH_UHURA_ARC_EASY_TASKS",
    "AFROBENCH_XLSUM_TASKS",
    "AGIEVAL_TASKS",
    "ANLI_TASKS",
    "ARAB_CULTURE_TASKS",
    "ARABIC_LEADERBOARD_ACVA_TASKS",
    "ARABIC_LEADERBOARD_ACVA_LIGHT_TASKS",
    "ARABIC_LEADERBOARD_COMPLETE_TASKS",
    "ARABIC_LEADERBOARD_LIGHT_TASKS",
    "ARABICMMLU_TASKS",
    "ARADICE_TASKS",
    "ARC_TASKS",
    "ARITHMETIC_TASKS",
    "BASQUE_BENCH_TASKS",
    "BBH_TASKS",
    "BBQ_TASKS",
    "BELEBELE_TASKS",
    "BERTAQA_TASKS",
    "BIGBENCH_TASKS",
    "BLIMP_TASKS",
    "CAREQA_TASKS",
    "CATALAN_BENCH_TASKS",
    "CEVAL_VALID_TASKS",
    "CMMLU_TASKS",
    "CODE_X_GLUE_TASKS",
    "COPAL_ID_TASKS",
    "CROWS_PAIRS_TASKS",
    "CSATQA_TASKS",
    "DARIJA_TASKS",
    "DARIJAMMLU_TASKS",
    "EGYMMLU_TASKS",
    "EUS_TASKS",
    "EVALITA_MP_TASKS",
    "FLD_TASKS",
    "FLORES_TASKS",
    "FREEBASE_TASKS",
    "FRENCH_BENCH_TASKS",
    "GALICIAN_BENCH_TASKS",
    "GLIANOREX_TASKS",
    "GLOBAL_TASKS",
    "GPQA_TASKS",
    "GSM8K_TASKS",
    "GSM8K_PLATINUM_TASKS",
    "HAERAE_TASKS",
    "HEADQA_TASKS",
    "HELLASWAG_TASKS",
    "HENDRYCKS_ETHICS_TASKS",
    "HENDRYCKS_MATH_TASKS",
    "HRM8K_TASKS",
    "INVERSE_TASKS",
    "JAPANESE_LEADERBOARD_TASKS",
    "JSONSCHEMA_BENCH_TASKS",
    "KBL_TASKS",
    "KMMLU_TASKS",
    "KOBEST_TASKS",
    "KORMEDMCQA_TASKS",
    "LAMBADA_TASKS",
    "LEADERBOARD_TASKS",
    "LIBRA_TASKS",
    "LINGOLY_TASKS",
    "LONGBENCH_TASKS",
    "M_TASKS",
    "MASTERMIND_TASKS",
    "MED_TASKS",
    "MEDDIALOG_TASKS",
    "MEDQA_TASKS",
    "MELA_TASKS",
    "METABENCH_TASKS",
    "MGSM_TASKS",
    "MINERVA_MATH_TASKS",
    "MLQA_TASKS",
    "MMLU_TASKS",
    "MMLU_PRO_TASKS",
    "MMLU_PRO_PLUS_TASKS",
    "MMLU_PROX_TASKS",
    "MMLUSR_TASKS",
    "MMMU_TASKS",
    "MODEL_WRITTEN_EVALS_TASKS",
    "MULTIBLIMP_TASKS",
    "NON_TASKS",
    "NOREVAL_TASKS",
    "NORIDIOM_TASKS",
    "NORTRUTHFULQA_TASKS",
    "NRK_TASKS",
    "OKAPI_TASKS",
    "OKAPI_ARC_MULTILINGUAL_TASKS",
    "OKAPI_HELLASWAG_MULTILINGUAL_TASKS",
    "OKAPI_MMLU_MULTILINGUAL_TASKS",
    "OKAPI_TRUTHFULQA_MULTILINGUAL_TASKS",
    "PALOMA_TASKS",
    "PAWSX_TASKS",
    "PERSONA_TASKS",
    "PILE_TASKS",
    "POLEMO2_TASKS",
    "PORTUGUESE_BENCH_TASKS",
    "PROMPT_TASKS",
    "QA4MRE_TASKS",
    "QASPER_TASKS",
    "RU_TASKS",
    "RULER_TASKS",
    "SCORE_TASKS",
    "SCROLLS_TASKS",
    "SELF_CONSISTENCY_TASKS",
    "SPANISH_BENCH_TASKS",
    "STORYCLOZE_TASKS",
    "TINYBENCHMARKS_TASKS",
    "TMLU_TASKS",
    "TMMLUPLUS_TASKS",
    "TRANSLATION_TASKS",
    "TRUTHFULQA_MULTI_TASKS",
    "TRUTHFULQA_TASKS",
    "TURKISHMMLU_TASKS",
    "UNSCRAMBLE_TASKS",
    "UNITXT_TASKS",
    "WINOGENDER_TASKS",
    "WMDP_TASKS",
    "WMT14_TASKS",
    "WMT16_TASKS",
    "XCOPA_TASKS",
    "XNLI_TASKS",
    "XNLI_EU_TASKS",
    "XQUAD_TASKS",
    "XSTORYCLOZE_TASKS",
    "XWINOGRAD_TASKS",
    "SUPER_GLUE_T5_PROMPT_TASKS",
    "get_all_group_task_mappings",
]


def get_all_group_task_mappings() -> dict[str, str]:
    """
    Get all group task to extractor mappings.

    Returns:
        Dictionary mapping task names to extractor module paths.
    """
    all_mappings = {}
    all_mappings.update(ACLUE_TASKS)
    all_mappings.update(ACP_TASKS)
    all_mappings.update(ADVANCED_AI_RISK_TASKS)
    all_mappings.update(ACPBENCH_TASKS)
    all_mappings.update(AEXAMS_TASKS)
    all_mappings.update(AFRIMGSM_TASKS)
    all_mappings.update(AFRIMMLU_TASKS)
    all_mappings.update(AFRIXNLI_TASKS)
    all_mappings.update(AFROBENCH_TASKS)
    all_mappings.update(AFROBENCH_ADR_TASKS)
    all_mappings.update(AFROBENCH_AFRIQA_TASKS)
    all_mappings.update(AFROBENCH_AFRISENTI_TASKS)
    all_mappings.update(AFROBENCH_BELEBELE_TASKS)
    all_mappings.update(AFROBENCH_FLORES_TASKS)
    all_mappings.update(AFROBENCH_INJONGOINTENT_TASKS)
    all_mappings.update(AFROBENCH_MAFAND_TASKS)
    all_mappings.update(AFROBENCH_MASAKHANER_TASKS)
    all_mappings.update(AFROBENCH_MASAKHANEWS_TASKS)
    all_mappings.update(AFROBENCH_MASAKHAPOS_TASKS)
    all_mappings.update(AFROBENCH_NAIJARC_TASKS)
    all_mappings.update(AFROBENCH_NOLLYSENTI_TASKS)
    all_mappings.update(AFROBENCH_NTREX_TASKS)
    all_mappings.update(AFROBENCH_OPENAI_MMLU_TASKS)
    all_mappings.update(AFROBENCH_SALT_TASKS)
    all_mappings.update(AFROBENCH_SIB_TASKS)
    all_mappings.update(AFROBENCH_UHURA_ARC_EASY_TASKS)
    all_mappings.update(AFROBENCH_XLSUM_TASKS)
    all_mappings.update(AGIEVAL_TASKS)
    all_mappings.update(ANLI_TASKS)
    all_mappings.update(ARAB_CULTURE_TASKS)
    all_mappings.update(ARABIC_LEADERBOARD_ACVA_TASKS)
    all_mappings.update(ARABIC_LEADERBOARD_ACVA_LIGHT_TASKS)
    all_mappings.update(ARABIC_LEADERBOARD_COMPLETE_TASKS)
    all_mappings.update(ARABIC_LEADERBOARD_LIGHT_TASKS)
    all_mappings.update(ARABICMMLU_TASKS)
    all_mappings.update(ARADICE_TASKS)
    all_mappings.update(ARC_TASKS)
    all_mappings.update(ARITHMETIC_TASKS)
    all_mappings.update(BASQUE_BENCH_TASKS)
    all_mappings.update(BBH_TASKS)
    all_mappings.update(BBQ_TASKS)
    all_mappings.update(BELEBELE_TASKS)
    all_mappings.update(BERTAQA_TASKS)
    all_mappings.update(BIGBENCH_TASKS)
    all_mappings.update(BLIMP_TASKS)
    all_mappings.update(CAREQA_TASKS)
    all_mappings.update(CATALAN_BENCH_TASKS)
    all_mappings.update(CEVAL_VALID_TASKS)
    all_mappings.update(CMMLU_TASKS)
    all_mappings.update(CODE_X_GLUE_TASKS)
    all_mappings.update(COPAL_ID_TASKS)
    all_mappings.update(CROWS_PAIRS_TASKS)
    all_mappings.update(CSATQA_TASKS)
    all_mappings.update(DARIJA_TASKS)
    all_mappings.update(DARIJAMMLU_TASKS)
    all_mappings.update(EGYMMLU_TASKS)
    all_mappings.update(EUS_TASKS)
    all_mappings.update(EVALITA_MP_TASKS)
    all_mappings.update(FLD_TASKS)
    all_mappings.update(FLORES_TASKS)
    all_mappings.update(FREEBASE_TASKS)
    all_mappings.update(FRENCH_BENCH_TASKS)
    all_mappings.update(GALICIAN_BENCH_TASKS)
    all_mappings.update(GLIANOREX_TASKS)
    all_mappings.update(GLOBAL_TASKS)
    all_mappings.update(GPQA_TASKS)
    all_mappings.update(GSM8K_TASKS)
    all_mappings.update(GSM8K_PLATINUM_TASKS)
    all_mappings.update(HAERAE_TASKS)
    all_mappings.update(HEADQA_TASKS)
    all_mappings.update(HELLASWAG_TASKS)
    all_mappings.update(HENDRYCKS_ETHICS_TASKS)
    all_mappings.update(HENDRYCKS_MATH_TASKS)
    all_mappings.update(HRM8K_TASKS)
    all_mappings.update(INVERSE_TASKS)
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
