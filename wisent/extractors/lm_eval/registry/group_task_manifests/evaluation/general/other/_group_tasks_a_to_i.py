"""Group task imports - Part 1 (A-I benchmarks)."""
from __future__ import annotations

from .aclue import ACLUE_TASKS
from .acp import ACP_TASKS
from .advanced_ai_risk import ADVANCED_AI_RISK_TASKS
from .acpbench import ACPBENCH_TASKS
from .aexams import AEXAMS_TASKS
from .afrimgsm import AFRIMGSM_TASKS
from .afrimmlu import AFRIMMLU_TASKS
from .inverse import INVERSE_TASKS
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


def get_a_to_i_mappings() -> dict[str, str]:
    """Get group task mappings for Part 1 benchmarks."""
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
    # FLORES_TASKS contains broken refs to flores:FloresExtractor (HF only); register
    # FIRST so that BASQUE_BENCH_TASKS, CATALAN_BENCH_TASKS, GALICIAN_BENCH_TASKS etc.
    # can override flores_<lang> entries with their working bench-specific extractors.
    all_mappings.update(FLORES_TASKS)
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
    return all_mappings
