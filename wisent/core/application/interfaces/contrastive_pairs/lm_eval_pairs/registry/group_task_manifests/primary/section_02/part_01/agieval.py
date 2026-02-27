"""Agieval group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

AGIEVAL_TASKS = {
    "agieval": f"{BASE_IMPORT}agieval:AgievalExtractor",
    "agieval_aqua_rat": f"{BASE_IMPORT}agieval:AgievalExtractor",
    "agieval_cn": f"{BASE_IMPORT}agieval:AgievalExtractor",
    "agieval_en": f"{BASE_IMPORT}agieval:AgievalExtractor",
    "agieval_gaokao_biology": f"{BASE_IMPORT}agieval:AgievalExtractor",
    "agieval_gaokao_chemistry": f"{BASE_IMPORT}agieval:AgievalExtractor",
    "agieval_gaokao_chinese": f"{BASE_IMPORT}agieval:AgievalExtractor",
    "agieval_gaokao_english": f"{BASE_IMPORT}agieval:AgievalExtractor",
    "agieval_gaokao_geography": f"{BASE_IMPORT}agieval:AgievalExtractor",
    "agieval_gaokao_history": f"{BASE_IMPORT}agieval:AgievalExtractor",
    "agieval_gaokao_mathcloze": f"{BASE_IMPORT}agieval:AgievalExtractor",
    "agieval_gaokao_mathqa": f"{BASE_IMPORT}agieval:AgievalExtractor",
    "agieval_gaokao_physics": f"{BASE_IMPORT}agieval:AgievalExtractor",
    "agieval_jec_qa_ca": f"{BASE_IMPORT}agieval:AgievalExtractor",
    "agieval_jec_qa_kd": f"{BASE_IMPORT}agieval:AgievalExtractor",
    "agieval_logiqa_en": f"{BASE_IMPORT}agieval:AgievalExtractor",
    "agieval_logiqa_zh": f"{BASE_IMPORT}agieval:AgievalExtractor",
    "agieval_lsat_ar": f"{BASE_IMPORT}agieval:AgievalExtractor",
    "agieval_lsat_lr": f"{BASE_IMPORT}agieval:AgievalExtractor",
    "agieval_lsat_rc": f"{BASE_IMPORT}agieval:AgievalExtractor",
    "agieval_math": f"{BASE_IMPORT}agieval:AgievalExtractor",
    "agieval_nous": f"{BASE_IMPORT}agieval:AgievalExtractor",
    "agieval_sat_en": f"{BASE_IMPORT}agieval:AgievalExtractor",
    "agieval_sat_en_without_passage": f"{BASE_IMPORT}agieval:AgievalExtractor",
    "agieval_sat_math": f"{BASE_IMPORT}agieval:AgievalExtractor",
}
