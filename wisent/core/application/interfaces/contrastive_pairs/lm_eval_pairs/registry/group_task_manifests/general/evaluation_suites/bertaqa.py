"""Bertaqa group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

BERTAQA_TASKS = {
    "bertaqa": f"{BASE_IMPORT}bertaqa:BertaqaExtractor",
    "bertaqa_en": f"{BASE_IMPORT}bertaqa:BertaqaExtractor",
    "bertaqa_en_mt_gemma-7b": f"{BASE_IMPORT}bertaqa:BertaqaExtractor",
    "bertaqa_en_mt_hitz": f"{BASE_IMPORT}bertaqa:BertaqaExtractor",
    "bertaqa_en_mt_itzuli": f"{BASE_IMPORT}bertaqa:BertaqaExtractor",
    "bertaqa_en_mt_latxa-13b-v1": f"{BASE_IMPORT}bertaqa:BertaqaExtractor",
    "bertaqa_en_mt_latxa-13b-v1.1": f"{BASE_IMPORT}bertaqa:BertaqaExtractor",
    "bertaqa_en_mt_latxa-70b-v1": f"{BASE_IMPORT}bertaqa:BertaqaExtractor",
    "bertaqa_en_mt_latxa-70b-v1.1": f"{BASE_IMPORT}bertaqa:BertaqaExtractor",
    "bertaqa_en_mt_latxa-7b-v1": f"{BASE_IMPORT}bertaqa:BertaqaExtractor",
    "bertaqa_en_mt_latxa-7b-v1.1": f"{BASE_IMPORT}bertaqa:BertaqaExtractor",
    "bertaqa_en_mt_llama-2-13b": f"{BASE_IMPORT}bertaqa:BertaqaExtractor",
    "bertaqa_en_mt_llama-2-70b": f"{BASE_IMPORT}bertaqa:BertaqaExtractor",
    "bertaqa_en_mt_llama-2-7b": f"{BASE_IMPORT}bertaqa:BertaqaExtractor",
    "bertaqa_en_mt_madlad": f"{BASE_IMPORT}bertaqa:BertaqaExtractor",
    "bertaqa_en_mt_nllb": f"{BASE_IMPORT}bertaqa:BertaqaExtractor",
    "bertaqa_eu": f"{BASE_IMPORT}bertaqa:BertaqaExtractor",
}
