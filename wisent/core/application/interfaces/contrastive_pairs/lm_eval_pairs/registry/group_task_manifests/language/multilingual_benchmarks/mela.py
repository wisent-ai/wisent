"""Mela group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

MELA_TASKS = {
    "mela": f"{BASE_IMPORT}mela:MelaExtractor",
    "mela_en": f"{BASE_IMPORT}mela:MelaExtractor",
    "mela_zh": f"{BASE_IMPORT}mela:MelaExtractor",
    "mela_it": f"{BASE_IMPORT}mela:MelaExtractor",
    "mela_ru": f"{BASE_IMPORT}mela:MelaExtractor",
    "mela_de": f"{BASE_IMPORT}mela:MelaExtractor",
    "mela_fr": f"{BASE_IMPORT}mela:MelaExtractor",
    "mela_es": f"{BASE_IMPORT}mela:MelaExtractor",
    "mela_ja": f"{BASE_IMPORT}mela:MelaExtractor",
    "mela_ar": f"{BASE_IMPORT}mela:MelaExtractor",
}
