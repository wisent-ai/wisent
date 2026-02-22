"""Xquad group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

XQUAD_TASKS = {
    "xquad": f"{BASE_IMPORT}xquad:XquadExtractor",
    "xquad_ar": f"{BASE_IMPORT}xquad:XquadExtractor",
    "xquad_ca": f"{BASE_IMPORT}xquad:XquadExtractor",
    "xquad_de": f"{BASE_IMPORT}xquad:XquadExtractor",
    "xquad_el": f"{BASE_IMPORT}xquad:XquadExtractor",
    "xquad_en": f"{BASE_IMPORT}xquad:XquadExtractor",
    "xquad_es": f"{BASE_IMPORT}xquad:XquadExtractor",
    "xquad_hi": f"{BASE_IMPORT}xquad:XquadExtractor",
    "xquad_ro": f"{BASE_IMPORT}xquad:XquadExtractor",
    "xquad_ru": f"{BASE_IMPORT}xquad:XquadExtractor",
    "xquad_th": f"{BASE_IMPORT}xquad:XquadExtractor",
    "xquad_tr": f"{BASE_IMPORT}xquad:XquadExtractor",
    "xquad_vi": f"{BASE_IMPORT}xquad:XquadExtractor",
    "xquad_zh": f"{BASE_IMPORT}xquad:XquadExtractor",
}
