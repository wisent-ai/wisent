"""Xnli group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

XNLI_TASKS = {
    "xnli": f"{BASE_IMPORT}xnli:XNLIExtractor",
    "xnli_ar": f"{BASE_IMPORT}xnli:XNLIExtractor",
    "xnli_bg": f"{BASE_IMPORT}xnli:XNLIExtractor",
    "xnli_ca": f"{BASE_IMPORT}xnli:XNLIExtractor",
    "xnli_de": f"{BASE_IMPORT}xnli:XNLIExtractor",
    "xnli_el": f"{BASE_IMPORT}xnli:XNLIExtractor",
    "xnli_en": f"{BASE_IMPORT}xnli:XNLIExtractor",
    "xnli_es": f"{BASE_IMPORT}xnli:XNLIExtractor",
    "xnli_es_spanish_bench": f"{BASE_IMPORT}xnli:XNLIExtractor",
    "xnli_eu": f"{BASE_IMPORT}xnli:XNLIExtractor",
    "xnli_fr": f"{BASE_IMPORT}xnli:XNLIExtractor",
    "xnli_gl": f"{BASE_IMPORT}xnli:XNLIExtractor",
    "xnli_hi": f"{BASE_IMPORT}xnli:XNLIExtractor",
    "xnli_ru": f"{BASE_IMPORT}xnli:XNLIExtractor",
    "xnli_sw": f"{BASE_IMPORT}xnli:XNLIExtractor",
    "xnli_th": f"{BASE_IMPORT}xnli:XNLIExtractor",
    "xnli_tr": f"{BASE_IMPORT}xnli:XNLIExtractor",
    "xnli_ur": f"{BASE_IMPORT}xnli:XNLIExtractor",
    "xnli_vi": f"{BASE_IMPORT}xnli:XNLIExtractor",
    "xnli_zh": f"{BASE_IMPORT}xnli:XNLIExtractor",
}
