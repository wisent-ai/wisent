"""Lambada group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

LAMBADA_TASKS = {
    "lambada_openai": f"{BASE_IMPORT}lambada:LambadaExtractor",
    "lambada_standard": f"{BASE_IMPORT}lambada:LambadaExtractor",
    "lambada": f"{BASE_IMPORT}lambada:LambadaExtractor",
    "lambada_cloze": f"{BASE_IMPORT}lambada_cloze:LambadaClozeExtractor",
    "lambada_openai_cloze_yaml": f"{BASE_IMPORT}lambada_cloze:LambadaClozeExtractor",
    "lambada_standard_cloze_yaml": f"{BASE_IMPORT}lambada_cloze:LambadaClozeExtractor",
    "lambada_multilingual": f"{BASE_IMPORT}lambada_multilingual:LambadaMultilingualExtractor",
    "lambada_openai_mt_de": f"{BASE_IMPORT}lambada_multilingual:LambadaMultilingualExtractor",
    "lambada_openai_mt_en": f"{BASE_IMPORT}lambada_multilingual:LambadaMultilingualExtractor",
    "lambada_openai_mt_es": f"{BASE_IMPORT}lambada_multilingual:LambadaMultilingualExtractor",
    "lambada_openai_mt_fr": f"{BASE_IMPORT}lambada_multilingual:LambadaMultilingualExtractor",
    "lambada_openai_mt_it": f"{BASE_IMPORT}lambada_multilingual:LambadaMultilingualExtractor",
    "lambada_multilingual_stablelm": f"{BASE_IMPORT}lambada_multilingual_stablelm:LambadaMultilingualStablelmExtractor",
    "lambada_openai_mt_stablelm_de": f"{BASE_IMPORT}lambada_multilingual_stablelm:LambadaMultilingualStablelmExtractor",
    "lambada_openai_mt_stablelm_en": f"{BASE_IMPORT}lambada_multilingual_stablelm:LambadaMultilingualStablelmExtractor",
    "lambada_openai_mt_stablelm_es": f"{BASE_IMPORT}lambada_multilingual_stablelm:LambadaMultilingualStablelmExtractor",
    "lambada_openai_mt_stablelm_fr": f"{BASE_IMPORT}lambada_multilingual_stablelm:LambadaMultilingualStablelmExtractor",
    "lambada_openai_mt_stablelm_it": f"{BASE_IMPORT}lambada_multilingual_stablelm:LambadaMultilingualStablelmExtractor",
    "lambada_openai_mt_stablelm_nl": f"{BASE_IMPORT}lambada_multilingual_stablelm:LambadaMultilingualStablelmExtractor",
    "lambada_openai_mt_stablelm_pt": f"{BASE_IMPORT}lambada_multilingual_stablelm:LambadaMultilingualStablelmExtractor",
}
