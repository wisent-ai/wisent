"""Wmt14 group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

WMT14_TASKS = {
    "wmt14": f"{BASE_IMPORT}wmt14:Wmt14Extractor",
    "wmt2014": f"{BASE_IMPORT}wmt14:Wmt14Extractor",
    "wmt14-en-fr": f"{BASE_IMPORT}wmt14:Wmt14Extractor",
    "wmt14-fr-en": f"{BASE_IMPORT}wmt14:Wmt14Extractor",
    "wmt14_en_fr": f"{BASE_IMPORT}wmt14:Wmt14Extractor",
    "wmt14_fr_en": f"{BASE_IMPORT}wmt14:Wmt14Extractor",
    "wmt14_en-fr": f"{BASE_IMPORT}wmt14:Wmt14Extractor",
    "wmt14_fr-en": f"{BASE_IMPORT}wmt14:Wmt14Extractor",
}
