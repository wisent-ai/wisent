"""Wmt16 group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

WMT16_TASKS = {
    "wmt16": f"{BASE_IMPORT}wmt16:Wmt16Extractor",
    "wmt2016": f"{BASE_IMPORT}wmt16:Wmt16Extractor",
    "wmt16-de-en": f"{BASE_IMPORT}wmt16:Wmt16Extractor",
    "wmt16-en-de": f"{BASE_IMPORT}wmt16:Wmt16Extractor",
    "wmt16-en-ro": f"{BASE_IMPORT}wmt16:Wmt16Extractor",
    "wmt16-ro-en": f"{BASE_IMPORT}wmt16:Wmt16Extractor",
    "wmt16_de_en": f"{BASE_IMPORT}wmt16:Wmt16Extractor",
    "wmt16_en_de": f"{BASE_IMPORT}wmt16:Wmt16Extractor",
    "wmt16_en_ro": f"{BASE_IMPORT}wmt16:Wmt16Extractor",
    "wmt16_ro_en": f"{BASE_IMPORT}wmt16:Wmt16Extractor",
    "wmt16_de-en": f"{BASE_IMPORT}wmt16:Wmt16Extractor",
    "wmt16_en-de": f"{BASE_IMPORT}wmt16:Wmt16Extractor",
    "wmt16_en-ro": f"{BASE_IMPORT}wmt16:Wmt16Extractor",
    "wmt16_ro-en": f"{BASE_IMPORT}wmt16:Wmt16Extractor",
}
