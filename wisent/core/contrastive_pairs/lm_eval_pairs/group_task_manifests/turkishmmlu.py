"""TurkishMMLU group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

TURKISHMMLU_TASKS = {
    "turkishmmlu": f"{BASE_IMPORT}turkishmmlu:TurkishmmluExtractor",
    "turkishmmlu_biology": f"{BASE_IMPORT}turkishmmlu:TurkishmmluExtractor",
    "turkishmmlu_chemistry": f"{BASE_IMPORT}turkishmmlu:TurkishmmluExtractor",
    "turkishmmlu_geography": f"{BASE_IMPORT}turkishmmlu:TurkishmmluExtractor",
    "turkishmmlu_history": f"{BASE_IMPORT}turkishmmlu:TurkishmmluExtractor",
    "turkishmmlu_mathematics": f"{BASE_IMPORT}turkishmmlu:TurkishmmluExtractor",
    "turkishmmlu_philosophy": f"{BASE_IMPORT}turkishmmlu:TurkishmmluExtractor",
    "turkishmmlu_physics": f"{BASE_IMPORT}turkishmmlu:TurkishmmluExtractor",
    "turkishmmlu_religion_and_ethics": f"{BASE_IMPORT}turkishmmlu:TurkishmmluExtractor",
    "turkishmmlu_turkish_language_and_literature": f"{BASE_IMPORT}turkishmmlu:TurkishmmluExtractor",
}
