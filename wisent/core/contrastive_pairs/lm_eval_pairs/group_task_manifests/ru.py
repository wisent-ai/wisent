"""Ru group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

RU_TASKS = {
    "ru_2wikimultihopqa": f"{BASE_IMPORT}libra:LibraExtractor",
    "ru_babilong_qa1": f"{BASE_IMPORT}libra:LibraExtractor",
    "ru_babilong_qa2": f"{BASE_IMPORT}libra:LibraExtractor",
    "ru_babilong_qa3": f"{BASE_IMPORT}libra:LibraExtractor",
    "ru_babilong_qa4": f"{BASE_IMPORT}libra:LibraExtractor",
    "ru_babilong_qa5": f"{BASE_IMPORT}libra:LibraExtractor",
    "ru_gsm100": f"{BASE_IMPORT}libra:LibraExtractor",
    "ru_qasper": f"{BASE_IMPORT}libra:LibraExtractor",
    "ru_quality": f"{BASE_IMPORT}libra:LibraExtractor",
    "ru_sci_abstract_retrieval": f"{BASE_IMPORT}libra:LibraExtractor",
    "ru_sci_passage_count": f"{BASE_IMPORT}libra:LibraExtractor",
}
