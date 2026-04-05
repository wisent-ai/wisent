"""Afrobench group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

AFROBENCH_TASKS = {
    "afrobench_afrisenti": f"{BASE_IMPORT}afrobench_mc:AfroBenchMultipleChoiceExtractor",
    "afrobench_belebele": f"{BASE_IMPORT}afrobench_mc:AfroBenchMultipleChoiceExtractor",
    "afrobench_injongointent": f"{BASE_IMPORT}afrobench_mc:AfroBenchMultipleChoiceExtractor",
    "afrobench_masakhanews": f"{BASE_IMPORT}afrobench_mc:AfroBenchMultipleChoiceExtractor",
    "afrobench_naijarc": f"{BASE_IMPORT}afrobench_mc:AfroBenchMultipleChoiceExtractor",
    "afrobench_nollysenti": f"{BASE_IMPORT}afrobench_mc:AfroBenchMultipleChoiceExtractor",
    "afrobench_openai_mmlu": f"{BASE_IMPORT}afrobench_mc:AfroBenchMultipleChoiceExtractor",
    "afrobench_sib": f"{BASE_IMPORT}afrobench_mc:AfroBenchMultipleChoiceExtractor",
    "afrobench_uhura-arc-easy": f"{BASE_IMPORT}afrobench_mc:AfroBenchMultipleChoiceExtractor",
    "afrobench_adr": f"{BASE_IMPORT}afrobench_cot:AfroBenchCotExtractor",
    "afrobench_afriqa": f"{BASE_IMPORT}afrobench_cot:AfroBenchCotExtractor",
    "afrobench_flores": f"{BASE_IMPORT}afrobench_cot:AfroBenchCotExtractor",
    "afrobench_mafand": f"{BASE_IMPORT}afrobench_cot:AfroBenchCotExtractor",
    "afrobench_masakhaner": f"{BASE_IMPORT}afrobench_cot:AfroBenchCotExtractor",
    "afrobench_masakhapos": f"{BASE_IMPORT}afrobench_cot:AfroBenchCotExtractor",
    "afrobench_ntrex": f"{BASE_IMPORT}afrobench_cot:AfroBenchCotExtractor",
    "afrobench_salt": f"{BASE_IMPORT}afrobench_cot:AfroBenchCotExtractor",
    "afrobench_xlsum": f"{BASE_IMPORT}afrobench_cot:AfroBenchCotExtractor",
}
