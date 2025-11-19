"""Darija group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

DARIJA_TASKS = {
    "darija_bench": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "darija_sentiment": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "darija_sentiment_mac": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "darija_sentiment_myc": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "darija_sentiment_msac": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "darija_sentiment_msda": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "darija_sentiment_electrom": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "darija_sentiment_tasks": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "darija_summarization": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "darija_summarization_task": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "darija_translation": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "darija_translation_doda": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "darija_translation_flores": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "darija_translation_madar": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "darija_translation_seed": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "darija_translation_tasks_doda": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "darija_translation_tasks_flores": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "darija_translation_tasks_madar": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "darija_translation_tasks_seed": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "darija_transliteration": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "darija_transliteration_tasks": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
}
