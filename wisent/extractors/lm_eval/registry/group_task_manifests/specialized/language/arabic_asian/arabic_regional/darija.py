"""Darija group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

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
    # Darija-bench upstream typo: "trasnlation_*" subtasks
    "trasnlation_all_flores": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "trasnlation_dr_en_flores": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "trasnlation_dr_fr_flores": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "trasnlation_dr_msa_flores": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "trasnlation_en_dr_flores": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "trasnlation_fr_dr_flores": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "trasnlation_msa_dr_flores": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "trasnlation_dr_en_doda": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "trasnlation_en_dr_doda": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "trasnlation_msa_dr_madar": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
    "trasnlation_dr_msa_madar": f"{BASE_IMPORT}darija_bench:DarijaBenchExtractor",
}
