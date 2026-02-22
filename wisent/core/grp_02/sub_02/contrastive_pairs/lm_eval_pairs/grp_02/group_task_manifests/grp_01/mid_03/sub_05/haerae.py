"""Haerae group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

HAERAE_TASKS = {
    "haerae": f"{BASE_IMPORT}haerae:HaeraeExtractor",
    "haerae_general_knowledge": f"{BASE_IMPORT}haerae:HaeraeExtractor",
    "haerae_history": f"{BASE_IMPORT}haerae:HaeraeExtractor",
    "haerae_loan_word": f"{BASE_IMPORT}haerae:HaeraeExtractor",
    "haerae_rare_word": f"{BASE_IMPORT}haerae:HaeraeExtractor",
    "haerae_standard_nomenclature": f"{BASE_IMPORT}haerae:HaeraeExtractor",
}
