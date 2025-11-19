"""Hendrycks ethics group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

HENDRYCKS_ETHICS_TASKS = {
    "hendrycks_ethics": f"{BASE_IMPORT}hendrycks_ethics:HendrycksEthicsExtractor",
    "ethics_cm": f"{BASE_IMPORT}hendrycks_ethics:HendrycksEthicsExtractor",
    "ethics_deontology": f"{BASE_IMPORT}hendrycks_ethics:HendrycksEthicsExtractor",
    "ethics_justice": f"{BASE_IMPORT}hendrycks_ethics:HendrycksEthicsExtractor",
    "ethics_utilitarianism": f"{BASE_IMPORT}hendrycks_ethics:HendrycksEthicsExtractor",
    "ethics_virtue": f"{BASE_IMPORT}hendrycks_ethics:HendrycksEthicsExtractor",
}
