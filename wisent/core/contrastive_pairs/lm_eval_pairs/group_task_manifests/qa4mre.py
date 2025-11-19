"""Qa4mre group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

QA4MRE_TASKS = {
    "qa4mre": f"{BASE_IMPORT}qa4mre:QA4MREExtractor",
    "qa4mre_2011": f"{BASE_IMPORT}qa4mre:QA4MREExtractor",
    "qa4mre_2012": f"{BASE_IMPORT}qa4mre:QA4MREExtractor",
    "qa4mre_2013": f"{BASE_IMPORT}qa4mre:QA4MREExtractor",
}
