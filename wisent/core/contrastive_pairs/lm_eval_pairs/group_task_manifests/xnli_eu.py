"""Xnli eu group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

XNLI_EU_TASKS = {
    "xnli_eu": f"{BASE_IMPORT}xnli:XNLIExtractor",
    "xnli_eu_mt": f"{BASE_IMPORT}xnli:XNLIExtractor",
    "xnli_eu_mt_native": f"{BASE_IMPORT}xnli:XNLIExtractor",
    "xnli_eu_native": f"{BASE_IMPORT}xnli:XNLIExtractor",
}
