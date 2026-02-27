"""Wmdp group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

WMDP_TASKS = {
    "wmdp": f"{BASE_IMPORT}wmdp:WmdpExtractor",
    "wmdp_bio": f"{BASE_IMPORT}wmdp:WmdpExtractor",
    "wmdp_chem": f"{BASE_IMPORT}wmdp:WmdpExtractor",
    "wmdp_cyber": f"{BASE_IMPORT}wmdp:WmdpExtractor",
}
