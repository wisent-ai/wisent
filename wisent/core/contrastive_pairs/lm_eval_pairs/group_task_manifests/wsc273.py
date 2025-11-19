"""Wsc273 group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

WSC273_TASKS = {
    "wsc273": f"{BASE_IMPORT}wsc273:Wsc273Extractor",
}
