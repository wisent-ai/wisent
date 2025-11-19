"""Libra group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

LIBRA_TASKS = {
    "libra": f"{BASE_IMPORT}libra:LibraExtractor",
}
