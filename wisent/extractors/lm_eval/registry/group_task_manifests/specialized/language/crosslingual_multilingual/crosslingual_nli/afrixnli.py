"""Afrixnli group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

AFRIXNLI_TASKS = {
    "afrixnli": f"{BASE_IMPORT}afrixnli:AfrixnliExtractor",
}
