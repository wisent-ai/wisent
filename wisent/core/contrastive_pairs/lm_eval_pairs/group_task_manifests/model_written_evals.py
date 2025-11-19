"""Model written evals group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

MODEL_WRITTEN_EVALS_TASKS = {
    "model_written_evals": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
}
