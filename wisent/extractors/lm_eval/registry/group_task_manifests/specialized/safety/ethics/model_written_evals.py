"""Model written evals group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

MODEL_WRITTEN_EVALS_TASKS = {
    "model_written_evals": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
    # winogenerated uses the same answer_matching_behavior /
    # answer_not_matching_behavior MCQ schema as model_written_evals.
    "winogenerated": f"{BASE_IMPORT}model_written_evals:ModelWrittenEvalsExtractor",
}
