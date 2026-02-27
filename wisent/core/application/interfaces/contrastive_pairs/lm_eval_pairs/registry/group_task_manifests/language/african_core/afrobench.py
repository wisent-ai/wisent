"""Afrobench group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

# NO PARENT TASK MAPPING - each subtask mapped individually based on output_type

AFROBENCH_TASKS = {
    # Multiple_choice subtasks → afrobench_mc.py (evaluator_name = "log_likelihoods")

    # Generate_until (CoT) subtasks → afrobench_cot.py (evaluator_name = "generation")

}
