"""Resolve evaluation config for tasks not in task-evaluator.json.

Derives evaluation_type and primary_metric from the extractor's evaluator_name
attribute, using the extractor registry as the source of truth.

Only standard evaluation tasks can be derived this way; docker_execution,
personalization, and refusal tasks MUST be explicitly listed in task-evaluator.json.
"""

from __future__ import annotations

from typing import Tuple

from wisent.core.utils.infra_tools.errors import TaskNotFoundError


# Maps extractor evaluator_name → (evaluation_type, primary_metric)
_EVALUATOR_NAME_TO_EVAL_CONFIG = {
    "exact_match": ("generate_until", "exact_match"),
    "log_likelihoods": ("loglikelihood", "acc"),
    "generation": ("generate_until", "generation"),
    "truthfulqa_gen": ("generate_until", "generation"),
    "f1": ("multiple_choice", "f1"),
    "perplexity": ("loglikelihood_rolling", "perplexity"),
}


def derive_eval_config_from_extractor(task_name: str) -> Tuple[str, str]:
    """Derive evaluation_type and primary_metric from the extractor's evaluator_name.

    Falls back to the extractor registry when a task is not in task-evaluator.json.

    Args:
        task_name: The benchmark/task name.

    Returns:
        Tuple of (evaluation_type, primary_metric).

    Raises:
        TaskNotFoundError: If no extractor found or evaluator_name not mappable.
    """
    from wisent.extractors.lm_eval.lm_extractor_registry import (
        get_extractor as get_lm_extractor,
        UnsupportedLMEvalBenchmarkError,
    )
    from wisent.extractors.hf.hf_extractor_registry import (
        get_extractor as get_hf_extractor,
        UnsupportedHuggingFaceBenchmarkError,
    )

    extractor = None
    try:
        extractor = get_lm_extractor(task_name)
    except UnsupportedLMEvalBenchmarkError:
        try:
            extractor = get_hf_extractor(task_name)
        except UnsupportedHuggingFaceBenchmarkError:
            pass

    if extractor is None:
        raise TaskNotFoundError(task_name=task_name)

    evaluator_name = getattr(extractor, "evaluator_name", None)
    if not evaluator_name:
        raise TaskNotFoundError(task_name=task_name)

    config = _EVALUATOR_NAME_TO_EVAL_CONFIG.get(evaluator_name)
    if config is None:
        raise TaskNotFoundError(task_name=task_name)

    print(
        f"   Task '{task_name}' not in task-evaluator.json, "
        f"derived from extractor evaluator_name='{evaluator_name}'"
    )
    return config
