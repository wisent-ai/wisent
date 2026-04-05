"""Resolve evaluation config for tasks not in task-evaluator.json.

Derives evaluation_type and primary_metric from the extractor's evaluator_name
attribute, using the extractor registry as the source of truth.

Only standard evaluation tasks can be derived this way; docker_execution,
personalization, and refusal tasks MUST be explicitly listed in task-evaluator.json.
"""

from __future__ import annotations

import json
from pathlib import Path
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
    "agentharm": ("generate_until", "generation"),
    # AIME-style evaluators use exact integer match, evaluated as generate_until
    "aime": ("generate_until", "exact_match"),
}


def _load_task_evaluator_json() -> dict:
    """Load task-evaluator.json, searching upward from this file then via the wisent package."""
    current_dir = Path(__file__).resolve()
    for parent in current_dir.parents:
        task_eval_file = parent / "task-evaluator.json"
        if task_eval_file.exists():
            with open(task_eval_file) as f:
                return json.load(f).get("tasks", {})

    try:
        import wisent
        pkg_task_eval = Path(wisent.__file__).resolve().parent / "task-evaluator.json"
        if pkg_task_eval.exists():
            with open(pkg_task_eval) as f:
                return json.load(f).get("tasks", {})
    except Exception:
        pass

    return {}


def derive_eval_config_from_extractor(task_name: str) -> Tuple[str, str]:
    """Derive evaluation_type and primary_metric for a task.

    Checks task-evaluator.json first (for tasks explicitly registered there),
    then falls back to deriving the config from the extractor's evaluator_name.

    Args:
        task_name: The benchmark/task name.

    Returns:
        Tuple of (evaluation_type, primary_metric).

    Raises:
        TaskNotFoundError: If no extractor found or evaluator_name not mappable.
    """
    # Check task-evaluator.json first
    tasks = _load_task_evaluator_json()
    task_config = tasks.get(task_name, {})
    evaluation_type_from_json = task_config.get("evaluation_type")
    primary_metric_from_json = task_config.get("primary_metric")
    if (
        evaluation_type_from_json
        and evaluation_type_from_json not in ("docker_execution", "personalization", "refusal")
        and primary_metric_from_json
    ):
        return (evaluation_type_from_json, primary_metric_from_json)

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
