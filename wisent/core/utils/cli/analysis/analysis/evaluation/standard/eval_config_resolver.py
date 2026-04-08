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
    # Core
    "exact_match": ("generate_until", "exact_match"),
    "log_likelihoods": ("loglikelihood", "acc"),
    "generation": ("generate_until", "generation"),
    "truthfulqa_gen": ("generate_until", "generation"),
    "f1": ("multiple_choice", "f1"),
    "perplexity": ("loglikelihood_rolling", "perplexity"),
    "agentharm": ("generate_until", "generation"),
    "aime": ("generate_until", "exact_match"),
    # Math / coding (generate + exact match against expected answer)
    "math": ("generate_until", "exact_match"),
    "coding": ("generate_until", "exact_match"),
    "cnmo": ("generate_until", "exact_match"),
    "polymath": ("generate_until", "exact_match"),
    "olympiadbench": ("generate_until", "exact_match"),
    # Loglikelihood / multi-choice variants
    "eus_exams": ("loglikelihood", "acc"),
    "inverse_scaling": ("loglikelihood", "acc"),
    "lambada_multilingual": ("loglikelihood", "acc"),
    "paws_x": ("loglikelihood", "acc"),
    "okapi_hellaswag": ("loglikelihood", "acc"),
    "okapi_mmlu": ("loglikelihood", "acc"),
    "okapi_truthfulqa": ("loglikelihood", "acc"),
    "mmlu_redux": ("loglikelihood", "acc"),
    "med_concepts_qa": ("loglikelihood", "acc"),
    "cluewsc": ("loglikelihood", "acc"),
    "mmmu": ("loglikelihood", "acc"),
    # Generation-based
    "darija_bench": ("generate_until", "generation"),
    "libra_score": ("generate_until", "generation"),
    "multi_swe_bench": ("generate_until", "generation"),
    "agentbench": ("generate_until", "generation"),
    "aider_polyglot": ("generate_until", "generation"),
    "bfcl": ("generate_until", "generation"),
    "browsecomp": ("generate_until", "generation"),
    "chinese_simpleqa": ("generate_until", "generation"),
    "codeforces": ("generate_until", "exact_match"),
    "conala": ("generate_until", "generation"),
    "curate": ("generate_until", "generation"),
    "donotanswer": ("generate_until", "generation"),
    "facts_grounding": ("generate_until", "generation"),
    "faithbench": ("generate_until", "generation"),
    "finsearchcomp": ("generate_until", "generation"),
    "flames": ("generate_until", "generation"),
    "hallucinations_leaderboard": ("generate_until", "generation"),
    "halueval": ("generate_until", "generation"),
    "halulens": ("generate_until", "generation"),
    "harmbench": ("generate_until", "generation"),
    "jailbreakbench": ("generate_until", "generation"),
    "livecodebench_v6": ("generate_until", "exact_match"),
    "longform_writing": ("generate_until", "generation"),
    "mercury": ("generate_until", "generation"),
    "mlqa": ("generate_until", "generation"),
    "nl2bash": ("generate_until", "generation"),
    "ojbench": ("generate_until", "exact_match"),
    "or_bench": ("generate_until", "generation"),
    "planbench": ("generate_until", "generation"),
    "politicalbias": ("generate_until", "generation"),
    "polyglot_toxicity": ("generate_until", "generation"),
    "recode": ("generate_until", "generation"),
    "refusalbench": ("generate_until", "generation"),
    "scicode": ("generate_until", "exact_match"),
    "seal": ("generate_until", "generation"),
    "sorry_bench": ("generate_until", "generation"),
    "swe_bench": ("generate_until", "generation"),
    "swe_bench_verified": ("generate_until", "generation"),
    "sycophancy_eval": ("generate_until", "generation"),
    "tau_bench": ("generate_until", "generation"),
    "terminal_bench": ("generate_until", "generation"),
    "toolbench": ("generate_until", "generation"),
    "toolemu": ("generate_until", "generation"),
    "travelplanner": ("generate_until", "generation"),
    "wildguard": ("generate_until", "generation"),
    "tag": ("generate_until", "generation"),
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
