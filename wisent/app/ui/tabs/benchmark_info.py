"""Benchmark info panel — detailed metadata for the debug tab."""

import inspect
from typing import Any

from wisent.core.utils.config_tools.constants import (
    INDEX_FIRST,
    TEST_EXTRACTOR_EVALUATOR_DEFAULT_LIMIT,
    DISPLAY_TRUNCATION_MEDIUM,
)


def _truncate(s: str, limit: int = DISPLAY_TRUNCATION_MEDIUM) -> str:
    """Truncate a string to limit characters."""
    return s[:limit] + "..." if len(s) > limit else s


def get_dataset_columns(extractor: Any) -> list[str]:
    """Try to load one doc and return its keys."""
    try:
        pairs = extractor.extract_contrastive_pairs(
            limit=INDEX_FIRST + INDEX_FIRST)
        if not pairs:
            try:
                pairs = extractor.extract_contrastive_pairs(
                    lm_eval_task_data=None,
                    limit=INDEX_FIRST + INDEX_FIRST)
            except Exception:
                pass
        if pairs:
            pair = pairs[INDEX_FIRST]
            meta = pair.metadata if hasattr(pair, "metadata") else {}
            return {
                "prompt_preview": _truncate(pair.prompt),
                "correct_preview": _truncate(
                    pair.positive_response.model_response),
                "incorrect_preview": _truncate(
                    pair.negative_response.model_response),
                "metadata_keys": list(meta.keys()) if isinstance(meta, dict) else [],
            }
    except Exception:
        pass
    return {}


def get_contrastive_pair_logic(extractor: Any) -> str:
    """Extract the contrastive pair generation approach from source."""
    cls = type(extractor)
    doc = cls.extract_contrastive_pairs.__doc__
    if doc:
        return doc.strip()
    # Try to find _extract_pair_from_doc or _create_incorrect
    for method_name in ("_extract_pair_from_doc",
                        "_create_incorrect_answer",
                        "_create_incorrect_response"):
        method = getattr(cls, method_name, None)
        if method and method.__doc__:
            return f"{method_name}: {method.__doc__.strip()}"
    return "No documentation available"


def get_evaluator_logic(evaluator_name: str) -> str:
    """Get the evaluator methodology description."""
    from wisent.core.reading.evaluators.core.atoms import BaseEvaluator
    import wisent.core.reading.evaluators.core.benchmark_specific  # noqa: F401
    try:
        cls = BaseEvaluator.get(evaluator_name)
        parts = []
        if cls.__doc__:
            parts.append(cls.__doc__.strip())
        if hasattr(cls, "requires_judge"):
            parts.append(f"Requires judge: {cls.requires_judge}")
        evaluate_doc = cls.evaluate.__doc__
        if evaluate_doc:
            parts.append(f"evaluate(): {evaluate_doc.strip()}")
        return "\n".join(parts) if parts else "No documentation"
    except Exception as exc:
        return f"Error: {exc}"


def get_example_pairs(extractor: Any, count: int) -> list[dict]:
    """Extract a few pairs and return them as dicts."""
    try:
        pairs = extractor.extract_contrastive_pairs(limit=count)
    except TypeError:
        try:
            pairs = extractor.extract_contrastive_pairs(
                lm_eval_task_data=None, limit=count)
        except Exception:
            return []
    except Exception:
        return []

    results = []
    for pair in pairs:
        results.append({
            "prompt": _truncate(pair.prompt),
            "correct": _truncate(
                pair.positive_response.model_response),
            "incorrect": _truncate(
                pair.negative_response.model_response),
        })
    return results


def format_full_info(task_name: str) -> str:
    """Build complete benchmark info string."""
    from wisent.extractors.lm_eval.lm_extractor_registry import (
        get_extractor, _REGISTRY,
    )
    from wisent.core.utils.services.benchmarks.registry.benchmark_registry import (
        get_working_benchmarks_with_categories,
    )

    if " (" in task_name and task_name.endswith(")"):
        task_name = task_name.split(" (")[INDEX_FIRST]

    all_names = sorted(_REGISTRY.keys())
    categories = get_working_benchmarks_with_categories()
    cat = categories.get(task_name, "uncategorized")

    lines = [f"**{task_name}** — `{cat}`\n"]

    try:
        ext = get_extractor(task_name)
    except Exception as exc:
        return f"**{task_name}**\n\nError loading extractor: {exc}"

    ev_name = getattr(ext, "evaluator_name", None)
    lines.append(f"**Extractor:** `{type(ext).__name__}` "
                 f"(`{type(ext).__module__}`)")
    lines.append(f"**Evaluator:** `{ev_name}`")

    # Aliases
    ref = _REGISTRY.get(task_name)
    aliases = [n for n in all_names if _REGISTRY.get(n) == ref and n != task_name]
    if aliases:
        lines.append(f"**Aliases:** {', '.join(aliases)}")

    # Subtasks
    prefix = task_name + "_"
    subtasks = [n for n in all_names if n.startswith(prefix)]
    if subtasks:
        lines.append(f"**Subtasks:** {len(subtasks)}")

    # Contrastive pair logic
    lines.append(f"\n**Contrastive Pair Logic:**")
    lines.append(f"```\n{get_contrastive_pair_logic(ext)}\n```")

    # Evaluator logic
    if ev_name:
        lines.append(f"\n**Evaluator Logic ({ev_name}):**")
        lines.append(f"```\n{get_evaluator_logic(ev_name)}\n```")

    # Example pairs
    examples = get_example_pairs(ext, INDEX_FIRST + INDEX_FIRST + INDEX_FIRST)
    if examples:
        lines.append(f"\n**Example Pairs ({len(examples)}):**")
        for i, ex in enumerate(examples):
            lines.append(f"\n*Pair {i}:*")
            lines.append(f"- Prompt: {ex['prompt']}")
            lines.append(f"- Correct: {ex['correct']}")
            lines.append(f"- Incorrect: {ex['incorrect']}")

    return "\n".join(lines)
