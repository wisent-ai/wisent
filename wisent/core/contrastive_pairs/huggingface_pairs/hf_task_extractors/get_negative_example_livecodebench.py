"""
Extract negative (incorrect) examples from LiveCodeBench dataset.

This module loads pre-computed model outputs from the LiveCodeBench HuggingFace Space
and extracts examples that failed test cases.
"""
from __future__ import annotations

from wisent.core.cli_logger import setup_logger
import random
import string
from typing import Any

from wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors.get_positive_example_livecodebench import (
    load_all_outputs_cache,
)

__all__ = ["get_negative_example"]

log = setup_logger(__name__)


def get_negative_example(
    problem_idx: int,
    model_name: str | None = None,
    cache_dir: str | None = None,
) -> dict[str, Any] | None:
    """
    Get a negative (incorrect) code example for a given problem.

    Args:
        problem_idx: Index of the problem in the dataset
        model_name: Optional model name to get solution from. If None, gets first failing solution.
        cache_dir: Optional directory to cache downloaded data

    Returns:
        Dictionary with keys:
            - code: The incorrect code solution
            - metadata: Additional information about the solution
            - pass1: Boolean indicating if it passed (always False for negative examples)
        Returns None if no negative example found.
    """
    try:
        # Load all_outputs.json (uses shared cache)
        all_outputs = load_all_outputs_cache(cache_dir=cache_dir)

        if not all_outputs:
            log.warning("Failed to load all_outputs.json")
            return None

        # all_outputs structure: {model: [list of problems], ...}
        # Each problem is a dict: {code_list, pass1_list, metadata_list}

        # If model specified, try to get from that model first
        if model_name and model_name in all_outputs:
            model_data = all_outputs[model_name]
            if problem_idx < len(model_data):
                result = _extract_failing_example(model_data[problem_idx])
                if result:
                    return result

        # Otherwise, iterate through all models to find a failing example
        for current_model, model_data in all_outputs.items():
            if problem_idx < len(model_data):
                result = _extract_failing_example(model_data[problem_idx])
                if result:
                    log.debug(f"Found negative example from model: {current_model}")
                    return result

        log.warning(f"No negative examples found for problem {problem_idx}, generating synthetic gibberish")
        return _generate_gibberish_negative(problem_idx)

    except Exception as exc:
        log.error(f"Error loading negative example: {exc}", exc_info=True)
        return None


def _extract_failing_example(outputs: dict[str, list]) -> dict[str, Any] | None:
    """
    Extract the first failing example from model outputs.

    Args:
        outputs: Dictionary with code_list, pass1_list, metadata_list

    Returns:
        Dictionary with code, pass1, and metadata, or None if no failing example found.
    """
    code_list = outputs.get("code_list", [])
    pass1_list = outputs.get("pass1_list", [])
    metadata_list = outputs.get("metadata_list", [])

    # Find first failing example
    for code, pass1, metadata in zip(code_list, pass1_list, metadata_list):
        if not pass1:  # pass1 is False for failing examples
            return {
                "code": code,
                "pass1": pass1,
                "metadata": metadata,
            }

    return None


def _generate_gibberish_negative(problem_idx: int) -> dict[str, Any]:
    """
    Generate a synthetic negative example with random gibberish code.

    Used when no failing examples exist in pre-computed outputs (all models passed).

    Args:
        problem_idx: Index of the problem (used in metadata)

    Returns:
        Dictionary with synthetic gibberish code that will definitely fail.
    """
    words = []
    for _ in range(random.randint(10, 30)):
        word_len = random.randint(3, 12)
        word = ''.join(random.choices(string.ascii_lowercase, k=word_len))
        words.append(word)

    gibberish_lines = []
    gibberish_lines.append(f"def {random.choice(words)}():")
    for _ in range(random.randint(3, 8)):
        indent = "    "
        line_type = random.choice(["assign", "return", "print", "comment"])
        if line_type == "assign":
            gibberish_lines.append(f"{indent}{random.choice(words)} = '{random.choice(words)}'")
        elif line_type == "return":
            gibberish_lines.append(f"{indent}return '{random.choice(words)}'")
        elif line_type == "print":
            gibberish_lines.append(f"{indent}print('{random.choice(words)}')")
        else:
            gibberish_lines.append(f"{indent}# {random.choice(words)} {random.choice(words)}")

    gibberish_code = "\n".join(gibberish_lines)

    return {
        "code": gibberish_code,
        "pass1": False,
        "metadata": {
            "synthetic": True,
            "reason": "no_failing_model_outputs",
            "problem_idx": problem_idx,
        },
    }
