"""
Extract positive (correct) examples from LiveCodeBench dataset.

This module loads pre-computed model outputs from the LiveCodeBench HuggingFace Space
and extracts examples that passed all test cases.
"""
from __future__ import annotations

import json
from wisent.core.cli_logger import setup_logger
import os
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

__all__ = ["get_positive_example", "load_all_outputs_cache"]

log = setup_logger(__name__)

# Global cache for all_outputs.json to avoid re-downloading
_ALL_OUTPUTS_CACHE = None


def load_all_outputs_cache(cache_dir: str | None = None) -> dict[str, Any]:
    """
    Load and cache the all_outputs.json file from LiveCodeBench HuggingFace Space.

    Args:
        cache_dir: Optional directory to cache downloaded data

    Returns:
        Dictionary with structure: {model: {problem_idx: {code_list, pass1_list, metadata_list}}}
    """
    global _ALL_OUTPUTS_CACHE

    if _ALL_OUTPUTS_CACHE is not None:
        return _ALL_OUTPUTS_CACHE

    try:
        # Download all_outputs.json from the HuggingFace Space
        file_path = hf_hub_download(
            repo_id="livecodebench/code_generation_samples",
            filename="all_outputs.json",
            repo_type="space",
            cache_dir=cache_dir,
        )

        log.info(f"Loading all_outputs.json from {file_path}")

        with open(file_path, "r") as f:
            _ALL_OUTPUTS_CACHE = json.load(f)

        log.info(f"Loaded all_outputs.json with {len(_ALL_OUTPUTS_CACHE)} models")
        return _ALL_OUTPUTS_CACHE

    except Exception as exc:
        log.error(f"Error loading all_outputs.json: {exc}", exc_info=True)
        return {}


def get_positive_example(
    problem_idx: int,
    model_name: str | None = None,
    cache_dir: str | None = None,
) -> dict[str, Any] | None:
    """
    Get a positive (correct) code example for a given problem.

    Args:
        problem_idx: Index of the problem in the dataset
        model_name: Optional model name to get solution from. If None, gets first passing solution.
        cache_dir: Optional directory to cache downloaded data

    Returns:
        Dictionary with keys:
            - code: The correct code solution
            - metadata: Additional information about the solution
            - pass1: Boolean indicating if it passed (always True for positive examples)
        Returns None if no positive example found.
    """
    try:
        # Load all_outputs.json
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
                result = _extract_passing_example(model_data[problem_idx])
                if result:
                    return result

        # Otherwise, iterate through all models to find a passing example
        for current_model, model_data in all_outputs.items():
            if problem_idx < len(model_data):
                result = _extract_passing_example(model_data[problem_idx])
                if result:
                    log.debug(f"Found positive example from model: {current_model}")
                    return result

        log.warning(f"No positive examples found for problem {problem_idx}")
        return None

    except Exception as exc:
        log.error(f"Error loading positive example: {exc}", exc_info=True)
        return None


def _extract_passing_example(outputs: dict[str, list]) -> dict[str, Any] | None:
    """
    Extract the first passing example from model outputs.

    Args:
        outputs: Dictionary with code_list, pass1_list, metadata_list

    Returns:
        Dictionary with code, pass1, and metadata, or None if no passing example found.
    """
    code_list = outputs.get("code_list", [])
    pass1_list = outputs.get("pass1_list", [])
    metadata_list = outputs.get("metadata_list", [])

    # Find first passing example
    for code, pass1, metadata in zip(code_list, pass1_list, metadata_list):
        if pass1:  # pass1 is True for passing examples
            return {
                "code": code,
                "pass1": pass1,
                "metadata": metadata,
            }

    return None
