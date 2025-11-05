"""
Generate contrastive pairs from LiveCodeBench pre-computed model outputs.

This module creates contrastive pairs by loading existing correct and incorrect
code solutions from the LiveCodeBench dataset's all_outputs.json file.
"""
from __future__ import annotations

import logging
from typing import Any

from datasets import load_dataset

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors.get_positive_example_livecodebench import (
    get_positive_example,
)
from wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors.get_negative_example_livecodebench import (
    get_negative_example,
)

__all__ = ["generate_livecodebench_pairs"]

log = logging.getLogger(__name__)


def generate_livecodebench_pairs(
    limit: int | None = None,
    cache_dir: str | None = None,
) -> list[ContrastivePair]:
    """
    Generate contrastive pairs from LiveCodeBench dataset.

    This loads pre-computed model outputs (correct and incorrect solutions)
    from the LiveCodeBench dataset and creates contrastive pairs.

    Args:
        limit: Optional maximum number of pairs to generate
        cache_dir: Optional directory to cache downloaded data

    Returns:
        List of ContrastivePair objects with positive (passing) and negative (failing) examples
    """
    try:
        # Load the problems dataset
        dataset = load_dataset(
            "livecodebench/code_generation_lite",
            "release_latest",
            split="test",
            cache_dir=cache_dir,
        )

        # Also load problems.json from the Space to get proper mappings
        from huggingface_hub import hf_hub_download
        import json

        problems_path = hf_hub_download(
            repo_id="livecodebench/code_generation_samples",
            filename="problems.json",
            repo_type="space",
            cache_dir=cache_dir,
        )

        with open(problems_path, "r") as f:
            problems_json = json.load(f)

        max_items = min(limit, len(problems_json)) if limit else len(problems_json)

        pairs: list[ContrastivePair] = []

        log.info(f"Generating contrastive pairs from {max_items} livecodebench problems")

        for problem_idx in range(max_items):
            pair = _create_pair_for_problem(problem_idx, problems_json, cache_dir)
            if pair is not None:
                pairs.append(pair)
                log.debug(f"Created pair {len(pairs)}/{max_items}")

        log.info(f"Generated {len(pairs)} livecodebench contrastive pairs")
        return pairs

    except Exception as exc:
        log.error(f"Error generating livecodebench pairs: {exc}", exc_info=True)
        return []


def _create_pair_for_problem(
    problem_idx: int,
    problems_json: list[dict],
    cache_dir: str | None = None,
) -> ContrastivePair | None:
    """
    Create a contrastive pair for a single problem.

    Args:
        problem_idx: Index of the problem in the problems.json
        problems_json: List of problem dictionaries from problems.json
        cache_dir: Optional cache directory

    Returns:
        ContrastivePair or None if unable to create pair
    """
    try:
        # Get the problem data from problems.json
        if problem_idx >= len(problems_json):
            log.warning(f"Problem index {problem_idx} out of range")
            return None

        problem = problems_json[problem_idx]
        question = problem.get("question_content", "").strip()

        if not question:
            log.warning(f"Problem {problem_idx} has no question content")
            return None

        # Get positive and negative examples from pre-computed outputs
        positive_example = get_positive_example(problem_idx, cache_dir=cache_dir)
        negative_example = get_negative_example(problem_idx, cache_dir=cache_dir)

        if not positive_example:
            log.warning(f"No positive example found for problem {problem_idx}")
            return None

        if not negative_example:
            log.warning(f"No negative example found for problem {problem_idx}")
            return None

        # Get test cases from livecodebench dataset
        from datasets import load_dataset
        import json as json_lib

        dataset = load_dataset(
            "livecodebench/code_generation_lite",
            "release_latest",
            split="test",
            cache_dir=cache_dir,
        )

        if problem_idx < len(dataset):
            problem_data = dataset[problem_idx]
            public_test_cases_str = problem_data.get("public_test_cases", "[]")

            # Parse test cases (they're stored as JSON string)
            try:
                test_cases = json_lib.loads(public_test_cases_str)
                # Build test_code from test cases
                test_code = _build_test_code(test_cases)
            except Exception as e:
                log.warning(f"Could not parse test cases for problem {problem_idx}: {e}")
                test_code = None
        else:
            test_code = None

        # Format the prompt
        formatted_question = f"Question: {question}\n\nWrite a solution:"

        # Create responses
        positive_response = PositiveResponse(model_response=positive_example["code"])
        negative_response = NegativeResponse(model_response=negative_example["code"])

        # Build metadata
        metadata = {
            "label": "livecodebench",
            "source": "livecodebench/code_generation_samples",
            "problem_idx": problem_idx,
            "question_id": problem.get("question_id"),
            "difficulty": problem.get("difficulty"),
            "positive_metadata": positive_example.get("metadata"),
            "negative_metadata": negative_example.get("metadata"),
            "test_code": test_code,
        }

        # Create the contrastive pair
        pair = ContrastivePair(
            prompt=formatted_question,
            positive_response=positive_response,
            negative_response=negative_response,
            label="livecodebench",
            metadata=metadata,
        )

        return pair

    except Exception as exc:
        log.error(f"Error creating pair for problem {problem_idx}: {exc}", exc_info=True)
        return None


def _build_test_code(test_cases: list[dict]) -> str | None:
    """Build test code from livecodebench test cases.

    Uses subprocess-based testing like the LiveCodeBench provider.

    Args:
        test_cases: List of test case dicts with 'input' and 'output' keys

    Returns:
        Test code string or None if no valid test cases
    """
    if not test_cases:
        return None

    # Build test code using subprocess approach (matches livecodebench provider)
    test_code = """import subprocess
import sys

def test_stdin():
    test_cases = [
"""

    for i, test_case in enumerate(test_cases):
        input_data = test_case.get("input", "")
        expected_output = test_case.get("output", "")

        if not input_data or not expected_output:
            continue

        test_code += f"        # Test case {i + 1}\n"
        test_code += f"        ({repr(input_data)}, {repr(expected_output)}),\n"

    test_code += """    ]

    for i, (input_data, expected_output) in enumerate(test_cases):
        # Run solution with input
        proc = subprocess.run(
            [sys.executable, "solution.py"],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=5
        )

        actual_output = proc.stdout.strip()
        expected_output = expected_output.strip()

        assert actual_output == expected_output, (
            f"Test case {i + 1} failed:\\n"
            f"  Input: {input_data[:100]}\\n"
            f"  Expected: {expected_output[:200]}\\n"
            f"  Got: {actual_output[:200]}"
        )

    print(f'All {len(test_cases)} test(s) passed!')

if __name__ == '__main__':
    test_stdin()
"""

    return test_code
