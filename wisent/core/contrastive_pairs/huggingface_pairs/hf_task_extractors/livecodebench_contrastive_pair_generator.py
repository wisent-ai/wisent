"""
Generate contrastive pairs from LiveCodeBench pre-computed model outputs.

This module creates contrastive pairs by loading existing correct and incorrect
code solutions from the LiveCodeBench dataset's all_outputs.json file.
"""
from __future__ import annotations

from wisent.core.cli_logger import setup_logger
from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors.get_positive_example_livecodebench import (
    get_positive_example,
)
from wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors.get_negative_example_livecodebench import (
    get_negative_example,
)

__all__ = ["generate_livecodebench_pairs"]

log = setup_logger(__name__)


def _load_livecodebench_data(cache_dir: str | None = None) -> list[dict]:
    """
    Load LiveCodeBench data from HuggingFace datasets.

    Args:
        cache_dir: Optional cache directory

    Returns:
        List of problem dictionaries with test cases
    """
    # First, try to load from cached arrow files (most reliable)
    try:
        from datasets import Dataset
        import os

        # Look for cached dataset in standard HuggingFace cache location
        hf_cache = os.path.expanduser("~/.cache/huggingface/datasets")
        lcb_cache_base = os.path.join(hf_cache, "livecodebench___code_generation_lite")

        if os.path.exists(lcb_cache_base):
            # Find the latest release directory
            for release_dir in sorted(os.listdir(lcb_cache_base), reverse=True):
                release_path = os.path.join(lcb_cache_base, release_dir)
                if not os.path.isdir(release_path):
                    continue

                # Find version directory
                for version_dir in os.listdir(release_path):
                    version_path = os.path.join(release_path, version_dir)
                    if not os.path.isdir(version_path):
                        continue

                    # Look for arrow files
                    arrow_files = sorted([
                        f for f in os.listdir(version_path)
                        if f.startswith("code_generation_lite-test") and f.endswith(".arrow")
                    ])

                    if arrow_files:
                        all_data = []
                        for arrow_file in arrow_files:
                            arrow_path = os.path.join(version_path, arrow_file)
                            try:
                                ds = Dataset.from_file(arrow_path)
                                all_data.extend([dict(row) for row in ds])
                            except Exception as e:
                                log.warning(f"Could not load arrow file {arrow_file}: {e}")

                        if all_data:
                            log.info(f"Loaded {len(all_data)} problems from cached arrow files")
                            return all_data

    except Exception as e:
        log.warning(f"Could not load from cached arrow files: {e}")

    # Second, try downloading JSONL files directly from HuggingFace
    try:
        from huggingface_hub import hf_hub_download
        import json

        # Download the test.jsonl file (contains problems with test cases)
        jsonl_path = hf_hub_download(
            repo_id="livecodebench/code_generation_lite",
            filename="test.jsonl",
            repo_type="dataset",
            cache_dir=cache_dir,
        )

        all_data = []
        with open(jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        all_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if all_data:
            log.info(f"Loaded {len(all_data)} problems from test.jsonl")
            return all_data

    except Exception as e1:
        log.warning(f"Could not load via JSONL download: {e1}")

    # Third, try standard datasets library
    try:
        from datasets import load_dataset

        # Load using standard datasets library (uses cache automatically)
        ds = load_dataset("livecodebench/code_generation_lite", split="test")
        log.info(f"Loaded {len(ds)} problems from livecodebench/code_generation_lite")
        return [dict(row) for row in ds]

    except Exception as e2:
        log.warning(f"Could not load via datasets library: {e2}")

        # Fallback: try to load from the Space's problems.json
        # Note: This fallback does NOT have public_test_cases field
        try:
            from huggingface_hub import hf_hub_download
            import json

            problems_path = hf_hub_download(
                repo_id="livecodebench/code_generation_samples",
                filename="problems.json",
                repo_type="space",
                cache_dir=cache_dir,
            )

            with open(problems_path, "r") as f:
                data = json.load(f)
                log.warning(
                    "Loaded from problems.json which does NOT contain test cases. "
                    "Code evaluation will not work properly."
                )
                return data
        except Exception as e3:
            log.error(f"Could not load problems.json fallback: {e3}")
            return []


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
        # Load problems.json from the Space to get proper mappings
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

        # Load dataset for test cases
        dataset_data = _load_livecodebench_data(cache_dir)

        # Build question_id -> dataset row mapping for proper matching
        dataset_by_question_id = {}
        for row in dataset_data:
            qid = row.get("question_id")
            if qid:
                dataset_by_question_id[qid] = row

        log.info(f"Loaded {len(dataset_data)} dataset rows, {len(dataset_by_question_id)} with question_id")

        for problem_idx in range(max_items):
            pair = _create_pair_for_problem(problem_idx, problems_json, dataset_data, dataset_by_question_id, cache_dir)
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
    dataset_data: list[dict],
    dataset_by_question_id: dict[str, dict],
    cache_dir: str | None = None,
) -> ContrastivePair | None:
    """
    Create a contrastive pair for a single problem.

    Args:
        problem_idx: Index of the problem in the problems.json
        problems_json: List of problem dictionaries from problems.json
        dataset_data: List of problem data from dataset (for test cases)
        dataset_by_question_id: Dict mapping question_id to dataset row
        cache_dir: Optional cache directory

    Returns:
        ContrastivePair or None if unable to create pair
    """
    import json as json_lib

    try:
        # Get the problem data from problems.json
        if problem_idx >= len(problems_json):
            log.warning(f"Problem index {problem_idx} out of range")
            return None

        problem = problems_json[problem_idx]
        question = problem.get("question_content", "").strip()
        question_id = problem.get("question_id")

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

        # Get test cases from dataset data using question_id for proper matching
        test_code = None
        starter_code = ""

        # Try to match by question_id first
        problem_data = dataset_by_question_id.get(question_id) if question_id else None

        # Fallback to index-based matching
        if problem_data is None and problem_idx < len(dataset_data):
            problem_data = dataset_data[problem_idx]

        if problem_data:
            public_test_cases_str = problem_data.get("public_test_cases", "[]")
            starter_code = problem_data.get("starter_code", "")

            # Parse test cases (they're stored as JSON string)
            try:
                test_cases = json_lib.loads(public_test_cases_str) if isinstance(public_test_cases_str, str) else public_test_cases_str
                # Build test_code from test cases
                test_code = _build_test_code(test_cases, starter_code)
            except Exception as e:
                log.warning(f"Could not parse test cases for problem {problem_idx}: {e}")
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
            "question_id": question_id,
            "difficulty": problem.get("difficulty"),
            "positive_metadata": positive_example.get("metadata"),
            "negative_metadata": negative_example.get("metadata"),
            "test_code": test_code,
            "starter_code": starter_code,
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


def _build_test_code(test_cases: list[dict], starter_code: str = "") -> str | None:
    """Build test code from livecodebench test cases.

    Handles both stdin (AtCoder/Codeforces) and functional (LeetCode) test types.

    Args:
        test_cases: List of test case dicts with 'input', 'output', and 'testtype' keys
        starter_code: Optional starter code containing class/method definition

    Returns:
        Test code string or None if no valid test cases
    """
    if not test_cases:
        return None

    # Determine test type from first test case
    test_type = test_cases[0].get("testtype", "stdin")

    if test_type == "functional":
        return _build_functional_test_code(test_cases, starter_code)
    else:
        return _build_stdin_test_code(test_cases)


def _build_stdin_test_code(test_cases: list[dict]) -> str:
    """Build test code for stdin-based problems (AtCoder/Codeforces)."""
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
            f"  Input: {input_data}\\n"
            f"  Expected: {expected_output}\\n"
            f"  Got: {actual_output}"
        )

    print(f'All {len(test_cases)} test(s) passed!')

if __name__ == '__main__':
    test_stdin()
"""

    return test_code


def _extract_method_name(starter_code: str) -> str | None:
    """Extract method name from LeetCode starter code."""
    import re
    match = re.search(r'def\s+(\w+)\s*\(\s*self', starter_code)
    return match.group(1) if match else None


def _build_functional_test_code(test_cases: list[dict], starter_code: str) -> str | None:
    """Build test code for functional problems (LeetCode).

    These problems have a Solution class with a method to call.
    """
    method_name = _extract_method_name(starter_code)
    if not method_name:
        return None

    test_code = """import json
import ast
from typing import List, Optional, Dict, Tuple, Set

# Import the solution
from solution import Solution

def parse_input(input_str):
    \"\"\"Parse input string into Python objects.

    Input can be:
    - Single line: one argument
    - Multiple lines: multiple arguments (one per line)
    \"\"\"
    lines = input_str.strip().split('\\n')
    args = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            # Try JSON parsing first (handles arrays, strings, etc.)
            args.append(json.loads(line))
        except json.JSONDecodeError:
            try:
                # Fall back to ast.literal_eval for Python literals
                args.append(ast.literal_eval(line))
            except (ValueError, SyntaxError):
                # Keep as string if nothing else works
                args.append(line)
    return args

def parse_output(output_str):
    \"\"\"Parse expected output into Python object.\"\"\"
    output_str = output_str.strip()
    try:
        return json.loads(output_str)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(output_str)
        except (ValueError, SyntaxError):
            return output_str

def test_functional():
    test_cases = [
"""

    for i, test_case in enumerate(test_cases):
        input_data = test_case.get("input", "")
        expected_output = test_case.get("output", "")

        if expected_output == "":
            continue

        test_code += f"        # Test case {i + 1}\n"
        test_code += f"        ({repr(input_data)}, {repr(expected_output)}),\n"

    test_code += f"""    ]

    sol = Solution()

    for i, (input_str, expected_str) in enumerate(test_cases):
        args = parse_input(input_str)
        expected = parse_output(expected_str)

        # Call the method with parsed arguments
        actual = sol.{method_name}(*args)

        # Compare results
        assert actual == expected, (
            f"Test case {{i + 1}} failed:\\n"
            f"  Input: {{input_str}}\\n"
            f"  Expected: {{expected}}\\n"
            f"  Got: {{actual}}"
        )

    print(f'All {{len(test_cases)}} test(s) passed!')

if __name__ == '__main__':
    test_functional()
"""

    return test_code
