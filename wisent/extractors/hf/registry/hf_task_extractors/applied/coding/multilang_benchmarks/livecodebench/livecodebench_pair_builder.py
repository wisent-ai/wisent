"""LiveCodeBench pair creation and test code builders."""
from __future__ import annotations

from wisent.core import constants as _C
from wisent.core.utils.cli.cli_logger import setup_logger
from wisent.core.utils.infra_tools.infra.core.hardware import docker_code_exec_timeout_s as _get_code_exec_timeout
from typing import Any

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from .get_positive_example_livecodebench import (
    get_positive_example,
)
from .get_negative_example_livecodebench import (
    get_negative_example,
)

log = setup_logger(__name__)

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
            timeout=TIMEOUT_VAL
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
""".replace("TIMEOUT_VAL", str(_get_code_exec_timeout()))

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
