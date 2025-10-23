# coding/providers/livecodebench/provider.py
from __future__ import annotations
import json
from typing import Iterable, Optional
from ..core.atoms import CodingTask, Language


class LiveCodeBenchProvider:
    """
    LiveCodeBench provider: loads real coding problems from HuggingFace.

    Dataset: livecodebench/code_generation_lite
    Supports Python problems from LeetCode, AtCoder, and CodeForces.
    """
    name = "livecodebench"

    def __init__(
        self,
        language: Language = "python",
        release_version: str = "all",
        limit: Optional[int] = None,
        platform: Optional[str] = None,
    ):
        """
        Initialize LiveCodeBench provider.

        Arguments:
            language: Programming language (currently only "python" supported)
            release_version: Version to load ("release_v1", "release_v2", "all")
            limit: Maximum number of problems to load
            platform: Filter by platform ("leetcode", "codeforces", "atcoder")
        """
        self.language = language
        self.release_version = release_version
        self.limit = limit
        self.platform = platform

        if language != "python":
            raise NotImplementedError(
                f"LiveCodeBench currently only supports Python. Got: {language}"
            )

    def iter_tasks(self, split: str = "test") -> Iterable[CodingTask]:
        """
        Iterate over LiveCodeBench coding tasks.

        Arguments:
            split: Dataset split (only "test" is available for LiveCodeBench)

        Yields:
            CodingTask objects with solution file, test file, and options
        """
        from datasets import load_dataset

        # Load dataset from HuggingFace
        dataset = load_dataset("livecodebench/code_generation_lite", split=split)

        # Filter by version (date range)
        if self.release_version == "release_v1":
            dataset = dataset.filter(
                lambda x: x["contest_date"] >= "2023-05-01" and x["contest_date"] <= "2023-10-31"
            )
        elif self.release_version == "release_v2":
            dataset = dataset.filter(
                lambda x: x["contest_date"] >= "2023-11-01" and x["contest_date"] <= "2024-04-30"
            )

        # Filter by platform if specified
        if self.platform:
            platform_lower = self.platform.lower()
            dataset = dataset.filter(
                lambda x: x["platform"].lower() == platform_lower
            )

        # Apply limit
        if self.limit:
            dataset = dataset.select(range(min(self.limit, len(dataset))))

        # Convert each problem to a CodingTask
        for idx, problem in enumerate(dataset):
            task = self._problem_to_task(problem, idx)
            if task:
                yield task

    def _problem_to_task(self, problem: dict, idx: int) -> Optional[CodingTask]:
        """
        Convert a LiveCodeBench problem to a CodingTask.

        Arguments:
            problem: Problem dictionary from HuggingFace dataset
            idx: Problem index

        Returns:
            CodingTask or None if conversion fails
        """
        try:
            platform = problem["platform"].lower()
            question_id = problem["question_id"]

            # Parse test cases
            public_tests = json.loads(problem["public_test_cases"])

            if not public_tests:
                return None

            # Determine test type and generate appropriate test file
            test_type = public_tests[0].get("testtype", "stdin")

            if test_type == "functional":
                # LeetCode-style: function calls with arguments
                test_file = self._generate_functional_test(problem, public_tests)
            else:
                # stdin: CodeForces/AtCoder style
                test_file = self._generate_stdin_test(problem, public_tests)

            if not test_file:
                return None

            # Generate solution file template
            solution_file = self._generate_solution_template(problem)

            files = {
                "solution.py": solution_file,
                "tests.py": test_file,
            }

            options = {
                "problem_id": question_id,
                "platform": platform,
                "difficulty": problem.get("difficulty", "unknown"),
            }

            return CodingTask(
                language=self.language,
                files=files,
                options=options,
            )

        except Exception as e:
            # Skip problematic problems
            import logging
            logging.warning(f"Failed to convert problem {idx}: {e}")
            return None

    def _generate_solution_template(self, problem: dict) -> str:
        """
        Generate a solution template from starter code or problem description.

        Arguments:
            problem: Problem dictionary

        Returns:
            Python solution template as string
        """
        starter_code = problem.get("starter_code", "").strip()

        if starter_code:
            # Use provided starter code
            return starter_code
        else:
            # Generate minimal template for stdin problems
            return """# Read input and solve the problem
import sys

def solve():
    # Read input from stdin
    lines = sys.stdin.read().strip().split('\\n')

    # TODO: Implement solution
    pass

if __name__ == "__main__":
    solve()
"""

    def _generate_functional_test(self, problem: dict, test_cases: list) -> str:
        """
        Generate test file for LeetCode-style functional tests.

        Arguments:
            problem: Problem dictionary
            test_cases: List of test case dictionaries

        Returns:
            Python test file content
        """
        starter_code = problem.get("starter_code", "").strip()

        if not starter_code:
            return ""

        # Extract class and method name from starter code
        import re
        class_match = re.search(r"class\s+(\w+)", starter_code)
        method_match = re.search(r"def\s+(\w+)\s*\(", starter_code)

        if not class_match or not method_match:
            return ""

        class_name = class_match.group(1)
        method_name = method_match.group(1)

        # Generate test file
        test_code = f"""from solution import {class_name}

def test_functional():
    solution = {class_name}()

"""

        for i, test in enumerate(test_cases):
            input_str = test.get("input", "")
            expected_output = test.get("output", "")

            # Parse input (typically JSON array where first element is the actual argument)
            try:
                # Try to evaluate as Python literal
                import ast
                parsed = ast.literal_eval(input_str)

                # If it's a list with one element that's also a list, use that inner list
                if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], list):
                    args = [parsed[0]]
                elif isinstance(parsed, list):
                    args = [parsed]
                else:
                    args = [parsed]
            except:
                # Fallback: use raw string
                args = [input_str]

            # Parse expected output
            try:
                import ast
                expected = ast.literal_eval(expected_output)
            except:
                expected = expected_output

            # Generate assertion
            args_str = ", ".join(repr(arg) for arg in args)
            test_code += f"    # Test case {i + 1}\n"
            test_code += f"    result = solution.{method_name}({args_str})\n"
            test_code += f"    assert result == {repr(expected)}, f\"Test {i + 1} failed: {{result}} != {repr(expected)}\"\n\n"

        test_code += "if __name__ == '__main__':\n"
        test_code += "    test_functional()\n"
        test_code += "    print('All tests passed!')\n"

        return test_code

    def _generate_stdin_test(self, problem: dict, test_cases: list) -> str:
        """
        Generate test file for stdin-based tests (CodeForces/AtCoder style).

        Arguments:
            problem: Problem dictionary
            test_cases: List of test case dictionaries

        Returns:
            Python test file content
        """
        # For stdin tests, we run the solution and compare output
        test_code = """import subprocess
import sys

def test_stdin():
    test_cases = [
"""

        for i, test in enumerate(test_cases):
            input_data = test.get("input", "")
            expected_output = test.get("output", "")

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
