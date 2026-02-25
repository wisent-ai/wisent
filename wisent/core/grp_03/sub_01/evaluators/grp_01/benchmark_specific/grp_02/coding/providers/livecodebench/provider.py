# coding/providers/livecodebench/provider.py
from __future__ import annotations
import json
from typing import Iterable, Optional
from ..core.atoms import CodingTask, Language
from wisent.core.constants import DISPLAY_TRUNCATION_MEDIUM, DISPLAY_TRUNCATION_COMPACT, SUBPROCESS_TIMEOUT_SHORT


class LiveCodeBenchProvider:
    """LiveCodeBench provider: loads real coding problems from HuggingFace."""
    name = "livecodebench"

    def __init__(self, language: Language = "python", release_version: str = "all",
                 limit: Optional[int] = None, platform: Optional[str] = None):
        self.language = language
        self.release_version = release_version
        self.limit = limit
        self.platform = platform
        if language != "python":
            raise NotImplementedError(f"LiveCodeBench currently only supports Python. Got: {language}")

    def iter_tasks(self, split: str = "test") -> Iterable[CodingTask]:
        """Iterate over LiveCodeBench coding tasks."""
        from datasets import load_dataset
        dataset = load_dataset("livecodebench/code_generation_lite", split=split)
        if self.release_version == "release_v1":
            dataset = dataset.filter(
                lambda x: x["contest_date"] >= "2023-05-01" and x["contest_date"] <= "2023-10-31"
            )
        elif self.release_version == "release_v2":
            dataset = dataset.filter(
                lambda x: x["contest_date"] >= "2023-11-01" and x["contest_date"] <= "2024-04-30"
            )
        if self.platform:
            platform_lower = self.platform.lower()
            dataset = dataset.filter(lambda x: x["platform"].lower() == platform_lower)
        if self.limit:
            dataset = dataset.select(range(min(self.limit, len(dataset))))
        for idx, problem in enumerate(dataset):
            task = self._problem_to_task(problem, idx)
            if task:
                yield task

    def _problem_to_task(self, problem: dict, idx: int) -> Optional[CodingTask]:
        """Convert a LiveCodeBench problem to a CodingTask."""
        try:
            platform = problem["platform"].lower()
            question_id = problem["question_id"]
            public_tests = json.loads(problem["public_test_cases"])
            if not public_tests:
                return None
            test_type = public_tests[0].get("testtype", "stdin")
            if test_type == "functional":
                test_file = self._generate_functional_test(problem, public_tests)
            else:
                test_file = self._generate_stdin_test(problem, public_tests)
            if not test_file:
                return None
            solution_file = self._generate_solution_template(problem)
            files = {"solution.py": solution_file, "tests.py": test_file}
            options = {"problem_id": question_id, "platform": platform, "difficulty": problem.get("difficulty", "unknown")}
            return CodingTask(language=self.language, files=files, options=options)
        except Exception as e:
            import logging
            logging.warning(f"Failed to convert problem {idx}: {e}")
            return None

    def _generate_solution_template(self, problem: dict) -> str:
        """Generate a solution template from starter code or problem description."""
        starter_code = problem.get("starter_code", "").strip()
        if starter_code:
            return starter_code
        return """# Read input and solve the problem
import sys

def solve():
    lines = sys.stdin.read().strip().split('\\n')
    # TODO: Implement solution
    pass

if __name__ == "__main__":
    solve()
"""

    def _generate_functional_test(self, problem: dict, test_cases: list) -> str:
        """Generate test file for LeetCode-style functional tests."""
        starter_code = problem.get("starter_code", "").strip()
        if not starter_code:
            return ""
        import re
        class_match = re.search(r"class\s+(\w+)", starter_code)
        method_match = re.search(r"def\s+(\w+)\s*\(", starter_code)
        if not class_match or not method_match:
            return ""
        class_name = class_match.group(1)
        method_name = method_match.group(1)
        test_code = f"""from solution import {class_name}

def test_functional():
    solution = {class_name}()

"""
        for i, test in enumerate(test_cases):
            input_str = test.get("input", "")
            expected_output = test.get("output", "")
            try:
                import ast
                parsed = ast.literal_eval(input_str)
                if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], list):
                    args = [parsed[0]]
                elif isinstance(parsed, list):
                    args = [parsed]
                else:
                    args = [parsed]
            except Exception:
                args = [input_str]
            try:
                import ast
                expected = ast.literal_eval(expected_output)
            except Exception:
                expected = expected_output
            args_str = ", ".join(repr(arg) for arg in args)
            test_code += f"    # Test case {i + 1}\n"
            test_code += f"    result = solution.{method_name}({args_str})\n"
            test_code += f"    assert result == {repr(expected)}, f\"Test {i + 1} failed: {{result}} != {repr(expected)}\"\n\n"
        test_code += "if __name__ == '__main__':\n"
        test_code += "    test_functional()\n"
        test_code += "    print('All tests passed!')\n"
        return test_code

    def _generate_stdin_test(self, problem: dict, test_cases: list) -> str:
        """Generate test file for stdin-based tests (CodeForces/AtCoder style)."""
        limit_secs = SUBPROCESS_TIMEOUT_SHORT
        test_code = f"""import subprocess
import sys

_LIMIT = {limit_secs}

def test_stdin():
    test_cases = [
"""
        for i, test in enumerate(test_cases):
            input_data = test.get("input", "")
            expected_output = test.get("output", "")
            test_code += f"        # Test case {i + 1}\n"
            test_code += f"        ({repr(input_data)}, {repr(expected_output)}),\n"
        _trunc = DISPLAY_TRUNCATION_MEDIUM
        _trunc_c = DISPLAY_TRUNCATION_COMPACT
        test_code += f"""    ]

    for i, (input_data, expected_output) in enumerate(test_cases):
        proc = subprocess.run(
            [sys.executable, "solution.py"],
            input=input_data,
            capture_output=True,
            text=True,
        )
        actual_output = proc.stdout.strip()
        expected_output = expected_output.strip()
        assert actual_output == expected_output, (
            f"Test case {{i + 1}} failed:\\n"
            f"  Input: {{input_data[:{_trunc_c}]}}\\n"
            f"  Expected: {{expected_output[:{_trunc}]}}\\n"
            f"  Got: {{actual_output[:{_trunc}]}}"
        )
    print(f'All {{len(test_cases)}} test(s) passed!')

if __name__ == '__main__':
    test_stdin()
"""
        return test_code
