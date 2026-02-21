"""Test script creation mixin for BigCode evaluator."""
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BigCodeScriptsMixin:
    """Mixin providing test script creation for BigCodeEvaluator."""

    def _create_test_script(self, sample: Dict, generation: str, task_name: str) -> str:
        """Create a test script for the sample."""
        if "humaneval" in task_name:
            script = self._create_humaneval_test_script(sample, generation)
        elif "mbpp" in task_name:
            script = self._create_mbpp_test_script(sample, generation)
        elif "apps" in task_name:
            script = self._create_apps_test_script(sample, generation)
        else:
            # Default format
            script = self._create_humaneval_test_script(sample, generation)

        logger.debug(f"📝 Test script for {task_name}:\n{script}\n")
        return script

    def _create_humaneval_test_script(self, sample: Dict, generation: str) -> str:
        """Create test script for HumanEval format."""
        entry_point = sample.get("entry_point", "solution")
        test_code = sample.get("test", "")
        prompt = sample.get("prompt", "")

        # The prompt contains the function signature, and generation should be the function body
        # We need to combine them properly
        script = f"""
{prompt}{generation}

{test_code}

if __name__ == "__main__":
    check({entry_point})
    print("All tests passed!")
"""
        return script

    def _create_mbpp_test_script(self, sample: Dict, generation: str) -> str:
        """Create test script for MBPP format."""
        test_imports = sample.get("test_imports", [])
        test_list = sample.get("test_list", [])

        # Fix function name mismatch before creating test script
        fixed_generation = self._fix_function_name_mismatch(generation, test_list)

        imports = "\n".join(test_imports)
        tests = "\n    ".join(test_list)

        script = f"""
{imports}

{fixed_generation}

if __name__ == "__main__":
    {tests}
    print("All tests passed!")
"""
        return script

    def _create_apps_test_script(self, sample: Dict, generation: str) -> str:
        """Create test script for APPS format."""
        # APPS has input/output pairs
        io_data = json.loads(sample.get("input_output", "{}"))
        inputs = io_data.get("inputs", [])
        outputs = io_data.get("outputs", [])

        tests = []
        for inp, out in zip(inputs, outputs):
            tests.append(f"assert str(solution({inp})) == '{out}'")

        test_code = "\n    ".join(tests)

        script = f"""
{generation}

if __name__ == "__main__":
    {test_code}
    print("All tests passed!")
"""
        return script

    def _fix_function_name_mismatch(self, code: str, test_list: List[str]) -> str:
        """
        Fix function name mismatches between generated code and test cases.

        Uses wrapper function approach for robustness across different code structures.

        Args:
            code: Generated code that may have wrong function name
            test_list: List of test assertions that specify expected function name

        Returns:
            Fixed code with wrapper function if needed
        """
        import re

        if not test_list or not code.strip():
            return code

        # Extract expected function name from test assertions
        expected_name = None
        # Built-in functions to skip when looking for the target function
        builtin_functions = {
            "set",
            "len",
            "str",
            "int",
            "float",
            "list",
            "tuple",
            "dict",
            "sum",
            "max",
            "min",
            "abs",
            "round",
            "sorted",
            "reversed",
        }

        for test in test_list:
            # Find all function calls in assert statements
            function_calls = re.findall(r"(\w+)\s*\(", test)

            for func_name in function_calls:
                # Skip built-in functions and common test functions
                if func_name not in builtin_functions and func_name not in {
                    "assert",
                    "assertEqual",
                    "assertTrue",
                    "assertFalse",
                }:
                    expected_name = func_name
                    break

            if expected_name:
                break

        if not expected_name:
            return code  # No function name found in tests

        # Extract actual function name from generated code
        actual_name = None
        func_match = re.search(r"def\s+(\w+)\s*\(", code)
        if func_match:
            actual_name = func_match.group(1)

        if not actual_name:
            return code  # No function definition found

        if actual_name == expected_name:
            return code  # Names already match

        logger.debug(f"🔧 Function name mismatch detected: {actual_name} → {expected_name}")
        logger.debug("   Adding wrapper function for compatibility")

        # Add wrapper function to bridge the name gap
        wrapper = f"""
# Wrapper function for test compatibility
def {expected_name}(*args, **kwargs):
    return {actual_name}(*args, **kwargs)
"""

        return code + wrapper

