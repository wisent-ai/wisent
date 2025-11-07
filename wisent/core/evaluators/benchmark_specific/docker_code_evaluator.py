"""Docker-based code evaluator that integrates with BaseEvaluator."""

import logging
from typing import Any

from wisent.core.evaluators.core.atoms import BaseEvaluator, EvalResult
from wisent.core.evaluators.benchmark_specific.coding.safe_docker.core.runtime import DockerSandboxExecutor
from wisent.core.evaluators.benchmark_specific.coding.safe_docker.core.atoms import Job

logger = logging.getLogger(__name__)


class DockerCodeEvaluator(BaseEvaluator):
    """Evaluator for code generation that executes code in Docker sandbox.

    Compatible with:
    - HumanEval: Python function completion with unit tests
    - MBPP: Python programming problems
    - All Python coding benchmarks
    """

    name = "docker_code"
    description = "Code execution evaluator using Docker sandbox"
    task_names = (
        "humaneval", "humaneval_plus", "instruct_humaneval",
        "mbpp", "mbpp_plus",
        "apps",
        "ds_1000", "ds1000",
        "multipl_e", "multiple_py", "multiple_js", "multiple_java", "multiple_cpp", "multiple_rs", "multiple_go",
        "livecodebench",
        # New task families from lm-eval-harness (code generation with pass@k)
        "instructhumaneval", "humanevalpack", "recode", "conala", "concode", "mercury", "codexglue"
    )

    def __init__(self, image: str = "coding/sandbox:polyglot-1.0"):
        """Initialize Docker code evaluator.

        Args:
            image: Docker image to use (default: coding/sandbox:polyglot-1.0)
        """
        self.default_image = image
        self.executor = None  # Will be initialized per-task based on benchmark

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate code by executing it in Docker.

        Args:
            response: Generated code
            expected: Expected correct code
            **kwargs:
                model: WisentModel instance (for choices evaluation)
                question: The prompt/question
                choices: [correct_code, incorrect_code] for log-likelihood comparison
                task_name: Task name

                # For direct execution:
                test_code: Test code to run
                entry_point: Function name being tested

        Returns:
            EvalResult with TRUTHFUL/UNTRUTHFUL/UNKNOWN
        """
        logger.debug(f"DockerCodeEvaluator.evaluate called with kwargs keys: {list(kwargs.keys())}")

        # Select appropriate Docker image based on task
        task_name = kwargs.get('task_name', '')
        if 'ds1000' in task_name.lower() or 'ds_1000' in task_name.lower():
            image = "coding/sandbox:ds1000"
        else:
            image = self.default_image

        # Initialize executor with selected image
        if self.executor is None or self.executor.image != image:
            self.executor = DockerSandboxExecutor(image=image)

        # Docker code evaluator ONLY does code execution, no log-likelihood fallback
        test_code = kwargs.get('test_code')
        entry_point = kwargs.get('entry_point')

        if not test_code:
            error_msg = f"No test code provided for Docker code execution. kwargs keys: {list(kwargs.keys())}"
            logger.error(error_msg)
            return EvalResult(
                ground_truth="UNKNOWN",
                method_used=self.name,
                confidence=0.0,
                details=error_msg,
            )

        # Execute code in Docker
        try:
            # Use 'expected' as the code to execute if 'response' is empty
            # This happens when we're comparing against an expected answer
            code_to_test = response if response else expected

            # Debug: print what we're testing
            print(f"\n[DEBUG] Testing code length: {len(code_to_test)}")
            print(f"[DEBUG] Test code length: {len(test_code)}")
            print(f"[DEBUG] Test code content:\n{test_code[:500]}\n")

            # Prepare files for execution
            # Note: recipes expect tests.py (plural) not test.py
            # For HumanEval, test_code defines check(candidate) but doesn't call it
            # We need to import the solution function and call check() on it

            # Extract the entry_point (function name) from kwargs
            entry_point_name = kwargs.get('entry_point')
            task_name = kwargs.get('task_name', '')

            # For livecodebench or when entry_point is None, use test_code as-is
            # (it runs solution.py as subprocess, doesn't import functions)
            if entry_point_name is None or 'livecodebench' in task_name.lower():
                test_file_content = test_code
            else:
                # For HumanEval-style function testing, wrap with import
                test_file_content = f"""from solution import {entry_point_name}

{test_code}

# Call the check function with the solution
if __name__ == "__main__":
    check({entry_point_name})
    print("All tests passed!")
"""

            files = {
                "solution.py": code_to_test,
                "tests.py": test_file_content,
            }

            # Create job configuration
            # DS-1000 needs more time and higher nproc for numpy/scipy multithreading
            task_name = kwargs.get('task_name', '')
            if 'ds1000' in task_name.lower() or 'ds_1000' in task_name.lower():
                cpu_limit_s = 60
                wall_timeout_s = 120
                nproc = 512  # Higher limit for scientific libraries
            else:
                cpu_limit_s = 3
                wall_timeout_s = 5
                nproc = 64

            job = Job(
                language="python",
                compile_argv=None,
                run_argv=["python3", "tests.py"],  # Changed to tests.py (plural)
                cpu_limit_s=cpu_limit_s,
                wall_timeout_s=wall_timeout_s,
                mem_limit_mb=256,
                fsize_mb=10,
                nproc=nproc,
                nofile=128,
            )

            # Execute
            result = self.executor.run(files, job)

            # Determine outcome based on test execution results
            # For code evaluation:
            # - TRUTHFUL means code is correct (tests passed)
            # - UNTRUTHFUL means code is incorrect (tests failed)
            if result.status == "ok":
                # Tests passed - code is correct
                return EvalResult(
                    ground_truth="TRUTHFUL",
                    method_used=self.name,
                    confidence=1.0,
                    details=f"Code executed successfully. Status: {result.status}",
                    meta={
                        "exit_code": result.exit_code,
                        "elapsed": result.elapsed,
                        "stdout": result.stdout[:500],
                        "stderr": result.stderr[:500],
                    }
                )
            else:
                # Tests failed - code is incorrect
                error_info = f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
                return EvalResult(
                    ground_truth="UNTRUTHFUL",
                    method_used=self.name,
                    confidence=1.0,
                    details=f"Code execution failed. Status: {result.status}\n{error_info}",
                    meta={
                        "exit_code": result.exit_code,
                        "elapsed": result.elapsed,
                        "stdout": result.stdout[:500],
                        "stderr": result.stderr[:500],
                    }
                )

        except Exception as e:
            logger.error(f"Docker execution failed: {e}")
            return EvalResult(
                ground_truth="UNKNOWN",
                method_used=self.name,
                confidence=0.0,
                details=f"Docker execution error: {str(e)}",
            )
