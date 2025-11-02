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
        "apps", "conala", "concode",
        "ds_1000", "mercury", "recode",
        "multipl_e", "codexglue",
        "livecodebench"
    )

    def __init__(self, image: str = "coding/sandbox:polyglot-1.0"):
        """Initialize Docker code evaluator.

        Args:
            image: Docker image to use (default: coding/sandbox:polyglot-1.0)
        """
        self.executor = DockerSandboxExecutor(image=image)

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
        # If we have choices, do log-likelihood comparison first
        choices = kwargs.get('choices')
        model = kwargs.get('model')
        question = kwargs.get('question', '')

        if choices and model and len(choices) == 2:
            # Use log-likelihood to compare the two code choices
            correct_code = choices[0]
            incorrect_code = choices[1]

            # Expected is what we're comparing against
            # If expected == correct_code, we expect TRUTHFUL
            # If expected == incorrect_code, we expect UNTRUTHFUL

            try:
                # Get log-likelihoods for both choices
                correct_logprob = model.get_logprob(question, correct_code)
                incorrect_logprob = model.get_logprob(question, incorrect_code)

                # Model prefers the one with higher log-likelihood
                model_prefers_correct = correct_logprob > incorrect_logprob

                # Check if this matches expectation
                expects_correct = (expected == correct_code)

                is_truthful = model_prefers_correct == expects_correct
                confidence = abs(correct_logprob - incorrect_logprob)

                return EvalResult(
                    ground_truth="TRUTHFUL" if is_truthful else "UNTRUTHFUL",
                    method_used=self.name,
                    confidence=confidence,
                    details=f"Log-likelihood comparison: correct={correct_logprob:.4f}, incorrect={incorrect_logprob:.4f}",
                    meta={
                        "correct_logprob": correct_logprob,
                        "incorrect_logprob": incorrect_logprob,
                        "model_prefers_correct": model_prefers_correct,
                    }
                )
            except Exception as e:
                logger.warning(f"Log-likelihood comparison failed: {e}, falling back to Docker execution")

        # If no choices or log-likelihood failed, try Docker execution
        test_code = kwargs.get('test_code')
        entry_point = kwargs.get('entry_point')

        if not test_code:
            return EvalResult(
                ground_truth="UNKNOWN",
                method_used=self.name,
                confidence=0.0,
                details="No test code provided and no choices for log-likelihood comparison",
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
            entry_point_name = kwargs.get('entry_point', 'solution')

            # Create test file that imports solution and calls check()
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
            job = Job(
                language="python",
                compile_argv=None,
                run_argv=["python3", "tests.py"],  # Changed to tests.py (plural)
                cpu_limit_s=3,
                wall_timeout_s=5,
                mem_limit_mb=256,
                fsize_mb=10,
                nproc=64,
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
