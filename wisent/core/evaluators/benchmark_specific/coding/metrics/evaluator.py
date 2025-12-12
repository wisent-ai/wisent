from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, TYPE_CHECKING
import logging

from wisent.core.evaluators.core.atoms import BaseEvaluator, EvalResult
from wisent.core.evaluators.benchmark_specific.coding.safe_docker.core.runtime import DockerSandboxExecutor
from wisent.core.evaluators.benchmark_specific.coding.safe_docker.recipes import RECIPE_REGISTRY
from wisent.core.evaluators.benchmark_specific.coding.metrics.core.atoms import SampleOutcome
from wisent.core.errors import MissingParameterError

from wisent.core.evaluators.benchmark_specific.coding.output_sanitizer.core.atoms import TaskSchema
from wisent.core.evaluators.benchmark_specific.coding.output_sanitizer.python_sanitizer import PythonStandardizer
from wisent.core.evaluators.benchmark_specific.coding.output_sanitizer.cpp_sanitizer import CppStandardizer
from wisent.core.evaluators.benchmark_specific.coding.output_sanitizer.java_sanitizer import JavaStandardizer

if TYPE_CHECKING:
    from wisent.core.evaluators.benchmark_specific.coding.safe_docker.core.atoms import Result
    from wisent.core.evaluators.benchmark_specific.coding.providers.core.atoms import Provider, CodingTask
    from wisent.core.evaluators.benchmark_specific.coding.output_sanitizer.core.atoms import CodeStandardizer

logger = logging.getLogger(__name__)

RepairFn = Callable[[str, dict[str,str], str], dict[str,str]]

@dataclass
class EvaluatorConfig:
    """
    Configuration for CodingEvaluator.

    attributes:
        image:
            Docker image to use for code execution (default: "coding/sandbox:polyglot-1.0").
        runtime:
            Optional Docker runtime (e.g., "runsc" for gVisor).
        feedback_max_chars:
            Maximum characters of feedback to pass to the repair function (default: 2000).
        self_repair:
            Whether to perform a single self-repair turn (default: True). It means the we provide feedback to the model for one iteration.
        time_limit_s:
            Time limit in seconds for each code execution (default: 8s).
        cpu_limit_s:
            CPU time limit in seconds for each code execution (default: 3s).
        mem_limit_mb:
            Memory limit in megabytes for each code execution (default: 768MB).
        pre_sanitize:
            Whether to run LLM output through a sanitizer before execution (default: True).
    """
    image: str = "coding/sandbox:polyglot-1.0"
    runtime: Optional[str] = None
    feedback_max_chars: int = 2000
    self_repair: bool = True
    time_limit_s: int = 8
    cpu_limit_s: int = 3
    mem_limit_mb: int = 768
    pre_sanitize: bool = True

_SANITIZERS = {
    "python": PythonStandardizer(),
    "cpp":    CppStandardizer(),
    "java":   JavaStandardizer(),
}

def _default_filename(lang: str) -> str:
    """
    Returns the default source file name for a given programming language.

    arguments:
        lang:
            Programming language ("python", "cpp", or "java").

    returns:
        Default filename as a string.
    """
    return {"python":"solution.py","cpp":"solution.cpp","java":"Solution.java"}.get(lang, "solution.py")

def _make_schema(task: "CodingTask") -> TaskSchema:
    """
    Constructs a TaskSchema from a CodingTask, using task options or defaults.
    """
    entry = str(task.options.get("entry_point", "solve"))
    file_name = str(task.options.get("file_name", _default_filename(task.language)))
    java_class = str(task.options.get("java_class", "Solution"))
    return TaskSchema(language=task.language, file_name=file_name, entry_point=entry,
                      java_class=java_class, prefer_rename=True, allow_wrapper=True)


class CodingEvaluator(BaseEvaluator):
    """
    Unified evaluator for coding tasks.

    Supports two modes:
    1. Single evaluation via evaluate(response, expected, **kwargs) - for steering/contrastive pairs
    2. Batch evaluation via evaluate_all() - for full benchmark evaluation with self-repair

    Features:
    - Docker sandbox execution
    - Code sanitization
    - Self-repair (optional)
    - Multiple language support (Python, C++, Java)
    """

    name = "coding"
    description = "Code execution evaluator with Docker sandbox, sanitization and self-repair"

    def __init__(
        self,
        provider: Optional["Provider"] = None,
        model_fn: Optional[Callable[["CodingTask"], dict[str,str]]] = None,
        repair_fn: Optional[RepairFn] = None,
        cfg: EvaluatorConfig = None
    ):
        """Initialize coding evaluator.

        Args:
            provider: Task provider for batch evaluation (optional)
            model_fn: Function to generate code from task (optional, for batch mode)
            repair_fn: Function to repair code after failure (optional)
            cfg: Evaluator configuration
        """
        self.provider = provider
        self.model_fn = model_fn
        self.repair_fn = repair_fn
        self.cfg = cfg or EvaluatorConfig()
        self.exec = DockerSandboxExecutor(image=self.cfg.image, runtime=self.cfg.runtime)

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate code by executing it in Docker.

        This is the BaseEvaluator interface for single evaluation.

        Args:
            response: Generated code to test
            expected: Expected correct code (or reference)
            **kwargs:
                test_code: Test code to run
                entry_point: Function name being tested
                task_name: Task name (for selecting Docker image)
                language: Programming language (default: python)

        Returns:
            EvalResult with TRUTHFUL (tests pass) / UNTRUTHFUL (tests fail) / UNKNOWN (error)
        """
        test_code = kwargs.get('test_code')
        entry_point = kwargs.get('entry_point')
        task_name = kwargs.get('task_name', '')
        language = kwargs.get('language', 'python')

        # Debug logging for test_code
        logger.debug(f"CodingEvaluator.evaluate() called with test_code={'present (' + str(len(test_code)) + ' chars)' if test_code else 'MISSING'}, task_name={task_name}")
        if not test_code:
            logger.warning(f"CodingEvaluator: No test_code provided. kwargs keys: {list(kwargs.keys())}")

        if not test_code:
            return EvalResult(
                ground_truth="UNKNOWN",
                method_used=self.name,
                confidence=0.0,
                details="No test code provided for code execution",
            )

        # Select appropriate Docker image based on task
        if 'ds1000' in task_name.lower() or 'ds_1000' in task_name.lower():
            image = "coding/sandbox:ds1000"
        else:
            image = self.cfg.image

        # Re-initialize executor if image changed
        if self.exec.image != image:
            self.exec = DockerSandboxExecutor(image=image, runtime=self.cfg.runtime)

        # Code to test
        code_to_test = response if response else str(expected)

        # For livecodebench functional tests, prepend typing imports if needed
        if 'livecodebench' in task_name.lower() and 'from solution import Solution' in (test_code or ''):
            # This is a functional (LeetCode-style) problem - add typing imports
            typing_imports = "from typing import List, Optional, Dict, Tuple, Set, Any\n\n"
            if not code_to_test.strip().startswith("from typing") and "List[" in code_to_test:
                code_to_test = typing_imports + code_to_test

        # Prepare test file
        if entry_point is None or 'livecodebench' in task_name.lower():
            test_file_content = test_code
        else:
            test_file_content = f"""from solution import {entry_point}

{test_code}

if __name__ == "__main__":
    check({entry_point})
    print("All tests passed!")
"""

        files = {
            "solution.py": code_to_test,
            "tests.py": test_file_content,
        }

        # Optionally sanitize code
        # Skip sanitization for:
        # - livecodebench stdin-style tests (they run solution.py as a script via subprocess)
        # - DS-1000 tests (they use exec() with exec_context template, not function imports)
        is_stdin_test = 'livecodebench' in task_name.lower() and 'subprocess' in (test_code or '')
        is_ds1000 = 'ds1000' in task_name.lower() or 'ds_1000' in task_name.lower()

        if self.cfg.pre_sanitize and language in _SANITIZERS and not is_stdin_test and not is_ds1000:
            schema = TaskSchema(
                language=language,
                file_name="solution.py",
                entry_point=entry_point or "solve",
                java_class="Solution",
                prefer_rename=True,
                allow_wrapper=True
            )
            sanitizer = _SANITIZERS[language]
            raw = files.get("solution.py")
            if raw:
                out = sanitizer.normalize(raw, schema)
                files["solution.py"] = out.files.get("solution.py", raw)

        # Execute
        try:
            from wisent.core.evaluators.benchmark_specific.coding.safe_docker.core.atoms import Job

            # Set limits based on task or kwargs
            timeout_override = kwargs.get('timeout')
            if 'ds1000' in task_name.lower() or 'ds_1000' in task_name.lower():
                cpu_limit_s = 60
                wall_timeout_s = 120
                nproc = 512
            elif timeout_override:
                cpu_limit_s = timeout_override
                wall_timeout_s = timeout_override
                nproc = 64
            else:
                cpu_limit_s = self.cfg.cpu_limit_s
                wall_timeout_s = self.cfg.time_limit_s
                nproc = 64

            job = Job(
                language=language,
                compile_argv=None,
                run_argv=["python3", "tests.py"],
                cpu_limit_s=cpu_limit_s,
                wall_timeout_s=wall_timeout_s,
                mem_limit_mb=self.cfg.mem_limit_mb,
                fsize_mb=10,
                nproc=nproc,
                nofile=128,
            )

            result = self.exec.run(files, job)

            if result.status == "ok":
                return EvalResult(
                    ground_truth="TRUTHFUL",
                    method_used=self.name,
                    confidence=1.0,
                    details=f"Code executed successfully. Status: {result.status}",
                    meta={
                        "exit_code": result.exit_code,
                        "elapsed": result.elapsed,
                        "stdout": result.stdout[:500] if result.stdout else "",
                        "stderr": result.stderr[:500] if result.stderr else "",
                    }
                )
            else:
                error_info = f"stdout: {result.stdout[:500] if result.stdout else ''}\nstderr: {result.stderr[:500] if result.stderr else ''}"
                return EvalResult(
                    ground_truth="UNTRUTHFUL",
                    method_used=self.name,
                    confidence=1.0,
                    details=f"Code execution failed. Status: {result.status}\n{error_info}",
                    meta={
                        "exit_code": result.exit_code,
                        "elapsed": result.elapsed,
                        "stdout": result.stdout[:500] if result.stdout else "",
                        "stderr": result.stderr[:500] if result.stderr else "",
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

    # --- Batch evaluation methods (for full benchmark runs) ---

    def _feedback(self, res: "Result") -> str:
        """Generates feedback text from a Result object for use in self-repair."""
        if res.status == "timeout":
            return f"Timeout after {res.elapsed:.2f}s."
        body = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
        if res.status == "compile_error":
            prefix = "Compilation failed:\n"
        else:
            prefix = "Tests failed:\n"
        return (prefix + body)[: self.cfg.feedback_max_chars]

    def _run_once(self, task: "CodingTask", files: dict[str,str]) -> "Result":
        """Runs a single evaluation job for the given task and files."""
        recipe = RECIPE_REGISTRY[task.language]
        job = recipe.make_job(**task.options,
                              time_limit_s=self.cfg.time_limit_s,
                              cpu_limit_s=self.cfg.cpu_limit_s,
                              mem_limit_mb=self.cfg.mem_limit_mb)
        return self.exec.run(files, job)

    def _maybe_sanitize(self, task: "CodingTask", files: dict[str,str]) -> dict[str,str]:
        """Optionally sanitizes the generated files based on the task schema."""
        if not self.cfg.pre_sanitize:
            return files
        schema = _make_schema(task)
        sanitizer: "CodeStandardizer" = _SANITIZERS.get(task.language)
        if sanitizer is None:
            return files

        raw = files.get(schema.file_name) or files.get("__raw__")
        if not raw:
            return files

        out = sanitizer.normalize(raw, schema)
        files = {**files, schema.file_name: out.files.get(schema.file_name, raw)}
        return files

    def evaluate_all(self) -> Iterable[SampleOutcome]:
        """
        Evaluates all tasks from the provider, performing optional self-repair.

        This is for full benchmark evaluation (e.g., computing pass@k).

        yields:
            SampleOutcome for each task, indicating pass/fail status and elapsed time.
        """
        if self.provider is None or self.model_fn is None:
            raise MissingParameterError(params=["provider", "model_fn"], context="batch evaluation")

        for idx, task in enumerate(self.provider.iter_tasks()):
            files0 = self.model_fn(task)
            files0 = {**task.files, **files0}
            files0 = self._maybe_sanitize(task, files0)

            r0 = self._run_once(task, files0)
            if r0.status == "ok":
                yield SampleOutcome(task_id=f"{self.provider.name}:{idx}", status=r0.status, passed=True, elapsed=r0.elapsed)
                continue

            if not self.cfg.self_repair or self.repair_fn is None:
                yield SampleOutcome(task_id=f"{self.provider.name}:{idx}", status=r0.status, passed=False, elapsed=r0.elapsed)
                continue

            fb = self._feedback(r0)
            files1 = self.repair_fn(task.language, files0, fb)
            files1 = {**task.files, **files1}
            files1 = self._maybe_sanitize(task, files1)

            r1 = self._run_once(task, files1)
            passed = (r0.status == "ok") or (r1.status == "ok")
            yield SampleOutcome(task_id=f"{self.provider.name}:{idx}", status=r1.status, passed=passed, elapsed=(r0.elapsed + r1.elapsed))


# Backward compatibility alias
DockerCodeEvaluator = CodingEvaluator
