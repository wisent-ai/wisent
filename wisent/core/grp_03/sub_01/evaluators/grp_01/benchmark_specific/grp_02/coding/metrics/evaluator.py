from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, TYPE_CHECKING
import logging

from wisent.core.evaluators.core.atoms import BaseEvaluator, EvalResult
from wisent.core.constants import FEEDBACK_MAX_CHARS, EVAL_TIME_LIMIT_S, EVAL_CPU_LIMIT_S, EVAL_MEM_LIMIT_MB, SAFE_DOCKER_NPROC_DEFAULT, DS1000_CPU_LIMIT_S, DS1000_WALL_TIMEOUT_S, DS1000_NPROC, SAFE_DOCKER_FSIZE_MB, SAFE_DOCKER_NOFILE, DISPLAY_TRUNCATION_LARGE
from wisent.core.evaluators.benchmark_specific.coding.safe_docker.core.runtime import DockerSandboxExecutor
from wisent.core.evaluators.benchmark_specific.coding.metrics.core.atoms import SampleOutcome

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
            Whether to perform a single self-repair turn (default: True).
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
    feedback_max_chars: int = FEEDBACK_MAX_CHARS
    self_repair: bool = True
    time_limit_s: int = EVAL_TIME_LIMIT_S
    cpu_limit_s: int = EVAL_CPU_LIMIT_S
    mem_limit_mb: int = EVAL_MEM_LIMIT_MB
    pre_sanitize: bool = True

_SANITIZERS = {
    "python": PythonStandardizer(),
    "cpp":    CppStandardizer(),
    "java":   JavaStandardizer(),
}

def _default_filename(lang: str) -> str:
    """Returns the default source file name for a given programming language."""
    return {"python":"solution.py","cpp":"solution.cpp","java":"Solution.java"}.get(lang, "solution.py")

def _make_schema(task: "CodingTask") -> TaskSchema:
    """Constructs a TaskSchema from a CodingTask, using task options or defaults."""
    entry = str(task.options.get("entry_point", "solve"))
    file_name = str(task.options.get("file_name", _default_filename(task.language)))
    java_class = str(task.options.get("java_class", "Solution"))
    return TaskSchema(language=task.language, file_name=file_name, entry_point=entry,
                      java_class=java_class, prefer_rename=True, allow_wrapper=True)


class CodingEvaluator(BaseEvaluator):
    """
    Unified evaluator for coding tasks.

    Supports two modes:
    1. Single evaluation via evaluate(response, expected, **kwargs)
    2. Batch evaluation via evaluate_all()
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
        """Initialize coding evaluator."""
        self.provider = provider
        self.model_fn = model_fn
        self.repair_fn = repair_fn
        self.cfg = cfg or EvaluatorConfig()
        self.exec = DockerSandboxExecutor(image=self.cfg.image, runtime=self.cfg.runtime)

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate code by executing it in Docker."""
        test_code = kwargs.get('test_code')
        entry_point = kwargs.get('entry_point')
        task_name = kwargs.get('task_name', '')
        language = kwargs.get('language', 'python')

        logger.debug(f"CodingEvaluator.evaluate() called with test_code={'present (' + str(len(test_code)) + ' chars)' if test_code else 'MISSING'}, task_name={task_name}")
        if not test_code:
            logger.warning(f"CodingEvaluator: No test_code provided. kwargs keys: {list(kwargs.keys())}")

        if not test_code:
            return EvalResult(
                ground_truth="UNKNOWN", method_used=self.name,
                confidence=0.0, details="No test code provided for code execution",
            )

        if 'ds1000' in task_name.lower() or 'ds_1000' in task_name.lower():
            image = "coding/sandbox:ds1000"
        else:
            image = self.cfg.image

        if self.exec.image != image:
            self.exec = DockerSandboxExecutor(image=image, runtime=self.cfg.runtime)

        code_to_test = response if response else str(expected)

        if 'livecodebench' in task_name.lower() and 'from solution import Solution' in (test_code or ''):
            typing_imports = "from typing import List, Optional, Dict, Tuple, Set, Any\n\n"
            if not code_to_test.strip().startswith("from typing") and "List[" in code_to_test:
                code_to_test = typing_imports + code_to_test

        if entry_point is None or 'livecodebench' in task_name.lower():
            test_file_content = test_code
        else:
            test_file_content = f"""from solution import {entry_point}

{test_code}

if __name__ == "__main__":
    check({entry_point})
    print("All tests passed!")
"""

        files = {"solution.py": code_to_test, "tests.py": test_file_content}

        is_stdin_test = 'livecodebench' in task_name.lower() and 'subprocess' in (test_code or '')
        is_ds1000 = 'ds1000' in task_name.lower() or 'ds_1000' in task_name.lower()

        if self.cfg.pre_sanitize and language in _SANITIZERS and not is_stdin_test and not is_ds1000:
            schema = TaskSchema(
                language=language, file_name="solution.py",
                entry_point=entry_point or "solve", java_class="Solution",
                prefer_rename=True, allow_wrapper=True
            )
            sanitizer = _SANITIZERS[language]
            raw = files.get("solution.py")
            if raw:
                out = sanitizer.normalize(raw, schema)
                files["solution.py"] = out.files.get("solution.py", raw)

        return self._execute_in_docker(files, task_name, language, kwargs)

    def _execute_in_docker(self, files, task_name, language, kwargs):
        """Execute code in Docker sandbox and return EvalResult."""
        try:
            from wisent.core.evaluators.benchmark_specific.coding.safe_docker.core.atoms import Job

            timeout_override = kwargs.get('timeout')
            if 'ds1000' in task_name.lower() or 'ds_1000' in task_name.lower():
                cpu_limit_s, wall_timeout_s, nproc = DS1000_CPU_LIMIT_S, DS1000_WALL_TIMEOUT_S, DS1000_NPROC
            elif timeout_override:
                cpu_limit_s, wall_timeout_s, nproc = timeout_override, timeout_override, SAFE_DOCKER_NPROC_DEFAULT
            else:
                cpu_limit_s = self.cfg.cpu_limit_s
                wall_timeout_s = self.cfg.time_limit_s
                nproc = SAFE_DOCKER_NPROC_DEFAULT

            job = Job(
                language=language, compile_argv=None,
                run_argv=["python3", "tests.py"],
                cpu_limit_s=cpu_limit_s, wall_timeout_s=wall_timeout_s,
                mem_limit_mb=self.cfg.mem_limit_mb,
                fsize_mb=SAFE_DOCKER_FSIZE_MB, nproc=nproc, nofile=SAFE_DOCKER_NOFILE,
            )

            result = self.exec.run(files, job)

            if result.status == "ok":
                return EvalResult(
                    ground_truth="TRUTHFUL", method_used=self.name, confidence=1.0,
                    details=f"Code executed successfully. Status: {result.status}",
                    meta={"exit_code": result.exit_code, "elapsed": result.elapsed,
                          "stdout": result.stdout[:DISPLAY_TRUNCATION_LARGE] if result.stdout else "",
                          "stderr": result.stderr[:DISPLAY_TRUNCATION_LARGE] if result.stderr else ""},
                )
            else:
                error_info = f"stdout: {result.stdout[:DISPLAY_TRUNCATION_LARGE] if result.stdout else ''}\nstderr: {result.stderr[:DISPLAY_TRUNCATION_LARGE] if result.stderr else ''}"
                return EvalResult(
                    ground_truth="UNTRUTHFUL", method_used=self.name, confidence=1.0,
                    details=f"Code execution failed. Status: {result.status}\n{error_info}",
                    meta={"exit_code": result.exit_code, "elapsed": result.elapsed,
                          "stdout": result.stdout[:DISPLAY_TRUNCATION_LARGE] if result.stdout else "",
                          "stderr": result.stderr[:DISPLAY_TRUNCATION_LARGE] if result.stderr else ""},
                )

        except Exception as e:
            logger.error(f"Docker execution failed: {e}")
            return EvalResult(
                ground_truth="UNKNOWN", method_used=self.name,
                confidence=0.0, details=f"Docker execution error: {str(e)}",
            )

    # --- Batch evaluation methods (from _evaluator_helpers.py) ---
    from wisent.core.evaluators.benchmark_specific.coding.metrics._evaluator_helpers import (
        _feedback, _run_once, _maybe_sanitize, evaluate_all,
    )


# Backward compatibility alias
DockerCodeEvaluator = CodingEvaluator
