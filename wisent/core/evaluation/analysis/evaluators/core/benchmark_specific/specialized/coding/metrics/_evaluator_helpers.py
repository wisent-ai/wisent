"""
Batch evaluation helpers for CodingEvaluator.

Split from evaluator.py to meet 300-line limit.
Contains _feedback, _run_once, _maybe_sanitize, evaluate_all, and backward compat alias.
"""

from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

from wisent.core.evaluators.benchmark_specific.coding.safe_docker.recipes import RECIPE_REGISTRY
from wisent.core.evaluators.benchmark_specific.coding.metrics.core.atoms import SampleOutcome
from wisent.core.errors import MissingParameterError
from wisent.core.evaluators.benchmark_specific.coding.output_sanitizer.core.atoms import TaskSchema

if TYPE_CHECKING:
    from wisent.core.evaluators.benchmark_specific.coding.safe_docker.core.atoms import Result
    from wisent.core.evaluators.benchmark_specific.coding.providers.core.atoms import CodingTask
    from wisent.core.evaluators.benchmark_specific.coding.output_sanitizer.core.atoms import CodeStandardizer


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


def _run_once(self, task: "CodingTask", files: dict[str, str]) -> "Result":
    """Runs a single evaluation job for the given task and files."""
    recipe = RECIPE_REGISTRY[task.language]
    job = recipe.make_job(**task.options,
                          time_limit_s=self.cfg.time_limit_s,
                          cpu_limit_s=self.cfg.cpu_limit_s,
                          mem_limit_mb=self.cfg.mem_limit_mb)
    return self.exec.run(files, job)


def _maybe_sanitize(self, task: "CodingTask", files: dict[str, str]) -> dict[str, str]:
    """Optionally sanitizes the generated files based on the task schema."""
    from wisent.core.evaluators.benchmark_specific.coding.metrics.evaluator import (
        _SANITIZERS, _make_schema,
    )

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
