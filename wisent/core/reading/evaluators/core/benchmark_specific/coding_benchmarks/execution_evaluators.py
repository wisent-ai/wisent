"""Execution-based coding benchmark evaluators.

Each evaluator delegates code execution to CodingEvaluator's Docker
sandbox and adds benchmark-specific logic:
- AiderPolyglot: two-attempt execution
- Mercury: pass@k + runtime measurement
- Codeforces: all test cases must pass
- LiveCodeBenchV6: multi-candidate pass@k
- ReCode: worst-case across perturbations
- SciCode: numerical domain-specific tests
"""

from typing import Any
import logging

from wisent.core.reading.evaluators.core.atoms import BaseEvaluator, EvalResult
from wisent.core.reading.evaluators.benchmark_specific.coding.metrics.evaluator import (
    CodingEvaluator, EvaluatorConfig,
)
from wisent.core.utils.config_tools.constants import (
    EVAL_CONFIDENCE_FULL,
    EVAL_CONFIDENCE_ZERO,
    EVAL_NUM_CONTRASTIVE_CHOICES,
    EVAL_CONTRASTIVE_CORRECT_IDX,
    EVAL_CONTRASTIVE_INCORRECT_IDX,
    FEEDBACK_MAX_CHARS,
    SAFE_DOCKER_FSIZE_MB,
    SAFE_DOCKER_NOFILE,
)

logger = logging.getLogger(__name__)

_CODING_EVAL = None


def _get_coding_evaluator() -> CodingEvaluator:
    """Lazy-load shared CodingEvaluator instance."""
    global _CODING_EVAL
    if _CODING_EVAL is None:
        cfg = EvaluatorConfig(
            feedback_max_chars=FEEDBACK_MAX_CHARS,
            fsize_mb=SAFE_DOCKER_FSIZE_MB,
            nofile=SAFE_DOCKER_NOFILE,
        )
        _CODING_EVAL = CodingEvaluator(cfg=cfg)
    return _CODING_EVAL


def _exec_code(response: str, expected: Any, bench_name: str, **kwargs) -> EvalResult:
    """Execute code via CodingEvaluator Docker sandbox."""
    kwargs.pop("task_name", None)
    evaluator = _get_coding_evaluator()
    return evaluator.evaluate(response, expected, task_name=bench_name, **kwargs)


class AiderPolyglotEvaluator(BaseEvaluator):
    """Aider Polyglot: unit test execution with two attempts.

    Paper: Aider Polyglot. Method: Execute generated code against unit
    tests. If first attempt fails, model gets error feedback for a
    second attempt. Metric: Pass rate.
    """

    name = "aider_polyglot"
    description = "Aider Polyglot: unit test execution with two attempts"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        result = _exec_code(response, expected, "aider_polyglot", **kwargs)
        if result.ground_truth == "TRUTHFUL":
            return EvalResult(
                ground_truth="TRUTHFUL", method_used=self.name,
                confidence=result.confidence,
                details=f"First attempt passed. {result.details}",
                meta=result.meta)
        repair_fn = kwargs.get("repair_fn")
        if repair_fn and result.meta:
            feedback = result.meta.get("stderr", "") or result.details
            repaired = repair_fn(response, feedback)
            if repaired:
                result2 = _exec_code(repaired, expected, "aider_polyglot", **kwargs)
                return EvalResult(
                    ground_truth=result2.ground_truth, method_used=self.name,
                    confidence=result2.confidence,
                    details=f"Second attempt: {result2.details}",
                    meta=result2.meta)
        return EvalResult(
            ground_truth=result.ground_truth, method_used=self.name,
            confidence=result.confidence,
            details=f"Failed (no second attempt available). {result.details}",
            meta=result.meta)


class MercuryEvaluator(BaseEvaluator):
    """Mercury: pass@k + runtime-percentile-weighted Beyond@k.

    Paper: Mercury. Method: Execution-based Pass@k plus runtime
    measurement for efficiency scoring. Metric: Pass@k, Beyond@k.
    """

    name = "mercury"
    description = "Mercury: pass@k with runtime efficiency measurement"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        result = _exec_code(response, expected, "mercury", **kwargs)
        elapsed = result.meta.get("elapsed") if result.meta else None
        meta = dict(result.meta) if result.meta else {}
        meta["runtime_s"] = elapsed
        return EvalResult(
            ground_truth=result.ground_truth, method_used=self.name,
            confidence=result.confidence,
            details=f"{result.details} Runtime: {elapsed}s" if elapsed else result.details,
            meta=meta)


class CodeforcesEvaluator(BaseEvaluator):
    """Codeforces/CodeElo: all test cases must pass.

    Paper: CodeElo. Method: Solutions executed against comprehensive
    test cases, must pass ALL. Metric: pass@k, Elo.
    """

    name = "codeforces"
    description = "Codeforces: all-or-nothing test case execution"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        result = _exec_code(response, expected, "codeforces", **kwargs)
        return EvalResult(
            ground_truth=result.ground_truth, method_used=self.name,
            confidence=result.confidence, details=result.details,
            meta=result.meta)


class LiveCodeBenchV6Evaluator(BaseEvaluator):
    """LiveCodeBench v6: test case execution, pass@k.

    Paper: LiveCodeBench. Method: Test case execution with multiple
    candidates, fraction passing all tests. Metric: pass@k.
    """

    name = "livecodebench_v6"
    description = "LiveCodeBench v6: test case execution"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        result = _exec_code(response, expected, "livecodebench_v6", **kwargs)
        return EvalResult(
            ground_truth=result.ground_truth, method_used=self.name,
            confidence=result.confidence, details=result.details,
            meta=result.meta)


class RecodeEvaluator(BaseEvaluator):
    """ReCode: worst-case robustness across input perturbations.

    Paper: ReCode. Method: Code executed against test cases, worst-case
    across perturbations. Metric: Robust Pass@k, Robust Drop%.
    """

    name = "recode"
    description = "ReCode: worst-case robustness code execution"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        perturbations = kwargs.get("perturbations", [response])
        worst_result = None
        for perturbed in perturbations:
            result = _exec_code(perturbed, expected, "recode", **kwargs)
            if worst_result is None or result.ground_truth != "TRUTHFUL":
                worst_result = result
                if result.ground_truth != "TRUTHFUL":
                    break
        if worst_result is None:
            worst_result = _exec_code(response, expected, "recode", **kwargs)
        return EvalResult(
            ground_truth=worst_result.ground_truth, method_used=self.name,
            confidence=worst_result.confidence,
            details=f"Worst-case: {worst_result.details}",
            meta=worst_result.meta)


class SciCodeEvaluator(BaseEvaluator):
    """SciCode: execution with numerical domain-specific tests.

    Paper: SciCode. Method: Execution-based with numerical and
    domain-specific tests. Metric: pass@k.
    """

    name = "scicode"
    description = "SciCode: execution with numerical domain tests"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        result = _exec_code(response, expected, "scicode", **kwargs)
        return EvalResult(
            ground_truth=result.ground_truth, method_used=self.name,
            confidence=result.confidence, details=result.details,
            meta=result.meta)
