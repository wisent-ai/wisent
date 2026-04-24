"""SWE-bench family and related code execution evaluators.

Each evaluator delegates to CodingEvaluator Docker sandbox:
- SWEBenchVerified: patch application + FAIL_TO_PASS/PASS_TO_PASS tests
- SWEBench: same methodology as verified
- MultiSWEBench: SWE-bench across multiple languages
- OJBench: competition test cases, pass ALL
- TerminalBench: pytest on Docker container final state
- NL2Bash: character-based BLEU (text metric, not execution)
"""

from typing import Any
import logging

import evaluate as hf_evaluate

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
    EVAL_NL2BASH_MATCH_THRESHOLD,
    BLEU_MAX_ORDER,
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


class SWEBenchVerifiedEvaluator(BaseEvaluator):
    """SWE-bench Verified: apply patches, run FAIL_TO_PASS + PASS_TO_PASS.

    Paper: SWE-bench. Method: Apply patches in Docker, run both
    FAIL_TO_PASS (must now pass) and PASS_TO_PASS (must still pass)
    unit tests. Metric: % Resolved.
    """

    name = "swe_bench_verified"
    description = "SWE-bench Verified: patch + dual test suite execution"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        result = _exec_code(response, expected, "swe_bench_verified", **kwargs)
        return EvalResult(
            ground_truth=result.ground_truth, method_used=self.name,
            confidence=result.confidence, details=result.details,
            meta=result.meta)


class SWEBenchEvaluator(BaseEvaluator):
    """SWE-bench: same as Verified, original dataset.

    Paper: SWE-bench. Method: Apply patches in Docker, run unit tests.
    Metric: % Resolved.
    """

    name = "swe_bench"
    description = "SWE-bench: patch application and test execution"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        result = _exec_code(response, expected, "swe_bench", **kwargs)
        return EvalResult(
            ground_truth=result.ground_truth, method_used=self.name,
            confidence=result.confidence, details=result.details,
            meta=result.meta)


class MultiSWEBenchEvaluator(BaseEvaluator):
    """Multi-SWE-bench: SWE-bench across multiple languages.

    Paper: Multi-SWE-bench. Method: Same as SWE-bench but supports
    Python, Java, JS, TypeScript, Go, Rust, C++. Metric: Resolved Rate.
    """

    name = "multi_swe_bench"
    description = "Multi-SWE-bench: multilingual patch + test execution"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        language = kwargs.get("language", "python")
        result = _exec_code(
            response, expected, "multi_swe_bench", language=language, **kwargs)
        return EvalResult(
            ground_truth=result.ground_truth, method_used=self.name,
            confidence=result.confidence,
            details=f"[{language}] {result.details}",
            meta=result.meta)


class OJBenchEvaluator(BaseEvaluator):
    """OJBench: competition test cases, must pass ALL.

    Paper: OJBench. Method: Pass ALL test cases from competition
    organizers. Metric: Pass@n.
    """

    name = "ojbench"
    description = "OJBench: all-or-nothing competition test execution"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        result = _exec_code(response, expected, "ojbench", **kwargs)
        return EvalResult(
            ground_truth=result.ground_truth, method_used=self.name,
            confidence=result.confidence, details=result.details,
            meta=result.meta)


class TerminalBenchEvaluator(BaseEvaluator):
    """Terminal-Bench: pytest on Docker container final state.

    Paper: Terminal-Bench. Method: Execute commands in Docker, then
    run pytest on the resulting container state. Metric: pass@k.
    """

    name = "terminal_bench"
    description = "Terminal-Bench: pytest on Docker container state"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        result = _exec_code(response, expected, "terminal_bench", **kwargs)
        return EvalResult(
            ground_truth=result.ground_truth, method_used=self.name,
            confidence=result.confidence, details=result.details,
            meta=result.meta)


class NL2BashEvaluator(BaseEvaluator):
    """NL2Bash: character-based BLEU for bash command generation.

    Paper: NL2Bash. Method: Character-based BLEU, template matching.
    Metric: BLEU, Template Accuracy.
    """

    name = "nl2bash"
    description = "NL2Bash: character-based BLEU for bash generation"

    def _compute_char_bleu(self, candidate: str, reference: str) -> float:
        """Compute character-level BLEU as specified in the NL2Bash paper."""
        bleu_metric = hf_evaluate.load("bleu")
        result = bleu_metric.compute(
            predictions=[candidate],
            references=[[reference]],
            tokenizer=lambda s: list(s),
            max_order=BLEU_MAX_ORDER,
            smooth=False,
        )
        return result["bleu"]

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        choices = kwargs.get("choices")
        if choices and len(choices) == EVAL_NUM_CONTRASTIVE_CHOICES:
            return self._eval_contrastive(choices, expected)
        return self._eval_direct(response, str(expected))

    def _eval_contrastive(self, choices: list, expected: Any) -> EvalResult:
        exp_str = str(expected).strip()
        c_bleu = self._compute_char_bleu(
            str(choices[EVAL_CONTRASTIVE_CORRECT_IDX]).strip(), exp_str)
        i_bleu = self._compute_char_bleu(
            str(choices[EVAL_CONTRASTIVE_INCORRECT_IDX]).strip(), exp_str)
        meta = {"correct_bleu": c_bleu, "incorrect_bleu": i_bleu}
        if c_bleu > i_bleu:
            return EvalResult(
                ground_truth="TRUTHFUL", method_used=self.name,
                confidence=c_bleu,
                details=f"Correct BLEU={c_bleu:.4f} > Incorrect={i_bleu:.4f}",
                meta=meta)
        if i_bleu > c_bleu:
            return EvalResult(
                ground_truth="UNTRUTHFUL", method_used=self.name,
                confidence=i_bleu,
                details=f"Incorrect BLEU={i_bleu:.4f} > Correct={c_bleu:.4f}",
                meta=meta)
        return EvalResult(
            ground_truth="UNKNOWN", method_used=self.name,
            confidence=EVAL_CONFIDENCE_ZERO,
            details="Equal character BLEU scores", meta=meta)

    def _eval_direct(self, response: str, expected: str) -> EvalResult:
        if self.normalize_text(response) == self.normalize_text(expected):
            return EvalResult(
                ground_truth="TRUTHFUL", method_used=self.name,
                confidence=EVAL_CONFIDENCE_FULL,
                details="Exact match after normalization")
        bleu = self._compute_char_bleu(response.strip(), expected.strip())
        gt = "TRUTHFUL" if bleu >= EVAL_NL2BASH_MATCH_THRESHOLD else "UNTRUTHFUL"
        return EvalResult(
            ground_truth=gt, method_used=self.name, confidence=bleu,
            details=f"Character BLEU: {bleu:.4f}",
            meta={"char_bleu": bleu})
