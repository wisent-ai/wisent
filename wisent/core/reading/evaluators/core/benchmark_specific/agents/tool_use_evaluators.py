"""Tool-use evaluators for function calling and tool benchmarks.

Covers BFCL (Berkeley Function Calling Leaderboard) and ToolBench (ToolLLM).
"""

import ast
import re
from typing import Any
import logging

from wisent.core.reading.evaluators.core.atoms import BaseEvaluator, EvalResult
from wisent.core.utils.config_tools.constants import (
    EVAL_CONFIDENCE_FULL,
    EVAL_CONFIDENCE_ZERO,
    EVAL_CONFIDENCE_CONTAINMENT,
    EVAL_FUNCTION_NAME_WEIGHT,
    EVAL_PARAM_PRESENCE_WEIGHT,
    EVAL_NUM_CONTRASTIVE_CHOICES,
    EVAL_CONTRASTIVE_CORRECT_IDX,
    EVAL_CONTRASTIVE_INCORRECT_IDX,
    EVAL_BEST_SCORE_INIT,
    EVAL_COUNTER_INIT,
    BINARY_CLASS_POSITIVE,
    DISPLAY_TRUNCATION_MEDIUM,
)

logger = logging.getLogger(__name__)


def _parse_func_call(text: str) -> dict | None:
    """Parse a function call string into structured form using ast.parse.

    Returns dict with 'name' (str) and 'kwargs' (dict of param_name -> value)
    or None if parsing fails.
    """
    try:
        tree = ast.parse(text.strip(), mode="eval")
        if not isinstance(tree.body, ast.Call):
            return None
        call = tree.body
        name = ast.dump(call.func) if not isinstance(call.func, ast.Name) else call.func.id
        if isinstance(call.func, ast.Attribute):
            name = ast.unparse(call.func)
        kwargs = {}
        for kw in call.keywords:
            if kw.arg is not None:
                kwargs[kw.arg] = ast.literal_eval(kw.value)
        for i, arg in enumerate(call.args):
            try:
                kwargs[f"__pos_{i}"] = ast.literal_eval(arg)
            except (ValueError, TypeError):
                kwargs[f"__pos_{i}"] = ast.unparse(arg)
        return {"name": name, "kwargs": kwargs}
    except Exception:
        return None


class BFCLEvaluator(BaseEvaluator):
    """Evaluator for BFCL (Berkeley Function Calling Leaderboard).

    Paper: BFCL. Metric: Accuracy/pass rate (all-or-nothing).
    Method: AST evaluation + executable evaluation. For contrastive eval,
    compares function call structure: function name matching and parameter presence.
    """

    name = "bfcl"
    description = "BFCL evaluator using function call structure comparison"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate function call correctness.

        Args:
            response: Generated function call string.
            expected: Expected function call string or list of acceptable calls.
            **kwargs:
                choices: [correct_answer, incorrect_answer] for contrastive eval.

        Returns:
            EvalResult with TRUTHFUL if function call matches, UNTRUTHFUL otherwise.
        """
        choices = kwargs.get("choices")

        if choices and len(choices) == EVAL_NUM_CONTRASTIVE_CHOICES:
            return self._evaluate_contrastive(choices, expected)

        return self._evaluate_single(response, expected)

    def _evaluate_single(self, response: str, expected: Any) -> EvalResult:
        """Check if a single response matches the expected function call."""
        expected_list = expected if isinstance(expected, list) else [expected]
        resp_str = str(response)

        for exp in expected_list:
            score = self._call_similarity(resp_str, str(exp))
            if score >= EVAL_CONFIDENCE_FULL:
                return EvalResult(
                    ground_truth="TRUTHFUL",
                    method_used=self.name,
                    confidence=EVAL_CONFIDENCE_FULL,
                    details=f"Function call matches expected '{exp}'",
                    meta={"matched_call": exp, "score": score},
                )

        best_score = EVAL_BEST_SCORE_INIT
        for exp in expected_list:
            score = self._call_similarity(resp_str, str(exp))
            best_score = max(best_score, score)

        if best_score >= EVAL_CONFIDENCE_CONTAINMENT:
            return EvalResult(
                ground_truth="TRUTHFUL",
                method_used=self.name,
                confidence=best_score,
                details=f"Partial function call match (score={best_score})",
                meta={"score": best_score},
            )

        return EvalResult(
            ground_truth="UNTRUTHFUL",
            method_used=self.name,
            confidence=EVAL_CONFIDENCE_ZERO,
            details=f"Function call does not match expected",
        )

    def _evaluate_contrastive(self, choices: list, expected: Any) -> EvalResult:
        """Contrastive evaluation comparing two function call choices."""
        correct_answer = choices[EVAL_CONTRASTIVE_CORRECT_IDX]
        incorrect_answer = choices[EVAL_CONTRASTIVE_INCORRECT_IDX]
        expected_list = expected if isinstance(expected, list) else [expected]

        correct_score = self._best_call_score(str(correct_answer), expected_list)
        incorrect_score = self._best_call_score(str(incorrect_answer), expected_list)

        meta = {
            "correct_answer": correct_answer,
            "incorrect_answer": incorrect_answer,
            "correct_score": correct_score,
            "incorrect_score": incorrect_score,
        }

        if correct_score > incorrect_score:
            return EvalResult(
                ground_truth="TRUTHFUL", method_used=self.name, confidence=correct_score,
                details="Correct call closer to expected", meta=meta,
            )
        elif incorrect_score > correct_score:
            return EvalResult(
                ground_truth="UNTRUTHFUL", method_used=self.name, confidence=incorrect_score,
                details="Incorrect call closer to expected", meta=meta,
            )

        return EvalResult(
            ground_truth="UNKNOWN", method_used=self.name, confidence=EVAL_CONFIDENCE_ZERO,
            details="Ambiguous: both scores equal", meta=meta,
        )

    def _call_similarity(self, candidate: str, expected: str) -> float:
        """Compare function calls using AST parsing (all-or-nothing per BFCL).

        Parses both strings as Python function calls, then checks:
        - Function name must match
        - All expected keyword arguments must be present with matching values
        Returns EVAL_CONFIDENCE_FULL for match, EVAL_CONFIDENCE_ZERO otherwise.
        """
        if self.normalize_text(candidate) == self.normalize_text(expected):
            return EVAL_CONFIDENCE_FULL
        cand_parsed = _parse_func_call(candidate)
        exp_parsed = _parse_func_call(expected)
        if cand_parsed is None or exp_parsed is None:
            return EVAL_CONFIDENCE_ZERO
        if cand_parsed["name"] != exp_parsed["name"]:
            return EVAL_CONFIDENCE_ZERO
        for key, exp_val in exp_parsed["kwargs"].items():
            if key not in cand_parsed["kwargs"]:
                return EVAL_CONFIDENCE_ZERO
            if cand_parsed["kwargs"][key] != exp_val:
                return EVAL_CONFIDENCE_ZERO
        return EVAL_CONFIDENCE_FULL

    def _best_call_score(self, candidate: str, expected_list: list) -> float:
        """Return the best call similarity score across expected list."""
        best = EVAL_BEST_SCORE_INIT
        for exp in expected_list:
            score = self._call_similarity(candidate, str(exp))
            best = max(best, score)
        return best


_TOOLBENCH_JUDGE = (
    "You are evaluating tool-use task completion quality "
    "(per ToolLLM/ToolBench ToolEval methodology).\n\n"
    "Task: {prompt}\n"
    "Expected outcome: {expected}\n"
    "Agent response: {response}\n\n"
    "Did the agent successfully complete the task using the "
    "appropriate tools? Reply with exactly PASS or FAIL."
)


class ToolBenchEvaluator(BaseEvaluator):
    """ToolBench (ToolLLM): LLM-as-judge task completion evaluation.

    Paper: ToolLLM. Method: ChatGPT ToolEval for binary pass/fail
    and preference comparison. Uses LLM-as-judge.
    Metric: Pass Rate + Win Rate.
    """

    name = "toolbench"
    description = "ToolBench: LLM-as-judge tool completion evaluation"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        from wisent.core.reading.evaluators.benchmark_specific.judge_utils import (
            require_judge_model, call_judge, parse_binary_verdict,
        )
        model = require_judge_model(kwargs, self.name)
        question = kwargs.get("question", str(expected))
        prompt = _TOOLBENCH_JUDGE.format(
            prompt=question, expected=str(expected), response=response)
        verdict = call_judge(model, prompt)
        gt = parse_binary_verdict(verdict, ("pass",), ("fail",))
        conf = EVAL_CONFIDENCE_FULL if gt != "UNKNOWN" else EVAL_CONFIDENCE_ZERO
        return EvalResult(
            ground_truth=gt, method_used=self.name, confidence=conf,
            details=f"Judge: {verdict[:DISPLAY_TRUNCATION_MEDIUM]}",
            meta={"judge_output": verdict, "expected": str(expected)})
