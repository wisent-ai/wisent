"""Stateful agent evaluators using LLM-as-judge.

- tau-bench: stateful database comparison via LLM judge
- TravelPlanner: constraint checking via LLM judge
"""

from typing import Any
import logging

from wisent.core.reading.evaluators.core.atoms import BaseEvaluator, EvalResult
from wisent.core.utils.config_tools.constants import (
    EVAL_CONFIDENCE_FULL,
    EVAL_CONFIDENCE_ZERO,
    DISPLAY_TRUNCATION_MEDIUM,
)

logger = logging.getLogger(__name__)

_TAU_BENCH_JUDGE = (
    "You are evaluating agent task completion for a stateful task "
    "(per tau-bench methodology). The agent should modify a database "
    "or system state to match the expected outcome.\n\n"
    "Task: {prompt}\n"
    "Expected final state: {expected}\n"
    "Agent output: {response}\n\n"
    "Does the agent's output indicate the task was completed "
    "successfully (state matches expected)?\n"
    "Reply with exactly PASS or FAIL."
)

_TRAVELPLANNER_JUDGE = (
    "You are evaluating a travel plan (per TravelPlanner methodology). "
    "Check if the plan satisfies all constraints.\n\n"
    "Travel request: {prompt}\n"
    "Expected plan constraints: {expected}\n"
    "Generated plan: {response}\n\n"
    "Constraints to check: budget limits, time constraints, "
    "location validity, transportation feasibility, "
    "accommodation availability.\n"
    "Does the plan satisfy ALL constraints?\n"
    "Reply with exactly PASS or FAIL."
)


class TauBenchEvaluator(BaseEvaluator):
    """tau-bench: stateful database task completion via LLM judge.

    Paper: tau-bench. Method: Stateful database comparison, binary
    task success. Uses LLM-as-judge for state comparison.
    Metric: pass^k.
    """

    name = "tau_bench"
    description = "tau-bench: stateful task completion via LLM judge"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        from wisent.core.reading.evaluators.benchmark_specific.judge_utils import (
            require_judge_model, call_judge, parse_binary_verdict,
        )
        model = require_judge_model(kwargs, self.name)
        question = kwargs.get("question", str(expected))
        prompt = _TAU_BENCH_JUDGE.format(
            prompt=question, expected=str(expected), response=response)
        verdict = call_judge(model, prompt)
        gt = parse_binary_verdict(verdict, ("pass",), ("fail",))
        conf = EVAL_CONFIDENCE_FULL if gt != "UNKNOWN" else EVAL_CONFIDENCE_ZERO
        return EvalResult(
            ground_truth=gt, method_used=self.name, confidence=conf,
            details=f"Judge: {verdict[:DISPLAY_TRUNCATION_MEDIUM]}",
            meta={"judge_output": verdict, "expected": str(expected)})


class TravelPlannerEvaluator(BaseEvaluator):
    """TravelPlanner: constraint satisfaction via LLM judge.

    Paper: TravelPlanner. Method: Programmatic constraint checking,
    binary pass/fail per constraint. Uses LLM-as-judge for constraint
    assessment. Metric: Final Pass Rate.
    """

    name = "travelplanner"
    description = "TravelPlanner: constraint checking via LLM judge"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        from wisent.core.reading.evaluators.benchmark_specific.judge_utils import (
            require_judge_model, call_judge, parse_binary_verdict,
        )
        model = require_judge_model(kwargs, self.name)
        question = kwargs.get("question", str(expected))
        prompt = _TRAVELPLANNER_JUDGE.format(
            prompt=question, expected=str(expected), response=response)
        verdict = call_judge(model, prompt)
        gt = parse_binary_verdict(verdict, ("pass",), ("fail",))
        conf = EVAL_CONFIDENCE_FULL if gt != "UNKNOWN" else EVAL_CONFIDENCE_ZERO
        return EvalResult(
            ground_truth=gt, method_used=self.name, confidence=conf,
            details=f"Judge: {verdict[:DISPLAY_TRUNCATION_MEDIUM]}",
            meta={"judge_output": verdict, "expected": str(expected)})
