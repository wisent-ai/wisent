"""AgentBench evaluator using LLM-as-judge.

Paper: AgentBench. Metric: Success Rate (binary per task).
Method: Environment-specific evaluation across multiple domains.
Uses LLM-as-judge to assess task completion quality.
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

_AGENTBENCH_JUDGE = (
    "You are evaluating agent task completion (per AgentBench methodology). "
    "AgentBench tests agents across multiple environments including OS, "
    "database, knowledge graph, web, and more.\n\n"
    "Task: {prompt}\n"
    "Expected outcome: {expected}\n"
    "Agent response: {response}\n\n"
    "Did the agent successfully complete the task? Compare the "
    "agent's output against the expected outcome.\n"
    "Reply with exactly SUCCESS or FAILURE."
)


class AgentBenchEvaluator(BaseEvaluator):
    """AgentBench: multi-environment task completion via LLM judge.

    Paper: AgentBench. Method: Environment-specific evaluators across
    OS, DB, KG, web, etc. with binary success/failure.
    Uses LLM-as-judge for task completion assessment.
    Metric: Success Rate, weighted average.
    """

    name = "agentbench"
    description = "AgentBench: multi-environment task evaluation via LLM judge"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        from wisent.core.reading.evaluators.benchmark_specific.judge_utils import (
            require_judge_model, call_judge, parse_binary_verdict,
        )
        model = require_judge_model(kwargs, self.name)
        question = kwargs.get("question", str(expected))
        prompt = _AGENTBENCH_JUDGE.format(
            prompt=question, expected=str(expected), response=response)
        verdict = call_judge(model, prompt)
        gt = parse_binary_verdict(
            verdict, ("success",), ("failure",))
        conf = EVAL_CONFIDENCE_FULL if gt != "UNKNOWN" else EVAL_CONFIDENCE_ZERO
        return EvalResult(
            ground_truth=gt, method_used=self.name, confidence=conf,
            details=f"Judge: {verdict[:DISPLAY_TRUNCATION_MEDIUM]}",
            meta={"judge_output": verdict, "expected": str(expected)})
