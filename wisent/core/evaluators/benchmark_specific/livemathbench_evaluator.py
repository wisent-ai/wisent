"""LiveMathBench evaluator for mathematical olympiad problems.

This evaluator implements two evaluation methods for LiveMathBench:
1. Answer extraction + math_equal comparison (like PolyMath)
2. LLM-as-a-judge evaluation

Also implements G-Pass@k metrics from the LiveMathBench paper (arxiv.org/abs/2412.13147):
- Greedy accuracy: Single-shot accuracy with temperature=0
- Pass@k: Probability that at least one of k samples is correct (G-Pass@k with τ=0)
- G-Pass@k(τ): Probability that at least τ*k of k samples are correct
- mG-Pass@k: Mean G-Pass@k integrated over τ ∈ [0.5, 1.0]

Implementation follows the official GPassK repository:
https://github.com/open-compass/GPassK
"""

import logging
import numpy as np
from typing import Any, List
from scipy.stats import hypergeom

from wisent.core.evaluators.benchmark_specific.math_parsing.scripts import multi_math_equal
from wisent.core.evaluators.core.atoms import BaseEvaluator, EvalResult

logger = logging.getLogger(__name__)


# Language-specific prompts for boxed answer format
LANGUAGE_PROMPTS = {
    "en": "Note: Please put the final answer in the $\\boxed{}$.",
    "cn": "注意：请将最终答案放在 $\\boxed{}$ 中。",
}

# LLM-as-a-judge prompts from the LiveMathBench paper
JUDGE_PROMPT_EN = """Please act as an expert in grading mathematics exam papers, and judge whether the following answers match the standard answers, i.e., whether the examinee answered correctly. Here are some evaluation criteria:
1. Some answers may contain multiple parts, such as single-choice questions, multiple-choice questions, fill-in-the-blank questions, and problem-solving questions. As long as the answer matches the standard answer, it is considered correct. For multiple-choice questions and fill-in-the-blank questions with multiple blanks, the examinee must answer all corresponding options or blanks correctly to be considered correct.
2. Some answers may be expressed in different ways; for example, some answers may be mathematical expressions, while others may be textual descriptions. As long as the meaning conveyed is consistent, it is considered correct. Additionally, some formulas may be expressed differently but are equivalent, which is also considered correct.
3. You do not need to recalculate the problem answers, as the standard answers are already provided. You only need to judge whether the examinee's answer matches the standard answer based on the form of the question and whether it is correct.
Please judge whether the following answer matches the standard answer according to the above criteria. If they match, output \\boxed{{yes}}, otherwise output \\boxed{{no}}. If it is difficult to judge, also output \\boxed{{no}}.
Original Question: {question}
Standard Answer: {reference_answer}
Examinee's Answer: {candidate_answer}
Analysis:"""

JUDGE_PROMPT_CN = """请你作为一个数学阅卷专家，判断下面的答案是否与标准答案一致，即考生是否回答正确。下面是一些评判标准：
1. 有些答案可能包含多项内容，可能有单选题，多选题，填空题和问答题，只要答案与标准答案一致即可, 对于多选题和多个空的填空题，需要考生对应的选项或空都回答正确才算正确。
2. 有些答案可能通过不同的方式表达，比如有些答案可能是一个数学表达式，有些答案可能是一个文字描述，只要表达的意思一致即可。且有些公式通过不同的方式表达，但等价，也是正确的。
3. 你不需要重新计算问题答案，因为标准答案已经给出，只需要根据问题形式来判断考生的答案是否与标准答案一致，是否正确即可。
请你根据上述标准，判断下面的答案是否与标准答案一致，如果一致，请在最后输出\\boxed{{yes}}, 否则输出\\boxed{{no}}, 如果难以判断，请输出\\boxed{{no}}.
原问题：{question}
标准答案：{reference_answer}
考生答案：{candidate_answer}
分析："""

JUDGE_PROMPTS = {
    "en": JUDGE_PROMPT_EN,
    "cn": JUDGE_PROMPT_CN,
}


def _compute_g_pass_at_k(n: int, c: int, k: int, m: int) -> float:
    """Internal G-Pass@k computation using hypergeometric survival function.

    Computes the probability that at least m of k randomly selected samples
    are correct, given n total samples with c correct ones.

    Uses scipy.stats.hypergeom.sf (survival function) for efficient computation.

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: Number of samples to select
        m: Minimum number of correct samples required

    Returns:
        Probability in [0, 1]
    """
    if m > min(c, k) or k > n or c < 0 or n <= 0 or m < 0:
        return 0.0
    # hypergeom.sf(m-1, n, c, k) = P(X >= m) where X ~ Hypergeometric(n, c, k)
    return hypergeom.sf(m - 1, n, c, k)


def compute_g_pass_at_k(n: int, c: int, k: int, tau: float) -> float:
    """Compute G-Pass@k with threshold τ.

    G-Pass@k(τ) is the probability that at least ⌈τ·k⌉ of k randomly selected
    samples are correct (with minimum 1).

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: Number of samples to select
        tau: Threshold fraction in [0, 1]

    Returns:
        G-Pass@k(τ) probability in [0, 1]
    """
    m = max(int(np.ceil(k * tau)), 1)
    return _compute_g_pass_at_k(n, c, k, m)


def compute_mg_pass_at_k(n: int, c: int, k: int) -> float:
    """Compute mG-Pass@k (mean G-Pass@k).

    mG-Pass@k = (2/k) * Σ(i=⌈k*0.5⌉+1 to k) G-Pass@k(i)

    This metric sums G-Pass@k values for thresholds from just above 50% to 100%,
    providing a single score that captures performance across majority voting thresholds.

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: Number of samples to select

    Returns:
        mG-Pass@k score in [0, 1]
    """
    low = int(np.ceil(k * 0.5))
    high = k

    mg_pass_at_k = 0.0
    for i in range(low + 1, high + 1):
        mg_pass_at_k += _compute_g_pass_at_k(n, c, k, i)
    mg_pass_at_k = 2 * mg_pass_at_k / k

    return mg_pass_at_k


def compute_metrics_for_problem(
    n: int,
    c: int,
    k_values: List[int] = [4, 8, 16],
    tau_values: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
) -> dict[str, float]:
    """Compute G-Pass@k metrics for a single problem.

    Args:
        n: Total number of samples
        c: Number of correct samples
        k_values: List of k values to compute metrics for
        tau_values: List of tau thresholds for G-Pass@k

    Returns:
        Dictionary with metrics for this problem
    """
    metrics = {}

    for k in k_values:
        if k > n:
            continue

        for tau in tau_values:
            metrics[f"G-Pass@{k}_{tau}"] = compute_g_pass_at_k(n, c, k, tau)

        metrics[f"mG-Pass@{k}"] = compute_mg_pass_at_k(n, c, k)

    return metrics


def compute_all_metrics(
    correct_counts: List[int],
    total_samples: int,
    k_values: List[int] = [4, 8, 16],
    tau_values: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
) -> dict[str, float]:
    """Compute all G-Pass@k metrics aggregated over a set of problems.

    Args:
        correct_counts: List of correct sample counts per problem (c_i for each problem)
        total_samples: Total samples per problem (n)
        k_values: List of k values to compute metrics for
        tau_values: List of tau thresholds for G-Pass@k

    Returns:
        Dictionary with all metrics (averaged across problems):
        - G-Pass@k_tau for each k and tau (tau=0 is equivalent to Pass@k)
        - mG-Pass@k for each k
    """
    num_problems = len(correct_counts)
    if num_problems == 0:
        return {}

    n = total_samples

    # Initialize accumulators
    metric_sums = {}

    for c in correct_counts:
        problem_metrics = compute_metrics_for_problem(n, c, k_values, tau_values)
        for key, value in problem_metrics.items():
            if key not in metric_sums:
                metric_sums[key] = 0.0
            metric_sums[key] += value

    # Average across problems
    metrics = {key: value / num_problems for key, value in metric_sums.items()}

    return metrics


class LiveMathBenchEvaluator(BaseEvaluator):
    """Evaluator for LiveMathBench mathematical olympiad benchmark.

    Supports two evaluation modes:
    1. Math extraction mode: Extract answer from \\boxed{} and compare using math_equal
    2. LLM-as-a-judge mode: Use an LLM to judge if the answer is correct

    Uses multi_math_equal from math_parsing for robust mathematical comparison.
    """

    name = "livemathbench"
    description = "LiveMathBench evaluator for mathematical olympiad problems"

    @staticmethod
    def get_prompt(question: str, language: str = "en") -> str:
        """Create prompt by appending language-specific instruction to the question.

        Args:
            question: The math question from the dataset
            language: Language code (en or cn)

        Returns:
            Question with language-specific instruction appended
        """
        instruction = LANGUAGE_PROMPTS.get(language, LANGUAGE_PROMPTS["en"])
        return f"{question}\n\n{instruction}"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate model response against expected LiveMathBench answer.

        Args:
            response: Model-generated response (may contain \\boxed{answer})
            expected: Expected answer string
            **kwargs:
                mode: "math" (default) or "llm_judge"

        Returns:
            EvalResult with TRUTHFUL/UNTRUTHFUL/UNKNOWN
        """
        mode = kwargs.get("mode", "math")

        if mode == "llm_judge":
            return self._evaluate_llm_judge(response, expected, **kwargs)
        else:
            return self._evaluate_math(response, expected, **kwargs)

    def _evaluate_math(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate using math extraction and comparison."""
        if not response or not response.strip():
            return EvalResult(
                ground_truth="UNKNOWN",
                method_used=self.name,
                confidence=0.0,
                details="Empty response",
                meta={
                    "response_preview": None,
                    "expected": expected,
                }
            )

        expected_str = str(expected).strip()

        # Use multi_math_equal which handles:
        # - Answer extraction (boxed, multilingual patterns, last number)
        # - Multiple comparison methods (numeric, symbolic, etc.)
        is_correct, predictions, no_boxed = multi_math_equal(expected_str, response)

        # Extract the model answer from predictions for logging
        model_answer = predictions[0][0] if predictions and predictions[0] else None

        return EvalResult(
            ground_truth="TRUTHFUL" if is_correct else "UNTRUTHFUL",
            method_used=self.name,
            confidence=1.0 if is_correct else 0.0,
            details=f"Model: '{model_answer}' vs Expected: '{expected_str}'",
            meta={
                "model_answer": model_answer,
                "expected_answer": expected_str,
                "is_equivalent": is_correct,
                "predictions": predictions,
                "no_boxed": no_boxed,
            }
        )

    def _evaluate_llm_judge(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate using LLM as a judge.

        Uses the official LiveMathBench judge prompts from the paper.

        Required kwargs:
            judge_model: The model to use for judging
            question: The original question

        Optional kwargs:
            language: "en" (default) or "cn" for Chinese prompt
        """
        from wisent.core.evaluators.benchmark_specific.math_parsing.extract_boxed import extract_boxed_answer

        model = kwargs.get("judge_model")
        if model is None:
            return EvalResult(
                ground_truth="UNKNOWN",
                method_used=f"{self.name}_llm_judge",
                confidence=0.0,
                details="No judge model provided",
                meta={"expected": expected}
            )

        question = kwargs.get("question", "")
        language = kwargs.get("language", "en")
        expected_str = str(expected).strip()

        # Get the appropriate judge prompt template
        judge_template = JUDGE_PROMPTS.get(language, JUDGE_PROMPT_EN)

        # Format the judge prompt
        judge_prompt = judge_template.format(
            question=question,
            reference_answer=expected_str,
            candidate_answer=response,
        )

        try:
            judge_response = model.generate(
                inputs=judge_prompt,
                max_new_tokens=1024,
                temperature=0.0,
                do_sample=False,
                prompt_is_formatted=True,
            )

            judge_output = judge_response[0] if judge_response else ""

            # Extract the verdict from \boxed{yes} or \boxed{no}
            boxed_answer = extract_boxed_answer(judge_output)
            if boxed_answer is not None:
                is_correct = boxed_answer.lower().strip() == "yes"
            else:
                # Fallback: check for yes/no in the output
                lower_output = judge_output.lower()
                is_correct = "yes" in lower_output and "no" not in lower_output.split("yes")[-1]

            return EvalResult(
                ground_truth="TRUTHFUL" if is_correct else "UNTRUTHFUL",
                method_used=f"{self.name}_llm_judge",
                confidence=1.0 if boxed_answer else 0.5,
                details=f"Judge verdict: {boxed_answer or 'unclear'}",
                meta={
                    "expected_answer": expected_str,
                    "judge_output": judge_output,
                    "boxed_verdict": boxed_answer,
                    "is_correct": is_correct,
                }
            )
        except Exception as e:
            logger.error(f"LLM judge evaluation failed: {e}")
            return EvalResult(
                ground_truth="UNKNOWN",
                method_used=f"{self.name}_llm_judge",
                confidence=0.0,
                details=f"Judge evaluation failed: {str(e)}",
                meta={"expected": expected}
            )

    def evaluate_multiple_samples(
        self,
        responses: list[str],
        expected: Any,
        **kwargs
    ) -> tuple[list[EvalResult], int]:
        """Evaluate multiple samples for the same problem.

        Used for computing G-Pass@k metrics.

        Args:
            responses: List of model responses for the same problem
            expected: Expected answer
            **kwargs: Additional arguments passed to evaluate()

        Returns:
            Tuple of (list of EvalResults, number of correct samples)
        """
        results = []
        correct_count = 0

        for response in responses:
            result = self.evaluate(response, expected, **kwargs)
            results.append(result)
            if result.ground_truth == "TRUTHFUL":
                correct_count += 1

        return results, correct_count
