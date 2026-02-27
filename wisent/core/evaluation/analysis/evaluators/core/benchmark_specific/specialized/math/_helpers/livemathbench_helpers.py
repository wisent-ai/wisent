"""LiveMathBenchEvaluator class for mathematical olympiad problems.

Extracted from livemathbench_evaluator.py to keep file under 300 lines.
"""

import logging
from typing import Any

from wisent.core.evaluators.benchmark_specific.math_parsing.scripts import multi_math_equal
from wisent.core.evaluators.core.atoms import BaseEvaluator, EvalResult

from wisent.core.constants import CHANCE_LEVEL_ACCURACY
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


class LiveMathBenchEvaluator(BaseEvaluator):
    """Evaluator for LiveMathBench mathematical olympiad benchmark."""

    name = "livemathbench"
    description = "LiveMathBench evaluator for mathematical olympiad problems"

    @staticmethod
    def get_prompt(question: str, language: str = "en") -> str:
        """Create prompt by appending language-specific instruction to the question."""
        instruction = LANGUAGE_PROMPTS.get(language, LANGUAGE_PROMPTS["en"])
        return f"{question}\n\n{instruction}"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate model response against expected LiveMathBench answer."""
        mode = kwargs.get("mode", "math")
        if mode == "llm_judge":
            return self._evaluate_llm_judge(response, expected, **kwargs)
        else:
            return self._evaluate_math(response, expected, **kwargs)

    def _evaluate_math(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate using math extraction and comparison."""
        if not response or not response.strip():
            return EvalResult(
                ground_truth="UNKNOWN", method_used=self.name,
                confidence=0.0, details="Empty response",
                meta={"response_preview": None, "expected": expected}
            )
        expected_str = str(expected).strip()
        is_correct, predictions, no_boxed = multi_math_equal(expected_str, response)
        model_answer = predictions[0][0] if predictions and predictions[0] else None
        return EvalResult(
            ground_truth="TRUTHFUL" if is_correct else "UNTRUTHFUL",
            method_used=self.name,
            confidence=1.0 if is_correct else 0.0,
            details=f"Model: '{model_answer}' vs Expected: '{expected_str}'",
            meta={
                "model_answer": model_answer, "expected_answer": expected_str,
                "is_equivalent": is_correct, "predictions": predictions,
                "no_boxed": no_boxed,
            }
        )

    def _evaluate_llm_judge(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate using LLM as a judge."""
        from wisent.core.evaluators.benchmark_specific.math_parsing.extract_boxed import extract_boxed_answer
        model = kwargs.get("judge_model")
        if model is None:
            return EvalResult(
                ground_truth="UNKNOWN", method_used=f"{self.name}_llm_judge",
                confidence=0.0, details="No judge model provided",
                meta={"expected": expected}
            )
        question = kwargs.get("question", "")
        language = kwargs.get("language", "en")
        expected_str = str(expected).strip()
        judge_template = JUDGE_PROMPTS.get(language, JUDGE_PROMPT_EN)
        judge_prompt = judge_template.format(
            question=question, reference_answer=expected_str,
            candidate_answer=response,
        )
        try:
            judge_response = model.generate(
                inputs=judge_prompt, do_sample=False, prompt_is_formatted=True,
            )
            judge_output = judge_response[0] if judge_response else ""
            boxed_answer = extract_boxed_answer(judge_output)
            if boxed_answer is not None:
                is_correct = boxed_answer.lower().strip() == "yes"
            else:
                lower_output = judge_output.lower()
                is_correct = "yes" in lower_output and "no" not in lower_output.split("yes")[-1]
            return EvalResult(
                ground_truth="TRUTHFUL" if is_correct else "UNTRUTHFUL",
                method_used=f"{self.name}_llm_judge",
                confidence=1.0 if boxed_answer else CHANCE_LEVEL_ACCURACY,
                details=f"Judge verdict: {boxed_answer or 'unclear'}",
                meta={
                    "expected_answer": expected_str, "judge_output": judge_output,
                    "boxed_verdict": boxed_answer, "is_correct": is_correct,
                }
            )
        except Exception as e:
            logger.error(f"LLM judge evaluation failed: {e}")
            return EvalResult(
                ground_truth="UNKNOWN", method_used=f"{self.name}_llm_judge",
                confidence=0.0, details=f"Judge evaluation failed: {str(e)}",
                meta={"expected": expected}
            )

    def evaluate_multiple_samples(
        self, responses: list[str], expected: Any, **kwargs
    ) -> tuple[list[EvalResult], int]:
        """Evaluate multiple samples for the same problem."""
        results = []
        correct_count = 0
        for response in responses:
            result = self.evaluate(response, expected, **kwargs)
            results.append(result)
            if result.ground_truth == "TRUTHFUL":
                correct_count += 1
        return results, correct_count
