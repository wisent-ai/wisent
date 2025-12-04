"""PolyMath evaluator for multilingual math benchmarks.

This evaluator handles answer comparison for PolyMath (Qwen/PolyMath) benchmarks
where answers may be in various formats and need mathematical equivalence checking.

Uses math_equal from math_parsing for robust comparison.
"""

import logging
from typing import Any

from wisent.core.evaluators.benchmark_specific.math_parsing.scripts import multi_math_equal
from wisent.core.evaluators.core.atoms import BaseEvaluator, EvalResult

logger = logging.getLogger(__name__)


LANGUAGE_PROMPTS = {
    "en": "Note: Please put the final answer in the $\\boxed{}$.",
    "zh": "注意：请将最终答案放在 $\\boxed{}$ 中。",
    "ar": "ملاحظة: يُرجى وضع الإجابة النهائية في $\\boxed{}$.",
    "bn": "বিঃদ্রিঃ: অনুগ্রহ করে চূড়ান্ত উত্তরটি $\\boxed{}$ এর মধ্যে রাখুন।",
    "de": "Hinweis: Bitte setzen Sie die endgültige Antwort in $\\boxed{}$.",
    "es": "Nota: Por favor, coloque la respuesta final en el $\\boxed{}$.",
    "fr": "Remarque : Veuillez mettre la réponse finale dans le $\\boxed{}$.",
    "id": "Catatan: Silakan letakkan jawaban akhir di dalam $\\boxed{}$.",
    "it": "Nota: Per favore, metti la risposta finale nel $\\boxed{}$.",
    "ja": "注意：最終的な答えを $\\boxed{}$ に入れてください。",
    "ko": "참고: 최종 답안을 $\\boxed{}$ 안에 넣어 주세요.",
    "ms": "Nota: Sila letakkan jawapan akhir dalam $\\boxed{}$.",
    "pt": "Nota: Por favor, coloque a resposta final no $\\boxed{}$.",
    "ru": "Примечание: Пожалуйста, поместите окончательный ответ в $\\boxed{}$.",
    "sw": "Kumbuka: Tafadhali weka jibu la mwisho katika $\\boxed{}$.",
    "te": "గమనిక: దయచేసి తుది జవాబును $\\boxed{}$ లో ఉంచండి.",
    "th": "หมายเหตุ: กรุณาใส่คำตอบสุดท้ายใน $\\boxed{}$.",
    "vi": "Lưu ý: Vui lòng đặt câu trả lời cuối cùng trong $\\boxed{}$.",
}


class PolyMathEvaluator(BaseEvaluator):
    """Evaluator for PolyMath multilingual math benchmark.

    Designed for PolyMath benchmarks where:
    - Answers can be numbers, fractions, or LaTeX expressions
    - Model outputs may contain answers in \\boxed{} format
    - Answers need mathematical equivalence checking

    Uses math_equal from math_parsing for robust comparison.
    """

    name = "polymath"
    description = "PolyMath evaluator for multilingual math answers"

    @staticmethod
    def get_prompt(question: str, language: str = "en") -> str:
        """Create prompt by appending language-specific instruction to the question.

        Args:
            question: The math question from the dataset
            language: Language code (en, zh, ar, bn, de, es, fr, id, it, ja, ko, ms, pt, ru, sw, te, th, vi)

        Returns:
            Question with language-specific instruction appended
        """
        instruction = LANGUAGE_PROMPTS.get(language, LANGUAGE_PROMPTS["en"])
        return f"{question} {instruction}"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate model response against expected PolyMath answer.

        Args:
            response: Model-generated response (may contain \\boxed{answer})
            expected: Expected answer string

        Returns:
            EvalResult with TRUTHFUL/UNTRUTHFUL/UNKNOWN
        """
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

        # Convert expected to string
        expected_str = str(expected).strip()

        # Use multi_math_equal which handles:
        # - Answer extraction (boxed, multilingual patterns, last number)
        # - Multiple comparison methods (numeric, symbolic, etc.)
        is_correct, predictions, no_boxed = multi_math_equal(expected_str, response)

        # Extract the model answer from predictions for logging
        # predictions is ((prediction1, result1), (prediction2, result2))
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
