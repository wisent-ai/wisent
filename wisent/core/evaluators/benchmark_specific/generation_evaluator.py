"""Generation-based evaluator for benchmarks that require text generation.

This evaluator handles tasks like GSM8K, DROP, TriviaQA where the model generates
free-form text that must be parsed and compared to reference answers.
"""

import re
from typing import Any, Dict
import logging

from wisent.core.evaluators.core.atoms import BaseEvaluator, EvalResult

logger = logging.getLogger(__name__)


class GenerationEvaluator(BaseEvaluator):
    """Evaluator for generation-based benchmarks.

    Handles tasks that require text generation followed by answer extraction:
    - GSM8K: Math word problems requiring numerical answers
    - DROP: Reading comprehension with discrete reasoning
    - TriviaQA: Trivia questions requiring factual answers

    Methods:
    - Exact match: Direct string comparison
    - Normalized match: Lenient comparison after normalization
    - Numerical match: For math problems
    """

    name = "generation"
    description = "Generation-based evaluator for text generation tasks"
    task_names = (
        # Math Tasks
        "gsm8k", "asdiv", "arithmetic", "math", "math_500", "math500",
        "aime", "aime2024", "aime2025",
        "hmmt", "hmmt_feb_2025",
        "polymath", "polymath_en_medium", "polymath_zh_medium", "polymath_en_high", "polymath_zh_high",
        "livemathbench", "livemathbench_cnmo_en", "livemathbench_cnmo_zh",
        "hendrycks_math",
        # Reading Comprehension & QA
        "drop", "triviaqa", "record", "squadv2", "squad2",
        "webqs", "nq_open", "coqa",
        # Code-to-Text (Code Summarization)
        "codexglue_code_to_text_python", "codexglue_code_to_text_go", "codexglue_code_to_text_ruby",
        "codexglue_code_to_text_java", "codexglue_code_to_text_javascript", "codexglue_code_to_text_php",
        # Code Translation
        "mercury",
        # Code Generation (BLEU/Exact Match evaluation)
        "conala", "concode", "recode",
        # New task families from lm-eval-harness (generate_until output_type)
        "kmmlu", "bbh", "flores", "afrimgsm", "mgsm", "ja", "phrases", "polemo2", "gsm",
        "anagrams1", "anagrams2", "babi", "cycle", "logieval", "random", "reversed",
        "flan", "xquad", "minerva", "scrolls", "code2text", "cabreu", "evalita-sp",
        "paloma", "ifeval", "gpt3", "iwslt2017", "iwslt2017-ar-en", "iwslt2017-en-ar",
        "translation", "wmt14", "wmt16", "wmt14-en-fr", "wmt14-fr-en", "wmt16-de-en",
        "wmt16-en-de", "wmt16-en-ro", "wmt16-ro-en", "wmt-ro-en-t5-prompt",
        "chain", "hendrycks", "self", "unscramble", "20", "ag", "argument", "atis",
        "banking77", "bec2016eu", "bhtc", "boolq-seq2seq", "catalanqa", "claim",
        "cnn", "cocoteros", "coedit", "cola", "commonsense", "coqcat", "dbpedia",
        "doc", "epec", "eq", "ethos", "fda", "financial", "groundcocoa", "histoires",
        "law", "ledgar", "medical", "medmcqa", "moral", "noticia", "parafraseja",
        "parafrases", "qnlieu", "realtoxicityprompts", "sglue", "squad", "stsb",
        "summarization", "swde", "teca", "tinyGSM8k", "toxigen", "unfair", "vaxx",
        "wiceu", "wsc273", "xlsum", "xsum", "yahoo", "t0", "super", "csatqa",
        "multiple", "afrixnli", "evalita-mp", "freebase", "llama", "math",
        "super-glue-lm-eval-v1", "super-glue-lm-eval-v1-seq2seq", "super-glue-t5-prompt",
        "tinyBenchmarks", "Tag", "pythia"
    )

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate generated response against expected answer.

        Args:
            response: Generated model response (can be empty if using choices)
            expected: Expected answer (str, int, float, or list of acceptable answers)
            **kwargs:
                task_name: Task name for task-specific logic
                answer_type: Type of answer ("numerical", "text", "exact")
                normalize: Whether to normalize strings before comparison
                choices: List of [correct_answer, incorrect_answer] for contrastive evaluation
                model: WisentModel instance for generating responses

        Returns:
            EvalResult with TRUTHFUL/UNTRUTHFUL/UNKNOWN
        """
        task_name = kwargs.get('task_name', '')
        answer_type = kwargs.get('answer_type', 'text')
        normalize = kwargs.get('normalize', True)
        choices = kwargs.get('choices')
        model = kwargs.get('model')
        question = kwargs.get('question', '')

        # If we have choices, do direct comparison (for contrastive evaluation)
        if choices and len(choices) == 2:
            correct_answer = choices[0]
            incorrect_answer = choices[1]

            # Determine answer type based on task or if it looks numerical
            if task_name in ["gsm8k", "asdiv", "arithmetic", "math", "math_500", "math500", "aime", "aime2024", "aime2025", "hmmt", "hmmt_feb_2025", "polymath", "polymath_en_medium", "polymath_zh_medium", "polymath_en_high", "polymath_zh_high", "livemathbench", "livemathbench_cnmo_en", "livemathbench_cnmo_zh", "hendrycks_math"]:
                answer_type = "numerical"

            # For generation tasks, we compare the pre-generated answers directly
            # Expected is what we're comparing against
            # Extract answers from both choices
            extracted_correct = self._extract_answer(correct_answer, task_name, answer_type)
            extracted_incorrect = self._extract_answer(incorrect_answer, task_name, answer_type)
            extracted_expected = self._extract_answer(str(expected), task_name, answer_type)

            if extracted_expected is None:
                return EvalResult(
                    ground_truth="UNKNOWN",
                    method_used=self.name,
                    confidence=0.0,
                    details="Could not extract answer from expected response",
                )

            # Check which choice matches the expected answer
            if isinstance(expected, list):
                expected_answers = expected
            else:
                expected_answers = [expected]

            # For numerical answers, we've already extracted, so compare directly
            if answer_type == "numerical":
                # Compare extracted numbers directly
                correct_matches = (extracted_correct is not None and
                                 extracted_expected is not None and
                                 abs(extracted_correct - extracted_expected) < 1e-6)
                incorrect_matches = (extracted_incorrect is not None and
                                   extracted_expected is not None and
                                   abs(extracted_incorrect - extracted_expected) < 1e-6)
                conf_correct = 1.0 if correct_matches else 0.0
                conf_incorrect = 1.0 if incorrect_matches else 0.0
            else:
                # For text answers, check match against the list
                correct_matches, _, conf_correct = self._check_match(
                    extracted_correct, expected_answers, answer_type, normalize
                )
                incorrect_matches, _, conf_incorrect = self._check_match(
                    extracted_incorrect, expected_answers, answer_type, normalize
                )

            # If correct matches expected, return TRUTHFUL
            # If incorrect matches expected, return UNTRUTHFUL
            if correct_matches and not incorrect_matches:
                return EvalResult(
                    ground_truth="TRUTHFUL",
                    method_used=self.name,
                    confidence=conf_correct,
                    details=f"Correct answer '{correct_answer}' matches expected '{expected}'",
                    meta={
                        "correct_answer": correct_answer,
                        "incorrect_answer": incorrect_answer,
                        "expected": expected,
                    }
                )
            elif incorrect_matches and not correct_matches:
                return EvalResult(
                    ground_truth="UNTRUTHFUL",
                    method_used=self.name,
                    confidence=conf_incorrect,
                    details=f"Incorrect answer '{incorrect_answer}' matches expected '{expected}'",
                    meta={
                        "correct_answer": correct_answer,
                        "incorrect_answer": incorrect_answer,
                        "expected": expected,
                    }
                )
            else:
                # Both match or neither matches - unclear
                return EvalResult(
                    ground_truth="UNKNOWN",
                    method_used=self.name,
                    confidence=0.0,
                    details=f"Ambiguous: correct={correct_matches}, incorrect={incorrect_matches}",
                    meta={
                        "correct_answer": correct_answer,
                        "incorrect_answer": incorrect_answer,
                        "expected": expected,
                    }
                )

        # If no choices, extract answer from generated response
        extracted_answer = self._extract_answer(response, task_name, answer_type)

        if extracted_answer is None:
            return EvalResult(
                ground_truth="UNKNOWN",
                method_used=self.name,
                confidence=0.0,
                details="Could not extract answer from generated response",
            )

        # Normalize expected answer
        if isinstance(expected, list):
            expected_answers = expected
        else:
            expected_answers = [expected]

        # Check if extracted answer matches any expected answer
        is_correct, matched_answer, confidence = self._check_match(
            extracted_answer, expected_answers, answer_type, normalize
        )

        return EvalResult(
            ground_truth="TRUTHFUL" if is_correct else "UNTRUTHFUL",
            method_used=self.name,
            confidence=confidence,
            details=f"Extracted: '{extracted_answer}', Expected: '{matched_answer or expected_answers[0]}'",
            meta={
                "extracted_answer": extracted_answer,
                "expected_answers": expected_answers,
                "matched_answer": matched_answer,
                "answer_type": answer_type,
            }
        )

    def _extract_answer(self, response: str, task_name: str, answer_type: str) -> Any:
        """Extract answer from generated response."""
        if answer_type == "numerical" or task_name == "gsm8k":
            return self._extract_numerical_answer(response)
        else:
            return self._extract_text_answer(response)

    def _extract_numerical_answer(self, response: str) -> float:
        """Extract numerical answer from response (for math problems)."""
        # Handle LaTeX fractions: \frac{a}{b}
        frac_pattern = r'\\frac\{([-+]?\d+\.?\d*)\}\{([-+]?\d+\.?\d*)\}'
        frac_match = re.search(frac_pattern, response)
        if frac_match:
            try:
                numerator = float(frac_match.group(1))
                denominator = float(frac_match.group(2))
                if denominator != 0:
                    return numerator / denominator
            except (ValueError, ZeroDivisionError):
                pass

        # Handle LaTeX degree notation: number^{\circ} or number^\circ
        degree_pattern = r'([-+]?\d+\.?\d*)\^\{?\\circ\}?'
        degree_match = re.search(degree_pattern, response)
        if degree_match:
            try:
                return float(degree_match.group(1))
            except ValueError:
                pass

        # Look for common patterns (specific markers only)
        patterns = [
            r'####\s*([-+]?\d*\.?\d+)',  # GSM8K format: #### 42
            r'answer\s*is\s*:?\s*([-+]?\d*\.?\d+)',  # "answer is 42" or "answer is: 42"
            r'=\s*([-+]?\d*\.?\d+)\s*$',  # Ends with "= 42"
            r'^\s*([-+]?\d*\.?\d+)\s*$',  # Just a number by itself (entire response)
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        # Fallback: find last number in response
        # This handles responses without explicit markers
        numbers = re.findall(r'[-+]?\d*\.?\d+', response)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass

        return None

    def _extract_text_answer(self, response: str) -> str:
        """Extract text answer from response."""
        # Look for explicit answer markers
        patterns = [
            r'answer\s*is:?\s*(.+?)(?:\n|$)',
            r'final\s+answer:?\s*(.+?)(?:\n|$)',
            r'(?:^|\n)answer:?\s*(.+?)(?:\n|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fallback: use first sentence
        sentences = re.split(r'[.!?]\s+', response)
        if sentences:
            return sentences[0].strip()

        return response.strip()

    def _check_match(
        self, extracted: Any, expected_list: list, answer_type: str, normalize: bool
    ) -> tuple:
        """Check if extracted answer matches any expected answer.

        Returns:
            (is_correct, matched_answer, confidence)
        """
        if answer_type == "numerical":
            return self._check_numerical_match(extracted, expected_list)
        else:
            return self._check_text_match(extracted, expected_list, normalize)

    def _check_numerical_match(self, extracted: float, expected_list: list) -> tuple:
        """Check numerical match with tolerance."""
        if extracted is None:
            return False, None, 0.0

        for expected in expected_list:
            try:
                expected_num = float(expected)
                # Check if close enough (tolerance for floating point)
                if abs(extracted - expected_num) < 1e-6:
                    return True, expected, 1.0
            except (ValueError, TypeError):
                continue

        return False, None, 0.0

    def _check_text_match(self, extracted: str, expected_list: list, normalize: bool) -> tuple:
        """Check text match with optional normalization."""
        if extracted is None:
            return False, None, 0.0

        if normalize:
            extracted_norm = self.normalize_text(extracted)
        else:
            extracted_norm = extracted

        for expected in expected_list:
            expected_str = str(expected)
            if normalize:
                expected_norm = self.normalize_text(expected_str)
            else:
                expected_norm = expected_str

            # Exact match
            if extracted_norm == expected_norm:
                return True, expected, 1.0

            # Substring match
            if extracted_norm in expected_norm or expected_norm in extracted_norm:
                return True, expected, 0.8

        return False, None, 0.0
