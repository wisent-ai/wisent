"""
Answer extractors for TaskInterface benchmarks.

These extractors parse model outputs to extract answers for validation.
Different from LMEvalBenchmarkExtractor which creates contrastive pairs.
"""

from abc import ABC, abstractmethod
from typing import Optional
import re

from wisent.core.errors import NumericalExtractionError, TextExtractionError, ExtractorNotFoundError


class BenchmarkExtractor(ABC):
    """Base class for benchmark answer extraction."""

    @abstractmethod
    def extract_answer(self, text: str) -> Optional[str]:
        """Extract answer from model's generated text."""
        pass

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        if answer is None:
            return ""
        return answer.lower().strip()

    def check_answer(self, predicted: str, expected: str) -> bool:
        """Check if predicted answer matches expected."""
        return self.normalize_answer(predicted) == self.normalize_answer(expected)

    def extract_qa_pair(
        self,
        sample: dict,
        task: any = None,
    ) -> Optional[dict]:
        """
        Extract a question-answer pair from a sample dictionary.
        
        This method handles common field names across different benchmarks.
        Subclasses can override for benchmark-specific extraction.
        
        Args:
            sample: A document/sample dictionary from a benchmark.
            task: Optional task object (may be needed for some extractors).
            
        Returns:
            A dict with keys "formatted_question" and "correct_answer",
            or None if extraction fails.
        """
        question = None
        answer = None
        
        # Try common question field names
        for q_field in ["question", "prompt", "text", "input", "problem", "query"]:
            if q_field in sample and sample[q_field]:
                question = str(sample[q_field]).strip()
                break
        
        # Try common answer field names
        for a_field in ["answer", "target", "label", "output", "solution", "expected"]:
            if a_field in sample and sample[a_field]:
                answer = str(sample[a_field]).strip()
                # For GSM8K style answers with "####", extract the numerical part
                if "####" in answer:
                    answer = answer.split("####")[-1].strip()
                break
        
        if not question or not answer:
            return None
            
        return {
            "formatted_question": f"Question: {question}",
            "correct_answer": answer,
        }

    def extract_contrastive_pair(
        self,
        sample: dict,
        task: any = None,
    ) -> Optional[dict]:
        """
        Extract a contrastive pair (question, correct, incorrect) from a sample.
        
        This is used for steering vector training. Returns question with both
        correct and incorrect answers.
        
        Args:
            sample: A document/sample dictionary from a benchmark.
            task: Optional task object (may be needed for some extractors).
            
        Returns:
            A dict with keys "question", "correct_answer", "incorrect_answer",
            or None if extraction fails.
        """
        qa_pair = self.extract_qa_pair(sample, task)
        if not qa_pair:
            return None
            
        # Generate a simple incorrect answer by modifying the correct one
        correct = qa_pair["correct_answer"]
        try:
            # For numerical answers, add 1
            num_val = float(correct.replace(',', ''))
            incorrect_val = num_val + 1
            incorrect = str(int(incorrect_val)) if incorrect_val == int(incorrect_val) else str(incorrect_val)
        except (ValueError, TypeError):
            # For text answers, just use a placeholder
            incorrect = "I don't know"
            
        return {
            "question": qa_pair["formatted_question"],
            "correct_answer": correct,
            "incorrect_answer": incorrect,
        }


class GSM8KExtractor(BenchmarkExtractor):
    """Extractor for GSM8K and math tasks."""

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract numerical answer from text."""
        if not text:
            return None

        # Strategy 1: JSON format {"final_answer": "123"}
        try:
            import json
            if "{" in text and "}" in text:
                # Find JSON-like structures
                json_match = re.search(r'\{[^}]*"final_answer"[^}]*\}', text)
                if json_match:
                    data = json.loads(json_match.group(0))
                    answer = data.get("final_answer")
                    if answer:
                        # Remove commas and non-numeric characters except decimal and minus
                        answer = re.sub(r'[^\d.\-]', '', str(answer))
                        if answer and answer.replace('.', '').replace('-', '').isdigit():
                            return answer
        except Exception:
            pass

        # Strategy 2: GSM8K format "#### 123"
        hash_match = re.search(r'####\s*([\d,.\-]+)', text)
        if hash_match:
            answer = hash_match.group(1).replace(',', '')
            return answer

        # Strategy 3: "The answer is 123" or "The final answer is 123"
        answer_patterns = [
            r'(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*([\d,.\-]+)',
            r'(?:therefore|thus|so),?\s+(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*([\d,.\-]+)',
            r'=\s*([\d,.\-]+)\s*$',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer = match.group(1).replace(',', '')
                return answer

        # No fallback - raise error if numerical answer cannot be extracted
        raise NumericalExtractionError(response=text)

    def check_answer(self, predicted: str, expected: str) -> bool:
        """Compare numerical answers with tolerance."""
        if predicted is None:
            return False
        try:
            pred_float = float(predicted)
            expected_float = float(expected)
            return abs(pred_float - expected_float) < 1e-6
        except (ValueError, TypeError):
            return self.normalize_answer(predicted) == self.normalize_answer(expected)


class LiveCodeBenchExtractor(BenchmarkExtractor):
    """Extractor for coding tasks."""

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract code from model response."""
        if not text:
            return None

        # Strategy 1: Extract from markdown code blocks
        code_block_patterns = [
            r'```python\s*(.*?)```',
            r'```\s*(.*?)```',
            r'`([^`]+)`',
        ]

        for pattern in code_block_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                # Return the longest code block found
                code = max(matches, key=len)
                return code.strip()

        # Strategy 2: Look for function definitions
        func_match = re.search(r'(def\s+\w+.*?)(?:\n\n|\Z)', text, re.DOTALL)
        if func_match:
            return func_match.group(1).strip()

        # Strategy 3: Look for class definitions
        class_match = re.search(r'(class\s+\w+.*?)(?:\n\n|\Z)', text, re.DOTALL)
        if class_match:
            return class_match.group(1).strip()

        # Strategy 4: Return entire text if it looks like code
        if 'def ' in text or 'class ' in text or 'import ' in text:
            return text.strip()

        return None

    def check_answer(self, predicted: str, expected: str) -> bool:
        """For coding tasks, basic check: code is not empty."""
        if predicted is None:
            return False
        # Basic check: code is not empty
        return len(predicted.strip()) > 0


class HLEExtractor(BenchmarkExtractor):
    """Extractor for HLE tasks."""

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract answer from HLE response."""
        if not text:
            return None

        # Strategy 1: JSON format {"answer": "X"}
        try:
            import json
            if "{" in text and "}" in text:
                json_match = re.search(r'\{[^}]*"answer"[^}]*\}', text)
                if json_match:
                    data = json.loads(json_match.group(0))
                    return str(data.get("answer", ""))
        except Exception:
            pass

        # Strategy 2: "Answer: X" format
        answer_match = re.search(r'answer\s*:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()

        # Strategy 3: Multiple choice (A, B, C, D)
        mc_match = re.search(r'\b([A-D])\b', text)
        if mc_match:
            return mc_match.group(1)

        # No fallback - raise error if answer cannot be extracted
        raise TextExtractionError(response=text)

    def check_answer(self, predicted: str, expected: str) -> bool:
        """Compare answers with case-insensitive substring matching."""
        if predicted is None:
            return False
        pred_norm = self.normalize_answer(predicted)
        exp_norm = self.normalize_answer(expected)
        return exp_norm in pred_norm or pred_norm in exp_norm


class SuperGPQAExtractor(BenchmarkExtractor):
    """Extractor for SuperGPQA science tasks."""

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract answer from SuperGPQA response."""
        if not text:
            return None

        # Strategy 1: "Answer: A" format
        answer_match = re.search(r'answer\s*:\s*([A-D])', text, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).upper()

        # Strategy 2: (A) or [A] format
        bracket_match = re.search(r'[\(\[]\s*([A-D])\s*[\)\]]', text, re.IGNORECASE)
        if bracket_match:
            return bracket_match.group(1).upper()

        # Strategy 3: Standalone letter
        letter_match = re.search(r'\b([A-D])\b', text)
        if letter_match:
            return letter_match.group(1).upper()

        # Strategy 4: First character if it's A-D
        first_char = text.strip()[0].upper() if text.strip() else None
        if first_char in ['A', 'B', 'C', 'D']:
            return first_char

        return None

    def check_answer(self, predicted: str, expected: str) -> bool:
        """Compare answers (case-insensitive letter comparison)."""
        if predicted is None:
            return False
        return self.normalize_answer(predicted) == self.normalize_answer(expected)


# Registry mapping task names to extractors
_EXTRACTOR_REGISTRY = {}


def _populate_registry():
    """Populate the extractor registry."""
    global _EXTRACTOR_REGISTRY

    # Math tasks use GSM8KExtractor
    math_tasks = [
        "gsm8k", "math", "math500", "hendrycks_math",
        "aime", "aime2024", "aime2025",
        "hmmt", "hmmt_feb_2025",
        "polymath", "polymath_en_medium", "polymath_zh_medium",
        "polymath_en_high", "polymath_zh_high",
        "livemathbench", "livemathbench_cnmo_en", "livemathbench_cnmo_zh",
    ]
    for task in math_tasks:
        _EXTRACTOR_REGISTRY[task] = GSM8KExtractor()

    # Coding tasks use LiveCodeBenchExtractor
    coding_tasks = [
        "livecodebench", "humaneval", "mbpp", "humaneval_plus", "mbpp_plus",
        "instructhumaneval", "apps", "ds1000",
        "multiple_py", "multiple_js", "multiple_java", "multiple_cpp", "multiple_rs", "multiple_go",
        "conala", "concode", "mercury", "recode",
        "codexglue_code_to_text_python", "codexglue_code_to_text_go",
        "codexglue_code_to_text_ruby", "codexglue_code_to_text_java",
        "codexglue_code_to_text_javascript", "codexglue_code_to_text_php",
    ]
    for task in coding_tasks:
        _EXTRACTOR_REGISTRY[task] = LiveCodeBenchExtractor()

    # HLE tasks use HLEExtractor
    hle_tasks = ["hle", "hle_exact_match", "hle_multiple_choice"]
    for task in hle_tasks:
        _EXTRACTOR_REGISTRY[task] = HLEExtractor()

    # Science tasks use SuperGPQAExtractor
    science_tasks = ["supergpqa", "supergpqa_physics", "supergpqa_chemistry", "supergpqa_biology"]
    for task in science_tasks:
        _EXTRACTOR_REGISTRY[task] = SuperGPQAExtractor()

    # QA tasks use generic extractor (HLEExtractor works well for these)
    qa_tasks = ["truthfulqa_mc1", "mmlu", "squad2"]
    for task in qa_tasks:
        _EXTRACTOR_REGISTRY[task] = HLEExtractor()


# Populate on module load
_populate_registry()


def get_extractor(task_name: str) -> BenchmarkExtractor:
    """Get the appropriate extractor for a task."""
    if task_name in _EXTRACTOR_REGISTRY:
        return _EXTRACTOR_REGISTRY[task_name]

    # Pattern-based extractor selection
    task_lower = task_name.lower()
    if any(keyword in task_lower for keyword in ["math", "aime", "hmmt", "gsm", "arithmetic"]):
        return GSM8KExtractor()
    elif any(keyword in task_lower for keyword in ["code", "human", "mbpp", "programming"]):
        return LiveCodeBenchExtractor()
    elif "hle" in task_lower:
        return HLEExtractor()
    elif any(keyword in task_lower for keyword in ["gpqa", "science", "physics", "chemistry", "biology"]):
        return SuperGPQAExtractor()
    
    # No fallback - raise error if no extractor found
    raise ExtractorNotFoundError(task_name=task_name)
