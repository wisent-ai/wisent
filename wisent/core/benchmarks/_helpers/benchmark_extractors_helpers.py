"""SuperGPQA extractor and registry functions for benchmark extractors.

Extracted from benchmark_extractors.py to keep file under 300 lines.
"""

import re
from typing import Optional

from wisent.core.errors import ExtractorNotFoundError


class SuperGPQAExtractor:
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

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        if answer is None:
            return ""
        return answer.lower().strip()

    def check_answer(self, predicted: str, expected: str) -> bool:
        """Compare answers (case-insensitive letter comparison)."""
        if predicted is None:
            return False
        return self.normalize_answer(predicted) == self.normalize_answer(expected)


def _populate_registry(registry: dict):
    """Populate the extractor registry."""
    from wisent.core.benchmarks.benchmark_extractors import (
        GSM8KExtractor, LiveCodeBenchExtractor, HLEExtractor,
    )

    # Math tasks
    math_tasks = [
        "gsm8k", "math", "math500", "hendrycks_math",
        "aime", "aime2024", "aime2025",
        "hmmt", "hmmt_feb_2025",
        "polymath", "polymath_en_medium", "polymath_zh_medium",
        "polymath_en_high", "polymath_zh_high",
        "livemathbench", "livemathbench_cnmo_en", "livemathbench_cnmo_zh",
    ]
    for task in math_tasks:
        registry[task] = GSM8KExtractor()

    # Coding tasks
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
        registry[task] = LiveCodeBenchExtractor()

    # HLE tasks
    hle_tasks = ["hle", "hle_exact_match", "hle_multiple_choice"]
    for task in hle_tasks:
        registry[task] = HLEExtractor()

    # Science tasks
    science_tasks = ["supergpqa", "supergpqa_physics", "supergpqa_chemistry", "supergpqa_biology"]
    for task in science_tasks:
        registry[task] = SuperGPQAExtractor()

    # QA tasks
    qa_tasks = ["truthfulqa_mc1", "mmlu", "squad2"]
    for task in qa_tasks:
        registry[task] = HLEExtractor()


def get_extractor(task_name: str, registry: dict):
    """Get the appropriate extractor for a task."""
    from wisent.core.benchmarks.benchmark_extractors import (
        GSM8KExtractor, LiveCodeBenchExtractor, HLEExtractor,
    )

    if task_name in registry:
        return registry[task_name]

    task_lower = task_name.lower()
    if any(keyword in task_lower for keyword in ["math", "aime", "hmmt", "gsm", "arithmetic"]):
        return GSM8KExtractor()
    elif any(keyword in task_lower for keyword in ["code", "human", "mbpp", "programming"]):
        return LiveCodeBenchExtractor()
    elif "hle" in task_lower:
        return HLEExtractor()
    elif any(keyword in task_lower for keyword in ["gpqa", "science", "physics", "chemistry", "biology"]):
        return SuperGPQAExtractor()

    raise ExtractorNotFoundError(task_name=task_name)
