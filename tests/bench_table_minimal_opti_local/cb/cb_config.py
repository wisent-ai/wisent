"""
CB Configuration and Baseline Performance Evaluation.

Provides:
- Baseline performance evaluator
- Benchmark configuration (data splits, etc.)
"""

from __future__ import annotations

import sys
import os
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import re
from lm_eval import tasks
from typing import List, Dict, Any

from wisent.core.contrastive_pairs.lm_eval_pairs import atoms
from tests.bench_table_minimal_opti_local.utils.baseline_performance import BaselinePerformanceEvaluator
from tests.bench_table_minimal_opti_local.utils.config import BenchmarkConfig


class CBConfig(BenchmarkConfig):
    """CB benchmark configuration."""

    @staticmethod
    def get_benchmark_name() -> str:
        """Return benchmark name."""
        return "cb"

    @staticmethod
    def get_data_config() -> Dict[str, Any]:
        """
        Return data configuration for CB.

        Returns:
            Dictionary with data split sources
        """
        return {
            "train_val_source": "training",
            "test_source": "validation",
        }

    @staticmethod
    def get_baseline_config() -> Dict[str, Any]:
        """
        Return baseline evaluation configuration for CB.

        Returns:
            Dictionary with baseline parameters
        """
        return {
            "num_test": 56,
            "max_new_tokens": 700,
        }

    @staticmethod
    def get_optimization_config() -> Dict[str, Any]:
        """
        Return optimization configuration for CB.

        Returns:
            Dictionary with optimization parameters
        """
        return {
            "num_train": 200,
            "num_val": 50,
            "num_test": 56,
            "n_trials": 40,
            "n_runs": 10,
        }


class CBEvaluator(BaselinePerformanceEvaluator):
    """CB baseline performance evaluator."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        num_questions: int = 56,
        max_new_tokens: int = 700,
        preferred_doc: str = "validation",
    ):
        super().__init__(
            benchmark_name="cb",
            model_name=model_name,
            num_questions=num_questions,
            max_new_tokens=max_new_tokens,
            preferred_doc=preferred_doc,
        )

    def load_questions(self, limit: int, preferred_doc: str) -> List[Dict[str, Any]]:
        """Load CB questions from specified document source."""
        print(f"Loading {self.benchmark_name} task from {preferred_doc} docs...")

        task_dict = tasks.get_task_dict([self.benchmark_name])
        cb_task = task_dict[self.benchmark_name]

        docs = atoms.LMEvalBenchmarkExtractor.load_docs(
            lm_eval_task_data=cb_task,
            limit=limit,
            preferred_doc=preferred_doc
        )

        print(f"Successfully loaded {len(docs)} questions")

        # CB uses numerical labels: 0 = True, 1 = False, 2 = Neither
        label_map = {0: "True", 1: "False", 2: "Neither"}

        questions = []
        for doc in docs:
            premise = doc.get("premise", "")
            hypothesis = doc.get("hypothesis", "")
            label_idx = doc.get("label")
            label = label_map.get(label_idx, "Neither")

            questions.append({
                "premise": premise,
                "hypothesis": hypothesis,
                "answer": label,
            })

        return questions

    def extract_answer(self, text: str) -> str | None:
        """Extract True/False/Neither answer from model response for CB."""
        if not text:
            return None

        text_lower = text.lower()

        # Strategy 1: Try to extract from JSON format
        try:
            matches = list(re.finditer(r'"final_answer"\s*:\s*"([^"]+)"', text_lower, re.IGNORECASE))
            if not matches:
                matches = list(re.finditer(r'"final_answer"\s*:\s*([^,}\s]+)', text_lower, re.IGNORECASE))

            if matches:
                json_match = matches[-1]
                answer = json_match.group(1).strip().strip('"').strip("'").lower()
                if answer == "true":
                    return "True"
                elif answer == "false":
                    return "False"
                elif answer == "neither":
                    return "Neither"
        except Exception:
            pass

        # Strategy 2: Look for common CB answer patterns
        answer_patterns = [
            r'(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*(true|false|neither)',
            r'(?:therefore|thus|so),?\s+(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*(true|false|neither)',
            r'answer\s*:\s*(true|false|neither)',
            r'(?:the\s+)?(?:relationship|label)\s+is\s*:?\s*(true|false|neither)',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                answer = match.group(1).lower()
                return answer.capitalize()

        # Strategy 3: First occurrence (fallback)
        true_match = re.search(r'\b(true)\b', text_lower)
        false_match = re.search(r'\b(false)\b', text_lower)
        neither_match = re.search(r'\b(neither)\b', text_lower)

        matches = []
        if true_match:
            matches.append((true_match.start(), "True"))
        if false_match:
            matches.append((false_match.start(), "False"))
        if neither_match:
            matches.append((neither_match.start(), "Neither"))

        if matches:
            return min(matches, key=lambda x: x[0])[1]

        return None

    def create_prompt(self, question_data: Dict[str, Any]) -> str:
        """Create the prompt for CB."""
        premise = question_data["premise"]
        hypothesis = question_data["hypothesis"]
        return (
            f"{premise}\n"
            f"Question: {hypothesis}. True, False, or Neither?\n\n"
            f"After your reasoning, provide the final answer in this exact JSON format:\n"
            f"{{\n  \"final_answer\": \"<True, False, or Neither>\"\n}}\n\nAnswer:"
        )

    def format_example(self, result: Dict[str, Any]) -> str:
        """Format an example result for display."""
        return (
            f"  Premise: {result['question']['premise'][:50]}...\n"
            f"  Hypothesis: {result['question']['hypothesis'][:50]}...\n"
            f"  Correct answer: {result['correct_answer']}\n"
            f"  Model predicted: {result['predicted_answer']}"
        )


if __name__ == "__main__":
    evaluator = CBEvaluator(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        num_questions=56,
        max_new_tokens=700,
    )

    results = evaluator.evaluate(
        output_path="results/cb/cb_baseline_results.json"
    )
