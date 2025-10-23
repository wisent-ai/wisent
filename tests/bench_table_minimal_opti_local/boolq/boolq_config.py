"""
BoolQ Configuration and Baseline Performance Evaluation.

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


class BoolQConfig(BenchmarkConfig):
    """BoolQ benchmark configuration."""

    @staticmethod
    def get_benchmark_name() -> str:
        """Return benchmark name."""
        return "boolq"

    @staticmethod
    def get_data_config() -> Dict[str, Any]:
        """
        Return data configuration for BoolQ.

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
        Return baseline evaluation configuration for BoolQ.

        Returns:
            Dictionary with baseline parameters
        """
        return {
            "num_test": 150,
            "max_new_tokens": 700,
        }

    @staticmethod
    def get_optimization_config() -> Dict[str, Any]:
        """
        Return optimization configuration for BoolQ.

        Returns:
            Dictionary with optimization parameters
        """
        return {
            "num_train": 250,
            "num_val": 50,
            "num_test": 150,
            "n_trials": 40,
            "n_runs": 10,
        }


class BoolQEvaluator(BaselinePerformanceEvaluator):
    """BoolQ baseline performance evaluator."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        num_questions: int = 150,
        max_new_tokens: int = 700,
        preferred_doc: str = "validation",
    ):
        super().__init__(
            benchmark_name="boolq",
            model_name=model_name,
            num_questions=num_questions,
            max_new_tokens=max_new_tokens,
            preferred_doc=preferred_doc,
        )

    def load_questions(self, limit: int, preferred_doc: str) -> List[Dict[str, Any]]:
        """Load BoolQ questions from specified document source."""
        print(f"Loading {self.benchmark_name} task from {preferred_doc} docs...")

        task_dict = tasks.get_task_dict([self.benchmark_name])
        boolq_task = task_dict[self.benchmark_name]

        docs = atoms.LMEvalBenchmarkExtractor.load_docs(
            lm_eval_task_data=boolq_task,
            limit=limit,
            preferred_doc=preferred_doc
        )

        print(f"Successfully loaded {len(docs)} questions")

        # BoolQ uses numerical labels: 0 = no, 1 = yes
        label_map = {0: "no", 1: "yes"}

        questions = []
        for doc in docs:
            question_text = doc.get("question", "")
            answer_idx = doc.get("label")
            answer_text = label_map.get(answer_idx, "no")

            questions.append({
                "question": question_text,
                "answer": answer_text,
            })

        return questions

    def extract_answer(self, text: str) -> str | None:
        """Extract yes/no answer from model response for BoolQ."""
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
                answer = json_match.group(1).strip().strip('"').strip("'")
                if answer.startswith('<') and answer.endswith('>'):
                    pass
                elif answer in ["yes", "no"]:
                    return answer
        except Exception:
            pass

        # Strategy 2: Look for common BoolQ answer patterns
        answer_patterns = [
            r'(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*(yes|no)',
            r'(?:therefore|thus|so),?\s+(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*(yes|no)',
            r'answer\s*:\s*(yes|no)',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                return match.group(1)

        # Strategy 3: First occurrence of "yes" or "no" (fallback)
        yes_match = re.search(r'\b(yes)\b', text_lower)
        no_match = re.search(r'\b(no)\b', text_lower)

        if yes_match and no_match:
            return "yes" if yes_match.start() < no_match.start() else "no"
        elif yes_match:
            return "yes"
        elif no_match:
            return "no"

        return None

    def create_prompt(self, question_data: Dict[str, Any]) -> str:
        """Create the prompt for BoolQ."""
        question = question_data["question"]
        return (
            f"Question: {question}\n\n"
            f"Answer yes or no. After your reasoning, provide the final answer in this exact JSON format:\n"
            f"{{\n  \"final_answer\": \"<yes or no>\"\n}}\n\nAnswer:"
        )

    def format_example(self, result: Dict[str, Any]) -> str:
        """Format an example result for display."""
        return (
            f"  Q: {result['question']}...\n"
            f"  Correct answer: {result['correct_answer']}\n"
            f"  Model predicted: {result['predicted_answer']}"
        )


if __name__ == "__main__":
    evaluator = BoolQEvaluator(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        num_questions=150,
        max_new_tokens=700,
    )

    results = evaluator.evaluate(
        output_path="results/boolq/boolq_baseline_results.json"
    )
