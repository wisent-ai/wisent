"""
SST2 Configuration and Baseline Performance Evaluation.

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


class SST2Config(BenchmarkConfig):
    """SST2 benchmark configuration."""

    @staticmethod
    def get_benchmark_name() -> str:
        """Return benchmark name."""
        return "sst2"

    @staticmethod
    def get_data_config() -> Dict[str, Any]:
        """
        Return data configuration for SST2.

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
        Return baseline evaluation configuration for SST2.

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
        Return optimization configuration for SST2.

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


class SST2Evaluator(BaselinePerformanceEvaluator):
    """SST2 baseline performance evaluator."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        num_questions: int = 150,
        max_new_tokens: int = 700,
        preferred_doc: str = "validation",
    ):
        super().__init__(
            benchmark_name="sst2",
            model_name=model_name,
            num_questions=num_questions,
            max_new_tokens=max_new_tokens,
            preferred_doc=preferred_doc,
        )

    def load_questions(self, limit: int, preferred_doc: str) -> List[Dict[str, Any]]:
        """Load SST2 questions from specified document source."""
        print(f"Loading {self.benchmark_name} task from {preferred_doc} docs...")

        task_dict = tasks.get_task_dict([self.benchmark_name])
        sst2_task = task_dict[self.benchmark_name]

        docs = atoms.LMEvalBenchmarkExtractor.load_docs(
            lm_eval_task_data=sst2_task,
            limit=limit,
            preferred_doc=preferred_doc
        )

        print(f"Successfully loaded {len(docs)} questions")

        # SST2 uses numerical labels: 0 = negative, 1 = positive
        label_map = {0: "negative", 1: "positive"}

        questions = []
        for doc in docs:
            sentence_text = doc.get("sentence", "")
            label_idx = doc.get("label")
            label = label_map.get(label_idx, "negative")

            questions.append({
                "sentence": sentence_text,
                "answer": label,
            })

        return questions

    def extract_answer(self, text: str) -> str | None:
        """Extract positive/negative sentiment from model response for SST2."""
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
                if answer in ["positive", "negative"]:
                    return answer
        except Exception:
            pass

        # Strategy 2: Look for common SST2 answer patterns
        answer_patterns = [
            r'(?:the\s+)?(?:final\s+)?(?:answer|sentiment)\s+is\s*:?\s*(positive|negative)',
            r'(?:therefore|thus|so),?\s+(?:the\s+)?(?:final\s+)?(?:answer|sentiment)\s+is\s*:?\s*(positive|negative)',
            r'(?:answer|sentiment)\s*:\s*(positive|negative)',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                return match.group(1)

        # Strategy 3: First occurrence (fallback)
        positive_match = re.search(r'\b(positive)\b', text_lower)
        negative_match = re.search(r'\b(negative)\b', text_lower)

        if positive_match and negative_match:
            return "positive" if positive_match.start() < negative_match.start() else "negative"
        elif positive_match:
            return "positive"
        elif negative_match:
            return "negative"

        return None

    def create_prompt(self, question_data: Dict[str, Any]) -> str:
        """Create the prompt for SST2."""
        sentence = question_data["sentence"]
        return (
            f"Sentence: {sentence}\n\n"
            f"Analyze the sentiment. After your reasoning, provide the final answer in this exact JSON format:\n"
            f"{{\n  \"final_answer\": \"<positive or negative>\"\n}}\n\nAnswer:"
        )

    def format_example(self, result: Dict[str, Any]) -> str:
        """Format an example result for display."""
        return (
            f"  Sentence: {result['question']['sentence'][:50]}...\n"
            f"  Correct answer: {result['correct_answer']}\n"
            f"  Model predicted: {result['predicted_answer']}"
        )


if __name__ == "__main__":
    evaluator = SST2Evaluator(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        num_questions=150,
        max_new_tokens=700,
    )

    results = evaluator.evaluate(
        output_path="results/sst2/sst2_baseline_results.json"
    )
