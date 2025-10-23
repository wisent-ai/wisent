"""
GSM8K Configuration and Baseline Performance Evaluation.

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


class GSM8KConfig(BenchmarkConfig):
    """GSM8K benchmark configuration."""

    @staticmethod
    def get_benchmark_name() -> str:
        """Return benchmark name."""
        return "gsm8k"

    @staticmethod
    def get_data_config() -> Dict[str, Any]:
        """
        Return data configuration for GSM8K.

        Returns:
            Dictionary with data split sources
        """
        return {
            "train_val_source": "training",
            "test_source": "test",
        }

    @staticmethod
    def get_baseline_config() -> Dict[str, Any]:
        """
        Return baseline evaluation configuration for GSM8K.

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
        Return optimization configuration for GSM8K.

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


class GSM8KEvaluator(BaselinePerformanceEvaluator):
    """GSM8K baseline performance evaluator."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        num_questions: int = 150,
        max_new_tokens: int = 700,
        preferred_doc: str = "test",
    ):
        super().__init__(
            benchmark_name="gsm8k",
            model_name=model_name,
            num_questions=num_questions,
            max_new_tokens=max_new_tokens,
            preferred_doc=preferred_doc,
        )

    def load_questions(self, limit: int, preferred_doc: str) -> List[Dict[str, Any]]:
        """Load GSM8K questions from specified document source."""
        print(f"Loading {self.benchmark_name} task from {preferred_doc} docs...")

        task_dict = tasks.get_task_dict([self.benchmark_name])
        gsm8k_task = task_dict[self.benchmark_name]

        docs = atoms.LMEvalBenchmarkExtractor.load_docs(
            lm_eval_task_data=gsm8k_task,
            limit=limit,
            preferred_doc=preferred_doc
        )

        print(f"Successfully loaded {len(docs)} questions")

        questions = []
        for doc in docs:
            question_text = doc.get("question", "")
            answer_text = doc.get("answer", "")

            # GSM8K answers are in format "explanation #### numerical_answer"
            if "####" in answer_text:
                numerical_answer = answer_text.split("####")[-1].strip()
            else:
                numerical_answer = answer_text.strip()

            questions.append({
                "question": question_text,
                "answer": numerical_answer,
            })

        return questions

    def extract_answer(self, text: str) -> str | None:
        """Extract numerical answer from model response for GSM8K."""
        if not text:
            return None

        # Strategy 1: Try to extract from JSON format
        try:
            json_match = re.search(r'\{[^}]*"final_answer"[^}]*:\s*([^,}\s]+)[^}]*\}', text, re.IGNORECASE)
            if json_match:
                answer = json_match.group(1).strip().strip('"').strip("'")
                answer = re.sub(r'[^\d.\-]', '', answer)
                if answer and answer.replace('.', '').replace('-', '').isdigit():
                    return answer
        except Exception:
            pass

        # Strategy 2: Look for GSM8K answer patterns
        # Pattern: "#### 123" (standard GSM8K format)
        hash_match = re.search(r'####\s*([\d,\.\-]+)', text)
        if hash_match:
            answer = hash_match.group(1).replace(',', '')
            return answer

        # Pattern: "The answer is 123" or "The final answer is 123"
        answer_patterns = [
            r'(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*([\d,\.\-]+)',
            r'(?:therefore|thus|so),?\s+(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*([\d,\.\-]+)',
            r'=\s*([\d,\.\-]+)\s*$',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer = match.group(1).replace(',', '')
                return answer

        # Strategy 3: Last number in the text (fallback)
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]

        return None

    def create_prompt(self, question_data: Dict[str, Any]) -> str:
        """Create the prompt for GSM8K."""
        question = question_data["question"]
        return (
            f"Question: {question}\n\n"
            f"Solve this step by step. After your reasoning, provide the final numerical answer in this exact JSON format:\n"
            f"{{\n  \"final_answer\": <your_number_here>\n}}\n\nAnswer:"
        )

    def format_example(self, result: Dict[str, Any]) -> str:
        """Format an example result for display."""
        return (
            f"  Q: {result['question']['question'][:50]}...\n"
            f"  Correct answer: {result['correct_answer']}\n"
            f"  Model predicted: {result['predicted_answer']}"
        )

    def check_answer_correctness(self, predicted: str, correct: str) -> bool:
        """Check if predicted answer matches correct answer for GSM8K."""
        if predicted is None:
            return False

        try:
            pred_float = float(predicted)
            correct_float = float(correct)
            # Allow small floating point differences
            return abs(pred_float - correct_float) < 1e-6
        except (ValueError, TypeError):
            # Fall back to string comparison
            return predicted.strip() == correct.strip()


if __name__ == "__main__":
    evaluator = GSM8KEvaluator(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        num_questions=150,
        max_new_tokens=700,
    )

    results = evaluator.evaluate(
        output_path="results/gsm8k/gsm8k_baseline_results.json"
    )
