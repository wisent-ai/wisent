"""
Baseline Performance Evaluation Utilities.

Common code for evaluating model baseline performance on benchmarks.
Based on bench_table_rewrite/utils/baseline_performance.py
"""

from __future__ import annotations

import sys
import os
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import json

from wisent.core.models.wisent_model import WisentModel


class BaselinePerformanceEvaluator(ABC):
    """Abstract base class for baseline performance evaluation on benchmarks.

    Each benchmark should inherit from this class and implement:
    - load_questions(): Load and parse questions from the benchmark
    - extract_answer(): Extract the answer from model's response
    - create_prompt(): Format the prompt for the model
    - compare_answers(): Compare predicted vs correct answers (optional override)
    - format_example(): Format example output for display (optional override)
    """

    def __init__(
        self,
        benchmark_name: str,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        num_questions: int = 150,
        max_new_tokens: int = 700,
        preferred_doc: str = "validation",
    ):
        """Initialize the evaluator.

        Args:
            benchmark_name: Name of the benchmark (e.g., "boolq", "cb")
            model_name: HuggingFace model name or path
            num_questions: Number of questions to evaluate
            max_new_tokens: Maximum tokens to generate
            preferred_doc: Document source ("validation", "test", "training", "fewshot")
        """
        self.benchmark_name = benchmark_name
        self.model_name = model_name
        self.num_questions = num_questions
        self.max_new_tokens = max_new_tokens
        self.preferred_doc = preferred_doc

    @abstractmethod
    def load_questions(self, limit: int, preferred_doc: str) -> List[Dict[str, Any]]:
        """Load questions from the benchmark.

        Args:
            limit: Number of questions to load
            preferred_doc: Document source to use

        Returns:
            List of question dictionaries. Each dict should have an 'answer' key
            with the correct answer.
        """
        pass

    @abstractmethod
    def extract_answer(self, text: str) -> str | None:
        """Extract the answer from model's response.

        Args:
            text: Model's response text

        Returns:
            Extracted answer or None if not found
        """
        pass

    @abstractmethod
    def create_prompt(self, question_data: Dict[str, Any]) -> str:
        """Create the prompt for the model.

        Args:
            question_data: Dictionary containing question fields

        Returns:
            Formatted prompt string
        """
        pass

    def compare_answers(self, predicted: str | None, correct: str) -> bool:
        """Compare predicted answer with correct answer.

        Default implementation: case-insensitive string comparison.
        Override for custom comparison logic (e.g., numerical comparison).

        Args:
            predicted: Predicted answer from model
            correct: Correct answer

        Returns:
            True if answers match, False otherwise
        """
        if predicted is None:
            return False
        try:
            return predicted.strip().lower() == correct.strip().lower()
        except (ValueError, TypeError, AttributeError):
            return False

    def format_example(self, result: Dict[str, Any]) -> str:
        """Format an example result for display.

        Default implementation: shows first field (after question_id) and answers.
        Override for custom formatting.

        Args:
            result: Result dictionary

        Returns:
            Formatted string for display
        """
        # Find the first field that's not question_id, answer, or internal fields
        question_field = None
        for key in result.keys():
            if key not in ['question_id', 'answer', 'correct_answer', 'predicted_answer',
                          'model_response', 'is_correct', 'error']:
                question_field = key
                break

        output = []
        if question_field:
            output.append(f"  {question_field.capitalize()}: {result[question_field]}...")
        output.append(f"  Correct answer: {result['correct_answer']}")
        output.append(f"  Model predicted: {result['predicted_answer']}")
        return "\n".join(output)

    def evaluate(self, output_path: str | None = None) -> Dict:
        """Run the evaluation.

        Args:
            output_path: Optional path to save results

        Returns:
            Dict containing evaluation results and statistics
        """
        print(f"Evaluating baseline performance for {self.model_name}")
        print(f"Benchmark: {self.benchmark_name.upper()}")
        print(f"Questions: {self.num_questions}")
        print(f"Generation params: max_new_tokens={self.max_new_tokens}, do_sample=False")
        print("=" * 80)

        questions = self.load_questions(
            limit=self.num_questions,
            preferred_doc=self.preferred_doc
        )

        if len(questions) < self.num_questions:
            print(f"Warning: Only loaded {len(questions)} questions, expected {self.num_questions}")
            self.num_questions = len(questions)

        print(f"\nLoading model: {self.model_name}")
        model = WisentModel(model_name=self.model_name)
        print(f"Model loaded. Hidden size: {model.hidden_size}, Layers: {model.num_layers}")

        results: List[Dict] = []
        correct_count = 0

        print(f"\nEvaluating {len(questions)} questions...")
        print("=" * 80)

        for idx, question_data in enumerate(questions):
            correct_answer = question_data["answer"]

            prompt = self.create_prompt(question_data)
            messages = [[{"role": "user", "content": prompt}]]

            try:
                # Generate with do_sample=False for deterministic output
                responses = model.generate(
                    inputs=messages,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_steering=False,
                )
                response = responses[0]

                predicted_answer = self.extract_answer(response)
                is_correct = self.compare_answers(predicted_answer, correct_answer)

                if is_correct:
                    correct_count += 1

                result = {
                    "question_id": idx,
                    **question_data,
                    "correct_answer": correct_answer,
                    "model_response": response,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct,
                }
                results.append(result)

                if (idx + 1) % 10 == 0 or idx == 0:
                    current_accuracy = (correct_count / (idx + 1)) * 100
                    status = "✓" if is_correct else "✗"
                    print(f"{status} Question {idx + 1}/{len(questions)}: {correct_count}/{idx + 1} correct ({current_accuracy:.1f}%)")

            except Exception as e:
                print(f"✗ Error processing question {idx}: {e}")
                result = {
                    "question_id": idx,
                    **question_data,
                    "correct_answer": correct_answer,
                    "model_response": None,
                    "predicted_answer": None,
                    "is_correct": False,
                    "error": str(e),
                }
                results.append(result)

        total_questions = len(results)
        accuracy = (correct_count / total_questions * 100) if total_questions > 0 else 0.0

        summary = {
            "benchmark": self.benchmark_name,
            "model_name": self.model_name,
            "total_questions": total_questions,
            "correct_answers": correct_count,
            "incorrect_answers": total_questions - correct_count,
            "accuracy": accuracy / 100.0,  # As fraction
            "accuracy_percent": accuracy,
        }

        evaluation_result = {
            "summary": summary,
            "results": results,
        }

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            print(f"\nSaving results to {output_path}...")
            with open(output_path, "w") as f:
                json.dump(evaluation_result, f, indent=2)
            print("Results saved successfully!")

        self._print_summary(summary, results)

        del model

        return evaluation_result

    def _print_summary(self, summary: Dict, results: List[Dict]):
        """Print evaluation summary and examples."""
        print("\n" + "=" * 80)
        print("BASELINE PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"Benchmark: {summary['benchmark'].upper()}")
        print(f"Model: {summary['model_name']}")
        print(f"Total questions: {summary['total_questions']}")
        print(f"Correct answers: {summary['correct_answers']}")
        print(f"Incorrect answers: {summary['incorrect_answers']}")
        print(f"Accuracy: {summary['accuracy_percent']:.2f}%")
        print("=" * 80)

        print("\nSample Results:")
        print("-" * 80)
        correct_examples = [r for r in results if r["is_correct"]][:2]
        incorrect_examples = [r for r in results if not r["is_correct"]][:2]

        if correct_examples:
            print("\n✓ CORRECT EXAMPLES:")
            for i, result in enumerate(correct_examples):
                print(f"\nExample {i+1}:")
                print(self.format_example(result))

        if incorrect_examples:
            print("\n✗ INCORRECT EXAMPLES:")
            for i, result in enumerate(incorrect_examples):
                print(f"\nExample {i+1}:")
                print(self.format_example(result))
