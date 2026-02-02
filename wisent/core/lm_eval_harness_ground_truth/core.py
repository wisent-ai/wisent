"""LM-Eval-Harness Ground Truth Evaluation - Core module."""

import logging
import re
import json
from typing import Any, Dict

from wisent.core.activations import ExtractionStrategy

logger = logging.getLogger(__name__)


class LMEvalHarnessGroundTruth:
    """Ground truth evaluator using lm-eval-harness tasks."""

    def __init__(self, task_name: str, evaluation_method: str = None, model=None):
        self.task_name = task_name
        self.evaluation_method = evaluation_method
        self.model = model
        if not self.evaluation_method:
            self.evaluation_method = self._get_evaluation_method_for_task(task_name)

    def evaluate_classifier_on_task(self, classifier, task_name: str, num_samples: int = 100,
                                     model=None, layer: int = 15, token_aggregation: str = "average") -> Dict[str, Any]:
        """Evaluate a classifier on the specified lm-eval task."""
        evaluation_model = model or self.model
        if self.evaluation_method == "log-likelihoods":
            return self._evaluate_log_likelihoods(classifier, task_name, num_samples, evaluation_model, layer, token_aggregation)
        if self.evaluation_method == "text-generation":
            from .text_generation import evaluate_text_generation
            return evaluate_text_generation(self, classifier, task_name, num_samples, evaluation_model, layer, token_aggregation)
        if self.evaluation_method == "perplexity":
            from .perplexity import evaluate_perplexity
            return evaluate_perplexity(self, classifier, task_name, num_samples, evaluation_model, layer, token_aggregation)
        if self.evaluation_method == "code-execution":
            from .code_execution import evaluate_code_execution
            return evaluate_code_execution(self, classifier, task_name, num_samples, evaluation_model, layer, token_aggregation)
        return {"ground_truth": "UNKNOWN", "method_used": "lm-eval-harness-unsupported", "confidence": 0.0,
                "details": f"Unsupported evaluation method: {self.evaluation_method}",
                "task_name": task_name, "evaluation_method": self.evaluation_method}

    def _evaluate_log_likelihoods(self, classifier, task_name: str, num_samples: int, model, layer: int,
                                   token_aggregation: str = "average") -> Dict[str, Any]:
        """Evaluate classifier using log-likelihoods approach."""
        try:
            from ..log_likelihoods_evaluator import LogLikelihoodsEvaluator
            evaluator = LogLikelihoodsEvaluator(task_name, model=model)
            results = evaluator.evaluate_classifier_on_task(classifier, task_name, num_samples=num_samples,
                                                            model=model, layer=layer, token_aggregation=token_aggregation)
            print(results)
            return results
        except Exception as e:
            logger.error(f"Error in log-likelihoods evaluation: {e}")
            return {"ground_truth": "UNKNOWN", "method_used": "lm-eval-harness-error", "confidence": 0.0,
                    "details": f"Log-likelihoods evaluation failed: {e!s}", "task_name": task_name,
                    "evaluation_method": "log-likelihoods"}

    def _get_evaluation_method_for_task(self, task_name: str) -> str:
        """Get the evaluation method for a task from the benchmark configuration."""
        try:
            eval_methods_path = "wisent/parameters/benchmarks/benchmark_evaluation_methods.json"
            with open(eval_methods_path) as f:
                benchmark_methods = json.load(f)
                return benchmark_methods.get(task_name, "text-generation")
        except Exception as e:
            logger.debug(f"Could not load benchmark evaluation methods: {e}")
            return "text-generation"

    def _error_result(self, error_message: str) -> Dict[str, Any]:
        """Return a standardized error result."""
        return {"ground_truth": "ERROR", "method_used": "lm-eval-harness-error", "confidence": 0.0,
                "details": error_message, "task_name": self.task_name, "evaluation_method": self.evaluation_method}

    def _map_token_aggregation_to_activation_method(self, token_aggregation: str):
        """Map token aggregation string to ExtractionStrategy."""
        try:
            return ExtractionStrategy(token_aggregation)
        except ValueError:
            return ExtractionStrategy.CHAT_LAST

    def _is_task_interface_task(self, task_name: str) -> bool:
        """Check if this is a TaskInterface task (not an lm-eval task)."""
        task_interface_tasks = {"hle", "hle_exact_match", "hle_multiple_choice", "livecodebench",
            "math500", "math", "hendrycks_math", "aime", "aime2025", "aime2024", "hmmt",
            "hmmt_feb_2025", "polymath", "polymath_en_medium", "polymath_zh_medium",
            "polymath_en_high", "polymath_zh_high", "livemathbench", "livemathbench_cnmo_en",
            "livemathbench_cnmo_zh", "multirc", "arithmetic_1dc", "arithmetic_2dm",
            "arithmetic_2ds", "arithmetic_3da", "arithmetic_3ds", "arithmetic_4da",
            "arithmetic_4ds", "arithmetic_5da", "arithmetic_5ds", "qa4mre_2013"}
        return task_name in task_interface_tasks

    def _load_task_interface_data(self, task_name: str, num_samples: int):
        """Load data from TaskInterface tasks."""
        try:
            from wisent.core.tasks.base.task_interface import get_task
            task = get_task(task_name)
            docs = task.load_data(limit=num_samples)
            return docs, task
        except Exception as e:
            logger.error(f"Failed to load TaskInterface task {task_name}: {e}")
            return [], None

    def evaluate_with_lm_eval_metrics(self, task_name: str, response_data: list, task_data) -> Dict[str, Any]:
        """Evaluate responses using task-specific evaluation metrics."""
        try:
            correct, total = 0, len(response_data)
            evaluation_details = []
            for response in response_data:
                generated, ground_truth = response["generated_response"], response["ground_truth"]
                if task_name == "gsm8k":
                    is_correct = self._evaluate_gsm8k_response(generated, ground_truth)
                elif task_name.startswith("math") or task_name in ["hendrycks_math"]:
                    is_correct = self._evaluate_gsm8k_response(generated, ground_truth)
                elif task_name in ["arc_easy", "arc_challenge"]:
                    is_correct = self._evaluate_arc_response(generated, ground_truth)
                elif task_name == "hellaswag":
                    is_correct = self._evaluate_hellaswag_response(generated, ground_truth)
                elif task_name == "mathqa":
                    is_correct = self._evaluate_mathqa_response(generated, ground_truth)
                elif task_name == "drop":
                    is_correct = self._evaluate_drop_response(generated, ground_truth)
                elif task_name.startswith("gpqa"):
                    is_correct = self._evaluate_multiple_choice_response(generated, ground_truth)
                elif task_name.startswith("hle") and "multiple_choice" in task_name:
                    is_correct = self._evaluate_multiple_choice_response(generated, ground_truth)
                elif task_name.startswith("truthfulqa") or task_name == "truthfulqa_mc1":
                    is_correct = self._evaluate_multiple_choice_response(generated, ground_truth)
                else:
                    is_correct = self._evaluate_default_response(generated, ground_truth)
                if is_correct:
                    correct += 1
                evaluation_details.append({"question": response["question"][:100], "generated": generated[-50:],
                                          "ground_truth": ground_truth, "correct": is_correct})
            accuracy = correct / total if total > 0 else 0.0
            return {"accuracy": accuracy, "correct_predictions": correct, "total_samples": total,
                    "evaluation_details": evaluation_details[:5], "task_name": task_name}
        except Exception as e:
            logger.error(f"Error in metrics evaluation: {e}")
            return {"accuracy": 0.0, "correct_predictions": 0, "total_samples": len(response_data), "error": str(e)}

    def _evaluate_gsm8k_response(self, generated: str, ground_truth) -> bool:
        """Evaluate GSM8K response using numerical answer extraction."""
        try:
            generated_answer = self._extract_numerical_answer(generated)
            ground_truth_answer = self._extract_numerical_answer(str(ground_truth))
            if generated_answer is not None and ground_truth_answer is not None:
                return abs(generated_answer - ground_truth_answer) < 1e-6
            return generated.strip().lower() == str(ground_truth).strip().lower()
        except Exception as e:
            logger.error(f"Error evaluating GSM8K response: {e}")
            return False

    def _extract_numerical_answer(self, text: str) -> float:
        """Extract numerical answer from text."""
        try:
            pattern = r"####\s*([+-]?\d+(?:\.\d+)?)"
            match = re.search(pattern, text)
            if match:
                return float(match.group(1))
            numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
            if numbers:
                return float(numbers[-1])
            return None
        except Exception:
            return None

    def _evaluate_arc_response(self, generated: str, ground_truth) -> bool:
        """Evaluate ARC response using exact match."""
        try:
            gen_clean, gt_clean = generated.strip().lower(), str(ground_truth).strip().lower()
            if gen_clean == gt_clean or gt_clean in gen_clean:
                return True
            gen_match = re.search(r"[abcd]|\d+", gen_clean)
            gt_match = re.search(r"[abcd]|\d+", gt_clean)
            return gen_match and gt_match and gen_match.group() == gt_match.group()
        except Exception:
            return False

    def _evaluate_hellaswag_response(self, generated: str, ground_truth) -> bool:
        """Evaluate HellaSwag response using exact match."""
        try:
            gen_clean, gt_clean = generated.strip().lower(), str(ground_truth).strip().lower()
            return gen_clean == gt_clean or gt_clean in gen_clean
        except Exception:
            return False

    def _evaluate_mathqa_response(self, generated: str, ground_truth) -> bool:
        """Evaluate MATH_QA response using choice matching."""
        try:
            gt_str, gen_clean = str(ground_truth).strip(), generated.strip().lower()
            if gt_str in gen_clean:
                return True
            choice_map = {"a": "0", "b": "1", "c": "2", "d": "3"}
            for letter, index in choice_map.items():
                if index == gt_str and letter in gen_clean:
                    return True
            choice_patterns = [rf"\b{gt_str}\b", rf"choice\s*{choice_map.get(gt_str, gt_str)}",
                              rf"answer\s*is\s*{gt_str}", rf"option\s*{gt_str}"]
            return any(re.search(pattern, gen_clean) for pattern in choice_patterns)
        except Exception:
            return False

    def _evaluate_drop_response(self, generated: str, ground_truth) -> bool:
        """Evaluate DROP response using structured answer format."""
        try:
            if isinstance(ground_truth, str):
                try:
                    if ground_truth.startswith("{"):
                        gt_dict = json.loads(ground_truth)
                    else:
                        return False
                except:
                    return False
            elif isinstance(ground_truth, dict):
                gt_dict = ground_truth
            else:
                return False
            gen_clean = generated.strip().lower()
            if gt_dict.get("number"):
                number_str = str(gt_dict["number"]).strip()
                if number_str and (number_str.lower() in gen_clean or number_str in re.findall(r"\b\d+\b", generated)):
                    return True
            if gt_dict.get("spans"):
                spans = gt_dict["spans"]
                if isinstance(spans, list):
                    for span in spans:
                        if str(span).strip().lower() in gen_clean:
                            return True
                elif isinstance(spans, str) and spans.strip().lower() in gen_clean:
                    return True
            return False
        except Exception:
            return False

    def _evaluate_default_response(self, generated: str, ground_truth) -> bool:
        """Default evaluation using flexible string matching."""
        try:
            gen_clean = generated.strip().lower()
            if isinstance(ground_truth, list):
                for gt_option in ground_truth:
                    gt_clean = str(gt_option).strip().lower()
                    if gen_clean == gt_clean or gt_clean in gen_clean or gen_clean in gt_clean:
                        return True
                return False
            gt_clean = str(ground_truth).strip().lower()
            return gen_clean == gt_clean or gt_clean in gen_clean or gen_clean in gt_clean
        except Exception:
            return False

    def _evaluate_multiple_choice_response(self, generated: str, ground_truth) -> bool:
        """Evaluate multiple choice response by extracting choice letter."""
        try:
            gen_clean, gt_str = generated.strip(), str(ground_truth).strip()
            gt_match = re.search(r"[ABCDE]", gt_str.upper())
            if not gt_match:
                return False
            expected_letter = gt_match.group()
            patterns = [r"(?:answer|choice|option)\s*(?:is\s+|:\s*)(?:\()?([ABCDE])(?:\))?",
                       r"the\s+(?:correct\s+)?answer\s+is\s*(?:\()?([ABCDE])(?:\))?",
                       r"(?:select|choose)\s+(?:\()?([ABCDE])(?:\))?",
                       r"(?:^|\n)([ABCDE])(?:\s*$)", r"^([ABCDE])[.,;!?)\s]*$",
                       r"^(?:\()?([ABCDE])(?:\))?\s*$"]
            for pattern in patterns:
                for match in re.finditer(pattern, gen_clean.upper(), re.IGNORECASE | re.MULTILINE):
                    if match.group(1).upper() == expected_letter:
                        return True
            return False
        except Exception:
            return False
