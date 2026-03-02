"""
BigCode Evaluation Harness integration for Wisent.

Provides integration with bigcode-evaluation-harness for code generation benchmarks.
"""
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from wisent.core.utils.infra_tools.errors import TaskNotFoundError, InsufficientDataError
from wisent.core.utils.config_tools.constants import BIGCODE_K_VALUES

logger = logging.getLogger(__name__)

# Import task loader and task from extracted module
from wisent.core.utils.services.benchmarks.bigcode._bigcode_loader import (
    BigCodeTaskLoader,
    BigCodeTask,
)

# Import evaluator mixins
from wisent.core.utils.services.benchmarks.bigcode._bigcode_docker import BigCodeDockerMixin
from wisent.core.utils.services.benchmarks.bigcode._bigcode_scripts import BigCodeScriptsMixin
from wisent.core.utils.services.benchmarks.bigcode._bigcode_text_eval import BigCodeTextEvalMixin


class BigCodeEvaluator(BigCodeDockerMixin, BigCodeScriptsMixin, BigCodeTextEvalMixin):
    """Evaluates model outputs on BigCode benchmarks."""

    def __init__(self, docker_executor=None):
        self.docker_executor = docker_executor

    def evaluate(self, task: BigCodeTask, generations: List[str], k_values: List[int] = list(BIGCODE_K_VALUES)) -> Dict[str, Any]:
        """Evaluate generations on a BigCode task."""
        results = {
            "task": task.task_name,
            "num_samples": len(task),
            "num_generations": len(generations),
            "pass_at_k": {},
        }
        if self._is_code_execution_task(task.task_name):
            results["execution_results"] = self._evaluate_code_execution(task, generations)
            for k in k_values:
                if k <= len(generations):
                    pass_rate = self._calculate_pass_at_k(results["execution_results"], k)
                    results["pass_at_k"][f"pass@{k}"] = pass_rate
        else:
            results["bleu_scores"] = self._evaluate_text_generation(task, generations)
        return results

    def _is_code_execution_task(self, task_name: str) -> bool:
        non_execution_tasks = {
            "codexglue_code_to_text",
            "codexglue_code_to_text_python",
            "codexglue_code_to_text_go",
            "codexglue_code_to_text_ruby",
            "codexglue_code_to_text_java",
            "codexglue_code_to_text_javascript",
            "codexglue_code_to_text_php",
        }
        return task_name not in non_execution_tasks

    def _evaluate_code_execution(self, task: BigCodeTask, generations: List[str]) -> List[Dict]:
        results = []
        for i, sample in enumerate(task.get_samples()):
            sample_results = []
            for j, generation in enumerate(generations[i] if i < len(generations) else []):
                result = self._execute_and_test(sample, generation, task.task_name)
                sample_results.append(result)
            results.append({"sample_id": i, "results": sample_results})
        return results

    def _execute_and_test(self, sample: Dict, generation: str, task_name: str) -> Dict:
        if self.docker_executor:
            return self._execute_in_docker(sample, generation, task_name)
        return self._execute_in_subprocess(sample, generation, task_name)

    def _calculate_pass_at_k(self, execution_results: List[Dict], k: int) -> float:
        total_passed = 0
        total_samples = len(execution_results)
        for result in execution_results:
            sample_results = result["results"][:k]
            if any(r["passed"] for r in sample_results):
                total_passed += 1
        return total_passed / total_samples if total_samples > 0 else 0.0


def get_bigcode_loader() -> BigCodeTaskLoader:
    """Get singleton BigCodeTaskLoader."""
    return BigCodeTaskLoader()


def get_bigcode_evaluator(docker_executor=None) -> BigCodeEvaluator:
    """Get BigCodeEvaluator instance."""
    return BigCodeEvaluator(docker_executor=docker_executor)


def is_bigcode_task(task_name: str) -> bool:
    """Check if a task name is a BigCode task."""
    return get_bigcode_loader().is_bigcode_task(task_name)


def load_bigcode_task(task_name: str, limit: Optional[int] = None) -> BigCodeTask:
    """Load a BigCode task by name."""
    return get_bigcode_loader().load_task(task_name, limit=limit)


def evaluate_bigcode_task(task: BigCodeTask, generations: List[str], docker_executor=None) -> Dict[str, Any]:
    """Evaluate BigCode task with generated code."""
    evaluator = get_bigcode_evaluator(docker_executor=docker_executor)
    return evaluator.evaluate(task, generations)
