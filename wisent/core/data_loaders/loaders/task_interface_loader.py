from __future__ import annotations
from typing import Any, Optional
import logging
import random

from wisent.core.data_loaders.core.atoms import BaseDataLoader, DataLoaderError, LoadDataResult
from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.task_interface import get_task, list_tasks
from wisent.core.task_interface import TaskInterface

__all__ = [
    "TaskInterfaceDataLoader",
]

log = logging.getLogger(__name__)


class TaskInterfaceDataLoader(BaseDataLoader):
    """
    Load contrastive pairs from TaskInterface tasks (AIME, HMMT, LiveCodeBench, etc.).

    This loader bridges TaskInterface tasks with the CLI training pipeline by:
    1. Loading problem data from TaskInterface tasks
    2. Converting problems into contrastive pairs (correct/incorrect answers)
    3. Splitting into train/test sets

    Usage:
        wisent train model meta-llama/Llama-3.2-1B-Instruct loader task_interface task gsm8k training_limit 100
        wisent train model meta-llama/Llama-3.2-1B-Instruct loader task_interface task aime training_limit 50
        wisent train model meta-llama/Llama-3.2-1B-Instruct loader task_interface task livecodebench training_limit 200
    """

    name = "task_interface"
    description = "Load from TaskInterface tasks (AIME, HMMT, LiveCodeBench, GSM8K, etc.)"

    def load(
        self,
        task: Optional[str] = None,
        split_ratio: Optional[float] = None,
        seed: int = 42,
        limit: Optional[int] = None,
        training_limit: Optional[int] = None,
        testing_limit: Optional[int] = None,
        **kwargs: Any,
    ) -> LoadDataResult:
        """
        Load contrastive pairs from a TaskInterface task.

        Arguments:
            task: Name of the TaskInterface task (e.g., 'gsm8k', 'aime', 'livecodebench')
            split_ratio: Fraction of data for training (default: 0.8)
            seed: Random seed for splitting
            limit: Total number of problems to load
            training_limit: Maximum training examples
            testing_limit: Maximum testing examples
            **kwargs: Additional arguments passed to the task

        Returns:
            LoadDataResult with train/test contrastive pairs

        Raises:
            DataLoaderError: If task is not specified or not found
        """
        if not task:
            available = list_tasks()
            raise DataLoaderError(
                f"TaskInterface loader requires a 'task' parameter. "
                f"Available tasks: {', '.join(available[:10])}..."
            )

        # Ensure split ratio is valid
        split_ratio = self._effective_split(split_ratio)

        # Load the task
        try:
            task_obj: TaskInterface = get_task(task, limit=limit)
        except ValueError as e:
            available = list_tasks()
            raise DataLoaderError(
                f"TaskInterface task '{task}' not found. "
                f"Available tasks: {', '.join(available[:20])}..."
            ) from e

        # Load problem data
        log.info(f"Loading data from TaskInterface task: {task}")
        problems = task_obj.load_data(limit=limit)

        if not problems:
            raise DataLoaderError(f"TaskInterface task '{task}' returned no data")

        log.info(f"Loaded {len(problems)} problems from {task}")

        # Convert problems to contrastive pairs
        pairs = self._convert_to_contrastive_pairs(task_obj, problems)

        if not pairs:
            raise DataLoaderError(
                f"Could not generate any contrastive pairs from {task}. "
                f"Problems may be missing required fields."
            )

        log.info(f"Generated {len(pairs)} contrastive pairs")

        # Shuffle and split
        random.seed(seed)
        random.shuffle(pairs)

        split_idx = int(len(pairs) * split_ratio)
        train_pairs = pairs[:split_idx]
        test_pairs = pairs[split_idx:]

        # Apply limits
        if training_limit:
            train_pairs = train_pairs[:training_limit]
        if testing_limit:
            test_pairs = test_pairs[:testing_limit]

        log.info(f"Split: {len(train_pairs)} train, {len(test_pairs)} test")

        # Create ContrastivePairSets
        train_set = ContrastivePairSet(name=f"{task}_train", pairs=train_pairs, task_type="classification")
        test_set = ContrastivePairSet(name=f"{task}_test", pairs=test_pairs, task_type="classification")

        return LoadDataResult(
            train_qa_pairs=train_set,
            test_qa_pairs=test_set,
            task_type="classification",
            lm_task_data=None,  # TaskInterface tasks don't use lm-eval format
        )

    def _convert_to_contrastive_pairs(
        self,
        task_obj: TaskInterface,
        problems: list[dict[str, Any]],
    ) -> list[ContrastivePair]:
        """
        Convert task problems into contrastive pairs.

        For each problem, we create a contrastive pair with:
        - Positive response: The correct answer
        - Negative response: An incorrect answer (generated or from problem data)

        Arguments:
            task_obj: The TaskInterface task object
            problems: List of problem dictionaries

        Returns:
            List of ContrastivePair objects
        """
        pairs = []
        extractor = task_obj.get_extractor()
        task_name = task_obj.get_name()

        for idx, problem in enumerate(problems):
            try:
                # Get the prompt/question
                if hasattr(task_obj, 'doc_to_text'):
                    prompt = task_obj.doc_to_text(problem)
                else:
                    # Fallback: extract prompt from problem dict
                    prompt = self._extract_prompt_from_problem(problem, task_name)

                if not prompt:
                    log.warning(f"Problem {idx} has no prompt, skipping")
                    continue

                # Get correct answer
                correct_answer = self._get_correct_answer(problem, task_name)
                if not correct_answer:
                    log.warning(f"Problem {idx} has no correct answer, skipping")
                    continue

                # Generate incorrect answer
                incorrect_answer = self._generate_incorrect_answer(
                    problem, correct_answer, task_name, extractor
                )
                if not incorrect_answer:
                    log.warning(f"Problem {idx}: could not generate incorrect answer, skipping")
                    continue

                # Create contrastive pair
                pair = ContrastivePair(
                    prompt=prompt,
                    positive_response=PositiveResponse(model_response=correct_answer),
                    negative_response=NegativeResponse(model_response=incorrect_answer),
                    label="correct",
                )
                pairs.append(pair)

            except Exception as e:
                log.warning(f"Problem {idx}: failed to create pair: {e}")
                continue

        return pairs

    def _extract_prompt_from_problem(self, problem: dict[str, Any], task_name: str) -> Optional[str]:
        """Extract prompt/question from a problem dict."""
        # Try common field names for prompts
        prompt_fields = ["question", "prompt", "problem", "text", "input", "query", "doc"]

        for field in prompt_fields:
            if field in problem:
                value = problem[field]
                if isinstance(value, str) and value.strip():
                    return value.strip()

        # If no prompt found, try to construct from problem data
        if task_name == "gsm8k" and "question" in problem:
            return problem["question"]

        return None

    def _get_correct_answer(self, problem: dict[str, Any], task_name: str) -> Optional[str]:
        """Extract the correct answer from a problem."""
        # Try common field names
        answer_fields = ["answer", "target", "label", "solution", "expected_output"]

        for field in answer_fields:
            if field in problem:
                answer = problem[field]
                if isinstance(answer, (str, int, float)):
                    return str(answer)
                elif isinstance(answer, dict):
                    # Try to extract from nested dict
                    if "answer" in answer:
                        return str(answer["answer"])
                    if "text" in answer:
                        return str(answer["text"])

        return None

    def _generate_incorrect_answer(
        self,
        problem: dict[str, Any],
        correct_answer: str,
        task_name: str,
        extractor: Any,
    ) -> Optional[str]:
        """
        Generate an incorrect answer for a problem.

        Strategy:
        1. Check if problem has bad_code/incorrect answer field (for LiveCodeBench)
        2. If problem has multiple choices, use an incorrect choice
        3. For numerical answers, perturb the number
        4. For text answers, use a generic incorrect response
        """
        # Strategy 0: Check for explicit bad_code field (LiveCodeBench)
        if "bad_code" in problem and problem["bad_code"]:
            return problem["bad_code"]

        # Strategy 1: Check for multiple choice options
        choices_fields = ["choices", "options", "mc1_targets", "mc2_targets"]
        for field in choices_fields:
            if field in problem:
                choices = problem[field]
                if isinstance(choices, dict):
                    # Handle mc1_targets/mc2_targets format
                    if "choices" in choices and "labels" in choices:
                        incorrect_indices = [
                            i for i, label in enumerate(choices["labels"]) if label == 0
                        ]
                        if incorrect_indices:
                            return choices["choices"][random.choice(incorrect_indices)]
                elif isinstance(choices, list) and choices:
                    # Filter out correct answer
                    incorrect_choices = [
                        c for c in choices
                        if self._normalize_for_comparison(str(c)) != self._normalize_for_comparison(correct_answer)
                    ]
                    if incorrect_choices:
                        return str(random.choice(incorrect_choices))

        # Strategy 2: Numerical answer perturbation
        try:
            correct_num = float(correct_answer.strip())
            # Perturb by 10-50%
            perturbation = random.uniform(1.1, 1.5) if random.random() > 0.5 else random.uniform(0.5, 0.9)
            incorrect_num = correct_num * perturbation
            return str(int(incorrect_num) if correct_num == int(correct_num) else round(incorrect_num, 2))
        except (ValueError, AttributeError):
            pass

        # Strategy 3: Generic incorrect responses by task type
        if task_name in ["gsm8k", "math500", "aime", "hmmt", "polymath", "livemathbench"]:
            # Math tasks: slightly wrong number
            return str(random.randint(0, 1000))
        elif task_name in ["livecodebench", "humaneval", "mbpp"]:
            # Coding tasks: empty or syntax error
            return "# Incomplete solution\npass"
        else:
            # Generic incorrect response
            return "I don't know"

    @staticmethod
    def _normalize_for_comparison(text: str) -> str:
        """Normalize text for comparison."""
        return text.lower().strip()
