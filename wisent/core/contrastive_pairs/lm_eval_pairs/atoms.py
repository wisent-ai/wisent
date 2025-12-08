from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Sequence, TYPE_CHECKING
from abc import ABC, abstractmethod

from wisent.core.utils.dataset_splits import (
    get_train_docs as _get_train_docs,
    get_all_docs_from_task,
    create_deterministic_split,
)
from wisent.core.errors import NoDocsAvailableError, DatasetLoadError


if TYPE_CHECKING:
    from wisent.core.contrastive_pairs.core.pair import ContrastivePair
    from lm_eval.api.task import ConfigurableTask


__all__ = [
    "UnsupportedLMEvalBenchmarkError",
    "NoLabelledDocsAvailableError",
    "LMEvalBenchmarkExtractor",
]


class UnsupportedLMEvalBenchmarkError(Exception):
    """Raised when a benchmark/task does not have a compatible extractor."""


class NoLabelledDocsAvailableError(UnsupportedLMEvalBenchmarkError):
    """
    Raised when no labeled documents can be found for a given lm-eval task.

    This typically indicates the task does not expose any of:
    validation/test/training/fewshot docs, nor sufficient dataset metadata
    to load a split directly.
    """

class LMEvalBenchmarkExtractor(ABC):
    """
    Abstract base class for lm-eval benchmark-specific extractors.

    Subclasses should implement :meth:'extract_contrastive_pairs' to transform
    task documents into a list of :class:'ContrastivePair' instances.

    Documents are loaded using our unified split strategy: all available splits
    are combined and then split 80/20 into train/test. For contrastive pair
    extraction, we use the TRAINING portion to avoid data leakage with evaluation.

    Subclasses should declare:
        evaluator_name (str): Name of the evaluator to use (e.g., "log_likelihoods")
    """

    # Default evaluator - subclasses should override
    evaluator_name: str = "log_likelihoods"

    @abstractmethod
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Extract contrastive pairs from the provided lm-eval task.

        arguments:
            lm_eval_task_data:
                An lm-eval task instance.
            limit:
                Optional upper bound on the number of pairs to return.
                Values <= 0 are treated as "no limit".

        returns:
            A list of :class:'ContrastivePair'.
        """
        raise NotImplementedError

    def extract_contrastive_pair(
        self,
        sample: dict[str, Any],
        task: Any = None,
    ) -> dict[str, str] | None:
        """
        Extract a contrastive pair from a single sample dictionary.
        
        This method is used by the optuna pipeline for sample-by-sample processing.
        Subclasses should override this if they have custom extraction logic.
        
        Default implementation calls _extract_pair_from_doc if available.

        arguments:
            sample: A single document/sample dictionary.
            task: Optional task object (may be needed for some extractors).

        returns:
            A dict with keys: "question", "correct_answer", "incorrect_answer"
            or None if extraction fails.
        """
        if hasattr(self, '_extract_pair_from_doc'):
            pair = self._extract_pair_from_doc(sample)
            if pair is not None:
                return {
                    "question": pair.prompt,
                    "correct_answer": pair.positive_response.model_response,
                    "incorrect_answer": pair.negative_response.model_response,
                }
        return None

    def extract_qa_pair(
        self,
        sample: dict[str, Any],
        task: Any = None,
    ) -> dict[str, str] | None:
        """
        Extract a question-answer pair from a single sample dictionary.
        
        This method is used for evaluation - it extracts the question and
        correct answer without needing an incorrect answer.

        arguments:
            sample: A single document/sample dictionary.
            task: Optional task object (may be needed for some extractors).

        returns:
            A dict with keys: "formatted_question", "correct_answer"
            or None if extraction fails.
        """
        contrastive = self.extract_contrastive_pair(sample, task)
        if contrastive:
            return {
                "formatted_question": contrastive["question"],
                "correct_answer": contrastive["correct_answer"],
            }
        return None


    @classmethod
    def load_docs(
        cls,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Load TRAINING documents for contrastive pair extraction.

        This method combines ALL available splits from the task and applies
        our own 80/20 train/test split. Only the TRAINING portion is returned
        to ensure no data leakage with evaluation (which uses the test portion).

        arguments:
            lm_eval_task_data:
                Task object from lm-eval.
            limit:
                Optional maximum number of documents to return.
                Values <= 0 are treated as "no limit".
            preferred_doc:
                DEPRECATED - ignored. All splits are now combined.

        returns:
            A list of document dictionaries (training portion only).

        raises:
            NoLabelledDocsAvailableError:
                If no labeled documents are available.
        """
        max_items = cls._normalize_limit(limit)

        # Get benchmark name for deterministic splitting
        benchmark_name = getattr(
            lm_eval_task_data, "NAME",
            getattr(lm_eval_task_data, "TASK_NAME", type(lm_eval_task_data).__name__)
        )

        # Get ALL docs from all splits
        all_docs, split_counts = get_all_docs_from_task(lm_eval_task_data)

        if not all_docs:
            raise NoDocsAvailableError(task_name=benchmark_name)

        # Apply our 80/20 split and get TRAINING docs only
        train_docs, _ = create_deterministic_split(all_docs, benchmark_name)

        # Coerce to dicts
        docs_list = cls._coerce_docs_to_dicts(train_docs, max_items)

        total = len(all_docs)
        train_count = len(train_docs)
        returned = len(docs_list)
        print(f"Loaded {returned} training docs from {benchmark_name} "
              f"(total: {total}, train split: {train_count}, original splits: {split_counts})")

        return docs_list

    @staticmethod
    def _normalize_limit(limit: int | None) -> int | None:
        """
        Normalize limit semantics:
          - None → None (unbounded)
          - <= 0 → None (unbounded)
          - > 0 → limit
        """
        if limit is None or limit <= 0:
            return None
        return int(limit)

    @staticmethod
    def _has_callable(obj: Any, name: str) -> bool:
        """Return True if obj has a callable attribute with the given name."""
        return hasattr(obj, name) and callable(getattr(obj, name))

    @staticmethod
    def _has_true(obj: Any, name: str) -> bool:
        """Return True if obj has an attribute that evaluates to True when called or read."""
        attr = getattr(obj, name, None)
        try:
            return bool(attr() if callable(attr) else attr)
        except Exception:  # pragma: no cover (defensive)
            return False

    @classmethod
    def _coerce_docs_to_dicts(
        cls,
        docs_iter: Iterable[Any] | None,
        max_items: int | None,
    ) -> list[dict[str, Any]]:
        """
        Materialize an iterable of docs into a list of dictionaries,
        applying an optional limit.
        """
        if docs_iter is None:
            return []

        out: list[dict[str, Any]] = []
        for idx, item in enumerate(docs_iter):
            if max_items is not None and idx >= max_items:
                break
            if isinstance(item, Mapping):
                out.append(dict(item))
            else:
                try:
                    out.append(dict(item))  
                except Exception as exc:  
                    raise TypeError(
                        "Expected each document to be a mapping-like object that can "
                        "be converted to dict. Got type "
                        f"{type(item).__name__} with value {item!r}"
                    ) from exc
        return out

    @classmethod
    def _fallback_load_from_dataset(
        cls,
        lm_eval_task_data: ConfigurableTask,
        max_items: int | None,
    ) -> list[dict[str, Any]]:
        """
        DEPRECATED: Fallback dataset loading is not permitted.
        
        Raises:
            DatasetLoadError: Always raises an error - fallback loading is not permitted.
        """
        task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
        raise DatasetLoadError(task_name=task_name)