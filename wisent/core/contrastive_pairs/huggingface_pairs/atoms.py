from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, TYPE_CHECKING
from abc import ABC, abstractmethod

from wisent.core.errors import FileLoadError, DatasetLoadError

if TYPE_CHECKING:
    from wisent.core.contrastive_pairs.core.pair import ContrastivePair

__all__ = [
    "UnsupportedHuggingFaceBenchmarkError",
    "HuggingFaceBenchmarkExtractor",
]


class UnsupportedHuggingFaceBenchmarkError(Exception):
    """Raised when a HuggingFace benchmark/task does not have a compatible extractor."""


class HuggingFaceBenchmarkExtractor(ABC):
    """
    Abstract base class for HuggingFace benchmark-specific extractors.

    Subclasses should implement :meth:`extract_contrastive_pairs` to transform
    dataset examples into a list of :class:`ContrastivePair` instances.

    This is for datasets that are NOT in lm-eval-harness but are available
    on HuggingFace directly.
    """

    @abstractmethod
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list:
        """
        Extract contrastive pairs from the HuggingFace dataset.

        arguments:
            limit:
                Optional upper bound on the number of pairs to return.
                Values <= 0 are treated as "no limit".

        returns:
            A list of :class:`ContrastivePair`.
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
    def load_dataset(
        cls,
        dataset_name: str,
        dataset_config: str | None = None,
        split: str = "test",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Load a HuggingFace dataset and convert to list of dicts.

        arguments:
            dataset_name:
                HuggingFace dataset identifier (e.g., "openai_humaneval").
            dataset_config:
                Optional dataset configuration/subset name.
            split:
                Dataset split to load (default: "test").
            limit:
                Optional maximum number of examples to return.

        returns:
            A list of document dictionaries.

        raises:
            RuntimeError:
                If the dataset cannot be loaded.
        """
        max_items = cls._normalize_limit(limit)

        try:
            from datasets import load_dataset
        except Exception as exc:
            raise FileLoadError(
                file_path="datasets library",
                cause=exc
            )

        try:
            dataset = load_dataset(
                dataset_name,
                dataset_config if dataset_config else None,
                split=split,
                trust_remote_code=True,
            )
        except ValueError as exc:
            # Handle deprecated 'List' feature type by patching features dict
            if "Feature type 'List' not found" in str(exc) or "Type mismatch" in str(exc):
                import datasets.features.features as features_module
                # Temporarily register 'List' as alias for 'LargeList'
                orig_feature_types = features_module._FEATURE_TYPES.copy()
                features_module._FEATURE_TYPES['List'] = features_module._FEATURE_TYPES['LargeList']
                try:
                    # Force download_mode to redownload to avoid cached metadata issues
                    dataset = load_dataset(
                        dataset_name,
                        dataset_config if dataset_config else None,
                        split=split,
                        trust_remote_code=True,
                        download_mode='force_redownload',
                    )
                finally:
                    # Restore original feature types
                    features_module._FEATURE_TYPES = orig_feature_types
            else:
                raise DatasetLoadError(task_name=dataset_name)
        except Exception as exc:
            raise DatasetLoadError(task_name=dataset_name)

        return cls._coerce_docs_to_dicts(dataset, max_items)

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

    @staticmethod
    def _build_pair(
        question: str,
        correct: str,
        incorrect: str,
        metadata: dict[str, Any] | None = None,
    ) -> "ContrastivePair":
        """
        Build a ContrastivePair from question and responses.

        This is a shared utility method used by most extractors to construct
        ContrastivePair objects with consistent structure.

        Arguments:
            question: The prompt/question text.
            correct: The correct/positive response.
            incorrect: The incorrect/negative response.
            metadata: Optional metadata dict (should contain 'label' key).

        Returns:
            A ContrastivePair with positive and negative responses.
        """
        from wisent.core.contrastive_pairs.core.pair import ContrastivePair
        from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse

        positive_response = PositiveResponse(model_response=correct)
        negative_response = NegativeResponse(model_response=incorrect)
        return ContrastivePair(
            prompt=question,
            positive_response=positive_response,
            negative_response=negative_response,
            label=metadata.get("label") if metadata else None,
            metadata=metadata,
        )
