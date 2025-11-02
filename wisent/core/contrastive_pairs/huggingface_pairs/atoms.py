from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any
from abc import ABC, abstractmethod


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
            raise RuntimeError(
                f"The 'datasets' library is not available. "
                "Install it via 'pip install datasets' to use HuggingFace loaders."
            ) from exc

        try:
            dataset = load_dataset(
                dataset_name,
                dataset_config if dataset_config else None,
                split=split,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load HuggingFace dataset '{dataset_name}'. "
                f"Arguments were: config={dataset_config!r}, split={split!r}. "
                f"Underlying error: {exc}"
            ) from exc

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
