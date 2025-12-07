"""Data loader for HuggingFace datasets not in lm-eval-harness."""
from __future__ import annotations
from typing import Any
import logging

from wisent.core.data_loaders.core.atoms import BaseDataLoader, DataLoaderError, LoadDataResult
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.errors import InvalidRangeError

__all__ = [
    "HuggingFaceDataLoader",
]

log = logging.getLogger(__name__)


class HuggingFaceDataLoader(BaseDataLoader):
    """
    Load datasets directly from HuggingFace that aren't in lm-eval-harness.

    This loader imports extractors from wisent/core/contrastive_pairs/huggingface_pairs/
    to convert HuggingFace datasets into contrastive pairs.
    """
    name = "huggingface"
    description = "Load from HuggingFace datasets not in lm-eval"

    def _load_one_task(
        self,
        task_name: str,
        split_ratio: float,
        seed: int,
        limit: int | None,
        training_limit: int | None,
        testing_limit: int | None,
    ) -> LoadDataResult:
        """
        Load a HuggingFace dataset by name, convert to contrastive pairs,
        split into train/test, and return a LoadDataResult.

        Args:
            task_name: Name of the task (e.g., "humaneval", "mbpp")
            split_ratio: Fraction of data for training
            seed: Random seed
            limit: Optional limit on total pairs
            training_limit: Optional limit on training pairs
            testing_limit: Optional limit on testing pairs

        Returns:
            LoadDataResult with train/test pairs
        """
        from wisent.core.contrastive_pairs.huggingface_pairs.hf_extractor_registry import get_extractor

        # Get the extractor for this task
        try:
            extractor = get_extractor(task_name)
        except Exception as e:
            raise DataLoaderError(f"No extractor found for HuggingFace task '{task_name}': {e}")

        # Extract contrastive pairs
        try:
            pairs = extractor.extract_contrastive_pairs(limit=limit)
        except Exception as e:
            raise DataLoaderError(f"Failed to extract pairs from '{task_name}': {e}")

        if not pairs:
            raise DataLoaderError(f"No pairs extracted from '{task_name}'")

        # Split into train/test
        train_pairs, test_pairs = self._split_pairs(
            pairs, split_ratio, seed, training_limit, testing_limit
        )

        if not train_pairs or not test_pairs:
            raise DataLoaderError("One of the splits is empty after splitting.")

        train_set = ContrastivePairSet("hf_train", train_pairs, task_type=task_name)
        test_set = ContrastivePairSet("hf_test", test_pairs, task_type=task_name)

        train_set.validate(raise_on_critical=False)
        test_set.validate(raise_on_critical=False)

        return LoadDataResult(
            train_qa_pairs=train_set,
            test_qa_pairs=test_set,
            task_type=task_name,
            lm_task_data=None,
        )

    def load(
        self,
        task_name: str,
        split_ratio: float | None = None,
        seed: int | None = None,
        limit: int | None = None,
        training_limit: int | None = None,
        testing_limit: int | None = None,
        **_: Any,
    ) -> LoadDataResult:
        """Load a HuggingFace dataset."""
        split = self._effective_split(split_ratio)
        seed = self._effective_seed(seed)

        return self._load_one_task(
            task_name=task_name,
            split_ratio=split,
            seed=seed,
            limit=limit,
            training_limit=training_limit,
            testing_limit=testing_limit,
        )

    def _split_pairs(
        self,
        pairs: list[ContrastivePair],
        split_ratio: float,
        seed: int,
        training_limit: int | None,
        testing_limit: int | None,
    ) -> tuple[list[ContrastivePair], list[ContrastivePair]]:
        """Split pairs into train/test sets."""
        if not pairs:
            return [], []

        if not (0.0 <= split_ratio <= 1.0):
            raise InvalidRangeError(param_name="split_ratio", actual=split_ratio, min_val=0.0, max_val=1.0)

        # Shuffle
        try:
            from numpy.random import default_rng
            rng = default_rng(seed)
            indices = rng.permutation(len(pairs)).tolist()
        except Exception:
            import random
            rnd = random.Random(seed)
            indices = list(range(len(pairs)))
            rnd.shuffle(indices)

        # Split
        split_at = int(len(pairs) * split_ratio)
        train_indices = indices[:split_at]
        test_indices = indices[split_at:]

        train_pairs = [pairs[i] for i in train_indices]
        test_pairs = [pairs[i] for i in test_indices]

        # Apply limits
        if training_limit is not None:
            train_pairs = train_pairs[:max(0, int(training_limit))]
        if testing_limit is not None:
            test_pairs = test_pairs[:max(0, int(testing_limit))]

        return train_pairs, test_pairs
