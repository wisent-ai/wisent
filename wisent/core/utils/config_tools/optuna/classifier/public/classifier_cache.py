"""
Classifier model caching system for efficient Optuna optimization.

This module provides intelligent caching of trained classifier models to avoid
retraining identical configurations across optimization runs and trials.
"""

import hashlib
import json
import logging
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from wisent.core.reading.classifiers.core.atoms import Classifier
from wisent.core.utils.config_tools.constants import CACHE_MAX_SIZE_GB_DEFAULT, CACHE_MAX_AGE_DAYS_DEFAULT, CACHE_MEMORY_SIZE_DEFAULT, HASH_DIGEST_PREFIX, BYTES_PER_MB, SECONDS_PER_HOUR

logger = logging.getLogger(__name__)


from wisent.core.utils.config_tools.optuna.classifier._helpers.classifier_cache_helpers import ClassifierCacheHelpersMixin

@dataclass
class CacheMetadata:
    """Metadata for cached classifier models."""

    cache_key: str
    model_name: str
    task_name: str
    model_type: str
    layer: int
    aggregation: str
    threshold: float
    hyperparameters: dict[str, Any]
    performance_metrics: dict[str, float]
    training_samples: int
    data_hash: str
    timestamp: float
    file_size_mb: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheMetadata":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CacheConfig:
    """Configuration for classifier cache."""

    cache_dir: str = "./classifier_cache"
    max_cache_size_gb: float = CACHE_MAX_SIZE_GB_DEFAULT
    max_age_days: float = CACHE_MAX_AGE_DAYS_DEFAULT
    memory_cache_size: int = CACHE_MEMORY_SIZE_DEFAULT  # Number of models to keep in memory

    def __post_init__(self):
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)


class ClassifierCache(ClassifierCacheHelpersMixin):
    """
    Intelligent caching system for trained classifier models.

    Features:
    - Hash-based cache keys for deterministic caching
    - Persistent disk storage with metadata
    - In-memory hot cache for frequently used models
    - Automatic cleanup based on size and age limits
    - Performance metrics tracking
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.memory_cache: dict[str, Classifier] = {}
        self.access_times: dict[str, float] = {}

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Load existing metadata
        self.metadata = self._load_metadata()

        # Cleanup old/large cache if needed
        self._cleanup_cache()

    def get_cache_key(
        self,
        model_name: str,
        task_name: str,
        model_type: str,
        layer: int,
        aggregation: str,
        threshold: float,
        hyperparameters: dict[str, Any],
        data_hash: str,
    ) -> str:
        """
        Generate deterministic cache key for classifier configuration.

        Args:
            model_name: Name of the base model
            task_name: Task being optimized
            model_type: Type of classifier ("logistic", "mlp")
            layer: Layer index used
            aggregation: Token aggregation method
            threshold: Classification threshold
            hyperparameters: Model-specific hyperparameters
            data_hash: Hash of the training data

        Returns:
            Unique cache key string
        """
        # Normalize model name
        clean_model_name = model_name.replace("/", "_").replace(":", "_")

        # Sort hyperparameters for consistent hashing
        sorted_hyperparams = json.dumps(hyperparameters, sort_keys=True)

        # Create cache key components
        key_components = [
            clean_model_name,
            task_name,
            model_type,
            str(layer),
            aggregation,
            f"{threshold:.3f}",
            sorted_hyperparams,
            data_hash,
        ]

        # Generate hash
        key_string = "_".join(key_components)
        cache_key = hashlib.sha256(key_string.encode()).hexdigest()[:HASH_DIGEST_PREFIX]

        return cache_key

    def has_cached_model(self, cache_key: str) -> bool:
        """Check if a model with the given cache key exists."""
        return cache_key in self.metadata or cache_key in self.memory_cache

    def save_classifier(
        self,
        cache_key: str,
        classifier: Classifier,
        model_name: str,
        task_name: str,
        layer: int,
        aggregation: str,
        threshold: float,
        hyperparameters: dict[str, Any],
        performance_metrics: dict[str, float],
        training_samples: int,
        data_hash: str,
    ) -> None:
        """
        Save a trained classifier to cache.

        Args:
            cache_key: Unique cache key
            classifier: Trained classifier model
            model_name: Name of base model
            task_name: Task name
            layer: Layer index
            aggregation: Aggregation method
            threshold: Classification threshold
            hyperparameters: Model hyperparameters
            performance_metrics: Training/validation metrics
            training_samples: Number of training samples
            data_hash: Hash of training data
        """
        try:
            # Save model to disk
            model_file = self.cache_dir / f"{cache_key}.pkl"
            with open(model_file, "wb") as f:
                pickle.dump(classifier, f)

            # Calculate file size
            file_size_mb = model_file.stat().st_size / BYTES_PER_MB

            # Create metadata
            metadata = CacheMetadata(
                cache_key=cache_key,
                model_name=model_name,
                task_name=task_name,
                model_type=classifier.model_type,
                layer=layer,
                aggregation=aggregation,
                threshold=threshold,
                hyperparameters=hyperparameters,
                performance_metrics=performance_metrics,
                training_samples=training_samples,
                data_hash=data_hash,
                timestamp=time.time(),
                file_size_mb=file_size_mb,
            )

            # Update metadata
            self.metadata[cache_key] = metadata
            self._save_metadata()

            # Add to memory cache if space available
            if len(self.memory_cache) < self.config.memory_cache_size:
                self.memory_cache[cache_key] = classifier
                self.access_times[cache_key] = time.time()

            self.logger.info(
                f"Cached classifier {cache_key}: {model_name}/{task_name} "
                f"layer_{layer} {classifier.model_type} ({file_size_mb:.2f}MB)"
            )

        except Exception as e:
            self.logger.error(f"Failed to save classifier {cache_key}: {e}")
            raise

    def load_classifier(self, cache_key: str) -> Optional[Classifier]:
        """
        Load a cached classifier model.

        Args:
            cache_key: Cache key to load

        Returns:
            Loaded classifier or None if not found
        """
        # Try memory cache first
        if cache_key in self.memory_cache:
            self.access_times[cache_key] = time.time()
            self.logger.debug(f"Loaded classifier {cache_key} from memory cache")
            return self.memory_cache[cache_key]

        # Try disk cache
        if cache_key not in self.metadata:
            return None

        model_file = self.cache_dir / f"{cache_key}.pkl"
        if not model_file.exists():
            self.logger.warning(f"Cache file missing for {cache_key}")
            # Remove from metadata
            del self.metadata[cache_key]
            self._save_metadata()
            return None

        try:
            with open(model_file, "rb") as f:
                classifier = pickle.load(f)

            # Add to memory cache (evict oldest if needed)
            if len(self.memory_cache) >= self.config.memory_cache_size:
                # Evict oldest accessed model
                oldest_key = min(self.access_times.keys(), key=self.access_times.get)
                del self.memory_cache[oldest_key]
                del self.access_times[oldest_key]

            self.memory_cache[cache_key] = classifier
            self.access_times[cache_key] = time.time()

            self.logger.debug(f"Loaded classifier {cache_key} from disk cache")
            return classifier

        except Exception as e:
            self.logger.error(f"Failed to load classifier {cache_key}: {e}")
            return None

    def get_cache_info(self) -> dict[str, Any]:
        """Get comprehensive cache information."""
        total_size_mb = sum(metadata.file_size_mb for metadata in self.metadata.values())

        # Group by task and model type
        task_counts = {}
        model_type_counts = {}

        for metadata in self.metadata.values():
            task_counts[metadata.task_name] = task_counts.get(metadata.task_name, 0) + 1
            model_type_counts[metadata.model_type] = model_type_counts.get(metadata.model_type, 0) + 1

        return {
            "total_models": len(self.metadata),
            "total_size_mb": total_size_mb,
            "memory_cache_size": len(self.memory_cache),
            "cache_dir": str(self.cache_dir),
            "task_distribution": task_counts,
            "model_type_distribution": model_type_counts,
            "oldest_cache_age_hours": (
                time.time() - min((m.timestamp for m in self.metadata.values()), default=time.time())
            )
            / SECONDS_PER_HOUR,
            "config": asdict(self.config),
        }

