"""Cache management helpers for ClassifierCache."""

import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


class ClassifierCacheHelpersMixin:
    """Mixin providing cache management methods for ClassifierCache."""

    def find_similar_models(
        self,
        model_name: str,
        task_name: str,
        model_type: Optional[str] = None,
        layer: Optional[int] = None,
        top_k: int = 5,
    ) -> list[tuple[str, CacheMetadata, float]]:
        """
        Find similar cached models based on configuration.

        Args:
            model_name: Base model name
            task_name: Task name
            model_type: Optional model type filter
            layer: Optional layer filter
            top_k: Maximum number of results

        Returns:
            List of (cache_key, metadata, similarity_score) tuples
        """
        candidates = []

        for cache_key, metadata in self.metadata.items():
            # Calculate similarity score
            score = 0.0

            # Model name match (highest weight)
            if metadata.model_name == model_name:
                score += 0.4

            # Task name match
            if metadata.task_name == task_name:
                score += 0.3

            # Model type match
            if model_type and metadata.model_type == model_type:
                score += 0.2

            # Layer proximity
            if layer is not None:
                layer_diff = abs(metadata.layer - layer)
                layer_score = max(0, 1.0 - layer_diff / 10.0)  # Decay with distance
                score += 0.1 * layer_score

            # Only include models with some similarity
            if score > 0.1:
                candidates.append((cache_key, metadata, score))

        # Sort by similarity score and return top_k
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:top_k]

    def clear_cache(self, keep_recent_hours: float = 0) -> int:
        """
        Clear cached models.

        Args:
            keep_recent_hours: Keep models newer than this many hours

        Returns:
            Number of models removed
        """
        cutoff_time = time.time() - (keep_recent_hours * 3600)
        removed_count = 0

        keys_to_remove = []
        for cache_key, metadata in self.metadata.items():
            if metadata.timestamp < cutoff_time:
                keys_to_remove.append(cache_key)

        for cache_key in keys_to_remove:
            try:
                # Remove from disk
                model_file = self.cache_dir / f"{cache_key}.pkl"
                if model_file.exists():
                    model_file.unlink()

                # Remove from memory cache
                if cache_key in self.memory_cache:
                    del self.memory_cache[cache_key]
                if cache_key in self.access_times:
                    del self.access_times[cache_key]

                # Remove from metadata
                del self.metadata[cache_key]
                removed_count += 1

            except Exception as e:
                self.logger.warning(f"Failed to remove cached model {cache_key}: {e}")

        self._save_metadata()
        self.logger.info(f"Cleared {removed_count} cached models")
        return removed_count

    def _load_metadata(self) -> dict[str, CacheMetadata]:
        """Load cache metadata from disk."""
        if not self.metadata_file.exists():
            return {}

        try:
            with open(self.metadata_file) as f:
                data = json.load(f)

            metadata = {}
            for cache_key, metadata_dict in data.items():
                metadata[cache_key] = CacheMetadata.from_dict(metadata_dict)

            self.logger.debug(f"Loaded metadata for {len(metadata)} cached models")
            return metadata

        except Exception as e:
            self.logger.warning(f"Failed to load cache metadata: {e}")
            return {}

    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        try:
            data = {}
            for cache_key, metadata in self.metadata.items():
                data[cache_key] = metadata.to_dict()

            with open(self.metadata_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save cache metadata: {e}")

    def _cleanup_cache(self) -> None:
        """Clean up cache based on size and age limits."""
        current_time = time.time()
        total_size_mb = sum(metadata.file_size_mb for metadata in self.metadata.values())

        # Remove old models
        old_threshold = current_time - (self.config.max_age_days * 24 * 3600)
        old_models = [cache_key for cache_key, metadata in self.metadata.items() if metadata.timestamp < old_threshold]

        if old_models:
            for cache_key in old_models:
                try:
                    model_file = self.cache_dir / f"{cache_key}.pkl"
                    if model_file.exists():
                        model_file.unlink()
                    del self.metadata[cache_key]
                except Exception as e:
                    self.logger.warning(f"Failed to remove old model {cache_key}: {e}")

            self.logger.info(f"Removed {len(old_models)} old cached models")
            total_size_mb = sum(metadata.file_size_mb for metadata in self.metadata.values())

        # Remove largest models if over size limit
        if total_size_mb > self.config.max_cache_size_gb * 1024:
            # Sort by size (largest first)
            models_by_size = sorted(self.metadata.items(), key=lambda x: x[1].file_size_mb, reverse=True)

            removed_count = 0
            for cache_key, metadata in models_by_size:
                if total_size_mb <= self.config.max_cache_size_gb * 1024:
                    break

                try:
                    model_file = self.cache_dir / f"{cache_key}.pkl"
                    if model_file.exists():
                        model_file.unlink()

                    total_size_mb -= metadata.file_size_mb
                    del self.metadata[cache_key]
                    removed_count += 1

                except Exception as e:
                    self.logger.warning(f"Failed to remove large model {cache_key}: {e}")

            if removed_count > 0:
                self.logger.info(f"Removed {removed_count} large cached models to free space")

        # Save updated metadata
        self._save_metadata()

    def compute_data_hash(self, X: torch.Tensor, y: torch.Tensor) -> str:
        """
        Compute hash of training data for cache key generation.

        Args:
            X: Training features (torch tensor)
            y: Training labels (torch tensor)

        Returns:
            Hash string representing the data
        """
        # Work directly with tensors - no numpy conversion needed
        # Use shape and sample of data for hashing (efficient for large datasets)
        x_hash = hashlib.md5(str(tuple(X.shape)).encode()).hexdigest()[:8]
        y_hash = hashlib.md5(str(tuple(y.shape)).encode()).hexdigest()[:8]

        # Sample some data points for more unique hash (tensor operations)
        if X.size(0) > 10:
            # Use tensor indexing instead of numpy.linspace
            sample_indices = torch.linspace(0, X.size(0) - 1, 10, dtype=torch.long)
            x_sample = X[sample_indices].flatten()[:100]  # First 100 values
            y_sample = y[sample_indices]
        else:
            x_sample = X.flatten()[:100]
            y_sample = y

        # Convert tensor data to bytes for hashing (float32 required, bfloat16 not supported)
        x_sample_bytes = x_sample.detach().cpu().float().numpy().tobytes()
        y_sample_bytes = y_sample.detach().cpu().float().numpy().tobytes()

        x_sample_hash = hashlib.md5(x_sample_bytes).hexdigest()[:8]
        y_sample_hash = hashlib.md5(y_sample_bytes).hexdigest()[:8]

        return f"{x_hash}_{y_hash}_{x_sample_hash}_{y_sample_hash}"
