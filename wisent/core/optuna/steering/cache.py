"""
Activation caching system for the steering optimization pipeline.
"""

import hashlib
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class ActivationCache:
    """Efficient activation caching system with proper cache keys."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _generate_cache_key(
        self, split: str, layer_id: int, tokenization_config: dict[str, Any], prompt_variant: str = "default"
    ) -> str:
        """Generate unique cache key for activations."""
        config_str = json.dumps(tokenization_config, sort_keys=True)
        key_data = f"{split}_{layer_id}_{config_str}_{prompt_variant}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for key."""
        return self.cache_dir / f"activations_{cache_key}.pkl"

    def has_cached_activations(
        self, split: str, layer_id: int, tokenization_config: dict[str, Any], prompt_variant: str = "default"
    ) -> bool:
        """Check if activations are cached."""
        cache_key = self._generate_cache_key(split, layer_id, tokenization_config, prompt_variant)
        return self._get_cache_path(cache_key).exists()

    def save_activations(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        split: str,
        layer_id: int,
        tokenization_config: dict[str, Any],
        prompt_variant: str = "default",
    ):
        """Save activations to cache."""
        cache_key = self._generate_cache_key(split, layer_id, tokenization_config, prompt_variant)
        cache_path = self._get_cache_path(cache_key)

        cache_data = {
            "activations": activations,
            "labels": labels,
            "metadata": {
                "split": split,
                "layer_id": layer_id,
                "tokenization_config": tokenization_config,
                "prompt_variant": prompt_variant,
                "timestamp": datetime.now().isoformat(),
                "shape": activations.shape,
            },
        }

        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)

        self.logger.info(f"Cached activations for {split} layer {layer_id}: {activations.shape}")

    def load_activations(
        self, split: str, layer_id: int, tokenization_config: dict[str, Any], prompt_variant: str = "default"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load activations from cache."""
        cache_key = self._generate_cache_key(split, layer_id, tokenization_config, prompt_variant)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            raise FileNotFoundError(f"No cached activations found for key: {cache_key}")

        with open(cache_path, "rb") as f:
            cache_data = pickle.load(f)

        self.logger.info(f"Loaded cached activations for {split} layer {layer_id}: {cache_data['activations'].shape}")
        return cache_data["activations"], cache_data["labels"]
