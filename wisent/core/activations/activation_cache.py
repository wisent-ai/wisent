"""
Activation cache for geometry search.

Caches activations for ALL layers once per (benchmark, strategy) pair.
Layer combinations are then tested from cache without re-extraction.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch

from wisent.core.activations.core.atoms import LayerActivations, RawActivationMap
from wisent.core.activations.extraction_strategy import ExtractionStrategy
from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.utils.device import resolve_default_device


@dataclass
class CachedActivations:
    """
    Cached activations for a single (benchmark, strategy) pair.
    
    Contains activations for ALL layers for all pairs.
    Layer combinations can be extracted without re-running the model.
    """
    benchmark: str
    strategy: ExtractionStrategy
    model_name: str
    num_layers: int
    
    # List of (positive_activations, negative_activations) per pair
    # Each activation is a dict: layer_name -> tensor [hidden_size]
    pair_activations: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = field(default_factory=list)
    
    # Metadata
    num_pairs: int = 0
    hidden_size: int = 0
    
    def add_pair(self, positive: LayerActivations, negative: LayerActivations) -> None:
        """Add activations for a contrastive pair."""
        pos_dict = {k: v.clone() for k, v in positive.items() if v is not None}
        neg_dict = {k: v.clone() for k, v in negative.items() if v is not None}
        self.pair_activations.append((pos_dict, neg_dict))
        self.num_pairs = len(self.pair_activations)
        
        # Infer hidden size from first tensor
        if self.hidden_size == 0 and pos_dict:
            first_tensor = next(iter(pos_dict.values()))
            self.hidden_size = first_tensor.shape[-1]
    
    def get_layer_subset(self, layers: List[int]) -> "CachedActivations":
        """
        Get a new CachedActivations with only the specified layers.
        
        Args:
            layers: List of layer indices (0-based)
            
        Returns:
            New CachedActivations with only the specified layers
        """
        layer_names = [str(l) for l in layers]
        
        new_pairs = []
        for pos_dict, neg_dict in self.pair_activations:
            new_pos = {k: v for k, v in pos_dict.items() if k in layer_names}
            new_neg = {k: v for k, v in neg_dict.items() if k in layer_names}
            new_pairs.append((new_pos, new_neg))
        
        result = CachedActivations(
            benchmark=self.benchmark,
            strategy=self.strategy,
            model_name=self.model_name,
            num_layers=len(layers),
            hidden_size=self.hidden_size,
        )
        result.pair_activations = new_pairs
        result.num_pairs = len(new_pairs)
        return result
    
    def get_available_layers(self) -> List[str]:
        """Get list of available layer names."""
        if not self.pair_activations:
            return []
        return list(self.pair_activations[0][0].keys())
    
    def get_positive_activations(self, layer: int | str) -> torch.Tensor:
        """
        Get stacked positive activations for a single layer.
        
        Args:
            layer: Layer index (int) or layer name (str)
            
        Returns:
            Tensor of shape [num_pairs, hidden_size]
        """
        layer_name = str(layer)
        tensors = [pos[layer_name] for pos, _ in self.pair_activations if layer_name in pos]
        if not tensors:
            raise KeyError(f"Layer {layer_name} not found. Available: {self.get_available_layers()}")
        return torch.stack(tensors, dim=0)
    
    def get_negative_activations(self, layer: int | str) -> torch.Tensor:
        """
        Get stacked negative activations for a single layer.
        
        Args:
            layer: Layer index (int) or layer name (str)
            
        Returns:
            Tensor of shape [num_pairs, hidden_size]
        """
        layer_name = str(layer)
        tensors = [neg[layer_name] for _, neg in self.pair_activations if layer_name in neg]
        if not tensors:
            raise KeyError(f"Layer {layer_name} not found. Available: {self.get_available_layers()}")
        return torch.stack(tensors, dim=0)
    
    def get_diff_activations(self, layer: int | str) -> torch.Tensor:
        """
        Get positive - negative activation differences for a layer.
        
        Args:
            layer: Layer index (int) or layer name (str)
            
        Returns:
            Tensor of shape [num_pairs, hidden_size]
        """
        return self.get_positive_activations(layer) - self.get_negative_activations(layer)
    
    def get_all_layers_diff(self) -> Dict[str, torch.Tensor]:
        """
        Get activation differences for all layers.
        
        Returns:
            Dict mapping layer_name -> tensor [num_pairs, hidden_size]
        """
        result = {}
        if not self.pair_activations:
            return result
        
        # Get layer names from first pair
        layer_names = list(self.pair_activations[0][0].keys())
        for layer_name in layer_names:
            pos_tensors = []
            neg_tensors = []
            for pos, neg in self.pair_activations:
                if layer_name in pos and layer_name in neg:
                    pos_tensors.append(pos[layer_name])
                    neg_tensors.append(neg[layer_name])
            if pos_tensors:
                result[layer_name] = torch.stack(pos_tensors) - torch.stack(neg_tensors)
        return result
    
    def to_device(self, device: str) -> "CachedActivations":
        """Move all tensors to a device."""
        new_pairs = []
        for pos, neg in self.pair_activations:
            new_pos = {k: v.to(device) for k, v in pos.items()}
            new_neg = {k: v.to(device) for k, v in neg.items()}
            new_pairs.append((new_pos, new_neg))
        
        result = CachedActivations(
            benchmark=self.benchmark,
            strategy=self.strategy,
            model_name=self.model_name,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
        )
        result.pair_activations = new_pairs
        result.num_pairs = self.num_pairs
        return result


class ActivationCache:
    """
    Disk-backed cache for activations.
    
    Saves/loads activations per (model, benchmark, strategy) tuple.
    """
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, CachedActivations] = {}
    
    def _get_cache_key(self, model_name: str, benchmark: str, strategy: ExtractionStrategy) -> str:
        """Generate a unique cache key."""
        key_str = f"{model_name}_{benchmark}_{strategy.value}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path for a cache file."""
        return self.cache_dir / f"{cache_key}.pt"
    
    def _get_metadata_path(self, cache_key: str) -> Path:
        """Get path for cache metadata."""
        return self.cache_dir / f"{cache_key}.json"
    
    def has(self, model_name: str, benchmark: str, strategy: ExtractionStrategy) -> bool:
        """Check if activations are cached."""
        key = self._get_cache_key(model_name, benchmark, strategy)
        if key in self._memory_cache:
            return True
        return self._get_cache_path(key).exists()
    
    def get(
        self, 
        model_name: str, 
        benchmark: str, 
        strategy: ExtractionStrategy,
        load_to_memory: bool = True,
    ) -> Optional[CachedActivations]:
        """
        Get cached activations if they exist.
        
        Args:
            model_name: Model identifier
            benchmark: Benchmark name
            strategy: Extraction strategy
            load_to_memory: If True, keep in memory cache after loading
            
        Returns:
            CachedActivations or None if not cached
        """
        key = self._get_cache_key(model_name, benchmark, strategy)
        
        # Check memory cache first
        if key in self._memory_cache:
            return self._memory_cache[key]
        
        # Check disk cache
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None
        
        # Load from disk
        data = torch.load(cache_path, map_location=resolve_default_device(), weights_only=False)
        
        cached = CachedActivations(
            benchmark=data["benchmark"],
            strategy=ExtractionStrategy(data["strategy"]),
            model_name=data["model_name"],
            num_layers=data["num_layers"],
            hidden_size=data["hidden_size"],
        )
        cached.pair_activations = data["pair_activations"]
        cached.num_pairs = data["num_pairs"]
        
        if load_to_memory:
            self._memory_cache[key] = cached
        
        return cached
    
    def put(
        self, 
        cached: CachedActivations,
        save_to_disk: bool = True,
    ) -> None:
        """
        Store cached activations.
        
        Args:
            cached: CachedActivations to store
            save_to_disk: If True, persist to disk
        """
        key = self._get_cache_key(cached.model_name, cached.benchmark, cached.strategy)
        
        # Store in memory
        self._memory_cache[key] = cached
        
        if save_to_disk:
            # Save to disk
            data = {
                "benchmark": cached.benchmark,
                "strategy": cached.strategy.value,
                "model_name": cached.model_name,
                "num_layers": cached.num_layers,
                "hidden_size": cached.hidden_size,
                "num_pairs": cached.num_pairs,
                "pair_activations": cached.pair_activations,
            }
            torch.save(data, self._get_cache_path(key))
            
            # Save metadata as JSON
            metadata = {
                "benchmark": cached.benchmark,
                "strategy": cached.strategy.value,
                "model_name": cached.model_name,
                "num_layers": cached.num_layers,
                "hidden_size": cached.hidden_size,
                "num_pairs": cached.num_pairs,
            }
            with open(self._get_metadata_path(key), "w") as f:
                json.dump(metadata, f, indent=2)
    
    def clear_memory(self) -> None:
        """Clear the in-memory cache."""
        self._memory_cache.clear()
    
    def list_cached(self) -> List[Dict[str, Any]]:
        """List all cached activations."""
        result = []
        for meta_path in self.cache_dir.glob("*.json"):
            with open(meta_path) as f:
                result.append(json.load(f))
        return result
    
    def get_cache_size_bytes(self) -> int:
        """Get total size of cache on disk."""
        total = 0
        for path in self.cache_dir.glob("*.pt"):
            total += path.stat().st_size
        return total


def collect_and_cache_activations(
    model: "WisentModel",
    pairs: List[ContrastivePair],
    benchmark: str,
    strategy: ExtractionStrategy,
    cache: Optional[ActivationCache] = None,
    cache_dir: Optional[str] = None,
    show_progress: bool = True,
) -> CachedActivations:
    """
    Collect activations for all pairs and all layers, then cache.
    
    Args:
        model: WisentModel instance
        pairs: List of contrastive pairs
        benchmark: Benchmark name
        strategy: Extraction strategy
        cache: Optional existing cache to use
        cache_dir: Cache directory (used if cache not provided)
        show_progress: Print progress
        
    Returns:
        CachedActivations with all layers for all pairs
    """
    from wisent.core.activations.activations_collector import ActivationCollector
    
    # Check cache first
    if cache is None and cache_dir:
        cache = ActivationCache(cache_dir)
    
    if cache and cache.has(model.model_name, benchmark, strategy):
        if show_progress:
            print(f"Loading cached activations for {benchmark}/{strategy.value}")
        return cache.get(model.model_name, benchmark, strategy)
    
    # Collect activations for ALL layers (preserve model's native dtype)
    collector = ActivationCollector(model=model)
    
    cached = CachedActivations(
        benchmark=benchmark,
        strategy=strategy,
        model_name=model.model_name,
        num_layers=model.num_layers,
    )
    
    for i, pair in enumerate(pairs):
        if show_progress and i % 10 == 0:
            print(f"Collecting activations: {i+1}/{len(pairs)}", end="\r", flush=True)
        
        # Collect ALL layers (layers=None)
        updated = collector.collect(pair, strategy=strategy, layers=None)
        cached.add_pair(
            updated.positive_response.layers_activations,
            updated.negative_response.layers_activations,
        )
    
    if show_progress:
        print(f"Collected activations: {len(pairs)}/{len(pairs)} pairs, {cached.num_layers} layers")
    
    # Cache the result
    if cache:
        cache.put(cached)
        if show_progress:
            print(f"Cached to {cache.cache_dir}")
    
    return cached


# Type hint for WisentModel (avoid circular import)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wisent.core.models.wisent_model import WisentModel
