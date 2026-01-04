"""
Activation cache for geometry search.

Two cache types:
1. CachedActivations - legacy, stores extracted vectors per (benchmark, strategy)
2. RawCachedActivations - stores full hidden states per (benchmark, text_family)
   Allows extracting any strategy from same family without re-running model.

Text families (strategies that share same input text):
- "chat": chat_mean, chat_first, chat_last, chat_max_norm, chat_weighted
- "role_play": role_play  
- "mc": mc_balanced, mc_completion
- "completion": completion_last, completion_mean
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
from wisent.core.activations.extraction_strategy import ExtractionStrategy, extract_activation
from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.utils.device import resolve_default_device


def get_strategy_text_family(strategy: ExtractionStrategy) -> str:
    """
    Get the text family for a strategy.
    
    Strategies in the same family use the same input text (prompt construction),
    they only differ in which token(s) they extract from the output.
    
    Families:
    - "chat": chat_mean, chat_first, chat_last, chat_max_norm, chat_weighted
    - "role_play": role_play
    - "mc": mc_balanced, mc_completion  
    - "completion": completion_last, completion_mean
    
    Returns:
        Family name string
    """
    if strategy in (
        ExtractionStrategy.CHAT_MEAN,
        ExtractionStrategy.CHAT_FIRST,
        ExtractionStrategy.CHAT_LAST,
        ExtractionStrategy.CHAT_MAX_NORM,
        ExtractionStrategy.CHAT_WEIGHTED,
    ):
        return "chat"
    elif strategy == ExtractionStrategy.ROLE_PLAY:
        return "role_play"
    elif strategy in (ExtractionStrategy.MC_BALANCED, ExtractionStrategy.MC_COMPLETION):
        return "mc"
    elif strategy in (ExtractionStrategy.COMPLETION_LAST, ExtractionStrategy.COMPLETION_MEAN):
        return "completion"
    else:
        # Unknown strategy, use its own name
        return strategy.value


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


@dataclass
class RawPairData:
    """Raw hidden states and metadata for a single contrastive pair."""
    # Full hidden states: layer_name -> [seq_len, hidden_size]
    pos_hidden_states: Dict[str, torch.Tensor]
    neg_hidden_states: Dict[str, torch.Tensor]
    # Metadata for extraction
    pos_answer_text: str
    neg_answer_text: str
    pos_prompt_len: int
    neg_prompt_len: int


@dataclass
class RawCachedActivations:
    """
    Cache full hidden states per (benchmark, text_family).
    
    Stores complete sequences so any extraction strategy in the same
    text family can be applied without re-running the model.
    
    This is more memory-intensive but allows testing multiple strategies
    (e.g., chat_last, chat_mean, chat_first) from a single forward pass.
    """
    benchmark: str
    text_family: str  # "chat", "role_play", "mc", "completion"
    model_name: str
    num_layers: int
    
    # Raw data per pair
    pairs: List[RawPairData] = field(default_factory=list)
    
    # Metadata
    num_pairs: int = 0
    hidden_size: int = 0
    
    def add_pair(
        self,
        pos_hidden_states: Dict[str, torch.Tensor],
        neg_hidden_states: Dict[str, torch.Tensor],
        pos_answer_text: str,
        neg_answer_text: str,
        pos_prompt_len: int,
        neg_prompt_len: int,
    ) -> None:
        """Add raw hidden states for a contrastive pair."""
        pair_data = RawPairData(
            pos_hidden_states={k: v.clone() for k, v in pos_hidden_states.items()},
            neg_hidden_states={k: v.clone() for k, v in neg_hidden_states.items()},
            pos_answer_text=pos_answer_text,
            neg_answer_text=neg_answer_text,
            pos_prompt_len=pos_prompt_len,
            neg_prompt_len=neg_prompt_len,
        )
        self.pairs.append(pair_data)
        self.num_pairs = len(self.pairs)
        
        # Infer hidden size
        if self.hidden_size == 0 and pos_hidden_states:
            first_tensor = next(iter(pos_hidden_states.values()))
            self.hidden_size = first_tensor.shape[-1]
    
    def extract_with_strategy(
        self,
        strategy: ExtractionStrategy,
        tokenizer,
        layer: int | str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract activations using a specific strategy.
        
        Args:
            strategy: Extraction strategy (must be in same text family)
            tokenizer: Tokenizer for computing answer token positions
            layer: Layer to extract from
            
        Returns:
            Tuple of (pos_activations, neg_activations) each [num_pairs, hidden_size]
        """
        # Verify strategy is compatible
        if get_strategy_text_family(strategy) != self.text_family:
            raise ValueError(
                f"Strategy {strategy.value} (family: {get_strategy_text_family(strategy)}) "
                f"incompatible with cached family: {self.text_family}"
            )
        
        layer_name = str(layer)
        pos_acts = []
        neg_acts = []
        
        for pair_data in self.pairs:
            # Extract positive
            pos_hs = pair_data.pos_hidden_states.get(layer_name)
            if pos_hs is not None:
                pos_vec = extract_activation(
                    strategy, pos_hs, pair_data.pos_answer_text, 
                    tokenizer, pair_data.pos_prompt_len
                )
                pos_acts.append(pos_vec)
            
            # Extract negative
            neg_hs = pair_data.neg_hidden_states.get(layer_name)
            if neg_hs is not None:
                neg_vec = extract_activation(
                    strategy, neg_hs, pair_data.neg_answer_text,
                    tokenizer, pair_data.neg_prompt_len
                )
                neg_acts.append(neg_vec)
        
        return torch.stack(pos_acts), torch.stack(neg_acts)
    
    def to_cached_activations(
        self,
        strategy: ExtractionStrategy,
        tokenizer,
    ) -> CachedActivations:
        """
        Convert to CachedActivations for a specific strategy.
        
        Args:
            strategy: Extraction strategy
            tokenizer: Tokenizer for extraction
            
        Returns:
            CachedActivations with extracted vectors
        """
        cached = CachedActivations(
            benchmark=self.benchmark,
            strategy=strategy,
            model_name=self.model_name,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
        )
        
        for pair_data in self.pairs:
            pos_dict = {}
            neg_dict = {}
            
            for layer_name in pair_data.pos_hidden_states.keys():
                pos_hs = pair_data.pos_hidden_states[layer_name]
                neg_hs = pair_data.neg_hidden_states[layer_name]
                
                pos_dict[layer_name] = extract_activation(
                    strategy, pos_hs, pair_data.pos_answer_text,
                    tokenizer, pair_data.pos_prompt_len
                )
                neg_dict[layer_name] = extract_activation(
                    strategy, neg_hs, pair_data.neg_answer_text,
                    tokenizer, pair_data.neg_prompt_len
                )
            
            cached.pair_activations.append((pos_dict, neg_dict))
        
        cached.num_pairs = len(cached.pair_activations)
        return cached
    
    def get_available_layers(self) -> List[str]:
        """Get list of available layer names."""
        if not self.pairs:
            return []
        return list(self.pairs[0].pos_hidden_states.keys())


class RawActivationCache:
    """
    Disk-backed cache for raw hidden states.
    
    Saves/loads per (model, benchmark, text_family) tuple.
    Allows extracting any strategy in the family without re-running model.
    """
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, RawCachedActivations] = {}
    
    def _get_cache_key(self, model_name: str, benchmark: str, text_family: str) -> str:
        """Generate a unique cache key."""
        key_str = f"{model_name}_{benchmark}_{text_family}_raw"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path for a cache file."""
        return self.cache_dir / f"{cache_key}.pt"
    
    def _get_metadata_path(self, cache_key: str) -> Path:
        """Get path for cache metadata."""
        return self.cache_dir / f"{cache_key}_meta.json"
    
    def has(self, model_name: str, benchmark: str, text_family: str) -> bool:
        """Check if raw activations are cached."""
        key = self._get_cache_key(model_name, benchmark, text_family)
        if key in self._memory_cache:
            return True
        return self._get_cache_path(key).exists()
    
    def has_for_strategy(self, model_name: str, benchmark: str, strategy: ExtractionStrategy) -> bool:
        """Check if cache exists for a strategy's text family."""
        text_family = get_strategy_text_family(strategy)
        return self.has(model_name, benchmark, text_family)
    
    def get(
        self,
        model_name: str,
        benchmark: str,
        text_family: str,
        load_to_memory: bool = True,
    ) -> Optional[RawCachedActivations]:
        """Get cached raw activations if they exist."""
        key = self._get_cache_key(model_name, benchmark, text_family)
        
        # Check memory cache
        if key in self._memory_cache:
            return self._memory_cache[key]
        
        # Check disk cache
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None
        
        # Load from disk
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        
        cached = RawCachedActivations(
            benchmark=data["benchmark"],
            text_family=data["text_family"],
            model_name=data["model_name"],
            num_layers=data["num_layers"],
            hidden_size=data["hidden_size"],
        )
        
        # Reconstruct pairs
        for pair_data in data["pairs"]:
            cached.pairs.append(RawPairData(
                pos_hidden_states=pair_data["pos_hidden_states"],
                neg_hidden_states=pair_data["neg_hidden_states"],
                pos_answer_text=pair_data["pos_answer_text"],
                neg_answer_text=pair_data["neg_answer_text"],
                pos_prompt_len=pair_data["pos_prompt_len"],
                neg_prompt_len=pair_data["neg_prompt_len"],
            ))
        cached.num_pairs = len(cached.pairs)
        
        if load_to_memory:
            self._memory_cache[key] = cached
        
        return cached
    
    def get_for_strategy(
        self,
        model_name: str,
        benchmark: str,
        strategy: ExtractionStrategy,
        load_to_memory: bool = True,
    ) -> Optional[RawCachedActivations]:
        """Get cached raw activations for a strategy's text family."""
        text_family = get_strategy_text_family(strategy)
        return self.get(model_name, benchmark, text_family, load_to_memory)
    
    def put(
        self,
        cached: RawCachedActivations,
        save_to_disk: bool = True,
    ) -> None:
        """Store raw cached activations."""
        key = self._get_cache_key(cached.model_name, cached.benchmark, cached.text_family)
        
        # Store in memory
        self._memory_cache[key] = cached
        
        if save_to_disk:
            # Serialize pairs
            pairs_data = []
            for pair in cached.pairs:
                pairs_data.append({
                    "pos_hidden_states": pair.pos_hidden_states,
                    "neg_hidden_states": pair.neg_hidden_states,
                    "pos_answer_text": pair.pos_answer_text,
                    "neg_answer_text": pair.neg_answer_text,
                    "pos_prompt_len": pair.pos_prompt_len,
                    "neg_prompt_len": pair.neg_prompt_len,
                })
            
            data = {
                "benchmark": cached.benchmark,
                "text_family": cached.text_family,
                "model_name": cached.model_name,
                "num_layers": cached.num_layers,
                "hidden_size": cached.hidden_size,
                "num_pairs": cached.num_pairs,
                "pairs": pairs_data,
            }
            torch.save(data, self._get_cache_path(key))
            
            # Save metadata
            metadata = {
                "benchmark": cached.benchmark,
                "text_family": cached.text_family,
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


def collect_and_cache_raw_activations(
    model: "WisentModel",
    pairs: List[ContrastivePair],
    benchmark: str,
    strategy: ExtractionStrategy,
    cache: Optional[RawActivationCache] = None,
    cache_dir: Optional[str] = None,
    show_progress: bool = True,
) -> RawCachedActivations:
    """
    Collect RAW hidden states for all pairs and all layers, then cache.
    
    Unlike collect_and_cache_activations, this stores full sequences
    allowing extraction with any strategy in the same text family.
    
    Args:
        model: WisentModel instance
        pairs: List of contrastive pairs
        benchmark: Benchmark name
        strategy: Extraction strategy (determines text family)
        cache: Optional existing RawActivationCache
        cache_dir: Cache directory (used if cache not provided)
        show_progress: Print progress
        
    Returns:
        RawCachedActivations with full hidden states
    """
    from wisent.core.activations.activations_collector import ActivationCollector
    
    text_family = get_strategy_text_family(strategy)
    
    # Check cache first
    if cache is None and cache_dir:
        cache = RawActivationCache(cache_dir)
    
    if cache and cache.has(model.model_name, benchmark, text_family):
        if show_progress:
            print(f"Loading cached raw activations for {benchmark}/{text_family}")
        return cache.get(model.model_name, benchmark, text_family)
    
    # Collect RAW activations for ALL layers
    collector = ActivationCollector(model=model)
    
    cached = RawCachedActivations(
        benchmark=benchmark,
        text_family=text_family,
        model_name=model.model_name,
        num_layers=model.num_layers,
    )
    
    for i, pair in enumerate(pairs):
        if show_progress and i % 10 == 0:
            print(f"Collecting raw activations: {i+1}/{len(pairs)}", end="\r", flush=True)
        
        # Collect RAW hidden states (full sequences)
        raw_data = collector.collect_raw(pair, strategy=strategy, layers=None)
        cached.add_pair(
            pos_hidden_states=raw_data["pos_hidden_states"],
            neg_hidden_states=raw_data["neg_hidden_states"],
            pos_answer_text=raw_data["pos_answer_text"],
            neg_answer_text=raw_data["neg_answer_text"],
            pos_prompt_len=raw_data["pos_prompt_len"],
            neg_prompt_len=raw_data["neg_prompt_len"],
        )
    
    if show_progress:
        print(f"Collected raw activations: {len(pairs)}/{len(pairs)} pairs, {cached.num_layers} layers")
    
    # Cache the result
    if cache:
        cache.put(cached)
        if show_progress:
            print(f"Cached raw to {cache.cache_dir}")
    
    return cached


# Type hint for WisentModel (avoid circular import)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wisent.core.models.wisent_model import WisentModel
