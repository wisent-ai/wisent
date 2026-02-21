"""
Disk-backed caches for activations.

RawActivationCache: full hidden state sequences per (model, benchmark, text_family)
ActivationCache: extracted activation vectors per (model, benchmark, strategy)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch

from wisent.core.activations import ExtractionStrategy
from wisent.core.utils import resolve_default_device
from .cached_activations import CachedActivations, get_strategy_text_family
from .raw_cached_activations import RawCachedActivations, RawPairData


class RawActivationCache:
    """Disk-backed cache for raw hidden states."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, RawCachedActivations] = {}

    def _get_cache_key(self, model_name: str, benchmark: str, text_family: str,
                       component: str = "residual_stream") -> str:
        key_str = f"{model_name}_{benchmark}_{text_family}_raw"
        if component != "residual_stream":
            key_str += f"_{component}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.pt"

    def _get_metadata_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}_meta.json"

    def has(self, model_name: str, benchmark: str, text_family: str,
            component: str = "residual_stream") -> bool:
        key = self._get_cache_key(model_name, benchmark, text_family, component)
        if key in self._memory_cache:
            return True
        return self._get_cache_path(key).exists()

    def has_for_strategy(self, model_name: str, benchmark: str, strategy: ExtractionStrategy,
                         component: str = "residual_stream") -> bool:
        text_family = get_strategy_text_family(strategy)
        return self.has(model_name, benchmark, text_family, component)

    def get(
        self, model_name: str, benchmark: str, text_family: str,
        load_to_memory: bool = True, component: str = "residual_stream",
    ) -> Optional[RawCachedActivations]:
        key = self._get_cache_key(model_name, benchmark, text_family, component)

        if key in self._memory_cache:
            return self._memory_cache[key]

        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None

        data = torch.load(cache_path, map_location="cpu", weights_only=False)

        cached = RawCachedActivations(
            benchmark=data["benchmark"],
            text_family=data["text_family"],
            model_name=data["model_name"],
            num_layers=data["num_layers"],
            hidden_size=data["hidden_size"],
        )

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
        self, model_name: str, benchmark: str,
        strategy: ExtractionStrategy, load_to_memory: bool = True,
        component: str = "residual_stream",
    ) -> Optional[RawCachedActivations]:
        text_family = get_strategy_text_family(strategy)
        return self.get(model_name, benchmark, text_family, load_to_memory, component)

    def put(self, cached: RawCachedActivations, save_to_disk: bool = True,
            component: str = "residual_stream") -> None:
        key = self._get_cache_key(cached.model_name, cached.benchmark, cached.text_family, component)
        self._memory_cache[key] = cached

        if save_to_disk:
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
        self._memory_cache.clear()


class ActivationCache:
    """Disk-backed cache for extracted activation vectors."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, CachedActivations] = {}

    def _get_cache_key(self, model_name: str, benchmark: str, strategy: ExtractionStrategy,
                       component: str = "residual_stream") -> str:
        key_str = f"{model_name}_{benchmark}_{strategy.value}"
        if component != "residual_stream":
            key_str += f"_{component}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.pt"

    def _get_metadata_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def has(self, model_name: str, benchmark: str, strategy: ExtractionStrategy,
            component: str = "residual_stream") -> bool:
        key = self._get_cache_key(model_name, benchmark, strategy, component)
        if key in self._memory_cache:
            return True
        return self._get_cache_path(key).exists()

    def get(
        self, model_name: str, benchmark: str,
        strategy: ExtractionStrategy, load_to_memory: bool = True,
        component: str = "residual_stream",
    ) -> Optional[CachedActivations]:
        key = self._get_cache_key(model_name, benchmark, strategy, component)

        if key in self._memory_cache:
            return self._memory_cache[key]

        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None

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

    def put(self, cached: CachedActivations, save_to_disk: bool = True,
            component: str = "residual_stream") -> None:
        key = self._get_cache_key(cached.model_name, cached.benchmark, cached.strategy, component)
        self._memory_cache[key] = cached

        if save_to_disk:
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
        self._memory_cache.clear()

    def list_cached(self) -> List[Dict[str, Any]]:
        result = []
        for meta_path in self.cache_dir.glob("*.json"):
            with open(meta_path) as f:
                result.append(json.load(f))
        return result

    def get_cache_size_bytes(self) -> int:
        total = 0
        for path in self.cache_dir.glob("*.pt"):
            total += path.stat().st_size
        return total
