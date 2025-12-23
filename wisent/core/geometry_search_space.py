"""
Configuration for geometry search space.

Defines all parameters to search over when testing if a unified "goodness" 
direction exists across benchmarks.

Strategy:
- Extract activations for ALL layers once per (benchmark, strategy) pair
- Cache activations to disk/memory
- Test all layer combinations from cached activations (fast, just tensor math)
- This reduces extraction time from O(layer_combos) to O(1) per benchmark
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from pathlib import Path
import json

from wisent.core.activations.extraction_strategy import ExtractionStrategy
from wisent.core.utils.layer_combinations import get_layer_combinations
from wisent.core.benchmark_registry import get_all_benchmarks
from wisent.core.activations.activation_cache import ActivationCache, CachedActivations


@dataclass
class GeometrySearchConfig:
    """Configuration for a single geometry search run."""
    
    # Pairs settings
    pairs_per_benchmark: int = 50
    random_seed: int = 42
    
    # Layer settings
    max_layer_combo_size: int = 3
    
    # Caching
    cache_activations: bool = True
    cache_dir: Optional[str] = None
    
    # Estimation
    estimated_time_per_extraction_seconds: float = 120.0  # ~2 min per (benchmark, strategy)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pairs_per_benchmark": self.pairs_per_benchmark,
            "random_seed": self.random_seed,
            "max_layer_combo_size": self.max_layer_combo_size,
            "cache_activations": self.cache_activations,
            "cache_dir": self.cache_dir,
            "estimated_time_per_extraction_seconds": self.estimated_time_per_extraction_seconds,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeometrySearchConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class GeometrySearchSpace:
    """
    Search space configuration for geometry testing.
    
    Combines:
    - Models to test
    - Extraction strategies
    - Layer combinations
    - Benchmarks
    
    With activation caching:
    - Extract ALL layers once per (benchmark, strategy)
    - Test layer combinations from cache (no re-extraction needed)
    """
    
    # Default models to test
    DEFAULT_MODELS = [
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-2-7b-chat-hf",
        "Qwen/Qwen3-8B",
        "openai/gpt-oss-20b",
    ]
    
    # Extraction strategies for instruct models
    INSTRUCT_STRATEGIES = [
        ExtractionStrategy.CHAT_MEAN,
        ExtractionStrategy.CHAT_FIRST,
        ExtractionStrategy.CHAT_LAST,
        ExtractionStrategy.CHAT_MAX_NORM,
        ExtractionStrategy.CHAT_WEIGHTED,
        ExtractionStrategy.ROLE_PLAY,
        ExtractionStrategy.MC_BALANCED,
    ]
    
    # Extraction strategies for base models
    BASE_STRATEGIES = [
        ExtractionStrategy.COMPLETION_LAST,
        ExtractionStrategy.COMPLETION_MEAN,
        ExtractionStrategy.MC_COMPLETION,
    ]
    
    def __init__(
        self,
        models: Optional[List[str]] = None,
        strategies: Optional[List[ExtractionStrategy]] = None,
        benchmarks: Optional[List[str]] = None,
        config: Optional[GeometrySearchConfig] = None,
    ):
        """
        Initialize the search space.
        
        Args:
            models: List of model names to test. Defaults to DEFAULT_MODELS.
            strategies: List of extraction strategies. Defaults to INSTRUCT_STRATEGIES.
            benchmarks: List of benchmarks. Defaults to all available benchmarks.
            config: Search configuration (pairs, caching, etc.)
        """
        self.models = models or self.DEFAULT_MODELS
        self.strategies = strategies or self.INSTRUCT_STRATEGIES
        self.benchmarks = benchmarks or get_all_benchmarks()
        self.config = config or GeometrySearchConfig()
    
    def get_layer_combinations_for_model(self, model_name: str, num_layers: int) -> List[List[int]]:
        """
        Get all layer combinations to test for a given model.
        
        Args:
            model_name: Name of the model
            num_layers: Number of layers in the model
            
        Returns:
            List of layer combinations
        """
        return get_layer_combinations(num_layers, self.config.max_layer_combo_size)
    
    def get_extraction_count(self) -> int:
        """
        Calculate number of activation extractions needed (with caching).
        
        With caching, we extract ALL layers once per (benchmark, strategy).
        Layer combinations are tested from cache without re-extraction.
        
        Returns:
            Number of (benchmark, strategy) pairs = extraction operations
        """
        return len(self.benchmarks) * len(self.strategies)
    
    def get_total_configurations(self, num_layers: int) -> int:
        """
        Calculate total number of configurations to test.
        
        Total = strategies * layer_combos * benchmarks
        (Layer combos are tested from cached activations)
        """
        from wisent.core.utils.layer_combinations import get_layer_combinations_count
        
        layer_combos = get_layer_combinations_count(num_layers, self.config.max_layer_combo_size)
        return len(self.strategies) * layer_combos * len(self.benchmarks)
    
    def estimate_time_hours(self) -> float:
        """
        Estimate total time for geometry search (per model).
        
        With caching:
        - Extract once per (benchmark, strategy) 
        - Layer combo testing is fast (from cache)
        
        Returns:
            Estimated hours per model
        """
        extractions = self.get_extraction_count()
        seconds = extractions * self.config.estimated_time_per_extraction_seconds
        return seconds / 3600
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "models": self.models,
            "strategies": [s.value for s in self.strategies],
            "benchmarks": self.benchmarks,
            "config": self.config.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeometrySearchSpace":
        """Deserialize from dictionary."""
        strategies = [ExtractionStrategy(s) for s in data.get("strategies", [])]
        config = GeometrySearchConfig.from_dict(data.get("config", {}))
        return cls(
            models=data.get("models"),
            strategies=strategies if strategies else None,
            benchmarks=data.get("benchmarks"),
            config=config,
        )
    
    def summary(self) -> str:
        """Return a human-readable summary of the search space."""
        lines = [
            "Geometry Search Space:",
            f"  Models: {len(self.models)}",
            f"  Strategies: {len(self.strategies)}",
            f"  Benchmarks: {len(self.benchmarks)}",
            f"  Pairs per benchmark: {self.config.pairs_per_benchmark}",
            f"  Max layer combo size: {self.config.max_layer_combo_size}",
            f"  Cache activations: {self.config.cache_activations}",
            f"",
            f"  Extractions needed (per model): {self.get_extraction_count()}",
            f"  Estimated time (per model): {self.estimate_time_hours():.1f} hours",
        ]
        return "\n".join(lines)
    
    def save(self, path: str) -> None:
        """Save search space to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "GeometrySearchSpace":
        """Load search space from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# Default search space instance
DEFAULT_SEARCH_SPACE = GeometrySearchSpace()


if __name__ == "__main__":
    # Print summary of default search space
    space = GeometrySearchSpace()
    print(space.summary())
    print()
    
    # Example with 16 layers (Llama-3.2-1B)
    num_layers = 16
    layer_combos = space.get_layer_combinations_for_model("test", num_layers)
    print(f"For a {num_layers}-layer model:")
    print(f"  Layer combinations: {len(layer_combos)}")
    print(f"  Total configs to test: {space.get_total_configurations(num_layers)}")
