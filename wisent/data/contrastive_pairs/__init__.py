"""
Contrastive pairs storage module.

Storage structure:
- personalization/   Personality trait pairs (british, evil, flirty, left_wing, custom)
- synthetic/         Other synthetically generated pairs
- benchmarks/        Pairs extracted from evaluation benchmarks

Each trait directory can contain multiple JSON files with pairs.
File naming convention: {trait}_{model}_{timestamp}.json or {trait}_pairs.json
"""

from pathlib import Path

# Base directory for contrastive pairs
PAIRS_DIR = Path(__file__).parent

# Category directories
PERSONALIZATION_DIR = PAIRS_DIR / "personalization"
SYNTHETIC_DIR = PAIRS_DIR / "synthetic"
BENCHMARKS_DIR = PAIRS_DIR / "benchmarks"

# Personalization trait directories
TRAIT_DIRS = {
    "british": PERSONALIZATION_DIR / "british",
    "evil": PERSONALIZATION_DIR / "evil",
    "flirty": PERSONALIZATION_DIR / "flirty",
    "left_wing": PERSONALIZATION_DIR / "left_wing",
    "custom": PERSONALIZATION_DIR / "custom",
}


def get_trait_dir(trait: str) -> Path:
    """Get the directory for a specific personality trait.

    Args:
        trait: The trait name (british, evil, flirty, left_wing, or custom)

    Returns:
        Path to the trait's storage directory

    Raises:
        ValueError: If trait is not recognized
    """
    trait_lower = trait.lower().replace("-", "_").replace(" ", "_")
    if trait_lower in TRAIT_DIRS:
        return TRAIT_DIRS[trait_lower]
    # For custom traits, use the custom directory
    return TRAIT_DIRS["custom"]


def list_available_traits() -> list[str]:
    """List all available personality traits with stored pairs."""
    available = []
    for trait, trait_dir in TRAIT_DIRS.items():
        if trait_dir.exists() and any(trait_dir.glob("*.json")):
            available.append(trait)
    return available


# Import storage functions for convenience
from .storage import (
    save_personalization_pairs,
    load_personalization_pairs,
    save_synthetic_pairs,
    load_synthetic_pairs,
    save_benchmark_pairs,
    load_benchmark_pairs,
    list_stored_pairs,
    get_pair_count,
)

__all__ = [
    # Path constants
    "PAIRS_DIR",
    "PERSONALIZATION_DIR",
    "SYNTHETIC_DIR",
    "BENCHMARKS_DIR",
    "TRAIT_DIRS",
    # Path utilities
    "get_trait_dir",
    "list_available_traits",
    # Storage functions
    "save_personalization_pairs",
    "load_personalization_pairs",
    "save_synthetic_pairs",
    "load_synthetic_pairs",
    "save_benchmark_pairs",
    "load_benchmark_pairs",
    "list_stored_pairs",
    "get_pair_count",
]
