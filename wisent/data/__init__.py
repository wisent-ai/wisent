"""
Wisent data storage module.

This module provides centralized storage locations for:
- Contrastive pairs (personalization, synthetic, benchmarks)
- Steering vectors
- Evaluation results
"""

from pathlib import Path

# Base data directory
DATA_DIR = Path(__file__).parent

# Contrastive pairs storage
CONTRASTIVE_PAIRS_DIR = DATA_DIR / "contrastive_pairs"
PERSONALIZATION_PAIRS_DIR = CONTRASTIVE_PAIRS_DIR / "personalization"
SYNTHETIC_PAIRS_DIR = CONTRASTIVE_PAIRS_DIR / "synthetic"
BENCHMARK_PAIRS_DIR = CONTRASTIVE_PAIRS_DIR / "benchmarks"

# Personalization trait directories
BRITISH_PAIRS_DIR = PERSONALIZATION_PAIRS_DIR / "british"
EVIL_PAIRS_DIR = PERSONALIZATION_PAIRS_DIR / "evil"
FLIRTY_PAIRS_DIR = PERSONALIZATION_PAIRS_DIR / "flirty"
LEFT_WING_PAIRS_DIR = PERSONALIZATION_PAIRS_DIR / "left_wing"
CUSTOM_PAIRS_DIR = PERSONALIZATION_PAIRS_DIR / "custom"

# Legacy personalization directory
PERSONALIZATION_DIR = DATA_DIR / "personalization"

__all__ = [
    "DATA_DIR",
    "CONTRASTIVE_PAIRS_DIR",
    "PERSONALIZATION_PAIRS_DIR",
    "SYNTHETIC_PAIRS_DIR",
    "BENCHMARK_PAIRS_DIR",
    "BRITISH_PAIRS_DIR",
    "EVIL_PAIRS_DIR",
    "FLIRTY_PAIRS_DIR",
    "LEFT_WING_PAIRS_DIR",
    "CUSTOM_PAIRS_DIR",
    "PERSONALIZATION_DIR",
]
