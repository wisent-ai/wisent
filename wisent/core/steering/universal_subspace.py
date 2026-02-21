"""
Universal Subspace Analysis for Steering Vectors.

Based on "The Universal Weight Subspace Hypothesis" (Kaushik et al., 2025).
See _subspace_*.py files for implementation details.
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
import torch.nn.functional as F

from wisent.core.cli.cli_logger import setup_logger, bind
from wisent.core.activations.core.atoms import LayerActivations, LayerName

# Constants
UNIVERSAL_SUBSPACE_RANK = 16
VARIANCE_EXPLAINED_THRESHOLD = 0.80
MARGINAL_VARIANCE_THRESHOLD = 0.05

# Re-exports from extracted modules
from wisent.core.steering._subspace_analysis import (
    SubspaceAnalysisConfig,
    SubspaceAnalysisResult,
    analyze_steering_vector_subspace,
    check_vector_quality,
)
from wisent.core.steering._subspace_compression import (
    UniversalBasis,
    compute_universal_basis,
    compress_steering_vectors,
    decompress_steering_vectors,
    save_compressed_vectors,
    load_compressed_vectors,
)
from wisent.core.steering._subspace_directions import (
    explained_variance_analysis,
    compute_optimal_num_directions,
    get_cached_universal_basis,
    initialize_from_universal_basis,
)
from wisent.core.steering._subspace_validation import (
    UNIVERSAL_SUBSPACE_THRESHOLDS,
    compute_subspace_alignment,
    verify_subspace_preservation,
    get_recommended_geometry_thresholds,
)

__all__ = [
    "SubspaceAnalysisConfig",
    "SubspaceAnalysisResult",
    "analyze_steering_vector_subspace",
    "check_vector_quality",
    "UniversalBasis",
    "compress_steering_vectors",
    "decompress_steering_vectors",
    "save_compressed_vectors",
    "load_compressed_vectors",
    "compute_optimal_num_directions",
    "explained_variance_analysis",
    "compute_universal_basis",
    "initialize_from_universal_basis",
    "get_cached_universal_basis",
    "verify_subspace_preservation",
    "compute_subspace_alignment",
    "UNIVERSAL_SUBSPACE_THRESHOLDS",
    "get_recommended_geometry_thresholds",
]
