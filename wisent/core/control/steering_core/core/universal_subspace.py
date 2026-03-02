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

from wisent.core.utils.cli.cli_logger import setup_logger, bind
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations, LayerName
from wisent.core.utils.config_tools.constants import DEFAULT_VARIANCE_THRESHOLD, UNIVERSAL_SUBSPACE_RANK, MARGINAL_VARIANCE_THRESHOLD

# Backward-compatible aliases
VARIANCE_EXPLAINED_THRESHOLD = DEFAULT_VARIANCE_THRESHOLD

# Re-exports from extracted modules
from wisent.core.control.steering_core._subspace_analysis import (
    SubspaceAnalysisConfig,
    SubspaceAnalysisResult,
    analyze_steering_vector_subspace,
    check_vector_quality,
)
from wisent.core.control.steering_core._subspace_compression import (
    UniversalBasis,
    compute_universal_basis,
    compress_steering_vectors,
    decompress_steering_vectors,
    save_compressed_vectors,
    load_compressed_vectors,
)
from wisent.core.control.steering_core._subspace_directions import (
    explained_variance_analysis,
    compute_optimal_num_directions,
    get_cached_universal_basis,
    initialize_from_universal_basis,
)
from wisent.core.control.steering_core._subspace_validation import (
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
