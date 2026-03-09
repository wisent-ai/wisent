"""
Hierarchical steering optimization with guarantees.

Instead of random/Bayesian search over the full space, we search systematically:

Stage 1: Layer Sweep
    - For each method, test ALL layers with fixed strength=1.0
    - Find the best layer per method
    - ~num_layers × num_methods configs

Stage 2: Strength Sweep
    - At best layer, test all strength values
    - Find best strength per method
    - ~num_strengths × num_methods configs

Stage 3: Method-Specific Tuning
    - At best layer+strength, grid search method-specific params
    - CAA/Ostrze: just normalize (2 configs each)
    - MLP: hidden_dim × num_layers × normalize
    - TECZA: num_directions × optimization_steps × normalize
    - TETNO: threshold × temperature × normalize
    - GROM: num_directions × max_alpha × temperature × normalize

This gives FULL coverage of the search space in a tractable way.
"""

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import (
    run_pipeline,
    MethodConfig,
    CAAConfig,
    OstrzeConfig,
    MLPConfig,
    TECZAConfig,
    TETNOConfig,
    GROMConfig,
    OptimizationResult,
)


from wisent.core.utils.cli.optimize_steering.hierarchical_config import (
    HierarchicalResult, HierarchicalConfig,
    count_hierarchical_configs,
)
from wisent.core.utils.cli.optimize_steering.hierarchical_runner import (
    run_hierarchical_optimization,
)


def execute_hierarchical_optimization(args):
    """Execute hierarchical optimization from CLI args."""
    enriched_pairs_file = getattr(args, 'enriched_pairs_file', None)
    task = getattr(args, 'task', None) or "custom"
    methods = getattr(args, 'methods', None) or ["CAA"]

    # Get num_layers
    num_layers = getattr(args, 'num_layers', None)
    if not num_layers and enriched_pairs_file:
        with open(enriched_pairs_file) as f:
            data = json.load(f)
        num_layers = len(data.get("layers", []))
    if not num_layers:
        raise ValueError("num_layers must be specified (via --num-layers or enriched_pairs_file)")

    return run_hierarchical_optimization(
        model=args.model,
        task=task,
        methods=methods,
        num_layers=num_layers,
        min_clusters=getattr(args, 'min_clusters', None),
        limit=None,
        device=getattr(args, 'device', None),
        enriched_pairs_file=enriched_pairs_file,
        output_dir=getattr(args, 'output_dir', './hierarchical_results'),
        verbose=getattr(args, 'verbose', False),
    )
