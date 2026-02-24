"""Train unified goodness vector from pooled multi-benchmark data."""

from __future__ import annotations

import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch

from wisent.core.constants import BENCHMARK_DISPLAY_LIMIT


def get_checkpoint_dir(output_path: str) -> Path:
    """Get checkpoint directory based on output path."""
    output_dir = Path(output_path).parent
    checkpoint_dir = output_dir / ".unified_goodness_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def save_checkpoint(checkpoint_dir: Path, name: str, data: Any):
    """Save checkpoint data."""
    checkpoint_path = checkpoint_dir / f"{name}.pkl"
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"   💾 Checkpoint saved: {name}")


def load_checkpoint(checkpoint_dir: Path, name: str) -> Optional[Any]:
    """Load checkpoint data if it exists."""
    checkpoint_path = checkpoint_dir / f"{name}.pkl"
    if checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        print(f"   📂 Checkpoint loaded: {name}")
        return data
    return None


def clear_checkpoints(checkpoint_dir: Path):
    """Clear all checkpoints after successful completion."""
    import shutil
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        print(f"   🧹 Checkpoints cleared")


def load_all_benchmarks():
    """
    Load ALL benchmarks from the central registry.
    
    Returns:
        Tuple of (filtered_benchmarks, broken_benchmarks)
    """
    from wisent.core.benchmarks import load_all_benchmarks as _load_all_benchmarks
    return _load_all_benchmarks()


from wisent.core.cli.analysis.training.train_unified_data import collect_pairs_and_train
from wisent.core.cli.analysis.training.train_unified_eval import run_evaluation


def execute_train_unified_goodness(args):
    """
    Execute the train-unified-goodness command.

    Pipeline:
    1. Load ALL 327 benchmarks (same as test_all_benchmarks.py)
    2. Generate contrastive pairs from ALL selected benchmarks (pooled)
    3. Collect activations for all pairs
    4. Train single unified steering vector from pooled data
    5. Evaluate vector across ALL benchmarks (pooled evaluation)
    """
    # Expand task if it's a skill or risk name
    from wisent.core.tasks.base.task_selector import expand_task_if_skill_or_risk
    if args.task:
        args.task = expand_task_if_skill_or_risk(args.task)
    
    from wisent.core.data_loaders.loaders.lm_eval.lm_loader import LMEvalDataLoader
    from wisent.core.models.wisent_model import WisentModel
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations import ExtractionStrategy
    
    from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
    from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import lm_build_contrastive_pairs
    from wisent.core.steering_methods.methods.caa import CAAMethod
    from wisent.core.evaluators.rotator import EvaluatorRotator
    from wisent.core.models import get_generate_kwargs

    pipeline_start = time.time() if args.timing else None

    print("\n" + "=" * 70)
    print("🎯 UNIFIED GOODNESS VECTOR TRAINING")
    print("=" * 70)
    print("Training a single steering vector that improves performance")
    print("across ALL benchmarks simultaneously.")
    print("=" * 70 + "\n")

    # Setup checkpointing
    checkpoint_dir = get_checkpoint_dir(args.output)

    # =========================================================================
    # Step 1: Select benchmarks - use ALL 327 benchmarks by default
    # =========================================================================
    print("📋 Step 1/5: Selecting benchmarks...")

    # Load ALL benchmarks from parameter files (same as test_all_benchmarks.py)
    all_benchmark_names, broken_benchmarks = load_all_benchmarks()
    print(f"   ✓ Loaded {len(all_benchmark_names)} total benchmarks")
    if broken_benchmarks:
        print(f"   ✓ Skipping {len(broken_benchmarks)} broken benchmarks")

    # Determine benchmarks from --task argument
    if args.task:
        # Parse comma-separated benchmarks
        task_benchmarks = [b.strip() for b in args.task.split(",")]
        selected_benchmark_names = [name for name in task_benchmarks if name in all_benchmark_names]
        unknown = [name for name in task_benchmarks if name not in all_benchmark_names]
        for name in unknown:
            print(f"   ⚠️  Unknown benchmark: {name}, skipping")
        print(f"   ✓ Using specified benchmarks: {', '.join(selected_benchmark_names)}")
    else:
        # Use ALL benchmarks by default
        selected_benchmark_names = all_benchmark_names.copy()

    # Apply exclusions
    if args.exclude_benchmarks:
        for name in args.exclude_benchmarks:
            if name in selected_benchmark_names:
                selected_benchmark_names.remove(name)
                if args.verbose:
                    print(f"   Excluded: {name}")

    # Apply max limit
    if args.max_benchmarks and len(selected_benchmark_names) > args.max_benchmarks:
        selected_benchmark_names = selected_benchmark_names[:args.max_benchmarks]

    print(f"   ✓ Selected {len(selected_benchmark_names)} benchmarks for training")
    if args.verbose:
        for name in selected_benchmark_names[:BENCHMARK_DISPLAY_LIMIT]:
            print(f"      • {name}")
        if len(selected_benchmark_names) > BENCHMARK_DISPLAY_LIMIT:
            print(f"      ... and {len(selected_benchmark_names) - BENCHMARK_DISPLAY_LIMIT} more")

    # =========================================================================
    # Step 2: Load model
    # =========================================================================
    print(f"\n🤖 Step 2/5: Loading model '{args.model}'...")
    model = WisentModel(args.model, device=args.device)
    print(f"   ✓ Model loaded with {model.num_layers} layers")
    print(f"   ✓ Hidden size: {model.hidden_size}")

    # Determine layer(s) to use
    if args.layer is not None:
        layers = [str(args.layer)]
    elif args.layers:
        # Parse layer specification
        layers = []
        for part in args.layers.replace(" ", "").split(","):
            if "-" in part or ".." in part:
                a, b = part.replace("..", "-").split("-")
                layers.extend(str(i) for i in range(int(a), int(b) + 1))
            else:
                layers.append(part)
    else:
        # Use ALL layers by default
        layers = [str(i) for i in range(model.num_layers)]
        print(f"   Using ALL layers: 0 to {model.num_layers - 1}")

    print(f"   ✓ Target layers: {layers}")

    # =========================================================================
    # Step 3: Collect contrastive pairs from ALL benchmarks (POOLED)
    # =========================================================================
    print(f"\n📊 Step 3/5: Collecting contrastive pairs from all benchmarks...")

    # Steps 3-4: Collect pairs and train
    all_layer_vectors, train_pairs, eval_pairs, benchmarks_used = collect_pairs_and_train(
        args, wisent_model, layers, checkpoint_dir, benchmarks, loader,
    )

    # Step 5: Evaluation + Summary
    run_evaluation(
        args, wisent_model, all_layer_vectors, train_pairs,
        eval_pairs, benchmarks_used, checkpoint_dir,
    )
