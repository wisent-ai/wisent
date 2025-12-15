"""
Exhaustive layer combination analysis.

Tests all 2^N - 1 layer combinations to find optimal layer subsets
for geometry detection.

Uses CLI commands for pair generation and activation extraction.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import torch
from datetime import datetime
from typing import Dict, List


def run_exhaustive_layer_analysis(
    task: str = "truthfulqa_gen",
    model: str = "meta-llama/Llama-3.2-1B-Instruct",
    num_pairs: int = 50,
    max_layers: int | None = None,
    output_dir: str = "/home/ubuntu/output",
):
    """
    Run exhaustive layer combination analysis.
    
    Tests all 2^N - 1 layer combinations to find which layer subsets
    produce the strongest geometric structure detection.
    
    Uses CLI commands:
    - generate-pairs-from-task: Generate contrastive pairs
    - get-activations: Extract activations for all layers
    
    Automatically detects the model's layer count.
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    WARNING: DO NOT SET max_layers TO REDUCE THE NUMBER OF LAYERS TESTED.
    
    The whole point of this analysis is to test ALL layer combinations.
    If you need to reduce combinations for feasibility:
    1. Use a larger instance (g6e.2xlarge = 64GB, g6e.4xlarge = 128GB, g6e.12xlarge = 384GB)
    2. Wait longer - it's supposed to take hours/days
    3. DO NOT artificially cap layers - that defeats the purpose
    
    max_layers exists ONLY for debugging/testing purposes, NOT for production runs.
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    from wisent.core.contrastive_pairs.diagnostics.control_vectors import (
        detect_geometry_exhaustive,
    )
    
    sys.stdout.reconfigure(line_buffering=True)
    
    print("=" * 80)
    print("EXHAUSTIVE LAYER COMBINATION ANALYSIS")
    print("=" * 80)
    print(f"Task: {task}")
    print(f"Model: {model}")
    print(f"Num pairs: {num_pairs}")
    print(f"Output dir: {output_dir}")
    
    # Auto-detect model layer count from config (without loading weights)
    print(f"\n[0] Detecting model layer count from config...")
    start = time.time()
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    # Different models use different config keys for layer count
    model_layers = getattr(config, 'num_hidden_layers', None) or \
                   getattr(config, 'n_layer', None) or \
                   getattr(config, 'num_layers', None) or 32
    print(f"    Model has {model_layers} layers (detected in {time.time() - start:.1f}s)")
    
    # Determine layers to use
    if max_layers is not None:
        num_layers = min(max_layers, model_layers)
        print(f"    Using {num_layers} layers (limited by --max-layers)")
    else:
        num_layers = model_layers
    
    print(f"    Total combinations to test: {2**num_layers - 1:,}")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pairs_file = os.path.join(tmpdir, "pairs.json")
        activations_file = os.path.join(tmpdir, "activations.json")
        
        # Step 1: Generate pairs using CLI
        print(f"\n[1] Generating {num_pairs} pairs for {task}...")
        start = time.time()
        result = subprocess.run(
            [
                sys.executable, "-m", "wisent.core.main", "generate-pairs-from-task",
                task,
                "--output", pairs_file,
                "--limit", str(num_pairs),
            ],
            capture_output=True,
            text=True,
            timeout=600
        )
        if result.returncode != 0:
            print(f"ERROR: Pair generation failed: {result.stderr}")
            return
        print(f"    Generated pairs in {time.time() - start:.1f}s")
        
        # Step 2: Get activations for ALL layers using CLI
        # Build layers string: "1,2,3,...,num_layers"
        layers_str = ",".join(str(i) for i in range(1, num_layers + 1))
        
        print(f"\n[2] Extracting activations for layers 1-{num_layers}...")
        start = time.time()
        result = subprocess.run(
            [
                sys.executable, "-m", "wisent.core.main", "get-activations",
                pairs_file,
                "--output", activations_file,
                "--model", model,
                "--layers", layers_str,
                "--token-aggregation", "final",
            ],
            capture_output=True,
            text=True,
            timeout=1800  # 30 min timeout for activation extraction
        )
        if result.returncode != 0:
            print(f"ERROR: Activation extraction failed: {result.stderr}")
            return
        print(f"    Extracted activations in {time.time() - start:.1f}s")
        
        # Step 3: Load activations from JSON
        print("\n[3] Loading activations from file...")
        with open(activations_file, 'r') as f:
            data = json.load(f)
        
        pairs_list = data.get('pairs', [])
        print(f"    Loaded {len(pairs_list)} pairs with activations")
        
        # Step 4: Convert to tensors by layer
        print("\n[4] Converting to tensors by layer...")
        pos_by_layer: Dict[int, List[torch.Tensor]] = {}
        neg_by_layer: Dict[int, List[torch.Tensor]] = {}
        
        for pair in pairs_list:
            pos_la = pair.get('positive_response', {}).get('layers_activations', {})
            neg_la = pair.get('negative_response', {}).get('layers_activations', {})
            
            for layer_key in pos_la:
                layer = int(layer_key)
                if max_layers is not None and layer > max_layers:
                    continue
                    
                if layer not in pos_by_layer:
                    pos_by_layer[layer] = []
                    neg_by_layer[layer] = []
                
                if layer_key in pos_la and layer_key in neg_la:
                    pos_by_layer[layer].append(torch.tensor(pos_la[layer_key]).reshape(-1))
                    neg_by_layer[layer].append(torch.tensor(neg_la[layer_key]).reshape(-1))
        
        # Stack into tensors
        pos_tensors = {}
        neg_tensors = {}
        layers_available = sorted(pos_by_layer.keys())
        
        for layer in layers_available:
            if pos_by_layer[layer] and neg_by_layer[layer]:
                pos_tensors[layer] = torch.stack(pos_by_layer[layer])
                neg_tensors[layer] = torch.stack(neg_by_layer[layer])
                print(f"    Layer {layer}: {pos_tensors[layer].shape}")
        
        num_layers = len(pos_tensors)
        actual_combos = 2 ** num_layers - 1
        print(f"\n    {num_layers} layers available -> {actual_combos} combinations to test")
        
        # Step 5: Run exhaustive analysis
        print(f"\n[5] Running exhaustive analysis ({actual_combos} combinations)...")
        start = time.time()
        
        last_progress = [0]  # Use list for closure
        def progress_callback(current: int, total: int):
            pct = int(100 * current / total)
            if pct >= last_progress[0] + 5:  # Report every 5%
                elapsed = time.time() - start
                rate = current / elapsed if elapsed > 0 else 0
                remaining = (total - current) / rate if rate > 0 else 0
                print(f"    Progress: {current}/{total} ({pct}%) - {rate:.1f} combos/sec - ETA: {remaining:.0f}s")
                last_progress[0] = pct
        
        result = detect_geometry_exhaustive(
            pos_tensors,
            neg_tensors,
            max_layers=num_layers,
            combination_method="concat",
            progress_callback=progress_callback,
        )
        
        elapsed = time.time() - start
        print(f"\n    Completed in {elapsed:.1f}s ({actual_combos / elapsed:.1f} combos/sec)")
        
        # Print results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        
        print(f"\nTotal combinations tested: {result.total_combinations}")
        print(f"\nBest combination: {result.best_combination}")
        print(f"Best score: {result.best_score:.4f}")
        print(f"Best structure: {result.best_structure.value}")
        
        print(f"\nBest single layer: L{result.single_layer_best}")
        print(f"Best single layer score: {result.single_layer_best_score:.4f}")
        print(f"Combination beats single: {result.combination_beats_single}")
        print(f"Improvement over single: {result.improvement_over_single:.4f}")
        
        print("\n--- Top 10 Combinations ---")
        for i, r in enumerate(result.top_10):
            layers_str = "+".join(f"L{l}" for l in r.layers)
            print(f"  {i+1}. {layers_str}: {r.best_structure.value} = {r.best_score:.4f}")
        
        print("\n--- Patterns ---")
        print(f"  Most important layers: {result.patterns.get('most_important_layers', [])}")
        print(f"  Optimal combination size: {result.patterns.get('optimal_combination_size', 1)}")
        print(f"  Dominant structure: {result.patterns.get('dominant_structure', 'unknown')}")
        print(f"  Best score by size: {result.patterns.get('best_score_by_size', {})}")
        print(f"  Early vs late ratio: {result.patterns.get('early_vs_late_ratio', 0):.2f}")
        
        print(f"\n--- Recommendation ---")
        print(f"  {result.recommendation}")
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"exhaustive_geometry_{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Convert to serializable format
        results_json = {
            "task": task,
            "model": model,
            "num_pairs": num_pairs,
            "max_layers": num_layers,
            "total_combinations": result.total_combinations,
            "elapsed_seconds": elapsed,
            "best_combination": list(result.best_combination),
            "best_score": result.best_score,
            "best_structure": result.best_structure.value,
            "single_layer_best": result.single_layer_best,
            "single_layer_best_score": result.single_layer_best_score,
            "combination_beats_single": result.combination_beats_single,
            "improvement_over_single": result.improvement_over_single,
            "top_10": [
                {
                    "layers": list(r.layers),
                    "best_structure": r.best_structure.value,
                    "best_score": r.best_score,
                    "all_scores": r.all_scores,
                }
                for r in result.top_10
            ],
            "top_100": [
                {
                    "layers": list(r.layers),
                    "best_structure": r.best_structure.value,
                    "best_score": r.best_score,
                }
                for r in result.all_results[:100]
            ],
            "patterns": {
                k: v if not isinstance(v, float) or not (v != v) else None  # Handle NaN
                for k, v in result.patterns.items()
            },
            "recommendation": result.recommendation,
        }
        
        with open(output_file, "w") as f:
            json.dump(results_json, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        
        return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="truthfulqa_gen")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--num-pairs", type=int, default=50)
    # WARNING: Do NOT use --max-layers in production runs!
    # The whole point of exhaustive analysis is to test ALL layers.
    # If you need more memory, use a larger instance type instead.
    parser.add_argument("--max-layers", type=int, default=None, 
                        help="DEBUG ONLY - DO NOT USE IN PRODUCTION. Use larger instance instead.")
    parser.add_argument("--output-dir", default="/home/ubuntu/output")
    args = parser.parse_args()
    
    # Print loud warning if max_layers is set
    if args.max_layers is not None:
        print("!" * 80)
        print("WARNING: --max-layers is set! This should ONLY be used for debugging.")
        print("For production runs, use a larger instance type instead of capping layers.")
        print("!" * 80)
    
    run_exhaustive_layer_analysis(
        task=args.task,
        model=args.model,
        num_pairs=args.num_pairs,
        max_layers=args.max_layers,
        output_dir=args.output_dir,
    )
