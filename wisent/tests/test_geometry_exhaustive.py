"""
Exhaustive layer combination analysis.

Tests all 2^N - 1 layer combinations to find optimal layer subsets
for geometry detection.

Uses CLI commands for pair generation and activation extraction.

===============================================================================
DEBUGGING NOTES - READ BEFORE MAKING ASSUMPTIONS
===============================================================================

On Dec 15, 2025, a Qwen3-8B run (36 layers = 68 billion combinations) became
unresponsive after starting step [5]. The instance lost SSM connection, SSH
timed out, and required a reboot.

WHAT WE KNOW (facts with evidence):
- Step [5] started: "Running exhaustive analysis (68719476735 combinations)..."
- No further output after that line
- Instance became unreachable (SSM ConnectionLost, SSH timeout)
- After reboot, dmesg.0 showed NO OOM messages
- kern.log had no errors between 18:30 (step 5 start) and 19:58 (reboot)

WHAT WE DO NOT KNOW (no evidence):
- Whether the process was running or stuck
- Whether memory was exhausted (no OOM in logs)
- Whether CPU was pegged
- The actual cause of unresponsiveness

DO NOT ASSUME:
- That 68 billion combinations is "too many" without measuring
- That the list allocation caused OOM (no evidence)
- That the loop is slow (no benchmarks)
- ANY root cause without actual evidence from logs/metrics

If investigating future failures:
1. Check dmesg BEFORE rebooting for OOM messages
2. Check /var/log/kern.log for errors
3. Try to SSH and run 'top', 'free -h', 'ps aux' before assuming crash
4. Get actual memory/CPU metrics, don't guess

The instance may have been working fine but just not producing output.
===============================================================================
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
        
        last_report = [0, time.time()]  # [last_count, last_time]
        def progress_callback(current: int, total: int):
            # Report every 10000 combinations OR every 30 seconds, whichever comes first
            now = time.time()
            if current - last_report[0] >= 10000 or now - last_report[1] >= 30:
                elapsed = now - start
                rate = current / elapsed if elapsed > 0 else 0
                remaining = (total - current) / rate if rate > 0 else float('inf')
                pct = 100 * current / total
                print(f"    Progress: {current:,}/{total:,} ({pct:.4f}%) - {rate:.1f} combos/sec - ETA: {remaining:.0f}s")
                last_report[0] = current
                last_report[1] = now
        
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


def run_limited_layer_analysis(
    task: str = "truthfulqa_gen",
    model: str = "meta-llama/Llama-3.2-1B-Instruct",
    num_pairs: int = 50,
    max_combo_size: int = 3,
    output_dir: str = "/home/ubuntu/output",
):
    """
    Run limited layer combination analysis.
    
    Tests 1-layer, 2-layer, 3-layer combinations plus all layers combined.
    Much faster than exhaustive: O(N^3) instead of O(2^N).
    
    For 36 layers with max_combo_size=3:
    - 36 + 630 + 7140 + 1 = 7,807 combinations (vs 68 billion exhaustive)
    """
    from wisent.core.contrastive_pairs.diagnostics.control_vectors import (
        detect_geometry_limited,
    )
    from math import comb
    
    sys.stdout.reconfigure(line_buffering=True)
    
    print("=" * 80)
    print("LIMITED LAYER COMBINATION ANALYSIS")
    print("=" * 80)
    print(f"Task: {task}")
    print(f"Model: {model}")
    print(f"Num pairs: {num_pairs}")
    print(f"Max combo size: {max_combo_size}")
    print(f"Output dir: {output_dir}")
    
    # Auto-detect model layer count from config
    print(f"\n[0] Detecting model layer count from config...")
    start = time.time()
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    model_layers = getattr(config, 'num_hidden_layers', None) or \
                   getattr(config, 'n_layer', None) or \
                   getattr(config, 'num_layers', None) or 32
    print(f"    Model has {model_layers} layers (detected in {time.time() - start:.1f}s)")
    
    # Calculate expected combinations
    total_combos = sum(comb(model_layers, r) for r in range(1, min(max_combo_size, model_layers) + 1))
    if max_combo_size < model_layers:
        total_combos += 1  # all layers
    print(f"    Will test {total_combos:,} combinations (1 to {max_combo_size} layers + all {model_layers})")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pairs_file = os.path.join(tmpdir, "pairs.json")
        activations_file = os.path.join(tmpdir, "activations.json")
        
        # Step 1: Generate pairs
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
        
        # Step 2: Get activations for ALL layers
        layers_str = ",".join(str(i) for i in range(1, model_layers + 1))
        
        print(f"\n[2] Extracting activations for layers 1-{model_layers}...")
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
            timeout=1800
        )
        if result.returncode != 0:
            print(f"ERROR: Activation extraction failed: {result.stderr}")
            return
        print(f"    Extracted activations in {time.time() - start:.1f}s")
        
        # Step 3: Load activations
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
                if layer not in pos_by_layer:
                    pos_by_layer[layer] = []
                    neg_by_layer[layer] = []
                
                if layer_key in pos_la and layer_key in neg_la:
                    pos_by_layer[layer].append(torch.tensor(pos_la[layer_key]).reshape(-1))
                    neg_by_layer[layer].append(torch.tensor(neg_la[layer_key]).reshape(-1))
        
        pos_tensors: Dict[int, torch.Tensor] = {}
        neg_tensors: Dict[int, torch.Tensor] = {}
        for layer in sorted(pos_by_layer.keys()):
            if pos_by_layer[layer]:
                pos_tensors[layer] = torch.stack(pos_by_layer[layer])
                neg_tensors[layer] = torch.stack(neg_by_layer[layer])
                print(f"    Layer {layer}: {pos_tensors[layer].shape}")
        
        num_layers = len(pos_tensors)
        print(f"\n    {num_layers} layers available")
        
        # Step 5: Run limited analysis
        print(f"\n[5] Running limited analysis ({total_combos:,} combinations)...")
        start = time.time()
        
        last_report = [0, time.time()]
        def progress_callback(current: int, total: int):
            now = time.time()
            if current - last_report[0] >= 100 or now - last_report[1] >= 30 or current == total:
                elapsed = now - start
                rate = current / elapsed if elapsed > 0 else 0
                remaining = (total - current) / rate if rate > 0 else 0
                pct = 100 * current / total
                print(f"    Progress: {current:,}/{total:,} ({pct:.1f}%) - {rate:.1f} combos/sec - ETA: {remaining:.0f}s")
                last_report[0] = current
                last_report[1] = now
        
        result = detect_geometry_limited(
            pos_tensors,
            neg_tensors,
            max_combo_size=max_combo_size,
            combination_method="concat",
            progress_callback=progress_callback,
        )
        
        elapsed = time.time() - start
        print(f"\n    Completed in {elapsed:.1f}s ({total_combos / elapsed:.1f} combos/sec)")
        
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
            print(f"{i+1}. {layers_str}: {r.best_score:.4f} ({r.best_structure.value})")
        
        print(f"\nRecommendation: {result.recommendation}")
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"geometry_limited_{task}_{timestamp}.json")
        
        results_json = {
            "task": task,
            "model": model,
            "num_pairs": num_pairs,
            "max_combo_size": max_combo_size,
            "total_combinations": result.total_combinations,
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
                    "best_score": r.best_score,
                    "best_structure": r.best_structure.value,
                    "all_scores": r.all_scores,
                }
                for r in result.top_10
            ],
            "top_100": [
                {
                    "layers": list(r.layers),
                    "best_score": r.best_score,
                    "best_structure": r.best_structure.value,
                }
                for r in result.all_results[:100]
            ],
            "patterns": result.patterns,
            "recommendation": result.recommendation,
        }
        
        with open(output_file, "w") as f:
            json.dump(results_json, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        
        return result


def run_contiguous_layer_analysis(
    task: str = "truthfulqa_gen",
    model: str = "meta-llama/Llama-3.2-1B-Instruct",
    num_pairs: int = 50,
    output_dir: str = "/home/ubuntu/output",
):
    """
    Run contiguous layer combination analysis.
    
    Only tests combinations where layers are adjacent (1-2, 2-3, 1-5, etc.).
    Very fast: O(N^2) = N*(N+1)/2 combinations.
    
    For 36 layers: 666 combinations
    For 24 layers: 300 combinations
    """
    from wisent.core.contrastive_pairs.diagnostics.control_vectors import (
        detect_geometry_contiguous,
    )
    
    sys.stdout.reconfigure(line_buffering=True)
    
    print("=" * 80)
    print("CONTIGUOUS LAYER COMBINATION ANALYSIS")
    print("=" * 80)
    print(f"Task: {task}")
    print(f"Model: {model}")
    print(f"Num pairs: {num_pairs}")
    print(f"Output dir: {output_dir}")
    
    # Auto-detect model layer count from config
    print(f"\n[0] Detecting model layer count from config...")
    start = time.time()
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    model_layers = getattr(config, 'num_hidden_layers', None) or \
                   getattr(config, 'n_layer', None) or \
                   getattr(config, 'num_layers', None) or 32
    print(f"    Model has {model_layers} layers (detected in {time.time() - start:.1f}s)")
    
    # Calculate expected combinations
    total_combos = model_layers * (model_layers + 1) // 2
    print(f"    Will test {total_combos:,} contiguous combinations")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pairs_file = os.path.join(tmpdir, "pairs.json")
        activations_file = os.path.join(tmpdir, "activations.json")
        
        # Step 1: Generate pairs
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
        
        # Step 2: Get activations for ALL layers
        layers_str = ",".join(str(i) for i in range(1, model_layers + 1))
        
        print(f"\n[2] Extracting activations for layers 1-{model_layers}...")
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
            timeout=1800
        )
        if result.returncode != 0:
            print(f"ERROR: Activation extraction failed: {result.stderr}")
            return
        print(f"    Extracted activations in {time.time() - start:.1f}s")
        
        # Step 3: Load activations
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
                if layer not in pos_by_layer:
                    pos_by_layer[layer] = []
                    neg_by_layer[layer] = []
                
                if layer_key in pos_la and layer_key in neg_la:
                    pos_by_layer[layer].append(torch.tensor(pos_la[layer_key]).reshape(-1))
                    neg_by_layer[layer].append(torch.tensor(neg_la[layer_key]).reshape(-1))
        
        pos_tensors: Dict[int, torch.Tensor] = {}
        neg_tensors: Dict[int, torch.Tensor] = {}
        for layer in sorted(pos_by_layer.keys()):
            if pos_by_layer[layer]:
                pos_tensors[layer] = torch.stack(pos_by_layer[layer])
                neg_tensors[layer] = torch.stack(neg_by_layer[layer])
                print(f"    Layer {layer}: {pos_tensors[layer].shape}")
        
        num_layers = len(pos_tensors)
        print(f"\n    {num_layers} layers available")
        
        # Step 5: Run contiguous analysis
        print(f"\n[5] Running contiguous analysis ({total_combos:,} combinations)...")
        start = time.time()
        
        last_report = [0, time.time()]
        def progress_callback(current: int, total: int):
            now = time.time()
            if current - last_report[0] >= 50 or now - last_report[1] >= 30 or current == total:
                elapsed = now - start
                rate = current / elapsed if elapsed > 0 else 0
                remaining = (total - current) / rate if rate > 0 else 0
                pct = 100 * current / total
                print(f"    Progress: {current:,}/{total:,} ({pct:.1f}%) - {rate:.1f} combos/sec - ETA: {remaining:.0f}s")
                last_report[0] = current
                last_report[1] = now
        
        result = detect_geometry_contiguous(
            pos_tensors,
            neg_tensors,
            combination_method="concat",
            progress_callback=progress_callback,
        )
        
        elapsed = time.time() - start
        print(f"\n    Completed in {elapsed:.1f}s ({total_combos / elapsed:.1f} combos/sec)")
        
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
            layers_str = f"L{r.layers[0]}-L{r.layers[-1]}" if len(r.layers) > 1 else f"L{r.layers[0]}"
            print(f"{i+1}. {layers_str} ({len(r.layers)} layers): {r.best_score:.4f} ({r.best_structure.value})")
        
        print(f"\nRecommendation: {result.recommendation}")
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"geometry_contiguous_{task}_{timestamp}.json")
        
        results_json = {
            "task": task,
            "model": model,
            "num_pairs": num_pairs,
            "mode": "contiguous",
            "total_combinations": result.total_combinations,
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
                    "best_score": r.best_score,
                    "best_structure": r.best_structure.value,
                    "all_scores": r.all_scores,
                }
                for r in result.top_10
            ],
            "top_100": [
                {
                    "layers": list(r.layers),
                    "best_score": r.best_score,
                    "best_structure": r.best_structure.value,
                }
                for r in result.all_results[:100]
            ],
            "patterns": result.patterns,
            "recommendation": result.recommendation,
        }
        
        with open(output_file, "w") as f:
            json.dump(results_json, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        
        return result


TOKEN_AGGREGATIONS = ["final", "average", "first", "max", "min", "max_score"]
PROMPT_STRATEGIES = ["chat_template", "direct_completion", "instruction_following", "multiple_choice", "role_playing"]


def run_smart_layer_analysis(
    task: str = "truthfulqa_gen",
    model: str = "meta-llama/Llama-3.2-1B-Instruct",
    num_pairs: int = 50,
    max_combo_size: int = 3,
    token_aggregation: str = "final",
    prompt_strategy: str = "chat_template",
    output_dir: str = "/home/ubuntu/output",
):
    """
    Run smart layer combination analysis.
    
    Combines contiguous + limited search: tests all contiguous ranges (L1-L5, L3-L8, etc.)
    plus all 1,2,3-layer non-contiguous combinations. Deduplicates overlaps.
    
    For 36 layers: ~7,800 unique combinations
    For 24 layers: ~2,600 unique combinations
    """
    from wisent.core.contrastive_pairs.diagnostics.control_vectors import (
        detect_geometry_smart,
    )
    from math import comb
    
    sys.stdout.reconfigure(line_buffering=True)
    
    print("=" * 80)
    print("SMART LAYER COMBINATION ANALYSIS")
    print("(Contiguous + Limited 1,2,3-layer combinations)")
    print("=" * 80)
    print(f"Task: {task}")
    print(f"Model: {model}")
    print(f"Num pairs: {num_pairs}")
    print(f"Max combo size: {max_combo_size}")
    print(f"Token aggregation: {token_aggregation}")
    print(f"Prompt strategy: {prompt_strategy}")
    print(f"Output dir: {output_dir}")
    
    # Auto-detect model layer count from config
    print(f"\n[0] Detecting model layer count from config...")
    start = time.time()
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    model_layers = getattr(config, 'num_hidden_layers', None) or \
                   getattr(config, 'n_layer', None) or \
                   getattr(config, 'num_layers', None) or 32
    print(f"    Model has {model_layers} layers (detected in {time.time() - start:.1f}s)")
    
    # Calculate expected combinations (estimate, actual will be less due to deduplication)
    contiguous = model_layers * (model_layers + 1) // 2
    limited = sum(comb(model_layers, r) for r in range(1, min(max_combo_size, model_layers) + 1))
    print(f"    Contiguous: {contiguous:,}, Limited 1-{max_combo_size}: {limited:,}")
    print(f"    (Actual will be less due to deduplication)")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pairs_file = os.path.join(tmpdir, "pairs.json")
        activations_file = os.path.join(tmpdir, "activations.json")
        
        # Step 1: Generate pairs
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
        
        # Step 2: Get activations for ALL layers
        layers_str = ",".join(str(i) for i in range(1, model_layers + 1))
        
        print(f"\n[2] Extracting activations for layers 1-{model_layers}...")
        print(f"    Token aggregation: {token_aggregation}, Prompt strategy: {prompt_strategy}")
        start = time.time()
        result = subprocess.run(
            [
                sys.executable, "-m", "wisent.core.main", "get-activations",
                pairs_file,
                "--output", activations_file,
                "--model", model,
                "--layers", layers_str,
                "--token-aggregation", token_aggregation,
                "--prompt-strategy", prompt_strategy,
            ],
            capture_output=True,
            text=True,
            timeout=1800
        )
        if result.returncode != 0:
            print(f"ERROR: Activation extraction failed: {result.stderr}")
            return
        print(f"    Extracted activations in {time.time() - start:.1f}s")
        
        # Step 3: Load activations
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
                if layer not in pos_by_layer:
                    pos_by_layer[layer] = []
                    neg_by_layer[layer] = []
                
                if layer_key in pos_la and layer_key in neg_la:
                    pos_by_layer[layer].append(torch.tensor(pos_la[layer_key]).reshape(-1))
                    neg_by_layer[layer].append(torch.tensor(neg_la[layer_key]).reshape(-1))
        
        pos_tensors: Dict[int, torch.Tensor] = {}
        neg_tensors: Dict[int, torch.Tensor] = {}
        for layer in sorted(pos_by_layer.keys()):
            if pos_by_layer[layer]:
                pos_tensors[layer] = torch.stack(pos_by_layer[layer])
                neg_tensors[layer] = torch.stack(neg_by_layer[layer])
                print(f"    Layer {layer}: {pos_tensors[layer].shape}")
        
        num_layers = len(pos_tensors)
        print(f"\n    {num_layers} layers available")
        
        # Step 5: Run smart analysis
        print(f"\n[5] Running smart analysis...")
        start = time.time()
        
        last_report = [0, time.time()]
        def progress_callback(current: int, total: int):
            now = time.time()
            if current - last_report[0] >= 100 or now - last_report[1] >= 30 or current == total:
                elapsed = now - start
                rate = current / elapsed if elapsed > 0 else 0
                remaining = (total - current) / rate if rate > 0 else 0
                pct = 100 * current / total
                print(f"    Progress: {current:,}/{total:,} ({pct:.1f}%) - {rate:.1f} combos/sec - ETA: {remaining:.0f}s")
                last_report[0] = current
                last_report[1] = now
        
        result = detect_geometry_smart(
            pos_tensors,
            neg_tensors,
            max_combo_size=max_combo_size,
            combination_method="concat",
            progress_callback=progress_callback,
        )
        
        elapsed = time.time() - start
        print(f"\n    Completed in {elapsed:.1f}s ({result.total_combinations / elapsed:.1f} combos/sec)")
        
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
            if len(r.layers) > 1 and r.layers[-1] - r.layers[0] == len(r.layers) - 1:
                # Contiguous
                layers_str = f"L{r.layers[0]}-L{r.layers[-1]}"
            else:
                layers_str = "+".join(f"L{l}" for l in r.layers)
            print(f"{i+1}. {layers_str} ({len(r.layers)} layers): {r.best_score:.4f} ({r.best_structure.value})")
        
        print(f"\nRecommendation: {result.recommendation}")
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"geometry_smart_{task}_{token_aggregation}_{prompt_strategy}_{timestamp}.json")
        
        results_json = {
            "task": task,
            "model": model,
            "num_pairs": num_pairs,
            "mode": "smart",
            "max_combo_size": max_combo_size,
            "token_aggregation": token_aggregation,
            "prompt_strategy": prompt_strategy,
            "total_combinations": result.total_combinations,
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
                    "best_score": r.best_score,
                    "best_structure": r.best_structure.value,
                    "all_scores": r.all_scores,
                }
                for r in result.top_10
            ],
            "top_100": [
                {
                    "layers": list(r.layers),
                    "best_score": r.best_score,
                    "best_structure": r.best_structure.value,
                }
                for r in result.all_results[:100]
            ],
            "patterns": result.patterns,
            "recommendation": result.recommendation,
        }
        
        with open(output_file, "w") as f:
            json.dump(results_json, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        
        return result


def run_comprehensive_sweep(
    task: str = "truthfulqa_gen",
    model: str = "meta-llama/Llama-3.2-1B-Instruct",
    num_pairs: int = 50,
    max_combo_size: int = 3,
    output_dir: str = "/home/ubuntu/output",
):
    """
    Run comprehensive sweep across all token aggregations and prompt strategies.
    
    Tests 6 token aggregations x 5 prompt strategies = 30 configurations,
    each with smart layer combination search.
    """
    sys.stdout.reconfigure(line_buffering=True)
    
    print("=" * 80)
    print("COMPREHENSIVE GEOMETRY SWEEP")
    print("=" * 80)
    print(f"Task: {task}")
    print(f"Model: {model}")
    print(f"Num pairs: {num_pairs}")
    print(f"Token aggregations: {TOKEN_AGGREGATIONS}")
    print(f"Prompt strategies: {PROMPT_STRATEGIES}")
    print(f"Total configurations: {len(TOKEN_AGGREGATIONS) * len(PROMPT_STRATEGIES)}")
    print("=" * 80)
    
    all_results = []
    total_configs = len(TOKEN_AGGREGATIONS) * len(PROMPT_STRATEGIES)
    config_idx = 0
    
    for token_agg in TOKEN_AGGREGATIONS:
        for prompt_strat in PROMPT_STRATEGIES:
            config_idx += 1
            print(f"\n{'='*80}")
            print(f"CONFIG {config_idx}/{total_configs}: {token_agg} + {prompt_strat}")
            print("=" * 80)
            
            try:
                result = run_smart_layer_analysis(
                    task=task,
                    model=model,
                    num_pairs=num_pairs,
                    max_combo_size=max_combo_size,
                    token_aggregation=token_agg,
                    prompt_strategy=prompt_strat,
                    output_dir=output_dir,
                )
                
                if result:
                    all_results.append({
                        "token_aggregation": token_agg,
                        "prompt_strategy": prompt_strat,
                        "best_combination": list(result.best_combination),
                        "best_score": result.best_score,
                        "best_structure": result.best_structure.value,
                        "single_layer_best": result.single_layer_best,
                        "single_layer_best_score": result.single_layer_best_score,
                        "improvement_over_single": result.improvement_over_single,
                    })
            except Exception as e:
                print(f"ERROR in config {token_agg}+{prompt_strat}: {e}")
                all_results.append({
                    "token_aggregation": token_agg,
                    "prompt_strategy": prompt_strat,
                    "error": str(e),
                })
    
    # Save summary
    print("\n" + "=" * 80)
    print("SWEEP SUMMARY")
    print("=" * 80)
    
    # Sort by best_score
    successful = [r for r in all_results if "best_score" in r]
    successful.sort(key=lambda x: x["best_score"], reverse=True)
    
    print(f"\nCompleted {len(successful)}/{total_configs} configurations")
    print("\n--- Top 10 Configurations ---")
    for i, r in enumerate(successful[:10]):
        print(f"{i+1}. {r['token_aggregation']}+{r['prompt_strategy']}: {r['best_score']:.4f} ({r['best_structure']}) - layers {r['best_combination']}")
    
    # Save sweep summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(output_dir, f"geometry_sweep_summary_{task}_{timestamp}.json")
    
    summary = {
        "task": task,
        "model": model,
        "num_pairs": num_pairs,
        "max_combo_size": max_combo_size,
        "token_aggregations": TOKEN_AGGREGATIONS,
        "prompt_strategies": PROMPT_STRATEGIES,
        "total_configurations": total_configs,
        "successful_configurations": len(successful),
        "all_results": all_results,
        "top_10": successful[:10],
    }
    
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSweep summary saved to: {summary_file}")
    
    return summary


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
    parser.add_argument("--sweep", action="store_true",
                        help="Run comprehensive sweep across all token aggregations and prompt strategies")
    parser.add_argument("--smart", action="store_true", default=True,
                        help="Use smart search (contiguous + 1,2,3-layer) - DEFAULT")
    parser.add_argument("--limited", action="store_true",
                        help="Use limited search (1,2,3-layer combos + all layers)")
    parser.add_argument("--contiguous", action="store_true",
                        help="Use contiguous search (adjacent layers only)")
    parser.add_argument("--exhaustive", action="store_true",
                        help="Use exhaustive search (all 2^N combinations) - VERY SLOW")
    parser.add_argument("--max-combo-size", type=int, default=3,
                        help="Max combination size for limited/smart search (default: 3)")
    parser.add_argument("--token-aggregation", default="final", choices=TOKEN_AGGREGATIONS,
                        help="Token aggregation method (default: final)")
    parser.add_argument("--prompt-strategy", default="chat_template", choices=PROMPT_STRATEGIES,
                        help="Prompt construction strategy (default: chat_template)")
    args = parser.parse_args()
    
    # Print loud warning if max_layers is set
    if args.max_layers is not None:
        print("!" * 80)
        print("WARNING: --max-layers is set! This should ONLY be used for debugging.")
        print("For production runs, use a larger instance type instead of capping layers.")
        print("!" * 80)
    
    if args.sweep:
        run_comprehensive_sweep(
            task=args.task,
            model=args.model,
            num_pairs=args.num_pairs,
            max_combo_size=args.max_combo_size,
            output_dir=args.output_dir,
        )
    elif args.exhaustive:
        run_exhaustive_layer_analysis(
            task=args.task,
            model=args.model,
            num_pairs=args.num_pairs,
            max_layers=args.max_layers,
            output_dir=args.output_dir,
        )
    elif args.contiguous:
        run_contiguous_layer_analysis(
            task=args.task,
            model=args.model,
            num_pairs=args.num_pairs,
            output_dir=args.output_dir,
        )
    elif args.limited:
        run_limited_layer_analysis(
            task=args.task,
            model=args.model,
            num_pairs=args.num_pairs,
            max_combo_size=args.max_combo_size,
            output_dir=args.output_dir,
        )
    else:
        # Default: smart search
        run_smart_layer_analysis(
            task=args.task,
            model=args.model,
            num_pairs=args.num_pairs,
            max_combo_size=args.max_combo_size,
            token_aggregation=args.token_aggregation,
            prompt_strategy=args.prompt_strategy,
            output_dir=args.output_dir,
        )
