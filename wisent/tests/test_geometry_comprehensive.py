"""
Comprehensive geometry analysis across the full search space.

Sweeps all layers, token aggregations, and prompt strategies to find
optimal configurations for each geometry type.
"""

import subprocess
import tempfile
import os
import json
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from itertools import product


@dataclass
class GeometryResult:
    layer: int
    token_aggregation: str
    prompt_strategy: str
    num_pairs: int
    scores: Dict[str, float]
    best_structure: str
    best_score: float


def run_comprehensive_geometry_analysis(
    task: str = "truthfulqa_gen",
    model: str = "meta-llama/Llama-3.2-1B-Instruct",
    num_pairs: int = 50,
    output_dir: str = "/home/ubuntu/output",
):
    """
    Run comprehensive geometry analysis across full search space.
    
    Sweeps:
    - All layers (every 2nd layer for efficiency)
    - All token aggregations: average, final, first, max, min
    - All prompt strategies: chat_template, direct_completion, instruction_following
    
    Returns best configuration for each geometry type.
    """
    from wisent.core.models.wisent_model import WisentModel
    from wisent.core.contrastive_pairs.diagnostics import detect_geometry_structure
    from wisent.core.steering_methods.preflight import run_preflight_check
    
    print("=" * 80)
    print("COMPREHENSIVE GEOMETRY ANALYSIS")
    print("=" * 80)
    print(f"Task: {task}")
    print(f"Model: {model}")
    print(f"Num pairs: {num_pairs}")
    print("=" * 80)
    
    # Load model to get layer count
    print("\n[1] Loading model to determine layer count...")
    wisent_model = WisentModel(model)
    num_layers = wisent_model.num_layers
    print(f"    Model has {num_layers} layers")
    
    # Define search space - ALL options
    # ALL layers (1 to num_layers)
    layers_to_test = list(range(1, num_layers + 1))
    # ALL token aggregations
    token_aggregations = ["average", "final", "first", "max", "min", "max_score"]
    # ALL prompt strategies
    prompt_strategies = ["chat_template", "direct_completion", "instruction_following", "multiple_choice", "role_playing"]
    
    print(f"\n[2] Search space:")
    print(f"    Layers: {layers_to_test} ({len(layers_to_test)} layers)")
    print(f"    Token aggregations: {token_aggregations}")
    print(f"    Prompt strategies: {prompt_strategies}")
    
    total_configs = len(layers_to_test) * len(token_aggregations) * len(prompt_strategies)
    print(f"    Total configurations: {total_configs}")
    
    # Generate pairs once (pairs don't depend on layer/aggregation)
    print(f"\n[3] Generating {num_pairs} pairs for {task}...")
    with tempfile.TemporaryDirectory() as tmpdir:
        pairs_file = os.path.join(tmpdir, "pairs.json")
        
        result = subprocess.run(
            [
                "python", "-m", "wisent.core.main", "generate-pairs-from-task",
                task,
                "--output", pairs_file,
                "--limit", str(num_pairs),
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode != 0:
            print(f"ERROR: Pair generation failed: {result.stderr}")
            return
        print(f"    Generated pairs saved to {pairs_file}")
        
        # Store all results
        all_results: List[GeometryResult] = []
        
        # Track best for each geometry type
        best_by_geometry: Dict[str, GeometryResult] = {}
        
        config_num = 0
        
        # Sweep search space
        print(f"\n[4] Running geometry analysis across {total_configs} configurations...")
        print("-" * 80)
        
        for prompt_strategy in prompt_strategies:
            for token_agg in token_aggregations:
                for layer in layers_to_test:
                    config_num += 1
                    
                    activations_file = os.path.join(tmpdir, f"act_l{layer}_{token_agg}_{prompt_strategy}.json")
                    
                    print(f"\n[{config_num}/{total_configs}] Layer={layer}, Agg={token_agg}, Prompt={prompt_strategy}")
                    
                    # Collect activations for this configuration
                    result = subprocess.run(
                        [
                            "python", "-m", "wisent.core.main", "get-activations",
                            pairs_file,
                            "--output", activations_file,
                            "--model", model,
                            "--layers", str(layer),
                            "--token-aggregation", token_agg,
                            "--prompt-strategy", prompt_strategy,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=600
                    )
                    
                    if result.returncode != 0:
                        print(f"    ERROR: {result.stderr[:200]}")
                        continue
                    
                    # Load activations
                    try:
                        with open(activations_file, 'r') as f:
                            data = json.load(f)
                        
                        pairs_list = data.get('pairs', [])
                        
                        # Extract activations
                        pos_acts = []
                        neg_acts = []
                        layer_key = str(layer)
                        
                        for pair in pairs_list:
                            pos_la = pair.get('positive_response', {}).get('layers_activations', {})
                            neg_la = pair.get('negative_response', {}).get('layers_activations', {})
                            if layer_key in pos_la and layer_key in neg_la:
                                pos_acts.append(torch.tensor(pos_la[layer_key]).reshape(-1))
                                neg_acts.append(torch.tensor(neg_la[layer_key]).reshape(-1))
                        
                        if len(pos_acts) < 5:
                            print(f"    SKIP: Only {len(pos_acts)} valid pairs")
                            continue
                        
                        pos_tensor = torch.stack(pos_acts)
                        neg_tensor = torch.stack(neg_acts)
                        
                        # Run geometry detection
                        geo_result = detect_geometry_structure(pos_tensor, neg_tensor)
                        
                        # Store result
                        scores = {k: v.score for k, v in geo_result.all_scores.items()}
                        result_obj = GeometryResult(
                            layer=layer,
                            token_aggregation=token_agg,
                            prompt_strategy=prompt_strategy,
                            num_pairs=len(pos_acts),
                            scores=scores,
                            best_structure=geo_result.best_structure.value,
                            best_score=geo_result.best_score,
                        )
                        all_results.append(result_obj)
                        
                        # Update best for each geometry type
                        for geo_type, score in scores.items():
                            if geo_type not in best_by_geometry or score > best_by_geometry[geo_type].scores[geo_type]:
                                best_by_geometry[geo_type] = result_obj
                        
                        print(f"    Best: {geo_result.best_structure.value} ({geo_result.best_score:.3f})")
                        print(f"    Scores: " + ", ".join(f"{k}={v:.2f}" for k, v in sorted(scores.items())))
                        
                    except Exception as e:
                        print(f"    ERROR: {e}")
                        continue
        
        # Print summary
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        
        print(f"\nTotal configurations tested: {len(all_results)}/{total_configs}")
        
        # Best configuration for each geometry type
        print("\n" + "-" * 80)
        print("BEST CONFIGURATION FOR EACH GEOMETRY TYPE:")
        print("-" * 80)
        
        for geo_type in sorted(best_by_geometry.keys()):
            r = best_by_geometry[geo_type]
            print(f"\n  {geo_type.upper()}:")
            print(f"    Score: {r.scores[geo_type]:.4f}")
            print(f"    Layer: {r.layer}")
            print(f"    Token aggregation: {r.token_aggregation}")
            print(f"    Prompt strategy: {r.prompt_strategy}")
            print(f"    Num pairs: {r.num_pairs}")
        
        # Overall best configurations
        print("\n" + "-" * 80)
        print("TOP 10 CONFIGURATIONS BY BEST SCORE:")
        print("-" * 80)
        
        sorted_results = sorted(all_results, key=lambda x: x.best_score, reverse=True)
        for i, r in enumerate(sorted_results[:10]):
            print(f"\n  #{i+1}: {r.best_structure} = {r.best_score:.4f}")
            print(f"      Layer={r.layer}, Agg={r.token_aggregation}, Prompt={r.prompt_strategy}")
        
        # Pre-flight recommendations
        print("\n" + "-" * 80)
        print("PRE-FLIGHT RECOMMENDATIONS:")
        print("-" * 80)
        
        # Get the overall best config
        if sorted_results:
            best = sorted_results[0]
            print(f"\nBest overall config: Layer={best.layer}, Agg={best.token_aggregation}, Prompt={best.prompt_strategy}")
            print(f"Detected structure: {best.best_structure} ({best.best_score:.3f})")
            
            # Re-run preflight on best config
            best_acts_file = os.path.join(tmpdir, f"act_l{best.layer}_{best.token_aggregation}_{best.prompt_strategy}.json")
            if os.path.exists(best_acts_file):
                with open(best_acts_file, 'r') as f:
                    data = json.load(f)
                pairs_list = data.get('pairs', [])
                pos_acts = []
                neg_acts = []
                layer_key = str(best.layer)
                for pair in pairs_list:
                    pos_la = pair.get('positive_response', {}).get('layers_activations', {})
                    neg_la = pair.get('negative_response', {}).get('layers_activations', {})
                    if layer_key in pos_la and layer_key in neg_la:
                        pos_acts.append(torch.tensor(pos_la[layer_key]).reshape(-1))
                        neg_acts.append(torch.tensor(neg_la[layer_key]).reshape(-1))
                
                if pos_acts:
                    pos_tensor = torch.stack(pos_acts)
                    neg_tensor = torch.stack(neg_acts)
                    
                    print("\nSteering method compatibility:")
                    for method in ["caa", "titan", "prism", "pulse"]:
                        try:
                            check = run_preflight_check(pos_tensor, neg_tensor, method)
                            print(f"  {method.upper()}: {check.compatibility_score:.0%} compatible")
                            if check.warnings:
                                for w in check.warnings[:2]:
                                    print(f"    - {w}")
                        except Exception as e:
                            print(f"  {method.upper()}: Error - {e}")
        
        # Save full results to JSON
        results_file = os.path.join(output_dir, "geometry_analysis_results.json")
        results_data = {
            "task": task,
            "model": model,
            "num_pairs": num_pairs,
            "search_space": {
                "layers": layers_to_test,
                "token_aggregations": token_aggregations,
                "prompt_strategies": prompt_strategies,
            },
            "total_configs": total_configs,
            "configs_tested": len(all_results),
            "best_by_geometry": {
                geo_type: {
                    "score": r.scores[geo_type],
                    "layer": r.layer,
                    "token_aggregation": r.token_aggregation,
                    "prompt_strategy": r.prompt_strategy,
                }
                for geo_type, r in best_by_geometry.items()
            },
            "all_results": [
                {
                    "layer": r.layer,
                    "token_aggregation": r.token_aggregation,
                    "prompt_strategy": r.prompt_strategy,
                    "num_pairs": r.num_pairs,
                    "scores": r.scores,
                    "best_structure": r.best_structure,
                    "best_score": r.best_score,
                }
                for r in all_results
            ],
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\nFull results saved to: {results_file}")
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="truthfulqa_gen")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--num-pairs", type=int, default=50)
    parser.add_argument("--output-dir", default="/home/ubuntu/output")
    args = parser.parse_args()
    
    run_comprehensive_geometry_analysis(
        task=args.task,
        model=args.model,
        num_pairs=args.num_pairs,
        output_dir=args.output_dir,
    )
