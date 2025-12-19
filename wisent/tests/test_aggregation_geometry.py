"""
Test how different token aggregation strategies affect geometry detection
using simple positive vs negative word pairs.

Uses CLI commands instead of reimplementing logic.
"""

import subprocess
import sys
import json
import tempfile
import os
import torch

POSITIVE_WORDS = [
    "happy", "joyful", "excited", "wonderful", "amazing",
    "brilliant", "fantastic", "excellent", "delightful", "cheerful",
    "grateful", "optimistic", "peaceful", "loving", "confident",
    "hopeful", "radiant", "thriving", "blessed", "ecstatic"
]

NEGATIVE_WORDS = [
    "sad", "miserable", "depressed", "terrible", "awful",
    "dreadful", "horrible", "disappointing", "gloomy", "sorrowful",
    "hopeless", "pessimistic", "anxious", "hateful", "insecure",
    "fearful", "dark", "suffering", "cursed", "devastated"
]

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
LAYERS = [4, 8, 12, 16]
AGGREGATIONS = ["average", "max", "min", "final", "first"]
PROMPT_STRATEGIES = ["chat_template", "direct_completion", "instruction_following", "multiple_choice", "role_playing"]


def create_pairs_file(output_path):
    """Create contrastive pairs JSON file."""
    pairs = []
    for pos, neg in zip(POSITIVE_WORDS, NEGATIVE_WORDS):
        pairs.append({
            "prompt": "How do you feel?",
            "positive_response": {"model_response": f"I feel {pos} today."},
            "negative_response": {"model_response": f"I feel {neg} today."}
        })
    
    with open(output_path, "w") as f:
        json.dump({"pairs": pairs}, f, indent=2)
    
    return len(pairs)


def run_cli(args):
    """Run a CLI command and return stdout."""
    result = subprocess.run(
        [sys.executable, "-m", "wisent.core.main"] + args,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


def convert_to_pt(json_path, pt_path, layer):
    """Convert enriched pairs JSON to .pt format for geometry detection."""
    with open(json_path) as f:
        data = json.load(f)
    
    pos_acts = []
    neg_acts = []
    layer_key = str(layer)
    
    for pair in data["pairs"]:
        pos_la = pair["positive_response"]["layers_activations"][layer_key]
        neg_la = pair["negative_response"]["layers_activations"][layer_key]
        pos_acts.append(pos_la)
        neg_acts.append(neg_la)
    
    torch.save({
        "positive": torch.tensor(pos_acts),
        "negative": torch.tensor(neg_acts),
    }, pt_path)


def main():
    print("=" * 80)
    print("AGGREGATION STRATEGY vs GEOMETRY STRUCTURE TEST")
    print("=" * 80)
    print(f"Model: {MODEL}")
    print(f"Layers: {LAYERS}")
    print(f"Aggregations: {AGGREGATIONS}")
    print(f"Prompt strategies: {PROMPT_STRATEGIES}")
    print(f"Total configs: {len(LAYERS)} x {len(AGGREGATIONS)} x {len(PROMPT_STRATEGIES)} = {len(LAYERS) * len(AGGREGATIONS) * len(PROMPT_STRATEGIES)}")
    print()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pairs_file = os.path.join(tmpdir, "pairs.json")
        
        # Create pairs file
        num_pairs = create_pairs_file(pairs_file)
        print(f"Created {num_pairs} contrastive pairs")
        print(f"Example: 'I feel happy today.' vs 'I feel sad today.'")
        print()
        
        results = {}
        
        for prompt_strategy in PROMPT_STRATEGIES:
            print(f"\n{'#'*80}")
            print(f"PROMPT STRATEGY: {prompt_strategy}")
            print("#" * 80)
            
            for layer in LAYERS:
                print(f"\n{'='*80}")
                print(f"LAYER {layer}")
                print("=" * 80)
                
                for agg in AGGREGATIONS:
                    acts_file = os.path.join(tmpdir, f"acts_L{layer}_{agg}_{prompt_strategy}.json")
                    vector_file = os.path.join(tmpdir, f"vector_L{layer}_{agg}_{prompt_strategy}.json")
                    pt_file = os.path.join(tmpdir, f"acts_L{layer}_{agg}_{prompt_strategy}.pt")
                    
                    # Step 1: Get activations
                    ret, out, err = run_cli([
                        "get-activations", pairs_file,
                        "--output", acts_file,
                        "--model", MODEL,
                        "--layers", str(layer),
                        "--token-aggregation", agg,
                        "--prompt-strategy", prompt_strategy,
                    ])
                    if ret != 0:
                        print(f"  {agg}: ERROR getting activations - {err}")
                        continue
                    
                    # Step 2: Create steering vector
                    ret, out, err = run_cli([
                        "create-steering-vector", acts_file,
                        "--output", vector_file,
                    ])
                    if ret != 0:
                        print(f"  {agg}: ERROR creating vector - {err}")
                        continue
                    
                    # Step 3: Convert to .pt format
                    convert_to_pt(acts_file, pt_file, layer)
                    
                    # Step 4: Detect geometry
                    ret, out, err = run_cli([
                        "diagnose-vectors", vector_file,
                        "--activations-file", pt_file,
                        "--detect-geometry",
                    ])
                    if ret != 0:
                        print(f"  {agg}: ERROR detecting geometry - {err}")
                        continue
                    
                    # Parse results from output
                    scores = {}
                    best_structure = None
                    best_score = 0
                    
                    for line in out.split("\n"):
                        line = line.strip()
                        if line.startswith("Best Structure:"):
                            best_structure = line.split(":")[1].strip()
                        elif line.startswith("Best Score:"):
                            best_score = float(line.split(":")[1].strip())
                        # Parse score lines like "   linear       0.636    0.800"
                        parts = line.split()
                        if len(parts) >= 2:
                            struct = parts[0].lower()
                            if struct in ["linear", "cone", "cluster", "manifold", "sparse", "bimodal", "orthogonal"]:
                                try:
                                    scores[struct] = float(parts[1])
                                except ValueError:
                                    pass
                    
                    results[(prompt_strategy, layer, agg)] = {
                        "scores": scores,
                        "best_structure": best_structure,
                        "best_score": best_score,
                    }
                    
                    print(f"\n{agg.upper()}:")
                    print(f"  Best: {best_structure} ({best_score:.4f})")
                    if scores:
                        print(f"  All: " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(scores.items())))
        
        # Summary table
        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        print(f"{'Prompt':<20} {'Layer':<8} {'Agg':<10} {'Best':<15} {'Score':<8} {'Linear':<8}")
        print("-" * 80)
        
        for prompt_strategy in PROMPT_STRATEGIES:
            for layer in LAYERS:
                for agg in AGGREGATIONS:
                    r = results.get((prompt_strategy, layer, agg), {})
                    if r:
                        linear_score = r.get("scores", {}).get("linear", 0)
                        print(f"{prompt_strategy:<20} {layer:<8} {agg:<10} {r['best_structure']:<15} {r['best_score']:.4f}   {linear_score:.4f}")
        
        # Best config per prompt strategy
        print("\n" + "=" * 80)
        print("BEST CONFIG PER PROMPT STRATEGY")
        print("=" * 80)
        
        for prompt_strategy in PROMPT_STRATEGIES:
            best_config = None
            best_score = 0
            for layer in LAYERS:
                for agg in AGGREGATIONS:
                    r = results.get((prompt_strategy, layer, agg), {})
                    if r and r.get("best_score", 0) > best_score:
                        best_score = r["best_score"]
                        best_config = (layer, agg, r["best_structure"])
            
            if best_config:
                print(f"{prompt_strategy:<25}: Layer {best_config[0]}, {best_config[1]}, {best_config[2]} = {best_score:.4f}")
        
        # Best config for LINEAR structure specifically
        print("\n" + "=" * 80)
        print("BEST CONFIG FOR LINEAR STRUCTURE")
        print("=" * 80)
        
        linear_results = []
        for key, r in results.items():
            linear_score = r.get("scores", {}).get("linear", 0)
            if linear_score > 0:
                linear_results.append((key, linear_score))
        
        linear_results.sort(key=lambda x: x[1], reverse=True)
        for (prompt_strategy, layer, agg), score in linear_results[:10]:
            print(f"{prompt_strategy:<20} Layer {layer:<3} {agg:<10}: {score:.4f}")


if __name__ == "__main__":
    main()
