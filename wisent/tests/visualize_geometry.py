"""Visualize the geometric structure of activations across multiple configs."""

import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "matplotlib", "scikit-learn"])

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path

from wisent.core.models.wisent_model import WisentModel
from wisent.core.activations.activations_collector import ActivationCollector
from wisent.core.activations.extraction_strategy import ExtractionStrategy
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_extractor_registry import get_extractor
from wisent.core.contrastive_pairs.diagnostics.control_vectors import (
    detect_geometry_structure,
    GeometryAnalysisConfig,
)


# Best configs loaded from comprehensive test results
BEST_CONFIGS = None  # Will be loaded from geometry_analysis_results.json


def load_best_configs(results_file: str = "/home/ubuntu/output/geometry_analysis_results.json", configs_json: str = None):
    """Load best configs from comprehensive geometry analysis results or JSON string."""
    global BEST_CONFIGS
    
    # If JSON string provided, use that
    if configs_json:
        try:
            data = json.loads(configs_json)
            BEST_CONFIGS = {}
            for structure, info in data.items():
                agg = info["token_aggregation"]
                if agg == "final":
                    agg = "last"
                elif agg == "average":
                    agg = "mean"
                BEST_CONFIGS[structure] = {
                    "layer": info["layer"],
                    "aggregation": agg,
                    "prompt": info["prompt_strategy"],
                }
            print(f"Loaded best configs for {len(BEST_CONFIGS)} structures from JSON argument")
            return True
        except Exception as e:
            print(f"ERROR parsing configs JSON: {e}")
            return False
    
    # Otherwise load from file
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        BEST_CONFIGS = {}
        for structure, info in data.get("best_by_geometry", {}).items():
            agg = info["token_aggregation"]
            if agg == "final":
                agg = "last"
            elif agg == "average":
                agg = "mean"
            BEST_CONFIGS[structure] = {
                "layer": info["layer"],
                "aggregation": agg,
                "prompt": info["prompt_strategy"],
            }
        
        print(f"Loaded best configs for {len(BEST_CONFIGS)} structures from {results_file}")
        return True
    except FileNotFoundError:
        print(f"WARNING: {results_file} not found. Run test_geometry_comprehensive.py first.")
        return False
    except Exception as e:
        print(f"ERROR loading configs: {e}")
        return False

AGG_MAP = {
    "first": ExtractionStrategy.CHAT_FIRST,
    "final": ExtractionStrategy.CHAT_LAST,
    "last": ExtractionStrategy.CHAT_LAST,
    "mean": ExtractionStrategy.CHAT_MEAN,
    "average": ExtractionStrategy.CHAT_MEAN,
    "max": ExtractionStrategy.CHAT_MAX_NORM,
    "min": ExtractionStrategy.CHAT_FIRST,  # fallback
    "max_score": ExtractionStrategy.CHAT_MAX_NORM,
}

PROMPT_MAP = {
    "chat_template": ExtractionStrategy.CHAT_LAST,
    "instruction_following": ExtractionStrategy.CHAT_LAST,
    "direct_completion": ExtractionStrategy.CHAT_LAST,
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="truthfulqa_gen")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--num-pairs", type=int, default=50)
    parser.add_argument("--layer", type=int, default=None, help="Single layer (overrides multi-config)")
    parser.add_argument("--aggregation", default=None, help="Single aggregation (overrides multi-config)")
    parser.add_argument("--prompt-strategy", default=None, help="Single prompt strategy (overrides multi-config)")
    parser.add_argument("--output", default="geometry_visualization.png")
    parser.add_argument("--multi-config", action="store_true", help="Run all best configs for each structure")
    parser.add_argument("--configs-json", default=None, help="JSON string with best_by_geometry configs")
    args = parser.parse_args()

    print(f"Loading model {args.model}...")
    wisent_model = WisentModel(model_name=args.model)

    print(f"Generating {args.num_pairs} contrastive pairs from {args.task}...")
    extractor = get_extractor(args.task)
    pairs = extractor.extract_contrastive_pairs(limit=args.num_pairs)

    collector = ActivationCollector(model=wisent_model)

    if args.multi_config:
        # Run for each structure's best config
        run_multi_config(args, wisent_model, pairs, collector, args.configs_json)
    else:
        # Single config mode
        layer = args.layer if args.layer is not None else 6
        agg = AGG_MAP.get(args.aggregation, ExtractionStrategy.CHAT_LAST)
        prompt_strat = PROMPT_MAP.get(args.prompt_strategy, ExtractionStrategy.CHAT_LAST)
        
        run_single_config(args, wisent_model, pairs, collector, layer, agg, prompt_strat, args.output)


def run_multi_config(args, wisent_model, pairs, collector, configs_json=None):
    """Run visualization and detection for each structure's best config."""
    
    # Load best configs from comprehensive test results or JSON argument
    if not load_best_configs(configs_json=configs_json):
        print("ERROR: Cannot run multi-config without geometry_analysis_results.json")
        print("Run test_geometry_comprehensive.py first to generate the results file.")
        return
    
    print("\n" + "="*80)
    print("MULTI-CONFIG COMPARISON: Visualization vs Detection at each structure's best config")
    print("="*80)
    
    results = {}
    
    for structure_name, config in BEST_CONFIGS.items():
        layer = config["layer"]
        agg = AGG_MAP[config["aggregation"]]
        prompt_strat = PROMPT_MAP[config["prompt"]]
        
        print(f"\n{'='*60}")
        print(f"Config for {structure_name.upper()}: Layer={layer}, Agg={config['aggregation']}, Prompt={config['prompt']}")
        print("="*60)
        
        output_file = f"geometry_{structure_name}_L{layer}_{config['aggregation']}_{config['prompt']}.png"
        
        viz_metrics, detector_scores = run_single_config(
            args, wisent_model, pairs, collector, layer, agg, prompt_strat, output_file
        )
        
        results[structure_name] = {
            "config": config,
            "visualization": viz_metrics,
            "detection": detector_scores,
        }
    
    # Print comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY: Visualization vs Detection")
    print("="*80)
    print(f"{'Structure':<12} {'Config':<30} {'Viz Cosine':<12} {'Det Cone':<10} {'Det Score':<10}")
    print("-"*80)
    
    for structure_name, data in results.items():
        cfg = data["config"]
        cfg_str = f"L{cfg['layer']}/{cfg['aggregation']}/{cfg['prompt'][:10]}"
        viz_cos = data["visualization"].get("cosine_sim", 0)
        det_cone = data["detection"].get("cone", 0)
        det_best = data["detection"].get(structure_name, 0)
        print(f"{structure_name:<12} {cfg_str:<30} {viz_cos:<12.3f} {det_cone:<10.3f} {det_best:<10.3f}")


def run_single_config(args, wisent_model, pairs, collector, layer, aggregation, prompt_strategy, output_file):
    """Run visualization and detection for a single config."""
    
    print(f"Collecting activations from layer {layer}...")
    
    pos_activations = []
    neg_activations = []
    layer_str = str(layer)
    
    for i, pair in enumerate(pairs):
        updated_pair = collector.collect(
            pair, strategy=aggregation,
        )
        
        pos_vec = updated_pair.positive_response.layers_activations[layer_str].float().cpu().numpy()
        neg_vec = updated_pair.negative_response.layers_activations[layer_str].float().cpu().numpy()
        
        pos_activations.append(pos_vec)
        neg_activations.append(neg_vec)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(pairs)} pairs")

    pos_arr = np.array(pos_activations)
    neg_arr = np.array(neg_activations)
    all_arr = np.vstack([pos_arr, neg_arr])
    
    # Compute difference vectors
    diff_arr = pos_arr - neg_arr
    
    # Compute visualization metrics
    pos_sparsity = (np.abs(pos_arr) < 0.1).mean(axis=1)
    diff_norm = diff_arr / (np.linalg.norm(diff_arr, axis=1, keepdims=True) + 1e-8)
    cos_sim = diff_norm @ diff_norm.T
    cos_sim_flat = cos_sim[np.triu_indices(len(cos_sim), k=1)]
    
    pca = PCA(n_components=2)
    pca.fit(all_arr)
    
    viz_metrics = {
        "sparsity": float(pos_sparsity.mean()),
        "cosine_sim": float(cos_sim_flat.mean()),
        "cosine_std": float(cos_sim_flat.std()),
        "pca_variance": float(pca.explained_variance_ratio_.sum()),
    }
    
    # Run detector
    pos_tensor = torch.from_numpy(pos_arr).float()
    neg_tensor = torch.from_numpy(neg_arr).float()
    
    config = GeometryAnalysisConfig(num_components=5, optimization_steps=50)
    result = detect_geometry_structure(pos_tensor, neg_tensor, config)
    
    detector_scores = {name: score.score for name, score in result.all_scores.items()}
    detector_scores["best"] = result.best_structure.value
    
    # Print comparison
    print(f"\n--- Visualization Metrics ---")
    print(f"  Sparsity: {viz_metrics['sparsity']:.2%}")
    print(f"  Cosine similarity (mean): {viz_metrics['cosine_sim']:.3f}")
    print(f"  Cosine similarity (std): {viz_metrics['cosine_std']:.3f}")
    print(f"  PCA variance (2 PCs): {viz_metrics['pca_variance']:.1%}")
    
    print(f"\n--- Detector Scores ---")
    print(f"  Best structure: {detector_scores['best']}")
    for name, score in sorted(detector_scores.items(), key=lambda x: x[1] if isinstance(x[1], float) else 0, reverse=True):
        if name != "best":
            print(f"  {name}: {score:.3f}")
    
    # Check consistency
    print(f"\n--- Consistency Check ---")
    cone_det = detector_scores.get("cone", 0)
    viz_cos = viz_metrics["cosine_sim"]
    
    # Cone score should correlate with cosine similarity
    if abs(cone_det - viz_cos) < 0.2:
        print(f"  Cone: CONSISTENT (detector={cone_det:.3f}, viz_cosine={viz_cos:.3f})")
    else:
        print(f"  Cone: MISMATCH (detector={cone_det:.3f}, viz_cosine={viz_cos:.3f})")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    agg_name = {v: k for k, v in AGG_MAP.items()}.get(aggregation, str(aggregation))
    fig.suptitle(f'Geometry of {args.task} (Layer {layer}, Agg={agg_name}, {args.num_pairs} pairs)', fontsize=14)

    # 1. PCA of all activations
    pca_full = PCA(n_components=2)
    all_pca = pca_full.fit_transform(all_arr)
    ax = axes[0, 0]
    ax.scatter(all_pca[:len(pos_arr), 0], all_pca[:len(pos_arr), 1], c='blue', alpha=0.6, label='True')
    ax.scatter(all_pca[len(pos_arr):, 0], all_pca[len(pos_arr):, 1], c='red', alpha=0.6, label='False')
    ax.set_title(f'PCA of Activations\n(var explained: {pca_full.explained_variance_ratio_.sum():.1%})')
    ax.legend()
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    # 2. PCA of difference vectors
    pca_diff = PCA(n_components=2)
    diff_pca = pca_diff.fit_transform(diff_arr)
    ax = axes[0, 1]
    ax.scatter(diff_pca[:, 0], diff_pca[:, 1], c='green', alpha=0.6)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.set_title(f'PCA of Difference Vectors\n(var explained: {pca_diff.explained_variance_ratio_.sum():.1%})')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    # 3. t-SNE
    tsne = TSNE(n_components=2, perplexity=min(30, len(all_arr)-1), random_state=42)
    all_tsne = tsne.fit_transform(all_arr)
    ax = axes[0, 2]
    ax.scatter(all_tsne[:len(pos_arr), 0], all_tsne[:len(pos_arr), 1], c='blue', alpha=0.6, label='True')
    ax.scatter(all_tsne[len(pos_arr):, 0], all_tsne[len(pos_arr):, 1], c='red', alpha=0.6, label='False')
    ax.set_title('t-SNE of Activations')
    ax.legend()

    # 4. Sparsity histogram
    ax = axes[1, 0]
    neg_sparsity = (np.abs(neg_arr) < 0.1).mean(axis=1)
    ax.hist(pos_sparsity, bins=20, alpha=0.6, label='True', color='blue')
    ax.hist(neg_sparsity, bins=20, alpha=0.6, label='False', color='red')
    ax.set_title(f'Sparsity Distribution\n(mean: {pos_sparsity.mean():.2%})')
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Count')
    ax.legend()

    # 5. Cosine similarity histogram
    ax = axes[1, 1]
    ax.hist(cos_sim_flat, bins=50, alpha=0.7, color='green')
    ax.axvline(x=cos_sim_flat.mean(), color='red', linestyle='--', label=f'Mean: {cos_sim_flat.mean():.2f}')
    ax.set_title(f'Cosine Sim of Diff Vectors\n(detector cone={cone_det:.2f})')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Count')
    ax.legend()

    # 6. Detector scores bar chart
    ax = axes[1, 2]
    scores = [(k, v) for k, v in detector_scores.items() if k != "best"]
    scores.sort(key=lambda x: x[1], reverse=True)
    names = [s[0] for s in scores]
    values = [s[1] for s in scores]
    colors = ['green' if n == detector_scores['best'] else 'gray' for n in names]
    ax.barh(names, values, color=colors)
    ax.set_title(f'Detector Scores\n(best: {detector_scores["best"]})')
    ax.set_xlabel('Score')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_file}")
    
    return viz_metrics, detector_scores


if __name__ == "__main__":
    main()
