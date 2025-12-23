"""Test that geometry detectors match visualization metrics."""

import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "matplotlib", "scikit-learn"])

import json
import torch
import numpy as np
from pathlib import Path

from wisent.core.models.wisent_model import WisentModel
from wisent.core.activations.activations_collector import ActivationCollector
from wisent.core.activations.extraction_strategy import ExtractionStrategy
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_extractor_registry import get_extractor
from wisent.core.contrastive_pairs.diagnostics.control_vectors import (
    detect_geometry_structure,
    GeometryAnalysisConfig,
)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="truthfulqa_gen")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--num-pairs", type=int, default=50)
    parser.add_argument("--layer", type=int, default=6)
    args = parser.parse_args()

    print(f"Loading model {args.model}...")
    wisent_model = WisentModel(model_name=args.model)

    print(f"Generating {args.num_pairs} contrastive pairs from {args.task}...")
    extractor = get_extractor(args.task)
    pairs = extractor.extract_contrastive_pairs(limit=args.num_pairs)

    print(f"Collecting activations from layer {args.layer}...")
    collector = ActivationCollector(model=wisent_model)
    
    pos_activations = []
    neg_activations = []
    layer_str = str(args.layer)
    
    for i, pair in enumerate(pairs):
        updated_pair = collector.collect(
            pair, strategy=ExtractionStrategy.CHAT_LAST,
        )
        
        pos_vec = updated_pair.positive_response.layers_activations[layer_str].float()
        neg_vec = updated_pair.negative_response.layers_activations[layer_str].float()
        
        pos_activations.append(pos_vec)
        neg_activations.append(neg_vec)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(pairs)} pairs")

    pos_tensor = torch.stack(pos_activations)
    neg_tensor = torch.stack(neg_activations)
    
    # Compute raw visualization metrics
    print("\n" + "="*70)
    print("RAW VISUALIZATION METRICS (ground truth)")
    print("="*70)
    
    pos_np = pos_tensor.numpy()
    neg_np = neg_tensor.numpy()
    diff_np = pos_np - neg_np
    
    # 1. Sparsity
    sparsity = (np.abs(pos_np) < 0.1).mean()
    print(f"Sparsity (fraction < 0.1): {sparsity:.2%}")
    
    # 2. Cosine similarity of difference vectors (for cone detection)
    diff_norm = diff_np / (np.linalg.norm(diff_np, axis=1, keepdims=True) + 1e-8)
    cos_sim = diff_norm @ diff_norm.T
    cos_sim_flat = cos_sim[np.triu_indices(len(cos_sim), k=1)]
    mean_cos_sim = cos_sim_flat.mean()
    print(f"Mean cosine similarity of diff vectors: {mean_cos_sim:.3f}")
    print(f"  -> If high (>0.5): CONE structure")
    print(f"  -> If near zero: ORTHOGONAL structure")
    
    # 3. PCA variance (for linear detection)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(np.vstack([pos_np, neg_np]))
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"PCA variance explained (2 PCs): {var_explained:.1%}")
    print(f"  -> If high (>70%): LINEAR structure")
    
    # Run detector
    print("\n" + "="*70)
    print("DETECTOR SCORES (should match above)")
    print("="*70)
    
    config = GeometryAnalysisConfig(
        num_components=5,
        optimization_steps=50,
    )
    result = detect_geometry_structure(pos_tensor, neg_tensor, config)
    
    print(f"\nBest structure: {result.best_structure.value} (score: {result.best_score:.3f})")
    print(f"\nAll scores:")
    for name, score in sorted(result.all_scores.items(), key=lambda x: x[1].score, reverse=True):
        s = score.score
        details = score.details
        print(f"  {name:12}: {s:.3f}")
        # Print key details for cone and orthogonal
        if name == "cone" and "raw_mean_cosine_similarity" in details:
            print(f"                raw_cosine_sim={details['raw_mean_cosine_similarity']:.3f}")
        elif name == "orthogonal" and "raw_abs_mean_cosine_similarity" in details:
            print(f"                raw_abs_cosine_sim={details['raw_abs_mean_cosine_similarity']:.3f}")
        elif name == "manifold" and "pca_top2_variance" in details:
            print(f"                pca_top2_var={details['pca_top2_variance']:.3f}")
        elif name == "cluster" and "best_silhouette" in details:
            print(f"                silhouette={details['best_silhouette']:.3f}")
    
    # Verify consistency
    print("\n" + "="*70)
    print("CONSISTENCY CHECK")
    print("="*70)
    
    cone_details = result.all_scores.get("cone", {})
    if hasattr(cone_details, 'details') and "raw_mean_cosine_similarity" in cone_details.details:
        detector_cos_sim = cone_details.details["raw_mean_cosine_similarity"]
        diff = abs(detector_cos_sim - mean_cos_sim)
        status = "PASS" if diff < 0.05 else "FAIL"
        print(f"Cone cosine sim: viz={mean_cos_sim:.3f}, detector={detector_cos_sim:.3f}, diff={diff:.3f} [{status}]")
    
    orthogonal_details = result.all_scores.get("orthogonal", {})
    if hasattr(orthogonal_details, 'details') and "raw_abs_mean_cosine_similarity" in orthogonal_details.details:
        detector_abs_cos = orthogonal_details.details["raw_abs_mean_cosine_similarity"]
        viz_abs_cos = np.abs(cos_sim_flat).mean()
        diff = abs(detector_abs_cos - viz_abs_cos)
        status = "PASS" if diff < 0.05 else "FAIL"
        print(f"Orthogonal abs cosine: viz={viz_abs_cos:.3f}, detector={detector_abs_cos:.3f}, diff={diff:.3f} [{status}]")
    
    # Final verdict
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if mean_cos_sim < 0.15:
        expected = "orthogonal"
    elif mean_cos_sim > 0.5:
        expected = "cone"
    elif var_explained > 0.7:
        expected = "linear"
    else:
        expected = "sparse or manifold"
    
    detected = result.best_structure.value
    print(f"Based on raw metrics, expected structure: {expected}")
    print(f"Detector chose: {detected}")
    
    match = (expected == detected) or (expected in detected) or (detected in expected)
    print(f"Match: {'YES' if match else 'NO - investigate further'}")


if __name__ == "__main__":
    main()
