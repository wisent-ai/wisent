"""Cone and geometry analysis for diagnose-vectors."""
from typing import Dict

import torch
from wisent.core import constants as _C
from wisent.core.constants import CONE_THRESHOLD, CONE_DIRECTIONS, DIAG_NUM_COMPONENTS, MAX_CLUSTERS, MANIFOLD_NEIGHBORS, DEFAULT_OPTIMIZATION_STEPS


def _run_cone_analysis(
    activations_file: str, 
    verbose: bool = False,
    cone_threshold: float = CONE_THRESHOLD,
    cone_directions: int = CONE_DIRECTIONS,
):
    """Run cone structure analysis on activations."""
    from wisent.core.contrastive_pairs.diagnostics.control_vectors import (
        check_cone_structure,
        ConeAnalysisConfig,
    )
    
    print(f"\n🔺 Cone Structure Analysis")
    print(f"   Activations file: {activations_file}")
    
    try:
        if not os.path.exists(activations_file):
            print(f"   ❌ Activations file not found: {activations_file}")
            return
        
        # Load activations (supports .pt or .json)
        if activations_file.endswith('.pt'):
            activations_data = torch.load(activations_file, weights_only=True)
        else:
            with open(activations_file, 'r') as f:
                activations_data = json.load(f)
        
        # Extract positive and negative activations
        pos_acts = None
        neg_acts = None
        
        if isinstance(activations_data, dict):
            # Format 1: {"positive": [...], "negative": [...]}
            if 'positive' in activations_data and 'negative' in activations_data:
                pos_acts = activations_data['positive']
                neg_acts = activations_data['negative']
            # Format 2: {"pos_activations": [...], "neg_activations": [...]}
            elif 'pos_activations' in activations_data and 'neg_activations' in activations_data:
                pos_acts = activations_data['pos_activations']
                neg_acts = activations_data['neg_activations']
            # Format 3: Per-layer format {"layer_15": {"pos": [...], "neg": [...]}}
            elif any(k.startswith('layer_') for k in activations_data.keys()):
                # Use the first layer found
                for key, layer_data in activations_data.items():
                    if isinstance(layer_data, dict) and 'pos' in layer_data and 'neg' in layer_data:
                        pos_acts = layer_data['pos']
                        neg_acts = layer_data['neg']
                        print(f"   Using layer: {key}")
                        break
        
        if pos_acts is None or neg_acts is None:
            print(f"   ❌ Could not find positive/negative activations in file")
            print(f"   Expected format: {{'positive': [...], 'negative': [...]}}")
            return
        
        # Convert to tensors if needed
        dtype = preferred_dtype()
        if not isinstance(pos_acts, torch.Tensor):
            pos_acts = torch.tensor(pos_acts, dtype=dtype)
        if not isinstance(neg_acts, torch.Tensor):
            neg_acts = torch.tensor(neg_acts, dtype=dtype)
        
        print(f"   Positive samples: {pos_acts.shape[0]}")
        print(f"   Negative samples: {neg_acts.shape[0]}")
        print(f"   Hidden dimension: {pos_acts.shape[1]}")
        
        # Run cone analysis
        config = ConeAnalysisConfig(
            num_directions=cone_directions,
            optimization_steps=DEFAULT_OPTIMIZATION_STEPS,
            cone_threshold=cone_threshold,
        )
        
        print(f"\n   Running cone analysis...")
        result = check_cone_structure(pos_acts, neg_acts, config)
        
        # Display results
        print(f"\n📊 Cone Analysis Results:")
        
        if result.has_cone_structure:
            print(f"   ✅ CONE STRUCTURE DETECTED")
        else:
            print(f"   ❌ No cone structure (linear subspace is sufficient)")
        
        print(f"\n   Cone Score: {result.cone_score:.3f} (threshold: {config.cone_threshold})")
        print(f"   PCA Explained Variance: {result.pca_explained_variance:.3f}")
        print(f"   Cone Explained Variance: {result.cone_explained_variance:.3f}")
        print(f"   Half-Space Consistency: {result.half_space_consistency:.3f}")
        print(f"   Avg Cosine Similarity: {result.avg_cosine_similarity:.3f}")
        print(f"   Positive Combination Score: {result.positive_combination_score:.3f}")
        print(f"   Directions Found: {result.num_directions_found}")
        
        # Separation scores
        print(f"\n   Per-Direction Separation Scores:")
        for i, score in enumerate(result.separation_scores):
            significance = "***" if abs(score) > _C.DIAG_SIGNIFICANCE_STRONG else "**" if abs(score) > _C.DIAG_SIGNIFICANCE_MODERATE else "*" if abs(score) > _C.DIAG_SIGNIFICANCE_WEAK else ""
            print(f"      Direction {i}: {score:.4f} {significance}")
        
        # Interpretation
        print(f"\n📝 Interpretation:")
        if result.has_cone_structure:
            print(f"   - Multiple directions mediate this behavior")
            print(f"   - Consider using TECZA for multi-directional steering")
            print(f"   - CAA may capture only partial behavior")
        else:
            print(f"   - Single direction (CAA) is sufficient")
            print(f"   - Behavior is well-represented by linear subspace")
        
        if verbose:
            print(f"\n   Cosine Similarity Matrix:")
            for i, row in enumerate(result.direction_cosine_similarities):
                print(f"      {[f'{x:.2f}' for x in row]}")
        
    except Exception as e:
        print(f"   ❌ Cone analysis failed: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()


def _run_geometry_analysis(
    activations_file: str,
    verbose: bool = False,
    num_components: int = DIAG_NUM_COMPONENTS,
    max_clusters: int = MAX_CLUSTERS,
    manifold_neighbors: int = MANIFOLD_NEIGHBORS,
):
    """Run comprehensive geometry structure analysis on activations."""
    from wisent.core.contrastive_pairs.diagnostics.control_vectors import (
        detect_geometry_structure,
        GeometryAnalysisConfig,
    )
    
    print(f"\n🔷 Comprehensive Geometry Analysis")
    print(f"   Activations file: {activations_file}")
    
    try:
        if not os.path.exists(activations_file):
            print(f"   ❌ Activations file not found: {activations_file}")
            return
        
        # Load activations (supports .pt or .json)
        if activations_file.endswith('.pt'):
            activations_data = torch.load(activations_file, weights_only=True)
        else:
            with open(activations_file, 'r') as f:
                activations_data = json.load(f)
        
        # Extract positive and negative activations
        pos_acts = None
        neg_acts = None
        
        if isinstance(activations_data, dict):
            if 'positive' in activations_data and 'negative' in activations_data:
                pos_acts = activations_data['positive']
                neg_acts = activations_data['negative']
            elif 'pos_activations' in activations_data and 'neg_activations' in activations_data:
                pos_acts = activations_data['pos_activations']
                neg_acts = activations_data['neg_activations']
            elif any(k.startswith('layer_') for k in activations_data.keys()):
                for key, layer_data in activations_data.items():
                    if isinstance(layer_data, dict) and 'pos' in layer_data and 'neg' in layer_data:
                        pos_acts = layer_data['pos']
                        neg_acts = layer_data['neg']
                        print(f"   Using layer: {key}")
                        break
        
        if pos_acts is None or neg_acts is None:
            print(f"   ❌ Could not find positive/negative activations in file")
            return
        
        # Convert to tensors
        dtype = preferred_dtype()
        if not isinstance(pos_acts, torch.Tensor):
            pos_acts = torch.tensor(pos_acts, dtype=dtype)
        if not isinstance(neg_acts, torch.Tensor):
            neg_acts = torch.tensor(neg_acts, dtype=dtype)
        
        print(f"   Positive samples: {pos_acts.shape[0]}")
        print(f"   Negative samples: {neg_acts.shape[0]}")
        print(f"   Hidden dimension: {pos_acts.shape[1]}")
        
        # Run geometry analysis
        config = GeometryAnalysisConfig(
            num_components=num_components,
            max_clusters=max_clusters,
            manifold_neighbors=manifold_neighbors,
        )
        
        print(f"\n   Running geometry analysis...")
        result = detect_geometry_structure(pos_acts, neg_acts, config)
        
        # Display results
        print(f"\n📊 Geometry Analysis Results:")
        print(f"\n   Best Structure: {result.best_structure.value.upper()}")
        print(f"   Best Score: {result.best_score:.3f}")
        
        print(f"\n   All Structure Scores (ranked):")
        print(f"   {'Structure':<12} {'Score':<8} {'Confidence':<10}")
        print(f"   {'-'*32}")
        
        for name, score in sorted(result.all_scores.items(), key=lambda x: x[1].score, reverse=True):
            marker = "***" if score.score > _C.DIAG_SCORE_HIGH else "**" if score.score > _C.DIAG_SCORE_MODERATE else "*" if score.score > _C.DIAG_SCORE_LOW else ""
            print(f"   {name:<12} {score.score:<8.3f} {score.confidence:<10.3f} {marker}")
        
        print(f"\n📝 Recommendation:")
        print(f"   {result.recommendation}")
        
        if verbose:
            print(f"\n   Detailed Structure Analysis:")
            for name, score in result.all_scores.items():
                if score.details:
                    print(f"\n   {name.upper()}:")
                    for key, value in score.details.items():
                        if isinstance(value, float):
                            print(f"      {key}: {value:.4f}")
                        elif isinstance(value, list) and len(value) <= 10:
                            if all(isinstance(v, float) for v in value):
                                print(f"      {key}: [{', '.join(f'{v:.3f}' for v in value)}]")
                            else:
                                print(f"      {key}: {value}")
                        elif isinstance(value, dict):
                            print(f"      {key}:")
                            for k, v in value.items():
                                print(f"         {k}: {v}")
                        else:
                            print(f"      {key}: {value}")
        
    except Exception as e:
        print(f"   ❌ Geometry analysis failed: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
