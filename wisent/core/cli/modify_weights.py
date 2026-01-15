"""
CLI command for modifying model weights using steering vectors.

This module implements the modify-weights command which permanently modifies
model weights using either directional projection or additive methods.

By default, uses automatic method selection based on geometry analysis
which picks the optimal steering method for the data structure.
"""

import json
import sys
import time
from pathlib import Path
import torch

from wisent.core.utils.device import resolve_default_device
from wisent.core.cli_logger import setup_logger, bind
from wisent.core.models.wisent_model import WisentModel
from wisent.core.weight_modification import (
    project_weights,
    project_with_kernel,
    bake_steering_into_weights,
    bake_steering_with_kernel,
    export_modified_model,
)

_LOG = setup_logger(__name__)


def _auto_select_steering_method(pairs, model, args):
    """
    Automatically select the best steering method based on repscan geometry analysis.
    
    Uses compute_geometry_metrics and compute_recommendation from the geometry module.
    
    Returns:
        tuple: (steering_method, modification_method, metrics)
            - steering_method: 'caa', 'titan', 'pulse', or 'prism'
            - modification_method: 'directional', 'titan', 'pulse', or 'prism'
            - metrics: dict of geometry metrics
    """
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations.extraction_strategy import ExtractionStrategy
    from wisent.core.geometry import (
        compute_geometry_metrics,
        compute_recommendation,
        compute_concept_coherence,
    )
    
    if args.verbose:
        print("\n" + "=" * 60)
        print("🔍 AUTO-SELECTING STEERING METHOD (repscan)")
        print("=" * 60)
        print("   Analyzing activation geometry...")
    
    # Collect activations from a sample of pairs for analysis
    collector = ActivationCollector(model=model)
    sample_pairs = pairs[:min(50, len(pairs))]
    
    # Collect activations at 75% layer (where steering is most effective)
    num_layers = model.num_layers if hasattr(model, 'num_layers') else 36
    analysis_layer = str(int(num_layers * 0.75))
    
    pos_activations = []
    neg_activations = []
    
    for pair in sample_pairs:
        enriched = collector.collect(
            pair, 
            strategy=ExtractionStrategy.CHAT_LAST,
            layers=[analysis_layer]
        )
        
        if enriched.positive_response.layers_activations.get(analysis_layer) is not None:
            pos_activations.append(enriched.positive_response.layers_activations[analysis_layer])
        if enriched.negative_response.layers_activations.get(analysis_layer) is not None:
            neg_activations.append(enriched.negative_response.layers_activations[analysis_layer])
    
    if len(pos_activations) < 10 or len(neg_activations) < 10:
        if args.verbose:
            print("   ⚠️  Insufficient activations for analysis, defaulting to TITAN")
        return "titan", "titan", None
    
    # Stack activations
    pos_tensor = torch.stack(pos_activations)
    neg_tensor = torch.stack(neg_activations)
    
    # Run full repscan geometry analysis
    metrics = compute_geometry_metrics(
        pos_tensor, neg_tensor,
        include_expensive=False,  # Skip expensive metrics for speed
        n_folds=3,
    )
    
    # Get recommendation from repscan
    recommendation = compute_recommendation(metrics)
    recommended_method = recommendation.get("recommended_method", "TITAN").upper()
    confidence = recommendation.get("confidence", 0.5)
    reasoning = recommendation.get("reasoning", "")
    
    # Also compute coherence for more detail
    coherence = compute_concept_coherence(pos_tensor, neg_tensor)
    
    if args.verbose:
        print(f"\n   Repscan Analysis Results:")
        print(f"   ├─ Linear probe accuracy: {metrics.get('linear_probe_accuracy', 0):.3f}")
        print(f"   ├─ Signal strength:       {metrics.get('signal_strength', 0):.3f}")
        print(f"   ├─ Concept coherence:     {coherence:.3f}")
        print(f"   ├─ Steerability score:    {metrics.get('steer_steerability_score', 0):.3f}")
        print(f"   ├─ ICD:                   {metrics.get('icd_icd', 0):.1f}")
        print(f"   └─ Recommendation:        {recommended_method} (confidence={confidence:.2f})")
        print(f"       Reasoning: {reasoning}")
    
    # Map recommendation to steering/modification methods
    if recommended_method == "CAA":
        steering_method = "caa"
        modification_method = "directional"
        reason = f"Repscan recommends CAA (linear_probe={metrics.get('linear_probe_accuracy', 0):.2f})"
    elif recommended_method == "PRISM":
        steering_method = "prism"
        modification_method = "prism"
        reason = f"Repscan recommends PRISM (signal={metrics.get('signal_strength', 0):.2f})"
    else:  # TITAN or unknown
        steering_method = "titan"
        modification_method = "titan"
        reason = f"Repscan recommends TITAN (adaptive steering)"
    
    if args.verbose:
        print(f"\n   ✓ Selected: {steering_method.upper()} / {modification_method}")
        print(f"   └─ Reason: {reason}")
        print("=" * 60 + "\n")
    
    return steering_method, modification_method, metrics


def _generate_pairs_for_titan(args, wisent_model):
    """
    Generate contrastive pairs for TITAN training.
    
    Uses existing contrastive pair generation infrastructure.
    """
    from wisent.core.contrastive_pairs.core.pair import ContrastivePair
    from wisent.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse
    
    pairs = []
    
    if args.task:
        task_lower = args.task.lower()
        
        if task_lower == "personalization" and args.trait:
            # Synthetic pairs from trait
            from wisent.core.contrastive_pairs.generators.llm_synthetic import LLMSyntheticGenerator
            
            generator = LLMSyntheticGenerator(model_id=args.model)
            raw_pairs = generator.generate_pairs(
                trait=args.trait,
                num_pairs=args.num_pairs,
            )
            
            for raw_pair in raw_pairs:
                pair = ContrastivePair(
                    prompt=raw_pair.get('prompt', ''),
                    positive_response=PositiveResponse(model_response=raw_pair.get('positive_response', '')),
                    negative_response=NegativeResponse(model_response=raw_pair.get('negative_response', '')),
                    label=args.trait,
                )
                pairs.append(pair)
                
        else:
            # Task-based generation using lm-eval pairs
            from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import build_contrastive_pairs
            
            raw_pairs = build_contrastive_pairs(
                task_name=args.task,
                limit=args.num_pairs,
            )
            
            for raw_pair in raw_pairs:
                # raw_pair is already a ContrastivePair
                pairs.append(raw_pair)
    
    elif args.trait:
        # Trait-based synthetic generation
        from wisent.core.contrastive_pairs.generators.llm_synthetic import LLMSyntheticGenerator
        
        generator = LLMSyntheticGenerator(model_id=args.model)
        raw_pairs = generator.generate_pairs(
            trait=args.trait,
            num_pairs=args.num_pairs,
        )
        
        for raw_pair in raw_pairs:
            pair = ContrastivePair(
                prompt=raw_pair.get('prompt', ''),
                positive_response=PositiveResponse(model_response=raw_pair.get('positive_response', '')),
                negative_response=NegativeResponse(model_response=raw_pair.get('negative_response', '')),
                label=args.trait,
            )
            pairs.append(pair)
    
    return pairs


def _get_all_layers(model) -> list[str]:
    """Get all layer indices as strings for a model (1-indexed for collector API)."""
    if hasattr(model, 'hf_model'):
        config = model.hf_model.config
    elif hasattr(model, 'config'):
        config = model.config
    else:
        return [str(i) for i in range(1, 37)]  # Default fallback (1-indexed)
    
    num_layers = getattr(config, 'num_hidden_layers', None) or \
                 getattr(config, 'n_layer', None) or \
                 getattr(config, 'num_layers', None) or 36
    
    return [str(i) for i in range(1, num_layers + 1)]  # 1-indexed


def _train_titan_for_task(args, model, pairs):
    """
    Train TITAN on contrastive pairs and return the TITANResult.
    
    Args:
        args: Command arguments
        model: WisentModel instance
        pairs: List of contrastive pairs
        
    Returns:
        TITANResult object
    """
    from wisent.core.steering_methods.methods.titan import TITANMethod
    from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
    
    if args.verbose:
        print("\n🔥 Training TITAN steering method...")
    
    # Create pair set
    pair_set = ContrastivePairSet(
        name=getattr(args, 'trait_label', 'steering'),
        pairs=pairs,
        task_type=args.task if hasattr(args, 'task') else None,
    )
    
    # Collect activations for pairs (required for TITAN)
    if args.verbose:
        print("  Collecting activations for TITAN training...")
    
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations.extraction_strategy import ExtractionStrategy
    
    # Default to all layers if not specified
    if args.layers is None:
        layers = _get_all_layers(model)
    else:
        layers = [str(l) for l in str(args.layers).split(',')]
    strategy = ExtractionStrategy.CHAT_LAST
    
    collector = ActivationCollector(model=model)
    enriched_pairs = []
    for i, pair in enumerate(pair_set.pairs):
        enriched_pair = collector.collect(pair, strategy=strategy, layers=layers)
        enriched_pairs.append(enriched_pair)
        if args.verbose and (i + 1) % 10 == 0:
            print(f"    Collected {i + 1}/{len(pair_set.pairs)} pairs")
    
    # Update pair set with enriched pairs
    pair_set.pairs = enriched_pairs
    if args.verbose:
        print(f"  ✓ Collected activations for {len(enriched_pairs)} pairs")
    
    # Configure TITAN with explicit layer specification
    layer_indices = [int(l) for l in layers]
    titan_method = TITANMethod(
        model=model,
        num_directions=getattr(args, 'titan_num_directions', 8),
        manifold_method="pca",
        steering_layers=layer_indices,  # Use the same layers we collected activations for
        sensor_layer=layer_indices[0],  # Use first layer as sensor
    )
    
    # Train TITAN
    titan_result = titan_method.train_titan(pair_set)
    
    if args.verbose:
        print(f"✓ TITAN trained on {len(pairs)} pairs")
        print(f"  Layers: {len(titan_result.layer_order)}")
        print(f"  Directions per layer: {titan_result.directions[titan_result.layer_order[0]].shape[0]}")
    
    return titan_result


def _train_pulse_for_task(args, wisent_model, pairs):
    """Train PULSE steering for a task."""
    from wisent.core.steering_methods.methods.pulse import PULSEMethod
    from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations.extraction_strategy import ExtractionStrategy
    
    model = wisent_model.hf_model
    # Default to all layers if not specified
    if args.layers:
        layers = args.layers.split(',')
    else:
        layers = _get_all_layers(wisent_model)
    
    # Collect activations for pairs
    if args.verbose:
        print(f"  Collecting activations for PULSE training...")
    
    collector = ActivationCollector(model=wisent_model)
    enriched_pairs = []
    
    for i, pair in enumerate(pairs):
        enriched = collector.collect(pair, strategy=ExtractionStrategy.CHAT_LAST, layers=layers)
        enriched_pairs.append(enriched)
        if args.verbose and (i + 1) % 10 == 0:
            print(f"    Collected {i + 1}/{len(pairs)} pairs")
    
    pair_set = ContrastivePairSet(pairs=enriched_pairs, name="pulse_training")
    
    if args.verbose:
        print(f"  ✓ Collected activations for {len(enriched_pairs)} pairs")
    
    # Configure PULSE with explicit layer specification
    layer_indices = [int(l) for l in layers]
    pulse_method = PULSEMethod(
        model=model,
        steering_layers=layer_indices,
        sensor_layer=layer_indices[0],
    )
    
    # Train PULSE
    pulse_result = pulse_method.train_pulse(pair_set)
    
    if args.verbose:
        print(f"✓ PULSE trained on {len(pairs)} pairs")
        print(f"  Layers: {len(pulse_result.behavior_vectors)}")
        print(f"  Optimal threshold: {pulse_result.optimal_threshold:.3f}")
    
    return pulse_result


def _train_prism_for_task(args, wisent_model, pairs):
    """Train PRISM steering for a task."""
    from wisent.core.steering_methods.methods.prism import PRISMMethod
    from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations.extraction_strategy import ExtractionStrategy
    
    model = wisent_model.hf_model
    # Default to all layers if not specified
    if args.layers:
        layers = args.layers.split(',')
    else:
        layers = _get_all_layers(wisent_model)
    
    # Collect activations for pairs
    if args.verbose:
        print(f"  Collecting activations for PRISM training...")
    
    collector = ActivationCollector(model=wisent_model)
    enriched_pairs = []
    
    for i, pair in enumerate(pairs):
        enriched = collector.collect(pair, strategy=ExtractionStrategy.CHAT_LAST, layers=layers)
        enriched_pairs.append(enriched)
        if args.verbose and (i + 1) % 10 == 0:
            print(f"    Collected {i + 1}/{len(pairs)} pairs")
    
    pair_set = ContrastivePairSet(pairs=enriched_pairs, name="prism_training")
    
    if args.verbose:
        print(f"  ✓ Collected activations for {len(enriched_pairs)} pairs")
    
    # Configure PRISM
    num_directions = getattr(args, 'prism_num_directions', 3)
    prism_method = PRISMMethod(
        model=model,
        num_directions=num_directions,
    )
    
    # Train PRISM
    prism_result = prism_method.train(pair_set)
    
    if args.verbose:
        num_dirs = next(iter(prism_result.directions.values())).shape[0]
        print(f"✓ PRISM trained on {len(pairs)} pairs")
        print(f"  Layers: {len(prism_result.directions)}")
        print(f"  Directions per layer: {num_dirs}")
    
    return prism_result


def execute_modify_weights(args):
    """
    Execute weight modification command.

    Pipeline:
    1. Generate/load steering vectors (from task, trait, or file)
    2. Optionally load harmless vectors for biprojection
    3. Load model
    4. Modify weights (norm-preserving directional projection or additive)
    5. Export modified model
    
    If method is 'auto', analyzes activation geometry and picks the best method.
    """
    # Expand task if it's a skill or risk name
    from wisent.core.task_selector import expand_task_if_skill_or_risk
    if getattr(args, 'task', None):
        args.task = expand_task_if_skill_or_risk(args.task)
    
    log = bind(_LOG)
    start_time = time.time()

    # Determine norm_preserve and use_biprojection from args
    norm_preserve = not getattr(args, 'no_norm_preserve', False)
    use_biprojection = not getattr(args, 'no_biprojection', False)
    
    # Track if we need to run auto-selection (requires loading model and generating pairs first)
    needs_auto_selection = (args.method == "auto" or getattr(args, 'steering_method', 'auto') == "auto")
    
    # Store original method for later reference
    original_method = args.method
    original_steering_method = getattr(args, 'steering_method', 'auto')

    if args.verbose:
        print("\n" + "=" * 80)
        print("WEIGHT MODIFICATION")
        print("=" * 80)
        if needs_auto_selection:
            print(f"Method: AUTO (will analyze geometry to select best method)")
        else:
            print(f"Method: {args.method}")
            if args.method == "directional":
                print(f"Norm-Preserving: {norm_preserve} {'(RECOMMENDED)' if norm_preserve else '(NOT recommended)'}")
                print(f"Biprojection: {use_biprojection}")
        print(f"Model: {args.model}")
        print(f"Output: {args.output_dir}")
        print("=" * 80 + "\n")

    # Step 1: Get steering vectors
    # For auto mode, we'll skip vector generation here and handle it after auto-selection
    # since TITAN/PULSE/PRISM have their own vector generation flow
    skip_vector_generation = needs_auto_selection and not args.steering_vectors
    steering_vectors = None
    
    if skip_vector_generation:
        if args.verbose:
            print("Skipping initial vector generation (will generate after auto-selection)\n")
    elif args.steering_vectors:
        # Load pre-computed steering vectors
        if args.verbose:
            print(f"Loading steering vectors from {args.steering_vectors}...")

        vector_path = Path(args.steering_vectors)

        if vector_path.suffix == '.pt':
            # Load PyTorch format (from train-unified-goodness or similar)
            checkpoint = torch.load(args.steering_vectors, map_location=resolve_default_device(), weights_only=False)

            # Handle different .pt file formats
            if 'steering_vectors' in checkpoint:
                # Format from train-unified-goodness: {layer_idx: tensor}
                raw_vectors = checkpoint['steering_vectors']
                # Check if keys are already 0-indexed or need conversion
                first_key = next(iter(raw_vectors.keys()))
                if isinstance(first_key, str):
                    # String keys like "14" - convert to int, assume 0-indexed
                    steering_vectors = {
                        int(layer): vec if isinstance(vec, torch.Tensor) else torch.tensor(vec)
                        for layer, vec in raw_vectors.items()
                    }
                else:
                    # Integer keys - already 0-indexed
                    steering_vectors = {
                        layer: vec if isinstance(vec, torch.Tensor) else torch.tensor(vec)
                        for layer, vec in raw_vectors.items()
                    }
            elif 'vector' in checkpoint:
                # Single vector format: needs layer info
                layer = checkpoint.get('layer', checkpoint.get('best_layer', 14))
                steering_vectors = {layer: checkpoint['vector']}
            else:
                # Assume direct dict format {layer: vector}
                steering_vectors = {
                    int(k): v if isinstance(v, torch.Tensor) else torch.tensor(v)
                    for k, v in checkpoint.items()
                    if isinstance(k, (int, str)) and str(k).isdigit()
                }

            if args.verbose:
                print(f"✓ Loaded {len(steering_vectors)} steering vectors from .pt file")
                if 'metadata' in checkpoint:
                    meta = checkpoint['metadata']
                    if 'benchmarks_used' in meta:
                        print(f"  Trained on {len(meta['benchmarks_used'])} benchmarks")
                    if 'optimal_scale' in meta:
                        print(f"  Optimal steering scale: {meta['optimal_scale']}")
                print()
        else:
            # Load JSON format
            with open(args.steering_vectors, 'r') as f:
                vector_data = json.load(f)

            # Handle both formats: "steering_vectors" (old) and "vectors" (steering object)
            if "steering_vectors" in vector_data:
                vectors_dict = vector_data["steering_vectors"]
            elif "vectors" in vector_data:
                vectors_dict = vector_data["vectors"]
            else:
                raise ValueError("No steering vectors found in file (expected 'steering_vectors' or 'vectors' key)")

            # Convert layer numbers from JSON to 0-indexed for internal use
            # Steering objects use 0-indexed, old format uses 1-indexed
            steering_vectors = {}
            for layer, vector in vectors_dict.items():
                layer_int = int(layer)
                # Check if this looks like 1-indexed (layers typically start at 1 in old format)
                # Steering objects from create-steering-vector use the actual layer index
                steering_vectors[layer_int] = torch.tensor(vector)

            if args.verbose:
                print(f"✓ Loaded {len(steering_vectors)} steering vectors (layers: {list(steering_vectors.keys())})\n")

    elif args.task:
        # Parse task type
        task_lower = args.task.lower()
        
        if task_lower == "personalization":
            # Personalization: requires --trait
            if not args.trait:
                raise ValueError("--trait is required when --task personalization")
            
            if args.verbose:
                print(f"Generating steering vectors from trait '{args.trait}'...")

            from wisent.core.cli.generate_vector_from_synthetic import execute_generate_vector_from_synthetic

            # Create temp args for vector generation
            class VectorArgs:
                pass

            vector_args = VectorArgs()
            vector_args.trait = args.trait
            vector_args.model = args.model
            vector_args.num_pairs = args.num_pairs
            vector_args.similarity_threshold = getattr(args, 'similarity_threshold', 0.8)
            vector_args.layers = str(args.layers) if args.layers is not None else "all"
            vector_args.token_aggregation = args.token_aggregation
            vector_args.prompt_strategy = args.prompt_strategy
            vector_args.method = getattr(args, 'steering_method', 'caa')
            vector_args.normalize = args.normalize_vectors
            vector_args.verbose = args.verbose
            vector_args.timing = getattr(args, 'timing', False)
            vector_args.intermediate_dir = None
            vector_args.keep_intermediate = False
            vector_args.device = None
            vector_args.accept_low_quality_vector = getattr(args, 'accept_low_quality_vector', False)
            vector_args.pairs_cache_dir = getattr(args, 'pairs_cache_dir', None)
            vector_args.force_regenerate = getattr(args, 'force_regenerate', False)

            # Use temp file for steering vectors
            import tempfile
            temp_vector_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            vector_args.output = temp_vector_file.name
            temp_vector_file.close()

            # Generate vectors
            execute_generate_vector_from_synthetic(vector_args)

            # Load generated vectors
            with open(vector_args.output, 'r') as f:
                vector_data = json.load(f)

            # Handle both "steering_vectors" (old format) and "vectors" (new steering object format)
            vectors_dict = vector_data.get("steering_vectors") or vector_data.get("vectors", {})
            
            # Steering objects use 1-indexed layers, convert to 0-indexed for internal use
            steering_vectors = {
                int(layer) - 1: torch.tensor(vector)
                for layer, vector in vectors_dict.items()
            }

            # Optionally save steering vectors
            if getattr(args, 'save_steering_vectors', None):
                import shutil
                shutil.copy(vector_args.output, args.save_steering_vectors)
                if args.verbose:
                    print(f"✓ Saved steering vectors to {args.save_steering_vectors}")

            # Clean up temp file
            import os
            os.unlink(vector_args.output)

            if args.verbose:
                print(f"✓ Generated {len(steering_vectors)} steering vectors\n")

        elif task_lower == "refusal":
            # Refusal: use synthetic pairs with refusal trait
            if args.verbose:
                print("Generating steering vectors for refusal/compliance...")

            from wisent.core.cli.generate_vector_from_synthetic import execute_generate_vector_from_synthetic

            class VectorArgs:
                pass

            vector_args = VectorArgs()
            vector_args.trait = "refusal"
            vector_args.model = args.model
            vector_args.num_pairs = args.num_pairs
            vector_args.similarity_threshold = getattr(args, 'similarity_threshold', 0.8)
            vector_args.layers = str(args.layers) if args.layers is not None else "all"
            vector_args.token_aggregation = args.token_aggregation
            vector_args.prompt_strategy = args.prompt_strategy
            vector_args.method = getattr(args, 'steering_method', 'caa')
            vector_args.normalize = args.normalize_vectors
            vector_args.verbose = args.verbose
            vector_args.timing = getattr(args, 'timing', False)
            vector_args.intermediate_dir = None
            vector_args.keep_intermediate = False
            vector_args.device = None
            vector_args.accept_low_quality_vector = getattr(args, 'accept_low_quality_vector', False)
            vector_args.pairs_cache_dir = getattr(args, 'pairs_cache_dir', None)
            vector_args.force_regenerate = getattr(args, 'force_regenerate', False)

            import tempfile
            temp_vector_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            vector_args.output = temp_vector_file.name
            temp_vector_file.close()

            execute_generate_vector_from_synthetic(vector_args)

            with open(vector_args.output, 'r') as f:
                vector_data = json.load(f)

            # Handle both "steering_vectors" (old format) and "vectors" (new steering object format)
            vectors_dict = vector_data.get("steering_vectors") or vector_data.get("vectors", {})
            
            steering_vectors = {
                int(layer) - 1: torch.tensor(vector)
                for layer, vector in vectors_dict.items()
            }

            if getattr(args, 'save_steering_vectors', None):
                import shutil
                shutil.copy(vector_args.output, args.save_steering_vectors)
                if args.verbose:
                    print(f"✓ Saved steering vectors to {args.save_steering_vectors}")

            import os
            os.unlink(vector_args.output)

            if args.verbose:
                print(f"✓ Generated {len(steering_vectors)} steering vectors\n")

        elif task_lower == "custom":
            # Custom evaluator: requires --trait for vector generation
            if not args.trait:
                raise ValueError("--trait is required when --task custom (needed to generate steering vectors)")
            
            if args.verbose:
                print(f"Generating steering vectors from trait '{args.trait}' for custom evaluation...")

            from wisent.core.cli.generate_vector_from_synthetic import execute_generate_vector_from_synthetic

            class VectorArgs:
                pass

            vector_args = VectorArgs()
            vector_args.trait = args.trait
            vector_args.model = args.model
            vector_args.num_pairs = args.num_pairs
            vector_args.similarity_threshold = getattr(args, 'similarity_threshold', 0.8)
            vector_args.layers = str(args.layers) if args.layers is not None else "all"
            vector_args.token_aggregation = args.token_aggregation
            vector_args.prompt_strategy = args.prompt_strategy
            vector_args.method = getattr(args, 'steering_method', 'caa')
            vector_args.normalize = args.normalize_vectors
            vector_args.verbose = args.verbose
            vector_args.timing = getattr(args, 'timing', False)
            vector_args.intermediate_dir = None
            vector_args.keep_intermediate = False
            vector_args.device = None
            vector_args.accept_low_quality_vector = getattr(args, 'accept_low_quality_vector', False)
            vector_args.pairs_cache_dir = getattr(args, 'pairs_cache_dir', None)
            vector_args.force_regenerate = getattr(args, 'force_regenerate', False)

            import tempfile
            temp_vector_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            vector_args.output = temp_vector_file.name
            temp_vector_file.close()

            execute_generate_vector_from_synthetic(vector_args)

            with open(vector_args.output, 'r') as f:
                vector_data = json.load(f)

            # Handle both "steering_vectors" (old format) and "vectors" (new steering object format)
            vectors_dict = vector_data.get("steering_vectors") or vector_data.get("vectors", {})
            
            steering_vectors = {
                int(layer) - 1: torch.tensor(vector)
                for layer, vector in vectors_dict.items()
            }

            if getattr(args, 'save_steering_vectors', None):
                import shutil
                shutil.copy(vector_args.output, args.save_steering_vectors)
                if args.verbose:
                    print(f"✓ Saved steering vectors to {args.save_steering_vectors}")

            import os
            os.unlink(vector_args.output)

            if args.verbose:
                print(f"✓ Generated {len(steering_vectors)} steering vectors\n")

        elif "," in args.task:
            # Multiple benchmarks: use unified goodness training
            benchmarks = [b.strip() for b in args.task.split(",")]
            if args.verbose:
                print(f"Generating steering vectors from {len(benchmarks)} benchmarks: {', '.join(benchmarks)}...")

            from wisent.core.cli.train_unified_goodness import execute_train_unified_goodness

            class UnifiedArgs:
                pass

            unified_args = UnifiedArgs()
            unified_args.task = args.task  # Pass comma-separated list
            unified_args.exclude_benchmarks = None
            unified_args.max_benchmarks = getattr(args, 'max_benchmarks', None)
            unified_args.cap_pairs_per_benchmark = getattr(args, 'cap_pairs_per_benchmark', None)
            unified_args.train_ratio = getattr(args, 'train_ratio', 0.8)
            unified_args.seed = getattr(args, 'seed', 42)
            unified_args.model = args.model
            unified_args.device = getattr(args, 'device', None)
            unified_args.layer = None
            unified_args.layers = args.layers
            unified_args.token_aggregation = args.token_aggregation if hasattr(args, 'token_aggregation') else 'continuation'
            unified_args.prompt_strategy = args.prompt_strategy if hasattr(args, 'prompt_strategy') else 'chat_template'
            unified_args.method = "caa"
            unified_args.normalize = args.normalize_vectors if hasattr(args, 'normalize_vectors') else False
            unified_args.no_normalize = not unified_args.normalize
            unified_args.skip_evaluation = True
            unified_args.evaluate_steering_scales = "0.0,1.0"
            unified_args.save_pairs = None
            unified_args.save_report = None
            unified_args.verbose = args.verbose
            unified_args.timing = args.timing if hasattr(args, 'timing') else False

            import tempfile
            import os
            temp_vector_file = tempfile.NamedTemporaryFile(mode='w', suffix='.pt', delete=False)
            unified_args.output = temp_vector_file.name
            temp_vector_file.close()

            execute_train_unified_goodness(unified_args)

            checkpoint = torch.load(unified_args.output, map_location=resolve_default_device(), weights_only=False)
            
            if 'steering_vectors' in checkpoint:
                raw_vectors = checkpoint['steering_vectors']
            elif 'all_layer_vectors' in checkpoint:
                raw_vectors = checkpoint['all_layer_vectors']
            elif 'steering_vector' in checkpoint and 'layer_index' in checkpoint:
                raw_vectors = {checkpoint['layer_index']: checkpoint['steering_vector']}
            else:
                raw_vectors = {
                    k: v for k, v in checkpoint.items()
                    if isinstance(k, (int, str)) and str(k).isdigit()
                }
            
            steering_vectors = {}
            for layer, vec in raw_vectors.items():
                layer_idx = int(layer) if isinstance(layer, str) else layer
                steering_vectors[layer_idx] = vec if isinstance(vec, torch.Tensor) else torch.tensor(vec)

            if hasattr(args, 'save_steering_vectors') and args.save_steering_vectors:
                import shutil
                shutil.copy(unified_args.output, args.save_steering_vectors)
                if args.verbose:
                    print(f"✓ Saved steering vectors to {args.save_steering_vectors}")

            os.unlink(unified_args.output)

            if args.verbose:
                print(f"✓ Generated {len(steering_vectors)} steering vectors from {len(benchmarks)} benchmarks\n")

        else:
            # Single benchmark: use task-based generation
            # Check for optimal config first
            optimal_config = None
            use_optimal = getattr(args, 'use_optimal', True)
            
            if use_optimal:
                try:
                    from wisent.core.config_manager import get_cached_optimization
                    optimal_result = get_cached_optimization(args.model, args.task, method="*")
                    if optimal_result:
                        optimal_config = {
                            "method": optimal_result.method,
                            "layer": optimal_result.layer,
                            "strength": optimal_result.strength,
                            "strategy": optimal_result.strategy,
                            "token_aggregation": optimal_result.token_aggregation,
                            "prompt_strategy": optimal_result.prompt_strategy,
                            "score": optimal_result.score,
                            # Method-specific params
                            "num_directions": optimal_result.num_directions,
                            "direction_weighting": optimal_result.direction_weighting,
                            "retain_weight": optimal_result.retain_weight,
                            "sensor_layer": optimal_result.sensor_layer,
                            "condition_threshold": optimal_result.condition_threshold,
                            "gate_temperature": optimal_result.gate_temperature,
                            "gate_hidden_dim": optimal_result.gate_hidden_dim,
                            "intensity_hidden_dim": optimal_result.intensity_hidden_dim,
                        }
                        if args.verbose:
                            print(f"\n📊 Found optimal steering config for {args.task}!")
                            print(f"   Method: {optimal_config['method']}")
                            print(f"   Layer: {optimal_config['layer']}")
                            print(f"   Strength: {optimal_config['strength']}")
                            print(f"   Score: {optimal_config['score']:.3f}")
                            print(f"   → Using optimal settings\n")
                except Exception:
                    pass
            
            if args.verbose:
                print(f"Generating steering vectors from task '{args.task}'...")

            from wisent.core.cli.generate_vector_from_task import execute_generate_vector_from_task

            class VectorArgs:
                pass

            vector_args = VectorArgs()
            vector_args.task = args.task
            vector_args.trait_label = getattr(args, 'trait_label', 'correctness')
            vector_args.model = args.model
            vector_args.num_pairs = args.num_pairs
            
            # Use optimal config if available, otherwise use provided args
            if optimal_config and args.layers is None:
                vector_args.layers = str(optimal_config['layer'])
            else:
                vector_args.layers = str(args.layers) if args.layers is not None else "all"
            
            # Map token aggregation from stored format
            if optimal_config:
                token_agg_map = {
                    "last_token": "final",
                    "mean_pooling": "average",
                    "first_token": "first",
                    "max_pooling": "max",
                }
                vector_args.token_aggregation = token_agg_map.get(
                    optimal_config['token_aggregation'], 
                    args.token_aggregation
                )
                vector_args.method = optimal_config['method'].lower()
            else:
                vector_args.token_aggregation = args.token_aggregation
                # Don't pass 'auto' to vector generation - use 'caa' as default for directional mode
                steering_method = getattr(args, 'steering_method', 'caa')
                if steering_method == 'auto':
                    steering_method = 'caa'  # Default to CAA for directional projection
                vector_args.method = steering_method
            
            vector_args.prompt_strategy = args.prompt_strategy
            vector_args.normalize = args.normalize_vectors
            vector_args.verbose = args.verbose
            vector_args.timing = getattr(args, 'timing', False)
            vector_args.intermediate_dir = None
            vector_args.keep_intermediate = False
            vector_args.device = None
            vector_args.accept_low_quality_vector = getattr(args, 'accept_low_quality_vector', False)
            vector_args.use_optimal = use_optimal
            vector_args.extraction_strategy = getattr(args, 'extraction_strategy', 'chat_last')
            
            # Pass optimal config for method-specific params
            if optimal_config:
                vector_args._optimal_config = optimal_config

            # Use temp file for steering vectors
            import tempfile
            temp_vector_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            vector_args.output = temp_vector_file.name
            temp_vector_file.close()

            # Generate vectors
            execute_generate_vector_from_task(vector_args)

            # Load generated vectors
            with open(vector_args.output, 'r') as f:
                vector_data = json.load(f)

            # Handle different steering object formats
            method_name = vector_data.get("method", "caa")
            
            if method_name == "titan":
                # TITAN stores directions as multi-dimensional manifold
                # Extract effective direction (mean of weighted directions)
                directions = vector_data.get("directions", {})
                direction_weights = vector_data.get("direction_weights", {})
                
                steering_vectors = {}
                for layer_str, dirs in directions.items():
                    layer = int(layer_str)
                    dirs_tensor = torch.tensor(dirs)  # Shape: [num_directions, hidden_dim]
                    
                    # Get weights for this layer (if available)
                    weights = direction_weights.get(layer_str)
                    if weights is not None:
                        weights_tensor = torch.tensor(weights)
                        # Weighted sum of directions
                        effective_dir = (dirs_tensor * weights_tensor.unsqueeze(-1)).sum(0)
                    else:
                        # Simple mean if no weights
                        effective_dir = dirs_tensor.mean(0)
                    
                    steering_vectors[layer] = effective_dir
            else:
                # Standard format: "steering_vectors" or "vectors"
                vectors_dict = vector_data.get("steering_vectors") or vector_data.get("vectors", {})
                
                steering_vectors = {
                    int(layer) - 1: torch.tensor(vector)
                    for layer, vector in vectors_dict.items()
                }

            # Optionally save steering vectors
            if getattr(args, 'save_steering_vectors', None):
                import shutil
                shutil.copy(vector_args.output, args.save_steering_vectors)
                if args.verbose:
                    print(f"✓ Saved steering vectors to {args.save_steering_vectors}")

            # Clean up temp file
            import os
            os.unlink(vector_args.output)

            if args.verbose:
                print(f"✓ Generated {len(steering_vectors)} steering vectors")
                if optimal_config:
                    print(f"   (using optimal {optimal_config['method']} config with score={optimal_config['score']:.3f})\n")
                else:
                    print()

    # Step 1.5: Load harmless vectors for biprojection (if provided)
    harmless_vectors = None
    if args.method == "directional" and use_biprojection and hasattr(args, 'harmless_vectors') and args.harmless_vectors:
        if args.verbose:
            print(f"Loading harmless vectors from {args.harmless_vectors}...")

        with open(args.harmless_vectors, 'r') as f:
            harmless_data = json.load(f)

        # Handle both "steering_vectors" (old format) and "vectors" (new steering object format)
        harmless_dict = harmless_data.get("steering_vectors") or harmless_data.get("vectors", {})
        
        harmless_vectors = {
            int(layer) - 1: torch.tensor(vector)
            for layer, vector in harmless_dict.items()
        }

        if args.verbose:
            print(f"✓ Loaded {len(harmless_vectors)} harmless vectors for biprojection\n")

    # Step 2: Load model
    if args.verbose:
        print(f"Loading model '{args.model}'...")

    # For additive bias method, we need to enable biases in model config
    enable_bias = (args.method == "additive" and getattr(args, 'additive_method', 'bias') == 'bias')
    
    if enable_bias:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        # Enable biases in attention layers
        if hasattr(config, 'attention_bias'):
            config.attention_bias = True
        if hasattr(config, 'mlp_bias'):
            config.mlp_bias = True
        # Load model with updated config
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        # Create a simple wrapper to get num_layers
        class ModelInfo:
            def __init__(self, model):
                if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                    self.num_layers = len(model.model.layers)
                elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                    self.num_layers = len(model.transformer.h)
                else:
                    self.num_layers = 32  # fallback
        wisent_model = ModelInfo(model)
        if args.verbose:
            print(f"  (Enabled attention biases for additive method)")
    else:
        wisent_model = WisentModel(args.model, device=getattr(args, 'device', None))
        model = wisent_model.hf_model  # Get underlying HF model for weight modification
        tokenizer = wisent_model.tokenizer

    if args.verbose:
        print(f"✓ Model loaded with {wisent_model.num_layers} layers\n")

    # Step 2.4: AUTO-SELECTION MODE - Analyze geometry and pick best method
    if needs_auto_selection and args.task:
        # Generate pairs for analysis
        if args.verbose:
            print("Generating pairs for auto-selection analysis...")
        
        pairs = _generate_pairs_for_titan(args, wisent_model)
        
        if pairs and len(pairs) >= 10:
            # Run auto-selection
            steering_method, modification_method, geo_result = _auto_select_steering_method(
                pairs, wisent_model, args
            )
            
            # Update args with selected methods
            args.steering_method = steering_method
            args.method = modification_method
            
            if args.verbose:
                print(f"✓ Auto-selected: steering={steering_method}, modification={modification_method}\n")
            
            # If directional mode selected, generate CAA vectors now
            if modification_method == "directional" and steering_vectors is None:
                if args.verbose:
                    print("Generating CAA steering vectors for directional projection...")
                
                from wisent.core.activations.activations_collector import ActivationCollector
                from wisent.core.activations.extraction_strategy import ExtractionStrategy
                from wisent.core.steering_methods.methods.caa import CAAMethod
                from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
                
                # Collect activations from all layers
                collector = ActivationCollector(model=wisent_model)
                all_layers = _get_all_layers(wisent_model)
                
                enriched_pairs = []
                for pair in pairs:
                    enriched = collector.collect(pair, strategy=ExtractionStrategy.CHAT_LAST, layers=all_layers)
                    enriched_pairs.append(enriched)
                
                pair_set = ContrastivePairSet(pairs=enriched_pairs, name="auto_caa")
                
                # Train CAA
                caa_method = CAAMethod()
                caa_result = caa_method.train(pair_set)
                
                # Convert to steering_vectors format
                steering_vectors = {}
                for layer_name, vector in caa_result.directions.items():
                    layer_idx = int(layer_name.replace("layer_", "")) if "layer_" in str(layer_name) else int(layer_name)
                    steering_vectors[layer_idx] = vector
                
                if args.verbose:
                    print(f"✓ Generated CAA vectors for {len(steering_vectors)} layers\n")
        else:
            # Not enough pairs - default to TITAN
            if args.verbose:
                print("⚠️  Not enough pairs for analysis, defaulting to TITAN\n")
            args.steering_method = "titan"
            args.method = "titan"

    # Step 2.5: GUIDED MODE - Use linearity diagnostics for data-driven modification
    if getattr(args, 'guided', False):
        stats = _execute_guided_modification(args, wisent_model, model, tokenizer)
        return  # Guided mode handles export internally
    
    # Step 2.6: MULTI-CONCEPT MODE - Modify multiple concepts simultaneously
    if getattr(args, 'concepts', None):
        stats = _execute_multi_concept_modification(args, wisent_model, model, tokenizer, steering_vectors)
        return  # Multi-concept mode handles export internally

    # Step 2.7: TITAN MODE - Full TITAN with dynamic gating
    if args.method == "titan":
        steering_method = getattr(args, 'steering_method', 'caa')
        
        if steering_method != "titan":
            print(f"⚠ Warning: --method titan requires --steering-method titan")
            print(f"  Falling back to directional projection with {steering_method} vectors\n")
            args.method = "directional"
        else:
            # Full TITAN mode - train TITAN and export with hooks
            if args.verbose:
                print("Training TITAN for full dynamic steering...")
            
            # Generate pairs for TITAN training
            pairs = _generate_pairs_for_titan(args, wisent_model)
            
            if not pairs:
                print("✗ Error: Could not generate pairs for TITAN")
                sys.exit(1)
            
            # Train TITAN
            titan_result = _train_titan_for_task(args, wisent_model, pairs)
            
            # Export with TITAN-specific function
            from wisent.core.weight_modification.export import export_titan_model
            
            titan_mode = getattr(args, 'titan_mode', 'hybrid')
            
            if args.verbose:
                print(f"\nExporting TITAN model (mode={titan_mode})...")
            
            export_titan_model(
                model=model,
                titan_result=titan_result,
                save_path=args.output_dir,
                tokenizer=tokenizer,
                mode=titan_mode,
                push_to_hub=args.push_to_hub,
                repo_id=args.repo_id if args.push_to_hub else None,
                commit_message=args.commit_message,
            )
            
            if args.verbose:
                print(f"\n✓ TITAN model exported to {args.output_dir}")
                print(f"  Mode: {titan_mode}")
                print(f"  Layers: {len(titan_result.layer_order)}")
                print(f"  Load with: load_titan_model('{args.output_dir}')")
            
            return  # TITAN mode handles export internally

    # Step 2.8: PULSE MODE - Conditional steering with gating
    if args.method == "pulse":
        steering_method = getattr(args, 'steering_method', 'caa')
        
        if steering_method != "pulse":
            print(f"⚠ Warning: --method pulse requires --steering-method pulse")
            print(f"  Falling back to directional projection with {steering_method} vectors\n")
            args.method = "directional"
        else:
            if args.verbose:
                print("Training PULSE for conditional steering...")
            
            # Generate pairs for PULSE training
            pairs = _generate_pairs_for_titan(args, wisent_model)  # Same pair generation
            
            if not pairs:
                print("✗ Error: Could not generate pairs for PULSE")
                sys.exit(1)
            
            # Train PULSE
            pulse_result = _train_pulse_for_task(args, wisent_model, pairs)
            
            # Export with PULSE-specific function
            from wisent.core.weight_modification.export import export_pulse_model
            
            pulse_mode = getattr(args, 'titan_mode', 'hybrid')  # Reuse titan_mode arg
            
            if args.verbose:
                print(f"\nExporting PULSE model (mode={pulse_mode})...")
            
            export_pulse_model(
                model=model,
                pulse_result=pulse_result,
                save_path=args.output_dir,
                tokenizer=tokenizer,
                mode=pulse_mode,
                strength=args.strength,
                push_to_hub=args.push_to_hub,
                repo_id=args.repo_id if args.push_to_hub else None,
                commit_message=args.commit_message,
            )
            
            if args.verbose:
                print(f"\n✓ PULSE model exported to {args.output_dir}")
                print(f"  Mode: {pulse_mode}")
                print(f"  Layers: {len(pulse_result.behavior_vectors)}")
                print(f"  Threshold: {pulse_result.optimal_threshold:.3f}")
                print(f"  Load with: load_pulse_model('{args.output_dir}')")
            
            return

    # Step 2.9: PRISM MODE - Multi-directional steering
    if args.method == "prism":
        steering_method = getattr(args, 'steering_method', 'caa')
        
        if steering_method != "prism":
            print(f"⚠ Warning: --method prism requires --steering-method prism")
            print(f"  Falling back to directional projection with {steering_method} vectors\n")
            args.method = "directional"
        else:
            if args.verbose:
                print("Training PRISM for multi-directional steering...")
            
            # Generate pairs for PRISM training
            pairs = _generate_pairs_for_titan(args, wisent_model)
            
            if not pairs:
                print("✗ Error: Could not generate pairs for PRISM")
                sys.exit(1)
            
            # Train PRISM
            prism_result = _train_prism_for_task(args, wisent_model, pairs)
            
            # Export with PRISM-specific function
            from wisent.core.weight_modification.export import export_prism_model
            
            prism_mode = getattr(args, 'prism_mode', 'weighted')
            
            if args.verbose:
                print(f"\nExporting PRISM model (mode={prism_mode})...")
            
            export_prism_model(
                model=model,
                prism_result=prism_result,
                save_path=args.output_dir,
                tokenizer=tokenizer,
                mode=prism_mode,
                strength=args.strength,
                push_to_hub=args.push_to_hub,
                repo_id=args.repo_id if args.push_to_hub else None,
                commit_message=args.commit_message,
            )
            
            if args.verbose:
                num_dirs = next(iter(prism_result.directions.values())).shape[0]
                print(f"\n✓ PRISM model exported to {args.output_dir}")
                print(f"  Mode: {prism_mode}")
                print(f"  Layers: {len(prism_result.directions)}")
                print(f"  Directions per layer: {num_dirs}")
                print(f"  Load with: load_prism_model('{args.output_dir}')")
            
            return

    # Step 3: Modify weights (standard mode)
    if args.verbose:
        print(f"Modifying weights using {args.method} method...")
        print()

    if args.method == "directional":
        # Directional projection method (norm-preserving by default)
        if args.use_kernel:
            # Use kernel-based layer weighting
            stats = project_with_kernel(
                model,
                steering_vectors,
                harmless_vectors=harmless_vectors,
                max_weight=args.max_weight,
                max_weight_position=args.max_weight_position,
                min_weight=args.min_weight,
                min_weight_distance=args.min_weight_distance,
                components=args.components,
                normalize_vectors=args.normalize_vectors,
                norm_preserve=norm_preserve,
                use_biprojection=use_biprojection,
                verbose=args.verbose,
            )
        else:
            # Uniform directional projection
            stats = project_weights(
                model,
                steering_vectors,
                harmless_vectors=harmless_vectors,
                components=args.components,
                strength=args.strength,
                normalize_vectors=args.normalize_vectors,
                norm_preserve=norm_preserve,
                use_biprojection=use_biprojection,
                verbose=args.verbose,
            )

    elif args.method == "additive":
        # Additive method
        if args.use_kernel:
            # Use kernel-based layer weighting
            stats = bake_steering_with_kernel(
                model,
                steering_vectors,
                max_alpha=args.max_weight,  # Use max_weight for alpha
                max_alpha_position=args.max_weight_position,
                min_alpha=args.min_weight,
                components=args.components,
                method=args.additive_method,
                verbose=args.verbose,
            )
        else:
            # Uniform additive
            stats = bake_steering_into_weights(
                model,
                steering_vectors,
                components=args.components,
                alpha=args.alpha,
                method=args.additive_method,
                verbose=args.verbose,
            )

    if args.verbose:
        print()
        print("✓ Weight modification complete!")
        print(f"  Layers modified: {stats['layers_modified']}")
        print(f"  Components modified: {stats['components_modified']}")
        print(f"  Parameters modified: {stats['total_parameters_modified']:,}")
        if args.method == "directional":
            print(f"  Norms preserved: {stats.get('norm_preserved', 'N/A')}")
        print()

    # Step 4: Export model
    if args.verbose:
        print(f"Exporting modified model to {args.output_dir}...")

    # Validate push-to-hub requirements
    if args.push_to_hub and not args.repo_id:
        print("✗ Error: --repo-id required when using --push-to-hub")
        sys.exit(1)

    export_modified_model(
        model,
        args.output_dir,
        tokenizer=tokenizer,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id if args.push_to_hub else None,
        commit_message=args.commit_message,
    )

    if args.verbose:
        print(f"✓ Model exported to {args.output_dir}")
        if args.push_to_hub:
            print(f"✓ Model uploaded to HuggingFace Hub: {args.repo_id}")

    # Timing
    if args.timing:
        elapsed = time.time() - start_time
        print(f"\n⏱  Total time: {elapsed:.2f}s")

    if args.verbose:
        print("\n" + "=" * 80)
        print("WEIGHT MODIFICATION COMPLETE")
        print("=" * 80)
        print(f"Modified model: {args.output_dir}")
        print(f"Method: {args.method}")
        if args.method == "directional":
            print(f"Norm-preserving: {norm_preserve}")
            print(f"Biprojection: {use_biprojection and harmless_vectors is not None}")
        print(f"Layers modified: {stats['layers_modified']}")
        print(f"Parameters modified: {stats['total_parameters_modified']:,}")
        print("=" * 80 + "\n")

    log.info("Weight modification complete", extra={
        "method": args.method,
        "output_dir": args.output_dir,
        "norm_preserve": norm_preserve if args.method == "directional" else None,
        "stats": stats,
    })


def _execute_guided_modification(args, wisent_model, model, tokenizer):
    """
    Execute linearity-guided weight modification.
    
    This is a novel approach that uses diagnostic signals to perform
    surgical, targeted weight modification without expensive optimization.
    
    Key innovations:
    1. Layer selection based on measured linear separability
    2. Fisher ratio-weighted ablation strength
    3. Surgical modification of only high-signal layers
    4. Optional collateral damage validation
    """
    from wisent.core.weight_modification.guided import (
        GuidedModificationConfig,
        AblationMode,
        run_guided_modification,
    )
    from wisent.core.weight_modification import export_modified_model
    from wisent.core.contrastive_pairs.core.pair import ContrastivePair
    from wisent.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse
    
    log = bind(_LOG)
    start_time = time.time()
    
    if args.verbose:
        print("\n" + "=" * 80)
        print("GUIDED WEIGHT MODIFICATION (Linearity-Driven)")
        print("=" * 80)
        print("This mode uses diagnostic signals for data-driven modification:")
        print("  - Layer selection based on measured linear separability")
        print("  - Fisher ratio-weighted ablation strength")
        print("  - Surgical modification of only high-signal layers")
        print("=" * 80 + "\n")
    
    # Step 1: Generate contrastive pairs for the task
    pairs = _generate_pairs_for_guided_mode(args)
    
    if not pairs:
        print("Error: No contrastive pairs generated for guided mode")
        sys.exit(1)
    
    if args.verbose:
        print(f"Generated {len(pairs)} contrastive pairs for diagnostics\n")
    
    # Step 2: Configure guided modification
    mode_map = {
        "full": AblationMode.FULL,
        "surgical": AblationMode.SURGICAL,
        "adaptive": AblationMode.ADAPTIVE,
    }
    
    config = GuidedModificationConfig(
        mode=mode_map.get(args.guided_mode, AblationMode.ADAPTIVE),
        surgical_top_k=args.surgical_top_k,
        min_linear_score=args.min_linear_score,
        use_fisher_weights=not getattr(args, 'no_fisher_weights', False),
        extraction_strategy=args.extraction_strategy,
        validate_collateral=getattr(args, 'validate_collateral', False),
        max_allowed_degradation=getattr(args, 'max_degradation', 0.1),
        base_strength=args.strength,
        normalize_vectors=getattr(args, 'normalize_vectors', True),
        verbose=args.verbose,
    )
    
    # Step 3: Run guided modification
    result = run_guided_modification(
        model=model,
        pairs=pairs,
        wisent_model=wisent_model,
        config=config,
        components=args.components,
    )
    
    # Step 4: Save diagnostics if requested
    if getattr(args, 'save_diagnostics', None):
        diagnostics_data = {
            "layers": {
                str(layer): {
                    "linear_score": diag.linear_score,
                    "knn_score": diag.knn_score,
                    "fisher_ratio": diag.fisher_ratio,
                    "cohens_d": diag.cohens_d,
                    "variance_explained": diag.variance_explained,
                    "recommended_weight": diag.recommended_weight,
                    "extraction_strategy": diag.extraction_strategy,
                }
                for layer, diag in result.layer_diagnostics.items()
            },
            "layer_weights": {str(k): v for k, v in result.layer_weights.items()},
            "mode_used": result.mode_used.value,
            "recommendation": result.recommendation,
        }
        
        with open(args.save_diagnostics, 'w') as f:
            json.dump(diagnostics_data, f, indent=2)
        
        if args.verbose:
            print(f"\n✓ Saved diagnostics to {args.save_diagnostics}")
    
    # Step 5: Export model
    if args.verbose:
        print(f"\nExporting modified model to {args.output_dir}...")
    
    if args.push_to_hub and not args.repo_id:
        print("✗ Error: --repo-id required when using --push-to-hub")
        sys.exit(1)
    
    export_modified_model(
        model,
        args.output_dir,
        tokenizer=tokenizer,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id if args.push_to_hub else None,
        commit_message=args.commit_message,
    )
    
    # Step 6: Print summary
    if args.timing:
        elapsed = time.time() - start_time
        print(f"\n⏱  Total time: {elapsed:.2f}s")
    
    if args.verbose:
        print("\n" + "=" * 80)
        print("GUIDED MODIFICATION COMPLETE")
        print("=" * 80)
        print(f"Mode: {result.mode_used.value}")
        print(f"Layers modified: {result.layers_modified}")
        print(f"Parameters modified: {result.total_parameters_modified:,}")
        print(f"\nRecommendation: {result.recommendation}")
        print("=" * 80 + "\n")
    
    log.info("Guided weight modification complete", extra={
        "mode": result.mode_used.value,
        "layers_modified": result.layers_modified,
        "output_dir": args.output_dir,
    })
    
    return {
        "layers_modified": result.layers_modified,
        "total_parameters_modified": result.total_parameters_modified,
    }


def _execute_multi_concept_modification(args, wisent_model, model, tokenizer, base_steering_vectors):
    """
    Execute multi-concept weight modification.
    
    This mode allows modifying multiple concepts simultaneously:
    - Suppress some directions (e.g., refusal)
    - Enhance others (e.g., truthfulness)
    - Handle interference between concepts via orthogonalization
    """
    from wisent.core.weight_modification.multi_concept import (
        MultiConceptConfig,
        ConceptSpec,
        ConceptAction,
        run_multi_concept_modification,
    )
    from wisent.core.weight_modification import export_modified_model
    
    log = bind(_LOG)
    start_time = time.time()
    
    if args.verbose:
        print("\n" + "=" * 80)
        print("MULTI-CONCEPT WEIGHT MODIFICATION")
        print("=" * 80)
        print("Modifying multiple concepts simultaneously with:")
        print("  - Interference minimization via orthogonalization")
        print("  - Bidirectional ablation (suppress + enhance)")
        print("=" * 80 + "\n")
    
    # Parse concept specifications
    concepts = []
    
    for concept_str in args.concepts:
        parts = concept_str.split(":")
        if len(parts) < 2:
            print(f"Error: Invalid concept format '{concept_str}'. Use 'name:action' or 'name:action:strength'")
            sys.exit(1)
        
        name = parts[0]
        action_str = parts[1].lower()
        strength = float(parts[2]) if len(parts) > 2 else 1.0
        
        action_map = {
            "suppress": ConceptAction.SUPPRESS,
            "enhance": ConceptAction.ENHANCE,
            "neutral": ConceptAction.NEUTRAL,
        }
        
        if action_str not in action_map:
            print(f"Error: Unknown action '{action_str}'. Use: suppress, enhance, neutral")
            sys.exit(1)
        
        # Generate steering vectors for this concept
        # For now, use the base steering vectors if name matches task
        # In full implementation, would generate per-concept vectors
        if name == args.task or name == "base":
            vectors = base_steering_vectors
        else:
            # Would need to generate vectors for this concept
            print(f"Warning: Using base steering vectors for concept '{name}'")
            vectors = base_steering_vectors
        
        concepts.append(ConceptSpec(
            name=name,
            steering_vectors=vectors,
            action=action_map[action_str],
            strength=strength,
        ))
    
    if args.verbose:
        print(f"Concepts to modify: {len(concepts)}")
        for c in concepts:
            print(f"  - {c.name}: {c.action.value} (strength={c.strength})")
        print()
    
    # Configure multi-concept modification
    config = MultiConceptConfig(
        orthogonalize=not getattr(args, 'no_orthogonalize', False),
        components=args.components,
        norm_preserve=not getattr(args, 'no_norm_preserve', False),
        verbose=args.verbose,
    )
    
    # Run multi-concept modification
    result = run_multi_concept_modification(
        model=model,
        concepts=concepts,
        config=config,
    )
    
    # Export model
    if args.verbose:
        print(f"\nExporting modified model to {args.output_dir}...")
    
    if args.push_to_hub and not args.repo_id:
        print("✗ Error: --repo-id required when using --push-to-hub")
        sys.exit(1)
    
    export_modified_model(
        model,
        args.output_dir,
        tokenizer=tokenizer,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id if args.push_to_hub else None,
        commit_message=args.commit_message,
    )
    
    # Print summary
    if args.timing:
        elapsed = time.time() - start_time
        print(f"\n⏱  Total time: {elapsed:.2f}s")
    
    if args.verbose:
        print("\n" + "=" * 80)
        print("MULTI-CONCEPT MODIFICATION COMPLETE")
        print("=" * 80)
        print(f"Concepts modified: {result.concepts_modified}")
        print(f"Layers modified: {result.layers_modified}")
        print(f"Parameters modified: {result.total_parameters_modified:,}")
        print(f"Orthogonalized: {result.orthogonalized}")
        if result.warnings:
            print(f"\nWarnings:")
            for w in result.warnings:
                print(f"  - {w}")
        print("=" * 80 + "\n")
    
    log.info("Multi-concept modification complete", extra={
        "concepts": result.concepts_modified,
        "layers_modified": result.layers_modified,
        "output_dir": args.output_dir,
    })
    
    return {
        "layers_modified": result.layers_modified,
        "total_parameters_modified": result.total_parameters_modified,
    }


def _generate_pairs_for_guided_mode(args):
    """Generate contrastive pairs for guided mode diagnostics."""
    from wisent.core.contrastive_pairs.core.pair import ContrastivePair
    from wisent.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse
    
    pairs = []
    
    # Try to load from task
    if args.task:
        try:
            # Use the task-based pair generation
            from wisent.core.data_loaders import load_contrastive_pairs
            
            task_pairs = load_contrastive_pairs(
                task=args.task,
                num_pairs=args.num_pairs,
                model_name=args.model,
            )
            
            for p in task_pairs:
                if hasattr(p, 'prompt') and hasattr(p, 'positive_response') and hasattr(p, 'negative_response'):
                    pairs.append(p)
        except Exception as e:
            print(f"Warning: Could not load pairs from task: {e}")
            
            # Fallback: try synthetic generation
            try:
                from wisent.core.synthetic.generators.pairs_generator import generate_synthetic_pairs
                
                synthetic_pairs = generate_synthetic_pairs(
                    trait=args.task,
                    num_pairs=args.num_pairs,
                )
                
                for sp in synthetic_pairs:
                    pairs.append(ContrastivePair(
                        prompt=sp.get('prompt', ''),
                        positive_response=PositiveResponse(
                            model_response=sp.get('positive', '')
                        ),
                        negative_response=NegativeResponse(
                            model_response=sp.get('negative', '')
                        ),
                    ))
            except Exception as e2:
                print(f"Warning: Synthetic generation also failed: {e2}")
    
    return pairs
