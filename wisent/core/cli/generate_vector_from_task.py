"""Generate steering vector from task command execution logic - unified pipeline."""

import sys
import os
import time
import tempfile
from argparse import Namespace

from wisent.core.cli.generate_pairs_from_task import execute_generate_pairs_from_task
from wisent.core.cli.get_activations import execute_get_activations
from wisent.core.cli.create_steering_vector import execute_create_steering_vector


def _load_optimal_defaults(model_name: str, task_name: str, args):
    """
    Load optimal steering config from cache and apply as defaults.
    
    Returns the optimal config dict if found, None otherwise.
    """
    try:
        from wisent.core.config_manager import get_cached_optimization
        
        # Try to get cached optimization for any method
        result = get_cached_optimization(model_name, task_name, method="*")
        if result is None:
            return None
        
        optimal_config = {
            "method": result.method,
            "layer": result.layer,
            "strength": result.strength,
            "strategy": result.strategy,
            "extraction_strategy": getattr(result, 'extraction_strategy', None),
            "score": result.score,
        }
        
        # Add method-specific params
        if result.method.upper() == "PRISM":
            optimal_config.update({
                "num_directions": result.num_directions,
                "direction_weighting": result.direction_weighting,
                "retain_weight": result.retain_weight,
            })
        elif result.method.upper() == "PULSE":
            optimal_config.update({
                "sensor_layer": result.sensor_layer,
                "condition_threshold": result.condition_threshold,
                "gate_temperature": result.gate_temperature,
            })
        elif result.method.upper() == "TITAN":
            optimal_config.update({
                "num_directions": result.num_directions,
                "gate_hidden_dim": result.gate_hidden_dim,
                "intensity_hidden_dim": result.intensity_hidden_dim,
            })
        
        return optimal_config
    except Exception:
        return None


def execute_generate_vector_from_task(args):
    """
    Execute the generate-vector-from-task command - full pipeline in one command.
    
    Pipeline:
    1. Generate contrastive pairs from lm-eval task
    2. Collect activations from those pairs
    3. Create steering vectors from the activations
    
    If optimal steering config exists for model/task, those settings are used as defaults.
    """
    # Expand task if it's a skill or risk name
    from wisent.core.task_selector import expand_task_if_skill_or_risk
    args.task = expand_task_if_skill_or_risk(args.task)
    
    # Check for optimal defaults from previous optimization
    use_optimal = getattr(args, 'use_optimal', True)  # Default to using optimal if available
    optimal_config = None
    
    if use_optimal:
        optimal_config = _load_optimal_defaults(args.model, args.task, args)
        
        if optimal_config:
            print(f"\n{'='*60}")
            print(f"📊 Found optimal steering config from previous optimization!")
            print(f"{'='*60}")
            print(f"   Method: {optimal_config['method']}")
            print(f"   Layer: {optimal_config['layer']}")
            print(f"   Strength: {optimal_config['strength']}")
            if optimal_config.get('extraction_strategy'):
                print(f"   Extraction Strategy: {optimal_config['extraction_strategy']}")
            print(f"   Score: {optimal_config['score']:.3f}")
            print(f"{'='*60}")

            # Apply optimal defaults if user didn't explicitly override
            if not getattr(args, '_layers_set_by_user', False) and args.layers is None:
                args.layers = str(optimal_config['layer'])
                print(f"   → Using optimal layer: {args.layers}")

            if not getattr(args, '_extraction_strategy_set_by_user', False) and optimal_config.get('extraction_strategy'):
                args.extraction_strategy = optimal_config['extraction_strategy']
                print(f"   → Using optimal extraction strategy: {args.extraction_strategy}")

            if not getattr(args, '_method_set_by_user', False):
                args.method = optimal_config['method'].lower()
                print(f"   → Using optimal method: {args.method}")

            # Store optimal config for later use
            args._optimal_config = optimal_config
            print()
    
    print(f"\n{'='*60}")
    print(f"🎯 Generating Steering Vector from Task (Full Pipeline)")
    print(f"{'='*60}")
    print(f"   Task: {args.task}")
    print(f"   Trait Label: {args.trait_label}")
    print(f"   Model: {args.model}")
    print(f"   Num Pairs: {args.num_pairs}")
    if optimal_config:
        print(f"   Using Optimal Config: YES (score={optimal_config['score']:.3f})")
    print(f"{'='*60}\n")
    
    pipeline_start = time.time() if args.timing else None
    
    try:
        # Determine intermediate file paths
        if args.intermediate_dir:
            intermediate_dir = args.intermediate_dir
        else:
            intermediate_dir = os.path.dirname(os.path.abspath(args.output))
        
        os.makedirs(intermediate_dir, exist_ok=True)
        
        # Create intermediate file paths
        if args.keep_intermediate:
            pairs_file = os.path.join(intermediate_dir, f"{args.task}_{args.trait_label}_pairs.json")
            enriched_file = os.path.join(intermediate_dir, f"{args.task}_{args.trait_label}_pairs_with_activations.json")
        else:
            # Use temporary files that will be deleted
            pairs_file = tempfile.NamedTemporaryFile(mode='w', suffix='_pairs.json', delete=False).name
            enriched_file = tempfile.NamedTemporaryFile(mode='w', suffix='_enriched.json', delete=False).name

        # Step 1: Generate pairs from task
        print(f"{'='*60}")
        print(f"Step 1/3: Generating contrastive pairs from task...")
        print(f"{'='*60}\n")
        
        pairs_args = Namespace(
            task_name=args.task,
            limit=args.num_pairs,
            output=pairs_file,
            seed=42,
            verbose=args.verbose,
        )
        
        execute_generate_pairs_from_task(pairs_args)
        print(f"\n✓ Step 1 complete: Pairs saved to {pairs_file}\n")
        
        # Step 2: Collect activations
        print(f"{'='*60}")
        print(f"Step 2/3: Collecting activations from pairs...")
        print(f"{'='*60}\n")
        
        activations_args = Namespace(
            pairs_file=pairs_file,
            output=enriched_file,
            model=args.model,
            device=args.device,
            layers=args.layers,
            extraction_strategy=args.extraction_strategy,
            verbose=args.verbose,
            timing=args.timing,
        )
        
        execute_get_activations(activations_args)
        print(f"\n✓ Step 2 complete: Enriched pairs saved to {enriched_file}\n")
        
        # Step 3: Create steering vector
        print(f"{'='*60}")
        print(f"Step 3/3: Creating steering vector...")
        print(f"{'='*60}\n")
        
        vector_args = Namespace(
            enriched_pairs_file=enriched_file,
            output=args.output,
            method=args.method,
            normalize=args.normalize,
            verbose=args.verbose,
            timing=args.timing,
            accept_low_quality_vector=getattr(args, 'accept_low_quality_vector', False),
            # Universal Subspace options for PRISM/TITAN
            auto_num_directions=getattr(args, 'auto_num_directions', False),
            use_universal_basis_init=getattr(args, 'use_universal_basis_init', False),
            num_directions=getattr(args, 'num_directions', 3),
        )
        
        execute_create_steering_vector(vector_args)
        print(f"\n✓ Step 3 complete: Steering vector saved to {args.output}\n")
        
        # Clean up intermediate files if not keeping them
        if not args.keep_intermediate:
            if args.verbose:
                print(f"\n🧹 Cleaning up intermediate files...")
            try:
                os.unlink(pairs_file)
                os.unlink(enriched_file)
                if args.verbose:
                    print(f"   ✓ Removed temporary files")
            except Exception as e:
                if args.verbose:
                    print(f"   ⚠️  Warning: Could not remove some temporary files: {e}")
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"✅ Full Pipeline Completed Successfully!")
        print(f"{'='*60}")
        print(f"   Final steering vector: {args.output}")
        if args.keep_intermediate:
            print(f"   Intermediate pairs: {pairs_file}")
            print(f"   Intermediate enriched: {enriched_file}")
        if args.timing and pipeline_start:
            total_time = time.time() - pipeline_start
            print(f"   ⏱️  Total pipeline time: {total_time:.2f}s")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        
        # Clean up on failure
        if not args.keep_intermediate:
            try:
                if 'pairs_file' in locals() and os.path.exists(pairs_file):
                    os.unlink(pairs_file)
                if 'enriched_file' in locals() and os.path.exists(enriched_file):
                    os.unlink(enriched_file)
            except:
                pass
        
        sys.exit(1)
