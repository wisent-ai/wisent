"""Generate steering vector from task command execution logic - unified pipeline."""

import sys
import os
import time
import tempfile
from argparse import Namespace

from wisent.core.cli.generate_pairs_from_task import execute_generate_pairs_from_task
from wisent.core.cli.get_activations import execute_get_activations
from wisent.core.cli.create_steering_vector import execute_create_steering_vector


def execute_generate_vector_from_task(args):
    """
    Execute the generate-vector-from-task command - full pipeline in one command.
    
    Pipeline:
    1. Generate contrastive pairs from lm-eval task
    2. Collect activations from those pairs
    3. Create steering vectors from the activations
    """
    print(f"\n{'='*60}")
    print(f"🎯 Generating Steering Vector from Task (Full Pipeline)")
    print(f"{'='*60}")
    print(f"   Task: {args.task}")
    print(f"   Trait Label: {args.trait_label}")
    print(f"   Model: {args.model}")
    print(f"   Num Pairs: {args.num_pairs}")
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
            token_aggregation=args.token_aggregation,
            prompt_strategy=args.prompt_strategy,
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
