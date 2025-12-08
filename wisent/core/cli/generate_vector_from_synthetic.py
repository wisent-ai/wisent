"""Generate steering vector from synthetic pairs command execution logic - unified pipeline."""

import sys
import os
import time
import tempfile
import hashlib
import re
from argparse import Namespace

from wisent.core.cli.generate_pairs import execute_generate_pairs
from wisent.core.cli.get_activations import execute_get_activations
from wisent.core.cli.create_steering_vector import execute_create_steering_vector


def _get_pairs_cache_filename(trait: str, num_pairs: int) -> str:
    """Generate a consistent filename for cached pairs based on trait and num_pairs."""
    # Create a safe filename from trait
    safe_trait = re.sub(r'[^\w\s-]', '', trait.lower())
    safe_trait = re.sub(r'[\s]+', '_', safe_trait)[:50]  # Limit length
    # Add hash for uniqueness if trait is complex
    trait_hash = hashlib.md5(trait.encode()).hexdigest()[:8]
    return f"pairs_{safe_trait}_{trait_hash}_n{num_pairs}.json"


def execute_generate_vector_from_synthetic(args):
    """
    Execute the generate-vector-from-synthetic command - full pipeline in one command.

    Pipeline:
    1. Generate synthetic contrastive pairs for a trait (or load from cache)
    2. Collect activations from those pairs
    3. Create steering vectors from the activations
    """
    print(f"\n{'='*60}")
    print(f"🎯 Generating Steering Vector from Synthetic Pairs (Full Pipeline)")
    print(f"{'='*60}")
    print(f"   Trait: {args.trait}")
    print(f"   Model: {args.model}")
    print(f"   Num Pairs: {args.num_pairs}")
    print(f"{'='*60}\n")

    pipeline_start = time.time() if args.timing else None

    # Check for pairs cache
    pairs_cache_dir = getattr(args, 'pairs_cache_dir', None)
    force_regenerate = getattr(args, 'force_regenerate', False)
    use_cached_pairs = False
    cached_pairs_file = None

    if pairs_cache_dir and not force_regenerate:
        os.makedirs(pairs_cache_dir, exist_ok=True)
        cache_filename = _get_pairs_cache_filename(args.trait, args.num_pairs)
        cached_pairs_file = os.path.join(pairs_cache_dir, cache_filename)

        if os.path.exists(cached_pairs_file):
            print(f"📦 Found cached pairs: {cached_pairs_file}")
            use_cached_pairs = True
        else:
            print(f"📦 Pairs cache dir: {pairs_cache_dir}")
            print(f"   No cached pairs found, will generate and cache")

    try:
        # Determine intermediate file paths
        if args.intermediate_dir:
            intermediate_dir = args.intermediate_dir
        else:
            intermediate_dir = os.path.dirname(os.path.abspath(args.output))

        os.makedirs(intermediate_dir, exist_ok=True)

        # Create intermediate file paths
        if args.keep_intermediate:
            pairs_file = os.path.join(intermediate_dir, f"{args.trait.replace(' ', '_')}_pairs.json")
            enriched_file = os.path.join(intermediate_dir, f"{args.trait.replace(' ', '_')}_pairs_with_activations.json")
        else:
            # Use temporary files that will be deleted
            pairs_file = tempfile.NamedTemporaryFile(mode='w', suffix='_pairs.json', delete=False).name
            enriched_file = tempfile.NamedTemporaryFile(mode='w', suffix='_enriched.json', delete=False).name

        # Step 1: Generate synthetic pairs (or use cached)
        print(f"{'='*60}")
        if use_cached_pairs:
            print(f"Step 1/3: Using cached contrastive pairs...")
            pairs_file = cached_pairs_file
            print(f"   ✓ Loaded from: {cached_pairs_file}")
        else:
            print(f"Step 1/3: Generating synthetic contrastive pairs...")
        print(f"{'='*60}\n")
        
        if not use_cached_pairs:
            # Determine output path for pairs - use cache dir if specified
            if pairs_cache_dir:
                pairs_output = cached_pairs_file
            else:
                pairs_output = pairs_file

            pairs_args = Namespace(
                trait=args.trait,
                num_pairs=args.num_pairs,
                output=pairs_output,
                model=args.model,
                device=args.device,
                similarity_threshold=args.similarity_threshold,
                verbose=args.verbose,
                timing=args.timing,
                nonsense=getattr(args, 'nonsense', False),
                nonsense_mode=getattr(args, 'nonsense_mode', None),
            )

            execute_generate_pairs(pairs_args)

            # If we cached to a different location, use that as pairs_file
            if pairs_cache_dir:
                pairs_file = cached_pairs_file
                print(f"\n✓ Step 1 complete: Pairs cached to {pairs_file}")
            else:
                print(f"\n✓ Step 1 complete: Pairs saved to {pairs_file}")
        else:
            print(f"\n✓ Step 1 complete: Using cached pairs from {pairs_file}")
        
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
            accept_low_quality_vector=getattr(args, 'accept_low_quality_vector', False),
        )
        
        execute_create_steering_vector(vector_args)
        print(f"\n✓ Step 3 complete: Steering vector saved to {args.output}\n")
        
        # Clean up intermediate files if not keeping them
        # Never delete cached pairs (they're meant to be reused)
        if not args.keep_intermediate:
            if args.verbose:
                print(f"\n🧹 Cleaning up intermediate files...")
            try:
                # Only delete pairs_file if it's NOT in the cache dir
                if not (pairs_cache_dir and pairs_file == cached_pairs_file):
                    if os.path.exists(pairs_file):
                        os.unlink(pairs_file)
                        if args.verbose:
                            print(f"   ✓ Removed temporary pairs file")
                else:
                    if args.verbose:
                        print(f"   📦 Keeping cached pairs: {pairs_file}")
                # Always clean up enriched file if not keeping intermediate
                if os.path.exists(enriched_file):
                    os.unlink(enriched_file)
                    if args.verbose:
                        print(f"   ✓ Removed temporary enriched file")
            except Exception as e:
                if args.verbose:
                    print(f"   ⚠️  Warning: Could not remove some temporary files: {e}")
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"✅ Full Pipeline Completed Successfully!")
        print(f"{'='*60}")
        print(f"   Final steering vector: {args.output}")
        if pairs_cache_dir:
            print(f"   Cached pairs: {cached_pairs_file}")
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

        # Clean up on failure (but never delete cached pairs)
        if not args.keep_intermediate:
            try:
                # Only delete pairs_file if it's NOT in the cache dir
                if 'pairs_file' in locals() and os.path.exists(pairs_file):
                    if not (pairs_cache_dir and pairs_file == cached_pairs_file):
                        os.unlink(pairs_file)
                if 'enriched_file' in locals() and os.path.exists(enriched_file):
                    os.unlink(enriched_file)
            except:
                pass

        sys.exit(1)
