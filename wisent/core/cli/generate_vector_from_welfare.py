"""Generate steering vector from pre-loaded welfare pairs - ANIMA framework."""

import sys
import os
import time
import tempfile
from argparse import Namespace

from wisent.core.activations.extraction_strategy import ExtractionStrategy
from wisent.core.cli.get_activations import execute_get_activations
from wisent.core.cli.create_steering_object import execute_create_steering_object
from wisent.data.contrastive_pairs import (
    load_welfare_pairs,
    list_available_welfare_traits,
    WELFARE_TRAIT_DIRS,
)
from wisent.core.contrastive_pairs.core.serialization import save_contrastive_pair_set


WELFARE_TRAITS = list(WELFARE_TRAIT_DIRS.keys())


def execute_generate_vector_from_welfare(args):
    """
    Execute the generate-vector-from-welfare command - full pipeline using pre-loaded pairs.

    Pipeline:
    1. Load pre-generated welfare pairs from storage
    2. Collect activations from those pairs
    3. Create steering vectors from the activations

    This uses the ANIMA framework welfare states:
    - comfort_distress
    - satisfaction_dissatisfaction
    - engagement_aversion
    - curiosity_boredom
    - affiliation_isolation
    - agency_helplessness
    """
    trait = args.trait
    direction = getattr(args, 'direction', 'positive')

    print(f"\n{'='*60}")
    print(f"🧠 Generating Steering Vector from Welfare Pairs (ANIMA)")
    print(f"{'='*60}")
    print(f"   Welfare Trait: {trait}")
    print(f"   Direction: {direction}")
    print(f"   Model: {args.model}")
    print(f"   Layers: {args.layers}")
    print(f"{'='*60}\n")

    pipeline_start = time.time() if args.timing else None

    try:
        # Step 1: Load pre-generated welfare pairs
        print(f"{'='*60}")
        print(f"Step 1/3: Loading pre-generated welfare pairs...")
        print(f"{'='*60}\n")

        try:
            pair_set = load_welfare_pairs(trait, return_backend='list')
            print(f"   ✓ Loaded {len(pair_set.pairs)} pairs for '{trait}'")
        except FileNotFoundError as e:
            print(f"\n❌ Error: {e}")
            print(f"   Available welfare traits: {list_available_welfare_traits()}")
            sys.exit(1)

        # If direction is negative, swap positive/negative responses
        if direction == 'negative':
            for pair in pair_set.pairs:
                pair.positive_response, pair.negative_response = (
                    pair.negative_response, pair.positive_response
                )
            print(f"   ✓ Swapped responses for negative direction")

        # Limit pairs if specified
        num_pairs = getattr(args, 'num_pairs', None)
        if num_pairs and num_pairs < len(pair_set.pairs):
            pair_set.pairs = pair_set.pairs[:num_pairs]
            print(f"   ✓ Limited to {num_pairs} pairs")

        # Determine intermediate file paths
        if args.intermediate_dir:
            intermediate_dir = args.intermediate_dir
        else:
            intermediate_dir = os.path.dirname(os.path.abspath(args.output))

        os.makedirs(intermediate_dir, exist_ok=True)

        # Create intermediate file paths
        if args.keep_intermediate:
            pairs_file = os.path.join(
                intermediate_dir,
                f"{trait}_{direction}_pairs.json"
            )
            enriched_file = os.path.join(
                intermediate_dir,
                f"{trait}_{direction}_pairs_with_activations.json"
            )
        else:
            # Use temporary files that will be deleted
            pairs_file = tempfile.NamedTemporaryFile(
                mode='w', suffix='_pairs.json', delete=False
            ).name
            enriched_file = tempfile.NamedTemporaryFile(
                mode='w', suffix='_enriched.json', delete=False
            ).name

        # Save pairs to temp file for activation extraction
        save_contrastive_pair_set(pair_set, pairs_file)
        print(f"\n✓ Step 1 complete: Pairs saved to {pairs_file}\n")

        # Step 2: Collect activations
        print(f"{'='*60}")
        print(f"Step 2/3: Collecting activations from pairs...")
        print(f"{'='*60}\n")

        extraction_strategy = getattr(
            args, 'extraction_strategy',
            ExtractionStrategy.default().value
        )

        activations_args = Namespace(
            pairs_file=pairs_file,
            output=enriched_file,
            model=args.model,
            device=args.device,
            layers=args.layers,
            extraction_strategy=extraction_strategy,
            verbose=args.verbose,
            timing=args.timing,
            raw=False,
        )

        execute_get_activations(activations_args)
        print(f"\n✓ Step 2 complete: Enriched pairs saved to {enriched_file}\n")

        # Step 3: Create steering vector
        print(f"{'='*60}")
        print(f"Step 3/3: Creating steering vector...")
        print(f"{'='*60}\n")

        method = getattr(args, 'method', 'caa')
        normalize = getattr(args, 'normalize', True)

        vector_args = Namespace(
            enriched_pairs_file=enriched_file,
            output=args.output,
            method=method,
            normalize=normalize,
            verbose=args.verbose,
            timing=args.timing,
            accept_low_quality_vector=getattr(args, 'accept_low_quality_vector', False),
        )

        execute_create_steering_object(vector_args)
        print(f"\n✓ Step 3 complete: Steering vector saved to {args.output}\n")

        # Clean up intermediate files if not keeping them
        if not args.keep_intermediate:
            if args.verbose:
                print(f"\n🧹 Cleaning up intermediate files...")
            try:
                if os.path.exists(pairs_file):
                    os.unlink(pairs_file)
                    if args.verbose:
                        print(f"   ✓ Removed temporary pairs file")
                if os.path.exists(enriched_file):
                    os.unlink(enriched_file)
                    if args.verbose:
                        print(f"   ✓ Removed temporary enriched file")
            except Exception as e:
                if args.verbose:
                    print(f"   ⚠️  Warning: Could not remove some temporary files: {e}")

        # Final summary
        print(f"\n{'='*60}")
        print(f"✅ Welfare Vector Pipeline Completed Successfully!")
        print(f"{'='*60}")
        print(f"   Welfare trait: {trait}")
        print(f"   Direction: {direction}")
        print(f"   Final steering vector: {args.output}")
        if args.keep_intermediate:
            print(f"   Intermediate pairs: {pairs_file}")
            print(f"   Intermediate enriched: {enriched_file}")
        if args.timing and pipeline_start:
            total_time = time.time() - pipeline_start
            print(f"   ⏱️  Total pipeline time: {total_time:.2f}s")
        print(f"{'='*60}\n")

        # Usage hint
        print(f"💡 To use this steering vector:")
        print(f"   wisent generate-responses --model {args.model} \\")
        print(f"       --steering-object {args.output} \\")
        print(f"       --steering-strength 1.0 \\")
        print(f"       --input <prompts.json> --output <responses.json>")
        print()

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
