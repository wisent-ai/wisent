"""Generate pairs command execution logic - synthetic generation."""

import sys
import json
import os


def execute_generate_pairs(args):
    """Execute the generate-pairs command - generate synthetic contrastive pairs from trait description."""
    from wisent.core.models.wisent_model import WisentModel
    from wisent.core.synthetic.generators.pairs_generator import SyntheticContrastivePairsGenerator
    from wisent.core.synthetic.db_instructions.mini_dp import Default_DB_Instructions
    from wisent.core.synthetic.cleaners.pairs_cleaner import PairsCleaner
    from wisent.core.synthetic.cleaners.refusaler_cleaner import RefusalerCleaner
    from wisent.core.synthetic.cleaners.deduper_cleaner import DeduperCleaner
    from wisent.core.synthetic.generators.diversities.methods.fast_diversity import FastDiversity

    print(f"\n🎨 Generating synthetic contrastive pairs")
    print(f"   Trait: {args.trait}")
    print(f"   Number of pairs: {args.num_pairs}")
    print(f"   Model: {args.model}")

    try:
        # 1. Load model
        print(f"\n🤖 Loading model '{args.model}'...")
        model = WisentModel(args.model, device=args.device)
        print(f"   ✓ Model loaded with {model.num_layers} layers")

        # 2. Set up generation config
        # Scale max_new_tokens based on number of pairs (roughly 150 tokens per pair + buffer)
        estimated_tokens = args.num_pairs * 150 + 500
        max_tokens = max(2048, min(estimated_tokens, 8192))  # Between 2048 and 8192

        generation_config = {
            "max_new_tokens": max_tokens,
            "temperature": 0.9,
            "do_sample": True,
        }

        # 3. Set up cleaning pipeline
        print(f"\n🧹 Setting up cleaning pipeline...")
        from wisent.core.synthetic.cleaners.methods.base_dedupers import SimHashDeduper

        cleaning_steps = [
            DeduperCleaner(deduper=SimHashDeduper(threshold_bits=3)),
        ]
        cleaner = PairsCleaner(steps=cleaning_steps)

        # 4. Set up components
        db_instructions = Default_DB_Instructions()
        diversity = FastDiversity()

        # 5. Create generator
        print(f"\n⚙️  Initializing generator...")
        generator = SyntheticContrastivePairsGenerator(
            model=model,
            generation_config=generation_config,
            contrastive_set_name=f"synthetic_{args.trait[:20].replace(' ', '_')}",
            trait_description=args.trait,
            trait_label=args.trait[:50],
            db_instructions=db_instructions,
            cleaner=cleaner,
            diversity=diversity,
        )

        # 6. Generate pairs
        print(f"\n🎯 Generating {args.num_pairs} contrastive pairs...")
        if args.timing:
            import time
            start_time = time.time()

        pair_set, report = generator.generate(num_pairs=args.num_pairs)

        if args.timing:
            elapsed = time.time() - start_time
            print(f"   ⏱️  Generation time: {elapsed:.2f}s")

        print(f"   ✓ Generated {len(pair_set.pairs)} pairs")

        # 7. Print generation report
        if args.verbose and len(pair_set.pairs) > 0:
            print(f"\n📊 Generation Report:")
            print(f"   Requested: {report.requested}")
            print(f"   Kept after dedupe: {report.kept_after_dedupe}")
            print(f"   Retries for refusals: {report.retries_for_refusals}")
            if report.diversity:
                print(f"   Diversity:")
                print(f"     • Unique unigrams: {report.diversity.unique_unigrams:.3f}")
                print(f"     • Unique bigrams: {report.diversity.unique_bigrams:.3f}")
                print(f"     • Avg Jaccard: {report.diversity.avg_jaccard_prompt:.3f}")

        # 8. Convert pairs to dict format for JSON serialization
        print(f"\n💾 Saving pairs to '{args.output}'...")
        pairs_data = []
        for pair in pair_set.pairs:
            pair_dict = pair.to_dict()
            pairs_data.append(pair_dict)

        # 9. Save to JSON file
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        save_data = {
            'trait_description': args.trait,
            'trait_label': pair_set.task_type,
            'num_pairs': len(pairs_data),
            'generation_config': generation_config,
            'requested': report.requested,
            'kept_after_dedupe': report.kept_after_dedupe,
            'pairs': pairs_data
        }
        if report.diversity:
            save_data['diversity'] = {
                'unique_unigrams': report.diversity.unique_unigrams,
                'unique_bigrams': report.diversity.unique_bigrams,
                'avg_jaccard': report.diversity.avg_jaccard_prompt
            }

        with open(args.output, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"   ✓ Saved {len(pairs_data)} pairs to: {args.output}")
        print(f"\n✅ Synthetic pair generation completed successfully!\n")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
