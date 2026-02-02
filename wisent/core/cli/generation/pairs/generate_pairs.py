"""Generate pairs command execution logic - synthetic generation."""

import sys
import json
import os

from wisent.core.models import get_generate_kwargs
from wisent.data.contrastive_pairs import save_personalization_pairs, save_synthetic_pairs


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

    # Check if nonsense mode is enabled
    nonsense_mode = getattr(args, 'nonsense_mode', None) if getattr(args, 'nonsense', False) else None

    try:
        # If nonsense mode, use programmatic generator (no LLM needed!)
        if nonsense_mode:
            print(f"   🎲 Nonsense mode: {nonsense_mode} (programmatic generation)")
            from wisent.core.synthetic.generators.nonsense_generator import ProgrammaticNonsenseGenerator
            from wisent.core.synthetic.generators.core.atoms import GenerationReport

            # 1. Create programmatic generator
            print(f"\n⚙️  Initializing programmatic nonsense generator...")
            generator = ProgrammaticNonsenseGenerator(
                nonsense_mode=nonsense_mode,
                contrastive_set_name=f"nonsense_{nonsense_mode}_{args.trait[:20].replace(' ', '_')}",
                trait_label=args.trait[:50],
                trait_description=args.trait,
            )

            # 2. Generate pairs (fast - no LLM!)
            print(f"\n🎯 Generating {args.num_pairs} nonsense pairs programmatically...")
            if args.timing:
                import time
                start_time = time.time()

            pair_set = generator.generate(num_pairs=args.num_pairs)

            if args.timing:
                elapsed = time.time() - start_time
                print(f"   ⏱️  Generation time: {elapsed:.2f}s")

            # Create a simple report (no diversity metrics for pure nonsense)
            report = GenerationReport(
                requested=args.num_pairs,
                kept_after_dedupe=len(pair_set.pairs),
                retries_for_refusals=0,
                diversity=None,
            )

        else:
            # Normal LLM-based generation
            print(f"   Model: {args.model}")

            # 1. Load model
            print(f"\n🤖 Loading model '{args.model}'...")
            model = WisentModel(args.model, device=args.device)
            print(f"   ✓ Model loaded with {model.num_layers} layers")

            # 2. Set up generation config
            # Scale max_new_tokens based on number of pairs (roughly 150 tokens per pair + buffer)
            estimated_tokens = args.num_pairs * 150 + 500
            max_tokens = max(2048, min(estimated_tokens, 8192))  # Between 2048 and 8192

            # Get generation config from centralized inference config
            generation_config = get_generate_kwargs(max_new_tokens=max_tokens)

            # 3. Set up cleaning pipeline
            print(f"\n🧹 Setting up cleaning pipeline...")
            from wisent.core.synthetic.cleaners.methods.base_dedupers import SimHashDeduper

            cleaning_steps = [
                DeduperCleaner(deduper=SimHashDeduper(threshold_bits=10)),  # Relaxed threshold to keep more diverse pairs
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
                nonsense_mode=None,  # Not used for LLM-based generation
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
            'requested': report.requested,
            'kept_after_dedupe': report.kept_after_dedupe,
            'pairs': pairs_data
        }

        # Only add generation_config if it was used (LLM-based generation)
        if not nonsense_mode:
            save_data['generation_config'] = generation_config
        else:
            save_data['generation_method'] = f'programmatic_nonsense_{nonsense_mode}'
        if report.diversity:
            save_data['diversity'] = {
                'unique_unigrams': report.diversity.unique_unigrams,
                'unique_bigrams': report.diversity.unique_bigrams,
                'avg_jaccard': report.diversity.avg_jaccard_prompt
            }

        with open(args.output, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"   ✓ Saved {len(pairs_data)} pairs to: {args.output}")

        # Also save to centralized storage if requested or by default for personalization traits
        store_centrally = getattr(args, 'store_centrally', True)
        if store_centrally and len(pairs_data) > 0:
            try:
                # Determine if this is a personalization trait
                trait_lower = args.trait.lower()
                known_traits = ['british', 'evil', 'flirty', 'left_wing', 'left-wing', 'leftist']
                is_personalization = any(t in trait_lower for t in known_traits)

                metadata = {
                    'source': 'generate_pairs_cli',
                    'model': getattr(args, 'model', 'unknown'),
                    'requested': report.requested,
                    'kept_after_dedupe': report.kept_after_dedupe,
                }
                if report.diversity:
                    metadata['diversity'] = {
                        'unique_unigrams': report.diversity.unique_unigrams,
                        'unique_bigrams': report.diversity.unique_bigrams,
                        'avg_jaccard': report.diversity.avg_jaccard_prompt,
                    }

                if is_personalization:
                    # Map to trait directory name
                    if 'british' in trait_lower:
                        trait_name = 'british'
                    elif 'evil' in trait_lower:
                        trait_name = 'evil'
                    elif 'flirty' in trait_lower:
                        trait_name = 'flirty'
                    elif 'left' in trait_lower:
                        trait_name = 'left_wing'
                    else:
                        trait_name = 'custom'

                    stored_path = save_personalization_pairs(
                        pairs=pair_set,
                        trait=trait_name,
                        model=getattr(args, 'model', None),
                        metadata=metadata,
                    )
                    print(f"   ✓ Also stored in centralized location: {stored_path}")
                else:
                    stored_path = save_synthetic_pairs(
                        pairs=pair_set,
                        name=args.trait[:30].replace(' ', '_'),
                        model=getattr(args, 'model', None),
                        metadata=metadata,
                    )
                    print(f"   ✓ Also stored in centralized location: {stored_path}")
            except Exception as e:
                print(f"   ⚠ Warning: Could not save to centralized storage: {e}")

        print(f"\n✅ Synthetic pair generation completed successfully!\n")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
