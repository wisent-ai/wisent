"""Generate synthetic contrastive pairs for agent."""

from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.synthetic.generators.core.atoms import GenerationReport
from wisent.core.errors import PairGenerationError
from wisent.core.constants import AGENT_MAX_WORKERS, GENERATE_PAIRS_MIN_TOKENS, GENERATE_PAIRS_MAX_TOKENS, SIMHASH_DEFAULT_THRESHOLD_BITS, SYNTHETIC_PAIRS_TEMPERATURE, DISPLAY_TRUNCATION_SHORT, TRAIT_LABEL_MAX_LENGTH, MIN_SYNTHETIC_PAIRS, SYNTHETIC_PAIRS_TIME_MULTIPLIER, TOKENS_PER_PAIR_SYNTHETIC, TOKENS_BASE_SYNTHETIC


def generate_synthetic_pairs(
    model,
    prompt: str,
    time_budget: float,
    verbose: bool = False,
    num_pairs: int = None,
    similarity_threshold: float = None,
    max_workers: int = AGENT_MAX_WORKERS
) -> tuple[ContrastivePairSet, GenerationReport]:
    """
    Generate synthetic contrastive pairs for the given prompt.

    Uses the same logic as the generate-pairs CLI command.

    arguments:
        model:
            WisentModel instance to use for generation
        prompt:
            The trait/behavior to generate pairs for
        time_budget:
            Time budget in minutes
        verbose:
            Enable verbose output
        num_pairs:
            Override number of pairs (default: based on time_budget)
        similarity_threshold:
            Similarity threshold for deduplication (default: None uses SimHash threshold_bits=3)
        max_workers:
            Number of parallel workers (default: 4, currently not used by generator)

    returns:
        Tuple of (ContrastivePairSet, GenerationReport)
    """
    from wisent.core.synthetic.generators.pairs_generator import SyntheticContrastivePairsGenerator
    from wisent.core.synthetic.cleaners.pairs_cleaner import PairsCleaner
    from wisent.core.synthetic.cleaners.deduper_cleaner import DeduperCleaner
    from wisent.core.synthetic.cleaners.methods.base_dedupers import SimHashDeduper
    from wisent.core.synthetic.db_instructions.mini_dp import Default_DB_Instructions
    from wisent.core.synthetic.generators.diversities.methods.fast_diversity import FastDiversity

    print(f"\n📝 Step 1: Creating contrastive pairs synthetically")
    print(f"   Trait: {prompt}")
    print(f"   Time budget: {time_budget} minutes")

    # Determine number of pairs: use parameter override or estimate from time budget
    if num_pairs is None:
        num_pairs = max(MIN_SYNTHETIC_PAIRS, int(time_budget * SYNTHETIC_PAIRS_TIME_MULTIPLIER))
        print(f"   Generating {num_pairs} pairs (based on time budget)")
    else:
        print(f"   Generating {num_pairs} pairs (user specified)")

    print(f"   ✓ Model loaded with {model.num_layers} layers")

    # Scale max_new_tokens based on number of pairs (same as generate-pairs CLI)
    estimated_tokens = num_pairs * TOKENS_PER_PAIR_SYNTHETIC + TOKENS_BASE_SYNTHETIC
    max_tokens = max(GENERATE_PAIRS_MIN_TOKENS, min(estimated_tokens, GENERATE_PAIRS_MAX_TOKENS))

    generation_config = {
        "max_new_tokens": max_tokens,
        "temperature": SYNTHETIC_PAIRS_TEMPERATURE,
        "do_sample": True,
    }

    # Set up cleaning pipeline (same as generate-pairs CLI)
    print(f"\n🧹 Setting up cleaning pipeline...")

    # Use similarity threshold if provided, otherwise use SimHash threshold_bits
    if similarity_threshold is not None:
        print(f"   Using similarity threshold: {similarity_threshold}")
        deduper = SimHashDeduper(threshold=similarity_threshold)
    else:
        print(f"   Using SimHash deduper with threshold_bits={SIMHASH_DEFAULT_THRESHOLD_BITS}")
        deduper = SimHashDeduper(threshold_bits=SIMHASH_DEFAULT_THRESHOLD_BITS)

    cleaning_steps = [
        DeduperCleaner(deduper=deduper),
    ]
    cleaner = PairsCleaner(steps=cleaning_steps)

    # Set up components
    db_instructions = Default_DB_Instructions()
    diversity = FastDiversity()

    print(f"\n⚙️  Initializing generator...")

    generator = SyntheticContrastivePairsGenerator(
        model=model,
        generation_config=generation_config,
        contrastive_set_name=f"agent_synthetic_{prompt[:TRAIT_LABEL_MAX_LENGTH].replace(' ', '_')}",
        trait_description=prompt,
        trait_label=prompt[:DISPLAY_TRUNCATION_SHORT],
        db_instructions=db_instructions,
        cleaner=cleaner,
        diversity=diversity,
        nonsense_mode=None,
    )

    print(f"\n🎯 Generating {num_pairs} contrastive pairs...")
    pair_set, report = generator.generate(num_pairs=num_pairs)
    print(f"   ✓ Generated {len(pair_set.pairs)} pairs")

    if verbose:
        print(f"\n📊 Generation Report:")
        print(f"   Requested: {report.requested}")
        print(f"   Kept after dedupe: {report.kept_after_dedupe}")
        print(f"   Retries for refusals: {report.retries_for_refusals}")
        if report.diversity:
            print(f"   Diversity:")
            print(f"     • Unique unigrams: {report.diversity.unique_unigrams:.3f}")
            print(f"     • Unique bigrams: {report.diversity.unique_bigrams:.3f}")
            print(f"     • Avg Jaccard: {report.diversity.avg_jaccard_prompt:.3f}")

    if len(pair_set.pairs) == 0:
        raise PairGenerationError(reason="Failed to generate pairs")

    return pair_set, report
