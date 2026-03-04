"""Generate synthetic contrastive pairs for agent."""

from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.control.generation.synthetic.generators.core.atoms import GenerationReport
from wisent.core.utils.infra_tools.errors import PairGenerationError
from wisent.core.primitives.models import get_generate_kwargs
from wisent.core.utils.config_tools.constants import DISPLAY_TRUNCATION_SHORT


def generate_synthetic_pairs(
    model,
    prompt: str,
    time_budget: float,
    max_workers: int,
    trait_label_max_length: int,
    verbose: bool = False,
    similarity_threshold: float = None,
    *,
    agent_synth_min_pairs: int,
    agent_synth_time_multiplier: int,
    generate_pairs_min_tokens: int,
    simhash_default_threshold_bits: int,
    tokens_per_pair_estimate: int,
    tokens_base_offset: int,
    dedup_word_ngram: int,
    dedup_char_ngram: int,
    simhash_num_bands: int,
    fast_diversity_seed: int,
    diversity_max_sample_size: int,
    retry_multiplier: int,
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
        max_workers:
            Number of parallel workers
        verbose:
            Enable verbose output
        similarity_threshold:
            Similarity threshold for deduplication (default: None uses SimHash threshold_bits)

    returns:
        Tuple of (ContrastivePairSet, GenerationReport)
    """
    from wisent.core.control.generation.synthetic.generators.pairs_generator import SyntheticContrastivePairsGenerator
    from wisent.core.control.generation.synthetic.cleaners.pairs_cleaner import PairsCleaner
    from wisent.core.control.generation.synthetic.cleaners.deduper_cleaner import DeduperCleaner
    from wisent.core.control.generation.synthetic.cleaners.methods.base_dedupers import SimHashDeduper
    from wisent.core.control.generation.synthetic.db_instructions.mini_dp import Default_DB_Instructions
    from wisent.core.control.generation.synthetic.generators.diversities.methods.fast_diversity import FastDiversity

    print(f"\n📝 Step 1: Creating contrastive pairs synthetically")
    print(f"   Trait: {prompt}")
    print(f"   Time budget: {time_budget} minutes")

    # Determine number of pairs from time budget
    num_pairs = max(agent_synth_min_pairs, int(time_budget * agent_synth_time_multiplier))
    print(f"   Generating {num_pairs} pairs (based on time budget)")

    print(f"   ✓ Model loaded with {model.num_layers} layers")

    # Scale max_new_tokens based on number of pairs (same as generate-pairs CLI)
    gen_kwargs = get_generate_kwargs()
    estimated_tokens = num_pairs * tokens_per_pair_estimate + tokens_base_offset
    max_tokens = max(generate_pairs_min_tokens, min(estimated_tokens, gen_kwargs["max_new_tokens"]))

    generation_config = {**gen_kwargs, "max_new_tokens": max_tokens}

    # Set up cleaning pipeline (same as generate-pairs CLI)
    print(f"\n🧹 Setting up cleaning pipeline...")

    # Use similarity threshold if provided, otherwise use SimHash threshold_bits
    if similarity_threshold is not None:
        print(f"   Using similarity threshold (bits): {simhash_default_threshold_bits}")
    tb = simhash_default_threshold_bits
    print(f"   Using SimHash deduper with threshold_bits={tb}")
    deduper = SimHashDeduper(
        threshold_bits=tb,
        word_ngram=dedup_word_ngram,
        char_ngram=dedup_char_ngram,
        num_bands=simhash_num_bands,
    )

    cleaning_steps = [
        DeduperCleaner(deduper=deduper),
    ]
    cleaner = PairsCleaner(steps=cleaning_steps)

    # Set up components
    db_instructions = Default_DB_Instructions()
    diversity = FastDiversity(seed=fast_diversity_seed, max_sample_size=diversity_max_sample_size)

    print(f"\n⚙️  Initializing generator...")

    generator = SyntheticContrastivePairsGenerator(
        model=model,
        generation_config=generation_config,
        contrastive_set_name=f"agent_synthetic_{prompt[:trait_label_max_length].replace(' ', '_')}",
        trait_description=prompt,
        trait_label=prompt[:DISPLAY_TRUNCATION_SHORT],
        db_instructions=db_instructions,
        cleaner=cleaner,
        diversity=diversity,
        retry_multiplier=retry_multiplier,
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
