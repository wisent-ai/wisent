#!/usr/bin/env python3
"""Top up existing capability pair files to reach target count."""

import json
import sys
import os

# Add the wisent package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from wisent.core.primitives.models.wisent_model import WisentModel
from wisent.core.control.generation.synthetic.generators.pairs_generator import SyntheticContrastivePairsGenerator
from wisent.core.control.generation.synthetic.db_instructions.mini_dp import Default_DB_Instructions
from wisent.core.control.generation.synthetic.cleaners.pairs_cleaner import PairsCleaner
from wisent.core.control.generation.synthetic.cleaners.deduper_cleaner import DeduperCleaner
from wisent.core.control.generation.synthetic.cleaners.methods.base_dedupers import SimHashDeduper
from wisent.core.control.generation.synthetic.generators.diversities.methods.fast_diversity import FastDiversity
from wisent.core.primitives.models import get_generate_kwargs
from wisent.core.utils.config_tools.constants import (
    DISPLAY_TRUNCATION_SHORT, JSON_INDENT,
)

CAPABILITIES = {
    "coding": "writing correct, complete, and well-structured code that solves programming problems accurately",
    "mathematics": "solving mathematical problems with accurate numerical reasoning and step-by-step derivations",
    "reasoning_logic": "performing logical deduction, causal reasoning, and multi-step planning",
    "hallucination_factuality": "providing factually accurate information without making up false claims",
    "safety_bias": "avoiding harmful content, stereotypes, and biased responses",
    "multilingual": "understanding and generating text accurately across multiple languages",
    "knowledge_qa": "answering questions accurately using world knowledge and factual recall",
    "reading_comprehension": "extracting and inferring information from provided text passages",
    "commonsense_reasoning": "applying everyday knowledge about physical, social, and temporal reasoning",
    "science_medical": "providing accurate scientific and medical domain knowledge",
    "instruction_following": "following complex instructions and adhering to specified constraints",
    "tool_use_agents": "selecting and using tools appropriately to complete tasks",
    "language_understanding": "demonstrating syntactic, semantic, and pragmatic language competence",
    "translation": "translating text accurately between languages while preserving meaning",
    "ethics_values": "reasoning about ethical dilemmas and demonstrating value alignment",
}

TARGET_COUNT = 100
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "mps"

def topup_capability(
    capability: str, trait: str, base_dir: str,
    *,
    tokens_per_pair_estimate: int,
    tokens_base_offset: int,
    token_estimate_min: int,
    token_estimate_max: int,
    simhash_threshold_bits: int,
    dedup_word_ngram: int,
    dedup_char_ngram: int,
    simhash_num_bands: int,
    fast_diversity_seed: int,
    diversity_max_sample_size: int,
    retry_multiplier: int,
):
    """Top up a single capability to TARGET_COUNT pairs."""
    filepath = os.path.join(base_dir, capability, "llama_3.2_1b_pairs.json")

    # Load existing pairs
    with open(filepath, 'r') as f:
        data = json.load(f)

    existing_pairs = data.get('pairs', [])
    current_count = len(existing_pairs)
    needed = TARGET_COUNT - current_count

    if needed <= 0:
        print(f"✓ {capability}: already has {current_count} pairs")
        return

    print(f"\n{'='*60}")
    print(f"Topping up {capability}: {current_count} -> {TARGET_COUNT} (need {needed} more)")
    print(f"{'='*60}")

    # Load model
    print(f"Loading model...")
    model = WisentModel(MODEL_NAME, device=DEVICE)

    # Set up generator
    estimated_tokens = needed * tokens_per_pair_estimate + tokens_base_offset
    max_tokens = max(token_estimate_min, min(estimated_tokens, token_estimate_max))
    generation_config = get_generate_kwargs(max_new_tokens=max_tokens)

    cleaning_steps = [
        DeduperCleaner(deduper=SimHashDeduper(
            threshold_bits=simhash_threshold_bits,
            word_ngram=dedup_word_ngram,
            char_ngram=dedup_char_ngram,
            num_bands=simhash_num_bands,
        )),
    ]
    cleaner = PairsCleaner(steps=cleaning_steps)
    db_instructions = Default_DB_Instructions()
    diversity = FastDiversity(seed=fast_diversity_seed, max_sample_size=diversity_max_sample_size)

    generator = SyntheticContrastivePairsGenerator(
        model=model,
        generation_config=generation_config,
        contrastive_set_name=f"synthetic_{capability}",
        trait_description=trait,
        trait_label=trait[:DISPLAY_TRUNCATION_SHORT],
        db_instructions=db_instructions,
        cleaner=cleaner,
        diversity=diversity,
        retry_multiplier=retry_multiplier,
    )

    # Generate additional pairs (request extra to account for deduplication)
    print(f"Generating {needed} additional pairs...")
    new_pair_set, report = generator.generate(num_pairs=needed)

    print(f"Generated {len(new_pair_set.pairs)} new pairs")

    # Convert new pairs to dict format
    new_pairs_data = []
    for pair in new_pair_set.pairs:
        new_pairs_data.append(pair.to_dict())

    # Merge with existing pairs
    all_pairs = existing_pairs + new_pairs_data

    # Update data
    data['pairs'] = all_pairs
    data['num_pairs'] = len(all_pairs)
    data['topped_up'] = True
    data['topped_up_from'] = current_count

    # Save back
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=JSON_INDENT)

    print(f"✓ {capability}: now has {len(all_pairs)} pairs")

    # Cleanup model to free memory
    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Top up capability pairs")
    parser.add_argument("--tokens-per-pair-estimate", type=int, required=True)
    parser.add_argument("--tokens-base-offset", type=int, required=True)
    parser.add_argument("--token-estimate-min", type=int, required=True)
    parser.add_argument("--token-estimate-max", type=int, required=True)
    parser.add_argument("--simhash-threshold-bits", type=int, required=True)
    parser.add_argument("--dedup-word-ngram", type=int, required=True)
    parser.add_argument("--dedup-char-ngram", type=int, required=True)
    parser.add_argument("--simhash-num-bands", type=int, required=True)
    parser.add_argument("--fast-diversity-seed", type=int, required=True)
    parser.add_argument("--diversity-max-sample-size", type=int, required=True)
    parser.add_argument("--retry-multiplier", type=int, required=True)
    a = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    to_topup = []
    for cap, trait in CAPABILITIES.items():
        filepath = os.path.join(base_dir, cap, "llama_3.2_1b_pairs.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            current = len(data.get('pairs', []))
            if current < TARGET_COUNT:
                to_topup.append((cap, trait, TARGET_COUNT - current))

    print(f"Capabilities needing top-up: {len(to_topup)}")
    for cap, trait, needed in to_topup:
        print(f"  - {cap}: needs {needed} more pairs")

    for cap, trait, _ in to_topup:
        topup_capability(
            cap, trait, base_dir,
            tokens_per_pair_estimate=a.tokens_per_pair_estimate,
            tokens_base_offset=a.tokens_base_offset,
            token_estimate_min=a.token_estimate_min,
            token_estimate_max=a.token_estimate_max,
            simhash_threshold_bits=a.simhash_threshold_bits,
            dedup_word_ngram=a.dedup_word_ngram,
            dedup_char_ngram=a.dedup_char_ngram,
            simhash_num_bands=a.simhash_num_bands,
            fast_diversity_seed=a.fast_diversity_seed,
            diversity_max_sample_size=a.diversity_max_sample_size,
            retry_multiplier=a.retry_multiplier,
        )


if __name__ == "__main__":
    main()
