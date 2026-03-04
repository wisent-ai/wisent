#!/usr/bin/env python3
"""Generate capability pairs for all models."""

import json
import sys
import os

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

# Models to generate for (excluding llama_3.2_1b and llama_3.1_8b which are already done)
MODELS = [
    # ("meta-llama/Llama-3.1-8B-Instruct", "llama_3.1_8b_pairs.json"),  # Complete
    ("Qwen/Qwen3-8B", "qwen3_8b_pairs.json"),
    # ("gpt-oss-20b", "gpt_oss_20b_pairs.json"),  # Using separate API script
]

TARGET_COUNT = 100
DEVICE = "mps"


def generate_for_model(
    model_name: str,
    filename: str,
    base_dir: str,
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
    """Generate pairs for all capabilities for a single model."""
    print(f"\n{'='*70}")
    print(f"GENERATING FOR MODEL: {model_name}")
    print(f"{'='*70}")

    # Load model once
    print(f"Loading model {model_name}...")
    try:
        model = WisentModel(model_name, device=DEVICE)
        print(f"Model loaded with {model.num_layers} layers")
    except Exception as e:
        print(f"ERROR loading model {model_name}: {e}")
        return

    for capability, trait in CAPABILITIES.items():
        filepath = os.path.join(base_dir, capability, filename)

        # Skip if already exists
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            if len(data.get('pairs', [])) >= TARGET_COUNT:
                print(f"✓ {capability}: already has {len(data['pairs'])} pairs, skipping")
                continue

        print(f"\n{'-'*60}")
        print(f"Generating {capability}: {trait[:DISPLAY_TRUNCATION_SHORT]}...")
        print(f"{'-'*60}")

        try:
            # Set up generator
            estimated_tokens = TARGET_COUNT * tokens_per_pair_estimate + tokens_base_offset
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

            # Generate pairs
            pair_set, report = generator.generate(num_pairs=TARGET_COUNT)
            print(f"Generated {len(pair_set.pairs)} pairs")

            # Convert to dict format
            pairs_data = [pair.to_dict() for pair in pair_set.pairs]

            # Build save data with model info
            save_data = {
                'model': model_name,
                'trait_description': trait,
                'trait_label': trait[:DISPLAY_TRUNCATION_SHORT],
                'num_pairs': len(pairs_data),
                'requested': report.requested,
                'kept_after_dedupe': report.kept_after_dedupe,
                'generation_config': generation_config,
                'pairs': pairs_data
            }

            if report.diversity:
                save_data['diversity'] = {
                    'unique_unigrams': report.diversity.unique_unigrams,
                    'unique_bigrams': report.diversity.unique_bigrams,
                    'avg_jaccard': report.diversity.avg_jaccard_prompt,
                }

            # Save
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=JSON_INDENT)

            print(f"✓ Saved {len(pairs_data)} pairs to {filepath}")

        except Exception as e:
            print(f"ERROR generating {capability}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Cleanup
    del model
    import torch
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate capability pairs for all models")
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

    for model_name, filename in MODELS:
        generate_for_model(
            model_name, filename, base_dir,
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

    print("ALL DONE!")


if __name__ == "__main__":
    main()
