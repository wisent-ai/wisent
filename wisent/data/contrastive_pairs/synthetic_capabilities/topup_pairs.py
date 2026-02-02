#!/usr/bin/env python3
"""Top up existing capability pair files to reach target count."""

import json
import sys
import os

# Add the wisent package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from wisent.core.models.wisent_model import WisentModel
from wisent.core.synthetic.generators.pairs_generator import SyntheticContrastivePairsGenerator
from wisent.core.synthetic.db_instructions.mini_dp import Default_DB_Instructions
from wisent.core.synthetic.cleaners.pairs_cleaner import PairsCleaner
from wisent.core.synthetic.cleaners.deduper_cleaner import DeduperCleaner
from wisent.core.synthetic.cleaners.methods.base_dedupers import SimHashDeduper
from wisent.core.synthetic.generators.diversities.methods.fast_diversity import FastDiversity
from wisent.core.models import get_generate_kwargs

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

def topup_capability(capability: str, trait: str, base_dir: str):
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
    estimated_tokens = needed * 150 + 500
    max_tokens = max(2048, min(estimated_tokens, 8192))
    generation_config = get_generate_kwargs(max_new_tokens=max_tokens)

    cleaning_steps = [
        DeduperCleaner(deduper=SimHashDeduper(threshold_bits=10)),
    ]
    cleaner = PairsCleaner(steps=cleaning_steps)
    db_instructions = Default_DB_Instructions()
    diversity = FastDiversity()

    generator = SyntheticContrastivePairsGenerator(
        model=model,
        generation_config=generation_config,
        contrastive_set_name=f"synthetic_{capability}",
        trait_description=trait,
        trait_label=trait[:50],
        db_instructions=db_instructions,
        cleaner=cleaner,
        diversity=diversity,
        nonsense_mode=None,
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
        json.dump(data, f, indent=2)

    print(f"✓ {capability}: now has {len(all_pairs)} pairs")

    # Cleanup model to free memory
    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Check which capabilities need topping up
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

    print(f"\nStarting top-up process...")

    for cap, trait, _ in to_topup:
        topup_capability(cap, trait, base_dir)

    print(f"\n{'='*60}")
    print("ALL DONE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
