"""Generate contrastive pairs from real human vs AI text datasets for humanization optimization."""

import json
import random
from argparse import Namespace
from typing import List, Dict

from datasets import load_dataset


def generate_humanization_pairs(
    num_pairs: int = 100,
    output_path: str = None,
    seed: int = 42,
    min_length: int = 100,
    max_length: int = 500,
) -> List[Dict]:
    """Generate contrastive pairs from GPT-wiki-intro dataset.
    
    Uses real human-written Wikipedia intros as positive examples
    and GPT-generated versions as negative examples.
    
    Args:
        num_pairs: Number of pairs to generate
        output_path: Path to save pairs JSON (optional)
        seed: Random seed for reproducibility
        min_length: Minimum text length in characters
        max_length: Maximum text length in characters
    
    Returns:
        List of contrastive pair dicts
    """
    random.seed(seed)
    
    print(f"Loading GPT-wiki-intro dataset...")
    ds = load_dataset('aadityaubhat/GPT-wiki-intro', split='train')
    print(f"  Loaded {len(ds)} examples")
    
    # Filter by length
    valid_indices = []
    for i, example in enumerate(ds):
        human_len = len(example['wiki_intro'])
        ai_len = len(example['generated_intro'])
        if min_length <= human_len <= max_length and min_length <= ai_len <= max_length:
            valid_indices.append(i)
    
    print(f"  {len(valid_indices)} examples within length range [{min_length}, {max_length}]")
    
    # Sample pairs
    if len(valid_indices) < num_pairs:
        print(f"  Warning: Only {len(valid_indices)} valid examples, using all")
        selected_indices = valid_indices
    else:
        selected_indices = random.sample(valid_indices, num_pairs)
    
    # Create contrastive pairs
    pairs = []
    for idx in selected_indices:
        example = ds[idx]
        
        # Human text is positive (what we want), AI text is negative
        pair = {
            "prompt": f"Write an introduction about: {example['title']}",
            "positive_response": {
                "model_response": example['wiki_intro'],
                "metadata": {
                    "source": "wikipedia",
                    "is_human": True,
                }
            },
            "negative_response": {
                "model_response": example['generated_intro'],
                "metadata": {
                    "source": "gpt-3",
                    "is_human": False,
                }
            },
            "metadata": {
                "title": example['title'],
                "dataset": "aadityaubhat/GPT-wiki-intro",
            }
        }
        pairs.append(pair)
    
    print(f"  Generated {len(pairs)} contrastive pairs")
    
    # Save if output path provided
    if output_path:
        output_data = {
            "pairs": pairs,
            "metadata": {
                "source": "aadityaubhat/GPT-wiki-intro",
                "num_pairs": len(pairs),
                "positive_label": "human_written",
                "negative_label": "ai_generated",
                "min_length": min_length,
                "max_length": max_length,
            }
        }
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"  Saved to {output_path}")
    
    return pairs


def execute_generate_humanization_pairs(args):
    """CLI entry point."""
    pairs = generate_humanization_pairs(
        num_pairs=args.num_pairs,
        output_path=args.output,
        seed=getattr(args, 'seed', 42),
        min_length=getattr(args, 'min_length', 100),
        max_length=getattr(args, 'max_length', 500),
    )
    return pairs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-pairs", type=int, default=100)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-length", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=500)
    args = parser.parse_args()
    execute_generate_humanization_pairs(args)
