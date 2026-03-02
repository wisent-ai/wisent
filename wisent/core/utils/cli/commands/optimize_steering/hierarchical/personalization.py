"""Personalization steering optimization."""
import json
import os
import tempfile
from typing import Optional

from wisent.core.utils.cli.optimize_steering.method_configs import CAAConfig
from wisent.core.utils.cli.optimize_steering.pipeline import run_pipeline
from wisent.core.utils.config_tools.constants import (JSON_INDENT, PERSONALIZATION_N_TRIALS, DEFAULT_N_TRIALS, WELFARE_LIMIT,
    DEFAULT_NUM_HIDDEN_LAYERS, DEFAULT_NUM_STRENGTH_STEPS,
    PARSER_STRENGTH_RANGE_PERSONALIZATION, SEPARATOR_WIDTH_REPORT,
    LAYER_STRIDE_DEFAULT)


def _execute_personalization_optimization(args):
    """
    Execute personalization steering optimization.

    This generates synthetic pairs for the specified personality trait
    and optimizes steering parameters.
    """
    import optuna

    trait = args.trait
    trait_name = getattr(args, 'trait_name', trait.split()[0].lower())
    model = args.model
    num_pairs = getattr(args, 'num_pairs', PERSONALIZATION_N_TRIALS)
    n_trials = getattr(args, 'n_trials', DEFAULT_N_TRIALS)
    limit = getattr(args, 'limit', WELFARE_LIMIT)
    device = getattr(args, 'device', None)
    output_dir = getattr(args, 'output_dir', './personalization_optimization')

    print(f"\n{'=' * SEPARATOR_WIDTH_REPORT}")
    print(f"🎭 PERSONALIZATION STEERING OPTIMIZATION")
    print(f"{'=' * SEPARATOR_WIDTH_REPORT}")
    print(f"   Model: {model}")
    print(f"   Trait: {trait}")
    print(f"   Trait Name: {trait_name}")
    print(f"   Num Pairs: {num_pairs}")
    print(f"   Trials: {n_trials}")
    print(f"   Output: {output_dir}")
    print(f"{'=' * SEPARATOR_WIDTH_REPORT}\n")

    # Try to load existing personalization pairs first
    try:
        from wisent.data.contrastive_pairs import load_personalization_pairs, TRAIT_DIRS
        trait_lower = trait_name.lower().replace("-", "_").replace(" ", "_")
        if trait_lower in TRAIT_DIRS:
            pair_set = load_personalization_pairs(trait_lower, return_backend='list')
            print(f"   Loaded {len(pair_set.pairs)} existing pairs for '{trait_lower}'")
        else:
            raise FileNotFoundError(f"No pre-generated pairs for '{trait}'")
    except FileNotFoundError:
        # Generate synthetic pairs
        print(f"   Generating {num_pairs} synthetic pairs for '{trait}'...")
        from wisent.core.primitives.contrastive_pairs.synthetic import generate_trait_pairs

        pairs_list = generate_trait_pairs(
            trait_description=trait,
            num_pairs=num_pairs,
            model=model,
        )

        from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet
        pair_set = ContrastivePairSet(
            name=f"{trait_name}_personalization",
            pairs=pairs_list,
            task_type="personalization",
        )
        print(f"   Generated {len(pair_set.pairs)} pairs")

    # Save pairs to temp file
    os.makedirs(output_dir, exist_ok=True)
    pairs_file = os.path.join(output_dir, f"{trait_name}_pairs.json")

    from wisent.core.primitives.contrastive_pairs.core.io.serialization import save_contrastive_pair_set
    save_contrastive_pair_set(pair_set, pairs_file)

    # Get model's number of layers
    from transformers import AutoConfig
    try:
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        num_layers = getattr(config, 'num_hidden_layers', DEFAULT_NUM_HIDDEN_LAYERS)
    except Exception:
        num_layers = DEFAULT_NUM_HIDDEN_LAYERS

    # Determine layers to search
    layers = getattr(args, 'layers', None)
    if layers is None:
        layers = list(range(0, num_layers, LAYER_STRIDE_DEFAULT))

    # Strength range
    strength_range = getattr(args, 'strength_range', list(PARSER_STRENGTH_RANGE_PERSONALIZATION))
    num_strength_steps = getattr(args, 'num_strength_steps', DEFAULT_NUM_STRENGTH_STEPS)
    strengths = [
        strength_range[0] + i * (strength_range[1] - strength_range[0]) / (num_strength_steps - 1)
        for i in range(num_strength_steps)
    ]

    print(f"   Layers to search: {layers}")
    print(f"   Strengths to search: {[f'{s:.2f}' for s in strengths]}")

    # Grid search (simpler for personalization)
    best_score = 0.0
    best_params = {}
    total_configs = len(layers) * len(strengths)
    current = 0

    for layer in layers:
        for strength in strengths:
            current += 1
            config = CAAConfig(
                method="CAA",
                layer=layer,
                extraction_strategy="chat_last",
                steering_strategy="constant",
            )

            try:
                with tempfile.TemporaryDirectory() as work_dir:
                    result = run_pipeline(
                        model=model,
                        task="personalization",
                        config=config,
                        work_dir=work_dir,
                        limit=min(limit, len(pair_set.pairs)),
                        device=device,
                        strength=strength,
                    )
                    if result.score > best_score:
                        best_score = result.score
                        best_params = {
                            "layer": layer,
                            "strength": strength,
                        }
                        print(f"   [{current}/{total_configs}] New best: {best_score:.4f} @ layer={layer}, strength={strength:.2f}")
            except Exception as e:
                print(f"   [{current}/{total_configs}] Failed: {e}")

    # Print results
    print(f"\n{'=' * SEPARATOR_WIDTH_REPORT}")
    print(f"📊 PERSONALIZATION OPTIMIZATION COMPLETE")
    print(f"{'=' * SEPARATOR_WIDTH_REPORT}")
    print(f"\n✅ Best configuration for '{trait}':")
    print(f"   Score: {best_score:.4f}")
    for k, v in best_params.items():
        print(f"   {k}: {v}")

    # Save results
    results_file = os.path.join(output_dir, f"{trait_name}_results.json")
    output_data = {
        "model": model,
        "trait": trait,
        "trait_name": trait_name,
        "best_score": best_score,
        "best_params": best_params,
    }
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=JSON_INDENT)
    print(f"\n💾 Results saved to: {results_file}")

    return output_data

