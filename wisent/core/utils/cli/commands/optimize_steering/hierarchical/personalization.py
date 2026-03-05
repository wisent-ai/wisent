"""Personalization steering optimization."""
import json
import os
import tempfile

from wisent.core.control.steering_methods.configs.validated_defaults import VALIDATED_EXTRACTION_STRATEGY
from wisent.core.utils.cli.optimize_steering.method_configs import CAAConfig
from wisent.core.utils.cli.optimize_steering.pipeline import run_pipeline
from wisent.core.utils.config_tools.constants import (JSON_INDENT,
    SEPARATOR_WIDTH_REPORT,
    PARSER_DEFAULT_LAYER_START)


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
    num_pairs = getattr(args, 'num_pairs', None)
    if num_pairs is None:
        raise ValueError("num_pairs is required (set via --num-pairs)")
    n_trials = args.n_trials
    limit = getattr(args, 'limit', None)
    if limit is None:
        raise ValueError("limit is required (set via --limit)")
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
        num_layers = getattr(config, 'num_hidden_layers', None)
        if num_layers is None:
            raise ValueError(f"num_layers must be specified: model '{model}' config has no num_hidden_layers")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"num_layers must be specified: failed to load model config ({e})")

    # Determine layers to search
    layer_stride = args.layer_stride
    layers = list(range(PARSER_DEFAULT_LAYER_START, num_layers, layer_stride))

    # Strength range
    strength_range = args.strength_range
    num_strength_steps = args.num_strength_steps
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
                extraction_strategy=VALIDATED_EXTRACTION_STRATEGY,
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

