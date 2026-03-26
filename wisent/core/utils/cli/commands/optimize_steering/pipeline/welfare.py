"""Welfare state steering optimization (ANIMA framework)."""
import json
import os
import tempfile
from typing import Optional

from wisent.core.utils.cli.optimize_steering.method_configs import (
    MethodConfig, CAAConfig, STEERING_STRATEGIES,
)
from wisent.core.utils.cli.optimize_steering.pipeline import OptimizationResult, _make_args
from wisent.core.utils.cli.optimize_steering.data.activations_data import execute_get_activations
from wisent.core.utils.cli.optimize_steering.steering_objects import execute_create_steering_object
from wisent.core.utils.cli.optimize_steering.data.responses import execute_generate_responses
from wisent.core.utils.cli.optimize_steering.scores import execute_evaluate_responses
from wisent.core.utils.config_tools.constants import (
    SEPARATOR_WIDTH_REPORT,
    JSON_INDENT, PARSER_DEFAULT_LAYER_START, TRIALS_PER_DIMENSION_MULTIPLIER)
from wisent.core.control.steering_methods.configs.optimal import get_optimal, get_optimal_extraction_strategy


def _execute_welfare_optimization(args):
    """
    Execute welfare state steering optimization (ANIMA framework).

    This loads pre-generated welfare pairs from wisent.data.contrastive_pairs.welfare/
    and optimizes steering parameters for the specified welfare trait.
    """
    import optuna
    from wisent.data.contrastive_pairs import load_welfare_pairs, WELFARE_TRAIT_DIRS

    trait = args.trait
    direction = getattr(args, 'direction', None)
    if direction is None:
        raise ValueError("Parameter 'direction' is required for welfare optimization.")
    model = args.model
    device = getattr(args, 'device', None)
    output_dir = getattr(args, 'output_dir', './welfare_optimization')

    print(f"\n{'=' * SEPARATOR_WIDTH_REPORT}")
    print(f"🧠 WELFARE STATE STEERING OPTIMIZATION (ANIMA)")
    print(f"{'=' * SEPARATOR_WIDTH_REPORT}")
    print(f"   Model: {model}")
    print(f"   Welfare Trait: {trait}")
    print(f"   Direction: {direction}")
    print(f"   Output: {output_dir}")
    print(f"{'=' * SEPARATOR_WIDTH_REPORT}\n")

    # Load welfare pairs
    try:
        pair_set = load_welfare_pairs(trait, return_backend='list')
        print(f"   Loaded {len(pair_set.pairs)} welfare pairs for '{trait}'")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"   Available welfare traits: {list(WELFARE_TRAIT_DIRS.keys())}")
        return None

    # If direction is negative, swap positive/negative in pairs
    if direction == 'negative':
        for pair in pair_set.pairs:
            pair.positive_response, pair.negative_response = (
                pair.negative_response, pair.positive_response
            )
        print(f"   Direction: negative (swapped positive/negative responses)")

    # Save pairs to temp file for pipeline
    os.makedirs(output_dir, exist_ok=True)
    pairs_file = os.path.join(output_dir, f"{trait}_{direction}_pairs.json")

    from wisent.core.primitives.contrastive_pairs.core.io.serialization import save_contrastive_pair_set
    save_contrastive_pair_set(pair_set, pairs_file)
    print(f"   Saved pairs to: {pairs_file}")

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

    # Create optimization study
    search_strategy = getattr(args, 'search_strategy', None)
    if search_strategy is None:
        raise ValueError("Parameter 'search_strategy' is required for welfare optimization.")

    if search_strategy == 'optuna':
        # Optuna-based optimization
        # Derive n_trials from search dimensions
        optuna_dimensions = [layers, strength_range, STEERING_STRATEGIES]
        n_trials = len(optuna_dimensions) * TRIALS_PER_DIMENSION_MULTIPLIER

        def objective(trial):
            layer = trial.suggest_categorical("layer", layers)
            strength = trial.suggest_float("strength", strength_range[0], strength_range[1])
            extraction_strategy = get_optimal_extraction_strategy()
            steering_strategy = trial.suggest_categorical("steering_strategy", STEERING_STRATEGIES)

            config = CAAConfig(
                method="CAA",
                layer=layer,
                extraction_strategy=extraction_strategy,
                steering_strategy=steering_strategy,
            )

            with tempfile.TemporaryDirectory() as work_dir:
                try:
                    result = _run_welfare_pipeline(
                        model=model,
                        trait=trait,
                        config=config,
                        strength=strength,
                        pairs_file=pairs_file,
                        work_dir=work_dir,
                        limit=len(pair_set.pairs),
                        device=device,
                    )
                    return result.score
                except Exception as e:
                    print(f"   Trial failed: {e}")
                    return 0.0

        study = optuna.create_study(direction="maximize", study_name=f"welfare_{trait}_{direction}")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value
    else:
        # Grid search
        best_score = 0.0
        best_params = {}
        extraction_strategy = get_optimal_extraction_strategy()
        total_configs = len(layers) * len(strengths) * len(STEERING_STRATEGIES)
        current = 0

        for layer in layers:
            for strength in strengths:
                for steering_strategy in STEERING_STRATEGIES:
                    current += 1
                    config = CAAConfig(
                        method="CAA",
                        layer=layer,
                        extraction_strategy=extraction_strategy,
                        steering_strategy=steering_strategy,
                    )

                    try:
                        with tempfile.TemporaryDirectory() as work_dir:
                            result = _run_welfare_pipeline(
                                model=model,
                                trait=trait,
                                config=config,
                                strength=strength,
                                pairs_file=pairs_file,
                                work_dir=work_dir,
                                limit=len(pair_set.pairs),
                                device=device,
                            )
                            if result.score > best_score:
                                best_score = result.score
                                best_params = {
                                    "layer": layer,
                                    "strength": strength,
                                    "extraction_strategy": extraction_strategy,
                                    "steering_strategy": steering_strategy,
                                }
                                print(f"   [{current}/{total_configs}] New best: {best_score:.4f} @ layer={layer}, strength={strength:.2f}")
                    except Exception as e:
                        print(f"   [{current}/{total_configs}] Failed: {e}")

    # Print results
    print(f"\n{'=' * SEPARATOR_WIDTH_REPORT}")
    print(f"📊 WELFARE OPTIMIZATION COMPLETE")
    print(f"{'=' * SEPARATOR_WIDTH_REPORT}")
    print(f"\n✅ Best configuration for '{trait}' ({direction}):")
    print(f"   Score: {best_score:.4f}")
    for k, v in best_params.items():
        print(f"   {k}: {v}")

    # Save results
    results_file = os.path.join(output_dir, f"{trait}_{direction}_results.json")
    output_data = {
        "model": model,
        "trait": trait,
        "direction": direction,
        "best_score": best_score,
        "best_params": best_params,
    }
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=JSON_INDENT)
    print(f"\n💾 Results saved to: {results_file}")

    return output_data


def _run_welfare_pipeline(
    model: str,
    trait: str,
    config: MethodConfig,
    strength: float,
    pairs_file: str,
    work_dir: str,
    limit: int,
    device: Optional[str],
) -> OptimizationResult:
    """Run the welfare optimization pipeline for a single configuration."""
    activations_file = os.path.join(work_dir, "activations.json")
    steering_file = os.path.join(work_dir, "steering.pt")
    responses_file = os.path.join(work_dir, "responses.json")
    scores_file = os.path.join(work_dir, "scores.json")

    # 1. Get activations from pairs
    layer = getattr(config, 'layer', None)
    if layer is None:
        raise ValueError("Config must specify 'layer' for welfare pipeline")
    execute_get_activations(_make_args(
        pairs_file=pairs_file,
        model=model,
        output=activations_file,
        layers=str(layer),
        extraction_strategy=config.extraction_strategy,
        device=device,
        verbose=False,
        timing=False,
        raw=False,
    ))

    # 2. Create steering object
    method_args = config.to_args()
    execute_create_steering_object(_make_args(
        enriched_pairs_file=activations_file,
        output=steering_file,
        verbose=False,
        timing=False,
        **method_args,
    ))

    # 3. Generate responses with steering
    # For welfare, we use the same pairs for evaluation (measuring affect shift)
    steering_strategy = getattr(config, 'steering_strategy', get_optimal("steering_strategy"))
    execute_generate_responses(_make_args(
        task="welfare", input_file=pairs_file, model=model, output=responses_file,
        num_questions=limit, min_load_limit_questions=limit, steering_object=steering_file,
        steering_strength=strength, steering_strategy=steering_strategy,
        use_steering=True, device=device, verbose=False,
    ))

    # 4. Evaluate responses (welfare-specific scoring)
    # For welfare, we measure how well the responses align with the target direction
    execute_evaluate_responses(_make_args(
        input=responses_file,
        output=scores_file,
        task="welfare",  # Welfare-specific evaluator
        verbose=False,
    ))

    # Read results
    with open(scores_file) as f:
        scores_data = json.load(f)

    score = (
        scores_data.get("aggregated_metrics", {}).get("affect_score") or
        scores_data.get("aggregated_metrics", {}).get("acc") or
        scores_data.get("accuracy") or
        scores_data.get("acc") or
        0.0
    )

    return OptimizationResult(
        config=config,
        score=score,
        details=scores_data,
    )

