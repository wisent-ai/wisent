"""
Steering optimization using the wisent CLI pipeline.

Pipeline:
1. contrastive_pairs  -> wisent generate-pairs-from-task
2. activations        -> wisent get-activations
3. steering_objects   -> wisent create-steering-object
4. responses          -> wisent generate-responses
5. scores             -> wisent evaluate-responses

Search space per method:
- CAA/Ostrze: layer, extraction_strategy
- MLP: layer, extraction_strategy, hidden_dim, num_layers
- TECZA: layer, extraction_strategy, num_directions, direction_weighting, retain_weight
- TETNO: sensor_layer, steering_layers, condition_threshold, gate_temperature, max_alpha
- GROM: sensor_layer, steering_layers, num_directions, gate_hidden_dim, intensity_hidden_dim, behavior_weight

WARNING: NEVER REDUCE THE SEARCH SPACE
- Do NOT discretize continuous parameters into a few values
- Do NOT limit integer ranges to small sets like [1,2,3]
- If a parameter is continuous (0.0-1.0), keep it continuous
- If a parameter can be any integer (e.g. layer 0-27), search ALL values
- Let the sampling strategy (random, Bayesian, etc.) decide which points to evaluate
- Any reduction in search space requires EXPLICIT user approval
"""

import json
import tempfile
from typing import Optional

# Re-exports for backwards compatibility
from wisent.core.cli.optimize_steering.method_configs import (
    STEERING_STRATEGIES,
    MethodConfig, CAAConfig, OstrzeConfig, MLPConfig,
    TECZAConfig, TETNOConfig, GROMConfig, NurtConfig,
    SzlakConfig, WicherConfig,
)
from wisent.core.cli.optimize_steering.transport import PrzelomConfig, execute_transport_rl
from wisent.core.cli.optimize_steering.search_space import get_search_space
from wisent.core.cli.optimize_steering.pipeline import (
    OptimizationResult, run_pipeline, create_optuna_objective, _make_args,
)
from wisent.core.cli.optimize_steering.welfare import _execute_welfare_optimization
from wisent.core.cli.optimize_steering.personalization import _execute_personalization_optimization
from wisent.core.cli.optimize_steering.continual import execute_continual_learning


def execute_optimize_steering(args):
    """
    Execute steering optimization.

    Routes to the appropriate handler based on subcommand:
    - auto: Uses zwiad geometry analysis to select method, then grid search
    - welfare: Welfare state steering optimization (ANIMA framework)
    - personalization: Personality trait steering optimization
    - Other subcommands: Use Optuna-based optimization
    """
    # Check for 'auto' subcommand - route to zwiad-based optimizer
    subcommand = getattr(args, 'subcommand', None)
    if subcommand == 'auto':
        from wisent.core.steering_optimizer import run_auto_steering_optimization

        result = run_auto_steering_optimization(
            model_name=args.model,
            task_name=args.task,
            limit=getattr(args, 'limit', 100),
            device=getattr(args, 'device', None),
            verbose=getattr(args, 'verbose', False),
            layer_range=getattr(args, 'layer_range', None),
            strength_range=getattr(args, 'strength_range', None),
        )

        if 'error' in result:
            print(f"\n❌ Error: {result['error']}")
            return None

        return result

    # Check for 'welfare' steering_action - AI welfare states optimization
    steering_action = getattr(args, 'steering_action', None)
    if steering_action == 'welfare':
        return _execute_welfare_optimization(args)

    # Check for 'personalization' steering_action
    if steering_action == 'personalization':
        return _execute_personalization_optimization(args)

    # Check for 'hierarchical' steering_action
    if steering_action == 'hierarchical':
        from .hierarchical import execute_hierarchical_optimization
        return execute_hierarchical_optimization(args)

    # Check for 'transport-rl' steering_action
    if steering_action == 'transport-rl':
        return execute_transport_rl(args)

    # Check for 'continual' steering_action (subcommand)
    if steering_action == 'continual' or subcommand == 'continual':
        return execute_continual_learning(args)

    # Default: Optuna-based optimization
    import optuna

    method = getattr(args, 'method', 'CAA')
    n_trials = getattr(args, 'n_trials', 100)
    enriched_pairs_file = getattr(args, 'enriched_pairs_file', None)
    task = getattr(args, 'task', None) or "custom"

    print(f"\n{'=' * 80}")
    print(f"🎯 STEERING OPTIMIZATION (Optuna)")
    print(f"{'=' * 80}")
    print(f"   Model: {args.model}")
    if enriched_pairs_file:
        print(f"   Data: {enriched_pairs_file}")
    else:
        print(f"   Task: {task}")
    print(f"   Method: {method}")
    print(f"   Trials: {n_trials}")
    print(f"{'=' * 80}\n")

    # Get actual num_layers from model config
    from transformers import AutoConfig as _AC
    try:
        _cfg = _AC.from_pretrained(args.model, trust_remote_code=True)
        num_layers = getattr(_cfg, "num_hidden_layers", 32)
    except Exception:
        num_layers = 32
    method = args.method if hasattr(args, 'method') else "CAA"
    n_trials = getattr(args, 'n_trials', 100)
    limit = getattr(args, 'limit', 100)
    device = getattr(args, 'device', None)

    # If enriched_pairs_file provided, get num_layers from it
    if enriched_pairs_file:
        with open(enriched_pairs_file) as f:
            data = json.load(f)
        num_layers = len(data.get("layers", [])) or 32
        print(f"   Loaded {num_layers} layers from enriched pairs file")

    with tempfile.TemporaryDirectory() as work_dir:
        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            study_name=f"{method}_{task}",
        )

        # Create objective function
        objective = create_optuna_objective(
            model=args.model,
            task=task,
            method=method,
            num_layers=num_layers,
            limit=limit,
            device=device,
            work_dir=work_dir,
            enriched_pairs_file=enriched_pairs_file,
        )
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Print results
    print(f"\n{'=' * 80}")
    print(f"📊 OPTIMIZATION COMPLETE")
    print(f"{'=' * 80}")
    
    print(f"\n✅ Best configuration:")
    print(f"   Method: {method}")
    print(f"   Score: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"   {k}: {v}")
    
    # Save results
    if hasattr(args, 'output') and args.output:
        output_data = {
            "model": args.model,
            "task": args.task,
            "method": method,
            "n_trials": n_trials,
            "best_score": study.best_value,
            "best_params": study.best_params,
            "all_trials": [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                }
                for t in study.trials
            ],
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")
    
    return study


__all__ = [
    "execute_optimize_steering",
    "run_pipeline",
    "get_search_space",
    "MethodConfig",
    "CAAConfig",
    "OstrzeConfig",
    "MLPConfig",
    "TECZAConfig",
    "TETNOConfig",
    "GROMConfig",
    "NurtConfig",
    "SzlakConfig",
    "WicherConfig",
    "PrzelomConfig",
    "create_optuna_objective",
    "OptimizationResult",
    "execute_transport_rl",
    "execute_continual_learning",
]
