import os as _os
_base = _os.path.dirname(__file__)
for _root, _dirs, _files in _os.walk(_base):
    _dirs[:] = sorted(d for d in _dirs if not d.startswith((".", "_")))
    if _root != _base:
        __path__.append(_root)

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
from wisent.core.utils.cli.optimize_steering.method_configs import (
    STEERING_STRATEGIES,
    MethodConfig, CAAConfig, OstrzeConfig, MLPConfig,
    TECZAConfig, TETNOConfig, GROMConfig, NurtConfig,
    SzlakConfig, WicherConfig,
)
from wisent.core.utils.cli.optimize_steering.transport import PrzelomConfig, execute_transport_rl
from wisent.core.utils.cli.optimize_steering.search_space import get_search_space
from wisent.core.utils.cli.optimize_steering.pipeline import (
    OptimizationResult, run_pipeline, create_optuna_objective, _make_args,
)
from wisent.core.utils.cli.optimize_steering.welfare import _execute_welfare_optimization
from wisent.core.utils.cli.optimize_steering.personalization import _execute_personalization_optimization
from wisent.core.utils.cli.optimize_steering.continual import execute_continual_learning
from wisent.core.utils.config_tools.constants import JSON_INDENT, SEPARATOR_WIDTH_REPORT


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
        from wisent.core.control.steering_optimizer import run_auto_steering_optimization

        _GROM_PARAM_NAMES = [
            "num_directions", "optimization_steps", "learning_rate",
            "warmup_steps", "behavior_weight", "retain_weight",
            "sparse_weight", "smooth_weight", "independence_weight",
            "max_alpha", "gate_temperature", "max_grad_norm",
            "eta_min_factor", "linear_threshold",
            "adapt_cone_threshold", "adapt_manifold_threshold",
            "adapt_linear_directions", "adapt_complex_directions",
            "adapt_max_directions", "significant_directions_default",
            "min_adapted_directions", "caa_similarity_skip",
            "contrastive_margin", "contrastive_weight", "utility_weight",
            "concentration_weight", "gate_warmup_weight", "caa_alignment_weight",
            "gate_dim_min", "gate_dim_max", "gate_dim_divisor",
            "intensity_dim_min", "intensity_dim_max", "intensity_dim_divisor",
            "log_interval", "weight_decay",
        ]
        _TETNO_PARAM_NAMES = [
            "condition_threshold", "gate_temperature", "max_alpha",
            "entropy_floor", "entropy_ceiling", "threshold_search_steps",
            "learning_rate", "condition_margin", "min_layer_scale", "log_interval",
            "optimization_steps",
        ]
        grom_params = {k: getattr(args, f"grom_{k}") for k in _GROM_PARAM_NAMES}
        tetno_params = {k: getattr(args, f"tetno_{k}") for k in _TETNO_PARAM_NAMES}
        result = run_auto_steering_optimization(
            model_name=args.model,
            task_name=args.task,
            limit=args.limit,
            min_norm_threshold=args.min_norm_threshold,
            device=getattr(args, 'device', None),
            verbose=getattr(args, 'verbose', False),
            layer_range=getattr(args, 'layer_range', None),
            strength_range=getattr(args, 'strength_range', None),
            min_clusters=getattr(args, 'min_clusters', None),
            grom_params=grom_params,
            tetno_params=tetno_params,
            tecza_params={
                "tecza_num_directions": args.tecza_num_directions,
                "tecza_learning_rate": args.tecza_learning_rate,
                "tecza_retain_weight": args.tecza_retain_weight,
                "tecza_independence_weight": args.tecza_independence_weight,
                "tecza_min_cosine_sim": args.tecza_min_cosine_sim,
                "tecza_max_cosine_sim": args.tecza_max_cosine_sim,
                "tecza_marginal_threshold": args.tecza_marginal_threshold,
                "tecza_max_directions": args.tecza_max_directions,
                "tecza_ablation_weight": args.tecza_ablation_weight,
                "tecza_addition_weight": args.tecza_addition_weight,
                "tecza_separation_margin": args.tecza_separation_margin,
                "tecza_perturbation_scale": args.tecza_perturbation_scale,
                "tecza_universal_basis_noise": args.tecza_universal_basis_noise,
                "tecza_optimization_steps": args.tecza_optimization_steps,
            },
            train_ratio=args.train_ratio,
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
    n_trials = args.n_trials
    enriched_pairs_file = getattr(args, 'enriched_pairs_file', None)
    task = getattr(args, 'task', None) or "custom"

    print(f"\n{'=' * SEPARATOR_WIDTH_REPORT}")
    print(f"🎯 STEERING OPTIMIZATION (Optuna)")
    print(f"{'=' * SEPARATOR_WIDTH_REPORT}")
    print(f"   Model: {args.model}")
    if enriched_pairs_file:
        print(f"   Data: {enriched_pairs_file}")
    else:
        print(f"   Task: {task}")
    print(f"   Method: {method}")
    print(f"   Trials: {n_trials}")
    print(f"{'=' * SEPARATOR_WIDTH_REPORT}\n")

    # Get actual num_layers from model config
    from transformers import AutoConfig as _AC
    _cfg = _AC.from_pretrained(args.model, trust_remote_code=True)
    num_layers = _cfg.num_hidden_layers
    method = args.method if hasattr(args, 'method') else "CAA"
    n_trials = args.n_trials
    limit = args.limit
    device = getattr(args, 'device', None)

    # If enriched_pairs_file provided, get num_layers from it
    if enriched_pairs_file:
        with open(enriched_pairs_file) as f:
            data = json.load(f)
        num_layers = len(data.get("layers", [])) or num_layers
        print(f"   Loaded {num_layers} layers from enriched pairs file")

    with tempfile.TemporaryDirectory() as work_dir:
        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            study_name=f"{method}_{task}",
        )

        # Create objective function
        objective = create_optuna_objective(
            model=args.model, task=task, method=method, num_layers=num_layers,
            limit=limit, device=device, work_dir=work_dir,
            lr_lower_bound=args.lr_lower_bound, lr_upper_bound=args.lr_upper_bound,
            alpha_lower_bound=args.alpha_lower_bound, alpha_upper_bound=args.alpha_upper_bound,
            optuna_szlak_reg_min=args.optuna_szlak_reg_min,
            optuna_nurt_steps_min=args.optuna_nurt_steps_min,
            optuna_nurt_steps_max=args.optuna_nurt_steps_max,
            optuna_wicher_concept_dims=args.optuna_wicher_concept_dims,
            optuna_wicher_steps_min=args.optuna_wicher_steps_min,
            optuna_wicher_steps_max=args.optuna_wicher_steps_max,
            optuna_przelom_target_modes=args.optuna_przelom_target_modes,
            optuna_mlp_hidden_dim_min=args.optuna_mlp_hidden_dim_min,
            optuna_mlp_hidden_dim_max=args.optuna_mlp_hidden_dim_max,
            optuna_mlp_num_layers_min=args.optuna_mlp_num_layers_min,
            optuna_mlp_num_layers_max=args.optuna_mlp_num_layers_max,
            mlp_input_divisor=args.mlp_input_divisor,
            mlp_early_stopping_patience=args.mlp_early_stopping_patience,
            mlp_gating_hidden_dim_divisor=args.mlp_gating_hidden_dim_divisor,
            enriched_pairs_file=enriched_pairs_file,
            optuna_grom_gate_dim_min=args.optuna_grom_gate_dim_min,
            optuna_grom_gate_dim_max=args.optuna_grom_gate_dim_max,
            optuna_grom_intensity_dim_min=args.optuna_grom_intensity_dim_min,
            optuna_grom_intensity_dim_max=args.optuna_grom_intensity_dim_max,
            optuna_grom_sparse_weight_min=args.optuna_grom_sparse_weight_min,
            optuna_grom_sparse_weight_max=args.optuna_grom_sparse_weight_max,
        )
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Print results
    print(f"\n{'=' * SEPARATOR_WIDTH_REPORT}")
    print(f"📊 OPTIMIZATION COMPLETE")
    print(f"{'=' * SEPARATOR_WIDTH_REPORT}")
    
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
            json.dump(output_data, f, indent=JSON_INDENT)
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
