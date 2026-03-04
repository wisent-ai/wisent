"""Pipeline and Optuna objective for steering optimization."""
import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

from wisent.core.utils.cli.optimize_steering.method_configs import (
    MethodConfig, CAAConfig, OstrzeConfig, MLPConfig, TECZAConfig,
    TETNOConfig, GROMConfig, NurtConfig, SzlakConfig, WicherConfig,
    STEERING_STRATEGIES,
)
from wisent.core.utils.cli.optimize_steering.transport.method_configs_transport import PrzelomConfig
from wisent.core.utils.cli.optimize_steering.data.contrastive_pairs_data import execute_generate_pairs_from_task
from wisent.core.utils.cli.optimize_steering.data.activations_data import execute_get_activations
from wisent.core.utils.cli.optimize_steering.steering_objects import execute_create_steering_object
from wisent.core.utils.cli.optimize_steering.data.responses import execute_generate_responses
from wisent.core.utils.cli.optimize_steering.scores import execute_evaluate_responses
from wisent.core.utils.config_tools.constants import COMPARE_TOL


@dataclass
class OptimizationResult:
    """Result of a single optimization trial."""
    config: MethodConfig
    score: float
    details: dict


def _make_args(**kwargs):
    """Create an argparse.Namespace from kwargs."""
    args = argparse.Namespace()
    for k, v in kwargs.items():
        setattr(args, k, v)
    return args


def run_pipeline(
    model: str,
    task: str,
    config: MethodConfig,
    work_dir: str,
    strength: float,
    limit: int,
    device: Optional[str] = None,
    enriched_pairs_file: Optional[str] = None,
    cached_model=None,  # Pre-loaded WisentModel to avoid reloading
) -> OptimizationResult:
    """
    Run the full optimization pipeline for a single configuration.

    If enriched_pairs_file is provided (JSON with pairs + activations),
    skips pair generation and activation collection steps.
    """
    pairs_file = os.path.join(work_dir, "pairs.json")
    activations_file = os.path.join(work_dir, "activations.json")
    steering_file = os.path.join(work_dir, "steering.pt")
    responses_file = os.path.join(work_dir, "responses.json")
    scores_file = os.path.join(work_dir, "scores.json")

    if enriched_pairs_file:
        # Skip pair generation and activation collection - use provided file
        activations_file = enriched_pairs_file
        pairs_file = enriched_pairs_file  # Contains pairs too
    else:
        # 1. Generate contrastive pairs
        execute_generate_pairs_from_task(_make_args(
            task_name=task,
            output=pairs_file,
            limit=limit,
            verbose=False,
        ))

        # 2. Collect activations
        layer = getattr(config, 'layer', None) or getattr(config, 'sensor_layer', None)
        if layer is None:
            raise ValueError("Config must specify 'layer' or 'sensor_layer'")
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
            cached_model=cached_model,
        ))
    
    # 3. Create steering object
    method_args = config.to_args()
    execute_create_steering_object(_make_args(
        enriched_pairs_file=activations_file,
        output=steering_file,
        verbose=False,
        timing=False,
        **method_args,
    ))
    
    # 4. Generate responses with steering
    steering_strategy = getattr(config, 'steering_strategy', 'constant')
    execute_generate_responses(_make_args(
        task=task, input_file=pairs_file, model=model, output=responses_file,
        num_questions=limit, min_load_limit_questions=limit, steering_object=steering_file,
        steering_strength=strength, steering_strategy=steering_strategy,
        use_steering=True, device=device, verbose=False, cached_model=cached_model,
    ))
    
    # 5. Evaluate responses
    execute_evaluate_responses(_make_args(
        input=responses_file,
        output=scores_file,
        task=task,
        verbose=False,
    ))
    
    # Read results
    with open(scores_file) as f:
        scores_data = json.load(f)
    
    # Try different accuracy keys - check aggregated_metrics first
    score = (
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


def create_optuna_objective(
    model: str, task: str, method: str, num_layers: int, limit: int, device: Optional[str], work_dir: str,
    lr_lower_bound: float, lr_upper_bound: float, alpha_lower_bound: float, alpha_upper_bound: float,
    optuna_szlak_reg_min: float, optuna_nurt_steps_min: int, optuna_nurt_steps_max: int,
    optuna_wicher_concept_dims: tuple, optuna_wicher_steps_min: int, optuna_wicher_steps_max: int,
    optuna_przelom_target_modes: tuple, optuna_mlp_hidden_dim_min: int = None, optuna_mlp_hidden_dim_max: int = None,
    optuna_mlp_num_layers_min: int = None, optuna_mlp_num_layers_max: int = None,
    mlp_input_divisor: int = None, mlp_early_stopping_patience: int = None, mlp_gating_hidden_dim_divisor: int = None,
    enriched_pairs_file: Optional[str] = None, cached_model=None, *,
    optuna_grom_gate_dim_min: int, optuna_grom_gate_dim_max: int,
    optuna_grom_intensity_dim_min: int, optuna_grom_intensity_dim_max: int,
    optuna_grom_sparse_weight_min: float, optuna_grom_sparse_weight_max: float,
    search_strength_range_min: float, search_strength_range_max: float,
    optuna_num_directions_min: int, optuna_num_directions_max: int,
    optuna_retain_weight_min: float, optuna_retain_weight_max: float,
    optuna_opt_steps_min: int, optuna_opt_steps_max: int,
    optuna_temperature_min: float, optuna_temperature_max: float,
    optuna_max_alpha_min: float, optuna_max_alpha_max: float,
    optuna_variance_min: float, optuna_variance_max: float,
    optuna_inference_k_min: int, optuna_inference_k_max: int,
):
    """Create an Optuna objective function for a given method."""
    def objective(trial: "optuna.Trial") -> float:
        extraction_strategy = trial.suggest_categorical("extraction_strategy", ["chat_last", "chat_mean"])
        steering_strategy = trial.suggest_categorical("steering_strategy", STEERING_STRATEGIES)
        trial_strength = trial.suggest_float("strength", search_strength_range_min, search_strength_range_max)
        
        if method.upper() == "CAA":
            config = CAAConfig(
                method="CAA",
                layer=trial.suggest_int("layer", 1, num_layers),
                extraction_strategy=extraction_strategy,
                steering_strategy=steering_strategy,
            )
        elif method.upper() == "OSTRZE":
            config = OstrzeConfig(
                method="Ostrze",
                layer=trial.suggest_int("layer", 1, num_layers),
                extraction_strategy=extraction_strategy,
                steering_strategy=steering_strategy,
            )
        elif method.upper() == "MLP":
            config = MLPConfig(
                method="MLP",
                layer=trial.suggest_int("layer", 1, num_layers),
                hidden_dim=trial.suggest_int("hidden_dim", optuna_mlp_hidden_dim_min, optuna_mlp_hidden_dim_max),
                num_layers=trial.suggest_int("num_layers", optuna_mlp_num_layers_min, optuna_mlp_num_layers_max),
                mlp_input_divisor=mlp_input_divisor,
                mlp_early_stopping_patience=mlp_early_stopping_patience,
                mlp_gating_hidden_dim_divisor=mlp_gating_hidden_dim_divisor,
                extraction_strategy=extraction_strategy, steering_strategy=steering_strategy,
            )
        elif method.upper() == "TECZA":
            config = TECZAConfig(
                method="TECZA",
                layer=trial.suggest_int("layer", 1, num_layers),
                num_directions=trial.suggest_int("num_directions", optuna_num_directions_min, optuna_num_directions_max),
                direction_weighting=trial.suggest_categorical("direction_weighting", ["primary_only", "equal"]),
                retain_weight=trial.suggest_float("retain_weight", optuna_retain_weight_min, optuna_retain_weight_max),
                optimization_steps=trial.suggest_int("optimization_steps", optuna_opt_steps_min, optuna_opt_steps_max),
                extraction_strategy=extraction_strategy,
                steering_strategy=steering_strategy,
            )
        elif method.upper() == "TETNO":
            sensor_layer = trial.suggest_int("sensor_layer", 1, num_layers)
            steering_start = trial.suggest_int("steering_start", 1, num_layers)
            steering_end = trial.suggest_int("steering_end", steering_start, num_layers)
            steering_layers = list(range(steering_start, steering_end + 1))
            
            config = TETNOConfig(
                method="TETNO",
                sensor_layer=sensor_layer,
                steering_layers=steering_layers,
                condition_threshold=trial.suggest_float("condition_threshold", optuna_retain_weight_min, optuna_retain_weight_max),
                gate_temperature=trial.suggest_float("gate_temperature", optuna_temperature_min, optuna_temperature_max),
                max_alpha=trial.suggest_float("max_alpha", optuna_max_alpha_min, optuna_max_alpha_max),
                extraction_strategy=extraction_strategy,
                steering_strategy=steering_strategy,
            )
        elif method.upper() == "GROM":
            sensor_layer = trial.suggest_int("sensor_layer", 1, num_layers)
            steering_start = trial.suggest_int("steering_start", 1, num_layers)
            steering_end = trial.suggest_int("steering_end", steering_start, num_layers)
            steering_layers = list(range(steering_start, steering_end + 1))
            
            config = GROMConfig(
                method="GROM",
                sensor_layer=sensor_layer,
                steering_layers=steering_layers,
                num_directions=trial.suggest_int("num_directions", optuna_num_directions_min, optuna_num_directions_max),
                gate_hidden_dim=trial.suggest_int("gate_hidden_dim", optuna_grom_gate_dim_min, optuna_grom_gate_dim_max),
                intensity_hidden_dim=trial.suggest_int("intensity_hidden_dim", optuna_grom_intensity_dim_min, optuna_grom_intensity_dim_max),
                behavior_weight=trial.suggest_float("behavior_weight", optuna_temperature_min, optuna_temperature_max),
                retain_weight=trial.suggest_float("retain_weight", optuna_retain_weight_min, optuna_retain_weight_max),
                sparse_weight=trial.suggest_float("sparse_weight", optuna_grom_sparse_weight_min, optuna_grom_sparse_weight_max),
                max_alpha=trial.suggest_float("max_alpha", optuna_max_alpha_min, optuna_max_alpha_max),
                optimization_steps=trial.suggest_int("optimization_steps", optuna_opt_steps_min, optuna_opt_steps_max),
                extraction_strategy=extraction_strategy,
                steering_strategy=steering_strategy,
            )

        elif method.upper() == "NURT" or method.upper() == "CONCEPT FLOW":
            config = NurtConfig(
                method="nurt",
                layer=trial.suggest_int("layer", 1, num_layers),
                variance_threshold=trial.suggest_float("variance_threshold", optuna_variance_min, optuna_variance_max),
                training_epochs=trial.suggest_int("training_epochs", optuna_opt_steps_min, optuna_opt_steps_max),
                lr=trial.suggest_float("lr", lr_lower_bound, lr_upper_bound, log=True),
                num_integration_steps=trial.suggest_int("num_integration_steps", optuna_nurt_steps_min, optuna_nurt_steps_max),
                t_max=trial.suggest_float("t_max", optuna_temperature_min, optuna_temperature_max),
                extraction_strategy=extraction_strategy,
                steering_strategy=steering_strategy,
            )

        elif method.upper() == "SZLAK" or method.upper() == "GEODESIC OT":
            config = SzlakConfig(
                method="szlak",
                layer=trial.suggest_int("layer", 1, num_layers),
                sinkhorn_reg=trial.suggest_float("sinkhorn_reg", optuna_szlak_reg_min, optuna_retain_weight_max, log=True),
                inference_k=trial.suggest_int("inference_k", optuna_inference_k_min, optuna_inference_k_max),
                extraction_strategy=extraction_strategy,
                steering_strategy=steering_strategy,
            )
        elif method.upper() == "WICHER":
            config = WicherConfig(
                method="wicher",
                layer=trial.suggest_int("layer", 1, num_layers),
                concept_dim=trial.suggest_categorical("concept_dim", list(optuna_wicher_concept_dims)),
                variance_threshold=trial.suggest_float("variance_threshold", optuna_variance_min, optuna_variance_max),
                num_steps=trial.suggest_int("num_steps", optuna_wicher_steps_min, optuna_wicher_steps_max),
                alpha=trial.suggest_float("alpha", alpha_lower_bound, alpha_upper_bound, log=True),
                eta=trial.suggest_float("eta", optuna_temperature_min, optuna_temperature_max),
                beta=trial.suggest_float("beta", optuna_retain_weight_min, optuna_retain_weight_max),
                alpha_decay=trial.suggest_float("alpha_decay", optuna_retain_weight_min, optuna_retain_weight_max),
                extraction_strategy=extraction_strategy,
                steering_strategy=steering_strategy,
            )
        elif method.upper() == "PRZELOM":
            config = PrzelomConfig(
                method="przelom",
                layer=trial.suggest_int("layer", 1, num_layers),
                epsilon=trial.suggest_float("epsilon", optuna_max_alpha_min, optuna_max_alpha_max, log=True),
                target_mode=trial.suggest_categorical("target_mode", list(optuna_przelom_target_modes)),
                regularization=trial.suggest_float("regularization", COMPARE_TOL, lr_upper_bound, log=True),
                inference_k=trial.suggest_int("inference_k", optuna_inference_k_min, optuna_inference_k_max),
                extraction_strategy=extraction_strategy,
                steering_strategy=steering_strategy,
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        return run_pipeline(
            model=model, task=task, config=config, work_dir=work_dir, limit=limit,
            device=device, enriched_pairs_file=enriched_pairs_file,
            cached_model=cached_model, strength=trial_strength,
        ).score
    return objective
