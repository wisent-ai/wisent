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
from wisent.core import constants as _C
from wisent.core.utils.config_tools.constants import COMPARE_TOL, LR_LOWER_BOUND, LR_UPPER_BOUND, ALPHA_LOWER_BOUND, ALPHA_UPPER_BOUND, DEFAULT_LIMIT


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
    limit: int = DEFAULT_LIMIT,
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
        task=task,
        input_file=pairs_file,  # Load from pairs file instead of task
        model=model,
        output=responses_file,
        num_questions=limit,
        steering_object=steering_file,
        steering_strength=strength,
        steering_strategy=steering_strategy, use_steering=True, device=device,
        verbose=False, cached_model=cached_model,
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
    model: str,
    task: str,
    method: str,
    num_layers: int,
    limit: int,
    device: Optional[str],
    work_dir: str,
    enriched_pairs_file: Optional[str] = None,
    cached_model=None,
):
    """Create an Optuna objective function for a given method."""

    def objective(trial: "optuna.Trial") -> float:
        # Common parameters for all methods
        extraction_strategy = trial.suggest_categorical("extraction_strategy", ["chat_last", "chat_mean"])
        steering_strategy = trial.suggest_categorical("steering_strategy", STEERING_STRATEGIES)
        trial_strength = trial.suggest_float("strength", _C.SEARCH_STRENGTH_RANGE_MIN, _C.SEARCH_STRENGTH_RANGE_MAX)
        
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
                hidden_dim=trial.suggest_int("hidden_dim", _C.OPTUNA_MLP_HIDDEN_DIM_MIN, _C.OPTUNA_MLP_HIDDEN_DIM_MAX),
                num_layers=trial.suggest_int("num_layers", _C.OPTUNA_MLP_NUM_LAYERS_MIN, _C.OPTUNA_MLP_NUM_LAYERS_MAX),
                extraction_strategy=extraction_strategy,
                steering_strategy=steering_strategy,
            )
        elif method.upper() == "TECZA":
            config = TECZAConfig(
                method="TECZA",
                layer=trial.suggest_int("layer", 1, num_layers),
                num_directions=trial.suggest_int("num_directions", _C.OPTUNA_NUM_DIRECTIONS_MIN, _C.OPTUNA_NUM_DIRECTIONS_MAX),
                direction_weighting=trial.suggest_categorical("direction_weighting", ["primary_only", "equal"]),
                retain_weight=trial.suggest_float("retain_weight", _C.OPTUNA_RETAIN_WEIGHT_MIN, _C.OPTUNA_RETAIN_WEIGHT_MAX),
                optimization_steps=trial.suggest_int("optimization_steps", _C.OPTUNA_OPT_STEPS_MIN, _C.OPTUNA_OPT_STEPS_MAX),
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
                condition_threshold=trial.suggest_float("condition_threshold", _C.OPTUNA_RETAIN_WEIGHT_MIN, _C.OPTUNA_RETAIN_WEIGHT_MAX),
                gate_temperature=trial.suggest_float("gate_temperature", _C.OPTUNA_TEMPERATURE_MIN, _C.OPTUNA_TEMPERATURE_MAX),
                max_alpha=trial.suggest_float("max_alpha", _C.OPTUNA_MAX_ALPHA_MIN, _C.OPTUNA_MAX_ALPHA_MAX),
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
                num_directions=trial.suggest_int("num_directions", _C.OPTUNA_NUM_DIRECTIONS_MIN, _C.OPTUNA_NUM_DIRECTIONS_MAX),
                gate_hidden_dim=trial.suggest_int("gate_hidden_dim", _C.OPTUNA_GROM_GATE_DIM_MIN, _C.OPTUNA_GROM_GATE_DIM_MAX),
                intensity_hidden_dim=trial.suggest_int("intensity_hidden_dim", _C.OPTUNA_GROM_INTENSITY_DIM_MIN, _C.OPTUNA_GROM_INTENSITY_DIM_MAX),
                behavior_weight=trial.suggest_float("behavior_weight", _C.OPTUNA_TEMPERATURE_MIN, _C.OPTUNA_TEMPERATURE_MAX),
                retain_weight=trial.suggest_float("retain_weight", _C.OPTUNA_RETAIN_WEIGHT_MIN, _C.OPTUNA_RETAIN_WEIGHT_MAX),
                sparse_weight=trial.suggest_float("sparse_weight", _C.OPTUNA_GROM_SPARSE_WEIGHT_MIN, _C.OPTUNA_GROM_SPARSE_WEIGHT_MAX),
                max_alpha=trial.suggest_float("max_alpha", _C.OPTUNA_MAX_ALPHA_MIN, _C.OPTUNA_MAX_ALPHA_MAX),
                optimization_steps=trial.suggest_int("optimization_steps", _C.OPTUNA_OPT_STEPS_MIN, _C.OPTUNA_OPT_STEPS_MAX),
                extraction_strategy=extraction_strategy,
                steering_strategy=steering_strategy,
            )

        elif method.upper() == "NURT" or method.upper() == "CONCEPT FLOW":
            config = NurtConfig(
                method="nurt",
                layer=trial.suggest_int("layer", 1, num_layers),
                variance_threshold=trial.suggest_float("variance_threshold", _C.OPTUNA_VARIANCE_MIN, _C.OPTUNA_VARIANCE_MAX),
                training_epochs=trial.suggest_int("training_epochs", _C.OPTUNA_OPT_STEPS_MIN, _C.OPTUNA_OPT_STEPS_MAX),
                lr=trial.suggest_float("lr", LR_LOWER_BOUND, LR_UPPER_BOUND, log=True),
                num_integration_steps=trial.suggest_int("num_integration_steps", _C.OPTUNA_NURT_STEPS_MIN, _C.OPTUNA_NURT_STEPS_MAX),
                t_max=trial.suggest_float("t_max", _C.OPTUNA_TEMPERATURE_MIN, _C.OPTUNA_TEMPERATURE_MAX),
                extraction_strategy=extraction_strategy,
                steering_strategy=steering_strategy,
            )

        elif method.upper() == "SZLAK" or method.upper() == "GEODESIC OT":
            config = SzlakConfig(
                method="szlak",
                layer=trial.suggest_int("layer", 1, num_layers),
                sinkhorn_reg=trial.suggest_float("sinkhorn_reg", _C.OPTUNA_SZLAK_REG_MIN, _C.OPTUNA_RETAIN_WEIGHT_MAX, log=True),
                inference_k=trial.suggest_int("inference_k", _C.OPTUNA_INFERENCE_K_MIN, _C.OPTUNA_INFERENCE_K_MAX),
                extraction_strategy=extraction_strategy,
                steering_strategy=steering_strategy,
            )
        elif method.upper() == "WICHER":
            config = WicherConfig(
                method="wicher",
                layer=trial.suggest_int("layer", 1, num_layers),
                concept_dim=trial.suggest_categorical("concept_dim", list(_C.OPTUNA_WICHER_CONCEPT_DIMS)),
                variance_threshold=trial.suggest_float("variance_threshold", _C.OPTUNA_VARIANCE_MIN, _C.OPTUNA_VARIANCE_MAX),
                num_steps=trial.suggest_int("num_steps", _C.OPTUNA_WICHER_STEPS_MIN, _C.OPTUNA_WICHER_STEPS_MAX),
                alpha=trial.suggest_float("alpha", ALPHA_LOWER_BOUND, ALPHA_UPPER_BOUND, log=True),
                eta=trial.suggest_float("eta", _C.OPTUNA_TEMPERATURE_MIN, _C.OPTUNA_TEMPERATURE_MAX),
                beta=trial.suggest_float("beta", _C.OPTUNA_RETAIN_WEIGHT_MIN, _C.OPTUNA_RETAIN_WEIGHT_MAX),
                alpha_decay=trial.suggest_float("alpha_decay", _C.OPTUNA_RETAIN_WEIGHT_MIN, _C.OPTUNA_RETAIN_WEIGHT_MAX),
                extraction_strategy=extraction_strategy,
                steering_strategy=steering_strategy,
            )
        elif method.upper() == "PRZELOM":
            config = PrzelomConfig(
                method="przelom",
                layer=trial.suggest_int("layer", 1, num_layers),
                epsilon=trial.suggest_float("epsilon", _C.OPTUNA_MAX_ALPHA_MIN, _C.OPTUNA_MAX_ALPHA_MAX, log=True),
                target_mode=trial.suggest_categorical("target_mode", list(_C.OPTUNA_PRZELOM_TARGET_MODES)),
                regularization=trial.suggest_float("regularization", COMPARE_TOL, LR_UPPER_BOUND, log=True),
                inference_k=trial.suggest_int("inference_k", _C.OPTUNA_INFERENCE_K_MIN, _C.OPTUNA_INFERENCE_K_MAX),
                extraction_strategy=extraction_strategy,
                steering_strategy=steering_strategy,
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Run pipeline and return score
        result = run_pipeline(
            model=model,
            task=task,
            config=config,
            work_dir=work_dir,
            limit=limit,
            device=device,
            enriched_pairs_file=enriched_pairs_file,
            cached_model=cached_model,
            strength=trial_strength,
        )
        return result.score

    return objective
