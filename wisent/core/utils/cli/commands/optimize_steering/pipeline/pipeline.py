"""Pipeline and objective function for steering optimization."""
import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional

from wisent.core.utils.cli.optimize_steering.method_configs import (
    MethodConfig, CAAConfig, OstrzeConfig, MLPConfig, TECZAConfig,
    TETNOConfig, GROMConfig, NurtConfig, SzlakConfig, WicherConfig,
)
from wisent.core.utils.cli.optimize_steering.transport.method_configs_transport import PrzelomConfig
from wisent.core.utils.cli.optimize_steering.data.contrastive_pairs_data import execute_generate_pairs_from_task
from wisent.core.utils.cli.optimize_steering.data.activations_data import execute_get_activations
from wisent.core.utils.cli.optimize_steering.steering_objects import execute_create_steering_object
from wisent.core.utils.cli.optimize_steering.data.responses import execute_generate_responses
from wisent.core.utils.cli.optimize_steering.scores import execute_evaluate_responses
from wisent.core.utils.config_tools.constants import (
    SCORE_RANGE_MIN, GENERATION_DEFAULT_MAX_NEW_TOKENS,
    GENERATION_DEFAULT_TEMPERATURE, GENERATION_DEFAULT_TOP_P,
    EVAL_F1_THRESHOLD, EVAL_GENERATION_EMBEDDING_WEIGHT,
    EVAL_GENERATION_NLI_WEIGHT, SPLIT_RATIO_TRAIN_DEFAULT,
)


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
    train_pairs_file: Optional[str] = None,
    test_pairs_file: Optional[str] = None,
    cached_model=None,
) -> OptimizationResult:
    """Run the full optimization pipeline for a single configuration.

    When train_pairs_file and test_pairs_file are set, the steering object
    is trained on train pairs and evaluated on test pairs (no data leakage).
    """
    activations_file = os.path.join(work_dir, "activations.json")
    steering_file = os.path.join(work_dir, "steering.pt")
    responses_file = os.path.join(work_dir, "responses.json")
    scores_file = os.path.join(work_dir, "scores.json")

    if enriched_pairs_file:
        activations_file = enriched_pairs_file
        eval_pairs_file = enriched_pairs_file
        eval_limit = limit
    elif train_pairs_file:
        layer = getattr(config, 'layer', None) or getattr(config, 'sensor_layer', None)
        if layer is None:
            raise ValueError("Config must specify 'layer' or 'sensor_layer'")
        from wisent.core.reading.modules.utilities.data.enriched_builder import (
            build_enriched_from_hf, build_enriched_from_db,
        )
        cached = build_enriched_from_hf(
            model, task, layer, config.extraction_strategy, work_dir,
            train_pairs_file=train_pairs_file, limit=limit)
        if not cached:
            cached = build_enriched_from_db(
                model, task, work_dir, config.extraction_strategy, limit=limit)
        if cached:
            activations_file = cached
        else:
            execute_get_activations(_make_args(
                pairs_file=train_pairs_file, model=model, output=activations_file,
                layers=str(layer), extraction_strategy=config.extraction_strategy,
                device=device, verbose=False, timing=False, raw=False,
                cached_model=cached_model,
            ))
        eval_pairs_file = test_pairs_file or train_pairs_file
        with open(eval_pairs_file) as f:
            eval_limit = len(json.load(f).get("pairs", []))
    else:
        pairs_file = os.path.join(work_dir, "pairs.json")
        execute_generate_pairs_from_task(_make_args(
            task_name=task, output=pairs_file, limit=limit, verbose=False,
        ))
        layer = getattr(config, 'layer', None) or getattr(config, 'sensor_layer', None)
        if layer is None:
            raise ValueError("Config must specify 'layer' or 'sensor_layer'")
        execute_get_activations(_make_args(
            pairs_file=pairs_file, model=model, output=activations_file,
            layers=str(layer), extraction_strategy=config.extraction_strategy,
            device=device, verbose=False, timing=False, raw=False,
            cached_model=cached_model,
        ))
        eval_pairs_file = pairs_file
        eval_limit = limit

    method_args = config.to_args()
    execute_create_steering_object(_make_args(
        enriched_pairs_file=activations_file, output=steering_file,
        verbose=False, timing=False, **method_args,
    ))

    max_new_tokens = GENERATION_DEFAULT_MAX_NEW_TOKENS
    try:
        import wisent as _pkg
        _te = os.path.join(os.path.dirname(_pkg.__file__), "task-evaluator.json")
        if os.path.exists(_te):
            with open(_te) as f:
                _tgt = json.load(f).get("tasks", {}).get(task, {}).get("max_gen_toks")
            if _tgt is not None:
                max_new_tokens = _tgt
    except Exception:
        pass
    steering_strategy = getattr(config, 'steering_strategy', 'constant')
    execute_generate_responses(_make_args(
        task=task, input_file=eval_pairs_file, model=model, output=responses_file,
        num_questions=eval_limit, min_load_limit_questions=eval_limit,
        steering_object=steering_file, steering_strength=strength,
        steering_strategy=steering_strategy, use_steering=True,
        device=device, verbose=False, cached_model=cached_model,
        max_new_tokens=max_new_tokens,
        temperature=GENERATION_DEFAULT_TEMPERATURE, top_p=GENERATION_DEFAULT_TOP_P,
    ))

    execute_evaluate_responses(_make_args(
        input=responses_file, output=scores_file, task=task, verbose=False,
        f1_threshold=EVAL_F1_THRESHOLD,
        generation_embedding_weight=EVAL_GENERATION_EMBEDDING_WEIGHT,
        generation_nli_weight=EVAL_GENERATION_NLI_WEIGHT,
        train_ratio=SPLIT_RATIO_TRAIN_DEFAULT,
    ))

    with open(scores_file) as f:
        scores_data = json.load(f)

    score = (
        scores_data.get("aggregated_metrics", {}).get("acc")
        or scores_data.get("accuracy")
        or scores_data.get("acc")
        or SCORE_RANGE_MIN
    )

    return OptimizationResult(config=config, score=score, details=scores_data)


def _build_config(method: str, params: dict) -> tuple[MethodConfig, float]:
    """Build a MethodConfig from flat params dict. Returns (config, strength)."""
    strength = params.get("strength", SCORE_RANGE_MIN)
    ext = params.get("extraction_strategy", "chat_last")
    steer = params.get("steering_strategy", "constant")
    m = method.upper()

    if m == "CAA":
        cfg = CAAConfig(method="CAA", layer=int(params["layer"]),
                        extraction_strategy=ext, steering_strategy=steer)
    elif m == "OSTRZE":
        cfg = OstrzeConfig(method="Ostrze", layer=int(params["layer"]),
                           extraction_strategy=ext, steering_strategy=steer)
    elif m == "MLP":
        cfg = MLPConfig(
            method="MLP", layer=int(params["layer"]),
            hidden_dim=int(params["hidden_dim"]),
            num_layers=int(params["num_layers"]),
            mlp_input_divisor=int(params.get("mlp_input_divisor", params.get("layer"))),
            mlp_early_stopping_patience=int(params.get("mlp_early_stopping_patience", params.get("layer"))),
            mlp_gating_hidden_dim_divisor=int(params.get("gating_hidden_dim_divisor", params.get("layer"))),
            extraction_strategy=ext, steering_strategy=steer,
        )
    elif m == "TECZA":
        cfg = TECZAConfig(
            method="TECZA", layer=int(params["layer"]),
            num_directions=int(params["num_directions"]),
            direction_weighting=params.get("direction_weighting", "primary_only"),
            retain_weight=float(params["retain_weight"]),
            optimization_steps=int(params["optimization_steps"]),
            extraction_strategy=ext, steering_strategy=steer,
        )
    elif m == "TETNO":
        start = int(params["steering_start"])
        end = int(params["steering_end"])
        if end < start:
            start, end = end, start
        cfg = TETNOConfig(
            method="TETNO", sensor_layer=int(params["sensor_layer"]),
            steering_layers=list(range(start, end + _LAYER_OFFSET)),
            condition_threshold=float(params["condition_threshold"]),
            gate_temperature=float(params["gate_temperature"]),
            max_alpha=float(params["max_alpha"]),
            extraction_strategy=ext, steering_strategy=steer,
        )
    elif m == "GROM":
        start = int(params["steering_start"])
        end = int(params["steering_end"])
        if end < start:
            start, end = end, start
        cfg = GROMConfig(
            method="GROM", sensor_layer=int(params["sensor_layer"]),
            steering_layers=list(range(start, end + _LAYER_OFFSET)),
            num_directions=int(params["num_directions"]),
            gate_hidden_dim=int(params["gate_hidden_dim"]),
            intensity_hidden_dim=int(params["intensity_hidden_dim"]),
            behavior_weight=float(params.get("behavior_weight", SCORE_RANGE_MIN)),
            retain_weight=float(params.get("retain_weight", SCORE_RANGE_MIN)),
            sparse_weight=float(params.get("sparse_weight", SCORE_RANGE_MIN)),
            max_alpha=float(params.get("max_alpha", SCORE_RANGE_MIN)),
            optimization_steps=int(params["optimization_steps"]),
            extraction_strategy=ext, steering_strategy=steer,
        )
    elif m == "NURT":
        cfg = NurtConfig(
            method="nurt", layer=int(params["layer"]),
            variance_threshold=float(params.get("variance_threshold", SCORE_RANGE_MIN)),
            training_epochs=int(params.get("training_epochs", params.get("layer"))),
            lr=float(params.get("lr", SCORE_RANGE_MIN)),
            num_integration_steps=int(params.get("num_integration_steps", params.get("layer"))),
            t_max=float(params.get("t_max", SCORE_RANGE_MIN)),
            extraction_strategy=ext, steering_strategy=steer,
        )
    elif m == "SZLAK":
        cfg = SzlakConfig(
            method="szlak", layer=int(params["layer"]),
            sinkhorn_reg=float(params["sinkhorn_reg"]),
            inference_k=int(params["inference_k"]),
            extraction_strategy=ext, steering_strategy=steer,
        )
    elif m == "WICHER":
        cfg = WicherConfig(
            method="wicher", layer=int(params["layer"]),
            concept_dim=int(params["concept_dim"]),
            variance_threshold=float(params["variance_threshold"]),
            num_steps=int(params["num_steps"]),
            alpha=float(params["alpha"]),
            eta=float(params["eta"]),
            beta=float(params["beta"]),
            alpha_decay=float(params["alpha_decay"]),
            extraction_strategy=ext, steering_strategy=steer,
        )
    elif m == "PRZELOM":
        cfg = PrzelomConfig(
            method="przelom", layer=int(params["layer"]),
            epsilon=float(params["epsilon"]),
            target_mode=params["target_mode"],
            regularization=float(params["regularization"]),
            inference_k=int(params["inference_k"]),
            extraction_strategy=ext, steering_strategy=steer,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return cfg, strength


from wisent.core.utils.config_tools.constants import COMBO_OFFSET as _LAYER_OFFSET


def create_objective(
    method: str,
    model: str,
    task: str,
    num_layers: int,
    limit: int,
    device: Optional[str],
    work_dir: str,
    enriched_pairs_file: Optional[str] = None,
    train_pairs_file: Optional[str] = None,
    test_pairs_file: Optional[str] = None,
    cached_model=None,
):
    """Create an objective function for the UnifiedOptimizer.

    Returns a callable that takes a flat params dict and returns a score.
    When train_pairs_file/test_pairs_file are set, trains on train and
    evaluates on test (no data leakage).
    """
    def objective(params: dict) -> float:
        config, strength = _build_config(method, params)
        return run_pipeline(
            model=model, task=task, config=config, work_dir=work_dir,
            strength=strength, limit=limit, device=device,
            enriched_pairs_file=enriched_pairs_file,
            train_pairs_file=train_pairs_file,
            test_pairs_file=test_pairs_file,
            cached_model=cached_model,
        ).score
    return objective
