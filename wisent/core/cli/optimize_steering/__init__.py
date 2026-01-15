"""
Steering optimization using the wisent CLI pipeline.

Pipeline:
1. contrastive_pairs  -> wisent generate-pairs-from-task
2. activations        -> wisent get-activations
3. steering_objects   -> wisent create-steering-object
4. responses          -> wisent generate-responses
5. scores             -> wisent evaluate-responses

Search space per method:
- CAA/Hyperplane: layer, extraction_strategy
- MLP: layer, extraction_strategy, hidden_dim, num_layers
- PRISM: layer, extraction_strategy, num_directions, direction_weighting, retain_weight
- PULSE: sensor_layer, steering_layers, condition_threshold, gate_temperature, max_alpha
- TITAN: sensor_layer, steering_layers, num_directions, gate_hidden_dim, intensity_hidden_dim, behavior_weight

WARNING: NEVER REDUCE THE SEARCH SPACE
- Do NOT discretize continuous parameters into a few values
- Do NOT limit integer ranges to small sets like [1,2,3]
- If a parameter is continuous (0.0-1.0), keep it continuous
- If a parameter can be any integer (e.g. layer 0-27), search ALL values
- Let the sampling strategy (random, Bayesian, etc.) decide which points to evaluate
- Any reduction in search space requires EXPLICIT user approval
"""

import argparse
import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Iterator

from .contrastive_pairs import execute_generate_pairs_from_task
from .activations import execute_get_activations
from .steering_objects import execute_create_steering_object
from .responses import execute_generate_responses
from .scores import execute_evaluate_responses


# Steering application strategies (how steering is applied during generation)
STEERING_STRATEGIES = ["constant", "initial_only", "diminishing", "increasing", "gaussian"]


@dataclass
class MethodConfig:
    """Base configuration for all methods."""
    method: str
    extraction_strategy: str = "chat_last"
    steering_strategy: str = "constant"  # How steering is applied during generation


@dataclass
class CAAConfig(MethodConfig):
    """CAA-specific parameters."""
    layer: int = 16
    
    def to_args(self) -> Dict[str, Any]:
        return {
            "method": "CAA",
            "layer": self.layer,
            "steering_strategy": self.steering_strategy,
        }


@dataclass
class HyperplaneConfig(MethodConfig):
    """Hyperplane-specific parameters."""
    layer: int = 16
    
    def to_args(self) -> Dict[str, Any]:
        return {
            "method": "Hyperplane",
            "layer": self.layer,
            "steering_strategy": self.steering_strategy,
        }


@dataclass
class MLPConfig(MethodConfig):
    """MLP-specific parameters."""
    layer: int = 16
    hidden_dim: int = 256
    num_layers: int = 2
    
    def to_args(self) -> Dict[str, Any]:
        return {
            "method": "MLP",
            "layer": self.layer,
            "mlp_hidden_dim": self.hidden_dim,
            "mlp_num_layers": self.num_layers,
            "steering_strategy": self.steering_strategy,
        }


@dataclass
class PRISMConfig(MethodConfig):
    """PRISM-specific parameters."""
    layer: int = 16
    num_directions: int = 3
    direction_weighting: str = "primary_only"
    retain_weight: float = 0.1
    optimization_steps: int = 100
    
    def to_args(self) -> Dict[str, Any]:
        return {
            "method": "PRISM",
            "layer": self.layer,
            "num_directions": self.num_directions,
            "direction_weighting": self.direction_weighting,
            "retain_weight": self.retain_weight,
            "prism_optimization_steps": self.optimization_steps,
            "steering_strategy": self.steering_strategy,
        }


@dataclass
class PULSEConfig(MethodConfig):
    """PULSE-specific parameters."""
    sensor_layer: int = 16
    steering_layers: List[int] = field(default_factory=lambda: [20, 21, 22])
    condition_threshold: float = 0.5
    gate_temperature: float = 0.5
    max_alpha: float = 2.0
    
    def to_args(self) -> Dict[str, Any]:
        return {
            "method": "PULSE",
            "sensor_layer": self.sensor_layer,
            "steering_layers": ",".join(str(l) for l in self.steering_layers),
            "condition_threshold": self.condition_threshold,
            "gate_temperature": self.gate_temperature,
            "max_alpha": self.max_alpha,
            "steering_strategy": self.steering_strategy,
        }


@dataclass
class TITANConfig(MethodConfig):
    """TITAN-specific parameters."""
    sensor_layer: int = 16
    steering_layers: List[int] = field(default_factory=lambda: [20, 21, 22])
    num_directions: int = 3
    gate_hidden_dim: int = 64
    intensity_hidden_dim: int = 32
    behavior_weight: float = 1.0
    retain_weight: float = 0.2
    sparse_weight: float = 0.05
    max_alpha: float = 2.0
    optimization_steps: int = 200
    
    def to_args(self) -> Dict[str, Any]:
        return {
            "method": "TITAN",
            "sensor_layer": self.sensor_layer,
            "steering_layers": ",".join(str(l) for l in self.steering_layers),
            "num_directions": self.num_directions,
            "gate_hidden_dim": self.gate_hidden_dim,
            "intensity_hidden_dim": self.intensity_hidden_dim,
            "behavior_weight": self.behavior_weight,
            "retain_weight": self.retain_weight,
            "sparse_weight": self.sparse_weight,
            "max_alpha": self.max_alpha,
            "titan_optimization_steps": self.optimization_steps,
            "steering_strategy": self.steering_strategy,
        }


def get_search_space(method: str, num_layers: int) -> Iterator[MethodConfig]:
    """
    Generate search space for a method.
    
    Includes:
    - extraction_strategy: how to collect activations (chat_last, chat_mean)
    - steering_strategy: how to apply steering during generation (constant, diminishing, etc.)
    - method-specific parameters
    """
    extraction_strategies = ["chat_last", "chat_mean"]
    steering_strategies = STEERING_STRATEGIES  # constant, initial_only, diminishing, increasing, gaussian
    
    if method.upper() == "CAA":
        for layer in range(num_layers):
            for ext_strategy in extraction_strategies:
                for steer_strategy in steering_strategies:
                    yield CAAConfig(
                        method="CAA",
                        layer=layer,
                        extraction_strategy=ext_strategy,
                        steering_strategy=steer_strategy,
                    )
    
    elif method.upper() == "HYPERPLANE":
        for layer in range(num_layers):
            for ext_strategy in extraction_strategies:
                for steer_strategy in steering_strategies:
                    yield HyperplaneConfig(
                        method="Hyperplane",
                        layer=layer,
                        extraction_strategy=ext_strategy,
                        steering_strategy=steer_strategy,
                    )
    
    elif method.upper() == "MLP":
        for layer in range(num_layers):
            for hidden_dim in [128, 256, 512]:
                for num_layers_mlp in [1, 2, 3]:
                    for ext_strategy in extraction_strategies:
                        for steer_strategy in steering_strategies:
                            yield MLPConfig(
                                method="MLP",
                                layer=layer,
                                hidden_dim=hidden_dim,
                                num_layers=num_layers_mlp,
                                extraction_strategy=ext_strategy,
                                steering_strategy=steer_strategy,
                            )
    
    elif method.upper() == "PRISM":
        for layer in range(num_layers):
            for num_directions in [1, 2, 3, 5]:
                for direction_weighting in ["primary_only", "equal"]:
                    for retain_weight in [0.0, 0.1, 0.3]:
                        for ext_strategy in extraction_strategies:
                            for steer_strategy in steering_strategies:
                                yield PRISMConfig(
                                    method="PRISM",
                                    layer=layer,
                                    num_directions=num_directions,
                                    direction_weighting=direction_weighting,
                                    retain_weight=retain_weight,
                                    extraction_strategy=ext_strategy,
                                    steering_strategy=steer_strategy,
                                )
    
    elif method.upper() == "PULSE":
        for sensor_pos in [0.5, 0.75]:  # middle, late
            sensor_layer = int(num_layers * sensor_pos)
            for steering_range in [3, 5]:
                steering_start = int(num_layers * 0.75)
                steering_layers = list(range(steering_start, min(steering_start + steering_range, num_layers)))
                for threshold in [0.3, 0.5, 0.7]:
                    for gate_temp in [0.1, 0.5, 1.0]:
                        for max_alpha in [1.5, 2.0, 3.0]:
                            for ext_strategy in extraction_strategies:
                                for steer_strategy in steering_strategies:
                                    yield PULSEConfig(
                                        method="PULSE",
                                        sensor_layer=sensor_layer,
                                        steering_layers=steering_layers,
                                        condition_threshold=threshold,
                                        gate_temperature=gate_temp,
                                        max_alpha=max_alpha,
                                        extraction_strategy=ext_strategy,
                                        steering_strategy=steer_strategy,
                                    )
    
    elif method.upper() == "TITAN":
        for sensor_pos in [0.5, 0.75]:
            sensor_layer = int(num_layers * sensor_pos)
            for steering_range in [3, 5]:
                steering_start = int(num_layers * 0.75)
                steering_layers = list(range(steering_start, min(steering_start + steering_range, num_layers)))
                for num_directions in [2, 3, 5]:
                    for gate_hidden in [32, 64]:
                        for intensity_hidden in [16, 32]:
                            for behavior_weight in [0.5, 1.0]:
                                for ext_strategy in extraction_strategies:
                                    for steer_strategy in steering_strategies:
                                        yield TITANConfig(
                                            method="TITAN",
                                            sensor_layer=sensor_layer,
                                            steering_layers=steering_layers,
                                            num_directions=num_directions,
                                            gate_hidden_dim=gate_hidden,
                                            intensity_hidden_dim=intensity_hidden,
                                            behavior_weight=behavior_weight,
                                            extraction_strategy=ext_strategy,
                                            steering_strategy=steer_strategy,
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
    limit: int = 100,
    device: Optional[str] = None,
) -> OptimizationResult:
    """
    Run the full optimization pipeline for a single configuration.
    """
    pairs_file = os.path.join(work_dir, "pairs.json")
    activations_file = os.path.join(work_dir, "activations.json")
    steering_file = os.path.join(work_dir, "steering.pt")
    responses_file = os.path.join(work_dir, "responses.json")
    scores_file = os.path.join(work_dir, "scores.json")
    
    # 1. Generate contrastive pairs
    execute_generate_pairs_from_task(_make_args(
        task_name=task,
        output=pairs_file,
        limit=limit,
        verbose=False,
    ))
    
    # 2. Collect activations
    layer = getattr(config, 'layer', None) or getattr(config, 'sensor_layer', 16)
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
        steering_strength=1.0,  # No strength scaling for methods that don't use it
        steering_strategy=steering_strategy,
        use_steering=True,
        device=device,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.95,
        verbose=False,
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
):
    """Create an Optuna objective function for a given method."""
    
    def objective(trial: "optuna.Trial") -> float:
        # Common parameters for all methods
        extraction_strategy = trial.suggest_categorical("extraction_strategy", ["chat_last", "chat_mean"])
        steering_strategy = trial.suggest_categorical("steering_strategy", STEERING_STRATEGIES)
        
        if method.upper() == "CAA":
            config = CAAConfig(
                method="CAA",
                layer=trial.suggest_int("layer", 1, num_layers),
                extraction_strategy=extraction_strategy,
                steering_strategy=steering_strategy,
            )
        
        elif method.upper() == "HYPERPLANE":
            config = HyperplaneConfig(
                method="Hyperplane",
                layer=trial.suggest_int("layer", 1, num_layers),
                extraction_strategy=extraction_strategy,
                steering_strategy=steering_strategy,
            )
        
        elif method.upper() == "MLP":
            config = MLPConfig(
                method="MLP",
                layer=trial.suggest_int("layer", 1, num_layers),
                hidden_dim=trial.suggest_int("hidden_dim", 32, 1024),
                num_layers=trial.suggest_int("num_layers", 1, 5),
                extraction_strategy=extraction_strategy,
                steering_strategy=steering_strategy,
            )
        
        elif method.upper() == "PRISM":
            config = PRISMConfig(
                method="PRISM",
                layer=trial.suggest_int("layer", 1, num_layers),
                num_directions=trial.suggest_int("num_directions", 1, 10),
                direction_weighting=trial.suggest_categorical("direction_weighting", ["primary_only", "equal"]),
                retain_weight=trial.suggest_float("retain_weight", 0.0, 1.0),
                optimization_steps=trial.suggest_int("optimization_steps", 50, 500),
                extraction_strategy=extraction_strategy,
                steering_strategy=steering_strategy,
            )
        
        elif method.upper() == "PULSE":
            sensor_layer = trial.suggest_int("sensor_layer", 1, num_layers)
            steering_start = trial.suggest_int("steering_start", 1, num_layers)
            steering_end = trial.suggest_int("steering_end", steering_start, num_layers)
            steering_layers = list(range(steering_start, steering_end + 1))
            
            config = PULSEConfig(
                method="PULSE",
                sensor_layer=sensor_layer,
                steering_layers=steering_layers,
                condition_threshold=trial.suggest_float("condition_threshold", 0.0, 1.0),
                gate_temperature=trial.suggest_float("gate_temperature", 0.01, 2.0),
                max_alpha=trial.suggest_float("max_alpha", 0.5, 5.0),
                extraction_strategy=extraction_strategy,
                steering_strategy=steering_strategy,
            )
        
        elif method.upper() == "TITAN":
            sensor_layer = trial.suggest_int("sensor_layer", 1, num_layers)
            steering_start = trial.suggest_int("steering_start", 1, num_layers)
            steering_end = trial.suggest_int("steering_end", steering_start, num_layers)
            steering_layers = list(range(steering_start, steering_end + 1))
            
            config = TITANConfig(
                method="TITAN",
                sensor_layer=sensor_layer,
                steering_layers=steering_layers,
                num_directions=trial.suggest_int("num_directions", 1, 10),
                gate_hidden_dim=trial.suggest_int("gate_hidden_dim", 16, 256),
                intensity_hidden_dim=trial.suggest_int("intensity_hidden_dim", 8, 128),
                behavior_weight=trial.suggest_float("behavior_weight", 0.0, 2.0),
                retain_weight=trial.suggest_float("retain_weight", 0.0, 1.0),
                sparse_weight=trial.suggest_float("sparse_weight", 0.0, 0.5),
                max_alpha=trial.suggest_float("max_alpha", 0.5, 5.0),
                optimization_steps=trial.suggest_int("optimization_steps", 50, 500),
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
        )
        
        return result.score
    
    return objective


def execute_optimize_steering(args):
    """
    Execute steering optimization.

    Routes to the appropriate handler based on subcommand:
    - auto: Uses repscan geometry analysis to select method, then grid search
    - Other subcommands: Use Optuna-based optimization
    """
    # Check for 'auto' subcommand - route to repscan-based optimizer
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

    # Default: Optuna-based optimization
    import optuna

    method = getattr(args, 'method', 'CAA')
    n_trials = getattr(args, 'n_trials', 100)

    print(f"\n{'=' * 80}")
    print(f"🎯 STEERING OPTIMIZATION (Optuna)")
    print(f"{'=' * 80}")
    print(f"   Model: {args.model}")
    print(f"   Task: {args.task}")
    print(f"   Method: {method}")
    print(f"   Trials: {n_trials}")
    print(f"{'=' * 80}\n")
    
    num_layers = getattr(args, 'num_layers', 32)
    method = args.method if hasattr(args, 'method') else "CAA"
    n_trials = getattr(args, 'n_trials', 100)
    limit = getattr(args, 'limit', 100)
    device = getattr(args, 'device', None)
    
    with tempfile.TemporaryDirectory() as work_dir:
        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            study_name=f"{method}_{args.task}",
        )
        
        # Create objective function
        objective = create_optuna_objective(
            model=args.model,
            task=args.task,
            method=method,
            num_layers=num_layers,
            limit=limit,
            device=device,
            work_dir=work_dir,
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
    "HyperplaneConfig",
    "MLPConfig",
    "PRISMConfig",
    "PULSEConfig",
    "TITANConfig",
    "OptimizationResult",
]
