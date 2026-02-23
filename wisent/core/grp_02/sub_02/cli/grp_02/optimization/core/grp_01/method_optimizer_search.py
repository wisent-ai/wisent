"""Search space generation for MethodOptimizer."""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

from wisent.core.activations import ExtractionStrategy
from wisent.core.cli.optimization.core.method_optimizer_config import OptimizationConfig
from wisent.core.steering_methods.registry import SteeringMethodRegistry

logger = logging.getLogger(__name__)


def _log(self, msg: str):
    if self.verbose:
        print(msg)


def generate_search_space(
    self,
    num_layers: int,
    custom_layers: Optional[List[int]] = None,
    custom_strengths: Optional[List[float]] = None,
    custom_token_aggregations: Optional[List[str]] = None,
    custom_prompt_strategies: Optional[List[str]] = None,
    custom_method_params: Optional[Dict[str, List[Any]]] = None,
    evidence_reductions: Optional[Dict] = None,
) -> List[OptimizationConfig]:
    """
    Generate search space for optimization.

    Args:
        num_layers: Number of layers in the model
        custom_*: Override default search values
        evidence_reductions: Dict[str, AxisReduction] from EvidenceLedger

    Returns:
        List of OptimizationConfig to test
    """
    layers = custom_layers or self._get_full_layers(num_layers)
    strengths = custom_strengths or [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    token_aggs = custom_token_aggregations or ["last_token", "mean_pooling", "first_token", "max_pooling", "continuation_token"]
    prompt_strats = custom_prompt_strategies or ["chat_template", "direct_completion", "multiple_choice", "role_playing", "instruction_following"]
    steering_strategies = ["constant", "initial_only", "diminishing", "increasing", "gaussian"]

    # Apply evidence-based reductions (only when no custom override)
    if evidence_reductions:
        if not custom_token_aggregations and "extraction_strategy" in evidence_reductions:
            keep = evidence_reductions["extraction_strategy"].keep_values
            token_aggs = [v for v in token_aggs if v in keep] or token_aggs
            logger.info("Evidence: extraction_strategy -> %s", token_aggs)
        if not custom_prompt_strategies and "prompt_strategy" in evidence_reductions:
            keep = evidence_reductions["prompt_strategy"].keep_values
            prompt_strats = [v for v in prompt_strats if v in keep] or prompt_strats
            logger.info("Evidence: prompt_strategy -> %s", prompt_strats)
        if "steering_strategy" in evidence_reductions:
            keep = evidence_reductions["steering_strategy"].keep_values
            steering_strategies = [v for v in steering_strategies if v in keep] or steering_strategies
            logger.info("Evidence: steering_strategy -> %s", steering_strategies)
        if not custom_strengths and "strength" in evidence_reductions:
            keep = evidence_reductions["strength"].keep_values
            strengths = [s for s in strengths if str(s) in keep] or strengths
            logger.info("Evidence: strength -> %s", strengths)

    # Get method-specific parameter ranges
    method_param_ranges = self._get_method_param_ranges(custom_method_params)

    # Apply evidence reductions to method-specific params
    if evidence_reductions and method_param_ranges:
        for param_name in list(method_param_ranges.keys()):
            if param_name in evidence_reductions:
                keep = evidence_reductions[param_name].keep_values
                orig = method_param_ranges[param_name]
                filtered = [v for v in orig if str(v) in keep]
                if filtered:
                    method_param_ranges[param_name] = filtered
                    logger.info("Evidence: %s -> %s", param_name, filtered)

    # Generate all configurations
    configs = []

    # Convert to enums
    token_agg_map = {
        "last_token": ExtractionStrategy.CHAT_LAST,
        "mean_pooling": ExtractionStrategy.CHAT_MEAN,
        "first_token": ExtractionStrategy.CHAT_FIRST,
        "max_pooling": ExtractionStrategy.CHAT_MAX_NORM,
        "continuation_token": ExtractionStrategy.CHAT_FIRST,  # First answer token
    }

    prompt_strat_map = {
        "chat_template": ExtractionStrategy.CHAT_LAST,
        "direct_completion": ExtractionStrategy.CHAT_LAST,
        "multiple_choice": ExtractionStrategy.MC_BALANCED,
        "role_playing": ExtractionStrategy.ROLE_PLAY,
        "instruction_following": ExtractionStrategy.CHAT_LAST,
    }

    # Generate method param combinations
    method_param_combos = self._generate_param_combinations(method_param_ranges)

    for layer in layers:
        for strength in strengths:
            for token_agg_name in token_aggs:
                for prompt_strat_name in prompt_strats:
                    for steering_strat in steering_strategies:
                        for method_params in method_param_combos:
                            # Determine which layers to collect activations for
                            # For multi-layer methods, collect all needed layers
                            activation_layers = self._determine_activation_layers(
                                layer, num_layers, method_params
                            )

                            config = OptimizationConfig(
                                method_name=self.method_name,
                                layers=[str(l) for l in activation_layers],
                                token_aggregation=token_agg_map.get(
                                    token_agg_name, ExtractionStrategy.CHAT_LAST
                                ),
                                prompt_strategy=prompt_strat_map.get(
                                    prompt_strat_name, ExtractionStrategy.CHAT_LAST
                                ),
                                strength=strength,
                                strategy=steering_strat,
                                method_params=method_params,
                            )
                            configs.append(config)

    return configs


def _get_full_layers(self, num_layers: int) -> List[int]:
    """Get full layer set for comprehensive search."""
    # Test ALL layers from 0 to num_layers-1
    return list(range(num_layers))


def _get_method_param_ranges(
    self,
    custom: Optional[Dict[str, List[Any]]] = None,
) -> Dict[str, List[Any]]:
    """Get method-specific parameter ranges."""
    custom = custom or {}

    if self.method_name == "caa":
        return {
            "normalize": custom.get("normalize", [True]),
        }

    elif self.method_name == "tecza":
        return {
            "num_directions": custom.get("num_directions", [1, 2, 3, 5]),
            "optimization_steps": custom.get("optimization_steps", [50, 100]),
            "retain_weight": custom.get("retain_weight", [0.0, 0.1, 0.3]),
            "learning_rate": custom.get("learning_rate", [0.01]),
            "independence_weight": custom.get("independence_weight", [0.05]),
            "use_caa_init": custom.get("use_caa_init", [True]),
        }

    elif self.method_name == "tetno":
        nl = self.model.num_layers
        sensor_defaults = sorted(set(
                list(range(0, nl, max(1, nl // 4))) + [nl - 1]
            ))
        return {
            "sensor_layer": custom.get("sensor_layer", sensor_defaults),
            "steering_layers": custom.get("steering_layers", ["single", "range_3", "range_5"]),
            "condition_threshold": custom.get("condition_threshold", [0.3, 0.5, 0.7]),
            "gate_temperature": custom.get("gate_temperature", [0.1, 0.5, 1.0]),
            "per_layer_scaling": custom.get("per_layer_scaling", [True, False]),
            "use_entropy_scaling": custom.get("use_entropy_scaling", [True, False]),
            "max_alpha": custom.get("max_alpha", [1.5, 2.0, 3.0]),
        }

    elif self.method_name == "grom":
        nl = self.model.num_layers
        sensor_defaults = sorted(set(
                list(range(0, nl, max(1, nl // 4))) + [nl - 1]
            ))
        return {
            "num_directions": custom.get("num_directions", [2, 3, 5]),
            "sensor_layer": custom.get("sensor_layer", sensor_defaults),
            "steering_layers": custom.get("steering_layers", ["range_3", "range_5"]),
            "gate_hidden_dim": custom.get("gate_hidden_dim", [32, 64, 128]),
            "intensity_hidden_dim": custom.get("intensity_hidden_dim", [16, 32, 64]),
            "optimization_steps": custom.get("optimization_steps", [100, 200]),
            "behavior_weight": custom.get("behavior_weight", [0.5, 1.0]),
            "retain_weight": custom.get("retain_weight", [0.1, 0.2, 0.5]),
            "sparse_weight": custom.get("sparse_weight", [0.0, 0.05]),
            "max_alpha": custom.get("max_alpha", [2.0, 3.0]),
        }

    # Default for unknown methods - empty params
    return {}


def _generate_param_combinations(
    self,
    param_ranges: Dict[str, List[Any]],
) -> List[Dict[str, Any]]:
    """Generate all combinations of method parameters."""
    if not param_ranges:
        return [{}]

    import itertools

    keys = list(param_ranges.keys())
    values = [param_ranges[k] for k in keys]

    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


def _determine_activation_layers(
    self,
    base_layer: int,
    num_layers: int,
    method_params: Dict[str, Any],
) -> List[int]:
    """Determine which layers to collect activations for."""
    # For methods that need multi-layer activations
    steering_layers_config = method_params.get("steering_layers", "single")

    if steering_layers_config == "single":
        return [base_layer]
    elif steering_layers_config == "range_3":
        return list(range(max(0, base_layer - 1), min(num_layers, base_layer + 2)))
    elif steering_layers_config == "range_5":
        return list(range(max(0, base_layer - 2), min(num_layers, base_layer + 3)))
    elif isinstance(steering_layers_config, list):
        return steering_layers_config
    else:
        return [base_layer]


def collect_activations(
    self,
    pairs: ContrastivePairSet,
    config: OptimizationConfig,
) -> ContrastivePairSet:
    """
    Collect activations for a pair set using the given config.

    Args:
        pairs: ContrastivePairSet to collect activations for
        config: Configuration specifying layers, aggregation, etc.

    Returns:
        ContrastivePairSet with activations populated
    """
    # Store activations on CPU - GROM and other methods expect CPU tensors for training
    collector = ActivationCollector(model=self.model, store_device="cpu")

    updated_pairs = []
    for pair in pairs.pairs:
        updated_pair = collector.collect(
            pair,
            strategy=config.token_aggregation,
            layers=config.layers,
            normalize=False,
        )
        updated_pairs.append(updated_pair)

    return ContrastivePairSet(
        name=pairs.name if hasattr(pairs, 'name') else "collected",
        pairs=updated_pairs,
        task_type=pairs.task_type if hasattr(pairs, 'task_type') else None,
    )


def _load_evidence_reductions(self, task_name: str):
    """Load evidence-based search-space reductions (returns empty on error)."""
    try:
        from wisent.core.steering_optimizer.constants_registry.evidence import (
            EvidenceLedger,
        )
        ledger = EvidenceLedger()
        reductions = ledger.get_reductions(
            model_name=self.model.model_name,
            task_name=task_name,
            method_name=self.method_name,
        )
        if reductions:
            self._log(
                f"Evidence ledger: {len(reductions)} axis reduction(s) loaded"
            )
        return reductions
    except Exception as exc:
        logger.debug("Evidence ledger unavailable: %s", exc)
        return {}
