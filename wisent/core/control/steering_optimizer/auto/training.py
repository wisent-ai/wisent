"""Training helpers for auto steering optimization."""
from __future__ import annotations
import logging
from typing import List, Dict, Any
from wisent.core.utils.config_tools.constants import (
    ARCHITECTURE_MODULE_LIMIT,
    COMBO_OFFSET, PROGRESS_LOG_INTERVAL_25, RECURSION_INITIAL_DEPTH,
)

logger = logging.getLogger(__name__)


def train_recommended_method(
    wisent_model: Any, pairs: List[Any], method: str, layer: int,
    verbose: bool = False, *,
    tetno_condition_threshold: float, tetno_gate_temperature: float,
    tetno_max_alpha: float, tetno_entropy_floor: float,
    tetno_entropy_ceiling: float, tetno_threshold_search_steps: int,
    tetno_learning_rate: float, tetno_condition_margin: float,
    tetno_min_layer_scale: float, tetno_log_interval: int,
    tetno_optimization_steps: int,
    grom_num_directions: int, grom_optimization_steps: int,
    grom_learning_rate: float, grom_warmup_steps: int,
    grom_behavior_weight: float, grom_retain_weight: float,
    grom_sparse_weight: float, grom_smooth_weight: float,
    grom_independence_weight: float, grom_max_alpha: float,
    grom_gate_temperature: float, grom_max_grad_norm: float,
    grom_eta_min_factor: float, grom_linear_threshold: float,
    grom_adapt_cone_threshold: float, grom_adapt_manifold_threshold: float,
    grom_adapt_linear_directions: int, grom_adapt_complex_directions: int,
    grom_adapt_max_directions: int, grom_significant_directions_default: int,
    grom_min_adapted_directions: int, grom_caa_similarity_skip: float,
    grom_contrastive_margin: float, grom_contrastive_weight: float,
    grom_utility_weight: float, grom_concentration_weight: float,
    grom_gate_warmup_weight: float, grom_caa_alignment_weight: float,
    grom_gate_dim_min: int, grom_gate_dim_max: int,
    grom_gate_dim_divisor: int, grom_intensity_dim_min: int,
    grom_intensity_dim_max: int, grom_intensity_dim_divisor: int,
    grom_log_interval: int, grom_weight_decay: float,
    tecza_num_directions: int, tecza_learning_rate: float,
    tecza_retain_weight: float, tecza_independence_weight: float,
    tecza_min_cosine_sim: float, tecza_max_cosine_sim: float,
    tecza_marginal_threshold: float, tecza_max_directions: int,
    tecza_ablation_weight: float, tecza_addition_weight: float,
    tecza_separation_margin: float, tecza_perturbation_scale: float,
    tecza_universal_basis_noise: float, tecza_optimization_steps: int,
) -> Dict[str, Any]:
    """Train the recommended steering method (CAA, GROM, TECZA, or TETNO)."""
    from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet
    from wisent.core.primitives.model_interface.core.activations.activations_collector import ActivationCollector
    from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy
    num_layers = wisent_model.num_layers
    all_layers = [str(i) for i in range(COMBO_OFFSET, num_layers + COMBO_OFFSET)]
    if verbose:
        print(f"   Collecting activations for {len(pairs)} pairs...")
    collector = ActivationCollector(model=wisent_model, architecture_module_limit=ARCHITECTURE_MODULE_LIMIT)
    enriched_pairs = []
    for i, pair in enumerate(pairs):
        enriched = collector.collect(pair, strategy=ExtractionStrategy.CHAT_LAST, layers=all_layers)
        enriched_pairs.append(enriched)
        if verbose and (i + COMBO_OFFSET) % PROGRESS_LOG_INTERVAL_25 == RECURSION_INITIAL_DEPTH:
            print(f"     {i + COMBO_OFFSET}/{len(pairs)} pairs processed")
    pair_set = ContrastivePairSet(pairs=enriched_pairs, name=f"{method.lower()}_training")
    if verbose:
        print(f"   Collected activations for {len(enriched_pairs)} pairs")
    if method == "CAA":
        return _train_caa(pair_set, verbose)
    elif method == "GROM":
        return _train_grom(
            wisent_model, pair_set, all_layers, verbose,
            num_directions=grom_num_directions, optimization_steps=grom_optimization_steps,
            learning_rate=grom_learning_rate, warmup_steps=grom_warmup_steps,
            behavior_weight=grom_behavior_weight, retain_weight=grom_retain_weight,
            sparse_weight=grom_sparse_weight, smooth_weight=grom_smooth_weight,
            independence_weight=grom_independence_weight, max_alpha=grom_max_alpha,
            gate_temperature=grom_gate_temperature, max_grad_norm=grom_max_grad_norm,
            eta_min_factor=grom_eta_min_factor, linear_threshold=grom_linear_threshold,
            min_cosine_similarity=tecza_min_cosine_sim,
            max_cosine_similarity=tecza_max_cosine_sim,
            adapt_cone_threshold=grom_adapt_cone_threshold,
            adapt_manifold_threshold=grom_adapt_manifold_threshold,
            adapt_linear_directions=grom_adapt_linear_directions,
            adapt_complex_directions=grom_adapt_complex_directions,
            adapt_max_directions=grom_adapt_max_directions,
            significant_directions_default=grom_significant_directions_default,
            min_adapted_directions=grom_min_adapted_directions,
            caa_similarity_skip=grom_caa_similarity_skip,
            contrastive_margin=grom_contrastive_margin,
            contrastive_weight=grom_contrastive_weight,
            utility_weight=grom_utility_weight,
            concentration_weight=grom_concentration_weight,
            gate_warmup_weight=grom_gate_warmup_weight,
            caa_alignment_weight=grom_caa_alignment_weight,
            gate_dim_min=grom_gate_dim_min, gate_dim_max=grom_gate_dim_max,
            gate_dim_divisor=grom_gate_dim_divisor,
            intensity_dim_min=grom_intensity_dim_min,
            intensity_dim_max=grom_intensity_dim_max,
            intensity_dim_divisor=grom_intensity_dim_divisor,
            log_interval=grom_log_interval, weight_decay=grom_weight_decay,
        )
    elif method == "TECZA":
        return _train_tecza(
            wisent_model, pair_set, verbose,
            num_directions=tecza_num_directions, optimization_steps=tecza_optimization_steps,
            learning_rate=tecza_learning_rate,
            retain_weight=tecza_retain_weight, independence_weight=tecza_independence_weight,
            min_cosine_similarity=tecza_min_cosine_sim,
            max_cosine_similarity=tecza_max_cosine_sim,
            marginal_threshold=tecza_marginal_threshold, max_directions=tecza_max_directions,
            ablation_weight=tecza_ablation_weight, addition_weight=tecza_addition_weight,
            separation_margin=tecza_separation_margin,
            perturbation_scale=tecza_perturbation_scale,
            universal_basis_noise=tecza_universal_basis_noise,
        )
    elif method == "TETNO":
        return _train_tetno(
            wisent_model, pair_set, all_layers, verbose,
            condition_threshold=tetno_condition_threshold,
            gate_temperature=tetno_gate_temperature, max_alpha=tetno_max_alpha,
            entropy_floor=tetno_entropy_floor, entropy_ceiling=tetno_entropy_ceiling,
            threshold_search_steps=tetno_threshold_search_steps,
            optimization_steps=tetno_optimization_steps, learning_rate=tetno_learning_rate,
            condition_margin=tetno_condition_margin, min_layer_scale=tetno_min_layer_scale,
            log_interval=tetno_log_interval,
        )
    else:
        logger.warning(f"Unknown method {method}, falling back to CAA")
        return _train_caa(pair_set, verbose)


def _train_caa(pair_set: Any, verbose: bool) -> Dict[str, Any]:
    """Train CAA method."""
    from wisent.core.control.steering_methods.methods.caa import CAAMethod
    caa_method = CAAMethod()
    result = caa_method.train(pair_set)
    if verbose:
        print(f"   CAA trained on {len(pair_set.pairs)} pairs")
        print(f"     Layers: {len(result.directions)}")
    return {"method": "CAA", "layers": len(result.directions), "result": result,
            "method_params": {"normalize": True}}


def _train_grom(
    wisent_model: Any, pair_set: Any, all_layers: List[str], verbose: bool, *,
    num_directions: int, optimization_steps: int, learning_rate: float,
    warmup_steps: int, behavior_weight: float, retain_weight: float,
    sparse_weight: float, smooth_weight: float, independence_weight: float,
    max_alpha: float, gate_temperature: float, max_grad_norm: float,
    eta_min_factor: float, linear_threshold: float,
    min_cosine_similarity: float, max_cosine_similarity: float,
    adapt_cone_threshold: float, adapt_manifold_threshold: float,
    adapt_linear_directions: int, adapt_complex_directions: int,
    adapt_max_directions: int, significant_directions_default: int,
    min_adapted_directions: int, caa_similarity_skip: float,
    contrastive_margin: float, contrastive_weight: float,
    utility_weight: float, concentration_weight: float,
    gate_warmup_weight: float, caa_alignment_weight: float,
    gate_dim_min: int, gate_dim_max: int, gate_dim_divisor: int,
    intensity_dim_min: int, intensity_dim_max: int, intensity_dim_divisor: int,
    log_interval: int, weight_decay: float,
) -> Dict[str, Any]:
    """Train GROM method."""
    from wisent.core.control.steering_methods.methods.grom import GROMMethod
    layer_indices = [int(l) for l in all_layers]
    first_layer = next(iter(layer_indices))
    grom_method = GROMMethod(
        model=wisent_model, num_directions=num_directions,
        steering_layers=layer_indices, sensor_layer=first_layer,
        optimization_steps=optimization_steps, learning_rate=learning_rate,
        warmup_steps=warmup_steps, behavior_weight=behavior_weight,
        retain_weight=retain_weight, sparse_weight=sparse_weight,
        smooth_weight=smooth_weight, independence_weight=independence_weight,
        max_alpha=max_alpha, gate_temperature=gate_temperature,
        min_cosine_similarity=min_cosine_similarity,
        max_cosine_similarity=max_cosine_similarity,
        weight_decay=weight_decay, max_grad_norm=max_grad_norm,
        eta_min_factor=eta_min_factor, linear_threshold=linear_threshold,
        adapt_cone_threshold=adapt_cone_threshold,
        adapt_manifold_threshold=adapt_manifold_threshold,
        adapt_linear_directions=adapt_linear_directions,
        adapt_complex_directions=adapt_complex_directions,
        adapt_max_directions=adapt_max_directions,
        significant_directions_default=significant_directions_default,
        min_adapted_directions=min_adapted_directions,
        caa_similarity_skip=caa_similarity_skip,
        contrastive_margin=contrastive_margin,
        contrastive_weight=contrastive_weight,
        utility_weight=utility_weight,
        concentration_weight=concentration_weight,
        gate_warmup_weight=gate_warmup_weight,
        caa_alignment_weight=caa_alignment_weight,
        gate_dim_min=gate_dim_min, gate_dim_max=gate_dim_max,
        gate_dim_divisor=gate_dim_divisor,
        intensity_dim_min=intensity_dim_min,
        intensity_dim_max=intensity_dim_max,
        intensity_dim_divisor=intensity_dim_divisor,
        log_interval=log_interval,
    )
    result = grom_method.train_grom(pair_set)
    if verbose:
        first_order_layer = result.layer_order[RECURSION_INITIAL_DEPTH]
        print(f"   GROM trained on {len(pair_set.pairs)} pairs")
        print(f"     Layers: {len(result.layer_order)}")
        print(f"     Directions per layer: {result.directions[first_order_layer].shape[RECURSION_INITIAL_DEPTH]}")
    return {"method": "GROM", "layers": len(result.layer_order), "result": result,
            "method_params": grom_method.config.__dict__}


def _train_tecza(
    wisent_model: Any, pair_set: Any, verbose: bool, *,
    num_directions: int, learning_rate: float, retain_weight: float,
    independence_weight: float, optimization_steps: int,
    min_cosine_similarity: float, max_cosine_similarity: float, marginal_threshold: float,
    max_directions: int, ablation_weight: float, addition_weight: float,
    separation_margin: float, perturbation_scale: float,
    universal_basis_noise: float,
    variance_threshold: float = None,
) -> Dict[str, Any]:
    """Train TECZA method."""
    if variance_threshold is None:
        raise ValueError("variance_threshold is required")
    from wisent.core.control.steering_methods.methods.advanced import TECZAMethod
    tecza_method = TECZAMethod(
        model=wisent_model.hf_model, num_directions=num_directions,
        optimization_steps=optimization_steps, learning_rate=learning_rate,
        retain_weight=retain_weight, independence_weight=independence_weight,
        min_cosine_similarity=min_cosine_similarity,
        max_cosine_similarity=max_cosine_similarity,
        variance_threshold=variance_threshold,
        marginal_threshold=marginal_threshold, max_directions=max_directions,
        ablation_weight=ablation_weight, addition_weight=addition_weight,
        separation_margin=separation_margin, perturbation_scale=perturbation_scale,
        universal_basis_noise=universal_basis_noise,
    )
    result = tecza_method.train(pair_set)
    if verbose:
        first_layer_dirs = next(iter(result.directions.values()))
        print(f"   TECZA trained on {len(pair_set.pairs)} pairs")
        print(f"     Layers: {len(result.directions)}, Directions: {first_layer_dirs.shape[RECURSION_INITIAL_DEPTH]}")
    return {"method": "TECZA", "layers": len(result.directions), "result": result,
            "method_params": tecza_method.config.__dict__}


def _train_tetno(
    wisent_model: Any, pair_set: Any, all_layers: List[str], verbose: bool, *,
    condition_threshold: float, gate_temperature: float, max_alpha: float,
    optimization_steps: int, entropy_floor: float, entropy_ceiling: float,
    threshold_search_steps: int, learning_rate: float, condition_margin: float,
    min_layer_scale: float, log_interval: int,
) -> Dict[str, Any]:
    """Train TETNO method."""
    from wisent.core.control.steering_methods.methods.advanced import TETNOMethod
    layer_indices = [int(l) for l in all_layers]
    first_layer = next(iter(layer_indices))
    tetno_method = TETNOMethod(
        model=wisent_model.hf_model, steering_layers=layer_indices,
        sensor_layer=first_layer, condition_threshold=condition_threshold,
        gate_temperature=gate_temperature, max_alpha=max_alpha,
        optimization_steps=optimization_steps, learning_rate=learning_rate,
        entropy_floor=entropy_floor, entropy_ceiling=entropy_ceiling,
        threshold_search_steps=threshold_search_steps,
        condition_margin=condition_margin, min_layer_scale=min_layer_scale,
        log_interval=log_interval,
    )
    result = tetno_method.train_tetno(pair_set)
    if verbose:
        print(f"   TETNO trained on {len(pair_set.pairs)} pairs")
        print(f"     Layers: {len(result.behavior_vectors)}")
        print(f"     Optimal threshold: {result.optimal_threshold:.3f}")
    return {"method": "TETNO", "layers": len(result.behavior_vectors), "result": result,
            "method_params": tetno_method.config.__dict__}
