"""Per-method search space definitions using distribution-based parameters."""
from __future__ import annotations

from wisent.core.utils.services.optimization.core.parameters import (
    CategoricalParam as _Cat,
    FloatParam as _Fp,
    IntParam as _Ip,
    Param,
)
from wisent.core.utils.config_tools import constants as _C

EXTRACTION_STRATEGIES = ["chat_last", "chat_mean"]
_SS = ["constant", "initial_only", "diminishing", "increasing", "gaussian"]
_DW = ["primary_only", "equal"]


def _ln(mu, sigma):
    return _Fp(distribution="lognormal", mu=mu, sigma=sigma)


def _nm(mu, sigma):
    return _Fp(distribution="normal", mu=mu, sigma=sigma)


def _uf(lo, hi):
    return _Fp(distribution="uniform", low=lo, high=hi)


def _ri(lo, hi):
    return _Ip(distribution="randint", low=lo, high=hi)


def _qln(mu, sigma):
    return _Ip(distribution="qlognormal", mu=mu, sigma=sigma)


def _base(num_layers: int) -> dict[str, Param]:
    return {
        "extraction_strategy": _Cat(choices=EXTRACTION_STRATEGIES),
        "steering_strategy": _Cat(choices=_SS),
        "strength": _ln(_C.SP_STRENGTH_MU, _C.SP_STRENGTH_SIGMA),
        "layer": _ri(_C.SP_LAYER_MIN, num_layers),
    }


def _sensor_base(num_layers: int) -> dict[str, Param]:
    return {
        "extraction_strategy": _Cat(choices=EXTRACTION_STRATEGIES),
        "steering_strategy": _Cat(choices=_SS),
        "strength": _ln(_C.SP_STRENGTH_MU, _C.SP_STRENGTH_SIGMA),
        "sensor_layer": _ri(_C.SP_LAYER_MIN, num_layers),
        "steering_start": _ri(_C.SP_LAYER_MIN, num_layers),
        "steering_end": _ri(_C.SP_LAYER_MIN, num_layers),
    }


def caa_space(num_layers: int) -> dict[str, Param]:
    return _base(num_layers)


def ostrze_space(num_layers: int) -> dict[str, Param]:
    s = _base(num_layers)
    s["C"] = _ln(_C.SP_OSTRZE_C_MU, _C.SP_OSTRZE_C_SIGMA)
    return s


def mlp_space(num_layers: int) -> dict[str, Param]:
    s = _base(num_layers)
    s["hidden_dim"] = _qln(_C.SP_MLP_HIDDEN_DIM_MU, _C.SP_MLP_HIDDEN_DIM_SIGMA)
    s["num_layers"] = _ri(_C.SP_MLP_NUM_LAYERS_MIN, _C.SP_MLP_NUM_LAYERS_MAX)
    s["dropout"] = _nm(_C.SP_MLP_DROPOUT_MU, _C.SP_MLP_DROPOUT_SIGMA)
    s["epochs"] = _qln(_C.SP_MLP_EPOCHS_MU, _C.SP_MLP_EPOCHS_SIGMA)
    s["learning_rate"] = _ln(_C.SP_LEARNING_RATE_MU, _C.SP_LEARNING_RATE_SIGMA)
    s["weight_decay"] = _ln(_C.SP_MLP_WEIGHT_DECAY_MU, _C.SP_MLP_WEIGHT_DECAY_SIGMA)
    s["early_stop_tol"] = _ln(_C.SP_MLP_EARLY_STOP_TOL_MU, _C.SP_MLP_EARLY_STOP_TOL_SIGMA)
    s["mlp_input_divisor"] = _ri(_C.SP_MLP_INPUT_DIVISOR_MIN, _C.SP_MLP_INPUT_DIVISOR_MAX)
    s["gating_hidden_dim_divisor"] = _ri(_C.SP_MLP_GATING_DIVISOR_MIN, _C.SP_MLP_GATING_DIVISOR_MAX)
    s["mlp_early_stopping_patience"] = _ri(_C.SP_MLP_PATIENCE_MIN, _C.SP_MLP_PATIENCE_MAX)
    return s


def tecza_space(num_layers: int) -> dict[str, Param]:
    s = _base(num_layers)
    s["num_directions"] = _qln(_C.SP_TECZA_NUM_DIR_MU, _C.SP_TECZA_NUM_DIR_SIGMA)
    s["optimization_steps"] = _qln(_C.SP_TECZA_OPT_STEPS_MU, _C.SP_TECZA_OPT_STEPS_SIGMA)
    s["retain_weight"] = _nm(_C.SP_TECZA_RETAIN_MU, _C.SP_TECZA_RETAIN_SIGMA)
    s["learning_rate"] = _ln(_C.SP_LEARNING_RATE_MU, _C.SP_LEARNING_RATE_SIGMA)
    s["independence_weight"] = _ln(_C.SP_TECZA_INDEP_MU, _C.SP_TECZA_INDEP_SIGMA)
    s["min_cosine_similarity"] = _uf(_C.SP_TECZA_MIN_COS_LOW, _C.SP_TECZA_MIN_COS_HIGH)
    s["max_cosine_similarity"] = _uf(_C.SP_TECZA_MAX_COS_LOW, _C.SP_TECZA_MAX_COS_HIGH)
    s["variance_threshold"] = _uf(_C.SP_TECZA_VAR_THRESH_LOW, _C.SP_TECZA_VAR_THRESH_HIGH)
    s["marginal_threshold"] = _uf(_C.SP_TECZA_MARG_THRESH_LOW, _C.SP_TECZA_MARG_THRESH_HIGH)
    s["max_directions"] = _ri(_C.SP_TECZA_MAX_DIR_MIN, _C.SP_TECZA_MAX_DIR_MAX)
    s["ablation_weight"] = _ln(_C.SP_TECZA_ABL_WEIGHT_MU, _C.SP_TECZA_ABL_WEIGHT_SIGMA)
    s["addition_weight"] = _ln(_C.SP_TECZA_ADD_WEIGHT_MU, _C.SP_TECZA_ADD_WEIGHT_SIGMA)
    s["separation_margin"] = _nm(_C.SP_TECZA_SEP_MARGIN_MU, _C.SP_TECZA_SEP_MARGIN_SIGMA)
    s["perturbation_scale"] = _ln(_C.SP_TECZA_PERTURB_MU, _C.SP_TECZA_PERTURB_SIGMA)
    s["universal_basis_noise"] = _ln(_C.SP_TECZA_NOISE_MU, _C.SP_TECZA_NOISE_SIGMA)
    s["log_interval"] = _ri(_C.SP_TECZA_LOG_INTERVAL_MIN, _C.SP_TECZA_LOG_INTERVAL_MAX)
    s["direction_weighting"] = _Cat(choices=_DW)
    return s


def tetno_space(num_layers: int) -> dict[str, Param]:
    s = _sensor_base(num_layers)
    s["condition_threshold"] = _nm(_C.SP_TETNO_COND_THRESH_MU, _C.SP_TETNO_COND_THRESH_SIGMA)
    s["gate_temperature"] = _ln(_C.SP_TETNO_GATE_TEMP_MU, _C.SP_TETNO_GATE_TEMP_SIGMA)
    s["entropy_floor"] = _nm(_C.SP_TETNO_ENT_FLOOR_MU, _C.SP_TETNO_ENT_FLOOR_SIGMA)
    s["entropy_ceiling"] = _nm(_C.SP_TETNO_ENT_CEIL_MU, _C.SP_TETNO_ENT_CEIL_SIGMA)
    s["max_alpha"] = _ln(_C.SP_TETNO_MAX_ALPHA_MU, _C.SP_TETNO_MAX_ALPHA_SIGMA)
    s["optimization_steps"] = _qln(_C.SP_TETNO_OPT_STEPS_MU, _C.SP_TETNO_OPT_STEPS_SIGMA)
    s["learning_rate"] = _ln(_C.SP_LEARNING_RATE_MU, _C.SP_LEARNING_RATE_SIGMA)
    s["threshold_search_steps"] = _ri(_C.SP_TETNO_THRESH_STEPS_MIN, _C.SP_TETNO_THRESH_STEPS_MAX)
    s["condition_margin"] = _nm(_C.SP_TETNO_COND_MARGIN_MU, _C.SP_TETNO_COND_MARGIN_SIGMA)
    s["min_layer_scale"] = _uf(_C.SP_TETNO_MIN_LAYER_SCALE_LOW, _C.SP_TETNO_MIN_LAYER_SCALE_HIGH)
    s["log_interval"] = _ri(_C.SP_TETNO_LOG_INTERVAL_MIN, _C.SP_TETNO_LOG_INTERVAL_MAX)
    return s


def grom_space(num_layers: int) -> dict[str, Param]:
    s = _sensor_base(num_layers)
    s["num_directions"] = _qln(_C.SP_GROM_NUM_DIR_MU, _C.SP_GROM_NUM_DIR_SIGMA)
    s["gate_hidden_dim"] = _qln(_C.SP_GROM_GATE_DIM_MU, _C.SP_GROM_GATE_DIM_SIGMA)
    s["intensity_hidden_dim"] = _qln(_C.SP_GROM_INTENSITY_DIM_MU, _C.SP_GROM_INTENSITY_DIM_SIGMA)
    s["optimization_steps"] = _qln(_C.SP_GROM_OPT_STEPS_MU, _C.SP_GROM_OPT_STEPS_SIGMA)
    s["learning_rate"] = _ln(_C.SP_LEARNING_RATE_MU, _C.SP_LEARNING_RATE_SIGMA)
    s["warmup_steps"] = _qln(_C.SP_GROM_WARMUP_MU, _C.SP_GROM_WARMUP_SIGMA)
    s["behavior_weight"] = _ln(_C.SP_GROM_BEHAV_WEIGHT_MU, _C.SP_GROM_BEHAV_WEIGHT_SIGMA)
    s["retain_weight"] = _nm(_C.SP_GROM_RETAIN_WEIGHT_MU, _C.SP_GROM_RETAIN_WEIGHT_SIGMA)
    s["sparse_weight"] = _ln(_C.SP_GROM_SPARSE_WEIGHT_MU, _C.SP_GROM_SPARSE_WEIGHT_SIGMA)
    s["smooth_weight"] = _ln(_C.SP_GROM_SMOOTH_WEIGHT_MU, _C.SP_GROM_SMOOTH_WEIGHT_SIGMA)
    s["independence_weight"] = _ln(_C.SP_GROM_INDEP_WEIGHT_MU, _C.SP_GROM_INDEP_WEIGHT_SIGMA)
    s["max_alpha"] = _ln(_C.SP_GROM_MAX_ALPHA_MU, _C.SP_GROM_MAX_ALPHA_SIGMA)
    s["gate_temperature"] = _ln(_C.SP_GROM_GATE_TEMP_MU, _C.SP_GROM_GATE_TEMP_SIGMA)
    s["max_grad_norm"] = _ln(_C.SP_GROM_MAX_GRAD_NORM_MU, _C.SP_GROM_MAX_GRAD_NORM_SIGMA)
    s["eta_min_factor"] = _uf(_C.SP_GROM_ETA_MIN_FACTOR_LOW, _C.SP_GROM_ETA_MIN_FACTOR_HIGH)
    s["linear_threshold"] = _uf(_C.SP_GROM_LINEAR_THRESH_LOW, _C.SP_GROM_LINEAR_THRESH_HIGH)
    s["adapt_cone_threshold"] = _uf(_C.SP_GROM_ADAPT_CONE_LOW, _C.SP_GROM_ADAPT_CONE_HIGH)
    s["adapt_manifold_threshold"] = _uf(_C.SP_GROM_ADAPT_MANIFOLD_LOW, _C.SP_GROM_ADAPT_MANIFOLD_HIGH)
    s["adapt_linear_directions"] = _ri(_C.SP_GROM_ADAPT_LINEAR_DIR_MIN, _C.SP_GROM_ADAPT_LINEAR_DIR_MAX)
    s["adapt_complex_directions"] = _ri(_C.SP_GROM_ADAPT_COMPLEX_DIR_MIN, _C.SP_GROM_ADAPT_COMPLEX_DIR_MAX)
    s["adapt_max_directions"] = _ri(_C.SP_GROM_ADAPT_MAX_DIR_MIN, _C.SP_GROM_ADAPT_MAX_DIR_MAX)
    s["significant_directions_default"] = _ri(_C.SP_GROM_SIG_DIR_DEFAULT_MIN, _C.SP_GROM_SIG_DIR_DEFAULT_MAX)
    s["min_adapted_directions"] = _ri(_C.SP_GROM_MIN_ADAPTED_DIR_MIN, _C.SP_GROM_MIN_ADAPTED_DIR_MAX)
    s["caa_similarity_skip"] = _uf(_C.SP_GROM_CAA_SIM_SKIP_LOW, _C.SP_GROM_CAA_SIM_SKIP_HIGH)
    s["contrastive_margin"] = _nm(_C.SP_GROM_CONTRASTIVE_MARGIN_MU, _C.SP_GROM_CONTRASTIVE_MARGIN_SIGMA)
    s["contrastive_weight"] = _ln(_C.SP_GROM_CONTRASTIVE_WEIGHT_MU, _C.SP_GROM_CONTRASTIVE_WEIGHT_SIGMA)
    s["utility_weight"] = _ln(_C.SP_GROM_UTILITY_WEIGHT_MU, _C.SP_GROM_UTILITY_WEIGHT_SIGMA)
    s["concentration_weight"] = _ln(_C.SP_GROM_CONC_WEIGHT_MU, _C.SP_GROM_CONC_WEIGHT_SIGMA)
    s["gate_warmup_weight"] = _ln(_C.SP_GROM_GATE_WARMUP_MU, _C.SP_GROM_GATE_WARMUP_SIGMA)
    s["caa_alignment_weight"] = _ln(_C.SP_GROM_CAA_ALIGN_MU, _C.SP_GROM_CAA_ALIGN_SIGMA)
    s["gate_dim_min"] = _ri(_C.SP_GROM_DIM_MIN, _C.SP_GROM_DIM_MAX)
    s["gate_dim_max"] = _ri(_C.SP_GROM_DIM_MIN, _C.SP_GROM_DIM_MAX)
    s["gate_dim_divisor"] = _ri(_C.SP_GROM_DIM_DIVISOR_MIN, _C.SP_GROM_DIM_DIVISOR_MAX)
    s["intensity_dim_min"] = _ri(_C.SP_GROM_DIM_MIN, _C.SP_GROM_DIM_MAX)
    s["intensity_dim_max"] = _ri(_C.SP_GROM_DIM_MIN, _C.SP_GROM_DIM_MAX)
    s["intensity_dim_divisor"] = _ri(_C.SP_GROM_DIM_DIVISOR_MIN, _C.SP_GROM_DIM_DIVISOR_MAX)
    s["gate_shrink_factor"] = _ri(_C.SP_GROM_SHRINK_MIN, _C.SP_GROM_SHRINK_MAX)
    s["weight_decay"] = _ln(_C.SP_GROM_WEIGHT_DECAY_MU, _C.SP_GROM_WEIGHT_DECAY_SIGMA)
    s["min_cosine_sim"] = _uf(_C.SP_GROM_MIN_COS_LOW, _C.SP_GROM_MIN_COS_HIGH)
    s["max_cosine_sim"] = _uf(_C.SP_GROM_MAX_COS_LOW, _C.SP_GROM_MAX_COS_HIGH)
    s["create_noise_scale"] = _ln(_C.SP_GROM_NOISE_SCALE_MU, _C.SP_GROM_NOISE_SCALE_SIGMA)
    s["create_gate_threshold"] = _uf(_C.SP_GROM_GATE_THRESH_LOW, _C.SP_GROM_GATE_THRESH_HIGH)
    s["log_interval"] = _ri(_C.SP_GROM_LOG_INTERVAL_MIN, _C.SP_GROM_LOG_INTERVAL_MAX)
    return s


def nurt_space(num_layers: int) -> dict[str, Param]:
    s = _base(num_layers)
    s["num_dims"] = _ri(_C.SP_NURT_CONCEPT_DIM_MIN, _C.SP_NURT_CONCEPT_DIM_MAX)
    s["max_concept_dim"] = _ri(_C.SP_NURT_MAX_CONCEPT_DIM_MIN, _C.SP_NURT_MAX_CONCEPT_DIM_MAX)
    s["variance_threshold"] = _uf(_C.SP_NURT_VAR_THRESH_LOW, _C.SP_NURT_VAR_THRESH_HIGH)
    s["training_epochs"] = _qln(_C.SP_NURT_EPOCHS_MU, _C.SP_NURT_EPOCHS_SIGMA)
    s["lr"] = _ln(_C.SP_LEARNING_RATE_MU, _C.SP_LEARNING_RATE_SIGMA)
    s["lr_min"] = _ln(_C.SP_NURT_LR_MIN_MU, _C.SP_NURT_LR_MIN_SIGMA)
    s["num_integration_steps"] = _ri(_C.SP_NURT_INTEGRATION_STEPS_MIN, _C.SP_NURT_INTEGRATION_STEPS_MAX)
    s["t_max"] = _nm(_C.SP_NURT_T_MAX_MU, _C.SP_NURT_T_MAX_SIGMA)
    s["flow_hidden_dim"] = _ri(_C.SP_NURT_FLOW_HIDDEN_DIM_MIN, _C.SP_NURT_FLOW_HIDDEN_DIM_MAX)
    s["weight_decay"] = _ln(_C.SP_NURT_WEIGHT_DECAY_MU, _C.SP_NURT_WEIGHT_DECAY_SIGMA)
    s["max_grad_norm"] = _ln(_C.SP_NURT_MAX_GRAD_NORM_MU, _C.SP_NURT_MAX_GRAD_NORM_SIGMA)
    return s


def szlak_space(num_layers: int) -> dict[str, Param]:
    s = _base(num_layers)
    s["sinkhorn_reg"] = _ln(_C.SP_SZLAK_SINKHORN_REG_MU, _C.SP_SZLAK_SINKHORN_REG_SIGMA)
    s["max_iter"] = _ri(_C.SP_SZLAK_MAX_ITER_MIN, _C.SP_SZLAK_MAX_ITER_MAX)
    s["inference_k"] = _ri(_C.SP_SZLAK_INFERENCE_K_MIN, _C.SP_SZLAK_INFERENCE_K_MAX)
    return s


def wicher_space(num_layers: int) -> dict[str, Param]:
    s = _base(num_layers)
    s["concept_dim"] = _Cat(choices=list(_C.SP_WICHER_CONCEPT_DIMS))
    s["variance_threshold"] = _uf(_C.SP_WICHER_VAR_THRESH_LOW, _C.SP_WICHER_VAR_THRESH_HIGH)
    s["num_steps"] = _ri(_C.SP_WICHER_NUM_STEPS_MIN, _C.SP_WICHER_NUM_STEPS_MAX)
    s["alpha"] = _ln(_C.SP_WICHER_ALPHA_MU, _C.SP_WICHER_ALPHA_SIGMA)
    s["eta"] = _nm(_C.SP_WICHER_ETA_MU, _C.SP_WICHER_ETA_SIGMA)
    s["beta"] = _uf(_C.SP_WICHER_BETA_LOW, _C.SP_WICHER_BETA_HIGH)
    s["alpha_decay"] = _uf(_C.SP_WICHER_ALPHA_DECAY_LOW, _C.SP_WICHER_ALPHA_DECAY_HIGH)
    return s


def przelom_space(num_layers: int) -> dict[str, Param]:
    s = _base(num_layers)
    s["epsilon"] = _ln(_C.SP_PRZELOM_EPSILON_MU, _C.SP_PRZELOM_EPSILON_SIGMA)
    s["target_mode"] = _Cat(choices=list(_C.SP_PRZELOM_TARGET_MODES))
    s["regularization"] = _ln(_C.SP_PRZELOM_REG_MU, _C.SP_PRZELOM_REG_SIGMA)
    s["inference_k"] = _ri(_C.SP_PRZELOM_INFERENCE_K_MIN, _C.SP_PRZELOM_INFERENCE_K_MAX)
    return s


def get_method_space(method: str, num_layers: int) -> dict[str, Param]:
    """Look up the search space for a method by name."""
    dispatch = {
        "CAA": caa_space, "OSTRZE": ostrze_space, "MLP": mlp_space,
        "TECZA": tecza_space, "TETNO": tetno_space, "GROM": grom_space,
        "NURT": nurt_space, "SZLAK": szlak_space,
        "WICHER": wicher_space, "PRZELOM": przelom_space,
    }
    fn = dispatch.get(method.upper())
    if fn is None:
        raise ValueError(f"No search space for method: {method}")
    return fn(num_layers)
