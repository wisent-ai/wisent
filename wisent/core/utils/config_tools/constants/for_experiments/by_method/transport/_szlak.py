"""SZLAK (Trail) steering method constants and search priors."""
from wisent.core.utils.config_tools.constants.for_experiments.by_method.transport._przelom import *  # noqa: F401,F403

# --- SZLAK search priors ---
SP_SZLAK_SINKHORN_REG_MU = -2.0
SP_SZLAK_SINKHORN_REG_SIGMA = 1.5
SP_SZLAK_MAX_ITER_MIN = 10
SP_SZLAK_MAX_ITER_MAX = 200
SP_SZLAK_INFERENCE_K_MIN = 1
SP_SZLAK_INFERENCE_K_MAX = 10
