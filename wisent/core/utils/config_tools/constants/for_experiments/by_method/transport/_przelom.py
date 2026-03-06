"""PRZELOM (Breakthrough) steering method constants and search priors."""
from wisent.core.utils.config_tools.constants.for_experiments.by_method.transport._wicher import *  # noqa: F401,F403

# --- PRZELOM search priors ---
SP_PRZELOM_EPSILON_MU = -2.0
SP_PRZELOM_EPSILON_SIGMA = 1.5
SP_PRZELOM_TARGET_MODES = ("uniform", "nearest")
SP_PRZELOM_REG_MU = -5.0
SP_PRZELOM_REG_SIGMA = 2.0
SP_PRZELOM_INFERENCE_K_MIN = 1
SP_PRZELOM_INFERENCE_K_MAX = 10
