"""WICHER (Whirlwind) steering method constants and search priors."""
from wisent.core.utils.config_tools.constants.for_experiments.by_method.simple import *  # noqa: F401,F403

# --- WICHER search priors ---
SP_WICHER_CONCEPT_DIMS = (0, 8, 16, 32, 64)
SP_WICHER_VAR_THRESH_LOW = 0.8
SP_WICHER_VAR_THRESH_HIGH = 0.99
SP_WICHER_NUM_STEPS_MIN = 1
SP_WICHER_NUM_STEPS_MAX = 20
SP_WICHER_ALPHA_MU = -3.0
SP_WICHER_ALPHA_SIGMA = 2.0
SP_WICHER_ETA_MU = 0.0
SP_WICHER_ETA_SIGMA = 0.5
SP_WICHER_BETA_LOW = 0.0
SP_WICHER_BETA_HIGH = 1.0
SP_WICHER_ALPHA_DECAY_LOW = 0.8
SP_WICHER_ALPHA_DECAY_HIGH = 1.0
