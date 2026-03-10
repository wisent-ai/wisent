"""PRZELOM (Breakthrough) steering method constants and search space bounds."""
from wisent.core.utils.config_tools.constants.for_experiments.by_method.transport._wicher import *  # noqa: F401,F403

# --- PRZELOM code-enforced bounds ---
SS_PRZELOM_TARGET_MODES = ("uniform", "nearest")
SS_PRZELOM_INFERENCE_K_MIN = 1
SS_PRZELOM_INFERENCE_K_MAX = 10
