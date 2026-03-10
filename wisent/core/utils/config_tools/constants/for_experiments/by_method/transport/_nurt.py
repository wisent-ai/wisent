"""NURT (Current) steering method constants and search space bounds."""
from wisent.core.utils.config_tools.constants.for_experiments.by_method.transport._szlak import *  # noqa: F401,F403

# --- NURT code-enforced bounds ---
SS_NURT_VAR_THRESH_LOW = 0.0
SS_NURT_VAR_THRESH_HIGH = 1.0
SS_NURT_INTEGRATION_STEPS_MIN = 1
SS_NURT_INTEGRATION_STEPS_MAX = 50
SS_NURT_T_MAX_LOW = -1.0
SS_NURT_T_MAX_HIGH = 1.0
SS_NURT_FLOW_HIDDEN_DIM_MIN = 0
SS_NURT_FLOW_HIDDEN_DIM_MAX = 256
