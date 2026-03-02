"""
Named numeric constants for the wisent package.
ALL numeric defaults MUST be defined here and imported by name.
"""
from wisent.core.utils.config_tools.constants.validated import *  # noqa: F401,F403
from wisent.core.utils.config_tools.constants.for_experiments import *  # noqa: F401,F403
from wisent.core.utils.config_tools.constants.cannot_be_optimized import *  # noqa: F401,F403

# --- Numerical stability ---
NORM_EPS = 1e-8
LOG_EPS = 1e-12
ZERO_THRESHOLD = 1e-10
COMPARE_TOL = 1e-6
NEAR_ZERO_TOL = 1e-9
SHERMAN_MORRISON_EPS = 1e-12

# --- Regularization / tolerances ---
TIKHONOV_REG = 1e-4
EARLY_STOP_TOL = 1e-4
MIN_NORM_THRESHOLD = 1e-4
DEFAULT_CLASSIFIER_LR = 1e-3
L1_DEFAULT_COEF = 1e-3
MATH_REL_TOL = 1e-4
MATH_PERCENT_REL_TOL = 1e-3
SYMPY_REL_TOL = 1e-9

# --- Optuna search bounds ---
LR_LOWER_BOUND = 1e-4
LR_UPPER_BOUND = 1e-2
ALPHA_LOWER_BOUND = 1e-4
ALPHA_UPPER_BOUND = 1e-1
