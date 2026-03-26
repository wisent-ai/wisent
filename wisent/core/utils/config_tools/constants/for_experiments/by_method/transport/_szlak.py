"""SZLAK (Trail) steering method constants and search space bounds."""
from wisent.core.utils.config_tools.constants.for_experiments.by_method.transport._przelom import *  # noqa: F401,F403

# --- SZLAK code-enforced bounds ---
SS_SZLAK_MAX_ITER_MIN = 1
SS_SZLAK_MAX_ITER_MAX = 200
SS_SZLAK_INFERENCE_K_MIN = 1
SS_SZLAK_INFERENCE_K_MAX = 10

# Methods that require Q/K projection capture during activation collection
METHODS_REQUIRING_QK_CAPTURE = frozenset({"szlak", "Szlak", "SZLAK", "przelom", "Przelom", "PRZELOM"})
