"""SZLAK (Trail) steering method constants."""
from wisent.core.utils.constants.for_experiments.by_method.transport._przelom import *  # noqa: F401,F403

# --- Core SZLAK parameters ---
SZLAK_SINKHORN_REG = 0.1
SZLAK_INFERENCE_K = 5
SZLAK_MAX_ITER = 100
SZLAK_HIGH_REG = 1.0
SZLAK_SPARSE_K = 10

# --- SZLAK Optuna bounds ---
OPTUNA_SZLAK_REG_MIN = 0.01

# --- SZLAK transport ---
SZLAK_GEODESIC_INF_MULTIPLIER = 2.0

# --- SZLAK parser defaults ---
SZLAK_DEFAULT_EPSILON = 0.80
SZLAK_DEFAULT_MAX_ITER = 300
SZLAK_DEFAULT_STEP_SIZE = 0.001
SZLAK_DEFAULT_ITER_PER_SCALE = 4
