"""NURT (Current) steering method constants."""
from wisent.core.utils.constants.for_experiments.by_method.transport._szlak import *  # noqa: F401,F403

# --- Core NURT parameters ---
NURT_NUM_DIMS = 0
NURT_TRAINING_EPOCHS = 300
NURT_LR_MIN = 0.0001
NURT_NUM_INTEGRATION_STEPS = 4
NURT_T_MAX = 1.0

# --- NURT Optuna bounds ---
OPTUNA_NURT_STEPS_MIN = 2
OPTUNA_NURT_STEPS_MAX = 8

# --- NURT architecture ---
NURT_MAX_CONCEPT_DIM = 10
NURT_FLOW_HIDDEN_MIN = 32
NURT_FLOW_HIDDEN_MAX = 128
NURT_FLOW_HIDDEN_MULTIPLIER = 4

# --- NURT parser defaults ---
NURT_DEFAULT_STRENGTH = 3.0
NURT_DEFAULT_TEMPERATURE = 0.5
