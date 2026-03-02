"""MLP steering method constants."""
from wisent.core.infrastructure.constant_definitions.for_experiments.by_method.simple._caa import *  # noqa: F401,F403

# --- Core MLP parameters ---
MLP_HIDDEN_DIM = 256
MLP_NUM_LAYERS = 2
MLP_DROPOUT = 0.1
MLP_LEARNING_RATE = 0.001

# --- MLP training ---
MLP_EARLY_STOPPING_PATIENCE = 20
MLP_INPUT_DIVISOR = 4
MLP_EARLY_STOPPING_MIN_SAMPLES = 20
MLP_PROBE_MAX_ITER = 500

# --- MLP Optuna bounds ---
OPTUNA_MLP_HIDDEN_DIM_MIN = 32
OPTUNA_MLP_HIDDEN_DIM_MAX = 1024
OPTUNA_MLP_NUM_LAYERS_MIN = 1
OPTUNA_MLP_NUM_LAYERS_MAX = 5

# --- MLP search/hierarchical ---
SEARCH_MLP_HIDDEN_DIMS = (128, 256, 512)
SEARCH_MLP_NUM_LAYERS = (1, 2, 3)
HIERARCHICAL_MLP_HIDDEN_DIMS = (64, 128, 256, 512)
HIERARCHICAL_MLP_NUM_LAYERS_LIST = (1, 2, 3)

# --- MLP min pairs ---
MIN_PAIRS_MLP = 50
