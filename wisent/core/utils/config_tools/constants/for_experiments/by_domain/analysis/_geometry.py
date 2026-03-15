"""Geometry, linearity, stability, and subspace constants."""
from wisent.core.utils.config_tools.constants.for_experiments.by_domain.analysis._analysis import *  # noqa: F401,F403

# --- Geometry metric constants (A: fixed by definition, B: data-derivable) ---
# Category A reuse validated constants (CHANCE_LEVEL_ACCURACY, SCORE_RANGE_MIN, COMBO_OFFSET)
# Category C (non-derivable design choices) are now required CLI arguments

GEOMETRY_VARIANCE_EXPLAINED_90PCT = 0.9  # "90%" in "dims for 90% variance"

# Category B: min_cloud_points (minimum for centroid + spread, need >= 3 for std)
GEOMETRY_MIN_CLOUD_POINTS = 3

# Category B: CV folds standard (maximum)
GEOMETRY_CV_FOLDS_MAX = 5
GEOMETRY_CV_FOLDS_MIN = 2

# Category B: KNN heuristic bounds
GEOMETRY_KNN_K_MIN = 3
GEOMETRY_KNN_K_MAX = 15

# Category B: PCA null model cap
GEOMETRY_PCA_NULL_CAP = 50

# Category B: probe validation floor
GEOMETRY_PROBE_VALIDATION_MIN = 0.1
GEOMETRY_PROBE_VALIDATION_MIN_SAMPLES = 2

# Category B: classifier self-configuration bounds (data-adaptive)
CLASSIFIER_HIDDEN_DIM_MIN = 32
CLASSIFIER_HIDDEN_DIM_MAX = 512
CLASSIFIER_BATCH_SIZE_MIN = 8
CLASSIFIER_BATCH_SIZE_MAX = 64
CLASSIFIER_BATCH_DIVISOR = 4
CLASSIFIER_TEST_SIZE_MIN = 0.1
CLASSIFIER_TEST_SIZE_MAX = 0.3
CLASSIFIER_MIN_TEST_SAMPLES = 10
CLASSIFIER_DROPOUT_MIN = 0.05
CLASSIFIER_DROPOUT_MAX = 0.5
CLASSIFIER_LOG_DIVISOR = 10
CLASSIFIER_DEFAULT_LR = 0.001

# --- Steering method scorer constants (structural, not calibratable) ---
SCORER_SIGNAL_Z_NORMALIZER = 10.0
SCORER_NEUTRAL_DEFAULT = 0.5
SCORER_TOP_RECOMMENDED = 3
SCORER_CLOUD_SEP_NORMALIZER = 3.0
