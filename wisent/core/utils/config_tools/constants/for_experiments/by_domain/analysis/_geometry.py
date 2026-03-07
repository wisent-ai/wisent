"""Geometry, linearity, stability, and subspace constants."""
from wisent.core.utils.config_tools.constants.for_experiments.by_domain.analysis._analysis import *  # noqa: F401,F403

# --- Geometry metric defaults (Category C: temporary, needs empirical validation) ---
# Category A constants (fixed by definition) reuse validated constants:
#   blend_default -> CHANCE_LEVEL_ACCURACY (0.5)
#   default_score -> SCORE_RANGE_MIN (0.0)
#   variance_explained_90pct -> GEOMETRY_VARIANCE_EXPLAINED_90PCT (0.9)
#   feature_dim_index -> COMBO_OFFSET (1)

GEOMETRY_VARIANCE_EXPLAINED_90PCT = 0.9  # "90%" in "dims for 90% variance"

# Category B: min_cloud_points (minimum for centroid + spread, need >= 3 for std)
GEOMETRY_MIN_CLOUD_POINTS = 3

# Category C: probe hidden layer sizes
GEOMETRY_PROBE_SMALL_HIDDEN_CAP = 64
GEOMETRY_PROBE_MLP_HIDDEN_CAP = 128

# Category C: spectral/manifold neighborhood cap
GEOMETRY_SPECTRAL_NEIGHBORS_CAP = 15

# Category C: bootstrap for direction stability
GEOMETRY_DIRECTION_N_BOOTSTRAP = 100
GEOMETRY_DIRECTION_SUBSET_FRACTION = 0.632  # 1 - 1/e, bootstrap theory
GEOMETRY_DIRECTION_STD_PENALTY = 0.5

# Category C: consistency weights (equal, no validation data)
GEOMETRY_CONSISTENCY_W_EQUAL = 1 / 3

# Category C: sparsity and detection thresholds
GEOMETRY_SPARSITY_THRESHOLD_FRACTION = 0.01

# Category C: direction moderate similarity threshold
GEOMETRY_DIRECTION_MODERATE_SIMILARITY = 0.5

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
