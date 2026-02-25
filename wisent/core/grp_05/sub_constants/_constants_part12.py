"""
Overflow named constants for the wisent package (part 12).
Imported by _constants_part11.py via star-import so all constants remain
accessible from wisent.core.constants.
"""
from wisent.core.grp_05.sub_constants.sub_01._constants_part13 import *  # noqa: F401,F403

# --- Polymath evaluation ---
POLYMATH_DEFAULT_TOTAL = 125
POLYMATH_DEFAULT_K = 16

# --- Agent classifier pair gen ---
AGENT_CLASSIFIER_NUM_PAIRS = 15

# --- Process pool ---
MAX_TASKS_PER_PROC = 1024

# --- Codeforces dataset defaults ---
CODEFORCES_DEFAULT_TIME_LIMIT = 1.0
CODEFORCES_DEFAULT_MEMORY_LIMIT = 256.0

# --- Mercury runtime sentinel ---
MERCURY_RUNTIME_SENTINEL = 999
MERCURY_RUNTIME_SENTINEL_STR = "999ms"

# --- NLP evaluator embedding confidence ---
NLP_EMB_CONFIDENCE_CAP = 0.8
NLP_EMB_CONFIDENCE_BASE = 0.5

# --- Task selector quality score ---
TASK_QUALITY_SCORE_MAX = 5.0

# --- Refusal family default weight ---
REFUSAL_FAMILY_DEFAULT_WEIGHT = 0.5

# --- Bias scale ---
BIAS_NEUTRAL_MIDPOINT = 2

# --- PCA dimension reduction threshold ---
PACMAP_PCA_DIM_THRESHOLD = 50

# --- Geometry interpretation thresholds (percent) ---
GEOMETRY_MAJORITY_PCT = 50
GEOMETRY_MINORITY_PCT = 30

# --- Personalization score thresholds ---
PERSONALIZATION_GOOD_THRESHOLD = 70
PERSONALIZATION_FAIR_THRESHOLD = 50

# --- Benchmark loading time threshold ---
BENCHMARK_FAST_LOAD_THRESHOLD = 50

# --- Minimum text lengths ---
MIN_PAGE_TEXT_LENGTH = 50
MIN_RESPONSE_TEXT_LENGTH = 50
MIN_SENTENCE_LENGTH = 20

# --- Angle deviation ---
ANGLE_DEVIATION_DEGREES_THRESHOLD = 60

# --- Display truncation (model/task names) ---
DISPLAY_TRUNCATION_MODEL = 40
DISPLAY_TRUNCATION_TASK = 20
DISPLAY_TRUNCATION_XLARGE = 400

# --- Answer length ---
ANSWER_MAX_DISPLAY_LENGTH = 27

# --- Benchmark display limit ---
BENCHMARK_DISPLAY_LIMIT = 20

# --- Display truncation (compact, for log/debug previews) ---
DISPLAY_TRUNCATION_COMPACT = 100

# --- Display truncation (large, for error/output previews) ---
DISPLAY_TRUNCATION_LARGE = 500

# --- Display truncation (response preview in viz) ---
DISPLAY_TRUNCATION_RESPONSE = 150

# --- Context preview max length ---
CONTEXT_MAX_PREVIEW = 1000

# --- Classifier cache hash sample size ---
HASH_SAMPLE_SIZE = 100

# --- Task search limit ---
TASK_SEARCH_LIMIT = 100

# --- Display truncation (error messages) ---
DISPLAY_TRUNCATION_ERROR = 80

# --- Display truncation (eval preview) ---
DISPLAY_TRUNCATION_EVAL = 60

# --- Context max length (SQuAD, HaluLens, tag generation) ---
CONTEXT_MAX_LENGTH = 1500

# --- Detector max text length ---
DETECTOR_MAX_TEXT_LENGTH = 2000

# --- Hash digest prefix length ---
HASH_DIGEST_PREFIX = 16

# --- Device hash prefix length ---
DEVICE_HASH_PREFIX = 12

# --- Display top-N items (list slicing) ---
DISPLAY_TOP_N_SMALL = 10
DISPLAY_TOP_N_MEDIUM = 20

# --- SAE top features display ---
SAE_TOP_FEATURES_DISPLAY = 15
SAE_TOP_FEATURES_MAX = 40

# --- Enrichment max pairs ---
ENRICHMENT_MAX_PAIRS = 50

# --- Task selector search limit ---
TASK_SELECTOR_LIMIT = 50

# --- Trait label max length ---
TRAIT_LABEL_MAX_LENGTH = 20

# --- Trait name max length ---
TRAIT_NAME_MAX_LENGTH = 30

# --- Database text field max length ---
DB_TEXT_FIELD_MAX_LENGTH = 65000

# --- Z-score significance threshold ---
Z_SCORE_SIGNIFICANCE = 2.0

# --- Report formatting ---
REPORT_LINE_WIDTH = 80

# --- Clustering defaults ---
MIN_CLUSTERS = 2

# --- Layer stride default ---
LAYER_STRIDE_DEFAULT = 2

# --- Baseline accuracy (chance level) ---
CHANCE_LEVEL_ACCURACY = 0.5

# --- Math evaluator checks ---
MATH_EVAL_N_CHECKS = 5

# --- Subprocess timeout (short, seconds) ---
SUBPROCESS_TIMEOUT_SHORT = 5

# --- Agent max tasks to process ---
MAX_TASKS_TO_PROCESS = 5

# --- Diagnostics total checks ---
DIAGNOSTICS_TOTAL_CHECKS = 4

# --- Default max retries ---
DEFAULT_MAX_RETRIES = 3

# --- Database connectivity ---
DB_CONNECT_WAIT_S = 30

# --- Synthetic pair token estimation ---
TOKENS_PER_PAIR_SYNTHETIC = 150
TOKENS_BASE_SYNTHETIC = 500
MIN_SYNTHETIC_PAIRS = 5
SYNTHETIC_PAIRS_TIME_MULTIPLIER = 3

# --- DIP test simulations ---
DIP_TEST_N_SIMULATIONS = 1000

# --- Confidence vote thresholds ---
CONFIDENCE_HIGH_VOTES = 4
CONFIDENCE_MEDIUM_VOTES = 3

# --- Batch size cap ---
BATCH_SIZE_CAP = 32

# --- Evaluation limits ---
EVAL_PAIR_LIMIT = 1500

# --- Minimum quality score ---
MIN_QUALITY_SCORE_DEFAULT = 2

# --- Robustness thresholds ---
ROBUSTNESS_THRESHOLD_STRICT = 0.1
ROBUSTNESS_THRESHOLD_MODERATE = 0.2
ROBUSTNESS_THRESHOLD_LOOSE = 0.3

# --- Code evaluation memory limit ---
CODE_EVAL_MEM_LIMIT_MB = 768

# --- CLI banner width ---
BANNER_WIDTH = 64

# --- Agent retry attempts (autonomous) ---
AGENT_RETRY_ATTEMPTS = 2

# --- Benchmark test sample sizes ---
BENCH_TEST_SAMPLE_SIZE = 3
BENCH_TEST_SAMPLE_MIN = 2

# --- TETNO steering base strength ---
STEERING_BASE_STRENGTH_DEFAULT = 0.5

# --- Visualization marker sizes (large/xlarge) ---
VIZ_MARKER_SIZE_LARGE = 60
VIZ_MARKER_SIZE_XLARGE = 80
VIZ_MARKER_SIZE_MOVEMENT = 40

# --- Visualization arrow properties ---
VIZ_ARROW_LINEWIDTH = 1.5
VIZ_BBOX_ALPHA = 0.8
VIZ_ARROW_SIZE_DEFAULT = 1

# --- Visualization font sizes ---
VIZ_FONTSIZE_SMALL = 8
VIZ_FONTSIZE_ANNOTATION = 9
VIZ_FONTSIZE_TITLE = 11
VIZ_FONTSIZE_SUPTITLE = 14

# --- Scoring weights ---
STEERING_ICD_WEIGHT = 2.0
STRENGTH_ACCURACY_WEIGHT = 0.5
STEERING_EFFECT_CAP = 100.0
ADAPTIVE_STRENGTH_MULTIPLIER = 2
BINARY_CLASSIFICATION_THRESHOLD = 0.5
CONTRASTIVE_SCORE_OFFSET = 2
CONTRASTIVE_SCORE_RANGE = 4

# --- Marketplace sample estimation ---
MARKETPLACE_MAX_SAMPLES_NEEDED = 500
MARKETPLACE_BASE_SAMPLES_NEEDED = 100
MARKETPLACE_PER_BENCHMARK_SAMPLES = 30

# --- Minimum sample thresholds ---
MIN_SAMPLES_PCA = 3
MIN_CLOUD_POINTS = 3
MIN_CLOUD_CLUSTER_POINTS = 6

# --- HDBSCAN clustering ---
HDBSCAN_MIN_CLUSTER_FLOOR = 5
HDBSCAN_ADAPTIVE_DIVISOR = 20
HDBSCAN_MIN_SAMPLES_DEFAULT = 3
ELBOW_CONSECUTIVE_DECREASES = 3
MAX_SILHOUETTE_CLUSTERS = 6

# --- PCA/subspace analysis ---
PCA_QUALITY_COMPONENTS = 5
DISCOVERY_MAX_PCA_COMPONENTS = 20
SUBSPACE_DIRECTION_PADDING = 5
CUMULATIVE_VARIANCE_TOP_N = 3

# --- Retry/buffer multipliers ---
PAIR_GEN_RETRY_MULTIPLIER = 50
DATA_OVERSAMPLE_MULTIPLIER = 3
SAMPLE_LOADING_BUFFER = 50

# --- Coherence/text quality thresholds ---
MIN_RESPONSE_TOKENS = 4
MIN_TOKENS_TRIGRAM = 6
MIN_CONTENT_WORD_LENGTH = 4
MIN_TOKENS_NONSENSE = 5
MIN_TOKEN_LENGTH_NONSENSE = 4

# --- Display/formatting ---
EIGENVALUE_DISPLAY_LIMIT = 20
HASH_DISPLAY_LENGTH = 8
MAX_TAGS_PER_BENCHMARK = 3
BAR_CHART_SCALE = 20
ROUNDING_PRECISION = 3
ROUNDING_PRECISION_FINE = 4

# --- Layer range offsets ---
LAYER_RANGE_SMALL_OFFSET = 2
LAYER_RANGE_LARGE_OFFSET = 3

# --- Display top-N (tiny/mini/brief) ---
DISPLAY_TOP_N_TINY = 3
DISPLAY_TOP_N_MINI = 5
DISPLAY_TOP_N_BRIEF = 8

# --- Visualization figure sizes ---
VIZ_FIGSIZE_STANDARD = (10, 8)
VIZ_FIGSIZE_WIDE = (10, 6)
VIZ_FIGSIZE_COMPACT = (8, 6)
VIZ_FIGSIZE_MIN_WIDTH = 8
VIZ_FIGSIZE_MIN_HEIGHT = 4
VIZ_FIGSIZE_HEATMAP_SCALE_W = 0.5
VIZ_FIGSIZE_HEATMAP_SCALE_H = 0.8
VIZ_FIGSIZE_CONCEPT_MIN_W = 6
VIZ_FIGSIZE_CONCEPT_SCALE = 1.2
