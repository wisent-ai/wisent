"""Constants that cannot be experimentally optimized."""
from wisent.core.grp_05.sub_constants.cannot_be_optimized.sub_fixed._fixed_04 import *  # noqa: F401,F403

MERCURY_RUNTIME_SENTINEL_STR = "999ms"

# --- Bias scale ---
BIAS_NEUTRAL_MIDPOINT = 2

# --- PCA dimension reduction threshold ---
PACMAP_PCA_DIM_THRESHOLD = 50

# --- Score midpoint (0-100 scale: majority, neutral, fair) ---
SCORE_MIDPOINT_PCT = 50

# --- Geometry interpretation thresholds (percent) ---
GEOMETRY_MINORITY_PCT = 30

# --- Personalization score thresholds ---
PERSONALIZATION_GOOD_THRESHOLD = 70

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

# --- Clustering defaults ---
MIN_CLUSTERS = 2

# --- Layer stride default ---
LAYER_STRIDE_DEFAULT = 2

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

# --- Batch size cap ---
BATCH_SIZE_CAP = 32

# --- CLI banner width ---
BANNER_WIDTH = 64

# --- Agent retry attempts (autonomous) ---
AGENT_RETRY_ATTEMPTS = 2

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


# Single-token verification — literally needs exactly 1 token
MAX_NEW_TOKENS_VERIFY_SINGLE = 1

# Judge outputs a short numeric score
STEERING_OPTI_JUDGE_MAX_TOKENS = 8

# Nonsense baseline must be short by definition
NONSENSE_MAX_TOKENS = 25

# Whisper model architecture constraint
AUDIO_WHISPER_MAX_TOKENS = 448
