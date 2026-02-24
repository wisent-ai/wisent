"""
Overflow named constants for the wisent package (part 11).
Imported by _constants_part10.py via star-import so all constants remain
accessible from wisent.core.constants.
"""
from wisent.core.grp_05.sub_constants._constants_part12 import *  # noqa: F401,F403

# --- Classification threshold range ---
CLASSIFICATION_THRESHOLD_RANGE = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# --- Weight optimization search space (string-encoded min,max ranges) ---
WEIGHT_OPT_STRENGTH_RANGE = "0.3,2.0"
WEIGHT_OPT_MAX_WEIGHT_RANGE = "0.5,3.0"
WEIGHT_OPT_MIN_WEIGHT_RANGE = "0.0,0.5"
WEIGHT_OPT_POSITION_RANGE = "0.3,0.7"

# --- Steering evaluation scales (string-encoded) ---
STEERING_EVAL_SCALES_DEFAULT = "0.0,0.5,1.0,1.5,2.0"

# --- Timeout variants ---
DEFAULT_TIMEOUT_SHORT = 10
DEFAULT_TIMEOUT_QUICK = 5
DEFAULT_TIMEOUT_SUBPROCESS = 60
DEFAULT_TIMEOUT_MATH_SCRIPT = 1

# --- Subprocess/bigcode test timeout ---
BIGCODE_TEST_TIMEOUT = 10

# --- Priority scoring ---
PRIORITY_HIGH = 3
PRIORITY_MEDIUM = 2
PRIORITY_LOW = 1

# --- Test/runner limits ---
TEST_DEFAULT_LIMIT = 10
QWEN3_4B_DEFAULT_LAYER = 17

# --- SAE analysis ---
SAE_HIDDEN_DIM_MULTIPLIER = 4
SAE_TOP_K_ANALYSIS = 20

# --- Abliteration search defaults ---
ABLITERATION_NUM_PAIRS = 300
ABLITERATION_DEFAULT_POSITION = 8.0
ABLITERATION_DEFAULT_DISTANCE = 6.0
ABLITERATION_BINARY_SEARCH_LOW = 0.5
ABLITERATION_BINARY_SEARCH_HIGH = 3.0
ABLITERATION_BINARY_SEARCH_ITERS = 5

# --- Dimensionality reduction ---
N_COMPONENTS_2D = 2

# --- Concept experiment ---
N_PAIRS_PER_CONCEPT_DEFAULT = 100

# --- Visualization gallery ---
VIZ_GALLERY_N_PER_TYPE = 2

# --- Threshold analysis ---
THRESHOLD_HIDDEN_DIM_LARGE = 4096

# --- Benchmark loading time ---
BENCHMARK_LOADING_TIME_DEFAULT = 60.0

# --- Synthetic generation defaults ---
SYNTHETIC_GENERATION_MAX_TOKENS = 256
SYNTHETIC_GENERATION_TEMPERATURE = 1.0
SYNTHETIC_GENERATION_TOP_P = 1.0

# --- Steering weight scheduling ---
STEERING_WEIGHT_INITIAL_TOKENS = 10
STEERING_WEIGHT_DECAY_RATE = 0.1
STEERING_WEIGHT_GAUSSIAN_CENTER = 0.5
STEERING_WEIGHT_GAUSSIAN_WIDTH = 0.2

# --- Concept detection ---
CONCEPT_DETECTION_DEFAULT_N = 2

# --- Extractor defaults ---
EXTRACTOR_DEFAULT_LIMIT = 500
EXTRACTOR_MIN_TEXT_LENGTH = 500

# --- Recursion/retry defaults ---
MAX_RECURSION_DEPTH = 3
DEFAULT_API_RETRIES = 3

# --- Fast diversity ---
FAST_DIVERSITY_SEED = 13

# --- Nonsense generator ---
NONSENSE_DEFAULT_NUM_PAIRS = 10

# --- Classifier recency ---
CLASSIFIER_RECENCY_DAYS = 30

# --- Quality evaluation bounds ---
QUALITY_EVAL_LAYER_MIN = 8
QUALITY_EVAL_LAYER_MAX = 20
QUALITY_EVAL_SAMPLES_MIN = 10
QUALITY_EVAL_SAMPLES_MAX = 50

# --- Zwiad analysis limits ---
ZWIAD_ANALYSIS_LIMIT = 500
ZWIAD_ANALYSIS_LIMIT_SMALL = 200

# --- Token estimation ---
TOKENS_PER_PAIR_ESTIMATE = 150
TOKENS_BASE_OFFSET = 500
TOKEN_ESTIMATE_MIN = 2048
TOKEN_ESTIMATE_MAX = 8192

# --- SimHash dedup thresholds ---
SIMHASH_THRESHOLD_AGGRESSIVE = 10
SIMHASH_THRESHOLD_CONSERVATIVE = 3

# --- Subprocess timeout (long) ---
SUBPROCESS_TIMEOUT_LONG = 600

# --- Docker sandbox limits ---
DOCKER_SANDBOX_TIME_LIMIT = 10
DOCKER_SANDBOX_CPU_LIMIT = 5
DOCKER_SANDBOX_MEM_LIMIT_MB = 512

# --- t-SNE ---
TSNE_PERPLEXITY_MAX = 30

# --- Dedup item threshold ---
DEDUP_ITEM_THRESHOLD = 1000

# --- Display truncation ---
DISPLAY_TRUNCATION_SHORT = 50
DISPLAY_TRUNCATION_MEDIUM = 200
DISPLAY_TRUNCATION_LONG = 300

# --- Training logging frequency ---
TRAINING_LOG_FREQUENCY = 10

# --- Gradient accumulation ---
GRADIENT_ACCUMULATION_STEPS_DEFAULT = 1

# --- Geometry structure thresholds ---
GEOMETRY_THRESHOLD_DEFAULT = 0.5
GEOMETRY_THRESHOLD_CLUSTER = 0.6
GEOMETRY_THRESHOLD_SPARSE = 0.7
GEOMETRY_THRESHOLD_MANIFOLD = 0.3

# --- Benchmark test limits ---
BENCHMARK_TEST_DATA_LIMIT = 300
BENCHMARK_SPLIT_LIMIT = 15

# --- Null distribution ---
NULL_DISTRIBUTION_SAMPLES = 100

# --- Max benchmarks (single match) ---
MAX_BENCHMARKS_SINGLE = 1

# --- Eval harness num_samples ---
EVAL_HARNESS_NUM_SAMPLES = 100
EVAL_HARNESS_NUM_SAMPLES_SMALL = 5

# --- Top-k test matching ---
TEST_BENCHMARK_TOP_K = 3

# --- Prompt/response min length ---
PAIR_MIN_TEXT_LENGTH = 10

# --- Gap threshold candidates ---
GAP_THRESHOLD_CANDIDATES = [0.05, 0.10, 0.15, 0.20, 0.25]

# --- JSON serialization limit ---
JSON_ARRAY_LIMIT = 100

# --- Steering alpha range ---
STEERING_ALPHA_MIN = -3.0
STEERING_ALPHA_MAX = 3.0

# --- Model architecture defaults ---
DEFAULT_NUM_ATTENTION_HEADS = 32
DEFAULT_MODEL_LAYER_COUNT = 32

# --- Database ---
DEFAULT_DB_PORT = 5432

# --- Video ---
DEFAULT_VIDEO_FPS = 30.0

# --- Personalization ---
DIFFERENCE_SCORE_NEUTRAL = 50.0

# --- Visualization figure sizes ---
VIZ_FIGSIZE_SUMMARY = (18, 18)
VIZ_FIGSIZE_DETAIL = (12, 10)
VIZ_GRID_SUMMARY_ROWS = 3
VIZ_GRID_SUMMARY_COLS = 3

# --- Diagnostics report ---
DIAGNOSTICS_LINE_WIDTH = 60

# --- Optimization time (long) ---
MAX_OPTIMIZATION_TIME_MINUTES = 60.0

# --- Subspace validation sample thresholds ---
SUBSPACE_SAMPLE_CONSERVATIVE = 20
SUBSPACE_SAMPLE_RELAXED = 100

# --- PCA ---
MAX_PCA_COMPONENTS_ANALYSIS = 20
PCA_TOP_N_COMPONENTS = 10

# --- Policy layers ---
POLICY_LAYER_COUNT = 5

# --- Display truncation (description) ---
DISPLAY_TRUNCATION_DESCRIPTION = 25

# --- Architecture detection ---
ARCHITECTURE_MODULE_LIMIT = 50

# --- Progress reporting interval ---
PROGRESS_REPORT_INTERVAL = 10

# --- Signal detection thresholds ---
SIGNAL_EXISTENCE_THRESHOLD = 0.6
LINEAR_GAP_THRESHOLD = 0.15

# --- Null distribution per-class samples ---
NULL_DISTRIBUTION_SAMPLES_PER_CLASS = 50

# --- Threshold analysis existence grid ---
EXISTENCE_THRESHOLD_GRID = [0.5, 0.55, 0.6, 0.65, 0.7]

# --- Separator line width ---
SEPARATOR_LINE_WIDTH = 70

# --- Quality threshold (agent task selector) ---
TASK_SELECTOR_QUALITY_THRESHOLD = 1.5

# --- Threshold analysis default hidden dim ---
THRESHOLD_HIDDEN_DIM_DEFAULT = 100

# --- PaCMAP alternative implementation ---
PACMAP_ALT_NUM_ITERS = 100
PACMAP_LEARNING_RATE_DEFAULT = 1.0
PACMAP_N_MID_PAIRS = 5
PACMAP_N_FAR_PAIRS = 2

# --- Model generation defaults ---
DEFAULT_NO_REPEAT_NGRAM_SIZE = 4
MODEL_COLLECT_TOPK = 5
CODE_GENERATION_MAX_TOKENS = 512

# --- Retry/cleaning ---
REFUSAL_CLEANER_MAX_RETRIES = 2
DEFAULT_RETRY_DELAY = 1.0

# --- Classifier selection ---
DEFAULT_MAX_CLASSIFIERS_SELECT = 5
CLASSIFIER_CACHE_TOP_K = 5

# --- Validation ---
MIN_CHOICES_VALIDATION = 2

# --- Code formatting ---
JAVA_INDENT_SPACES = 4

# --- Test harness ---
TEST_MAX_COMBO_SIZE = 3
PROGRESS_CALLBACK_THRESHOLD = 100

# --- Concept/noise analysis ---
CONCEPT_NAMING_N_SAMPLES = 5
NOISE_BASELINE_N_SAMPLES = 5

# --- Marketplace ---
MARKETPLACE_MIN_QUALITY_DEFAULT = 0.0

# --- Steering optimization ---
STEERING_OPT_NUM_EPOCHS = 10

# --- Synthetic evaluation ---
SYNTHETIC_EVAL_NUM_PROMPTS = 10

# --- Optuna classifier ---
OPTUNA_CLASSIFIER_CV_FOLDS = 3

# --- Signal baseline ---
SIGNAL_BASELINE_RATIO_DEFAULT = 1.0

# --- Score range ---
SCORE_RANGE_MIN = 0.0
SCORE_RANGE_MAX = 1.0

# --- KNN default k ---
KNN_DEFAULT_K = 10
