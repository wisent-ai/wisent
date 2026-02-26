"""Constants that cannot be experimentally optimized."""
from wisent.core.grp_05.sub_constants.cannot_be_optimized.sub_fixed import *  # noqa: F401,F403

VIZ_FIGURE_HEIGHT_PX = 800
VIZ_LEGEND_Y_TOP = 0.99
# --- Visualization histogram bins ---
VIZ_HISTOGRAM_BINS_20 = 20
# --- Visualization scale factors ---
VIZ_MIN_SCALE = 0.1
VIZ_SCALE_PADDING = 0.8
# --- Database cursor ---
DB_CURSOR_ITERSIZE = 500
# --- Generate responses min load ---
MIN_LOAD_LIMIT_QUESTIONS = 20
# --- QA parser lookahead ---
QA_PARSER_LOOKAHEAD_LINES = 10
# --- Nonsense generator ranges ---
NONSENSE_CHAR_LEN_MIN = 20
NONSENSE_CHAR_LEN_MAX = 50
NONSENSE_WORD_FILTER_MAX_LEN = 15
NONSENSE_REPETITION_MIN = 10
NONSENSE_REPETITION_MAX = 30
NONSENSE_WORD_SALAD_MIN = 3
NONSENSE_WORD_SALAD_MAX = 10
NONSENSE_MIXED_COMPONENTS_MIN = 2
NONSENSE_MIXED_COMPONENTS_MAX = 4
NONSENSE_MIXED_CHAR_LEN_MIN = 5
NONSENSE_MIXED_CHAR_LEN_MAX = 15
NONSENSE_MIXED_REPS_MIN = 3
NONSENSE_MIXED_REPS_MAX = 6
NONSENSE_MIXED_WORDS_MIN = 3
NONSENSE_MIXED_WORDS_MAX = 6
# --- ChartQA perturbation ---
CHARTQA_PCT_DELTAS = (-30, -20, -10, 10, 20, 30)
CHARTQA_INT_DELTAS = (-5, -3, -2, -1, 1, 2, 3, 5)
CHARTQA_DECIMAL_DELTAS = (-0.2, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.2)
# --- GSM8K extractor ---
GSM8K_DEFAULT_LIMIT = 500
GSM8K_PERTURBATION_MIN = 1
GSM8K_PERTURBATION_MAX = 10
# --- Distractor generation ---
DISTRACTOR_NEARBY_MIN = 2
DISTRACTOR_NEARBY_MAX = 10
DISTRACTOR_MAX_COUNT = 3
# --- Frames extractor ---
FRAMES_NUMERIC_DELTA_LARGE = 100
FRAMES_NUMERIC_DELTA_MEDIUM = 50
FRAMES_NUMERIC_DELTA_SMALL = 10
FRAMES_NUMERIC_DELTA_TINY = 5
FRAMES_SHORT_ANSWER_THRESHOLD = 100
FRAMES_ALT_ANSWER_LENGTH = 20

# --- Paper-derived defaults (arbitrary choices by paper authors) ---
DPO_DEFAULT_BETA = 0.1

# --- Convention-derived defaults (arbitrary choices) ---
JSON_INDENT = 2
DEFAULT_VIDEO_FPS = 30.0
JAVA_INDENT_SPACES = 4
GEO_DEFAULT_DIRECTION_ANGLE = 90.0

# --- Gemma BOS features ---
GEMMA_2B_BOS_FEATURES_PAPER = (11087, 3220, 11752, 12160, 11498)
GEMMA_2B_BOS_FEATURES_DETECTED = (
    1041, 7507, 11087, 3220, 11767, 11752,
    14669, 6889, 12160, 13700, 2747, 11498,
)
GEMMA_9B_BOS_FEATURES_DETECTED = (
    8032, 11906, 7768, 14845, 14483, 10562,
    8892, 9151, 5721, 15738, 5285, 13895,
)

# --- Visualization bar/direction ---
VIZ_BAR_WIDTH = 0.35
VIZ_DIRECTION_N_PCA = 5
VIZ_DIRECTION_N_RANDOM_SEARCH = 10
VIZ_TRUTHFUL_REGION_THRESHOLD = 0.5

# --- Agent defaults ---
AGENT_TIMEOUT_CHECK_INTERVAL = 10

# --- Weight modification default components ---
WEIGHT_MOD_DEFAULT_COMPONENTS = ("self_attn.o_proj", "mlp.down_proj")

# --- Role play tokens ---
ROLE_PLAY_TOKENS = (
    "I", "Well", "The", "Sure", "Let",
    "That", "It", "This", "My", "To",
)
COMPARISON_MAX_BATCH_SIZE = 8

STEERING_OPTI_BATCH_SIZE = 16

# --- Extraction batch sizes ---
EXTRACTION_RAW_BATCH_SIZE = 5
EXTRACTION_SMALL_BATCH_SIZE = 10

# --- Classifier cache defaults ---
CACHE_MAX_SIZE_GB_DEFAULT = 5.0
CACHE_MAX_AGE_DAYS_DEFAULT = 30.0
CACHE_MEMORY_SIZE_DEFAULT = 10
COMPARISON_TRAINING_BATCH_SIZE = 2

# --- Visualization ---
VIZ_PLOT_DPI = 300

# --- Comparison module logging ---
COMPARISON_LOGGING_STEPS = 10

# --- Progress logging interval ---
PROGRESS_LOG_INTERVAL = 50

# --- Comparison default batch size (sequential) ---
COMPARISON_DEFAULT_BATCH_SIZE = 1

# --- Visualization marker sizes ---
VIZ_MARKER_SIZE = 50
VIZ_MARKER_SIZE_SMALL = 30

PARSER_DEFAULT_LAYER_START = 0
MONITOR_DEFAULT_INTERVAL = 1.0

# --- SAE batch size ---
SAE_BATCH_SIZE_DEFAULT = 64

# --- Histogram bins ---
VIZ_HISTOGRAM_BINS_LARGE = 50

# --- Visualization alpha values ---
VIZ_ALPHA_LIGHT = 0.3
VIZ_ALPHA_MEDIUM = 0.6
VIZ_ALPHA_HIGH = 0.7
VIZ_ALPHA_HALF = 0.5

# --- Visualization linewidth ---
VIZ_LINEWIDTH_THIN = 0.8
VIZ_LINEWIDTH_NORMAL = 2

# --- Visualization marker sizes (extra small) ---
VIZ_MARKER_SIZE_TINY = 10
VIZ_MARKER_SIZE_XS = 20
PARSER_DEFAULT_WORKERS = 4

# --- Nonsense/quality detection thresholds ---
NONSENSE_MAX_WORD_LENGTH = 20

# --- Monitoring defaults ---
MONITOR_DEFAULT_DURATION = 60

# --- Core thresholds ---
DEFAULT_TIMEOUT_DOCKER = 30

# --- Timeout variants ---
DEFAULT_TIMEOUT_SHORT = 10
DEFAULT_TIMEOUT_QUICK = 5
DEFAULT_TIMEOUT_MATH_SCRIPT = 1

# --- Subprocess/bigcode test timeout ---
BIGCODE_TEST_TIMEOUT = 10

# --- Priority scoring ---
PRIORITY_HIGH = 3
PRIORITY_MEDIUM = 2
PRIORITY_LOW = 1

# --- Visualization gallery ---
VIZ_GALLERY_N_PER_TYPE = 2

# --- Benchmark loading time ---
BENCHMARK_LOADING_TIME_DEFAULT = 60.0

# --- Recursion/retry defaults ---
MAX_RECURSION_DEPTH = 3
DEFAULT_API_RETRIES = 3

# --- t-SNE ---
TSNE_PERPLEXITY_MAX = 30

# --- Display truncation ---
DISPLAY_TRUNCATION_SHORT = 50
DISPLAY_TRUNCATION_MEDIUM = 200
DISPLAY_TRUNCATION_LONG = 300

# --- Training logging frequency ---
TRAINING_LOG_FREQUENCY = 10

# --- Gradient accumulation ---
GRADIENT_ACCUMULATION_STEPS_DEFAULT = 1

# --- Max benchmarks (single match) ---
MAX_BENCHMARKS_SINGLE = 1

# --- JSON serialization limit ---
JSON_ARRAY_LIMIT = 100

# --- Visualization figure sizes ---
VIZ_FIGSIZE_SUMMARY = (18, 18)
VIZ_FIGSIZE_DETAIL = (12, 10)
VIZ_GRID_SUMMARY_ROWS = 3
VIZ_GRID_SUMMARY_COLS = 3

# --- Display truncation (description) ---
DISPLAY_TRUNCATION_DESCRIPTION = 25

# --- Architecture detection ---
ARCHITECTURE_MODULE_LIMIT = 50

# --- Progress reporting interval ---
PROGRESS_REPORT_INTERVAL = 10

# --- Retry/cleaning ---
REFUSAL_CLEANER_MAX_RETRIES = 2
DEFAULT_RETRY_DELAY = 1.0

# --- Validation ---
MIN_CHOICES_VALIDATION = 2

# --- Test harness ---
TEST_MAX_COMBO_SIZE = 3
PROGRESS_CALLBACK_THRESHOLD = 100

# --- Concept/noise analysis ---
CONCEPT_NAMING_N_SAMPLES = 5

# --- Polymath evaluation ---
POLYMATH_DEFAULT_TOTAL = 125

# --- Process pool ---
MAX_TASKS_PER_PROC = 1024

# --- Codeforces dataset defaults ---
CODEFORCES_DEFAULT_TIME_LIMIT = 1.0
CODEFORCES_DEFAULT_MEMORY_LIMIT = 256.0

# --- Mercury runtime sentinel ---
MERCURY_RUNTIME_SENTINEL = 999
