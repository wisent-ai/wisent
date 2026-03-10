"""Infrastructure constants: timeouts, retries, batch sizes, caching, logging."""

# --- Core defaults ---
DEFAULT_RANDOM_SEED = 42
SPLIT_RATIO_FULL = 1.0
N_JOBS_SINGLE = 1
PERCENT_MULTIPLIER = 100
BINARY_CLASS_NEGATIVE = 0
BINARY_CLASS_POSITIVE = 1
# --- Batch sizes ---
COMPARISON_MAX_BATCH_SIZE = 8
COMPARISON_DEFAULT_BATCH_SIZE = 1
GENERATION_BATCH_SIZE = 8
# --- Docker / sandbox ---
FEEDBACK_MAX_CHARS = 2000
SAFE_DOCKER_FSIZE_MB = 10
SAFE_DOCKER_NOFILE = 128
DOCKER_CPU_QUOTA_50PCT_US = 50000
# --- Combinatorics ---
COMBO_BASE = 2
COMBO_OFFSET = 1
# --- Exit codes ---
EXIT_CODE_ERROR = 1
# --- Data loading ---
QA_PARSER_LOOKAHEAD_LINES = 10
# --- Timing ---
DEFAULT_PROMPT_LENGTH = 0
# --- Geometry search ---
RECURSION_INITIAL_DEPTH = 0
# --- Parser defaults ---
PARSER_DEFAULT_LAYER_START = 0
# --- Priority scoring ---
PRIORITY_HIGH = 3
PRIORITY_MEDIUM = 2
PRIORITY_LOW = 1
# --- Convention defaults ---
JSON_INDENT = 2
JAVA_INDENT_SPACES = 4
# --- Gradient accumulation ---
GRADIENT_ACCUMULATION_STEPS_DEFAULT = 1
# --- Limits ---
MAX_BENCHMARKS_SINGLE = 1
MIN_CHOICES_VALIDATION = 2
DIAGNOSTICS_TOTAL_CHECKS = 4
# --- Token limits ---
CHARS_PER_TOKEN_ESTIMATE = 3.5
# --- Session transcript report formatting ---
REPORT_COL_TYPE_WIDTH = 20
REPORT_COL_NUMERIC_WIDTH = 12
REPORT_COL_COUNT_WIDTH = 8
MAX_NEW_TOKENS_VERIFY_SINGLE = 1
AUDIO_WHISPER_MAX_TOKENS = 448
# --- Numpy axis and index constants ---
SEARCH_INIT_NEGATIVE = -1.0
AXIS_ROWS = 0
AXIS_COLS = 1
INDEX_FIRST = 0
INDEX_SECOND = 1
INDEX_THIRD = 2
INDEX_LAST = -1
# --- Tensor dimensionality constants ---
NDIM_VECTOR = 1
NDIM_MATRIX = 2
NDIM_BATCH_SEQ = 3
# --- Split ratios ---
SPLIT_RATIO_HALF = 0.5
SPLIT_RATIO_TRAIN_DEFAULT = 0.8
# --- Sensor layer computation ---
SENSOR_STRIDE_MIN = 1
SENSOR_RANGE_START = 0
SENSOR_LAST_OFFSET = 1
# --- Temporal ramp ---
TEMPORAL_RAMP_MAX = 1.0
# --- Thresholds ---
SCORE_MIDPOINT_PCT = 50
# --- String identity constants ---
# Evaluator names
EVALUATOR_NAME_TRUTHFULQA_GEN = "truthfulqa_gen"
EVALUATOR_NAME_LOG_LIKELIHOODS = "log_likelihoods"
# Extractor manifest base imports
HF_EXTRACTOR_BASE_IMPORT = "wisent.extractors.hf.hf_task_extractors."
LM_EVAL_EXTRACTOR_BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."
# Base class / registry identity names
BASE_CLASS_NAME = "base"
CLEAN_STEP_DEFAULT_NAME = "step"
BASE_OPTIMIZER_NAME = "base-optimizer"
# LiveMathBench defaults
LIVEMATHBENCH_DEFAULT_DATASET_CONFIG = "v202412_CNMO_en"
LIVEMATHBENCH_DEFAULT_CONFIG_LABEL = "cnmo_en"
# Config field names
NESTED_CONFIG_NAME_FIELD = "name"
TASK_CONFIG_NAME_FIELD = "task_name"
TRAIT_CONFIG_NAME_FIELD = "trait_name"
# --- Unified optimizer defaults ---
QUANTIZATION_STEP_DEFAULT = 1
OPTUNA_SIGMA_SPREAD_FACTOR = 4
UNINFORMATIVE_MU = 0.0
UNINFORMATIVE_SIGMA = 3.0
HYPEROPT_BACKEND_NAME = "hyperopt"
OPTUNA_BACKEND_NAME = "optuna"
TRIALS_PER_DIMENSION_MULTIPLIER = 50
# --- Steering defaults ---
STEERING_SCALE_IDENTITY = 1.0
STEERING_DEFAULT_INTENSITY = 1.0
# --- Steering strategy config defaults ---
STEERING_STRATEGY_DEFAULT_RATE = 0.02
STEERING_STRATEGY_DEFAULT_INITIAL_TOKENS = 10
STEERING_STRATEGY_DEFAULT_GAUSSIAN_CENTER = 0.5
STEERING_STRATEGY_DEFAULT_GAUSSIAN_WIDTH = 0.167
# --- Generation defaults (matches InferenceConfig) ---
GENERATION_DEFAULT_MAX_NEW_TOKENS = 32768
GENERATION_DEFAULT_TEMPERATURE = 0.7
GENERATION_DEFAULT_TOP_P = 0.9
# --- Evaluation defaults (balanced equal-weight scoring) ---
EVAL_F1_THRESHOLD = 0.5
EVAL_GENERATION_EMBEDDING_WEIGHT = 0.5
EVAL_GENERATION_NLI_WEIGHT = 0.5
# --- Architecture detection ---
ARCHITECTURE_MODULE_LIMIT_DEFAULT = 200
# --- API fetch limits ---
SUPABASE_REST_LAYER_QUERY_LIMIT = 2000
# --- Evaluator similarity defaults ---
EVAL_MIN_SIMILARITY_THRESHOLD_DEFAULT = 0.5
# --- Early rejection ---
EARLY_REJECTION_CV_THRESHOLD_DEFAULT = 0.1
# --- HuggingFace upload retry defaults ---
HF_RETRY_MAX_RETRIES = 5
HF_RETRY_BASE_WAIT = 2
HF_RETRY_BACKOFF_MAX_EXPONENT = 4
HF_RETRY_JITTER_MIN = 0.8
HF_RETRY_JITTER_MAX = 1.2
HF_RETRY_RETRYABLE_PATTERNS = ("429", "412", "timeout", "Timeout", "ConnectionError")
# --- Database statement controls ---
PG_STATEMENT_NO_LIMIT = "0"
# --- Gradio app defaults ---
GRADIO_SERVER_PORT = 7860
GRADIO_SERVER_HOST = "0.0.0.0"
GRADIO_APPEND_LINES = 3
GRADIO_MODEL_PLACEHOLDER = "e.g. meta-llama/Llama-3.2-1B-Instruct"
GRADIO_PIP_SPEC = "gradio>=4.0.0"
GRADIO_GALLERY_COLUMNS = 2
WISENT_COLOR_MINT = "#b0f0c0"
WISENT_COLOR_MINT_LIGHT = "#d0f8dc"
WISENT_COLOR_MINT_DARK = "#7ae094"
WISENT_COLOR_CHARCOAL = "#333333"
WISENT_COLOR_DARK_BG = "#2b2b2b"
WISENT_COLOR_DARK_SURFACE = "#3a3a3a"
WISENT_COLOR_TEXT_LIGHT = "#f0f0f0"
WISENT_COLOR_TEXT_MUTED = "#b0b0b0"
WISENT_COLOR_LIGHT_BG = "#f8faf8"
WISENT_COLOR_LIGHT_SURFACE = "#ffffff"
WISENT_COLOR_LIGHT_TEXT = "#1a1a1a"
WISENT_COLOR_LIGHT_TEXT_MUTED = "#555555"
WISENT_COLOR_MINT_ACCENT_DARK = "#2d8a4e"
WISENT_LOGO_FILENAME = "wisent_logo.png"
WISENT_LOGO_DISPLAY_WIDTH = 80
GRADIO_MODEL_EXAMPLES = (
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3.5-0.8B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3.5-2B",
)
# --- Degeneration detection (StoppingCriteria) ---
DEGEN_WARMUP_TOKENS = 100
DEGEN_NGRAM_SIZE = 4
DEGEN_MAX_REPEATS = 3
DEGEN_WINDOW_SIZE = 50
DEGEN_DIVERSITY_THRESHOLD = 0.4
