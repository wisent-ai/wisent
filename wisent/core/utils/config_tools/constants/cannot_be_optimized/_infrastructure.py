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
# --- Docker / sandbox ---
FEEDBACK_MAX_CHARS = 2000
SAFE_DOCKER_FSIZE_MB = 10
SAFE_DOCKER_NOFILE = 128
DOCKER_CPU_QUOTA_50PCT_US = 50000
# --- Combinatorics ---
COMBO_BASE = 2
COMBO_OFFSET = 1
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
ARCHITECTURE_MODULE_LIMIT = 50
MIN_CHOICES_VALIDATION = 2
DIAGNOSTICS_TOTAL_CHECKS = 4
# --- Token limits ---
MAX_NEW_TOKENS_VERIFY_SINGLE = 1
AUDIO_WHISPER_MAX_TOKENS = 448
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
