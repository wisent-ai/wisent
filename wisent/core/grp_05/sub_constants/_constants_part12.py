"""
Overflow named constants for the wisent package (part 12).
Imported by _constants_part11.py via star-import so all constants remain
accessible from wisent.core.constants.
"""

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
