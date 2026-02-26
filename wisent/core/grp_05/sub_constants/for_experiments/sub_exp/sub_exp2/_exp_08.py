"""Constants requiring experimental optimization."""
from wisent.core.grp_05.sub_constants.for_experiments.by_method import *  # noqa: F401,F403


# --- Benchmark test limits ---
BENCHMARK_TEST_DATA_LIMIT = 300
BENCHMARK_SPLIT_LIMIT = 15

# --- Null distribution ---
NULL_DISTRIBUTION_SAMPLES = 100

# --- Eval harness num_samples ---
EVAL_HARNESS_NUM_SAMPLES = 100
EVAL_HARNESS_NUM_SAMPLES_SMALL = 5

# --- Top-k test matching ---
TEST_BENCHMARK_TOP_K = 3

# --- Prompt/response min length ---
PAIR_MIN_TEXT_LENGTH = 10

# --- Gap threshold candidates ---
GAP_THRESHOLD_CANDIDATES = [0.05, 0.10, 0.15, 0.20, 0.25]

# --- Steering alpha range ---
STEERING_ALPHA_MIN = -3.0
STEERING_ALPHA_MAX = 3.0

# --- Subspace validation sample thresholds ---
SUBSPACE_SAMPLE_CONSERVATIVE = 20
SUBSPACE_SAMPLE_RELAXED = 100

# --- PCA ---
PCA_TOP_N_COMPONENTS = 10

# --- Policy layers ---
POLICY_LAYER_COUNT = 5

# --- Null distribution per-class samples ---
NULL_DISTRIBUTION_SAMPLES_PER_CLASS = 50

# --- Threshold analysis existence grid ---
EXISTENCE_THRESHOLD_GRID = [0.5, 0.55, 0.6, 0.65, 0.7]

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
MODEL_COLLECT_TOPK = 5

# --- Classifier selection ---
DEFAULT_MAX_CLASSIFIERS_SELECT = 5
CLASSIFIER_CACHE_TOP_K = 5
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

# --- KNN default k ---
KNN_DEFAULT_K = 10
POLYMATH_DEFAULT_K = 16

# --- Agent classifier pair gen ---
AGENT_CLASSIFIER_NUM_PAIRS = 15

# --- NLP evaluator embedding confidence ---
NLP_EMB_CONFIDENCE_CAP = 0.8
NLP_EMB_CONFIDENCE_BASE = 0.5

# --- Task selector quality score ---
TASK_QUALITY_SCORE_MAX = 5.0

# --- Refusal family default weight ---
REFUSAL_FAMILY_DEFAULT_WEIGHT = 0.5

# --- Classifier cache hash sample size ---
HASH_SAMPLE_SIZE = 100

# --- Hash digest prefix length ---
HASH_DIGEST_PREFIX = 16

# --- Device hash prefix length ---
DEVICE_HASH_PREFIX = 12

# --- Confidence vote thresholds ---
CONFIDENCE_HIGH_VOTES = 4
CONFIDENCE_MEDIUM_VOTES = 3

# --- Evaluation limits ---
EVAL_PAIR_LIMIT = 1500

# --- Minimum quality score ---
MIN_QUALITY_SCORE_DEFAULT = 2

# --- Robustness thresholds ---
ROBUSTNESS_THRESHOLD_STRICT = 0.1
ROBUSTNESS_THRESHOLD_MODERATE = 0.2
ROBUSTNESS_THRESHOLD_LOOSE = 0.3

# --- Benchmark test sample sizes ---
BENCH_TEST_SAMPLE_SIZE = 3
BENCH_TEST_SAMPLE_MIN = 2

# --- TETNO steering base strength ---
STEERING_BASE_STRENGTH_DEFAULT = 0.5

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
MIN_TOKEN_LENGTH_NONSENSE = 4
