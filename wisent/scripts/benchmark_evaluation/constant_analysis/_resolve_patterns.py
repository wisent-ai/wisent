"""Pattern definitions for constant classification rules.

Used by resolve_constant.py to classify constants into:
    Rule 2 (hardware-dependent), Rule 3 (model-dependent),
    Rule 4 (definition/formatting).
"""
import re

# ── Rule 2: hardware-dependent patterns ──────────────────────────
# Test: "If I move from A100 to Raspberry Pi, should X change?"
HARDWARE_PATTERNS = [
    re.compile(r"BATCH_SIZE", re.IGNORECASE),
    re.compile(r"MAX_WORKERS"),
    re.compile(r"PARALLEL_WORKERS"),
    re.compile(r"PROCESS_POOL"),
    re.compile(r"CACHE_MAX_SIZE_GB"),
    re.compile(r"DOCKER_CPU_QUOTA"),
    re.compile(r"EXTRACTION_RAW_BATCH"),
    re.compile(r"EXTRACTION_SMALL_BATCH"),
    re.compile(r"GPU_MEMORY"),
    re.compile(r"_WORKER_COUNT"),
]

# ── Rule 3: model-dependent patterns ─────────────────────────────
# Test: "If I switch from Llama-1B to Qwen3-8B, should X change?"
MODEL_PATTERNS = [
    re.compile(r"DEFAULT_LAYER$"),
    re.compile(r"DEFAULT_STEERING_LAYERS$"),
    re.compile(r"SEARCH_MAX_LAYER_CAP$"),
    re.compile(r"CLASSIFIER_LAYER_RANGE_END$"),
    re.compile(r"LAYER_THRESHOLD_SMALL$"),
    re.compile(r"LAYER_THRESHOLD_MEDIUM$"),
    re.compile(r"LAYER_THRESHOLD_LARGE$"),
    re.compile(r"CLUSTER_SMALL_MAX_LAYERS$"),
    re.compile(r"CLUSTER_MEDIUM_MAX_LAYERS$"),
    re.compile(r"CLUSTER_LAYERS_"),
    re.compile(r"QUALITY_CLASSIFIER_DEFAULT_LAYER$"),
    re.compile(r"QWEN3_4B_DEFAULT_LAYER$"),
    re.compile(r"COMPARISON_STEERING_LAYER$"),
    re.compile(r"SUBSPACE_HIDDEN_DIM_LARGE$"),
    re.compile(r"AUDIO_WHISPER_MAX_TOKENS$"),
    re.compile(r"LLAMA_PAD_TOKEN_ID$"),
    re.compile(r"GEMMA_.*_BOS_FEATURES"),
]

# ── Rule 4: definition patterns ──────────────────────────────────
# Changing V would change the CONCEPT, not improve quality.
DEFINITION_PATTERNS_DISPLAY = [
    re.compile(r"^VIZ_"),
    re.compile(r"^DISPLAY_"),
    re.compile(r"^SEPARATOR_"),
    re.compile(r"^BANNER_"),
    re.compile(r"^PROGRESS_LOG_"),
    re.compile(r"^TRAINING_LOG_"),
    re.compile(r"^HEATMAP_TEXT_"),
    re.compile(r"^ROUNDING_PRECISION"),
    re.compile(r"^HASH_DISPLAY_LENGTH$"),
    re.compile(r"^JSON_INDENT$"),
    re.compile(r"^JAVA_INDENT_SPACES$"),
    re.compile(r"^EIGENVALUE_DISPLAY_LIMIT$"),
    re.compile(r"^BAR_CHART_SCALE$"),
    re.compile(r"^MAX_TAGS_PER_BENCHMARK$"),
    re.compile(r"^DISPLAY_DECIMAL_PRECISION$"),
    re.compile(r"^PRIORITY_HIGH$"),
    re.compile(r"^PRIORITY_MEDIUM$"),
    re.compile(r"^PRIORITY_LOW$"),
    re.compile(r"^PERCENTILE_P\d+$"),
    re.compile(r"^95$"),
    re.compile(r"^99$"),
    re.compile(r"^WM_BRANDING_WIDTH$"),
    re.compile(r"^TSNE_"),
    re.compile(r"^TECZA_LOGGING_INTERVAL$"),
    re.compile(r"^TETNO_CONDITION_LOGGING_INTERVAL$"),
    re.compile(r"^CLUSTER_PROMPT_TRUNCATION$"),
    re.compile(r"^CLUSTER_RESPONSE_TRUNCATION$"),
    re.compile(r"^AUTOTUNER_PROGRESS_INTERVAL$"),
    re.compile(r"^CLUSTER_PROGRESS_INTERVAL$"),
    re.compile(r"^COMPARISON_LOGGING_STEPS$"),
    re.compile(r"^TRAINING_LOG_FREQUENCY$"),
]

DEFINITION_PATTERNS_PHYSICAL = [
    re.compile(r"^SECONDS_PER_"),
    re.compile(r"^HOURS_PER_"),
    re.compile(r"^MS_PER_"),
    re.compile(r"^BYTES_PER_"),
    re.compile(r"^HTTP_STATUS_"),
    re.compile(r"^DEFAULT_DB_PORT$"),
    re.compile(r"^DOCKER_CPU_PERIOD_US$"),
    re.compile(r"^DOCKER_TMPFS_MODE$"),
    re.compile(r"^DOCKER_TMPFS_.*_SIZE_BYTES$"),
    re.compile(r"^SIMHASH_BIT_WIDTH$"),
    re.compile(r"^BLAKE2B_DIGEST_SIZE$"),
    re.compile(r"^SIMPLEQA_YEAR_DIGIT_LENGTH$"),
    re.compile(r"^MERCURY_RUNTIME_SENTINEL"),
]

DEFINITION_PATTERNS_STATISTICAL = [
    re.compile(r"^STAT_ALPHA$"),
    re.compile(r"^CONFIDENCE_LEVEL$"),
    re.compile(r"^TARGET_POWER$"),
    re.compile(r"^Z_CRITICAL_"),
    re.compile(r"^CI_PERCENTILE_"),
    re.compile(r"^EFFECT_SIZE_"),
    re.compile(r"^SIGNIFICANCE_ALPHA$"),
    re.compile(r"^NULL_TEST_SIGNIFICANCE_THRESHOLD$"),
    re.compile(r"^NULL_TEST_Z_SCORE_SIGNIFICANT$"),
    re.compile(r"^Z_SCORE_SIGNIFICANCE$"),
    re.compile(r"^POWER_ADEQUATE_THRESHOLD$"),
    re.compile(r"^CHANCE_LEVEL_ACCURACY$"),
    re.compile(r"^NONSENSE_BASELINE_ACCURACY$"),
    re.compile(r"^STABILITY_BINARY_VARIANCE$"),
    re.compile(r"^BLEU_MAX_"),
    re.compile(r"^MMLU_PRO_MAX_OPTIONS$"),
    re.compile(r"^AUDIO_SAMPLE_RATE$"),
    re.compile(r"^DEFAULT_AUDIO_SAMPLE_RATE$"),
    re.compile(r"^N_COMPONENTS_2D$"),
    re.compile(r"^SCORE_RANGE_"),
    re.compile(r"^SCORE_SCALE_"),
]

DEFINITION_PATTERNS_NUMERICAL = [
    re.compile(r"^NORM_EPS$"),
    re.compile(r"^LOG_EPS$"),
    re.compile(r"^ZERO_THRESHOLD$"),
    re.compile(r"^COMPARE_TOL$"),
    re.compile(r"^NEAR_ZERO_TOL$"),
    re.compile(r"^SHERMAN_MORRISON_EPS$"),
    re.compile(r"^MATH_REL_TOL$"),
    re.compile(r"^MATH_PERCENT_REL_TOL$"),
    re.compile(r"^SYMPY_REL_TOL$"),
]

DEFINITION_PATTERNS_ENUMS = [
    re.compile(r"^STEERING_STRATEGIES$"),
    re.compile(r"^TOKEN_AGGREGATIONS$"),
    re.compile(r"^PROMPT_CONSTRUCTIONS$"),
    re.compile(r"^DIRECTION_WEIGHTING_OPTIONS$"),
    re.compile(r"^OPTIMIZER_AGGREGATION_METHODS$"),
    re.compile(r"^OPTIMIZER_TOKEN_TARGETING$"),
    re.compile(r"^ROLE_PLAY_TOKENS$"),
    re.compile(r"^CHARTQA_.*_DELTAS$"),
    re.compile(r"^MAX_NEW_TOKENS_VERIFY_SINGLE$"),
    re.compile(r"^NONSENSE_MAX_TOKENS$"),
]

ALL_DEFINITION_PATTERNS = (
    DEFINITION_PATTERNS_DISPLAY
    + DEFINITION_PATTERNS_PHYSICAL
    + DEFINITION_PATTERNS_STATISTICAL
    + DEFINITION_PATTERNS_NUMERICAL
    + DEFINITION_PATTERNS_ENUMS
)


def classify_rule2(name):
    """Rule 2: hardware-dependent?"""
    return any(pat.search(name) for pat in HARDWARE_PATTERNS)


def classify_rule3(name):
    """Rule 3: model-dependent?"""
    return any(pat.search(name) for pat in MODEL_PATTERNS)


def classify_rule4(name):
    """Rule 4: definition, not a choice?"""
    return any(pat.search(name) for pat in ALL_DEFINITION_PATTERNS)


def half_double_trivially_passes(raw_val):
    """Non-numeric values (tuple/list/string) skip half/double test."""
    v = raw_val.rstrip(",")
    if v.startswith("(") or v.startswith("["):
        return True
    if v.startswith('"') or v.startswith("'"):
        return True
    if v.startswith("0o") or v.startswith("0x"):
        return True
    return False
