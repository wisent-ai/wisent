"""
Overflow named constants for the wisent package (part 13).
Imported by _constants_part12.py via star-import so all constants remain
accessible from wisent.core.constants.
"""

# --- Time conversions ---
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400
HOURS_PER_DAY = 24

# --- Byte conversions ---
BYTES_PER_KB = 1024
BYTES_PER_MB = 1024 * 1024
BYTES_PER_GB = 1024 ** 3

# --- Separator widths (print formatting) ---
SEPARATOR_WIDTH_COMPACT = 30
SEPARATOR_WIDTH_HALF = 35
SEPARATOR_WIDTH_NARROW = 40
SEPARATOR_WIDTH_MEDIUM = 50
SEPARATOR_WIDTH_MEDIUM_PLUS = 55
SEPARATOR_WIDTH_TABLE = 56
SEPARATOR_WIDTH_STANDARD = 60
SEPARATOR_WIDTH_PLUS = 65
SEPARATOR_WIDTH_WIDE = 70
SEPARATOR_WIDTH_WIDE_PLUS = 75
SEPARATOR_WIDTH_REPORT = 80
SEPARATOR_WIDTH_SENSITIVITY = 89
SEPARATOR_WIDTH_FULL = 100
SEPARATOR_WIDTH_ULTRA = 128
SEPARATOR_WIDTH_MAX = 150
SEPARATOR_PAD_SMALL = 20

# --- HTTP status codes ---
HTTP_STATUS_OK = 200

# --- JSON formatting ---
JSON_INDENT = 2

# --- Display decimal precision ---
DISPLAY_DECIMAL_PRECISION = 2

# --- Progress logging intervals ---
PROGRESS_LOG_INTERVAL_10 = 10
PROGRESS_LOG_INTERVAL_20 = 20

# --- Visualization font sizes (additional) ---
VIZ_FONTSIZE_TINY = 7
VIZ_FONTSIZE_BODY = 10
VIZ_FONTSIZE_SUBTITLE = 12
VIZ_FONTSIZE_LARGE = 16

# --- Visualization line widths (additional) ---
VIZ_LINEWIDTH_FINE = 0.5
VIZ_LINEWIDTH_HAIRLINE = 1

# --- Visualization colorbar ---
VIZ_COLORBAR_FRACTION = 0.046
VIZ_COLORBAR_PAD = 0.04

# --- Visualization suptitle y-offset ---
VIZ_SUPTITLE_Y_OFFSET = 1.01
VIZ_SUPTITLE_Y_OFFSET_HIGH = 1.02

# --- Heatmap accuracy bounds ---
VIZ_HEATMAP_ACCURACY_VMIN = 0.5
VIZ_HEATMAP_ACCURACY_VMAX = 1.0

# --- Histogram alpha ---
VIZ_HISTOGRAM_ALPHA = 0.5

# --- Percentile values ---
PERCENTILE_HIGH = 95
PERCENTILE_CRITICAL = 99

# --- Scale/split defaults ---
SPLIT_RATIO_FULL = 1.0
DEFAULT_SCALE_FACTOR = 1.0

# --- Single-threaded job count ---
N_JOBS_SINGLE = 1

# --- Pixel max value (image processing) ---
PIXEL_MAX_VALUE = 255

# --- Score adjustments ---
CLASSIFIER_RECENCY_BONUS = 0.1
COHERENCE_DEGENERATION_PENALTY = 0.5

# --- Visualization alpha values (additional) ---
VIZ_ALPHA_HIST = 0.4

# --- Time unit conversions (ms) ---
MS_PER_SECOND = 1000

# --- Correlation heatmap bounds ---
VIZ_CORRELATION_VMIN = -1
VIZ_CORRELATION_VMAX = 1

# --- Contrastive pair limits ---
MAX_INCORRECT_PER_CORRECT = 2

# --- Evidence train/test split fraction ---
EVIDENCE_TRAIN_SPLIT_NUM = 2
EVIDENCE_TRAIN_SPLIT_DEN = 3

# --- Dimension thresholds ---
HIGH_DIM_THRESHOLD = 1000

# --- Visualization line plot marker sizes ---
VIZ_MARKERSIZE_LINE_SMALL = 6
VIZ_MARKERSIZE_LINE = 8

# --- Heatmap generic bounds ---
VIZ_HEATMAP_VMIN_ZERO = 0
VIZ_HEATMAP_VMAX_ONE = 1

# --- Comparison script defaults ---
COMPARISON_OPTIMAL_LAYER = 4
COMPARISON_OPTIMAL_STRENGTH = 0.5
COMPARISON_TOTAL_PAIRS = 80

# --- Database connection retries ---
DB_CONNECTION_MAX_RETRIES = 5
