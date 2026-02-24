"""Zwiad protocol configuration and adaptive thresholds."""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from wisent.core.constants import (
    ZWIAD_GAP_K,
    ZWIAD_GAP_MIN,
    ZWIAD_GAP_MAX,
    ZWIAD_SIL_BASE,
    ZWIAD_SIL_SLOPE,
    ZWIAD_SIL_SCALE_SAMPLES,
    ZWIAD_SIL_MAX,
    DEFAULT_SCALE,
    STAT_ALPHA,
)


def adaptive_gap_threshold(n_samples: int) -> float:
    """Adaptive geometry gap threshold based on sample size.

    With more samples, accuracy estimates are more stable, so we can
    require a smaller gap to declare nonlinearity meaningful.
    Based on: threshold = k / sqrt(n_samples) where k controls sensitivity.
    """
    threshold = ZWIAD_GAP_K / np.sqrt(n_samples)
    return max(ZWIAD_GAP_MIN, min(threshold, ZWIAD_GAP_MAX))


def adaptive_min_silhouette(n_samples: int) -> float:
    """Adaptive silhouette threshold based on sample size.

    With more samples, silhouette scores are more reliable, so we can
    require higher scores to declare fragmentation.
    """
    base = ZWIAD_SIL_BASE + ZWIAD_SIL_SLOPE * min(DEFAULT_SCALE, n_samples / ZWIAD_SIL_SCALE_SAMPLES)
    return max(ZWIAD_SIL_BASE, min(base, ZWIAD_SIL_MAX))


@dataclass
class ZwiadProtocolConfig:
    """Configuration for Zwiad protocol."""
    signal_keys: List[str] = None
    p_threshold: float = STAT_ALPHA
    gap_threshold: Optional[float] = None
    min_silhouette: Optional[float] = None

    def __post_init__(self):
        if self.signal_keys is None:
            self.signal_keys = ["knn_accuracy", "knn_pca_accuracy", "mlp_probe_accuracy"]
