"""Zwiad protocol configuration and adaptive thresholds."""
from dataclasses import dataclass, field
from typing import List
import numpy as np
from wisent.core.utils.config_tools.constants import STAT_ALPHA


def adaptive_gap_threshold(n_samples: int, *, zwiad_gap_k: float, zwiad_gap_min: float, zwiad_gap_max: float) -> float:
    """Adaptive geometry gap threshold based on sample size."""
    threshold = zwiad_gap_k / np.sqrt(n_samples)
    return max(zwiad_gap_min, min(threshold, zwiad_gap_max))


def adaptive_min_silhouette(n_samples: int, *, default_scale: float, zwiad_sil_base: float, zwiad_sil_slope: float, zwiad_sil_scale_samples: float, zwiad_sil_max: float) -> float:
    """Adaptive silhouette threshold based on sample size."""
    base = zwiad_sil_base + zwiad_sil_slope * min(default_scale, n_samples / zwiad_sil_scale_samples)
    return max(zwiad_sil_base, min(base, zwiad_sil_max))


@dataclass
class ZwiadProtocolConfig:
    """Configuration for Zwiad protocol."""
    signal_keys: List[str] = field(
        default_factory=lambda: ["knn_accuracy", "knn_pca_accuracy", "mlp_probe_accuracy"]
    )
