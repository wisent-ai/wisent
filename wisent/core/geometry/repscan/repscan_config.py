"""RepScan protocol configuration and adaptive thresholds."""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


def adaptive_gap_threshold(n_samples: int) -> float:
    """Adaptive geometry gap threshold based on sample size.

    With more samples, accuracy estimates are more stable, so we can
    require a smaller gap to declare nonlinearity meaningful.
    Based on: threshold = k / sqrt(n_samples) where k controls sensitivity.
    """
    threshold = 0.5 / np.sqrt(n_samples)
    return max(0.02, min(threshold, 0.1))


def adaptive_min_silhouette(n_samples: int) -> float:
    """Adaptive silhouette threshold based on sample size.

    With more samples, silhouette scores are more reliable, so we can
    require higher scores to declare fragmentation.
    """
    base = 0.05 + 0.1 * min(1.0, n_samples / 500)
    return max(0.05, min(base, 0.2))


@dataclass
class RepScanProtocolConfig:
    """Configuration for RepScan protocol."""
    signal_keys: List[str] = None
    p_threshold: float = 0.05
    gap_threshold: Optional[float] = None
    min_silhouette: Optional[float] = None
    rigorous_geometry: bool = False

    def __post_init__(self):
        if self.signal_keys is None:
            self.signal_keys = ["knn_accuracy", "knn_pca_accuracy", "mlp_probe_accuracy"]
