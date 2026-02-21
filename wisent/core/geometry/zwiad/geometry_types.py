"""Activation space geometry types derived from empirical clustering.

K-means on 208 benchmarks x 16 zwiad metrics identified two
granularity levels: 5 coarse types (best silhouette) and 8 fine
types (k=8). classify_geometry(fine=True) uses the 8-type system.
"""
from __future__ import annotations

import json
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class GeometryType(Enum):
    """Five coarse activation space geometry types (k=5)."""
    LINEAR_FLAT = "linear_flat"
    LINEAR_MULTI = "linear_multi"
    CURVED_DIFFUSE = "curved_diffuse"
    UNSTABLE_CONCENTRATED = "unstable_concentrated"
    CONCENTRATED_FEW = "concentrated_few"


class GeometryTypeFine(Enum):
    """Eight fine-grained geometry types (k=8).

    Maps to geometry_runner.py shape scores:
      LINEAR_STABLE      -> linear    (flat hyperplane)
      LINEAR_CONCENTRATED -> cone     (narrow linear cone)
      LINEAR_MULTIDIR    -> orthogonal (multi-dir linear)
      CONCENTRATED_FEW   -> bimodal   (2 concentrated poles)
      MID_MULTI_CONCEPT  -> cluster   (many concept clusters)
      DIFFUSE_CURVED     -> manifold  (curved surface)
      LOW_LINEAR_UNSTABLE -> sparse   (noisy, unstable)
      MANY_CONCEPT_CURVED -> manifold+cluster (fragmented)
    """
    LINEAR_STABLE = "linear_stable"
    DIFFUSE_CURVED = "diffuse_curved"
    CONCENTRATED_FEW = "concentrated_few"
    LINEAR_CONCENTRATED = "linear_concentrated"
    MID_MULTI_CONCEPT = "mid_multi_concept"
    LINEAR_MULTIDIR = "linear_multidir"
    LOW_LINEAR_UNSTABLE = "low_linear_unstable"
    MANY_CONCEPT_CURVED = "many_concept_curved"


# Maps fine geometry types to theoretical activation space shapes
# (as computed by geometry_runner.py structure scores)
SHAPE_MAP = {
    GeometryTypeFine.LINEAR_STABLE: "linear",
    GeometryTypeFine.LINEAR_CONCENTRATED: "cone",
    GeometryTypeFine.LINEAR_MULTIDIR: "orthogonal",
    GeometryTypeFine.CONCENTRATED_FEW: "bimodal",
    GeometryTypeFine.MID_MULTI_CONCEPT: "cluster",
    GeometryTypeFine.DIFFUSE_CURVED: "manifold",
    GeometryTypeFine.LOW_LINEAR_UNSTABLE: "sparse",
    GeometryTypeFine.MANY_CONCEPT_CURVED: "manifold",
}

# Maps fine geometry types to recommended steering methods.
# Populated from ground truth experiments via collect_ground_truth().
# Empty until experimental data is available.
METHOD_MAP: Dict[GeometryTypeFine, List[str]] = {}


_CENTROIDS_5 = {
    GeometryType.LINEAR_FLAT: {
        "linear": 0.980, "stability": 0.975,
        "n_concepts": 5, "var_pc1": 0.155,
        "curvature": 0.044, "coherence": 0.85,
    },
    GeometryType.LINEAR_MULTI: {
        "linear": 0.828, "stability": 0.968,
        "n_concepts": 8, "var_pc1": 0.148,
        "curvature": 0.116, "coherence": 0.72,
    },
    GeometryType.CURVED_DIFFUSE: {
        "linear": 0.565, "stability": 0.888,
        "n_concepts": 13, "var_pc1": 0.128,
        "curvature": 0.206, "coherence": 0.55,
    },
    GeometryType.UNSTABLE_CONCENTRATED: {
        "linear": 0.578, "stability": 0.844,
        "n_concepts": 8, "var_pc1": 0.344,
        "curvature": 0.204, "coherence": 0.58,
    },
    GeometryType.CONCENTRATED_FEW: {
        "linear": 0.751, "stability": 0.967,
        "n_concepts": 3, "var_pc1": 0.369,
        "curvature": 0.037, "coherence": 0.78,
    },
}

_CENTROIDS_8 = {
    GeometryTypeFine.LINEAR_STABLE: {
        "linear": 0.986, "stability": 0.997,
        "n_concepts": 19, "var_pc1": 0.229,
        "curvature": 0.102, "coherence": 0.707,
    },
    GeometryTypeFine.DIFFUSE_CURVED: {
        "linear": 0.529, "stability": 0.773,
        "n_concepts": 12, "var_pc1": 0.178,
        "curvature": 0.789, "coherence": 0.122,
    },
    GeometryTypeFine.CONCENTRATED_FEW: {
        "linear": 0.692, "stability": 0.948,
        "n_concepts": 2, "var_pc1": 0.877,
        "curvature": 0.190, "coherence": 0.825,
    },
    GeometryTypeFine.LINEAR_CONCENTRATED: {
        "linear": 0.947, "stability": 0.989,
        "n_concepts": 5, "var_pc1": 0.471,
        "curvature": 0.321, "coherence": 0.592,
    },
    GeometryTypeFine.MID_MULTI_CONCEPT: {
        "linear": 0.779, "stability": 0.983,
        "n_concepts": 14, "var_pc1": 0.262,
        "curvature": 0.569, "coherence": 0.226,
    },
    GeometryTypeFine.LINEAR_MULTIDIR: {
        "linear": 0.915, "stability": 0.994,
        "n_concepts": 4, "var_pc1": 0.368,
        "curvature": 0.274, "coherence": 0.388,
    },
    GeometryTypeFine.LOW_LINEAR_UNSTABLE: {
        "linear": 0.549, "stability": 0.722,
        "n_concepts": 4, "var_pc1": 0.543,
        "curvature": 0.533, "coherence": 0.388,
    },
    GeometryTypeFine.MANY_CONCEPT_CURVED: {
        "linear": 0.602, "stability": 0.834,
        "n_concepts": 92, "var_pc1": 0.205,
        "curvature": 0.794, "coherence": 0.139,
    },
}

_METRIC_ALIASES = {
    "linear": ["linear_probe_accuracy", "linear_probe"],
    "stability": ["direction_stability_score",
                   "direction_stability"],
    "n_concepts": ["n_concepts"],
    "var_pc1": ["manifold_variance_pc1"],
    "curvature": ["manifold_curvature",
                   "manifold_curvature_proxy"],
    "coherence": ["concept_coherence"],
}

_RANGES = {
    "linear": (0.3, 1.0), "stability": (0.5, 1.0),
    "n_concepts": (1, 100), "var_pc1": (0.05, 0.9),
    "curvature": (0.0, 0.9), "coherence": (0.0, 1.0),
}
_WEIGHTS = {
    "linear": 3.0, "stability": 1.5, "n_concepts": 1.0,
    "var_pc1": 2.0, "curvature": 2.0, "coherence": 1.5,
}


def _extract(metrics: Dict[str, Any], key: str) -> Optional[float]:
    for alias in _METRIC_ALIASES[key]:
        if alias in metrics and metrics[alias] is not None:
            return float(metrics[alias])
    return None


def _classify(metrics, centroids, default_type):
    vals = {}
    for key in _METRIC_ALIASES:
        v = _extract(metrics, key)
        if v is None:
            return default_type, 0.0
        vals[key] = v
    distances = {}
    for gtype, centroid in centroids.items():
        dist = 0.0
        for key in centroid:
            lo, hi = _RANGES[key]
            span = hi - lo if hi > lo else 1.0
            nv = (vals[key] - lo) / span
            nc = (centroid[key] - lo) / span
            dist += _WEIGHTS[key] * (nv - nc) ** 2
        distances[gtype] = dist ** 0.5
    ranked = sorted(distances.items(), key=lambda x: x[1])
    best, best_d = ranked[0]
    second_d = ranked[1][1]
    conf = min((second_d - best_d) / best_d, 1.0) if best_d > 0 else 1.0
    return best, round(conf, 3)


def classify_geometry(
    metrics: Dict[str, Any], fine: bool = False,
) -> Tuple:
    """Classify geometry. fine=True uses 8 types, False uses 5."""
    if fine:
        return _classify(metrics, _CENTROIDS_8,
                         GeometryTypeFine.LINEAR_STABLE)
    return _classify(metrics, _CENTROIDS_5,
                     GeometryType.LINEAR_FLAT)


def select_representative_benchmarks(
    zwiad_dir: str, model_slug: str,
    per_type: int = 2, fine: bool = False,
) -> Dict:
    """Select representative benchmarks covering all types."""
    zwiad_path = Path(zwiad_dir)
    prefix = model_slug.replace("/", "_")
    enum_cls = GeometryTypeFine if fine else GeometryType
    typed = {t: [] for t in enum_cls}
    for f in sorted(zwiad_path.glob(f"{prefix}__*.json")):
        bench = f.stem.replace(f"{prefix}__", "")
        data = json.loads(f.read_text())
        m = data.get("metrics", data)
        gtype, conf = classify_geometry(m, fine=fine)
        typed[gtype].append((bench, conf))
    selected = {}
    for gtype in enum_cls:
        cands = typed[gtype]
        cands.sort(key=lambda x: x[1], reverse=True)
        selected[gtype] = [b for b, _ in cands[:per_type]]
    return selected


@dataclass
class GeometryProfile:
    """Full geometry characterization of a benchmark."""
    geometry_type: str
    geometry_type_fine: str
    activation_shape: str
    recommended_methods: List[str]
    confidence: float
    confidence_fine: float
    metrics: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "geometry_type": self.geometry_type,
            "geometry_type_fine": self.geometry_type_fine,
            "activation_shape": self.activation_shape,
            "recommended_methods": self.recommended_methods,
            "confidence": self.confidence,
            "confidence_fine": self.confidence_fine,
            "metrics": self.metrics,
        }

    @staticmethod
    def _lookup_methods(gt_fine) -> List[str]:
        """Look up methods from METHOD_MAP, empty if not yet populated."""
        return list(METHOD_MAP.get(gt_fine, []))

    @classmethod
    def from_dict(cls, d: dict) -> GeometryProfile:
        return cls(**d)


def profile_benchmark(
    metrics: Dict[str, Any],
) -> GeometryProfile:
    """Build full geometry profile with both granularities."""
    gt5, c5 = classify_geometry(metrics, fine=False)
    gt8, c8 = classify_geometry(metrics, fine=True)
    key_metrics = {}
    for key in _METRIC_ALIASES:
        v = _extract(metrics, key)
        if v is not None:
            key_metrics[key] = v
    return GeometryProfile(
        geometry_type=gt5.value,
        geometry_type_fine=gt8.value,
        activation_shape=SHAPE_MAP[gt8],
        recommended_methods=GeometryProfile._lookup_methods(gt8),
        confidence=c5,
        confidence_fine=c8,
        metrics=key_metrics,
    )
