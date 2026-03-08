"""Derive geometry metric parameters from data shape at runtime.

Parameters are split into two categories:
  A. Fixed by definition (mathematically determined)
  B. Derivable from data (textbook formulas, data-shape constraints)

Non-derivable design choices (probe sizes, bootstrap counts, consistency
weights, thresholds) are required keyword arguments on compute_geometry_metrics.
"""

from typing import Dict, Any

from wisent.core.utils.config_tools.constants import (
    CHANCE_LEVEL_ACCURACY, SCORE_RANGE_MIN, COMBO_OFFSET,
    GEOMETRY_VARIANCE_EXPLAINED_90PCT, GEOMETRY_MIN_CLOUD_POINTS,
    GEOMETRY_CV_FOLDS_MAX, GEOMETRY_CV_FOLDS_MIN,
    GEOMETRY_KNN_K_MIN, GEOMETRY_KNN_K_MAX,
    GEOMETRY_PCA_NULL_CAP, GEOMETRY_PROBE_VALIDATION_MIN,
    GEOMETRY_PROBE_VALIDATION_MIN_SAMPLES,
)


def derive_geometry_params(n_samples: int, n_features: int) -> Dict[str, Any]:
    """Compute all geometry metric parameters from data dimensions.

    Args:
        n_samples: Number of positive (or negative) activation vectors.
        n_features: Dimensionality of each activation vector.

    Returns:
        Dict with all parameters needed by compute_geometry_metrics internals.
    """
    # --- Category A: Fixed by definition ---
    blend_default = CHANCE_LEVEL_ACCURACY
    default_score = SCORE_RANGE_MIN
    variance_explained_90pct = GEOMETRY_VARIANCE_EXPLAINED_90PCT
    feature_dim_index = COMBO_OFFSET

    # --- Category B: Derivable from data ---
    subsample_threshold = n_samples
    pca_dims_limit = min(n_samples - COMBO_OFFSET, n_features)
    cv_folds = min(GEOMETRY_CV_FOLDS_MAX, max(GEOMETRY_CV_FOLDS_MIN, n_samples))
    probe_knn_k = max(
        GEOMETRY_KNN_K_MIN,
        min(int(n_samples ** CHANCE_LEVEL_ACCURACY), GEOMETRY_KNN_K_MAX),
    )
    probe_min_per_class = max(GEOMETRY_CV_FOLDS_MIN, cv_folds)
    knn_min_class_offset = probe_knn_k
    min_cloud_points = GEOMETRY_MIN_CLOUD_POINTS
    probe_validation_fraction = max(
        GEOMETRY_PROBE_VALIDATION_MIN,
        GEOMETRY_PROBE_VALIDATION_MIN_SAMPLES / n_samples,
    )
    pca_max_components_null = min(
        n_samples - COMBO_OFFSET, n_features, GEOMETRY_PCA_NULL_CAP,
    )

    return {
        "blend_default": blend_default,
        "default_score": default_score,
        "variance_explained_90pct": variance_explained_90pct,
        "feature_dim_index": feature_dim_index,
        "subsample_threshold": subsample_threshold,
        "pca_dims_limit": pca_dims_limit,
        "cv_folds": cv_folds,
        "probe_knn_k": probe_knn_k,
        "probe_min_per_class": probe_min_per_class,
        "knn_min_class_offset": knn_min_class_offset,
        "min_cloud_points": min_cloud_points,
        "probe_validation_fraction": probe_validation_fraction,
        "pca_max_components_null": pca_max_components_null,
    }
