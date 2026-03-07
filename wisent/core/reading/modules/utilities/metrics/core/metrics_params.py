"""Derive geometry metric parameters from data shape at runtime.

Parameters are split into three categories:
  A. Fixed by definition (mathematically determined)
  B. Derivable from data (textbook formulas, data-shape constraints)
  C. Arbitrary design choices (temporary placeholders, need empirical validation)
"""

from typing import Dict, Any

from wisent.core.utils.config_tools.constants import (
    CHANCE_LEVEL_ACCURACY, SCORE_RANGE_MIN, COMBO_OFFSET, MATH_REL_TOL,
    GEOMETRY_VARIANCE_EXPLAINED_90PCT, GEOMETRY_MIN_CLOUD_POINTS,
    GEOMETRY_PROBE_SMALL_HIDDEN_CAP, GEOMETRY_PROBE_MLP_HIDDEN_CAP,
    GEOMETRY_SPECTRAL_NEIGHBORS_CAP, GEOMETRY_DIRECTION_N_BOOTSTRAP,
    GEOMETRY_DIRECTION_SUBSET_FRACTION, GEOMETRY_DIRECTION_STD_PENALTY,
    GEOMETRY_CONSISTENCY_W_EQUAL, GEOMETRY_SPARSITY_THRESHOLD_FRACTION,
    GEOMETRY_DIRECTION_MODERATE_SIMILARITY, GEOMETRY_CV_FOLDS_MAX,
    GEOMETRY_CV_FOLDS_MIN, GEOMETRY_KNN_K_MIN, GEOMETRY_KNN_K_MAX,
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

    # --- Category C: Arbitrary (TEMPORARY, needs empirical validation) ---
    probe_small_hidden = min(n_features, GEOMETRY_PROBE_SMALL_HIDDEN_CAP)
    probe_mlp_hidden = min(n_features, GEOMETRY_PROBE_MLP_HIDDEN_CAP)
    probe_mlp_alpha = MATH_REL_TOL
    spectral_n_neighbors = min(
        n_samples - COMBO_OFFSET, GEOMETRY_SPECTRAL_NEIGHBORS_CAP,
    )
    direction_n_bootstrap = GEOMETRY_DIRECTION_N_BOOTSTRAP
    direction_subset_fraction = GEOMETRY_DIRECTION_SUBSET_FRACTION
    direction_std_penalty = GEOMETRY_DIRECTION_STD_PENALTY
    consistency_w_cosine = GEOMETRY_CONSISTENCY_W_EQUAL
    consistency_w_positive = GEOMETRY_CONSISTENCY_W_EQUAL
    consistency_w_high_sim = GEOMETRY_CONSISTENCY_W_EQUAL
    sparsity_threshold_fraction = GEOMETRY_SPARSITY_THRESHOLD_FRACTION
    detection_threshold = CHANCE_LEVEL_ACCURACY
    direction_moderate_similarity = GEOMETRY_DIRECTION_MODERATE_SIMILARITY

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
        "probe_small_hidden": probe_small_hidden,
        "probe_mlp_hidden": probe_mlp_hidden,
        "probe_mlp_alpha": probe_mlp_alpha,
        "spectral_n_neighbors": spectral_n_neighbors,
        "direction_n_bootstrap": direction_n_bootstrap,
        "direction_subset_fraction": direction_subset_fraction,
        "direction_std_penalty": direction_std_penalty,
        "consistency_w_cosine": consistency_w_cosine,
        "consistency_w_positive": consistency_w_positive,
        "consistency_w_high_sim": consistency_w_high_sim,
        "sparsity_threshold_fraction": sparsity_threshold_fraction,
        "detection_threshold": detection_threshold,
        "direction_moderate_similarity": direction_moderate_similarity,
    }
