"""Minerva math group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

MINERVA_MATH_TASKS = {
    "minerva_math": f"{BASE_IMPORT}minerva_math:MinervaMathExtractor",
    "minerva_math_algebra": f"{BASE_IMPORT}minerva_math:MinervaMathExtractor",
    "minerva_math_counting_and_prob": f"{BASE_IMPORT}minerva_math:MinervaMathExtractor",
    "minerva_math_geometry": f"{BASE_IMPORT}minerva_math:MinervaMathExtractor",
    "minerva_math_intermediate_algebra": f"{BASE_IMPORT}minerva_math:MinervaMathExtractor",
    "minerva_math_num_theory": f"{BASE_IMPORT}minerva_math:MinervaMathExtractor",
    "minerva_math_prealgebra": f"{BASE_IMPORT}minerva_math:MinervaMathExtractor",
    "minerva_math_precalc": f"{BASE_IMPORT}minerva_math:MinervaMathExtractor",
}
