"""Analysis modules for contrastive pair diagnostics."""

from .concept_analysis import (
    ConceptAnalysisResult,
    compute_icd,
    compute_eigenvalue_spectrum,
    decompose_concepts,
    compute_concept_correlations,
    analyze_concepts,
    analyze_concept_interference,
)
from .linearity import (
    LinearityConfig,
    LinearityResult,
    LinearityVerdict,
    check_linearity,
    check_linearity_from_activations,
)
from .vector_quality import (
    VectorQualityConfig,
    VectorQualityReport,
    run_vector_quality_diagnostics,
)

__all__ = [
    # Concept analysis
    "ConceptAnalysisResult",
    "compute_icd",
    "compute_eigenvalue_spectrum",
    "decompose_concepts",
    "compute_concept_correlations",
    "analyze_concepts",
    "analyze_concept_interference",
    # Linearity
    "LinearityConfig",
    "LinearityResult",
    "LinearityVerdict",
    "check_linearity",
    "check_linearity_from_activations",
    # Vector quality
    "VectorQualityConfig",
    "VectorQualityReport",
    "run_vector_quality_diagnostics",
]
