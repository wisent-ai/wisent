"""Re-export diagnostics from wisent.core.reading.diagnostics for backward compat.

This package acts as an alias: any import of
``wisent.core.primitives.contrastive_pairs.diagnostics.X``
is transparently redirected to ``wisent.core.reading.diagnostics.X``.
"""

import sys
import importlib

_CANONICAL = "wisent.core.reading.diagnostics"
_ALIAS = __name__  # wisent.core.primitives.contrastive_pairs.diagnostics

# Force-import the canonical tree so all sub-modules are in sys.modules.
importlib.import_module(_CANONICAL)
importlib.import_module(_CANONICAL + ".base")
importlib.import_module(_CANONICAL + ".metrics")
importlib.import_module(_CANONICAL + ".control_vectors")
importlib.import_module(_CANONICAL + ".analysis")
importlib.import_module(_CANONICAL + ".analysis.concept_analysis")
importlib.import_module(_CANONICAL + ".analysis.linearity")
importlib.import_module(_CANONICAL + ".analysis.vector_quality")
importlib.import_module(_CANONICAL + ".analysis._concept_analysis_part2")
importlib.import_module(_CANONICAL + ".analysis._linearity_from_activations")
importlib.import_module(_CANONICAL + ".analysis._vector_quality_helpers")
importlib.import_module(_CANONICAL + ".analysis._vector_quality_runner")

# Create sys.modules aliases for every sub-module.
for _key, _mod in list(sys.modules.items()):
    if _key.startswith(_CANONICAL + "."):
        _suffix = _key[len(_CANONICAL):]
        sys.modules[_ALIAS + _suffix] = _mod

# Re-export top-level names.
from wisent.core.reading.diagnostics import *  # noqa: F401,F403
from wisent.core.reading.diagnostics import (  # noqa: F401
    DiagnosticsConfig,
    DiagnosticsReport,
    run_all_diagnostics,
    ControlVectorDiagnosticsConfig,
    run_control_vector_diagnostics,
    run_control_steering_diagnostics,
    ConeAnalysisConfig,
    ConeAnalysisResult,
    check_cone_structure,
    GeometryAnalysisConfig,
    GeometryAnalysisResult,
    StructureType,
    detect_geometry_structure,
    VectorQualityConfig,
    VectorQualityReport,
    run_vector_quality_diagnostics,
    LinearityConfig,
    LinearityResult,
    LinearityVerdict,
    check_linearity,
    check_linearity_from_activations,
    ConceptAnalysisResult,
    compute_icd,
    compute_eigenvalue_spectrum,
    decompose_concepts,
    compute_concept_correlations,
    analyze_concepts,
    analyze_concept_interference,
)
