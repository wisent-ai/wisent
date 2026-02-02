"""Control vector diagnostics package.

This package provides tools for analyzing steering/control vectors:
- Basic diagnostics (norm, zero-fraction, health checks)
- Cone structure analysis
- Geometry structure detection (linear, cone, cluster, manifold, etc.)
- Multi-layer geometry analysis
- Exhaustive layer combination analysis
"""

from .core import (
    ControlVectorDiagnosticsConfig,
    run_control_vector_diagnostics,
    run_control_steering_diagnostics,
)

from .cone import (
    ConeAnalysisConfig,
    ConeAnalysisResult,
    check_cone_structure,
)

from .geometry import (
    StructureType,
    StructureScore,
    GeometryAnalysisConfig,
    GeometryAnalysisResult,
    detect_geometry_structure,
)

from .multi_layer import (
    MultiLayerGeometryConfig,
    MultiLayerGeometryResult,
    LayerGeometryResult,
    detect_geometry_multi_layer,
    detect_geometry_all_layers,
)

from .exhaustive import (
    ExhaustiveCombinationResult,
    ExhaustiveGeometryAnalysisResult,
    detect_geometry_exhaustive,
    detect_geometry_limited,
    detect_geometry_contiguous,
    detect_geometry_smart,
)

__all__ = [
    # Core diagnostics
    "ControlVectorDiagnosticsConfig",
    "run_control_vector_diagnostics",
    "run_control_steering_diagnostics",
    # Cone analysis
    "ConeAnalysisConfig",
    "ConeAnalysisResult",
    "check_cone_structure",
    # Geometry detection
    "StructureType",
    "StructureScore",
    "GeometryAnalysisConfig",
    "GeometryAnalysisResult",
    "detect_geometry_structure",
    # Multi-layer
    "MultiLayerGeometryConfig",
    "MultiLayerGeometryResult",
    "LayerGeometryResult",
    "detect_geometry_multi_layer",
    "detect_geometry_all_layers",
    # Exhaustive
    "ExhaustiveCombinationResult",
    "ExhaustiveGeometryAnalysisResult",
    "detect_geometry_exhaustive",
    "detect_geometry_limited",
    "detect_geometry_contiguous",
    "detect_geometry_smart",
]
