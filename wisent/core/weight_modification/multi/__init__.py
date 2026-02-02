"""Multi-concept and multi-direction weight modification implementations."""

from .multi_concept import (
    ConceptAction,
    ConceptSpec,
    MultiConceptConfig,
    MultiConceptResult,
    compute_interference_matrix,
    orthogonalize_concept_directions,
    bidirectional_projection,
    run_multi_concept_modification,
)
from .multi_direction import (
    MultiDirectionConfig,
    MultiDirectionResult,
    combine_directions,
    bake_multi_directions,
    train_and_bake,
)

__all__ = [
    'ConceptAction',
    'ConceptSpec',
    'MultiConceptConfig',
    'MultiConceptResult',
    'compute_interference_matrix',
    'orthogonalize_concept_directions',
    'bidirectional_projection',
    'run_multi_concept_modification',
    'MultiDirectionConfig',
    'MultiDirectionResult',
    'combine_directions',
    'bake_multi_directions',
    'train_and_bake',
]
