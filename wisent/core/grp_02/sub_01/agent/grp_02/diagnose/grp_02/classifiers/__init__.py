import os as _os
_base = _os.path.dirname(__file__)
for _root, _dirs, _files in _os.walk(_base):
    _dirs[:] = sorted(d for d in _dirs if d.startswith(("grp_", "sub_", "mid_")))
    if _root != _base:
        __path__.append(_root)

"""Classifier creation and management for diagnostic agents."""

from .classifier_marketplace import (
    ClassifierListing,
    ClassifierCreationEstimate,
    ClassifierMarketplace,
)
from .create_classifier import (
    TrainingConfig,
    TrainingResult,
    ClassifierCreator,
    create_classifier_on_demand,
)
from .select_classifiers import (
    ClassifierInfo,
    SelectionCriteria,
    ClassifierSelector,
    auto_select_classifiers_for_agent,
)
from .synthetic_classifier_option import (
    TraitDiscoveryResult,
    SyntheticClassifierResult,
    AutomaticTraitDiscovery,
    SyntheticClassifierFactory,
    SyntheticClassifierSystem,
    get_time_budget_from_manager,
    create_synthetic_classifier_system,
    create_classifiers_for_prompt,
    apply_classifiers_to_response,
    create_classifier_from_trait_description,
    evaluate_response_with_trait_classifier,
)

__all__ = [
    # Marketplace
    'ClassifierListing',
    'ClassifierCreationEstimate',
    'ClassifierMarketplace',
    # Creation
    'TrainingConfig',
    'TrainingResult',
    'ClassifierCreator',
    'create_classifier_on_demand',
    # Selection
    'ClassifierInfo',
    'SelectionCriteria',
    'ClassifierSelector',
    'auto_select_classifiers_for_agent',
    # Synthetic classifiers
    'TraitDiscoveryResult',
    'SyntheticClassifierResult',
    'AutomaticTraitDiscovery',
    'SyntheticClassifierFactory',
    'SyntheticClassifierSystem',
    'get_time_budget_from_manager',
    'create_synthetic_classifier_system',
    'create_classifiers_for_prompt',
    'apply_classifiers_to_response',
    'create_classifier_from_trait_description',
    'evaluate_response_with_trait_classifier',
]
