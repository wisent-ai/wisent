"""
Synthetic Classifier Option System.

Re-exports from split modules for backward compatibility.
"""
from wisent.core.agent.diagnose.classifiers._synthetic_classes import (
    TraitDiscoveryResult,
    SyntheticClassifierResult,
    AutomaticTraitDiscovery,
    SyntheticClassifierFactory,
)
from wisent.core.agent.diagnose.classifiers._synthetic_system import (
    SyntheticClassifierSystem,
)
from wisent.core.agent.diagnose.classifiers._synthetic_functions import (
    get_time_budget_from_manager,
    create_synthetic_classifier_system,
    create_classifiers_for_prompt,
    apply_classifiers_to_response,
    create_classifier_from_trait_description,
    evaluate_response_with_trait_classifier,
)
