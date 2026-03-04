"""SyntheticClassifierSystem - orchestrates trait-based classification."""
import logging
import time
from typing import List, Tuple
from wisent.core.reading.classifiers.core.atoms import ActivationClassifier
from wisent.core.utils.config_tools.constants import SECONDS_PER_MINUTE
from wisent.core.utils.infra_tools.errors import MissingParameterError
from wisent.core.experimental.agent.diagnose.classifiers._synthetic_classes import (
    TraitDiscoveryResult, SyntheticClassifierResult,
    AutomaticTraitDiscovery, SyntheticClassifierFactory,
)

logger = logging.getLogger(__name__)

class SyntheticClassifierSystem:
    """
    Creates synthetic classifiers based on prompt analysis and applies them to response activations.

    Analyzes prompts to discover relevant traits, creates classifiers using synthetic
    contrastive pairs, and applies them to response activations only.
    """

    def __init__(self, model, layer: int = None, min_pairs: int = None, time_multiplier: float = None, trait_discovery_cost_s: float = None, data_gen_cost_per_pair_s: float = None, classifier_training_cost_s: float = None):
        for _n, _v in [("min_pairs", min_pairs), ("time_multiplier", time_multiplier), ("trait_discovery_cost_s", trait_discovery_cost_s), ("data_gen_cost_per_pair_s", data_gen_cost_per_pair_s), ("classifier_training_cost_s", classifier_training_cost_s)]:
            if _v is None:
                raise ValueError(f"{_n} is required")
        self.model = model
        self.layer = layer
        self._min_pairs = min_pairs
        self._time_multiplier = time_multiplier
        self._trait_discovery_cost_s = trait_discovery_cost_s
        self._data_gen_cost_per_pair_s = data_gen_cost_per_pair_s
        self._classifier_training_cost_s = classifier_training_cost_s
        self.trait_discovery = AutomaticTraitDiscovery(model, layer=layer)
        self.classifier_factory = SyntheticClassifierFactory(model, layer=layer)

    def create_classifiers_for_prompt(
        self, prompt: str, time_budget_minutes: float, pairs_per_trait: int = None
    ) -> Tuple[List[ActivationClassifier], TraitDiscoveryResult]:
        """
        Create synthetic classifiers for a prompt by discovering relevant traits.
        Uses budget-aware planning to make intelligent decisions about what operations to perform.

        Args:
            prompt: The prompt to analyze and create classifiers for
            time_budget_minutes: Time budget for classifier creation in minutes
            pairs_per_trait: Number of contrastive pairs per trait

        Returns:
            Tuple of (list of trained classifiers, trait discovery result)
        """
        if pairs_per_trait is None:
            raise ValueError("pairs_per_trait is required")
        logging.info(f"Creating synthetic classifiers for prompt (budget: {time_budget_minutes:.1f} minutes)...")

        # Get cost estimates from device benchmarks
        try:
            from ..budget import estimate_task_time_direct
            model_loading_cost = estimate_task_time_direct("model_loading", 1)
            trait_discovery_cost = self._trait_discovery_cost_s
            data_generation_cost = estimate_task_time_direct("data_generation", 1)
            classifier_training_cost = estimate_task_time_direct("classifier_training", 100) / 100
            logging.info("Cost estimates per unit:")
            logging.info(f"Trait discovery: ~{trait_discovery_cost:.0f}s")
            logging.info(f"Data generation: ~{data_generation_cost:.0f}s per pair")
            logging.info(f"Classifier training: ~{classifier_training_cost:.0f}s per classifier")
        except Exception as e:
            logging.info(f"Could not get benchmark data: {e}")
            logging.info("Using configured estimates")
            trait_discovery_cost = self._trait_discovery_cost_s
            data_generation_cost = self._data_gen_cost_per_pair_s
            classifier_training_cost = self._classifier_training_cost_s

        budget_seconds = time_budget_minutes * SECONDS_PER_MINUTE

        # Step 1: Budget-aware trait discovery
        logging.info("Discovering relevant traits for this prompt...")

        # Estimate if we have enough budget for even basic operations
        min_required_time = trait_discovery_cost + (data_generation_cost * self._time_multiplier) + classifier_training_cost

        if budget_seconds < min_required_time:
            logging.info(f"Budget ({budget_seconds:.0f}s) too small for full classifier training")
            logging.info(f"Minimum required: {min_required_time:.0f}s")
            logging.info("Falling back to simple trait analysis only...")

            # Just do trait discovery without training classifiers
            discovery_result = self.trait_discovery.discover_relevant_traits(prompt, time_budget_minutes)
            logging.info(
                f"Discovered {len(discovery_result.traits_discovered)} traits: {discovery_result.traits_discovered}"
            )
            logging.info("Skipping classifier training due to budget constraints")
            return [], discovery_result

        # Calculate how many traits we can afford
        cost_per_trait = (data_generation_cost * pairs_per_trait) + classifier_training_cost
        available_for_traits = budget_seconds - trait_discovery_cost
        max_affordable_traits = max(1, int(available_for_traits / cost_per_trait))

        logging.info("Budget analysis:")
        logging.info(f"• Available time: {budget_seconds:.0f}s")
        logging.info(f"• Cost per trait ({pairs_per_trait} pairs): {cost_per_trait:.0f}s")
        logging.info(f"• Max affordable traits: {max_affordable_traits}")

        discovery_result = self.trait_discovery.discover_relevant_traits(prompt, time_budget_minutes)

        if not discovery_result.traits_discovered:
            logging.info("No traits discovered, cannot create classifiers")
            return [], discovery_result

        # Limit traits to what we can afford
        affordable_traits = discovery_result.traits_discovered[:max_affordable_traits]
        if len(affordable_traits) < len(discovery_result.traits_discovered):
            logging.info(
                f"Budget limiting to {len(affordable_traits)}/{len(discovery_result.traits_discovered)} traits"
            )

        logging.info(f"Processing {len(affordable_traits)} traits: {affordable_traits}")

        # Step 2: Create classifiers for affordable traits with smart resource allocation
        classifiers = []
        remaining_budget = budget_seconds - trait_discovery_cost

        for i, trait_description in enumerate(affordable_traits):
            logging.info(f"Creating classifier {i + 1}/{len(affordable_traits)}: {trait_description}")
            logging.info(f"Remaining budget: {remaining_budget:.0f}s")

            # Estimate cost for this specific classifier
            estimated_pairs_cost = data_generation_cost * pairs_per_trait
            estimated_training_cost = classifier_training_cost
            total_estimated_cost = estimated_pairs_cost + estimated_training_cost

            if total_estimated_cost > remaining_budget:
                # Try with fewer pairs
                max_affordable_pairs = max(self._min_pairs, int((remaining_budget - classifier_training_cost) / data_generation_cost))
                if max_affordable_pairs < self._min_pairs:
                    logging.info(f"Insufficient budget ({remaining_budget:.0f}s) for training, skipping")
                    continue
                logging.info(f"Reducing pairs from {pairs_per_trait} to {max_affordable_pairs} to fit budget")
                actual_pairs = max_affordable_pairs
            else:
                actual_pairs = pairs_per_trait

            try:
                start_time = time.time()

                # Create classifier for this trait
                logging.info(f"Creating classifier with {actual_pairs} pairs...")
                classifier, pairs_count = self.classifier_factory.create_classifier_from_trait(
                    trait_description=trait_description, num_pairs=actual_pairs
                )

                actual_time = time.time() - start_time
                remaining_budget -= actual_time

                # Store trait info in classifier for later reference
                classifier._trait_description = trait_description
                classifier._pairs_count = pairs_count

                classifiers.append(classifier)

                logging.info(f"Classifier created with {pairs_count} training pairs ({actual_time:.0f}s)")

            except Exception as e:
                logging.info(f"Error creating classifier for trait '{trait_description}': {e}")
                continue

        logging.info(f"Created {len(classifiers)} synthetic classifiers within budget")

        # Update discovery result to reflect what we actually processed
        final_discovery = TraitDiscoveryResult(traits_discovered=affordable_traits)
        return classifiers, final_discovery

    def apply_classifiers_to_response(
        self, response_text: str, classifiers: List[ActivationClassifier], trait_discovery: TraitDiscoveryResult
    ) -> List[SyntheticClassifierResult]:
        """
        Apply pre-trained synthetic classifiers to a response.

        Args:
            response_text: The response to analyze (only used for activation extraction)
            classifiers: List of trained classifiers to apply
            trait_discovery: Original trait discovery result for context

        Returns:
            List of classification results
        """
        logging.info(f"Applying {len(classifiers)} synthetic classifiers to response...")

        # Extract activations from the response ONCE
        logging.info("Extracting activations from response...")
        try:
            if self.layer is None:
                raise MissingParameterError(params=["layer"], context="apply_classifiers_to_response")
            response_activations, _ = self.model.extract_activations(response_text, layer=self.layer)
        except Exception as e:
            logging.info(f"Error extracting response activations: {e}")
            return []

        results = []

        for i, classifier in enumerate(classifiers):
            trait_description = getattr(classifier, "_trait_description", f"trait_{i}")
            pairs_count = getattr(classifier, "_pairs_count", 0)

            logging.info(f"Applying classifier {i + 1}/{len(classifiers)}: {trait_description}")

            try:
                start_time = time.time()

                # Apply classifier to response activations
                prediction = classifier.predict(response_activations)
                confidence = classifier.predict_proba(response_activations)

                # Handle confidence score (could be array or scalar)
                if hasattr(confidence, "__iter__") and len(confidence) > 1:
                    confidence_score = float(max(confidence))
                else:
                    confidence_score = float(confidence)

                generation_time = time.time() - start_time

                result = SyntheticClassifierResult(
                    trait_description=trait_description,
                    classifier_confidence=confidence_score,
                    prediction=int(prediction),
                    confidence_score=confidence_score,
                    training_pairs_count=pairs_count,
                    generation_time=generation_time,
                )

                results.append(result)

                logging.info(f"Result: prediction={prediction}, confidence={confidence_score:.3f}")

            except Exception as e:
                logging.info(f"Error applying classifier for trait '{trait_description}': {e}")
                continue

        logging.info(f"Applied {len(results)} classifiers successfully")
        return results


