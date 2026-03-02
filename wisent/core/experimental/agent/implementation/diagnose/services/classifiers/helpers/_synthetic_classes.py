"""
Synthetic Classifier Option System

Creates custom classifiers from automatically discovered traits using synthetic contrastive pairs.
The model analyzes prompts to determine relevant traits for responses, then creates classifiers for those traits.
The actual response is NEVER analyzed as text - only its activations are classified.
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Tuple

from wisent.core.reading.classifiers.core.atoms import ActivationClassifier
from wisent.core.utils.config_tools.constants import DEFAULT_LAYER, AGENT_SYNTH_MIN_PAIRS, AGENT_SYNTH_DEFAULT_PAIRS, DISPLAY_TRUNCATION_MEDIUM, DISPLAY_TRUNCATION_COMPACT, TRAIT_LABEL_MAX_LENGTH
from wisent.core.primitives.models.config import get_generate_kwargs
from wisent.core.utils.infra_tools.errors import InsufficientDataError, MissingParameterError, ExecutionError

from ....core.agent.budget import ResourceType, calculate_max_tasks_for_time_budget, get_budget_manager
from ....core.contrastive_pairs.generate_synthetically import SyntheticContrastivePairGenerator


@dataclass
class TraitDiscoveryResult:
    """Result of automatic trait discovery."""

    traits_discovered: List[str]


@dataclass
class SyntheticClassifierResult:
    """Result of synthetic classifier creation and diagnosis."""

    trait_description: str
    classifier_confidence: float
    prediction: int
    confidence_score: float
    training_pairs_count: int
    generation_time: float


class AutomaticTraitDiscovery:
    """Automatically discovers relevant traits for prompt response analysis."""

    def __init__(self, model):
        self.model = model

    def discover_relevant_traits(self, prompt: str, time_budget_minutes: float) -> TraitDiscoveryResult:
        """
        Analyze a prompt to automatically discover relevant quality traits for responses.

        Args:
            prompt: The prompt/question to analyze for trait discovery
            time_budget_minutes: Time budget for classifier creation in minutes

        Returns:
            TraitDiscoveryResult with discovered traits
        """
        # Calculate max traits based on time budget
        max_traits = calculate_max_tasks_for_time_budget("classifier_training", time_budget_minutes)
        max_traits = max(1, min(max_traits, 5))  # Cap between 1-5 traits
        logging.info(f"Budget system: {time_budget_minutes:.1f} min budget → max {max_traits} traits")

        # Generate dynamic trait prompt based on budget
        trait_lines = "\n".join([f"TRAIT_{i + 1}:" for i in range(max_traits)])

        discovery_prompt = f"""USER PROMPT: {prompt}

List {max_traits} quality traits for responses:
{trait_lines}"""

        try:
            analysis, _ = self.model.generate(
                discovery_prompt, layer_index=DEFAULT_LAYER
            )

            logging.info(f"Model generated analysis: {analysis[:DISPLAY_TRUNCATION_MEDIUM]}...")
            return self._parse_discovery_result(analysis)

        except Exception as e:
            logging.info(f"Error in trait discovery: {e}")
            # Fallback to general traits
            return TraitDiscoveryResult(traits_discovered=["accuracy and truthfulness", "helpfulness", "safety"])

    def _parse_discovery_result(self, analysis: str) -> TraitDiscoveryResult:
        """Parse the model's trait discovery response."""
        traits = []

        lines = analysis.split("\n")

        for line in lines:
            line = line.strip()

            if line.startswith("TRAIT_"):
                # Extract trait description
                if ":" in line:
                    trait = line.split(":", 1)[1].strip()
                    if len(trait) > 3:
                        traits.append(trait)

        return TraitDiscoveryResult(traits_discovered=traits)


class SyntheticClassifierFactory:
    """Creates custom classifiers from trait descriptions using synthetic contrastive pairs."""

    def __init__(self, model):
        self.model = model
        self.pair_generator = SyntheticContrastivePairGenerator(model)

    def create_classifier_from_trait(
        self, trait_description: str, num_pairs: int = AGENT_SYNTH_DEFAULT_PAIRS
    ) -> Tuple[ActivationClassifier, int]:
        """
        Create a classifier for a specific trait using synthetic contrastive pairs.

        Args:
            trait_description: Natural language description of the trait
            num_pairs: Number of contrastive pairs to generate

        Returns:
            Tuple of (trained classifier, number of training pairs)
        """
        try:
            # Generate synthetic contrastive pairs for this trait
            pair_set = self.pair_generator.generate_contrastive_pair_set(
                trait_description=trait_description,
                num_pairs=num_pairs,
                name=f"synthetic_{trait_description[:TRAIT_LABEL_MAX_LENGTH].replace(' ', '_')}",
            )

            if len(pair_set.pairs) < AGENT_SYNTH_MIN_PAIRS:
                raise InsufficientDataError(reason="training pairs", required=AGENT_SYNTH_MIN_PAIRS, actual=len(pair_set.pairs))

            # Extract activations for training
            positive_activations = []
            negative_activations = []

            logging.info(f"Extracting activations from {len(pair_set.pairs)} pairs...")

            # Create Layer object for activation extraction
            from wisent.core.primitives.models.core.layer import Layer

            layer_obj = Layer(index=DEFAULT_LAYER, type="transformer")
            logging.info(f"Created Layer object: index={layer_obj.index}, type={layer_obj.type}")

            for i, pair in enumerate(pair_set.pairs):
                logging.debug(f"Processing pair {i + 1}/{len(pair_set.pairs)}...")
                try:
                    # Get activations for positive response
                    logging.debug(f"Extracting positive activations for: {pair.positive_response.text[:DISPLAY_TRUNCATION_COMPACT]!r}")
                    pos_activations = self.model.extract_activations(pair.positive_response.text, layer_obj)
                    logging.debug(
                        f"Positive activations shape: {pos_activations.shape if hasattr(pos_activations, 'shape') else 'N/A'}"
                    )
                    positive_activations.append(pos_activations)

                    # Get activations for negative response
                    logging.debug(f"Extracting negative activations for: {pair.negative_response.text[:DISPLAY_TRUNCATION_COMPACT]!r}")
                    neg_activations = self.model.extract_activations(pair.negative_response.text, layer_obj)
                    logging.debug(
                        f"Negative activations shape: {neg_activations.shape if hasattr(neg_activations, 'shape') else 'N/A'}"
                    )
                    negative_activations.append(neg_activations)

                    logging.debug(f"Successfully processed pair {i + 1}")

                except Exception as e:
                    logging.debug(f"Error extracting activations for pair {i + 1}: {e}")
                    import traceback

                    error_details = traceback.format_exc()
                    logging.debug(f"Full error traceback:\n{error_details}")
                    continue

            logging.info("ACTIVATION EXTRACTION SUMMARY:")
            logging.info(f"Positive activations collected: {len(positive_activations)}")
            logging.info(f"Negative activations collected: {len(negative_activations)}")
            logging.info(f"Total pairs processed: {len(pair_set.pairs)}")
            logging.info(f"Success rate: {(len(positive_activations) / len(pair_set.pairs) * 100):.1f}%")

            if len(positive_activations) < 2 or len(negative_activations) < 2:
                error_msg = f"Insufficient activation data for training: {len(positive_activations)} positive, {len(negative_activations)} negative"
                logging.info(f"ERROR: {error_msg}")
                raise InsufficientDataError(reason=error_msg)

            # Train classifier on activations
            logging.info(
                f"Training classifier on {len(positive_activations)} positive, {len(negative_activations)} negative activations..."
            )

            logging.info("Creating ActivationClassifier instance...")
            classifier = ActivationClassifier()
            logging.info("ActivationClassifier created")

            logging.info("Starting classifier training...")
            try:
                # Convert activations to the format expected by train_on_activations method
                from wisent.core.primitives.model_interface.core.activations.activations import Activations

                # Convert torch tensors to Activations objects if needed
                harmful_activations = []
                harmless_activations = []

                from wisent.core.primitives.models.core.layer import Layer

                layer_obj = Layer(index=DEFAULT_LAYER, type="transformer")

                for pos_act in positive_activations:
                    if hasattr(pos_act, "shape"):  # It's a torch tensor
                        # Create Activations object from tensor
                        act_obj = Activations(pos_act, layer_obj)
                        harmful_activations.append(act_obj)
                    else:
                        harmful_activations.append(pos_act)

                for neg_act in negative_activations:
                    if hasattr(neg_act, "shape"):  # It's a torch tensor
                        # Create Activations object from tensor
                        act_obj = Activations(neg_act, layer_obj)
                        harmless_activations.append(act_obj)
                    else:
                        harmless_activations.append(neg_act)

                logging.info(
                    f"Converted to Activations objects: {len(harmful_activations)} harmful, {len(harmless_activations)} harmless"
                )

                # Train using the correct method
                training_result = classifier.train_on_activations(harmful_activations, harmless_activations)
                logging.info(f"Classifier training completed successfully! Result: {training_result}")
            except Exception as e:
                logging.info(f"ERROR during classifier training: {e}")
                import traceback

                error_details = traceback.format_exc()
                logging.info(f"Full training error traceback:\n{error_details}")
                raise

            return classifier, len(pair_set.pairs)

        except Exception as e:
            logging.info(f"Error creating classifier for trait '{trait_description}': {e}")
            raise


