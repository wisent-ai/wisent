"""Public functions for synthetic classifier creation."""
import logging, time
from typing import List, Tuple
from wisent.core.classifier.classifier import ActivationClassifier
from wisent.core.constants import DEFAULT_LAYER, AGENT_SYNTH_MIN_PAIRS, AGENT_SYNTH_PAIRS_PER_TRAIT, AGENT_CLASSIFIER_NUM_PAIRS, TRAIT_LABEL_MAX_LENGTH
from wisent.core.errors import InsufficientDataError, MissingParameterError, ExecutionError
from wisent.core.agent.diagnose.classifiers._synthetic_classes import (
    TraitDiscoveryResult, SyntheticClassifierResult,
    AutomaticTraitDiscovery, SyntheticClassifierFactory)
from wisent.core.agent.diagnose.classifiers._synthetic_system import SyntheticClassifierSystem
from wisent.core import constants as _C
logger = logging.getLogger(__name__)

def get_time_budget_from_manager() -> float:
    """Get time budget from the global budget manager."""
    budget_manager = get_budget_manager()
    time_budget = budget_manager.get_budget(ResourceType.TIME)
    if not time_budget:
        raise MissingParameterError(params=["time_budget"], context="Budget manager")
    return time_budget.remaining_budget / _C.SECONDS_PER_MINUTE  # Convert to minutes


# Main interface functions
def create_synthetic_classifier_system(model) -> SyntheticClassifierSystem:
    """Create a synthetic classifier system instance."""
    return SyntheticClassifierSystem(model)


def create_classifiers_for_prompt(
    model, prompt: str, pairs_per_trait: int = AGENT_SYNTH_PAIRS_PER_TRAIT
) -> Tuple[List[ActivationClassifier], TraitDiscoveryResult]:
    """
    Convenience function to create synthetic classifiers for a prompt.

    Args:
        model: The language model instance
        prompt: Prompt to analyze and create classifiers for
        pairs_per_trait: Number of contrastive pairs per trait

    Returns:
        Tuple of (trained classifiers, trait discovery result)
    """
    time_budget_minutes = get_time_budget_from_manager()
    system = create_synthetic_classifier_system(model)
    return system.create_classifiers_for_prompt(prompt, time_budget_minutes, pairs_per_trait)


def apply_classifiers_to_response(
    model, response_text: str, classifiers: List[ActivationClassifier], trait_discovery: TraitDiscoveryResult
) -> List[SyntheticClassifierResult]:
    """
    Convenience function to apply classifiers to a response.

    Args:
        model: The language model instance
        response_text: Response to analyze
        classifiers: Pre-trained classifiers
        trait_discovery: Original trait discovery result

    Returns:
        List of classification results
    """
    system = create_synthetic_classifier_system(model)
    return system.apply_classifiers_to_response(response_text, classifiers, trait_discovery)


def create_classifier_from_trait_description(
    model, trait_description: str, num_pairs: int = AGENT_CLASSIFIER_NUM_PAIRS
) -> ActivationClassifier:
    """
    Direct function to create a classifier from a trait description.

    Args:
        model: The language model instance
        trait_description: Natural language description of the trait (e.g., "accuracy and truthfulness")
        num_pairs: Number of contrastive pairs to generate for training

    Returns:
        Trained ActivationClassifier
    """
    import datetime

    # Setup logging to file
    log_file = f"synthetic_classifier_debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    def log_and_print(message):
        print(message)
        with open(log_file, "a") as f:
            f.write(f"{datetime.datetime.now().isoformat()}: {message}\n")

    log_and_print(f"🎯 Creating classifier for trait: '{trait_description}'")
    log_and_print(f"📋 Parameters: num_pairs={num_pairs}")

    # Create synthetic contrastive pair generator
    log_and_print("🏭 Creating SyntheticContrastivePairGenerator...")
    pair_generator = SyntheticContrastivePairGenerator(model)
    log_and_print("✅ SyntheticContrastivePairGenerator created successfully")

    # Generate contrastive pairs for this trait
    log_and_print(f"📝 Generating {num_pairs} contrastive pairs...")
    pair_set = pair_generator.generate_contrastive_pair_set(
        trait_description=trait_description,
        num_pairs=num_pairs,
        name=f"synthetic_{trait_description[:TRAIT_LABEL_MAX_LENGTH].replace(' ', '_')}",
    )

    log_and_print(f"✅ Generated {len(pair_set.pairs)} pairs total")

    # Log all generated pairs in detail
    log_and_print("=" * _C.REPORT_LINE_WIDTH)
    log_and_print("DETAILED PAIR ANALYSIS:")
    log_and_print("=" * _C.REPORT_LINE_WIDTH)

    for i, pair in enumerate(pair_set.pairs):
        log_and_print(f"\n--- PAIR {i + 1}/{len(pair_set.pairs)} ---")
        log_and_print(f"Prompt: {pair.prompt!r}")
        log_and_print(f"Positive Response: {pair.positive_response.text!r}")
        log_and_print(f"Negative Response: {pair.negative_response.text!r}")
        log_and_print(f"Positive Response Type: {type(pair.positive_response)}")
        log_and_print(f"Negative Response Type: {type(pair.negative_response)}")
        log_and_print(
            f"Positive Response Length: {len(pair.positive_response.text) if hasattr(pair.positive_response, 'text') else 'N/A'}"
        )
        log_and_print(
            f"Negative Response Length: {len(pair.negative_response.text) if hasattr(pair.negative_response, 'text') else 'N/A'}"
        )

        # Check for any special attributes
        if hasattr(pair, "_prompt_pair"):
            log_and_print(f"Has _prompt_pair: {pair._prompt_pair}")
        if hasattr(pair, "_prompt_strategy"):
            log_and_print(f"Has _prompt_strategy: {pair._prompt_strategy}")

    log_and_print("=" * _C.REPORT_LINE_WIDTH)

    if len(pair_set.pairs) < AGENT_SYNTH_MIN_PAIRS:
        error_msg = f"Insufficient training pairs generated: {len(pair_set.pairs)}"
        log_and_print(f"❌ ERROR: {error_msg}")
        raise InsufficientDataError(reason="training pairs", required=AGENT_SYNTH_MIN_PAIRS, actual=len(pair_set.pairs))

    # Extract activations for training
    positive_activations = []
    negative_activations = []

    log_and_print(f"🧠 Extracting activations from {len(pair_set.pairs)} pairs...")

    # Create Layer object for activation extraction
    from wisent.core.layer import Layer

    layer_obj = Layer(index=DEFAULT_LAYER, type="transformer")
    log_and_print(f"🔧 Created Layer object: index={layer_obj.index}, type={layer_obj.type}")

    for i, pair in enumerate(pair_set.pairs):
        log_and_print(f"\n🔍 Processing pair {i + 1}/{len(pair_set.pairs)}...")
        try:
            # Get activations for positive response
            log_and_print(f"   📊 Extracting positive activations for: {pair.positive_response.text[:_C.DISPLAY_TRUNCATION_COMPACT]!r}")
            pos_activations = model.extract_activations(pair.positive_response.text, layer_obj)
            log_and_print(
                f"   ✅ Positive activations shape: {pos_activations.shape if hasattr(pos_activations, 'shape') else 'N/A'}"
            )
            positive_activations.append(pos_activations)

            # Get activations for negative response
            log_and_print(f"   📊 Extracting negative activations for: {pair.negative_response.text[:_C.DISPLAY_TRUNCATION_COMPACT]!r}")
            neg_activations = model.extract_activations(pair.negative_response.text, layer_obj)
            log_and_print(
                f"   ✅ Negative activations shape: {neg_activations.shape if hasattr(neg_activations, 'shape') else 'N/A'}"
            )
            negative_activations.append(neg_activations)

            log_and_print(f"   ✅ Successfully processed pair {i + 1}")

        except Exception as e:
            log_and_print(f"   ⚠️ Error extracting activations for pair {i + 1}: {e}")
            import traceback

            error_details = traceback.format_exc()
            log_and_print(f"   📜 Full error traceback:\n{error_details}")
            continue

    log_and_print("\n📊 ACTIVATION EXTRACTION SUMMARY:")
    log_and_print(f"   Positive activations collected: {len(positive_activations)}")
    log_and_print(f"   Negative activations collected: {len(negative_activations)}")
    log_and_print(f"   Total pairs processed: {len(pair_set.pairs)}")
    log_and_print(f"   Success rate: {(len(positive_activations) / len(pair_set.pairs) * 100):.1f}%")

    if len(positive_activations) < 2 or len(negative_activations) < 2:
        error_msg = f"Insufficient activation data for training: {len(positive_activations)} positive, {len(negative_activations)} negative"
        log_and_print(f"❌ ERROR: {error_msg}")
        raise InsufficientDataError(reason=error_msg)

    # Train classifier on activations
    log_and_print(
        f"🏋️ Training classifier on {len(positive_activations)} positive, {len(negative_activations)} negative activations..."
    )

    log_and_print("🔧 Creating ActivationClassifier instance...")
    classifier = ActivationClassifier()
    log_and_print("✅ ActivationClassifier created")

    log_and_print("🎯 Starting classifier training...")
    try:
        # Convert activations to the format expected by train_on_activations method
        from wisent.core.activations import Activations

        # Convert torch tensors to Activations objects if needed
        harmful_activations = []
        harmless_activations = []

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

        log_and_print(
            f"🔧 Converted to Activations objects: {len(harmful_activations)} harmful, {len(harmless_activations)} harmless"
        )

        # Train using the correct method
        training_result = classifier.train_on_activations(harmful_activations, harmless_activations)
        log_and_print(f"✅ Classifier training completed successfully! Result: {training_result}")
    except Exception as e:
        log_and_print(f"❌ ERROR during classifier training: {e}")
        import traceback

        error_details = traceback.format_exc()
        log_and_print(f"📜 Full training error traceback:\n{error_details}")
        raise

    # Store metadata
    classifier._trait_description = trait_description
    classifier._pairs_count = len(pair_set.pairs)
    log_and_print(f"📝 Stored metadata: trait='{trait_description}', pairs_count={len(pair_set.pairs)}")

    log_and_print(f"🎉 Classifier creation completed successfully! Debug log saved to: {log_file}")

    return classifier


def evaluate_response_with_trait_classifier(
    model, response_text: str, trait_classifier: ActivationClassifier
) -> SyntheticClassifierResult:
    """
    Evaluate a response using a trait-specific classifier.

    Args:
        model: The language model instance
        response_text: Response to analyze
        trait_classifier: Pre-trained classifier for a specific trait

    Returns:
        Classification result
    """
    trait_description = getattr(trait_classifier, "_trait_description", "unknown_trait")
    pairs_count = getattr(trait_classifier, "_pairs_count", 0)

    logging.info(f"Evaluating response with '{trait_description}' classifier...")

    # Extract activations from response
    try:
        response_activations, _ = model.extract_activations(response_text, layer=DEFAULT_LAYER)
    except Exception as e:
        raise ExecutionError(reason=f"Error extracting response activations: {e}", cause=e)

    # Apply classifier
    start_time = time.time()
    prediction = trait_classifier.predict(response_activations)
    confidence = trait_classifier.predict_proba(response_activations)

    # Handle confidence score
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

    logging.info(f"Result: prediction={prediction}, confidence={confidence_score:.3f}")
    return result
