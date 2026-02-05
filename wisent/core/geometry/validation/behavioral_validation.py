"""Behavioral validation for steering - Step 5 of RepScan protocol.

Actually tests if steering changes behavior, not just geometry.
Diagnoses whether activation movement translates to behavioral change.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch


@dataclass
class BehavioralValidationResult:
    """Result of behavioral validation."""
    diagnosis: str  # EFFECTIVE, IMPROPERLY_IDENTIFIED, INEFFECTIVE
    activations_moved: bool
    behavior_improved: bool
    activation_movement_rate: float
    base_success_rate: float
    steered_success_rate: float
    behavior_delta: float
    confidence: float
    details: Dict[str, Any]


def compute_activation_movement(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    base_activations: torch.Tensor,
    steered_activations: torch.Tensor,
) -> Dict[str, float]:
    """Measure if steered activations moved toward positive region."""
    pos = pos_activations.float().cpu().numpy()
    neg = neg_activations.float().cpu().numpy()
    base = base_activations.float().cpu().numpy()
    steered = steered_activations.float().cpu().numpy()

    pos_centroid = pos.mean(axis=0)
    neg_centroid = neg.mean(axis=0)

    base_to_pos = np.linalg.norm(base - pos_centroid, axis=1)
    base_to_neg = np.linalg.norm(base - neg_centroid, axis=1)
    steered_to_pos = np.linalg.norm(steered - pos_centroid, axis=1)
    steered_to_neg = np.linalg.norm(steered - neg_centroid, axis=1)

    moved_toward_pos = np.sum(steered_to_pos < base_to_pos)
    moved_away_from_neg = np.sum(steered_to_neg > base_to_neg)
    total = len(base)

    # Train simple classifier to check region membership
    from sklearn.linear_model import LogisticRegression
    X_train = np.vstack([pos, neg])
    y_train = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    clf = LogisticRegression( random_state=42)
    clf.fit(X_train, y_train)

    base_probs = clf.predict_proba(base)[:, 1]
    steered_probs = clf.predict_proba(steered)[:, 1]

    base_in_pos_region = np.sum(base_probs >= 0.5)
    steered_in_pos_region = np.sum(steered_probs >= 0.5)

    return {
        "moved_toward_pos_count": int(moved_toward_pos),
        "moved_toward_pos_rate": moved_toward_pos / total,
        "moved_away_from_neg_rate": moved_away_from_neg / total,
        "base_in_pos_region": int(base_in_pos_region),
        "steered_in_pos_region": int(steered_in_pos_region),
        "region_shift": (steered_in_pos_region - base_in_pos_region) / total,
        "base_mean_prob": float(base_probs.mean()),
        "steered_mean_prob": float(steered_probs.mean()),
        "prob_shift": float(steered_probs.mean() - base_probs.mean()),
        "total": total,
    }


def compute_behavioral_change(
    base_evaluations: List[str],
    steered_evaluations: List[str],
    positive_label: str = "TRUTHFUL",
) -> Dict[str, float]:
    """Measure if behavior actually improved."""
    base_success = sum(1 for e in base_evaluations if e == positive_label)
    steered_success = sum(1 for e in steered_evaluations if e == positive_label)
    total = len(base_evaluations)

    base_rate = base_success / total if total > 0 else 0
    steered_rate = steered_success / total if total > 0 else 0
    delta = steered_rate - base_rate

    # Statistical significance via McNemar's test approximation
    improved = sum(1 for b, s in zip(base_evaluations, steered_evaluations)
                   if b != positive_label and s == positive_label)
    worsened = sum(1 for b, s in zip(base_evaluations, steered_evaluations)
                   if b == positive_label and s != positive_label)

    return {
        "base_success_count": base_success,
        "steered_success_count": steered_success,
        "base_success_rate": base_rate,
        "steered_success_rate": steered_rate,
        "delta": delta,
        "improved_count": improved,
        "worsened_count": worsened,
        "net_change": improved - worsened,
        "total": total,
    }


def validate_steering_behavioral(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    base_activations: torch.Tensor,
    steered_activations: torch.Tensor,
    base_evaluations: List[str],
    steered_evaluations: List[str],
    positive_label: str = "TRUTHFUL",
    movement_threshold: float = 0.6,
    behavior_threshold: float = 0.0,
) -> BehavioralValidationResult:
    """
    Validate if steering actually works by comparing activation movement vs behavior.

    Args:
        pos_activations: Positive examples from contrastive pairs
        neg_activations: Negative examples from contrastive pairs
        base_activations: Activations from unsteered model on test prompts
        steered_activations: Activations from steered model on test prompts
        base_evaluations: Evaluation labels for unsteered outputs
        steered_evaluations: Evaluation labels for steered outputs
        positive_label: The label indicating success (e.g., "TRUTHFUL")
        movement_threshold: Min rate of samples moving toward positive
        behavior_threshold: Min delta in success rate to consider improved

    Returns:
        BehavioralValidationResult with diagnosis
    """
    activation_metrics = compute_activation_movement(
        pos_activations, neg_activations, base_activations, steered_activations
    )
    behavior_metrics = compute_behavioral_change(
        base_evaluations, steered_evaluations, positive_label
    )

    activations_moved = activation_metrics["moved_toward_pos_rate"] >= movement_threshold
    behavior_improved = behavior_metrics["delta"] > behavior_threshold

    # Diagnosis
    if activations_moved and behavior_improved:
        diagnosis = "EFFECTIVE"
        confidence = min(activation_metrics["moved_toward_pos_rate"],
                        0.5 + behavior_metrics["delta"])
    elif activations_moved and not behavior_improved:
        diagnosis = "IMPROPERLY_IDENTIFIED"
        confidence = activation_metrics["moved_toward_pos_rate"] * 0.5
    elif not activations_moved and behavior_improved:
        diagnosis = "UNEXPECTED_IMPROVEMENT"
        confidence = 0.3
    else:
        diagnosis = "INEFFECTIVE"
        confidence = 0.1

    return BehavioralValidationResult(
        diagnosis=diagnosis,
        activations_moved=activations_moved,
        behavior_improved=behavior_improved,
        activation_movement_rate=activation_metrics["moved_toward_pos_rate"],
        base_success_rate=behavior_metrics["base_success_rate"],
        steered_success_rate=behavior_metrics["steered_success_rate"],
        behavior_delta=behavior_metrics["delta"],
        confidence=confidence,
        details={
            "activation_metrics": activation_metrics,
            "behavior_metrics": behavior_metrics,
        },
    )


def run_behavioral_validation(
    adapter,
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    steering_vector: torch.Tensor,
    test_prompts: List[str],
    evaluator,
    layer_name: str,
    strength: float = 1.0,
    max_new_tokens: int = 100,
    positive_label: str = "TRUTHFUL",
    extraction_strategy: str = "chat_last",
) -> BehavioralValidationResult:
    """
    Full behavioral validation: generate outputs, extract activations FROM RESPONSE, evaluate.

    IMPORTANT: Extracts activations from the GENERATED RESPONSE using the same
    extraction strategy that was used for the contrastive pair activations.

    Args:
        adapter: Model adapter with steering capability
        pos_activations: Positive contrastive pair activations
        neg_activations: Negative contrastive pair activations
        steering_vector: The steering vector to apply
        test_prompts: Prompts to test on
        evaluator: Evaluator to judge outputs
        layer_name: Layer to steer (e.g., "layer.8")
        strength: Steering strength
        max_new_tokens: Max tokens to generate
        positive_label: Success label
        extraction_strategy: Same strategy used for contrastive pairs (e.g., "chat_last", "chat_mean")

    Returns:
        BehavioralValidationResult
    """
    from wisent.core.activations.core.atoms import LayerActivations
    from wisent.core.adapters.base import SteeringConfig
    from wisent.core.activations import ExtractionStrategy, extract_activation

    strategy = ExtractionStrategy(extraction_strategy)

    base_acts, steered_acts = [], []
    base_evals, steered_evals = [], []

    steering_vectors = LayerActivations({layer_name: steering_vector})
    config = SteeringConfig(scale={layer_name: strength})

    for prompt in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted = adapter.apply_chat_template(messages, add_generation_prompt=True)

        # Generate outputs FIRST
        base_full_response = adapter._generate_unsteered(
            formatted, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=True
        )
        steered_full_response = adapter.forward_with_steering(
            formatted, steering_vectors=steering_vectors, config=config
        )

        # Extract just the response part for evaluation
        base_response = base_full_response
        steered_response = steered_full_response
        if formatted in base_response:
            base_response = base_response[len(formatted):]
        if formatted in steered_response:
            steered_response = steered_response[len(formatted):]

        # Extract activations from the FULL RESPONSE using the SAME extraction strategy
        base_layer_acts = adapter.extract_activations(base_full_response, layers=[layer_name])
        steered_layer_acts = adapter.extract_activations(steered_full_response, layers=[layer_name])

        base_act = base_layer_acts.get(layer_name)
        steered_act = steered_layer_acts.get(layer_name)

        if base_act is not None and steered_act is not None:
            # Use the same extraction strategy as the contrastive pairs
            prompt_len = len(adapter.tokenizer(formatted, add_special_tokens=False)["input_ids"])
            base_extracted = extract_activation(
                strategy, base_act[0], base_response, adapter.tokenizer, prompt_len
            )
            steered_extracted = extract_activation(
                strategy, steered_act[0], steered_response, adapter.tokenizer, prompt_len
            )
            base_acts.append(base_extracted.cpu())
            steered_acts.append(steered_extracted.cpu())

        # Evaluate
        base_result = evaluator.evaluate(base_response, "", correct_answers=[], incorrect_answers=[])
        steered_result = evaluator.evaluate(steered_response, "", correct_answers=[], incorrect_answers=[])

        base_evals.append(base_result.ground_truth)
        steered_evals.append(steered_result.ground_truth)

    if not base_acts:
        return BehavioralValidationResult(
            diagnosis="NO_ACTIVATIONS",
            activations_moved=False,
            behavior_improved=False,
            activation_movement_rate=0.0,
            base_success_rate=0.0,
            steered_success_rate=0.0,
            behavior_delta=0.0,
            confidence=0.0,
            details={"error": "No activations extracted"},
        )

    base_activations = torch.stack(base_acts)
    steered_activations = torch.stack(steered_acts)

    return validate_steering_behavioral(
        pos_activations, neg_activations,
        base_activations, steered_activations,
        base_evals, steered_evals,
        positive_label=positive_label,
    )
