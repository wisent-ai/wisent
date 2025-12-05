"""
PersonalizationEvaluator: Evaluates the effectiveness of steering control vectors.

This evaluator assesses steering on three key criteria:
1. Difference: Is the steered response different from baseline?
2. Quality: Is the response coherent (not lobotomized/repetitive)?
3. Alignment: Does the response match the intended trait direction?
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
import numpy as np

from wisent.core.cli_logger import setup_logger, bind
from wisent.core.models.inference_config import get_generate_kwargs
from wisent.core.evaluators.personalization import (
    evaluate_alignment,
    evaluate_quality,
    evaluate_difference,
)
from wisent.core.evaluators.personalization.alignment import estimate_alignment

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

__all__ = ["PersonalizationEvaluator", "SteeringEvaluationResult"]

_LOG = setup_logger(__name__)


class SteeringEvaluationResult:
    """Results from evaluating a steering control vector."""

    def __init__(
        self,
        trait_name: str,
        difference_score: float,
        quality_score: float,
        alignment_score: float,
        baseline_response: str,
        steered_response: str,
        metadata: dict[str, Any] | None = None,
    ):
        self.trait_name = trait_name
        self.difference_score = difference_score  # 1-100, higher = more different
        self.quality_score = quality_score  # 1-100, higher = better quality
        self.alignment_score = alignment_score  # 1-100, higher = better alignment
        self.baseline_response = baseline_response
        self.steered_response = steered_response
        self.metadata = metadata or {}

    @property
    def overall_score(self) -> float:
        """
        Weighted average of all scores.
        Returns 0 if difference_score < 70.
        """
        # If difference score is too low, steering is ineffective
        if self.difference_score < 70:
            return 0.0

        # Weight: difference=0.2, quality=0.3, alignment=0.5
        return (
            0.2 * self.difference_score
            + 0.3 * self.quality_score
            + 0.5 * self.alignment_score
        )

    def __repr__(self) -> str:
        return (
            f"SteeringEvaluationResult(trait={self.trait_name}, "
            f"difference={self.difference_score:.3f}, "
            f"quality={self.quality_score:.3f}, "
            f"alignment={self.alignment_score:.3f}, "
            f"overall={self.overall_score:.3f})"
        )


class PersonalizationEvaluator:
    """Evaluates steering control vectors on difference, quality, and alignment."""

    def __init__(
        self,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        device: str | torch.device = "cuda",
    ):
        """
        Initialize the evaluator.

        Args:
            model: The language model to evaluate (optional, only needed for evaluate_steering)
            tokenizer: Tokenizer for the model (optional, only needed for evaluate_steering)
            device: Device to run evaluation on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device) if isinstance(device, str) else device
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def evaluate_steering(
        self,
        control_vector: torch.Tensor,
        trait_name: str,
        trait_description: str,
        test_prompts: list[str],
        steering_strength: float = 1.0,
        max_new_tokens: int = 100,
    ) -> SteeringEvaluationResult:
        """
        Evaluate a control vector's effectiveness.

        Args:
            control_vector: The steering vector to evaluate
            trait_name: Name of the trait (e.g., "British", "Mean", "Evil")
            trait_description: Description of what the trait means
            test_prompts: List of prompts to test steering on
            steering_strength: Strength multiplier for the control vector
            max_new_tokens: Maximum tokens to generate

        Returns:
            SteeringEvaluationResult with scores and responses
        """
        log = bind(_LOG, trait=trait_name, num_prompts=len(test_prompts))
        log.info("Starting steering evaluation")

        # Generate baseline and steered responses
        baseline_responses = []
        steered_responses = []

        for prompt in test_prompts:
            baseline = self._generate_response(prompt, None, max_new_tokens)
            steered = self._generate_response(
                prompt, control_vector * steering_strength, max_new_tokens
            )
            baseline_responses.append(baseline)
            steered_responses.append(steered)

        # Evaluate on three criteria using the modular evaluation functions
        difference_scores = []
        quality_scores = []
        alignment_scores = []

        for baseline, steered in zip(baseline_responses, steered_responses):
            diff = evaluate_difference(baseline, steered, self.model, self.tokenizer, self.device)
            qual = evaluate_quality(steered, self.model, self.tokenizer, self.device)
            align = evaluate_alignment(steered, trait_name, trait_description, self.model, self.tokenizer, self.device)
            difference_scores.append(diff)
            quality_scores.append(qual)
            alignment_scores.append(align)

        difference_score = float(np.mean(difference_scores))
        quality_score = float(np.mean(quality_scores))
        alignment_score = float(np.mean(alignment_scores))

        log.info(
            "Evaluation complete",
            extra={
                "difference": difference_score,
                "quality": quality_score,
                "alignment": alignment_score,
            },
        )

        return SteeringEvaluationResult(
            trait_name=trait_name,
            difference_score=difference_score,
            quality_score=quality_score,
            alignment_score=alignment_score,
            baseline_response="\n\n".join(baseline_responses[:3]),  # Sample
            steered_response="\n\n".join(steered_responses[:3]),  # Sample
            metadata={
                "num_prompts": len(test_prompts),
                "steering_strength": steering_strength,
                "max_new_tokens": max_new_tokens,
            },
        )

    def _generate_response(
        self,
        prompt: str,
        control_vector: torch.Tensor | None,
        max_new_tokens: int,
    ) -> str:
        """Generate a response with optional steering."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            if control_vector is not None:
                # Apply steering by adding control vector to activations
                # This is a simplified version - real implementation would hook into model layers
                outputs = self.model.generate(
                    **inputs,
                    **get_generate_kwargs(max_new_tokens=max_new_tokens),
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    **get_generate_kwargs(max_new_tokens=max_new_tokens),
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the response
        response = response[len(prompt) :].strip()
        return response

    def evaluate_response_pair(
        self,
        baseline_response: str,
        steered_response: str,
        trait_name: str,
        trait_description: str,
        positive_examples: list[str],
        negative_examples: list[str],
    ) -> SteeringEvaluationResult:
        """
        Evaluate a pair of pre-generated responses (baseline vs steered).

        This method evaluates the effectiveness of steering by comparing
        baseline and steered responses.

        Args:
            baseline_response: The baseline model response
            steered_response: The steered model response
            trait_name: Name of the trait (e.g., "British", "Mean")
            trait_description: Description of what the trait means
            positive_examples: List of positive example responses for alignment
            negative_examples: List of negative example responses for alignment

        Returns:
            SteeringEvaluationResult with scores and analysis
        """
        log = bind(_LOG, trait=trait_name)
        log.info("Evaluating response pair")

        # Evaluate on three criteria using the modular evaluation functions
        difference_score = evaluate_difference(
            baseline_response, steered_response, self.model, self.tokenizer, self.device
        )
        quality_score = evaluate_quality(
            steered_response, self.model, self.tokenizer, self.device
        )
        alignment_score = evaluate_alignment(
            steered_response, trait_name, trait_description,
            self.model, self.tokenizer, self.device, positive_examples, negative_examples
        )

        log.info(
            "Evaluation complete",
            extra={
                "difference": difference_score,
                "quality": quality_score,
                "alignment": alignment_score,
            },
        )

        return SteeringEvaluationResult(
            trait_name=trait_name,
            difference_score=difference_score,
            quality_score=quality_score,
            alignment_score=alignment_score,
            baseline_response=baseline_response,
            steered_response=steered_response,
            metadata={
                "evaluation_mode": "response_pair",
            },
        )

    @staticmethod
    def estimate_alignment(
        responses: list[str],
        trait_description: str,
        positive_examples: list[str],
        negative_examples: list[str],
    ) -> float:
        """
        Estimate trait alignment using contrastive embedding similarity.

        Computes how much closer response embeddings are to positive examples
        versus negative examples.

        Args:
            responses: List of model responses to evaluate
            trait_description: Description of the trait (unused, kept for API compatibility)
            positive_examples: List of positive example responses for the trait
            negative_examples: List of negative example responses for the trait

        Returns:
            Float score between 0 and 1 indicating alignment with trait
        """
        return estimate_alignment(responses, trait_description, positive_examples, negative_examples)
