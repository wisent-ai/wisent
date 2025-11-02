"""
PersonalizationEvaluator: Evaluates the effectiveness of steering control vectors.

This evaluator assesses steering on three key criteria:
1. Difference: Is the steered response different from baseline?
2. Quality: Is the response coherent (not lobotomized/repetitive)?
3. Alignment: Does the response match the intended trait direction?
"""

from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING
from collections import Counter

import torch
import numpy as np

from wisent.core.cli_logger import setup_logger, bind

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
        self.difference_score = difference_score  # 0-1, higher = more different
        self.quality_score = quality_score  # 0-1, higher = better quality
        self.alignment_score = alignment_score  # 0-1, higher = better alignment
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

        # Evaluate on three criteria
        difference_score = self._evaluate_difference(
            baseline_responses, steered_responses
        )
        quality_score = self._evaluate_quality(steered_responses)
        alignment_score = self._evaluate_alignment(
            steered_responses, trait_name, trait_description
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
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the response
        response = response[len(prompt) :].strip()
        return response

    def _evaluate_difference(
        self, baseline_responses: list[str], steered_responses: list[str]
    ) -> float:
        """
        Evaluate how different steered responses are from baseline.

        Returns score between 0 (identical) and 1 (very different).
        """
        if not baseline_responses or not steered_responses:
            return 0.0

        differences = []
        for baseline, steered in zip(baseline_responses, steered_responses):
            # Calculate token-level Jaccard distance
            baseline_tokens = set(baseline.lower().split())
            steered_tokens = set(steered.lower().split())

            if not baseline_tokens and not steered_tokens:
                diff = 0.0
            elif not baseline_tokens or not steered_tokens:
                diff = 1.0
            else:
                intersection = len(baseline_tokens & steered_tokens)
                union = len(baseline_tokens | steered_tokens)
                jaccard_similarity = intersection / union if union > 0 else 0.0
                diff = 1.0 - jaccard_similarity

            differences.append(diff)

        avg_difference = np.mean(differences)
        return float(avg_difference)

    def _evaluate_quality(self, responses: list[str]) -> float:
        """
        Evaluate if responses are coherent (not lobotomized).

        Checks for:
        - Repetitive tokens
        - Nonsensical patterns
        - Too short/empty responses
        - Repeated phrases

        Returns score between 0 (poor quality) and 1 (good quality).
        """
        if not responses:
            return 0.0

        quality_scores = []

        for response in responses:
            score = 1.0  # Start with perfect score

            # Check 1: Empty or too short
            if len(response.strip()) < 10:
                score *= 0.1
                quality_scores.append(score)
                continue

            tokens = response.lower().split()

            # Check 2: Repetitive tokens
            if len(tokens) > 0:
                token_counts = Counter(tokens)
                most_common_count = token_counts.most_common(1)[0][1]
                repetition_ratio = most_common_count / len(tokens)

                # Penalize if any token appears more than 30% of the time
                if repetition_ratio > 0.3:
                    score *= 1.0 - (repetition_ratio - 0.3)

            # Check 3: Repeated n-grams (phrases)
            bigrams = [
                f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)
            ]
            if bigrams:
                bigram_counts = Counter(bigrams)
                most_common_bigram_count = bigram_counts.most_common(1)[0][1]
                if most_common_bigram_count > 2:  # Same phrase repeated 3+ times
                    score *= 0.5

            # Check 4: Nonsensical patterns (too many special chars)
            special_char_ratio = len(re.findall(r"[^a-zA-Z0-9\s.,!?']", response)) / max(
                len(response), 1
            )
            if special_char_ratio > 0.2:
                score *= 0.7

            # Check 5: Single repeated character
            if re.search(r"(.)\1{10,}", response):  # Same char 10+ times in a row
                score *= 0.3

            quality_scores.append(max(0.0, score))

        avg_quality = np.mean(quality_scores)
        return float(avg_quality)

    def _evaluate_alignment(
        self, responses: list[str], trait_name: str, trait_description: str
    ) -> float:
        """
        Evaluate if responses align with the intended trait.

        Uses a simple LLM-as-a-judge approach with the same model.

        Returns score between 0 (not aligned) and 1 (well aligned).
        """
        if not responses:
            return 0.0

        alignment_scores = []

        for response in responses:
            # Create evaluation prompt
            eval_prompt = f"""Rate how strongly this response demonstrates the trait "{trait_name}" ({trait_description}).

Response: {response}

Does this response exhibit "{trait_name}" characteristics? Rate from 0 (not at all) to 10 (strongly).

Rating (0-10):"""

            # Get rating from model
            inputs = self.tokenizer(eval_prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,  # Greedy for consistency
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            rating_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            # Extract numeric rating
            rating = self._extract_rating(rating_text)
            normalized_score = rating / 10.0  # Normalize to 0-1

            alignment_scores.append(normalized_score)

        avg_alignment = np.mean(alignment_scores)
        return float(avg_alignment)

    def _extract_rating(self, text: str, max_rating: int = 100) -> float:
        """Extract a numeric rating from text."""
        # Try to find numbers in the text
        numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", text)

        if numbers:
            rating = float(numbers[0])
            # Clamp to 1-max_rating range
            return max(1.0, min(float(max_rating), rating))

        # Default to middle rating if no number found
        return float(max_rating) / 2.0

    def evaluate_response_pair(
        self,
        baseline_response: str,
        steered_response: str,
        trait_name: str,
        trait_description: str,
    ) -> SteeringEvaluationResult:
        """
        Evaluate a pair of pre-generated responses (baseline vs steered).

        This method evaluates the effectiveness of steering by comparing
        baseline and steered responses. Requires a model to be loaded.

        Args:
            baseline_response: The baseline model response
            steered_response: The steered model response
            trait_name: Name of the trait (e.g., "British", "Mean")
            trait_description: Description of what the trait means

        Returns:
            SteeringEvaluationResult with scores and analysis
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer required for evaluate_response_pair. Initialize with model and tokenizer.")

        log = bind(_LOG, trait=trait_name)
        log.info("Evaluating response pair")

        # Import evaluation functions
        from wisent.core.evaluators.personalization import (
            evaluate_difference,
            evaluate_quality,
            evaluate_alignment,
        )

        # Evaluate on three criteria using model self-evaluation
        difference_score = evaluate_difference(
            baseline_response, steered_response, self.model, self.tokenizer, self.device
        )
        quality_score = evaluate_quality(
            steered_response, self.model, self.tokenizer, self.device
        )
        alignment_score = evaluate_alignment(
            steered_response, trait_name, trait_description,
            self.model, self.tokenizer, self.device
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
                "evaluation_mode": "response_pair_model_judge",
            },
        )

