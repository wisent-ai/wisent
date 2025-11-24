"""
KL divergence measurement for evaluating steering impact.

Measures how much steering changes the model's probability distribution
compared to the original model (preservation of capabilities).

Adapted from Heretic's evaluator.py KL divergence computation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["compute_kl_divergence", "KLDivergenceEvaluator"]


def compute_kl_divergence(
    baseline_logprobs: Tensor,
    modified_logprobs: Tensor,
    reduction: str = "batchmean",
) -> float:
    """
    Compute KL divergence between baseline and modified model distributions.

    Measures: KL(P_baseline || P_modified)

    This quantifies how much the modified model's distribution differs from
    the baseline. Lower values indicate better preservation of original
    model capabilities.

    Args:
        baseline_logprobs: Log-probabilities from baseline model, shape [B, V]
                          where B is batch size, V is vocab size
        modified_logprobs: Log-probabilities from modified/steered model, shape [B, V]
        reduction: How to reduce across batch:
                  - "batchmean": Average over batch (default, matches F.kl_div)
                  - "sum": Sum over batch
                  - "mean": Mean over all elements
                  - "none": No reduction, return per-sample

    Returns:
        KL divergence value (scalar if reduction != "none", else tensor)

    Example:
        >>> baseline = model.get_logprobs(prompts)
        >>> steered = steered_model.get_logprobs(prompts)
        >>> kl = compute_kl_divergence(baseline, steered)
        >>> print(f"KL divergence: {kl:.4f}")
    """
    # F.kl_div expects (input, target, log_target=True)
    # KL(P || Q) = sum(P * log(P/Q)) = sum(P * (log(P) - log(Q)))
    # When both are in log-space: F.kl_div(log_Q, log_P, log_target=True)
    kl_div = F.kl_div(
        modified_logprobs,
        baseline_logprobs,
        log_target=True,
        reduction=reduction,
    )

    if reduction == "none":
        return kl_div
    else:
        return kl_div.item()


class KLDivergenceEvaluator:
    """
    Evaluator for measuring KL divergence between baseline and steered models.

    This class handles:
    1. Collecting baseline log-probabilities (once)
    2. Computing KL divergence for steered models
    3. Tracking divergence across multiple evaluations

    Adapted from Heretic's Evaluator class.

    Usage:
        evaluator = KLDivergenceEvaluator(model, prompts)
        kl = evaluator.evaluate(steered_model)
    """

    def __init__(
        self,
        model,
        prompts: list[str],
        tokenizer=None,
        device: str | None = None,
    ):
        """
        Initialize KL divergence evaluator.

        Args:
            model: Baseline model (unsteered)
            prompts: List of prompts to evaluate on (should be "harmless")
            tokenizer: Tokenizer (if None, extracted from model)
            device: Device to run on (if None, extracted from model)
        """
        self.model = model
        self.prompts = prompts
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.device = device or next(model.parameters()).device

        # Collect baseline log-probabilities once
        self.baseline_logprobs = self._collect_logprobs(model, prompts)

    def _collect_logprobs(self, model, prompts: list[str]) -> Tensor:
        """
        Collect log-probabilities of first generated token for each prompt.

        Args:
            model: Model to evaluate
            prompts: List of prompts

        Returns:
            Log-probabilities tensor, shape [len(prompts), vocab_size]
        """
        logprobs_list = []

        for prompt in prompts:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            # Generate 1 token with logits
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    do_sample=False,
                )

            # Extract first token logits and convert to log-probabilities
            first_token_logits = outputs.scores[0][0]  # [vocab_size]
            logprobs = F.log_softmax(first_token_logits, dim=-1)
            logprobs_list.append(logprobs)

        return torch.stack(logprobs_list, dim=0)  # [B, V]

    def evaluate(self, modified_model) -> float:
        """
        Evaluate KL divergence between baseline and modified model.

        Args:
            modified_model: Model with steering applied

        Returns:
            KL divergence (lower is better - less deviation from baseline)
        """
        modified_logprobs = self._collect_logprobs(modified_model, self.prompts)
        return compute_kl_divergence(self.baseline_logprobs, modified_logprobs)

    def evaluate_with_steering(
        self,
        model,
        steering_vectors: dict[int, Tensor],
        alpha: float,
    ) -> float:
        """
        Evaluate KL divergence with steering applied via hooks.

        Args:
            model: Base model
            steering_vectors: Dictionary mapping layer index to steering vector
            alpha: Steering strength

        Returns:
            KL divergence
        """
        # Apply steering hooks
        from wisent.core.steering.application import apply_steering

        handles = apply_steering(model, steering_vectors, alpha)

        try:
            kl_div = self.evaluate(model)
        finally:
            # Remove hooks
            for handle in handles:
                handle.remove()

        return kl_div

    def __repr__(self) -> str:
        return f"KLDivergenceEvaluator(prompts={len(self.prompts)})"
