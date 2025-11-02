"""
Difference evaluator for personalization steering.

Evaluates how different the steered response is from the baseline response.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wisent.core.synthetic.generators.diversities.methods.fast_diversity import FastDiversity

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    import torch

__all__ = ["evaluate_difference"]

# Initialize FastDiversity instance
_diversity = FastDiversity()


def evaluate_difference(
    baseline_response: str,
    steered_response: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: "torch.device",
) -> float:
    """
    Evaluate how different two responses are using Jaccard distance on a scale of 1-100.

    Uses token-level Jaccard distance to measure lexical difference between responses.
    This is fast, objective, and produces varied scores.

    Args:
        baseline_response: The baseline response
        steered_response: The steered response
        model: The model (not used, kept for API compatibility)
        tokenizer: The tokenizer (not used, kept for API compatibility)
        device: Device (not used, kept for API compatibility)

    Returns:
        Difference score between 1 and 100
        - 1 = Nearly identical (high Jaccard similarity)
        - 100 = Completely different (low Jaccard similarity)
    """
    # Calculate Jaccard similarity (0-1 scale)
    similarity = _diversity._jaccard(baseline_response, steered_response)

    # Convert similarity to difference: higher similarity = lower difference
    # Scale to 1-100 range
    difference = (1.0 - similarity) * 99.0 + 1.0

    return difference
