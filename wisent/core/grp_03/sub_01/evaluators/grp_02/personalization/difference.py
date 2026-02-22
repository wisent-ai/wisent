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
    baseline_response: "str | list[str]",
    steered_response: "str | list[str]",
    model: "PreTrainedModel | None" = None,
    tokenizer: "PreTrainedTokenizer | None" = None,
    device: "torch.device | None" = None,
) -> float:
    """
    Evaluate how different two responses are using Jaccard distance on a scale of 1-100.

    Uses token-level Jaccard distance to measure lexical difference between responses.
    This is fast, objective, and produces varied scores.

    Args:
        baseline_response: The baseline response (string or list of strings)
        steered_response: The steered response (string or list of strings)
        model: The model (not used, kept for API compatibility)
        tokenizer: The tokenizer (not used, kept for API compatibility)
        device: Device (not used, kept for API compatibility)

    Returns:
        Difference score between 1 and 100
        - 1 = Nearly identical (high Jaccard similarity)
        - 100 = Completely different (low Jaccard similarity)
    """
    # Handle list inputs - compute average difference
    if isinstance(baseline_response, list) and isinstance(steered_response, list):
        if len(baseline_response) != len(steered_response):
            # Different lengths - compare what we can
            min_len = min(len(baseline_response), len(steered_response))
            baseline_response = baseline_response[:min_len]
            steered_response = steered_response[:min_len]
        
        if not baseline_response:
            return 50.0  # Default if empty
        
        differences = []
        for b, s in zip(baseline_response, steered_response):
            similarity = _diversity._jaccard(b, s)
            diff = (1.0 - similarity) * 99.0 + 1.0
            differences.append(diff)
        return sum(differences) / len(differences)
    
    # Single string inputs
    similarity = _diversity._jaccard(baseline_response, steered_response)

    # Convert similarity to difference: higher similarity = lower difference
    # Scale to 1-100 range
    difference = (1.0 - similarity) * 99.0 + 1.0

    return difference
