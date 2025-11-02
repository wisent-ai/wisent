"""
Alignment evaluator for personalization steering.

Evaluates how well the response exhibits the target trait.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

__all__ = ["evaluate_alignment"]


def evaluate_alignment(
    response: str,
    trait_name: str,
    trait_description: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
) -> float:
    """
    Use the model to evaluate how well the response exhibits a trait on a scale of 1-100.

    Args:
        response: The response to evaluate
        trait_name: Target trait name
        trait_description: Trait description
        model: The model to use for evaluation
        tokenizer: The tokenizer for the model
        device: Device to run on

    Returns:
        Alignment score between 1 and 100
    """
    eval_prompt = f"""Rate how strongly this response demonstrates the trait "{trait_name}" on a scale of 1-100.

Trait: {trait_name}
Description: {trait_description}

Response: {response}

Does this response exhibit "{trait_name}" characteristics?
1 = Not at all
50 = Moderately
100 = Strongly

Rating (1-100):"""

    rating = _get_model_rating(eval_prompt, model, tokenizer, device, max_rating=100)
    return rating


def _get_model_rating(
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    max_rating: int = 100,
) -> float:
    """
    Get a rating from the model for an evaluation prompt.

    Args:
        prompt: The evaluation prompt
        model: The model to use
        tokenizer: The tokenizer
        device: Device to run on
        max_rating: Maximum rating value (default 100)

    Returns:
        Rating between 1 and max_rating
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,  # Greedy for consistency
            pad_token_id=tokenizer.eos_token_id,
        )

    rating_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    # Extract numeric rating
    return _extract_rating(rating_text, max_rating)


def _extract_rating(text: str, max_rating: int = 100) -> float:
    """Extract a numeric rating from text."""
    # Try to find numbers in the text
    numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", text)

    if numbers:
        rating = float(numbers[0])
        # Clamp to 1-max_rating range
        return max(1.0, min(float(max_rating), rating))

    # Default to middle rating if no number found
    return float(max_rating) / 2.0
