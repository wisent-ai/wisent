"""
Alignment evaluator for personalization steering.

Evaluates how well the response exhibits the target trait using contrastive
embedding similarity against synthetic positive and negative examples.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    import torch

from wisent.core.errors import MissingParameterError

__all__ = ["evaluate_alignment", "estimate_alignment"]

# Embedding model (lazily loaded)
_embedding_model = None


def _get_embedding_model():
    """Lazily load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model


def evaluate_alignment(
    response: str,
    trait_name: str,
    trait_description: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: "torch.device",
    positive_examples: list[str] | None = None,
    negative_examples: list[str] | None = None,
) -> float:
    """
    Evaluate how well the response exhibits a trait using contrastive embedding similarity.

    Compares the response embedding against positive and negative examples.
    The score measures how much closer the response is to positive examples
    versus negative examples.

    Args:
        response: The response to evaluate
        trait_name: Target trait name (unused, kept for API compatibility)
        trait_description: Trait description (unused when examples provided)
        model: The model (unused, kept for API compatibility)
        tokenizer: The tokenizer (unused, kept for API compatibility)
        device: Device (unused, kept for API compatibility)
        positive_examples: List of positive example responses for the trait
        negative_examples: List of negative example responses for the trait

    Returns:
        Alignment score between 1 and 100
    """
    if not response:
        return 1.0

    if not positive_examples or not negative_examples:
        raise MissingParameterError(params=["positive_examples", "negative_examples"], context="alignment evaluation")

    score = _compute_contrastive_alignment(response, positive_examples, negative_examples)

    # Scale from 0-1 to 1-100
    return score * 99.0 + 1.0


def estimate_alignment(
    responses: list[str],
    trait_description: str,
    positive_examples: list[str] | None = None,
    negative_examples: list[str] | None = None,
) -> float:
    """
    Estimate trait alignment using contrastive embedding similarity.

    Computes how much closer response embeddings are to positive examples
    versus negative examples.

    Args:
        responses: List of model responses to evaluate
        trait_description: Description of the trait (unused when examples provided)
        positive_examples: List of positive example responses for the trait
        negative_examples: List of negative example responses for the trait

    Returns:
        Float score between 0 and 1 indicating alignment with trait
    """
    if not responses:
        return 0.0

    if not positive_examples or not negative_examples:
        raise MissingParameterError(params=["positive_examples", "negative_examples"], context="alignment evaluation")

    scores = [_compute_contrastive_alignment(r, positive_examples, negative_examples) for r in responses]

    return float(np.mean(scores))


def _compute_contrastive_alignment(
    response: str,
    positive_examples: list[str],
    negative_examples: list[str],
) -> float:
    """
    Compute alignment score by comparing response to positive vs negative examples.

    Args:
        response: The response to evaluate
        positive_examples: List of positive example responses
        negative_examples: List of negative example responses

    Returns:
        Float score between 0 and 1
        - 1.0 = response is much more similar to positive than negative
        - 0.5 = response is equally similar to both
        - 0.0 = response is much more similar to negative than positive
    """
    import torch

    model = _get_embedding_model()

    # Encode response
    response_embedding = model.encode(
        response,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    # Encode positive examples
    positive_embeddings = model.encode(
        positive_examples,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    # Encode negative examples
    negative_embeddings = model.encode(
        negative_examples,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    # Compute mean cosine similarity to positive examples
    positive_similarities = torch.matmul(positive_embeddings, response_embedding)
    mean_positive_sim = positive_similarities.mean().item()

    # Compute mean cosine similarity to negative examples
    negative_similarities = torch.matmul(negative_embeddings, response_embedding)
    mean_negative_sim = negative_similarities.mean().item()

    # Contrastive score: how much more similar to positive than negative
    # Range: [-2, 2] -> normalize to [0, 1]
    contrastive_score = mean_positive_sim - mean_negative_sim
    normalized_score = (contrastive_score + 2) / 4

    return max(0.0, min(1.0, normalized_score))
