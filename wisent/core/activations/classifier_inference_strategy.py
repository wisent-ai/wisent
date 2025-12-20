"""
Classifier inference strategies for runtime classification.

These strategies determine how to extract activations from generated text
at inference time when classifying responses.

Based on empirical testing across 3 models (Llama-3.2-1B, Llama-2-7b, Qwen3-8B)
and 4 tasks (truthfulqa, happy, left_wing, livecodebench):

Results:
- last_token: Best performer (77% with chat_last training on truthfulqa)
- all_mean: Poor (~50%) - dominated by shared prompt tokens
- all_max/all_min: Poor (~50%)
- first_token: BROKEN (50%) - BOS token is identical for all inputs

Recommendation: Use LAST_TOKEN (default) - it works best with chat_last training strategy.

IMPORTANT: These strategies operate on the FULL sequence (prompt + response).
At inference time, we typically don't know where the answer starts, so we
can only use strategies that work on the whole sequence.
"""

from enum import Enum
from typing import Optional
import argparse
import torch
import numpy as np


class ClassifierInferenceStrategy(str, Enum):
    """
    Strategies for extracting activations at inference time for classification.
    """
    
    LAST_TOKEN = "last_token"
    """Extract activation from the last token only. Best overall performance."""
    
    FIRST_TOKEN = "first_token"
    """Extract activation from the first token only. NOT RECOMMENDED - BOS token has no variance."""
    
    ALL_MEAN = "all_mean"
    """Classify each token, return mean of all scores."""
    
    ALL_MAX = "all_max"
    """Classify each token, return max score (most confident positive)."""
    
    ALL_MIN = "all_min"
    """Classify each token, return min score (most confident negative)."""

    @property
    def description(self) -> str:
        descriptions = {
            ClassifierInferenceStrategy.LAST_TOKEN: "Last token activation (recommended)",
            ClassifierInferenceStrategy.FIRST_TOKEN: "First token activation (not recommended)",
            ClassifierInferenceStrategy.ALL_MEAN: "Mean of all token scores",
            ClassifierInferenceStrategy.ALL_MAX: "Max of all token scores",
            ClassifierInferenceStrategy.ALL_MIN: "Min of all token scores",
        }
        return descriptions.get(self, "Unknown strategy")
    
    @classmethod
    def default(cls) -> "ClassifierInferenceStrategy":
        """Return the default strategy (last_token performs best)."""
        return cls.LAST_TOKEN
    
    @classmethod
    def list_all(cls) -> list[str]:
        """List all strategy names."""
        return [s.value for s in cls]


def extract_inference_activation(
    strategy: ClassifierInferenceStrategy,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """
    Extract activation for classification at inference time.
    
    Args:
        strategy: The inference strategy to use
        hidden_states: Hidden states tensor of shape [seq_len, hidden_dim]
    
    Returns:
        Activation vector of shape [hidden_dim]
    """
    seq_len = hidden_states.shape[0]
    
    if strategy == ClassifierInferenceStrategy.LAST_TOKEN:
        return hidden_states[-1]
    
    elif strategy == ClassifierInferenceStrategy.FIRST_TOKEN:
        return hidden_states[0]
    
    elif strategy == ClassifierInferenceStrategy.ALL_MEAN:
        return hidden_states.mean(dim=0)
    
    elif strategy == ClassifierInferenceStrategy.ALL_MAX:
        # Token with max norm
        norms = torch.norm(hidden_states, dim=1)
        return hidden_states[torch.argmax(norms)]
    
    elif strategy == ClassifierInferenceStrategy.ALL_MIN:
        # Token with min norm
        norms = torch.norm(hidden_states, dim=1)
        return hidden_states[torch.argmin(norms)]
    
    else:
        raise ValueError(f"Unknown classifier inference strategy: {strategy}")


def get_inference_score(
    classifier,
    hidden_states: torch.Tensor,
    strategy: ClassifierInferenceStrategy,
) -> float:
    """
    Get classifier score using the specified inference strategy.
    
    For single-token strategies (last_token, first_token), returns the classifier
    probability for that token.
    
    For all_* strategies, classifies each token and aggregates scores.
    
    Args:
        classifier: A trained classifier with predict_proba method
        hidden_states: Hidden states tensor of shape [seq_len, hidden_dim]
        strategy: The inference strategy to use
    
    Returns:
        Classification score (probability of positive class)
    """
    hidden_np = hidden_states.cpu().float().numpy()
    seq_len = hidden_np.shape[0]
    
    if strategy == ClassifierInferenceStrategy.LAST_TOKEN:
        return float(classifier.predict_proba([hidden_np[-1]])[0, 1])
    
    elif strategy == ClassifierInferenceStrategy.FIRST_TOKEN:
        return float(classifier.predict_proba([hidden_np[0]])[0, 1])
    
    elif strategy in (ClassifierInferenceStrategy.ALL_MEAN, 
                      ClassifierInferenceStrategy.ALL_MAX,
                      ClassifierInferenceStrategy.ALL_MIN):
        # Classify all tokens
        all_scores = []
        for t in range(seq_len):
            score = classifier.predict_proba([hidden_np[t]])[0, 1]
            all_scores.append(score)
        
        if strategy == ClassifierInferenceStrategy.ALL_MEAN:
            return float(np.mean(all_scores))
        elif strategy == ClassifierInferenceStrategy.ALL_MAX:
            return float(np.max(all_scores))
        elif strategy == ClassifierInferenceStrategy.ALL_MIN:
            return float(np.min(all_scores))
    
    raise ValueError(f"Unknown classifier inference strategy: {strategy}")


def get_recommended_inference_strategy(train_strategy) -> ClassifierInferenceStrategy:
    """
    Get the recommended inference strategy for a given training strategy.
    
    Based on empirical testing:
    - chat_last, role_play, mc_balanced -> last_token
    - chat_mean, chat_weighted, chat_max_norm, chat_first -> all_mean
    
    Args:
        train_strategy: ExtractionStrategy used for training
    
    Returns:
        Recommended ClassifierInferenceStrategy
    """
    # Import here to avoid circular dependency
    from wisent.core.activations.extraction_strategy import ExtractionStrategy
    
    if train_strategy in (ExtractionStrategy.CHAT_LAST, 
                          ExtractionStrategy.ROLE_PLAY, 
                          ExtractionStrategy.MC_BALANCED):
        return ClassifierInferenceStrategy.LAST_TOKEN
    else:
        return ClassifierInferenceStrategy.ALL_MEAN


def add_classifier_inference_strategy_args(parser: argparse.ArgumentParser) -> None:
    """
    Add --classifier-inference-strategy argument to an argument parser.
    """
    parser.add_argument(
        "--classifier-inference-strategy",
        type=str,
        default=ClassifierInferenceStrategy.default().value,
        choices=ClassifierInferenceStrategy.list_all(),
        help=f"Inference strategy for classifier. Options: {', '.join(ClassifierInferenceStrategy.list_all())}. Default: {ClassifierInferenceStrategy.default().value}",
    )
