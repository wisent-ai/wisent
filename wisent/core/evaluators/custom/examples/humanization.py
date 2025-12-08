"""
Humanization evaluator for optimizing text to appear more human-written.

Uses GPTZero API exclusively for AI detection scoring.

Usage:
    from wisent.core.evaluators.custom.examples.humanization import create_humanization_evaluator
    
    evaluator = create_humanization_evaluator(api_key="your-gptzero-key")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from wisent.core.evaluators.custom.custom_evaluator import (
    CustomEvaluator,
    CustomEvaluatorConfig,
)

__all__ = ["HumanizationEvaluator", "create_humanization_evaluator"]

logger = logging.getLogger(__name__)


class HumanizationEvaluator(CustomEvaluator):
    """GPTZero-based evaluator for humanization optimization.
    
    Uses GPTZero API to score how "human-like" text appears.
    Score is normalized to [0, 1] where higher = more human-like.
    
    Args:
        api_key: GPTZero API key (required)
    """
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GPTZero API key is required for HumanizationEvaluator")
        
        config = CustomEvaluatorConfig(
            name="humanization",
            description="GPTZero-based humanization evaluator (higher = more human-like)",
        )
        super().__init__(name="humanization", description=config.description, config=config)
        
        from wisent.core.evaluators.custom.examples.gptzero import GPTZeroEvaluator
        self._gptzero = GPTZeroEvaluator(api_key=api_key)
    
    def evaluate_response(self, response: str, **kwargs) -> Dict[str, Any]:
        """Evaluate response using GPTZero API."""
        gptzero_result = self._gptzero(response)
        
        return {
            "score": gptzero_result["score"],
            "human_prob": gptzero_result.get("human_prob", gptzero_result["score"]),
            "ai_prob": gptzero_result.get("ai_prob", 1.0 - gptzero_result["score"]),
            "details": gptzero_result,
        }


def create_humanization_evaluator(api_key: str, **kwargs) -> HumanizationEvaluator:
    """Create a humanization evaluator using GPTZero.
    
    Args:
        api_key: GPTZero API key (required)
        **kwargs: Additional arguments (ignored, kept for compatibility)
    
    Returns:
        HumanizationEvaluator instance
    """
    return HumanizationEvaluator(api_key=api_key)


def create_evaluator(api_key: str = None, **kwargs) -> HumanizationEvaluator:
    """Factory function for module-based loading."""
    if not api_key:
        raise ValueError("api_key is required for humanization evaluator")
    return create_humanization_evaluator(api_key=api_key, **kwargs)
