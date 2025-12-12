"""
Humanization evaluator with coherence check.

Combines AI detection (Desklib) with coherence evaluation to ensure
outputs are both human-like AND readable/coherent.

Usage:
    from wisent.core.evaluators.custom.examples.humanization_coherent import HumanizationCoherentEvaluator
    
    evaluator = HumanizationCoherentEvaluator()
    result = evaluator("Some text to analyze")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from wisent.core.evaluators.custom.custom_evaluator import (
    CustomEvaluator,
    CustomEvaluatorConfig,
)
from wisent.core.evaluators.personalization.coherence import evaluate_quality

__all__ = ["HumanizationCoherentEvaluator", "create_humanization_coherent_evaluator"]

logger = logging.getLogger(__name__)


class HumanizationCoherentEvaluator(CustomEvaluator):
    """Combined humanization + coherence evaluator.
    
    Uses Desklib for AI detection and coherence metrics to ensure
    outputs are both human-like AND readable.
    
    Final score = human_score * coherence_weight if coherence passes threshold,
    otherwise 0.
    
    Args:
        coherence_threshold: Minimum coherence score (0-100) to accept output (default: 50)
        coherence_weight: How much coherence affects final score (default: 0.5)
        device: Device for Desklib model
    """
    
    def __init__(
        self,
        coherence_threshold: float = 50.0,
        coherence_weight: float = 0.5,
        device: Optional[str] = None,
    ):
        config = CustomEvaluatorConfig(
            name="humanization_coherent",
            description="Humanization with coherence check (higher = more human-like AND coherent)",
        )
        super().__init__(name="humanization_coherent", description=config.description, config=config)
        
        self.coherence_threshold = coherence_threshold
        self.coherence_weight = coherence_weight
        
        # Load Desklib detector
        from wisent.core.evaluators.custom.examples.desklib_detector import DesklibDetectorEvaluator
        self._desklib = DesklibDetectorEvaluator(device=device)
    
    def evaluate_response(self, response: str, **kwargs) -> Dict[str, Any]:
        """Evaluate response for humanization AND coherence."""
        
        # Use existing coherence evaluator from personalization
        coherence_score = evaluate_quality(response)
        
        # If coherence is below threshold, return 0
        if coherence_score < self.coherence_threshold:
            logger.warning(f"Coherence check failed: {coherence_score:.1f} < {self.coherence_threshold}")
            logger.warning(f"Response preview: {response[:100]}...")
            return {
                "score": 0.0,
                "human_prob": 0.0,
                "ai_prob": 1.0,
                "coherence_score": coherence_score,
                "rejected_reason": f"Coherence {coherence_score:.1f} below threshold {self.coherence_threshold}",
            }
        
        # Get Desklib score
        desklib_result = self._desklib(response)
        human_prob = desklib_result["human_prob"]
        
        # Combine scores: human_prob * (coherence_factor)
        # coherence_factor scales from coherence_weight to 1.0 based on coherence
        coherence_normalized = coherence_score / 100.0  # 0-1
        coherence_factor = self.coherence_weight + (1.0 - self.coherence_weight) * coherence_normalized
        
        final_score = human_prob * coherence_factor
        
        return {
            "score": final_score,
            "human_prob": human_prob,
            "ai_prob": desklib_result["ai_prob"],
            "coherence_score": coherence_score,
            "coherence_factor": coherence_factor,
        }


def create_humanization_coherent_evaluator(
    coherence_threshold: float = 50.0,
    coherence_weight: float = 0.5,
    device: Optional[str] = None,
    **kwargs
) -> HumanizationCoherentEvaluator:
    """Create a humanization + coherence evaluator.
    
    Args:
        coherence_threshold: Minimum coherence score to accept (default: 50)
        coherence_weight: How much coherence affects score (default: 0.5)
        device: Device for Desklib model
    
    Returns:
        HumanizationCoherentEvaluator instance
    """
    return HumanizationCoherentEvaluator(
        coherence_threshold=coherence_threshold,
        coherence_weight=coherence_weight,
        device=device,
    )


def create_evaluator(**kwargs) -> HumanizationCoherentEvaluator:
    """Factory function for module-based loading."""
    return create_humanization_coherent_evaluator(**kwargs)
