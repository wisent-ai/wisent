"""
Humanization evaluator with coherence check.

Combines AI detection (Desklib) with semantic coherence evaluation to ensure
outputs are both human-like AND actually answer the prompt coherently.

Uses enochlev/coherence-all-mpnet-base-v2 model to check if the response
is semantically relevant to the prompt.

Usage:
    from wisent.core.evaluators.custom.examples.humanization_coherent import HumanizationCoherentEvaluator
    
    evaluator = HumanizationCoherentEvaluator()
    result = evaluator("Some text to analyze", prompt="What was the question?")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from wisent.core.evaluators.custom.custom_evaluator import (
    CustomEvaluator,
    CustomEvaluatorConfig,
)

__all__ = ["HumanizationCoherentEvaluator", "create_humanization_coherent_evaluator"]

logger = logging.getLogger(__name__)

# Cache for coherence model
_coherence_model_cache = {}


def _get_coherence_model(device: Optional[str] = None):
    """Get cached coherence model."""
    if "model" not in _coherence_model_cache:
        model_name = "enochlev/coherence-all-mpnet-base-v2"
        logger.info(f"Loading coherence model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        if device:
            model = model.to(device)
        _coherence_model_cache["model"] = model
        _coherence_model_cache["tokenizer"] = tokenizer
    return _coherence_model_cache["model"], _coherence_model_cache["tokenizer"]


class HumanizationCoherentEvaluator(CustomEvaluator):
    """Combined humanization + semantic coherence evaluator.
    
    Uses Desklib for AI detection and a cross-encoder model to check if
    the response actually answers the prompt coherently.
    
    Final score = human_score if coherence passes threshold, otherwise 0.
    
    Args:
        coherence_threshold: Minimum coherence score (0-1) to accept output (default: 0.3)
        device: Device for models
    """
    
    def __init__(
        self,
        coherence_threshold: float = 0.3,
        device: Optional[str] = None,
    ):
        config = CustomEvaluatorConfig(
            name="humanization_coherent",
            description="Humanization with semantic coherence check (higher = more human-like AND coherent)",
        )
        super().__init__(name="humanization_coherent", description=config.description, config=config)
        
        self.coherence_threshold = coherence_threshold
        self.device = device
        
        # Load Desklib detector
        from wisent.core.evaluators.custom.examples.desklib_detector import DesklibDetectorEvaluator
        self._desklib = DesklibDetectorEvaluator(device=device)
        
        # Load coherence model
        self._coherence_model, self._coherence_tokenizer = _get_coherence_model(device)
    
    def _score_coherence(self, prompt: str, response: str) -> float:
        """Score how coherent the response is to the prompt."""
        inputs = self._coherence_tokenizer(
            prompt, response, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        if self.device:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._coherence_model(**inputs)
        
        score = torch.sigmoid(outputs.logits).item()
        return score
    
    def evaluate_response(self, response: str, prompt: str = None, **kwargs) -> Dict[str, Any]:
        """Evaluate response for humanization AND coherence.
        
        Args:
            response: The generated response text
            prompt: The original prompt (required for coherence check)
            **kwargs: Additional arguments
            
        Returns:
            Dict with score, human_prob, coherence_score, etc.
        """
        # If no prompt provided, skip coherence check (fallback behavior)
        if not prompt:
            logger.warning("No prompt provided, skipping coherence check")
            desklib_result = self._desklib(response)
            return {
                "score": desklib_result["human_prob"],
                "human_prob": desklib_result["human_prob"],
                "ai_prob": desklib_result["ai_prob"],
                "coherence_score": None,
                "coherence_skipped": True,
            }
        
        # Check semantic coherence - does response actually answer the prompt?
        coherence_score = self._score_coherence(prompt, response)
        
        # If coherence is below threshold, return 0
        if coherence_score < self.coherence_threshold:
            logger.warning(f"Coherence check failed: {coherence_score:.3f} < {self.coherence_threshold}")
            logger.warning(f"Prompt: {prompt[:50]}...")
            logger.warning(f"Response preview: {response[:100]}...")
            return {
                "score": 0.0,
                "human_prob": 0.0,
                "ai_prob": 1.0,
                "coherence_score": coherence_score,
                "rejected_reason": f"Coherence {coherence_score:.3f} below threshold {self.coherence_threshold}",
            }
        
        # Get Desklib score
        desklib_result = self._desklib(response)
        human_prob = desklib_result["human_prob"]
        
        # Score is human_prob weighted by coherence
        # Higher coherence = closer to full human_prob score
        final_score = human_prob * coherence_score
        
        return {
            "score": final_score,
            "human_prob": human_prob,
            "ai_prob": desklib_result["ai_prob"],
            "coherence_score": coherence_score,
        }


def create_humanization_coherent_evaluator(
    coherence_threshold: float = 0.3,
    device: Optional[str] = None,
    **kwargs
) -> HumanizationCoherentEvaluator:
    """Create a humanization + coherence evaluator.
    
    Args:
        coherence_threshold: Minimum coherence score (0-1) to accept (default: 0.3)
        device: Device for models
    
    Returns:
        HumanizationCoherentEvaluator instance
    """
    return HumanizationCoherentEvaluator(
        coherence_threshold=coherence_threshold,
        device=device,
    )


def create_evaluator(**kwargs) -> HumanizationCoherentEvaluator:
    """Factory function for module-based loading."""
    return create_humanization_coherent_evaluator(**kwargs)
