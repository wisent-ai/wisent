"""
GPTZero API evaluator for AI detection.

This evaluator uses the GPTZero API to detect AI-generated content.
The score represents how "human-like" the text appears.

Usage:
    from wisent.core.evaluators.custom.examples.gptzero import create_gptzero_evaluator
    
    evaluator = create_gptzero_evaluator(api_key="your-api-key")
    
    # Use with optimize-steering:
    wisent optimize-steering comprehensive model_name \\
        --custom-evaluator "wisent.core.evaluators.custom.examples.gptzero" \\
        --custom-evaluator-kwargs '{"api_key": "your-key"}'

For API documentation, see: https://gptzero.me/docs
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from wisent.core.evaluators.custom.custom_evaluator import (
    APIEvaluator,
    CustomEvaluatorConfig,
)

__all__ = ["GPTZeroEvaluator", "create_gptzero_evaluator"]

logger = logging.getLogger(__name__)


class GPTZeroEvaluator(APIEvaluator):
    """GPTZero API evaluator for AI detection.
    
    Scores responses based on how "human-like" they appear according to GPTZero.
    
    Metrics:
    - completely_generated_prob: Probability that entire text is AI-generated [0, 1]
    - average_generated_prob: Average probability per sentence [0, 1]
    
    The returned score is inverted: higher score = more human-like.
    
    Args:
        api_key: GPTZero API key (or set GPTZERO_API_KEY env var)
        optimize_for: Which metric to optimize ('human_prob', 'mixed_prob', 'avg_human_prob')
        version: API version ('v2' recommended)
    """
    
    GPTZERO_BASE_URL = "https://api.gptzero.me"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        optimize_for: str = "human_prob",
        model_version: str = "2025-11-28-base",
        **kwargs,
    ):
        config = CustomEvaluatorConfig(
            name="gptzero",
            description="GPTZero AI detection evaluator (higher = more human-like)",
            invert_score=False,  # We handle inversion in _call_api
            min_score=0.0,
            max_score=1.0,
        )
        
        api_key = api_key or os.environ.get("GPTZERO_API_KEY")
        if not api_key:
            raise ValueError(
                "GPTZero API key required. Set GPTZERO_API_KEY env var or pass api_key argument."
            )
        
        super().__init__(
            name="gptzero",
            description="GPTZero AI detection evaluator",
            config=config,
            api_key=api_key,
            base_url=self.GPTZERO_BASE_URL,
            **kwargs,
        )
        
        self.optimize_for = optimize_for
        self.model_version = model_version
    
    def _call_api(self, response: str, **kwargs) -> Dict[str, Any]:
        """Call GPTZero API to analyze text."""
        import requests
        
        if len(response.strip()) < 50:
            logger.warning("Text too short for GPTZero analysis, returning neutral score")
            return {"score": 0.5, "error": "text_too_short"}
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        payload = {
            "document": response,
            "version": self.model_version,
        }
        
        url = f"{self.base_url}/v2/predict/text"
        
        try:
            resp = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"GPTZero API error: {e}")
            raise
        
        documents = data.get("documents", [{}])
        doc = documents[0] if documents else {}
        
        ai_prob = doc.get("completely_generated_prob", 0.5)
        avg_ai_prob = doc.get("average_generated_prob", ai_prob)
        mixed_prob = doc.get("overall_burstiness", 0.5)
        
        human_prob = 1.0 - ai_prob
        avg_human_prob = 1.0 - avg_ai_prob
        
        if self.optimize_for == "human_prob":
            score = human_prob
        elif self.optimize_for == "mixed_prob":
            score = 1.0 - (ai_prob * 0.7 + avg_ai_prob * 0.3)
        elif self.optimize_for == "avg_human_prob":
            score = avg_human_prob
        else:
            score = human_prob
        
        return {
            "score": score,
            "human_prob": human_prob,
            "ai_prob": ai_prob,
            "avg_human_prob": avg_human_prob,
            "avg_ai_prob": avg_ai_prob,
            "burstiness": doc.get("overall_burstiness"),
            "sentences": doc.get("sentences", []),
            "raw_response": data,
        }
    
    def _call_api_batch(self, responses: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Batch API call for multiple texts (if supported)."""
        return [self._call_api(r, **kwargs) for r in responses]


def create_gptzero_evaluator(
    api_key: Optional[str] = None,
    optimize_for: str = "human_prob",
    **kwargs,
) -> GPTZeroEvaluator:
    """Create a GPTZero evaluator instance.
    
    Args:
        api_key: GPTZero API key (or set GPTZERO_API_KEY env var)
        optimize_for: Metric to optimize:
            - 'human_prob': Probability text is human-written
            - 'mixed_prob': Weighted combination of metrics
            - 'avg_human_prob': Average human probability per sentence
        **kwargs: Additional arguments passed to GPTZeroEvaluator
    
    Returns:
        GPTZeroEvaluator instance
    
    Example:
        evaluator = create_gptzero_evaluator(api_key="your-key")
        result = evaluator("This is some text to analyze")
        print(f"Human-like score: {result['score']}")
    """
    return GPTZeroEvaluator(api_key=api_key, optimize_for=optimize_for, **kwargs)


def create_evaluator(**kwargs) -> GPTZeroEvaluator:
    """Factory function for module-based loading."""
    return create_gptzero_evaluator(**kwargs)
