"""
RoBERTa-based AI text detector using Hugging Face's free inference API.

Uses openai-community/roberta-base-openai-detector model.

Usage:
    from wisent.core.evaluators.custom.examples.roberta_detector import RobertaDetectorEvaluator
    
    evaluator = RobertaDetectorEvaluator()  # No API key needed!
    result = evaluator("Some text to analyze")
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from wisent.core.evaluators.custom.custom_evaluator import (
    CustomEvaluator,
    CustomEvaluatorConfig,
)

__all__ = ["RobertaDetectorEvaluator", "create_roberta_detector_evaluator"]

logger = logging.getLogger(__name__)


class RobertaDetectorEvaluator(CustomEvaluator):
    """Free AI detector using Hugging Face's RoBERTa model.
    
    Uses the openai-community/roberta-base-openai-detector model
    via Hugging Face's free inference API or local inference.
    
    Score is normalized to [0, 1] where higher = more human-like.
    
    Args:
        use_local: If True, load model locally instead of using API (default: False)
        hf_token: Optional Hugging Face token for API rate limits
    """
    
    def __init__(self, use_local: bool = True, hf_token: Optional[str] = None):
        config = CustomEvaluatorConfig(
            name="roberta_detector",
            description="RoBERTa-based AI detector (free, higher = more human-like)",
        )
        super().__init__(name="roberta_detector", description=config.description, config=config)
        
        self.use_local = use_local
        self.hf_token = hf_token
        self.model_id = "openai-community/roberta-base-openai-detector"
        
        self._pipeline = None
        self._api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        
        if use_local:
            self._init_local_model()
    
    def _init_local_model(self):
        """Initialize local model pipeline."""
        try:
            from transformers import pipeline
            self._pipeline = pipeline("text-classification", model=self.model_id)
            logger.info(f"Loaded local model: {self.model_id}")
        except Exception as e:
            logger.warning(f"Failed to load local model, falling back to API: {e}")
            self.use_local = False
    
    def _query_api(self, text: str, retries: int = 3) -> Dict[str, Any]:
        """Query Hugging Face inference API."""
        import requests
        
        headers = {}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        
        for attempt in range(retries):
            try:
                response = requests.post(
                    self._api_url,
                    headers=headers,
                    json={"inputs": text},
                    timeout=30,
                )
                
                if response.status_code == 503:
                    # Model loading, wait and retry
                    wait_time = response.json().get("estimated_time", 20)
                    logger.info(f"Model loading, waiting {wait_time}s...")
                    time.sleep(min(wait_time, 30))
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"API call failed after {retries} attempts: {e}")
        
        raise RuntimeError("API call failed: max retries exceeded")
    
    def _parse_result(self, result: Any) -> Dict[str, float]:
        """Parse API or local result into standardized format."""
        # Result is list of [{"label": "LABEL_0/1", "score": float}, ...]
        # LABEL_0 = Fake (AI), LABEL_1 = Real (Human)
        
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):
                result = result[0]
            
            scores = {item["label"]: item["score"] for item in result}
            
            # Real = human, Fake = AI
            human_prob = scores.get("Real", scores.get("LABEL_1", 0.5))
            ai_prob = scores.get("Fake", scores.get("LABEL_0", 0.5))
            
            return {
                "human_prob": human_prob,
                "ai_prob": ai_prob,
            }
        
        return {"human_prob": 0.5, "ai_prob": 0.5}
    
    def evaluate_response(self, response: str, **kwargs) -> Dict[str, Any]:
        """Evaluate response for AI detection."""
        if len(response.strip()) < 50:
            logger.warning("Text too short for reliable detection")
            return {
                "score": 0.5,
                "human_prob": 0.5,
                "ai_prob": 0.5,
                "warning": "Text too short for reliable detection",
            }
        
        # Truncate to model's max length (512 tokens ~ 2000 chars)
        text = response[:2000]
        
        if self.use_local and self._pipeline:
            result = self._pipeline(text)
        else:
            result = self._query_api(text)
        
        parsed = self._parse_result(result)
        
        return {
            "score": parsed["human_prob"],  # Higher = more human-like
            "human_prob": parsed["human_prob"],
            "ai_prob": parsed["ai_prob"],
        }


def create_roberta_detector_evaluator(
    use_local: bool = True,
    hf_token: Optional[str] = None,
    **kwargs
) -> RobertaDetectorEvaluator:
    """Create a RoBERTa detector evaluator.
    
    Args:
        use_local: Load model locally instead of using API
        hf_token: Optional Hugging Face token
    
    Returns:
        RobertaDetectorEvaluator instance
    """
    return RobertaDetectorEvaluator(use_local=use_local, hf_token=hf_token)


def create_evaluator(**kwargs) -> RobertaDetectorEvaluator:
    """Factory function for module-based loading."""
    return create_roberta_detector_evaluator(**kwargs)
