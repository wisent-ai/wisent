"""
RoBERTa-based AI text detector using Hugging Face's free inference API.

Uses openai-community/roberta-base-openai-detector model.

Usage:
    from wisent.core.reading.evaluators.custom.examples.roberta_detector import RobertaDetectorEvaluator
    
    evaluator = create_roberta_detector_evaluator(api_retries=retries, exponential_backoff_base=backoff, api_max_wait_time=max_wait)
    result = evaluator("Some text to analyze")
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from wisent.core.reading.evaluators.custom.custom_evaluator import (
    CustomEvaluator,
    CustomEvaluatorConfig,
)
from wisent.core.utils.infra_tools.infra.core.hardware import docker_code_exec_timeout_s

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
    
    def __init__(self, api_retries: int, exponential_backoff_base: int, api_max_wait_time: int, min_response_text_length: int, detector_max_text_length: int, use_local: bool = True, hf_token: Optional[str] = None):
        config = CustomEvaluatorConfig(
            name="roberta_detector",
            description="RoBERTa-based AI detector (free, higher = more human-like)",
        )
        super().__init__(name="roberta_detector", description=config.description, config=config)

        self.use_local = use_local
        self.hf_token = hf_token
        self._api_retries = api_retries
        self._exponential_backoff_base = exponential_backoff_base
        self._api_max_wait_time = api_max_wait_time
        self._min_response_text_length = min_response_text_length
        self._detector_max_text_length = detector_max_text_length
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
    
    def _query_api(self, text: str) -> Dict[str, Any]:
        """Query Hugging Face inference API."""
        import requests

        headers = {}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        last_error: Exception = RuntimeError("No attempts made")
        for attempt in range(self._api_retries):
            try:
                response = requests.post(
                    self._api_url,
                    headers=headers,
                    json={"inputs": text},
                    timeout=docker_code_exec_timeout_s(),
                )
                
                if response.status_code == 503:
                    # Model loading, wait and retry
                    wait_time = response.json()["estimated_time"]
                    logger.info(f"Model loading, waiting {wait_time}s...")
                    time.sleep(min(wait_time, self._api_max_wait_time))
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                last_error = e
                time.sleep(self._exponential_backoff_base ** attempt)
        raise RuntimeError(f"API call failed after {self._api_retries} attempts: {last_error}")
    
    def _parse_result(self, result: Any) -> Dict[str, float]:
        """Parse API or local result into standardized format."""
        # Result is list of [{"label": "LABEL_0/1", "score": float}, ...]
        # LABEL_0 = Fake (AI), LABEL_1 = Real (Human)
        
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):
                result = result[0]
            
            scores = {item["label"]: item["score"] for item in result}
            
            # Real = human, Fake = AI
            human_prob = scores.get("Real", scores.get("LABEL_1"))
            ai_prob = scores.get("Fake", scores.get("LABEL_0"))
            if human_prob is None:
                raise KeyError("RoBERTa detector response missing both 'Real' and 'LABEL_1' keys")
            if ai_prob is None:
                raise KeyError("RoBERTa detector response missing both 'Fake' and 'LABEL_0' keys")
            
            return {
                "human_prob": human_prob,
                "ai_prob": ai_prob,
            }
        
        return {"human_prob": 0.5, "ai_prob": 0.5}
    
    def evaluate_response(self, response: str, **kwargs) -> Dict[str, Any]:
        """Evaluate response for AI detection."""
        if len(response.strip()) < self._min_response_text_length:
            logger.warning("Text too short for reliable detection")
            return {
                "score": 0.5,
                "human_prob": 0.5,
                "ai_prob": 0.5,
                "warning": "Text too short for reliable detection",
            }
        
        # Truncate to model's max length (512 tokens ~ 2000 chars)
        text = response[:self._detector_max_text_length]
        
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
    api_retries: int,
    exponential_backoff_base: int,
    api_max_wait_time: int,
    min_response_text_length: int,
    detector_max_text_length: int,
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
    return RobertaDetectorEvaluator(api_retries=api_retries, exponential_backoff_base=exponential_backoff_base, api_max_wait_time=api_max_wait_time, min_response_text_length=min_response_text_length, detector_max_text_length=detector_max_text_length, use_local=use_local, hf_token=hf_token)


def create_evaluator(**kwargs) -> RobertaDetectorEvaluator:
    """Factory function for module-based loading."""
    return create_roberta_detector_evaluator(api_retries=kwargs.pop("api_retries"), exponential_backoff_base=kwargs.pop("exponential_backoff_base"), api_max_wait_time=kwargs.pop("api_max_wait_time"), min_response_text_length=kwargs.pop("min_response_text_length"), detector_max_text_length=kwargs.pop("detector_max_text_length"), **kwargs)
