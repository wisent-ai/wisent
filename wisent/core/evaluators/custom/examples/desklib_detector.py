"""
Desklib AI text detector - top performer on RAID benchmark.

Uses desklib/ai-text-detector-v1.01 model (fine-tuned DeBERTa-v3-large).

Usage:
    from wisent.core.evaluators.custom.examples.desklib_detector import DesklibDetectorEvaluator
    
    evaluator = DesklibDetectorEvaluator()
    result = evaluator("Some text to analyze")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel

from wisent.core.evaluators.custom.custom_evaluator import (
    CustomEvaluator,
    CustomEvaluatorConfig,
)

__all__ = ["DesklibDetectorEvaluator", "create_desklib_detector_evaluator"]

logger = logging.getLogger(__name__)


class DesklibAIDetectionModel(PreTrainedModel):
    """Custom model class for Desklib AI detector."""
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())

        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        return output


class DesklibDetectorEvaluator(CustomEvaluator):
    """Desklib AI detector - top performer on RAID benchmark.
    
    Uses desklib/ai-text-detector-v1.01 (fine-tuned DeBERTa-v3-large).
    
    Score is normalized to [0, 1] where higher = more human-like.
    """
    
    def __init__(self, device: Optional[str] = None, max_length: int = 768):
        config = CustomEvaluatorConfig(
            name="desklib_detector",
            description="Desklib AI detector (RAID benchmark leader, higher = more human-like)",
        )
        super().__init__(name="desklib_detector", description=config.description, config=config)
        
        self.model_id = "desklib/ai-text-detector-v1.01"
        self.max_length = max_length
        
        # Set device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        self._load_model()
    
    def _load_model(self):
        """Load the Desklib model and tokenizer."""
        logger.info(f"Loading Desklib AI detector from {self.model_id}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = DesklibAIDetectionModel.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Desklib detector loaded on {self.device}")
    
    def _predict(self, text: str) -> float:
        """Get AI probability for text."""
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            ai_probability = torch.sigmoid(logits).item()
        
        return ai_probability
    
    def evaluate_response(self, response: str, **kwargs) -> Dict[str, Any]:
        """Evaluate response for AI detection.
        
        Returns score where higher = more human-like.
        """
        if len(response.strip()) < 50:
            logger.warning("Text too short for reliable detection")
            return {
                "score": 0.5,
                "human_prob": 0.5,
                "ai_prob": 0.5,
                "warning": "Text too short for reliable detection",
            }
        
        ai_prob = self._predict(response)
        human_prob = 1.0 - ai_prob
        
        return {
            "score": human_prob,  # Higher = more human-like
            "human_prob": human_prob,
            "ai_prob": ai_prob,
        }


def create_desklib_detector_evaluator(
    device: Optional[str] = None,
    max_length: int = 768,
    **kwargs
) -> DesklibDetectorEvaluator:
    """Create a Desklib detector evaluator.
    
    Args:
        device: Device to run on (cuda, mps, cpu)
        max_length: Max token length (default 768)
    
    Returns:
        DesklibDetectorEvaluator instance
    """
    return DesklibDetectorEvaluator(device=device, max_length=max_length)


def create_evaluator(**kwargs) -> DesklibDetectorEvaluator:
    """Factory function for module-based loading."""
    return create_desklib_detector_evaluator(**kwargs)
