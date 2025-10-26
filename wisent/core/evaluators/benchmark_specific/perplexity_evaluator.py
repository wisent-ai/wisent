"""Perplexity evaluator for language modeling benchmarks.

Used for tasks that measure language modeling performance.
"""

from typing import Any
import logging
import math
import torch

from wisent.core.evaluators.core.atoms import BaseEvaluator, EvalResult

logger = logging.getLogger(__name__)


class PerplexityEvaluator(BaseEvaluator):
    """Evaluator using perplexity for language modeling tasks.

    Compatible with:
    - WikiText: Language modeling
    - LAMBADA: Word prediction in context
    - Any loglikelihood_rolling task
    """

    name = "perplexity"
    description = "Perplexity evaluator for language modeling"
    task_names = ("wikitext", "lambada_openai", "lambada_standard")

    def __init__(self, model=None):
        """Initialize perplexity evaluator.

        Args:
            model: Model with loglikelihood capabilities
        """
        super().__init__()
        self.model = model

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate using perplexity.

        Args:
            response: Text to evaluate (for language modeling)
            expected: NOT USED (perplexity is computed on response)
            **kwargs:
                model: Model instance (WisentModel or similar, overrides self.model)
                context: Optional context for conditional generation

        Returns:
            EvalResult with perplexity as confidence metric (lower is better)
        """
        model = kwargs.get('model', self.model)
        context = kwargs.get('context', '')

        if model is None:
            raise ValueError(
                "No model provided for perplexity computation. "
                "Please provide a model via __init__ or as a kwarg."
            )

        try:
            # Compute perplexity
            full_text = f"{context}{response}" if context else response
            perplexity = self._compute_perplexity(model, full_text)

            # Lower perplexity is better, so we use negative for confidence
            # (higher confidence = lower perplexity)
            confidence = -perplexity

            return EvalResult(
                ground_truth="EVALUATED",
                method_used=self.name,
                confidence=confidence,
                details=f"Perplexity: {perplexity:.4f} (lower is better)",
            )

        except Exception as e:
            logger.error(f"Error computing perplexity: {e}")
            return EvalResult(
                ground_truth="ERROR",
                method_used=self.name,
                confidence=0.0,
                details=f"Perplexity computation failed: {str(e)}",
            )

    def _compute_perplexity(self, model, text: str) -> float:
        """Compute perplexity for text.

        Args:
            model: Model with HuggingFace interface (WisentModel or similar)
            text: Text to evaluate

        Returns:
            Perplexity value (lower is better)
        """
        # Get model and tokenizer from WisentModel
        if hasattr(model, 'hf_model') and hasattr(model, 'tokenizer'):
            hf_model = model.hf_model
            tokenizer = model.tokenizer
        else:
            # Assume model is directly a HuggingFace model
            hf_model = model
            tokenizer = getattr(model, 'tokenizer', None)
            if tokenizer is None:
                raise ValueError("Model must have a tokenizer attribute")

        # Tokenize the text
        encodings = tokenizer(text, return_tensors='pt')
        input_ids = encodings['input_ids'].to(hf_model.device)

        # Get model outputs (logits)
        with torch.no_grad():
            outputs = hf_model(input_ids)
            logits = outputs.logits

        # Shift logits and labels for next-token prediction
        # logits: [batch, seq_len, vocab_size]
        # We want to predict tokens 1..N from tokens 0..N-1
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

        # Gather the log probabilities of the actual tokens
        # shift_labels: [batch, seq_len-1]
        # We need to gather from log_probs: [batch, seq_len-1, vocab_size]
        batch_size, seq_len = shift_labels.shape
        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Compute negative log-likelihood (NLL)
        nll = -token_log_probs.sum()

        # Compute perplexity = exp(NLL / num_tokens)
        num_tokens = seq_len
        perplexity = torch.exp(nll / num_tokens)

        return float(perplexity)
