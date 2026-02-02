"""Perplexity evaluator for language modeling benchmarks.

Used for tasks that measure language modeling performance.
"""

from typing import Any
import logging
import math
import torch

from wisent.core.evaluators.core.atoms import BaseEvaluator, EvalResult
from wisent.core.errors import ModelNotProvidedError, InvalidValueError

logger = logging.getLogger(__name__)


class PerplexityEvaluator(BaseEvaluator):
    """Evaluator using perplexity for language modeling tasks.

    Compatible with:
    - WikiText: Language modeling
    - LAMBADA: Word prediction in context
    - Any loglikelihood_rolling task

    Note: Task names are declared in extractors (via evaluator_name attribute),
    not in evaluators. This maintains proper separation of concerns.
    """

    name = "perplexity"
    description = "Perplexity evaluator for language modeling"

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
            expected: Expected continuation (for comparison)
            **kwargs:
                model: Model instance (WisentModel or similar, overrides self.model)
                context: Optional context for conditional generation
                question: Context/prompt before the continuation
                choices: List of possible continuations for comparison

        Returns:
            EvalResult with perplexity as confidence metric (lower is better)
        """
        model = kwargs.get('model', self.model)
        context = kwargs.get('context', '')
        question = kwargs.get('question', '')
        choices = kwargs.get('choices', None)

        if model is None:
            raise ModelNotProvidedError()

        try:
            # If choices are provided, compare perplexities (for contrastive evaluation)
            if choices and len(choices) >= 2:
                prompt = question if question else context
                perplexities = {}

                for choice in choices:
                    full_text = f"{prompt} {choice}".strip()
                    ppl = self._compute_perplexity(model, full_text)
                    perplexities[choice] = ppl

                # Pick the choice with lowest perplexity
                best_choice = min(perplexities, key=perplexities.get)
                best_perplexity = perplexities[best_choice]

                # Check if best choice matches expected
                is_correct = best_choice == expected
                ground_truth = "TRUTHFUL" if is_correct else "UNTRUTHFUL"

                # Calculate confidence (inverse of perplexity difference)
                perp_diff = abs(perplexities[choices[0]] - perplexities[choices[1]])
                confidence = 1.0 if perp_diff > 1.0 else perp_diff

                details = f"Predicted: '{best_choice}' (ppl={best_perplexity:.2f}), Expected: '{expected}'"

                return EvalResult(
                    ground_truth=ground_truth,
                    method_used=self.name,
                    confidence=confidence,
                    details=details,
                )
            else:
                # Single perplexity computation (original behavior)
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
                raise InvalidValueError(param_name="model.tokenizer", actual=None, expected="tokenizer attribute")

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
