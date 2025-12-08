"""Log Likelihoods Evaluator for multiple choice tasks.

This evaluator handles tasks like BoolQ, MMLU, ARC where evaluation is done
by comparing log likelihoods of different answer choices rather than generating text.
Works with steering by computing log probabilities with steering applied.
"""

import logging
import torch
from typing import Any, List

from wisent.core.evaluators.core.atoms import BaseEvaluator, EvalResult
from wisent.core.errors.error_handler import (
    ModelNotProvidedError,
    validate_choices,
    require_all_parameters
)

logger = logging.getLogger(__name__)


class LogLikelihoodsEvaluator(BaseEvaluator):
    """Evaluator for multiple choice tasks using log likelihood comparison.

    Compatible with:
    - BoolQ: Boolean questions with yes/no choices
    - MMLU: Multiple choice questions
    - ARC: Science questions with multiple choices
    - Any task requiring log likelihood comparison

    This evaluator computes the log likelihood of each choice and selects
    the one with the highest probability. Can apply steering before computing
    log likelihoods.

    Note: Task names are declared in extractors (via evaluator_name attribute),
    not in evaluators. This maintains proper separation of concerns.
    """

    name = "log_likelihoods"
    description = "Log likelihood evaluator for multiple choice tasks"

    def __init__(self, model=None):
        """Initialize with optional model for log likelihood computation.

        Args:
            model: WisentModel instance that can compute log likelihoods
        """
        self.model = model

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate using log likelihood comparison of choices.

        Args:
            response: Not used for log likelihood evaluation
            expected: Expected answer
            **kwargs:
                model: WisentModel instance (REQUIRED)
                question: The question/context (REQUIRED)
                choices: List of answer choices (REQUIRED)
                steering_plan: Optional steering plan to apply

        Returns:
            EvalResult with TRUTHFUL/UNTRUTHFUL

        Raises:
            ModelNotProvidedError: If model is not provided
            MissingParameterError: If question is not provided
            InvalidChoicesError: If choices are invalid or missing
        """
        model = kwargs.get('model') or self.model
        question = kwargs.get('question')
        choices = kwargs.get('choices')
        steering_plan = kwargs.get('steering_plan')
        task_name = kwargs.get('task_name', 'unknown')

        # NO FALLBACKS - require all parameters
        if not model:
            raise ModelNotProvidedError(evaluator_name=self.name, task_name=task_name)

        require_all_parameters(
            {'question': question},
            context=f"{self.name} evaluator",
            task_name=task_name
        )

        validate_choices(choices, task_name=task_name, min_choices=2)

        return self._evaluate_log_likelihood(
            model, question, choices, expected, steering_plan
        )

    def _evaluate_log_likelihood(
        self, model, question: str, choices: List[str], expected: Any, steering_plan=None
    ) -> EvalResult:
        """Evaluate by comparing log likelihoods of choices."""
        try:
            # Apply steering if provided
            if steering_plan:
                model.apply_steering(steering_plan)

            # Check if we should use mock log probabilities for testing
            import os
            use_mock_logprobs = os.environ.get('WISENT_USE_MOCK_LOGPROBS', 'false').lower() == 'true'

            if use_mock_logprobs:
                # For framework testing: always favor the FIRST choice (assumed to be correct/positive)
                # This ensures consistent behavior regardless of what 'expected' is set to
                log_probs = []
                for i, choice in enumerate(choices):
                    if i == 0:
                        log_probs.append(-1.0)  # Highest likelihood for first choice
                    else:
                        log_probs.append(-5.0)  # Lower likelihood for other choices
            else:
                # Compute log likelihood for each choice
                log_probs = []
                for choice in choices:
                    log_prob = self._compute_choice_log_likelihood(model, question, choice)
                    log_probs.append(log_prob)

            # Detach steering
            if steering_plan:
                model.detach()

            # Select choice with highest log likelihood
            predicted_idx = log_probs.index(max(log_probs))
            predicted_choice = choices[predicted_idx]

            # Normalize expected answer for comparison
            expected_normalized = str(expected).strip().lower()
            predicted_normalized = predicted_choice.strip().lower()

            is_correct = predicted_normalized == expected_normalized

            return EvalResult(
                ground_truth="TRUTHFUL" if is_correct else "UNTRUTHFUL",
                method_used=self.name,
                confidence=1.0 if is_correct else 0.0,
                details=f"Predicted: '{predicted_choice}' (log_prob={log_probs[predicted_idx]:.3f}), Expected: '{expected}'",
                meta={
                    "predicted": predicted_choice,
                    "expected": expected,
                    "log_probs": {choice: lp for choice, lp in zip(choices, log_probs)},
                }
            )

        except Exception as e:
            logger.error(f"Error in log likelihood evaluation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # NO FALLBACK - raise the error
            raise

    def _compute_choice_log_likelihood(self, model, question: str, choice: str) -> float:
        """Compute log likelihood of a choice given a question.

        Args:
            model: WisentModel instance
            question: The question/context
            choice: The answer choice

        Returns:
            Log likelihood (higher = more likely)
        """
        # Format as: question + choice
        full_text = f"{question}\n{choice}"

        # Tokenize question and choice separately
        question_inputs = model.tokenizer(question, return_tensors="pt", add_special_tokens=True).to(model.device)
        choice_tokens = model.tokenizer(choice, return_tensors="pt", add_special_tokens=False).to(model.device)

        # Get model logits for the full sequence
        with torch.no_grad():
            # Tokenize full sequence
            full_inputs = model.tokenizer(full_text, return_tensors="pt", add_special_tokens=True).to(model.device)
            outputs = model.hf_model(**full_inputs)
            logits = outputs.logits

            # Compute log probability of the choice tokens
            # logits shape: [batch, seq_len, vocab_size]
            # We want log prob of choice tokens given question

            question_len = question_inputs.input_ids.shape[1]
            choice_len = choice_tokens.input_ids.shape[1]

            # Get logits at positions where we're predicting choice tokens
            log_prob = 0.0
            for i in range(choice_len):
                # Position in full sequence where we predict token i of choice
                # Subtract 1 because we predict the next token
                pos = question_len + i - 1
                if pos >= 0 and pos < logits.shape[1]:
                    token_logits = logits[0, pos, :]  # Logits at this position
                    token_log_probs = torch.nn.functional.log_softmax(token_logits, dim=-1)
                    # Get log prob of the actual choice token at this position
                    actual_token_id = choice_tokens.input_ids[0, i]
                    log_prob += token_log_probs[actual_token_id].item()

            # Normalize by length to avoid bias toward shorter choices
            normalized_log_prob = log_prob / max(choice_len, 1)

            return normalized_log_prob
