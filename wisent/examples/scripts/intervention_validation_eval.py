"""Evaluation helpers for intervention_validation."""

from typing import List, Tuple

import torch
import numpy as np

from wisent.examples.scripts.intervention_validation_helpers import (
    apply_steering_to_model,
)


def get_model_logprobs(
    model: "WisentModel",
    prompt: str,
    completion: str,
) -> float:
    """
    Get log probability of completion given prompt.
    
    Args:
        model: WisentModel instance
        prompt: Input prompt
        completion: Completion to score
        
    Returns:
        Average log probability of completion tokens
    """
    full_text = prompt + completion
    
    inputs = model.tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)
    
    prompt_tokens = model.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.hf_model(**inputs)
        logits = outputs.logits
    
    # Get logprobs for completion tokens only
    shift_logits = logits[:, prompt_tokens-1:-1, :].contiguous()
    shift_labels = inputs.input_ids[:, prompt_tokens:].contiguous()
    
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    return token_log_probs.mean().item()


def evaluate_steering(
    model: "WisentModel",
    test_pairs: List,
    layer: int,
    direction: torch.Tensor,
    coefficient: float,
) -> Tuple[float, float, float]:
    """
    Evaluate steering effect on test pairs using WisentModel's built-in steering.
    
    Args:
        model: WisentModel instance
        test_pairs: List of ContrastivePair objects
        layer: Layer to apply steering
        direction: Steering direction
        coefficient: Steering strength
        
    Returns:
        (accuracy, avg_correct_logprob, avg_incorrect_logprob)
    """
    # Apply steering using WisentModel's built-in method
    apply_steering_to_model(model, layer, direction, coefficient)
    
    try:
        correct = 0
        correct_logprobs = []
        incorrect_logprobs = []
        
        for pair in test_pairs:
            prompt = pair.prompt
            correct_completion = pair.positive_response.model_response
            incorrect_completion = pair.negative_response.model_response
            
            correct_lp = get_model_logprobs(model, prompt, correct_completion)
            incorrect_lp = get_model_logprobs(model, prompt, incorrect_completion)
            
            correct_logprobs.append(correct_lp)
            incorrect_logprobs.append(incorrect_lp)
            
            if correct_lp > incorrect_lp:
                correct += 1
        
        accuracy = correct / len(test_pairs) if test_pairs else 0.0
        avg_correct = np.mean(correct_logprobs) if correct_logprobs else 0.0
        avg_incorrect = np.mean(incorrect_logprobs) if incorrect_logprobs else 0.0
        
        return accuracy, avg_correct, avg_incorrect
    
    finally:
        # Remove steering
        model.detach()


def evaluate_baseline(
    model: "WisentModel",
    test_pairs: List,
) -> Tuple[float, float, float]:
    """
    Evaluate baseline (no steering) on test pairs.
    
    Args:
        model: WisentModel instance
        test_pairs: List of ContrastivePair objects
        
    Returns:
        (accuracy, avg_correct_logprob, avg_incorrect_logprob)
    """
    correct = 0
    correct_logprobs = []
    incorrect_logprobs = []
    
    for pair in test_pairs:
        prompt = pair.prompt
        correct_completion = pair.positive_response.model_response
        incorrect_completion = pair.negative_response.model_response
        
        correct_lp = get_model_logprobs(model, prompt, correct_completion)
        incorrect_lp = get_model_logprobs(model, prompt, incorrect_completion)
        
        correct_logprobs.append(correct_lp)
        incorrect_logprobs.append(incorrect_lp)
        
        if correct_lp > incorrect_lp:
            correct += 1
    
    accuracy = correct / len(test_pairs) if test_pairs else 0.0
    avg_correct = np.mean(correct_logprobs) if correct_logprobs else 0.0
    avg_incorrect = np.mean(incorrect_logprobs) if incorrect_logprobs else 0.0
    
    return accuracy, avg_correct, avg_incorrect

