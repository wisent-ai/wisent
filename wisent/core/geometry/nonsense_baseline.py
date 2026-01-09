"""
Nonsense baseline for validating that metrics detect real signal.

Generate random token sequences to create a null distribution,
then compare benchmark metrics against this baseline.
"""

import torch
import numpy as np
import random
from typing import Dict, Tuple, Any


def generate_nonsense_activations(
    model,
    tokenizer,
    n_pairs: int = 50,
    layer: int = None,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate activations from random token sequences (nonsense baseline).
    
    This creates a null baseline - what activation differences look like
    when there is NO semantic concept.
    """
    vocab_size = tokenizer.vocab_size
    if layer is None:
        layer = model.config.num_hidden_layers // 2
    
    def generate_random_tokens(min_tokens=5, max_tokens=25):
        n_tokens = random.randint(min_tokens, max_tokens)
        token_ids = [random.randint(100, vocab_size - 100) for _ in range(n_tokens)]
        return tokenizer.decode(token_ids)
    
    def get_activation(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hs = outputs.hidden_states[layer + 1]
            return hs[0, -1, :]
    
    pos_acts = []
    neg_acts = []
    
    for _ in range(n_pairs):
        try:
            pos_text = generate_random_tokens()
            neg_text = generate_random_tokens()
            pos_acts.append(get_activation(pos_text))
            neg_acts.append(get_activation(neg_text))
        except Exception:
            continue
    
    if len(pos_acts) < 10:
        raise ValueError(f"Could only generate {len(pos_acts)} nonsense pairs")
    
    return torch.stack(pos_acts), torch.stack(neg_acts)


def compute_nonsense_baseline(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    nonsense_pos: torch.Tensor,
    nonsense_neg: torch.Tensor,
    n_folds: int = 5,
) -> Dict[str, float]:
    """
    Compare metrics against nonsense baseline.
    
    Returns z-scores: how many standard deviations above nonsense.
    """
    from .probe_metrics import compute_linear_probe_accuracy, compute_signal_strength
    
    real_linear = compute_linear_probe_accuracy(pos_activations, neg_activations, n_folds)
    real_signal = compute_signal_strength(pos_activations, neg_activations, n_folds)
    
    nonsense_linear = compute_linear_probe_accuracy(nonsense_pos, nonsense_neg, n_folds)
    nonsense_signal = compute_signal_strength(nonsense_pos, nonsense_neg, n_folds)
    
    linear_z = (real_linear - nonsense_linear) / 0.1  # Assume std ~0.1
    signal_z = (real_signal - nonsense_signal) / 0.1
    
    return {
        "real_linear_probe": real_linear,
        "real_signal_strength": real_signal,
        "nonsense_linear_probe": nonsense_linear,
        "nonsense_signal_strength": nonsense_signal,
        "linear_z_score": float(linear_z),
        "signal_z_score": float(signal_z),
        "is_significant": linear_z > 2.0 or signal_z > 2.0,
    }


def analyze_with_nonsense_baseline(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    model=None,
    tokenizer=None,
    layer: int = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Full analysis comparing real data to nonsense baseline.
    """
    from .probe_metrics import (
        compute_linear_probe_accuracy,
        compute_signal_strength,
        compute_mlp_probe_accuracy,
    )
    from .icd import compute_icd
    
    real_metrics = {
        "linear_probe": compute_linear_probe_accuracy(pos_activations, neg_activations),
        "signal_strength": compute_signal_strength(pos_activations, neg_activations),
        "mlp_probe": compute_mlp_probe_accuracy(pos_activations, neg_activations),
        "icd": compute_icd(pos_activations, neg_activations)["icd"],
    }
    
    if model is not None and tokenizer is not None:
        try:
            nonsense_pos, nonsense_neg = generate_nonsense_activations(
                model, tokenizer, n_pairs=50, layer=layer, device=device
            )
            
            nonsense_metrics = {
                "linear_probe": compute_linear_probe_accuracy(nonsense_pos, nonsense_neg),
                "signal_strength": compute_signal_strength(nonsense_pos, nonsense_neg),
                "mlp_probe": compute_mlp_probe_accuracy(nonsense_pos, nonsense_neg),
                "icd": compute_icd(nonsense_pos, nonsense_neg)["icd"],
            }
            
            z_scores = {}
            for key in real_metrics:
                diff = real_metrics[key] - nonsense_metrics[key]
                z_scores[f"{key}_z"] = diff / 0.1
            
            return {
                "real": real_metrics,
                "nonsense": nonsense_metrics,
                "z_scores": z_scores,
                "is_real_signal": z_scores.get("linear_probe_z", 0) > 2.0,
            }
        except Exception as e:
            return {
                "real": real_metrics,
                "nonsense": None,
                "error": str(e),
            }
    
    return {"real": real_metrics, "nonsense": None}
