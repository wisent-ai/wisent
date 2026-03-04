"""
FGAA (Feature Guided Activation Addition) steering method.

Implements the method from "Interpretable Steering of Large Language Models
with Feature Guided Activation Additions" (arXiv:2501.09929).

Uses Gemma Scope SAEs and pre-computed effect approximators.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from huggingface_hub import hf_hub_download

from wisent.comparison.utils import (
    apply_steering_to_model,
    remove_steering,
    convert_to_lm_eval_format,
    generate_contrastive_pairs,
    load_model_and_tokenizer,
    load_sae,
    SAE_CONFIGS,
)
from wisent.comparison._helpers.fgaa_helpers import (
    compute_v_target,
    compute_v_opt,
    get_residual_stream_activations as _get_residual_stream_activations,
)
from wisent.core.utils.config_tools.constants import (
    GEMMA_2B_BOS_FEATURES_PAPER,
    GEMMA_2B_BOS_FEATURES_DETECTED,
    GEMMA_9B_BOS_FEATURES_DETECTED,
    PROGRESS_LOG_INTERVAL_10,
)

if TYPE_CHECKING:
    from wisent.core.primitives.models.wisent_model import WisentModel

__all__ = ["generate_steering_vector", "apply_steering_to_model", "remove_steering", "convert_to_lm_eval_format"]


# BOS feature indices - these features activate most strongly on the BOS token
# Paper features from Appendix G (5 features)
BOS_FEATURES_PAPER = {
    "google/gemma-2-2b": list(GEMMA_2B_BOS_FEATURES_PAPER),
    "google/gemma-2-9b": [],  # Not listed in paper
}

# Detected features from running detect_bos_features.py (top 12 by mean activation)
BOS_FEATURES_DETECTED = {
    "google/gemma-2-2b": list(GEMMA_2B_BOS_FEATURES_DETECTED),
    "google/gemma-2-9b": list(GEMMA_9B_BOS_FEATURES_DETECTED),
}

# FGAA-specific: effect approximator config (adapter files)
FGAA_ADAPTER_FILES = {
    "google/gemma-2-2b": "adapter_2b_layer_12.pt",
    "google/gemma-2-9b": "adapter_9b_layer_12.pt",
}


def load_effect_approximator(model_name: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load the pre-trained effect approximator (adapter) from HuggingFace.

    Args:
        model_name: HuggingFace model name
        device: Device to load on

    Returns:
        Tuple of (W, b) tensors
    """
    if model_name not in FGAA_ADAPTER_FILES:
        raise ValueError(f"No effect approximator for model '{model_name}'")

    adapter_file = FGAA_ADAPTER_FILES[model_name]

    print(f"   Loading adapter from schalnev/sae-ts-effects / {adapter_file}")
    path = hf_hub_download(
        repo_id="schalnev/sae-ts-effects",
        filename=adapter_file,
        repo_type="dataset",
    )

    adapter = torch.load(path, map_location=device, weights_only=False)

    # Adapter is OrderedDict with 'W' and 'b'
    W = adapter["W"].to(device)  # [d_model, d_sae]
    b = adapter["b"].to(device)  # [d_sae]

    print(f"   Adapter W shape: {W.shape}, b shape: {b.shape}")

    return W, b


def compute_v_diff(
    model,
    tokenizer,
    sae,
    pairs: list[dict],
    layer_idx: int,
    device: str,
) -> torch.Tensor:
    """
    Compute v_diff: the difference vector between positive and negative examples in SAE space.

    v_diff = mean(f(h_l(x+))) - mean(f(h_l(x-)))
    """
    pos_features_list = []
    neg_features_list = []

    print(f"   Computing v_diff from {len(pairs)} pairs...")

    for i, pair in enumerate(pairs):
        prompt = pair["prompt"]
        pos_response = pair["positive_response"]["model_response"]
        neg_response = pair["negative_response"]["model_response"]

        pos_text = f"{prompt} {pos_response}"
        neg_text = f"{prompt} {neg_response}"

        # Get activations and encode through SAE
        pos_acts = _get_residual_stream_activations(model, tokenizer, pos_text, layer_idx, device)
        pos_acts = pos_acts.to(device).to(sae.W_enc.dtype)
        pos_latents = sae.encode(pos_acts)
        pos_features_list.append(pos_latents.mean(dim=1).detach())

        neg_acts = _get_residual_stream_activations(model, tokenizer, neg_text, layer_idx, device)
        neg_acts = neg_acts.to(device).to(sae.W_enc.dtype)
        neg_latents = sae.encode(neg_acts)
        neg_features_list.append(neg_latents.mean(dim=1).detach())

        if (i + 1) % PROGRESS_LOG_INTERVAL_10 == 0:
            print(f"      Processed {i + 1}/{len(pairs)} pairs")

    pos_features = torch.cat(pos_features_list, dim=0)
    neg_features = torch.cat(neg_features_list, dim=0)

    v_diff = pos_features.mean(dim=0) - neg_features.mean(dim=0)

    print(f"   v_diff computed, shape: {v_diff.shape}")
    print(f"   v_diff stats: mean={v_diff.mean():.6f}, std={v_diff.std():.6f}, "
          f"min={v_diff.min():.6f}, max={v_diff.max():.6f}")

    return v_diff


def generate_steering_vector(
    task: str,
    model_name: str,
    output_path: str | Path,
    bos_features_source: str,
    trait_label: str,
    method: str,
    device: str,
    num_pairs: int,
    layers: str,
    density_threshold: float,
    top_k_positive: int,
    top_k_negative: int,
    *,
    keep_intermediate: bool = False,
    **kwargs,
) -> Path:
    """Generate a steering vector using the FGAA method."""
    from wisent.comparison._helpers.fgaa_helpers import generate_steering_vector as _gen_sv
    return _gen_sv(
        task=task, model_name=model_name, output_path=output_path,
        sae_configs=SAE_CONFIGS, bos_features_paper=BOS_FEATURES_PAPER,
        bos_features_detected=BOS_FEATURES_DETECTED,
        load_model_fn=load_model_and_tokenizer,
        load_effect_approximator_fn=load_effect_approximator,
        load_sae_fn=load_sae, generate_pairs_fn=generate_contrastive_pairs,
        compute_v_diff_fn=compute_v_diff,
        trait_label=trait_label, num_pairs=num_pairs, layers=layers,
        device=device, keep_intermediate=keep_intermediate,
        density_threshold=density_threshold, top_k_positive=top_k_positive,
        top_k_negative=top_k_negative, bos_features_source=bos_features_source,
        **kwargs,
    )
