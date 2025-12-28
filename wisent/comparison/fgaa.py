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

if TYPE_CHECKING:
    from wisent.core.models.wisent_model import WisentModel

__all__ = ["generate_steering_vector", "apply_steering_to_model", "remove_steering", "convert_to_lm_eval_format"]


# BOS feature indices - these features activate most strongly on the BOS token
# Paper features from Appendix G (5 features)
BOS_FEATURES_PAPER = {
    "google/gemma-2-2b": [11087, 3220, 11752, 12160, 11498],
    "google/gemma-2-9b": [],  # Not listed in paper
}

# Detected features from running detect_bos_features.py (top 12 by mean activation)
BOS_FEATURES_DETECTED = {
    "google/gemma-2-2b": [1041, 7507, 11087, 3220, 11767, 11752, 14669, 6889, 12160, 13700, 2747, 11498],
    "google/gemma-2-9b": [8032, 11906, 7768, 14845, 14483, 10562, 8892, 9151, 5721, 15738, 5285, 13895],
}

# FGAA-specific: effect approximator config (adapter files)
FGAA_ADAPTER_FILES = {
    "google/gemma-2-2b": "adapter_2b_layer_12.pt",
    "google/gemma-2-9b": "adapter_9b_layer_12.pt",
}


def load_effect_approximator(model_name: str, device: str = "cuda:0") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load the pre-trained effect approximator (adapter) from HuggingFace.

    The adapter contains:
    - W: [d_model, d_sae] - maps SAE feature space to model activation space
    - b: [d_sae] - bias term

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

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        sae: SAE object from sae_lens
        pairs: List of contrastive pairs
        layer_idx: Layer to extract activations from
        device: Device

    Returns:
        v_diff tensor of shape [d_sae]
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
        # SAE encode: latents = (x - b_dec) @ W_enc + b_enc
        pos_latents = sae.encode(pos_acts)
        # Mean over sequence dimension
        pos_features_list.append(pos_latents.mean(dim=1).detach())  # [1, d_sae]

        neg_acts = _get_residual_stream_activations(model, tokenizer, neg_text, layer_idx, device)
        neg_acts = neg_acts.to(device).to(sae.W_enc.dtype)
        neg_latents = sae.encode(neg_acts)
        neg_features_list.append(neg_latents.mean(dim=1).detach())

        if (i + 1) % 10 == 0:
            print(f"      Processed {i + 1}/{len(pairs)} pairs")

    # Stack and compute mean
    pos_features = torch.cat(pos_features_list, dim=0)  # [num_pairs, d_sae]
    neg_features = torch.cat(neg_features_list, dim=0)

    v_diff = pos_features.mean(dim=0) - neg_features.mean(dim=0)  # [d_sae]

    print(f"   v_diff computed, shape: {v_diff.shape}")
    print(f"   v_diff stats: mean={v_diff.mean():.6f}, std={v_diff.std():.6f}, "
          f"min={v_diff.min():.6f}, max={v_diff.max():.6f}")

    return v_diff


def compute_v_target(
    v_diff: torch.Tensor,
    sparsity: torch.Tensor,
    model_name: str,
    bos_features_source: str = "detected",
    density_threshold: float = 0.01,
    top_k_positive: int = 50,
    top_k_negative: int = 0,
) -> torch.Tensor:
    """
    Compute v_target by filtering v_diff.

    Three filtering stages:
    1. Density filtering: zero out features with activation density > threshold
    2. BOS token filtering: zero out features that activate mainly on BOS token
    3. Top-k selection: keep top positive and negative features

    Args:
        v_diff: Difference vector in SAE space [d_sae]
        sparsity: Feature sparsity/density values from SAE [d_sae]
        model_name: Model name to look up BOS features
        bos_features_source: Source of BOS features - "paper" (5 features), "detected" (12 features), or "none"
        density_threshold: Zero out features with density above this (default 0.01)
        top_k_positive: Number of top positive features to keep
        top_k_negative: Number of top negative features to keep (paper uses 0)

    Returns:
        v_target tensor of shape [d_sae]
    """
    v_filtered = v_diff.clone()

    # Stage 1: Density filtering
    # Zero out features that are too commonly activated (not specific enough)
    if sparsity is not None:
        density_mask = sparsity > density_threshold
        num_filtered = density_mask.sum().item()
        v_filtered[density_mask] = 0
        print(f"   Density filtering: zeroed {num_filtered} features (density > {density_threshold})")

    # Stage 2: BOS filtering
    # Zero out features that activate mainly on BOS tokens
    if bos_features_source == "paper":
        bos_features = BOS_FEATURES_PAPER.get(model_name, [])
    elif bos_features_source == "detected":
        bos_features = BOS_FEATURES_DETECTED.get(model_name, [])
    else:  # "none"
        bos_features = []
    if bos_features:
        for idx in bos_features:
            v_filtered[idx] = 0
        print(f"   BOS filtering: zeroed {len(bos_features)} features {bos_features}")
    else:
        print(f"   BOS filtering: no known BOS features for {model_name}")

    # Stage 3: Top-k selection
    v_target = torch.zeros_like(v_filtered)

    # Get top positive features
    if top_k_positive > 0:
        pos_values = v_filtered.clone()
        pos_values[pos_values < 0] = 0
        top_pos_values, top_pos_indices = pos_values.topk(min(top_k_positive, (pos_values > 0).sum().item()))
        v_target[top_pos_indices] = v_filtered[top_pos_indices]
        print(f"   Selected top {len(top_pos_indices)} positive features")

    # Get top negative features (paper uses 0)
    if top_k_negative > 0:
        neg_values = -v_filtered.clone()
        neg_values[neg_values < 0] = 0
        top_neg_values, top_neg_indices = neg_values.topk(min(top_k_negative, (neg_values > 0).sum().item()))
        v_target[top_neg_indices] = v_filtered[top_neg_indices]
        print(f"   Selected top {len(top_neg_indices)} negative features")

    num_nonzero = (v_target != 0).sum().item()
    print(f"   v_target: {num_nonzero} non-zero features")

    return v_target


def compute_v_opt(
    v_target: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Compute v_opt using the effect approximator.

    From paper: v_opt = (W @ v_target_norm) / ||W @ v_target_norm|| - (W @ b) / ||W @ b||

    Args:
        v_target: Target vector in SAE space [d_sae]
        W: Effect approximator weight matrix [d_model, d_sae]
        b: Effect approximator bias [d_sae]

    Returns:
        v_opt tensor of shape [d_model]
    """
    # L1 normalize v_target (as specified in paper)
    v_target_norm = v_target / (v_target.abs().sum() + 1e-8)

    # W is [d_model, d_sae], v_target_norm is [d_sae]
    # W @ v_target_norm -> [d_model]
    Wv = W @ v_target_norm
    Wv_normalized = Wv / (Wv.norm() + 1e-8)

    # Bias term: W @ b -> [d_model]
    Wb = W @ b
    Wb_normalized = Wb / (Wb.norm() + 1e-8)

    # Final v_opt (paper formula)
    v_opt = Wv_normalized - Wb_normalized

    print(f"   v_opt computed, shape: {v_opt.shape}, norm: {v_opt.norm():.6f}")

    return v_opt


def _get_residual_stream_activations(
    model,
    tokenizer,
    text: str,
    layer_idx: int,
    device: str,
) -> torch.Tensor:
    """
    Get residual stream activations from a specific layer.

    Uses output_hidden_states=True (same as wisent's ActivationCollector).

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        text: Input text
        layer_idx: Layer index (0-indexed)
        device: Device

    Returns:
        Tensor of shape (1, seq_len, d_model)
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)

    # hidden_states is tuple: (embedding, layer0, layer1, ..., layerN)
    # layer_idx=0 -> hs[1], layer_idx=12 -> hs[13]
    hs = out.hidden_states
    return hs[layer_idx + 1]  # +1 because hs[0] is embedding layer


def generate_steering_vector(
    task: str,
    model_name: str,
    output_path: str | Path,
    trait_label: str = "correctness",
    num_pairs: int = 50,
    method: str = "fgaa",
    layers: str | None = None,
    device: str = "cuda:0",
    keep_intermediate: bool = False,
    density_threshold: float = 0.01,
    top_k_positive: int = 50,
    top_k_negative: int = 0,
    bos_features_source: str = "detected",
    **kwargs,  # Accept additional kwargs for compatibility (e.g., extraction_strategy)
) -> Path:
    """
    Generate a steering vector using the FGAA method.

    Args:
        task: lm-eval task name (e.g., 'boolq', 'cb')
        model_name: HuggingFace model name (must be Gemma 2B or 9B)
        output_path: Where to save the steering vector
        trait_label: Label for the trait being steered
        num_pairs: Number of contrastive pairs to use
        method: Method name (should be 'fgaa')
        layers: Layer(s) to use (e.g., '12' or '10,11,12')
        device: Device to run on
        keep_intermediate: Whether to keep intermediate files
        density_threshold: Density threshold for filtering (default 0.01)
        top_k_positive: Number of top positive features to keep
        top_k_negative: Number of top negative features to keep
        bos_features_source: Source of BOS features - "paper" (5), "detected" (12), or "none"

    Returns:
        Path to the saved steering vector
    """
    import gc

    output_path = Path(output_path)

    if model_name not in SAE_CONFIGS:
        raise ValueError(
            f"No SAE config for model '{model_name}'. "
            f"Supported models: {list(SAE_CONFIGS.keys())}"
        )

    config = SAE_CONFIGS[model_name]

    # Parse layers
    if layers is None:
        layer_indices = [config["default_layer"]]
    elif layers == "all":
        layer_indices = list(range(config["num_layers"]))
    else:
        layer_indices = [int(l.strip()) for l in layers.split(",")]

    # Step 1: Generate contrastive pairs
    print(f"Step 1: Generating contrastive pairs from task: {task}")
    pairs, pairs_file = generate_contrastive_pairs(task, num_pairs)
    print(f"   Loaded {len(pairs)} contrastive pairs")

    # Step 2: Load model
    print(f"\nStep 2: Loading model {model_name}...")
    model, tokenizer = load_model_and_tokenizer(model_name, device)

    # Step 3: Load effect approximator (shared across layers)
    print(f"\nStep 3: Loading effect approximator...")
    W, b = load_effect_approximator(model_name, device=device)

    steering_vectors = {}
    feature_info = {}

    for layer_idx in layer_indices:
        print(f"\nStep 4: Processing layer {layer_idx}")

        # Load SAE for this layer
        sae, sparsity = load_sae(model_name, layer_idx, device=device)

        # Compute v_diff
        print(f"\nStep 5: Computing v_diff for layer {layer_idx}...")
        v_diff = compute_v_diff(model, tokenizer, sae, pairs, layer_idx, device)

        # Compute v_target
        print(f"\nStep 6: Computing v_target for layer {layer_idx}...")
        v_target = compute_v_target(
            v_diff,
            sparsity,
            model_name,
            bos_features_source=bos_features_source,
            density_threshold=density_threshold,
            top_k_positive=top_k_positive,
            top_k_negative=top_k_negative,
        )

        # Compute v_opt
        print(f"\nStep 7: Computing v_opt for layer {layer_idx}...")
        v_opt = compute_v_opt(v_target, W, b)

        steering_vectors[str(layer_idx)] = v_opt.cpu().float().tolist()

        # Store feature info
        nonzero_mask = v_target != 0
        nonzero_indices = nonzero_mask.nonzero().squeeze(-1).tolist()
        feature_info[str(layer_idx)] = {
            "num_selected_features": len(nonzero_indices) if isinstance(nonzero_indices, list) else 1,
            "selected_feature_indices": nonzero_indices[:20] if isinstance(nonzero_indices, list) else [nonzero_indices],
            "v_diff_stats": {
                "mean": v_diff.mean().item(),
                "std": v_diff.std().item(),
                "min": v_diff.min().item(),
                "max": v_diff.max().item(),
            },
        }

        # Cleanup SAE
        del sae, sparsity, v_diff, v_target
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Cleanup
    del model, W, b
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if not keep_intermediate:
        import os
        os.unlink(pairs_file)

    # Save results
    result = {
        "steering_vectors": steering_vectors,
        "layers": [str(l) for l in layer_indices],
        "model": model_name,
        "method": "fgaa",
        "trait_label": trait_label,
        "task": task,
        "num_pairs": len(pairs),
        "fgaa_params": {
            "density_threshold": density_threshold,
            "top_k_positive": top_k_positive,
            "top_k_negative": top_k_negative,
            "bos_features_source": bos_features_source,
        },
        "feature_info": feature_info,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved FGAA steering vector to {output_path}")
    return output_path
