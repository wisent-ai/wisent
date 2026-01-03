"""
SAE-based steering method for comparison experiments.

Uses Sparse Autoencoders to identify steering directions from contrastive pairs.
Computes steering vector using SAE decoder features weighted by feature differences.

Supports Gemma models with Gemma Scope SAEs via sae_lens.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

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


def _get_residual_stream_activations(
    model,
    tokenizer,
    text: str,
    layer_idx: int,
    device: str,
) -> torch.Tensor:
    """
    Get residual stream activations from a specific layer.

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
    return hs[layer_idx + 1]


def compute_feature_diff(
    model,
    tokenizer,
    sae,
    pairs: list[dict],
    layer_idx: int,
    device: str,
) -> torch.Tensor:
    """
    Compute feature difference between positive and negative examples in SAE space.

    feature_diff = mean(encode(h_pos)) - mean(encode(h_neg))

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        sae: SAE object from sae_lens
        pairs: List of contrastive pairs
        layer_idx: Layer to extract activations from
        device: Device

    Returns:
        feature_diff tensor of shape [d_sae]
    """
    pos_features_list = []
    neg_features_list = []

    print(f"   Computing feature diff from {len(pairs)} pairs...")

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
        pos_features_list.append(pos_latents.mean(dim=1).detach())  # Mean over sequence

        neg_acts = _get_residual_stream_activations(model, tokenizer, neg_text, layer_idx, device)
        neg_acts = neg_acts.to(device).to(sae.W_enc.dtype)
        neg_latents = sae.encode(neg_acts)
        neg_features_list.append(neg_latents.mean(dim=1).detach())

        if (i + 1) % 10 == 0:
            print(f"      Processed {i + 1}/{len(pairs)} pairs")

    # Stack and compute mean difference
    pos_features = torch.cat(pos_features_list, dim=0)  # [num_pairs, d_sae]
    neg_features = torch.cat(neg_features_list, dim=0)

    feature_diff = pos_features.mean(dim=0) - neg_features.mean(dim=0)  # [d_sae]

    print(f"   feature_diff computed, shape: {feature_diff.shape}")
    print(f"   feature_diff stats: mean={feature_diff.mean():.6f}, std={feature_diff.std():.6f}")

    return feature_diff


def compute_steering_vector_from_decoder(
    feature_diff: torch.Tensor,
    sae,
    top_k: int = 4,
    normalize: bool = True,
) -> tuple[torch.Tensor, dict]:
    """
    Compute steering vector using SAE decoder features.

    steering_vector = sum(feature_diff[i] * W_dec[i]) for top-k features

    Args:
        feature_diff: Difference vector in SAE space [d_sae]
        sae: SAE object with W_dec decoder weights
        top_k: Number of top features to use
        normalize: Whether to normalize the final steering vector

    Returns:
        Tuple of (steering_vector [d_model], feature_info dict)
    """
    # Find top-k features by absolute difference
    abs_diff = feature_diff.abs()
    top_values, top_indices = abs_diff.topk(min(top_k, len(feature_diff)))

    print(f"   Selected top {len(top_indices)} features")
    print(f"   Top feature indices: {top_indices[:10].tolist()}...")
    print(f"   Top feature diff magnitudes: {top_values[:10].tolist()}")

    # Construct steering vector from decoder
    # W_dec shape: [d_sae, d_model]
    steering_vector = torch.zeros(sae.W_dec.shape[1], device=sae.W_dec.device, dtype=sae.W_dec.dtype)

    for feat_idx in top_indices:
        steering_vector += feature_diff[feat_idx] * sae.W_dec[feat_idx]

    if normalize:
        norm = steering_vector.norm()
        if norm > 0:
            steering_vector = steering_vector / norm
            print(f"   Normalized steering vector (original norm: {norm:.4f})")

    print(f"   steering_vector shape: {steering_vector.shape}, norm: {steering_vector.norm():.6f}")

    feature_info = {
        "top_k": top_k,
        "top_indices": top_indices.tolist(),
        "top_diff_values": [feature_diff[i].item() for i in top_indices],
    }

    return steering_vector, feature_info


def generate_steering_vector(
    task: str,
    model_name: str,
    output_path: str | Path,
    trait_label: str = "correctness",
    num_pairs: int = 50,
    method: str = "sae",
    layers: str | None = None,
    normalize: bool = True,
    device: str = "cuda:0",
    keep_intermediate: bool = False,
    top_k: int = 4,
    **kwargs,  # Accept additional kwargs for compatibility
) -> Path:
    """
    Generate a steering vector using SAE decoder features.

    Args:
        task: lm-eval task name (e.g., 'boolq', 'cb')
        model_name: HuggingFace model name (must be Gemma 2B or 9B)
        output_path: Where to save the steering vector
        trait_label: Label for the trait being steered
        num_pairs: Number of contrastive pairs to use
        method: Method name (should be 'sae')
        layers: Layer(s) to use (e.g., '12' or '10,11,12')
        normalize: Whether to normalize the steering vector
        device: Device to run on
        keep_intermediate: Whether to keep intermediate files
        top_k: Number of top SAE features to use

    Returns:
        Path to the saved steering vector
    """
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

    steering_vectors = {}
    feature_info = {}

    for layer_idx in layer_indices:
        print(f"\nStep 3: Processing layer {layer_idx}")

        # Load SAE for this layer
        sae, sparsity = load_sae(model_name, layer_idx, device=device)

        # Step 4: Compute feature difference
        print(f"\nStep 4: Computing feature diff for layer {layer_idx}...")
        feat_diff = compute_feature_diff(model, tokenizer, sae, pairs, layer_idx, device)

        # Step 5: Compute steering vector from decoder
        print(f"\nStep 5: Computing steering vector from decoder for layer {layer_idx}...")
        steering_vec, feat_info = compute_steering_vector_from_decoder(
            feat_diff, sae, top_k=top_k, normalize=normalize
        )

        steering_vectors[str(layer_idx)] = steering_vec.cpu().float().tolist()
        feature_info[str(layer_idx)] = feat_info

        # Cleanup SAE
        del sae, sparsity, feat_diff
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Cleanup model
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Cleanup temp files
    if not keep_intermediate:
        import os
        os.unlink(pairs_file)

    # Save results
    result = {
        "steering_vectors": steering_vectors,
        "layers": [str(l) for l in layer_indices],
        "model": model_name,
        "method": "sae",
        "trait_label": trait_label,
        "task": task,
        "num_pairs": len(pairs),
        "sae_config": {
            "release": config["sae_release"],
            "top_k": top_k,
            "normalize": normalize,
        },
        "feature_info": feature_info,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved SAE steering vector to {output_path}")
    return output_path
