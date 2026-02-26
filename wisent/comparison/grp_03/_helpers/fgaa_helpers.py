"""FGAA helpers: v_target computation, v_opt, residual stream extraction, vector generation."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from wisent.core.constants import NORM_EPS, FGAA_DENSITY_THRESHOLD, FGAA_TOP_K_POSITIVE, FGAA_TOP_K_NEGATIVE, COMPARISON_NUM_PAIRS, DISPLAY_TOP_N_MEDIUM, JSON_INDENT

__all__ = [
    "compute_v_target",
    "compute_v_opt",
    "get_residual_stream_activations",
    "generate_steering_vector",
]


def compute_v_target(
    v_diff: torch.Tensor,
    sparsity: torch.Tensor,
    model_name: str,
    bos_features_paper: dict,
    bos_features_detected: dict,
    bos_features_source: str = "detected",
    density_threshold: float = FGAA_DENSITY_THRESHOLD,
    top_k_positive: int = FGAA_TOP_K_POSITIVE,
    top_k_negative: int = FGAA_TOP_K_NEGATIVE,
) -> torch.Tensor:
    """Compute v_target by filtering v_diff with density, BOS, and top-k."""
    v_filtered = v_diff.clone()

    if sparsity is not None:
        density_mask = sparsity > density_threshold
        num_filtered = density_mask.sum().item()
        v_filtered[density_mask] = 0
        print(f"   Density filtering: zeroed {num_filtered} features (density > {density_threshold})")

    if bos_features_source == "paper":
        bos_features = bos_features_paper.get(model_name, [])
    elif bos_features_source == "detected":
        bos_features = bos_features_detected.get(model_name, [])
    else:
        bos_features = []
    if bos_features:
        for idx in bos_features:
            v_filtered[idx] = 0
        print(f"   BOS filtering: zeroed {len(bos_features)} features {bos_features}")
    else:
        print(f"   BOS filtering: no known BOS features for {model_name}")

    v_target = torch.zeros_like(v_filtered)

    if top_k_positive > 0:
        pos_values = v_filtered.clone()
        pos_values[pos_values < 0] = 0
        top_pos_values, top_pos_indices = pos_values.topk(min(top_k_positive, (pos_values > 0).sum().item()))
        v_target[top_pos_indices] = v_filtered[top_pos_indices]
        print(f"   Selected top {len(top_pos_indices)} positive features")

    if top_k_negative > 0:
        neg_values = -v_filtered.clone()
        neg_values[neg_values < 0] = 0
        top_neg_values, top_neg_indices = neg_values.topk(min(top_k_negative, (neg_values > 0).sum().item()))
        v_target[top_neg_indices] = v_filtered[top_neg_indices]
        print(f"   Selected top {len(top_neg_indices)} negative features")

    num_nonzero = (v_target != 0).sum().item()
    print(f"   v_target: {num_nonzero} non-zero features")
    return v_target


def compute_v_opt(v_target: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute v_opt using the effect approximator."""
    v_target_norm = v_target / (v_target.abs().sum() + NORM_EPS)
    Wv = W @ v_target_norm
    Wv_normalized = Wv / (Wv.norm() + NORM_EPS)
    Wb = W @ b
    Wb_normalized = Wb / (Wb.norm() + NORM_EPS)
    v_opt = Wv_normalized - Wb_normalized
    print(f"   v_opt computed, shape: {v_opt.shape}, norm: {v_opt.norm():.6f}")
    return v_opt


def get_residual_stream_activations(model, tokenizer, text: str, layer_idx: int, device: str) -> torch.Tensor:
    """Get residual stream activations from a specific layer."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    return hs[layer_idx + 1]


def generate_steering_vector(
    task: str, model_name: str, output_path: str | Path,
    sae_configs: dict, bos_features_paper: dict, bos_features_detected: dict,
    load_model_fn, load_effect_approximator_fn, load_sae_fn,
    generate_pairs_fn, compute_v_diff_fn,
    trait_label: str = "correctness", num_pairs: int = COMPARISON_NUM_PAIRS,
    layers: str | None = None, device: str = "cuda:0",
    keep_intermediate: bool = False, density_threshold: float = FGAA_DENSITY_THRESHOLD,
    top_k_positive: int = FGAA_TOP_K_POSITIVE, top_k_negative: int = FGAA_TOP_K_NEGATIVE,
    bos_features_source: str = "detected", **kwargs,
) -> Path:
    """Generate a steering vector using the FGAA method."""
    output_path = Path(output_path)
    if model_name not in sae_configs:
        raise ValueError(f"No SAE config for '{model_name}'. Supported: {list(sae_configs.keys())}")
    config = sae_configs[model_name]

    if layers is None:
        layer_indices = [config["default_layer"]]
    elif layers == "all":
        layer_indices = list(range(config["num_layers"]))
    else:
        layer_indices = [int(l.strip()) for l in layers.split(",")]

    print(f"Step 1: Generating contrastive pairs from task: {task}")
    pairs, pairs_file = generate_pairs_fn(task, num_pairs)

    print(f"\nStep 2: Loading model {model_name}...")
    model, tokenizer = load_model_fn(model_name, device)

    print(f"\nStep 3: Loading effect approximator...")
    W, b = load_effect_approximator_fn(model_name, device=device)

    steering_vectors = {}
    feature_info = {}

    for layer_idx in layer_indices:
        print(f"\nProcessing layer {layer_idx}")
        sae, sparsity = load_sae_fn(model_name, layer_idx, device=device)
        v_diff = compute_v_diff_fn(model, tokenizer, sae, pairs, layer_idx, device)
        v_target = compute_v_target(
            v_diff, sparsity, model_name, bos_features_paper, bos_features_detected,
            bos_features_source=bos_features_source, density_threshold=density_threshold,
            top_k_positive=top_k_positive, top_k_negative=top_k_negative)
        v_opt = compute_v_opt(v_target, W, b)
        steering_vectors[str(layer_idx)] = v_opt.cpu().float().tolist()

        nonzero_mask = v_target != 0
        nonzero_indices = nonzero_mask.nonzero().squeeze(-1).tolist()
        feature_info[str(layer_idx)] = {
            "num_selected_features": len(nonzero_indices) if isinstance(nonzero_indices, list) else 1,
            "selected_feature_indices": nonzero_indices[:DISPLAY_TOP_N_MEDIUM] if isinstance(nonzero_indices, list) else [nonzero_indices],
            "v_diff_stats": {"mean": v_diff.mean().item(), "std": v_diff.std().item(),
                            "min": v_diff.min().item(), "max": v_diff.max().item()},
        }
        del sae, sparsity, v_diff, v_target
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model, W, b
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if not keep_intermediate:
        import os
        os.unlink(pairs_file)

    result = {
        "steering_vectors": steering_vectors, "layers": [str(l) for l in layer_indices],
        "model": model_name, "method": "fgaa", "trait_label": trait_label,
        "task": task, "num_pairs": len(pairs),
        "fgaa_params": {"density_threshold": density_threshold, "top_k_positive": top_k_positive,
                       "top_k_negative": top_k_negative, "bos_features_source": bos_features_source},
        "feature_info": feature_info,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=JSON_INDENT)
    print(f"\nSaved FGAA steering vector to {output_path}")
    return output_path
