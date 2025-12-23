"""
Detect BOS (Beginning of Sequence) features in Gemma Scope SAEs.

BOS features are SAE features that activate most strongly at the BOS token position.
These should be filtered out when computing steering vectors as they introduce
artifacts without contributing to steering.

Reference: "Interpretable Steering of Large Language Models with Feature Guided
Activation Additions" (arXiv:2501.09929), Appendix G.

Known BOS features from paper (Gemma-2-2B, layer 12, 16k SAE):
- 11087, 3220, 11752, 12160, 11498

Usage:
    python -m wisent.comparison.detect_bos_features --model google/gemma-2-2b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm


# Known BOS feature indices from paper (Appendix G)
KNOWN_BOS_FEATURES = {
    "google/gemma-2-2b": [11087, 3220, 11752, 12160, 11498],
    "google/gemma-2-9b": [],  # Not listed in paper
}


def load_sample_texts(num_samples: int = 2000, min_length: int = 50) -> list[str]:
    """Load sample texts from WikiText dataset."""
    print(f"Loading up to {num_samples} sample texts from WikiText...")

    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")

    texts = []
    for item in dataset:
        if len(texts) >= num_samples:
            break
        text = item["text"].strip()
        if len(text) >= min_length:
            texts.append(text)

    print(f"   Loaded {len(texts)} texts")
    return texts


def detect_bos_features(
    model,
    tokenizer,
    sae,
    layer_idx: int,
    device: str,
    texts: list[str],
    top_k: int = 10,
    batch_size: int = 8,
) -> tuple[list[int], dict[str, torch.Tensor]]:
    """
    Detect BOS features by finding features that activate most strongly at position 0.

    Computes statistics (mean, variance, median) of activation at BOS position for each
    SAE feature across all samples, then returns the top-k features with highest mean
    BOS activation.

    Reference: FGAA paper (arXiv:2501.09929) identifies BOS features as those that
    "exclusively had the strongest activation on the BOS token".

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        sae: SAE object from sae_lens
        layer_idx: Layer index (0-indexed)
        device: Device
        texts: List of sample texts to analyze
        top_k: Number of top BOS features to return (default 10)
        batch_size: Batch size for processing

    Returns:
        Tuple of (top_bos_feature_indices, stats_dict) where stats_dict contains
        'mean', 'variance', and 'median' tensors of shape [d_sae].
    """
    d_sae = sae.cfg.d_sae

    # Collect all BOS activations on CPU for stable statistics computation
    all_bos_activations = []

    print(f"Detecting BOS features from {len(texts)} samples...")
    print(f"   Layer: {layer_idx}, d_sae: {d_sae}")

    # Use hook to capture only the layer we need (not all 26 layers)
    captured_acts = {}
    def capture_hook(module, input, output):
        captured_acts["hidden"] = output[0].detach()

    # Register hook on the specific layer
    target_layer = model.model.layers[layer_idx]
    hook_handle = target_layer.register_forward_hook(capture_hook)

    try:
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing"):
            batch_texts = texts[i:i + batch_size]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                # Use model.model to skip lm_head (saves ~66MB per forward)
                model.model(**inputs, use_cache=False)

            # Get captured hidden states (only this layer, not all 26)
            acts = captured_acts["hidden"].to(sae.W_enc.dtype)
            latents = sae.encode(acts)

            attention_mask = inputs.get("attention_mask")
            for j in range(latents.shape[0]):
                seq_len = int(attention_mask[j].sum().item()) if attention_mask is not None else latents.shape[1]
                sample_latents = latents[j, :seq_len, :]  # [seq_len, d_sae]

                # Collect BOS activation (position 0) - move to CPU immediately
                bos_act = sample_latents[0].float().cpu()
                all_bos_activations.append(bos_act)
    finally:
        hook_handle.remove()

    # Compute statistics (all on CPU)
    # Stack all activations for stable computation
    all_bos_tensor = torch.stack(all_bos_activations, dim=0)  # [num_samples, d_sae]
    mean_bos = all_bos_tensor.mean(dim=0)
    variance_bos = all_bos_tensor.var(dim=0)
    median_bos = all_bos_tensor.median(dim=0).values

    stats = {
        "mean": mean_bos,
        "variance": variance_bos,
        "median": median_bos,
    }

    # Select top features by BOS activation
    top_indices = mean_bos.topk(top_k).indices.tolist()

    print(f"\nDetected top {top_k} BOS features by mean activation")
    return top_indices, stats


def compare_with_known(model_name: str, detected: list[int], stats: dict[str, torch.Tensor]) -> None:
    """Compare detected BOS features with known list from paper."""
    mean_bos = stats["mean"]
    variance_bos = stats["variance"]
    median_bos = stats["median"]

    known = KNOWN_BOS_FEATURES.get(model_name, [])
    detected_set = set(detected)
    known_set = set(known)

    print(f"\n{'='*60}")
    print(f"BOS Feature Comparison for {model_name}")
    print(f"{'='*60}")
    print(f"Known (paper): {sorted(known_set)}")
    print(f"Detected:      {sorted(detected_set)}")
    print(f"{'='*60}")
    print(f"Common:        {sorted(detected_set & known_set)}")
    print(f"Only in paper: {sorted(known_set - detected_set)}")
    print(f"Only detected: {sorted(detected_set - known_set)}")

    if known:
        print(f"\nKnown features - BOS activation stats:")
        for idx in sorted(known):
            mean_val = mean_bos[idx].item()
            var_val = variance_bos[idx].item()
            median_val = median_bos[idx].item()
            status = "detected" if idx in detected_set else "missed"
            print(f"   Feature {idx}: mean={mean_val:.4f}, var={var_val:.4f}, median={median_val:.4f} ({status})")

    print(f"\nTop 20 features by mean BOS activation:")
    for rank, idx in enumerate(mean_bos.topk(20).indices.tolist(), 1):
        mean_val = mean_bos[idx].item()
        var_val = variance_bos[idx].item()
        median_val = median_bos[idx].item()
        marker = " (known)" if idx in known_set else ""
        print(f"   {rank:2}. Feature {idx}: mean={mean_val:.4f}, var={var_val:.4f}, median={median_val:.4f}{marker}")


def main():
    parser = argparse.ArgumentParser(description="Detect BOS features in Gemma Scope SAEs")
    parser.add_argument("--model", default="google/gemma-2-2b", help="Model name")
    parser.add_argument("--layer", type=int, default=12, help="Layer index")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of text samples")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top BOS features to detect")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--output-dir", default="wisent/comparison/results", help="Output directory")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Layer: {args.layer}")
    print(f"Top-k: {args.top_k}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    print(f"\nLoading SAE...")
    from sae_lens import SAE

    release = "gemma-scope-2b-pt-res-canonical" if "2b" in args.model.lower() else "gemma-scope-9b-pt-res-canonical"
    sae_id = f"layer_{args.layer}/width_16k/canonical"
    sae, _, _ = SAE.from_pretrained(release=release, sae_id=sae_id, device=args.device)

    texts = load_sample_texts(args.num_samples)

    bos_features, stats = detect_bos_features(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        layer_idx=args.layer,
        device=args.device,
        texts=texts,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )

    compare_with_known(args.model, bos_features, stats)

    # Build detected features with their stats
    detected_with_stats = [
        {
            "feature": idx,
            "mean_bos_activation": stats["mean"][idx].item(),
            "variance_bos_activation": stats["variance"][idx].item(),
            "median_bos_activation": stats["median"][idx].item(),
        }
        for idx in bos_features
    ]

    output = {
        "model": args.model,
        "layer": args.layer,
        "top_k": args.top_k,
        "num_samples": len(texts),
        "detected_bos_features": detected_with_stats,
        "known_bos_features": KNOWN_BOS_FEATURES.get(args.model, []),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"bos_features_{args.model.replace('/', '_')}_layer{args.layer}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
