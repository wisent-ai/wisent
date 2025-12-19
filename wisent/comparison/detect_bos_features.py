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

import torch
from datasets import load_dataset
from tqdm import tqdm


# Known BOS feature indices from paper (Appendix G)
KNOWN_BOS_FEATURES = {
    "google/gemma-2-2b": [11087, 3220, 11752, 12160, 11498],
    "google/gemma-2-9b": [],  # Not listed in paper
}


def load_sample_texts(num_samples: int = 5000, min_length: int = 50) -> list[str]:
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
    threshold: float = 0.5,
    batch_size: int = 8,
) -> tuple[list[int], torch.Tensor]:
    """
    Detect BOS features by finding features that activate most strongly at position 0.

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        sae: SAE object from sae_lens
        layer_idx: Layer index (0-indexed)
        device: Device
        texts: List of sample texts to analyze
        threshold: Fraction of samples where max must be at pos 0 to be BOS feature
        batch_size: Batch size for processing

    Returns:
        Tuple of (bos_feature_indices, bos_ratios_tensor)
    """
    d_sae = sae.cfg.d_sae
    bos_max_count = torch.zeros(d_sae, device=device)
    total_processed = 0

    print(f"Detecting BOS features from {len(texts)} samples...")
    print(f"   Layer: {layer_idx}, d_sae: {d_sae}")

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
            out = model(**inputs, output_hidden_states=True, use_cache=False)

        hs = out.hidden_states
        acts = hs[layer_idx + 1].to(sae.W_enc.dtype)
        latents = sae.encode(acts)

        attention_mask = inputs.get("attention_mask")
        for j in range(latents.shape[0]):
            seq_len = int(attention_mask[j].sum().item()) if attention_mask is not None else latents.shape[1]
            sample_latents = latents[j, :seq_len, :]
            max_positions = sample_latents.argmax(dim=0)
            bos_max_count += (max_positions == 0).float()
            total_processed += 1

    bos_ratio = bos_max_count / total_processed
    bos_features = (bos_ratio >= threshold).nonzero().squeeze(-1).tolist()

    if isinstance(bos_features, int):
        bos_features = [bos_features]

    print(f"\nDetected {len(bos_features)} BOS features (threshold={threshold})")
    return bos_features, bos_ratio


def compare_with_known(model_name: str, detected: list[int], bos_ratios: torch.Tensor) -> None:
    """Compare detected BOS features with known list from paper."""
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
    print(f"Only detected: {sorted(detected_set - known_set)}")
    print(f"Only in paper: {sorted(known_set - detected_set)}")

    if known:
        print(f"\nBOS ratios for known features:")
        for idx in sorted(known):
            ratio = bos_ratios[idx].item()
            status = "detected" if idx in detected_set else "missed"
            print(f"   Feature {idx}: {ratio:.3f} ({status})")

    print(f"\nTop 10 features by BOS ratio:")
    for idx in bos_ratios.topk(10).indices.tolist():
        ratio = bos_ratios[idx].item()
        marker = " (known)" if idx in known_set else ""
        print(f"   Feature {idx}: {ratio:.3f}{marker}")


def main():
    parser = argparse.ArgumentParser(description="Detect BOS features in Gemma Scope SAEs")
    parser.add_argument("--model", default="google/gemma-2-2b", help="Model name")
    parser.add_argument("--layer", type=int, default=12, help="Layer index")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of text samples")
    parser.add_argument("--threshold", type=float, default=0.5, help="BOS detection threshold")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default="cuda:0", help="Device")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Layer: {args.layer}")
    print(f"Threshold: {args.threshold}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
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

    bos_features, bos_ratios = detect_bos_features(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        layer_idx=args.layer,
        device=args.device,
        texts=texts,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )

    compare_with_known(args.model, bos_features, bos_ratios)

    output = {
        "model": args.model,
        "layer": args.layer,
        "threshold": args.threshold,
        "num_samples": len(texts),
        "detected_bos_features": bos_features,
        "known_bos_features": KNOWN_BOS_FEATURES.get(args.model, []),
    }

    output_path = f"bos_features_{args.model.replace('/', '_')}_layer{args.layer}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
