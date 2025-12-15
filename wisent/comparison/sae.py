"""
SAE-based steering method for comparison experiments.

Uses Sparse Autoencoders to identify steering directions from contrastive pairs.
Supports EleutherAI SAEs for Pythia models via the sparsify library.
"""

from __future__ import annotations

import json
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from wisent.core.models.wisent_model import WisentModel


# Default SAE configurations for supported models
# See: https://huggingface.co/collections/EleutherAI/sparse-autoencoders
SAE_CONFIGS = {
    "EleutherAI/pythia-70m": {
        "sae_repo": "EleutherAI/sae-pythia-70m-32k",
        "num_layers": 6,
        "default_layer": 3,
    },
    "EleutherAI/pythia-70m-deduped": {
        "sae_repo": "EleutherAI/sae-pythia-70m-deduped-32k",
        "num_layers": 6,
        "default_layer": 3,
    },
    "EleutherAI/pythia-160m": {
        "sae_repo": "EleutherAI/sae-pythia-160m-32k",
        "num_layers": 12,
        "default_layer": 6,
    },
    "EleutherAI/pythia-160m-deduped": {
        "sae_repo": "EleutherAI/sae-pythia-160m-deduped-32k",
        "num_layers": 12,
        "default_layer": 6,
    },
    "EleutherAI/pythia-410m": {
        "sae_repo": "EleutherAI/sae-pythia-410m-65k",
        "num_layers": 24,
        "default_layer": 12,
    },
    "meta-llama/Llama-3.2-1B": {
        "sae_repo": "EleutherAI/sae-Llama-3.2-1B-131k",
        "num_layers": 16,
        "default_layer": 8,
    },
}


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
) -> Path:
    """
    Generate a steering vector using SAE feature analysis.

    Args:
        task: lm-eval task name (e.g., 'boolq', 'cb')
        model_name: HuggingFace model name (must have SAE available)
        output_path: Where to save the steering vector
        trait_label: Label for the trait being steered
        num_pairs: Number of contrastive pairs to use
        method: Method name (should be 'sae')
        layers: Layer(s) to extract features from (e.g., '12' or '10,11,12')
        normalize: Whether to normalize the steering vector
        device: Device to run on
        keep_intermediate: Whether to keep intermediate files

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

    # Step 1: Generate contrastive pairs using wisent's existing infrastructure
    print(f"Step 1: Generating contrastive pairs from task: {task}")
    from wisent.core.cli.generate_pairs_from_task import execute_generate_pairs_from_task

    pairs_file = tempfile.NamedTemporaryFile(mode='w', suffix='_pairs.json', delete=False).name
    pairs_args = Namespace(
        task_name=task,
        limit=num_pairs,
        output=pairs_file,
        seed=42,
        verbose=False,
    )
    execute_generate_pairs_from_task(pairs_args)

    # Load the pairs
    with open(pairs_file) as f:
        pairs_data = json.load(f)

    pairs = pairs_data["pairs"]
    print(f"   Loaded {len(pairs)} contrastive pairs")

    # Step 2: Load model and SAE
    print(f"\nStep 2: Loading model and SAE...")
    from sparsify import SparseCoder
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    steering_vectors = {}
    top_features_info = {}

    for layer_idx in layer_indices:
        print(f"\nStep 3: Processing layer {layer_idx}")

        # Load SAE for this layer
        # Most SAEs use "layers.N.mlp" hookpoint format
        hookpoint = f"layers.{layer_idx}.mlp"
        print(f"   Loading SAE from {config['sae_repo']} for {hookpoint}")
        sae_device = device if isinstance(device, str) else "cuda:0"
        sae = SparseCoder.load_from_hub(config["sae_repo"], hookpoint, device=sae_device)

        # Collect SAE feature activations for positive and negative examples
        pos_features_list = []
        neg_features_list = []

        print(f"   Collecting SAE features from {len(pairs)} pairs...")
        for i, pair in enumerate(pairs):
            prompt = pair["prompt"]
            pos_response = pair["positive_response"]["model_response"]
            neg_response = pair["negative_response"]["model_response"]

            # Full text = prompt + response
            pos_text = f"{prompt} {pos_response}"
            neg_text = f"{prompt} {neg_response}"

            # Get activations for positive example
            pos_acts = _get_layer_activations(model, tokenizer, pos_text, layer_idx, device)
            pos_acts = pos_acts.to(sae_device).to(sae.W_dec.dtype)
            pos_sae_out = sae.encode(pos_acts)
            # pre_acts is the full feature activation [batch, seq, num_features]
            # Mean over sequence, keep batch dim
            pos_features_list.append(pos_sae_out.pre_acts.detach().mean(dim=1))

            # Get activations for negative example
            neg_acts = _get_layer_activations(model, tokenizer, neg_text, layer_idx, device)
            neg_acts = neg_acts.to(sae_device).to(sae.W_dec.dtype)
            neg_sae_out = sae.encode(neg_acts)
            neg_features_list.append(neg_sae_out.pre_acts.detach().mean(dim=1))

            if (i + 1) % 10 == 0:
                print(f"      Processed {i + 1}/{len(pairs)} pairs")

        # Stack and compute mean difference in SAE feature space
        pos_features = torch.cat(pos_features_list, dim=0)  # (num_pairs, num_features)
        neg_features = torch.cat(neg_features_list, dim=0)

        # Mean difference in feature space
        feature_diff = pos_features.mean(dim=0) - neg_features.mean(dim=0)

        # Find top-k most differentiating features
        top_k = 10
        top_values, top_indices = torch.abs(feature_diff).topk(top_k)
        top_feature_indices = top_indices

        print(f"   Top {top_k} differentiating features: {top_feature_indices.tolist()}")
        print(f"   Feature diff magnitudes: {top_values.tolist()}")

        # Construct steering vector from SAE decoder
        # steering_vector = sum of (feature_diff[i] * decoder_vector[i]) for top features
        steering_vector = torch.zeros(sae.W_dec.shape[1], device=sae_device, dtype=sae.W_dec.dtype)
        for feat_idx in top_feature_indices:
            steering_vector += feature_diff[feat_idx] * sae.W_dec[feat_idx]

        if normalize:
            norm = steering_vector.norm()
            if norm > 0:
                steering_vector = steering_vector / norm

        steering_vectors[str(layer_idx)] = steering_vector.cpu().float().tolist()
        top_features_info[str(layer_idx)] = {
            "indices": top_feature_indices.tolist(),
            "diff_values": [feature_diff[i].item() for i in top_feature_indices],
        }

        # Clean up SAE after each layer to free GPU memory
        del sae
        del pos_features_list, neg_features_list, pos_features, neg_features
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Clean up model to free GPU memory
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Clean up temp files
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
        "sae_repo": config["sae_repo"],
        "num_pairs": len(pairs),
        "top_features": top_features_info,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved steering vector to {output_path}")
    return output_path


def _get_layer_activations(
    model,
    tokenizer,
    text: str,
    layer_idx: int,
    device: str,
) -> torch.Tensor:
    """
    Get MLP activations from a specific layer for the given text.

    Returns:
        Tensor of shape (1, seq_len, hidden_dim)
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    activations = []

    def hook_fn(module, input, output):
        # output is typically (hidden_states, ...) tuple or just tensor
        if isinstance(output, tuple):
            activations.append(output[0].detach())
        else:
            activations.append(output.detach())

    # Register hook on the MLP of the target layer
    if hasattr(model, 'gpt_neox'):
        # Pythia models: gpt_neox.layers[N].mlp
        mlp = model.gpt_neox.layers[layer_idx].mlp
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Llama-style models: model.layers[N].mlp
        mlp = model.model.layers[layer_idx].mlp
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-2 style models: transformer.h[N].mlp
        mlp = model.transformer.h[layer_idx].mlp
    else:
        raise ValueError(f"Unknown model architecture: {type(model)}")

    handle = mlp.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(**inputs)

    handle.remove()

    return activations[0]


def load_steering_vector(path: str | Path) -> dict:
    """
    Load a steering vector from file.

    Args:
        path: Path to steering vector file (.json or .pt)

    Returns:
        Dictionary with steering vectors and metadata
    """
    path = Path(path)

    if path.suffix == ".pt":
        data = torch.load(path, map_location="cpu", weights_only=False)
        layer_idx = str(data.get("layer_index", data.get("layer", 1)))
        return {
            "steering_vectors": {layer_idx: data["steering_vector"].tolist()},
            "layers": [layer_idx],
            "model": data.get("model", "unknown"),
            "method": data.get("method", "sae"),
            "trait_label": data.get("trait_label", "unknown"),
        }
    else:
        with open(path) as f:
            return json.load(f)


def apply_steering_to_model(
    model: "WisentModel",
    steering_data: dict,
    scale: float = 1.0,
) -> None:
    """
    Apply loaded steering vectors to a WisentModel.

    Args:
        model: WisentModel instance
        steering_data: Dictionary from load_steering_vector()
        scale: Scaling factor for steering strength
    """
    raw_map = {}
    for layer_str, vec_list in steering_data["steering_vectors"].items():
        raw_map[layer_str] = torch.tensor(vec_list, dtype=torch.float32)

    model.set_steering_from_raw(raw_map, scale=scale, normalize=False)
    model.apply_steering()


def remove_steering(model: "WisentModel") -> None:
    """Remove steering from a WisentModel."""
    model.detach()
    model.clear_steering()


def convert_to_lm_eval_format(
    steering_data: dict,
    output_path: str | Path,
    scale: float = 1.0,
) -> Path:
    """
    Convert our steering vector format to lm-eval's steered model format.

    lm-eval expects:
    {
        "layers.N": {
            "steering_vector": tensor of shape (1, hidden_dim),
            "steering_coefficient": float,
            "action": "add"
        }
    }
    """
    output_path = Path(output_path)

    lm_eval_config = {}
    for layer_str, vec_list in steering_data["steering_vectors"].items():
        vec = torch.tensor(vec_list, dtype=torch.float32)
        # lm-eval expects shape (1, hidden_dim)
        if vec.dim() == 1:
            vec = vec.unsqueeze(0)

        layer_key = f"layers.{layer_str}"
        lm_eval_config[layer_key] = {
            "steering_vector": vec,
            "steering_coefficient": scale,
            "action": "add",
        }

    torch.save(lm_eval_config, output_path)
    return output_path
