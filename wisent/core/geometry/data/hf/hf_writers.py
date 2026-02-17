"""HuggingFace Hub upload functions for activation data."""
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .hf_config import (
    HF_REPO_ID,
    HF_REPO_TYPE,
    activation_hf_path,
    model_to_safe_name,
    pair_texts_hf_path,
    raw_activation_hf_path,
)


def _get_hf_token() -> str:
    """Get HuggingFace token (required for writes)."""
    token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )
    if not token:
        raise ValueError(
            "HF_TOKEN environment variable required for uploads. "
            "Set it to a HuggingFace write token."
        )
    return token


def _get_api():
    """Get authenticated HfApi instance."""
    from huggingface_hub import HfApi
    return HfApi(token=_get_hf_token())


def upload_activation_shard(
    model_name: str,
    benchmark: str,
    strategy: str,
    layer: int,
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    pair_ids: List[int],
    dry_run: bool = False,
) -> str:
    """Upload a single layer's activations as a safetensors file."""
    from safetensors.torch import save_file

    hf_path = activation_hf_path(model_name, benchmark, strategy, layer)
    n_pairs = pos_activations.shape[0]
    dim = pos_activations.shape[1]

    if dry_run:
        print(f"  [DRY RUN] Would upload {hf_path} ({n_pairs} pairs, dim={dim})")
        return hf_path

    tensors = {
        "pos_activations": pos_activations,
        "neg_activations": neg_activations,
    }
    metadata = {"pair_ids": json.dumps(pair_ids)}

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        save_file(tensors, tmp_path, metadata=metadata)
        api = _get_api()
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=hf_path,
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
        )
        print(f"  Uploaded {hf_path} ({n_pairs} pairs, dim={dim})")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return hf_path


def upload_raw_activation_shard(
    model_name: str,
    benchmark: str,
    prompt_format: str,
    layer: int,
    chunk: int,
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    pair_ids: List[int],
    dry_run: bool = False,
) -> str:
    """Upload a chunk of raw activations as safetensors."""
    from safetensors.torch import save_file

    hf_path = raw_activation_hf_path(
        model_name, benchmark, prompt_format, layer, chunk
    )

    if dry_run:
        print(f"  [DRY RUN] Would upload {hf_path} ({len(pair_ids)} pairs)")
        return hf_path

    tensors = {
        "pos_activations": pos_activations,
        "neg_activations": neg_activations,
    }
    metadata = {"pair_ids": json.dumps(pair_ids)}

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        save_file(tensors, tmp_path, metadata=metadata)
        api = _get_api()
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=hf_path,
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
        )
        print(f"  Uploaded {hf_path} ({len(pair_ids)} pairs)")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return hf_path


def upload_pair_texts(
    benchmark: str,
    pairs: Dict[int, Dict[str, str]],
    dry_run: bool = False,
) -> str:
    """Upload pair texts as a JSON file."""
    hf_path = pair_texts_hf_path(benchmark)

    if dry_run:
        print(f"  [DRY RUN] Would upload {hf_path} ({len(pairs)} pairs)")
        return hf_path

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        json.dump(pairs, tmp, indent=2)
        tmp_path = tmp.name

    try:
        api = _get_api()
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=hf_path,
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
        )
        print(f"  Uploaded {hf_path} ({len(pairs)} pairs)")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return hf_path


def update_index(
    model_name: str,
    benchmark: str,
    strategy: str,
    layers: List[int],
    dry_run: bool = False,
) -> None:
    """Update the repo-level index.json with newly uploaded layers.

    The index maps "{safe_model}/{benchmark}/{strategy}" -> [layers].
    Merges with existing entries rather than overwriting.
    """
    safe_model = model_to_safe_name(model_name)
    key = f"{safe_model}/{benchmark}/{strategy}"

    if dry_run:
        print(f"  [DRY RUN] Would update index.json: {key} -> {layers}")
        return

    api = _get_api()

    index = {}
    try:
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="index.json",
            repo_type=HF_REPO_TYPE,
            token=_get_hf_token(),
        )
        with open(local_path, "r") as f:
            index = json.load(f)
    except Exception:
        pass  # No existing index, start fresh

    existing_layers = set(index.get(key, []))
    existing_layers.update(layers)
    index[key] = sorted(existing_layers)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        json.dump(index, tmp, indent=2)
        tmp_path = tmp.name

    try:
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="index.json",
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
        )
        print(f"  Updated index.json: {key} -> {sorted(existing_layers)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
