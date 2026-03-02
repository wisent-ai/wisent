"""HuggingFace Hub read functions for activation data."""
import json
import os
from typing import Dict, List, Optional, Tuple

import torch

from wisent.core.utils.config_tools.constants import DATA_LOAD_LIMIT

from wisent.core.reading.modules.utilities.data.cache import get_cache_path, save_activations_cache, save_pair_texts_cache
from .hf_config import (
    HF_REPO_ID,
    HF_REPO_TYPE,
    activation_hf_path,
    model_to_safe_name,
    pair_texts_hf_path,
)


def _get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _hf_hub_download(repo_path: str) -> str:
    """Download a file from the HF repo and return local path."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=repo_path,
        repo_type=HF_REPO_TYPE,
        token=_get_hf_token(),
    )


def _load_safetensors_file(local_path: str) -> tuple:
    """Load tensors and metadata from a safetensors file."""
    from safetensors.torch import load_file
    from safetensors import safe_open

    tensors = load_file(local_path)
    metadata = {}
    with safe_open(local_path, framework="pt") as f:
        metadata = f.metadata() or {}
    return tensors, metadata


def load_activations_from_hf(
    model_name: str,
    task_name: str,
    layer: int,
    extraction_strategy: str,
    limit: Optional[int] = None,
    pair_ids: Optional[set] = None,
    use_cache: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load activations for a single layer from HuggingFace Hub.

    Downloads the safetensors shard, extracts pos/neg tensors,
    and saves to local cache for future use.

    Args:
        model_name: HuggingFace model ID
        task_name: Benchmark/task name
        layer: Layer number
        extraction_strategy: Extraction strategy used
        limit: Max pairs to return
        pair_ids: Optional set of pair IDs to filter
        use_cache: Whether to use local cache

    Returns:
        Tuple of (pos_activations, neg_activations) tensors
    """
    cache_path = get_cache_path(
        task_name, "activations", model_name=model_name, layer=layer
    )
    if use_cache and cache_path.exists():
        cached = torch.load(cache_path, weights_only=True)
        pos_tensor, neg_tensor = cached["pos"], cached["neg"]
        cached_pids = cached.get("pair_ids", list(range(len(pos_tensor))))
        if pair_ids is not None:
            pos_list, neg_list = [], []
            for i, pid in enumerate(cached_pids):
                if pid in pair_ids:
                    pos_list.append(pos_tensor[i])
                    neg_list.append(neg_tensor[i])
            if pos_list:
                return torch.stack(pos_list), torch.stack(neg_list)
        else:
            if limit and len(pos_tensor) > limit:
                return pos_tensor[:limit], neg_tensor[:limit]
            return pos_tensor, neg_tensor

    hf_path = activation_hf_path(
        model_name, task_name, extraction_strategy, layer
    )
    local_path = _hf_hub_download(hf_path)
    tensors, metadata = _load_safetensors_file(local_path)
    pos_tensor = tensors["pos_activations"]
    neg_tensor = tensors["neg_activations"]

    loaded_pair_ids = []
    if "pair_ids" in metadata:
        loaded_pair_ids = json.loads(metadata["pair_ids"])

    if use_cache:
        save_activations_cache(
            task_name, model_name, layer,
            pos_tensor, neg_tensor, loaded_pair_ids,
        )

    if pair_ids is not None:
        pos_list, neg_list = [], []
        for i, pid in enumerate(loaded_pair_ids):
            if pid in pair_ids:
                pos_list.append(pos_tensor[i])
                neg_list.append(neg_tensor[i])
        if pos_list:
            return torch.stack(pos_list), torch.stack(neg_list)
        return torch.tensor([]), torch.tensor([])

    if limit and len(pos_tensor) > limit:
        pos_tensor, neg_tensor = pos_tensor[:limit], neg_tensor[:limit]
    return pos_tensor, neg_tensor


def load_available_layers_from_hf(
    model_name: str,
    task_name: str,
    extraction_strategy: str,
) -> List[int]:
    """Query HF index.json to find available layers."""
    try:
        local_path = _hf_hub_download("index.json")
    except Exception:
        raise FileNotFoundError(
            f"index.json not found in {HF_REPO_ID}. "
            "Repository may not have been initialized yet."
        )

    with open(local_path, "r") as f:
        index = json.load(f)

    safe_model = model_to_safe_name(model_name)
    key = f"{safe_model}/{task_name}/{extraction_strategy}"

    if key not in index:
        raise FileNotFoundError(
            f"No layers found for {model_name}/{task_name}"
            f"/{extraction_strategy} in HF index."
        )
    return sorted(index[key])


def load_pair_texts_from_hf(
    task_name: str,
    limit: int = DATA_LOAD_LIMIT,
    use_cache: bool = True,
) -> Dict[int, Dict[str, str]]:
    """Load contrastive pair texts from HuggingFace Hub.

    Args:
        task_name: Benchmark/task name
        limit: Max pairs to return
        use_cache: Whether to use local cache

    Returns:
        Dict mapping pair_id -> {prompt, positive, negative}
    """
    cache_path = get_cache_path(task_name, "pair_texts")
    if use_cache and cache_path.exists():
        print(f"  Loading pair texts from cache: {cache_path}", flush=True)
        with open(cache_path, "r") as f:
            cached_data = json.load(f)
        pairs = {int(k): v for k, v in cached_data.items()}
        if limit and len(pairs) > limit:
            sorted_ids = sorted(pairs.keys())[:limit]
            pairs = {pid: pairs[pid] for pid in sorted_ids}
        return pairs

    hf_path = pair_texts_hf_path(task_name)
    local_path = _hf_hub_download(hf_path)

    with open(local_path, "r") as f:
        raw_data = json.load(f)

    pairs = {int(k): v for k, v in raw_data.items()}

    if use_cache:
        save_pair_texts_cache(task_name, pairs)

    if limit and len(pairs) > limit:
        sorted_ids = sorted(pairs.keys())[:limit]
        pairs = {pid: pairs[pid] for pid in sorted_ids}

    return pairs
