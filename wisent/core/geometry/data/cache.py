"""Local caching for database data."""
import json
from pathlib import Path
import torch

CACHE_DIR = Path.home() / ".wisent_cache"


def get_cache_path(task_name: str, cache_type: str, **kwargs) -> Path:
    """Get cache file path for a given task and type."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if cache_type == "pair_texts":
        return CACHE_DIR / f"{task_name}_pair_texts.json"
    elif cache_type == "activations":
        model_name = kwargs.get("model_name", "unknown").replace("/", "_")
        layer = kwargs.get("layer", 0)
        return CACHE_DIR / f"{task_name}_{model_name}_layer{layer}_activations.pt"
    else:
        return CACHE_DIR / f"{task_name}_{cache_type}.json"


def load_pair_texts_cache(task_name: str, limit: int = None):
    """Load pair texts from cache if available."""
    cache_path = get_cache_path(task_name, "pair_texts")
    if not cache_path.exists():
        return None

    print(f"  Loading pair texts from cache: {cache_path}")
    with open(cache_path, 'r') as f:
        cached_data = json.load(f)
    pairs = {int(k): v for k, v in cached_data.items()}

    if limit and len(pairs) > limit:
        pair_ids = sorted(pairs.keys())[:limit]
        pairs = {pid: pairs[pid] for pid in pair_ids}
    return pairs


def save_pair_texts_cache(task_name: str, pairs: dict):
    """Save pair texts to cache."""
    cache_path = get_cache_path(task_name, "pair_texts")
    print(f"  Caching {len(pairs)} pair texts to: {cache_path}")
    with open(cache_path, 'w') as f:
        json.dump(pairs, f)


def load_activations_cache(task_name: str, model_name: str, layer: int, pair_ids=None, limit=None):
    """Load activations from cache if available."""
    cache_path = get_cache_path(task_name, "activations", model_name=model_name, layer=layer)
    if not cache_path.exists():
        return None, None, None

    print(f"  Loading activations from cache: {cache_path}")
    cached = torch.load(cache_path, weights_only=True)
    pos_tensor = cached["pos"]
    neg_tensor = cached["neg"]
    cached_pair_ids = cached.get("pair_ids", list(range(len(pos_tensor))))

    if pair_ids is not None:
        pos_list, neg_list = [], []
        for i, pid in enumerate(cached_pair_ids):
            if pid in pair_ids:
                pos_list.append(pos_tensor[i])
                neg_list.append(neg_tensor[i])
        if pos_list:
            return torch.stack(pos_list), torch.stack(neg_list), None
        return None, None, None

    if limit and len(pos_tensor) > limit:
        pos_tensor = pos_tensor[:limit]
        neg_tensor = neg_tensor[:limit]

    return pos_tensor, neg_tensor, cached_pair_ids


def save_activations_cache(task_name: str, model_name: str, layer: int, pos_tensor, neg_tensor, pair_ids):
    """Save activations to cache."""
    cache_path = get_cache_path(task_name, "activations", model_name=model_name, layer=layer)
    print(f"  Caching {len(pos_tensor)} activations to: {cache_path}")
    torch.save({
        "pos": pos_tensor,
        "neg": neg_tensor,
        "pair_ids": pair_ids,
    }, cache_path)


def get_cached_layers(task_name: str, model_name: str) -> list:
    """Get list of layers available in cache for a task/model combination."""
    model_safe = model_name.replace("/", "_")
    pattern = f"{task_name}_{model_safe}_layer*_activations.pt"

    layers = []
    for cache_file in CACHE_DIR.glob(pattern):
        # Extract layer number from filename
        name = cache_file.stem  # e.g., "truthfulqa_custom_meta-llama_Llama-3.2-1B-Instruct_layer12_activations"
        parts = name.split("_layer")
        if len(parts) == 2:
            layer_part = parts[1].split("_")[0]  # e.g., "12"
            try:
                layers.append(int(layer_part))
            except ValueError:
                pass

    return sorted(layers)
