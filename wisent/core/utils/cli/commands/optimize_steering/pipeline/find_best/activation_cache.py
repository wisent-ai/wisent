"""Upload freshly extracted activations to HF for future cache hits."""
import json

import torch


def upload_extracted_activations(activations_file: str, model: str, task: str) -> None:
    """Parse activations.json and upload each layer shard to HF.

    Silently skips on any error (missing HF_TOKEN, upload failure, etc.)
    so optimization is never blocked by caching.
    """
    try:
        from wisent.core.reading.modules.utilities.data.sources.hf.hf_writers import (
            upload_activation_shard,
        )
    except ImportError:
        return

    try:
        with open(activations_file) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return

    strategy = data.get("extraction_strategy", "")
    component = data.get("extraction_component", "residual_stream")
    layers = data.get("layers", [])
    pairs = data.get("pairs", [])
    if not pairs or not layers:
        return

    for layer in layers:
        layer_str = str(layer)
        pos_list, neg_list, pair_ids = [], [], []
        for idx, pair in enumerate(pairs):
            pos_act = (pair.get("positive_response", {})
                       .get("layers_activations", {}).get(layer_str))
            neg_act = (pair.get("negative_response", {})
                       .get("layers_activations", {}).get(layer_str))
            if pos_act is not None and neg_act is not None:
                pos_list.append(torch.tensor(pos_act))
                neg_list.append(torch.tensor(neg_act))
                pair_ids.append(idx)
        if not pos_list:
            continue
        pos_tensor = torch.stack(pos_list)
        neg_tensor = torch.stack(neg_list)
        hf_strategy = (f"{strategy}/{component}"
                        if component != "residual_stream" else strategy)
        try:
            upload_activation_shard(
                model_name=model, benchmark=task, strategy=hf_strategy,
                layer=layer, pos_activations=pos_tensor,
                neg_activations=neg_tensor, pair_ids=pair_ids,
            )
        except Exception as exc:
            print(f"  [cache] Upload failed for "
                  f"{task}/{hf_strategy}/layer_{layer}: {exc}")
