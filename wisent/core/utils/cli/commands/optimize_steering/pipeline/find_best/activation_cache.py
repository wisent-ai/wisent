"""Upload freshly extracted activations and trial artifacts to HF."""
import json
import os

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


def upload_trial(
    model: str, benchmark: str, method: str, trial_idx: int, trial_dir: str,
) -> None:
    """Upload trial artifacts (responses, scores, meta) to HF."""
    try:
        from wisent.core.reading.modules.utilities.data.sources.hf.hf_config import (
            HF_REPO_ID, HF_REPO_TYPE, model_to_safe_name,
        )
        from wisent.core.reading.modules.utilities.data.sources.hf.hf_writers import (
            _get_api,
        )
        safe = model_to_safe_name(model)
        api = _get_api()
        for fname in ("trial_meta.json", "responses.json", "scores.json"):
            local = os.path.join(trial_dir, fname)
            if not os.path.exists(local):
                continue
            hf_path = f"trials/{safe}/{benchmark}/{method.lower()}/trial_{trial_idx:04d}/{fname}"
            api.upload_file(
                path_or_fileobj=local, path_in_repo=hf_path,
                repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE,
            )
        print(f"  [trial] Uploaded trial {trial_idx} to HF")
    except Exception as exc:
        print(f"  [trial] Upload failed for trial {trial_idx}: {exc}")
