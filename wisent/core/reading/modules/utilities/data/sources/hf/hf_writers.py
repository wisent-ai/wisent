"""HuggingFace Hub upload functions for activation data."""
import json, os, tempfile
from pathlib import Path
from typing import Dict, List, Optional

import torch
from wisent.core.utils.config_tools.constants import (
    COMBO_OFFSET,
    INDEX_FIRST,
    INDEX_LAST,
    JSON_INDENT,
    MIN_COMBO_PATH_PARTS,
)
from .hf_config import (
    HF_REPO_ID,
    HF_REPO_TYPE,
    activation_hf_path,
    model_to_safe_name,
    pair_texts_hf_path,
    raw_activation_hf_path,
    test_results_hf_path,
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
    staging_dir: Optional[str] = None,
) -> str:
    """Upload a single layer's activations as a safetensors file.
    If staging_dir is set, save locally instead of uploading (for batch mode)."""
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

    if staging_dir:
        out_path = Path(staging_dir) / hf_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(out_path), metadata=metadata)
        print(f"  Staged {hf_path} ({n_pairs} pairs, dim={dim})")
        return hf_path

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        save_file(tensors, tmp_path, metadata=metadata)
        api = _get_api()
        api.upload_file(
            path_or_fileobj=tmp_path, path_in_repo=hf_path,
            repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE,
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
    staging_dir: Optional[str] = None,
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

    if staging_dir:
        out_path = Path(staging_dir) / hf_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(out_path), metadata=metadata)
        print(f"  Staged {hf_path} ({len(pair_ids)} pairs)")
        return hf_path

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        save_file(tensors, tmp_path, metadata=metadata)
        api = _get_api()
        api.upload_file(
            path_or_fileobj=tmp_path, path_in_repo=hf_path,
            repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE,
        )
        print(f"  Uploaded {hf_path} ({len(pair_ids)} pairs)")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return hf_path


def upload_pair_texts(
    benchmark: str,
    pairs: Dict[int, Dict[str, str]],
    dry_run: bool = False,
    staging_dir: Optional[str] = None,
) -> str:
    """Upload pair texts as JSON. If staging_dir, save locally."""
    hf_path = pair_texts_hf_path(benchmark)
    if dry_run:
        print(f"  [DRY RUN] Would upload {hf_path} ({len(pairs)} pairs)")
        return hf_path

    if staging_dir:
        out_path = Path(staging_dir) / hf_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(pairs, f, indent=JSON_INDENT)
        print(f"  Staged {hf_path} ({len(pairs)} pairs)")
        return hf_path

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(pairs, tmp, indent=JSON_INDENT)
        tmp_path = tmp.name
    try:
        api = _get_api()
        api.upload_file(
            path_or_fileobj=tmp_path, path_in_repo=hf_path,
            repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE,
        )
        print(f"  Uploaded {hf_path} ({len(pairs)} pairs)")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return hf_path


def flush_staging_dir(staging_dir: str, path_in_repo: str = "."):
    """Upload everything in staging_dir as a single commit."""
    api = _get_api()
    api.upload_folder(
        folder_path=staging_dir,
        path_in_repo=path_in_repo,
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
    )
    print(f"  Flushed staging dir to HF ({path_in_repo})")


def write_marker(
    model_name: str,
    benchmark: str,
    strategy: str,
    layers: List[int],
    dry_run: bool = False,
) -> None:
    """Write a per-combo marker file (markers/{safe_model}/{benchmark}/{strategy}.json)."""
    safe_model = model_to_safe_name(model_name)
    marker_path = f"markers/{safe_model}/{benchmark}/{strategy}.json"
    payload = {"layers": sorted(layers)}

    if dry_run:
        print(f"  [DRY RUN] Would write marker: {marker_path} -> {payload}")
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(payload, tmp, indent=JSON_INDENT)
        tmp_path = tmp.name
    try:
        api = _get_api()
        api.upload_file(
            path_or_fileobj=tmp_path, path_in_repo=marker_path,
            repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE,
        )
        print(f"  Wrote marker: {marker_path} -> {payload}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _list_model_dirs(api) -> List[str]:
    """List model directories under activations/ using tree API."""
    entries = api.list_repo_tree(
        repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE,
        path_in_repo="activations", recursive=False,
    )
    return [e.path for e in entries if hasattr(e, "path")]


def consolidate_index(dry_run: bool = False) -> Dict[str, List[int]]:
    """Build unified index.json by scanning activation files in HF repo."""
    import re
    from huggingface_hub import HfApi

    token = _get_hf_token()
    api = HfApi(token=token)
    model_dirs = _list_model_dirs(api)
    print(f"Found {len(model_dirs)} model directories")
    layer_re = re.compile(
        r"layer_(?P<layer>\d+)\.safetensors$"
    )
    index: Dict[str, List[int]] = {}
    for model_dir in model_dirs:
        model_name = model_dir.split("/")[INDEX_LAST]
        print(f"  Scanning {model_name}...")
        entries = api.list_repo_tree(
            repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE,
            path_in_repo=model_dir, recursive=True,
        )
        for entry in entries:
            fp = getattr(entry, "path", None)
            if not fp:
                continue
            m = layer_re.search(fp)
            if not m:
                continue
            layer_num = int(m.group("layer"))
            rel = fp[len(model_dir) + COMBO_OFFSET:]
            parts = rel.split("/")
            if len(parts) < MIN_COMBO_PATH_PARTS:
                continue
            strategy = parts[INDEX_LAST - COMBO_OFFSET]
            benchmark = "/".join(parts[:INDEX_LAST - COMBO_OFFSET])
            key = f"{model_name}/{benchmark}/{strategy}"
            index.setdefault(key, []).append(layer_num)
    for key in index:
        index[key] = sorted(index[key])
    print(f"Consolidated index: {len(index)} entries (from activation files)")

    if dry_run:
        for k, v in sorted(index.items()):
            print(f"  {k}: {v}")
        return index

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(index, tmp, indent=JSON_INDENT)
        tmp_path = tmp.name
    try:
        api.upload_file(
            path_or_fileobj=tmp_path, path_in_repo="index.json",
            repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE,
        )
        print(f"  Uploaded consolidated index.json")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return index


def _retry_upload(upload_fn, max_retries, base_wait, backoff_max_exponent,
                  jitter_min, jitter_max, retryable_patterns):
    """Retry upload with exponential backoff on retryable errors."""
    import random, time
    for attempt in range(max_retries + COMBO_OFFSET):
        try:
            return upload_fn()
        except Exception as exc:
            if attempt >= max_retries or not any(p in str(exc) for p in retryable_patterns):
                raise
            time.sleep(base_wait * (COMBO_OFFSET << min(attempt, backoff_max_exponent)) * random.uniform(jitter_min, jitter_max))


def upload_test_results(benchmark: str, results: dict) -> str:
    """Upload benchmark test results as JSON."""
    hf_path = test_results_hf_path(benchmark)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(results, tmp, indent=JSON_INDENT)
        tmp_path = tmp.name
    try:
        api = _get_api()
        api.upload_file(
            path_or_fileobj=tmp_path, path_in_repo=hf_path,
            repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return hf_path
