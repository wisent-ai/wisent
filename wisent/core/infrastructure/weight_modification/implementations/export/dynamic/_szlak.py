"""Szlak (Geodesic OT) model export and loading functionality."""
from __future__ import annotations

import torch
from pathlib import Path
from wisent.core.cli.cli_logger import setup_logger, bind
from wisent.core.errors import MissingParameterError
from wisent.core.constants import DEFAULT_STRENGTH, JSON_INDENT, SZLAK_INFERENCE_K
from wisent.core.weight_modification.export._generic import (
    load_steered_model,
    _save_standalone_loader,
)

_LOG = setup_logger(__name__)

def export_szlak_model(
    model,
    szlak_steering,
    save_path,
    tokenizer=None,
    base_strength: float = DEFAULT_STRENGTH,
    push_to_hub: bool = False,
    repo_id=None,
    commit_message=None,
) -> None:
    """
    Export a SZLAK model with geodesic OT runtime hooks.

    Saves model weights unchanged plus source points and displacements.
    On load, runtime hooks apply k-NN IDW displacement interpolation.
    """
    import json

    log = bind(_LOG, save_path=str(save_path))
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    log.info("Exporting SZLAK model (dynamic, geodesic OT)")

    model.save_pretrained(save_path)
    log.info("Model saved successfully")

    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)
        log.info("Tokenizer saved successfully")

    szlak_data = {
        "base_strength": base_strength,
        "inference_k": szlak_steering.inference_k,
        "source_points": {
            str(k): v.cpu() for k, v in szlak_steering.source_points.items()
        },
        "displacements": {
            str(k): v.cpu() for k, v in szlak_steering.displacements.items()
        },
    }

    meta = szlak_steering.metadata
    szlak_data["metadata"] = {
        "method": meta.method,
        "model_name": meta.model_name,
        "benchmark": meta.benchmark,
        "category": meta.category,
        "layers": meta.layers,
        "hidden_dim": meta.hidden_dim,
    }

    steering_path = save_path / "szlak_steering.pt"
    torch.save(szlak_data, steering_path)
    log.info("Saved SZLAK steering data")

    config_path = save_path / "szlak_config.json"
    szlak_config = {
        "is_szlak_model": True,
        "mode": "dynamic",
        "num_layers": len(szlak_steering.source_points),
        "steered_layers": sorted(szlak_steering.source_points.keys()),
        "base_strength": base_strength,
        "inference_k": szlak_steering.inference_k,
    }
    with open(config_path, "w") as f:
        json.dump(szlak_config, f, indent=JSON_INDENT)
    log.info("Saved SZLAK config")

    _save_standalone_loader(save_path)

    if push_to_hub:
        if repo_id is None:
            raise MissingParameterError(
                params=["repo_id"], context="push_to_hub=True",
            )
        from wisent.core.weight_modification.export._hub import upload_to_hub
        upload_to_hub(
            model, repo_id,
            tokenizer=tokenizer, commit_message=commit_message,
        )
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(steering_path),
            path_in_repo="szlak_steering.pt", repo_id=repo_id,
        )
        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo="szlak_config.json", repo_id=repo_id,
        )
        log.info(f"Uploaded SZLAK files to {repo_id}")


def load_szlak_model(
    model_path,
    device_map: str = "auto",
    torch_dtype=None,
    install_hooks: bool = True,
):
    """
    Load a SZLAK model with geodesic OT runtime hooks.

    Returns (model, tokenizer, hooks).
    """
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download, HfFileSystem

    model_path_str = str(model_path)
    log = bind(_LOG, model_path=model_path_str)

    is_local = Path(model_path_str).exists()

    if is_local:
        config_path = Path(model_path_str) / "szlak_config.json"
        config_exists = config_path.exists()
    else:
        try:
            fs = HfFileSystem()
            files = fs.ls(model_path_str, detail=False)
            file_names = [f.split("/")[-1] for f in files]
            config_exists = "szlak_config.json" in file_names
        except Exception:
            config_exists = False

    if not config_exists:
        log.warning("No szlak_config.json found")
        if is_local:
            return load_steered_model(model_path_str, device_map, torch_dtype) + (None,)
        model = AutoModelForCausalLM.from_pretrained(
            model_path_str, device_map=device_map,
            torch_dtype=torch_dtype, trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path_str)
        return model, tokenizer, None

    if is_local:
        with open(config_path) as f:
            szlak_config = json.load(f)
    else:
        cf = hf_hub_download(repo_id=model_path_str, filename="szlak_config.json")
        with open(cf) as f:
            szlak_config = json.load(f)

    log.info(f"Loading SZLAK model (mode={szlak_config.get('mode', 'dynamic')})")

    model = AutoModelForCausalLM.from_pretrained(
        model_path_str, device_map=device_map,
        torch_dtype=torch_dtype, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path_str)

    hooks = None

    if is_local:
        data_path = Path(model_path_str) / "szlak_steering.pt"
        data_exists = data_path.exists()
    else:
        try:
            data_path = hf_hub_download(
                repo_id=model_path_str, filename="szlak_steering.pt",
            )
            data_exists = True
        except Exception:
            data_exists = False

    if install_hooks and data_exists:
        szlak_data = torch.load(data_path, map_location="cpu")

        from wisent.core.weight_modification.directional.hooks.transport.szlak import (
            SzlakRuntimeHooks,
        )

        def to_tensor(v):
            return v if isinstance(v, torch.Tensor) else torch.tensor(v)

        source_points = {
            int(k): to_tensor(v)
            for k, v in szlak_data["source_points"].items()
        }
        displacements = {
            int(k): to_tensor(v)
            for k, v in szlak_data["displacements"].items()
        }

        hooks = SzlakRuntimeHooks(
            model=model,
            source_points=source_points,
            displacements=displacements,
            inference_k=szlak_data.get("inference_k", SZLAK_INFERENCE_K),
            base_strength=szlak_data.get("base_strength", DEFAULT_STRENGTH),
        )
        hooks.install()
        log.info(
            f"Installed SZLAK hooks on layers "
            f"{sorted(source_points.keys())}",
        )

    return model, tokenizer, hooks


