"""Wicher (Broyden) model export and loading functionality."""
from __future__ import annotations

import torch
from pathlib import Path
from typing import Optional
from wisent.core.utils.cli.cli_logger import setup_logger, bind
from wisent.core.utils.infra_tools.errors import MissingParameterError, InsufficientDataError
from wisent.core.weight_modification.export._generic import (
    load_steered_model,
    _save_standalone_loader,
)
from wisent.core.utils.config_tools.constants import JSON_INDENT, WICHER_DEFAULT_SOLVER

_LOG = setup_logger(__name__)

def export_wicher_model(
    model,
    wicher_steering,
    save_path,
    base_strength: float,
    tokenizer=None,
    push_to_hub: bool = False,
    repo_id=None,
    commit_message=None,
) -> None:
    """
    Export a WICHER model with Broyden-based runtime hooks.

    Saves model weights unchanged plus concept subspace data.
    On load, runtime hooks apply Broyden iterations.
    """
    import json

    log = bind(_LOG, save_path=str(save_path))
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    log.info("Exporting WICHER model (dynamic, Broyden steering)")

    model.save_pretrained(save_path)
    log.info("Model saved successfully")

    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)
        log.info("Tokenizer saved successfully")

    wicher_data = {
        "base_strength": base_strength,
        "num_steps": wicher_steering.num_steps,
        "alpha": wicher_steering.alpha,
        "eta": wicher_steering.eta,
        "beta": wicher_steering.beta,
        "alpha_decay": wicher_steering.alpha_decay,
        "concept_directions": {
            str(k): v.cpu() for k, v in wicher_steering.concept_directions.items()
        },
        "concept_bases": {
            str(k): v.cpu() for k, v in wicher_steering.concept_bases.items()
        },
        "component_variances": {
            str(k): v.cpu() for k, v in wicher_steering.component_variances.items()
        },
        "layer_variance": {
            str(k): v for k, v in wicher_steering.layer_variance.items()
        },
        "solver": getattr(wicher_steering, "solver", WICHER_DEFAULT_SOLVER),
    }

    meta = wicher_steering.metadata
    wicher_data["metadata"] = {
        "method": meta.method,
        "model_name": meta.model_name,
        "benchmark": meta.benchmark,
        "category": meta.category,
        "layers": meta.layers,
        "hidden_dim": meta.hidden_dim,
    }

    steering_path = save_path / "wicher_steering.pt"
    torch.save(wicher_data, steering_path)
    log.info("Saved WICHER steering data")

    config_path = save_path / "wicher_config.json"
    wicher_config = {
        "is_wicher_model": True,
        "mode": "dynamic",
        "num_layers": len(wicher_steering.concept_directions),
        "steered_layers": sorted(wicher_steering.concept_directions.keys()),
        "base_strength": base_strength,
        "num_steps": wicher_steering.num_steps,
    }
    with open(config_path, "w") as f:
        json.dump(wicher_config, f, indent=JSON_INDENT)
    log.info("Saved WICHER config")

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
            path_in_repo="wicher_steering.pt", repo_id=repo_id,
        )
        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo="wicher_config.json", repo_id=repo_id,
        )
        log.info(f"Uploaded WICHER files to {repo_id}")


def load_wicher_model(
    model_path,
    device_map: Optional[str] = None,
    torch_dtype=None,
    install_hooks: bool = True,
):
    """
    Load a WICHER model with Broyden-based runtime hooks.

    Returns (model, tokenizer, hooks).
    """
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download, HfFileSystem

    model_path_str = str(model_path)
    log = bind(_LOG, model_path=model_path_str)

    is_local = Path(model_path_str).exists()

    if is_local:
        config_path = Path(model_path_str) / "wicher_config.json"
        config_exists = config_path.exists()
    else:
        try:
            fs = HfFileSystem()
            files = fs.ls(model_path_str, detail=False)
            file_names = [f.split("/")[-1] for f in files]
            config_exists = "wicher_config.json" in file_names
        except Exception:
            config_exists = False

    if not config_exists:
        log.warning("No wicher_config.json found")
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
            wicher_config = json.load(f)
    else:
        cf = hf_hub_download(repo_id=model_path_str, filename="wicher_config.json")
        with open(cf) as f:
            wicher_config = json.load(f)

    log.info(f"Loading WICHER model (mode={wicher_config.get('mode', 'dynamic')})")

    model = AutoModelForCausalLM.from_pretrained(
        model_path_str, device_map=device_map,
        torch_dtype=torch_dtype, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path_str)

    hooks = None

    if is_local:
        data_path = Path(model_path_str) / "wicher_steering.pt"
        data_exists = data_path.exists()
    else:
        try:
            data_path = hf_hub_download(
                repo_id=model_path_str, filename="wicher_steering.pt",
            )
            data_exists = True
        except Exception:
            data_exists = False

    if install_hooks and data_exists:
        wicher_data = torch.load(data_path, map_location="cpu", weights_only=False)

        from wisent.core.weight_modification.directional.hooks.transport.wicher import (
            WicherRuntimeHooks,
        )

        def to_tensor(v):
            return v if isinstance(v, torch.Tensor) else torch.tensor(v)

        concept_directions = {
            int(k): to_tensor(v)
            for k, v in wicher_data["concept_directions"].items()
        }
        concept_bases = {
            int(k): to_tensor(v)
            for k, v in wicher_data["concept_bases"].items()
        }
        component_variances = {
            int(k): to_tensor(v)
            for k, v in wicher_data["component_variances"].items()
        }
        layer_variance = {
            int(k): float(v)
            for k, v in wicher_data["layer_variance"].items()
        }


        _REQUIRED_KEYS = ["num_steps", "alpha", "eta", "beta", "alpha_decay"]
        for key in _REQUIRED_KEYS:
            if wicher_data.get(key) is None:
                raise InsufficientDataError(
                    reason=f"Missing '{key}' in saved data. Re-export the WICHER model.",
                )

        hooks = WicherRuntimeHooks(
            model=model,
            concept_directions=concept_directions,
            concept_bases=concept_bases,
            component_variances=component_variances,
            layer_variance=layer_variance,
            num_steps=wicher_data["num_steps"],
            alpha=wicher_data["alpha"],
            eta=wicher_data["eta"],
            beta=wicher_data["beta"],
            alpha_decay=wicher_data["alpha_decay"],
            base_strength=wicher_data["base_strength"],
            solver=wicher_data.get("solver", WICHER_DEFAULT_SOLVER),
        )
        hooks.install()
        log.info(
            f"Installed WICHER hooks on layers "
            f"{sorted(concept_directions.keys())}",
        )

    return model, tokenizer, hooks
