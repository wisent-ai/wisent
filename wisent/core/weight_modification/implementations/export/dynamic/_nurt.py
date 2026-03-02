"""Nurt (Concept Flow) model export and loading functionality."""
from __future__ import annotations

import torch
from pathlib import Path
from wisent.core.utils.cli.cli_logger import setup_logger, bind
from wisent.core.utils.infra_tools.errors import MissingParameterError
from wisent.core.utils.config_tools.constants import DEFAULT_STRENGTH, JSON_INDENT, NURT_NUM_INTEGRATION_STEPS, NURT_T_MAX
from wisent.core.weight_modification.export._generic import (
    load_steered_model,
    _save_standalone_loader,
)
_LOG = setup_logger(__name__)

def export_nurt_model(
    model: Module,
    nurt_steering: object,
    save_path: str | Path,
    tokenizer=None,
    base_strength: float = DEFAULT_STRENGTH,
    push_to_hub: bool = False,
    repo_id: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Export a Concept Flow model with full nonlinear transport.

    Saves the model weights (unchanged) plus concept flow components
    (flow networks, concept bases, metadata). On load, runtime hooks
    are installed to apply full nonlinear flow transport.

    No linearization — the velocity fields are preserved exactly.

    Args:
        model: Base model (weights are NOT modified)
        nurt_steering: NurtSteeringObject
        save_path: Directory to save model
        tokenizer: Optional tokenizer to save
        base_strength: Steering strength multiplier
        push_to_hub: Whether to upload to HuggingFace Hub
        repo_id: HuggingFace repo ID
        commit_message: Commit message for Hub upload
    """
    import json

    log = bind(_LOG, save_path=str(save_path))
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    log.info("Exporting Concept Flow model (dynamic, full nonlinear)")

    # Step 1: Save model weights unchanged
    log.info("Saving model to disk")
    model.save_pretrained(save_path)
    log.info("Model saved successfully")

    # Step 2: Save tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)
        log.info("Tokenizer saved successfully")

    # Step 3: Save concept flow components
    cf = nurt_steering
    flow_data = {
        "base_strength": base_strength,
        "t_max": cf.t_max,
        "num_integration_steps": cf.num_integration_steps,
        "layer_variance": {str(k): v for k, v in cf.layer_variance.items()},
    }

    # Flow networks: save state dicts and configs
    flow_nets = {}
    for layer, net in cf.flow_networks.items():
        flow_nets[str(layer)] = {
            "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
            "config": {
                "concept_dim": net.concept_dim,
                "hidden_dim": net.hidden_dim,
            },
        }
    flow_data["flow_networks"] = flow_nets

    # Concept bases and mean positions
    flow_data["concept_bases"] = {
        str(k): v.cpu() for k, v in cf.concept_bases.items()
    }
    flow_data["mean_neg"] = {
        str(k): v.cpu() for k, v in cf.mean_neg.items()
    }
    flow_data["mean_pos"] = {
        str(k): v.cpu() for k, v in cf.mean_pos.items()
    }

    # Layer weights (normalized variance)
    flow_data["_layer_weights"] = {
        str(k): v for k, v in cf._layer_weights.items()
    }

    # Metadata
    meta = cf.metadata
    flow_data["metadata"] = {
        "method": meta.method,
        "model_name": meta.model_name,
        "benchmark": meta.benchmark,
        "category": meta.category,
        "layers": meta.layers,
        "hidden_dim": meta.hidden_dim,
    }

    steering_path = save_path / "nurt_steering.pt"
    torch.save(flow_data, steering_path)
    log.info("Saved concept flow steering data")

    # Step 4: Save config
    config_path = save_path / "nurt_config.json"
    cf_config = {
        "is_nurt_model": True,
        "mode": "dynamic",
        "num_layers": len(cf.flow_networks),
        "steered_layers": sorted(cf.flow_networks.keys()),
        "base_strength": base_strength,
        "t_max": cf.t_max,
        "num_integration_steps": cf.num_integration_steps,
    }
    with open(config_path, "w") as f:
        json.dump(cf_config, f, indent=JSON_INDENT)
    log.info("Saved concept flow config")

    # Step 5: Save standalone loader
    _save_standalone_loader(save_path)
    log.info("Saved standalone_loader.py")

    # Step 6: Push to hub if requested
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
            path_in_repo="nurt_steering.pt",
            repo_id=repo_id,
        )
        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo="nurt_config.json",
            repo_id=repo_id,
        )
        log.info(f"Uploaded concept flow files to {repo_id}")


def load_nurt_model(
    model_path: str | Path,
    device_map: str = "auto",
    torch_dtype=None,
    install_hooks: bool = True,
):
    """
    Load a Concept Flow model with full nonlinear transport hooks.

    Loads the model and installs NurtRuntimeHooks that apply
    per-layer nonlinear flow transport (project -> Euler integrate ->
    reconstruct). No linearization.

    Args:
        model_path: Path to saved concept flow model
        device_map: Device map for model loading
        torch_dtype: Torch dtype for model weights
        install_hooks: Whether to install runtime hooks (default: True)

    Returns:
        Tuple of (model, tokenizer, hooks) where hooks is None if
        install_hooks=False or no concept flow data found.
    """
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download, HfFileSystem

    model_path_str = str(model_path)
    log = bind(_LOG, model_path=model_path_str)

    is_local = Path(model_path_str).exists()

    if is_local:
        cf_config_path = Path(model_path_str) / "nurt_config.json"
        config_exists = cf_config_path.exists()
    else:
        try:
            fs = HfFileSystem()
            files = fs.ls(model_path_str, detail=False)
            file_names = [f.split("/")[-1] for f in files]
            config_exists = "nurt_config.json" in file_names
        except Exception:
            config_exists = False

    if not config_exists:
        log.warning("No nurt_config.json found - loading as regular model")
        if is_local:
            return load_steered_model(model_path_str, device_map, torch_dtype) + (None,)
        model = AutoModelForCausalLM.from_pretrained(
            model_path_str, device_map=device_map,
            torch_dtype=torch_dtype, trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path_str)
        return model, tokenizer, None

    # Load config
    if is_local:
        with open(cf_config_path) as f:
            cf_config = json.load(f)
    else:
        config_file = hf_hub_download(
            repo_id=model_path_str, filename="nurt_config.json",
        )
        with open(config_file) as f:
            cf_config = json.load(f)

    log.info(f"Loading Concept Flow model (mode={cf_config.get('mode', 'dynamic')})")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path_str, device_map=device_map,
        torch_dtype=torch_dtype, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path_str)

    hooks = None

    # Load steering data
    if is_local:
        data_path = Path(model_path_str) / "nurt_steering.pt"
        data_exists = data_path.exists()
    else:
        try:
            data_path = hf_hub_download(
                repo_id=model_path_str, filename="nurt_steering.pt",
            )
            data_exists = True
        except Exception:
            data_exists = False

    if install_hooks and data_exists:
        flow_data = torch.load(data_path, map_location="cpu")

        from wisent.core.control.steering_methods.methods.nurt.flow_network import (
            FlowVelocityNetwork,
        )
        from wisent.core.weight_modification.directional.hooks.nurt import (
            NurtRuntimeHooks,
        )

        # Reconstruct flow networks
        flow_networks = {}
        for layer_str, net_data in flow_data["flow_networks"].items():
            cfg = net_data["config"]
            net = FlowVelocityNetwork(cfg["concept_dim"], cfg["hidden_dim"])
            sd = net_data["state_dict"]
            sd = {k: v if isinstance(v, torch.Tensor) else torch.tensor(v)
                  for k, v in sd.items()}
            net.load_state_dict(sd)
            net.eval()
            flow_networks[int(layer_str)] = net

        # Reconstruct concept bases
        concept_bases = {}
        for k, v in flow_data["concept_bases"].items():
            t = v if isinstance(v, torch.Tensor) else torch.tensor(v)
            concept_bases[int(k)] = t

        # Layer weights
        layer_weights = {
            int(k): float(v)
            for k, v in flow_data["_layer_weights"].items()
        }

        hooks = NurtRuntimeHooks(
            model=model,
            flow_networks=flow_networks,
            concept_bases=concept_bases,
            layer_weights=layer_weights,
            t_max=flow_data.get("t_max", NURT_T_MAX),
            num_integration_steps=flow_data.get("num_integration_steps", NURT_NUM_INTEGRATION_STEPS),
            base_strength=flow_data.get("base_strength", DEFAULT_STRENGTH),
        )
        hooks.install()
        log.info(
            f"Installed Concept Flow hooks on layers "
            f"{sorted(flow_networks.keys())}",
        )

    return model, tokenizer, hooks


