"""TETNO model export and loading functionality."""
from __future__ import annotations

import torch
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from wisent.core.utils.cli.cli_logger import setup_logger, bind
from wisent.core.utils.infra_tools.errors import MissingParameterError
from wisent.core.utils.config_tools.constants import JSON_INDENT
from wisent.core.utils.infra_tools.errors import InsufficientDataError
from wisent.core.weight_modification.export._generic import (
    load_steered_model,
    _save_standalone_loader,
)

if TYPE_CHECKING:
    from torch.nn import Module

_LOG = setup_logger(__name__)

def export_tetno_model(
    model: Module,
    tetno_result,
    save_path: str | Path,
    strength: float,
    mode: str,
    *,
    hybrid_strength_factor: float,
    tokenizer=None,
    push_to_hub: bool = False,
    repo_id: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Export a TETNO-steered model with conditional steering.
    
    TETNO uses a condition vector for input-dependent gating:
    - In "static" mode: Always apply steering at fixed strength
    - In "dynamic" mode: Use condition vector gating at runtime (requires hooks)
    - In "hybrid" mode: Bake partial steering + dynamic gating for remainder
    
    Args:
        model: Base model to modify
        tetno_result: TETNOResult from training
        save_path: Where to save the model
        tokenizer: Optional tokenizer to save
        mode: "static", "dynamic", or "hybrid"
        strength: Steering strength for static component
        push_to_hub: Whether to upload to HuggingFace Hub
        repo_id: HuggingFace repo ID
        commit_message: Commit message for Hub
    """
    import json
    from wisent.core.weight_modification.methods.additive import bake_steering_into_weights
    from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations
    
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    log = bind(_LOG, save_path=str(save_path))
    
    log.info(f"Exporting TETNO model (mode={mode})")
    
    # Step 1: Bake behavior vectors into weights (for static/hybrid mode)
    if mode in ("static", "hybrid"):
        bake_strength = strength if mode == "static" else strength * hybrid_strength_factor
        
        # Convert layer names to integer indices
        int_keyed_vectors = {}
        for layer_name, vec in tetno_result.behavior_vectors.items():
            try:
                layer_idx = int(str(layer_name).split("_")[-1])
                int_keyed_vectors[layer_idx] = vec
            except (ValueError, IndexError):
                pass
        
        steering_vectors = LayerActivations(int_keyed_vectors)
        
        bake_steering_into_weights(
            model=model,
            steering_vectors=steering_vectors,
            alpha=bake_strength,
            method="bias",
        )
        log.info(f"Baked TETNO vectors (strength={bake_strength})")
    
    # Step 2: Save model
    model.save_pretrained(save_path)
    log.info("Model saved successfully")
    
    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)
        log.info("Tokenizer saved successfully")
    
    # Step 3: Save TETNO components for dynamic steering
    tetno_save_path = save_path / "tetno_steering.pt"
    
    # Extract gate_temperature from metadata config if available
    config_dict = tetno_result.metadata.get("config", {}) if isinstance(tetno_result.metadata, dict) else {}
    gate_temp = config_dict.get("gate_temperature", getattr(tetno_result, "gate_temperature", None))
    if gate_temp is None:
        raise InsufficientDataError(reason="gate_temperature not found in TETNO result metadata. Re-train with current wisent version.")

    tetno_data = {
        "mode": mode,
        "behavior_vectors": {k: v.cpu() for k, v in tetno_result.behavior_vectors.items()},
        "condition_vector": tetno_result.condition_vector.cpu(),
        "layer_scales": tetno_result.layer_scales,
        "optimal_threshold": tetno_result.optimal_threshold,
        "gate_temperature": gate_temp,
        "metadata": tetno_result.metadata,
    }
    
    torch.save(tetno_data, tetno_save_path)
    log.info("Saved TETNO steering data")
    
    # Step 4: Save config
    config_path = save_path / "tetno_config.json"
    tetno_config = {
        "is_tetno_model": True,
        "mode": mode,
        "num_layers": len(tetno_result.behavior_vectors),
        "optimal_threshold": tetno_result.optimal_threshold,
        "layer_order": list(tetno_result.behavior_vectors.keys()),
    }
    with open(config_path, 'w') as f:
        json.dump(tetno_config, f, indent=JSON_INDENT)
    log.info("Saved TETNO config")
    
    # Step 5: Save standalone loader
    _save_standalone_loader(save_path)
    log.info("Saved standalone_loader.py")
    
    # Step 6: Push to hub if requested
    if push_to_hub:
        if repo_id is None:
            raise MissingParameterError(params=["repo_id"], context="push_to_hub=True")
        
        from wisent.core.weight_modification.export._hub import upload_to_hub
        upload_to_hub(model, repo_id, tokenizer=tokenizer, commit_message=commit_message)
        
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(path_or_fileobj=str(tetno_save_path), path_in_repo="tetno_steering.pt", repo_id=repo_id)
        api.upload_file(path_or_fileobj=str(config_path), path_in_repo="tetno_config.json", repo_id=repo_id)
        api.upload_file(path_or_fileobj=str(save_path / "standalone_loader.py"), path_in_repo="standalone_loader.py", repo_id=repo_id)
        log.info(f"Uploaded TETNO files to {repo_id}")


def load_tetno_model(
    model_path: str | Path,
    *,
    hybrid_strength_factor: float,
    dynamic_base_strength: float,
    device_map: Optional[str] = None,
    torch_dtype=None,
    install_hooks: bool = True,
):
    """
    Load a TETNO-steered model with optional runtime hooks.
    
    Args:
        model_path: Path to saved TETNO model (local or HuggingFace repo)
        device_map: Device map for model loading
        torch_dtype: Torch dtype for model weights
        install_hooks: Whether to install runtime hooks (default: True)
        
    Returns:
        Tuple of (model, tokenizer, hooks) where hooks is TETNORuntimeHooks or None
    """
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download, HfFileSystem
    
    model_path_str = str(model_path)
    log = bind(_LOG, model_path=model_path_str)
    
    # Check if local or HuggingFace
    is_local = Path(model_path_str).exists()
    
    if is_local:
        config_path = Path(model_path_str) / "tetno_config.json"
        config_exists = config_path.exists()
    else:
        try:
            fs = HfFileSystem()
            files = fs.ls(model_path_str, detail=False)
            file_names = [f.split("/")[-1] for f in files]
            config_exists = "tetno_config.json" in file_names
        except Exception:
            config_exists = False
    
    if not config_exists:
        log.warning("No tetno_config.json found - loading as regular model")
        if is_local:
            return load_steered_model(model_path_str, device_map, torch_dtype) + (None,)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path_str, device_map=device_map, torch_dtype=torch_dtype, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path_str)
            return model, tokenizer, None
    
    # Load config
    if is_local:
        with open(Path(model_path_str) / "tetno_config.json") as f:
            tetno_config = json.load(f)
    else:
        config_file = hf_hub_download(repo_id=model_path_str, filename="tetno_config.json")
        with open(config_file) as f:
            tetno_config = json.load(f)
    
    mode = tetno_config.get("mode", "hybrid")
    log.info(f"Loading TETNO model (mode={mode})")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path_str, device_map=device_map, torch_dtype=torch_dtype, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path_str)
    
    # Load TETNO data
    hooks = None
    
    if is_local:
        data_path = Path(model_path_str) / "tetno_steering.pt"
        data_exists = data_path.exists()
    else:
        try:
            data_path = hf_hub_download(repo_id=model_path_str, filename="tetno_steering.pt")
            data_exists = True
        except Exception:
            data_exists = False
    
    if install_hooks and mode in ("dynamic", "hybrid") and data_exists:
        tetno_data = torch.load(data_path, map_location="cpu")
        
        from wisent.core.weight_modification.directional import TETNORuntimeHooks
        from types import SimpleNamespace
        
        # Reconstruct TETNOResult-like object
        tetno_result = SimpleNamespace()
        tetno_result.behavior_vectors = {k: v.to(model.device) for k, v in tetno_data["behavior_vectors"].items()}
        tetno_result.condition_vector = tetno_data["condition_vector"].to(model.device)
        tetno_result.layer_scales = tetno_data["layer_scales"]
        tetno_result.optimal_threshold = tetno_data["optimal_threshold"]
        tetno_result.metadata = tetno_data.get("metadata", {})
        
        # Extract gate_temperature from saved data
        if "gate_temperature" not in tetno_data:
            raise InsufficientDataError(reason="gate_temperature missing from saved TETNO data. Re-export the model with current wisent version.")
        saved_gate_temperature = tetno_data["gate_temperature"]

        # Add methods
        import torch.nn.functional as F

        def compute_gate(hidden_state, temperature=saved_gate_temperature):
            if hidden_state.dim() > 1:
                hidden_state = hidden_state.reshape(-1, hidden_state.shape[-1]).mean(dim=0)
            h_norm = F.normalize(hidden_state, p=2, dim=-1)
            c_norm = F.normalize(tetno_result.condition_vector, p=2, dim=-1)
            similarity = (h_norm * c_norm).sum()
            return torch.sigmoid((similarity - tetno_result.optimal_threshold) / temperature)
        
        tetno_result.compute_gate = compute_gate
        
        hooks = TETNORuntimeHooks(
            model=model,
            tetno_result=tetno_result,
            base_strength=hybrid_strength_factor if mode == "hybrid" else dynamic_base_strength,
            gate_temperature=saved_gate_temperature,
        )
        hooks.install()
        log.info(f"Installed TETNO runtime hooks (threshold={tetno_result.optimal_threshold:.3f})")
    
    return model, tokenizer, hooks


