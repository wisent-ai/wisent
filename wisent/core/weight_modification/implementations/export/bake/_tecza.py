"""TECZA model export and loading functionality."""
from __future__ import annotations

import torch
from pathlib import Path
from typing import TYPE_CHECKING
from wisent.core.utils.cli.cli_logger import setup_logger, bind
from wisent.core.utils.infra_tools.errors import MissingParameterError
from wisent.core.utils.config_tools.constants import DEFAULT_STRENGTH, JSON_INDENT

if TYPE_CHECKING:
    from torch.nn import Module

_LOG = setup_logger(__name__)

def export_tecza_model(
    model: Module,
    tecza_result,
    save_path: str | Path,
    tokenizer=None,
    mode: str = "weighted",
    strength: float = DEFAULT_STRENGTH,
    push_to_hub: bool = False,
    repo_id: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Export a TECZA-steered model with multi-directional steering.
    
    TECZA has multiple directions per layer. Export modes:
    - "primary": Only bake the primary (strongest) direction
    - "weighted": Bake weighted sum of all directions
    - "full": Save all directions for runtime selection
    
    Args:
        model: Base model to modify
        tecza_result: MultiDirectionResult from training
        save_path: Where to save the model
        tokenizer: Optional tokenizer to save
        mode: "primary", "weighted", or "full"
        strength: Steering strength
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
    
    log.info(f"Exporting TECZA model (mode={mode})")
    
    # Step 1: Create effective steering vectors based on mode
    if mode == "primary":
        # Use only the first/primary direction
        effective_vectors = {layer: dirs[0] for layer, dirs in tecza_result.directions.items()}
    elif mode == "weighted":
        # Weighted average of directions (uniform weights)
        effective_vectors = {}
        for layer, dirs in tecza_result.directions.items():
            effective_vectors[layer] = dirs.mean(dim=0)
    else:  # full - bake mean but save all
        effective_vectors = {}
        for layer, dirs in tecza_result.directions.items():
            effective_vectors[layer] = dirs.mean(dim=0)
    
    # Step 2: Bake into weights (convert layer names to integer indices)
    int_keyed_vectors = {}
    for layer_name, vec in effective_vectors.items():
        try:
            layer_idx = int(str(layer_name).split("_")[-1])
            int_keyed_vectors[layer_idx] = vec
        except (ValueError, IndexError):
            pass
    
    steering_vectors = LayerActivations(int_keyed_vectors)
    bake_steering_into_weights(
        model=model,
        steering_vectors=steering_vectors,
        alpha=strength,
    )
    log.info(f"Baked TECZA vectors (mode={mode}, strength={strength})")
    
    # Step 3: Save model
    model.save_pretrained(save_path)
    log.info("Model saved successfully")
    
    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)
        log.info("Tokenizer saved successfully")
    
    # Step 4: Save full TECZA data (for analysis or runtime selection)
    tecza_save_path = save_path / "tecza_steering.pt"
    
    tecza_data = {
        "mode": mode,
        "directions": {k: v.cpu() for k, v in tecza_result.directions.items()},
        "metadata": tecza_result.metadata,
        "effective_vectors": {k: v.cpu() for k, v in effective_vectors.items()},
    }
    
    torch.save(tecza_data, tecza_save_path)
    log.info("Saved TECZA steering data")
    
    # Step 5: Save config
    config_path = save_path / "tecza_config.json"
    num_directions = next(iter(tecza_result.directions.values())).shape[0]
    tecza_config = {
        "is_tecza_model": True,
        "mode": mode,
        "num_layers": len(tecza_result.directions),
        "num_directions": num_directions,
        "layer_order": list(tecza_result.directions.keys()),
    }
    with open(config_path, 'w') as f:
        json.dump(tecza_config, f, indent=JSON_INDENT)
    log.info("Saved TECZA config")
    
    # Step 6: Push to hub
    if push_to_hub:
        if repo_id is None:
            raise MissingParameterError(params=["repo_id"], context="push_to_hub=True")
        
        from wisent.core.weight_modification.export._hub import upload_to_hub
        upload_to_hub(model, repo_id, tokenizer=tokenizer, commit_message=commit_message)
        
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(path_or_fileobj=str(tecza_save_path), path_in_repo="tecza_steering.pt", repo_id=repo_id)
        api.upload_file(path_or_fileobj=str(config_path), path_in_repo="tecza_config.json", repo_id=repo_id)
        log.info(f"Uploaded TECZA files to {repo_id}")


def load_tecza_model(
    model_path: str | Path,
    device_map: str = "auto",
    torch_dtype=None,
):
    """
    Load a TECZA-steered model.
    
    TECZA models have steering baked in but also save all directions
    for analysis. No runtime hooks needed - directions are already baked.
    
    Args:
        model_path: Path to saved TECZA model (local or HuggingFace repo)
        device_map: Device map for model loading
        torch_dtype: Torch dtype for model weights
        
    Returns:
        Tuple of (model, tokenizer, tecza_data) where tecza_data contains all directions
    """
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download, HfFileSystem
    
    model_path_str = str(model_path)
    log = bind(_LOG, model_path=model_path_str)
    
    is_local = Path(model_path_str).exists()
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path_str, device_map=device_map, torch_dtype=torch_dtype, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path_str)
    
    # Try to load TECZA data
    tecza_data = None
    
    if is_local:
        data_path = Path(model_path_str) / "tecza_steering.pt"
        if data_path.exists():
            tecza_data = torch.load(data_path, map_location="cpu")
            log.info(f"Loaded TECZA data ({tecza_data['mode']} mode)")
    else:
        try:
            data_path = hf_hub_download(repo_id=model_path_str, filename="tecza_steering.pt")
            tecza_data = torch.load(data_path, map_location="cpu")
            log.info(f"Loaded TECZA data from HuggingFace")
        except Exception:
            pass
    
    return model, tokenizer, tecza_data


