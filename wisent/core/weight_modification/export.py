"""
Export and save modified models.

Functions for saving modified models to disk and uploading to HuggingFace Hub.
"""

from __future__ import annotations

import torch
from pathlib import Path
from typing import TYPE_CHECKING
from wisent.core.cli_logger import setup_logger, bind
from wisent.core.errors import MissingParameterError

if TYPE_CHECKING:
    from torch.nn import Module
    from torch import Tensor

__all__ = [
    "export_modified_model",
    "export_titan_model",
    "export_pulse_model",
    "export_prism_model",
    "load_steered_model",
    "load_titan_model",
    "load_pulse_model",
    "load_prism_model",
    "save_modified_weights",
    "compare_models",
    "upload_to_hub",
]

_LOG = setup_logger(__name__)


def export_modified_model(
    model: Module,
    save_path: str | Path,
    tokenizer=None,
    push_to_hub: bool = False,
    repo_id: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Export modified model to disk and optionally to HuggingFace Hub.

    Args:
        model: Modified model to export
        save_path: Local path to save model
        tokenizer: Optional tokenizer to save alongside model
        push_to_hub: Whether to upload to HuggingFace Hub
        repo_id: HuggingFace repo ID (e.g., "username/model-name")
        commit_message: Commit message for Hub upload

    Example:
        >>> from wisent.core.weight_modification import (
        ...     project_weights,
        ...     export_modified_model
        ... )
        >>> # Modify model
        >>> project_weights(model, steering_vectors)
        >>> # Export
        >>> export_modified_model(
        ...     model,
        ...     "path/to/modified-model",
        ...     tokenizer=tokenizer,
        ...     push_to_hub=True,
        ...     repo_id="username/llama-3-8b-modified",
        ... )
    """
    log = bind(_LOG, save_path=str(save_path))

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    log.info("Saving model to disk", extra={"path": str(save_path)})

    # Save model (config is automatically saved with updated bias settings)
    model.save_pretrained(save_path)
    log.info("Model saved successfully")
    
    # Verify config has bias enabled if biases exist in model
    config_path = save_path / "config.json"
    if config_path.exists():
        import json
        with open(config_path) as f:
            config = json.load(f)
        # Check if model has attention bias parameters
        has_attn_bias = any('bias' in name for name in model.state_dict().keys() if 'self_attn' in name)
        # Check if model has MLP bias parameters
        has_mlp_bias = any('bias' in name for name in model.state_dict().keys() if 'mlp' in name)
        
        config_updated = False
        if has_attn_bias:
            config['attention_bias'] = True
            config_updated = True
        if has_mlp_bias:
            config['mlp_bias'] = True
            config_updated = True
        
        if config_updated:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            log.info(f"Updated config: attention_bias={has_attn_bias}, mlp_bias={has_mlp_bias}")

    # Save MLP biases separately (transformers doesn't support mlp_bias config)
    mlp_biases = {}
    state_dict = model.state_dict()
    for name, param in state_dict.items():
        if 'mlp' in name and 'bias' in name:
            mlp_biases[name] = param.cpu()
    
    if mlp_biases:
        import torch
        mlp_bias_path = save_path / "mlp_biases.pt"
        torch.save(mlp_biases, mlp_bias_path)
        log.info(f"Saved {len(mlp_biases)} MLP biases to mlp_biases.pt")

    # Save tokenizer if provided
    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)
        log.info("Tokenizer saved successfully")

    # Push to Hub if requested
    if push_to_hub:
        if repo_id is None:
            raise MissingParameterError(params=["repo_id"], context="push_to_hub=True")

        upload_to_hub(
            model,
            repo_id,
            tokenizer=tokenizer,
            commit_message=commit_message,
        )


def load_steered_model(
    model_path: str | Path,
    device_map: str = "auto",
    torch_dtype=None,
):
    """
    Load a steered model with MLP biases properly applied.
    
    This handles the case where MLP biases were added during steering
    but transformers doesn't support mlp_bias config natively.
    
    Args:
        model_path: Path to the saved steered model
        device_map: Device map for model loading
        torch_dtype: Torch dtype for model weights
        
    Returns:
        Tuple of (model, tokenizer)
        
    Example:
        >>> model, tokenizer = load_steered_model("./steered_model")
        >>> # Model now has MLP biases properly loaded
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_path = Path(model_path)
    log = bind(_LOG, model_path=str(model_path))
    
    # Load model normally
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Check for MLP biases file
    mlp_bias_path = model_path / "mlp_biases.pt"
    if mlp_bias_path.exists():
        log.info("Loading MLP biases from mlp_biases.pt")
        mlp_biases = torch.load(mlp_bias_path, map_location="cpu")
        
        # Get model layers
        if hasattr(model, "model"):
            layers = model.model.layers
        elif hasattr(model, "transformer"):
            layers = model.transformer.h
        else:
            layers = model.layers
        
        # Add biases to MLP layers
        for name, bias_value in mlp_biases.items():
            # Parse layer index from name like "model.layers.17.mlp.down_proj.bias"
            parts = name.split(".")
            layer_idx = None
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        break
                    except ValueError:
                        continue
            
            if layer_idx is not None and layer_idx < len(layers):
                layer = layers[layer_idx]
                # Find the component (e.g., mlp.down_proj)
                if "down_proj" in name:
                    component = layer.mlp.down_proj
                elif "gate_proj" in name:
                    component = layer.mlp.gate_proj
                elif "up_proj" in name:
                    component = layer.mlp.up_proj
                else:
                    continue
                
                # Add bias if it doesn't exist
                if component.bias is None:
                    component.bias = torch.nn.Parameter(
                        bias_value.to(device=component.weight.device, dtype=component.weight.dtype)
                    )
                    log.info(f"Added MLP bias to layer {layer_idx}: {name}")
        
        log.info(f"Loaded {len(mlp_biases)} MLP biases")
    
    return model, tokenizer


def _save_standalone_loader(save_path: Path) -> None:
    """Save standalone_loader.py to the model directory."""
    import shutil
    loader_src = Path(__file__).parent / "standalone_loader.py"
    loader_dst = save_path / "standalone_loader.py"
    if loader_src.exists():
        shutil.copy(loader_src, loader_dst)


def export_titan_model(
    model: Module,
    titan_result,
    save_path: str | Path,
    tokenizer=None,
    mode: str = "hybrid",
    push_to_hub: bool = False,
    repo_id: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Export a TITAN-steered model with full dynamic steering support.
    
    This saves:
    1. The model weights (with static directions baked in for hybrid/static mode)
    2. TITAN networks (gate network, intensity network)
    3. TITAN directions and metadata
    
    The saved model can be loaded with `load_titan_model()` to get full
    dynamic steering behavior at inference time.
    
    Args:
        model: Model to export
        titan_result: TITANResult from TITAN training
        save_path: Directory to save model
        tokenizer: Optional tokenizer to save
        mode: TITAN mode - "static", "dynamic", or "hybrid"
        push_to_hub: Whether to upload to HuggingFace Hub
        repo_id: HuggingFace repo ID
        commit_message: Commit message for Hub upload
        
    Example:
        >>> titan_result = titan_method.train_titan(pair_set)
        >>> export_titan_model(model, titan_result, "./my_titan_model", mode="hybrid")
        >>> 
        >>> # Later, load with hooks:
        >>> model, tokenizer, hooks = load_titan_model("./my_titan_model")
    """
    import json
    
    log = bind(_LOG, save_path=str(save_path))
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Apply TITAN steering to model (static component)
    if mode in ("static", "hybrid"):
        from wisent.core.weight_modification.directional import project_weights_titan
        
        log.info(f"Baking TITAN directions into weights (mode={mode})")
        project_weights_titan(
            model=model,
            titan_result=titan_result,
            base_strength=1.0,
            use_learned_intensities=True,
            verbose=False,
        )
    
    # Step 2: Save model weights
    log.info("Saving model to disk")
    model.save_pretrained(save_path)
    log.info("Model saved successfully")

    # Step 2b: Update config to enable biases if they were added
    config_path = save_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        # Check if model has attention bias parameters
        has_attn_bias = any('bias' in name for name in model.state_dict().keys() if 'self_attn' in name)
        # Check if model has MLP bias parameters
        has_mlp_bias = any('bias' in name for name in model.state_dict().keys() if 'mlp' in name)

        config_updated = False
        if has_attn_bias:
            config['attention_bias'] = True
            config_updated = True
        if has_mlp_bias:
            config['mlp_bias'] = True
            config_updated = True

        if config_updated:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            log.info(f"Updated config: attention_bias={has_attn_bias}, mlp_bias={has_mlp_bias}")

    # Step 2c: Save MLP biases to separate file for models that don't support mlp_bias config
    # This allows load_steered_model to manually add them after loading
    mlp_biases = {}
    for name, param in model.named_parameters():
        if 'mlp' in name and 'bias' in name:
            mlp_biases[name] = param.cpu()
    if mlp_biases:
        mlp_bias_path = save_path / "mlp_biases.pt"
        torch.save(mlp_biases, mlp_bias_path)
        log.info(f"Saved {len(mlp_biases)} MLP biases to mlp_biases.pt")

    # Step 3: Save tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)
        log.info("Tokenizer saved successfully")

    # Step 4: Save TITAN components
    titan_save_path = save_path / "titan_steering.pt"
    
    titan_data = {
        "mode": mode,
        "layer_order": titan_result.layer_order,
        "directions": {k: v.cpu() for k, v in titan_result.directions.items()},
        "direction_weights": {k: v.cpu() for k, v in titan_result.direction_weights.items()},
        "gate_temperature": getattr(titan_result, 'gate_temperature', 0.5),
        "max_alpha": getattr(titan_result, 'max_alpha', 3.0),
    }
    
    # Save networks if present
    if hasattr(titan_result, 'gate_network') and titan_result.gate_network is not None:
        titan_data["gate_network_state"] = titan_result.gate_network.state_dict()
        titan_data["gate_network_config"] = {
            "input_dim": titan_result.gate_network.net[0].in_features,
            "hidden_dim": titan_result.gate_network.net[0].out_features,
        }
    
    if hasattr(titan_result, 'intensity_network') and titan_result.intensity_network is not None:
        titan_data["intensity_network_state"] = titan_result.intensity_network.state_dict()
        titan_data["intensity_network_config"] = {
            "input_dim": titan_result.intensity_network.net[0].in_features,
            "num_layers": len(titan_result.layer_order),
            "hidden_dim": titan_result.intensity_network.net[0].out_features,
        }
    
    if hasattr(titan_result, 'sensor_layer'):
        titan_data["sensor_layer"] = titan_result.sensor_layer
    
    torch.save(titan_data, titan_save_path)
    log.info(f"Saved TITAN steering data to titan_steering.pt")
    
    # Step 5: Save config indicating this is a TITAN model
    config_path = save_path / "titan_config.json"
    titan_config = {
        "is_titan_model": True,
        "mode": mode,
        "num_layers": len(titan_result.layer_order),
        "layer_order": titan_result.layer_order,
    }
    with open(config_path, 'w') as f:
        json.dump(titan_config, f, indent=2)
    log.info("Saved TITAN config")
    
    # Step 6: Save standalone loader for use without wisent
    _save_standalone_loader(save_path)
    log.info("Saved standalone_loader.py")
    
    # Step 7: Push to hub if requested
    if push_to_hub:
        if repo_id is None:
            raise MissingParameterError(params=["repo_id"], context="push_to_hub=True")
        
        upload_to_hub(
            model,
            repo_id,
            tokenizer=tokenizer,
            commit_message=commit_message,
        )
        
        # Also upload TITAN files
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(titan_save_path),
            path_in_repo="titan_steering.pt",
            repo_id=repo_id,
        )
        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo="titan_config.json",
            repo_id=repo_id,
        )
        api.upload_file(
            path_or_fileobj=str(save_path / "standalone_loader.py"),
            path_in_repo="standalone_loader.py",
            repo_id=repo_id,
        )
        log.info(f"Uploaded TITAN files to {repo_id}")


def load_titan_model(
    model_path: str | Path,
    device_map: str = "auto",
    torch_dtype=None,
    install_hooks: bool = True,
):
    """
    Load a TITAN-steered model with optional runtime hooks.
    
    This loads the model and optionally installs TITAN runtime hooks
    for dynamic steering based on input content.
    
    Args:
        model_path: Path to saved TITAN model
        device_map: Device map for model loading
        torch_dtype: Torch dtype for model weights
        install_hooks: Whether to install runtime hooks (default: True)
        
    Returns:
        Tuple of (model, tokenizer, hooks) where hooks is None if install_hooks=False
        
    Example:
        >>> model, tokenizer, hooks = load_titan_model("./my_titan_model")
        >>> 
        >>> # Generate with dynamic steering
        >>> output = model.generate(...)
        >>> 
        >>> # Check gate value
        >>> print(f"Gate: {hooks.get_current_gate()}")
        >>> 
        >>> # Remove hooks when done
        >>> hooks.remove()
    """
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download, HfFileSystem
    
    model_path_str = str(model_path)
    log = bind(_LOG, model_path=model_path_str)
    
    # Check if this is a local path or HuggingFace repo
    is_local = Path(model_path_str).exists()
    
    if is_local:
        model_path = Path(model_path_str)
        titan_config_path = model_path / "titan_config.json"
        titan_data_path = model_path / "titan_steering.pt"
        config_exists = titan_config_path.exists()
    else:
        # HuggingFace Hub repo - check if titan files exist
        try:
            fs = HfFileSystem()
            files = fs.ls(model_path_str, detail=False)
            file_names = [f.split("/")[-1] for f in files]
            config_exists = "titan_config.json" in file_names
        except Exception:
            config_exists = False
    
    if not config_exists:
        log.warning("No titan_config.json found - loading as regular model")
        if is_local:
            return load_steered_model(model_path_str, device_map, torch_dtype) + (None,)
        else:
            # Load from HF without TITAN
            model = AutoModelForCausalLM.from_pretrained(model_path_str, device_map=device_map, torch_dtype=torch_dtype, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path_str)
            return model, tokenizer, None
    
    # Load TITAN config
    if is_local:
        with open(titan_config_path) as f:
            titan_config = json.load(f)
    else:
        # Download from HF Hub
        config_file = hf_hub_download(repo_id=model_path_str, filename="titan_config.json")
        with open(config_file) as f:
            titan_config = json.load(f)
    
    mode = titan_config.get("mode", "hybrid")
    log.info(f"Loading TITAN model (mode={mode})")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path_str,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path_str)
    
    # Load TITAN data
    hooks = None
    
    # Get titan_steering.pt path (local or download from HF)
    if is_local:
        titan_data_path = Path(model_path_str) / "titan_steering.pt"
        titan_data_exists = titan_data_path.exists()
    else:
        try:
            titan_data_path = hf_hub_download(repo_id=model_path_str, filename="titan_steering.pt")
            titan_data_exists = True
        except Exception:
            titan_data_exists = False
    
    if install_hooks and mode in ("dynamic", "hybrid") and titan_data_exists:
        titan_data = torch.load(titan_data_path, map_location="cpu")
        
        # Reconstruct TITANResult-like object for hooks
        from wisent.core.weight_modification.directional import TITANRuntimeHooks
        from types import SimpleNamespace
        
        # Create a minimal titan_result object with required attributes
        titan_result = SimpleNamespace()
        titan_result.layer_order = titan_data["layer_order"]
        titan_result.directions = {k: v.to(model.device) for k, v in titan_data["directions"].items()}
        titan_result.direction_weights = {k: v.to(model.device) for k, v in titan_data["direction_weights"].items()}
        titan_result.gate_temperature = titan_data.get("gate_temperature", 0.5)
        titan_result.max_alpha = titan_data.get("max_alpha", 3.0)
        
        # Reconstruct gate network
        if "gate_network_state" in titan_data:
            from wisent.core.steering_methods.methods.titan import GatingNetwork
            config = titan_data["gate_network_config"]
            titan_result.gate_network = GatingNetwork(
                config["input_dim"],
                config.get("hidden_dim", 128),
            )
            titan_result.gate_network.load_state_dict(titan_data["gate_network_state"])
            titan_result.gate_network = titan_result.gate_network.to(model.device)
        else:
            titan_result.gate_network = None
        
        # Reconstruct intensity network
        if "intensity_network_state" in titan_data:
            from wisent.core.steering_methods.methods.titan import IntensityNetwork
            config = titan_data["intensity_network_config"]
            titan_result.intensity_network = IntensityNetwork(
                config["input_dim"],
                config["num_layers"],
                config.get("hidden_dim", 64),
            )
            titan_result.intensity_network.load_state_dict(titan_data["intensity_network_state"])
            titan_result.intensity_network = titan_result.intensity_network.to(model.device)
        
        # Add required methods
        def get_effective_direction(layer_name):
            dirs = titan_result.directions.get(layer_name)
            weights = titan_result.direction_weights.get(layer_name)
            if dirs is None:
                return None
            if weights is not None:
                return (dirs * weights.unsqueeze(-1)).sum(0)
            return dirs.mean(0)
        
        def predict_gate(hidden_state):
            if titan_result.gate_network is not None:
                return titan_result.gate_network(hidden_state.float()).to(hidden_state.dtype)
            return torch.ones(hidden_state.shape[0], device=hidden_state.device, dtype=hidden_state.dtype)
        
        def predict_intensity(hidden_state):
            if titan_result.intensity_network is not None:
                intensities = titan_result.intensity_network(hidden_state.float())
                return {layer: intensities[:, i] for i, layer in enumerate(titan_result.layer_order)}
            return {layer: torch.ones(1, device=hidden_state.device) for layer in titan_result.layer_order}
        
        titan_result.get_effective_direction = get_effective_direction
        titan_result.predict_gate = predict_gate
        titan_result.predict_intensity = predict_intensity
        
        # Add metadata dict for TITANRuntimeHooks compatibility
        sensor_layer = titan_data.get("sensor_layer", titan_result.layer_order[len(titan_result.layer_order)//2])
        titan_result.metadata = {"sensor_layer": sensor_layer}
        titan_result.sensor_layer = sensor_layer
        
        # Install hooks
        hooks = TITANRuntimeHooks(
            model=model,
            titan_result=titan_result,
            base_strength=1.0,
            use_soft_gating=True,
        )
        hooks.install()
        log.info(f"Installed TITAN runtime hooks (sensor_layer={titan_result.sensor_layer})")
    
    return model, tokenizer, hooks


def export_pulse_model(
    model: Module,
    pulse_result,
    save_path: str | Path,
    tokenizer=None,
    mode: str = "hybrid",
    strength: float = 1.0,
    push_to_hub: bool = False,
    repo_id: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Export a PULSE-steered model with conditional steering.
    
    PULSE uses a condition vector for input-dependent gating:
    - In "static" mode: Always apply steering at fixed strength
    - In "dynamic" mode: Use condition vector gating at runtime (requires hooks)
    - In "hybrid" mode: Bake partial steering + dynamic gating for remainder
    
    Args:
        model: Base model to modify
        pulse_result: PULSEResult from training
        save_path: Where to save the model
        tokenizer: Optional tokenizer to save
        mode: "static", "dynamic", or "hybrid"
        strength: Steering strength for static component
        push_to_hub: Whether to upload to HuggingFace Hub
        repo_id: HuggingFace repo ID
        commit_message: Commit message for Hub
    """
    import json
    from wisent.core.weight_modification.additive import bake_steering_into_weights
    from wisent.core.activations.core.atoms import LayerActivations
    
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    log = bind(_LOG, save_path=str(save_path))
    
    log.info(f"Exporting PULSE model (mode={mode})")
    
    # Step 1: Bake behavior vectors into weights (for static/hybrid mode)
    if mode in ("static", "hybrid"):
        bake_strength = strength if mode == "static" else strength * 0.5
        
        # Convert layer names to integer indices
        int_keyed_vectors = {}
        for layer_name, vec in pulse_result.behavior_vectors.items():
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
        )
        log.info(f"Baked PULSE vectors (strength={bake_strength})")
    
    # Step 2: Save model
    model.save_pretrained(save_path)
    log.info("Model saved successfully")
    
    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)
        log.info("Tokenizer saved successfully")
    
    # Step 3: Save PULSE components for dynamic steering
    pulse_save_path = save_path / "pulse_steering.pt"
    
    pulse_data = {
        "mode": mode,
        "behavior_vectors": {k: v.cpu() for k, v in pulse_result.behavior_vectors.items()},
        "condition_vector": pulse_result.condition_vector.cpu(),
        "layer_scales": pulse_result.layer_scales,
        "optimal_threshold": pulse_result.optimal_threshold,
        "metadata": pulse_result.metadata,
    }
    
    torch.save(pulse_data, pulse_save_path)
    log.info("Saved PULSE steering data")
    
    # Step 4: Save config
    config_path = save_path / "pulse_config.json"
    pulse_config = {
        "is_pulse_model": True,
        "mode": mode,
        "num_layers": len(pulse_result.behavior_vectors),
        "optimal_threshold": pulse_result.optimal_threshold,
        "layer_order": list(pulse_result.behavior_vectors.keys()),
    }
    with open(config_path, 'w') as f:
        json.dump(pulse_config, f, indent=2)
    log.info("Saved PULSE config")
    
    # Step 5: Save standalone loader
    _save_standalone_loader(save_path)
    log.info("Saved standalone_loader.py")
    
    # Step 6: Push to hub if requested
    if push_to_hub:
        if repo_id is None:
            raise MissingParameterError(params=["repo_id"], context="push_to_hub=True")
        
        upload_to_hub(model, repo_id, tokenizer=tokenizer, commit_message=commit_message)
        
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(path_or_fileobj=str(pulse_save_path), path_in_repo="pulse_steering.pt", repo_id=repo_id)
        api.upload_file(path_or_fileobj=str(config_path), path_in_repo="pulse_config.json", repo_id=repo_id)
        api.upload_file(path_or_fileobj=str(save_path / "standalone_loader.py"), path_in_repo="standalone_loader.py", repo_id=repo_id)
        log.info(f"Uploaded PULSE files to {repo_id}")


def load_pulse_model(
    model_path: str | Path,
    device_map: str = "auto",
    torch_dtype=None,
    install_hooks: bool = True,
):
    """
    Load a PULSE-steered model with optional runtime hooks.
    
    Args:
        model_path: Path to saved PULSE model (local or HuggingFace repo)
        device_map: Device map for model loading
        torch_dtype: Torch dtype for model weights
        install_hooks: Whether to install runtime hooks (default: True)
        
    Returns:
        Tuple of (model, tokenizer, hooks) where hooks is PULSERuntimeHooks or None
    """
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download, HfFileSystem
    
    model_path_str = str(model_path)
    log = bind(_LOG, model_path=model_path_str)
    
    # Check if local or HuggingFace
    is_local = Path(model_path_str).exists()
    
    if is_local:
        config_path = Path(model_path_str) / "pulse_config.json"
        config_exists = config_path.exists()
    else:
        try:
            fs = HfFileSystem()
            files = fs.ls(model_path_str, detail=False)
            file_names = [f.split("/")[-1] for f in files]
            config_exists = "pulse_config.json" in file_names
        except Exception:
            config_exists = False
    
    if not config_exists:
        log.warning("No pulse_config.json found - loading as regular model")
        if is_local:
            return load_steered_model(model_path_str, device_map, torch_dtype) + (None,)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path_str, device_map=device_map, torch_dtype=torch_dtype, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path_str)
            return model, tokenizer, None
    
    # Load config
    if is_local:
        with open(Path(model_path_str) / "pulse_config.json") as f:
            pulse_config = json.load(f)
    else:
        config_file = hf_hub_download(repo_id=model_path_str, filename="pulse_config.json")
        with open(config_file) as f:
            pulse_config = json.load(f)
    
    mode = pulse_config.get("mode", "hybrid")
    log.info(f"Loading PULSE model (mode={mode})")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path_str, device_map=device_map, torch_dtype=torch_dtype, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path_str)
    
    # Load PULSE data
    hooks = None
    
    if is_local:
        data_path = Path(model_path_str) / "pulse_steering.pt"
        data_exists = data_path.exists()
    else:
        try:
            data_path = hf_hub_download(repo_id=model_path_str, filename="pulse_steering.pt")
            data_exists = True
        except Exception:
            data_exists = False
    
    if install_hooks and mode in ("dynamic", "hybrid") and data_exists:
        pulse_data = torch.load(data_path, map_location="cpu")
        
        from wisent.core.weight_modification.directional import PULSERuntimeHooks
        from types import SimpleNamespace
        
        # Reconstruct PULSEResult-like object
        pulse_result = SimpleNamespace()
        pulse_result.behavior_vectors = {k: v.to(model.device) for k, v in pulse_data["behavior_vectors"].items()}
        pulse_result.condition_vector = pulse_data["condition_vector"].to(model.device)
        pulse_result.layer_scales = pulse_data["layer_scales"]
        pulse_result.optimal_threshold = pulse_data["optimal_threshold"]
        pulse_result.metadata = pulse_data.get("metadata", {})
        
        # Add methods
        import torch.nn.functional as F
        
        def compute_gate(hidden_state, temperature=0.1):
            if hidden_state.dim() > 1:
                hidden_state = hidden_state.reshape(-1, hidden_state.shape[-1]).mean(dim=0)
            h_norm = F.normalize(hidden_state, p=2, dim=-1)
            c_norm = F.normalize(pulse_result.condition_vector, p=2, dim=-1)
            similarity = (h_norm * c_norm).sum()
            return torch.sigmoid((similarity - pulse_result.optimal_threshold) / temperature)
        
        pulse_result.compute_gate = compute_gate
        
        hooks = PULSERuntimeHooks(
            model=model,
            pulse_result=pulse_result,
            base_strength=0.5 if mode == "hybrid" else 1.0,
        )
        hooks.install()
        log.info(f"Installed PULSE runtime hooks (threshold={pulse_result.optimal_threshold:.3f})")
    
    return model, tokenizer, hooks


def export_prism_model(
    model: Module,
    prism_result,
    save_path: str | Path,
    tokenizer=None,
    mode: str = "weighted",
    strength: float = 1.0,
    push_to_hub: bool = False,
    repo_id: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Export a PRISM-steered model with multi-directional steering.
    
    PRISM has multiple directions per layer. Export modes:
    - "primary": Only bake the primary (strongest) direction
    - "weighted": Bake weighted sum of all directions
    - "full": Save all directions for runtime selection
    
    Args:
        model: Base model to modify
        prism_result: MultiDirectionResult from training
        save_path: Where to save the model
        tokenizer: Optional tokenizer to save
        mode: "primary", "weighted", or "full"
        strength: Steering strength
        push_to_hub: Whether to upload to HuggingFace Hub
        repo_id: HuggingFace repo ID
        commit_message: Commit message for Hub
    """
    import json
    from wisent.core.weight_modification.additive import bake_steering_into_weights
    from wisent.core.activations.core.atoms import LayerActivations
    
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    log = bind(_LOG, save_path=str(save_path))
    
    log.info(f"Exporting PRISM model (mode={mode})")
    
    # Step 1: Create effective steering vectors based on mode
    if mode == "primary":
        # Use only the first/primary direction
        effective_vectors = {layer: dirs[0] for layer, dirs in prism_result.directions.items()}
    elif mode == "weighted":
        # Weighted average of directions (uniform weights)
        effective_vectors = {}
        for layer, dirs in prism_result.directions.items():
            effective_vectors[layer] = dirs.mean(dim=0)
    else:  # full - bake mean but save all
        effective_vectors = {}
        for layer, dirs in prism_result.directions.items():
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
    log.info(f"Baked PRISM vectors (mode={mode}, strength={strength})")
    
    # Step 3: Save model
    model.save_pretrained(save_path)
    log.info("Model saved successfully")
    
    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)
        log.info("Tokenizer saved successfully")
    
    # Step 4: Save full PRISM data (for analysis or runtime selection)
    prism_save_path = save_path / "prism_steering.pt"
    
    prism_data = {
        "mode": mode,
        "directions": {k: v.cpu() for k, v in prism_result.directions.items()},
        "metadata": prism_result.metadata,
        "effective_vectors": {k: v.cpu() for k, v in effective_vectors.items()},
    }
    
    torch.save(prism_data, prism_save_path)
    log.info("Saved PRISM steering data")
    
    # Step 5: Save config
    config_path = save_path / "prism_config.json"
    num_directions = next(iter(prism_result.directions.values())).shape[0]
    prism_config = {
        "is_prism_model": True,
        "mode": mode,
        "num_layers": len(prism_result.directions),
        "num_directions": num_directions,
        "layer_order": list(prism_result.directions.keys()),
    }
    with open(config_path, 'w') as f:
        json.dump(prism_config, f, indent=2)
    log.info("Saved PRISM config")
    
    # Step 6: Push to hub
    if push_to_hub:
        if repo_id is None:
            raise MissingParameterError(params=["repo_id"], context="push_to_hub=True")
        
        upload_to_hub(model, repo_id, tokenizer=tokenizer, commit_message=commit_message)
        
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(path_or_fileobj=str(prism_save_path), path_in_repo="prism_steering.pt", repo_id=repo_id)
        api.upload_file(path_or_fileobj=str(config_path), path_in_repo="prism_config.json", repo_id=repo_id)
        log.info(f"Uploaded PRISM files to {repo_id}")


def load_prism_model(
    model_path: str | Path,
    device_map: str = "auto",
    torch_dtype=None,
):
    """
    Load a PRISM-steered model.
    
    PRISM models have steering baked in but also save all directions
    for analysis. No runtime hooks needed - directions are already baked.
    
    Args:
        model_path: Path to saved PRISM model (local or HuggingFace repo)
        device_map: Device map for model loading
        torch_dtype: Torch dtype for model weights
        
    Returns:
        Tuple of (model, tokenizer, prism_data) where prism_data contains all directions
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
    
    # Try to load PRISM data
    prism_data = None
    
    if is_local:
        data_path = Path(model_path_str) / "prism_steering.pt"
        if data_path.exists():
            prism_data = torch.load(data_path, map_location="cpu")
            log.info(f"Loaded PRISM data ({prism_data['mode']} mode)")
    else:
        try:
            data_path = hf_hub_download(repo_id=model_path_str, filename="prism_steering.pt")
            prism_data = torch.load(data_path, map_location="cpu")
            log.info(f"Loaded PRISM data from HuggingFace")
        except Exception:
            pass
    
    return model, tokenizer, prism_data


def save_modified_weights(
    model: Module,
    save_path: str | Path,
    save_format: str = "safetensors",
) -> None:
    """
    Save only the modified weights (not full model).

    Useful for distributing weight diffs or checkpoints.

    Args:
        model: Model with modified weights
        save_path: Path to save weights file
        save_format: "safetensors" or "pytorch"

    Example:
        >>> save_modified_weights(model, "modified_weights.safetensors")
    """
    log = bind(_LOG, save_path=str(save_path))

    save_path = Path(save_path)

    if save_format == "safetensors":
        try:
            from safetensors.torch import save_file

            state_dict = model.state_dict()
            save_file(state_dict, save_path)
            log.info("Saved weights as safetensors", extra={"path": str(save_path)})

        except ImportError:
            log.warning("safetensors not installed, falling back to pytorch")
            save_format = "pytorch"

    if save_format == "pytorch":
        torch.save(model.state_dict(), save_path)
        log.info("Saved weights as pytorch", extra={"path": str(save_path)})


def upload_to_hub(
    model: Module,
    repo_id: str,
    tokenizer=None,
    commit_message: str | None = None,
    private: bool = False,
) -> None:
    """
    Upload model to HuggingFace Hub.

    Args:
        model: Model to upload
        repo_id: Repository ID (e.g., "username/model-name")
        tokenizer: Optional tokenizer to upload
        commit_message: Commit message
        private: Whether repo should be private

    Example:
        >>> upload_to_hub(
        ...     model,
        ...     "username/llama-3-8b-math-steered",
        ...     tokenizer=tokenizer,
        ...     commit_message="Add math steering from Wisent contrastive pairs",
        ... )
    """
    log = bind(_LOG, repo_id=repo_id)

    if commit_message is None:
        commit_message = "Upload modified model from Wisent"

    log.info("Uploading to HuggingFace Hub", extra={"repo_id": repo_id})

    try:
        model.push_to_hub(
            repo_id,
            commit_message=commit_message,
            private=private,
        )
        log.info("Model uploaded successfully")

        if tokenizer is not None:
            tokenizer.push_to_hub(
                repo_id,
                commit_message=commit_message,
                private=private,
            )
            log.info("Tokenizer uploaded successfully")

    except Exception as e:
        log.error(f"Failed to upload to Hub: {e}", exc_info=e)
        raise


def compare_models(
    original_model: Module,
    modified_model: Module,
    sample_layers: list[int] | None = None,
) -> dict[str, float]:
    """
    Compare original and modified models to quantify changes.

    Args:
        original_model: Original unmodified model
        modified_model: Modified model
        sample_layers: Layers to sample (None = all layers)

    Returns:
        Dictionary with comparison metrics:
        - "mean_weight_diff": Mean absolute difference in weights
        - "max_weight_diff": Maximum weight difference
        - "total_params_changed": Number of parameters that changed
        - "fraction_changed": Fraction of parameters changed

    Example:
        >>> original = AutoModelForCausalLM.from_pretrained("llama-3-8b")
        >>> modified = copy.deepcopy(original)
        >>> project_weights(modified, steering_vectors)
        >>> metrics = compare_models(original, modified)
        >>> print(f"Changed {metrics['fraction_changed']:.2%} of parameters")
    """
    log = bind(_LOG)

    # Get state dicts
    original_state = original_model.state_dict()
    modified_state = modified_model.state_dict()

    # Compute differences
    total_params = 0
    params_changed = 0
    sum_abs_diff = 0.0
    max_diff = 0.0

    for key in original_state.keys():
        if key not in modified_state:
            log.warning(f"Key {key} not in modified model")
            continue

        orig_param = original_state[key]
        mod_param = modified_state[key]

        if orig_param.shape != mod_param.shape:
            log.warning(f"Shape mismatch for {key}: {orig_param.shape} vs {mod_param.shape}")
            continue

        # Compute difference
        diff = (mod_param - orig_param).abs()

        total_params += orig_param.numel()
        params_changed += (diff > 1e-6).sum().item()
        sum_abs_diff += diff.sum().item()
        max_diff = max(max_diff, diff.max().item())

    mean_diff = sum_abs_diff / total_params if total_params > 0 else 0.0
    fraction_changed = params_changed / total_params if total_params > 0 else 0.0

    metrics = {
        "mean_weight_diff": mean_diff,
        "max_weight_diff": max_diff,
        "total_params_changed": params_changed,
        "fraction_changed": fraction_changed,
        "total_params": total_params,
    }

    log.info("Model comparison complete", extra=metrics)

    return metrics


def create_model_card(
    repo_id: str,
    base_model: str,
    modification_type: str,
    steering_description: str,
    contrastive_pairs_info: dict,
    metrics: dict | None = None,
) -> str:
    """
    Create a HuggingFace model card for modified model.

    Args:
        repo_id: Repository ID
        base_model: Base model name
        modification_type: "directional" or "additive_steering"
        steering_description: Description of what the steering does
        contrastive_pairs_info: Info about contrastive pairs used
        metrics: Optional evaluation metrics

    Returns:
        Model card markdown string

    Example:
        >>> card = create_model_card(
        ...     "username/llama-3-8b-math",
        ...     "meta-llama/Llama-3-8b",
        ...     "additive_steering",
        ...     "Enhanced mathematical reasoning",
        ...     {"num_pairs": 1000, "source": "GSM8K"},
        ...     {"accuracy": 0.85, "kl_divergence": 0.12},
        ... )
    """
    metrics_section = ""
    if metrics:
        metrics_section = "\n## Metrics\n\n"
        for key, value in metrics.items():
            metrics_section += f"- **{key}**: {value}\n"

    card = f"""---
base_model: {base_model}
tags:
- wisent
- contrastive-activation-addition
- {modification_type}
- {steering_description.lower().replace(' ', '-')}
license: same-as-base
---

# {repo_id.split('/')[-1]}

This model is a modified version of [{base_model}](https://huggingface.co/{base_model}) using [Wisent](https://github.com/wisent) with {modification_type}.

## Modification

**Type**: {modification_type}

**Description**: {steering_description}

**Method**: This model was created using Wisent's weight modification capabilities, which permanently bake contrastive steering vectors into the model weights.

## Contrastive Pairs

The steering vectors were computed from {contrastive_pairs_info.get('num_pairs', 'N/A')} contrastive pairs.

**Source**: {contrastive_pairs_info.get('source', 'Custom dataset')}

**Positive examples**: {contrastive_pairs_info.get('positive_description', 'Correct task outputs')}

**Negative examples**: {contrastive_pairs_info.get('negative_description', 'Incorrect task outputs')}
{metrics_section}
## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

# Model is ready to use - no steering hooks needed!
```

## Citation

If you use this model, please cite Wisent:

```bibtex
@software{{wisent,
  title = {{Wisent: Contrastive Activation Addition}},
  author = {{...}},
  year = {{2025}},
  url = {{https://github.com/wisent}}
}}
```

## License

This model inherits the license from the base model: {base_model}
"""

    return card
