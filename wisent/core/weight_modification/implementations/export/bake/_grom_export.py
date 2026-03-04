"""GROM model export functionality."""
from __future__ import annotations

import torch
from pathlib import Path
from typing import TYPE_CHECKING
from wisent.core.utils.cli.cli_logger import setup_logger, bind
from wisent.core.utils.infra_tools.errors import MissingParameterError
from wisent.core.utils.config_tools.constants import JSON_INDENT, RECURSION_INITIAL_DEPTH
from wisent.core.weight_modification.export._generic import (
    _save_standalone_loader,
)

if TYPE_CHECKING:
    from torch.nn import Module

_LOG = setup_logger(__name__)

def export_grom_model(
    model: Module,
    grom_result,
    save_path: str | Path,
    mode: str,
    tokenizer=None,
    push_to_hub: bool = False,
    repo_id: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Export a GROM-steered model with full dynamic steering support.
    
    This saves:
    1. The model weights (with static directions baked in for hybrid/static mode)
    2. GROM networks (gate network, intensity network)
    3. GROM directions and metadata
    
    The saved model can be loaded with `load_grom_model()` to get full
    dynamic steering behavior at inference time.
    
    Args:
        model: Model to export
        grom_result: GROMResult from GROM training
        save_path: Directory to save model
        tokenizer: Optional tokenizer to save
        mode: GROM mode - "static", "dynamic", or "hybrid"
        push_to_hub: Whether to upload to HuggingFace Hub
        repo_id: HuggingFace repo ID
        commit_message: Commit message for Hub upload
        
    Example:
        >>> grom_result = grom_method.train_grom(pair_set)
        >>> export_grom_model(model, grom_result, "./my_grom_model", mode="hybrid")
        >>> 
        >>> # Later, load with hooks:
        >>> model, tokenizer, hooks = load_grom_model("./my_grom_model")
    """
    import json
    
    log = bind(_LOG, save_path=str(save_path))
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Apply GROM steering to model (static component)
    if mode in ("static", "hybrid"):
        from wisent.core.weight_modification.directional import project_weights_grom
        
        log.info(f"Baking GROM directions into weights (mode={mode})")
        project_weights_grom(
            model=model,
            grom_result=grom_result,
            base_strength=1.0,
            base_layer_weight=grom_result.metadata.get("base_layer_weight"),
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
                json.dump(config, f, indent=JSON_INDENT)
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

    # Step 4: Save GROM components
    grom_save_path = save_path / "grom_steering.pt"
    
    grom_data = {
        "mode": mode,
        "layer_order": grom_result.layer_order,
        "directions": {k: v.cpu() for k, v in grom_result.directions.items()},
        "direction_weights": {k: v.cpu() for k, v in grom_result.direction_weights.items()},
        "gate_temperature": grom_result.gate_temperature,
        "max_alpha": grom_result.metadata.get("config", {}).get("max_alpha", grom_result.intensity_network.max_alpha),
    }
    
    # Save networks if present
    if hasattr(grom_result, 'gate_network') and grom_result.gate_network is not None:
        grom_data["gate_network_state"] = grom_result.gate_network.state_dict()
        first_layer = grom_result.gate_network.net[RECURSION_INITIAL_DEPTH]
        grom_data["gate_network_config"] = {
            "shrink_factor": grom_result.gate_network.shrink_factor,
            "input_dim": first_layer.in_features,
            "hidden_dim": first_layer.out_features,
        }
    
    if hasattr(grom_result, 'intensity_network') and grom_result.intensity_network is not None:
        grom_data["intensity_network_state"] = grom_result.intensity_network.state_dict()
        grom_data["intensity_network_config"] = {
            "input_dim": grom_result.intensity_network.net[0].in_features,
            "num_layers": len(grom_result.layer_order),
            "hidden_dim": grom_result.intensity_network.net[0].out_features,
        }
    
    if hasattr(grom_result, 'sensor_layer'):
        grom_data["sensor_layer"] = grom_result.sensor_layer
    
    torch.save(grom_data, grom_save_path)
    log.info(f"Saved GROM steering data to grom_steering.pt")
    
    # Step 5: Save config indicating this is a GROM model
    config_path = save_path / "grom_config.json"
    grom_config = {
        "is_grom_model": True,
        "mode": mode,
        "num_layers": len(grom_result.layer_order),
        "layer_order": grom_result.layer_order,
    }
    with open(config_path, 'w') as f:
        json.dump(grom_config, f, indent=JSON_INDENT)
    log.info("Saved GROM config")
    
    # Step 6: Save standalone loader for use without wisent
    _save_standalone_loader(save_path)
    log.info("Saved standalone_loader.py")
    
    # Step 7: Push to hub if requested
    if push_to_hub:
        if repo_id is None:
            raise MissingParameterError(params=["repo_id"], context="push_to_hub=True")
        
        from wisent.core.weight_modification.export._hub import upload_to_hub
        upload_to_hub(
            model,
            repo_id,
            tokenizer=tokenizer,
            commit_message=commit_message,
        )

        # Also upload GROM files
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(grom_save_path),
            path_in_repo="grom_steering.pt",
            repo_id=repo_id,
        )
        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo="grom_config.json",
            repo_id=repo_id,
        )
        api.upload_file(
            path_or_fileobj=str(save_path / "standalone_loader.py"),
            path_in_repo="standalone_loader.py",
            repo_id=repo_id,
        )
        log.info(f"Uploaded GROM files to {repo_id}")

