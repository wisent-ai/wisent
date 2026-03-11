"""Generic export and load utilities for steered models."""
from __future__ import annotations

import torch
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from wisent.core.utils.cli.cli_logger import setup_logger, bind
from wisent.core.utils.config_tools.constants import JSON_INDENT
from wisent.core.utils.infra_tools.errors import MissingParameterError

if TYPE_CHECKING:
    from torch.nn import Module
    from torch import Tensor

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
                json.dump(config, f, indent=JSON_INDENT)
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

        from wisent.core.weight_modification.export._hub import upload_to_hub
        upload_to_hub(
            model,
            repo_id,
            tokenizer=tokenizer,
            commit_message=commit_message,
        )


def load_steered_model(
    model_path: str | Path,
    device_map: Optional[str] = None,
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
        mlp_biases = torch.load(mlp_bias_path, map_location="cpu", weights_only=False)
        
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


