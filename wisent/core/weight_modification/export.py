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

    # Save model
    model.save_pretrained(save_path)
    log.info("Model saved successfully")

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
