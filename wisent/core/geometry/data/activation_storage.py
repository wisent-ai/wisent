"""Storage utilities for steering activations.

Saves and loads base/steered activations alongside visualization outputs
so they don't need to be regenerated every time.
"""
from pathlib import Path
from typing import List, Dict, Any, Union
import torch


def save_steering_activations(
    output_path: Union[str, Path],
    base_activations: torch.Tensor,
    steered_activations: torch.Tensor,
    pos_ref: torch.Tensor,
    neg_ref: torch.Tensor,
    base_evaluations: List[str],
    steered_evaluations: List[str],
    base_space_probs: List[float],
    steered_space_probs: List[float],
    test_ids: List,
    extraction_strategy: str,
    model: str,
    task: str,
    layer: Union[int, List[str]],
    strength: float,
) -> Path:
    """
    Save steering activations and metadata for reuse.

    Args:
        output_path: Path to visualization output (will save as .activations.pt)
        base_activations: Activations from unsteered generation
        steered_activations: Activations from steered generation
        pos_ref: Positive reference activations (contrastive pairs)
        neg_ref: Negative reference activations (contrastive pairs)
        base_evaluations: Text evaluation labels for base outputs
        steered_evaluations: Text evaluation labels for steered outputs
        base_space_probs: Classifier probabilities for base activations
        steered_space_probs: Classifier probabilities for steered activations
        test_ids: IDs of test samples
        extraction_strategy: Strategy used (e.g., "chat_last")
        model: Model name
        task: Task name
        layer: Layer(s) used for steering
        strength: Steering strength

    Returns:
        Path to saved activations file
    """
    output_path = Path(output_path)
    activations_path = output_path.with_suffix('.activations.pt')

    torch.save({
        "base_activations": base_activations.cpu(),
        "steered_activations": steered_activations.cpu(),
        "pos_ref": pos_ref.cpu(),
        "neg_ref": neg_ref.cpu(),
        "base_evaluations": base_evaluations,
        "steered_evaluations": steered_evaluations,
        "base_space_probs": list(base_space_probs),
        "steered_space_probs": list(steered_space_probs),
        "test_ids": list(test_ids),
        "extraction_strategy": extraction_strategy,
        "model": model,
        "task": task,
        "layer": layer,
        "strength": strength,
    }, activations_path)

    print(f"Activations saved to: {activations_path}")
    return activations_path


def load_steering_activations(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load saved steering activations.

    Args:
        path: Path to .activations.pt file or the visualization .png file

    Returns:
        Dict with all saved data including activations and metadata
    """
    path = Path(path)
    if path.suffix != '.pt':
        path = path.with_suffix('.activations.pt')

    if not path.exists():
        raise FileNotFoundError(f"Activations file not found: {path}")

    data = torch.load(path, weights_only=False)
    print(f"Loaded activations from: {path}")
    print(f"  Model: {data['model']}, Task: {data['task']}")
    print(f"  Layer: {data['layer']}, Strength: {data['strength']}")
    print(f"  Samples: {len(data['base_activations'])}")
    print(f"  Extraction strategy: {data['extraction_strategy']}")

    return data


def get_activation_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get summary statistics from loaded activation data.

    Args:
        data: Loaded activation data from load_steering_activations

    Returns:
        Summary dict with key metrics
    """
    base_evals = data["base_evaluations"]
    steered_evals = data["steered_evaluations"]
    base_probs = data["base_space_probs"]
    steered_probs = data["steered_space_probs"]

    base_truthful = sum(1 for e in base_evals if e == "TRUTHFUL")
    steered_truthful = sum(1 for e in steered_evals if e == "TRUTHFUL")
    total = len(base_evals)

    base_in_region = sum(1 for p in base_probs if p >= 0.5)
    steered_in_region = sum(1 for p in steered_probs if p >= 0.5)

    return {
        "text_evaluation": {
            "base_truthful": base_truthful,
            "steered_truthful": steered_truthful,
            "total": total,
            "base_rate": base_truthful / total if total > 0 else 0,
            "steered_rate": steered_truthful / total if total > 0 else 0,
            "delta": (steered_truthful - base_truthful) / total if total > 0 else 0,
        },
        "activation_space": {
            "base_in_truthful_region": base_in_region,
            "steered_in_truthful_region": steered_in_region,
            "total": total,
            "base_rate": base_in_region / total if total > 0 else 0,
            "steered_rate": steered_in_region / total if total > 0 else 0,
            "delta": (steered_in_region - base_in_region) / total if total > 0 else 0,
        },
        "diagnosis": _diagnose(base_truthful, steered_truthful, base_in_region, steered_in_region),
    }


def _diagnose(base_text, steered_text, base_act, steered_act) -> str:
    """Diagnose steering effectiveness."""
    text_improved = steered_text > base_text
    acts_moved = steered_act > base_act

    if acts_moved and text_improved:
        return "EFFECTIVE"
    elif acts_moved and not text_improved:
        return "IMPROPERLY_IDENTIFIED"
    elif not acts_moved and text_improved:
        return "UNEXPECTED_IMPROVEMENT"
    else:
        return "INEFFECTIVE"
