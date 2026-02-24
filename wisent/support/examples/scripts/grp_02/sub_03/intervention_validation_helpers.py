"""Data classes and utility helpers for intervention_validation."""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field

import torch
import numpy as np
from wisent.core.constants import ZERO_THRESHOLD

GCS_BUCKET = "wisent-images-bucket"
GCS_PREFIX = "intervention_validation"


def gcs_upload_file(local_path: Path, model_name: str) -> None:
    """Upload a single file to GCS."""
    model_prefix = model_name.replace('/', '_')
    gcs_path = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/{model_prefix}/{local_path.name}"
    try:
        subprocess.run(
            ["gcloud", "storage", "cp", str(local_path), gcs_path, "--quiet"],
            check=True,
            capture_output=True,
        )
        print(f"  Uploaded to GCS: {gcs_path}")
    except Exception as e:
        print(f"  GCS upload failed: {e}")

@dataclass
class SteeringResult:
    """Result of a single steering experiment."""
    benchmark: str
    strategy: str
    layer: int
    diagnosis: str  # LINEAR, NONLINEAR, NO_SIGNAL
    
    # Before steering
    baseline_accuracy: float  # Model's baseline accuracy on task
    baseline_correct_logprob: float  # Avg logprob of correct answer
    baseline_incorrect_logprob: float  # Avg logprob of incorrect answer
    
    # After steering (with CAA)
    steered_accuracy: float
    steered_correct_logprob: float
    steered_incorrect_logprob: float
    
    # Steering effect
    accuracy_change: float  # steered - baseline (positive = improvement)
    logprob_shift: float  # Change in correct - incorrect gap
    steering_success: bool  # Did steering improve in expected direction?
    
    # Steering parameters
    steering_coefficient: float
    num_test_samples: int


@dataclass
class ValidationResults:
    """Results from full intervention validation."""
    model: str
    results: List[SteeringResult] = field(default_factory=list)
    
    # Summary statistics by diagnosis
    linear_success_rate: float = 0.0
    nonlinear_success_rate: float = 0.0
    no_signal_success_rate: float = 0.0
    
    def compute_summary(self):
        """Compute summary statistics."""
        linear = [r for r in self.results if r.diagnosis == "LINEAR"]
        nonlinear = [r for r in self.results if r.diagnosis == "NONLINEAR"]
        no_signal = [r for r in self.results if r.diagnosis == "NO_SIGNAL"]
        
        if linear:
            self.linear_success_rate = sum(r.steering_success for r in linear) / len(linear)
        if nonlinear:
            self.nonlinear_success_rate = sum(r.steering_success for r in nonlinear) / len(nonlinear)
        if no_signal:
            self.no_signal_success_rate = sum(r.steering_success for r in no_signal) / len(no_signal)


def compute_caa_direction(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> torch.Tensor:
    """
    Compute CAA (Contrastive Activation Addition) direction.
    
    This is the difference-in-means direction used for steering.
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        
    Returns:
        [hidden_dim] steering direction (normalized)
    """
    pos_mean = pos_activations.float().mean(dim=0)
    neg_mean = neg_activations.float().mean(dim=0)
    direction = pos_mean - neg_mean
    return direction / (direction.norm() + ZERO_THRESHOLD)


def apply_steering_to_model(
    model: "WisentModel",
    layer: int,
    direction: torch.Tensor,
    coefficient: float,
) -> None:
    """
    Apply steering to model using WisentModel's built-in steering.
    
    Args:
        model: WisentModel instance
        layer: Layer index to apply steering (0-based)
        direction: [hidden_dim] steering direction
        coefficient: Steering strength
    """
    from wisent.core.models.core.atoms import SteeringPlan
    
    # Create steering vector dict: layer_name -> tensor
    # WisentModel uses 1-based layer names
    layer_name = str(layer + 1)
    steering_dict = {layer_name: direction * coefficient}
    
    # Create and apply steering plan
    plan = SteeringPlan.from_raw(steering_dict, scale=1.0)
    model.apply_steering(plan)


def load_diagnosis_results(model_name: str, output_dir: Path) -> Dict[str, Any]:
    """Load Zwiad diagnosis results from GCS/local."""
    model_prefix = model_name.replace('/', '_')

    # Try to download from GCS first
    try:
        subprocess.run(
            ["gcloud", "storage", "rsync",
             f"gs://{GCS_BUCKET}/direction_discovery/{model_prefix}/",
             str(output_dir / "diagnosis"),
             "--quiet"],
            check=False,
            capture_output=True,
        )
    except Exception:
        pass
    
    # Load results
    results = {}
    diagnosis_dir = output_dir / "diagnosis"
    if diagnosis_dir.exists():
        for f in diagnosis_dir.glob(f"{model_prefix}_*.json"):
            if "summary" not in f.name:
                category = f.stem.replace(f"{model_prefix}_", "")
                with open(f) as fp:
                    results[category] = json.load(fp)
    
    return results


def get_diagnosis_for_benchmark(
    diagnosis_results: Dict[str, Any],
    benchmark: str,
    strategy: str = "chat_last",
) -> Tuple[str, int, float, float]:
    """
    Get Zwiad diagnosis for a specific benchmark.
    
    Args:
        diagnosis_results: Loaded diagnosis results
        benchmark: Benchmark name
        strategy: Extraction strategy
        
    Returns:
        (diagnosis, best_layer, signal_strength, linear_probe_accuracy)
    """
    for category, data in diagnosis_results.items():
        results = data.get("results", [])
        for r in results:
            if r["benchmark"] == benchmark and r["strategy"] == strategy:
                signal = r["signal_strength"]
                linear = r["linear_probe_accuracy"]
                layers = r["layers"]
                num_layers = len(layers) if layers else 36
                best_layer = num_layers // 2
                
                # Determine diagnosis
                if signal < 0.6:
                    diagnosis = "NO_SIGNAL"
                elif linear > 0.6 and (signal - linear) < 0.15:
                    diagnosis = "LINEAR"
                else:
                    diagnosis = "NONLINEAR"
                
                return diagnosis, best_layer, signal, linear
    
    return "UNKNOWN", 20, 0.5, 0.5


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

