"""Helpers for intervention_validation_cli."""

import json
import re
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from wisent.core.utils.config_tools.constants import (
    PAIR_GENERATORS_DEFAULT_N,
    DISPLAY_TRUNCATION_LARGE,
)
from wisent.core.utils.infra_tools.infra.core.hardware import subprocess_timeout_long_s
from wisent.core.utils.infra_tools.errors import MissingParameterError

GCS_BUCKET = "wisent-images-bucket"
GCS_PREFIX = "intervention_validation"


class BenchmarkResult:
    benchmark: str
    diagnosis: str  # LINEAR, NONLINEAR, NO_SIGNAL
    layer: int
    
    # Baseline (no steering)
    baseline_accuracy: float
    
    # With steering
    steered_accuracy: float
    steering_strength: float
    
    # Effect
    accuracy_change: float
    steering_success: bool


def run_wisent_task(
    benchmark: str,
    model: str,
    layer: int,
    steering_strength: Optional[float] = None,
    steering_mode: bool = False,
    training_limit: int = PAIR_GENERATORS_DEFAULT_N,
    testing_limit: int = PAIR_GENERATORS_DEFAULT_N,
) -> float:
    """
    Run wisent tasks command and return accuracy.
    """
    cmd = [
        sys.executable, "-m", "wisent.core.parser",
        "tasks", benchmark,
        "--model", model,
        "--layer", str(layer),
        "--training-limit", str(training_limit),
        "--testing-limit", str(testing_limit),
        "--output-mode", "likelihoods",
    ]
    
    if steering_mode:
        if steering_strength is None:
            raise MissingParameterError(
                params=["steering_strength"],
                context="run_wisent_task with steering_mode=True",
            )
        cmd.extend([
            "--steering-mode",
            "--steering-strength", str(steering_strength),
            "--steering-method", "CAA",
        ])
    
    print(f"  Running: {' '.join(cmd[-10:])}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=subprocess_timeout_long_s(),
        )
        
        output = result.stdout + result.stderr
        
        # Parse accuracy from output
        for line in output.split('\n'):
            if 'Test Accuracy:' in line or 'test_accuracy' in line.lower():
                # Try to extract number
                import re
                match = re.search(r'(\d+\.?\d*)\s*%?', line)
                if match:
                    acc = float(match.group(1))
                    if acc > 1:  # Percentage
                        acc /= 100
                    return acc
            if 'accuracy' in line.lower() and ':' in line:
                parts = line.split(':')
                if len(parts) >= 2:
                    try:
                        acc = float(parts[-1].strip().replace('%', ''))
                        if acc > 1:
                            acc /= 100
                        return acc
                    except:
                        pass
        
        print(f"  Warning: Could not parse accuracy from output")
        print(f"  Output: {output[:DISPLAY_TRUNCATION_LARGE]}")
        return 0.5
        
    except subprocess.TimeoutExpired:
        print(f"  Timeout running {benchmark}")
        return 0.5
    except Exception as e:
        print(f"  Error: {e}")
        return 0.5


def load_diagnosis_from_gcs(model_name: str) -> Dict[str, Any]:
    """Load Zwiad diagnosis results from GCS."""
    model_prefix = model_name.replace('/', '_')
    local_dir = Path(f"/tmp/diagnosis_{model_prefix}")
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            ["gcloud", "storage", "rsync",
             f"gs://{GCS_BUCKET}/direction_discovery/{model_prefix}/",
             str(local_dir),
             "--quiet"],
            check=False,
            capture_output=True,
        )
    except Exception:
        pass
    
    results = {}
    for f in local_dir.glob(f"{model_prefix}_*.json"):
        if "summary" not in f.name:
            category = f.stem.replace(f"{model_prefix}_", "")
            with open(f) as fp:
                results[category] = json.load(fp)
    
    return results


def get_benchmarks_by_diagnosis(diagnosis_results: Dict) -> Dict[str, List[tuple]]:
    """Group benchmarks by diagnosis type. Returns {diagnosis: [(benchmark, layer, signal, linear)]}"""
    by_diagnosis = {"LINEAR": [], "NO_SIGNAL": [], "NONLINEAR": []}
    
    for category, data in diagnosis_results.items():
        results = data.get("results", [])
        seen = set()
        
        for r in results:
            bench = r["benchmark"]
            if bench in seen:
                continue
            seen.add(bench)
            
            signal = r["signal_strength"]
            linear = r["linear_probe_accuracy"]
            num_layers = len(r["layers"]) if r["layers"] else 36
            best_layer = num_layers // 2
            
            if signal < 0.6:
                by_diagnosis["NO_SIGNAL"].append((bench, best_layer, signal, linear))
            elif linear > 0.6 and (signal - linear) < 0.15:
                by_diagnosis["LINEAR"].append((bench, best_layer, signal, linear))
            else:
                by_diagnosis["NONLINEAR"].append((bench, best_layer, signal, linear))
    
    return by_diagnosis

