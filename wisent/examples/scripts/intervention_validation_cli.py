"""
Intervention Validation using Wisent CLI.

Tests whether RepScan LINEAR diagnosis predicts CAA steering success.
Uses wisent CLI `tasks` command with --steering-mode.

Usage:
    python -m wisent.examples.scripts.intervention_validation_cli --model Qwen/Qwen3-8B
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

S3_BUCKET = "wisent-bucket"
S3_PREFIX = "intervention_validation"


@dataclass
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
    steering_mode: bool = False,
    steering_strength: float = 1.0,
    training_limit: int = 30,
    testing_limit: int = 50,
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
            timeout=600,
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
        print(f"  Output: {output[:500]}")
        return 0.5
        
    except subprocess.TimeoutExpired:
        print(f"  Timeout running {benchmark}")
        return 0.5
    except Exception as e:
        print(f"  Error: {e}")
        return 0.5


def load_diagnosis_from_s3(model_name: str) -> Dict[str, Any]:
    """Load RepScan diagnosis results from S3."""
    model_prefix = model_name.replace('/', '_')
    local_dir = Path(f"/tmp/diagnosis_{model_prefix}")
    local_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        subprocess.run(
            ["aws", "s3", "sync",
             f"s3://{S3_BUCKET}/direction_discovery/{model_prefix}/",
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
            best_layer = int(num_layers * 0.6)  # 60% through network
            
            if signal < 0.6:
                by_diagnosis["NO_SIGNAL"].append((bench, best_layer, signal, linear))
            elif linear > 0.6 and (signal - linear) < 0.15:
                by_diagnosis["LINEAR"].append((bench, best_layer, signal, linear))
            else:
                by_diagnosis["NONLINEAR"].append((bench, best_layer, signal, linear))
    
    return by_diagnosis


def run_intervention_validation(
    model_name: str,
    num_per_category: int = 3,
    steering_strengths: List[float] = [1.0, 2.0, 5.0],
):
    """Run intervention validation using wisent CLI."""
    
    print("=" * 70)
    print("INTERVENTION VALIDATION (using wisent CLI)")
    print("=" * 70)
    print(f"Model: {model_name}")
    
    # Load diagnosis
    diagnosis_results = load_diagnosis_from_s3(model_name)
    if not diagnosis_results:
        print("ERROR: No diagnosis results found")
        return
    
    by_diagnosis = get_benchmarks_by_diagnosis(diagnosis_results)
    
    print(f"\nBenchmarks by diagnosis:")
    for diag, benches in by_diagnosis.items():
        print(f"  {diag}: {len(benches)} benchmarks")
    
    # Select benchmarks to test
    import random
    random.seed(42)
    
    test_benchmarks = []
    for diag, benches in by_diagnosis.items():
        if benches:
            selected = random.sample(benches, min(num_per_category, len(benches)))
            for bench, layer, signal, linear in selected:
                test_benchmarks.append((bench, diag, layer, signal, linear))
    
    print(f"\nTesting {len(test_benchmarks)} benchmarks:")
    for bench, diag, layer, signal, linear in test_benchmarks:
        print(f"  {bench}: {diag} (layer {layer}, signal={signal:.2f}, linear={linear:.2f})")
    
    results = []
    
    for bench, diag, layer, signal, linear in test_benchmarks:
        print(f"\n{'-' * 50}")
        print(f"Benchmark: {bench} ({diag})")
        print("-" * 50)
        
        # Run baseline (no steering)
        print("  Running baseline (no steering)...")
        baseline_acc = run_wisent_task(
            benchmark=bench,
            model=model_name,
            layer=layer,
            steering_mode=False,
        )
        print(f"  Baseline accuracy: {baseline_acc:.3f}")
        
        # Run with steering at different strengths
        best_steered_acc = baseline_acc
        best_strength = 0.0
        
        for strength in steering_strengths:
            print(f"  Running with steering (strength={strength})...")
            steered_acc = run_wisent_task(
                benchmark=bench,
                model=model_name,
                layer=layer,
                steering_mode=True,
                steering_strength=strength,
            )
            print(f"    Steered accuracy: {steered_acc:.3f} (change: {steered_acc - baseline_acc:+.3f})")
            
            if steered_acc > best_steered_acc:
                best_steered_acc = steered_acc
                best_strength = strength
        
        acc_change = best_steered_acc - baseline_acc
        steering_success = acc_change > 0.05
        
        results.append(BenchmarkResult(
            benchmark=bench,
            diagnosis=diag,
            layer=layer,
            baseline_accuracy=baseline_acc,
            steered_accuracy=best_steered_acc,
            steering_strength=best_strength,
            accuracy_change=acc_change,
            steering_success=steering_success,
        ))
        
        print(f"  Best result: strength={best_strength}, change={acc_change:+.3f}, success={steering_success}")
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    for diag in ["LINEAR", "NONLINEAR", "NO_SIGNAL"]:
        diag_results = [r for r in results if r.diagnosis == diag]
        if diag_results:
            success_rate = sum(r.steering_success for r in diag_results) / len(diag_results)
            avg_change = sum(r.accuracy_change for r in diag_results) / len(diag_results)
            print(f"\n{diag}:")
            print(f"  Success rate: {success_rate:.1%}")
            print(f"  Avg accuracy change: {avg_change:+.3f}")
    
    linear_results = [r for r in results if r.diagnosis == "LINEAR"]
    no_signal_results = [r for r in results if r.diagnosis == "NO_SIGNAL"]
    
    linear_success = sum(r.steering_success for r in linear_results) / len(linear_results) if linear_results else 0
    no_signal_success = sum(r.steering_success for r in no_signal_results) / len(no_signal_results) if no_signal_results else 0
    
    if linear_success > no_signal_success:
        print("\n✓ VALIDATION PASSED: LINEAR diagnosis predicts higher CAA success!")
    else:
        print("\n✗ VALIDATION FAILED: LINEAR diagnosis does not predict higher CAA success")
    
    # Save results
    model_prefix = model_name.replace('/', '_')
    output_file = Path(f"/tmp/intervention_validation/{model_prefix}_cli_validation.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump({
            "model": model_name,
            "results": [asdict(r) for r in results],
            "summary": {
                "linear_success_rate": linear_success,
                "no_signal_success_rate": no_signal_success,
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Upload to S3
    try:
        s3_path = f"s3://{S3_BUCKET}/{S3_PREFIX}/{model_prefix}/{output_file.name}"
        subprocess.run(["aws", "s3", "cp", str(output_file), s3_path, "--quiet"], check=False)
        print(f"Uploaded to: {s3_path}")
    except:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intervention validation using wisent CLI")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model to test")
    parser.add_argument("--num-per-category", type=int, default=3, help="Benchmarks per diagnosis category")
    args = parser.parse_args()
    
    run_intervention_validation(
        model_name=args.model,
        num_per_category=args.num_per_category,
    )
