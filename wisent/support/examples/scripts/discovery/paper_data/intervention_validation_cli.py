"""Intervention Validation using Wisent CLI."""

import argparse
import json
import subprocess
from pathlib import Path
from dataclasses import asdict
from typing import List

from wisent.core.utils.config_tools.constants import DEFAULT_RANDOM_SEED, SEPARATOR_WIDTH_WIDE, SEPARATOR_WIDTH_MEDIUM, JSON_INDENT
from wisent.examples.scripts.intervention_validation_cli_helpers import (
    GCS_BUCKET,
    GCS_PREFIX,
    BenchmarkResult,
    run_wisent_task,
    load_diagnosis_from_gcs,
    get_benchmarks_by_diagnosis,
)


def run_intervention_validation(
    model_name: str,
    num_per_category: int = 3,
    steering_strengths: List[float] = [1.0, 2.0, 5.0],
):
    """Run intervention validation using wisent CLI."""
    
    print("=" * SEPARATOR_WIDTH_WIDE)
    print("INTERVENTION VALIDATION (using wisent CLI)")
    print("=" * SEPARATOR_WIDTH_WIDE)
    print(f"Model: {model_name}")
    
    # Load diagnosis
    diagnosis_results = load_diagnosis_from_gcs(model_name)
    if not diagnosis_results:
        print("ERROR: No diagnosis results found")
        return
    
    by_diagnosis = get_benchmarks_by_diagnosis(diagnosis_results)
    
    print(f"\nBenchmarks by diagnosis:")
    for diag, benches in by_diagnosis.items():
        print(f"  {diag}: {len(benches)} benchmarks")
    
    # Select benchmarks to test
    import random
    random.seed(DEFAULT_RANDOM_SEED)
    
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
        print(f"\n{'-' * SEPARATOR_WIDTH_MEDIUM}")
        print(f"Benchmark: {bench} ({diag})")
        print("-" * SEPARATOR_WIDTH_MEDIUM)
        
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
    print("\n" + "=" * SEPARATOR_WIDTH_WIDE)
    print("VALIDATION SUMMARY")
    print("=" * SEPARATOR_WIDTH_WIDE)
    
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
        }, f, indent=JSON_INDENT)
    
    print(f"\nResults saved to: {output_file}")
    
    # Upload to GCS
    try:
        gcs_path = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/{model_prefix}/{output_file.name}"
        subprocess.run(["gcloud", "storage", "cp", str(output_file), gcs_path, "--quiet"], check=False)
        print(f"Uploaded to: {gcs_path}")
    except:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intervention validation using wisent CLI")
    parser.add_argument("--model", type=str, required=True, help="Model to test")
    parser.add_argument("--num-per-category", type=int, required=True, help="Benchmarks per diagnosis category")
    args = parser.parse_args()
    
    run_intervention_validation(
        model_name=args.model,
        num_per_category=args.num_per_category,
    )
