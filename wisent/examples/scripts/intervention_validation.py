"""
Intervention Validation for RepScan.

Tests whether RepScan diagnosis predicts CAA steering success:
- LINEAR diagnosis -> CAA should work
- NONLINEAR diagnosis -> CAA should fail (but detection still works)
- NO_SIGNAL diagnosis -> neither should work

This is the CRITICAL missing piece identified by reviewers.

Usage:
    python -m wisent.examples.scripts.intervention_validation --model Qwen/Qwen3-8B
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import random

import torch
import numpy as np

S3_BUCKET = "wisent-bucket"
S3_PREFIX = "intervention_validation"


def s3_upload_file(local_path: Path, model_name: str) -> None:
    """Upload a single file to S3."""
    model_prefix = model_name.replace('/', '_')
    s3_path = f"s3://{S3_BUCKET}/{S3_PREFIX}/{model_prefix}/{local_path.name}"
    try:
        subprocess.run(
            ["aws", "s3", "cp", str(local_path), s3_path, "--quiet"],
            check=True,
            capture_output=True,
        )
        print(f"  Uploaded to S3: {s3_path}")
    except Exception as e:
        print(f"  S3 upload failed: {e}")


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
    return direction / (direction.norm() + 1e-10)


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


def get_model_logprobs(
    model: "WisentModel",
    prompt: str,
    completion: str,
) -> float:
    """
    Get log probability of completion given prompt.
    
    Args:
        model: WisentModel instance
        prompt: Input prompt
        completion: Completion to score
        
    Returns:
        Average log probability of completion tokens
    """
    full_text = prompt + completion
    
    inputs = model.tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)
    
    prompt_tokens = model.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.hf_model(**inputs)
        logits = outputs.logits
    
    # Get logprobs for completion tokens only
    shift_logits = logits[:, prompt_tokens-1:-1, :].contiguous()
    shift_labels = inputs.input_ids[:, prompt_tokens:].contiguous()
    
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    return token_log_probs.mean().item()


def evaluate_steering(
    model: "WisentModel",
    test_pairs: List,
    layer: int,
    direction: torch.Tensor,
    coefficient: float,
) -> Tuple[float, float, float]:
    """
    Evaluate steering effect on test pairs using WisentModel's built-in steering.
    
    Args:
        model: WisentModel instance
        test_pairs: List of ContrastivePair objects
        layer: Layer to apply steering
        direction: Steering direction
        coefficient: Steering strength
        
    Returns:
        (accuracy, avg_correct_logprob, avg_incorrect_logprob)
    """
    # Apply steering using WisentModel's built-in method
    apply_steering_to_model(model, layer, direction, coefficient)
    
    try:
        correct = 0
        correct_logprobs = []
        incorrect_logprobs = []
        
        for pair in test_pairs:
            prompt = pair.prompt
            correct_completion = pair.positive_response.model_response
            incorrect_completion = pair.negative_response.model_response
            
            correct_lp = get_model_logprobs(model, prompt, correct_completion)
            incorrect_lp = get_model_logprobs(model, prompt, incorrect_completion)
            
            correct_logprobs.append(correct_lp)
            incorrect_logprobs.append(incorrect_lp)
            
            if correct_lp > incorrect_lp:
                correct += 1
        
        accuracy = correct / len(test_pairs) if test_pairs else 0.0
        avg_correct = np.mean(correct_logprobs) if correct_logprobs else 0.0
        avg_incorrect = np.mean(incorrect_logprobs) if incorrect_logprobs else 0.0
        
        return accuracy, avg_correct, avg_incorrect
    
    finally:
        # Remove steering
        model.detach()


def evaluate_baseline(
    model: "WisentModel",
    test_pairs: List,
) -> Tuple[float, float, float]:
    """
    Evaluate baseline (no steering) on test pairs.
    
    Args:
        model: WisentModel instance
        test_pairs: List of ContrastivePair objects
        
    Returns:
        (accuracy, avg_correct_logprob, avg_incorrect_logprob)
    """
    correct = 0
    correct_logprobs = []
    incorrect_logprobs = []
    
    for pair in test_pairs:
        prompt = pair.prompt
        correct_completion = pair.positive_response.model_response
        incorrect_completion = pair.negative_response.model_response
        
        correct_lp = get_model_logprobs(model, prompt, correct_completion)
        incorrect_lp = get_model_logprobs(model, prompt, incorrect_completion)
        
        correct_logprobs.append(correct_lp)
        incorrect_logprobs.append(incorrect_lp)
        
        if correct_lp > incorrect_lp:
            correct += 1
    
    accuracy = correct / len(test_pairs) if test_pairs else 0.0
    avg_correct = np.mean(correct_logprobs) if correct_logprobs else 0.0
    avg_incorrect = np.mean(incorrect_logprobs) if incorrect_logprobs else 0.0
    
    return accuracy, avg_correct, avg_incorrect


def load_diagnosis_results(model_name: str, output_dir: Path) -> Dict[str, Any]:
    """Load RepScan diagnosis results from S3/local."""
    model_prefix = model_name.replace('/', '_')
    
    # Try to download from S3 first
    try:
        subprocess.run(
            ["aws", "s3", "sync", 
             f"s3://{S3_BUCKET}/direction_discovery/{model_prefix}/",
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
    Get RepScan diagnosis for a specific benchmark.
    
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
                # Use ~60% through the network (layer 20-22 for 36 layers)
                # This is typically where semantic representations are strongest
                num_layers = len(layers) if layers else 36
                best_layer = int(num_layers * 0.6)
                
                # Determine diagnosis
                if signal < 0.6:
                    diagnosis = "NO_SIGNAL"
                elif linear > 0.6 and (signal - linear) < 0.15:
                    diagnosis = "LINEAR"
                else:
                    diagnosis = "NONLINEAR"
                
                return diagnosis, best_layer, signal, linear
    
    return "UNKNOWN", 20, 0.5, 0.5


def run_intervention_validation(
    model_name: str,
    benchmarks_to_test: Optional[List[str]] = None,
    samples_per_benchmark: int = 20,
    test_samples: int = 30,
    steering_coefficients: List[float] = [1.0, 2.0, 5.0, 10.0],
):
    """
    Run intervention validation experiments.
    
    Args:
        model_name: Model to test
        benchmarks_to_test: Specific benchmarks (default: sample from each diagnosis)
        samples_per_benchmark: Pairs for computing steering direction
        test_samples: Pairs for evaluating steering
        steering_coefficients: Coefficients to test
    """
    from wisent.core.models.wisent_model import WisentModel
    from wisent.core.activations.extraction_strategy import ExtractionStrategy
    from wisent.core.activations.activation_cache import ActivationCache, collect_and_cache_activations
    from lm_eval.tasks import TaskManager
    from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import lm_build_contrastive_pairs
    
    print("=" * 70)
    print("INTERVENTION VALIDATION")
    print("=" * 70)
    print(f"Model: {model_name}")
    
    output_dir = Path("/tmp/intervention_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load diagnosis results
    diagnosis_results = load_diagnosis_results(model_name, output_dir)
    if not diagnosis_results:
        print("ERROR: No diagnosis results found. Run discover_directions first.")
        return
    
    # Select benchmarks to test (sample from each diagnosis type)
    if benchmarks_to_test is None:
        benchmarks_to_test = []
        
        # Collect all benchmarks with their diagnoses
        by_diagnosis = {"LINEAR": [], "NONLINEAR": [], "NO_SIGNAL": []}
        
        for category, data in diagnosis_results.items():
            results = data.get("results", [])
            seen_benchmarks = set()
            for r in results:
                bench = r["benchmark"]
                if bench in seen_benchmarks:
                    continue
                seen_benchmarks.add(bench)
                
                signal = r["signal_strength"]
                linear = r["linear_probe_accuracy"]
                
                if signal < 0.6:
                    by_diagnosis["NO_SIGNAL"].append(bench)
                elif linear > 0.6 and (signal - linear) < 0.15:
                    by_diagnosis["LINEAR"].append(bench)
                else:
                    by_diagnosis["NONLINEAR"].append(bench)
        
        # Sample 3 from each category
        random.seed(42)
        for diag, benches in by_diagnosis.items():
            if benches:
                sampled = random.sample(benches, min(3, len(benches)))
                benchmarks_to_test.extend(sampled)
                print(f"  {diag}: {sampled}")
    
    print(f"\nBenchmarks to test: {benchmarks_to_test}")
    
    # Load model
    print(f"\nLoading model: {model_name}")
    model = WisentModel(model_name, device="cuda")
    print(f"  Layers: {model.num_layers}, Hidden: {model.hidden_size}")
    
    # Cache directory
    model_prefix = model_name.replace('/', '_')
    cache_dir = f"/tmp/wisent_intervention_cache_{model_prefix}"
    cache = ActivationCache(cache_dir)
    
    # Results
    validation_results = ValidationResults(model=model_name)
    
    tm = TaskManager()
    strategy = ExtractionStrategy.CHAT_LAST
    
    for benchmark in benchmarks_to_test:
        print(f"\n{'-' * 50}")
        print(f"Benchmark: {benchmark}")
        print("-" * 50)
        
        # Get diagnosis
        diagnosis, best_layer, signal, linear_acc = get_diagnosis_for_benchmark(
            diagnosis_results, benchmark, strategy.value
        )
        print(f"  Diagnosis: {diagnosis}")
        print(f"  Signal: {signal:.3f}, Linear: {linear_acc:.3f}")
        print(f"  Best layer: {best_layer}")
        
        # Load pairs
        try:
            task_dict = tm.load_task_or_group([benchmark])
            task = list(task_dict.values())[0]
        except Exception:
            task = None
        
        try:
            all_pairs = lm_build_contrastive_pairs(
                benchmark, 
                task, 
                limit=samples_per_benchmark + test_samples
            )
        except Exception as e:
            print(f"  ERROR loading pairs: {e}")
            continue
        
        if len(all_pairs) < samples_per_benchmark + test_samples:
            print(f"  SKIP: Not enough pairs ({len(all_pairs)})")
            continue
        
        # Split into train (for direction) and test (for evaluation)
        random.shuffle(all_pairs)
        train_pairs = all_pairs[:samples_per_benchmark]
        test_pairs = all_pairs[samples_per_benchmark:samples_per_benchmark + test_samples]
        
        print(f"  Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")
        
        # Get activations for training pairs
        print(f"  Extracting activations...")
        try:
            cached = collect_and_cache_activations(
                model=model,
                pairs=train_pairs,
                benchmark=benchmark,
                strategy=strategy,
                cache=cache,
                show_progress=False,
            )
        except Exception as e:
            print(f"  ERROR extracting activations: {e}")
            continue
        
        # Get activations at best layer
        layer_name = str(best_layer + 1)  # 1-based
        try:
            pos_acts = cached.get_positive_activations(layer_name)
            neg_acts = cached.get_negative_activations(layer_name)
        except Exception as e:
            print(f"  ERROR getting activations: {e}")
            continue
        
        # Compute CAA direction
        direction = compute_caa_direction(pos_acts, neg_acts)
        print(f"  Direction norm: {direction.norm().item():.4f}")
        
        # Evaluate baseline
        print(f"  Evaluating baseline...")
        base_acc, base_correct_lp, base_incorrect_lp = evaluate_baseline(model, test_pairs)
        print(f"    Baseline accuracy: {base_acc:.3f}")
        print(f"    Baseline logprob gap: {base_correct_lp - base_incorrect_lp:.4f}")
        
        # Test steering at different coefficients
        best_result = None
        best_improvement = -float('inf')
        
        for coef in steering_coefficients:
            print(f"  Testing coefficient={coef}...")
            steered_acc, steered_correct_lp, steered_incorrect_lp = evaluate_steering(
                model, test_pairs, best_layer, direction, coef
            )
            
            acc_change = steered_acc - base_acc
            lp_shift = (steered_correct_lp - steered_incorrect_lp) - (base_correct_lp - base_incorrect_lp)
            
            print(f"    Steered accuracy: {steered_acc:.3f} (change: {acc_change:+.3f})")
            print(f"    Logprob shift: {lp_shift:+.4f}")
            
            # Steering is successful if it improves accuracy OR logprob gap
            steering_success = acc_change > 0.05 or lp_shift > 0.1
            
            if acc_change > best_improvement:
                best_improvement = acc_change
                best_result = SteeringResult(
                    benchmark=benchmark,
                    strategy=strategy.value,
                    layer=best_layer,
                    diagnosis=diagnosis,
                    baseline_accuracy=base_acc,
                    baseline_correct_logprob=base_correct_lp,
                    baseline_incorrect_logprob=base_incorrect_lp,
                    steered_accuracy=steered_acc,
                    steered_correct_logprob=steered_correct_lp,
                    steered_incorrect_logprob=steered_incorrect_lp,
                    accuracy_change=acc_change,
                    logprob_shift=lp_shift,
                    steering_success=steering_success,
                    steering_coefficient=coef,
                    num_test_samples=len(test_pairs),
                )
        
        if best_result:
            validation_results.results.append(best_result)
            print(f"\n  Best result: coef={best_result.steering_coefficient}, "
                  f"acc_change={best_result.accuracy_change:+.3f}, "
                  f"success={best_result.steering_success}")
    
    # Compute summary
    validation_results.compute_summary()
    
    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\nLinear diagnosis -> CAA success rate: {validation_results.linear_success_rate:.1%}")
    print(f"Nonlinear diagnosis -> CAA success rate: {validation_results.nonlinear_success_rate:.1%}")
    print(f"No signal diagnosis -> CAA success rate: {validation_results.no_signal_success_rate:.1%}")
    
    # Expected pattern:
    # LINEAR -> high success rate
    # NONLINEAR -> low success rate (CAA doesn't work, but detection does)
    # NO_SIGNAL -> low success rate
    
    if validation_results.linear_success_rate > validation_results.nonlinear_success_rate:
        print("\n✓ VALIDATION PASSED: LINEAR diagnosis predicts higher CAA success!")
    else:
        print("\n✗ VALIDATION FAILED: LINEAR diagnosis does not predict higher CAA success")
    
    # Save results
    results_file = output_dir / f"{model_prefix}_validation.json"
    
    # Convert results to JSON-serializable format
    def convert_to_serializable(obj):
        """Convert numpy types to Python native types."""
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
    
    results_dicts = [convert_to_serializable(asdict(r)) for r in validation_results.results]
    
    with open(results_file, "w") as f:
        json.dump({
            "model": model_name,
            "results": results_dicts,
            "summary": {
                "linear_success_rate": float(validation_results.linear_success_rate),
                "nonlinear_success_rate": float(validation_results.nonlinear_success_rate),
                "no_signal_success_rate": float(validation_results.no_signal_success_rate),
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    s3_upload_file(results_file, model_name)
    
    # Cleanup
    del model
    
    return validation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intervention validation for RepScan")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model to test")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=None, help="Specific benchmarks to test")
    parser.add_argument("--samples", type=int, default=20, help="Samples for direction computation")
    parser.add_argument("--test-samples", type=int, default=30, help="Samples for evaluation")
    args = parser.parse_args()
    
    run_intervention_validation(
        model_name=args.model,
        benchmarks_to_test=args.benchmarks,
        samples_per_benchmark=args.samples,
        test_samples=args.test_samples,
    )
