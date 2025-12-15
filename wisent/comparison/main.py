"""
Comparison of steering methods: Ours vs SAE-based.

This script:
1. Creates steering vectors using train split of pooled data
2. Runs base evaluation on test split (no overlap)
3. Runs steered evaluation on same test split
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import get_task_dict
from lm_eval.models.hf_steered import SteeredModel

from wisent.core.models.wisent_model import WisentModel
from wisent.core.utils.dataset_splits import get_test_docs
from wisent.comparison import ours
from wisent.comparison import sae

# Map method names to modules
METHOD_MODULES = {
    "caa": ours,
    "sae": sae,
}


def create_test_only_task(task_name: str, train_ratio: float = 0.8) -> dict:
    """
    Create a task that evaluates only on our test split.

    This ensures no overlap with the data used for steering vector training.
    """
    task_dict = get_task_dict([task_name])
    task = task_dict[task_name]

    # Get our test split
    test_docs = get_test_docs(task, benchmark_name=task_name, train_ratio=train_ratio)
    test_pct = round((1 - train_ratio) * 100)

    print(f"Test split size: {len(test_docs)} docs ({test_pct}% of pooled data)")

    # Override task's doc methods to use our test split
    # Return list (not iterator) so len() works
    task.test_docs = lambda: test_docs
    task.has_test_docs = lambda: True
    # Also override eval_docs property to return our test docs directly
    task._eval_docs = test_docs

    return {task_name: task}


def run_evaluation(
    model_name: str = None,
    wisent_model: WisentModel = None,
    task_dict: dict = None,
    task_name: str = None,
    device: str = "cuda:0",
    batch_size: int = 1,
    max_batch_size: int = 8,
    limit: int | None = None,
) -> dict:
    """
    Run evaluation using lm-eval-harness on our test split.

    Either provide model_name (loads fresh model) or wisent_model (uses existing).
    """
    if wisent_model is not None:
        # Use pre-loaded model (with steering hooks if applied)
        lm = HFLM(
            pretrained=wisent_model.hf_model,
            tokenizer=wisent_model.tokenizer,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
        )
    else:
        # Load fresh model
        lm = HFLM(
            pretrained=model_name,
            device=device,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
        )

    # Use lower-level evaluate() which accepts task_dict directly
    results = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
    )

    # Clean up model to free GPU memory
    if hasattr(lm, '_model'):
        del lm._model
    if hasattr(lm, 'model'):
        del lm.model
    del lm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return results


def extract_accuracy(results: dict, task: str) -> float:
    """Extract accuracy from lm-eval results."""
    task_results = results.get("results", {}).get(task, {})
    for key in ["acc", "acc,none", "accuracy", "acc_norm", "acc_norm,none"]:
        if key in task_results:
            return task_results[key]
    return 0.0


def run_single_task(
    model_name: str,
    task: str,
    methods: list[str] = None,
    num_pairs: int = 50,
    steering_scales: list[float] = None,
    device: str = "cuda:0",
    batch_size: int = 1,
    max_batch_size: int = 8,
    eval_limit: int | None = None,
    vectors_dir: Path = None,
    train_ratio: float = 0.8,
    layers: str | None = None,
) -> list[dict]:
    """
    Run comparison for a single task with multiple methods and scales.

    Returns list of result dicts, one per method/scale combination.
    """
    if methods is None:
        methods = ["caa"]
    if steering_scales is None:
        steering_scales = [1.0]

    results_list = []

    # Step 1: Create test task
    test_pct = round((1 - train_ratio) * 100)
    print(f"\n{'='*60}")
    print(f"Creating test task for: {task}")
    print(f"(using {test_pct}% of pooled data)")
    print(f"{'='*60}")

    task_dict = create_test_only_task(task, train_ratio=train_ratio)

    # Load model once and reuse
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"{'='*60}")
    wisent_model = WisentModel(model_name=model_name, device=device)

    # Step 2: Run base evaluation (no steering applied)
    print(f"\n{'='*60}")
    print(f"Running BASE evaluation for: {task}")
    print(f"{'='*60}")

    base_results = run_evaluation(
        wisent_model=wisent_model,
        task_dict=task_dict,
        task_name=task,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        limit=eval_limit,
    )
    base_acc = extract_accuracy(base_results, task)
    print(f"Base accuracy: {base_acc:.4f}")

    # Step 3: Generate steering vectors and evaluate for each method
    for method in methods:
        if method not in METHOD_MODULES:
            print(f"WARNING: Method '{method}' not implemented, skipping")
            continue

        method_module = METHOD_MODULES[method]

        train_pct = round(train_ratio * 100)
        print(f"\n{'='*60}")
        print(f"Generating steering vector for: {task} (method={method})")
        print(f"(using {train_pct}% of pooled data - no overlap with test)")
        if layers:
            print(f"Layers: {layers}")
        print(f"{'='*60}")

        vector_path = vectors_dir / f"{task}_{method}_steering_vector.json"
        method_module.generate_steering_vector(
            task=task,
            model_name=model_name,
            output_path=vector_path,
            num_pairs=num_pairs,
            device=device,
            layers=layers,
        )

        steering_data = method_module.load_steering_vector(vector_path)
        print(f"Loaded steering vector with layers: {steering_data['layers']}")

        # Step 4: Run steered evaluations for each scale
        for scale in steering_scales:
            print(f"\n{'='*60}")
            print(f"Running STEERED evaluation for: {task} (method={method}, scale={scale})")
            print(f"{'='*60}")

            # Apply steering to existing model
            method_module.apply_steering_to_model(wisent_model, steering_data, scale=scale)

            steered_results = run_evaluation(
                wisent_model=wisent_model,
                task_dict=task_dict,
                task_name=task,
                batch_size=batch_size,
                max_batch_size=max_batch_size,
                limit=eval_limit,
            )
            steered_acc = extract_accuracy(steered_results, task)
            print(f"Steered accuracy (wisent): {steered_acc:.4f}")

            # Remove steering for next iteration
            method_module.remove_steering(wisent_model)

            # Step 5: Run lm-eval native steered evaluation
            print(f"\n{'='*60}")
            print(f"Running lm-eval NATIVE steered for: {task} (method={method}, scale={scale})")
            print(f"{'='*60}")

            # Convert steering vector to lm-eval format
            lm_eval_steer_path = vectors_dir / f"{task}_{method}_lm_eval_steer_scale{scale}.pt"
            method_module.convert_to_lm_eval_format(steering_data, lm_eval_steer_path, scale=scale)

            lm_steered = SteeredModel(
                pretrained=model_name,
                steer_path=str(lm_eval_steer_path),
                device=device,
                batch_size=batch_size,
                max_batch_size=max_batch_size,
            )

            lm_eval_native_results = evaluator.evaluate(
                lm=lm_steered,
                task_dict=task_dict,
                limit=eval_limit,
            )
            lm_eval_native_acc = extract_accuracy(lm_eval_native_results, task)
            print(f"lm-eval native steered accuracy: {lm_eval_native_acc:.4f}")

            # Clean up
            del lm_steered
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Store results
            results_list.append({
                "task": task,
                "method": method,
                "model": model_name,
                "layers": steering_data['layers'],
                "num_pairs": num_pairs,
                "steering_scale": scale,
                "base_accuracy": base_acc,
                "steered_accuracy_wisent": steered_acc,
                "steered_accuracy_lm_eval_native": lm_eval_native_acc,
                "difference_wisent": steered_acc - base_acc,
                "difference_lm_eval_native": lm_eval_native_acc - base_acc,
            })

    # Clean up model before next task
    del wisent_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results_list


def run_comparison(
    model_name: str,
    tasks: list[str],
    methods: list[str] = None,
    num_pairs: int = 50,
    steering_scales: list[float] = None,
    device: str = "cuda:0",
    batch_size: int = 1,
    max_batch_size: int = 8,
    eval_limit: int | None = None,
    output_dir: str = "comparison_results",
    train_ratio: float = 0.8,
    layers: str | None = None,
) -> list[dict]:
    """
    Run full comparison for multiple tasks, methods, and scales.
    """
    if methods is None:
        methods = ["caa"]
    if steering_scales is None:
        steering_scales = [1.0]

    output_dir = Path(output_dir)
    vectors_dir = output_dir / "steering_vectors"
    results_dir = output_dir / "results"

    output_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for task in tasks:
        print(f"\n{'#'*60}")
        print(f"# TASK: {task}")
        print(f"{'#'*60}")

        task_results = run_single_task(
            model_name=model_name,
            task=task,
            methods=methods,
            num_pairs=num_pairs,
            steering_scales=steering_scales,
            device=device,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            eval_limit=eval_limit,
            vectors_dir=vectors_dir,
            train_ratio=train_ratio,
            layers=layers,
        )
        all_results.extend(task_results)

        # Save results for this task
        task_results_file = results_dir / f"{task}_results.json"
        with open(task_results_file, "w") as f:
            json.dump(task_results, f, indent=2)
        print(f"Results for {task} saved to: {task_results_file}")

    # Print final summary table
    print(f"\n{'='*90}")
    print(f"FINAL COMPARISON RESULTS")
    print(f"{'='*90}")
    print(f"Model: {model_name}")
    print(f"Num pairs: {num_pairs}")
    print(f"Layers: {layers or 'default (middle)'}")
    print(f"{'='*90}")
    print(f"{'Task':<10} {'Method':<8} {'Scale':<7} {'Base':<8} {'Wisent':<8} {'lm-eval':<8} {'Diff(W)':<9} {'Diff(L)':<9}")
    print(f"{'-'*90}")

    for r in all_results:
        print(f"{r['task']:<10} {r['method']:<8} {r['steering_scale']:<7.1f} {r['base_accuracy']:<8.4f} "
              f"{r['steered_accuracy_wisent']:<8.4f} {r['steered_accuracy_lm_eval_native']:<8.4f} "
              f"{r['difference_wisent']:+<9.4f} {r['difference_lm_eval_native']:+<9.4f}")

    print(f"{'='*90}")

    print(f"\nSteering vectors saved to: {vectors_dir}")
    print(f"Results saved to: {results_dir}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Compare steering methods")
    parser.add_argument("--model", default="EleutherAI/gpt-neo-125M", help="Model name")
    parser.add_argument("--tasks", default="boolq", help="Comma-separated lm-eval tasks (e.g., boolq,cb,copa)")
    parser.add_argument("--methods", default="caa", help="Comma-separated methods (e.g., caa,sae)")
    parser.add_argument("--num-pairs", type=int, default=50, help="Number of contrastive pairs")
    parser.add_argument("--scales", default="1.0", help="Comma-separated steering scales (e.g., 0.5,1.0,1.5)")
    parser.add_argument("--layers", default=None, help="Layer(s) for steering (e.g., '9' or '8,9,10' or 'all')")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--max-batch-size", type=int, default=8, help="Max batch size for lm-eval internal batching (reduce if OOM)")
    parser.add_argument("--limit", type=int, default=None, help="Limit eval examples")
    parser.add_argument("--output-dir", default="wisent/comparison/comparison_results", help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/test split ratio (default 0.8 = 80%% train, 20%% test)")

    args = parser.parse_args()

    # Parse comma-separated values
    tasks = [t.strip() for t in args.tasks.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]
    scales = [float(s.strip()) for s in args.scales.split(",")]

    run_comparison(
        model_name=args.model,
        tasks=tasks,
        methods=methods,
        num_pairs=args.num_pairs,
        steering_scales=scales,
        device=args.device,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        eval_limit=args.limit,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        layers=args.layers,
    )


if __name__ == "__main__":
    main()
