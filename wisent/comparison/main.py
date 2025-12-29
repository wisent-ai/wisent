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
from lm_eval.models.hf_steered import SteeredModel

from wisent.core.models.wisent_model import WisentModel
from wisent.comparison import ours
from wisent.comparison import sae
from wisent.comparison import fgaa
from wisent.comparison.utils import (
    load_steering_vector,
    apply_steering_to_model,
    remove_steering,
    convert_to_lm_eval_format,
    create_test_only_task,
    extract_accuracy,
    run_lm_eval_evaluation,
    run_ll_evaluation,
)

# Map method names to modules
METHOD_MODULES = {
    "caa": ours,
    "sae": sae,
    "fgaa": fgaa,
}


def run_single_task(
    model_name: str,
    task: str,
    methods: list[str] = None,
    num_pairs: int = 50,
    steering_scales: list[float] = None,
    device: str = "cuda:0",
    batch_size: int | str = 1,
    max_batch_size: int = 8,
    eval_limit: int | None = None,
    vectors_dir: Path = None,
    train_ratio: float = 0.8,
    caa_layers: str = "12",
    sae_layers: str = "12",
    extraction_strategies: list[str] = None,
    bos_features_source: str = "detected",
) -> list[dict]:
    """
    Run comparison for a single task with multiple methods, scales, and extraction strategies.

    Returns list of result dicts, one per method/scale/strategy combination.
    """
    if methods is None:
        methods = ["caa"]
    if steering_scales is None:
        steering_scales = [1.0]
    if extraction_strategies is None:
        extraction_strategies = ["mc_balanced"]

    results_list = []

    # Step 1: Create test task
    test_pct = round((1 - train_ratio) * 100)
    print(f"\n{'='*60}")
    print(f"Creating test task for: {task}")
    print(f"(using {test_pct}% of pooled data)")
    print(f"{'='*60}")

    task_dict = create_test_only_task(task, train_ratio=train_ratio)

    # Step 2: Generate ALL steering vectors FIRST for ALL strategies (subprocess frees GPU memory after each)
    # Structure: steering_vectors_data[strategy][method] = steering_data
    steering_vectors_data = {}
    train_pct = round(train_ratio * 100)

    for method in methods:
        if method not in METHOD_MODULES:
            print(f"WARNING: Method '{method}' not implemented, skipping")
            continue

        method_module = METHOD_MODULES[method]

        # CAA uses extraction strategy, FGAA/SAE don't
        for extraction_strategy in (extraction_strategies if method == "caa" else [None]):
            print(f"\n{'@'*60}")
            print(f"@ METHOD: {method}, EXTRACTION STRATEGY: {extraction_strategy or 'N/A'}")
            print(f"{'@'*60}")

            # Select layers based on method: CAA uses caa_layers (default=middle), SAE/FGAA use sae_layers (default=12)
            method_layers = caa_layers if method == "caa" else sae_layers

            print(f"\n{'='*60}")
            print(f"Generating steering vector for: {task} (method={method})")
            print(f"(using {train_pct}% of pooled data - no overlap with test)")
            print(f"Layers: {method_layers}")
            print(f"{'='*60}")

            suffix = f"_{extraction_strategy}" if extraction_strategy else ""
            vector_path = vectors_dir / f"{task}_{method}{suffix}_steering_vector.json"

            kwargs = {
                "task": task,
                "model_name": model_name,
                "output_path": vector_path,
                "num_pairs": num_pairs,
                "device": device,
                "layers": method_layers,
            }
            if extraction_strategy:
                kwargs["extraction_strategy"] = extraction_strategy
            if method == "fgaa":
                kwargs["bos_features_source"] = bos_features_source

            method_module.generate_steering_vector(**kwargs)

            steering_data = load_steering_vector(vector_path, default_method=method)
            if extraction_strategy not in steering_vectors_data:
                steering_vectors_data[extraction_strategy] = {}
            steering_vectors_data[extraction_strategy][method] = steering_data
            print(f"Loaded steering vector with layers: {steering_data['layers']}")

    # Step 3: Load model once for ALL evaluations
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"{'='*60}")
    wisent_model = WisentModel(model_name=model_name, device=device)

    # Step 4: Run base evaluation (no steering applied)
    print(f"\n{'='*60}")
    print(f"Running BASE evaluation for: {task}")
    print(f"{'='*60}")

    base_results = run_lm_eval_evaluation(
        wisent_model=wisent_model,
        task_dict=task_dict,
        task_name=task,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        limit=eval_limit,
    )
    base_acc = extract_accuracy(base_results, task)
    print(f"Base accuracy (lm-eval): {base_acc:.4f}")

    # Step 4b: Run base LL evaluation (no steering)
    print(f"\n{'='*60}")
    print(f"Running BASE LL evaluation for: {task}")
    print(f"{'='*60}")

    base_ll_acc = run_ll_evaluation(
        wisent_model=wisent_model,
        task_dict=task_dict,
        task_name=task,
        limit=eval_limit,
    )
    print(f"Base accuracy (LL): {base_ll_acc:.4f}")

    # Step 5: Run ALL wisent steered evaluations first (model stays loaded)
    # Structure: wisent_results[(strategy, method, scale)] = steered_acc
    wisent_results = {}
    for method in methods:
        # CAA uses extraction strategy, FGAA/SAE don't
        for extraction_strategy in (extraction_strategies if method == "caa" else [None]):
            if extraction_strategy not in steering_vectors_data:
                continue
            if method not in steering_vectors_data[extraction_strategy]:
                continue

            steering_data = steering_vectors_data[extraction_strategy][method]

            for scale in steering_scales:
                print(f"\n{'='*60}")
                print(f"Running STEERED evaluation for: {task} (strategy={extraction_strategy}, method={method}, scale={scale})")
                print(f"{'='*60}")

                # Apply steering to existing model
                apply_steering_to_model(wisent_model, steering_data, scale=scale)

                steered_results = run_lm_eval_evaluation(
                    wisent_model=wisent_model,
                    task_dict=task_dict,
                    task_name=task,
                    batch_size=batch_size,
                    max_batch_size=max_batch_size,
                    limit=eval_limit,
                )
                steered_acc = extract_accuracy(steered_results, task)
                print(f"Steered accuracy (lm-eval): {steered_acc:.4f}")

                # Run steered LL evaluation
                steered_ll_acc = run_ll_evaluation(
                    wisent_model=wisent_model,
                    task_dict=task_dict,
                    task_name=task,
                    limit=eval_limit,
                )
                print(f"Steered accuracy (LL): {steered_ll_acc:.4f}")

                # Remove steering for next iteration
                remove_steering(wisent_model)

                # Store wisent results
                wisent_results[(extraction_strategy, method, scale)] = {
                    "lm_eval": steered_acc,
                    "ll": steered_ll_acc,
                }

    # Step 6: Free wisent_model to make room for SteeredModel
    del wisent_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Step 7: Run ALL lm-eval native steered evaluations (one at a time)
    for method in methods:
        # CAA uses extraction strategy, FGAA/SAE don't
        for extraction_strategy in (extraction_strategies if method == "caa" else [None]):
            if extraction_strategy not in steering_vectors_data:
                continue
            if method not in steering_vectors_data[extraction_strategy]:
                continue

            steering_data = steering_vectors_data[extraction_strategy][method]

            for scale in steering_scales:
                print(f"\n{'='*60}")
                print(f"Running lm-eval NATIVE steered for: {task} (strategy={extraction_strategy}, method={method}, scale={scale})")
                print(f"{'='*60}")

                # Convert steering vector to lm-eval format
                suffix = f"_{extraction_strategy}" if extraction_strategy else ""
                lm_eval_steer_path = vectors_dir / f"{task}_{method}{suffix}_lm_eval_steer_scale{scale}.pt"
                convert_to_lm_eval_format(steering_data, lm_eval_steer_path, scale=scale)

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

                # Clean up SteeredModel to free GPU for next iteration
                del lm_steered
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Store combined results
                wisent_result = wisent_results[(extraction_strategy, method, scale)]
                steered_acc_lm_eval = wisent_result["lm_eval"]
                steered_acc_ll = wisent_result["ll"]
                results_list.append({
                    "task": task,
                    "extraction_strategy": extraction_strategy or "N/A",
                    "method": method,
                    "model": model_name,
                    "layers": steering_data['layers'],
                    "num_pairs": num_pairs,
                    "steering_scale": scale,
                    "base_accuracy_lm_eval": base_acc,
                    "base_accuracy_ll": base_ll_acc,
                    "steered_accuracy_lm_eval": steered_acc_lm_eval,
                    "steered_accuracy_ll": steered_acc_ll,
                    "steered_accuracy_lm_eval_native": lm_eval_native_acc,
                    "difference_lm_eval": steered_acc_lm_eval - base_acc,
                    "difference_ll": steered_acc_ll - base_ll_acc,
                    "difference_lm_eval_native": lm_eval_native_acc - base_acc,
                })

    return results_list


def run_comparison(
    model_name: str,
    tasks: list[str],
    methods: list[str] = None,
    num_pairs: int = 50,
    steering_scales: list[float] = None,
    device: str = "cuda:0",
    batch_size: int | str = 1,
    max_batch_size: int = 8,
    eval_limit: int | None = None,
    output_dir: str = "comparison_results",
    train_ratio: float = 0.8,
    caa_layers: str = "12",
    sae_layers: str = "12",
    extraction_strategies: list[str] = None,
    bos_features_source: str = "detected",
) -> list[dict]:
    """
    Run full comparison for multiple tasks, methods, scales, and extraction strategies.
    """
    if methods is None:
        methods = ["caa"]
    if steering_scales is None:
        steering_scales = [1.0]
    if extraction_strategies is None:
        extraction_strategies = ["mc_balanced"]

    output_dir = Path(output_dir)
    # Add model name to path (sanitize "/" -> "_")
    model_dir_name = model_name.replace("/", "_")
    output_dir = output_dir / model_dir_name
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
            caa_layers=caa_layers,
            sae_layers=sae_layers,
            extraction_strategies=extraction_strategies,
            bos_features_source=bos_features_source,
        )
        all_results.extend(task_results)

        # Save results for this task (includes all strategies)
        task_results_file = results_dir / f"{task}_results.json"
        with open(task_results_file, "w") as f:
            json.dump(task_results, f, indent=2)
        print(f"Results for {task} saved to: {task_results_file}")

    # Print final summary table
    print(f"\n{'='*150}")
    print(f"FINAL COMPARISON RESULTS")
    print(f"{'='*150}")
    print(f"Model: {model_name}")
    print(f"Num pairs: {num_pairs}")
    print(f"CAA Layers: {caa_layers}")
    print(f"SAE/FGAA Layers: {sae_layers}")
    print(f"Strategies: {', '.join(extraction_strategies)}")
    print(f"{'='*150}")
    print(f"{'Strategy':<16} {'Task':<10} {'Method':<8} {'Scale':<6} {'Base(E)':<8} {'Base(L)':<8} {'Steer(E)':<9} {'Steer(L)':<9} {'Native':<8} {'Diff(E)':<8} {'Diff(L)':<8} {'Diff(N)':<8}")
    print(f"{'-'*150}")

    for r in all_results:
        print(f"{r.get('extraction_strategy', 'N/A'):<16} {r['task']:<10} {r['method']:<8} {r['steering_scale']:<6.1f} "
              f"{r['base_accuracy_lm_eval']:<8.4f} {r['base_accuracy_ll']:<8.4f} "
              f"{r['steered_accuracy_lm_eval']:<9.4f} {r['steered_accuracy_ll']:<9.4f} {r['steered_accuracy_lm_eval_native']:<8.4f} "
              f"{r['difference_lm_eval']:+<8.4f} {r['difference_ll']:+<8.4f} {r['difference_lm_eval_native']:+<8.4f}")

    print(f"{'='*150}")

    print(f"\nSteering vectors saved to: {vectors_dir}")
    print(f"Results saved to: {results_dir}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Compare steering methods")
    parser.add_argument("--model", default="EleutherAI/gpt-neo-125M", help="Model name")
    parser.add_argument("--tasks", default="boolq", help="Comma-separated lm-eval tasks (e.g., boolq,cb,copa)")
    parser.add_argument("--methods", default="caa", help="Comma-separated methods (e.g., caa,sae,fgaa)")
    parser.add_argument("--num-pairs", type=int, default=50, help="Number of contrastive pairs")
    parser.add_argument("--scales", default="1.0", help="Comma-separated steering scales (e.g., 0.5,1.0,1.5)")
    parser.add_argument("--caa-layers", default="12", help="Layer(s) for CAA steering (default: 12)")
    parser.add_argument("--sae-layers", default="12", help="Layer(s) for SAE/FGAA steering (default: 12)")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--batch-size", default=1, help="Batch size (int or 'auto')")
    parser.add_argument("--max-batch-size", type=int, default=8, help="Max batch size for lm-eval internal batching (reduce if OOM)")
    parser.add_argument("--limit", type=int, default=None, help="Limit eval examples")
    parser.add_argument("--output-dir", default="wisent/comparison/comparison_results", help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/test split ratio (default 0.8 = 80%% train, 20%% test)")
    parser.add_argument("--extraction-strategy", default="mc_balanced",
                        help="Extraction strategy (comma-separated for multiple). Chat models: chat_mean, chat_first, chat_last, chat_max_norm, chat_weighted, role_play, mc_balanced. Base models: completion_last, completion_mean, mc_completion")
    parser.add_argument("--bos-features-source", default="detected",
                        help="BOS features source for FGAA: 'paper' (5 features), 'detected' (12 features), or 'none'")

    args = parser.parse_args()

    # Parse comma-separated values
    tasks = [t.strip() for t in args.tasks.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]
    scales = [float(s.strip()) for s in args.scales.split(",")]
    extraction_strategies = [s.strip() for s in args.extraction_strategy.split(",")]

    # Parse batch_size (can be int or "auto")
    batch_size = args.batch_size if args.batch_size == "auto" else int(args.batch_size)

    run_comparison(
        model_name=args.model,
        tasks=tasks,
        methods=methods,
        num_pairs=args.num_pairs,
        steering_scales=scales,
        device=args.device,
        batch_size=batch_size,
        max_batch_size=args.max_batch_size,
        eval_limit=args.limit,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        caa_layers=args.caa_layers,
        sae_layers=args.sae_layers,
        extraction_strategies=extraction_strategies,
        bos_features_source=args.bos_features_source,
    )


if __name__ == "__main__":
    main()
