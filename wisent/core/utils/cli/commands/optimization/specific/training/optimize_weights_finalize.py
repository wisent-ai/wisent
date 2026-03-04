"""GCS helpers and result finalization for optimize-weights."""
import json
import os
import subprocess
import time

from wisent.core.utils.config_tools.constants import DISPLAY_TRUNCATION_SHORT, JSON_INDENT, SECONDS_PER_MINUTE


def upload_to_gcs(local_path: str, gcs_bucket: str, gcs_key: str) -> bool:
    """Upload a file or directory to GCS."""
    try:
        if os.path.isdir(local_path):
            cmd = ["gcloud", "storage", "rsync", local_path, f"gs://{gcs_bucket}/{gcs_key}", "--quiet"]
        else:
            cmd = ["gcloud", "storage", "cp", local_path, f"gs://{gcs_bucket}/{gcs_key}", "--quiet"]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as e:
        print(f"   Warning: GCS upload failed: {e}")
        return False


def download_from_gcs(gcs_bucket: str, gcs_key: str, local_path: str) -> bool:
    """Download a file or directory from GCS."""
    try:
        gcs_path = f"gs://{gcs_bucket}/{gcs_key}"
        # Check if it exists
        check_cmd = ["gcloud", "storage", "ls", gcs_path]
        result = subprocess.run(check_cmd, capture_output=True)
        if result.returncode != 0:
            return False
        # Download
        if gcs_key.endswith('/'):
            cmd = ["gcloud", "storage", "rsync", gcs_path, local_path, "--quiet"]
        else:
            os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
            cmd = ["gcloud", "storage", "cp", gcs_path, local_path, "--quiet"]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception:
        return False


def _finalize_optimization(
    args, result, direction, evaluator_display, base_model,
    tokenizer, optimizer, steering_vectors, base_state_dict,
    num_layers, optimizer_config, start_time,
):
    """Finalize optimization: save model, upload to GCS, show comparisons.

    Returns:
        OptimizationResult
    """
    from wisent.core.utils.cli.optimization.specific.optimize_weights_comparisons import (
        _show_response_comparisons,
    )

    best_params = result.best_params
    best_value = result.best_value

    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nBest parameters:")
    for k, v in best_params.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
    print(f"\nBest {args.target_metric}: {best_value:.4f}")

    # Check if target achieved
    if direction == "maximize":
        target_achieved = best_value >= args.target_value
    else:
        target_achieved = best_value <= args.target_value
    print(f"\nTarget {args.target_value} achieved: {'YES' if target_achieved else 'NO'}")

    # Apply best parameters and save final model
    print(f"\n{'='*80}")
    print("SAVING OPTIMIZED MODEL")
    print(f"{'='*80}")

    optimizer.apply_best_params(best_params)

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nSaving optimized model to {args.output_dir}...")
    base_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save optimization metadata
    metadata = {
        "model": args.model,
        "task": args.task,
        "trait": getattr(args, 'trait', None),
        "evaluator_type": evaluator_display,
        "target_metric": args.target_metric,
        "target_value": args.target_value,
        "best_params": best_params,
        "best_score": best_value,
        "target_achieved": target_achieved,
        "total_trials": len(result.study.trials),
        "direction": direction,
    }

    with open(os.path.join(args.output_dir, "optimization_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=JSON_INDENT)

    print(f"   Model saved")
    print(f"   Metadata saved to optimization_metadata.json")

    # Upload to GCS if --gcs-bucket is provided
    gcs_bucket = getattr(args, 'gcs_bucket', None)
    if gcs_bucket:
        task_name = args.task.replace(',', '_')[:DISPLAY_TRUNCATION_SHORT] if args.task else (args.trait or 'unknown')[:DISPLAY_TRUNCATION_SHORT]
        gcs_key = f"optimization-results/{task_name}/{time.strftime('%Y%m%d-%H%M%S')}"
        print(f"\n   Uploading results to gs://{gcs_bucket}/{gcs_key}/...")
        if upload_to_gcs(args.output_dir, gcs_bucket, gcs_key):
            print(f"   Results uploaded to GCS")
        else:
            print(f"   GCS upload failed")

    # Save all trials if requested
    if args.save_trials:
        trials_data = [
            {
                "trial": t.number,
                "params": t.params,
                "score": t.value,
            }
            for t in result.study.trials
        ]
        with open(args.save_trials, "w") as f:
            json.dump(trials_data, f, indent=JSON_INDENT)
        print(f"   Trials saved to {args.save_trials}")

    # Push to hub if requested
    if args.push_to_hub:
        if not args.repo_id:
            print("\n   ERROR: --repo-id required for --push-to-hub")
        else:
            print(f"\n   Pushing to HuggingFace Hub: {args.repo_id}...")
            base_model.push_to_hub(args.repo_id)
            tokenizer.push_to_hub(args.repo_id)
            print(f"   Pushed successfully")

    # Show/save before/after comparisons if requested
    save_comparisons_path = getattr(args, 'save_comparisons', None)
    show_comparisons_count = args.show_comparisons
    if show_comparisons_count > 0 or save_comparisons_path:
        _show_response_comparisons(
            base_model=base_model,
            base_state_dict=base_state_dict,
            steering_vectors=steering_vectors,
            best_params=best_params,
            num_layers=num_layers,
            model_name=args.model,
            args=args,
            optimizer_config=optimizer_config,
            num_comparisons=show_comparisons_count if show_comparisons_count > 0 else None,
            save_path=save_comparisons_path,
        )

    total_time = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"Total optimization time: {total_time:.1f}s ({total_time/SECONDS_PER_MINUTE:.1f} min)")
    print(f"{'='*80}\n")

    from wisent.core.utils.cli.optimization.specific.optimize_weights import OptimizationResult
    return OptimizationResult(
        best_params=best_params,
        best_score=best_value,
        target_achieved=target_achieved,
        total_time=total_time,
        total_trials=len(result.study.trials),
    )

