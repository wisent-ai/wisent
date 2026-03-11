"""Single-method worker for parallel GCP steering optimization.

Loads pairs from storage (HF/Supabase/cache), runs optimization
for ONE method, uploads results to GCS.

Environment variables (all required):
    MODEL_NAME: HuggingFace model ID
    BENCHMARK: Benchmark task name
    METHOD: Steering method name (e.g. CAA, OSTRZE)
    JOB_ID: Shared job identifier for GCS paths
    TRIALS_MULTIPLIER: Trials per dimension
    BACKEND: Optimizer backend ('hyperopt' or 'optuna')
    GCS_BUCKET: GCS bucket name
"""
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from wisent.core.utils.config_tools.constants import (
    EXIT_CODE_ERROR,
    JSON_INDENT,
    SPLIT_RATIO_TRAIN_DEFAULT,
)
from wisent.core.utils.cli.optimize_steering.search_space import (
    get_method_space,
)
from wisent.core.utils.cli.optimize_steering.pipeline import (
    create_objective,
)
from wisent.core.utils.services.optimization.core.atoms import (
    BaseOptimizer,
    HPOConfig,
)
from wisent.core.utils.cli.optimize_steering.pipeline.comprehensive import (
    baseline_cache,
)


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: {name} environment variable is required")
        sys.exit(EXIT_CODE_ERROR)
    return val


def _gcs_upload(local_path: str, gcs_path: str):
    """Upload a file or directory to GCS."""
    cmd = ["gcloud", "storage", "cp"]
    if os.path.isdir(local_path):
        cmd.append("-r")
    cmd.extend([local_path, gcs_path])
    subprocess.run(cmd, check=True)


def main():
    model_name = _require_env("MODEL_NAME")
    benchmark = _require_env("BENCHMARK")
    method = _require_env("METHOD")
    job_id = _require_env("JOB_ID")
    trials_mult = int(_require_env("TRIALS_MULTIPLIER"))
    backend = _require_env("BACKEND")
    gcs_bucket = _require_env("GCS_BUCKET")

    output_dir = f"/home/ubuntu/output/{method}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    gcs_base = f"gs://{gcs_bucket}/find_best_method/{job_id}"

    # Load pairs from storage (HF/Supabase/cache) or generate
    pairs = _load_and_split_pairs(benchmark, output_dir)
    train_file, test_file, n_train = pairs

    from transformers import AutoConfig as _AC
    cfg = _AC.from_pretrained(model_name, trust_remote_code=True)
    num_layers = cfg.num_hidden_layers

    method_upper = method.upper()
    space = get_method_space(method_upper, num_layers)
    n_trials = len(space) * trials_mult

    print(f"Worker: {method_upper}")
    print(f"  Model: {model_name}")
    print(f"  Benchmark: {benchmark}")
    print(f"  Dims: {len(space)}, Trials: {n_trials}")
    print(f"  Train pairs: {n_train}")
    sys.stdout.flush()

    # Baseline (HF cache handles deduplication)
    baseline_score = _run_baseline(model_name, benchmark, test_file)
    print(f"  Baseline: {baseline_score:.4f}")
    sys.stdout.flush()

    # Run optimization
    method_start = time.time()
    optimizer = BaseOptimizer()
    optimizer.direction = "maximize"

    trials_dir = os.path.join(output_dir, "trials", method)
    workspace = os.path.join(trials_dir, "_workspace")
    os.makedirs(workspace, exist_ok=True)

    objective = create_objective(
        method=method_upper, model=model_name, task=benchmark,
        num_layers=num_layers, limit=n_train, device=None,
        work_dir=workspace,
        train_pairs_file=train_file,
        test_pairs_file=test_file,
    )

    trial_counter = []

    def persisted_objective(params):
        score = objective(params)
        trial_idx = len(trial_counter)
        trial_counter.append(trial_idx)
        trial_dir = os.path.join(trials_dir, f"trial_{trial_idx:04d}")
        os.makedirs(trial_dir, exist_ok=True)
        for fname in ("responses.json", "scores.json"):
            src = os.path.join(workspace, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(trial_dir, fname))
        meta = {
            "params": params,
            "score": score,
            "trial": trial_idx,
        }
        with open(os.path.join(trial_dir, "trial_meta.json"), "w") as f:
            json.dump(meta, f, indent=JSON_INDENT, default=str)
        return score

    result = optimizer.optimize_fn(persisted_objective, space, n_trials, cfg=HPOConfig(backend=backend))
    method_time = time.time() - method_start

    _save_and_upload(
        output_dir, method, result, method_time,
        baseline_score, gcs_base,
    )


def _load_and_split_pairs(benchmark, output_dir):
    """Load pairs via build_contrastive_pairs and split train/test."""
    from wisent.extractors.lm_eval.lm_task_pairs_generation import (
        build_contrastive_pairs,
    )

    pairs = build_contrastive_pairs(
        task_name=benchmark,
        train_ratio=SPLIT_RATIO_TRAIN_DEFAULT,
    )
    if not pairs:
        print(f"ERROR: No pairs for {benchmark}")
        sys.exit(EXIT_CODE_ERROR)

    split_idx = math.floor(len(pairs) * SPLIT_RATIO_TRAIN_DEFAULT)
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]

    def _save(pair_list, path):
        data = {
            "task_name": benchmark,
            "num_pairs": len(pair_list),
            "pairs": [p.to_dict() for p in pair_list],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=JSON_INDENT)

    train_path = os.path.join(output_dir, f"train_pairs_{benchmark}.json")
    test_path = os.path.join(output_dir, f"test_pairs_{benchmark}.json")
    _save(train_pairs, train_path)
    _save(test_pairs, test_path)
    print(f"  Pairs: {len(pairs)} total, {len(train_pairs)} train, {len(test_pairs)} test")
    return train_path, test_path, len(train_pairs)


def _save_and_upload(
    output_dir, method, result, method_time, baseline_score, gcs_base,
):
    """Save method result summary and upload to GCS."""
    summary = {
        "method": method,
        "best_score": result.best_score,
        "best_params": result.best_params,
        "n_trials": result.n_trials,
        "backend": result.backend,
        "time_seconds": method_time,
        "all_trials": result.all_trials,
        "baseline_score": baseline_score,
        "delta": result.best_score - baseline_score,
    }
    summary_path = os.path.join(output_dir, "method_result.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=JSON_INDENT, default=str)

    method_upper = method.upper()
    print(f"\n  {method_upper}: score={result.best_score:.4f} "
          f"delta={result.best_score - baseline_score:+.4f} "
          f"in {method_time:.1f}s")

    gcs_method = f"{gcs_base}/methods/{method}/"
    _gcs_upload(output_dir, gcs_method)
    print(f"  Uploaded to {gcs_method}")


def _run_baseline(model_name, benchmark, test_file):
    """Evaluate unsteered model, using HF cache if available."""
    if baseline_cache.check_baseline_exists(model_name, benchmark):
        print("  Loading cached baseline from HuggingFace...")
        _, scores, meta = baseline_cache.load_baseline_from_hf(
            model_name, benchmark,
        )
        return meta.get(
            "accuracy",
            sum(s["correct"] for s in scores) / len(scores),
        )
    print("  Generating baseline (no cache found)...")
    acc, _, _ = baseline_cache.generate_and_upload_baseline(
        model_name, benchmark, test_file, None,
        baseline_cache.build_default_hf_retry_config(),
    )
    return acc


if __name__ == "__main__":
    main()
