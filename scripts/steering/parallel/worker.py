"""Worker for GCP steering optimization (single or multi-method).

Loads pairs, runs optimization, uploads results to GCS.
Env vars: MODEL_NAME, BENCHMARK, METHOD, JOB_ID, TRIALS_MULTIPLIER, BACKEND, GCS_BUCKET
"""
import concurrent.futures
import gc
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from wisent.core.utils.config_tools.constants import (
    EXIT_CODE_ERROR, INDEX_FIRST, JSON_INDENT,
    MB_PER_GB, N_JOBS_SINGLE, OPTUNA_BACKEND_NAME,
    SCORE_RANGE_MIN, SPLIT_RATIO_TRAIN_DEFAULT,
)
from wisent.core.utils.cli.optimize_steering.search_space import (
    get_method_space,
)
from wisent.core.utils.cli.optimize_steering.pipeline import (
    create_objective,
)
from wisent.core.utils.services.optimization.core.atoms import (
    BaseOptimizer, HPOConfig,
)
from wisent.core.utils.cli.optimize_steering.pipeline.comprehensive import (
    baseline_cache,
)
from wisent.core.utils.infra_tools.infra.core.hardware import (
    detect_system_resources,
    estimate_max_gpu_workers,
    estimate_model_memory_mb,
)


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: {name} environment variable is required")
        sys.exit(EXIT_CODE_ERROR)
    return val


def _gcs_upload(local_path: str, gcs_path: str):
    cmd = ["gcloud", "storage", "cp"] + (["-r"] if os.path.isdir(local_path) else [])
    subprocess.run(cmd + [local_path, gcs_path], check=True)


def _run_single_method(
    method, model_name, benchmark, num_layers,
    train_file, test_file, n_train,
    trials_mult, backend, output_dir, gpu_id=None,
) -> dict:
    """Run optimization for one method. All args are primitives for spawn."""
    import torch
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        torch.cuda.reset_peak_memory_stats()
    method_upper = method.upper()
    space = get_method_space(method_upper, num_layers)
    n_trials = len(space) * trials_mult
    trials_dir = os.path.join(output_dir, "trials", method)
    workspace = os.path.join(trials_dir, "_workspace")
    os.makedirs(workspace, exist_ok=True)
    objective = create_objective(
        method=method_upper, model=model_name, task=benchmark,
        num_layers=num_layers, limit=n_train, device=None,
        work_dir=workspace,
        train_pairs_file=train_file, test_pairs_file=test_file,
    )
    trial_counter = []

    def persisted_objective(params):
        try:
            score = objective(params)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM on trial {len(trial_counter)}, returning min score")
            if has_gpu:
                torch.cuda.empty_cache()
            gc.collect()
            score = SCORE_RANGE_MIN
        trial_idx = len(trial_counter)
        trial_counter.append(trial_idx)
        trial_dir = os.path.join(trials_dir, f"trial_{trial_idx:04d}")
        os.makedirs(trial_dir, exist_ok=True)
        for fname in ("responses.json", "scores.json"):
            src = os.path.join(workspace, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(trial_dir, fname))
        meta = {
            "params": params, "score": score, "trial": trial_idx,
            "reserved_gpu_mb": torch.cuda.max_memory_reserved() // MB_PER_GB // MB_PER_GB if has_gpu else None,
            "allocated_gpu_mb": torch.cuda.max_memory_allocated() // MB_PER_GB // MB_PER_GB if has_gpu else None,
        }
        with open(os.path.join(trial_dir, "trial_meta.json"), "w") as f:
            json.dump(meta, f, indent=JSON_INDENT, default=str)
        if has_gpu:
            torch.cuda.empty_cache()
        gc.collect()
        return score

    method_start = time.time()
    optimizer = BaseOptimizer()
    optimizer.direction = "maximize"
    result = optimizer.optimize_fn(
        persisted_objective, space, n_trials,
        cfg=HPOConfig(backend=OPTUNA_BACKEND_NAME),
    )
    return {
        "method": method, "best_score": result.best_score,
        "best_params": result.best_params, "n_trials": result.n_trials,
        "backend": result.backend, "all_trials": result.all_trials,
        "time_seconds": time.time() - method_start,
        "reserved_gpu_mb": torch.cuda.max_memory_reserved() // MB_PER_GB // MB_PER_GB if has_gpu else None,
        "allocated_gpu_mb": torch.cuda.max_memory_allocated() // MB_PER_GB // MB_PER_GB if has_gpu else None,
    }


def _detect_gpu_layout(model_name):
    """Detect GPU count and compute workers per GPU and total."""
    import torch
    num_gpus = max(N_JOBS_SINGLE, torch.cuda.device_count())
    per_worker_mb = estimate_model_memory_mb(model_name)
    res = detect_system_resources()
    if not res.gpu_mem_mb or per_worker_mb > res.gpu_mem_mb:
        return num_gpus, N_JOBS_SINGLE, N_JOBS_SINGLE
    workers_per_gpu = res.gpu_mem_mb // per_worker_mb
    total_workers = max(N_JOBS_SINGLE, workers_per_gpu * num_gpus)
    return num_gpus, workers_per_gpu, total_workers


def _dispatch_parallel(
    methods, model_name, benchmark, num_layers,
    train_file, test_file, n_train,
    trials_mult, backend, output_dir, baseline_score, gcs_base,
):
    """Run multiple methods in parallel via ProcessPoolExecutor."""
    num_gpus, workers_per_gpu, total_workers = _detect_gpu_layout(
        model_name,
    )
    print(f"GPU layout: {num_gpus} GPU(s), {workers_per_gpu} workers/GPU, "
          f"{total_workers} total for {len(methods)} methods")
    sys.stdout.flush()
    ctx = mp.get_context("spawn")
    completed = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=total_workers, mp_context=ctx,
    ) as pool:
        futures = {}
        for idx, method in enumerate(methods):
            gpu_id = idx % num_gpus
            method_dir = os.path.join(output_dir, method)
            Path(method_dir).mkdir(parents=True, exist_ok=True)
            fut = pool.submit(
                _run_single_method, method, model_name, benchmark,
                num_layers, train_file, test_file, n_train,
                trials_mult, backend, method_dir, gpu_id,
            )
            futures[fut] = method
        for fut in concurrent.futures.as_completed(futures):
            method = futures[fut]
            try:
                result_dict = fut.result()
                print(f"  Completed {method.upper()}: "
                      f"score={result_dict['best_score']:.4f}")
                _save_and_upload(
                    os.path.join(output_dir, method), method,
                    result_dict, baseline_score, gcs_base,
                )
                completed.append(method)
            except Exception as exc:
                print(f"  FAILED {method.upper()}: {exc}")
            sys.stdout.flush()
    print(f"\nFinished: {len(completed)}/{len(methods)} methods succeeded")


def _save_and_upload(
    output_dir, method, result_dict, baseline_score, gcs_base,
):
    """Save method result summary and upload to GCS."""
    summary = dict(result_dict, baseline_score=baseline_score,
                   delta=result_dict["best_score"] - baseline_score)
    with open(os.path.join(output_dir, "method_result.json"), "w") as f:
        json.dump(summary, f, indent=JSON_INDENT, default=str)
    print(f"\n  {method.upper()}: score={result_dict['best_score']:.4f} "
          f"delta={summary['delta']:+.4f} reserved_gpu_mb={result_dict.get('reserved_gpu_mb')}")
    gcs_method = f"{gcs_base}/methods/{method}/"
    _gcs_upload(output_dir, gcs_method)
    print(f"  Uploaded to {gcs_method}")


def main():
    model_name = _require_env("MODEL_NAME")
    benchmark = _require_env("BENCHMARK")
    method_str = _require_env("METHOD")
    job_id = _require_env("JOB_ID")
    trials_mult = int(_require_env("TRIALS_MULTIPLIER"))
    backend = _require_env("BACKEND")
    gcs_bucket = _require_env("GCS_BUCKET")
    methods = [m.strip() for m in method_str.split(",")]
    output_dir = f"/home/ubuntu/output/{job_id}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    gcs_base = f"gs://{gcs_bucket}/find_best_method/{job_id}"
    # Load pairs and baseline ONCE (shared across all methods)
    pairs = _load_and_split_pairs(benchmark, output_dir)
    train_file, test_file, n_train = pairs
    from transformers import AutoConfig as _AC
    cfg = _AC.from_pretrained(model_name, trust_remote_code=True)
    num_layers = cfg.num_hidden_layers
    baseline_score = _run_baseline(model_name, benchmark, test_file)
    print(f"  Baseline: {baseline_score:.4f}")
    sys.stdout.flush()
    if len(methods) == N_JOBS_SINGLE:
        method = methods[INDEX_FIRST]
        method_dir = os.path.join(output_dir, method)
        Path(method_dir).mkdir(parents=True, exist_ok=True)
        print(f"Worker: {method.upper()} (single mode)")
        sys.stdout.flush()
        result_dict = _run_single_method(
            method, model_name, benchmark, num_layers,
            train_file, test_file, n_train,
            trials_mult, backend, method_dir,
        )
        _save_and_upload(
            method_dir, method, result_dict, baseline_score, gcs_base,
        )
    else:
        print(f"Worker: {len(methods)} methods (parallel mode)")
        sys.stdout.flush()
        _dispatch_parallel(
            methods, model_name, benchmark, num_layers,
            train_file, test_file, n_train,
            trials_mult, backend, output_dir,
            baseline_score, gcs_base,
        )


def _load_and_split_pairs(benchmark, output_dir):
    """Load pairs via build_contrastive_pairs and split train/test."""
    from wisent.extractors.lm_eval.lm_task_pairs_generation import (
        build_contrastive_pairs,
    )
    pairs = build_contrastive_pairs(
        task_name=benchmark, train_ratio=SPLIT_RATIO_TRAIN_DEFAULT,
    )
    if not pairs:
        print(f"ERROR: No pairs for {benchmark}")
        sys.exit(EXIT_CODE_ERROR)
    split_idx = math.floor(len(pairs) * SPLIT_RATIO_TRAIN_DEFAULT)
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]

    def _save(pair_list, path):
        data = {
            "task_name": benchmark, "num_pairs": len(pair_list),
            "pairs": [p.to_dict() for p in pair_list],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=JSON_INDENT)

    train_path = os.path.join(output_dir, f"train_pairs_{benchmark}.json")
    test_path = os.path.join(output_dir, f"test_pairs_{benchmark}.json")
    _save(train_pairs, train_path)
    _save(test_pairs, test_path)
    print(f"  Pairs: {len(pairs)} total, "
          f"{len(train_pairs)} train, {len(test_pairs)} test")
    return train_path, test_path, len(train_pairs)


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
    )
    return acc


if __name__ == "__main__":
    main()
