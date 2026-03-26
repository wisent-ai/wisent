"""Analyze ZWIAD profile validation experiment results.

Downloads find_best_method results from GCS for all benchmarks in
validation_experiment.json, compares actual winners vs predicted,
and computes accuracy metrics.

Usage:
    python3 scripts/steering/validation/analyze_results.py <experiment_id>
"""
import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from wisent.core.utils.config_tools.constants import (
    EXIT_CODE_ERROR,
    JSON_INDENT,
    SEPARATOR_WIDTH_REPORT,
    SCORE_RANGE_MIN,
    INDEX_FIRST,
    PERCENT_MULTIPLIER,
    COMBO_OFFSET,
    COMBO_BASE,
)


TOP_K_LENIENT = COMBO_BASE + COMBO_OFFSET


def _gcs_ls(gcs_path):
    result = subprocess.run(
        ["gcloud", "storage", "ls", gcs_path],
        capture_output=True, text=True,
    )
    if result.returncode:
        return []
    return [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]


def _gcs_cp(src, dst):
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    subprocess.run(["gcloud", "storage", "cp", src, dst], check=True)


def _load_experiment_config():
    pkg_dir = Path(__file__).resolve().parent.parent.parent.parent
    path = (
        pkg_dir / "wisent" / "support" / "parameters"
        / "lm_eval" / "profiles" / "validation_experiment.json"
    )
    with open(path) as f:
        return json.load(f)


def _collect_benchmark_results(experiment_id, gcs_bucket, local_dir):
    """Download and parse results for each benchmark job."""
    results = {}
    gcs_prefix = f"gs://{gcs_bucket}/find_best_method/{experiment_id}"
    jobs = _gcs_ls(f"{gcs_prefix}*/methods/")
    if not jobs:
        jobs = []
        all_dirs = _gcs_ls(f"gs://{gcs_bucket}/find_best_method/")
        for d in all_dirs:
            if experiment_id in d:
                methods = _gcs_ls(d + "methods/")
                jobs.extend(methods)

    for job_path in jobs:
        parts = job_path.rstrip("/").split("/")
        job_id = None
        for idx, p in enumerate(parts):
            if p == "find_best_method" and idx + COMBO_OFFSET < len(parts):
                job_id = parts[idx + COMBO_OFFSET]
                break
        if not job_id or "__" not in job_id:
            continue
        benchmark = job_id.split("__", maxsplit=COMBO_OFFSET)[-COMBO_OFFSET]
        method_name = parts[-COMBO_OFFSET].rstrip("/")
        result_gcs = f"{job_path}method_result.json"
        local_path = os.path.join(
            local_dir, benchmark, f"{method_name}.json",
        )
        try:
            _gcs_cp(result_gcs, local_path)
            with open(local_path) as f:
                data = json.load(f)
            if benchmark not in results:
                results[benchmark] = {}
            results[benchmark][method_name] = {
                "score": data.get("best_score", SCORE_RANGE_MIN),
                "delta": data.get("delta", SCORE_RANGE_MIN),
                "time": data.get("time_seconds", SCORE_RANGE_MIN),
            }
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            continue
    return results


def _analyze(config, results):
    """Compare predicted vs actual winners."""
    benchmark_to_profile = {}
    profile_to_predicted = {}
    for profile, info in config["profiles"].items():
        profile_to_predicted[profile] = info["predicted_method"]
        for bench in info["benchmarks"]:
            benchmark_to_profile[bench] = profile

    rows = []
    for benchmark, methods in results.items():
        if not methods:
            continue
        profile = benchmark_to_profile.get(benchmark, "UNKNOWN")
        predicted = profile_to_predicted.get(profile, "?")
        ranked = sorted(
            methods.items(), key=lambda x: x[COMBO_OFFSET]["score"],
            reverse=True,
        )
        winner = ranked[INDEX_FIRST][INDEX_FIRST]
        winner_score = ranked[INDEX_FIRST][COMBO_OFFSET]["score"]
        top_k_methods = [r[INDEX_FIRST] for r in ranked[:TOP_K_LENIENT]]
        exact_match = winner == predicted
        in_top_k = predicted in top_k_methods
        predicted_rank = None
        for rank_idx, (name, _) in enumerate(ranked):
            if name == predicted:
                predicted_rank = rank_idx + COMBO_OFFSET
                break
        rows.append({
            "benchmark": benchmark,
            "profile": profile,
            "predicted": predicted,
            "winner": winner,
            "winner_score": winner_score,
            "exact_match": exact_match,
            "in_top_k": in_top_k,
            "predicted_rank": predicted_rank,
            "ranking": [
                {"method": n, "score": d["score"]} for n, d in ranked
            ],
        })
    return rows


def _print_results(rows):
    n = len(rows)
    if not n:
        print("No results to analyze.")
        return

    exact_matches = sum(r["exact_match"] for r in rows)
    top_k_matches = sum(r["in_top_k"] for r in rows)
    pct_exact = exact_matches / n * PERCENT_MULTIPLIER
    pct_top_k = top_k_matches / n * PERCENT_MULTIPLIER

    print(f"\n{'=' * SEPARATOR_WIDTH_REPORT}")
    print("ZWIAD PROFILE VALIDATION RESULTS")
    print(f"{'=' * SEPARATOR_WIDTH_REPORT}")

    print(f"\n{'Benchmark':<30s} {'Profile':<16s} {'Predicted':<10s} "
          f"{'Winner':<10s} {'Score':>7s} {'Match':>6s} {'Rank':>5s}")
    print("-" * SEPARATOR_WIDTH_REPORT)

    for r in sorted(rows, key=lambda x: x["profile"]):
        mark = "YES" if r["exact_match"] else "no"
        rank_str = str(r["predicted_rank"]) if r["predicted_rank"] else "?"
        print(
            f"  {r['benchmark']:<28s} {r['profile']:<16s} "
            f"{r['predicted']:<10s} {r['winner']:<10s} "
            f"{r['winner_score']:>7.4f} {mark:>6s} {rank_str:>5s}"
        )

    print(f"\n{'=' * SEPARATOR_WIDTH_REPORT}")
    print("PER-PROFILE SUMMARY")
    print(f"{'=' * SEPARATOR_WIDTH_REPORT}")

    by_profile = defaultdict(list)
    for r in rows:
        by_profile[r["profile"]].append(r)

    profile_exact = {}
    for profile in sorted(by_profile.keys()):
        pr = by_profile[profile]
        hits = sum(r["exact_match"] for r in pr)
        total = len(pr)
        pct = hits / total * PERCENT_MULTIPLIER if total else SCORE_RANGE_MIN
        profile_exact[profile] = hits > INDEX_FIRST
        winners = [r["winner"] for r in pr]
        winner_set = ", ".join(sorted(set(winners)))
        print(
            f"  {profile:<16s}  {hits}/{total} exact  "
            f"predicted={pr[INDEX_FIRST]['predicted']:<10s}  "
            f"actual winners: {winner_set}"
        )

    profiles_correct = sum(profile_exact.values())
    profiles_total = len(profile_exact)

    print(f"\n{'=' * SEPARATOR_WIDTH_REPORT}")
    print("OVERALL METRICS")
    print(f"{'=' * SEPARATOR_WIDTH_REPORT}")
    print(f"  Benchmarks analyzed:    {n}")
    print(f"  Exact match (top-{COMBO_OFFSET}):    "
          f"{exact_matches}/{n} ({pct_exact:.1f}%)")
    print(f"  In top-{TOP_K_LENIENT}:              "
          f"{top_k_matches}/{n} ({pct_top_k:.1f}%)")
    print(f"  Profiles with >= {COMBO_OFFSET} hit: "
          f"{profiles_correct}/{profiles_total}")

    print(f"\n  VERDICT: ", end="")
    profiles_pct = (
        profiles_correct / profiles_total * PERCENT_MULTIPLIER
        if profiles_total else SCORE_RANGE_MIN
    )
    threshold_strong = PERCENT_MULTIPLIER * COMBO_BASE / (
        COMBO_BASE + COMBO_OFFSET
    )
    threshold_moderate = PERCENT_MULTIPLIER * COMBO_BASE / (
        COMBO_BASE + COMBO_BASE
    )
    if profiles_pct >= threshold_strong:
        print("STRONG VALIDATION")
    elif pct_top_k >= threshold_strong:
        print("MODERATE VALIDATION")
    elif pct_top_k >= threshold_moderate:
        print("WEAK VALIDATION")
    else:
        print("FAILED — profiles do not predict steering success")
    print(f"{'=' * SEPARATOR_WIDTH_REPORT}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id", help="Experiment ID prefix")
    parser.add_argument(
        "--output-dir", default=None,
        help="Local directory for downloaded results",
    )
    args = parser.parse_args()

    config = _load_experiment_config()
    gcs_bucket = os.environ.get("GCS_BUCKET", "wisent-gcp-bucket")
    local_dir = args.output_dir or f"./validation_results_{args.experiment_id}"

    print(f"Collecting results for experiment: {args.experiment_id}")
    results = _collect_benchmark_results(
        args.experiment_id, gcs_bucket, local_dir,
    )

    if not results:
        print("ERROR: No results found. Workers may still be running.")
        print(f"Check: gcloud storage ls "
              f"gs://{gcs_bucket}/find_best_method/{args.experiment_id}*/")
        sys.exit(EXIT_CODE_ERROR)

    print(f"Found results for {len(results)} benchmarks")
    rows = _analyze(config, results)
    _print_results(rows)

    out_path = os.path.join(local_dir, "validation_summary.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=JSON_INDENT, default=str)
    print(f"Saved detailed results to {out_path}")


if __name__ == "__main__":
    main()
