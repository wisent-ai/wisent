"""Collect results from parallel GCP steering optimization.

Downloads per-method results from GCS after all workers complete,
builds final ranking with delta vs baseline and response diff.

Usage: python3 scripts/steering/parallel/collect.py <job_id> [--output-dir DIR]
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

from wisent.core.utils.config_tools.constants import (
    EXIT_CODE_ERROR,
    JSON_INDENT,
    SCORE_RANGE_MIN,
    SEPARATOR_WIDTH_REPORT,
    SEPARATOR_WIDTH_WIDE,
)


def _gcs_download_dir(gcs_path: str, local_path: str):
    """Download a directory from GCS."""
    os.makedirs(local_path, exist_ok=True)
    subprocess.run(
        ["gcloud", "storage", "cp", "-r", gcs_path, local_path],
        check=True,
    )


def _gcs_ls(gcs_path: str) -> list:
    """List objects in GCS path."""
    result = subprocess.run(
        ["gcloud", "storage", "ls", gcs_path],
        capture_output=True, text=True,
    )
    if result.returncode:
        return []
    return [
        line.strip() for line in result.stdout.strip().split("\n")
        if line.strip()
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Collect parallel find_best_method results",
    )
    parser.add_argument("job_id", help="Job ID from launcher")
    parser.add_argument(
        "--output-dir", default=None,
        help="Local output directory",
    )
    args = parser.parse_args()

    job_id = args.job_id
    output_dir = args.output_dir or f"./results_{job_id}"
    gcs_bucket = os.environ.get("GCS_BUCKET", "wisent-gcp-bucket")
    gcs_base = f"gs://{gcs_bucket}/find_best_method/{job_id}"

    # Check which methods have completed
    methods_gcs = _gcs_ls(f"{gcs_base}/methods/")
    if not methods_gcs:
        print(f"No method results found at {gcs_base}/methods/")
        print("Workers may still be running.")
        sys.exit(EXIT_CODE_ERROR)

    completed = _extract_method_names(methods_gcs)
    print(f"Found results for {len(completed)} methods: {completed}")

    # Download all results
    os.makedirs(output_dir, exist_ok=True)
    for method in completed:
        local_method = os.path.join(output_dir, method)
        print(f"  Downloading {method}...")
        _gcs_download_dir(
            f"{gcs_base}/methods/{method}/", local_method,
        )

    # Load method results and build ranking
    method_results, baseline_score = _load_results(
        output_dir, completed,
    )

    if not method_results:
        print("ERROR: No method results loaded")
        sys.exit(EXIT_CODE_ERROR)

    benchmark = _get_benchmark_from_pairs(output_dir, gcs_base)

    _print_ranking(
        method_results, baseline_score, benchmark,
        output_dir, job_id,
    )


def _extract_method_names(gcs_paths):
    """Extract method names from GCS listing paths."""
    names = []
    for p in gcs_paths:
        stripped = p.rstrip("/")
        name = stripped.split("/")[-EXIT_CODE_ERROR]
        names.append(name)
    return names


def _load_results(output_dir, completed):
    """Load method results from downloaded directories."""
    method_results = {}
    baseline_score = None

    for method in completed:
        summary_path = os.path.join(
            output_dir, method, method, "method_result.json",
        )
        if not os.path.exists(summary_path):
            alt = os.path.join(
                output_dir, method, "method_result.json",
            )
            if os.path.exists(alt):
                summary_path = alt
            else:
                print(f"  WARNING: No result for {method}")
                continue

        with open(summary_path) as f:
            result = json.load(f)
        method_results[method] = result
        if baseline_score is None:
            baseline_score = result.get(
                "baseline_score", SCORE_RANGE_MIN,
            )

    return method_results, baseline_score


def _get_benchmark_from_pairs(output_dir, gcs_base):
    """Download pairs metadata to get benchmark name."""
    pairs_dir = os.path.join(output_dir, "pairs")
    _gcs_download_dir(f"{gcs_base}/pairs/", pairs_dir)
    for fname in os.listdir(pairs_dir):
        if fname.startswith("train_pairs_"):
            fpath = os.path.join(pairs_dir, fname)
            with open(fpath) as f:
                meta = json.load(f)
            return meta.get("task_name", "unknown")
    return "unknown"


def _print_ranking(
    method_results, baseline_score, benchmark, output_dir, job_id,
):
    """Build ranking, save report, and print results."""
    scored = {}
    for name, r in method_results.items():
        if "best_score" in r:
            scored[name] = r["best_score"]

    if not scored:
        print("ERROR: No method produced a score")
        sys.exit(EXIT_CODE_ERROR)

    winner = max(scored, key=scored.get)
    ranking = sorted(
        [
            {
                "method": n,
                "score": s,
                "delta": s - baseline_score,
            }
            for n, s in scored.items()
        ],
        key=lambda x: x["score"],
        reverse=True,
    )

    final = {
        "benchmark": benchmark,
        "baseline_score": baseline_score,
        "winner": winner,
        "winner_score": scored[winner],
        "winner_delta": scored[winner] - baseline_score,
        "timestamp": datetime.now().isoformat(),
        "job_id": job_id,
        "method_results": method_results,
        "ranking": ranking,
    }

    report_name = f"best_method_{benchmark}.json"
    final_path = os.path.join(output_dir, report_name)
    with open(final_path, "w") as f:
        json.dump(final, f, indent=JSON_INDENT, default=str)

    bstr = f"{baseline_score:.4f}" if baseline_score else "N/A"
    print(f"\n{'=' * SEPARATOR_WIDTH_WIDE}")
    print(f"RESULTS: {benchmark} (baseline: {bstr})")
    print(f"{'=' * SEPARATOR_WIDTH_WIDE}")
    for r in ranking:
        sign = "+" if r["delta"] >= SCORE_RANGE_MIN else ""
        mk = " <-- WINNER" if r["method"] == winner else ""
        nm = r["method"].rjust(SEPARATOR_WIDTH_REPORT)
        print(
            f"   {nm}: {r['score']:.4f} "
            f"({sign}{r['delta']:.4f}){mk}"
        )
    print(f"\n   Results: {final_path}")
    print(f"{'=' * SEPARATOR_WIDTH_WIDE}\n")


if __name__ == "__main__":
    main()
