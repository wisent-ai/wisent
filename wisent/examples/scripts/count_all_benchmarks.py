#!/usr/bin/env python3
"""Count maximum contrastive pairs for all benchmarks."""

import json
import os
import sys
from pathlib import Path

# Set environment variables
os.environ['HF_DATASETS_TRUST_REMOTE_CODE'] = '1'
os.environ['HF_ALLOW_CODE_EVAL'] = '1'

# Add wisent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from wisent.core.data_loaders.loaders.lm_loader import LMEvalDataLoader
from wisent.core.data_loaders.loaders.huggingface_loader import HuggingFaceDataLoader


def load_benchmarks():
    """Load benchmarks from central registry."""
    from wisent.core.benchmark_registry import get_all_benchmarks
    return get_all_benchmarks()


def count_pairs(task_name: str) -> int:
    """Count maximum contrastive pairs for a task."""
    try:
        # Determine loader type
        hf_tasks = [
            "math", "math_500", "aime", "hmmt", "polymath", "livemathbench",
            "humaneval", "humaneval_plus", "mbpp", "mbpp_plus",
            "instruct_humaneval", "apps", "conala", "concode",
            "ds", "ds1000", "ds_1000", "mercury", "recode",
            "multipl", "multiple_", "multipl_e",
            "codexglue", "livecodebench",
            "super_gpqa", "supergpqa", "hle",
            "tag",
            "meddialog",
            "mmlusr"
        ]

        lm_eval_only_tasks = [
            "minerva_math", "code_x_glue", "humaneval_infilling", "mathqa"
        ]

        if any(task_name.lower() == t or task_name.lower().startswith(t + "_") for t in lm_eval_only_tasks):
            loader = LMEvalDataLoader()
        elif any(task_name.lower().startswith(t) for t in hf_tasks):
            loader = HuggingFaceDataLoader()
        else:
            loader = LMEvalDataLoader()

        # Load with no limit to get full count
        result = loader._load_one_task(
            task_name=task_name,
            split_ratio=0.8,
            seed=42,
            limit=None,  # No limit
            training_limit=None,
            testing_limit=None
        )

        train_pairs = len(result['train_qa_pairs'].pairs) if result.get('train_qa_pairs') else 0
        test_pairs = len(result['test_qa_pairs'].pairs) if result.get('test_qa_pairs') else 0
        total_pairs = train_pairs + test_pairs

        return total_pairs

    except Exception as e:
        print(f"Error counting pairs for {task_name}: {e}", file=sys.stderr)
        return -1


def main():
    benchmarks = load_benchmarks()

    print(f"Counting maximum contrastive pairs for {len(benchmarks)} benchmarks...\n")

    results = {}

    for i, benchmark in enumerate(benchmarks, 1):
        print(f"[{i}/{len(benchmarks)}] Counting {benchmark}...", end=" ", flush=True)

        count = count_pairs(benchmark)
        results[benchmark] = count

        if count >= 0:
            print(f"{count:,} pairs")
        else:
            print("ERROR")

    # Save results
    output_file = Path(__file__).parent / "benchmark_pair_counts.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"\nResults saved to {output_file}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    successful = {k: v for k, v in results.items() if v >= 0}
    failed = {k: v for k, v in results.items() if v < 0}

    print(f"Successfully counted: {len(successful)}/{len(benchmarks)}")
    print(f"Failed: {len(failed)}/{len(benchmarks)}")

    if successful:
        total = sum(successful.values())
        avg = total / len(successful)
        print(f"\nTotal pairs across all benchmarks: {total:,}")
        print(f"Average pairs per benchmark: {avg:,.0f}")
        print(f"Max pairs: {max(successful.values()):,} ({max(successful, key=successful.get)})")
        print(f"Min pairs: {min(successful.values()):,} ({min(successful, key=successful.get)})")


if __name__ == "__main__":
    main()
