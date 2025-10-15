#!/usr/bin/env python3
"""
Check the number of questions available in all benchmark datasets.

This script loads the datasets directly to get accurate counts for each document type.
NO FALLBACK - reports exactly what exists in each split.
"""

from __future__ import annotations

import sys
import os
# Add project root to Python path for wisent_guard imports
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from lm_eval import tasks
from typing import Dict


def check_dataset_size(task_name: str) -> Dict[str, int]:
    """
    Check how many documents are available in each split of the benchmark.

    Args:
        task_name: Name of the benchmark task (e.g., "boolq", "gsm8k", "cb", "sst2")

    Returns:
        Dict mapping document type to count (0 if not available)
    """
    print(f"\n{'='*80}")
    print(f"Checking {task_name.upper()} dataset...")
    print(f"{'='*80}")

    try:
        # Load the task
        task_dict = tasks.get_task_dict([task_name])
        task = task_dict[task_name]

        print(f"✓ Task loaded: {task_name}")
    except Exception as e:
        print(f"✗ Failed to load task {task_name}: {e}")
        return {"training": 0, "validation": 0, "test": 0}

    doc_counts = {}

    # Map doc_type to actual split names that might exist
    split_mapping = {
        "training": ["train", "training", "train_hf"],
        "validation": ["validation", "val", "dev"],
        "test": ["test"],
    }

    print(f"\nChecking document splits...")
    print("-" * 80)

    for doc_type, possible_names in split_mapping.items():
        found = False

        # Method 1: Check if task has a dataset attribute with splits
        if hasattr(task, 'dataset') and hasattr(task.dataset, 'keys'):
            splits = list(task.dataset.keys())

            # Only print available splits once
            if doc_type == "training":
                print(f"Available splits in dataset: {splits}")
                print()

            for split_name in possible_names:
                if split_name in splits:
                    try:
                        count = len(task.dataset[split_name])
                        doc_counts[doc_type] = count
                        print(f"  ✓ {doc_type.upper():12} (split: '{split_name:12}'): {count:6,} documents")
                        found = True
                        break
                    except Exception as e:
                        print(f"  ✗ {doc_type.upper():12} (split: '{split_name}'): Error - {e}")

        # Method 2: Try using task methods like train_docs(), test_docs(), etc.
        if not found:
            for split_name in possible_names:
                method_name = f"{split_name}_docs"
                if hasattr(task, method_name):
                    try:
                        docs = list(getattr(task, method_name)())
                        count = len(docs)
                        doc_counts[doc_type] = count
                        print(f"  ✓ {doc_type.upper():12} (method: '{method_name}()'): {count:6,} documents")
                        found = True
                        break
                    except Exception as e:
                        print(f"  ✗ {doc_type.upper():12} (method: '{method_name}()'): Error - {e}")

        # If still not found, report as not available
        if not found:
            doc_counts[doc_type] = 0
            print(f"  ✗ {doc_type.upper():12}: NOT AVAILABLE (no such split)")

    return doc_counts


def main():
    """Check available questions in all benchmark datasets."""

    print("\n" + "#" * 80)
    print("# BENCHMARK DATASETS - SIZE CHECK")
    print("# Benchmarks: CB, GSM8K, BoolQ, SST2")
    print("#" * 80)

    # Check all benchmarks
    benchmarks = ["cb", "gsm8k", "boolq", "sst2"]
    all_results = {}

    for benchmark in benchmarks:
        doc_counts = check_dataset_size(benchmark)
        all_results[benchmark] = doc_counts

    # Create summary table
    print("\n\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print()

    # Header
    print(f"{'Benchmark':<15} {'Training':<15} {'Validation':<15} {'Test':<15}")
    print("-" * 80)

    # Rows
    for benchmark in benchmarks:
        counts = all_results[benchmark]
        train = counts.get("training", 0)
        val = counts.get("validation", 0)
        test = counts.get("test", 0)

        train_str = f"{train:,}" if train > 0 else "NOT AVAILABLE"
        val_str = f"{val:,}" if val > 0 else "NOT AVAILABLE"
        test_str = f"{test:,}" if test > 0 else "NOT AVAILABLE"

        print(f"{benchmark.upper():<15} {train_str:<15} {val_str:<15} {test_str:<15}")

    print("=" * 80)

    # Important note about contrastive pairs
    print("\n" + "!" * 80)
    print("IMPORTANT NOTE:")
    print("  - Each document generates 1 CONTRASTIVE PAIR (positive + negative response)")
    print("  - If a split has N documents, you can extract up to N pairs from it")
    print("  - 'NOT AVAILABLE' means that split does not exist in the dataset")
    print("!" * 80)

    # Analysis for common experiment sizes
    print("\n" + "=" * 80)
    print("ANALYSIS: Common Experiment Sizes")
    print("=" * 80)

    common_sizes = [50, 100, 150, 200, 250, 500, 1000]

    for benchmark in benchmarks:
        counts = all_results[benchmark]
        train = counts.get("training", 0)
        val = counts.get("validation", 0)
        test = counts.get("test", 0)

        print(f"\n{benchmark.upper()}:")
        print("-" * 40)

        # Training
        if train > 0:
            print(f"  Training ({train:,} available):")
            for size in common_sizes:
                if size <= train:
                    print(f"    ✓ Can use {size} pairs")
                elif size > train:
                    print(f"    ✗ Cannot use {size} pairs")
                    break
        else:
            print(f"  Training: NOT AVAILABLE")

        # Validation
        if val > 0:
            print(f"  Validation ({val:,} available):")
            for size in common_sizes:
                if size <= val:
                    print(f"    ✓ Can use {size} pairs")
                elif size > val:
                    print(f"    ✗ Cannot use {size} pairs")
                    break
        else:
            print(f"  Validation: NOT AVAILABLE")

        # Test
        if test > 0:
            print(f"  Test ({test:,} available):")
            for size in common_sizes:
                if size <= test:
                    print(f"    ✓ Can use {size} pairs")
                elif size > test:
                    print(f"    ✗ Cannot use {size} pairs")
                    break
        else:
            print(f"  Test: NOT AVAILABLE")

    print("\n" + "#" * 80)
    print("# DONE")
    print("#" * 80)


if __name__ == "__main__":
    main()
