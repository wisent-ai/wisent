#!/usr/bin/env python3
"""
General benchmark inspector for lm-eval tasks.

Usage:
    python inspect_benchmark.py <benchmark_name>

Example:
    python inspect_benchmark.py mmlu_abstract_algebra
    python inspect_benchmark.py hellaswag
    python inspect_benchmark.py boolq
"""

import argparse
import sys
from lm_eval.tasks import get_task_dict


def inspect_benchmark(benchmark_name: str, num_examples: int = 5) -> None:
    """Load a benchmark from lm-eval and display its structure and examples."""

    print(f"\n{'='*80}")
    print(f"INSPECTING BENCHMARK: {benchmark_name}")
    print(f"{'='*80}\n")

    # Load task
    try:
        tasks = get_task_dict([benchmark_name])
        task = tasks[benchmark_name]
    except Exception as e:
        print(f"Failed to load benchmark '{benchmark_name}': {e}")
        sys.exit(1)

    # Try all available splits
    split_methods = [
        ('test_docs', 'test'),
        ('validation_docs', 'validation'),
        ('training_docs', 'training'),
        ('fewshot_docs', 'fewshot'),
    ]

    print("AVAILABLE SPLITS:")
    print("-" * 40)

    available_splits = {}
    for method_name, split_name in split_methods:
        if hasattr(task, method_name):
            try:
                docs = list(getattr(task, method_name)())
                if docs:
                    available_splits[split_name] = (method_name, docs)
                    print(f"  {split_name}: {len(docs)} samples")
                else:
                    print(f"  {split_name}: empty")
            except Exception as e:
                print(f"  {split_name}: error - {e}")
        else:
            print(f"  {split_name}: not available")

    if not available_splits:
        print("\nNo data found in any split!")
        sys.exit(1)

    # For each available split, show examples
    for split_name, (method_name, docs) in available_splits.items():
        print(f"\n{'='*80}")
        print(f"SPLIT: {split_name.upper()} ({len(docs)} samples)")
        print(f"{'='*80}")

        # Show structure
        print(f"\nDocument keys: {list(docs[0].keys())}")
        print("\nField types:")
        for key in docs[0].keys():
            value = docs[0].get(key)
            print(f"  {key}: {type(value).__name__}")

        # Show examples
        print(f"\n{'-'*40}")
        print(f"FIRST {min(num_examples, len(docs))} EXAMPLES:")
        print(f"{'-'*40}")

        for i, doc in enumerate(docs[:num_examples]):
            print(f"\n--- EXAMPLE {i+1} ---")
            for key, value in doc.items():
                if isinstance(value, str):
                    print(f"{key}: {value}")
                else:
                    print(f"{key}: {value} (type: {type(value).__name__})")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect lm-eval benchmark format and data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python inspect_benchmark.py mmlu_abstract_algebra
    python inspect_benchmark.py hellaswag
    python inspect_benchmark.py boolq
    python inspect_benchmark.py --examples 10 truthfulqa_mc1
        """
    )
    parser.add_argument(
        "benchmark",
        help="Name of the lm-eval benchmark to inspect"
    )
    parser.add_argument(
        "--examples", "-n",
        type=int,
        default=2,
        help="Number of examples to display per split (default: 5)"
    )

    args = parser.parse_args()
    inspect_benchmark(args.benchmark, args.examples)


if __name__ == "__main__":
    main()
