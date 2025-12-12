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
import os
import sys

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

from lm_eval.api.task import TaskConfig
from lm_eval.tasks import get_task_dict

# Patch TaskConfig to ignore 'group' key (lm-eval 0.4.9 bug)
_original_taskconfig_init = TaskConfig.__init__
def _patched_taskconfig_init(self, **kwargs):
    kwargs.pop('group', None)
    return _original_taskconfig_init(self, **kwargs)
TaskConfig.__init__ = _patched_taskconfig_init


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

    max_docs_to_load = 100  # Limit to avoid loading millions of rows into memory
    available_splits = {}
    for method_name, split_name in split_methods:
        if hasattr(task, method_name):
            try:
                docs_iter = getattr(task, method_name)()
                docs = []
                for i, doc in enumerate(docs_iter):
                    docs.append(doc)
                    if i >= max_docs_to_load - 1:
                        break
                if docs:
                    has_more = len(docs) >= max_docs_to_load
                    available_splits[split_name] = (method_name, docs)
                    count_str = f"{len(docs)}+" if has_more else str(len(docs))
                    print(f"  {split_name}: {count_str} samples")
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

            # Formatted via doc_to_* methods
            print("\nFORMATTED (doc_to_* methods):")

            if hasattr(task, 'doc_to_text'):
                try:
                    text = task.doc_to_text(doc)
                    print(f"  doc_to_text: {repr(text)}")
                except Exception as e:
                    print(f"  doc_to_text: error - {e}")

            if hasattr(task, 'doc_to_target'):
                try:
                    target = task.doc_to_target(doc)
                    print(f"  doc_to_target: {repr(target)}")
                except Exception as e:
                    print(f"  doc_to_target: error - {e}")

            if hasattr(task, 'doc_to_choice'):
                try:
                    choices = task.doc_to_choice(doc)
                    if choices is not None:
                        print(f"  doc_to_choice: {choices}")
                except Exception as e:
                    print(f"  doc_to_choice: error - {e}")


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
        default=1,
        help="Number of examples to display per split (default: 5)"
    )

    args = parser.parse_args()
    inspect_benchmark(args.benchmark, args.examples)


if __name__ == "__main__":
    main()
