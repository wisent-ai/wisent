"""CLI entry point for benchmark testing."""

from __future__ import annotations

import json
import os
import sys
from typing import Dict

from ..registry import CORE_BENCHMARKS
from ..readme_parsing import update_all_benchmarks_from_readme
from ..matching import (
    find_most_relevant_benchmarks,
    get_benchmarks_by_priority,
    print_priority_summary,
)
from .tests import test_benchmark_creation, test_single_benchmark_direct, test_readme_updates


__all__ = ["main"]


def main() -> None:
    """Main function to test ALL core benchmarks comprehensively."""
    print("Testing ALL core benchmarks comprehensively")

    print("\nUpdating all benchmarks with README information...")
    updated_benchmarks = update_all_benchmarks_from_readme(CORE_BENCHMARKS)

    print(f"Processing {len(updated_benchmarks)} comprehensive benchmarks across all categories...")
    print("Categories: Benchmark Suites, Hallucination, Reasoning, QA/Reading, Knowledge, Math, Coding")
    print("           Bias/Toxicity, Adversarial, Multilingual, Medical, Language Modeling, Long Context")
    print("           Temporal, Linguistic, Translation, Dialogue")
    print("FAIL-HARD MODE: Script will exit with code 1 on first benchmark failure!")

    os.makedirs("test_results", exist_ok=True)

    results: Dict = {
        "dataset_creation": {"successful": [], "failed": []},
        "cli_testing": {"successful": [], "failed": []},
        "benchmark_tags": {},
    }

    total_benchmarks = len(updated_benchmarks)
    current_idx = 0

    for benchmark_name, benchmark_config in updated_benchmarks.items():
        current_idx += 1
        task_name = benchmark_config["task"]
        tags = benchmark_config["tags"]

        results["benchmark_tags"][benchmark_name] = {
            "task": task_name,
            "tags": tags,
            "original_tags": tags,
        }

        print(f"\n{'='*80}")
        print(f"BENCHMARK {current_idx}/{total_benchmarks}: {benchmark_name}")
        print("=" * 80)

        print(f"\n{'='*80}")
        print(f"STEP 1: Testing dataset creation for {benchmark_name}")
        print("=" * 80)

        dataset_success, actual_tags = test_benchmark_creation(benchmark_name, benchmark_config)

        if actual_tags != benchmark_config["tags"]:
            print(f"Updated tags from README: {actual_tags}")
            benchmark_config = benchmark_config.copy()
            benchmark_config["tags"] = actual_tags
            results["benchmark_tags"][benchmark_name]["tags"] = actual_tags

        if dataset_success:
            results["dataset_creation"]["successful"].append(benchmark_name)

            print(f"\n{'='*80}")
            print(f"STEP 2: Testing CLI integration for {benchmark_name}")
            print("=" * 80)

            cli_success = test_single_benchmark_direct(benchmark_name, benchmark_config)

            if cli_success:
                results["cli_testing"]["successful"].append(benchmark_name)
            else:
                results["cli_testing"]["failed"].append(benchmark_name)
                print(f"\nFATAL ERROR: CLI testing failed for {benchmark_name}")
                print("Script failing hard as requested!")
                print(f"Benchmark: {benchmark_name} ({task_name})")
                print(f"Tags: {', '.join(tags)}")
                sys.exit(1)
        else:
            results["dataset_creation"]["failed"].append(benchmark_name)
            results["cli_testing"]["failed"].append(benchmark_name)
            print(f"\nFATAL ERROR: Dataset creation failed for {benchmark_name}")
            print("Script failing hard as requested!")
            print(f"Benchmark: {benchmark_name} ({task_name})")
            print(f"Tags: {', '.join(tags)}")
            sys.exit(1)

        print(f"\n{'='*80}")
        print("CURRENT STATUS")
        print("=" * 80)
        print(f"Dataset creation successful: {len(results['dataset_creation']['successful'])}")
        print(f"Dataset creation failed: {len(results['dataset_creation']['failed'])}")
        print(f"CLI testing successful: {len(results['cli_testing']['successful'])}")
        print(f"CLI testing failed: {len(results['cli_testing']['failed'])}")

        print(f"\nSuccessfully completed testing {benchmark_name}")
        print("Moving to next benchmark...\n")

    _print_final_summary(results, updated_benchmarks)


def _print_final_summary(results: Dict, updated_benchmarks: Dict) -> None:
    """Print final summary of test results."""
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print("=" * 60)

    print(f"\nDataset Creation Results:")
    print(f"Successful: {len(results['dataset_creation']['successful'])}")
    for name in results["dataset_creation"]["successful"]:
        tags = results["benchmark_tags"][name]["tags"]
        print(f"  - {name} ({', '.join(tags)})")

    print(f"\nFailed: {len(results['dataset_creation']['failed'])}")
    for name in results["dataset_creation"]["failed"]:
        tags = results["benchmark_tags"][name]["tags"]
        print(f"  - {name} ({', '.join(tags)})")

    print(f"\nCLI Testing Results:")
    print(f"Successful: {len(results['cli_testing']['successful'])}")
    for name in results["cli_testing"]["successful"]:
        tags = results["benchmark_tags"][name]["tags"]
        print(f"  - {name} ({', '.join(tags)})")

    print(f"\nFailed: {len(results['cli_testing']['failed'])}")
    for name in results["cli_testing"]["failed"]:
        tags = results["benchmark_tags"][name]["tags"]
        print(f"  - {name} ({', '.join(tags)})")

    results_file = "test_results/benchmark_test_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    print(f"\nSUCCESS! All {len(updated_benchmarks)} benchmarks passed!")
    print("No failures detected - all benchmarks working with wisent CLI")
    print("Ready for production use!")


def cli_entry() -> None:
    """CLI entry point with subcommand handling."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "test_readme":
            test_readme_updates(CORE_BENCHMARKS)
        elif sys.argv[1] == "test_matching":
            from .tests import test_benchmark_matching
            test_benchmark_matching(
                lambda p, top_k=3: find_most_relevant_benchmarks(p, CORE_BENCHMARKS, top_k=top_k)
            )
        elif sys.argv[1] == "priorities":
            print_priority_summary(CORE_BENCHMARKS)
        elif sys.argv[1] == "high":
            high_priority = get_benchmarks_by_priority(CORE_BENCHMARKS, "high")
            print(f"HIGH PRIORITY BENCHMARKS ({len(high_priority)}):")
            for name in sorted(high_priority.keys()):
                print(f"   - {name}")
        elif sys.argv[1] == "medium":
            medium_priority = get_benchmarks_by_priority(CORE_BENCHMARKS, "medium")
            print(f"MEDIUM PRIORITY BENCHMARKS ({len(medium_priority)}):")
            for name in sorted(medium_priority.keys()):
                print(f"   - {name}")
        elif sys.argv[1] == "low":
            low_priority = get_benchmarks_by_priority(CORE_BENCHMARKS, "low")
            print(f"LOW PRIORITY BENCHMARKS ({len(low_priority)}):")
            for name in sorted(low_priority.keys()):
                print(f"   - {name}")
        elif sys.argv[1] == "find":
            if len(sys.argv) > 2:
                prompt = " ".join(sys.argv[2:])
                print(f"Finding benchmarks for: '{prompt}'")
                matches = find_most_relevant_benchmarks(prompt, CORE_BENCHMARKS)
                for i, match in enumerate(matches, 1):
                    print(f"\n{i}. **{match['benchmark']}** (score: {match['score']})")
                    print(f"   Description: {match['description']}")
                    print(f"   Reasons: {', '.join(match['reasons'])}")
                    print(f"   Tags: {match['tags']}")
                    print(f"   Task: {match['task']}")
                    if match["groups"]:
                        print(f"   Groups: {match['groups']}")
                    print(f"   Priority: {match['priority']}")
            else:
                print("Usage: python -m only_benchmarks find <prompt>")
        else:
            print("Usage: [test_readme|test_matching|priorities|high|medium|low|find <prompt>]")
    else:
        main()


if __name__ == "__main__":
    cli_entry()
