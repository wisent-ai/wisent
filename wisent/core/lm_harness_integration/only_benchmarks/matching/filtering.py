"""Priority filtering functions for benchmarks."""

from __future__ import annotations

from typing import Dict, Optional


__all__ = [
    "apply_priority_filtering",
    "get_benchmarks_by_priority",
    "get_priority_summary",
    "print_priority_summary",
]


def apply_priority_filtering(
    benchmarks: Dict[str, Dict],
    priority: str = "all",
    fast_only: bool = False,
    time_budget_minutes: Optional[float] = None,
) -> Dict[str, Dict]:
    """Apply priority-based filtering to benchmarks."""
    filtered_benchmarks = {}

    for name, config in benchmarks.items():
        benchmark_priority = config.get("priority", "unknown")

        if fast_only and benchmark_priority != "high":
            continue

        if priority != "all" and benchmark_priority != priority:
            continue

        if time_budget_minutes is not None:
            loading_time = config.get("loading_time", 60.0)
            max_time_per_benchmark = time_budget_minutes * 60 / 2
            if loading_time > max_time_per_benchmark:
                continue

        filtered_benchmarks[name] = config

    return filtered_benchmarks


def get_benchmarks_by_priority(
    core_benchmarks: Dict[str, Dict], priority: str = "high"
) -> Dict[str, Dict]:
    """Get benchmarks filtered by priority level."""
    return {
        name: config
        for name, config in core_benchmarks.items()
        if config.get("priority", "unknown") == priority
    }


def get_priority_summary(core_benchmarks: Dict[str, Dict]) -> Dict[str, int]:
    """Get summary of benchmark counts by priority level."""
    priority_counts = {"high": 0, "medium": 0, "low": 0, "unknown": 0}

    for config in core_benchmarks.values():
        priority = config.get("priority", "unknown")
        priority_counts[priority] += 1

    return priority_counts


def print_priority_summary(core_benchmarks: Dict[str, Dict]) -> None:
    """Print a summary of benchmark priorities for agentic optimization."""
    priority_counts = get_priority_summary(core_benchmarks)
    total = sum(priority_counts.values())

    print("BENCHMARK PRIORITY SUMMARY FOR AGENTIC OPTIMIZATION")
    print("=" * 65)
    print(f"Total benchmarks: {total}")
    print()

    print("HIGH PRIORITY (< 13.5s - optimal for agentic use):")
    print(f"   Count: {priority_counts['high']} ({priority_counts['high']/total*100:.1f}%)")
    high_priority = get_benchmarks_by_priority(core_benchmarks, "high")
    for name in sorted(high_priority.keys()):
        print(f"   - {name}")

    print(f"\nMEDIUM PRIORITY (13.5-60s - acceptable for agentic use):")
    print(f"   Count: {priority_counts['medium']} ({priority_counts['medium']/total*100:.1f}%)")
    medium_priority = get_benchmarks_by_priority(core_benchmarks, "medium")
    for name in sorted(medium_priority.keys()):
        print(f"   - {name}")

    print(f"\nLOW PRIORITY (> 60s - deprioritized for agentic use):")
    print(f"   Count: {priority_counts['low']} ({priority_counts['low']/total*100:.1f}%)")
    low_priority = get_benchmarks_by_priority(core_benchmarks, "low")
    for name in sorted(low_priority.keys()):
        print(f"   - {name}")

    if priority_counts["unknown"] > 0:
        print(f"\nUNKNOWN PRIORITY:")
        print(f"   Count: {priority_counts['unknown']}")

    print(f"\nAGENTIC OPTIMIZATION RECOMMENDATIONS:")
    print("   - Prefer HIGH priority benchmarks for quick responses")
    print("   - Use MEDIUM priority benchmarks for balanced evaluation")
    print("   - Avoid LOW priority benchmarks for interactive agentic flows")
    high_medium = priority_counts["high"] + priority_counts["medium"]
    pct = high_medium / total * 100
    print(f"   - HIGH + MEDIUM = {high_medium} benchmarks ({pct:.1f}%) suitable for agentic use")
