"""Run All benchmarks and return results as markdown table."""

import time

from wisent.core.utils.config_tools.constants import INDEX_FIRST


def run_all_benchmarks(category: str, limit: float | None) -> str:
    """Run tests for all benchmarks in a category. Returns markdown."""
    from wisent.app.ui.tabs.benchmark_debug import _get_benchmarks_for_category
    from wisent.support.examples.scripts.discovery.validation.test_single_benchmark import test_benchmark

    benchmarks = _get_benchmarks_for_category(category or "all")

    # Strip labels
    clean = []
    for b in benchmarks:
        if " (" in b and b.endswith(")"):
            clean.append(b.split(" (")[INDEX_FIRST])
        else:
            clean.append(b)

    lines = ["| Benchmark | Extraction | Evaluator | Details |",
             "|-----------|------------|-----------|---------|"]

    pass_count = INDEX_FIRST
    fail_count = INDEX_FIRST
    start = time.time()

    for task_name in clean:
        r = test_benchmark(task_name)
        ext = r.get("extraction", {}).get("status", "SKIP")
        evl = r.get("evaluator", {}).get("status", "SKIP")
        pairs = r.get("extraction", {}).get("pair_count", "-")
        detail = r.get("evaluator", {}).get("detail", "")[:100]

        overall = "PASS" if ext != "FAIL" and evl != "FAIL" else "FAIL"
        if overall == "PASS":
            pass_count += 1
        else:
            fail_count += 1

        lines.append(f"| {task_name} | {ext} ({pairs}) | {evl} | {detail} |")

    elapsed = time.time() - start
    summary = (f"\n**Summary:** {pass_count} PASS, {fail_count} FAIL, "
               f"{elapsed:.1f}s total")
    lines.append(summary)
    return "\n".join(lines)
