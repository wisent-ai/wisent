"""Run All benchmarks and return results as markdown table."""

import time

from wisent.core.utils.config_tools.constants import (
    INDEX_FIRST,
    TEST_EXTRACTOR_EVALUATOR_DEFAULT_LIMIT,
)


def run_all_benchmarks(category: str, limit: float | None) -> str:
    """Run tests for all benchmarks in a category. Returns markdown."""
    from wisent.app.ui.tabs.benchmark_debug import (
        _get_benchmarks_for_category, _find_subtasks, _test_single_task,
    )
    from wisent.extractors.lm_eval.lm_extractor_registry import _REGISTRY

    all_reg = sorted(_REGISTRY.keys())
    benchmarks = _get_benchmarks_for_category(category or "all")

    # Strip labels
    clean = []
    for b in benchmarks:
        if " (" in b and b.endswith(")"):
            clean.append(b.split(" (")[INDEX_FIRST])
        else:
            clean.append(b)

    lines = ["| Benchmark | Status | Pairs | Details |",
             "|-----------|--------|-------|---------|"]

    pass_count = INDEX_FIRST
    fail_count = INDEX_FIRST
    start = time.time()

    for task_name in clean:
        subtasks = _find_subtasks(task_name, all_reg)
        tasks = subtasks if subtasks else [task_name]

        for t in tasks:
            r = _test_single_task(t, limit)
            status = r["status"]
            pairs = r.get("pairs", "-")
            details = r.get("details", "")
            if r.get("correct"):
                details = f"correct={r['correct']} incorrect={r['incorrect']}"

            if status == "PASS":
                pass_count += INDEX_FIRST + INDEX_FIRST
                lines.append(f"| {t} | PASS | {pairs} | {details} |")
            else:
                fail_count += INDEX_FIRST + INDEX_FIRST
                short = details[:TEST_EXTRACTOR_EVALUATOR_DEFAULT_LIMIT
                                * TEST_EXTRACTOR_EVALUATOR_DEFAULT_LIMIT]
                lines.append(f"| {t} | FAIL | {pairs} | {short} |")

    elapsed = time.time() - start
    summary = (f"\n**Summary:** {pass_count} PASS, {fail_count} FAIL, "
               f"{elapsed:.1f}s total")
    lines.append(summary)
    return "\n".join(lines)
